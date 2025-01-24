import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
from pathlib import Path
import requests
import firebase_admin
from firebase_admin import credentials, auth
import logging
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2, TeamGameLog
from nba_api.stats.static import teams as nba_teams
import nfl_data_py as nfl
import cbbpy.mens_scraper as cbb

# Configure logging
logging.basicConfig(level=logging.INFO)

################################################################################
# FIREBASE CONFIGURATION
################################################################################

try:
    FIREBASE_API_KEY = st.secrets["general"]["firebaseApiKey"]
    service_account_info = {
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"],
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
    }
    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
except KeyError:
    st.warning("Firebase secrets not found or incomplete in st.secrets. Please verify your secrets.toml.")

def login_with_rest(email, password):
    try:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        response = requests.post(url, json=payload)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Login error: {e}")
        return None

def signup_user(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        st.success(f"User {email} created successfully!")
        return user
    except Exception as e:
        st.error(f"Signup error: {e}")

def logout_user():
    for key in ['email', 'logged_in']:
        st.session_state.pop(key, None)

################################################################################
# ENHANCED DATA PROCESSING
################################################################################

def preprocess_data_with_knn(data, n_neighbors=5):
    numeric_cols = ['off_rating', 'def_rating', 'pace', 'score']
    
    # Add rolling features
    data['rolling_mean_3'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    data['rolling_std_3'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(3, min_periods=1).std().fillna(0)
    )
    
    # KNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors)
    impute_cols = numeric_cols + ['rolling_mean_3', 'rolling_std_3']
    data[impute_cols] = imputer.fit_transform(data[impute_cols])
    return data

def add_contextual_factors(data):
    data['back_to_back'] = data.groupby('team')['gameday'].diff().dt.days <= 1
    return data

def round_half(number):
    return round(number * 2) / 2

################################################################################
# MODEL TRAINING & PREDICTION
################################################################################

@st.cache_data(ttl=14400)
def train_team_models(team_data: pd.DataFrame):
    gbr_models = {}
    arima_models = {}
    team_stats = {}
    global_mean = team_data['score'].mean()

    for team in team_data['team'].unique():
        try:
            df_team = team_data[team_data['team'] == team].copy()
            df_team.sort_values('gameday', inplace=True)
            scores = df_team['score'].reset_index(drop=True)

            # Team statistics with fallbacks
            team_stats[team] = {
                'mean': round_half(scores.mean() if len(scores) > 0 else global_mean),
                'std': round_half(scores.std() if len(scores) > 1 else 5),
                'recent_form': round_half(scores.tail(5).mean() if len(scores) >=5 else global_mean)
            }

            # GBR Model
            if len(scores) >= 10:
                features = ['off_rating', 'def_rating', 'rolling_mean_3', 'rolling_std_3']
                X_train = df_team[features].values[-10:]
                y_train = scores.values[-10:]
                
                gbr_model = GradientBoostingRegressor(random_state=42)
                gbr_model.fit(X_train, y_train)
                gbr_models[team] = gbr_model

            # ARIMA Model
            if len(scores) >= 7:
                arima_model = auto_arima(
                    scores,
                    seasonal=True,
                    m=7,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    max_p=3,
                    max_q=3,
                )
                arima_models[team] = arima_model

        except Exception as e:
            logging.error(f"Model training failed for {team}: {str(e)}")

    return gbr_models, arima_models, team_stats

def weighted_ensemble_prediction(gbr_pred, arima_pred, gbr_mae, arima_mae):
    if gbr_pred is None and arima_pred is None:
        return None
    if gbr_pred is None:
        return arima_pred
    if arima_pred is None:
        return gbr_pred

    gbr_weight = 1 / (gbr_mae + 1e-6)
    arima_weight = 1 / (arima_mae + 1e-6)
    return (gbr_pred * gbr_weight + arima_pred * arima_weight) / (gbr_weight + arima_weight)

def predict_team_score(team, gbr_models, arima_models, team_stats, team_data):
    if team not in team_stats:
        return team_data['score'].mean(), (0, 0)
    
    try:
        # GBR Prediction
        gbr_pred = None
        if team in gbr_models:
            df_team = team_data[team_data['team'] == team]
            features = ['off_rating', 'def_rating', 'rolling_mean_3', 'rolling_std_3']
            X_next = df_team[features].iloc[[-1]].values
            gbr_pred = gbr_models[team].predict(X_next)[0]

        # ARIMA Prediction
        arima_pred = None
        if team in arima_models:
            arima_pred = arima_models[team].predict(n_periods=1)[0]

        return weighted_ensemble_prediction(gbr_pred, arima_pred, 3, 2), (gbr_pred, arima_pred)
    except Exception as e:
        logging.error(f"Prediction failed for {team}: {str(e)}")
        return team_stats[team]['mean'], (None, None)

################################################################################
# IMPROVED EVALUATION & CONFIDENCE
################################################################################

def evaluate_models(team_data):
    gbr_errors, arima_errors = [], []
    tscv = TimeSeriesSplit(n_splits=3)

    for team in team_data['team'].unique():
        try:
            df_team = team_data[team_data['team'] == team].copy()
            df_team.sort_values('gameday', inplace=True)
            scores = df_team['score'].reset_index(drop=True)

            if len(scores) < 10:
                continue

            for train_index, test_index in tscv.split(scores):
                # GBR Evaluation
                features = ['off_rating', 'def_rating', 'rolling_mean_3']
                X_train = df_team.iloc[train_index][features]
                y_train = scores.iloc[train_index]
                X_test = df_team.iloc[test_index][features]
                y_test = scores.iloc[test_index]

                gbr_model = GradientBoostingRegressor(random_state=42)
                gbr_model.fit(X_train, y_train)
                gbr_errors.extend(abs(gbr_model.predict(X_test) - y_test))

                # ARIMA Evaluation
                arima_model = auto_arima(
                    scores.iloc[train_index],
                    seasonal=True,
                    m=7,
                    trace=False
                )
                arima_errors.extend(abs(arima_model.predict(len(test_index)) - scores.iloc[test_index]))

        except Exception as e:
            logging.error(f"Evaluation failed for {team}: {str(e)}")

    return (np.mean(gbr_errors) if gbr_errors else 5,
            np.mean(arima_errors) if arima_errors else 5)

def calculate_confidence(home_pred, away_pred, home_std, away_std, team_data):
    league_std = team_data['score'].std()
    combined_std = (home_std + away_std + league_std) / 3
    diff = home_pred - away_pred
    
    spread_conf = min(99, max(1, 50 + abs(diff)/(combined_std + 1e-6)*20))
    total_conf = min(99, max(1, 50 + (1/(league_std + 1e-6))*15))
    return round(spread_conf, 2), round(total_conf, 2)

def evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats, team_data):
    ou_threshold = team_data['score'].mean() * 2
    home_std = team_stats.get(home_team, {}).get('std', team_data['score'].std())
    away_std = team_stats.get(away_team, {}).get('std', team_data['score'].std())
    
    spread_conf, total_conf = calculate_confidence(
        home_pred, away_pred, home_std, away_std, team_data
    )
    
    return {
        'predicted_winner': home_team if home_pred > away_pred else away_team,
        'diff': round_half(home_pred - away_pred),
        'total_points': round_half(home_pred + away_pred),
        'spread_confidence': spread_conf,
        'total_confidence': total_conf,
        'spread_suggestion': f"Lean {home_team if home_pred > away_pred else away_team} by {round_half(home_pred - away_pred):.1f}",
        'ou_suggestion': f"Take {'Over' if (home_pred + away_pred) > ou_threshold else 'Under'} {ou_threshold:.1f}"
    }

################################################################################
# DATA LOADING & SCHEDULE FUNCTIONS
################################################################################

@st.cache_data(ttl=14400)
def load_nfl_schedule():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])
    schedule['gameday'] = pd.to_datetime(schedule['gameday']).dt.tz_localize(None)
    return schedule

@st.cache_data(ttl=14400)
def load_nba_data():
    nba_teams_list = nba_teams.get_teams()
    seasons = ['2022-23', '2023-24']
    all_rows = []
    
    for season in seasons:
        for team in nba_teams_list:
            try:
                gl = TeamGameLog(team_id=team['id'], season=season).get_data_frames()[0]
                gl['GAME_DATE'] = pd.to_datetime(gl['GAME_DATE'])
                
                for _, row in gl.iterrows():
                    all_rows.append({
                        'gameday': row['GAME_DATE'],
                        'team': team['abbreviation'],
                        'score': row['PTS'],
                        'off_rating': row['PTS'] / (row['FGA'] + 0.44 * row['FTA'] - row['OREB'] + row['TOV']) * 100,
                        'def_rating': row['PTS_OPP'] / (row['FGA'] + 0.44 * row['FTA'] - row['OREB'] + row['TOV']) * 100,
                        'pace': row['FGA'] + 0.44 * row['FTA'] - row['OREB'] + row['TOV']
                    })
            except Exception as e:
                logging.error(f"NBA data error for {team['abbreviation']}: {str(e)}")
    
    df = pd.DataFrame(all_rows)
    return preprocess_data_with_knn(df)

@st.cache_data(ttl=14400)
def load_ncaab_data_current_season():
    info_df, _, _ = cbb.get_games_season(season=2024, info=True, box=False, pbp=False)
    home_df = info_df.rename(columns={"home_team": "team", "home_score": "score", "game_day": "gameday"})[['gameday', 'team', 'score']]
    away_df = info_df.rename(columns={"away_team": "team", "away_score": "score", "game_day": "gameday"})[['gameday', 'team', 'score']]
    data = pd.concat([home_df, away_df], ignore_index=True)
    return preprocess_data_with_knn(data)

def fetch_upcoming_nfl_games(schedule):
    cutoff = datetime.now() - timedelta(days=1)
    upcoming = schedule[schedule['gameday'] > cutoff]
    return upcoming[['gameday', 'home_team', 'away_team']].head(5)

def fetch_upcoming_nba_games():
    try:
        sb = ScoreboardV2().get_data_frames()[0]
        sb['GAME_DATE'] = pd.to_datetime(sb['GAME_DATE_EST'])
        return sb[['GAME_DATE', 'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION']].rename(
            columns={'GAME_DATE': 'gameday', 'HOME_TEAM_ABBREVIATION': 'home_team', 'VISITOR_TEAM_ABBREVIATION': 'away_team'}
        )
    except:
        return pd.DataFrame([{'gameday': datetime.now() + timedelta(1), 'home_team': 'LAL', 'away_team': 'GSW'}])

def fetch_upcoming_ncaab_games():
    try:
        games, _, _ = cbb.get_games_date(datetime.now().strftime("%Y-%m-%d"))
        return games[['game_day', 'home_team', 'away_team']].rename(columns={'game_day': 'gameday'})
    except:
        return pd.DataFrame([{'gameday': datetime.now() + timedelta(1), 'home_team': 'Duke', 'away_team': 'UNC'}])

################################################################################
# MAIN APPLICATION
################################################################################

CSV_FILE = "predictions.csv"

def initialize_csv():
    if not Path(CSV_FILE).exists():
        pd.DataFrame(columns=[
            "date", "league", "home_team", "away_team", "home_pred", "away_pred",
            "predicted_winner", "total", "spread_suggestion", "ou_suggestion"
        ]).to_csv(CSV_FILE, index=False)

def save_predictions_to_csv(predictions):
    pd.DataFrame(predictions).to_csv(CSV_FILE, mode='a', index=False, header=False)
    st.success("Predictions saved!")

def run_league_pipeline(league_choice):
    results = []
    team_stats_global = {}

    st.header(f"{league_choice} Predictions")
    
    try:
        if league_choice == "NFL":
            schedule = load_nfl_schedule()
            team_data = preprocess_nfl_data(schedule)
            upcoming = fetch_upcoming_nfl_games(schedule)
        elif league_choice == "NBA":
            team_data = load_nba_data()
            upcoming = fetch_upcoming_nba_games()
        elif league_choice == "NCAAB":
            team_data = load_ncaab_data_current_season()
            upcoming = fetch_upcoming_ncaab_games()
        else:
            st.error("Invalid league choice")
            return

        gbr_models, arima_models, team_stats_global = train_team_models(team_data)
        gbr_mae, arima_mae = evaluate_models(team_data)

        for _, game in upcoming.iterrows():
            home_pred, (gbr_h, arima_h) = predict_team_score(
                game['home_team'], gbr_models, arima_models, team_stats_global, team_data
            )
            away_pred, (gbr_a, arima_a) = predict_team_score(
                game['away_team'], gbr_models, arima_models, team_stats_global, team_data
            )
            
            outcome = evaluate_matchup(
                game['home_team'], game['away_team'], 
                home_pred, away_pred, team_stats_global, team_data
            )
            
            results.append({
                **outcome,
                'date': game['gameday'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_pred': round_half(home_pred),
                'away_pred': round_half(away_pred),
            })

        st.dataframe(pd.DataFrame(results))
        
        if st.button("Save Predictions"):
            save_predictions_to_csv(results)

    except Exception as e:
        st.error(f"Error processing {league_choice}: {str(e)}")

def main():
    st.set_page_config(
        page_title="FoxEdge Sports Analytics",
        page_icon="🏀",
        layout="centered"
    )
    initialize_csv()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("FoxEdge Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                user = login_with_rest(email, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.email = user.get('email', '')
                    st.rerun()
        with col2:
            if st.button("Sign Up"):
                signup_user(email, password)
        return

    st.sidebar.title("Navigation")
    st.sidebar.write(f"User: {st.session_state.email}")
    if st.sidebar.button("Logout"):
        logout_user()
        st.rerun()

    st.title("🏈 FoxEdge Sports Betting Insights")
    league = st.sidebar.radio("Select League", ["NFL", "NBA", "NCAAB"])
    
    run_league_pipeline(league)
    
    st.sidebar.markdown("""
    ### About
    FoxEdge combines machine learning and sports analytics to deliver betting insights.
    """)

if __name__ == "__main__":
    main()
