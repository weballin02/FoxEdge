import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
from pathlib import Path
import requests
import firebase_admin
from firebase_admin import credentials, auth

# NBA API and NFL data imports
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2, TeamGameLog
from nba_api.stats.static import teams as nba_teams
import nfl_data_py as nfl

# cbbpy for NCAAB data scraping
import cbbpy.mens_scraper as cbb

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
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Invalid credentials.")
            return None
    except Exception as e:
        st.error(f"Error during login: {e}")
        return None

def signup_user(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        st.success(f"User {email} created successfully!")
        return user
    except Exception as e:
        st.error(f"Error: {e}")

def logout_user():
    for key in ['email', 'logged_in']:
        if key in st.session_state:
            del st.session_state[key]
################################################################################
# ENHANCED DATA PREPROCESSING FUNCTIONS
################################################################################

def preprocess_data_with_knn(data, n_neighbors=5):
    """
    Preprocesses data using KNN imputation for missing values.
    """
    numeric_cols = ['off_rating', 'def_rating', 'pace']
    imputer = KNNImputer(n_neighbors=n_neighbors)
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    return data

def add_contextual_factors(data):
    """
    Adds contextual factors like back-to-back games or injuries.
    """
    # Example: Adding a "back_to_back" column for NBA games
    data['back_to_back'] = data.groupby('team')['gameday'].diff().dt.days <= 1
    return data

def round_half(number):
    """
    Rounds a number to the nearest 0.5 increment.
    """
    return round(number * 2) / 2

################################################################################
# CSV MANAGEMENT FUNCTIONS (UNCHANGED)
################################################################################

CSV_FILE = "predictions.csv"

def initialize_csv(csv_file=CSV_FILE):
    """Initialize the CSV file if it doesn't exist."""
    if not Path(csv_file).exists():
        columns = [
            "date", "league", "home_team", "away_team", "home_pred", "away_pred",
            "predicted_winner", "total", "spread_suggestion", "ou_suggestion"
        ]
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

def save_predictions_to_csv(predictions, csv_file=CSV_FILE):
    """Save predictions to a CSV file."""
    df = pd.DataFrame(predictions)
    df.to_csv(csv_file, mode='a', index=False, header=not Path(csv_file).exists())
    st.success("Predictions have been saved to CSV!")
################################################################################
# MODEL TRAINING & PREDICTION FUNCTIONS (IMPROVED)
################################################################################

@st.cache_data(ttl=14400)
def train_team_models(team_data: pd.DataFrame):
    """
    Trains GradientBoostingRegressor + ARIMA for each team's 'score'.
    Returns: gbr_models, arima_models, team_stats.
    """
    gbr_models = {}
    arima_models = {}
    team_stats = {}

    all_teams = team_data['team'].unique()
    
    for team in all_teams:
        df_team = team_data[team_data['team'] == team].copy()
        df_team.sort_values('gameday', inplace=True)
        scores = df_team['score'].reset_index(drop=True)

        if len(scores) < 3:
            continue

        # Compute basic statistics for the team.
        team_stats[team] = {
            'mean': round_half(scores.mean()),
            'std': round_half(scores.std()),
            'max': round_half(scores.max()),
            'recent_form': round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
        }

        # Train GBR model if enough data is available.
        if len(scores) >= 10:
            X_train = np.arange(len(scores)).reshape(-1, 1)
            y_train = scores.values
            gbr_model = GradientBoostingRegressor(random_state=42)
            gbr_model.fit(X_train, y_train)
            gbr_models[team] = gbr_model

        # Train ARIMA model if enough data is available.
        if len(scores) >= 7:
            arima_model = auto_arima(
                scores,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                max_p=3,
                max_q=3,
            )
            arima_models[team] = arima_model

    return gbr_models, arima_models, team_stats


def weighted_ensemble_prediction(gbr_pred, arima_pred, gbr_mae, arima_mae):
    """
    Combines GBR and ARIMA predictions using inverse MAE as weights.
    """
    if gbr_pred is None and arima_pred is None:
        return None
    if gbr_pred is None:
        return arima_pred
    if arima_pred is None:
        return gbr_pred

    # Calculate weights based on inverse MAE (lower MAE gets higher weight).
    gbr_weight = 1 / gbr_mae if gbr_mae > 0 else 0.5
    arima_weight = 1 / arima_mae if arima_mae > 0 else 0.5

    total_weight = gbr_weight + arima_weight
    weighted_avg = (gbr_pred * gbr_weight + arima_pred * arima_weight) / total_weight

    return weighted_avg


def predict_team_score(team, gbr_models, arima_models, team_stats, team_data, gbr_mae, arima_mae):
    """Predict a team's next-game score by blending ARIMA & GBR outputs."""
    
    if team not in team_stats:
        return None, (None, None)

    df_team = team_data[team_data['team'] == team]
    
    # Predict using GBR model if available.
    gbr_pred = None
    if team in gbr_models:
        X_next = np.array([[len(df_team)]])
        gbr_pred = gbr_models[team].predict(X_next)[0]

    # Predict using ARIMA model if available.
    arima_pred = None
    if team in arima_models:
        forecast = arima_models[team].predict(n_periods=1)
        arima_pred = forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast.iloc[0]

    # Use weighted ensemble prediction logic here.
    ensemble_score = weighted_ensemble_prediction(gbr_pred, arima_pred, gbr_mae, arima_mae)

    mu = team_stats[team]['mean']
    sigma = team_stats[team]['std']
    
    conf_low = round_half(mu - 1.96 * sigma)
    conf_high = round_half(mu + 1.96 * sigma)

################################################################################
# CONFIDENCE METRICS AND BACKTESTING
################################################################################

def calculate_confidence(home_pred, away_pred, home_std, away_std):
    """
    Calculates confidence levels with model uncertainty included.
    """
    if home_pred is None or away_pred is None:
        return None

    diff = home_pred - away_pred
    combined_std = max(1.0, (home_std + away_std) / 2)

    # Spread confidence: Based on score difference and standard deviation
    spread_confidence = round(min(99, max(1, 50 + abs(diff) / combined_std * 15)), 2)

    # Total confidence: Based on variance in predictions (lower variance = higher confidence)
    total_variance = home_std**2 + away_std**2
    total_confidence = round(min(99, max(1, 50 + (1 / max(1, total_variance)) * 15)), 2)

    return spread_confidence, total_confidence


def evaluate_models(team_data):
    """
    Evaluates GBR and ARIMA models using historical data.
    Returns MAE and RMSE for both models.
    """
    all_teams = team_data['team'].unique()
    
    gbr_errors, arima_errors = [], []
    
    for team in all_teams:
        df_team = team_data[team_data['team'] == team].copy()
        df_team.sort_values('gameday', inplace=True)
        scores = df_team['score'].reset_index(drop=True)

        if len(scores) < 10:
            continue
        
        # GBR Evaluation
        X_train = np.arange(len(scores) - 1).reshape(-1, 1)
        y_train = scores[:-1]
        X_test = np.array([[len(scores) - 1]])
        y_test = scores[-1]

        gbr_model = GradientBoostingRegressor(random_state=42)
        gbr_model.fit(X_train, y_train)
        gbr_pred = gbr_model.predict(X_test)[0]
        
        gbr_errors.append(abs(gbr_pred - y_test))

        # ARIMA Evaluation
        try:
            arima_model = auto_arima(
                scores[:-1],
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                max_p=3,
                max_q=3,
            )
            arima_pred = arima_model.predict(n_periods=1)[0]
            arima_errors.append(abs(arima_pred - y_test))
        except Exception:
            continue
    
    # Calculate MAE and RMSE for both models
    gbr_mae = np.mean(gbr_errors) if gbr_errors else float('inf')
    arima_mae = np.mean(arima_errors) if arima_errors else float('inf')
    
    return gbr_mae, arima_mae
################################################################################
# MATCHUP EVALUATION FUNCTIONS
################################################################################

def evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats):
    """
    Compute predicted spread, total, and confidence for a single matchup.
    """
    if home_pred is None or away_pred is None:
        return None

    diff = home_pred - away_pred
    total_points = home_pred + away_pred

    home_std = team_stats.get(home_team, {}).get('std', 5)
    away_std = team_stats.get(away_team, {}).get('std', 5)
    
    # Calculate confidence metrics
    spread_confidence, total_confidence = calculate_confidence(home_pred, away_pred, home_std, away_std)

    winner = home_team if diff > 0 else away_team

    # Example threshold for NCAAB. Adjust if needed for NBA or NFL.
    ou_threshold = 145

    return {
        'predicted_winner': winner,
        'diff': round_half(diff),
        'total_points': round_half(total_points),
        'spread_confidence': spread_confidence,
        'total_confidence': total_confidence,
        'spread_suggestion': f"Lean {winner} by {round_half(diff):.1f}",
        'ou_suggestion': f"Take the {'Over' if total_points > ou_threshold else 'Under'} {round_half(total_points):.1f}"
    }

def find_top_bets(matchups, threshold=70.0):
    """
    Filters matchups to find top bets based on a confidence threshold.
    """
    df = pd.DataFrame(matchups)
    
    # Filter by spread confidence or total confidence
    df_top_spread = df[df['spread_confidence'] >= threshold]
    
    # Sort by highest confidence level first
    df_top_spread.sort_values('spread_confidence', ascending=False, inplace=True)
    
    return df_top_spread
################################################################################
# DATA LOADING FUNCTIONS FOR NFL, NBA, AND NCAAB
################################################################################

@st.cache_data(ttl=14400)
def load_nfl_schedule():
    """
    Loads NFL schedule data for the last 10 years using nfl_data_py.
    """
    current_year = datetime.now().year
    years = list(range(current_year - 10, current_year + 1))
    
    schedule = nfl.import_schedules(years)
    
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    
    if pd.api.types.is_datetime64tz_dtype(schedule['gameday']):
        schedule['gameday'] = schedule['gameday'].dt.tz_convert(None)
        
    return schedule


def preprocess_nfl_data(schedule):
    """
    Preprocesses NFL data into a format suitable for model training.
    """
    home_df = schedule[['gameday', 'home_team', 'home_score']].rename(
        columns={'home_team': 'team', 'home_score': 'score'}
    )
    away_df = schedule[['gameday', 'away_team', 'away_score']].rename(
        columns={'away_team': 'team', 'away_score': 'score'}
    )
    
    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=['score'], inplace=True)
    data.sort_values('gameday', inplace=True)
    
    return data


@st.cache_data(ttl=14400)
def load_nba_data():
    """
    Loads NBA team game logs for multiple seasons using nba_api.
    """
    nba_teams_list = nba_teams.get_teams()
    seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
    
    all_rows = []
    
    for season in seasons:
        for team in nba_teams_list:
            team_id = team['id']
            team_abbrev = team.get('abbreviation', str(team_id))
            
            try:
                gl = TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
                if gl.empty:
                    continue

                gl['GAME_DATE'] = pd.to_datetime(gl['GAME_DATE'])
                gl.sort_values('GAME_DATE', inplace=True)

                needed_cols = ['PTS', 'FGA', 'FTA', 'TOV', 'OREB', 'PTS_OPP']
                for col in needed_cols:
                    gl[col] = pd.to_numeric(gl.get(col, 0), errors='coerce').fillna(0)

                # Calculate possessions and ratings
                gl['TEAM_POSSESSIONS'] = gl['FGA'] + 0.44 * gl['FTA'] + gl['TOV'] - gl['OREB']
                gl['OFF_RATING'] = (gl['PTS'] / gl['TEAM_POSSESSIONS']) * 100
                gl['DEF_RATING'] = (gl['PTS_OPP'] / gl['TEAM_POSSESSIONS']) * 100
                gl['PACE'] = gl['TEAM_POSSESSIONS']

                for _, row in gl.iterrows():
                    all_rows.append({
                        'gameday': row['GAME_DATE'],
                        'team': team_abbrev,
                        'score': row['PTS'],
                        'off_rating': row.get('OFF_RATING'),
                        'def_rating': row.get('DEF_RATING'),
                        'pace': row.get('PACE')
                    })
            except Exception as e:
                print(f"Error processing {team_abbrev} for {season}: {e}")
                
    df = pd.DataFrame(all_rows)
    df.dropna(subset=['score'], inplace=True)
    df.sort_values('gameday', inplace=True)
    
    return preprocess_data_with_knn(df)


@st.cache_data(ttl=14400)
def load_ncaab_data_current_season(season=2025):
    """
    Loads NCAA basketball data for the current season using cbbpy.
    """
    info_df, _, _ = cbb.get_games_season(season=season, info=True, box=False, pbp=False)
    
    if info_df.empty:
        return pd.DataFrame()

    # Process home and away teams
    home_df = info_df.rename(columns={
        "home_team": "team",
        "home_score": "score",
        "game_day": "gameday"
    })[['gameday', 'team', 'score']]
    
    away_df = info_df.rename(columns={
        "away_team": "team",
        "away_score": "score",
        "game_day": "gameday"
    })[['gameday', 'team', 'score']]
    
    data = pd.concat([home_df, away_df], ignore_index=True)
    
    # Add rolling statistics
    data.sort_values(['team', 'gameday'], inplace=True)
    
    data['rolling_mean_3'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3).mean())
    
    return preprocess_data_with_knn(data)
################################################################################
# MAIN PIPELINE FOR LEAGUE ANALYSIS AND STREAMLIT UI INTEGRATION
################################################################################

def run_league_pipeline(league_choice):
    global results
    global team_stats_global

    st.header(f"{league_choice} Best Bets")

    if league_choice == "NFL":
        schedule = load_nfl_schedule()
        if schedule.empty:
            st.error("Unable to load NFL schedule.")
            return
        
        team_data = preprocess_nfl_data(schedule)
        upcoming_games = fetch_upcoming_nfl_games(schedule)

    elif league_choice == "NBA":
        team_data = load_nba_data()
        if team_data.empty:
            st.error("Unable to load NBA data.")
            return
        
        upcoming_games = fetch_upcoming_nba_games()

    elif league_choice == "NCAAB":
        team_data = load_ncaab_data_current_season()
        if team_data.empty:
            st.error("Unable to load NCAAB data.")
            return
        
        upcoming_games = fetch_upcoming_ncaab_games()

    else:
        st.error(f"League {league_choice} not supported.")
        return

    # Model training and prediction logic
    gbr_models, arima_models, team_stats_global = train_team_models(team_data)
    
    gbr_mae, arima_mae = evaluate_models(team_data)

    results.clear()
    
    for _, game in upcoming_games.iterrows():
        home_team, away_team = game['home_team'], game['away_team']
        
        home_pred, _ = predict_team_score(home_team, gbr_models, arima_models, team_stats_global, team_data, gbr_mae, arima_mae)
        away_pred, _ = predict_team_score(away_team, gbr_models, arima_models, team_stats_global, team_data, gbr_mae, arima_mae)

        outcome = evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats_global)

        if outcome:
            results.append({
                **outcome,
                'date': game['gameday'],
                'home_team': home_team,
                'away_team': away_team,
                'home_pred': home_pred,
                'away_pred': away_pred,
            })

################################################################################
# STREAMLIT MAIN
################################################################################
def main():
    st.set_page_config(
        page_title="FoxEdge Sports Betting Edge",
        page_icon="ð¦",
        layout="centered"
    )
    initialize_csv()

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        st.title("Login to FoxEdge Sports Betting Insights")

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                user_data = login_with_rest(email, password)
                if user_data:
                    st.session_state['logged_in'] = True
                    st.session_state['email'] = user_data['email']
                    st.success(f"Welcome, {user_data['email']}!")
                    st.rerun()
        with col2:
            if st.button("Sign Up"):
                signup_user(email, password)
        return
    else:
        st.sidebar.title("Account")
        st.sidebar.write(f"Logged in as: {st.session_state.get('email','Unknown')}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.rerun()

    st.title("ð¦ FoxEdge Sports Betting Insights")
    st.sidebar.header("Navigation")
    league_choice = st.sidebar.radio(
        "Select League",
        ["NFL", "NBA", "NCAAB"],
        help="Choose which league's games you'd like to analyze"
    )

    run_league_pipeline(league_choice)

    st.sidebar.markdown(
        "### About FoxEdge\n"
        "FoxEdge provides data-driven insights for NFL, NBA, and NCAAB games, helping bettors make informed decisions."
    )
    st.sidebar.markdown("#### Powered by AI & Statistical Analysis")

    if st.button("Save Predictions to CSV"):
        save_predictions_to_csv(results)

if __name__ == "__main__":
    main()
