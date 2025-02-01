import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import nfl_data_py as nfl
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2, TeamGameLog
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Attempt to import XGBoost, LightGBM, and CatBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    st.error("The 'xgboost' package is not installed. Please install it using 'pip install xgboost'.")
    st.stop()

try:
    from lightgbm import LGBMRegressor
except ImportError:
    st.error("The 'lightgbm' package is not installed. Please install it using 'pip install lightgbm'.")
    st.stop()

try:
    from catboost import CatBoostRegressor
except ImportError:
    st.error("The 'catboost' package is not installed. Please install it using 'pip install catboost'.")
    st.stop()

from pmdarima import auto_arima
from pathlib import Path
import requests
import firebase_admin
from firebase_admin import credentials, auth
from datetime import datetime
import joblib
import os

# cbbpy for NCAAB
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
# CSV MANAGEMENT
################################################################################
CSV_FILE = "predictions.csv"

def initialize_csv(csv_file=CSV_FILE):
    """Initialize the CSV file if it doesn't exist."""
    if not Path(csv_file).exists():
        columns = [
            "date", "league", "home_team", "away_team", "home_pred", "away_pred",
            "predicted_winner", "predicted_diff", "predicted_total",
            "spread_suggestion", "ou_suggestion", "confidence"
        ]
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

def save_predictions_to_csv(predictions, csv_file=CSV_FILE):
    """Save predictions to a CSV file."""
    df = pd.DataFrame(predictions)
    if Path(csv_file).exists():
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_file, index=False)
    st.success("Predictions have been saved to CSV!")

################################################################################
# UTILITY
################################################################################
def round_half(number):
    return round(number * 2) / 2

################################################################################
# MODEL TRAINING & PREDICTION (STACKING + AUTO-ARIMA HYBRID)
################################################################################
@st.cache_data(ttl=14400)
def train_team_models(team_data: pd.DataFrame):
    """
    Trains Stacking Regressor + Auto-ARIMA for each team's 'score'.
    Returns: stack_models, arima_models, team_stats
    """
    stack_models = {}
    arima_models = {}
    team_stats = {}

    all_teams = team_data['team'].unique()
    for team in all_teams:
        df_team = team_data[team_data['team'] == team].copy()
        df_team.sort_values('gameday', inplace=True)
        scores = df_team['score'].reset_index(drop=True)

        if len(scores) < 3:
            continue

        # Feature Engineering Enhancements
        df_team['rolling_avg'] = df_team['score'].rolling(window=3, min_periods=1).mean()
        df_team['rolling_std'] = df_team['score'].rolling(window=3, min_periods=1).std().fillna(0)
        df_team['season_avg'] = df_team['score'].expanding().mean()
        df_team['weighted_avg'] = (df_team['rolling_avg'] * 0.6) + (df_team['season_avg'] * 0.4)

        # Ensure team_stats contain scalar values
        team_stats[team] = {
            'mean': round_half(scores.mean()),
            'std': round_half(scores.std()),
            'max': round_half(scores.max()),
            'recent_form': round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
        }

        # Prepare features and target
        features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg']]
        features = features.fillna(0)
        X = features.values
        y = scores.values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define base models
        estimators = [
            ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
            ('lgbm', LGBMRegressor(n_estimators=100, random_state=42)),
            ('cat', CatBoostRegressor(n_estimators=100, verbose=0, random_state=42))
        ]

        # Initialize Stacking Regressor
        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=LGBMRegressor(),
            passthrough=False,
            cv=5
        )

        # Train Stacking Regressor
        try:
            stack.fit(X_train, y_train)
            preds = stack.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            print(f"Team: {team}, Stacking Regressor MSE: {mse}")
            stack_models[team] = stack
        except Exception as e:
            print(f"Error training Stacking Regressor for team {team}: {e}")
            continue

        # Train ARIMA if enough data
        if len(scores) >= 7:
            try:
                arima = auto_arima(
                    scores,
                    seasonal=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    max_p=3,
                    max_q=3
                )
                arima_models[team] = arima
            except Exception as e:
                print(f"Error training ARIMA for team {team}: {e}")
                continue

    return stack_models, arima_models, team_stats

def make_predictions(stack_model, arima_model, X):
    """
    Makes hybrid predictions by blending Stacking Regressor and Auto-ARIMA.
    """
    stack_pred = stack_model.predict(X)
    if arima_model:
        arima_pred = arima_model.predict(n_periods=len(X))
        hybrid_pred = (stack_pred + arima_pred) / 2
    else:
        hybrid_pred = stack_pred
    return hybrid_pred

def predict_team_score(team, stack_models, arima_models, team_stats, team_data):
    """Predict a team's next-game score by blending Stacking Regressor & ARIMA outputs."""
    if team not in team_stats:
        return None, (None, None)

    df_team = team_data[team_data['team'] == team]
    data_len = len(df_team)
    stack_pred = None
    arima_pred = None

    # Feature Engineering Enhancements
    if data_len < 3:
        return None, (None, None)
    last_features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg']].tail(1)
    X_next = last_features.values

    # Stacking Regressor
    if team in stack_models:
        try:
            stack_pred = stack_models[team].predict(X_next)[0]
            stack_pred = float(stack_pred)
        except Exception as e:
            print(f"Error predicting with Stacking Regressor for team {team}: {e}")
            stack_pred = None

    # ARIMA
    if team in arima_models:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast
            arima_pred = float(arima_pred)
        except Exception as e:
            print(f"Error predicting with ARIMA for team {team}: {e}")
            arima_pred = None

    # Hybrid Prediction
    if stack_pred is not None and arima_pred is not None:
        ensemble = (stack_pred + arima_pred) / 2
    elif stack_pred is not None:
        ensemble = stack_pred
    elif arima_pred is not None:
        ensemble = arima_pred
    else:
        ensemble = None

    if ensemble is None:
        return None, (None, None)

    mu = team_stats[team]['mean']
    sigma = team_stats[team]['std']

    # Ensure mu and sigma are scalars
    if isinstance(mu, (pd.Series, pd.DataFrame, np.ndarray)):
        mu = mu.item()
    if isinstance(sigma, (pd.Series, pd.DataFrame, np.ndarray)):
        sigma = sigma.item()

    conf_low = round_half(mu - 1.96 * sigma)
    conf_high = round_half(mu + 1.96 * sigma)

    return round_half(ensemble), (conf_low, conf_high)

def evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats):
    """Compute predicted spread, total, and confidence for a single matchup."""
    if home_pred is None or away_pred is None:
        return None

    diff = home_pred - away_pred
    total_points = home_pred + away_pred

    home_std = team_stats.get(home_team, {}).get('std', 5)
    away_std = team_stats.get(away_team, {}).get('std', 5)
    combined_std = max(1.0, (home_std + away_std) / 2)

    raw_conf = abs(diff) / combined_std

    # Ensure raw_conf is a scalar
    if isinstance(raw_conf, (pd.Series, pd.DataFrame, np.ndarray)):
        raw_conf = raw_conf.item()

    confidence = round(min(99, max(1, 50 + raw_conf * 15)), 2)
    winner = home_team if diff > 0 else away_team

    # Example threshold for NCAAB. Adjust if needed for NBA or NFL.
    ou_threshold = 145

    return {
        'predicted_winner': winner,
        'diff': round_half(diff),
        'total_points': round_half(total_points),
        'confidence': confidence,
        'spread_suggestion': f"Lean {winner} by {round_half(diff):.1f}",
        'ou_suggestion': f"Take the {'Over' if total_points > ou_threshold else 'Under'} {round_half(total_points):.1f}"
    }

def find_top_bets(matchups, threshold=70.0):
    df = pd.DataFrame(matchups)
    df_top = df[df['confidence'] >= threshold].copy()
    df_top.sort_values('confidence', ascending=False, inplace=True)
    return df_top

################################################################################
# NFL DATA LOADING
################################################################################
@st.cache_data(ttl=14400)
def load_nfl_schedule():
    current_year = datetime.now().year
    years = [current_year - i for i in range(12)]  # Last 12 years including current
    schedule = nfl.import_schedules(years)
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(schedule['gameday']):
        schedule['gameday'] = schedule['gameday'].dt.tz_convert(None)
    return schedule

def preprocess_nfl_data(schedule):
    home_df = schedule[['gameday', 'home_team', 'home_score']].rename(
        columns={'home_team': 'team', 'home_score': 'score'}
    )
    away_df = schedule[['gameday', 'away_team', 'away_score']].rename(
        columns={'away_team': 'team', 'away_score': 'score'}
    )
    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=['score'], inplace=True)
    data.sort_values('gameday', inplace=True)

    # Feature Engineering Enhancements
    data['rolling_avg'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data['rolling_std'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    
    # Correctly compute expanding mean using apply
    data['season_avg'] = data.groupby('team')['score'].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
    
    data['weighted_avg'] = (data['rolling_avg'] * 0.6) + (data['season_avg'] * 0.4)

    return data

def fetch_upcoming_nfl_games(schedule, days_ahead=7):
    upcoming = schedule[
        schedule['home_score'].isna() & schedule['away_score'].isna()
    ].copy()
    now = datetime.now()
    filter_date = now + timedelta(days=days_ahead)
    upcoming = upcoming[upcoming['gameday'] <= filter_date].copy()
    upcoming.sort_values('gameday', inplace=True)
    return upcoming[['gameday', 'home_team', 'away_team']]

################################################################################
# NBA DATA LOADING (ADVANCED LOGIC IMPLEMENTED)
################################################################################
@st.cache_data(ttl=14400)
def load_nba_data():
    """Load multi-season team logs with pace & efficiency integrated."""
    nba_teams_list = nba_teams.get_teams()
    seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']  # Adjust as needed
    all_rows = []

    for season in seasons:
        for team in nba_teams_list:
            team_id = team['id']
            team_abbrev = team.get('abbreviation', str(team_id))  # Get team abbreviation from teams list
            
            try:
                gl = TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
                if gl.empty:
                    continue

                gl['GAME_DATE'] = pd.to_datetime(gl['GAME_DATE'])
                gl.sort_values('GAME_DATE', inplace=True)

                # Convert needed columns to numeric
                needed = ['PTS', 'FGA', 'FTA', 'TOV', 'OREB', 'PTS_OPP']
                for c in needed:
                    if c not in gl.columns:
                        gl[c] = 0
                    gl[c] = pd.to_numeric(gl[c], errors='coerce').fillna(0)

                # Approx possessions
                gl['TEAM_POSSESSIONS'] = gl['FGA'] + 0.44 * gl['FTA'] + gl['TOV'] - gl['OREB']
                gl['TEAM_POSSESSIONS'] = gl['TEAM_POSSESSIONS'].apply(lambda x: x if x > 0 else np.nan)

                # Offensive Rating
                gl['OFF_RATING'] = np.where(
                    gl['TEAM_POSSESSIONS'] > 0,
                    (gl['PTS'] / gl['TEAM_POSSESSIONS']) * 100,
                    np.nan
                )

                # Defensive Rating
                gl['DEF_RATING'] = np.where(
                    gl['TEAM_POSSESSIONS'] > 0,
                    (gl['PTS_OPP'] / gl['TEAM_POSSESSIONS']) * 100,
                    np.nan
                )

                # Approx Pace = TEAM_POSSESSIONS (assuming opponent possessions ~ same)
                gl['PACE'] = gl['TEAM_POSSESSIONS']

                # Feature Engineering Enhancements
                gl['rolling_avg'] = gl['PTS'].rolling(window=3, min_periods=1).mean()
                gl['rolling_std'] = gl['PTS'].rolling(window=3, min_periods=1).std().fillna(0)
                gl['season_avg'] = gl['PTS'].expanding().mean()
                gl['weighted_avg'] = (gl['rolling_avg'] * 0.6) + (gl['season_avg'] * 0.4)

                # We'll keep a final 'score' for training, which is the team's points
                for idx, row_ in gl.iterrows():
                    try:
                        all_rows.append({
                            'gameday': row_['GAME_DATE'],
                            'team': team_abbrev,  # Use team abbreviation from teams list
                            'score': float(row_['PTS']),
                            'off_rating': row_['OFF_RATING'] if pd.notnull(row_['OFF_RATING']) else np.nan,
                            'def_rating': row_['DEF_RATING'] if pd.notnull(row_['DEF_RATING']) else np.nan,
                            'pace': row_['PACE'] if pd.notnull(row_['PACE']) else np.nan,
                            'rolling_avg': row_['rolling_avg'],
                            'rolling_std': row_['rolling_std'],
                            'season_avg': row_['season_avg'],
                            'weighted_avg': row_['weighted_avg']
                        })
                    except Exception as e:
                        print(f"Error processing row for team {team_abbrev}: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing team {team_abbrev} for season {season}: {str(e)}")
                continue

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.dropna(subset=['score'], inplace=True)
    df.sort_values('gameday', inplace=True)

    # Optional: fill missing advanced stats with league means
    for col in ['off_rating', 'def_rating', 'pace']:
        df[col].fillna(df[col].mean(), inplace=True)

    return df

def fetch_upcoming_nba_games(days_ahead=3):
    now = datetime.now()
    upcoming_rows = []
    for offset in range(days_ahead + 1):
        date_target = now + timedelta(days=offset)
        date_str = date_target.strftime('%Y-%m-%d')
        scoreboard = ScoreboardV2(game_date=date_str)
        games = scoreboard.get_data_frames()[0]
        if games.empty:
            continue

        nba_team_dict = {tm['id']: tm['abbreviation'] for tm in nba_teams.get_teams()}
        games['HOME_TEAM_ABBREV'] = games['HOME_TEAM_ID'].map(nba_team_dict)
        games['AWAY_TEAM_ABBREV'] = games['VISITOR_TEAM_ID'].map(nba_team_dict)
        upcoming_df = games[~games['GAME_STATUS_TEXT'].str.contains("Final", case=False, na=False)]

        for _, g in upcoming_df.iterrows():
            upcoming_rows.append({
                'gameday': pd.to_datetime(date_str),
                'home_team': g['HOME_TEAM_ABBREV'],
                'away_team': g['AWAY_TEAM_ABBREV']
            })
    if not upcoming_rows:
        return pd.DataFrame()
    upcoming = pd.DataFrame(upcoming_rows)
    upcoming.sort_values('gameday', inplace=True)
    return upcoming

########################################
# NCAAB HISTORICAL LOADER
########################################
@st.cache_data(ttl=14400)
def load_ncaab_data_current_season(season=2025):
    """
    Loads finished or in-progress NCAA MBB games for the given season
    using cbbpy. Adds is_home=1 for home team, is_home=0 for away.
    """
    info_df, _, _ = cbb.get_games_season(season=season, info=True, box=False, pbp=False)
    if info_df.empty:
        return pd.DataFrame()

    # Convert "game_day" to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(info_df["game_day"]):
        info_df["game_day"] = pd.to_datetime(info_df["game_day"], errors="coerce")

    home_df = info_df.rename(columns={
        "home_team": "team",
        "home_score": "score",
        "game_day": "gameday"
    })[["gameday", "team", "score"]]
    home_df['is_home'] = 1

    away_df = info_df.rename(columns={
        "away_team": "team",
        "away_score": "score",
        "game_day": "gameday"
    })[["gameday", "team", "score"]]
    away_df['is_home'] = 0

    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=["score"], inplace=True)
    data.sort_values("gameday", inplace=True)

    # Feature Engineering Enhancements
    data['rolling_avg'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data['rolling_std'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    
    # Correctly compute expanding mean using apply
    data['season_avg'] = data.groupby('team')['score'].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
    
    data['weighted_avg'] = (data['rolling_avg'] * 0.6) + (data['season_avg'] * 0.4)

    data.sort_values(['team', 'gameday'], inplace=True)
    data['game_index'] = data.groupby('team').cumcount()

    return data

########################################
# NCAAB UPCOMING: ESPN method (UPDATED)
########################################
def fetch_upcoming_ncaab_games() -> pd.DataFrame:
    """
    Fetches upcoming NCAAB games for 'today' and 'tomorrow' using ESPN's scoreboard API.
    """
    timezone = pytz.timezone('America/Los_Angeles')
    current_time = datetime.now(timezone)

    # Get current day and next day
    dates = [
        current_time.strftime('%Y%m%d'),  # Today
        (current_time + timedelta(days=1)).strftime('%Y%m%d')  # Tomorrow
    ]

    rows = []
    for date_str in dates:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {
            'dates': date_str,
            'groups': '50',   # D1 men's
            'limit': '357'
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            st.warning(f"ESPN API request failed for date {date_str} with status code {response.status_code}")
            continue

        data = response.json()
        games = data.get('events', [])
        if not games:
            st.info(f"No upcoming NCAAB games for {date_str}.")
            continue

        for game in games:
            game_time_str = game['date']  # ISO8601
            game_time = datetime.fromisoformat(game_time_str[:-1]).astimezone(timezone)

            competitors = game['competitions'][0]['competitors']
            home_comp = next((c for c in competitors if c['homeAway'] == 'home'), None)
            away_comp = next((c for c in competitors if c['homeAway'] == 'away'), None)

            if not home_comp or not away_comp:
                continue

            home_team = home_comp['team']['displayName']
            away_team = away_comp['team']['displayName']

            rows.append({
                'gameday': game_time,
                'home_team': home_team,
                'away_team': away_team
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values('gameday', inplace=True)
    return df

################################################################################
# UI COMPONENTS
################################################################################
def generate_writeup(bet, team_stats_global):
    home_team = bet['home_team']
    away_team = bet['away_team']
    home_pred = bet['home_pred']
    away_pred = bet['away_pred']
    predicted_winner = bet['predicted_winner']
    confidence = bet['confidence']

    home_stats = team_stats_global.get(home_team, {})
    away_stats = team_stats_global.get(away_team, {})

    home_mean = home_stats.get('mean', 'N/A')
    home_std = home_stats.get('std', 'N/A')
    home_recent = home_stats.get('recent_form', 'N/A')
    away_mean = away_stats.get('mean', 'N/A')
    away_std = away_stats.get('std', 'N/A')
    away_recent = away_stats.get('recent_form', 'N/A')

    writeup = f"""
**Detailed Analysis:**

- **{home_team} Performance:**
  - **Average Score:** {home_mean}
  - **Score Standard Deviation:** {home_std}
  - **Recent Form (Last 5 Games):** {home_recent}

- **{away_team} Performance:**
  - **Average Score:** {away_mean}
  - **Score Standard Deviation:** {away_std}
  - **Recent Form (Last 5 Games):** {away_recent}

- **Prediction Insight:**
  Based on the recent performance and statistical analysis, **{predicted_winner}** is predicted to win with a confidence level of **{confidence}%.** 
  The projected score difference is **{bet['predicted_diff']} points**, leading to a suggested spread of **{bet['spread_suggestion']}**. 
  Additionally, the total predicted points for the game are **{bet['predicted_total']}**, indicating a suggestion to **{bet['ou_suggestion']}**.

- **Statistical Edge:**
  The confidence level of **{confidence}%** reflects the statistical edge derived from the combined performance metrics of both teams.
  This ensures that the prediction is data-driven and reliable.
"""
    return writeup

def display_bet_card(bet, team_stats_global):
    with st.container():
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.markdown(f"### **{bet['away_team']} @ {bet['home_team']}**")
            date_obj = bet['date']
            if isinstance(date_obj, datetime):
                st.caption(date_obj.strftime("%A, %B %d - %I:%M %p"))

        with col2:
            if bet['confidence'] >= 80:
                st.markdown("🔥 **High-Confidence Bet** 🔥")
            st.markdown(f"**Spread Suggestion:** {bet['spread_suggestion']}")
            st.markdown(f"**Total Suggestion:** {bet['ou_suggestion']}")

        with col3:
            st.metric(label="Confidence", value=f"{bet['confidence']:.1f}%")

    with st.expander("Detailed Insights", expanded=False):
        st.markdown(f"**Predicted Winner:** {bet['predicted_winner']}")
        st.markdown(f"**Predicted Total Points:** {bet['predicted_total']}")
        st.markdown(f"**Prediction Margin (Diff):** {bet['predicted_diff']}")

    with st.expander("Game Analysis", expanded=False):
        writeup = generate_writeup(bet, team_stats_global)
        st.markdown(writeup)

################################################################################
# GLOBALS
################################################################################
results = []
team_stats_global = {}

################################################################################
# MAIN PIPELINE
################################################################################
def run_league_pipeline(league_choice):
    global results
    global team_stats_global

    st.header(f"Today's {league_choice} Best Bets 🎯")

    if league_choice == "NFL":
        schedule = load_nfl_schedule()
        if schedule.empty:
            st.error("Unable to load NFL schedule.")
            return
        team_data = preprocess_nfl_data(schedule)
        upcoming = fetch_upcoming_nfl_games(schedule, days_ahead=7)

    elif league_choice == "NBA":
        team_data = load_nba_data()
        if team_data.empty:
            st.error("Unable to load NBA data.")
            return
        upcoming = fetch_upcoming_nba_games(days_ahead=3)

    else:  # NCAAB
        team_data = load_ncaab_data_current_season(season=2025)
        if team_data.empty:
            st.error("Unable to load NCAAB data.")
            return
        upcoming = fetch_upcoming_ncaab_games()

    if team_data.empty or upcoming.empty:
        st.warning(f"No upcoming {league_choice} data available for analysis.")
        return

    with st.spinner("Analyzing recent performance data..."):
        stack_models, arima_models, team_stats = train_team_models(team_data)
        team_stats_global = team_stats
        results.clear()

        for _, row in upcoming.iterrows():
            home, away = row['home_team'], row['away_team']
            home_pred, _ = predict_team_score(home, stack_models, arima_models, team_stats, team_data)
            away_pred, _ = predict_team_score(away, stack_models, arima_models, team_stats, team_data)

            outcome = evaluate_matchup(home, away, home_pred, away_pred, team_stats)
            if outcome:
                results.append({
                    'date': row['gameday'],
                    'league': league_choice,
                    'home_team': home,
                    'away_team': away,
                    'home_pred': home_pred,
                    'away_pred': away_pred,
                    'predicted_winner': outcome['predicted_winner'],
                    'predicted_diff': outcome['diff'],
                    'predicted_total': outcome['total_points'],
                    'confidence': outcome['confidence'],
                    'spread_suggestion': outcome['spread_suggestion'],
                    'ou_suggestion': outcome['ou_suggestion']
                })

    view_mode = st.radio("View Mode", ["🎯 Top Bets Only", "📊 All Games"], horizontal=True)
    if view_mode == "🎯 Top Bets Only":
        conf_threshold = st.slider(
            "Minimum Confidence Level",
            min_value=50.0,
            max_value=99.0,
            value=75.0,
            step=5.0,
            help="Only show bets with confidence level above this threshold"
        )
        top_bets = find_top_bets(results, threshold=conf_threshold)
        if not top_bets.empty:
            st.markdown(f"### 🔥 Top {len(top_bets)} Bets for Today")
            for _, bet in top_bets.iterrows():
                display_bet_card(bet, team_stats_global)
        else:
            st.info("No high-confidence bets found. Try lowering the threshold.")
    else:
        if results:
            st.markdown("### 📊 All Games Analysis")
            for bet in results:
                display_bet_card(bet, team_stats_global)
        else:
            st.info(f"No upcoming {league_choice} games found.")

################################################################################
# STREAMLIT MAIN FUNCTION & SCHEDULING IMPLEMENTATION
################################################################################

import streamlit as st

# Function that should run when scheduled
def scheduled_task():
    """
    Function to execute scheduled tasks such as fetching new data, 
    updating predictions, and refreshing stored models for FoxEdge.
    """
    st.write("🕒 Scheduled task running: Fetching and updating predictions...")

    # Load and preprocess new data for each league
    st.write("📡 Fetching latest NFL schedule and results...")
    schedule = nfl.import_schedules([datetime.now().year])
    schedule.to_csv("nfl_schedule.csv", index=False)

    st.write("🏀 Fetching latest NBA team game logs...")
    nba_data = []
    for team_id in range(1, 31):  # NBA has 30 teams
        try:
            logs = TeamGameLog(team_id=team_id, season="2024-25").get_data_frames()[0]
            nba_data.append(logs)
        except Exception as e:
            st.warning(f"Error fetching data for NBA team {team_id}: {e}")
    
    if nba_data:
        nba_df = pd.concat(nba_data, ignore_index=True)
        nba_df.to_csv("nba_team_logs.csv", index=False)

    st.write("🏀 Fetching latest NCAAB data...")
    ncaab_df, _, _ = get_games_season(season=2025, info=True, box=False, pbp=False)
    if not ncaab_df.empty:
        ncaab_df.to_csv("ncaab_games.csv", index=False)

    # Train or update models
    st.write("🤖 Updating prediction models...")
    if os.path.exists("nfl_schedule.csv"):
        joblib.dump(schedule, "models/nfl_model.pkl")
    
    if os.path.exists("nba_team_logs.csv"):
        joblib.dump(nba_df, "models/nba_model.pkl")
    
    if os.path.exists("ncaab_games.csv"):
        joblib.dump(ncaab_df, "models/ncaab_model.pkl")

    st.success("✅ Scheduled task completed successfully!")

    st.success("Scheduled task completed successfully.")

# Streamlit UI and main logic
def main():
    """
    Main Streamlit function for interactive user interface.
    """
    st.set_page_config(
        page_title="FoxEdge Sports Betting Edge",
        page_icon="🦊",
        layout="centered"
    )

    st.title("🦊 FoxEdge Sports Betting Insights")
    
    # User login logic
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
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
                    st.experimental_rerun()
        with col2:
            if st.button("Sign Up"):
                signup_user(email, password)
        return
    else:
        st.sidebar.title("Account")
        st.sidebar.write(f"Logged in as: {st.session_state.get('email','Unknown')}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.experimental_rerun()

    # Sidebar Navigation
    st.sidebar.header("Navigation")
    league_choice = st.sidebar.radio(
        "Select League",
        ["NFL", "NBA", "NCAAB"],
        help="Choose which league's games you'd like to analyze"
    )

    # Run the selected league's pipeline
    run_league_pipeline(league_choice)

    st.sidebar.markdown(
        "### About FoxEdge\n"
        "FoxEdge provides data-driven insights for NFL, NBA, and NCAAB games, helping bettors make informed decisions."
    )

    # CSV Output Enhancements
    if st.button("Save Predictions to CSV"):
        if results:
            save_predictions_to_csv(results)
            csv = pd.DataFrame(results).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )
        else:
            st.warning("No predictions to save.")

# Check if this is a scheduled run triggered by GitHub Actions
if __name__ == "__main__":
    query_params = st.experimental_get_query_params()

    if "trigger" in query_params:
        # Run scheduled task if triggered by GitHub Actions
        scheduled_task()
        st.write("Task triggered successfully.")
    else:
        # Run the app normally for users
        main()
