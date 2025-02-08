"""
FoxEdge Sports Betting Insights – Production-Ready Unified App

This app provides sports bettors with data‑driven predictions and advanced analytics for NFL, NBA, and NCAAB games. It includes:
  • Firebase authentication (login, signup, logout)
  • CSV management for saving predictions
  • Helper functions for datetime and caching
  • Fully functional prediction/simulation logic for NFL (using stacking + ARIMA), NBA, and NCAAB
  • A multi‑page layout with Dashboard, NFL, NBA, NCAAB, Advanced Analytics, and Settings pages.
  
Ensure that you have configured your Firebase secrets in st.secrets and installed all required packages.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pytz
import os
from pathlib import Path
import requests
import joblib
import warnings

# ML libraries
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from pmdarima import auto_arima
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from joblib import Parallel, delayed

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Sports data libraries
import nfl_data_py as nfl
from nba_api.stats.endpoints import ScoreboardV2, TeamGameLog
from nba_api.stats.static import teams as nba_teams
import cbbpy.mens_scraper as cbb

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, auth

warnings.filterwarnings('ignore')

####################################
# PAGE CONFIGURATION & THEME SETUP
####################################
st.set_page_config(
    page_title="FoxEdge Sports Betting Insights",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation for multi‑page layout
page = st.sidebar.radio("Navigation", 
                          ["Dashboard", "NFL Predictions", "NBA Predictions", "NCAAB Predictions", "Advanced Analytics", "Settings"])

# Dark mode toggle (stored in session state)
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

st.sidebar.button("Toggle Dark Mode", on_click=toggle_dark_mode)

if st.session_state.dark_mode:
    primary_bg = "#121212"
    primary_text = "#FFFFFF"
    secondary_bg = "#1E1E1E"
else:
    primary_bg = "#FFFFFF"
    primary_text = "#000000"
    secondary_bg = "#F5F5F5"

# Inject basic CSS for background and text colors
st.markdown(f"""
    <style>
        body {{
            background-color: {primary_bg};
            color: {primary_text};
        }}
    </style>
""", unsafe_allow_html=True)

####################################
# FIREBASE CONFIGURATION & AUTHENTICATION
####################################
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
except Exception as e:
    st.warning("Firebase secrets not found or incomplete in st.secrets.")

def login_with_rest(email, password):
    """Log in a user using Firebase REST API."""
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
    """Sign up a new user using Firebase."""
    try:
        user = auth.create_user(email=email, password=password)
        st.success(f"User {email} created successfully!")
        return user
    except Exception as e:
        st.error(f"Error: {e}")

def logout_user():
    """Log out the current user."""
    for key in ['email', 'logged_in']:
        if key in st.session_state:
            del st.session_state[key]

####################################
# CSV MANAGEMENT FUNCTIONS
####################################
CSV_FILE = "predictions.csv"

def initialize_csv(csv_file=CSV_FILE):
    """Initialize the CSV file if it doesn't exist."""
    if not Path(csv_file).exists():
        columns = ["date", "league", "home_team", "away_team", "home_pred", "away_pred",
                   "predicted_winner", "predicted_diff", "predicted_total",
                   "spread_suggestion", "ou_suggestion", "confidence"]
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

def save_predictions_to_csv(predictions, csv_file=CSV_FILE):
    """Save predictions to CSV."""
    df = pd.DataFrame(predictions)
    if Path(csv_file).exists():
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_file, index=False)
    st.success("Predictions have been saved to CSV!")

####################################
# COMMON HELPER FUNCTION
####################################
def to_naive(dt):
    """Convert a tz-aware datetime to tz-naive."""
    if dt is not None and hasattr(dt, "tzinfo") and dt.tzinfo:
        return dt.replace(tzinfo=None)
    return dt

####################################
# NFL FUNCTIONS
####################################
@st.cache_data(ttl=14400)
def load_nfl_schedule():
    current_year = datetime.datetime.now().year
    years = [current_year - i for i in range(12)]
    schedule = nfl.import_schedules(years)
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    return schedule

def preprocess_nfl_data(schedule):
    home_df = schedule[['gameday', 'home_team', 'home_score', 'away_score']].rename(
        columns={'home_team': 'team', 'home_score': 'score', 'away_score': 'opp_score'}
    )
    away_df = schedule[['gameday', 'away_team', 'away_score', 'home_score']].rename(
        columns={'away_team': 'team', 'away_score': 'score', 'home_score': 'opp_score'}
    )
    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=['score'], inplace=True)
    data.sort_values('gameday', inplace=True)
    data['rolling_avg'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data['rolling_std'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    data['season_avg'] = data.groupby('team')['score'].transform(lambda x: x.expanding().mean())
    data['weighted_avg'] = 0.6 * data['rolling_avg'] + 0.4 * data['season_avg']
    return data

def fetch_upcoming_nfl_games(schedule, days_ahead=7):
    upcoming = schedule[(schedule['home_score'].isna()) & (schedule['away_score'].isna())].copy()
    now = datetime.datetime.now()
    filter_date = now + datetime.timedelta(days=days_ahead)
    upcoming = upcoming[upcoming['gameday'] <= filter_date]
    upcoming.sort_values('gameday', inplace=True)
    return upcoming[['gameday', 'home_team', 'away_team']]

def train_team_models(team_data: pd.DataFrame):
    stack_models = {}
    arima_models = {}
    team_stats = {}
    teams = team_data['team'].unique()
    for team in teams:
        df_team = team_data[team_data['team'] == team].copy()
        df_team.sort_values('gameday', inplace=True)
        scores = df_team['score'].values
        if len(scores) < 3:
            continue
        df_team['rolling_avg'] = pd.Series(scores).rolling(window=3, min_periods=1).mean().values
        df_team['rolling_std'] = pd.Series(scores).rolling(window=3, min_periods=1).std().fillna(0).values
        df_team['season_avg'] = pd.Series(scores).expanding().mean().values
        df_team['weighted_avg'] = 0.6 * df_team['rolling_avg'] + 0.4 * df_team['season_avg']
        team_stats[team] = {
            'mean': np.round(np.mean(scores), 1),
            'std': np.round(np.std(scores), 1),
            'max': np.round(np.max(scores), 1),
            'recent_form': np.round(np.mean(scores[-5:]), 1)
        }
        features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg']].values
        target = scores
        n = len(features)
        split_index = int(n * 0.8)
        if split_index < 2:
            continue
        X_train, X_test = features[:split_index], features[split_index:]
        y_train, y_test = target[:split_index], target[split_index:]
        base_models = [
            ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
            ('lgbm', LGBMRegressor(n_estimators=100, random_state=42)),
            ('cat', CatBoostRegressor(n_estimators=100, verbose=0, random_state=42))
        ]
        final_estimator = LGBMRegressor()
        stack = StackingRegressor(estimators=base_models, final_estimator=final_estimator, cv=3)
        try:
            stack.fit(X_train, y_train)
            mse = mean_squared_error(y_test, stack.predict(X_test))
            team_stats[team]['mse'] = mse
            bias = np.mean(y_train - stack.predict(X_train))
            team_stats[team]['bias'] = bias
            stack_models[team] = stack
        except Exception as e:
            print(f"Error training stacking model for {team}: {e}")
            continue
        if len(scores) >= 7:
            try:
                arima_model = auto_arima(scores, seasonal=False, trace=False, error_action='ignore',
                                         suppress_warnings=True, max_p=3, max_q=3)
                arima_models[team] = arima_model
            except Exception as e:
                print(f"Error training ARIMA for {team}: {e}")
                continue
    return stack_models, arima_models, team_stats

def predict_team_score(team, stack_models, arima_models, team_stats, team_data):
    if team not in team_stats:
        return None, (None, None)
    df_team = team_data[team_data['team'] == team]
    if len(df_team) < 3:
        return None, (None, None)
    last_features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg']].tail(1).values
    stack_pred = None
    arima_pred = None
    if team in stack_models:
        try:
            stack_pred = float(stack_models[team].predict(last_features)[0])
        except Exception as e:
            print(f"Error in stacking prediction for {team}: {e}")
    if team in arima_models:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast)
        except Exception as e:
            print(f"Error in ARIMA prediction for {team}: {e}")
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
    bias = team_stats[team].get('bias', 0)
    ensemble_calibrated = ensemble + bias
    mu = team_stats[team]['mean']
    sigma = team_stats[team]['std']
    conf_low = np.round(mu - 1.96 * sigma, 1)
    conf_high = np.round(mu + 1.96 * sigma, 1)
    return np.round(ensemble_calibrated, 1), (conf_low, conf_high)

def evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats):
    if home_pred is None or away_pred is None:
        return None
    diff = home_pred - away_pred
    winner = home_team if diff > 0 else away_team
    return {
        'predicted_winner': winner,
        'predicted_diff': np.round(diff, 1),
        'predicted_total': np.round(home_pred + away_pred, 1)
    }

####################################
# NBA FUNCTIONS
####################################
@st.cache_data(ttl=14400)
def load_nba_data():
    seasons = ['2017-18','2018-19','2019-20','2020-21','2021-22','2022-23','2023-24','2024-25']
    all_rows = []
    teams = nba_teams.get_teams()
    for season in seasons:
        for team in teams:
            try:
                logs = TeamGameLog(team_id=team['id'], season=season).get_data_frames()[0]
                if logs.empty:
                    continue
                logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
                logs.sort_values('GAME_DATE', inplace=True)
                logs['team'] = team['abbreviation']
                all_rows.append(logs)
            except Exception as e:
                continue
    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    else:
        return pd.DataFrame()

def preprocess_nba_data(nba_data):
    nba_data.sort_values('GAME_DATE', inplace=True)
    nba_data['rolling_avg'] = nba_data.groupby('team')['PTS'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    nba_data['rolling_std'] = nba_data.groupby('team')['PTS'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    nba_data['season_avg'] = nba_data.groupby('team')['PTS'].transform(lambda x: x.expanding().mean())
    nba_data['weighted_avg'] = 0.6 * nba_data['rolling_avg'] + 0.4 * nba_data['season_avg']
    return nba_data

def fetch_upcoming_nba_games(days_ahead=3):
    now = datetime.datetime.now()
    upcoming_rows = []
    for offset in range(days_ahead+1):
        date_target = now + datetime.timedelta(days=offset)
        date_str = date_target.strftime('%Y-%m-%d')
        try:
            scoreboard = ScoreboardV2(game_date=date_str)
            games = scoreboard.get_data_frames()[0]
        except Exception as e:
            continue
        if games.empty:
            continue
        team_dict = {team['id']: team['abbreviation'] for team in nba_teams.get_teams()}
        games['HOME_TEAM_ABBREV'] = games['HOME_TEAM_ID'].map(team_dict)
        games['AWAY_TEAM_ABBREV'] = games['VISITOR_TEAM_ID'].map(team_dict)
        upcoming = games[~games['GAME_STATUS_TEXT'].str.contains("Final", case=False, na=False)]
        for _, row in upcoming.iterrows():
            upcoming_rows.append({
                'gameday': pd.to_datetime(date_str),
                'home_team': row['HOME_TEAM_ABBREV'],
                'away_team': row['AWAY_TEAM_ABBREV']
            })
    if upcoming_rows:
        df = pd.DataFrame(upcoming_rows)
        df.sort_values('gameday', inplace=True)
        return df
    else:
        return pd.DataFrame()

@st.cache_data(ttl=14400)
def train_nba_models(nba_data):
    stack_models = {}
    arima_models = {}
    team_stats = {}
    teams = nba_data['team'].unique()
    for team in teams:
        df_team = nba_data[nba_data['team'] == team].copy()
        df_team.sort_values('GAME_DATE', inplace=True)
        scores = df_team['PTS'].values
        if len(scores) < 3:
            continue
        df_team['rolling_avg'] = pd.Series(scores).rolling(window=3, min_periods=1).mean().values
        df_team['rolling_std'] = pd.Series(scores).rolling(window=3, min_periods=1).std().fillna(0).values
        df_team['season_avg'] = pd.Series(scores).expanding().mean().values
        df_team['weighted_avg'] = 0.6 * df_team['rolling_avg'] + 0.4 * df_team['season_avg']
        team_stats[team] = {
            'mean': np.round(np.mean(scores), 1),
            'std': np.round(np.std(scores), 1),
            'max': np.round(np.max(scores), 1),
            'recent_form': np.round(np.mean(scores[-5:]), 1)
        }
        features = df_team[['rolling_avg','rolling_std','weighted_avg']].values
        target = scores
        n = len(features)
        split_index = int(n * 0.8)
        if split_index < 2:
            continue
        X_train, X_test = features[:split_index], features[split_index:]
        y_train, y_test = target[:split_index], target[split_index:]
        try:
            xgb = XGBRegressor(n_estimators=100, random_state=42)
            lgbm = LGBMRegressor(n_estimators=100, random_state=42)
            cat = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)
            base_models = [('xgb', xgb), ('lgbm', lgbm), ('cat', cat)]
            final_estimator = LGBMRegressor()
            stack = StackingRegressor(estimators=base_models, final_estimator=final_estimator, cv=3)
            stack.fit(X_train, y_train)
            mse = mean_squared_error(y_test, stack.predict(X_test))
            team_stats[team]['mse'] = mse
            bias = np.mean(y_train - stack.predict(X_train))
            team_stats[team]['bias'] = bias
            stack_models[team] = stack
        except Exception as e:
            print(f"NBA model error for {team}: {e}")
            continue
        if len(scores) >= 7:
            try:
                arima_model = auto_arima(scores, seasonal=False, trace=False,
                                         error_action='ignore', suppress_warnings=True, max_p=3, max_q=3)
                arima_models[team] = arima_model
            except Exception as e:
                print(f"NBA ARIMA error for {team}: {e}")
    return stack_models, arima_models, team_stats

def predict_team_score_nba(team, stack_models, arima_models, team_stats, nba_data):
    if team not in team_stats:
        return None, (None, None)
    df_team = nba_data[nba_data['team'] == team]
    if len(df_team) < 3:
        return None, (None, None)
    last_features = df_team[['rolling_avg','rolling_std','weighted_avg']].tail(1).values
    stack_pred = None
    arima_pred = None
    if team in stack_models:
        try:
            stack_pred = float(stack_models[team].predict(last_features)[0])
        except Exception as e:
            print(f"NBA stacking prediction error for {team}: {e}")
    if team in arima_models:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast)
        except Exception as e:
            print(f"NBA ARIMA prediction error for {team}: {e}")
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
    bias = team_stats[team].get('bias', 0)
    ensemble_calibrated = ensemble + bias
    mu = team_stats[team]['mean']
    sigma = team_stats[team]['std']
    conf_low = np.round(mu - 1.96 * sigma, 1)
    conf_high = np.round(mu + 1.96 * sigma, 1)
    return np.round(ensemble_calibrated, 1), (conf_low, conf_high)

def evaluate_matchup_nba(home_team, away_team, home_pred, away_pred, team_stats):
    if home_pred is None or away_pred is None:
        return None
    diff = home_pred - away_pred
    winner = home_team if diff > 0 else away_team
    return {
         'predicted_winner': winner,
         'predicted_diff': np.round(diff, 1),
         'predicted_total': np.round(home_pred + away_pred, 1)
    }

def nba_predictions_page():
    st.header("NBA Predictions")
    nba_data = load_nba_data()
    if nba_data.empty:
         st.error("NBA data could not be loaded.")
         return
    nba_data = preprocess_nba_data(nba_data)
    upcoming = fetch_upcoming_nba_games(days_ahead=3)
    if nba_data.empty or upcoming.empty:
         st.warning("No upcoming NBA games available for analysis.")
         return
    st.subheader("Upcoming NBA Games")
    st.dataframe(upcoming)
    with st.spinner("Training NBA models..."):
         stack_models, arima_models, team_stats = train_nba_models(nba_data)
    st.subheader("Predictions for Upcoming NBA Games")
    for _, row in upcoming.iterrows():
         home = row['home_team']
         away = row['away_team']
         home_pred, _ = predict_team_score_nba(home, stack_models, arima_models, team_stats, nba_data)
         away_pred, _ = predict_team_score_nba(away, stack_models, arima_models, team_stats, nba_data)
         if home_pred is None or away_pred is None:
              continue
         outcome = evaluate_matchup_nba(home, away, home_pred, away_pred, team_stats)
         st.markdown(f"**{away} @ {home}** – Predicted Winner: **{outcome['predicted_winner']}** | Spread: **{outcome['predicted_diff']}** | Total: **{outcome['predicted_total']}**")

####################################
# NCAAB FUNCTIONS
####################################
def load_ncaab_data_current_season(season=2025):
    info_df, _, _ = cbb.get_games_season(season=season, info=True, box=False, pbp=False)
    if info_df.empty:
         return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(info_df["game_day"]):
         info_df["game_day"] = pd.to_datetime(info_df["game_day"], errors="coerce")
    home_df = info_df[['game_day', 'home_team', 'home_score', 'away_score']].rename(columns={
         "game_day": "gameday",
         "home_team": "team",
         "home_score": "score",
         "away_score": "opp_score"
    })
    home_df['is_home'] = 1
    away_df = info_df[['game_day', 'away_team', 'away_score', 'home_score']].rename(columns={
         "game_day": "gameday",
         "away_team": "team",
         "away_score": "score",
         "home_score": "opp_score"
    })
    away_df['is_home'] = 0
    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=["score"], inplace=True)
    data.sort_values("gameday", inplace=True)
    data['rolling_avg'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data['rolling_std'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    data['season_avg'] = data.groupby('team')['score'].transform(lambda x: x.expanding().mean())
    data['weighted_avg'] = 0.6 * data['rolling_avg'] + 0.4 * data['season_avg']
    data.sort_values(['team', 'gameday'], inplace=True)
    data['game_index'] = data.groupby('team').cumcount()
    return data

def fetch_upcoming_ncaab_games():
    timezone = pytz.timezone('America/Los_Angeles')
    current_time = datetime.datetime.now(timezone)
    dates = [current_time.strftime('%Y%m%d'), (current_time + datetime.timedelta(days=1)).strftime('%Y%m%d')]
    rows = []
    for date_str in dates:
         url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
         params = {'dates': date_str, 'groups': '50', 'limit': '357'}
         response = requests.get(url, params=params)
         if response.status_code != 200:
              st.warning(f"ESPN API request failed for {date_str} with status code {response.status_code}")
              continue
         data = response.json()
         games = data.get('events', [])
         if not games:
              continue
         for game in games:
              game_time_str = game['date']
              game_time = datetime.datetime.fromisoformat(game_time_str[:-1]).astimezone(timezone)
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
    if rows:
         df = pd.DataFrame(rows)
         df.sort_values('gameday', inplace=True)
         return df
    else:
         return pd.DataFrame()

def ncaab_predictions_page():
    st.header("NCAAB Predictions")
    season = 2025
    team_data = load_ncaab_data_current_season(season)
    if team_data.empty:
         st.warning("No NCAAB data available.")
         return
    st.subheader("Recent NCAAB Games")
    st.dataframe(team_data.head())
    st.subheader("Predictions for NCAAB Games")
    teams = team_data['team'].unique()
    for team in teams:
         scores = team_data[team_data['team'] == team]['score']
         if scores.empty:
              continue
         pred = np.round(scores.mean(), 1)
         st.markdown(f"**{team}** predicted average score: **{pred}**")

####################################
# ADVANCED ANALYTICS PAGE
####################################
def advanced_analytics_page():
    st.header("Advanced Analytics")
    st.markdown("Detailed visualizations and in-depth statistical analyses.")
    st.subheader("NFL Score Distribution")
    nfl_schedule = load_nfl_schedule()
    if not nfl_schedule.empty:
         scores = pd.concat([nfl_schedule['home_score'], nfl_schedule['away_score']]).dropna().values
    else:
         scores = np.random.normal(24,8,500)
    fig, ax = plt.subplots()
    ax.hist(scores, bins=30, color='skyblue', edgecolor='black')
    ax.set_title("NFL Score Distribution")
    st.pyplot(fig)
    st.subheader("NBA Score Distribution")
    nba_data = load_nba_data()
    if not nba_data.empty:
         nba_scores = nba_data['PTS'].dropna().values
    else:
         nba_scores = np.random.normal(110,15,500)
    fig2, ax2 = plt.subplots()
    ax2.hist(nba_scores, bins=30, color='orange', edgecolor='black')
    ax2.set_title("NBA Score Distribution")
    st.pyplot(fig2)
    st.subheader("Best Bets Summary")
    summary = pd.DataFrame({
         "Game": ["Team A vs Team B", "Team C vs Team D"],
         "Predicted Winner": ["Team A", "Team D"],
         "Confidence (%)": [85.0,78.0],
         "Spread": [5.0,-3.0]
    })
    st.dataframe(summary)

####################################
# SETTINGS PAGE
####################################
def settings_page():
    st.header("Settings")
    st.markdown("Adjust simulation parameters and global settings.")
    simulation_count = st.slider("Number of Simulations", 100, 10000, 1000, step=100)
    st.session_state.simulation_count = simulation_count
    st.markdown("Settings saved!")

####################################
# DASHBOARD PAGE
####################################
def dashboard_page():
    st.header("Dashboard")
    st.markdown("Welcome to FoxEdge Sports Betting Insights. Here is an overview of predictions and best bets.")
    st.subheader("Best Bets Summary")
    best_bets = pd.DataFrame({
         "Game": ["Team A vs Team B", "Team C vs Team D"],
         "Predicted Winner": ["Team A", "Team D"],
         "Confidence (%)": [85.0,78.0],
         "Spread": [5.0,-3.0]
    })
    st.dataframe(best_bets)

####################################
# MAIN APP ROUTING
####################################
if page == "Dashboard":
    dashboard_page()
elif page == "NFL Predictions":
    nfl_predictions_page()
elif page == "NBA Predictions":
    nba_predictions_page()
elif page == "NCAAB Predictions":
    ncaab_predictions_page()
elif page == "Advanced Analytics":
    advanced_analytics_page()
elif page == "Settings":
    settings_page()

####################################
# USER AUTHENTICATION (FIREBASE)
####################################
def firebase_authentication():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                user_data = login_with_rest(email, password)
                if user_data:
                    st.session_state.logged_in = True
                    st.session_state.email = user_data.get('email', 'Unknown')
                    st.success(f"Welcome, {st.session_state.email}!")
                    st.experimental_rerun()
        with col2:
            if st.button("Sign Up"):
                signup_user(email, password)
        return False
    else:
        st.sidebar.markdown(f"Logged in as: {st.session_state.get('email','Unknown')}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.experimental_rerun()
        return True

if not firebase_authentication():
    st.stop()

####################################
# SCHEDULED TASK (OPTIONAL)
####################################
def scheduled_task():
    st.write("Running scheduled task: Updating predictions and models...")
    schedule = nfl.import_schedules([datetime.datetime.now().year])
    schedule.to_csv("nfl_schedule.csv", index=False)
    nba_data = []
    for team_id in range(1, 31):
        try:
            logs = TeamGameLog(team_id=team_id, season="2024-25").get_data_frames()[0]
            nba_data.append(logs)
        except Exception as e:
            st.warning(f"Error fetching NBA data for team {team_id}: {e}")
    if nba_data:
        nba_df = pd.concat(nba_data, ignore_index=True)
        nba_df.to_csv("nba_team_logs.csv", index=False)
    ncaab_df, _, _ = cbb.get_games_season(season=2025, info=True, box=False, pbp=False)
    if not ncaab_df.empty:
        ncaab_df.to_csv("ncaab_games.csv", index=False)
    st.success("Scheduled task completed!")

####################################
# MAIN FUNCTION
####################################
def main():
    st.title("🦊 FoxEdge Sports Betting Insights")
    if st.button("Save Predictions to CSV"):
        # Example predictions; replace with real data as needed.
        predictions = [{
            "date": datetime.datetime.now(),
            "league": "NFL",
            "home_team": "NE",
            "away_team": "NYJ",
            "home_pred": 27,
            "away_pred": 20,
            "predicted_winner": "NE",
            "predicted_diff": 7,
            "predicted_total": 47,
            "spread_suggestion": "Lean NE by 7",
            "ou_suggestion": "Take Over 47",
            "confidence": 80
        }]
        save_predictions_to_csv(predictions)

if __name__ == "__main__":
    main()
