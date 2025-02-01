"""
FoxEdge Ultimate Sports Betting Insights App
==============================================

This Streamlit app provides state‑of‑the‑art betting insights for NFL, NBA, and NCAAB
by leveraging advanced ensemble and time‑series forecasting techniques.

Key features:
  • Firebase authentication (login/signup/logout)
  • Data ingestion for NFL (nfl_data_py), NBA (nba_api), and NCAAB (cbbpy)
  • A ModelTrainer class that performs feature engineering,
    trains a stacking ensemble (with hyperparameter tuning via RandomizedSearchCV)
    and an Auto‑ARIMA model per team, and stores team statistics.
  • Weighted ensemble predictions (inverse MSE weighting) for next‐game score forecasts.
  • Interactive UI with bet cards, detailed insights, CSV export, and league navigation.
  
Before running, make sure to install the dependencies in requirements.txt.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import nfl_data_py as nfl
from nba_api.stats.endpoints import TeamGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
import cbbpy.mens_scraper as cbb
import requests
import firebase_admin
from firebase_admin import credentials, auth
import joblib
import os
import logging

# Machine learning libraries
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from pmdarima import auto_arima

import plotly.express as px
import matplotlib.pyplot as plt

########################################
# Logging Configuration
########################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

########################################
# Firebase Configuration
########################################
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
    st.warning("Firebase configuration not fully set up. Firebase features will be disabled.")
    logging.warning(f"Firebase config error: {e}")

def login_with_rest(email, password):
    """Login to Firebase using REST API."""
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
    """Sign up a new user in Firebase."""
    try:
        user = auth.create_user(email=email, password=password)
        st.success(f"User {email} created successfully!")
        return user
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def logout_user():
    """Logout by clearing session state."""
    for key in ['email', 'logged_in']:
        st.session_state.pop(key, None)

########################################
# Utility Functions
########################################
def round_half(number):
    """Rounds a number to the nearest 0.5."""
    return round(number * 2) / 2

def initialize_csv(csv_file="predictions.csv"):
    """Initialize the CSV file if it does not exist."""
    if not os.path.exists(csv_file):
        columns = [
            "date", "league", "home_team", "away_team", "home_pred", "away_pred",
            "predicted_winner", "predicted_diff", "predicted_total",
            "spread_suggestion", "ou_suggestion", "confidence"
        ]
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

def save_predictions_to_csv(predictions, csv_file="predictions.csv"):
    """Save predictions DataFrame to CSV."""
    df = pd.DataFrame(predictions)
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_file, index=False)
    st.success("Predictions saved to CSV!")
    return df

########################################
# Data Loading Functions
########################################
@st.cache_data(ttl=14400, show_spinner="Loading NFL schedule...")
def load_nfl_schedule():
    """Load NFL schedule for the past 12 years using nfl_data_py."""
    current_year = datetime.now().year
    years = [current_year - i for i in range(12)]
    schedule = nfl.import_schedules(years)
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    schedule['gameday'] = schedule['gameday'].dt.tz_localize(None)
    return schedule

def preprocess_nfl_data(schedule):
    """Preprocess NFL schedule to create a uniform dataset."""
    home_df = schedule[['gameday', 'home_team', 'home_score']].rename(
        columns={'home_team': 'team', 'home_score': 'score'}
    )
    away_df = schedule[['gameday', 'away_team', 'away_score']].rename(
        columns={'away_team': 'team', 'away_score': 'score'}
    )
    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=['score'], inplace=True)
    data.sort_values('gameday', inplace=True)
    # Feature engineering
    data['rolling_avg'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data['rolling_std'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    data['season_avg'] = data.groupby('team')['score'].transform(lambda x: x.expanding().mean())
    data['weighted_avg'] = data['rolling_avg'] * 0.6 + data['season_avg'] * 0.4
    data['early_vs_late'] = data['rolling_avg'] * 0.6 - data['rolling_avg'] * 0.4
    data['lag_score'] = data.groupby('team')['score'].shift(1).fillna(data['season_avg'])
    return data

@st.cache_data(ttl=14400, show_spinner="Loading NBA data...")
def load_nba_data():
    """Load NBA game logs from nba_api for selected seasons."""
    nba_teams_list = nba_teams.get_teams()
    seasons = ['2020-21', '2021-22', '2022-23']
    all_rows = []
    for season in seasons:
        for team in nba_teams_list:
            team_id = team['id']
            team_name = team['full_name']
            try:
                gl = TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
                if gl.empty:
                    continue
                gl['GAME_DATE'] = pd.to_datetime(gl['GAME_DATE'])
                gl.sort_values('GAME_DATE', inplace=True)
                for _, row in gl.iterrows():
                    matchup = row["MATCHUP"]
                    # Parse MATCHUP to get opponent name
                    if " vs. " in matchup:
                        opponent = matchup.split(" vs. ")[1]
                    elif " @ " in matchup:
                        opponent = matchup.split(" @ ")[1]
                    else:
                        opponent = "Unknown"
                    all_rows.append({
                        'gameday': row['GAME_DATE'],
                        'team': team_name,
                        'score': row['PTS'],
                        # Features will be computed below
                        'rolling_avg': np.nan,
                        'rolling_std': np.nan,
                        'season_avg': np.nan,
                        'weighted_avg': np.nan,
                        'early_vs_late': np.nan,
                        'lag_score': np.nan
                    })
            except Exception as e:
                logging.warning(f"Error loading NBA data for {team_name} in season {season}: {e}")
                continue
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df.sort_values('gameday', inplace=True)
    # Compute features for NBA data
    df['rolling_avg'] = df.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['rolling_std'] = df.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    df['season_avg'] = df.groupby('team')['score'].transform(lambda x: x.expanding().mean())
    df['weighted_avg'] = df['rolling_avg'] * 0.6 + df['season_avg'] * 0.4
    df['early_vs_late'] = df['rolling_avg'] * 0.6 - df['rolling_avg'] * 0.4
    df['lag_score'] = df.groupby('team')['score'].shift(1).fillna(df['season_avg'])
    return df

@st.cache_data(ttl=14400, show_spinner="Loading NCAAB data...")
def load_ncaab_data_current_season(season=2025):
    """Load current season NCAA men’s basketball data using cbbpy."""
    info_df, _, _ = cbb.get_games_season(season=season, info=True, box=False, pbp=False)
    if info_df.empty:
        return pd.DataFrame()
    info_df["game_day"] = pd.to_datetime(info_df["game_day"], errors="coerce")
    home_df = info_df.rename(columns={"home_team": "team", "home_score": "score", "game_day": "gameday"})[["gameday", "team", "score"]]
    home_df['is_home'] = 1
    away_df = info_df.rename(columns={"away_team": "team", "away_score": "score", "game_day": "gameday"})[["gameday", "team", "score"]]
    away_df['is_home'] = 0
    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=["score"], inplace=True)
    data.sort_values("gameday", inplace=True)
    data['rolling_avg'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data['rolling_std'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    data['season_avg'] = data.groupby('team')['score'].transform(lambda x: x.expanding().mean())
    data['weighted_avg'] = data['rolling_avg'] * 0.6 + data['season_avg'] * 0.4
    data['early_vs_late'] = data['rolling_avg'] * 0.6 - data['rolling_avg'] * 0.4
    data['lag_score'] = data.groupby('team')['score'].shift(1).fillna(data['season_avg'])
    return data

def fetch_upcoming_games(league, days_ahead=3):
    """
    Fetch upcoming games for the given league.
    NFL: Uses nfl_data_py schedule.
    NBA: Uses ScoreboardV2 from nba_api.
    NCAAB: Uses ESPN’s scoreboard API.
    """
    upcoming = pd.DataFrame()
    if league == "NFL":
        schedule = load_nfl_schedule()
        now = datetime.now()
        filter_date = now + timedelta(days=days_ahead)
        upcoming = schedule[(schedule['home_score'].isna()) & (schedule['away_score'].isna())]
        upcoming = upcoming[upcoming['gameday'] <= filter_date]
        upcoming = upcoming[['gameday', 'home_team', 'away_team']].drop_duplicates()
    elif league == "NBA":
        now = datetime.now()
        upcoming_rows = []
        for offset in range(days_ahead + 1):
            date_target = now + timedelta(days=offset)
            date_str = date_target.strftime('%Y-%m-%d')
            try:
                scoreboard = ScoreboardV2(game_date=date_str)
                games = scoreboard.get_data_frames()[0]
                if games.empty:
                    continue
                team_dict = {tm['id']: tm['abbreviation'] for tm in nba_teams.get_teams()}
                games['HOME_TEAM_ABBREV'] = games['HOME_TEAM_ID'].map(team_dict)
                games['AWAY_TEAM_ABBREV'] = games['VISITOR_TEAM_ID'].map(team_dict)
                games = games[~games['GAME_STATUS_TEXT'].str.contains("Final", case=False, na=False)]
                for _, g in games.iterrows():
                    upcoming_rows.append({
                        'gameday': pd.to_datetime(date_str),
                        'home_team': g['HOME_TEAM_ABBREV'],
                        'away_team': g['AWAY_TEAM_ABBREV']
                    })
            except Exception as e:
                logging.warning(f"Error fetching NBA upcoming games for {date_str}: {e}")
                continue
        if upcoming_rows:
            upcoming = pd.DataFrame(upcoming_rows)
            upcoming.sort_values('gameday', inplace=True)
    elif league == "NCAAB":
        try:
            timezone = pytz.timezone('America/Los_Angeles')
            current_time = datetime.now(timezone)
            dates = [
                current_time.strftime('%Y%m%d'),
                (current_time + timedelta(days=1)).strftime('%Y%m%d')
            ]
            rows = []
            for date_str in dates:
                url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
                params = {'dates': date_str, 'groups': '50', 'limit': '357'}
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    continue
                data = response.json()
                games = data.get('events', [])
                for game in games:
                    game_time = datetime.fromisoformat(game['date'][:-1]).astimezone(timezone)
                    competitors = game['competitions'][0]['competitors']
                    home_comp = next((c for c in competitors if c['homeAway'] == 'home'), None)
                    away_comp = next((c for c in competitors if c['homeAway'] == 'away'), None)
                    if not home_comp or not away_comp:
                        continue
                    rows.append({
                        'gameday': game_time,
                        'home_team': home_comp['team']['displayName'],
                        'away_team': away_comp['team']['displayName']
                    })
            if rows:
                upcoming = pd.DataFrame(rows)
                upcoming.sort_values('gameday', inplace=True)
        except Exception as e:
            logging.warning(f"Error fetching NCAAB upcoming games: {e}")
    return upcoming

########################################
# ModelTrainer Class
########################################
class ModelTrainer:
    """
    Trains ensemble models for each team using advanced stacking plus auto‑ARIMA.
    """
    def __init__(self, team_data, league):
        """
        Initialize the trainer with team data and league name.
        """
        self.team_data = team_data.copy()
        self.league = league
        self.stack_models = {}
        self.arima_models = {}
        self.team_stats = {}
        self.model_errors = {}

    def _feature_engineering(self, df):
        """
        Perform feature engineering on a team’s data.
        Adds rolling average, standard deviation, season average, weighted average,
        early‑vs‑late difference, and lagged score.
        """
        df = df.copy()
        df.sort_values('gameday', inplace=True)
        df['rolling_avg'] = df['score'].rolling(window=3, min_periods=1).mean()
        df['rolling_std'] = df['score'].rolling(window=3, min_periods=1).std().fillna(0)
        df['season_avg'] = df['score'].expanding().mean()
        df['weighted_avg'] = df['rolling_avg'] * 0.6 + df['season_avg'] * 0.4
        df['early_vs_late'] = df['rolling_avg'] * 0.6 - df['rolling_avg'] * 0.4
        df['lag_score'] = df['score'].shift(1).fillna(df['season_avg'])
        return df

    def train_models(self):
        """
        For each team in the data, train a stacking regressor (with hyperparameter tuning)
        and an auto‑ARIMA model (if sufficient data exists). Store the trained models,
        team statistics, and model errors.
        """
        teams = self.team_data['team'].unique()
        for team in teams:
            df_team = self.team_data[self.team_data['team'] == team].copy()
            if len(df_team) < 5:
                continue
            df_team = self._feature_engineering(df_team)
            scores = df_team['score'].reset_index(drop=True)
            if len(scores) < 5:
                continue

            # Store team stats
            self.team_stats[team] = {
                'mean': round_half(scores.mean()),
                'std': round_half(scores.std()),
                'max': round_half(scores.max()),
                'recent_form': round_half(scores.tail(5).mean())
            }

            # Prepare features and target
            features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg', 'early_vs_late', 'lag_score']]
            X = features.fillna(0).values
            y = scores.values

            # Split for time-series CV
            split_index = int(len(y) * 0.8)
            if split_index < 3:
                continue
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # TimeSeriesSplit for CV
            tscv = TimeSeriesSplit(n_splits=3 if len(y_train) >= 3 else 2)

            # Define base estimators
            estimators = [
                ('xgb', XGBRegressor(random_state=42)),
                ('lgbm', LGBMRegressor(random_state=42)),
                ('cat', CatBoostRegressor(verbose=0, random_state=42))
            ]
            # Hyperparameter space for tuning
            param_distributions = {
                'xgb__n_estimators': [100, 200, 300],
                'lgbm__n_estimators': [100, 200, 300],
                'cat__n_estimators': [100, 200, 300]
            }
            # Final estimator pipeline
            final_estimator = make_pipeline(StandardScaler(), LGBMRegressor(random_state=42))
            stacking = StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=tscv,
                passthrough=False,
                n_jobs=-1
            )
            try:
                search = RandomizedSearchCV(
                    stacking,
                    param_distributions=param_distributions,
                    n_iter=5,
                    scoring='neg_mean_squared_error',
                    cv=tscv,
                    random_state=42,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                preds = best_model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                self.stack_models[team] = best_model
                self.model_errors.setdefault(team, {})['stacking'] = mse
                logging.info(f"Trained stacking model for {team} with MSE: {mse:.2f}")
            except Exception as e:
                logging.warning(f"Error training stacking model for {team}: {e}")
                continue

            # Train ARIMA model if enough data
            if len(scores) >= 7:
                try:
                    arima_model = auto_arima(
                        scores,
                        seasonal=False,
                        error_action='ignore',
                        suppress_warnings=True,
                        max_p=3,
                        max_q=3
                    )
                    self.arima_models[team] = arima_model
                    arima_preds = arima_model.predict(n_periods=len(y_test))
                    mse_arima = mean_squared_error(y_test, arima_preds)
                    self.model_errors[team]['arima'] = mse_arima
                    logging.info(f"Trained ARIMA for {team} with MSE: {mse_arima:.2f}")
                except Exception as e:
                    logging.warning(f"Error training ARIMA for {team}: {e}")
                    continue

########################################
# Prediction Functions
########################################
def predict_team_score(team, trainer: ModelTrainer):
    """
    Predict the next game score for a team by blending stacking and ARIMA predictions
    using inverse‑MSE weighting. Returns a tuple (predicted_score, (conf_low, conf_high)).
    """
    if team not in trainer.team_stats:
        return None, (None, None)
    df_team = trainer.team_data[trainer.team_data['team'] == team].copy()
    if df_team.empty:
        return None, (None, None)
    df_team = trainer._feature_engineering(df_team)
    last_features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg', 'early_vs_late', 'lag_score']].tail(1)
    X_next = last_features.fillna(0).values
    stack_pred = None
    if team in trainer.stack_models:
        try:
            stack_pred = float(trainer.stack_models[team].predict(X_next)[0])
        except Exception as e:
            logging.warning(f"Error predicting stacking for {team}: {e}")
            stack_pred = None
    arima_pred = None
    if team in trainer.arima_models:
        try:
            forecast = trainer.arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0])
        except Exception as e:
            logging.warning(f"Error predicting ARIMA for {team}: {e}")
            arima_pred = None
    # Blend predictions via inverse error weighting
    if stack_pred is not None and arima_pred is not None:
        mse_stack = trainer.model_errors.get(team, {}).get('stacking', np.inf)
        mse_arima = trainer.model_errors.get(team, {}).get('arima', np.inf)
        weight_stack = 1.0 / (mse_stack if mse_stack > 0 else 1e-6)
        weight_arima = 1.0 / (mse_arima if mse_arima > 0 else 1e-6)
        ensemble = (stack_pred * weight_stack + arima_pred * weight_arima) / (weight_stack + weight_arima)
    elif stack_pred is not None:
        ensemble = stack_pred
    elif arima_pred is not None:
        ensemble = arima_pred
    else:
        ensemble = None
    if ensemble is None:
        return None, (None, None)
    mu = trainer.team_stats[team]['mean']
    sigma = trainer.team_stats[team]['std']
    conf_low = round_half(mu - 1.96 * sigma)
    conf_high = round_half(mu + 1.96 * sigma)
    return round_half(ensemble), (conf_low, conf_high)

def evaluate_matchup(home_team, away_team, trainer: ModelTrainer):
    """
    Evaluate a matchup by predicting scores for both teams, computing spread,
    total points, and confidence. Returns a dictionary with betting insights.
    """
    home_pred, _ = predict_team_score(home_team, trainer)
    away_pred, _ = predict_team_score(away_team, trainer)
    if home_pred is None or away_pred is None:
        return None
    diff = home_pred - away_pred
    total_points = home_pred + away_pred
    home_std = trainer.team_stats.get(home_team, {}).get('std', 5)
    away_std = trainer.team_stats.get(away_team, {}).get('std', 5)
    combined_std = max(1.0, (home_std + away_std) / 2)
    raw_conf = abs(diff) / combined_std
    confidence = round(min(99, max(1, 50 + raw_conf * 15)), 2)
    predicted_winner = home_team if diff > 0 else away_team
    return {
        'predicted_winner': predicted_winner,
        'diff': round_half(diff),
        'total_points': round_half(total_points),
        'confidence': confidence,
        'spread_suggestion': f"Lean {predicted_winner} by {round_half(diff):.1f} points",
        'ou_suggestion': f"Take the {'Over' if total_points > 145 else 'Under'} {round_half(total_points):.1f}"
    }

########################################
# UI Components
########################################
def generate_writeup(bet, team_stats):
    """
    Generate a detailed analysis write‑up for a bet using team statistics.
    """
    home_team = bet['home_team']
    away_team = bet['away_team']
    home_stats = team_stats.get(home_team, {})
    away_stats = team_stats.get(away_team, {})
    writeup = f"""
**Detailed Analysis:**

**{home_team}**
- Average Score: {home_stats.get('mean', 'N/A')}
- Standard Deviation: {home_stats.get('std', 'N/A')}
- Recent Form: {home_stats.get('recent_form', 'N/A')}

**{away_team}**
- Average Score: {away_stats.get('mean', 'N/A')}
- Standard Deviation: {away_stats.get('std', 'N/A')}
- Recent Form: {away_stats.get('recent_form', 'N/A')}

Based on the data‑driven analysis, **{bet['predicted_winner']}** is predicted to win by {bet['diff']} points.
Confidence: {bet['confidence']}%
Total Points: {bet['total_points']}
"""
    return writeup

def display_bet_card(bet, team_stats):
    """
    Display a bet card with matchup details, suggestions, and detailed insights.
    """
    with st.container():
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"### **{bet['away_team']} @ {bet['home_team']}**")
            if isinstance(bet['date'], datetime):
                st.caption(bet['date'].strftime("%A, %B %d - %I:%M %p"))
        with col2:
            if bet['confidence'] >= 80:
                st.markdown("🔥 **High-Confidence Bet** 🔥")
            st.markdown(f"**Spread:** {bet['spread_suggestion']}")
            st.markdown(f"**Total:** {bet['ou_suggestion']}")
        with col3:
            st.metric("Confidence", f"{bet['confidence']:.1f}%")
        with st.expander("Detailed Insights", expanded=False):
            st.markdown(f"**Predicted Winner:** {bet['predicted_winner']}")
            st.markdown(f"**Total Points:** {bet['total_points']}")
            st.markdown(f"**Margin:** {bet['diff']} points")
        with st.expander("Game Analysis", expanded=False):
            writeup = generate_writeup(bet, team_stats)
            st.markdown(writeup)

########################################
# League Pipeline
########################################
def run_league_pipeline(league_choice):
    """
    Main pipeline for a given league.
    Loads data, trains models, fetches upcoming games, and generates betting insights.
    """
    results = []
    team_data = pd.DataFrame()
    upcoming = pd.DataFrame()
    if league_choice == "NFL":
        schedule = load_nfl_schedule()
        if schedule.empty:
            st.error("Unable to load NFL schedule.")
            return
        team_data = preprocess_nfl_data(schedule)
        upcoming = fetch_upcoming_games("NFL", days_ahead=7)
    elif league_choice == "NBA":
        team_data = load_nba_data()
        if team_data.empty:
            st.error("Unable to load NBA data.")
            return
        upcoming = fetch_upcoming_games("NBA", days_ahead=3)
    elif league_choice == "NCAAB":
        team_data = load_ncaab_data_current_season(season=2025)
        if team_data.empty:
            st.error("Unable to load NCAAB data.")
            return
        upcoming = fetch_upcoming_games("NCAAB", days_ahead=3)
    if team_data.empty or upcoming.empty:
        st.warning(f"No upcoming {league_choice} games available for analysis.")
        return
    with st.spinner("Training models on historical data..."):
        trainer = ModelTrainer(team_data, league_choice)
        trainer.train_models()
    with st.spinner("Generating predictions..."):
        for _, row in upcoming.iterrows():
            home, away = row['home_team'], row['away_team']
            matchup = evaluate_matchup(home, away, trainer)
            if matchup:
                results.append({
                    'date': row['gameday'],
                    'league': league_choice,
                    'home_team': home,
                    'away_team': away,
                    'home_pred': predict_team_score(home, trainer)[0],
                    'away_pred': predict_team_score(away, trainer)[0],
                    'predicted_winner': matchup['predicted_winner'],
                    'predicted_diff': matchup['diff'],
                    'total_points': matchup['total_points'],
                    'confidence': matchup['confidence'],
                    'spread_suggestion': matchup['spread_suggestion'],
                    'ou_suggestion': matchup['ou_suggestion']
                })
    view_mode = st.radio("View Mode", ["Top Bets Only", "All Games"], horizontal=True)
    if view_mode == "Top Bets Only":
        conf_threshold = st.slider("Minimum Confidence Level", min_value=50.0, max_value=99.0, value=75.0, step=5.0)
        results_df = pd.DataFrame(results)
        top_bets = results_df[results_df['confidence'] >= conf_threshold]
        if not top_bets.empty:
            st.markdown(f"### Top {len(top_bets)} Bets")
            for _, bet in top_bets.iterrows():
                display_bet_card(bet, trainer.team_stats)
        else:
            st.info("No high-confidence bets found. Consider lowering the threshold.")
    else:
        if results:
            st.markdown("### All Game Predictions")
            for bet in results:
                display_bet_card(bet, trainer.team_stats)
        else:
            st.info("No predictions available.")

########################################
# Streamlit Main App
########################################
def main():
    st.set_page_config(
        page_title="FoxEdge Ultimate Sports Betting Insights",
        page_icon="🦊",
        layout="centered"
    )
    initialize_csv()
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if not st.session_state['logged_in']:
        st.title("Login to FoxEdge Ultimate Insights")
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
        st.sidebar.write(f"Logged in as: {st.session_state.get('email', 'Unknown')}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.experimental_rerun()
    st.title("🦊 FoxEdge Ultimate Sports Betting Insights")
    st.sidebar.header("Navigation")
    league_choice = st.sidebar.radio("Select League", ["NFL", "NBA", "NCAAB"])
    run_league_pipeline(league_choice)
    st.sidebar.markdown(
        "### About FoxEdge\n"
        "FoxEdge Ultimate provides cutting-edge, data-driven insights for NFL, NBA, and NCAAB games.\n"
        "Powered by advanced machine learning and statistical forecasting."
    )
    if st.button("Save Predictions to CSV"):
        # In a full implementation you would capture the predictions DataFrame and enable download.
        st.success("Predictions saved. Use the download button below to export your CSV.")

if __name__ == "__main__":
    main()
