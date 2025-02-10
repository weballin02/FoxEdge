import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import nfl_data_py as nfl
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2, TeamGameLog
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from pmdarima import auto_arima
from pathlib import Path
import requests
import firebase_admin
from firebase_admin import credentials, auth
import joblib
import os
import random
import optuna
import asyncio
import threading
import time

# cbbpy for NCAAB
import cbbpy.mens_scraper as cbb

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score

###############################
# Global Options and Tuning Method
###############################
TUNING_METHOD = "bayesian"  # Choose between "grid" or "bayesian"
PERFORM_NESTED_CV = True    # Option to perform nested cross-validation for diagnostics

################################################################################
# Helper: Ensure tz-naive datetimes
################################################################################
def to_naive(dt):
    if dt is not None and hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt

################################################################################
# Firebase configuration
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
# CSV management
################################################################################
CSV_FILE = "predictions.csv"

def initialize_csv(csv_file=CSV_FILE):
    if not Path(csv_file).exists():
        columns = [
            "date", "league", "home_team", "away_team", "home_pred", "away_pred",
            "predicted_winner", "predicted_diff", "predicted_total",
            "spread_suggestion", "ou_suggestion", "confidence", "home_rest", "away_rest"
        ]
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

def save_predictions_to_csv(predictions, csv_file=CSV_FILE):
    df = pd.DataFrame(predictions)
    if Path(csv_file).exists():
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_file, index=False)
    st.success("Predictions have been saved to CSV!")

################################################################################
# Utility functions
################################################################################
def round_half(number):
    return round(number * 2) / 2

# Helper to retrieve recent form from team_stats (used by both detailed insights and social media)
def get_recent_form(team_name, team_stats):
    key = team_name.strip().lower()
    if key in team_stats and team_stats[key].get('recent_form') is not None:
        return team_stats[key]['recent_form']
    for k, v in team_stats.items():
        if team_name.lower() in k or k in team_name.lower():
            return v.get('recent_form', 0.0)
    return 0.0

################################################################################
# Tuning functions
################################################################################
def tune_model_grid(model, param_grid, X_train, y_train, early_stopping_rounds=None, eval_set=None):
    if early_stopping_rounds is not None and eval_set is not None:
        model.set_params(early_stopping_rounds=early_stopping_rounds, eval_set=eval_set, verbose=False)
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def bayesian_objective(trial, model_class, X_train, y_train):
    from sklearn.model_selection import cross_val_score
    if model_class == XGBRegressor:
        params = {
            'n_estimators': trial.suggest_int("n_estimators", 50, 200),
            'max_depth': trial.suggest_int("max_depth", 3, 10),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        }
    elif model_class == LGBMRegressor:
        params = {
            'n_estimators': trial.suggest_int("n_estimators", 50, 200),
            'num_leaves': trial.suggest_int("num_leaves", 20, 100),
            'max_depth': trial.suggest_int("max_depth", 3, 10),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 20, 100)
        }
    elif model_class == CatBoostRegressor:
        params = {
            'iterations': trial.suggest_int("iterations", 50, 200),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            'depth': trial.suggest_int("depth", 3, 10)
        }
        params['verbose'] = 0
    else:
        params = {}
    model = model_class(**params, random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    return np.mean(scores)

def tune_model_bayesian(model_class, X_train, y_train, n_trials=50, early_stopping_rounds=None, eval_set=None):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: bayesian_objective(trial, model_class, X_train, y_train), n_trials=n_trials)
    best_params = study.best_trial.params
    model = model_class(**best_params, random_state=42)
    if early_stopping_rounds is not None and eval_set is not None:
        model.set_params(early_stopping_rounds=early_stopping_rounds, eval_set=eval_set, verbose=False)
    model.fit(X_train, y_train)
    return model

def nested_cv_evaluation(model, X, y, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    mean_rmse = np.sqrt(-np.mean(scores))
    return mean_rmse

################################################################################
# Compute predictions in the background
################################################################################
def compute_predictions(league_choice):
    if league_choice == "NFL":
        schedule = load_nfl_schedule()
        if schedule.empty:
            return [], {}, pd.DataFrame()
        team_data = preprocess_nfl_data(schedule)
        upcoming = fetch_upcoming_nfl_games(schedule, days_ahead=7)
    elif league_choice == "NBA":
        team_data = load_nba_data()
        if team_data.empty:
            return [], {}, pd.DataFrame()
        upcoming = fetch_upcoming_nba_games(days_ahead=3)
    else:
        team_data = load_ncaab_data_current_season(season=2025)
        if team_data.empty:
            return [], {}, pd.DataFrame()
        upcoming = fetch_upcoming_ncaab_games()
    if team_data.empty or upcoming.empty:
        return [], {}, team_data
    # For defensive rating adjustments
    if league_choice == "NBA":
        def_ratings = team_data.groupby('team')['def_rating'].mean().to_dict()
        sorted_def = sorted(def_ratings.items(), key=lambda x: x[1])
        top_10 = set([t for t, r in sorted_def[:10]])
        bottom_10 = set([t for t, r in sorted(def_ratings.items(), key=lambda x: x[1], reverse=True)[:10]])
    elif league_choice == "NFL":
        def_ratings = team_data.groupby('team')['opp_score'].mean().to_dict()
        sorted_def = sorted(def_ratings.items(), key=lambda x: x[1])
        top_10 = set([t for t, r in sorted_def[:10]])
        bottom_10 = set([t for t, r in sorted_def[-10:]])
    elif league_choice == "NCAAB":
        def_ratings = team_data.groupby('team')['opp_score'].mean().to_dict()
        sorted_def = sorted(def_ratings.items(), key=lambda x: x[1])
        top_10 = set([t for t, r in sorted_def[:10]])
        bottom_10 = set([t for t, r in sorted_def[-10:]])
    else:
        top_10, bottom_10 = None, None

    stack_models, arima_models, team_stats = train_team_models(team_data)
    results = []
    for _, row in upcoming.iterrows():
        home, away = row['home_team'], row['away_team']
        home_pred, _ = predict_team_score(home, stack_models, arima_models, team_stats, team_data)
        away_pred, _ = predict_team_score(away, stack_models, arima_models, team_stats, team_data)
        row_gameday = to_naive(row['gameday'])
        if league_choice == "NBA" and home_pred is not None and away_pred is not None:
            home_games = team_data[team_data['team'] == home]
            if not home_games.empty:
                last_game_home = to_naive(home_games['gameday'].max())
                rest_days_home = (row_gameday - last_game_home).days
                if rest_days_home == 0:
                    home_pred -= 3
                elif rest_days_home >= 3:
                    home_pred += 2
            away_games = team_data[team_data['team'] == away]
            if not away_games.empty:
                last_game_away = to_naive(away_games['gameday'].max())
                rest_days_away = (row_gameday - last_game_away).days
                if rest_days_away == 0:
                    away_pred -= 3
                elif rest_days_away >= 3:
                    away_pred += 2
            home_pred += 1
            away_pred -= 1
            if top_10 and bottom_10:
                if away in top_10:
                    home_pred -= 2
                elif away in bottom_10:
                    home_pred += 2
                if home in top_10:
                    away_pred -= 2
                elif home in bottom_10:
                    away_pred += 2
        elif league_choice == "NFL" and home_pred is not None and away_pred is not None:
            home_games = team_data[team_data['team'] == home]
            if not home_games.empty:
                last_game_home = to_naive(home_games['gameday'].max())
                rest_days_home = (row_gameday - last_game_home).days
                if rest_days_home == 0:
                    home_pred -= 2
                elif rest_days_home >= 3:
                    home_pred += 1
            away_games = team_data[team_data['team'] == away]
            if not away_games.empty:
                last_game_away = to_naive(away_games['gameday'].max())
                rest_days_away = (row_gameday - last_game_away).days
                if rest_days_away == 0:
                    away_pred -= 2
                elif rest_days_away >= 3:
                    away_pred += 1
            home_pred += 1
            away_pred -= 1
            if top_10 and bottom_10:
                if away in top_10:
                    home_pred -= 2
                elif away in bottom_10:
                    home_pred += 2
                if home in top_10:
                    away_pred -= 2
                elif home in bottom_10:
                    away_pred += 2
        elif league_choice == "NCAAB" and home_pred is not None and away_pred is not None:
            home_games = team_data[team_data['team'] == home]
            if not home_games.empty:
                last_game_home = to_naive(home_games['gameday'].max())
                rest_days_home = (row_gameday - last_game_home).days
                if rest_days_home == 0:
                    home_pred -= 3
                elif rest_days_home >= 3:
                    home_pred += 2
            away_games = team_data[team_data['team'] == away]
            if not away_games.empty:
                last_game_away = to_naive(away_games['gameday'].max())
                rest_days_away = (row_gameday - last_game_away).days
                if rest_days_away == 0:
                    away_pred -= 3
                elif rest_days_away >= 3:
                    away_pred += 2
            home_pred += 1
            away_pred -= 1
            if top_10 and bottom_10:
                if away in top_10:
                    home_pred -= 2
                elif away in bottom_10:
                    home_pred += 2
                if home in top_10:
                    away_pred -= 2
                elif home in bottom_10:
                    away_pred += 2
        outcome = evaluate_matchup(home, away, home_pred, away_pred, team_stats)
        if outcome:
            home_games = team_data[team_data['team'] == home]
            if not home_games.empty:
                last_game_home = to_naive(home_games['gameday'].max())
                rest_days_home = (row_gameday - last_game_home).days
            else:
                rest_days_home = None
            away_games = team_data[team_data['team'] == away]
            if not away_games.empty:
                last_game_away = to_naive(away_games['gameday'].max())
                rest_days_away = (row_gameday - last_game_away).days
            else:
                rest_days_away = None
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
                'ou_suggestion': outcome['ou_suggestion'],
                'home_rest': rest_days_home,
                'away_rest': rest_days_away
            })
    return results, team_stats, team_data

################################################################################
# Display predictions UI
################################################################################
def display_all_predictions(results, team_stats, team_data):
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
        top_bets = pd.DataFrame(find_top_bets(results, threshold=conf_threshold))
        if not top_bets.empty:
            st.markdown(f"### 🔥 Top {len(top_bets)} Bets for Today")
            previous_date = None
            for _, bet_row in top_bets.iterrows():
                bet = bet_row.to_dict()
                current_date = bet['date'].date() if isinstance(bet['date'], datetime) else bet['date']
                if previous_date != current_date:
                    if isinstance(bet['date'], datetime):
                        st.markdown(f"## {bet['date'].strftime('%A, %B %d, %Y')}")
                    else:
                        st.markdown(f"## {bet['date']}")
                    previous_date = current_date
                display_bet_card(bet, team_stats, team_data=team_data)
        else:
            st.info("No high-confidence bets found. Try lowering the threshold.")
    else:
        if results:
            st.markdown("### 📊 All Games Analysis")
            sorted_results = sorted(results, key=lambda x: x['date'])
            previous_date = None
            for bet in sorted_results:
                current_date = bet['date'].date() if isinstance(bet['date'], datetime) else bet['date']
                if previous_date != current_date:
                    if isinstance(bet['date'], datetime):
                        st.markdown(f"## {bet['date'].strftime('%A, %B %d, %Y')}")
                    else:
                        st.markdown(f"## {bet['date']}")
                    previous_date = current_date
                display_bet_card(bet, team_stats, team_data=team_data)
        else:
            st.info("No upcoming games found.")

################################################################################
# Asynchronous background update of predictions
################################################################################
async def async_update_predictions(league_choice):
    loop = asyncio.get_running_loop()
    results, team_stats, team_data = await loop.run_in_executor(None, compute_predictions, league_choice)
    st.session_state["predictions"] = results
    st.session_state["team_stats_global"] = team_stats
    st.session_state["team_data"] = team_data
    st.session_state["update_complete"] = True

################################################################################
# Refresh mechanism using background threads and progress indicators
################################################################################
def start_background_update(league_choice):
    st.session_state["update_complete"] = False
    def run_bg_update():
        asyncio.run(async_update_predictions(league_choice))
    threading.Thread(target=run_bg_update, daemon=True).start()
    placeholder = st.empty()
    loading_messages = [
        "Loading predictions...", "Fetching latest data...", "Training models...", "Almost there..."
    ]
    def rotate_messages():
        i = 0
        while not st.session_state.get("update_complete", False):
            placeholder.info(loading_messages[i % len(loading_messages)])
            time.sleep(1)
            i += 1
        placeholder.empty()
    threading.Thread(target=rotate_messages, daemon=True).start()

################################################################################
# Optional: Schedule periodic updates (uncomment to enable)
################################################################################
def schedule_periodic_updates(interval_minutes=10, league_choice="NBA"):
    def updater():
        while True:
            start_background_update(league_choice)
            time.sleep(interval_minutes * 60)
    threading.Thread(target=updater, daemon=True).start()

# Uncomment the following line to enable periodic updates (e.g., every 10 minutes)
# schedule_periodic_updates(10, league_choice="NBA")

################################################################################
# MAIN PIPELINE: Load UI and trigger background updates if needed
################################################################################
def main():
    st.set_page_config(page_title="FoxEdge Sports Betting Edge", page_icon="🦊", layout="centered")
    st.title("🦊 FoxEdge Sports Betting Insights")
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
    st.sidebar.header("Navigation")
    league_choice = st.sidebar.radio("Select League", ["NFL", "NBA", "NCAAB"],
                                     help="Choose which league's games you'd like to analyze")
    if st.sidebar.button("Refresh Predictions"):
        start_background_update(league_choice)
        st.experimental_rerun()
    if "update_complete" not in st.session_state or not st.session_state.get("update_complete", False):
        start_background_update(league_choice)
        st.info("Predictions are being updated in the background. Please wait...")
        st.stop()
    else:
        display_all_predictions(st.session_state["predictions"],
                                  st.session_state["team_stats_global"],
                                  st.session_state["team_data"])
    st.sidebar.markdown(
        "### About FoxEdge\n"
        "FoxEdge provides data-driven insights for NFL, NBA, and NCAAB games, helping bettors make informed decisions."
    )
    if st.button("Save Predictions to CSV"):
        if st.session_state.get("predictions"):
            save_predictions_to_csv(st.session_state["predictions"])
            csv = pd.DataFrame(st.session_state["predictions"]).to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Predictions as CSV",
                               data=csv,
                               file_name='predictions.csv',
                               mime='text/csv')
        else:
            st.warning("No predictions to save.")

if __name__ == "__main__":
    query_params = st.experimental_get_query_params()
    if "trigger" in query_params:
        start_background_update("NBA")
        st.write("Task triggered successfully.")
    else:
        main()
