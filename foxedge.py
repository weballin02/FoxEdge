import warnings
# Suppress known joblib/loky resource_tracker warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

import streamlit as st
# Set page configuration as the very first Streamlit command in the script.
st.set_page_config(page_title="FoxEdge Sports Betting Edge", page_icon="🦊", layout="centered")

import pandas as pd
import numpy as np
import pytz
import random
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
import optuna  # For Bayesian hyperparameter optimization

# cbbpy for NCAAB
import cbbpy.mens_scraper as cbb

# Additional imports for hyperparameter tuning and time-series cross validation
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# NEW: Import TensorFlow/Keras for the LSTM neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

################################################################################
# GLOBAL FLAGS FOR MODEL TUNING (Optimal Setup)
################################################################################
USE_RANDOMIZED_SEARCH = False    # Do not use RandomizedSearchCV (we rely on Bayesian search)
USE_OPTUNA_SEARCH = True           # Use Bayesian (Optuna) hyperparameter optimization
ENABLE_EARLY_STOPPING = True       # Enable early stopping for LightGBM models

################################################################################
# OPTIONAL SPORTSBOOK ODDS INTEGRATION (Non-Critical)
################################################################################
try:
    from pysbr import BestLines, CurrentLines, OpeningLines, LineHistory
    pysbr_available = True
except Exception as e:
    pysbr_available = False
    st.warning("PySBR not available; sportsbook odds integration will be skipped.")

def generate_game_key(game_date, home_team, away_team):
    """Creates a unique key for each game."""
    return f"{game_date.strftime('%Y-%m-%d')}_{home_team}_{away_team}"

def lookup_event_id(game):
    """
    Looks up the event_id for a given game.
    In production, implement proper lookup logic.
    Here, we return None so that no dummy data is provided.
    """
    return None

def integrate_sportsbook_odds(games):
    """
    Fetches sportsbook odds for a list of games using PySBR.
    If any error occurs, the function warns and returns an empty dictionary.
    """
    odds_data = {}
    try:
        for game in games:
            game_key = generate_game_key(game["gameday"], game["home_team"], game["away_team"])
            event_id = lookup_event_id(game)
            if event_id:
                best_lines = BestLines(event_id).records()
                current_lines = CurrentLines(event_id).records()
                opening_lines = OpeningLines(event_id).records()
            else:
                best_lines, current_lines, opening_lines = {}, {}, {}
            odds_data[game_key] = {
                "best_spread": best_lines.get("spread"),
                "best_total": best_lines.get("total"),
                "market_spread": current_lines.get("spread"),
                "market_total": current_lines.get("total"),
                "opening_spread": opening_lines.get("spread"),
                "opening_total": opening_lines.get("total"),
            }
    except Exception as e:
        st.warning(f"Error fetching sportsbook odds: {e}")
    return odds_data

def evaluate_odds_matchup(game, home_pred, away_pred, odds_data):
    """
    Compares model predictions against sportsbook lines.
    Returns a dictionary containing prediction values,
    sportsbook lines, and edge calculations.
    """
    game_key = generate_game_key(game["gameday"], game["home_team"], game["away_team"])
    odds = odds_data.get(game_key, {})
    diff = home_pred - away_pred
    total_points = home_pred + away_pred
    best_spread = odds.get("best_spread")
    best_total = odds.get("best_total")
    market_spread = odds.get("market_spread")
    market_total = odds.get("market_total")
    spread_edge = round(diff - market_spread, 1) if market_spread is not None else None
    ou_edge = round(total_points - market_total, 1) if market_total is not None else None
    return {
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "predicted_winner": game["home_team"] if diff > 0 else game["away_team"],
        "predicted_diff": round(diff, 1),
        "predicted_total": round(total_points, 1),
        "spread_edge": spread_edge,
        "ou_edge": ou_edge,
        "market_spread": market_spread,
        "market_total": market_total,
        "best_spread": best_spread,
        "best_total": best_total,
    }

def display_odds_bet_card(bet):
    """Displays game predictions vs. sportsbook odds in a styled bet card."""
    with st.container():
        st.markdown(f'<div class="bet-card">', unsafe_allow_html=True)
        st.markdown(f"<h3>{bet['away_team']} @ {bet['home_team']}</h3>", unsafe_allow_html=True)
        st.markdown('<div class="card-body">', unsafe_allow_html=True)
        st.markdown(f"**Predicted Spread:** {bet['predicted_diff']} vs. **Market Spread:** {bet['market_spread']}", unsafe_allow_html=True)
        st.markdown(f"**Spread Edge:** {bet['spread_edge']} pts", unsafe_allow_html=True)
        st.markdown(f"**Predicted Total:** {bet['predicted_total']} vs. **Market Total:** {bet['market_total']}", unsafe_allow_html=True)
        st.markdown(f"**O/U Edge:** {bet['ou_edge']} pts", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

################################################################################
# CUSTOM CSS & GLOBAL UI STYLING (DARK MODE & Vibrant Accents)
################################################################################
st.markdown("""
<style>
  /* Global Background & Typography */
  body {
      font-family: 'Helvetica Neue', Arial, sans-serif;
      background-color: #1e1e1e;
      color: #eee;
  }
  /* Header Banner */
  .header-banner {
      background: linear-gradient(90deg, #ff8c00, #ff0080);
      padding: 20px;
      border-radius: 8px;
      text-align: center;
      margin-bottom: 20px;
  }
  .header-banner h1 {
      color: white;
      margin: 0;
      font-size: 2.5em;
  }
  .header-banner p {
      color: white;
      margin: 0;
      font-size: 1.2em;
  }
  /* Bet Card Styling */
  .bet-card {
      background-color: #2a2a2a;
      border: 1px solid #444;
      border-radius: 8px;
      box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
      padding: 15px;
      margin-bottom: 20px;
      transition: transform 0.2s, box-shadow 0.2s;
  }
  .bet-card:hover {
      transform: scale(1.02);
      box-shadow: 4px 4px 20px rgba(0,0,0,0.4);
  }
  .bet-card .card-header {
      background: #3a3a3a;
      padding: 10px;
      border-radius: 5px;
      margin-bottom: 10px;
  }
  .bet-card .card-header h3, .bet-card .card-header p {
      margin: 0;
      color: #eee;
  }
  .bet-card .card-body {
      margin-bottom: 10px;
  }
  .bet-card .card-body p {
      margin: 5px 0;
  }
  .bet-card .card-footer {
      text-align: right;
  }
  /* Custom Buttons */
  .button-custom {
      background-color: #ff0080;
      color: white;
      border: none;
      border-radius: 5px;
      padding: 8px 16px;
      cursor: pointer;
      transition: background-color 0.2s;
  }
  .button-custom:hover {
      background-color: #e60073;
  }
  /* Confidence Badges */
  .badge {
      display: inline-block;
      padding: 5px 10px;
      border-radius: 5px;
      color: white;
      font-weight: bold;
  }
  .badge.green { background-color: #28a745; }
  .badge.orange { background-color: #fd7e14; }
  .badge.red { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

################################################################################
# HELPER FUNCTION TO ENSURE TZ-NAIVE DATETIMES
################################################################################
def to_naive(dt):
    if dt is not None and hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt

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
    if not Path(csv_file).exists():
        columns = [
            "date", "league", "home_team", "away_team", "home_pred", "away_pred",
            "predicted_winner", "predicted_diff", "predicted_total",
            "spread_suggestion", "ou_suggestion", "confidence"
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
# UTILITY
################################################################################
def round_half(number):
    return round(number * 2) / 2

################################################################################
# BAYESIAN HYPERPARAMETER OPTIMIZATION VIA OPTUNA
################################################################################
def optuna_tune_model(model, param_grid, X_train, y_train, n_trials=20, early_stopping=False):
    cv = TimeSeriesSplit(n_splits=3)
    def objective(trial):
        params = {}
        for key, values in param_grid.items():
            params[key] = trial.suggest_categorical(key, values)
        fit_params = {}
        X_train_used = X_train
        y_train_used = y_train
        if early_stopping and isinstance(model, LGBMRegressor):
            split = int(0.8 * len(X_train))
            X_train_used, X_val = X_train[:split], X_train[split:]
            y_train_used, y_val = y_train[:split], y_train[split:]
            fit_params = {'early_stopping_rounds': 10, 'eval_set': [(X_val, y_val)], 'verbose': False}
        scores = []
        for train_idx, val_idx in cv.split(X_train_used):
            X_tr, X_val_cv = X_train_used[train_idx], X_train_used[val_idx]
            y_tr, y_val_cv = y_train_used[train_idx], y_train_used[val_idx]
            trial_model = model.__class__(**params, random_state=42)
            trial_model.fit(X_tr, y_tr, **fit_params)
            preds = trial_model.predict(X_val_cv)
            score = -mean_squared_error(y_val_cv, preds)
            scores.append(score)
        return np.mean(scores)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trial.params
    best_model = model.__class__(**best_params, random_state=42)
    if early_stopping and isinstance(best_model, LGBMRegressor):
        split = int(0.8 * len(X_train))
        best_model.fit(X_train[:split], y_train[:split],
                       early_stopping_rounds=10, eval_set=[(X_train[split:], y_train[split:])],
                       verbose=False)
    else:
        best_model.fit(X_train, y_train)
    return best_model

################################################################################
# MODEL TUNING FUNCTION
################################################################################
def tune_model(model, param_grid, X_train, y_train, use_randomized=False, early_stopping=False):
    if USE_OPTUNA_SEARCH:
        return optuna_tune_model(model, param_grid, X_train, y_train, n_trials=20, early_stopping=early_stopping)
    else:
        cv = TimeSeriesSplit(n_splits=3)
        fit_params = {}
        if early_stopping and isinstance(model, LGBMRegressor):
            split = int(0.8 * len(X_train))
            X_train, X_val = X_train[:split], X_train[split:]
            y_train, y_val = y_train[:split], y_train[split:]
            fit_params = {'early_stopping_rounds': 10, 'eval_set': [(X_val, y_val)], 'verbose': False}
        if use_randomized:
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                model, param_distributions=param_grid, cv=cv,
                scoring='neg_mean_squared_error', n_jobs=-1, n_iter=20, random_state=42
            )
        else:
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(
                model, param_grid=param_grid, cv=cv,
                scoring='neg_mean_squared_error', n_jobs=-1
            )
        search.fit(X_train, y_train, **fit_params)
        return search.best_estimator_

################################################################################
# NESTED CROSS-VALIDATION EVALUATION
################################################################################
def nested_cv_evaluation(model, param_grid, X, y, use_randomized=False, early_stopping=False):
    from sklearn.model_selection import KFold
    outer_cv = KFold(n_splits=5, shuffle=False)
    scores = []
    for train_idx, test_idx in outer_cv.split(X):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        best_estimator = tune_model(model, param_grid, X_train_outer, y_train_outer,
                                    use_randomized=use_randomized, early_stopping=early_stopping)
        score = best_estimator.score(X_test_outer, y_test_outer)
        scores.append(score)
    return scores

################################################################################
# NEURAL NETWORK (LSTM) MODEL FOR SEQUENTIAL DATA
################################################################################
def create_lstm_model(input_shape):
    """
    Creates and compiles an LSTM neural network model.
    
    Args:
        input_shape (tuple): Shape of the input data (timesteps, features)
    
    Returns:
        model (tf.keras.Model): Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(25, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_team_sequences(team_data, team, window_size=3):
    """
    Prepare sequential data for a given team by creating overlapping windows.
    
    Args:
        team_data (pd.DataFrame): DataFrame containing game data with 'gameday' and 'score' columns.
        team (str): The team name.
        window_size (int): Number of past games to use as input.
    
    Returns:
        X (np.ndarray): 2D array of input sequences.
        y (np.ndarray): 1D array of target scores.
    """
    team_df = team_data[team_data['team'] == team].sort_values('gameday')
    scores = team_df['score'].values
    X, y = [], []
    for i in range(len(scores) - window_size):
        X.append(scores[i:i+window_size])
        y.append(scores[i+window_size])
    return np.array(X), np.array(y)

def train_team_lstm_models(team_data, window_size=3, epochs=50, batch_size=2):
    """
    Trains an LSTM model for each team using sequential data created from their game logs.
    Displays progress as each team is processed.
    
    Args:
        team_data (pd.DataFrame): DataFrame with team game data.
        window_size (int): Number of past games to use as input.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    
    Returns:
        dict: A dictionary mapping team names to a tuple (trained_model, window_size).
    """
    lstm_models = {}
    teams = team_data['team'].unique()
    total_teams = len(teams)
    
    progress_bar_lstm = st.progress(0)
    status_text_lstm = st.empty()
    metric_text_lstm = st.empty()
    
    for idx, team in enumerate(teams):
        status_text_lstm.text(f"Training LSTM model for team: {team} ({idx+1}/{total_teams})")
        X, y = prepare_team_sequences(team_data, team, window_size)
        if len(X) < 5:
            continue
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = create_lstm_model((X.shape[1], 1))
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        final_loss = history.history['loss'][-1]
        metric_text_lstm.text(f"Team {team} - Final Loss: {final_loss:.4f}")
        lstm_models[team] = (model, window_size)
        progress_bar_lstm.progress((idx + 1) / total_teams)
    
    status_text_lstm.text("All LSTM models trained!")
    metric_text_lstm.text("")
    return lstm_models

################################################################################
# MODEL TRAINING & PREDICTION (STACKING + AUTO-ARIMA HYBRID)
################################################################################
@st.cache_data(ttl=14400)
def train_team_models(team_data: pd.DataFrame):
    """
    Trains stacking regressors and ARIMA models for each team.
    Displays progress and status updates for each team.
    
    Returns:
        tuple: (stack_models, arima_models, team_stats)
    """
    stack_models = {}
    arima_models = {}
    team_stats = {}
    all_teams = team_data['team'].unique()
    total_teams = len(all_teams)
    
    progress_bar_stack = st.progress(0)
    status_text_stack = st.empty()
    
    for idx, team in enumerate(all_teams):
        status_text_stack.text(f"Training stacked & ARIMA models for team: {team} ({idx+1}/{total_teams})")
        df_team = team_data[team_data['team'] == team].copy()
        df_team.sort_values('gameday', inplace=True)
        scores = df_team['score'].reset_index(drop=True)
        if len(scores) < 3:
            continue
        df_team['rolling_avg'] = df_team['score'].rolling(window=3, min_periods=1).mean()
        df_team['rolling_std'] = df_team['score'].rolling(window=3, min_periods=1).std().fillna(0)
        df_team['season_avg'] = df_team['score'].expanding().mean()
        df_team['weighted_avg'] = (df_team['rolling_avg'] * 0.6) + (df_team['season_avg'] * 0.4)
        team_stats[team] = {
            'mean': round_half(scores.mean()),
            'std': round_half(scores.std()),
            'max': round_half(scores.max()),
            'recent_form': round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
        }
        features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg']].fillna(0)
        X = features.values
        y = scores.values
        n = len(X)
        split_index = int(n * 0.8)
        if split_index < 2 or n - split_index < 1:
            continue
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        try:
            xgb = XGBRegressor(random_state=42)
            xgb_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]}
            xgb_best = tune_model(xgb, xgb_grid, X_train, y_train,
                                  use_randomized=USE_RANDOMIZED_SEARCH, early_stopping=False)
        except Exception as e:
            print(f"Error tuning XGB for team {team}: {e}")
            xgb_best = XGBRegressor(n_estimators=100, random_state=42)
        try:
            lgbm = LGBMRegressor(random_state=42)
            lgbm_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 5, 10],
                'num_leaves': [31, 50, 70],
                'min_child_samples': [20, 30, 50],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
            lgbm_best = tune_model(lgbm, lgbm_grid, X_train, y_train,
                                   use_randomized=USE_RANDOMIZED_SEARCH, early_stopping=ENABLE_EARLY_STOPPING)
        except Exception as e:
            print(f"Error tuning LGBM for team {team}: {e}")
            lgbm_best = LGBMRegressor(n_estimators=100, random_state=42)
        try:
            cat = CatBoostRegressor(verbose=0, random_state=42)
            cat_grid = {'iterations': [50, 100, 150], 'learning_rate': [0.1, 0.05, 0.01]}
            cat_best = tune_model(cat, cat_grid, X_train, y_train,
                                  use_randomized=USE_RANDOMIZED_SEARCH, early_stopping=False)
        except Exception as e:
            print(f"Error tuning CatBoost for team {team}: {e}")
            cat_best = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)
        estimators = [
            ('xgb', xgb_best),
            ('lgbm', lgbm_best),
            ('cat', cat_best)
        ]
        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=LGBMRegressor(),
            passthrough=False,
            cv=3
        )
        try:
            stack.fit(X_train, y_train)
            preds = stack.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            print(f"Team: {team}, Stacking Regressor MSE: {mse}")
            stack_models[team] = stack
            team_stats[team]['mse'] = mse
            bias = np.mean(y_train - stack.predict(X_train))
            team_stats[team]['bias'] = bias
        except Exception as e:
            print(f"Error training Stacking Regressor for team {team}: {e}")
            continue
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
        progress_bar_stack.progress((idx + 1) / total_teams)
    status_text_stack.text("All stacked & ARIMA models trained!")
    return stack_models, arima_models, team_stats

def predict_team_score(team, stack_models, arima_models, lstm_models, team_stats, team_data):
    """
    Predicts a team's score using an ensemble of stacking, ARIMA, and LSTM predictions.
    
    Args:
        team (str): Team name.
        stack_models (dict): Trained stacking models per team.
        arima_models (dict): Trained ARIMA models per team.
        lstm_models (dict): Trained LSTM models per team.
        team_stats (dict): Dictionary of team statistics.
        team_data (pd.DataFrame): DataFrame with team game data.
    
    Returns:
        tuple: (ensemble prediction, (confidence lower, confidence upper))
    """
    if team not in team_stats:
        return None, (None, None)
    df_team = team_data[team_data['team'] == team]
    data_len = len(df_team)
    stack_pred = None
    arima_pred = None
    lstm_pred = None
    if data_len < 3:
        return None, (None, None)
    last_features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg']].tail(1)
    X_next = last_features.values
    if team in stack_models:
        try:
            stack_pred = float(stack_models[team].predict(X_next)[0])
        except Exception as e:
            print(f"Error predicting with Stacking Regressor for team {team}: {e}")
            stack_pred = None
    if team in arima_models:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast)
        except Exception as e:
            print(f"Error predicting with ARIMA for team {team}: {e}")
            arima_pred = None
    if team in lstm_models:
        try:
            model, window_size = lstm_models[team]
            team_df = df_team.sort_values('gameday')
            scores = team_df['score'].values
            if len(scores) >= window_size:
                last_window = scores[-window_size:]
                last_window = np.array(last_window).reshape((1, window_size, 1))
                lstm_pred = float(model.predict(last_window)[0][0])
        except Exception as e:
            print(f"Error predicting with LSTM for team {team}: {e}")
            lstm_pred = None

    predictions = [p for p in [stack_pred, arima_pred, lstm_pred] if p is not None]
    if not predictions:
        return None, (None, None)
    ensemble = np.mean(predictions)
    if team_stats[team].get('mse', 0) > 150:
        return None, (None, None)
    bias = team_stats[team].get('bias', 0)
    ensemble_calibrated = ensemble + bias
    mu = team_stats[team]['mean']
    sigma = team_stats[team]['std']
    if isinstance(mu, (pd.Series, pd.DataFrame, np.ndarray)):
        mu = mu.item()
    if isinstance(sigma, (pd.Series, pd.DataFrame, np.ndarray)):
        sigma = sigma.item()
    conf_low = round_half(mu - 1.96 * sigma)
    conf_high = round_half(mu + 1.96 * sigma)
    return round_half(ensemble_calibrated), (conf_low, conf_high)

def evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats):
    if home_pred is None or away_pred is None:
        return None
    diff = home_pred - away_pred
    total_points = home_pred + away_pred
    home_std = team_stats.get(home_team, {}).get('std', 5)
    away_std = team_stats.get(away_team, {}).get('std', 5)
    combined_std = max(1.0, (home_std + away_std) / 2)
    raw_conf = abs(diff) / combined_std
    confidence = round(min(99, max(1, 50 + raw_conf * 15)), 2)
    penalty = 0
    if team_stats.get(home_team, {}).get('mse', 0) > 120:
        penalty += 10
    if team_stats.get(away_team, {}).get('mse', 0) > 120:
        penalty += 10
    confidence = max(1, min(99, confidence - penalty))
    winner = home_team if diff > 0 else away_team
    ou_threshold = 145
    spread_value = -abs(round_half(diff))
    return {
        'predicted_winner': winner,
        'diff': round_half(diff),
        'total_points': round_half(total_points),
        'confidence': confidence,
        'spread_suggestion': f"Lean {winner} by {spread_value:.1f}",
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
@st.cache_data(ttl=3600)
def load_nfl_schedule():
    current_year = datetime.now().year
    years = [current_year - i for i in range(12)]
    schedule = nfl.import_schedules(years)
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(schedule['gameday']):
        schedule['gameday'] = schedule['gameday'].dt.tz_convert(None)
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
    data['season_avg'] = data.groupby('team')['score'].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
    data['weighted_avg'] = (data['rolling_avg'] * 0.6) + (data['season_avg'] * 0.4)
    return data

def fetch_upcoming_nfl_games(schedule, days_ahead=7):
    upcoming = schedule[schedule['home_score'].isna() & schedule['away_score'].isna()].copy()
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
    nba_teams_list = nba_teams.get_teams()
    seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
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
                needed = ['PTS', 'FGA', 'FTA', 'TOV', 'OREB', 'PTS_OPP']
                for c in needed:
                    if c not in gl.columns:
                        gl[c] = 0
                    gl[c] = pd.to_numeric(gl[c], errors='coerce').fillna(0)
                gl['TEAM_POSSESSIONS'] = gl['FGA'] + 0.44 * gl['FTA'] + gl['TOV'] - gl['OREB']
                gl['TEAM_POSSESSIONS'] = gl['TEAM_POSSESSIONS'].apply(lambda x: x if x > 0 else np.nan)
                gl['OFF_RATING'] = np.where(gl['TEAM_POSSESSIONS'] > 0, (gl['PTS'] / gl['TEAM_POSSESSIONS']) * 100, np.nan)
                gl['DEF_RATING'] = np.where(gl['TEAM_POSSESSIONS'] > 0, (gl['PTS_OPP'] / gl['TEAM_POSSESSIONS']) * 100, np.nan)
                gl['PACE'] = gl['TEAM_POSSESSIONS']
                gl['rolling_avg'] = gl['PTS'].rolling(window=3, min_periods=1).mean()
                gl['rolling_std'] = gl['PTS'].rolling(window=3, min_periods=1).std().fillna(0)
                gl['season_avg'] = gl['PTS'].expanding().mean()
                gl['weighted_avg'] = (gl['rolling_avg'] * 0.6) + (gl['season_avg'] * 0.4)
                for idx, row_ in gl.iterrows():
                    try:
                        all_rows.append({
                            'gameday': row_['GAME_DATE'],
                            'team': team_abbrev,
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

################################################################################
# NCAAB HISTORICAL LOADER (UPDATED)
################################################################################
@st.cache_data(ttl=14400)
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
    data['season_avg'] = data.groupby('team')['score'].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
    data['weighted_avg'] = (data['rolling_avg'] * 0.6) + (data['season_avg'] * 0.4)
    data.sort_values(['team', 'gameday'], inplace=True)
    data['game_index'] = data.groupby('team').cumcount()
    return data

def fetch_upcoming_ncaab_games() -> pd.DataFrame:
    timezone = pytz.timezone('America/Los_Angeles')
    current_time = datetime.now(timezone)
    dates = [
        current_time.strftime('%Y%m%d'),
        (current_time + timedelta(days=1)).strftime('%Y%m%d')
    ]
    rows = []
    for date_str in dates:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {
            'dates': date_str,
            'groups': '50',
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
            game_time_str = game['date']
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
# EXISTING UI COMPONENTS (Writeup, Bet Card, etc.)
################################################################################
def generate_writeup(bet, team_stats_global):
    home_team = bet['home_team']
    away_team = bet['away_team']
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
  - **Score Std Dev:** {home_std}
  - **Recent Form (Last 5 Games):** {home_recent}

- **{away_team} Performance:**
  - **Average Score:** {away_mean}
  - **Score Std Dev:** {away_std}
  - **Recent Form (Last 5 Games):** {away_recent}

- **Prediction Insight:**
  Based on recent performance, **{predicted_winner}** is predicted to win with **{confidence}%** confidence.
  The projected score difference is **{bet['predicted_diff']} points**, leading to a spread suggestion of **{bet['spread_suggestion']}**.
  The total predicted points for the game are **{bet['predicted_total']}** – suggesting a **{bet['ou_suggestion']}** bet.

- **Statistical Edge:**
  The confidence reflects a robust edge derived from combined team metrics.
"""
    return writeup

def generate_social_media_post(bet):
    conf = bet['confidence']
    if conf >= 85:
        tone = "This one’s a sure-fire winner! Don’t miss out!"
    elif conf >= 70:
        tone = "Looks promising – keep an eye on this one…"
    else:
        tone = "A cautious bet worth watching!"
    templates = [
        f"🔥 **Bet Alert!** 🔥\n\n"
        f"**Matchup:** {bet['away_team']} @ {bet['home_team']}\n\n"
        f"**Prediction Highlights:**\n"
        f"• **Winner:** {bet['predicted_winner']}\n"
        f"• **Spread:** {bet['spread_suggestion']}\n"
        f"• **Total Points:** {bet['predicted_total']}\n"
        f"• **Confidence:** {bet['confidence']:.1f}%\n\n"
        f"{tone}\n\n"
        f"💬 **Testimonial:** “I turned a $10 bet into $50 thanks to FoxEdge – total game changer!” – Alex\n\n"
        f"👉 **CTA:** {{cta}}\n\n"
        f"💡 **Did You Know?** Our model analyzes recent form and key metrics to give you the edge!\n\n"
        f"👥 **Join us:** Be part of a community of winning bettors!\n\n"
        f"🔎 {{hashtags}}",
        
        f"🚀 **Hot Tip Alert!** 🚀\n\n"
        f"**Game:** {bet['away_team']} @ {bet['home_team']}\n\n"
        f"• **Winner:** {bet['predicted_winner']}\n"
        f"• **Spread:** {bet['spread_suggestion']}\n"
        f"• **Total:** {bet['predicted_total']}\n"
        f"• **Confidence:** {bet['confidence']:.1f}%\n\n"
        f"{tone}\n\n"
        f"💡 **FYI:** Our predictions leverage advanced analytics for an unbeatable edge.\n\n"
        f"👉 **CTA:** {{cta}}\n\n"
        f"🏷️ {{hashtags}}",
        
        f"🎯 **Pro Pick Alert!** 🎯\n\n"
        f"Matchup: {bet['away_team']} vs {bet['home_team']}\n"
        f"Predicted Winner: {bet['predicted_winner']}\n"
        f"Spread: {bet['spread_suggestion']}\n"
        f"Total Points: {bet['predicted_total']}\n"
        f"Confidence: {bet['confidence']:.1f}%\n\n"
        f"{tone}\n\n"
        f"👉 **CTA:** {{cta}}\n\n"
        f"💡 **Did You Know?** Our system uses key metrics for that extra edge!\n\n"
        f"{{hashtags}}"
    ]
    selected_template = random.choice(templates)
    cta_options = [
        "Comment your prediction below!",
        "Tag a friend who needs this tip!",
        "Download now for real-time insights!",
        "Join the winning team and share your pick!"
    ]
    selected_cta = random.choice(cta_options)
    hashtag_pool = ["#SportsBetting", "#GameDay", "#BetSmart", "#WinningTips", "#Edge", "#BettingCommunity"]
    selected_hashtags = " ".join(random.sample(hashtag_pool, k=3))
    post = selected_template.replace("{cta}", selected_cta).replace("{hashtags}", selected_hashtags)
    return post

def display_bet_card(bet, team_stats_global, team_data=None):
    with st.container():
        st.markdown(f'<div class="bet-card">', unsafe_allow_html=True)
        date_obj = bet['date']
        date_str = date_obj.strftime("%A, %B %d - %I:%M %p") if isinstance(date_obj, datetime) else str(date_obj)
        st.markdown(
            f'<div class="card-header"><h3>{bet["away_team"]} @ {bet["home_team"]}</h3>'
            f'<p title="Game time">{date_str}</p></div>', unsafe_allow_html=True)
        conf_color = "green" if bet['confidence'] >= 80 else "red" if bet['confidence'] < 60 else "orange"
        st.markdown('<div class="card-body">', unsafe_allow_html=True)
        st.markdown(
            f'<p title="Spread Suggestion is based on the predicted margin">Spread Suggestion: {bet["spread_suggestion"]}</p>',
            unsafe_allow_html=True)
        st.markdown(
            f'<p title="Total Suggestion is based on the combined score prediction">Total Suggestion: {bet["ou_suggestion"]}</p>',
            unsafe_allow_html=True)
        st.markdown(
            f'<p><span class="badge {conf_color}" title="Confidence indicates the statistical edge from combined team metrics">{bet["confidence"]:.1f}% Confidence</span></p>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-footer">', unsafe_allow_html=True)
        if st.button("Generate Social Post", key=f"social_post_{bet['home_team']}_{bet['away_team']}_{bet['date']}"):
            post = generate_social_media_post(bet)
            st.code(post, language="markdown")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("Detailed Insights", expanded=False):
        st.markdown(f"**Predicted Winner:** {bet['predicted_winner']}")
        st.markdown(f"**Predicted Total Points:** {bet['predicted_total']}")
        st.markdown(f"**Prediction Margin (Diff):** {bet['predicted_diff']}")
    with st.expander("Game Analysis", expanded=False):
        writeup = generate_writeup(bet, team_stats_global)
        st.markdown(writeup)
    if team_data is not None:
        with st.expander("Recent Performance Trends", expanded=False):
            home_team_data = team_data[team_data['team'] == bet['home_team']].sort_values('gameday')
            if not home_team_data.empty:
                st.markdown(f"**{bet['home_team']} Recent Scores:**")
                home_scores = home_team_data['score'].tail(5).reset_index(drop=True)
                st.line_chart(home_scores)
            away_team_data = team_data[team_data['team'] == bet['away_team']].sort_values('gameday')
            if not away_team_data.empty:
                st.markdown(f"**{bet['away_team']} Recent Scores:**")
                away_scores = away_team_data['score'].tail(5).reset_index(drop=True)
                st.line_chart(away_scores)

################################################################################
# GLOBALS
################################################################################
results = []
team_stats_global = {}

################################################################################
# MAIN PIPELINE
################################################################################
def run_league_pipeline(league_choice):
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
    
    with st.spinner("Training models... This may take a while."):
        # Train stacking & ARIMA models with progress updates
        stack_models, arima_models, team_stats = train_team_models(team_data)
        # Train LSTM models with progress updates
        lstm_models = train_team_lstm_models(team_data, window_size=3, epochs=50, batch_size=2)
        team_stats_global = team_stats
        results.clear()
        for _, row in upcoming.iterrows():
            home, away = row['home_team'], row['away_team']
            home_pred, _ = predict_team_score(home, stack_models, arima_models, lstm_models, team_stats, team_data)
            away_pred, _ = predict_team_score(away, stack_models, arima_models, lstm_models, team_stats, team_data)
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
        conf_threshold = st.slider("Minimum Confidence Level", min_value=50.0, max_value=99.0, value=75.0, step=5.0,
                                    help="Only show bets with confidence level above this threshold")
        top_bets = find_top_bets(results, threshold=conf_threshold)
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
                display_bet_card(bet, team_stats_global, team_data=team_data)
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
                display_bet_card(bet, team_stats_global, team_data=team_data)
        else:
            st.info(f"No upcoming {league_choice} games found.")
    
    # Optional Sportsbook Odds Integration Section
    if st.sidebar.checkbox("Show Sportsbook Odds Comparison (Optional)"):
        if pysbr_available:
            try:
                games_for_odds = []
                for res in results:
                    games_for_odds.append({
                        "gameday": res["date"],
                        "home_team": res["home_team"],
                        "away_team": res["away_team"]
                    })
                odds_data = integrate_sportsbook_odds(games_for_odds)
                st.header("Sportsbook Odds Comparison")
                for res in results:
                    game_key = generate_game_key(res["date"], res["home_team"], res["away_team"])
                    if odds_data.get(game_key, {}).get("market_spread") is not None:
                        odds_bet = evaluate_odds_matchup(
                            {"gameday": res["date"], "home_team": res["home_team"], "away_team": res["away_team"]},
                            res["home_pred"],
                            res["away_pred"],
                            odds_data)
                        display_odds_bet_card(odds_bet)
            except Exception as e:
                st.warning(f"Skipping sportsbook odds integration due to error: {e}")
        else:
            st.info("Sportsbook odds integration is not available.")
    
################################################################################
# SCHEDULED TASK (for updating predictions)
################################################################################
def scheduled_task():
    st.write("🕒 Scheduled task running: Fetching and updating predictions...")
    st.write("📡 Fetching latest NFL schedule and results...")
    schedule = nfl.import_schedules([datetime.now().year])
    schedule.to_csv("nfl_schedule.csv", index=False)
    st.write("🏀 Fetching latest NBA team game logs...")
    nba_data = []
    for team_id in range(1, 31):
        try:
            logs = TeamGameLog(team_id=team_id, season="2024-25").get_data_frames()[0]
            nba_data.append(logs)
        except Exception as e:
            st.warning(f"Error fetching data for NBA team {team_id}: {e}")
    if nba_data:
        nba_df = pd.concat(nba_data, ignore_index=True)
        nba_df.to_csv("nba_team_logs.csv", index=False)
    st.write("🏀 Fetching latest NCAAB data...")
    ncaab_df, _, _ = cbb.get_games_season(season=2025, info=True, box=False, pbp=False)
    if not ncaab_df.empty:
        ncaab_df.to_csv("ncaab_games.csv", index=False)
    st.write("🤖 Updating prediction models...")
    if os.path.exists("nfl_schedule.csv"):
        joblib.dump(schedule, "models/nfl_model.pkl")
    if os.path.exists("nba_team_logs.csv"):
        joblib.dump(nba_df, "models/nba_model.pkl")
    if os.path.exists("ncaab_games.csv"):
        joblib.dump(ncaab_df, "models/ncaab_model.pkl")
    st.success("✅ Scheduled task completed successfully!")
    st.success("Scheduled task completed successfully.")

################################################################################
# MAIN FUNCTION (with Homepage as First Page)
################################################################################
def main():
    if 'homepage_done' not in st.session_state:
        st.title("Welcome to FoxEdge Sports Betting Insights")
        st.markdown("""
FoxEdge provides data-driven insights for NFL, NBA, and NCAAB games—helping you make informed betting decisions.  
Explore predictions, view detailed analyses, and generate social posts to share your picks.
""")
        if st.button("Get Started"):
            st.session_state.homepage_done = True
            st.experimental_rerun()
        st.stop()

    st.markdown("""
    <div class="header-banner">
        <h1>🦊 FoxEdge</h1>
        <p>Your Edge in Sports Betting</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if not st.session_state['logged_in']:
        st.sidebar.image("https://via.placeholder.com/100", width=100)
        st.sidebar.title("Account")
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
        st.sidebar.image("https://via.placeholder.com/100", width=100)
        st.sidebar.title("Account")
        st.sidebar.write(f"Logged in as: {st.session_state.get('email','Unknown')}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.experimental_rerun()
    
    st.sidebar.header("Navigation")
    league_options = {"🏈 NFL": "NFL", "🏀 NBA": "NBA", "🎓 NCAAB": "NCAAB"}
    league_choice_display = st.sidebar.radio("Select League", list(league_options.keys()),
                                             help="Choose which league's games you'd like to analyze")
    league_choice = league_options.get(league_choice_display, league_choice_display)
    
    run_league_pipeline(league_choice)
    st.sidebar.markdown(
        "### About FoxEdge\n"
        "FoxEdge provides data-driven insights for NFL, NBA, and NCAAB games, helping bettors make informed decisions."
    )
    if st.button("Save Predictions to CSV"):
        if results:
            save_predictions_to_csv(results)
            csv = pd.DataFrame(results).to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Predictions as CSV", data=csv,
                               file_name='predictions.csv', mime='text/csv')
        else:
            st.warning("No predictions to save.")

if __name__ == "__main__":
    query_params = st.query_params
    if "trigger" in query_params:
        scheduled_task()
        st.write("Task triggered successfully.")
    else:
        main()
