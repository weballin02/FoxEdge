import warnings
# Suppress known joblib/loky resource_tracker warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

import streamlit as st
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
import optuna
import inspect
import cbbpy.mens_scraper as cbb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from instagrapi import Client
import time
from PIL import Image  # For image processing

# =============================================================================
# Shared Utility Function (formerly in betting_utils.py)
# =============================================================================
def generate_social_media_post(bet):
    """
    Generate a social media post text from a betting prediction.
    """
    conf = bet.get('confidence', 50)
    if conf >= 85:
        tone = "This one’s a sure-fire winner! Don’t miss out!"
    elif conf >= 70:
        tone = "Looks promising – keep an eye on this one…"
    else:
        tone = "A cautious bet worth watching!"
    
    templates = [
        (
            "🔥 **Bet Alert!** 🔥\n\n"
            "**Matchup:** {away_team} @ {home_team}\n\n"
            "**Prediction:** {predicted_winner}\n"
            "• **Spread:** {spread_suggestion}\n"
            "• **Total Points:** {predicted_total}\n"
            "• **Confidence:** {confidence:.1f}%\n\n"
            "{tone}\n\n"
            "👉 **CTA:** {cta}\n\n"
            "🏷️ {hashtags}"
        ),
        (
            "🚀 **Hot Tip Alert!** 🚀\n\n"
            "Game: {away_team} @ {home_team}\n"
            "Winner: {predicted_winner}\n"
            "Spread: {spread_suggestion}\n"
            "Total Points: {predicted_total}\n"
            "Confidence: {confidence:.1f}%\n\n"
            "{tone}\n\n"
            "👉 **CTA:** {cta}\n\n"
            "🏷️ {hashtags}"
        )
    ]
    selected_template = random.choice(templates)
    cta_options = [
        "Comment your prediction below!",
        "Tag a friend who needs this tip!",
        "Download now for real-time insights!"
    ]
    selected_cta = random.choice(cta_options)
    hashtag_pool = ["#SportsBetting", "#GameDay", "#BetSmart", "#WinningTips"]
    selected_hashtags = " ".join(random.sample(hashtag_pool, k=3))
    return selected_template.format(
        home_team=bet['home_team'],
        away_team=bet['away_team'],
        predicted_winner=bet['predicted_winner'],
        spread_suggestion=bet['spread_suggestion'],
        predicted_total=bet['predicted_total'],
        confidence=bet['confidence'],
        tone=tone,
        cta=selected_cta,
        hashtags=selected_hashtags
    )

# =============================================================================
# Helper Function: Merge Two Team Logos
# =============================================================================
def merge_team_logos(team1, team2, output_path="merged_logo.png", target_height=200, bg_color=(255,255,255)):
    """
    Merge logos for two teams side-by-side.
    
    Parameters:
        team1 (str): Name of the first team (e.g., "Atlanta Hawks")
        team2 (str): Name of the second team
        output_path (str): File path to save the merged image
        target_height (int): Height in pixels for the resized logos
        bg_color (tuple): Background color for the merged image
        
    Returns:
        str: Path to the merged image if both logos are found; otherwise None.
    """
    def get_logo(team_name):
        base_dir = "nba_images"  # Folder in project directory
        filename = team_name.strip().replace(" ", "_") + ".png"
        path = os.path.join(base_dir, filename)
        return path if os.path.exists(path) else None

    logo1_path = get_logo(team1)
    logo2_path = get_logo(team2)
    if not logo1_path or not logo2_path:
        return None  # One or both logos not found.

    img1 = Image.open(logo1_path)
    img2 = Image.open(logo2_path)

    # Resize both images to the same height
    img1_ratio = img1.width / img1.height
    img2_ratio = img2.width / img2.height
    img1 = img1.resize((int(target_height * img1_ratio), target_height))
    img2 = img2.resize((int(target_height * img2_ratio), target_height))

    # Create a new image with width equal to sum of both widths
    total_width = img1.width + img2.width
    merged_img = Image.new('RGB', (total_width, target_height), color=bg_color)
    merged_img.paste(img1, (0, 0))
    merged_img.paste(img2, (img1.width, 0))
    merged_img.save(output_path)
    return output_path

# =============================================================================
# Global Flags and CSV File Name
# =============================================================================
USE_RANDOMIZED_SEARCH = True
USE_OPTUNA_SEARCH = True
ENABLE_EARLY_STOPPING = True
DISABLE_TUNING_FOR_NCAAB = False
USE_NBA_CSV_DATA = True
CSV_FILE = "predictions.csv"

# =============================================================================
# CSV and Utility Functions
# =============================================================================
def load_csv_data_safe(file_path, default_df=None):
    try:
        file = Path(file_path)
        if file.exists():
            return pd.read_csv(file)
    except Exception as e:
        print(f"Warning: Failed to load {file_path}. Error: {e}")
    return default_df if default_df is not None else pd.DataFrame()

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

def round_half(number):
    return round(number * 2) / 2

def supports_early_stopping(model):
    try:
        sig = inspect.signature(model.fit)
        return 'early_stopping_rounds' in sig.parameters
    except Exception:
        return False

# =============================================================================
# Hyperparameter Tuning Functions
# =============================================================================
def optuna_tune_model(model, param_grid, X_train, y_train, n_trials=20, early_stopping=False):
    cv = TimeSeriesSplit(n_splits=3)
    def objective(trial):
        params = {}
        for key, values in param_grid.items():
            params[key] = trial.suggest_categorical(key, values)
        fit_params = {}
        X_train_used = X_train
        y_train_used = y_train
        if early_stopping and isinstance(model, LGBMRegressor) and supports_early_stopping(model):
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
    if early_stopping and isinstance(best_model, LGBMRegressor) and supports_early_stopping(best_model):
        split = int(0.8 * len(X_train))
        best_model.fit(X_train[:split], y_train[:split],
                       early_stopping_rounds=10, eval_set=[(X_train[split:], y_train[split:])],
                       verbose=False)
    else:
        best_model.fit(X_train, y_train)
    return best_model

def tune_model(model, param_grid, X_train, y_train, use_randomized=False, early_stopping=False):
    if USE_OPTUNA_SEARCH:
        return optuna_tune_model(model, param_grid, X_train, y_train, n_trials=20, early_stopping=early_stopping)
    else:
        cv = TimeSeriesSplit(n_splits=3)
        fit_params = {}
        if early_stopping and isinstance(model, LGBMRegressor) and supports_early_stopping(model):
            split = int(0.8 * len(X_train))
            X_train, X_val = X_train[:split], X_train[split:]
            y_train, y_val = y_train[:split], y_train[split:]
            fit_params = {'early_stopping_rounds': 10, 'eval_set': [(X_val, y_val)], 'verbose': False}
        if use_randomized:
            search = RandomizedSearchCV(
                model, param_distributions=param_grid, cv=cv,
                scoring='neg_mean_squared_error', n_jobs=-1, n_iter=20, random_state=42
            )
        else:
            search = GridSearchCV(
                model, param_grid=param_grid, cv=cv,
                scoring='neg_mean_squared_error', n_jobs=-1
            )
        search.fit(X_train, y_train, **fit_params)
        return search.best_estimator_

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

# =============================================================================
# Model Training & Prediction Functions (Stacking + ARIMA)
# =============================================================================
@st.cache_data(ttl=3600)
def train_team_models(team_data: pd.DataFrame, disable_tuning=False):
    if USE_NBA_CSV_DATA and not team_data.empty:
        if 'spread1' in team_data.columns and 'spread2' in team_data.columns:
            team_data['spread_diff'] = abs(team_data['spread1'] - team_data['spread2'])
        if 'rolling_avg' not in team_data.columns:
            team_data['rolling_avg'] = team_data.groupby('team')['score'].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )

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

        if disable_tuning:
            xgb_best = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
            lgbm_best = LGBMRegressor(n_estimators=100, max_depth=5, random_state=42)
            cat_best = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)
        else:
            try:
                xgb = XGBRegressor(random_state=42)
                xgb_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]}
                xgb_best = tune_model(
                    xgb, xgb_grid, X_train, y_train,
                    use_randomized=USE_RANDOMIZED_SEARCH, early_stopping=False
                )
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
                lgbm_best = tune_model(
                    lgbm, lgbm_grid, X_train, y_train,
                    use_randomized=USE_RANDOMIZED_SEARCH, early_stopping=ENABLE_EARLY_STOPPING
                )
            except Exception as e:
                print(f"Error tuning LGBM for team {team}: {e}")
                lgbm_best = LGBMRegressor(n_estimators=100, random_state=42)
            try:
                cat = CatBoostRegressor(verbose=0, random_state=42)
                cat_grid = {'iterations': [50, 100, 150], 'learning_rate': [0.1, 0.05, 0.01]}
                cat_best = tune_model(
                    cat, cat_grid, X_train, y_train,
                    use_randomized=USE_RANDOMIZED_SEARCH, early_stopping=False
                )
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

        if not disable_tuning and len(scores) >= 7:
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
        else:
            arima_models[team] = None
    return stack_models, arima_models, team_stats

def predict_team_score(team, stack_models, arima_models, team_stats, team_data):
    if team not in team_stats:
        return None, (None, None)
    df_team = team_data[team_data['team'] == team]
    data_len = len(df_team)
    stack_pred = None
    arima_pred = None
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
    if team in arima_models and arima_models[team] is not None:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast)
        except Exception as e:
            print(f"Error predicting with ARIMA for team {team}: {e}")
            arima_pred = None
    ensemble = None
    if stack_pred is not None and arima_pred is not None:
        mse_stack = team_stats[team].get('mse', 1)
        try:
            resid = arima_models[team].resid()
            mse_arima = np.mean(np.square(resid))
        except Exception as e:
            mse_arima = None
        eps = 1e-6
        if mse_arima is not None and mse_arima > 0:
            weight_stack = 1 / (mse_stack + eps)
            weight_arima = 1 / (mse_arima + eps)
            ensemble = (stack_pred * weight_stack + arima_pred * weight_arima) / (weight_stack + weight_arima)
        else:
            ensemble = (stack_pred + arima_pred) / 2
    elif stack_pred is not None:
        ensemble = stack_pred
    elif arima_pred is not None:
        ensemble = arima_pred
    else:
        ensemble = None

    if team_stats[team].get('mse', 0) > 150:
        return None, (None, None)
    if ensemble is None:
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
    spread_suggestion = f"Lean {winner} by {round_half(diff):.1f}"
    ou_suggestion = f"Take the {'Over' if total_points > ou_threshold else 'Under'} {round_half(total_points):.1f}"
    return {
        'predicted_winner': winner,
        'diff': round_half(diff),
        'total_points': round_half(total_points),
        'confidence': confidence,
        'spread_suggestion': spread_suggestion,
        'ou_suggestion': ou_suggestion
    }

def find_top_bets(matchups, threshold=70.0):
    df = pd.DataFrame(matchups)
    if df.empty or 'confidence' not in df.columns:
        return pd.DataFrame()
    df_top = df[df['confidence'] >= threshold].copy()
    df_top.sort_values('confidence', ascending=False, inplace=True)
    return df_top

# =============================================================================
# NFL Data Loading Functions
# =============================================================================
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
    data['weighted_avg'] = (data['rolling_avg'] * 0.6) + (data.groupby('team')['score'].transform('mean') * 0.4)
    return data

def fetch_upcoming_nfl_games(schedule, days_ahead=7):
    upcoming = schedule[schedule['home_score'].isna() & schedule['away_score'].isna()].copy()
    now = datetime.now()
    filter_date = now + timedelta(days=days_ahead)
    upcoming = upcoming[upcoming['gameday'] <= filter_date].copy()
    upcoming.sort_values('gameday', inplace=True)
    return upcoming[['gameday', 'home_team', 'away_team']]

# =============================================================================
# NBA Data Loading Functions (with CSV fallback)
# =============================================================================
@st.cache_data(ttl=14400)
def load_nba_data():
    all_rows = []
    if USE_NBA_CSV_DATA:
        try:
            games_csv = load_csv_data_safe("data/nba_games_all.csv")
            teams_csv = load_csv_data_safe("data/nba_teams_all.csv")
            spreads_csv = load_csv_data_safe("data/nba_betting_spread.csv")
            if not games_csv.empty:
                print("Successfully loaded NBA games CSV.")
                games_csv['game_date'] = pd.to_datetime(games_csv['game_date'])
                games_csv.sort_values('game_date', inplace=True)
                if 'pts' in games_csv.columns:
                    games_csv['rolling_avg'] = games_csv.groupby('team_id')['pts'].transform(
                        lambda x: x.rolling(3, min_periods=1).mean()
                    )
                    games_csv['weighted_avg'] = (games_csv['rolling_avg'] * 0.6
                                                 + games_csv.groupby('team_id')['pts'].transform('mean') * 0.4)
                all_rows.extend(games_csv.to_dict('records'))
            if not spreads_csv.empty:
                print("Successfully loaded NBA betting spreads CSV.")
            if all_rows:
                df_csv = pd.DataFrame(all_rows)
                if 'game_date' in df_csv.columns:
                    df_csv.rename(columns={'game_date': 'gameday'}, inplace=True)
                if 'team_id' in df_csv.columns:
                    df_csv.rename(columns={'team_id': 'team'}, inplace=True)
                df_csv.dropna(subset=['pts'], inplace=True)
                df_csv.sort_values('gameday', inplace=True)
                df_csv['score'] = df_csv['pts'] if 'pts' in df_csv.columns else np.nan
                df_csv['score'].fillna(0, inplace=True)
                if 'rolling_avg' not in df_csv.columns:
                    df_csv['rolling_avg'] = df_csv.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
                if 'weighted_avg' not in df_csv.columns:
                    df_csv['weighted_avg'] = (df_csv['rolling_avg'] * 0.6
                                              + df_csv.groupby('team')['score'].transform('mean') * 0.4)
                return df_csv
        except Exception as e:
            print(f"Error loading NBA data from CSV: {e}")
    print("Falling back to NBA API data...")
    nba_teams_list = nba_teams.get_teams()
    seasons = [
        '2017-18', '2018-19', '2019-20', '2020-21',
        '2021-22', '2022-23', '2023-24', '2024-25'
    ]
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
                gl['OFF_RATING'] = np.where(
                    gl['TEAM_POSSESSIONS'] > 0,
                    (gl['PTS'] / gl['TEAM_POSSESSIONS']) * 100,
                    np.nan
                )
                gl['DEF_RATING'] = np.where(
                    gl['TEAM_POSSESSIONS'] > 0,
                    (gl['PTS_OPP'] / gl['TEAM_POSSESSIONS']) * 100,
                    np.nan
                )
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
                        print(f"Error processing row for team {team_abbrev}: {e}")
                        continue
            except Exception as e:
                print(f"Error processing team {team_abbrev} for season {season}: {e}")
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
    for offset in range(days_ahead + 7):
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

# =============================================================================
# NCAAB Data Loading Functions
# =============================================================================
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

# =============================================================================
# Compare Predictions with Bookmaker Odds
# =============================================================================
def compare_predictions_with_odds(predictions, league_choice, odds_api_key):
    sport_key_map = {
        "NFL": "americanfootball_nfl",
        "NBA": "basketball_nba",
        "NCAAB": "basketball_ncaab"
    }
    sport_key = sport_key_map.get(league_choice, "")
    selected_market = "spreads"
    odds_data = fetch_odds(odds_api_key, sport_key, selected_market)
    
    def map_team_name(team_name):
        default_mapping = {
            "New England Patriots": "New England Patriots",
            "Dallas Cowboys": "Dallas Cowboys"
        }
        nba_mapping = {
            "ATL": "Atlanta Hawks",
            "BKN": "Brooklyn Nets",
            "BOS": "Boston Celtics",
            "CHA": "Charlotte Hornets",
            "CHI": "Chicago Bulls",
            "CLE": "Cleveland Cavaliers",
            "DAL": "Dallas Mavericks",
            "DEN": "Denver Nuggets",
            "DET": "Detroit Pistons",
            "GSW": "Golden State Warriors",
            "HOU": "Houston Rockets",
            "IND": "Indiana Pacers",
            "LAC": "Los Angeles Clippers",
            "LAL": "Los Angeles Lakers",
            "MEM": "Memphis Grizzlies",
            "MIA": "Miami Heat",
            "MIL": "Milwaukee Bucks",
            "MIN": "Minnesota Timberwolves",
            "NOP": "New Orleans Pelicans",
            "NYK": "New York Knicks",
            "OKC": "Oklahoma City Thunder",
            "ORL": "Orlando Magic",
            "PHI": "Philadelphia 76ers",
            "PHX": "Phoenix Suns",
            "POR": "Portland Trail Blazers",
            "SAC": "Sacramento Kings",
            "SAS": "San Antonio Spurs",
            "TOR": "Toronto Raptors",
            "UTA": "Utah Jazz",
            "WAS": "Washington Wizards"
        }
        if league_choice == "NBA":
            return nba_mapping.get(team_name, team_name)
        else:
            return default_mapping.get(team_name, team_name)
    
    def get_bookmaker_spread(mapped_team_name, odds_data):
        for event in odds_data:
            home_api_name = map_team_name(event.get("home_team", ""))
            away_api_name = map_team_name(event.get("away_team", ""))
            if mapped_team_name in (home_api_name, away_api_name):
                for bm in event.get("bookmakers", []):
                    if bm.get("key") == "bovada":
                        for market in bm.get("markets", []):
                            if market.get("key") == "spreads":
                                for outcome in market.get("outcomes", []):
                                    if mapped_team_name == map_team_name(outcome.get("name", "")):
                                        return outcome.get("point")
        return None

    for pred in predictions:
        home_mapped = map_team_name(pred["home_team"])
        away_mapped = map_team_name(pred["away_team"])
        home_line = get_bookmaker_spread(home_mapped, odds_data)
        away_line = get_bookmaker_spread(away_mapped, odds_data)
        if home_line is not None and away_line is not None:
            bookmaker_spread = home_line - away_line
        else:
            bookmaker_spread = None
        pred["bookmaker_spread"] = bookmaker_spread
    return predictions

# =============================================================================
# UI Components
# =============================================================================
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
  Based on the analysis, **{predicted_winner}** is predicted to win with a confidence level of **{confidence}%**.
  The projected point difference is **{bet['predicted_diff']}**, leading to a spread suggestion of **{bet['spread_suggestion']}**.
  The total predicted points are **{bet['predicted_total']}**, indicating a suggestion to **{bet['ou_suggestion']}**.

- **Statistical Edge:**
  The confidence reflects the derived statistical edge.
"""
    return writeup

def generate_social_media_post_ui(bet):
    return generate_social_media_post(bet)

def display_bet_card(bet, team_stats_global, team_data=None):
    conf = bet['confidence']
    if conf >= 80:
        confidence_color = "green"
    elif conf < 60:
        confidence_color = "red"
    else:
        confidence_color = "orange"
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
            tooltip_text = "Confidence indicates the statistical edge."
            st.markdown(f"<h3 style='color:{confidence_color};' title='{tooltip_text}'>{bet['confidence']:.1f}% Confidence</h3>", unsafe_allow_html=True)
    with st.expander("Detailed Insights", expanded=False):
        st.markdown(f"**Predicted Winner:** {bet['predicted_winner']}")
        st.markdown(f"**Predicted Total Points:** {bet['predicted_total']}")
        st.markdown(f"**Prediction Margin:** {bet['predicted_diff']}")
    with st.expander("Generate Social Media Post", expanded=False):
        if st.button("Generate Post", key=f"social_post_{bet['home_team']}_{bet['away_team']}_{bet['date']}"):
            post = generate_social_media_post_ui(bet)
            st.markdown(post)
    if st.button("Manual Odds Update", key=f"manual_odds_{bet['date']}"):
        st.text_input("Enter Home Spread", key=f"manual_home_{bet['date']}")
        st.text_input("Enter Game Total", key=f"manual_total_{bet['date']}")
    if team_data is not None:
        with st.expander("Recent Performance Trends", expanded=False):
            home_team_data = team_data[team_data['team'] == bet['home_team']].sort_values('gameday')
            if not home_team_data.empty:
                st.markdown(f"**{bet['home_team']} Recent Scores:**")
                st.line_chart(home_team_data['score'].tail(5).reset_index(drop=True))
            away_team_data = team_data[team_data['team'] == bet['away_team']].sort_values('gameday')
            if not away_team_data.empty:
                st.markdown(f"**{bet['away_team']} Recent Scores:**")
                st.line_chart(away_team_data['score'].tail(5).reset_index(drop=True))

# =============================================================================
# Main Pipeline: Betting Predictions and UI
# =============================================================================
results = []
team_stats_global = {}

def run_league_pipeline(league_choice, odds_api_key):
    st.header(f"Today's {league_choice} Best Bets")
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
    else:
        team_data = load_ncaab_data_current_season(season=2025)
        if team_data.empty:
            st.error("Unable to load NCAAB data.")
            return
        upcoming = fetch_upcoming_ncaab_games()
    if team_data.empty or upcoming.empty:
        st.warning(f"No upcoming {league_choice} data available.")
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
    with st.spinner("Analyzing recent performance data..."):
        if league_choice == "NCAAB" and DISABLE_TUNING_FOR_NCAAB:
            stack_models, arima_models, team_stats = train_team_models(team_data, disable_tuning=True)
        else:
            stack_models, arima_models, team_stats = train_team_models(team_data, disable_tuning=False)
        team_stats_global.update(team_stats)
        results.clear()
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
    view_mode = st.radio("View Mode", ["Top Bets Only", "All Games"], horizontal=True)
    if view_mode == "Top Bets Only":
        conf_threshold = st.slider("Minimum Confidence Level", 50.0, 99.0, 75.0, 5.0)
        top_bets = find_top_bets(results, threshold=conf_threshold)
        if not top_bets.empty:
            st.markdown(f"### Top {len(top_bets)} Bets for Today")
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
            st.markdown("### All Games Analysis")
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
    if st.button("Compare to Bookmaker Odds"):
        compared_results = compare_predictions_with_odds(results.copy(), league_choice, st.secrets["odds_api"]["apiKey"])
        st.session_state["compared_results"] = compared_results
        st.markdown("## Comparison with Bookmaker Odds")
        for idx, bet in enumerate(compared_results):
            st.markdown("---")
            st.markdown(f"### {bet['away_team']} @ {bet['home_team']}")
            st.write(f"Predicted Spread: {bet['predicted_diff']}")
            if bet.get("bookmaker_spread") is not None:
                st.write(f"Bookmaker Spread: {bet['bookmaker_spread']}")
            else:
                st.write("Bookmaker Spread: Data not available")
            st.write(f"Confidence: {bet['confidence']}%")
    if st.button("Save Predictions to CSV"):
        if results:
            save_predictions_to_csv(results)
        else:
            st.warning("No predictions to save.")

# =============================================================================
# Instagram Scheduler Section
# =============================================================================
def login_to_instagram(username, password):
    client = Client()
    try:
        client.login(username, password)
        st.success("Instagram login successful!")
        return client
    except Exception as e:
        st.error(f"Instagram login failed: {e}")
        return None

def post_to_instagram(client, caption, image_path):
    try:
        media = client.photo_upload(image_path, caption)
        st.success("Post successfully uploaded to Instagram!")
        return media
    except Exception as e:
        st.error(f"Failed to post to Instagram: {e}")
        return None

def load_todays_predictions(csv_file="predictions.csv"):
    file = Path(csv_file)
    if not file.exists():
        st.info("No predictions file found in the project directory.")
        return []
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return []
    try:
        df['date'] = pd.to_datetime(df['date']).dt.date
    except Exception as e:
        st.error(f"Error parsing dates: {e}")
        return []
    today = datetime.now().date()
    todays_df = df[df['date'] == today]
    return todays_df.to_dict('records')

def show_instagram_scheduler_section():
    st.markdown("---")
    st.subheader("Instagram Scheduler")
    
    # Instagram Login
    ig_username = st.text_input("Instagram Username", key="ig_username")
    ig_password = st.text_input("Instagram Password", type="password", key="ig_password")
    if st.button("Login to Instagram"):
        client = login_to_instagram(ig_username, ig_password)
        if client:
            st.session_state['instagram_client'] = client

    # Load today's predictions from CSV
    if st.button("Load Today's Predictions"):
        predictions = load_todays_predictions()
        if predictions:
            st.session_state['todays_predictions'] = predictions
            st.success(f"Loaded {len(predictions)} predictions for today.")
        else:
            st.info("No predictions available for today.")

    if "todays_predictions" in st.session_state:
        selected_prediction = st.selectbox(
            "Select a game to generate post content",
            st.session_state['todays_predictions'],
            format_func=lambda p: f"{p['away_team']} @ {p['home_team']} - Winner: {p['predicted_winner']}"
        )
        if selected_prediction:
            post_text = generate_social_media_post(selected_prediction)
            st.markdown("### Generated Social Media Post")
            st.text_area("Post Content", post_text, height=200)
            
            # If the league is NBA, try to merge both team logos
            merged_image_path = None
            # Here we assume that the matchup logos are in nba_images folder.
            # Merge home and away logos.
            merged_image_path = merge_team_logos(selected_prediction['home_team'], selected_prediction['away_team'])
            if merged_image_path:
                st.markdown("Automatically merged team logos:")
                st.image(merged_image_path, width=300)
            else:
                st.info("Could not automatically merge logos; please provide an image path manually.")
            
            st.markdown("### Schedule / Post Now")
            scheduled_date = st.date_input("Scheduled Date", datetime.now().date())
            scheduled_time = st.time_input("Scheduled Time", datetime.now().time())
            # Default to merged image path if available.
            image_path = st.text_input("Image Path (local file path)", value=merged_image_path if merged_image_path else "", key="img_path")
            
            if st.button("Schedule Post"):
                scheduled_datetime = datetime.combine(scheduled_date, scheduled_time)
                client = st.session_state.get('instagram_client')
                if client is None:
                    st.error("Please log in to Instagram first.")
                elif not image_path or not Path(image_path).exists():
                    st.error("Please provide a valid image file path.")
                else:
                    st.info("Scheduling post...")
                    now = datetime.now()
                    delay = (scheduled_datetime - now).total_seconds()
                    if delay < 0:
                        st.warning("Scheduled time is in the past. Posting immediately.")
                        delay = 0
                    st.info(f"Post will be submitted in {int(delay)} seconds.")
                    time.sleep(delay)  # For demonstration; in production use APScheduler or background threading.
                    post_to_instagram(client, post_text, image_path)

# =============================================================================
# Helper to Merge Logos (from nba_images folder)
# =============================================================================
def get_team_logo(team_name):
    base_dir = "nba_images"
    filename = team_name.strip().replace(" ", "_") + ".png"
    path = os.path.join(base_dir, filename)
    return path if os.path.exists(path) else None

def merge_team_logos(team1, team2, output_path="merged_logo.png", target_height=200, bg_color=(255, 255, 255)):
    """
    Merge logos for team1 and team2 side-by-side.
    """
    logo1_path = get_team_logo(team1)
    logo2_path = get_team_logo(team2)
    if not logo1_path or not logo2_path:
        return None
    img1 = Image.open(logo1_path)
    img2 = Image.open(logo2_path)
    img1_ratio = img1.width / img1.height
    img2_ratio = img2.width / img2.height
    img1 = img1.resize((int(target_height * img1_ratio), target_height))
    img2 = img2.resize((int(target_height * img2_ratio), target_height))
    total_width = img1.width + img2.width
    merged_img = Image.new('RGB', (total_width, target_height), color=bg_color)
    merged_img.paste(img1, (0, 0))
    merged_img.paste(img2, (img1.width, 0))
    merged_img.save(output_path)
    return output_path

# =============================================================================
# Main Function
# =============================================================================
def main():
    st.set_page_config(page_title="FoxEdge Sports Betting Insights", page_icon="🦊", layout="centered")
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
    odds_api_key = st.sidebar.text_input(
        "Enter Odds API Key",
        type="password",
        value=st.secrets["odds_api"]["apiKey"] if "odds_api" in st.secrets else ""
    )
    st.sidebar.header("Navigation")
    league_choice = st.sidebar.radio("Select League", ["NFL", "NBA", "NCAAB"])
    run_league_pipeline(league_choice, odds_api_key)
    st.sidebar.markdown("### About FoxEdge\nFoxEdge provides data-driven insights for NFL, NBA, and NCAAB games.")
    
    # Add Instagram Scheduler at the bottom of the page.
    show_instagram_scheduler_section()

if __name__ == "__main__":
    initialize_csv()  # Ensure predictions.csv exists
    main()
