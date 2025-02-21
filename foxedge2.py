import os
import sys
import warnings
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
import optuna
import inspect
import cbbpy.mens_scraper as cbb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from instagrapi import Client
import time
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from moviepy.editor import ColorClip, TextClip, ImageClip, CompositeVideoClip, AudioFileClip
from gtts import gTTS
from contextlib import contextmanager

# --- Fix for sys.path ---
if sys.path and sys.path[0].endswith('.py'):
    sys.path.pop(0)
sys.path.insert(0, os.getcwd())

# =============================================================================
# Feature Flags and Global Constants
# =============================================================================
LOCAL_TEST_MODE = True            # Bypass Firebase login for local testing
TEST_FIRST_PREDICTION_FROM_CSV = False
USE_RANDOMIZED_SEARCH = True
USE_OPTUNA_SEARCH = False
ENABLE_EARLY_STOPPING = True
DISABLE_TUNING_FOR_NCAAB = True
USE_NBA_CSV_DATA = True
CSV_FILE = "predictions.csv"  # Used for saving all predictions

# =============================================================================
# NBA Mapping Dictionary for Logo Files
# =============================================================================
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

# =============================================================================
# Utility Functions
# =============================================================================
def to_naive(dt):
    """Convert timezone-aware datetime to naive datetime."""
    if dt is not None and hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt

def round_half(number):
    """Round number to nearest 0.5."""
    return round(number * 2) / 2

def supports_early_stopping(model):
    """Check if model's fit method supports early stopping."""
    try:
        sig = inspect.signature(model.fit)
        return 'early_stopping_rounds' in sig.parameters
    except Exception:
        return False

def updated_confidence(baseline_conf, monte_mean_diff, monte_std, win_probability, K=10, epsilon=1e-6):
    """Update the confidence metric using Monte Carlo simulation outputs."""
    snr = abs(monte_mean_diff) / (monte_std + epsilon)
    adjusted = baseline_conf + K * ((win_probability / 100) * snr - 1)
    return max(1, min(99, adjusted))

# =============================================================================
# Additional Feature Engineering
# =============================================================================
def add_additional_features(team_data: pd.DataFrame) -> pd.DataFrame:
    """Enhance team data with lag features, rolling averages, and rest days."""
    team_data['gameday'] = pd.to_datetime(team_data['gameday'], errors='coerce')
    team_data.sort_values(['team', 'gameday'], inplace=True)
    team_data['prev_score'] = team_data.groupby('team')['score'].shift(1)
    team_data['rolling_avg'] = team_data.groupby('team')['score'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    team_data['rolling_std'] = team_data.groupby('team')['score'].transform(lambda x: x.rolling(window=3, min_periods=1).std().fillna(0))
    team_data['prev_gameday'] = team_data.groupby('team')['gameday'].shift(1)
    team_data['rest_days'] = (team_data['gameday'] - team_data['prev_gameday']).dt.days.fillna(0)
    return team_data

# =============================================================================
# Dummy Odds Fetcher
# =============================================================================
def fetch_odds(api_key, sport_key, market):
    """Dummy function: Returns an empty list."""
    return []

# =============================================================================
# Cached Distribution Fitting Helper
# =============================================================================
@st.cache_data(show_spinner=False, hash_funcs={np.ndarray: lambda x: x.tobytes()})
def fit_distribution(simulation_array):
    """Fit a distribution to simulation results using Fitter."""
    from fitter import Fitter
    f = Fitter(simulation_array.flatten(), distributions=['norm', 't', 'lognorm'])
    f.fit()
    return f.get_best(method='sumsquare_error')

# =============================================================================
# Monte Carlo Simulation Functions
# =============================================================================
def monte_carlo_simulation_margin(model_home, model_away, X_home, X_away, n_simulations=10000,
                                  error_std_home=5, error_std_away=5, random_seed=42, run_fitter=False):
    """Run Monte Carlo simulation on the point margin (home minus away)."""
    np.random.seed(random_seed)
    base_pred_home = model_home.predict(X_home)
    base_pred_away = model_away.predict(X_away)
    simulated_diffs = []
    for _ in range(n_simulations):
        noise_home = np.random.normal(0, error_std_home, size=base_pred_home.shape)
        noise_away = np.random.normal(0, error_std_away, size=base_pred_away.shape)
        simulated_diffs.append((base_pred_home + noise_home) - (base_pred_away + noise_away))
    simulated_diffs = np.array(simulated_diffs)
    mean_diff = simulated_diffs.mean()
    ci_lower = np.percentile(simulated_diffs, 2.5)
    ci_upper = np.percentile(simulated_diffs, 97.5)
    ci = (ci_lower, ci_upper)
    if mean_diff >= 0:
        win_rate = (simulated_diffs > 0).mean() * 100
        win_margin_rate = (simulated_diffs >= mean_diff).mean() * 100
    else:
        win_rate = (simulated_diffs < 0).mean() * 100
        win_margin_rate = (simulated_diffs <= mean_diff).mean() * 100
    return {"mean_diff": mean_diff,
            "ci": ci,
            "median_diff": np.median(simulated_diffs),
            "std_diff": simulated_diffs.std(),
            "win_rate": win_rate,
            "win_margin_rate": win_margin_rate,
            "simulated_diffs": simulated_diffs,
            "fitted_distribution": fit_distribution(simulated_diffs) if run_fitter else None}

def monte_carlo_simulation_totals(model_home, model_away, X_home, X_away, n_simulations=10000,
                                  error_std_home=5, error_std_away=5, random_seed=42, run_fitter=False):
    """Run Monte Carlo simulation on the game total (home plus away)."""
    np.random.seed(random_seed)
    base_pred_home = model_home.predict(X_home)
    base_pred_away = model_away.predict(X_away)
    simulated_totals = []
    for _ in range(n_simulations):
        noise_home = np.random.normal(0, error_std_home, size=base_pred_home.shape)
        noise_away = np.random.normal(0, error_std_away, size=base_pred_away.shape)
        simulated_totals.append((base_pred_home + noise_home) + (base_pred_away + noise_away))
    simulated_totals = np.array(simulated_totals)
    return {"mean_total": simulated_totals.mean(),
            "median_total": np.median(simulated_totals),
            "std_total": simulated_totals.std(),
            "over_threshold": np.percentile(simulated_totals, 5),
            "under_threshold": np.percentile(simulated_totals, 95),
            "simulated_totals": simulated_totals,
            "fitted_distribution": fit_distribution(simulated_totals) if run_fitter else None}

# =============================================================================
# Betting Insights and Strategy Functions
# =============================================================================
def generate_betting_insights(bet):
    insights = []
    if bet.get('confidence', 0) >= 80 and bet.get('win_probability', 0) >= 75:
        insights.append(f"High confidence ({bet['confidence']:.1f}%) and a win probability of {bet['win_probability']:.1f}% indicate a strong play on {bet['predicted_winner']}.")
    else:
        insights.append(f"Moderate confidence ({bet['confidence']:.1f}%) and a win probability of {bet.get('win_probability', 0):.1f}% suggest caution.")
    if bet.get('monte_mean_diff') is not None and bet.get('monte_std') is not None:
        snr = abs(bet['monte_mean_diff']) / (bet['monte_std'] + 1e-6)
        insights.append(f"The signal-to-noise ratio is {snr:.2f}, indicating {'strong' if snr >= 1.5 else 'moderate'} predictive strength.")
        if bet.get('monte_ci'):
            try:
                ci_lower = float(bet['monte_ci'][0])
                ci_upper = float(bet['monte_ci'][1])
                ci_width = ci_upper - ci_lower
                rel_ci = ci_width / (abs(bet['monte_mean_diff']) + 1e-6)
                insights.append(f"The 95% confidence interval is {ci_width:.2f} points wide (about {rel_ci:.2f} times the mean margin).")
                insights.append("This narrow interval indicates high consistency in simulation outcomes." if rel_ci < 1.5 else "The wide interval signals significant uncertainty; consider a smaller stake.")
            except (ValueError, TypeError):
                insights.append("Confidence interval data unavailable or invalid.")
    if bet.get('simulated_diffs') is not None:
        try:
            sim_diffs = np.array(bet['simulated_diffs'], dtype=float).flatten()
            sim_skew = skew(sim_diffs)
            sim_kurt = kurtosis(sim_diffs)
            insights.append(f"The margin distribution is {sim_skew:+.2f} skewed with a kurtosis of {sim_kurt:.2f}, indicating {'a tendency for extreme outcomes' if sim_kurt > 3 else 'a relatively normal distribution'}.")
        except Exception:
            insights.append("Error processing simulated differences.")
    if (bet.get('monte_mean_diff') is not None and bet.get('monte_std') is not None and bet.get('win_probability') is not None):
        snr = abs(bet['monte_mean_diff']) / (bet['monte_std'] + 1e-6)
        value_score = (bet['win_probability'] / 100) * (bet['confidence'] / 100) * snr
        if bet.get('monte_ci'):
            try:
                ci_lower = float(bet['monte_ci'][0])
                ci_upper = float(bet['monte_ci'][1])
                ci_width = ci_upper - ci_lower
                rel_ci = ci_width / (abs(bet['monte_mean_diff']) + 1e-6)
                value_score /= rel_ci
            except (ValueError, TypeError):
                pass
        insights.append(f"Composite Betting Strength Index (Value Score): {value_score:.2f}.")
    if not insights:
        insights.append("Metrics are inconclusive; additional research is advised.")
    return " ".join(insights)

def recommended_betting_strategy(bet):
    if bet.get('win_probability') is None or bet.get('confidence') is None:
        return "Insufficient data to recommend a strategy."
    if bet['win_probability'] < 50:
        return "No Bet: Win probability is below 50%."
    if bet.get('monte_ci'):
        spread_range = f"between {bet['monte_ci'][0]:.1f} and {bet['monte_ci'][1]:.1f} points"
    else:
        spread_range = f"around {bet['predicted_diff']:.1f} points"
    spread_recommendation = f"Bet on {bet['predicted_winner']} to cover a spread of {spread_range}."
    over_under_recommendation = f"Bet the game total to be {'over' if bet['predicted_total'] >= 145 else 'under'} {bet['predicted_total']:.1f} points."
    if bet['confidence'] >= 85 and bet['win_probability'] >= 80:
        level = "Aggressive Bet"
        wager = "5% of your bankroll"
    elif bet['confidence'] >= 70 and bet['win_probability'] >= 70:
        level = "Moderate Bet"
        wager = "3% of your bankroll"
    elif bet['confidence'] >= 60 and bet['win_probability'] >= 60:
        level = "Cautious Bet"
        wager = "1.5% of your bankroll"
    else:
        return "No Bet: The metrics do not support a favorable betting opportunity."
    value_score = None
    if bet.get('monte_mean_diff') is not None and bet.get('monte_std') is not None:
        snr = abs(bet['monte_mean_diff']) / (bet['monte_std'] + 1e-6)
        value_score = (bet['win_probability'] / 100) * (bet['confidence'] / 100) * snr
        if bet.get('monte_ci'):
            ci_width = bet['monte_ci'][1] - bet['monte_ci'][0]
            rel_ci = ci_width / (abs(bet['monte_mean_diff']) + 1e-6)
            value_score /= rel_ci
    value_score_text = f"Value Score: {value_score:.2f}" if value_score is not None else "Value Score: N/A"
    return f"{level} Strategy:\n{spread_recommendation}\n{over_under_recommendation}\n{value_score_text}\nRecommended wager sizing: {wager}."

def find_top_bets(matchups, threshold=70.0):
    df = pd.DataFrame(matchups)
    if df.empty or 'confidence' not in df.columns:
        return pd.DataFrame()
    df_top = df[df['confidence'] >= threshold].copy()
    df_top.sort_values('confidence', ascending=False, inplace=True)
    return df_top

# =============================================================================
# New KPI Functions
# =============================================================================
def compute_mcciw(bet):
    if bet.get('monte_ci'):
        try:
            ci_lower = float(bet['monte_ci'][0])
            ci_upper = float(bet['monte_ci'][1])
            return ci_upper - ci_lower
        except (ValueError, TypeError):
            return None
    return None

def compute_psnr(bet):
    if bet.get('monte_mean_diff') is not None and bet.get('monte_std') is not None:
        return abs(bet['monte_mean_diff']) / (bet['monte_std'] + 1e-6)
    return None

def recommended_stake_percent(bet):
    conf = bet.get('confidence', 0)
    win_prob = bet.get('win_probability', 0)
    if conf >= 85 and win_prob >= 80:
        return 5.0
    elif conf >= 70 and win_prob >= 70:
        return 3.0
    elif conf >= 60 and win_prob >= 60:
        return 1.5
    else:
        return 0

def compute_value_score(bet):
    if bet.get('monte_mean_diff') is not None and bet.get('monte_std') is not None and bet.get('win_probability') is not None:
        snr = abs(bet['monte_mean_diff']) / (bet['monte_std'] + 1e-6)
        value_score = (bet['win_probability'] / 100) * (bet['confidence'] / 100) * snr
        if bet.get('monte_ci'):
            try:
                ci_lower = float(bet['monte_ci'][0])
                ci_upper = float(bet['monte_ci'][1])
                ci_width = ci_upper - ci_lower
                rel_ci = ci_width / (abs(bet['monte_mean_diff']) + 1e-6)
                value_score /= rel_ci
            except (ValueError, TypeError):
                pass
        return value_score
    return None

def compute_aser(bet):
    value_score = compute_value_score(bet)
    stake_percent = recommended_stake_percent(bet)
    if value_score is not None and stake_percent > 0:
        return value_score / stake_percent
    return None

def compute_evr(predictions_df):
    if 'predicted_diff' in predictions_df.columns and not predictions_df['predicted_diff'].empty:
        mean_edge = np.mean(np.abs(predictions_df['predicted_diff']))
        std_edge = np.std(predictions_df['predicted_diff'])
        if mean_edge > 0:
            return std_edge / mean_edge
    return None

def compute_pmdcr(predictions_df):
    if ('predicted_diff' in predictions_df.columns and 
        'bookmaker_spread' in predictions_df.columns and 
        not predictions_df['predicted_diff'].empty):
        spread_deviation = np.abs(predictions_df['predicted_diff'] - predictions_df['bookmaker_spread'])
        mean_deviation = np.mean(spread_deviation)
        mean_predicted = np.mean(np.abs(predictions_df['predicted_diff']))
        if mean_predicted > 0:
            return 100 * (1 - (mean_deviation / mean_predicted))
    return None

def compute_rosa(bet):
    value_score = compute_value_score(bet)
    stake_percent = recommended_stake_percent(bet)
    if value_score is not None and stake_percent > 0:
        return value_score / stake_percent
    return None

def compute_drar(predictions_df):
    if 'prediction_correct' not in predictions_df.columns:
        return None
    balance = 1.0
    cumulative = []
    for idx, row in predictions_df.iterrows():
        stake = recommended_stake_percent(row.to_dict()) / 100.0
        if row.get('prediction_correct'):
            balance += stake
        else:
            balance -= stake
        cumulative.append(balance)
    cumulative = np.array(cumulative)
    if len(cumulative) == 0:
        return None
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    max_drawdown = np.max(drawdown) if peak.max() > 0 else 0
    total_return = cumulative[-1] - 1.0
    if max_drawdown > 0:
        return total_return / max_drawdown
    return None

# =============================================================================
# CSV Data Functions (UPDATED)
# =============================================================================
def initialize_csv(csv_file=CSV_FILE):
    csv_columns = [
        "date", "league", "home_team", "away_team", "home_pred", "away_pred",
        "predicted_winner", "predicted_diff", "predicted_total", "spread_suggestion", "ou_suggestion",
        "monte_mean_diff", "monte_ci", "win_probability", "win_margin_rate",
        "monte_median", "monte_std", "mean_total", "over_threshold", "under_threshold",
        "median_total", "std_total",
        "margin_confidence", "totals_confidence", "overall_confidence",
        "actual_home_score", "actual_away_score", "actual_margin", "actual_total",
        "margin_delta", "total_delta", "prediction_correct",
        "mcciw", "psnr", "aser", "rosa", "evr", "pmdcr", "drar"
    ]
    file = Path(csv_file)
    if not file.exists():
        pd.DataFrame(columns=csv_columns).to_csv(csv_file, index=False)

def save_prediction_to_csv(prediction, csv_file="top_bets.csv"):
    prediction_copy = prediction.copy()
    if isinstance(prediction_copy.get('date'), datetime):
        prediction_copy['date'] = prediction_copy['date'].strftime('%m/%d/%y')
    elif isinstance(prediction_copy.get('date'), str):
        try:
            d = datetime.strptime(prediction_copy['date'], '%Y-%m-%d')
            prediction_copy['date'] = d.strftime('%m/%d/%y')
        except Exception:
            pass
    if prediction_copy.get("monte_ci") is not None:
        prediction_copy["monte_ci"] = str(prediction_copy["monte_ci"])
    csv_columns = [
        "date", "league", "home_team", "away_team", "home_pred", "away_pred",
        "predicted_winner", "predicted_diff", "predicted_total", "spread_suggestion", "ou_suggestion",
        "monte_mean_diff", "monte_ci", "win_probability", "win_margin_rate",
        "monte_median", "monte_std", "mean_total", "over_threshold", "under_threshold",
        "median_total", "std_total",
        "margin_confidence", "totals_confidence", "overall_confidence",
        "actual_home_score", "actual_away_score", "actual_margin", "actual_total",
        "margin_delta", "total_delta", "prediction_correct",
        "mcciw", "psnr", "aser", "rosa", "evr", "pmdcr", "drar"
    ]
    file = Path(csv_file)
    if not file.exists():
        df = pd.DataFrame([prediction_copy], columns=csv_columns)
        df.to_csv(csv_file, index=False)
    else:
        try:
            df_existing = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Warning: Could not read existing CSV due to: {e}")
            df_existing = pd.DataFrame(columns=csv_columns)
        df_new = pd.DataFrame([prediction_copy])
        for col in csv_columns:
            if col not in df_new.columns:
                df_new[col] = None
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined['date'] = pd.to_datetime(df_combined['date'], errors='coerce').dt.strftime('%m/%d/%y')
        df_combined = df_combined[csv_columns]
        df_combined.to_csv(csv_file, index=False)
    st.success("Prediction saved to CSV!")

def save_predictions_to_csv(predictions, csv_file=CSV_FILE):
    csv_columns = [
        "date", "league", "home_team", "away_team", "home_pred", "away_pred",
        "predicted_winner", "predicted_diff", "predicted_total", "spread_suggestion", "ou_suggestion",
        "monte_mean_diff", "monte_ci", "win_probability", "win_margin_rate",
        "monte_median", "monte_std", "mean_total", "over_threshold", "under_threshold",
        "median_total", "std_total",
        "margin_confidence", "totals_confidence", "overall_confidence",
        "actual_home_score", "actual_away_score", "actual_margin", "actual_total",
        "margin_delta", "total_delta", "prediction_correct",
        "mcciw", "psnr", "aser", "rosa", "evr", "pmdcr", "drar"
    ]
    df = pd.DataFrame(predictions)
    if "monte_ci" in df.columns:
        df["monte_ci"] = df["monte_ci"].apply(lambda x: str(x) if pd.notnull(x) else x)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%m/%d/%y')
    for col in csv_columns:
        if col not in df.columns:
            df[col] = None
    df = df[csv_columns]
    df.to_csv(csv_file, index=False)
    st.success("All predictions saved to CSV!")

# =============================================================================
# Team Models & Prediction Functions
# =============================================================================
@st.cache_data(ttl=3600)
def train_team_models(team_data: pd.DataFrame, disable_tuning=False):
    team_data = add_additional_features(team_data)
    if USE_NBA_CSV_DATA and not team_data.empty:
        if 'spread1' in team_data.columns and 'spread2' in team_data.columns:
            team_data['spread_diff'] = abs(team_data['spread1'] - team_data['spread2'])
        if 'rolling_avg' not in team_data.columns:
            team_data['rolling_avg'] = team_data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
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
                xgb_best = tune_model(xgb, xgb_grid, X_train, y_train, use_randomized=USE_RANDOMIZED_SEARCH, early_stopping=False)
            except Exception as e:
                print(f"Error tuning XGB for team {team}: {e}")
                xgb_best = XGBRegressor(n_estimators=100, random_state=42)
            try:
                lgbm = LGBMRegressor(random_state=42)
                lgbm_grid = {'n_estimators': [50, 100, 150],
                             'max_depth': [None, 5, 10],
                             'num_leaves': [31, 50, 70],
                             'min_child_samples': [20, 30, 50],
                             'reg_alpha': [0, 0.1, 0.5],
                             'reg_lambda': [0, 0.1, 0.5]}
                lgbm_best = tune_model(lgbm, lgbm_grid, X_train, y_train, use_randomized=USE_RANDOMIZED_SEARCH, early_stopping=ENABLE_EARLY_STOPPING)
            except Exception as e:
                print(f"Error tuning LGBM for team {team}: {e}")
                lgbm_best = LGBMRegressor(n_estimators=100, random_state=42)
            try:
                cat = CatBoostRegressor(verbose=0, random_state=42)
                cat_grid = {'iterations': [50, 100, 150], 'learning_rate': [0.1, 0.05, 0.01]}
                cat_best = tune_model(cat, cat_grid, X_train, y_train, use_randomized=USE_RANDOMIZED_SEARCH, early_stopping=False)
            except Exception as e:
                print(f"Error tuning CatBoost for team {team}: {e}")
                cat_best = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)
        estimators = [('xgb', xgb_best), ('lgbm', lgbm_best), ('cat', cat_best)]
        stack = StackingRegressor(estimators=estimators, final_estimator=LGBMRegressor(), passthrough=False, cv=3)
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
                arima = auto_arima(scores, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True, max_p=3, max_q=3)
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
    if len(df_team) < 3:
        return None, (None, None)
    last_features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg']].tail(1)
    X_next = last_features.values
    stack_pred = None
    arima_pred = None
    if team in stack_models:
        try:
            stack_pred = float(stack_models[team].predict(X_next)[0])
        except Exception as e:
            print(f"Error predicting with Stacking Regressor for team {team}: {e}")
    if team in arima_models and arima_models[team] is not None:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast)
        except Exception as e:
            print(f"Error predicting with ARIMA for team {team}: {e}")
    ensemble = None
    if stack_pred is not None and arima_pred is not None:
        mse_stack = team_stats[team].get('mse', 1)
        try:
            resid = arima_models[team].resid()
            mse_arima = np.mean(np.square(resid))
        except Exception:
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
    if team_stats[team].get('mse', 0) > 150 or ensemble is None:
        return None, (None, None)
    bias = team_stats[team].get('bias', 0)
    ensemble_calibrated = ensemble + bias
    mu = team_stats[team]['mean']
    sigma = team_stats[team]['std']
    if hasattr(mu, 'item'):
        mu = mu.item()
    if hasattr(sigma, 'item'):
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
    baseline_conf = round(min(99, max(1, 50 + raw_conf * 15)), 2)
    penalty = 0
    if team_stats.get(home_team, {}).get('mse', 0) > 120:
        penalty += 10
    if team_stats.get(away_team, {}).get('mse', 0) > 120:
        penalty += 10
    baseline_conf = max(1, min(99, baseline_conf - penalty))
    winner = home_team if diff > 0 else away_team
    ou_threshold = 145
    spread_suggestion = f"Lean {winner} by {round_half(diff):.1f}"
    ou_suggestion = f"Take the {'Over' if total_points > ou_threshold else 'Under'} {round_half(total_points):.1f} points"
    return {
        'predicted_winner': winner,
        'diff': round_half(diff),
        'total_points': round_half(total_points),
        'confidence': baseline_conf,
        'spread_suggestion': spread_suggestion,
        'ou_suggestion': ou_suggestion
    }

# =============================================================================
# Social Media Post Generation Functions
# =============================================================================
def generate_social_media_post(bet: dict) -> str:
    visual_text, _ = generate_social_media_texts(bet)
    return visual_text

def generate_social_media_texts(bet: dict) -> (str, str):
    confidence = bet.get('confidence', 0)
    home_team = bet.get('home_team', 'Unknown')
    away_team = bet.get('away_team', 'Unknown')
    league = bet.get('league', 'NBA')
    predicted_winner = bet.get('predicted_winner', 'Unknown')
    spread_suggestion = bet.get('spread_suggestion', 'N/A')
    predicted_total = bet.get('predicted_total', 'N/A')
    if confidence >= 85:
        tone_visual = "🚨 LOCK ALERT 🚨 This is the real deal—our elite model is flashing red hot signals on this pick! Don’t sit this one out!"
    elif confidence >= 70:
        tone_visual = "🔥 VALUE PICK 🔥 Our algorithm has identified a major edge—get in on this before the line shifts!"
    else:
        tone_visual = "⚠️ Sneaky Play ⚠️ This matchup is flying under the radar—sharp bettors are already moving in!"
    if confidence >= 85:
        tone_audio = "This is a high-confidence bet backed by elite data models and sharp money movement."
    elif confidence >= 70:
        tone_audio = "This play offers significant value based on our proprietary betting signals."
    else:
        tone_audio = "This play has solid potential, but we recommend watching for line movement."
    def get_full_team_name(team: str, league: str = "NBA") -> str:
        if league == "NBA" and team in nba_mapping:
            return nba_mapping[team]
        return team
    home_team_full = get_full_team_name(home_team, league)
    away_team_full = get_full_team_name(away_team, league)
    predicted_winner_full = get_full_team_name(predicted_winner, league)
    templates = [
        (
            "💰 SHARP MONEY ALERT 💰\n\n"
            "🚀 Matchup: {away_team} @ {home_team} 🚀\n\n"
            "🔥 AI Model Pick: {predicted_winner} 🔥\n\n"
            "📈 Spread: {spread_suggestion} (Syndicates are hammering this line!)\n"
            "🎯 Projected Total: {predicted_total} points (Model validation: 92% confidence!)\n"
            "💡 Sharp Money Confidence: {confidence:.1f}%\n\n"
            "{tone_visual}\n\n"
            "💸 LOCK IT IN NOW before the line moves! Drop a ‘🔥’ in the comments if you’re tailing!\n\n"
            "SportsBetting SharpPlays BettingStrategy AIpicks",
            "Tonight’s showdown between {away_team_full} and {home_team_full} is lit! \n\n"
            "Our elite AI model is backing {predicted_winner_full} to cover with a confidence level of {confidence:.1f}%. "
            "The spread recommendation is {spread_suggestion}, with a projected total of {predicted_total} points. "
            "{tone_audio} Don’t wait—get in before this line shifts!"
        ),
        (
            "⚡ BIG MONEY ALERT ⚡\n\n"
            "💥 Matchup: {away_team} vs. {home_team} 💥\n\n"
            "🚨 Pro Betting Signal: {predicted_winner} 🚨\n\n"
            "📊 Spread: {spread_suggestion} (Tracked by sharp bettors)\n"
            "🏆 Projected Total: {predicted_total} points\n"
            "📈 Confidence Level: {confidence:.1f}% (Elite play alert!)\n\n"
            "{tone_visual}\n\n"
            "🚀 DON’T GET LEFT BEHIND! Hit ‘💰’ in the comments if you’re locking it in!\n\n"
            "SharpMoney WinningPicks BetLikeAPro",
            "Sharp money is already flowing on this game between {away_team_full} and {home_team_full}. "
            "Our AI model has {predicted_winner_full} covering the spread at a {confidence:.1f}% confidence level. "
            "{tone_audio} This is the kind of play that moves lines—jump on it NOW!"
        ),
        (
            "💎 ELITE EDGE PLAY 💎\n\n"
            "📢 Breaking News: {away_team} vs. {home_team} 📢\n\n"
            "🏀 AI Model Pick: {predicted_winner}\n"
            "🔥 Spread: {spread_suggestion} (Locked in at peak value!)\n"
            "🎯 Predicted Total: {predicted_total} points\n"
            "💯 Confidence: {confidence:.1f}%\n\n"
            "{tone_visual}\n\n"
            "💸 THIS PLAY WON’T LAST—BET IT NOW! Drop a ‘🚀’ in the comments if you’re tailing!\n\n"
            "SmartBets ElitePicks SharpStrategy",
            "{away_team_full} and {home_team_full} face off in a high-stakes matchup. "
            "Our AI model gives {predicted_winner_full} the clear edge, with a {confidence:.1f}% confidence level. "
            "{tone_audio} This play is catching fire—move before the books adjust!"
        )
    ]
    visual_template, audio_template = random.choice(templates)
    visual_text = visual_template.format(
        away_team=away_team,
        home_team=home_team,
        predicted_winner=predicted_winner,
        spread_suggestion=spread_suggestion,
        predicted_total=predicted_total,
        confidence=confidence,
        tone_visual=tone_visual
    )
    audio_text = audio_template.format(
        away_team_full=away_team_full,
        home_team_full=home_team_full,
        predicted_winner_full=predicted_winner_full,
        spread_suggestion=spread_suggestion,
        predicted_total=predicted_total,
        confidence=confidence,
        tone_audio=tone_audio
    )
    return visual_text, audio_text

def display_generated_post(bet):
    post_text = generate_social_media_post(bet)
    st.write("### 🏆 Generated Social Media Post 🏆")
    st.text(post_text)

# =============================================================================
# Logo Merge Functions
# =============================================================================
def merge_ncaa_team_logos(team1, team2, output_path="merged_ncaa_logo.png", target_height=200, bg_color=(0,0,0)):
    base_dir = "ncaa_image"
    def get_logo(team_name):
        expected = team_name.strip().replace(" ", "_") + "_logo.png"
        for file in os.listdir(base_dir):
            if file.lower() == expected.lower():
                return os.path.join(base_dir, file)
        return None
    logo1_path = get_logo(team1)
    logo2_path = get_logo(team2)
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

def merge_team_logos(team1, team2, output_path="merged_logo.png", target_height=200, bg_color=(255,255,255)):
    def get_logo(abbrev):
        base_dir = "nba_images"
        if abbrev in nba_mapping:
            filename = nba_mapping[abbrev].replace(" ", "_") + ".png"
        else:
            filename = abbrev.strip().replace(" ", "_") + ".png"
        path = os.path.join(base_dir, filename)
        return path if os.path.exists(path) else None
    logo1_path = get_logo(team1)
    logo2_path = get_logo(team2)
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
# Video Generation Functions
# =============================================================================
def generate_audio_from_text(text, audio_output="generated_audio.mp3"):
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(audio_output)
        return audio_output
    except Exception as e:
        print(f"Error generating audio from text: {e}")
        return None

def generate_tiktok_video(bet: dict, output_file: str = None, default_duration: int = 10,
                          video_width: int = 1080, video_height: int = 1920, bg_color: tuple = (0, 0, 0)) -> str:
    if output_file is None:
        bet_date = bet.get('date')
        if isinstance(bet_date, datetime):
            date_str = bet_date.strftime('%Y%m%d')
        else:
            date_str = "unknown_date"
        home_team = bet.get('home_team', 'Home').replace(" ", "_")
        away_team = bet.get('away_team', 'Away').replace(" ", "_")
        output_file = f"tiktok_{date_str}_{home_team}_vs_{away_team}.mp4"
    visual_text, audio_text = generate_social_media_texts(bet)
    audio_file = generate_audio_from_text(audio_text, "generated_audio.mp3")
    if audio_file and Path(audio_file).exists():
        try:
            audio_clip = AudioFileClip(audio_file)
            video_duration = audio_clip.duration
        except Exception as e:
            print(f"Error loading generated audio: {e}")
            video_duration = default_duration
    else:
        print("No audio was generated.")
        video_duration = default_duration
    background = ColorClip(size=(video_width, video_height), color=bg_color, duration=video_duration)
    with smooth_transition("Text Glitch Effect"):
        text_clip = (TextClip(visual_text, fontsize=50, color='white', font='Arial-Bold',
                              method='caption', size=(video_width - 100, None))
                     .set_position(('center', 'center')).set_duration(video_duration).crossfadein(0.5))
    if bet.get('league') == "NCAAB":
        merged_logo_path = merge_ncaa_team_logos(bet.get('home_team', ''), bet.get('away_team', ''))
    else:
        merged_logo_path = merge_team_logos(bet.get('home_team', ''), bet.get('away_team', ''))
    with smooth_transition("Logo Swipe In"):
        if merged_logo_path and Path(merged_logo_path).exists():
            logo_clip = (ImageClip(merged_logo_path).resize(height=300)
                         .set_position(('center', 100)).set_duration(video_duration).crossfadein(0.7))
    with smooth_transition("Final CTA Explosion"):
        final_cta = (TextClip("💰 DOUBLE TAP IF TAILING! 💰", fontsize=60, color="yellow", font="Arial-Bold",
                              method="caption", size=(video_width - 100, None))
                     .set_position(('center', video_height - 300)).set_duration(3).crossfadein(0.5))
    if merged_logo_path and Path(merged_logo_path).exists():
        final_clip = CompositeVideoClip([background, logo_clip, text_clip, final_cta])
    else:
        final_clip = CompositeVideoClip([background, text_clip, final_cta])
    if audio_file and Path(audio_file).exists():
        try:
            final_clip = final_clip.set_audio(audio_clip)
        except Exception as e:
            print(f"Error setting audio: {e}")
    else:
        print("No audio to attach.")
    final_clip.write_videofile(output_file, fps=24, codec='libx264')
    return output_file

# =============================================================================
# Utility: Context Manager for Smooth Transitions
# =============================================================================
@contextmanager
def smooth_transition(label=""):
    start = time.time()
    yield
    end = time.time()
    print(f"⏳ Transition '{label}' took {end - start:.2f} seconds.")

# =============================================================================
# NFL, NBA, and NCAAB Data Functions
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

@st.cache_data(ttl=14400)
def load_nba_data():
    all_rows = []
    try:
        games_csv = pd.read_csv("data/nba_games_all.csv")
        teams_csv = pd.read_csv("data/nba_teams_all.csv")
        spreads_csv = pd.read_csv("data/nba_betting_spread.csv")
        if not games_csv.empty:
            print("Successfully loaded NBA games CSV.")
            games_csv['game_date'] = pd.to_datetime(games_csv['game_date'])
            games_csv.sort_values('game_date', inplace=True)
            if 'pts' in games_csv.columns:
                games_csv['rolling_avg'] = games_csv.groupby('team_id')['pts'].transform(lambda x: x.rolling(3, min_periods=1).mean())
                games_csv['weighted_avg'] = (games_csv['rolling_avg'] * 0.6 + games_csv.groupby('team_id')['pts'].transform('mean') * 0.4)
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
                df_csv['weighted_avg'] = (df_csv['rolling_avg'] * 0.6 + df_csv.groupby('team')['score'].transform('mean') * 0.4)
            return df_csv
    except Exception as e:
        print(f"Error loading NBA data from CSV: {e}")
    print("Falling back to NBA API data...")
    nba_teams_list = nba_teams.get_teams()
    seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
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
    pt_timezone = pytz.timezone('America/Los_Angeles')
    now = datetime.now(pt_timezone)
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
            if 'GAME_DATE' in games.columns:
                raw_date = g['GAME_DATE']
                if isinstance(raw_date, str):
                    raw_date = raw_date.replace("Z", "+00:00")
                    try:
                        game_date = datetime.fromisoformat(raw_date).astimezone(pt_timezone)
                    except Exception as e:
                        st.error(f"Error parsing NBA game time: {e}")
                        game_date = pt_timezone.localize(datetime.strptime(date_str, '%Y-%m-%d'))
                else:
                    game_date = pt_timezone.localize(raw_date) if raw_date.tzinfo is None else raw_date.astimezone(pt_timezone)
            else:
                game_date = pt_timezone.localize(datetime.strptime(date_str, '%Y-%m-%d'))
            upcoming_rows.append({
                'gameday': game_date,
                'home_team': g['HOME_TEAM_ABBREV'],
                'away_team': g['AWAY_TEAM_ABBREV']
            })
    if not upcoming_rows:
        return pd.DataFrame()
    upcoming = pd.DataFrame(upcoming_rows)
    upcoming.sort_values('gameday', inplace=True)
    return upcoming

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
    pt_timezone = pytz.timezone('America/Los_Angeles')
    current_time = datetime.now(pt_timezone)
    dates = [current_time.strftime('%Y%m%d'), (current_time + timedelta(days=1)).strftime('%Y%m%d')]
    rows = []
    for date_str in dates:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {'dates': date_str, 'groups': '50', 'limit': '357'}
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
            game_time_str = game['date'].replace("Z", "+00:00")
            try:
                game_time = datetime.fromisoformat(game_time_str).astimezone(pt_timezone)
            except Exception as e:
                st.error(f"Error parsing game time: {e}")
                continue
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

def update_predictions_with_results(predictions_csv="predictions.csv", league="NBA"):
    try:
        df_preds = pd.read_csv(predictions_csv, parse_dates=["date"])
    except Exception as e:
        print(f"Error loading predictions CSV: {e}")
        return pd.DataFrame()  # Return empty DataFrame if CSV load fails
    if league == "NBA":
        df_history = load_nba_data()
        if "pts" in df_history.columns:
            df_history = df_history.rename(columns={"pts": "score"})
    elif league == "NFL":
        schedule = load_nfl_schedule()
        df_history = preprocess_nfl_data(schedule)
    elif league == "NCAAB":
        df_history = load_ncaab_data_current_season(season=2025)
    else:
        print("Unsupported league specified.")
        return pd.DataFrame()
    updated_rows = []
    for idx, pred in df_preds.iterrows():
        try:
            pred_date = pd.to_datetime(pred["date"]).date()
        except Exception as e:
            print(f"Error converting prediction date: {e}")
            pred_date = None
        home_team = pred.get("home_team", None)
        away_team = pred.get("away_team", None)
        df_game = df_history[pd.to_datetime(df_history["gameday"]).dt.date == pred_date]
        game_home = df_game[df_game["team"] == home_team]
        game_away = df_game[df_game["team"] == away_team]
        if not game_home.empty and not game_away.empty:
            actual_home_score = game_home.iloc[0].get("score", None)
            actual_away_score = game_away.iloc[0].get("score", None)
            if actual_home_score is not None and actual_away_score is not None:
                actual_margin = actual_home_score - actual_away_score
                actual_total = actual_home_score + actual_away_score
                pred_margin = pred.get("predicted_diff", None)
                pred_total = pred.get("predicted_total", None)
                margin_delta = actual_margin - pred_margin if pred_margin is not None else None
                total_delta = actual_total - pred_total if pred_total is not None else None
                predicted_winner = pred.get("predicted_winner", None)
                actual_winner = home_team if actual_margin > 0 else away_team
                prediction_correct = (predicted_winner == actual_winner)
                pred_update = pred.to_dict()
                pred_update.update({
                    "actual_home_score": actual_home_score,
                    "actual_away_score": actual_away_score,
                    "actual_margin": actual_margin,
                    "actual_total": actual_total,
                    "margin_delta": margin_delta,
                    "total_delta": total_delta,
                    "prediction_correct": prediction_correct
                })
                updated_rows.append(pred_update)
            else:
                updated_rows.append(pred.to_dict())
        else:
            updated_rows.append(pred.to_dict())
    updated_df = pd.DataFrame(updated_rows)
    try:
        updated_df.to_csv(predictions_csv, index=False)
        updated_df.to_csv("top_bets.csv", index=False)
        print(f"Predictions CSV updated successfully and saved to both {predictions_csv} and top_bets.csv")
    except Exception as e:
        print(f"Error saving updated CSV: {e}")
    return updated_df

def compare_predictions_with_odds(predictions, league_choice, odds_api_key):
    sport_key_map = {"NFL": "americanfootball_nfl", "NBA": "basketball_nba", "NCAAB": "basketball_ncaab"}
    sport_key = sport_key_map.get(league_choice, "")
    selected_market = "spreads"
    odds_data = fetch_odds(odds_api_key, sport_key, selected_market)
    def map_team_name(team_name):
        default_mapping = {"New England Patriots": "New England Patriots", "Dallas Cowboys": "Dallas Cowboys"}
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

def show_instagram_scheduler_section(current_league):
    st.markdown("---")
    st.subheader("Instagram Scheduler")
    ig_username = st.text_input("Instagram Username", key="ig_username")
    ig_password = st.text_input("Instagram Password", type="password", key="ig_password")
    if st.button("Login to Instagram"):
        client = login_to_instagram(ig_username, ig_password)
        if client:
            st.session_state['instagram_client'] = client
    if st.button("Load Today's Predictions"):
        predictions = load_todays_predictions()
        if predictions:
            st.session_state['todays_predictions'] = predictions
            st.success(f"Loaded {len(predictions)} predictions for today.")
        else:
            st.info("No predictions available for today.")
    if "todays_predictions" in st.session_state:
        selected_prediction = st.selectbox("Select a game to generate post content",
                                             st.session_state['todays_predictions'],
                                             format_func=lambda p: f"{p['away_team']} @ {p['home_team']} - Winner: {p['predicted_winner']}")
        if selected_prediction:
            display_generated_post(selected_prediction)
            if current_league == "NCAAB":
                merged_image_path = merge_ncaa_team_logos(selected_prediction['home_team'], selected_prediction['away_team'])
            else:
                merged_image_path = merge_team_logos(selected_prediction['home_team'], selected_prediction['away_team'])
            if merged_image_path:
                st.markdown("Automatically merged team logos:")
                st.image(merged_image_path, width=300)
            else:
                st.info("Could not automatically merge logos; please provide an image path manually.")
            st.markdown("### Schedule / Post Now")
            scheduled_date = st.date_input("Scheduled Date", datetime.now().date())
            scheduled_time = st.time_input("Scheduled Time", datetime.now().time())
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
                    time.sleep(delay)
                    post_to_instagram(client, generate_social_media_post(selected_prediction), image_path)

def login_with_rest(email, password):
    try:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={st.secrets['general']['firebaseApiKey']}"
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
            st.markdown(f"### {bet['away_team']} @ {bet['home_team']}")
            date_obj = bet['date']
            if isinstance(date_obj, datetime):
                st.caption(date_obj.strftime("%A, %B %d - %I:%M %p"))
        with col2:
            if bet['confidence'] >= 80:
                st.markdown("🔥 High-Confidence Bet 🔥")
            st.markdown(f"Spread Suggestion: {bet['spread_suggestion']}")
            st.markdown(f"Total Suggestion: {bet['ou_suggestion']}")
        with col3:
            tooltip_text = "Confidence indicates the statistical edge."
            st.markdown(
                f"<h3 style='color:{confidence_color};' title='{tooltip_text}'>{bet['overall_confidence']:.1f}% Confidence</h3>",
                unsafe_allow_html=True,
            )
    st.expander("Detailed Insights", expanded=False).write(
        f"**Predicted Winner:** {bet['predicted_winner']}\n\n"
        f"**Predicted Total Points:** {bet['predicted_total']}\n\n"
        f"**Prediction Margin:** {bet['predicted_diff']}\n\n"
    )
    if bet.get('monte_mean_diff') is not None:
        st.expander("Monte Carlo Simulation (Margin) Details", expanded=False).write(
            f"- **Simulated Mean Margin:** {bet['monte_mean_diff']:.2f}\n"
            f"- **Median Margin:** {bet['monte_median']:.2f}\n"
            f"- **Standard Deviation:** {bet['monte_std']:.2f}\n"
            f"- **95% Confidence Interval (Margin):** ({bet['monte_ci'][0]:.2f}, {bet['monte_ci'][1]:.2f})\n"
            f"- **Margin Win Probability:** {bet['win_probability']:.1f}%\n"
        )
    if bet.get('mean_total') is not None:
        st.expander("Monte Carlo Simulation (Totals) Details", expanded=False).write(
            f"- **Simulated Mean Total:** {bet['mean_total']:.1f}\n"
            f"- **Over Threshold:** {bet['over_threshold']:.1f}\n"
            f"- **Under Threshold:** {bet['under_threshold']:.1f}\n"
        )
    if bet.get('margin_confidence') is not None:
        st.write(f"**Margin Confidence Score:** {bet['margin_confidence']:.1f}%")
    if bet.get('totals_confidence') is not None:
        st.write(f"**Totals Confidence Score:** {bet['totals_confidence']:.1f}%")
    st.write(f"**Overall Game Confidence:** {bet['overall_confidence']:.1f}%")
    st.expander("Betting Insights", expanded=False).write(generate_betting_insights(bet))
    st.expander("Recommended Betting Strategy", expanded=False).write(recommended_betting_strategy(bet))
    if st.button("Generate Social Media Post", key=f"social_post_{bet['home_team']}_{bet['away_team']}_{bet['date']}"):
        display_generated_post(bet)
    if st.button("Generate TikTok Video", key=f"tiktok_{bet['home_team']}_{bet['away_team']}_{bet['date']}"):
        video_path = generate_tiktok_video(bet)
        st.success(f"TikTok video saved to: {video_path}")
        st.video(video_path)
    if st.button("Manual Odds Update", key=f"manual_odds_{bet['date']}_{bet['home_team']}_{bet['away_team']}"):
        st.text_input("Enter Home Spread", key=f"manual_home_{bet['date']}_{bet['home_team']}_{bet['away_team']}")
        st.text_input("Enter Game Total", key=f"manual_total_{bet['date']}_{bet['home_team']}_{bet['away_team']}")
    if st.button("Save This Prediction", key=f"save_{bet['date']}_{bet['home_team']}_{bet['away_team']}"):
        save_prediction_to_csv(bet)
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
# Additional Post-Game Analysis & KPI Tracking
# =============================================================================
def display_additional_analytics():
    st.header("Additional Post-Game Analysis & KPI Tracking")
    file = Path(CSV_FILE)
    if not file.exists():
        st.info("No predictions file found for additional analysis.")
        return
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        st.error(f"Error loading predictions CSV: {e}")
        return
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    tab_err, tab_cal, tab_team, tab_profit, tab_spread, tab_corr = st.tabs([
        "Error Analysis", 
        "Calibration & Reliability", 
        "Team Trends", 
        "Profitability Simulation", 
        "Spread/OverUnder Analysis", 
        "Correlation Analysis"
    ])
    with tab_err:
        st.subheader("Error Analysis & Outlier Detection")
        if 'margin_delta' in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df['margin_delta'].dropna(), bins=20, color='skyblue', edgecolor='white')
            ax.set_title("Margin Delta Distribution")
            ax.set_xlabel("Margin Delta")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        if 'total_delta' in df.columns:
            fig2, ax2 = plt.subplots()
            ax2.hist(df['total_delta'].dropna(), bins=20, color='lightgreen', edgecolor='white')
            ax2.set_title("Total Delta Distribution")
            ax2.set_xlabel("Total Delta")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)
        threshold_margin = st.slider("Margin Delta Outlier Threshold", 5, 20, 10)
        threshold_total = st.slider("Total Delta Outlier Threshold", 5, 30, 15)
        outliers = df[(df['margin_delta'].abs() > threshold_margin) | (df['total_delta'].abs() > threshold_total)]
        st.subheader("Detected Outliers")
        st.dataframe(outliers)
    with tab_cal:
        st.subheader("Calibration & Reliability")
        if 'win_probability' in df.columns and 'prediction_correct' in df.columns:
            df['win_probability'] = pd.to_numeric(df['win_probability'], errors='coerce')
            df['win_prob_bin'] = pd.cut(df['win_probability'], bins=10)
            calibration = df.groupby('win_prob_bin').agg({
                'win_probability': 'mean',
                'prediction_correct': 'mean'
            }).reset_index()
            calibration['predicted_win_rate'] = calibration['win_probability']
            calibration['actual_win_rate'] = calibration['prediction_correct'] * 100
            st.subheader("Calibration Data")
            st.dataframe(calibration)
            fig3, ax3 = plt.subplots()
            ax3.plot(calibration['predicted_win_rate'], calibration['actual_win_rate'], marker='o', linestyle='-')
            ax3.plot([0, 100], [0, 100], linestyle='--', color='gray')
            ax3.set_xlabel("Predicted Win Probability")
            ax3.set_ylabel("Actual Win Rate (%)")
            ax3.set_title("Calibration Plot")
            st.pyplot(fig3)
        def parse_ci(ci_str):
            try:
                ci_str = ci_str.strip("()")
                lower, upper = ci_str.split(",")
                return float(lower), float(upper)
            except Exception:
                return None, None
        df['ci_lower'] = df['monte_ci'].apply(lambda x: parse_ci(x)[0] if pd.notnull(x) else None)
        df['ci_upper'] = df['monte_ci'].apply(lambda x: parse_ci(x)[1] if pd.notnull(x) else None)
        if 'actual_margin' in df.columns:
            df['within_ci'] = df.apply(lambda row: row['ci_lower'] <= row['actual_margin'] <= row['ci_upper']
                                       if row['ci_lower'] is not None and row['ci_upper'] is not None and pd.notnull(row['actual_margin'])
                                       else None, axis=1)
            coverage = df['within_ci'].dropna().mean() * 100
            st.subheader(f"Monte Carlo Interval Coverage: {coverage:.1f}%")
        else:
            st.info("Actual margin data not available for interval coverage.")
    with tab_team:
        st.subheader("Team Trends & Bias Analysis")
        if 'home_team' in df.columns and 'margin_delta' in df.columns:
            team_bias = df.groupby('home_team')['margin_delta'].mean().reset_index()
            st.subheader("Average Margin Delta by Home Team")
            st.dataframe(team_bias)
            fig4, ax4 = plt.subplots()
            ax4.bar(team_bias['home_team'], team_bias['margin_delta'], color='coral')
            ax4.set_title("Team Bias in Margin Predictions")
            ax4.set_xlabel("Team")
            ax4.set_ylabel("Average Margin Delta")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig4)
        else:
            st.info("Team-specific error data not available.")
    with tab_profit:
        st.subheader("Profitability & ROI Simulation")
        if 'prediction_correct' in df.columns:
            df['stake_percent'] = df.apply(lambda row: recommended_stake_percent(row.to_dict()) if 'confidence' in row else 0, axis=1)
            bankroll = [100]
            outcomes = []
            for i, row in df.iterrows():
                stake = row['stake_percent'] / 100 * bankroll[-1]
                if row['prediction_correct'] in [True, 1]:
                    profit = stake
                else:
                    profit = -stake
                new_bankroll = bankroll[-1] + profit
                bankroll.append(new_bankroll)
                outcomes.append(new_bankroll)
            st.subheader("Cumulative Bankroll Evolution")
            fig5, ax5 = plt.subplots()
            ax5.plot(outcomes, marker='o')
            ax5.set_title("Simulated Bankroll Over Bets")
            ax5.set_xlabel("Bet Number")
            ax5.set_ylabel("Bankroll Value")
            st.pyplot(fig5)
            roi = (outcomes[-1] - 100) / 100 * 100
            st.metric("Simulated ROI", f"{roi:.2f}%")
        else:
            st.info("Outcome data not available for ROI simulation.")
    with tab_spread:
        st.subheader("Spread & Over/Under Validity")
        if 'predicted_diff' in df.columns and 'actual_margin' in df.columns:
            fig6, ax6 = plt.subplots()
            ax6.scatter(df['predicted_diff'], df['actual_margin'], color='blue', alpha=0.6)
            ax6.set_xlabel("Predicted Margin")
            ax6.set_ylabel("Actual Margin")
            ax6.set_title("Predicted vs. Actual Margin")
            st.pyplot(fig6)
        if 'predicted_total' in df.columns and 'actual_total' in df.columns:
            fig7, ax7 = plt.subplots()
            ax7.scatter(df['predicted_total'], df['actual_total'], color='purple', alpha=0.6)
            ax7.set_xlabel("Predicted Total")
            ax7.set_ylabel("Actual Total")
            ax7.set_title("Predicted vs. Actual Total Points")
            st.pyplot(fig7)
    with tab_corr:
        st.subheader("Correlation & KPI Interactions")
        corr_columns = ["mcciw", "psnr", "aser", "rosa", "evr", "pmdcr", "drar", "margin_delta", "total_delta"]
        for col in corr_columns:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        corr_matrix = df[corr_columns].corr()
        st.subheader("Correlation Matrix")
        st.dataframe(corr_matrix)
        fig8, ax8 = plt.subplots()
        cax = ax8.matshow(corr_matrix, cmap='coolwarm')
        fig8.colorbar(cax)
        ax8.set_xticks(range(len(corr_columns)))
        ax8.set_xticklabels(corr_columns, rotation=90)
        ax8.set_yticks(range(len(corr_columns)))
        ax8.set_yticklabels(corr_columns)
        ax8.set_title("Correlation Matrix of KPIs and Errors", pad=20)
        st.pyplot(fig8)

def display_performance_dashboard():
    st.title("Performance Dashboard")
    file = Path(CSV_FILE)
    if not file.exists():
        st.info("No predictions file found.")
        return
    try:
        df = pd.read_csv(CSV_FILE, parse_dates=["date"])
    except Exception as e:
        st.error(f"Error loading predictions CSV: {e}")
        return
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')
    if 'prediction_correct' in df.columns:
        win_rate = df['prediction_correct'].mean() * 100
    else:
        win_rate = 0
    avg_confidence = df['overall_confidence'].mean() if 'overall_confidence' in df.columns else 0
    pred_count = df.shape[0]
    tab_overview, tab_historical, tab_distribution, tab_bookmaker, tab_high_confidence, tab_team_insights, tab_betting_insights, tab_additional = st.tabs(
        ["Overview", "Historical Trends", "Prediction Distribution", "Performance vs. Bookmaker", "High-Confidence Picks", "League & Team Insights", "Betting Insights", "Post-Game Analysis & KPIs"]
    )
    with tab_overview:
        st.header("Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Win Rate", f"{win_rate:.1f}%")
        col2.metric("Avg. Overall Confidence", f"{avg_confidence:.1f}%")
        col3.metric("Total Predictions", pred_count)
        st.subheader("Summary Table")
        st.dataframe(df)
        st.subheader("Performance & Betting Insights Summary")
        if 'actual_total' in df.columns and df['actual_total'].notna().all() and 'predicted_total' in df.columns:
            mae_val = np.mean(np.abs(df['actual_total'] - df['predicted_total']))
            rmse_val = np.sqrt(np.mean((df['actual_total'] - df['predicted_total'])**2))
            mape_val = np.mean(np.abs((df['actual_total'] - df['predicted_total']) / df['actual_total'])) * 100
            error_metrics_str = (
                f"Total Points Prediction Error Metrics:\n"
                f"• MAE: {mae_val:.2f}\n"
                f"• RMSE: {rmse_val:.2f}\n"
                f"• MAPE: {mape_val:.2f}%\n"
            )
        else:
            error_metrics_str = ""
        summary_text = (
            f"Aggregated Model Performance Summary:\n"
            f"• Overall Win Rate: {win_rate:.1f}% based on {pred_count} predictions.\n"
            f"• Average Overall Confidence: {avg_confidence:.1f}%.\n"
            f"{error_metrics_str}\n"
        )
        if win_rate >= 70 and avg_confidence >= 75:
            summary_text += "The model exhibits robust performance, indicating strong betting opportunities. Consider leveraging aggressive wagering strategies."
        elif win_rate >= 50 and avg_confidence >= 60:
            summary_text += "The model shows moderate performance; adjusting wager sizes and further analysis on select picks may optimize returns."
        else:
            summary_text += "Model performance appears limited; proceed with caution or conduct additional research to validate predictions before wagering."
        st.text(summary_text)
    with tab_historical:
        st.header("Historical Trends")
        if 'prediction_correct' in df.columns:
            daily_win_rate = df.groupby(df['date'].dt.date)['prediction_correct'].mean() * 100
            st.subheader("Win Rate Over Time")
            st.line_chart(daily_win_rate)
        if 'overall_confidence' in df.columns:
            daily_confidence = df.groupby(df['date'].dt.date)['overall_confidence'].mean()
            st.subheader("Average Confidence Over Time")
            st.line_chart(daily_confidence)
    with tab_distribution:
        st.header("Prediction Distribution")
        if 'predicted_diff' in df.columns:
            st.subheader("Histogram of Predicted Margin")
            fig, ax = plt.subplots()
            ax.hist(df['predicted_diff'], bins=20, color='skyblue', edgecolor='white')
            if 'monte_mean_diff' in df.columns and df['monte_mean_diff'].notna().all():
                ax.axvline(df['monte_mean_diff'].mean(), color='red', linestyle='--', label='Mean Margin')
                ax.legend()
            st.pyplot(fig)
        if 'actual_margin' in df.columns and df['actual_margin'].notna().all() and 'predicted_diff' in df.columns:
            st.subheader("Predicted vs. Actual Margin")
            fig2, ax2 = plt.subplots()
            ax2.scatter(df['predicted_diff'], df['actual_margin'], color='green', alpha=0.6)
            ax2.set_xlabel("Predicted Margin")
            ax2.set_ylabel("Actual Margin")
            ax2.set_title("Scatter Plot of Predicted vs. Actual Margin")
            st.pyplot(fig2)
    with tab_bookmaker:
        st.header("Performance vs. Bookmaker")
        if 'bookmaker_spread' in df.columns:
            st.subheader("Predicted Spread vs. Bookmaker Spread")
            fig3, ax3 = plt.subplots()
            ax3.scatter(df['predicted_diff'], df['bookmaker_spread'], color='purple', alpha=0.6)
            ax3.plot([df['predicted_diff'].min(), df['predicted_diff'].max()],
                     [df['predicted_diff'].min(), df['predicted_diff'].max()],
                     color='black', linestyle='--')
            ax3.set_xlabel("Predicted Spread")
            ax3.set_ylabel("Bookmaker Spread")
            ax3.set_title("Predicted vs. Bookmaker Spread")
            st.pyplot(fig3)
            st.subheader("Detailed Comparison")
            st.dataframe(df[['home_team','away_team','predicted_diff','bookmaker_spread']])
        else:
            st.info("Bookmaker odds data not available in predictions.")
    with tab_high_confidence:
        st.header("High-Confidence Picks & Upset Alerts")
        conf_threshold = st.slider("Minimum Confidence Level", 50.0, 99.0, 75.0, 5.0)
        top_bets = find_top_bets(df.to_dict('records'), threshold=conf_threshold)
        if not top_bets.empty:
            st.dataframe(top_bets)
        else:
            st.info("No high-confidence picks found. Try lowering the threshold.")
    with tab_team_insights:
        st.header("League & Team Insights")
        if 'home_team' in df.columns and 'predicted_diff' in df.columns:
            team_perf = df.groupby('home_team')['predicted_diff'].mean().reset_index()
            st.subheader("Average Predicted Margin by Home Team")
            fig4, ax4 = plt.subplots()
            ax4.bar(team_perf['home_team'], team_perf['predicted_diff'], color='teal')
            ax4.set_xlabel("Team")
            ax4.set_ylabel("Average Predicted Margin")
            ax4.set_title("Team Performance Overview")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig4)
    with tab_betting_insights:
        st.header("Betting Insights")
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        if 'predicted_total' in df.columns and 'actual_total' in df.columns and df['actual_total'].notna().all():
            mae = mean_absolute_error(df['actual_total'], df['predicted_total'])
            rmse = np.sqrt(mean_squared_error(df['actual_total'], df['predicted_total']))
            mape = np.mean(np.abs((df['actual_total'] - df['predicted_total']) / df['actual_total'])) * 100
            st.subheader("Error Metrics for Total Points Predictions")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAPE: {mape:.2f}%")
        if 'predicted_diff' in df.columns and 'bookmaker_spread' in df.columns:
            df['spread_deviation'] = df['predicted_diff'] - df['bookmaker_spread']
            st.subheader("Spread Deviation Analysis")
            st.write("The spread deviation highlights potential value plays when the predicted spread significantly differs from the bookmaker spread.")
            st.dataframe(df[['home_team','away_team','predicted_diff','bookmaker_spread','spread_deviation']])
        else:
            st.info("Spread deviation analysis not available.")
        st.subheader("Aggregated Betting Insights")
        insights_list = []
        for idx, row in df.iterrows():
            insights = generate_betting_insights(row.to_dict())
            insights_list.append(insights)
        st.write("\n\n".join(insights_list))
    with tab_additional:
        display_additional_analytics()

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
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce')
        df['date'] = df['date'].apply(lambda d: d.replace(year=d.year + 100) if d.year < 2000 else d)
        df['date'] = df['date'].dt.date
    except Exception as e:
        st.error(f"Error parsing dates: {e}")
        return []
    if TEST_FIRST_PREDICTION_FROM_CSV:
        if not df.empty:
            return df.iloc[[0]].to_dict('records')
        else:
            return []
    else:
        today = datetime.now().date()
        todays_df = df[df['date'] == today]
        return todays_df.to_dict('records')

# =============================================================================
# run_league_pipeline Function
# =============================================================================
def run_league_pipeline(league_choice, odds_api_key):
    st.header(f"Today's {league_choice} Best Bets")
    run_fitter = st.sidebar.checkbox("Enable Distribution Fitting (Manual Trigger)", value=False, key="run_fitter")
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
        try:
            upcoming = fetch_upcoming_nba_games(days_ahead=3)
        except Exception as e:
            st.error(f"Failed to fetch upcoming NBA games: {e}")
            upcoming = pd.DataFrame()
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
        # Make a copy to allow mutation of the cached DataFrame
        team_data = team_data.copy()
        # Check for 'def_rating' column; if missing, set default value
        if 'def_rating' not in team_data.columns:
            st.warning("def_rating column is missing. Defaulting to baseline value (e.g., 110) for all teams.")
            team_data['def_rating'] = 110
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
        global team_stats_global
        team_stats_global.update(team_stats)
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
            if outcome is None:
                continue
            if home in stack_models and away in stack_models:
                last_features_home = team_data[team_data['team'] == home][['rolling_avg','rolling_std','weighted_avg']].tail(1)
                last_features_away = team_data[team_data['team'] == away][['rolling_avg','rolling_std','weighted_avg']].tail(1)
                if not last_features_home.empty and not last_features_away.empty:
                    margin_sim = monte_carlo_simulation_margin(
                        model_home=stack_models[home],
                        model_away=stack_models[away],
                        X_home=last_features_home.values,
                        X_away=last_features_away.values,
                        n_simulations=10000,
                        error_std_home=team_stats[home]['std'],
                        error_std_away=team_stats[away]['std'],
                        random_seed=42,
                        run_fitter=run_fitter
                    )
                    total_sim = monte_carlo_simulation_totals(
                        model_home=stack_models[home],
                        model_away=stack_models[away],
                        X_home=last_features_home.values,
                        X_away=last_features_away.values,
                        n_simulations=10000,
                        error_std_home=team_stats[home]['std'],
                        error_std_away=team_stats[away]['std'],
                        random_seed=42,
                        run_fitter=run_fitter
                    )
                    monte_mean_diff = margin_sim["mean_diff"]
                    monte_ci = margin_sim["ci"]
                    win_rate = margin_sim["win_rate"]
                    win_margin_rate = margin_sim["win_margin_rate"]
                    monte_median = margin_sim["median_diff"]
                    monte_std = margin_sim["std_diff"]
                    simulated_diffs = margin_sim["simulated_diffs"]
                    mean_total = total_sim["mean_total"]
                    over_threshold = total_sim["over_threshold"]
                    under_threshold = total_sim["under_threshold"]
                    median_total = total_sim["median_total"]
                    std_total = total_sim["std_total"]
                    simulated_totals = total_sim["simulated_totals"]
                    total_range = under_threshold - over_threshold
                    totals_confidence = max(1, min(99, 100 - (total_range / mean_total * 100))) if mean_total != 0 else 50
                    margin_confidence = (win_rate + win_margin_rate) / 2
                    overall_confidence = (margin_confidence + totals_confidence) / 2
                else:
                    monte_mean_diff = monte_ci = win_rate = win_margin_rate = monte_median = monte_std = None
                    mean_total = over_threshold = under_threshold = median_total = std_total = None
                    margin_confidence = totals_confidence = overall_confidence = outcome['confidence']
            else:
                monte_mean_diff = monte_ci = win_rate = win_margin_rate = monte_median = monte_std = None
                mean_total = over_threshold = under_threshold = median_total = std_total = None
                margin_confidence = totals_confidence = overall_confidence = outcome['confidence']
            bet = {
                'date': row['gameday'],
                'league': league_choice,
                'home_team': home,
                'away_team': away,
                'home_pred': home_pred,
                'away_pred': away_pred,
                'predicted_winner': outcome['predicted_winner'],
                'predicted_diff': outcome['diff'],
                'predicted_total': outcome['total_points'],
                'confidence': overall_confidence,
                'spread_suggestion': outcome['spread_suggestion'],
                'ou_suggestion': outcome['ou_suggestion'],
                'monte_mean_diff': monte_mean_diff,
                'monte_ci': monte_ci,
                'win_probability': win_rate,
                'win_margin_rate': win_margin_rate,
                'monte_median': monte_median,
                'monte_std': monte_std,
                'simulated_diffs': simulated_diffs.tolist() if simulated_diffs is not None else None,
                'mean_total': mean_total,
                'over_threshold': over_threshold,
                'under_threshold': under_threshold,
                'median_total': median_total,
                'std_total': std_total,
                'simulated_totals': simulated_totals.tolist() if simulated_totals is not None else None,
                'margin_confidence': margin_confidence,
                'totals_confidence': totals_confidence,
                'overall_confidence': overall_confidence
            }
            results.append(bet)
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
        if st.button("Save All Predictions to CSV"):
            if results:
                save_predictions_to_csv(results, csv_file=CSV_FILE)
            else:
                st.warning("No predictions to save.")

def display_performance_dashboard():
    st.title("Performance Dashboard")
    file = Path(CSV_FILE)
    if not file.exists():
        st.info("No predictions file found.")
        return
    try:
        df = pd.read_csv(CSV_FILE, parse_dates=["date"])
    except Exception as e:
        st.error(f"Error loading predictions CSV: {e}")
        return
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')
    if 'prediction_correct' in df.columns:
        win_rate = df['prediction_correct'].mean() * 100
    else:
        win_rate = 0
    avg_confidence = df['overall_confidence'].mean() if 'overall_confidence' in df.columns else 0
    pred_count = df.shape[0]
    tab_overview, tab_historical, tab_distribution, tab_bookmaker, tab_high_confidence, tab_team_insights, tab_betting_insights, tab_additional = st.tabs(
        ["Overview", "Historical Trends", "Prediction Distribution", "Performance vs. Bookmaker", "High-Confidence Picks", "League & Team Insights", "Betting Insights", "Post-Game Analysis & KPIs"]
    )
    with tab_overview:
        st.header("Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Win Rate", f"{win_rate:.1f}%")
        col2.metric("Avg. Overall Confidence", f"{avg_confidence:.1f}%")
        col3.metric("Total Predictions", pred_count)
        st.subheader("Summary Table")
        st.dataframe(df)
        st.subheader("Performance & Betting Insights Summary")
        if 'actual_total' in df.columns and df['actual_total'].notna().all() and 'predicted_total' in df.columns:
            mae_val = np.mean(np.abs(df['actual_total'] - df['predicted_total']))
            rmse_val = np.sqrt(np.mean((df['actual_total'] - df['predicted_total'])**2))
            mape_val = np.mean(np.abs((df['actual_total'] - df['predicted_total']) / df['actual_total'])) * 100
            error_metrics_str = (
                f"Total Points Prediction Error Metrics:\n"
                f"• MAE: {mae_val:.2f}\n"
                f"• RMSE: {rmse_val:.2f}\n"
                f"• MAPE: {mape_val:.2f}%\n"
            )
        else:
            error_metrics_str = ""
        summary_text = (
            f"Aggregated Model Performance Summary:\n"
            f"• Overall Win Rate: {win_rate:.1f}% based on {pred_count} predictions.\n"
            f"• Average Overall Confidence: {avg_confidence:.1f}%.\n"
            f"{error_metrics_str}\n"
        )
        if win_rate >= 70 and avg_confidence >= 75:
            summary_text += "The model exhibits robust performance, indicating strong betting opportunities. Consider leveraging aggressive wagering strategies."
        elif win_rate >= 50 and avg_confidence >= 60:
            summary_text += "The model shows moderate performance; adjusting wager sizes and further analysis on select picks may optimize returns."
        else:
            summary_text += "Model performance appears limited; proceed with caution or conduct additional research to validate predictions before wagering."
        st.text(summary_text)
    with tab_historical:
        st.header("Historical Trends")
        if 'prediction_correct' in df.columns:
            daily_win_rate = df.groupby(df['date'].dt.date)['prediction_correct'].mean() * 100
            st.subheader("Win Rate Over Time")
            st.line_chart(daily_win_rate)
        if 'overall_confidence' in df.columns:
            daily_confidence = df.groupby(df['date'].dt.date)['overall_confidence'].mean()
            st.subheader("Average Confidence Over Time")
            st.line_chart(daily_confidence)
    with tab_distribution:
        st.header("Prediction Distribution")
        if 'predicted_diff' in df.columns:
            st.subheader("Histogram of Predicted Margin")
            fig, ax = plt.subplots()
            ax.hist(df['predicted_diff'], bins=20, color='skyblue', edgecolor='white')
            if 'monte_mean_diff' in df.columns and df['monte_mean_diff'].notna().all():
                ax.axvline(df['monte_mean_diff'].mean(), color='red', linestyle='--', label='Mean Margin')
                ax.legend()
            st.pyplot(fig)
        if 'actual_margin' in df.columns and df['actual_margin'].notna().all() and 'predicted_diff' in df.columns:
            st.subheader("Predicted vs. Actual Margin")
            fig2, ax2 = plt.subplots()
            ax2.scatter(df['predicted_diff'], df['actual_margin'], color='green', alpha=0.6)
            ax2.set_xlabel("Predicted Margin")
            ax2.set_ylabel("Actual Margin")
            ax2.set_title("Scatter Plot of Predicted vs. Actual Margin")
            st.pyplot(fig2)
    with tab_bookmaker:
        st.header("Performance vs. Bookmaker")
        if 'bookmaker_spread' in df.columns:
            st.subheader("Predicted Spread vs. Bookmaker Spread")
            fig3, ax3 = plt.subplots()
            ax3.scatter(df['predicted_diff'], df['bookmaker_spread'], color='purple', alpha=0.6)
            ax3.plot([df['predicted_diff'].min(), df['predicted_diff'].max()],
                     [df['predicted_diff'].min(), df['predicted_diff'].max()],
                     color='black', linestyle='--')
            ax3.set_xlabel("Predicted Spread")
            ax3.set_ylabel("Bookmaker Spread")
            ax3.set_title("Predicted vs. Bookmaker Spread")
            st.pyplot(fig3)
            st.subheader("Detailed Comparison")
            st.dataframe(df[['home_team','away_team','predicted_diff','bookmaker_spread']])
        else:
            st.info("Bookmaker odds data not available in predictions.")
    with tab_high_confidence:
        st.header("High-Confidence Picks & Upset Alerts")
        conf_threshold = st.slider("Minimum Confidence Level", 50.0, 99.0, 75.0, 5.0)
        top_bets = find_top_bets(df.to_dict('records'), threshold=conf_threshold)
        if not top_bets.empty:
            st.dataframe(top_bets)
        else:
            st.info("No high-confidence picks found. Try lowering the threshold.")
    with tab_team_insights:
        st.header("League & Team Insights")
        if 'home_team' in df.columns and 'predicted_diff' in df.columns:
            team_perf = df.groupby('home_team')['predicted_diff'].mean().reset_index()
            st.subheader("Average Predicted Margin by Home Team")
            fig4, ax4 = plt.subplots()
            ax4.bar(team_perf['home_team'], team_perf['predicted_diff'], color='teal')
            ax4.set_xlabel("Team")
            ax4.set_ylabel("Average Predicted Margin")
            ax4.set_title("Team Performance Overview")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig4)
    with tab_betting_insights:
        st.header("Betting Insights")
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        if 'predicted_total' in df.columns and 'actual_total' in df.columns and df['actual_total'].notna().all():
            mae = mean_absolute_error(df['actual_total'], df['predicted_total'])
            rmse = np.sqrt(mean_squared_error(df['actual_total'], df['predicted_total']))
            mape = np.mean(np.abs((df['actual_total'] - df['predicted_total']) / df['actual_total'])) * 100
            st.subheader("Error Metrics for Total Points Predictions")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAPE: {mape:.2f}%")
        if 'predicted_diff' in df.columns and 'bookmaker_spread' in df.columns:
            df['spread_deviation'] = df['predicted_diff'] - df['bookmaker_spread']
            st.subheader("Spread Deviation Analysis")
            st.write("The spread deviation highlights potential value plays when the predicted spread significantly differs from the bookmaker spread.")
            st.dataframe(df[['home_team','away_team','predicted_diff','bookmaker_spread','spread_deviation']])
        else:
            st.info("Spread deviation analysis not available.")
        st.subheader("Aggregated Betting Insights")
        insights_list = []
        for idx, row in df.iterrows():
            insights = generate_betting_insights(row.to_dict())
            insights_list.append(insights)
        st.write("\n\n".join(insights_list))
    with tab_additional:
        display_additional_analytics()

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
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce')
        df['date'] = df['date'].apply(lambda d: d.replace(year=d.year + 100) if d.year < 2000 else d)
        df['date'] = df['date'].dt.date
    except Exception as e:
        st.error(f"Error parsing dates: {e}")
        return []
    if TEST_FIRST_PREDICTION_FROM_CSV:
        if not df.empty:
            return df.iloc[[0]].to_dict('records')
        else:
            return []
    else:
        today = datetime.now().date()
        todays_df = df[df['date'] == today]
        return todays_df.to_dict('records')

# =============================================================================
# Main Pipeline: run_league_pipeline is now defined above main()
# =============================================================================

def main():
    st.set_page_config(page_title="FoxEdge Sports Betting Insights", page_icon="🦊", layout="centered")
    st.title("🦊 FoxEdge Sports Betting Insights")
    if LOCAL_TEST_MODE:
        st.info("LOCAL_TEST_MODE is enabled: bypassing Firebase login for local testing.")
        st.session_state['logged_in'] = True
        st.session_state['email'] = "local@test.com"
    else:
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
    odds_api_key = st.sidebar.text_input("Enter Odds API Key", type="password",
                                          value=st.secrets["odds_api"]["apiKey"] if "odds_api" in st.secrets else "")
    nav_option = st.sidebar.radio("Navigation", ["Prediction Pipeline", "Performance Dashboard"])
    league_choice = st.sidebar.radio("Select League", ["NFL", "NBA", "NCAAB"])
    if nav_option == "Prediction Pipeline":
        run_league_pipeline(league_choice, odds_api_key)
        show_instagram_scheduler_section(league_choice)
    else:
        display_performance_dashboard()

if __name__ == "__main__":
    initialize_csv()  # Ensure predictions.csv exists
    main()
    for sport in ["NBA", "NFL", "NCAAB"]:
        print(f"\nUpdating predictions for {sport}...")
        updated_df = update_predictions_with_results("predictions.csv", league=sport)
        if not updated_df.empty:
            print(f"Updated {sport} predictions:")
            print(updated_df.head())
        else:
            print(f"Could not update predictions for {sport}.")
