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
from joblib import Parallel, delayed

# cbbpy for NCAAB historical data
import cbbpy.mens_scraper as cbb

# For hyperparameter tuning and time-series cross validation
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

################################################################################
# HELPER FUNCTION TO ENSURE TZ-NAIVE DATETIMES
################################################################################
def to_naive(dt):
    """
    Converts a datetime object to tz-naive if it is tz-aware.
    
    Args:
        dt: A datetime object.
    
    Returns:
        A tz-naive datetime object.
    """
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
    """Login user using Firebase REST API."""
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
    """Logs out the current user."""
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
    """Rounds a number to the nearest 0.5."""
    return round(number * 2) / 2

################################################################################
# MODEL TUNING HELPER
################################################################################
def tune_model(model, param_grid, X_train, y_train):
    """
    Tunes a given model using GridSearchCV with TimeSeriesSplit.
    
    Args:
        model: The model to tune.
        param_grid: Dictionary of hyperparameters.
        X_train: Training features.
        y_train: Training target.
    
    Returns:
        The best estimator.
    """
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

################################################################################
# MODEL TRAINING & PREDICTION (STACKING + AUTO-ARIMA HYBRID)
################################################################################
def process_team(team, team_data):
    """
    Processes a single team's data to train models and compute statistics.
    
    Args:
        team: Team abbreviation (or name).
        team_data: DataFrame containing historical game data.
    
    Returns:
        A tuple (team, stacking_regressor, arima_model, team_stats) or None if not enough data.
    """
    df_team = team_data[team_data['team'] == team].copy()
    df_team.sort_values('gameday', inplace=True)
    scores = df_team['score'].reset_index(drop=True)

    if len(scores) < 3:
        return None

    # Enhanced Feature Engineering
    df_team['rolling_avg'] = df_team['score'].rolling(window=3, min_periods=1).mean()
    df_team['rolling_std'] = df_team['score'].rolling(window=3, min_periods=1).std().fillna(0)
    df_team['season_avg'] = df_team['score'].expanding().mean()
    df_team['weighted_avg'] = (df_team['rolling_avg'] * 0.6) + (df_team['season_avg'] * 0.4)

    # Save basic team stats
    team_stats_entry = {
        'mean': round_half(scores.mean()),
        'std': round_half(scores.std()),
        'max': round_half(scores.max()),
        'recent_form': round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
    }

    # Prepare features and target
    features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg']].fillna(0)
    X = features.values
    y = scores.values

    # Time-series split: first 80% for training, rest for testing
    n = len(X)
    split_index = int(n * 0.8)
    if split_index < 2 or n - split_index < 1:
        return None
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    # Hyperparameter tuning for base models
    try:
        xgb = XGBRegressor(random_state=42)
        xgb_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
        xgb_best = tune_model(xgb, xgb_grid, X_train, y_train)
    except Exception as e:
        print(f"Error tuning XGB for team {team}: {e}")
        xgb_best = XGBRegressor(n_estimators=100, random_state=42)

    try:
        lgbm = LGBMRegressor(random_state=42)
        lgbm_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5]}
        lgbm_best = tune_model(lgbm, lgbm_grid, X_train, y_train)
    except Exception as e:
        print(f"Error tuning LGBM for team {team}: {e}")
        lgbm_best = LGBMRegressor(n_estimators=100, random_state=42)

    try:
        cat = CatBoostRegressor(verbose=0, random_state=42)
        cat_grid = {'iterations': [50, 100], 'learning_rate': [0.1, 0.05]}
        cat_best = tune_model(cat, cat_grid, X_train, y_train)
    except Exception as e:
        print(f"Error tuning CatBoost for team {team}: {e}")
        cat_best = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)

    estimators = [
        ('xgb', xgb_best),
        ('lgbm', lgbm_best),
        ('cat', cat_best)
    ]

    # Initialize and train Stacking Regressor
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
        team_stats_entry['mse'] = mse
        # Compute bias from training data
        bias = np.mean(y_train - stack.predict(X_train))
        team_stats_entry['bias'] = bias
    except Exception as e:
        print(f"Error training Stacking Regressor for team {team}: {e}")
        return None

    # Train ARIMA if sufficient data
    arima_model = None
    if len(scores) >= 7:
        try:
            arima_model = auto_arima(
                scores,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                max_p=3,
                max_q=3
            )
        except Exception as e:
            print(f"Error training ARIMA for team {team}: {e}")
            arima_model = None

    return (team, stack, arima_model, team_stats_entry)

@st.cache_data(ttl=14400)
def train_team_models(team_data: pd.DataFrame):
    """
    Trains a hybrid model (Stacking Regressor + Auto-ARIMA) for each team's 'score'
    using time-series cross validation and hyperparameter optimization.
    
    Returns:
        stack_models: Dictionary of trained Stacking Regressors keyed by team.
        arima_models: Dictionary of trained ARIMA models keyed by team.
        team_stats: Dictionary containing statistical summaries for each team.
    """
    all_teams = team_data['team'].unique()
    results = Parallel(n_jobs=-1)(delayed(process_team)(team, team_data) for team in all_teams)
    stack_models = {}
    arima_models = {}
    team_stats = {}
    for res in results:
        if res is not None:
            team, stack_model, arima_model, stats = res
            stack_models[team] = stack_model
            if arima_model is not None:
                arima_models[team] = arima_model
            team_stats[team] = stats
    return stack_models, arima_models, team_stats

def predict_team_score(team, stack_models, arima_models, team_stats, team_data):
    """
    Predicts the next-game score for a given team by blending model outputs 
    and applying a bias correction.
    """
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

    if team in arima_models:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast)
        except Exception as e:
            print(f"Error predicting with ARIMA for team {team}: {e}")
            arima_pred = None

    ensemble = None
    if stack_pred is not None and arima_pred is not None:
        mse_stack = team_stats[team].get('mse', 1)
        # Attempt ARIMA MSE
        try:
            resid = arima_models[team].resid()
            mse_arima = np.mean(np.square(resid))
        except:
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

    # Large MSE filter
    if team_stats[team].get('mse', 0) > 150:
        return None, (None, None)

    if ensemble is None:
        return None, (None, None)

    # Apply bias correction
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
    """
    Evaluates a matchup by computing the predicted spread, total, and confidence.
    """
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

    return {
        'predicted_winner': winner,
        'diff': round_half(diff),
        'total_points': round_half(total_points),
        'confidence': confidence,
        'spread_suggestion': f"Lean {winner} by {round_half(diff):.1f}",
        'ou_suggestion': f"Take the {'Over' if total_points > ou_threshold else 'Under'} {round_half(total_points):.1f}"
    }

################################################################################
# UPDATED: Guard Clause in find_top_bets
################################################################################
def find_top_bets(matchups, threshold=70.0):
    """
    Filters and returns the matchups with confidence >= threshold.

    If 'matchups' is empty or has no 'confidence' column, return an empty DataFrame.
    """
    if not matchups:
        return pd.DataFrame()

    df = pd.DataFrame(matchups)
    if "confidence" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    df_top = df[df['confidence'] >= threshold].copy()
    df_top.sort_values('confidence', ascending=False, inplace=True)
    return df_top

################################################################################
# NFL DATA LOADING
################################################################################
@st.cache_data(ttl=14400)
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
    upcoming = schedule[
        schedule['home_score'].isna() & schedule['away_score'].isna()
    ].copy()
    now = datetime.now()
    filter_date = now + timedelta(days=days_ahead)
    upcoming = upcoming[upcoming['gameday'] <= filter_date].copy()
    upcoming.sort_values('gameday', inplace=True)
    return upcoming[['gameday', 'home_team', 'away_team']]

################################################################################
# NBA DATA LOADING
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

########################################
# NCAAB HISTORICAL LOADER (UPDATED)
########################################
@st.cache_data(ttl=14400)
def load_ncaab_data_current_season(season=2025):
    """
    Loads finished or in-progress NCAA MBB games for the given season
    using cbbpy. Adds is_home=1 for home team, is_home=0 for away.
    
    Now also includes opponent score (opp_score) for defensive metric calculations.
    """
    info_df, _, _ = cbb.get_games_season(season=season, info=True, box=False, pbp=False)
    if info_df.empty:
        return pd.DataFrame()

    # Convert "game_day" to datetime if needed
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
    """Generates a detailed analysis writeup for a given bet."""
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

def display_bet_card(bet, team_stats_global, team_data=None):
    """Displays a bet card with summary and expandable detailed insights."""
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
            st.markdown(
                f"**<span title='Spread Suggestion is based on the predicted point difference'>Spread Suggestion:</span>** {bet['spread_suggestion']}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**<span title='Total Suggestion indicates the recommended bet on the combined score'>Total Suggestion:</span>** {bet['ou_suggestion']}",
                unsafe_allow_html=True,
            )
    
        with col3:
            tooltip_text = "Confidence indicates the statistical edge derived from performance metrics."
            st.markdown(
                f"<h3 style='color:{confidence_color};' title='{tooltip_text}'>{bet['confidence']:.1f}% Confidence</h3>",
                unsafe_allow_html=True,
            )
    
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
    """
    Runs the data processing, model training, prediction, and UI display
    for the selected league.
    """
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
        # Load historical data from cbbpy for model training
        team_data = load_ncaab_data_current_season(season=2025)
        if team_data.empty:
            st.error("Unable to load historical NCAAB data from cbbpy.")
            return

        # Fetch upcoming NCAAB games from ESPN scoreboard
        upcoming = fetch_upcoming_ncaab_games()

    if team_data.empty or upcoming.empty:
        st.warning(f"No upcoming {league_choice} data available for analysis.")
        return

    # If league is NBA, NFL, or NCAAB, compute top/bottom defenses
    if league_choice in ["NBA", "NFL", "NCAAB"]:
        if league_choice == "NBA":
            def_ratings = team_data.groupby('team')['def_rating'].mean().to_dict()
            sorted_def = sorted(def_ratings.items(), key=lambda x: x[1])
            top_10 = set([t for t, r in sorted_def[:10]])
            bottom_10 = set([t for t, r in sorted_def[-10:]])
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
        models_path = "models/trained_models.pkl"
        if os.path.exists(models_path):
            try:
                stack_models, arima_models, team_stats = joblib.load(models_path)
            except Exception as e:
                print(f"Error loading cached models: {e}")
                stack_models, arima_models, team_stats = train_team_models(team_data)
                os.makedirs("models", exist_ok=True)
                joblib.dump((stack_models, arima_models, team_stats), models_path)
        else:
            stack_models, arima_models, team_stats = train_team_models(team_data)
            os.makedirs("models", exist_ok=True)
            joblib.dump((stack_models, arima_models, team_stats), models_path)

        team_stats_global = team_stats
        results.clear()

        for _, row in upcoming.iterrows():
            home, away = row['home_team'], row['away_team']
            home_pred, _ = predict_team_score(home, stack_models, arima_models, team_stats, team_data)
            away_pred, _ = predict_team_score(away, stack_models, arima_models, team_stats, team_data)

            row_gameday = to_naive(row['gameday'])

            # League-specific adjustments
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

                # Additional NBA tweaks
                home_pred += 1
                away_pred -= 1

                # Defensive rank adjustments
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

                # Additional NFL tweaks
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

                # Additional NCAAB tweaks
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
        conf_threshold = st.slider(
            "Minimum Confidence Level",
            min_value=50.0,
            max_value=99.0,
            value=75.0,
            step=5.0,
            help="Only show bets with confidence level above this threshold"
        )

        if not results or not any("confidence" in x for x in results):
            st.info("No predictions available or 'confidence' missing.")
            return

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

################################################################################
# STREAMLIT MAIN FUNCTION & SCHEDULING IMPLEMENTATION
################################################################################

def scheduled_task():
    """
    Executes scheduled tasks such as data fetching, updating predictions,
    and refreshing stored models.
    """
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

    st.write("🏀 Fetching latest NCAAB data (historical)...")
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

def main():
    """
    Main Streamlit function for the interactive user interface.
    """
    st.set_page_config(
        page_title="FoxEdge Sports Betting Edge",
        page_icon="🦊",
        layout="centered"
    )

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
                    # Re-run using st.rerun
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

    if st.button("Save Predictions to CSV"):
        if results:
            save_predictions_to_csv(results)
            csv_data = pd.DataFrame(results).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_data,
                file_name='predictions.csv',
                mime='text/csv',
            )
        else:
            st.warning("No predictions to save.")

if __name__ == "__main__":
    # Use st.query_params instead of st.experimental_get_query_params
    query_params = st.query_params
    if "trigger" in query_params:
        scheduled_task()
        st.write("Task triggered successfully.")
    else:
        main()
