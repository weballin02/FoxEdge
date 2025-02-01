import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import nfl_data_py as nfl
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2, TeamGameLog
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
    """
    Rounds a number to the nearest 0.5.
    """
    return round(number * 2) / 2

################################################################################
# MODEL TRAINING & PREDICTION (STACKING + AUTO-ARIMA HYBRID)
################################################################################
@st.cache_data(ttl=14400)
def train_team_models(team_data: pd.DataFrame):
    """
    Trains a Stacking Regressor and an Auto-ARIMA model for each team's 'score'
    using time-series–aware splitting and cross-validation. A 'lag_score' feature is added
    to capture recent momentum. Returns:
      - stack_models: dictionary of trained stacking regressors per team.
      - arima_models: dictionary of trained ARIMA models per team.
      - team_stats: dictionary of aggregated team statistics.
      - model_errors: dictionary of training errors for each model.
    """
    stack_models = {}
    arima_models = {}
    team_stats = {}
    model_errors = {}

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
        df_team['first_half_avg'] = df_team['rolling_avg'] * 0.6
        df_team['second_half_avg'] = df_team['rolling_avg'] * 0.4
        df_team['late_game_efficiency'] = df_team['score'] * 0.3 + df_team['season_avg'] * 0.7
        df_team['early_vs_late'] = df_team['first_half_avg'] - df_team['second_half_avg']
        df_team.rename(columns={'late_game_efficiency': 'late_game_impact'}, inplace=True)
        # Add previous game score as an additional feature (lag)
        df_team['lag_score'] = df_team['score'].shift(1).fillna(df_team['season_avg'])

        # Store team statistics
        team_stats[team] = {
            'mean': round_half(scores.mean()),
            'std': round_half(scores.std()),
            'max': round_half(scores.max()),
            'recent_form': round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
        }

        # Prepare features and target, including the new lag_score
        features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg', 'early_vs_late', 'late_game_impact', 'lag_score']]
        features = features.fillna(0)
        X = features
        y = scores

        # Time-series split (first 80% for training, remaining 20% for testing)
        split_index = int(len(scores) * 0.8)
        if split_index < 2:
            continue
        X_train = X.iloc[:split_index].values
        X_test = X.iloc[split_index:].values
        y_train = y.iloc[:split_index].values
        y_test = y.iloc[split_index:].values

        # Define a time-series cross-validator (3 splits if possible)
        n_splits = 3 if len(y_train) >= 3 else 2
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Define base models with increased iterations
        estimators = [
            ('xgb', XGBRegressor(n_estimators=200, random_state=42)),
            ('lgbm', LGBMRegressor(n_estimators=200, random_state=42)),
            ('cat', CatBoostRegressor(n_estimators=200, verbose=0, random_state=42))
        ]

        # Use a pipeline for the final estimator to scale features
        final_estimator = make_pipeline(StandardScaler(), LGBMRegressor(n_estimators=200, random_state=42))
        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            passthrough=False,
            cv=tscv
        )

        try:
            stack.fit(X_train, y_train)
            preds = stack.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            print(f"Team: {team}, Stacking Regressor MSE: {mse}")
            stack_models[team] = stack
            if team not in model_errors:
                model_errors[team] = {}
            model_errors[team]['stacking'] = mse
        except Exception as e:
            print(f"Error training Stacking Regressor for team {team}: {e}")
            continue

        # Train ARIMA if enough data exists
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
                if len(y_test) > 0:
                    arima_preds = arima.predict(n_periods=len(y_test))
                    mse_arima = mean_squared_error(y_test, arima_preds)
                    model_errors[team]['arima'] = mse_arima
            except Exception as e:
                print(f"Error training ARIMA for team {team}: {e}")
                continue

    return stack_models, arima_models, team_stats, model_errors

def make_predictions(stack_model, arima_model, X):
    """
    Makes hybrid predictions by blending Stacking Regressor and Auto-ARIMA outputs.
    This function is kept for reference.
    """
    stack_pred = stack_model.predict(X)
    if arima_model:
        arima_pred = arima_model.predict(n_periods=len(X))
        hybrid_pred = (stack_pred + arima_pred) / 2
    else:
        hybrid_pred = stack_pred
    return hybrid_pred

def predict_team_score(team, stack_models, arima_models, team_stats, team_data, model_errors):
    """
    Predicts a team's next-game score by blending the outputs of the Stacking Regressor
    and ARIMA model using error-based weighting.
    
    Returns:
      - The blended predicted score (rounded to the nearest 0.5)
      - A tuple of (lower_confidence_bound, upper_confidence_bound)
    """
    if team not in team_stats:
        return None, (None, None)

    df_team = team_data[team_data['team'] == team]
    data_len = len(df_team)
    if data_len < 3:
        return None, (None, None)

    # Use the most recent row to generate features for the next prediction.
    last_features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg', 'early_vs_late', 'late_game_impact', 'lag_score']].tail(1)
    X_next = last_features.values

    # Stacking Regressor prediction
    stack_pred = None
    if team in stack_models:
        try:
            stack_pred = float(stack_models[team].predict(X_next)[0])
        except Exception as e:
            print(f"Error predicting with Stacking Regressor for team {team}: {e}")
            stack_pred = None

    # ARIMA prediction
    arima_pred = None
    if team in arima_models:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast)
        except Exception as e:
            print(f"Error predicting with ARIMA for team {team}: {e}")
            arima_pred = None

    # Weighted ensemble using inverse MSE as weights
    if stack_pred is not None and arima_pred is not None:
        mse_stack = model_errors.get(team, {}).get('stacking', np.inf)
        mse_arima = model_errors.get(team, {}).get('arima', np.inf)
        if mse_stack <= 0:
            mse_stack = 1e-6
        if mse_arima <= 0:
            mse_arima = 1e-6
        weight_stack = 1.0 / mse_stack
        weight_arima = 1.0 / mse_arima
        ensemble = (stack_pred * weight_stack + arima_pred * weight_arima) / (weight_stack + weight_arima)
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
    if isinstance(raw_conf, (pd.Series, pd.DataFrame, np.ndarray)):
        raw_conf = raw_conf.item()

    confidence = round(min(99, max(1, 50 + raw_conf * 15)), 2)
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
    years = [current_year - i for i in range(12)]
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

    data['rolling_avg'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data['rolling_std'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    data['season_avg'] = data.groupby('team')['score'].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
    data['weighted_avg'] = (data['rolling_avg'] * 0.6) + (data['season_avg'] * 0.4)
    data['first_half_avg'] = data['rolling_avg'] * 0.6
    data['second_half_avg'] = data['rolling_avg'] * 0.4
    data['late_game_efficiency'] = data['score'] * 0.3 + data['season_avg'] * 0.7
    data['early_vs_late'] = data['first_half_avg'] - data['second_half_avg']
    data.rename(columns={'late_game_efficiency': 'late_game_impact'}, inplace=True)

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
                gl['first_half_avg'] = gl['rolling_avg'] * 0.6
                gl['second_half_avg'] = gl['rolling_avg'] * 0.4
                gl['late_game_efficiency'] = gl['PTS'] * 0.3 + gl['season_avg'] * 0.7
                gl['early_vs_late'] = gl['first_half_avg'] - gl['second_half_avg']
                gl.rename(columns={'late_game_efficiency': 'late_game_impact'}, inplace=True)

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
                            'weighted_avg': row_['weighted_avg'],
                            'early_vs_late': row_['early_vs_late'],
                            'late_game_impact': row_['late_game_impact']
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

    data['rolling_avg'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data['rolling_std'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    data['season_avg'] = data.groupby('team')['score'].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
    data['weighted_avg'] = (data['rolling_avg'] * 0.6) + (data['season_avg'] * 0.4)
    data['first_half_avg'] = data['rolling_avg'] * 0.6
    data['second_half_avg'] = data['rolling_avg'] * 0.4
    data['late_game_efficiency'] = data['score'] * 0.3 + data['season_avg'] * 0.7
    data['early_vs_late'] = data['first_half_avg'] - data['second_half_avg']
    data.rename(columns={'late_game_efficiency': 'late_game_impact'}, inplace=True)
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
# UI COMPONENTS
################################################################################
def generate_writeup(bet, team_stats_global, book_spread=None, book_total=None):
    """
    Generate an expanded detailed analysis for a given bet, including historical head-to-head,
    risk/volatility commentary and (if provided) a comparison to bookmaker odds.
    """
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

    # Expanded analysis with placeholders for additional data.
    writeup = f"""
**Detailed Analysis:**

- **{home_team} Performance:**
  - **Average Score:** {home_mean}  
    *(_Reflects the team's overall scoring consistency._)*
  - **Score Volatility:** {home_std}  
    *(_A higher number suggests greater variability._)*
  - **Recent Form (Last 5 Games):** {home_recent}

- **{away_team} Performance:**
  - **Average Score:** {away_mean}  
    *(_Shows overall offensive output._)*
  - **Score Volatility:** {away_std}  
    *(_A measure of unpredictability in performance._)*
  - **Recent Form (Last 5 Games):** {away_recent}

- **Prediction Insight:**
  Our analysis suggests that **{predicted_winner}** is likely to win with a confidence level of **{confidence}%**.  
  The model projects a spread and total points as shown in the bet details.

"""
    # If bookmaker odds are provided, add a comparison commentary.
    if book_spread is not None and book_total is not None:
        try:
            # Extract the model spread from the suggestion text
            model_spread = float(bet.get('spread_suggestion', '0').split("by")[1].strip().split()[0])
        except Exception:
            model_spread = None

        model_total = bet.get('predicted_total')
        if model_spread is not None and model_total is not None:
            spread_diff = model_spread - book_spread
            total_diff = model_total - book_total
            writeup += f"""
- **Bookmaker Odds Comparison:**
  - **Model Spread:** {model_spread:.1f} vs. **Bookmaker Spread:** {book_spread:.1f}  
    Difference: {spread_diff:.1f} points  
  - **Model Total:** {model_total:.1f} vs. **Bookmaker Total:** {book_total:.1f}  
    Difference: {total_diff:.1f} points

  **Commentary:**  
  """
            if abs(spread_diff) >= 3:
                writeup += ("The model's spread significantly deviates from the bookmaker's line, "
                            "indicating potential value if you trust our analysis.")
            else:
                writeup += "The spread prediction aligns closely with the bookmaker's line."
            writeup += "\n"
            if abs(total_diff) >= 5:
                writeup += ("There is a notable difference in the total points prediction, which might "
                            "offer an edge in over/under betting.")
            else:
                writeup += "The total points prediction is very similar to the bookmaker's line."
    else:
        writeup += "\n*No bookmaker odds were entered for comparison.*\n"

    return writeup

def display_bet_card(bet, team_stats_global, unique_key):
    """
    Displays a bet card for a given matchup with:
      - Game details and predictions.
      - Expanded analysis writeup.
      - Inline inputs for bookmaker spread and total with immediate comparison.
    
    The `unique_key` ensures that input widgets are distinct per game.
    """
    with st.container():
        st.markdown("---")
        # Game info and basic predictions
        col_game, col_inputs = st.columns([2, 1])
        with col_game:
            st.markdown(f"### **{bet['away_team']} @ {bet['home_team']}**")
            date_obj = bet['date']
            if isinstance(date_obj, datetime):
                st.caption(date_obj.strftime("%A, %B %d - %I:%M %p"))
            if bet['confidence'] >= 80:
                st.markdown("🔥 **High-Confidence Bet** 🔥")
            st.markdown(f"**Spread Suggestion:** {bet['spread_suggestion']}  " +
                        f"<span title='The suggestion is based on the predicted score difference.'>ℹ️</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**Total Suggestion:** {bet['ou_suggestion']}  " +
                        f"<span title='The total points prediction is the sum of the predicted scores.'>ℹ️</span>",
                        unsafe_allow_html=True)
            st.metric(label="Confidence", value=f"{bet['confidence']:.1f}%",
                      help="The model's confidence is derived from inverse error weighting.")
        with col_inputs:
            st.markdown("#### Enter Bookmaker Odds")
            # Unique keys ensure these inputs are per game.
            book_spread = st.number_input(
                "Spread (points)",
                min_value=-50.0,
                max_value=50.0,
                value=0.0,
                step=0.5,
                key=f"spread_{unique_key}",
                help="Enter the bookmaker's point spread."
            )
            book_total = st.number_input(
                "Total (points)",
                min_value=80.0,
                max_value=300.0,
                value=145.0,
                step=1.0,
                key=f"total_{unique_key}",
                help="Enter the bookmaker's total points line."
            )
            # Compute comparison differences (extracted similarly as before)
            try:
                model_spread = float(bet.get('spread_suggestion', '0').split("by")[1].strip().split()[0])
            except Exception:
                model_spread = None
            model_total = bet.get('predicted_total')
            if model_spread is not None and model_total is not None:
                spread_diff = model_spread - book_spread
                total_diff = model_total - book_total
                st.markdown("---")
                st.markdown("**Comparison Results:**")
                st.write(f"**Model Spread:** {model_spread:.1f} points  |  **Bookmaker Spread:** {book_spread:.1f} points")
                st.write(f"Difference (Model - Bookmaker): {spread_diff:.1f} points")
                st.write("---")
                st.write(f"**Model Total:** {model_total:.1f} points  |  **Bookmaker Total:** {book_total:.1f} points")
                st.write(f"Difference (Model - Bookmaker): {total_diff:.1f} points")
                # Provide commentary
                commentary = ""
                if abs(spread_diff) >= 3:
                    commentary += ("The model's spread significantly deviates from the bookmaker's line, "
                                   "suggesting potential value.")
                else:
                    commentary += "The spread is very close to the bookmaker's line."
                commentary += "\n"
                if abs(total_diff) >= 5:
                    commentary += ("There is a notable difference in the total points prediction, "
                                   "which might offer an edge in over/under betting.")
                else:
                    commentary += "The total prediction is similar to the bookmaker's line."
                st.info(commentary)
            else:
                st.warning("Unable to extract model predictions for comparison.")

        # Detailed insights and analysis expander
        with st.expander("Detailed Insights", expanded=False):
            st.markdown(f"**Predicted Winner:** {bet['predicted_winner']}")
            st.markdown(f"**Predicted Total Points:** {bet['predicted_total']}")
            st.markdown(f"**Prediction Margin (Diff):** {bet['predicted_diff']}")
        with st.expander("Game Analysis", expanded=False):
            # Pass the bookmaker odds to the analysis writeup so that comparison commentary appears.
            writeup = generate_writeup(bet, team_stats_global, book_spread=book_spread, book_total=book_total)
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
        stack_models, arima_models, team_stats, model_errors = train_team_models(team_data)
        team_stats_global = team_stats
        results.clear()

        for _, row in upcoming.iterrows():
            home, away = row['home_team'], row['away_team']
            home_pred, _ = predict_team_score(home, stack_models, arima_models, team_stats, team_data, model_errors)
            away_pred, _ = predict_team_score(away, stack_models, arima_models, team_stats, team_data, model_errors)

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
            for idx, bet in top_bets.iterrows():
                display_bet_card(bet, team_stats_global, unique_key=f"top_{idx}")
        else:
            st.info("No high-confidence bets found. Try lowering the threshold.")
    else:
        if results:
            st.markdown("### 📊 All Games Analysis")
            for idx, bet in enumerate(results):
                display_bet_card(bet, team_stats_global, unique_key=f"all_{idx}")
        else:
            st.info(f"No upcoming {league_choice} games found.")

################################################################################
# STREAMLIT MAIN
################################################################################
def main():
    st.set_page_config(
        page_title="FoxEdge Sports Betting Edge",
        page_icon="🦊",
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

    st.title("🦊 FoxEdge Sports Betting Insights")
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

if __name__ == "__main__":
    main()
