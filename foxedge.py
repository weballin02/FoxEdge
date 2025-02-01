import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import nfl_data_py as nfl
from nba_api.stats.endpoints import ScoreboardV2, TeamGameLog
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from pmdarima import auto_arima
from pathlib import Path
import requests
import firebase_admin
from firebase_admin import credentials, auth
import joblib
import os

# cbbpy for NCAAB data
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
except Exception as e:
    st.warning(f"Firebase configuration error: {e}")

def login_with_rest(email, password):
    """Logs in a user using Firebase REST API."""
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
    """Creates a new user in Firebase."""
    try:
        user = auth.create_user(email=email, password=password)
        st.success(f"User {email} created successfully!")
        return user
    except Exception as e:
        st.error(f"Error creating user: {e}")

def logout_user():
    """Logs out the current user."""
    for key in ['email', 'logged_in']:
        st.session_state.pop(key, None)

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
            "spread_suggestion", "ou_suggestion", "confidence", 
            "market_spread", "market_over_under"
        ]
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

def save_predictions_to_csv(predictions, csv_file=CSV_FILE):
    """Save predictions to a CSV file."""
    try:
        df = pd.DataFrame(predictions)
        if Path(csv_file).exists():
            existing_df = pd.read_csv(csv_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(csv_file, index=False)
        st.success("Predictions have been saved to CSV!")
    except Exception as e:
        st.error(f"Error saving CSV: {e}")

################################################################################
# UTILITY FUNCTIONS
################################################################################
def round_half(number):
    """Rounds a number to the nearest 0.5."""
    try:
        return round(number * 2) / 2
    except Exception:
        return number

def safe_numeric(val, default=0):
    """Convert val to a number or return default if conversion fails."""
    try:
        return float(val)
    except Exception:
        return default

################################################################################
# NFL SCORING LINES (MARKET DATA) LOADING
################################################################################
@st.cache_data(ttl=14400)
def load_nfl_scoring_lines():
    """
    Loads NFL scoring lines using nfl_data_py's import_sc_lines endpoint.
    Expected columns: game_id, season, week, home_team, away_team, spread, over_under.
    """
    try:
        sc_lines = nfl.import_sc_lines()
        if not sc_lines.empty:
            sc_lines['spread'] = pd.to_numeric(sc_lines['spread'], errors='coerce')
            sc_lines['over_under'] = pd.to_numeric(sc_lines['over_under'], errors='coerce')
        return sc_lines
    except Exception as e:
        st.warning(f"Error loading NFL scoring lines: {e}")
        return pd.DataFrame()

################################################################################
# MODEL TRAINING & PREDICTION (STACKING + AUTO-ARIMA HYBRID)
################################################################################
@st.cache_data(ttl=14400)
def train_team_models(team_data: pd.DataFrame):
    """
    Trains a stacking regressor and an auto-ARIMA model for each team's 'score'
    using time-series–aware splitting and cross-validation.
    
    In addition to standard features (rolling averages, lag scores, etc.),
    extra numeric predictors (if available) such as 'is_home', 'rank', 'wins', 'losses',
    'off_rating', 'def_rating', and 'pace' are used.
    
    To robustly handle missing data, for each team:
      - Missing-indicator columns (binary flags for each predictor) are added.
      - A KNN imputer is trained on the combined features.
      - The imputer is stored and later used for predicting the next game.
      
    Returns:
      - stack_models: dict of trained stacking regressors per team.
      - arima_models: dict of trained ARIMA models per team.
      - team_stats: dict of aggregated team statistics.
      - model_errors: dict of training errors for each model.
      - imputer_models: dict mapping each team to a tuple (imputer, feature_columns)
    """
    stack_models = {}
    arima_models = {}
    team_stats = {}
    model_errors = {}
    imputer_models = {}
    
    all_teams = team_data['team'].unique()
    for team in all_teams:
        try:
            df_team = team_data[team_data['team'] == team].copy()
            df_team.sort_values('gameday', inplace=True)
            # Ensure score is numeric
            df_team['score'] = pd.to_numeric(df_team['score'], errors='coerce')
            scores = df_team['score'].dropna().reset_index(drop=True)
            if len(scores) < 3:
                continue

            # Compute rolling and season aggregates (with error handling)
            try:
                df_team['rolling_avg'] = df_team['score'].rolling(window=3, min_periods=1).mean()
            except Exception as e:
                st.warning(f"Rolling avg error for team {team}: {e}")
                df_team['rolling_avg'] = df_team['score']

            try:
                df_team['rolling_std'] = df_team['score'].rolling(window=3, min_periods=1).std().fillna(0)
            except Exception as e:
                st.warning(f"Rolling std error for team {team}: {e}")
                df_team['rolling_std'] = 0

            try:
                df_team['season_avg'] = df_team['score'].expanding().mean()
            except Exception as e:
                st.warning(f"Season avg error for team {team}: {e}")
                df_team['season_avg'] = df_team['score']

            df_team['weighted_avg'] = (df_team.get('rolling_avg', df_team['score']) * 0.6) + \
                                      (df_team.get('season_avg', df_team['score']) * 0.4)
            df_team['first_half_avg'] = df_team.get('rolling_avg', df_team['score']) * 0.6
            df_team['second_half_avg'] = df_team.get('rolling_avg', df_team['score']) * 0.4
            df_team['late_game_impact'] = df_team['score'] * 0.3 + df_team.get('season_avg', df_team['score']) * 0.7
            df_team['early_vs_late'] = df_team['first_half_avg'] - df_team['second_half_avg']

            # Add lag score feature
            df_team['lag_score'] = df_team['score'].shift(1).fillna(df_team.get('season_avg', df_team['score']))

            # Store aggregated team statistics
            team_stats[team] = {
                'mean': round_half(scores.mean()),
                'std': round_half(scores.std()),
                'max': round_half(scores.max()),
                'recent_form': round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
            }

            # Ensure extra features exist (if missing, create with default value 0)
            for col in ['is_home', 'rank', 'wins', 'losses', 'off_rating', 'def_rating', 'pace']:
                if col not in df_team.columns:
                    df_team[col] = 0

            # Define the base set of feature columns
            feature_columns = [
                'rolling_avg', 'rolling_std', 'weighted_avg',
                'early_vs_late', 'late_game_impact', 'lag_score',
                'is_home', 'rank', 'wins', 'losses', 'off_rating', 'def_rating', 'pace'
            ]
            # Create missing indicators for each predictor
            X_original = df_team[feature_columns]
            indicators = X_original.isnull().astype(float)
            indicators.columns = [col + '_missing' for col in indicators.columns]
            X_combined = pd.concat([X_original, indicators], axis=1)

            # Use KNNImputer for advanced imputation
            imputer = KNNImputer(n_neighbors=3)
            X_imputed = imputer.fit_transform(X_combined)
            # Store the imputer and the column names (so we can later re-create the same feature set)
            imputer_models[team] = (imputer, X_combined.columns)

            X_train = X_imputed
            y = df_team['score']
            split_index = int(len(y) * 0.8)
            if split_index < 2:
                continue
            X_train_data = X_train[:split_index]
            X_test_data = X_train[split_index:]
            y_train = y.iloc[:split_index].values
            y_test = y.iloc[split_index:].values

            n_splits = 3 if len(y_train) >= 3 else 2
            tscv = TimeSeriesSplit(n_splits=n_splits)

            # Define base models
            estimators = [
                ('xgb', XGBRegressor(n_estimators=200, random_state=42)),
                ('lgbm', LGBMRegressor(n_estimators=200, random_state=42)),
                ('cat', CatBoostRegressor(n_estimators=200, verbose=0, random_state=42))
            ]
            final_estimator = make_pipeline(StandardScaler(), LGBMRegressor(n_estimators=200, random_state=42))
            stack = StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator,
                passthrough=False,
                cv=tscv
            )
            try:
                stack.fit(X_train_data, y_train)
                preds = stack.predict(X_test_data)
                mse = mean_squared_error(y_test, preds)
                st.write(f"Team: {team}, Stacking Regressor MSE: {mse}")
                stack_models[team] = stack
                model_errors.setdefault(team, {})['stacking'] = mse
            except Exception as e:
                st.write(f"Error training stacking model for team {team}: {e}")
                continue

            # Train ARIMA if sufficient data exists
            if len(y) >= 7:
                try:
                    arima = auto_arima(
                        y,
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
                    st.write(f"Error training ARIMA for team {team}: {e}")
                    continue
        except Exception as ex:
            st.write(f"Error processing team {team}: {ex}")
            continue

    return stack_models, arima_models, team_stats, model_errors, imputer_models

def make_predictions(stack_model, arima_model, X):
    """Makes hybrid predictions by blending stacking regressor and ARIMA outputs."""
    try:
        stack_pred = stack_model.predict(X)
    except Exception as e:
        st.write(f"Error in stacking prediction: {e}")
        stack_pred = np.zeros(len(X))
    try:
        if arima_model:
            arima_pred = arima_model.predict(n_periods=len(X))
        else:
            arima_pred = np.zeros(len(X))
    except Exception as e:
        st.write(f"Error in ARIMA prediction: {e}")
        arima_pred = np.zeros(len(X))
    hybrid_pred = (stack_pred + arima_pred) / 2
    return hybrid_pred

def predict_team_score(team, stack_models, arima_models, team_stats, team_data, model_errors, imputer_models, is_home=None):
    """
    Predicts a team's next-game score by blending stacking and ARIMA outputs using inverse-MSE weighting.
    
    The function extracts the most recent row's features from team_data, re-creates the combined
    feature set (including missing indicators), and uses the stored KNN imputer to impute missing values.
    
    An optional 'is_home' flag can override the existing value.
    
    Returns:
      - Blended predicted score (rounded to the nearest 0.5)
      - A tuple (lower_confidence_bound, upper_confidence_bound)
    """
    if team not in team_stats:
        return None, (None, None)
    df_team = team_data[team_data['team'] == team]
    if len(df_team) < 3:
        return None, (None, None)
    try:
        last_features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg', 'early_vs_late', 'late_game_impact', 'lag_score',
                                   'is_home', 'rank', 'wins', 'losses', 'off_rating', 'def_rating', 'pace']].tail(1)
    except Exception as e:
        st.write(f"Error extracting features for team {team}: {e}")
        return None, (None, None)
    # Override is_home if provided
    if 'is_home' in last_features.columns:
        last_features = last_features.copy()
        last_features['is_home'] = is_home if is_home is not None else last_features['is_home']
    # Create missing indicators for the same columns
    indicators = last_features.isnull().astype(float)
    indicators.columns = [col + '_missing' for col in indicators.columns]
    X_combined = pd.concat([last_features, indicators], axis=1)
    # Use stored imputer for this team
    if team in imputer_models:
        imputer, columns_used = imputer_models[team]
        # Ensure X_combined has the same columns as used during training
        X_combined = X_combined.reindex(columns=columns_used, fill_value=0)
        X_next = imputer.transform(X_combined)
    else:
        X_next = X_combined.fillna(0).values

    # Predict using stacking model
    stack_pred = None
    try:
        if team in stack_models:
            stack_pred = float(stack_models[team].predict(X_next)[0])
    except Exception as e:
        st.write(f"Error predicting with stacking model for team {team}: {e}")
    # Predict using ARIMA
    arima_pred = None
    try:
        if team in arima_models:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast)
    except Exception as e:
        st.write(f"Error predicting with ARIMA for team {team}: {e}")

    # Ensemble using inverse MSE weighting
    if stack_pred is not None and arima_pred is not None:
        mse_stack = model_errors.get(team, {}).get('stacking', np.inf)
        mse_arima = model_errors.get(team, {}).get('arima', np.inf)
        mse_stack = mse_stack if mse_stack > 0 else 1e-6
        mse_arima = mse_arima if mse_arima > 0 else 1e-6
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
    try:
        conf_low = round_half(mu - 1.96 * sigma)
        conf_high = round_half(mu + 1.96 * sigma)
    except Exception:
        conf_low, conf_high = mu, mu
    return round_half(ensemble), (conf_low, conf_high)

def evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats, market_spread=None, market_over_under=None):
    """
    Computes predicted spread, total points, and confidence for a matchup.
    If NFL market data is available, the predicted difference is compared with the market spread;
    if close, the confidence is boosted.
    """
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
    extra_info = {}
    if market_spread is not None:
        try:
            market_diff = abs(diff - safe_numeric(market_spread))
            if market_diff < combined_std:
                confidence = min(99, confidence + 5)
            extra_info['market_spread'] = round_half(safe_numeric(market_spread))
        except Exception:
            extra_info['market_spread'] = None
    else:
        extra_info['market_spread'] = None
    if market_over_under is not None:
        try:
            extra_info['market_over_under'] = round_half(safe_numeric(market_over_under))
        except Exception:
            extra_info['market_over_under'] = None
    else:
        extra_info['market_over_under'] = None
    ou_threshold = 145  # default threshold for total points suggestion
    result = {
        'predicted_winner': winner,
        'diff': round_half(diff),
        'total_points': round_half(total_points),
        'confidence': confidence,
        'spread_suggestion': f"Lean {winner} by {round_half(diff):.1f}",
        'ou_suggestion': f"Take the {'Over' if total_points > ou_threshold else 'Under'} {round_half(total_points):.1f}"
    }
    result.update(extra_info)
    return result

def find_top_bets(matchups, threshold=70.0):
    """Filters and sorts matchup predictions by a minimum confidence threshold."""
    try:
        df = pd.DataFrame(matchups)
        df_top = df[df['confidence'] >= threshold].copy()
        df_top.sort_values('confidence', ascending=False, inplace=True)
        return df_top
    except Exception as e:
        st.write(f"Error filtering top bets: {e}")
        return pd.DataFrame()

################################################################################
# NFL DATA LOADING & PREPROCESSING
################################################################################
@st.cache_data(ttl=14400)
def load_nfl_schedule():
    """Loads NFL schedule data from nfl_data_py."""
    try:
        current_year = datetime.now().year
        years = [current_year - i for i in range(12)]
        schedule = nfl.import_schedules(years)
        schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
        if pd.api.types.is_datetime64tz_dtype(schedule['gameday']):
            schedule['gameday'] = schedule['gameday'].dt.tz_convert(None)
        return schedule
    except Exception as e:
        st.error(f"Error loading NFL schedule: {e}")
        return pd.DataFrame()

def preprocess_nfl_data(schedule):
    """
    Preprocesses NFL schedule data by merging home and away scores.
    Adds an 'is_home' flag and adjusts for neutral-site games if available.
    """
    try:
        home_df = schedule[['gameday', 'home_team', 'home_score']].rename(
            columns={'home_team': 'team', 'home_score': 'score'}
        )
        home_df['is_home'] = 1
    except Exception as e:
        st.write(f"Error processing home data: {e}")
        home_df = pd.DataFrame()
    try:
        away_df = schedule[['gameday', 'away_team', 'away_score']].rename(
            columns={'away_team': 'team', 'away_score': 'score'}
        )
        away_df['is_home'] = 0
    except Exception as e:
        st.write(f"Error processing away data: {e}")
        away_df = pd.DataFrame()
    try:
        if 'is_neutral' in schedule.columns:
            neutral_mask = schedule['is_neutral'] == True
            home_df.loc[neutral_mask, 'is_home'] = 0.5
            away_df.loc[neutral_mask, 'is_home'] = 0.5
    except Exception as e:
        st.write(f"Error adjusting for neutral sites: {e}")
    try:
        data = pd.concat([home_df, away_df], ignore_index=True)
        data.dropna(subset=['score'], inplace=True)
        data.sort_values('gameday', inplace=True)
        data['rolling_avg'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        data['rolling_std'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
        data['season_avg'] = data.groupby('team')['score'].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
        data['weighted_avg'] = (data['rolling_avg'] * 0.6) + (data['season_avg'] * 0.4)
        data['first_half_avg'] = data['rolling_avg'] * 0.6
        data['second_half_avg'] = data['rolling_avg'] * 0.4
        data['late_game_impact'] = data['score'] * 0.3 + data['season_avg'] * 0.7
        data['early_vs_late'] = data['first_half_avg'] - data['second_half_avg']
        data.rename(columns={'late_game_impact': 'late_game_impact'}, inplace=True)
        return data
    except Exception as e:
        st.error(f"Error during NFL data preprocessing: {e}")
        return pd.DataFrame()

def fetch_upcoming_nfl_games(schedule, days_ahead=7):
    """Retrieves upcoming NFL games (without scores) within a specified timeframe."""
    try:
        upcoming = schedule[schedule['home_score'].isna() & schedule['away_score'].isna()].copy()
        now = datetime.now()
        filter_date = now + timedelta(days=days_ahead)
        upcoming = upcoming[upcoming['gameday'] <= filter_date].copy()
        upcoming.sort_values('gameday', inplace=True)
        return upcoming[['gameday', 'home_team', 'away_team']]
    except Exception as e:
        st.error(f"Error fetching upcoming NFL games: {e}")
        return pd.DataFrame()

################################################################################
# NBA DATA LOADING & PREPROCESSING
################################################################################
@st.cache_data(ttl=14400)
def load_nba_data():
    """
    Loads NBA team-level data across multiple seasons.
    Computes an 'is_home' flag using the MATCHUP column (if available) and advanced metrics.
    """
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
                gl['GAME_DATE'] = pd.to_datetime(gl['GAME_DATE'], errors='coerce')
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
                gl['late_game_impact'] = gl['PTS'] * 0.3 + gl['season_avg'] * 0.7
                gl['early_vs_late'] = gl['first_half_avg'] - gl['second_half_avg']
                gl.rename(columns={'late_game_impact': 'late_game_impact'}, inplace=True)
                for idx, row_ in gl.iterrows():
                    try:
                        is_home = 1 if ("@" not in row_.get('MATCHUP', '')) else 0
                    except Exception:
                        is_home = 0
                    all_rows.append({
                        'gameday': row_['GAME_DATE'],
                        'team': team_abbrev,
                        'score': safe_numeric(row_.get('PTS', 0)),
                        'off_rating': row_.get('OFF_RATING', np.nan),
                        'def_rating': row_.get('DEF_RATING', np.nan),
                        'pace': row_.get('PACE', np.nan),
                        'rolling_avg': row_.get('rolling_avg', 0),
                        'rolling_std': row_.get('rolling_std', 0),
                        'season_avg': row_.get('season_avg', 0),
                        'weighted_avg': row_.get('weighted_avg', 0),
                        'early_vs_late': row_.get('early_vs_late', 0),
                        'late_game_impact': row_.get('late_game_impact', 0),
                        'is_home': is_home
                    })
            except Exception as e:
                st.write(f"Error processing NBA team {team_abbrev} for season {season}: {e}")
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
    """Retrieves upcoming NBA games from ESPN's ScoreboardV2 API."""
    now = datetime.now()
    upcoming_rows = []
    for offset in range(days_ahead + 1):
        try:
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
                    'home_team': g.get('HOME_TEAM_ABBREV', ''),
                    'away_team': g.get('AWAY_TEAM_ABBREV', '')
                })
        except Exception as e:
            st.write(f"Error fetching NBA games for offset {offset}: {e}")
            continue
    if not upcoming_rows:
        return pd.DataFrame()
    upcoming = pd.DataFrame(upcoming_rows)
    upcoming.sort_values('gameday', inplace=True)
    return upcoming

################################################################################
# NCAAB DATA LOADING & PREPROCESSING
################################################################################
@st.cache_data(ttl=14400)
def load_ncaab_data_current_season(season=2025):
    """
    Loads finished or in-progress NCAA MBB games for the given season using cbbpy.
    Extracts basic scores and, if available, extra info (team ranking, win-loss record).
    """
    try:
        info_df, _, _ = cbb.get_games_season(season=season, info=True, box=False, pbp=False)
    except Exception as e:
        st.error(f"Error loading NCAAB season data: {e}")
        return pd.DataFrame()
    if info_df.empty:
        return pd.DataFrame()
    try:
        if not pd.api.types.is_datetime64_any_dtype(info_df["game_day"]):
            info_df["game_day"] = pd.to_datetime(info_df["game_day"], errors="coerce")
    except Exception as e:
        st.write(f"Error converting game_day: {e}")
    try:
        home_df = info_df.rename(columns={
            "home_team": "team",
            "home_score": "score",
            "game_day": "gameday"
        })[["gameday", "team", "score"]]
        home_df['is_home'] = 1
        if "home_rank" in info_df.columns:
            home_df["rank"] = pd.to_numeric(info_df["home_rank"], errors='coerce').fillna(0)
        if "home_record" in info_df.columns:
            rec = info_df["home_record"].str.split("-", expand=True)
            home_df["wins"] = pd.to_numeric(rec[0], errors='coerce').fillna(0)
            home_df["losses"] = pd.to_numeric(rec[1], errors='coerce').fillna(0)
    except Exception as e:
        st.write(f"Error processing NCAAB home data: {e}")
        home_df = pd.DataFrame()
    try:
        away_df = info_df.rename(columns={
            "away_team": "team",
            "away_score": "score",
            "game_day": "gameday"
        })[["gameday", "team", "score"]]
        away_df['is_home'] = 0
        if "away_rank" in info_df.columns:
            away_df["rank"] = pd.to_numeric(info_df["away_rank"], errors='coerce').fillna(0)
        if "away_record" in info_df.columns:
            rec = info_df["away_record"].str.split("-", expand=True)
            away_df["wins"] = pd.to_numeric(rec[0], errors='coerce').fillna(0)
            away_df["losses"] = pd.to_numeric(rec[1], errors='coerce').fillna(0)
    except Exception as e:
        st.write(f"Error processing NCAAB away data: {e}")
        away_df = pd.DataFrame()
    try:
        data = pd.concat([home_df, away_df], ignore_index=True)
        data.dropna(subset=["score"], inplace=True)
        data.sort_values("gameday", inplace=True)
        data['rolling_avg'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        data['rolling_std'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
        data['season_avg'] = data.groupby('team')['score'].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
        data['weighted_avg'] = (data['rolling_avg'] * 0.6) + (data['season_avg'] * 0.4)
        data['first_half_avg'] = data['rolling_avg'] * 0.6
        data['second_half_avg'] = data['rolling_avg'] * 0.4
        data['late_game_impact'] = data['score'] * 0.3 + data['season_avg'] * 0.7
        data['early_vs_late'] = data['first_half_avg'] - data['second_half_avg']
        data.rename(columns={'late_game_impact': 'late_game_impact'}, inplace=True)
        data.sort_values(['team', 'gameday'], inplace=True)
        data['game_index'] = data.groupby('team').cumcount()
        return data
    except Exception as e:
        st.error(f"Error processing NCAAB data: {e}")
        return pd.DataFrame()

def fetch_upcoming_ncaab_games() -> pd.DataFrame:
    """Fetches upcoming NCAAB games for 'today' and 'tomorrow' using ESPN's scoreboard API."""
    timezone = pytz.timezone('America/Los_Angeles')
    current_time = datetime.now(timezone)
    dates = [current_time.strftime('%Y%m%d'),
             (current_time + timedelta(days=1)).strftime('%Y%m%d')]
    rows = []
    for date_str in dates:
        try:
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
                try:
                    game_time_str = game.get('date', '')
                    game_time = datetime.fromisoformat(game_time_str[:-1]).astimezone(timezone)
                    competitors = game['competitions'][0]['competitors']
                    home_comp = next((c for c in competitors if c['homeAway'] == 'home'), None)
                    away_comp = next((c for c in competitors if c['homeAway'] == 'away'), None)
                    if not home_comp or not away_comp:
                        continue
                    home_team = home_comp['team'].get('displayName', '')
                    away_team = away_comp['team'].get('displayName', '')
                    rows.append({'gameday': game_time, 'home_team': home_team, 'away_team': away_team})
                except Exception as ex:
                    st.write(f"Error processing a NCAAB game: {ex}")
                    continue
        except Exception as e:
            st.write(f"Error fetching NCAAB games for date {date_str}: {e}")
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.sort_values('gameday', inplace=True)
    return df

################################################################################
# UI COMPONENTS & WRITEUP GENERATION
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
  Based on recent performance and statistical analysis, **{predicted_winner}** is predicted to win with a confidence of **{confidence}%**.
  The projected score difference is **{bet['predicted_diff']} points** and total points are **{bet['predicted_total']}**.
  Market data (if available) suggests a spread of **{bet.get('market_spread', 'N/A')}** and an over/under of **{bet.get('market_over_under', 'N/A')}**.

- **Statistical Edge:**
  The confidence reflects the edge derived from the combined performance metrics.
"""
    return writeup

def display_bet_card(bet, team_stats_global):
    with st.container():
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"### **{bet['away_team']} @ {bet['home_team']}**")
            date_obj = bet.get('date')
            if isinstance(date_obj, datetime):
                st.caption(date_obj.strftime("%A, %B %d - %I:%M %p"))
        with col2:
            if bet.get('confidence', 0) >= 80:
                st.markdown("🔥 **High-Confidence Bet** 🔥")
            st.markdown(f"**Spread Suggestion:** {bet.get('spread_suggestion', '')}")
            st.markdown(f"**Total Suggestion:** {bet.get('ou_suggestion', '')}")
        with col3:
            st.metric(label="Confidence", value=f"{bet.get('confidence', 0):.1f}%")
    with st.expander("Detailed Insights", expanded=False):
        st.markdown(f"**Predicted Winner:** {bet.get('predicted_winner', '')}")
        st.markdown(f"**Predicted Total Points:** {bet.get('predicted_total', '')}")
        st.markdown(f"**Prediction Margin (Diff):** {bet.get('predicted_diff', '')}")
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
    """
    Main pipeline that loads data, trains models, makes predictions,
    and displays betting insights.
    
    For NFL, market scoring lines are merged; for predictions the stored
    imputer (with missing-indicator features) is used.
    """
    global results, team_stats_global
    st.header(f"Today's {league_choice} Best Bets 🎯")
    if league_choice == "NFL":
        schedule = load_nfl_schedule()
        if schedule.empty:
            st.error("Unable to load NFL schedule.")
            return
        team_data = preprocess_nfl_data(schedule)
        upcoming = fetch_upcoming_nfl_games(schedule, days_ahead=7)
        sc_lines = load_nfl_scoring_lines()
        if not sc_lines.empty and not upcoming.empty:
            try:
                upcoming['home_team_lower'] = upcoming['home_team'].str.lower()
                upcoming['away_team_lower'] = upcoming['away_team'].str.lower()
                sc_lines['home_team_lower'] = sc_lines['home_team'].str.lower()
                sc_lines['away_team_lower'] = sc_lines['away_team'].str.lower()
                upcoming = upcoming.merge(
                    sc_lines[['home_team_lower', 'away_team_lower', 'spread', 'over_under']],
                    on=['home_team_lower', 'away_team_lower'],
                    how='left'
                )
            except Exception as e:
                st.write(f"Error merging NFL market data: {e}")
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
        (stack_models, arima_models, team_stats, model_errors,
         imputer_models) = train_team_models(team_data)
        team_stats_global = team_stats
        results.clear()
        for _, row in upcoming.iterrows():
            home = row.get('home_team')
            away = row.get('away_team')
            home_pred, _ = predict_team_score(home, stack_models, arima_models, team_stats, team_data, model_errors, imputer_models, is_home=1)
            away_pred, _ = predict_team_score(away, stack_models, arima_models, team_stats, team_data, model_errors, imputer_models, is_home=0)
            market_spread = row.get('spread') if league_choice == "NFL" else None
            market_over_under = row.get('over_under') if league_choice == "NFL" else None
            outcome = evaluate_matchup(home, away, home_pred, away_pred, team_stats, market_spread, market_over_under)
            if outcome:
                results.append({
                    'date': row.get('gameday'),
                    'league': league_choice,
                    'home_team': home,
                    'away_team': away,
                    'home_pred': home_pred,
                    'away_pred': away_pred,
                    'predicted_winner': outcome.get('predicted_winner'),
                    'predicted_diff': outcome.get('diff'),
                    'predicted_total': outcome.get('total_points'),
                    'confidence': outcome.get('confidence'),
                    'spread_suggestion': outcome.get('spread_suggestion'),
                    'ou_suggestion': outcome.get('ou_suggestion'),
                    'market_spread': outcome.get('market_spread'),
                    'market_over_under': outcome.get('market_over_under')
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
                    st.session_state['email'] = user_data.get('email', 'Unknown')
                    st.success(f"Welcome, {st.session_state['email']}!")
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
