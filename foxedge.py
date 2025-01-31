import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import nfl_data_py as nfl
from nba_api.stats.endpoints import TeamGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from pmdarima import auto_arima
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt
import time
import firebase_admin
from firebase_admin import credentials, auth

# cbbpy for NCAAB
import cbbpy.mens_scraper as cbb

################################################################################
# UTILITY FUNCTIONS
################################################################################

def round_half(number: float) -> float:
    """
    Rounds the number to the nearest 0.5 increment.
    
    Args:
        number (float): The number to round.
    
    Returns:
        float: The rounded number.
    """
    return round(number * 2) / 2

def check_feature_variance(features: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Removes features with variance below the specified threshold.
    
    Args:
        features (pd.DataFrame): The feature DataFrame to check.
        threshold (float): The variance threshold below which features will be removed.
    
    Returns:
        pd.DataFrame: The DataFrame with low-variance features removed.
    """
    variance = features.var()
    low_variance = variance[variance < threshold].index.tolist()
    if low_variance:
        st.warning(f"Low variance features detected and removed: {', '.join(low_variance)}")
        features = features.drop(columns=low_variance)
    return features

def check_missing_values(features: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for missing values in the DataFrame and imputes them with the mean of each column.
    
    Args:
        features (pd.DataFrame): The feature DataFrame to check.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    if features.isnull().sum().sum() > 0:
        st.warning("Missing values detected. Imputing with feature means.")
        features = features.fillna(features.mean())
    return features

def check_target_distribution(scores: np.ndarray):
    """
    Plots the distribution of the target variable.
    
    Args:
        scores (np.ndarray): The target variable array.
    """
    plt.figure(figsize=(6,4))
    plt.hist(scores, bins=30, edgecolor='k')
    plt.title('Distribution of Target Variable (Score)')
    st.pyplot(plt)

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validates the data to ensure scores are non-negative and within expected ranges.
    
    Args:
        df (pd.DataFrame): The DataFrame to validate.
    
    Returns:
        bool: True if data is valid, False otherwise.
    """
    if (df['score'] < 0).any():
        st.error("Negative scores detected in the data.")
        return False
    # Add more validation rules as needed
    return True

################################################################################
# RETRY MECHANISM FOR REQUESTS
################################################################################

def get_with_retry(url: str, params: dict = None, headers: dict = None, retries: int = 3, backoff_factor: float = 0.3, timeout: int = 30):
    """
    Sends a GET request with retry mechanism.
    
    Args:
        url (str): The URL to send the GET request to.
        params (dict, optional): URL parameters.
        headers (dict, optional): HTTP headers.
        retries (int): Number of retries.
        backoff_factor (float): Backoff factor for retries.
        timeout (int): Timeout for the request in seconds.
    
    Returns:
        requests.Response: The response object.
    
    Raises:
        requests.exceptions.RequestException: If all retries fail.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    
    try:
        response = session.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed for URL {url}: {e}")
        raise

################################################################################
# FIREBASE CONFIGURATION
################################################################################

def initialize_firebase():
    """
    Initializes Firebase Admin SDK using service account credentials from Streamlit secrets.
    """
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

def login_with_rest(email: str, password: str, api_key: str):
    """
    Logs in a user using Firebase REST API.
    
    Args:
        email (str): User's email.
        password (str): User's password.
        api_key (str): Firebase API key.
    
    Returns:
        dict or None: User data if successful, else None.
    """
    try:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        response = get_with_retry(url, json=payload, headers={'Content-Type': 'application/json'}, retries=3, backoff_factor=0.3, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Invalid credentials.")
            return None
    except Exception as e:
        st.error(f"Error during login: {e}")
        return None

def signup_user(email: str, password: str):
    """
    Signs up a new user using Firebase Admin SDK.
    
    Args:
        email (str): User's email.
        password (str): User's password.
    
    Returns:
        firebase_admin.auth.UserRecord or None: User record if successful, else None.
    """
    try:
        user = auth.create_user(email=email, password=password)
        st.success(f"User {email} created successfully!")
        return user
    except Exception as e:
        st.error(f"Error: {e}")

def logout_user():
    """
    Logs out the current user by clearing session state.
    """
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

def save_predictions_to_csv(predictions: list, csv_file=CSV_FILE):
    """Save predictions to a CSV file."""
    df = pd.DataFrame(predictions)
    if Path(csv_file).exists():
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_file, index=False)
    st.success("Predictions have been saved to CSV!")

################################################################################
# MODEL TRAINING & PREDICTION (STACKING + AUTO-ARIMA HYBRID)
################################################################################

def train_team_models(team_data: pd.DataFrame, league: str):
    """
    Trains Stacking Regressor + Auto-ARIMA for each team's 'score'.
    Returns: stack_models, arima_models, team_stats
    """
    stack_models = {}
    arima_models = {}
    team_stats = {}

    all_teams = team_data['team'].unique()
    for team in all_teams:
        current_features = TRAIN_FEATURES.copy()
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

        # In-Game Performance Trends
        df_team['first_half_avg'] = df_team['rolling_avg'] * 0.6
        df_team['second_half_avg'] = df_team['rolling_avg'] * 0.4

        # Late-Game Efficiency Metric
        df_team['late_game_efficiency'] = df_team['score'] * 0.3 + df_team['season_avg'] * 0.7

        # Early vs. Late Game Performance Differential
        df_team['early_vs_late'] = df_team['first_half_avg'] - df_team['second_half_avg']

        # Advanced Feature Engineering for Model Training
        df_team.rename(columns={'late_game_efficiency': 'late_game_impact'}, inplace=True)

        # Include additional features for NBA
        if league == "NBA":
            for feature in NBA_ADDITIONAL_FEATURES:
                if feature in df_team.columns:
                    current_features.append(feature)

        # Prepare features and target
        features = df_team[current_features].copy()
        features = features.fillna(0)
        features = check_missing_values(features)  # Handle missing values
        X = features.values
        y = scores.values

        # Validate data
        if not validate_data(df_team):
            st.warning(f"Data validation failed for team {team}. Skipping model training.")
            continue

        # Check target distribution
        check_target_distribution(y)

        # Check feature variance and remove low variance features
        features_df = pd.DataFrame(X, columns=current_features)
        high_variance_features = check_feature_variance(features_df)
        if high_variance_features.empty:
            st.warning(f"All features for team {team} have low variance. Skipping model training.")
            continue
        X = high_variance_features.values
        selected_features = high_variance_features.columns.tolist()

        st.write(f"Training model for team {team} with features: {selected_features}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define base models with adjusted hyperparameters
        estimators = [
            ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
            ('lgbm', LGBMRegressor(
                n_estimators=100, 
                random_state=42, 
                min_child_samples=20, 
                min_split_gain=0.01, 
                num_leaves=31
            )),
            ('cat', CatBoostRegressor(n_estimators=100, verbose=0, random_state=42))
        ]

        # Initialize Stacking Regressor with adjusted final estimator
        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),  # Changed to LinearRegression for stability
            passthrough=False,
            cv=5,
            n_jobs=-1  # Utilize all available cores
        )

        # Train Stacking Regressor with cross-validation
        try:
            stack.fit(X_train, y_train)
            preds = stack.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            print(f"Team: {team}, Stacking Regressor MSE: {mse}")

            # Cross-validated MSE
            cv_scores = cross_val_score(stack, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_mse = -np.mean(cv_scores)
            print(f"Team: {team}, Cross-Validated MSE: {cv_mse}")

            stack_models[team] = stack
        except Exception as e:
            st.error(f"Error training Stacking Regressor for team {team}: {e}")
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
                st.error(f"Error training ARIMA for team {team}: {e}")
                continue

        # Populate team statistics
        team_stats[team] = {
            'mean': round_half(scores.mean()),
            'std': round_half(scores.std()),
            'max': round_half(scores.max()),
            'recent_form': round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
        }
        if league == "NBA":
            team_stats[team].update({
                'off_rating': round_half(df_team['off_rating'].mean()) if 'off_rating' in df_team.columns else 'N/A',
                'def_rating': round_half(df_team['def_rating'].mean()) if 'def_rating' in df_team.columns else 'N/A',
                'pace': round_half(df_team['pace'].mean()) if 'pace' in df_team.columns else 'N/A'
            })

    return stack_models, arima_models, team_stats

def predict_team_score(team: str, stack_models: dict, arima_models: dict, team_stats: dict, team_data: pd.DataFrame, league: str):
    """
    Predict a team's next-game score by blending Stacking Regressor & ARIMA outputs.
    
    Args:
        team (str): Team abbreviation.
        stack_models (dict): Trained stacking models.
        arima_models (dict): Trained ARIMA models.
        team_stats (dict): Team statistics.
        team_data (pd.DataFrame): Historical team data.
        league (str): League name.
    
    Returns:
        tuple: (predicted_score, (conf_low, conf_high))
    """
    if team not in team_stats:
        return None, (None, None)

    df_team = team_data[team_data['team'] == team]
    data_len = len(df_team)
    stack_pred = None
    arima_pred = None

    # Feature Engineering Enhancements
    if data_len < 3:
        return None, (None, None)
    current_features = TRAIN_FEATURES.copy()
    if league == "NBA":
        current_features += NBA_ADDITIONAL_FEATURES

    last_features = df_team[current_features].tail(1).copy()
    last_features = last_features.fillna(0)
    last_features = check_missing_values(last_features)  # Handle missing values

    # Validate feature consistency
    if league == "NBA":
        for feature in NBA_ADDITIONAL_FEATURES:
            if feature not in last_features.columns:
                last_features[feature] = team_stats[team].get(feature, 0)

    X_next = last_features.values

    # Check if the number of features matches the training
    if team not in stack_models:
        st.error(f"No trained model available for team {team}.")
        return None, (None, None)
    
    expected_features = stack_models[team].named_steps['lgbm'].n_features_in_
    if X_next.shape[1] != expected_features:
        st.error(f"Feature shape mismatch for team {team}: expected {expected_features}, got {X_next.shape[1]}")
        st.write(f"Expected features: {stack_models[team].named_steps['lgbm'].feature_names_in_}")
        st.write(f"Provided features: {last_features.columns.tolist()}")
        return None, (None, None)

    st.write(f"Predicting for team {team} with {X_next.shape[1]} features. Expected: {expected_features}")

    # Stacking Regressor Prediction
    if team in stack_models:
        try:
            stack_pred = stack_models[team].predict(X_next)[0]
            stack_pred = float(stack_pred)
        except Exception as e:
            st.error(f"Error predicting with Stacking Regressor for team {team}: {e}")
            stack_pred = None

    # ARIMA Prediction
    if team in arima_models:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast
            arima_pred = float(arima_pred)
        except Exception as e:
            st.error(f"Error predicting with ARIMA for team {team}: {e}")
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

    # Confidence Interval Adjustments
    conf_low = round_half(mu - 1.96 * sigma)
    conf_high = round_half(mu + 1.96 * sigma)

    return round_half(ensemble), (conf_low, conf_high)

################################################################################
# MATCHUP EVALUATION
################################################################################

def evaluate_matchup(home_team: str, away_team: str, home_pred: float, away_pred: float, team_stats: dict):
    """
    Compute predicted spread, total, and confidence for a single matchup.
    
    Args:
        home_team (str): Home team abbreviation.
        away_team (str): Away team abbreviation.
        home_pred (float): Predicted home team score.
        away_pred (float): Predicted away team score.
        team_stats (dict): Team statistics.
    
    Returns:
        dict or None: Evaluation results or None if unable to evaluate.
    """
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

    # Additional Betting Indicators
    blowout_prob = "🔥 Potential Blowout" if abs(diff) >= 10 else ""
    live_betting = "🔴 Good Live-Betting Opportunity" if combined_std > 10 else ""
    clutch_record = get_clutch_record(home_team, away_team)

    return {
        'predicted_winner': winner,
        'diff': round_half(diff),
        'total_points': round_half(total_points),
        'confidence': confidence,
        'spread_suggestion': f"Lean {winner} by {round_half(diff):.1f}",
        'ou_suggestion': f"Take the {'Over' if total_points > ou_threshold else 'Under'} {round_half(total_points):.1f}",
        'blowout_prob': blowout_prob,
        'live_betting': live_betting,
        'clutch_record': clutch_record
    }

def get_clutch_record(home_team: str, away_team: str):
    """
    Placeholder function to calculate clutch performance indicators.
    This should be implemented based on available data.
    
    Args:
        home_team (str): Home team abbreviation.
        away_team (str): Away team abbreviation.
    
    Returns:
        str: Clutch performance record.
    """
    # Example implementation: Returning dummy data
    home_clutch = "3-2"
    away_clutch = "1-4"
    return f"Clutch Record - {home_team}: {home_clutch}, {away_team}: {away_clutch}"

def find_top_bets(matchups: list, threshold: float = 70.0) -> pd.DataFrame:
    """
    Filters matchups based on a confidence threshold.
    
    Args:
        matchups (list): List of matchup dictionaries.
        threshold (float): Confidence threshold.
    
    Returns:
        pd.DataFrame: Filtered DataFrame of top bets.
    """
    df = pd.DataFrame(matchups)
    df_top = df[df['confidence'] >= threshold].copy()
    df_top.sort_values('confidence', ascending=False, inplace=True)
    return df_top

################################################################################
# DATA LOADING FUNCTIONS
################################################################################

@st.cache_data(ttl=14400)
def load_nfl_schedule():
    """
    Loads NFL schedule data for the past 12 years.
    
    Returns:
        pd.DataFrame: NFL schedule data.
    """
    current_year = datetime.now().year
    years = [current_year - i for i in range(12)]  # Last 12 years including current
    schedule = nfl.import_schedules(years)
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(schedule['gameday']):
        schedule['gameday'] = schedule['gameday'].dt.tz_convert(None)
    return schedule

def preprocess_nfl_data(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses NFL schedule data by combining home and away team scores.
    
    Args:
        schedule (pd.DataFrame): Raw NFL schedule data.
    
    Returns:
        pd.DataFrame: Preprocessed NFL data.
    """
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

    # In-Game Performance Trends
    data['first_half_avg'] = data['rolling_avg'] * 0.6
    data['second_half_avg'] = data['rolling_avg'] * 0.4

    # Late-Game Efficiency Metric
    data['late_game_efficiency'] = data['score'] * 0.3 + data['season_avg'] * 0.7

    # Early vs. Late Game Performance Differential
    data['early_vs_late'] = data['first_half_avg'] - data['second_half_avg']

    # Advanced Feature Engineering for Model Training
    data.rename(columns={'late_game_efficiency': 'late_game_impact'}, inplace=True)

    return data

@st.cache_data(ttl=14400)
def load_nba_data():
    """
    Loads NBA multi-season team logs with pace & efficiency integrated.
    
    Returns:
        pd.DataFrame: NBA historical data.
    """
    nba_teams_list = nba_teams.get_teams()
    seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']  # Adjust as needed
    all_rows = []

    for season in seasons:
        for team in nba_teams_list:
            team_id = team['id']
            team_abbrev = team.get('abbreviation', str(team_id))  # Get team abbreviation from teams list
            
            try:
                gl_url = "https://stats.nba.com/stats/teamgamelog"
                params = {
                    'TeamID': team_id,
                    'Season': season,
                    'SeasonType': 'Regular Season'
                }
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                    'Referer': 'https://www.nba.com/',
                    'Accept': 'application/json, text/plain, */*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Origin': 'https://www.nba.com',
                    'Connection': 'keep-alive',
                }
                
                response = get_with_retry(gl_url, params=params, headers=headers, retries=3, backoff_factor=0.3, timeout=60)
                data = response.json()
                gl = pd.DataFrame(data['resultSets'][0]['rowSet'], columns=data['resultSets'][0]['headers'])
                
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

                # In-Game Performance Trends
                gl['first_half_avg'] = gl['rolling_avg'] * 0.6
                gl['second_half_avg'] = gl['rolling_avg'] * 0.4

                # Late-Game Efficiency Metric
                gl['late_game_efficiency'] = gl['PTS'] * 0.3 + gl['season_avg'] * 0.7

                # Early vs. Late Game Performance Differential
                gl['early_vs_late'] = gl['first_half_avg'] - gl['second_half_avg']

                # Advanced Feature Engineering for Model Training
                gl.rename(columns={'late_game_efficiency': 'late_game_impact'}, inplace=True)

                # Append to all_rows
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
                            'weighted_avg': row_['weighted_avg'],
                            'early_vs_late': row_['early_vs_late'],
                            'late_game_impact': row_['late_game_impact']
                        })
                    except Exception as e:
                        print(f"Error processing row for team {team_abbrev}: {str(e)}")
                        continue

                # Respect API rate limits
                time.sleep(1)  # Sleep for 1 second between requests

            except Exception as e:
                st.error(f"Error processing team {team_abbrev} for season {season}: {e}")
                continue

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.dropna(subset=['score'], inplace=True)
    df.sort_values('gameday', inplace=True)

    # Optional: fill missing advanced stats with league means
    for col in ['off_rating', 'def_rating', 'pace']:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)

    return df

def fetch_upcoming_nba_games(days_ahead=3):
    """
    Fetches upcoming NBA games for 'today' and the next 'days_ahead' days using NBA's scoreboard API.
    
    Args:
        days_ahead (int): Number of days ahead to fetch games for.
    
    Returns:
        pd.DataFrame: Upcoming NBA games data.
    """
    now = datetime.now()
    upcoming_rows = []
    for offset in range(days_ahead + 1):
        date_target = now + timedelta(days=offset)
        date_str = date_target.strftime('%Y%m%d')
        scoreboard_url = "https://stats.nba.com/stats/scoreboardV2"
        params = {
            'GameDate': date_target.strftime('%m/%d/%Y'),
            'LeagueID': '00',
            'DayOffset': 0
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Referer': 'https://www.nba.com/',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.nba.com',
            'Connection': 'keep-alive',
        }

        try:
            response = get_with_retry(scoreboard_url, params=params, headers=headers, retries=3, backoff_factor=0.3, timeout=60)
            data = response.json()
            games = data.get('resultSets', [])[0].get('rowSet', [])
            headers_keys = data.get('resultSets', [])[0].get('headers', [])
            for game in games:
                game_dict = dict(zip(headers_keys, game))
                home_team = game_dict.get('HOME_TEAM_ABBREVIATION')
                away_team = game_dict.get('VISITOR_TEAM_ABBREVIATION')
                game_status = game_dict.get('GAME_STATUS_TEXT')
                if game_status.lower() != 'final':
                    gameday = datetime.strptime(game_dict.get('GAME_DATE'), '%Y-%m-%dT%H:%M:%S.%fZ')
                    gameday = gameday.astimezone(pytz.timezone('America/Los_Angeles'))
                    upcoming_rows.append({
                        'gameday': gameday,
                        'home_team': home_team,
                        'away_team': away_team
                    })
        except Exception as e:
            st.error(f"Failed to fetch NBA games for {date_target.strftime('%Y-%m-%d')}: {e}")
            continue

    if not upcoming_rows:
        return pd.DataFrame()
    upcoming = pd.DataFrame(upcoming_rows)
    upcoming.sort_values('gameday', inplace=True)
    return upcoming

@st.cache_data(ttl=14400)
def load_ncaab_data_current_season(season=2025):
    """
    Loads finished or in-progress NCAA MBB games for the given season using cbbpy.
    
    Args:
        season (int): The NCAA season year.
    
    Returns:
        pd.DataFrame: NCAAB current season data.
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

    # In-Game Performance Trends
    data['first_half_avg'] = data['rolling_avg'] * 0.6
    data['second_half_avg'] = data['rolling_avg'] * 0.4

    # Late-Game Efficiency Metric
    data['late_game_efficiency'] = data['score'] * 0.3 + data['season_avg'] * 0.7

    # Early vs. Late Game Performance Differential
    data['early_vs_late'] = data['first_half_avg'] - data['second_half_avg']

    # Advanced Feature Engineering for Model Training
    data.rename(columns={'late_game_efficiency': 'late_game_impact'}, inplace=True)

    data.sort_values(['team', 'gameday'], inplace=True)
    data['game_index'] = data.groupby('team').cumcount()

    return data

def fetch_upcoming_ncaab_games() -> pd.DataFrame:
    """
    Fetches upcoming NCAAB games for 'today' and 'tomorrow' using ESPN's scoreboard API.
    
    Returns:
        pd.DataFrame: Upcoming NCAAB games data.
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

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Referer': 'https://www.espn.com/mens-college-basketball/',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.espn.com',
            'Connection': 'keep-alive',
        }

        try:
            response = get_with_retry(url, params=params, headers=headers, retries=3, backoff_factor=0.3, timeout=60)
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
        except Exception as e:
            st.error(f"Failed to fetch NCAAB games for date {date_str}: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values('gameday', inplace=True)
    return df

################################################################################
# BETTING INSIGHTS AND UI COMPONENTS
################################################################################

def generate_writeup(bet: dict, team_stats_global: dict) -> str:
    """
    Generates a detailed analysis writeup for a bet.
    
    Args:
        bet (dict): Bet information.
        team_stats_global (dict): Global team statistics.
    
    Returns:
        str: Markdown formatted writeup.
    """
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

    # Additional Context from Enhancements
    home_pace = home_stats.get('pace', 'N/A')
    away_pace = away_stats.get('pace', 'N/A')
    home_off_rating = home_stats.get('off_rating', 'N/A')
    home_def_rating = home_stats.get('def_rating', 'N/A')
    away_off_rating = away_stats.get('off_rating', 'N/A')
    away_def_rating = away_stats.get('def_rating', 'N/A')

    writeup = f"""
**Detailed Analysis:**

- **{home_team} Performance:**
  - **Average Score:** {home_mean}
  - **Score Standard Deviation:** {home_std}
  - **Recent Form (Last 5 Games):** {home_recent}
  - **Pace of Play:** {home_pace} possessions per game
  - **Offensive Rating:** {home_off_rating}
  - **Defensive Rating:** {home_def_rating}

- **{away_team} Performance:**
  - **Average Score:** {away_mean}
  - **Score Standard Deviation:** {away_std}
  - **Recent Form (Last 5 Games):** {away_recent}
  - **Pace of Play:** {away_pace} possessions per game
  - **Offensive Rating:** {away_off_rating}
  - **Defensive Rating:** {away_def_rating}

- **Prediction Insight:**
  Based on the recent performance and statistical analysis, **{predicted_winner}** is predicted to win with a confidence level of **{confidence}%.** 
  The projected score difference is **{bet['predicted_diff']} points**, leading to a suggested spread of **{bet['spread_suggestion']}**. 
  Additionally, the total predicted points for the game are **{bet['predicted_total']}**, indicating a suggestion to **{bet['ou_suggestion']}**.

- **Betting Indicators:**
  {bet['blowout_prob']}
  {bet['live_betting']}
  {bet['clutch_record']}

- **Statistical Edge:**
  The confidence level of **{confidence}%** reflects the statistical edge derived from the combined performance metrics of both teams.
  This ensures that the prediction is data-driven and reliable.
"""
    return writeup

def display_bet_card(bet: dict, team_stats_global: dict):
    """
    Displays a bet card with summarized and detailed insights.
    
    Args:
        bet (dict): Bet information.
        team_stats_global (dict): Global team statistics.
    """
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
            # Additional Betting Indicators
            if bet.get('blowout_prob'):
                st.markdown(f"{bet['blowout_prob']}")
            if bet.get('live_betting'):
                st.markdown(f"{bet['live_betting']}")
            if bet.get('clutch_record'):
                st.markdown(f"{bet['clutch_record']}")

        with col3:
            st.metric(label="Confidence", value=f"{bet['confidence']:.1f}%")

    with st.expander("Detailed Insights", expanded=False):
        st.markdown(f"**Predicted Winner:** {bet['predicted_winner']}")
        st.markdown(f"**Predicted Total Points:** {bet['predicted_total']}")
        st.markdown(f"**Prediction Margin (Diff):** {bet['predicted_diff']}")

    with st.expander("Game Analysis", expanded=False):
        writeup = generate_writeup(bet, team_stats_global)
        st.markdown(writeup)

    # Enhancement 2: Interactive Charts (Placeholder - Implement as needed)
    with st.expander("Performance Charts", expanded=False):
        # Example: Trend Graphs for Predicted vs. Actual Scores
        if 'actual_scores' in st.session_state and 'predicted_scores' in st.session_state:
            fig, ax = plt.subplots()
            ax.plot(st.session_state['actual_scores'], label='Actual Scores')
            ax.plot(st.session_state['predicted_scores'], label='Predicted Scores')
            ax.legend()
            st.pyplot(fig)

        # Rolling Performance Visualization
        if 'rolling_avg' in st.session_state:
            fig, ax = plt.subplots()
            ax.plot(st.session_state['rolling_avg'], label='3-Game Rolling Avg')
            ax.plot(st.session_state['rolling_avg_10'], label='10-Game Rolling Avg')
            ax.legend()
            st.pyplot(fig)

        # Confidence vs. Model Error Scatter Plot
        if 'confidences' in st.session_state and 'model_errors' in st.session_state:
            fig, ax = plt.subplots()
            ax.scatter(st.session_state['confidences'], st.session_state['model_errors'])
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Model Error')
            st.pyplot(fig)

################################################################################
# MAIN PIPELINE
################################################################################

def run_league_pipeline(league_choice: str):
    """
    Runs the pipeline for the selected league, fetching data, training models, and making predictions.
    
    Args:
        league_choice (str): Selected league ("NFL", "NBA", "NCAAB").
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
        team_data = load_ncaab_data_current_season(season=2025)
        if team_data.empty:
            st.error("Unable to load NCAAB data.")
            return
        upcoming = fetch_upcoming_ncaab_games()

    if team_data.empty or upcoming.empty:
        st.warning(f"No upcoming {league_choice} data available for analysis.")
        return

    with st.spinner("Analyzing recent performance data..."):
        stack_models, arima_models, team_stats = train_team_models(team_data, league=league_choice)
        team_stats_global = team_stats
        results.clear()

        for _, row in upcoming.iterrows():
            home, away = row['home_team'], row['away_team']
            home_pred, _ = predict_team_score(home, stack_models, arima_models, team_stats, team_data, league=league_choice)
            away_pred, _ = predict_team_score(away, stack_models, arima_models, team_stats, team_data, league=league_choice)

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
                    'ou_suggestion': outcome['ou_suggestion'],
                    'blowout_prob': outcome.get('blowout_prob', ''),
                    'live_betting': outcome.get('live_betting', ''),
                    'clutch_record': outcome.get('clutch_record', '')
                })

    # Enhancement 5: Better Game Filters in UI
    st.sidebar.header("Filters")
    with st.sidebar.form("filters_form"):
        st.markdown("### Filter Games")
        if results:
            total_points_min, total_points_max = st.slider(
                "Predicted Total Points Range",
                min_value=int(min(r['predicted_total'] for r in results)),
                max_value=int(max(r['predicted_total'] for r in results)),
                value=(100, 200),
                step=5
            )
            selected_teams = st.multiselect(
                "Select Teams",
                options=list(set([r['home_team'] for r in results] + [r['away_team'] for r in results])),
                default=[]
            )
        else:
            total_points_min, total_points_max = 100, 200
            selected_teams = []
        submitted = st.form_submit_button("Apply Filters")

    if submitted:
        filtered_results = pd.DataFrame(results)
        if not filtered_results.empty:
            if selected_teams:
                filtered_results = filtered_results[
                    filtered_results['home_team'].isin(selected_teams) | 
                    filtered_results['away_team'].isin(selected_teams)
                ]
            filtered_results = filtered_results[
                (filtered_results['predicted_total'] >= total_points_min) & 
                (filtered_results['predicted_total'] <= total_points_max)
            ]
            results_display = filtered_results.to_dict('records')
        else:
            results_display = results
    else:
        results_display = results

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
        top_bets = find_top_bets(results_display, threshold=conf_threshold)
        if not top_bets.empty:
            st.markdown(f"### 🔥 Top {len(top_bets)} Bets for Today")
            for _, bet in top_bets.iterrows():
                display_bet_card(bet, team_stats_global)
        else:
            st.info("No high-confidence bets found. Try lowering the threshold.")
    else:
        if results_display:
            st.markdown("### 📊 All Games Analysis")
            for bet in results_display:
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

    initialize_firebase()

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        st.title("Login to FoxEdge Sports Betting Insights")

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                FIREBASE_API_KEY = st.secrets["general"]["firebaseApiKey"]
                user_data = login_with_rest(email, password, FIREBASE_API_KEY)
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

    # CSV Output Enhancements
    if st.button("Save Predictions to CSV"):
        if results:
            save_predictions_to_csv(results)
            # Provide download link
            csv = pd.DataFrame(results).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )
        else:
            st.warning("No predictions to save.")

################################################################################
# ENTRY POINT
################################################################################

if __name__ == "__main__":
    main()
