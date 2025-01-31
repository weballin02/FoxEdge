import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import nfl_data_py as nfl
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2, TeamGameLog
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
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
    return round(number * 2) / 2

################################################################################
# ADVANCED FEATURE ENGINEERING
################################################################################
def enhance_features(df_team: pd.DataFrame, league: str) -> pd.DataFrame:
    """Enhanced feature engineering for better prediction accuracy"""
    
    # 1️⃣ Momentum Features
    df_team['last_5_trend'] = df_team['score'].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    
    # 2️⃣ Home/Away Performance
    df_team['home_avg'] = df_team[df_team['is_home'] == 1]['score'].expanding().mean()
    df_team['away_avg'] = df_team[df_team['is_home'] == 0]['score'].expanding().mean()
    
    # 3️⃣ Opponent Strength (Assuming 'opponent_rank' exists; otherwise, set to 0)
    if 'opponent_rank' in df_team.columns:
        df_team['opp_strength'] = df_team['opponent_rank'].rolling(5).mean()
    else:
        df_team['opp_strength'] = 0  # Placeholder if opponent_rank not available
    
    # 4️⃣ Rest Days Impact
    df_team['days_rest'] = df_team['gameday'].diff().dt.days
    df_team['rest_factor'] = df_team['days_rest'].apply(
        lambda x: 1.05 if x > 3 else (0.95 if x < 2 else 1)
    )
    
    # 5️⃣ Variance Features
    df_team['score_volatility'] = (
        df_team['score'].rolling(10).std() / df_team['score'].rolling(10).mean()
    )
    df_team['score_volatility'] = df_team['score_volatility'].fillna(0)
    
    # 6️⃣ Recent Form (Last 10 games weighted by recency)
    weights = np.array([1.1**i for i in range(10)])
    df_team['weighted_form'] = df_team['score'].rolling(10).apply(
        lambda x: np.average(x, weights=weights[:len(x)]) if len(x) > 0 else 0
    )
    
    return df_team

################################################################################
# ADVANCED MODEL ARCHITECTURE
################################################################################
def build_advanced_stack():
    """Enhanced stacking model architecture"""
    
    # Base Models with Tuned Parameters
    base_models = [
        ('xgb', XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )),
        ('lgbm', LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )),
        ('cat', CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            subsample=0.8,
            rsm=0.8,
            random_state=42,
            verbose=0
        ))
    ]
    
    # Meta-model
    meta_model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42
    )
    
    # Stacking with cross-validation
    stack = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    return stack

################################################################################
# MODEL TRAINING & PREDICTION (STACKING + AUTO-ARIMA HYBRID)
################################################################################
@st.cache_data(ttl=14400)
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

        # Apply advanced feature engineering
        df_team = enhance_features(df_team, league=league)

        # In-Game Performance Trends
        df_team['first_half_avg'] = df_team['rolling_avg'] * 0.6
        df_team['second_half_avg'] = df_team['rolling_avg'] * 0.4

        # Late-Game Efficiency Metric
        df_team['late_game_efficiency'] = df_team['score'] * 0.3 + df_team['season_avg'] * 0.7

        # Early vs. Late Game Performance Differential
        df_team['early_vs_late'] = df_team['first_half_avg'] - df_team['second_half_avg']

        # Advanced Feature Engineering for Model Training
        df_team.rename(columns={'late_game_efficiency': 'late_game_impact'}, inplace=True)

        # Populate team statistics
        team_stats[team] = {
            'mean': round_half(scores.mean()),
            'std': round_half(scores.std()),
            'max': round_half(scores.max()),
            'recent_form': round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
        }

        # Prepare features and target
        feature_columns = [
            'rolling_avg', 'rolling_std', 'weighted_avg', 'early_vs_late', 'late_game_impact',
            'last_5_trend', 'home_avg', 'away_avg', 'opp_strength',
            'rest_factor', 'score_volatility', 'weighted_form'
        ]
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in df_team.columns:
                df_team[col] = 0  # or appropriate default value

        features = df_team[feature_columns]
        features = features.fillna(0)
        X = features.values
        y = scores.values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize advanced stacking regressor
        stack = build_advanced_stack()

        # Train Stacking Regressor with cross-validation
        try:
            stack.fit(X_train, y_train)
            preds = stack.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            st.write(f"Team: {team}, Stacking Regressor MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
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

    return stack_models, arima_models, team_stats

def make_predictions(stack_model, arima_model, X):
    """
    Makes hybrid predictions by blending Stacking Regressor and Auto-ARIMA.
    """
    stack_pred = stack_model.predict(X)
    if arima_model:
        arima_pred = arima_model.predict(n_periods=len(X))
        hybrid_pred = (stack_pred + arima_pred) / 2
    else:
        hybrid_pred = stack_pred
    return hybrid_pred

def predict_team_score(team, stack_models, arima_models, team_stats, team_data, league):
    """Predict a team's next-game score by blending Stacking Regressor & ARIMA outputs."""
    if team not in team_stats:
        return None, (None, None)

    df_team = team_data[team_data['team'] == team]
    data_len = len(df_team)
    stack_pred = None
    arima_pred = None

    # Feature Engineering Enhancements
    if data_len < 3:
        return None, (None, None)
    last_features = df_team[[
        'rolling_avg', 'rolling_std', 'weighted_avg', 'early_vs_late', 'late_game_impact',
        'last_5_trend', 'home_avg', 'away_avg', 'opp_strength',
        'rest_factor', 'score_volatility', 'weighted_form'
    ]].tail(1)
    X_next = last_features.values

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
            arima_pred = float(forecast[0])
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

    # Enhanced Confidence Interval Adjustments
    conf_low = round_half(mu - 1.96 * sigma)
    conf_high = round_half(mu + 1.96 * sigma)

    return round_half(ensemble), (conf_low, conf_high)

################################################################################
# ENHANCED PREDICTION CONFIDENCE
################################################################################
def calculate_advanced_confidence(
    home_pred: float,
    away_pred: float,
    home_stats: dict,
    away_stats: dict,
    recent_games: pd.DataFrame,
    league: str
) -> float:
    """
    Calculate prediction confidence using multiple factors
    """
    # Base confidence from score difference
    diff = abs(home_pred - away_pred)
    combined_std = np.sqrt(home_stats['std']**2 + away_stats['std']**2)
    base_conf = 50 + (diff / combined_std) * 15 if combined_std > 0 else 50

    # Recent performance factor
    home_form_std = recent_games[recent_games['team'] == home_stats.get('team')]['score'].std()
    away_form_std = recent_games[recent_games['team'] == away_stats.get('team')]['score'].std()
    form_factor = 1 - ((home_form_std + away_form_std) / (2 * combined_std)) if combined_std > 0 else 1

    # Sample size factor
    home_games = len(recent_games[recent_games['team'] == home_stats.get('team')])
    away_games = len(recent_games[recent_games['team'] == away_stats.get('team')])
    sample_factor = min(1, (home_games + away_games) / 40)  # Normalize to reasonable game count

    # Head-to-head history factor
    h2h_games = recent_games[
        ((recent_games['team'] == home_stats.get('team')) & (recent_games['opponent'] == away_stats.get('team'))) |
        ((recent_games['team'] == away_stats.get('team')) & (recent_games['opponent'] == home_stats.get('team')))
    ]
    h2h_factor = 1 + (0.1 * len(h2h_games))  # Increase confidence with more H2H data

    # Combined confidence
    confidence = base_conf * form_factor * sample_factor * h2h_factor

    return min(99, max(50, confidence))

################################################################################
# LEAGUE-SPECIFIC ADJUSTMENTS
################################################################################
def apply_league_adjustments(
    prediction: float,
    league: str,
    is_home: bool,
    game_context: dict
) -> float:
    """Apply league-specific adjustments to predictions"""
    
    adjustments = {
        'NBA': {
            'home_advantage': 2.5,
            'back_to_back_penalty': -3.0,
            'rest_advantage': 1.5,
            'altitude_factor': 1.02  # Denver/Utah effect
        },
        'NFL': {
            'home_advantage': 2.0,
            'weather_factor': 0.95,  # Bad weather adjustment
            'division_game': 1.1,    # Division game intensity
            'prime_time': 1.05       # Prime time performance
        },
        'NCAAB': {
            'home_advantage': 3.5,
            'conference_game': 1.15,
            'rivalry_game': 1.1,
            'tournament': 1.05
        }
    }
    
    league_adj = adjustments.get(league, {})
    adjusted_pred = prediction
    
    # Apply home court/field advantage
    if is_home:
        adjusted_pred += league_adj.get('home_advantage', 0)
    
    # Apply league-specific context adjustments
    if league == 'NBA':
        if game_context.get('back_to_back'):
            adjusted_pred += league_adj.get('back_to_back_penalty', 0)
        if game_context.get('altitude_game'):
            adjusted_pred *= league_adj.get('altitude_factor', 1)
    
    elif league == 'NFL':
        if game_context.get('bad_weather'):
            adjusted_pred *= league_adj.get('weather_factor', 1)
        if game_context.get('division_game'):
            adjusted_pred *= league_adj.get('division_game', 1)
        if game_context.get('prime_time'):
            adjusted_pred *= league_adj.get('prime_time', 1)
    
    elif league == 'NCAAB':
        if game_context.get('conference_game'):
            adjusted_pred *= league_adj.get('conference_game', 1)
        if game_context.get('rivalry_game'):
            adjusted_pred *= league_adj.get('rivalry_game', 1)
        if game_context.get('tournament_game'):
            adjusted_pred *= league_adj.get('tournament', 1)
    
    return adjusted_pred

################################################################################
# VALIDATION AND MODEL SELECTION
################################################################################
def validate_predictions(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    league: str
) -> dict:
    """
    Comprehensive validation of prediction accuracy
    """
    predictions = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions)
    }
    
    # Placeholder for spread and total accuracy calculations
    # These would require additional data about actual spreads and totals
    spread_accuracy = np.nan  # To be implemented based on actual spread data
    total_accuracy = np.nan   # To be implemented based on actual total points data
    
    return {
        **metrics,
        'spread_accuracy': spread_accuracy,
        'total_accuracy': total_accuracy
    }

################################################################################
# MATCHUP EVALUATION
################################################################################
def evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats, league, game_context):
    """Compute predicted spread, total, and confidence for a single matchup."""
    if home_pred is None or away_pred is None:
        return None

    diff = home_pred - away_pred
    total_points = home_pred + away_pred

    home_std = team_stats.get(home_team, {}).get('std', 5)
    away_std = team_stats.get(away_team, {}).get('std', 5)
    combined_std = max(1.0, (home_std + away_std) / 2)

    # Enhanced Confidence Calculation
    # Assuming 'recent_games' DataFrame is available globally or passed appropriately
    # For this example, we'll use empty recent_games
    recent_games = pd.DataFrame()  # Placeholder: Replace with actual recent games data
    home_stats = team_stats.get(home_team, {})
    away_stats = team_stats.get(away_team, {})
    confidence = calculate_advanced_confidence(
        home_pred, away_pred, home_stats, away_stats, recent_games, league
    )

    winner = home_team if diff > 0 else away_team

    # Example threshold for NCAAB. Adjust if needed for NBA or NFL.
    ou_threshold = 145 if league == "NCAAB" else (200 if league == "NBA" else 55)

    # Apply league-specific adjustments
    game_context_updated = game_context.copy()  # Copy to prevent mutation
    adjusted_home_pred = apply_league_adjustments(home_pred, league, is_home=True, game_context=game_context_updated)
    adjusted_away_pred = apply_league_adjustments(away_pred, league, is_home=False, game_context=game_context_updated)
    
    # Recalculate diff and total_points after adjustments
    adjusted_diff = adjusted_home_pred - adjusted_away_pred
    adjusted_total_points = adjusted_home_pred + adjusted_away_pred

    # Suggestion based on adjusted predictions
    spread_suggestion = f"Lean {winner} by {round_half(adjusted_diff):.1f}"
    ou_suggestion = f"Take the {'Over' if adjusted_total_points > ou_threshold else 'Under'} {round_half(adjusted_total_points):.1f}"

    return {
        'predicted_winner': winner,
        'diff': round_half(adjusted_diff),
        'total_points': round_half(adjusted_total_points),
        'confidence': confidence,
        'spread_suggestion': spread_suggestion,
        'ou_suggestion': ou_suggestion
    }

################################################################################
# FIND TOP BETS
################################################################################
def find_top_bets(matchups, threshold=70.0):
    df = pd.DataFrame(matchups)
    df_top = df[df['confidence'] >= threshold].copy()
    df_top.sort_values('confidence', ascending=False, inplace=True)
    return df_top

################################################################################
# DATA LOADING FUNCTIONS
################################################################################
@st.cache_data(ttl=14400)
def load_nfl_schedule():
    current_year = datetime.now().year
    years = [current_year - i for i in range(12)]  # Last 12 years including current
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

    # Feature Engineering Enhancements
    data['is_home'] = data['team'].isin(schedule['home_team']).astype(int)
    data['rolling_avg'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data['rolling_std'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    
    # Correctly compute expanding mean using apply
    data['season_avg'] = data.groupby('team')['score'].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
    
    data['weighted_avg'] = (data['rolling_avg'] * 0.6) + (data['season_avg'] * 0.4)

    # Apply advanced feature engineering
    data = enhance_features(data, league='NFL')

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
# NBA DATA LOADING (ADVANCED LOGIC IMPLEMENTED)
################################################################################
@st.cache_data(ttl=14400)
def load_nba_data():
    """Load multi-season team logs with pace & efficiency integrated."""
    nba_teams_list = nba_teams.get_teams()
    seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']  # Adjust as needed
    all_rows = []

    for season in seasons:
        for team in nba_teams_list:
            team_id = team['id']
            team_abbrev = team.get('abbreviation', str(team_id))  # Get team abbreviation from teams list
            
            try:
                gl = TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
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

                # Apply advanced feature engineering
                gl = enhance_features(gl, league='NBA')

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
                            'late_game_impact': row_['late_game_impact'],
                            'last_5_trend': row_['last_5_trend'],
                            'home_avg': row_['home_avg'],
                            'away_avg': row_['away_avg'],
                            'opp_strength': row_['opp_strength'],
                            'rest_factor': row_['rest_factor'],
                            'score_volatility': row_['score_volatility'],
                            'weighted_form': row_['weighted_form']
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

    # Optional: fill missing advanced stats with league means
    for col in ['off_rating', 'def_rating', 'pace']:
        df[col].fillna(df[col].mean(), inplace=True)

    return df

def fetch_upcoming_nba_games(days_ahead=3):
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
        except Exception as e:
            st.error(f"Failed to fetch NBA games for date {date_str}: {e}")
            continue
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

    # Convert "game_day" to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(info_df["game_day"]):
        info_df["game_day"] = pd.to_datetime(info_df["game_day"], errors="coerce")

    home_df = info_df.rename(columns={
        "home_team": "team",
        "home_score": "score",
        "game_day": "gameday"
    })[[
        "gameday", "team", "score",
        "last_5_trend", "home_avg", "away_avg", "opp_strength",
        "rest_factor", "score_volatility", "weighted_form"
    ]]
    home_df['is_home'] = 1

    away_df = info_df.rename(columns={
        "away_team": "team",
        "away_score": "score",
        "game_day": "gameday"
    })[[
        "gameday", "team", "score",
        "last_5_trend", "home_avg", "away_avg", "opp_strength",
        "rest_factor", "score_volatility", "weighted_form"
    ]]
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

    # Apply advanced feature engineering
    data = enhance_features(data, league='NCAAB')

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

        try:
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
        except Exception as e:
            st.error(f"Failed to fetch NCAAB games for date {date_str}: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values('gameday', inplace=True)
    return df

################################################################################
# UI COMPONENTS
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
  Based on the recent performance and statistical analysis, **{predicted_winner}** is predicted to win with a confidence level of **{confidence}%.** 
  The projected score difference is **{bet['predicted_diff']} points**, leading to a suggested spread of **{bet['spread_suggestion']}**. 
  Additionally, the total predicted points for the game are **{bet['predicted_total']}**, indicating a suggestion to **{bet['ou_suggestion']}**.

- **Statistical Edge:**
  The confidence level of **{confidence}%** reflects the statistical edge derived from the combined performance metrics of both teams.
  This ensures that the prediction is data-driven and reliable.
"""
    return writeup

def display_bet_card(bet, team_stats_global):
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
            st.metric(label="Confidence", value=f"{bet['confidence']:.1f}%")

    with st.expander("Detailed Insights", expanded=False):
        st.markdown(f"**Predicted Winner:** {bet['predicted_winner']}")
        st.markdown(f"**Predicted Total Points:** {bet['predicted_total']}")
        st.markdown(f"**Prediction Margin (Diff):** {bet['predicted_diff']}")

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

            # Define game context based on league and other factors
            game_context = {}  # Placeholder: Populate based on actual game data

            outcome = evaluate_matchup(home, away, home_pred, away_pred, team_stats, league_choice, game_context)
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
