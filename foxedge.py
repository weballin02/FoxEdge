import streamlit as st 
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import nfl_data_py as nfl
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import firebase_admin
from firebase_admin import credentials, auth
import requests
import cbbpy.mens_scraper as cbb

########################################
# CONSTANTS AND CONFIGURATIONS
########################################
# NBA Specific Constants
NBA_KEY_NUMBERS = {
    'spreads': [2.5, 3.5, 5.5, 7.5, 9.5, 11.5],
    'totals': [205, 210, 215, 220, 225, 230]
}

# Model Configuration
MODEL_CONFIG = {
    'xgboost': {
        'learning_rate': 0.01,
        'max_depth': 5,
        'n_estimators': 200,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror'
    },
    'gradient_boosting': {
        'learning_rate': 0.1,
        'max_depth': 4,
        'n_estimators': 100,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
}

########################################
# ENHANCED PREDICTION MODELS
########################################
class EnhancedNBAPredictionModel:
    def __init__(self):
        self.xgb_model = None
        self.gbr_model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'rolling_mean_3', 'rolling_std_3',
            'rolling_mean_5', 'rolling_std_5',
            'season_avg', 'season_std',
            'home_court_advantage',
            'rest_days',
            'streak',
            'last_10_performance'
        ]
    
    def prepare_features(self, data):
        features = data[self.feature_columns].copy()
        return self.scaler.fit_transform(features)
    
    def train(self, data):
        X = self.prepare_features(data)
        y = data['score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(**MODEL_CONFIG['xgboost'])
        self.xgb_model.fit(X_train, y_train)
        
        # Train GradientBoosting
        self.gbr_model = GradientBoostingRegressor(**MODEL_CONFIG['gradient_boosting'])
        self.gbr_model.fit(X_train, y_train)
        
        # Calculate model weights based on performance
        xgb_score = self.xgb_model.score(X_test, y_test)
        gbr_score = self.gbr_model.score(X_test, y_test)
        
        total_score = xgb_score + gbr_score
        self.xgb_weight = xgb_score / total_score
        self.gbr_weight = gbr_score / total_score
    
    def predict(self, features):
        X = self.scaler.transform(features[self.feature_columns])
        xgb_pred = self.xgb_model.predict(X)
        gbr_pred = self.gbr_model.predict(X)
        
        # Weighted ensemble prediction
        final_pred = (xgb_pred * self.xgb_weight + 
                     gbr_pred * self.gbr_weight)
        
        return final_pred

########################################
# ENHANCED BETTING ANALYSIS
########################################
class BettingAnalyzer:
    def __init__(self):
        self.key_numbers = NBA_KEY_NUMBERS
    
    def calculate_edge(self, predicted_value, market_value, variance):
        """Calculate betting edge with Kelly Criterion"""
        edge = abs(predicted_value - market_value)
        edge_percentage = (edge / variance) * 100
        
        # Kelly Criterion calculation
        win_prob = self._calculate_win_probability(edge, variance)
        fair_odds = 1 / win_prob
        kelly_bet = max(0, min(0.05, (win_prob - (1 - win_prob)) / 1))  # Cap at 5%
        
        return {
            'edge_percentage': edge_percentage,
            'kelly_percentage': kelly_bet * 100,
            'win_probability': win_prob * 100
        }
    
    def _calculate_win_probability(self, edge, variance):
        """Calculate win probability using normal distribution"""
        z_score = edge / (variance ** 0.5)
        win_prob = 0.5 + (0.5 * np.tanh(z_score * np.pi / (2 * 2**0.5)))
        return win_prob
    
    def analyze_key_numbers(self, predicted_value, bet_type='spread'):
        """Analyze proximity to key betting numbers"""
        key_numbers = (self.key_numbers['spreads'] if bet_type == 'spread' 
                      else self.key_numbers['totals'])
        
        closest_key = min(key_numbers, key=lambda x: abs(x - predicted_value))
        distance = abs(predicted_value - closest_key)
        
        return {
            'closest_key': closest_key,
            'distance': distance,
            'is_key_number': distance < 0.5
        }
    
    def generate_betting_insight(self, game_data, predictions):
        """Generate comprehensive betting insights"""
        insights = []
        
        for pred in predictions:
            home_variance = pred.get('home_variance', 10)
            away_variance = pred.get('away_variance', 10)
            
            # Analyze spread
            spread_edge = self.calculate_edge(
                pred['predicted_diff'],
                pred.get('market_spread', 0),
                (home_variance + away_variance) ** 0.5
            )
            
            # Analyze total
            total_edge = self.calculate_edge(
                pred['predicted_total'],
                pred.get('market_total', 0),
                (home_variance + away_variance) ** 0.5
            )
            
            # Key number analysis
            spread_key = self.analyze_key_numbers(abs(pred['predicted_diff']), 'spread')
            total_key = self.analyze_key_numbers(pred['predicted_total'], 'total')
            
            insight = {
                'game_id': f"{pred['away_team']}@{pred['home_team']}",
                'spread_edge': spread_edge,
                'total_edge': total_edge,
                'spread_key_numbers': spread_key,
                'total_key_numbers': total_key,
                'recommended_bets': []
            }
            
            # Generate bet recommendations
            if spread_edge['edge_percentage'] > 5:
                insight['recommended_bets'].append({
                    'type': 'spread',
                    'edge': spread_edge['edge_percentage'],
                    'kelly': spread_edge['kelly_percentage'],
                    'win_prob': spread_edge['win_probability']
                })
            
            if total_edge['edge_percentage'] > 5:
                insight['recommended_bets'].append({
                    'type': 'total',
                    'edge': total_edge['edge_percentage'],
                    'kelly': total_edge['kelly_percentage'],
                    'win_prob': total_edge['win_probability']
                })
            
            insights.append(insight)
        
        return insights

########################################
# ENHANCED DATA PROCESSING
########################################
def enhance_team_data(data: pd.DataFrame) -> pd.DataFrame:
    """Add advanced features to team data"""
    enhanced = data.copy()
    
    # Add home court advantage
    enhanced['home_court_advantage'] = enhanced['is_home'].map({1: 2.5, 0: 0})
    
    # Calculate rest days (if game dates available)
    enhanced['rest_days'] = enhanced.groupby('team')['gameday'].diff().dt.days.fillna(2)
    
    # Calculate streak (positive for wins, negative for losses)
    enhanced['streak'] = enhanced.groupby('team')['score'].apply(
        lambda x: x.expanding().mean() - x.shift(1).expanding().mean()
    ).fillna(0)
    
    # Last 10 games performance
    enhanced['last_10_performance'] = enhanced.groupby('team')['score'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )
    
    # Add pace and efficiency metrics if available
    # This would require additional data sources
    
    return enhanced

########################################
# FIREBASE CONFIGURATION (unchanged)
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

########################################
# CSV MANAGEMENT (unchanged)
########################################
CSV_FILE = "predictions.csv"

def initialize_csv(csv_file=CSV_FILE):
    """Initialize the CSV file if it doesn't exist."""
    if not Path(csv_file).exists():
        columns = [
            "date", "league", "home_team", "away_team", "home_pred", "away_pred",
            "predicted_winner", "total", "spread_suggestion", "ou_suggestion"
        ]
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

def save_predictions_to_csv(predictions, csv_file=CSV_FILE):
    """Save predictions to a CSV file."""
    df = pd.DataFrame(predictions)
    df.to_csv(csv_file, mode='a', index=False, header=not Path(csv_file).exists())
    st.success("Predictions have been saved to CSV!")

########################################
# UTILITY
########################################
def round_half(number):
    return round(number * 2) / 2

########################################
# ENHANCEMENTS & DATA QUALITY CHECKS
########################################
def remove_outliers(df, score_col='score', z_threshold=3):
    """
    Removes outliers from the dataset based on a Z-threshold for the score column.
    Typically used prior to or after merging league data.
    """
    mean_val = df[score_col].mean()
    std_val = df[score_col].std()
    if std_val == 0 or np.isnan(std_val):
        return df  # No outlier removal if no variation
    df_filtered = df[np.abs(df[score_col] - mean_val) <= z_threshold * std_val]
    return df_filtered

def fallback_league_average(team_df, min_games_threshold=3):
    """
    If the team has fewer than `min_games_threshold` data points,
    replace their scores with the league average.
    """
    if len(team_df) < min_games_threshold:
        league_avg = team_df['score'].mean()  # This is a bit contrived in a single-team subset, see usage logic
        team_df['score'] = league_avg
    return team_df

########################################
# TRAINING & PREDICTIONS
########################################

# Toggle for weighted ensemble vs. simple average
USE_WEIGHTED_ENSEMBLE = True

@st.cache_data(ttl=3600)
def train_team_models(team_data: pd.DataFrame):
    """
    Trains GradientBoostingRegressor (with expanded hyperparameter tuning) 
    and ARIMA for each team. Returns gbr_models, arima_models, team_stats, and
    optional metrics such as RMSE to enable weighted ensembles.
    """
    gbr_models = {}
    arima_models = {}
    team_stats = {}
    # Store model performance metrics (RMSE) for dynamic weighting
    gbr_rmse_dict = {}
    arima_rmse_dict = {}

    all_teams = team_data['team'].unique()

    # --- Example for advanced feature engineering placeholders ---
    # If available, you could add rolling_mean_5, rolling_std_5, season_avg, etc. to your team_data,
    # same as you did rolling_mean_3. Already done for each league below, but further customization can happen here.
    
    for team in all_teams:
        # Extract data for this team & fallback if insufficient data
        team_df = team_data[team_data['team'] == team].copy()
        team_df.sort_values('gameday', inplace=True)
        # Optional outlier removal for each team
        team_df = remove_outliers(team_df, score_col='score', z_threshold=3)
        # Fallback if fewer than 3 games
        team_df = fallback_league_average(team_df, min_games_threshold=3)
        scores = team_df['score'].reset_index(drop=True)

        if len(scores) < 3:
            continue

        # Basic descriptive stats for this team
        recent_5 = scores.tail(5).mean() if len(scores) >= 5 else scores.mean()
        team_stats[team] = {
            'mean': round_half(scores.mean()),
            'std': round_half(scores.std()),
            'max': round_half(scores.max()),
            'recent_form': round_half(recent_5)
        }

        # ---------- Train GradientBoostingRegressor (GBR) ----------
        # Expanded hyperparameter grid
        if len(scores) >= 10:
            # Prepare features: game_index, is_home, rolling_mean_3, rolling_std_3
            # You can add rolling_mean_5, etc., if you’ve computed them in the data
            candidate_cols = ['game_index', 'is_home', 'rolling_mean_3', 'rolling_std_3',
                               'rolling_mean_5', 'rolling_std_5', 'season_avg', 'season_std']
            # Only use columns that actually exist
            feature_cols = [c for c in candidate_cols if c in team_df.columns]
            X = team_df[feature_cols].values
            y = scores.values

            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200],
                'max_depth': [2, 3, 4],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 3]
            }
            gbr = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(
                estimator=gbr,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=3,
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X, y)
            best_gbr = grid_search.best_estimator_
            gbr_models[team] = best_gbr

            # Evaluate via cross_val_score for an RMSE estimate
            cv_neg_mse = cross_val_score(best_gbr, X, y, cv=3, scoring='neg_mean_squared_error')
            cv_rmse = np.mean(np.sqrt(-cv_neg_mse))
            gbr_rmse_dict[team] = cv_rmse
        else:
            gbr_rmse_dict[team] = None

        # ---------- Train ARIMA (with optional seasonality) ----------
        # If data is too short, skip
        if len(scores) >= 7:
            # Example: enabling seasonality if periodic patterns exist.
            # If you have exogenous data (e.g., venue, injury, etc.), pass exog=some_df
            arima = auto_arima(
                scores,
                seasonal=True,  # Set to True if you suspect seasonal
                m=5,            # Season length; tune to league specifics
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                max_p=5,
                max_q=5
            )
            # You could compare AIC here or store it
            arima_models[team] = arima

            # ARIMA pseudo-evaluation:
            # A quick approach is to do an in-sample forecast. For real evaluation, do a train/test split.
            try:
                # Forecast last 3 points as a simple check
                steps = min(3, len(scores))
                train_cutoff = len(scores) - steps
                model_scores = scores.iloc[:train_cutoff]
                arima_fit = auto_arima(
                    model_scores,
                    seasonal=True,
                    m=5,
                    max_p=5,
                    max_q=5,
                    error_action='ignore',
                    suppress_warnings=True
                )
                preds = arima_fit.predict(n_periods=steps)
                actuals = scores.iloc[train_cutoff:]
                if len(preds) == len(actuals):
                    rmse_arima = mean_squared_error(actuals, preds, squared=False)
                    arima_rmse_dict[team] = rmse_arima
                else:
                    arima_rmse_dict[team] = None
            except:
                arima_rmse_dict[team] = None
        else:
            arima_rmse_dict[team] = None

    return gbr_models, arima_models, team_stats, gbr_rmse_dict, arima_rmse_dict

def get_next_features(team, team_data, is_home_flag=0):
    """
    Generate the next row of features for a future game for the given team:
    - game_index (one more than last game)
    - is_home (passed in)
    - rolling_mean_3, rolling_std_3 (based on the last 3 known scores)
    - rolling_mean_5, rolling_std_5, season_avg, season_std if available
    """
    team_df = team_data[team_data['team'] == team].copy()
    if team_df.empty:
        return None

    last_row = team_df.iloc[-1]
    next_game_index = last_row['game_index'] + 1

    # Grab last 3 & last 5 scores for rolling stats
    last_3_scores = team_df['score'].tail(3)
    next_rolling_mean_3 = last_3_scores.mean()
    next_rolling_std_3 = last_3_scores.std(ddof=1) if len(last_3_scores) > 1 else 0.0

    last_5_scores = team_df['score'].tail(5)
    next_rolling_mean_5 = last_5_scores.mean()
    next_rolling_std_5 = last_5_scores.std(ddof=1) if len(last_5_scores) > 1 else 0.0

    # Season stats if you define a “season” or if the dataset can delineate
    season_avg = team_df['score'].mean()
    season_std = team_df['score'].std()

    # Construct a single feature row, ensuring consistent order with training
    X_next = pd.DataFrame([{
        'game_index': next_game_index,
        'is_home': is_home_flag,
        'rolling_mean_3': next_rolling_mean_3,
        'rolling_std_3': next_rolling_std_3,
        'rolling_mean_5': next_rolling_mean_5,
        'rolling_std_5': next_rolling_std_5,
        'season_avg': season_avg if not np.isnan(season_avg) else 0.0,
        'season_std': season_std if not np.isnan(season_std) else 0.0
    }])

    return X_next

def weighted_ensemble_prediction(gbr_pred, gbr_rmse, arima_pred, arima_rmse):
    """
    Dynamically weight GBR and ARIMA predictions using inverse RMSE.
    If either RMSE is None or 0, fallback to simple averaging for that model.
    """
    if gbr_pred is None and arima_pred is None:
        return None

    if gbr_rmse is None or gbr_rmse == 0:
        w_gbr = 1.0 if gbr_pred is not None else 0
    else:
        w_gbr = 1.0 / gbr_rmse

    if arima_rmse is None or arima_rmse == 0:
        w_arima = 1.0 if arima_pred is not None else 0
    else:
        w_arima = 1.0 / arima_rmse

    if gbr_pred is None:
        return arima_pred
    elif arima_pred is None:
        return gbr_pred

    total_w = w_gbr + w_arima
    return (gbr_pred * w_gbr + arima_pred * w_arima) / total_w

def predict_team_score(team, gbr_models, arima_models, team_stats, team_data, 
                       gbr_rmse_dict, arima_rmse_dict, is_home=0):
    """
    Predict the next game score for the specified team. 
    is_home=1 if the upcoming game is at home, else 0.
    Integrates dynamic weighting if USE_WEIGHTED_ENSEMBLE=True.
    """
    if team not in team_stats:
        return None, (None, None)

    gbr_pred = None
    arima_pred = None

    # Generate features for next game
    feature_df = get_next_features(team, team_data, is_home_flag=is_home)
    if feature_df is None or feature_df.empty:
        return None, (None, None)

    # If we have a trained GBR model for this team, predict
    if team in gbr_models:
        gbr_pred = gbr_models[team].predict(feature_df.values)[0]

    # If we have an ARIMA model for this team, forecast
    if team in arima_models:
        forecast = arima_models[team].predict(n_periods=1)
        arima_pred = forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast.iloc[0]

    # Decide ensemble approach:
    if gbr_pred is not None and arima_pred is not None:
        if USE_WEIGHTED_ENSEMBLE:
            ensemble = weighted_ensemble_prediction(
                gbr_pred, gbr_rmse_dict.get(team), 
                arima_pred, arima_rmse_dict.get(team)
            )
        else:
            # Original simple average
            ensemble = (gbr_pred + arima_pred) / 2
    elif gbr_pred is not None:
        ensemble = gbr_pred
    elif arima_pred is not None:
        ensemble = arima_pred
    else:
        ensemble = None

    # Confidence interval placeholders using mean ± 1.96 * std
    mu = team_stats[team]['mean']
    sigma = team_stats[team]['std']

    # Optional refinement: if you have 10-game rolling std, you could do that:
    # recent_scores = team_data[team_data['team'] == team]['score'].tail(10)
    # sigma_10 = recent_scores.std()
    # if not np.isnan(sigma_10): sigma = sigma_10

    conf_low = round_half(mu - 1.96 * sigma)
    conf_high = round_half(mu + 1.96 * sigma)

    if ensemble is None:
        return None, (conf_low, conf_high)

    return round_half(ensemble), (conf_low, conf_high)

def evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats):
    """
    Existing matchup logic; calculates difference, total points, confidence, etc.
    Kept fully intact with minor expansions possible.
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
    winner = home_team if diff > 0 else away_team

    # For NCAAB, let's pick 145 as a typical threshold
    ou_threshold = 145
    spread_text = f"Lean {winner} by {round_half(diff):.1f}"
    ou_text = f"Take the {'Over' if total_points > ou_threshold else 'Under'} {round_half(total_points):.1f}"

    return {
        'predicted_winner': winner,
        'diff': round_half(diff),
        'total_points': round_half(total_points),
        'confidence': confidence,
        'spread_suggestion': spread_text,
        'ou_suggestion': ou_text
    }

def find_top_bets(matchups, threshold=70.0):
    df = pd.DataFrame(matchups)
    df_top = df[df['confidence'] >= threshold].copy()
    df_top.sort_values('confidence', ascending=False, inplace=True)
    return df_top

########################################
# NFL LOGIC
########################################
@st.cache_data(ttl=3600)
def load_nfl_schedule():
    current_year = datetime.now().year
    years = [current_year - 2, current_year - 1, current_year]
    schedule = nfl.import_schedules(years)
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(schedule['gameday']):
        schedule['gameday'] = schedule['gameday'].dt.tz_convert(None)
    return schedule

def preprocess_nfl_data(schedule):
    """
    Create a single DataFrame with columns:
      ['gameday', 'team', 'score', 'is_home', 'game_index', 
       'rolling_mean_3', 'rolling_std_3', 'rolling_mean_5', 'rolling_std_5',
       'season_avg', 'season_std'].

    Added additional rolling windows (5-game) and placeholders for season stats.
    """
    home_df = schedule[['gameday', 'home_team', 'home_score']].rename(
        columns={'home_team': 'team', 'home_score': 'score'}
    )
    home_df['is_home'] = 1

    away_df = schedule[['gameday', 'away_team', 'away_score']].rename(
        columns={'away_team': 'team', 'away_score': 'score'}
    )
    away_df['is_home'] = 0

    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=['score'], inplace=True)
    data.sort_values('gameday', inplace=True)

    # Sort by team + gameday for rolling features
    data.sort_values(['team', 'gameday'], inplace=True)
    # Create a 'game_index' that increments per team
    data['game_index'] = data.groupby('team').cumcount()

    # Rolling means & std dev (3 games)
    data['rolling_mean_3'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    data['rolling_std_3'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(3, min_periods=1).std(ddof=1)
    )

    # Rolling means & std dev (5 games)
    data['rolling_mean_5'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    data['rolling_std_5'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(5, min_periods=1).std(ddof=1)
    )

    # Season avg/std (placeholder: entire dataset)
    # If you want a single "season" column, you'd group by [team, season].
    data['season_avg'] = data.groupby('team')['score'].transform('mean')
    data['season_std'] = data.groupby('team')['score'].transform('std')

    # Fallbacks for the first few games
    data['rolling_mean_3'].fillna(data['score'], inplace=True)
    data['rolling_std_3'].fillna(0, inplace=True)
    data['rolling_mean_5'].fillna(data['score'], inplace=True)
    data['rolling_std_5'].fillna(0, inplace=True)
    data['season_std'].fillna(0, inplace=True)

    # (Optional) Insert placeholders for weather data if available

    return data

def enhance_nfl_data(schedule_df):
    """
    Enhances NFL data with key betting-specific features based on research.
    """
    df = schedule_df.copy()
    
    # Calculate key betting statistics
    df['margin'] = df['home_score'] - df['away_score']
    df['total_points'] = df['home_score'] + df['away_score']
    
    # Add margin buckets for key numbers analysis
    df['margin_bucket'] = df['margin'].apply(lambda x: round(x) if pd.notnull(x) else None)
    
    # Calculate historical probability distributions
    margin_dist = df['margin_bucket'].value_counts(normalize=True).to_dict()
    total_points_dist = df['total_points'].apply(lambda x: round(x/0.5)*0.5).value_counts(normalize=True).to_dict()
    
    # Add rolling team-specific metrics
    for team_type in ['home', 'away']:
        team_col = f'{team_type}_team'
        score_col = f'{team_type}_score'
        
        # Calculate team-specific rolling averages
        df[f'{team_type}_rolling_avg_5'] = df.groupby(team_col)[score_col].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df[f'{team_type}_rolling_std_5'] = df.groupby(team_col)[score_col].transform(
            lambda x: x.rolling(5, min_periods=1).std()
        )
        
    return df, margin_dist, total_points_dist

def calculate_key_number_edge(pred_margin, margin_dist):
    """
    Calculates betting edge based on key numbers in NFL.
    """
    key_numbers = {
        3: 0.1869,  # 18.69% of games
        7: 0.1147,  # 11.47% of games
        10: 0.0765, # 7.65% of games
        6: 0.0687,  # 6.87% of games
        14: 0.0597  # 5.97% of games
    }
    
    distances = {k: abs(pred_margin - k) for k in key_numbers.keys()}
    nearest_key = min(distances.items(), key=lambda x: x[1])
    
    if nearest_key[1] <= 0.5:
        return 1 + (key_numbers[nearest_key[0]] * 0.5)
    return 1.0

def enhance_nfl_prediction(gbr_pred, arima_pred, team_data, home_team, away_team, margin_dist):
    if gbr_pred is not None and arima_pred is not None:
        base_pred = (gbr_pred + arima_pred) / 2
    elif gbr_pred is not None:
        base_pred = gbr_pred
    elif arima_pred is not None:
        base_pred = arima_pred
    else:
        return None, 0
    
    margin = base_pred
    key_number_factor = calculate_key_number_edge(margin, margin_dist)
    
    base_confidence = 50 + (abs(margin) * 5)
    enhanced_confidence = base_confidence * key_number_factor
    enhanced_confidence = min(99, enhanced_confidence)
    
    return base_pred, enhanced_confidence

def get_nfl_betting_suggestion(margin, total, confidence):
    suggestions = []
    
    if confidence >= 70:
        if margin > 0:
            suggestions.append(f"Strong play on favorite -{abs(margin)}")
        else:
            suggestions.append(f"Strong play on underdog +{abs(margin)}")
    elif confidence >= 60:
        if margin > 0:
            suggestions.append(f"Lean on favorite -{abs(margin)}")
        else:
            suggestions.append(f"Lean on underdog +{abs(margin)}")
    
    if total:
        avg_nfl_total = 44.5
        if abs(total - avg_nfl_total) > 7:
            if total > avg_nfl_total:
                suggestions.append(f"Consider Under {total}")
            else:
                suggestions.append(f"Consider Over {total}")
    
    return suggestions

def evaluate_nfl_matchup(home_team, away_team, home_pred, away_pred, team_stats, margin_dist):
    if home_pred is None or away_pred is None:
        return None
        
    margin = home_pred - away_pred
    total = home_pred + away_pred
    
    _, confidence = enhance_nfl_prediction(
        home_pred, away_pred, team_stats, home_team, away_team, margin_dist
    )
    
    suggestions = get_nfl_betting_suggestion(margin, total, confidence)
    winner = home_team if margin > 0 else away_team
    
    return {
        'predicted_winner': winner,
        'diff': round_half(margin),
        'total_points': round_half(total),
        'confidence': confidence,
        'spread_suggestion': suggestions[0] if suggestions else "No strong lean",
        'ou_suggestion': suggestions[1] if len(suggestions) > 1 else "No strong lean",
        'key_number_analysis': calculate_key_number_edge(margin, margin_dist)
    }

def fetch_upcoming_nfl_games(schedule, days_ahead=7):
    upcoming = schedule[
        schedule['home_score'].isna() & schedule['away_score'].isna()
    ].copy()
    now = datetime.now()
    filter_date = now + timedelta(days=days_ahead)
    upcoming = upcoming[upcoming['gameday'] <= filter_date].copy()
    upcoming.sort_values('gameday', inplace=True)
    return upcoming[['gameday', 'home_team', 'away_team']]


##################################
# NBA-SPECIFIC LOGIC
##################################
@st.cache_data(ttl=3600)
def load_nba_data():
    """
    Loads multiple seasons from the NBA API with additional rolling windows
    for 3 & 5 games, plus placeholders for season stats.
    """
    seasons = ['2022-23', '2023-24', '2024-25']
    all_data = []

    for season in seasons:
        gamelog = LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season',
            player_or_team_abbreviation='T'
        )
        df = gamelog.get_data_frames()[0]
        if df.empty:
            continue

        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        new_df = df[['GAME_DATE', 'TEAM_ABBREVIATION', 'PTS']].copy()
        new_df.rename(columns={
            'GAME_DATE': 'gameday',
            'TEAM_ABBREVIATION': 'team',
            'PTS': 'score'
        }, inplace=True)
        # No direct home/away indicator from this dataset
        new_df['is_home'] = 0

        all_data.append(new_df)

    if not all_data:
        return pd.DataFrame()

    data = pd.concat(all_data, ignore_index=True)
    data.dropna(subset=['score'], inplace=True)

    # Sort by team + gameday
    data.sort_values(['team', 'gameday'], inplace=True)
    data['game_index'] = data.groupby('team').cumcount()

    # Rolling means & std dev (3 games)
    data['rolling_mean_3'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    data['rolling_std_3'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(3, min_periods=1).std(ddof=1)
    )

    # Rolling means & std dev (5 games)
    data['rolling_mean_5'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    data['rolling_std_5'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(5, min_periods=1).std(ddof=1)
    )

    # Season avg/std (placeholder: entire dataset)
    data['season_avg'] = data.groupby('team')['score'].transform('mean')
    data['season_std'] = data.groupby('team')['score'].transform('std')

    data['rolling_mean_3'].fillna(data['score'], inplace=True)
    data['rolling_std_3'].fillna(0, inplace=True)
    data['rolling_mean_5'].fillna(data['score'], inplace=True)
    data['rolling_std_5'].fillna(0, inplace=True)
    data['season_std'].fillna(0, inplace=True)

    # If you track pace-of-play or injuries, add them here as extra columns

    return data

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
@st.cache_data(ttl=3600)
def load_ncaab_data_current_season(season=2025):
    """
    Loads finished or in-progress NCAA MBB games for the given season
    using cbbpy. Adds rolling windows (3 & 5) plus placeholders for season stats.
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

    # Rolling features
    data.sort_values(['team', 'gameday'], inplace=True)
    data['game_index'] = data.groupby('team').cumcount()

    data['rolling_mean_3'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    data['rolling_std_3'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(3, min_periods=1).std(ddof=1)
    )

    data['rolling_mean_5'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    data['rolling_std_5'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(5, min_periods=1).std(ddof=1)
    )

    data['season_avg'] = data.groupby('team')['score'].transform('mean')
    data['season_std'] = data.groupby('team')['score'].transform('std')

    data['rolling_mean_3'].fillna(data['score'], inplace=True)
    data['rolling_std_3'].fillna(0, inplace=True)
    data['rolling_mean_5'].fillna(data['score'], inplace=True)
    data['rolling_std_5'].fillna(0, inplace=True)
    data['season_std'].fillna(0, inplace=True)

    # Optionally incorporate rankings or strength-of-schedule if available

    return data

########################################
# NCAAB UPCOMING: ESPN method (NEW)
########################################
def fetch_upcoming_ncaab_games() -> pd.DataFrame:
    """
    Fetches upcoming NCAAB games for 'today' using ESPN's scoreboard API.
    """
    timezone = pytz.timezone('America/Los_Angeles')
    current_time = datetime.now(timezone)

    date_str = current_time.strftime('%Y%m%d')  # e.g. 20231205
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    params = {
        'dates': date_str,
        'groups': '50',   # D1 men's
        'limit': '357'
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.warning(f"ESPN API request failed with status code {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    games = data.get('events', [])
    if not games:
        st.info(f"No upcoming NCAAB games for {current_time.strftime('%Y-%m-%d')}.")
        return pd.DataFrame()

    rows = []
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

########################################
# UI COMPONENTS (unchanged)
########################################
def generate_writeup(bet):
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

def display_bet_card(bet):
    with st.container():
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 1])

        # Game Info
        with col1:
            st.markdown(f"### **{bet['away_team']} @ {bet['home_team']}**")
            date_obj = bet['date']
            if isinstance(date_obj, datetime):
                st.caption(date_obj.strftime("%A, %B %d - %I:%M %p"))

        # Predictions
        with col2:
            if bet['confidence'] >= 80:
                st.markdown("🔥 **High-Confidence Bet** 🔥")
            st.markdown(f"**Spread Suggestion:** {bet['spread_suggestion']}")
            st.markdown(f"**Total Suggestion:** {bet['ou_suggestion']}")

        # Confidence Metric
        with col3:
            st.metric(label="Confidence", value=f"{bet['confidence']:.1f}%")

    # Optional Detailed Insights
    with st.expander("Detailed Insights", expanded=False):
        st.markdown(f"**Predicted Winner:** {bet['predicted_winner']}")
        st.markdown(f"**Predicted Total Points:** {bet['predicted_total']}")
        st.markdown(f"**Prediction Margin (Diff):** {bet['predicted_diff']}")

    # Detailed Writeup
    with st.expander("Game Analysis", expanded=False):
        writeup = generate_writeup(bet)
        st.markdown(writeup)

########################################
# GLOBALS
########################################
results = []
team_stats_global = {}

########################################
# MAIN PIPELINE
########################################
def run_league_pipeline(league_choice):
    global results
    global team_stats_global

    st.header(f"Today's {league_choice} Best Bets 🎯")
    
    # Initialize models
    nba_model = None
    betting_analyzer = BettingAnalyzer()
    
    if league_choice == "NFL":
        schedule = load_nfl_schedule()
        if schedule.empty:
            st.error("Unable to load NFL schedule. Please try again later.")
            return
        team_data = preprocess_nfl_data(schedule)
        upcoming = fetch_upcoming_nfl_games(schedule, days_ahead=7)

    
    elif league_choice == "NBA":
        # Load and enhance data
        team_data = load_nba_data()
        if team_data.empty:
            st.error("Unable to load NBA data. Please try again later.")
            return
        
        # Enhance data with additional features
        enhanced_data = enhance_team_data(team_data)
        
        # Initialize and train NBA model
        nba_model = EnhancedNBAPredictionModel()
        with st.spinner("Training advanced NBA prediction models..."):
            nba_model.train(enhanced_data)
        
        # Fetch upcoming games
        upcoming = fetch_upcoming_nba_games(days_ahead=3)
        
        # Generate predictions and betting insights
        results.clear()
        
        for _, row in upcoming.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            # Get team-specific enhanced data
            home_data = enhanced_data[enhanced_data['team'] == home].copy()
            away_data = enhanced_data[enhanced_data['team'] == away].copy()
            
            # Generate predictions
            home_pred = nba_model.predict(home_data.iloc[-1:])
            away_pred = nba_model.predict(away_data.iloc[-1:])
            
            # Traditional matchup evaluation
            outcome = evaluate_matchup(home, away, home_pred[0], away_pred[0], team_stats_global)
            
            if outcome:
                result = {
                    'date': row['gameday'],
                    'home_team': home,
                    'away_team': away,
                    'home_pred': home_pred[0],
                    'away_pred': away_pred[0],
                    'predicted_winner': outcome['predicted_winner'],
                    'predicted_diff': outcome['diff'],
                    'predicted_total': outcome['total_points'],
                    'confidence': outcome['confidence'],
                    'spread_suggestion': outcome['spread_suggestion'],
                    'ou_suggestion': outcome['ou_suggestion'],
                    'home_variance': home_data['score'].std(),
                    'away_variance': away_data['score'].std()
                }
                
                results.append(result)
        
        # Generate betting insights
        if results:
            betting_insights = betting_analyzer.generate_betting_insight(enhanced_data, results)
            
            # Display insights in sidebar
            st.sidebar.markdown("### 🎯 Advanced Betting Insights")
            for insight in betting_insights:
                if any(bet['edge'] > 5 for bet in insight['recommended_bets']):
                    st.sidebar.markdown(f"""
                    **{insight['game_id']}**
                    """)
                    
                    for bet in insight['recommended_bets']:
                        st.sidebar.markdown(f"""
                        - Type: {bet['type'].title()}
                        - Edge: {bet['edge']:.1f}%
                        - Kelly: {bet['kelly']:.1f}%
                        - Win Prob: {bet['win_prob']:.1f}%
                        """)
    else:  # NCAAB
        # 1) Load historical data via cbbpy
        team_data = load_ncaab_data_current_season(season=2025)
        if team_data.empty:
            st.error("Unable to load NCAAB data. Please try again later.")
            return

        # 2) Fetch upcoming games from ESPN scoreboard
        with st.spinner("Fetching upcoming NCAAB games..."):
            upcoming = fetch_upcoming_ncaab_games()

    if team_data.empty:
        st.warning(f"No {league_choice} data available for analysis.")
        return

    # Train models (GBR + ARIMA) with enhancements
    with st.spinner("Analyzing recent performance data..."):
        gbr_models, arima_models, team_stats, gbr_rmse_dict, arima_rmse_dict = train_team_models(team_data)
        team_stats_global = team_stats
        results.clear()

        for _, row in upcoming.iterrows():
            home = row['home_team']
            away = row['away_team']

            # Pass is_home=1 for home, 0 for away
            home_pred, _ = predict_team_score(home, gbr_models, arima_models, team_stats, team_data,
                                              gbr_rmse_dict, arima_rmse_dict, is_home=1)
            away_pred, _ = predict_team_score(away, gbr_models, arima_models, team_stats, team_data,
                                              gbr_rmse_dict, arima_rmse_dict, is_home=0)

            outcome = evaluate_matchup(home, away, home_pred, away_pred, team_stats)
            if outcome:
                results.append({
                    'date': row['gameday'],
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

    # UI: choose top bets or all
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
                display_bet_card(bet)
        else:
            st.info("No high-confidence bets found for today. Try lowering the confidence threshold.")
    else:
        if results:
            st.markdown("### 📊 All Games Analysis")
            for bet in results:
                display_bet_card(bet)
        else:
            st.info(f"No upcoming {league_choice} games found for analysis.")

def main():
    st.set_page_config(
        page_title="FoxEdge Sports Betting Edge",
        page_icon="🦊",
        layout="centered"
    )
    initialize_csv()

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # Simple authentication logic
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

    # Main UI
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
        "FoxEdge provides advanced data-driven insights for NFL, NBA, and NCAAB games, helping bettors make informed decisions with high confidence."
    )
    st.sidebar.markdown("#### Powered by 🧠 AI and 🔍 Statistical Analysis")
    st.sidebar.markdown("Feel free to reach out for feedback or support!")

    if st.button("Save Predictions to CSV"):
        save_predictions_to_csv(results)

if __name__ == "__main__":
    main()
