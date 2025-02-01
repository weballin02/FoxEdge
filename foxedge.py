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
from pmdarima import auto_arima
from pathlib import Path
import requests
import logging

# Firebase (for authentication)
import firebase_admin
from firebase_admin import credentials, auth

# cbbpy for NCAAB data
import cbbpy.mens_scraper as cbb

# ------------------------------------------------------------
# Configure Logging
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------
# Firebase Authentication Setup
try:
    # Ensure your st.secrets contains your Firebase configuration.
    FIREBASE_API_KEY = st.secrets["general"]["firebaseApiKey"]
    service_account_info = st.secrets["firebase"]
    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
except Exception as e:
    st.warning("Firebase not configured properly. Authentication will be disabled.")
    logging.warning(e)

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
        st.error("Error during login")
        logging.error(e)
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

# ------------------------------------------------------------
# CSV Management
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

# ------------------------------------------------------------
# Utility Function
def round_half(number):
    """Rounds a number to the nearest 0.5."""
    return round(number * 2) / 2

# ------------------------------------------------------------
# Model Training & Prediction Functions
@st.cache_data(ttl=14400)
def train_team_models(team_data: pd.DataFrame):
    """
    For each team in the provided dataset, train a stacking regressor (with a preprocessing pipeline)
    and an Auto-ARIMA model using time-series–aware splitting and cross-validation.
    Also compute basic team statistics.
    Returns:
      - stack_models: dict mapping team -> trained stacking model.
      - arima_models: dict mapping team -> trained ARIMA model.
      - team_stats: dict mapping team -> {'mean': ..., 'std': ..., 'recent_form': ...}.
      - model_errors: dict mapping team -> error metrics for stacking and ARIMA.
    """
    stack_models = {}
    arima_models = {}
    team_stats = {}
    model_errors = {}

    teams = team_data['team'].unique()
    for team in teams:
        df_team = team_data[team_data['team'] == team].copy().sort_values('gameday')
        scores = df_team['score'].reset_index(drop=True)
        if len(scores) < 3:
            continue

        # Feature Engineering: rolling averages, lag feature, etc.
        df_team['rolling_avg'] = df_team['score'].rolling(window=3, min_periods=1).mean()
        df_team['rolling_std'] = df_team['score'].rolling(window=3, min_periods=1).std().fillna(0)
        df_team['season_avg'] = df_team['score'].expanding().mean()
        df_team['weighted_avg'] = (df_team['rolling_avg'] * 0.6) + (df_team['season_avg'] * 0.4)
        df_team['first_half_avg'] = df_team['rolling_avg'] * 0.6
        df_team['second_half_avg'] = df_team['rolling_avg'] * 0.4
        df_team['late_game_impact'] = df_team['score'] * 0.3 + df_team['season_avg'] * 0.7
        df_team['early_vs_late'] = df_team['first_half_avg'] - df_team['second_half_avg']
        df_team['lag_score'] = df_team['score'].shift(1).fillna(df_team['season_avg'])

        # Determine features: include extra advanced stats if available (NBA data)
        base_features = ['rolling_avg', 'rolling_std', 'weighted_avg', 'early_vs_late', 'late_game_impact', 'lag_score']
        if 'off_rating' in df_team.columns:
            extra_features = ['off_rating', 'def_rating', 'pace']
            features_cols = base_features + extra_features
        else:
            features_cols = base_features

        team_stats[team] = {
            'mean': round_half(scores.mean()),
            'std': round_half(scores.std()),
            'recent_form': round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
        }

        features = df_team[features_cols].fillna(0)
        X = features
        y = scores

        split_index = int(len(scores) * 0.8)
        if split_index < 2:
            continue
        X_train = X.iloc[:split_index].values
        X_test = X.iloc[split_index:].values
        y_train = y.iloc[:split_index].values
        y_test = y.iloc[split_index:].values

        n_splits = 3 if len(y_train) >= 3 else 2
        tscv = TimeSeriesSplit(n_splits=n_splits)

        estimators = [
            ('xgb', XGBRegressor(n_estimators=200, random_state=42)),
            ('lgbm', LGBMRegressor(n_estimators=200, random_state=42)),
            ('cat', CatBoostRegressor(n_estimators=200, verbose=0, random_state=42))
        ]
        final_estimator = make_pipeline(StandardScaler(), LGBMRegressor(n_estimators=200, random_state=42))
        stack = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=tscv)

        try:
            stack.fit(X_train, y_train)
            preds = stack.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            stack_models[team] = stack
            model_errors.setdefault(team, {})['stacking'] = mse
        except Exception as e:
            logging.error(f"Error training stacking for {team}: {e}")
            continue

        if len(scores) >= 7:
            try:
                arima = auto_arima(scores, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True, max_p=3, max_q=3)
                arima_models[team] = arima
                if len(y_test) > 0:
                    arima_preds = arima.predict(n_periods=len(y_test))
                    mse_arima = mean_squared_error(y_test, arima_preds)
                    model_errors[team]['arima'] = mse_arima
            except Exception as e:
                logging.error(f"Error training ARIMA for {team}: {e}")
                continue

    return stack_models, arima_models, team_stats, model_errors

def predict_team_score(team, stack_models, arima_models, team_stats, team_data, model_errors):
    """
    For the given team, predict the next-game score by blending the stacking model
    and ARIMA prediction using inverse error weighting. Also returns a confidence interval.
    """
    if team not in team_stats:
        return None, (None, None)
    df_team = team_data[team_data['team'] == team].sort_values('gameday')
    if len(df_team) < 3:
        return None, (None, None)

    base_features = ['rolling_avg', 'rolling_std', 'weighted_avg', 'early_vs_late', 'late_game_impact', 'lag_score']
    if 'off_rating' in df_team.columns:
        features_cols = base_features + ['off_rating', 'def_rating', 'pace']
    else:
        features_cols = base_features

    X_next = df_team[features_cols].tail(1).values

    stack_pred = None
    if team in stack_models:
        try:
            stack_pred = float(stack_models[team].predict(X_next)[0])
        except Exception as e:
            logging.error(f"Stacking prediction error for {team}: {e}")
    arima_pred = None
    if team in arima_models:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast)
        except Exception as e:
            logging.error(f"ARIMA prediction error for {team}: {e}")

    if stack_pred is not None and arima_pred is not None:
        mse_stack = model_errors.get(team, {}).get('stacking', 1e-6)
        mse_arima = model_errors.get(team, {}).get('arima', 1e-6)
        if mse_stack <= 0: mse_stack = 1e-6
        if mse_arima <= 0: mse_arima = 1e-6
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
    """
    Computes the predicted margin (spread), total points, and a confidence metric for the matchup.
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

# ------------------------------------------------------------
# Data Loading Functions for Each League
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
    data['late_game_impact'] = data['score'] * 0.3 + data['season_avg'] * 0.7
    data['early_vs_late'] = data['first_half_avg'] - data['second_half_avg']
    data['lag_score'] = data.groupby('team')['score'].shift(1).fillna(data['season_avg'])
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
                gl['OFF_RATING'] = np.where(gl['TEAM_POSSESSIONS'] > 0, (gl['PTS'] / gl['TEAM_POSSESSIONS'])*100, np.nan)
                gl['DEF_RATING'] = np.where(gl['TEAM_POSSESSIONS'] > 0, (gl['PTS_OPP'] / gl['TEAM_POSSESSIONS'])*100, np.nan)
                gl['PACE'] = gl['TEAM_POSSESSIONS']
                gl['rolling_avg'] = gl['PTS'].rolling(window=3, min_periods=1).mean()
                gl['rolling_std'] = gl['PTS'].rolling(window=3, min_periods=1).std().fillna(0)
                gl['season_avg'] = gl['PTS'].expanding().mean()
                gl['weighted_avg'] = (gl['rolling_avg'] * 0.6) + (gl['season_avg'] * 0.4)
                gl['first_half_avg'] = gl['rolling_avg'] * 0.6
                gl['second_half_avg'] = gl['rolling_avg'] * 0.4
                gl['late_game_impact'] = gl['PTS'] * 0.3 + gl['season_avg'] * 0.7
                gl['early_vs_late'] = gl['first_half_avg'] - gl['second_half_avg']
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
                            'first_half_avg': row_['first_half_avg'],
                            'second_half_avg': row_['second_half_avg'],
                            'late_game_impact': row_['late_game_impact'],
                            'early_vs_late': row_['early_vs_late']
                        })
                    except Exception as e:
                        logging.error(f"Row processing error for team {team_abbrev}: {e}")
                        continue
            except Exception as e:
                logging.error(f"Team processing error for {team_abbrev} in season {season}: {e}")
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
    df = pd.DataFrame(upcoming_rows)
    df.sort_values('gameday', inplace=True)
    return df

@st.cache_data(ttl=14400)
def load_ncaab_data_current_season(season=2025):
    info_df, _, _ = cbb.get_games_season(season=season, info=True, box=False, pbp=False)
    if info_df.empty:
        return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(info_df["game_day"]):
        info_df["game_day"] = pd.to_datetime(info_df["game_day"], errors="coerce")
    home_df = info_df.rename(columns={"home_team": "team", "home_score": "score", "game_day": "gameday"})[["gameday", "team", "score"]]
    home_df['is_home'] = 1
    away_df = info_df.rename(columns={"away_team": "team", "away_score": "score", "game_day": "gameday"})[["gameday", "team", "score"]]
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
    data['late_game_impact'] = data['score'] * 0.3 + data['season_avg'] * 0.7
    data['early_vs_late'] = data['first_half_avg'] - data['second_half_avg']
    data.sort_values(['team', 'gameday'], inplace=True)
    data['game_index'] = data.groupby('team').cumcount()
    data['lag_score'] = data.groupby('team')['score'].shift(1).fillna(data['season_avg'])
    return data

def fetch_upcoming_ncaab_games():
    timezone = pytz.timezone('America/Los_Angeles')
    current_time = datetime.now(timezone)
    dates = [current_time.strftime('%Y%m%d'), (current_time + timedelta(days=1)).strftime('%Y%m%d')]
    rows = []
    for date_str in dates:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {'dates': date_str, 'groups': '50', 'limit': '357'}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            st.warning(f"ESPN API request failed for {date_str} with status {response.status_code}")
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
            rows.append({'gameday': game_time, 'home_team': home_team, 'away_team': away_team})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.sort_values('gameday', inplace=True)
    return df

# ------------------------------------------------------------
# UI Components
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
  - Average Score: {home_mean}
  - Score Std Dev: {home_std}
  - Recent Form (Last 5 Games): {home_recent}

- **{away_team} Performance:**
  - Average Score: {away_mean}
  - Score Std Dev: {away_std}
  - Recent Form (Last 5 Games): {away_recent}

- **Prediction Insight:**
  {predicted_winner} is predicted to win with {confidence}% confidence.
  Projected margin: {bet['predicted_diff']} points; Total points: {bet['predicted_total']}.
  Spread: {bet['spread_suggestion']}; O/U: {bet['ou_suggestion']}.
"""
    return writeup

def display_bet_card(bet, team_stats_global):
    with st.container():
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"### **{bet['away_team']} @ {bet['home_team']}**")
            if isinstance(bet['date'], datetime):
                st.caption(bet['date'].strftime("%A, %B %d - %I:%M %p"))
        with col2:
            if bet['confidence'] >= 80:
                st.markdown("🔥 **High-Confidence Bet** 🔥")
            st.markdown(f"**Spread:** {bet['spread_suggestion']}")
            st.markdown(f"**Total:** {bet['ou_suggestion']}")
        with col3:
            st.metric(label="Confidence", value=f"{bet['confidence']:.1f}%")
    with st.expander("Detailed Insights", expanded=False):
        st.markdown(f"**Predicted Winner:** {bet['predicted_winner']}")
        st.markdown(f"**Predicted Total Points:** {bet['predicted_total']}")
        st.markdown(f"**Margin:** {bet['predicted_diff']} points")
    with st.expander("Game Analysis", expanded=False):
        writeup = generate_writeup(bet, team_stats_global)
        st.markdown(writeup)

# ------------------------------------------------------------
# Global Variables
results = []
team_stats_global = {}

def run_league_pipeline(league_choice):
    global results, team_stats_global
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
            if home_pred is None or away_pred is None:
                continue
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
                                    help="Show only bets with confidence above this threshold")
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

# ------------------------------------------------------------
# Main App Function
def main():
    st.set_page_config(page_title="FoxEdge Sports Betting Edge", page_icon="🦊", layout="centered")
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
        st.sidebar.write(f"Logged in as: {st.session_state.get('email', 'Unknown')}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.experimental_rerun()
    st.title("🦊 FoxEdge Sports Betting Insights")
    st.sidebar.header("Navigation")
    league_choice = st.sidebar.radio("Select League", ["NFL", "NBA", "NCAAB"],
                                     help="Choose which league's games to analyze")
    run_league_pipeline(league_choice)
    st.sidebar.markdown("### About FoxEdge\nFoxEdge provides data-driven insights for NFL, NBA, and NCAAB games, helping bettors make informed decisions.")
    st.sidebar.markdown("#### Powered by AI & Statistical Analysis")
    if st.button("Save Predictions to CSV"):
        if results:
            save_predictions_to_csv(results)
            csv = pd.DataFrame(results).to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
        else:
            st.warning("No predictions to save.")

if __name__ == "__main__":
    main()
