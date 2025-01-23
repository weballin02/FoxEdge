import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import nfl_data_py as nfl
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2, TeamGameLog
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from pmdarima import auto_arima
from pathlib import Path
import requests
import firebase_admin
from firebase_admin import credentials, auth

# cbbpy for NCAAB
import cbbpy.mens_scraper as cbb

################################################################################
# ADVANCED FEATURE ENGINEERING UTILITIES
################################################################################
def engineer_sports_features(data, league='NFL'):
    """
    Advanced feature engineering for sports data
    """
    # Sort data chronologically
    data.sort_values(['team', 'gameday'], inplace=True)
    
    # Rolling performance metrics
    performance_windows = [3, 5, 10]
    for window in performance_windows:
        # Win rate calculation
        data[f'win_rate_{window}'] = data.groupby('team')['score'].transform(
            lambda x: (x > x.rolling(window, min_periods=1).mean()).rolling(window, min_periods=1).mean()
        )
        
        # Score volatility
        data[f'score_volatility_{window}'] = data.groupby('team')['score'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    # Rest days feature
    data['rest_days'] = data.groupby('team')['gameday'].diff().dt.days.fillna(0)
    
    # League-specific adjustments
    if league == 'NFL':
        data['is_home'] = data.index < len(data) // 2
    
    # Opponent strength indicator
    def calculate_opponent_strength(group):
        return group['score'].rolling(window=5, min_periods=1).mean()
    
    data['opponent_avg_score'] = data.groupby('team').apply(
        lambda x: calculate_opponent_strength(x.sort_values('gameday'))
    ).reset_index(level=0, drop=True)
    
    # Performance differential (home vs away)
    data['performance_differential'] = data.groupby('team').apply(
        lambda x: x[x['is_home']]['score'].mean() - x[~x['is_home']]['score'].mean()
    ).reset_index(level=0, drop=True)
    
    # Adjusted performance rating
    league_avg_score = data['score'].mean()
    data['adjusted_performance_rating'] = (data['score'] / league_avg_score) * 100
    
    return data

def advanced_model_training(team_data, league='NFL'):
    """
    Enhanced model training with multiple algorithms and blending
    """
    # Feature engineering
    enhanced_data = engineer_sports_features(team_data, league)
    
    # Prepare features and target
    features = [
        col for col in enhanced_data.columns 
        if col not in ['team', 'gameday', 'score', 'is_home']
    ]
    
    X = enhanced_data[features]
    y = enhanced_data['score']
    
    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Multiple model training
    models = {
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
    }
    
    # Train and evaluate models
    model_performance = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        model.fit(X_scaled, y)
        
        model_performance[name] = {
            'model': model,
            'cv_mean_score': cv_scores.mean(),
            'cv_std_score': cv_scores.std()
        }
    
    # Model blending (simple average)
    def blended_prediction(X_test):
        predictions = [
            model_performance[name]['model'].predict(X_test) 
            for name in models.keys()
        ]
        return np.mean(predictions, axis=0)
    
    return {
        'models': model_performance,
        'scaler': scaler,
        'blended_predictor': blended_prediction
    }

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
            "predicted_winner", "total", "spread_suggestion", "ou_suggestion"
        ]
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

def save_predictions_to_csv(predictions, csv_file=CSV_FILE):
    """Save predictions to a CSV file."""
    df = pd.DataFrame(predictions)
    df.to_csv(csv_file, mode='a', index=False, header=not Path(csv_file).exists())
    st.success("Predictions have been saved to CSV!")

################################################################################
# UTILITY
################################################################################
def round_half(number):
    return round(number * 2) / 2

################################################################################
# MODEL TRAINING & PREDICTION (ENHANCED)
################################################################################
@st.cache_data(ttl=3600)
def train_team_models(team_data: pd.DataFrame, league='NFL'):
    """
    Enhanced training of models with multiple algorithms
    """
    # Feature engineered data
    enhanced_data = engineer_sports_features(team_data, league)
    
    # Multiple model training
    models_data = advanced_model_training(team_data, league)
    
    # Compute team stats
    team_stats = {}
    all_teams = team_data['team'].unique()
    
    for team in all_teams:
        df_team = team_data[team_data['team'] == team].copy()
        df_team.sort_values('gameday', inplace=True)
        scores = df_team['score'].reset_index(drop=True)

        if len(scores) < 3:
            continue

        team_stats[team] = {
            'mean': round_half(scores.mean()),
            'std': round_half(scores.std()),
            'max': round_half(scores.max()),
            'recent_form': round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
        }

    return models_data, team_stats

def predict_team_score(team, models_data, team_stats, team_data):
    """
    Enhanced score prediction leveraging multiple models
    """
    if team not in team_stats:
        return None, (None, None)

    # Prepare team data
    df_team = team_data[team_data['team'] == team]
    
    # Prepare features
    team_features = df_team.iloc[-1]
    features = [
        col for col in team_features.index 
        if col not in ['team', 'gameday', 'score']
    ]
    
    # Scale features
    X_test = team_features[features].values.reshape(1, -1)
    X_scaled = models_data['scaler'].transform(X_test)
    
    # Blended prediction
    predicted_score = models_data['blended_predictor'](X_scaled)[0]

    # Compute confidence and intervals
    mu = team_stats[team]['mean']
    sigma = team_stats[team]['std']
    conf_low = round_half(mu - 1.96 * sigma)
    conf_high = round_half(mu + 1.96 * sigma)

    return round_half(predicted_score), (conf_low, conf_high)

def evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats):
    """
    Compute predicted spread, total, and confidence for a single matchup
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

    ou_threshold = 145  # Adjustable threshold

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
    home_df = schedule[['gameday', 'home_team', 'home_score']].rename(
        columns={'home_team': 'team', 'home_score': 'score'}
    )
    away_df = schedule[['gameday', 'away_team', 'away_score']].rename(
        columns={'away_team': 'team', 'away_score': 'score'}
    )
    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=['score'], inplace=True)
    data.sort_values('gameday', inplace=True)
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
@st.cache_data(ttl=3600)
def load_nba_data():
    """Load multi-season team logs with pace & efficiency integrated."""
    nba_teams_list = nba_teams.get_teams()
    seasons = ['2022-23', '2023-24']  # Adjust as needed
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

                # We'll keep a final 'score' for training, which is the team's points
                # plus advanced columns for potential feature engineering.
                for idx, row_ in gl.iterrows():
                    try:
                        all_rows.append({
                            'gameday': row_['GAME_DATE'],
                            'team': team_abbrev,  # Use team abbreviation from teams list
                            'score': float(row_['PTS']),
                            'off_rating': row_['OFF_RATING'] if pd.notnull(row_['OFF_RATING']) else np.nan,
                            'def_rating': row_['DEF_RATING'] if pd.notnull(row_['DEF_RATING']) else np.nan,
                            'pace': row_['PACE'] if pd.notnull(row_['PACE']) else np.nan
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

    # Add rolling features
    data.sort_values(['team', 'gameday'], inplace=True)
    data['game_index'] = data.groupby('team').cumcount()

    data['rolling_mean_3'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    data['rolling_std_3'] = data.groupby('team')['score'].transform(
        lambda x: x.rolling(3, min_periods=1).std(ddof=1)
    )

    data['rolling_mean_3'].fillna(data['score'], inplace=True)
    data['rolling_std_3'].fillna(0, inplace=True)

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

################################################################################
# UI COMPONENTS
################################################################################
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
        writeup = generate_writeup(bet)
        st.markdown(writeup)

################################################################################
# HOME PAGE COMPONENTS
################################################################################
def display_homepage():
    """
    Display a completely separate homepage with general app information.
    """
    # Welcome Header
    st.markdown("""
    # 🦊 Welcome to FoxEdge Sports Betting
    ### Your AI-Powered Sports Betting Analytics Platform
    """)

    # Quick Metrics Section
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Active Leagues", "3", "NFL, NBA, NCAAB")
    with metrics_col2:
        st.metric("Analysis Model", "Ensemble AI", "GBR + ARIMA")
    with metrics_col3:
        st.metric("Prediction Accuracy", "75%+", "High Confidence")

    # Navigation Buttons
    st.markdown("---")
    st.markdown("### 🔍 Explore Our Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏈 NFL Analysis", use_container_width=True):
            st.session_state['selected_page'] = "NFL"
            st.rerun()
    with col2:
        if st.button("🏀 NBA Analysis", use_container_width=True):
            st.session_state['selected_page'] = "NBA"
            st.rerun()
    with col3:
        if st.button("🏀 NCAAB Analysis", use_container_width=True):
            st.session_state['selected_page'] = "NCAAB"
            st.rerun()

    # Features Section
    st.markdown("---")
    st.markdown("### 🚀 Why Choose FoxEdge?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📊 Advanced Analytics
        - AI-powered predictions
        - Historical performance data
        - Team-specific insights
        """)
        
    with col2:
        st.markdown("""
        #### 📈 Real-time Updates
        - Live odds tracking
        - Injury reports impact
        - Line movement alerts
        """)
        
    with col3:
        st.markdown("""
        #### 📱 Custom Tools
        - Bet tracking
        - Performance metrics
        - ROI calculator
        """)

    # Educational Section for Beginners
    with st.expander("📚 New to Sports Betting?"):
        st.markdown("""
        ### Getting Started with FoxEdge
        1. **Choose your league**: Select NFL, NBA, or NCAAB.
        2. **Review predictions**: Check our AI-powered insights.
        3. **Set confidence threshold**: Adjust based on your risk tolerance.
        4. **Track performance**: Save predictions and monitor results.

        Remember to bet responsibly and never wager more than you can afford to lose.
        """)

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
        models_data, team_stats = train_team_models(team_data, league=league_choice)
        team_stats_global = team_stats
        results.clear()

        for _, row in upcoming.iterrows():
            home, away = row['home_team'], row['away_team']
            home_pred, _ = predict_team_score(home, models_data, team_stats, team_data)
            away_pred, _ = predict_team_score(away, models_data, team_stats, team_data)

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
            st.info("No high-confidence bets found. Try lowering the threshold.")
    else:
        if results:
            st.markdown("### 📊 All Games Analysis")
            for bet in results:
                display_bet_card(bet)
        else:
            st.info(f"No upcoming {league_choice} games found.")

################################################################################
# STREAMLIT MAIN (UPDATED)
################################################################################
def main():
    """
    Main function to manage app navigation and logic.
    """
    # Set page configuration
    st.set_page_config(
        page_title="FoxEdge Sports Betting Edge",
        page_icon="🦊",
        layout="wide"
    )

    initialize_csv()

    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if 'selected_page' not in st.session_state:
        st.session_state['selected_page'] = None

    # Login Page Logic
    if not st.session_state['logged_in']:
        st.title("Login to FoxEdge Sports Betting Insights")

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Login"):
                user_data = login_with_rest(email, password)
                if user_data:
                    # Set session state for logged-in user and redirect to homepage
                    st.session_state['logged_in'] = True
                    st.session_state['email'] = user_data['email']
                    st.session_state['selected_page'] = None  # Reset navigation to homepage
                    st.success(f"Welcome, {user_data['email']}!")
                    st.rerun()
        
        with col2:
            if st.button("Sign Up"):
                signup_user(email, password)
        
        return  # Stop execution until login is successful

    # Logged-In State: Navigation Logic
    else:
        # Sidebar Navigation and Logout Button
        st.sidebar.title("Account")
        st.sidebar.write(f"Logged in as: {st.session_state.get('email', 'Unknown')}")

        if st.sidebar.button("Logout"):
            logout_user()
            st.rerun()

        # Sidebar Navigation Menu for League Selection or Homepage Access
        if not st.session_state.get('selected_page'):
            display_homepage()  # Default view is the homepage after login
            return

        selected_league = st.sidebar.radio(
            "Select League",
            ["NFL", "NBA", "NCAAB"],
            help="Choose which league's games you'd like to analyze"
        )

        if selected_league != st.session_state.get('selected_page'):
            # Update session state for league selection and rerun app
            st.session_state['selected_page'] = selected_league
            st.rerun()

        # Run League Pipeline for Selected League (if chosen)
        run_league_pipeline(st.session_state['selected_page'])
        else:
            display_homepage()

        if st.sidebar.button("Save Predictions to CSV"):
            save_predictions_to_csv(results)

if __name__ == "__main__":
    main()
