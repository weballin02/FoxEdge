import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import nfl_data_py as nfl
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import GradientBoostingRegressor
from pmdarima import auto_arima
from pathlib import Path

import json
import firebase_admin
from firebase_admin import credentials, auth
import requests
import cbbpy.mens_scraper as cbb  # For historical NCAAB data

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
# TRAINING & PREDICTIONS (unchanged)
########################################
@st.cache_data(ttl=3600)
def train_team_models(team_data: pd.DataFrame):
    """
    Trains GradientBoostingRegressor and ARIMA for each team.
    Returns: gbr_models, arima_models, team_stats
    """
    gbr_models = {}
    arima_models = {}
    team_stats = {}

    all_teams = team_data['team'].unique()
    for team in all_teams:
        scores = team_data[team_data['team'] == team]['score'].reset_index(drop=True)
        if len(scores) < 3:
            continue

        team_stats[team] = {
            'mean': round_half(scores.mean()),
            'std': round_half(scores.std()),
            'max': round_half(scores.max()),
            'recent_form': round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
        }

        if len(scores) >= 10:
            X = np.arange(len(scores)).reshape(-1, 1)
            y = scores.values
            gbr = GradientBoostingRegressor()
            gbr.fit(X, y)
            gbr_models[team] = gbr

        if len(scores) >= 7:
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

    return gbr_models, arima_models, team_stats

def predict_team_score(team, gbr_models, arima_models, team_stats, team_data):
    if team not in team_stats:
        return None, (None, None)

    gbr_pred = None
    arima_pred = None

    if team in gbr_models:
        data_len = len(team_data[team_data['team'] == team])
        X_next = np.array([[data_len]])
        gbr_pred = gbr_models[team].predict(X_next)[0]

    if team in arima_models:
        forecast = arima_models[team].predict(n_periods=1)
        arima_pred = forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast.iloc[0]

    if gbr_pred is not None and arima_pred is not None:
        ensemble = (gbr_pred + arima_pred) / 2
    elif gbr_pred is not None:
        ensemble = gbr_pred
    elif arima_pred is not None:
        ensemble = arima_pred
    else:
        ensemble = None

    mu = team_stats[team]['mean']
    sigma = team_stats[team]['std']
    conf_low = round_half(mu - 1.96 * sigma)
    conf_high = round_half(mu + 1.96 * sigma)

    return round_half(ensemble), (conf_low, conf_high)

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
# NFL LOGIC (unchanged)
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

##################################
# NBA-SPECIFIC LOGIC
##################################

@st.cache_data(ttl=3600)
def load_nba_data():
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
        new_df.sort_values('gameday', inplace=True)
        all_data.append(new_df)

    if not all_data:
        return pd.DataFrame()

    data = pd.concat(all_data, ignore_index=True)
    data.dropna(subset=['score'], inplace=True)
    data.sort_values('gameday', inplace=True)
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
# NCAAB HISTORICAL LOADER (unchanged)
########################################
@st.cache_data(ttl=3600)
def load_ncaab_data_current_season(season=2025):
    """
    Loads finished or in-progress NCAA MBB games for the given season
    using cbbpy. 
    """
    # This call requires a mandatory 'season' param
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

    away_df = info_df.rename(columns={
        "away_team": "team",
        "away_score": "score",
        "game_day": "gameday"
    })[["gameday", "team", "score"]]

    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=["score"], inplace=True)
    data.sort_values("gameday", inplace=True)
    return data

########################################
# NCAAB UPCOMING: ESPN method (NEW)
########################################
def fetch_upcoming_ncaab_games() -> pd.DataFrame:
    """
    Fetches upcoming NCAAB games for 'today' using ESPN's scoreboard API
    (similar to the example provided).
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
        # Parse date/time
        game_time_str = game['date']          # ISO8601
        game_time = datetime.fromisoformat(game_time_str[:-1]).astimezone(timezone)

        # Parse teams
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
# MAIN PIPELINE (unchanged except NCAAB)
########################################
def run_league_pipeline(league_choice):
    global results
    global team_stats_global

    st.header(f"Today's {league_choice} Best Bets 🎯")

    if league_choice == "NFL":
        schedule = load_nfl_schedule()
        if schedule.empty:
            st.error("Unable to load NFL schedule. Please try again later.")
            return
        team_data = preprocess_nfl_data(schedule)
        upcoming = fetch_upcoming_nfl_games(schedule, days_ahead=7)

    elif league_choice == "NBA":
        team_data = load_nba_data()
        if team_data.empty:
            st.error("Unable to load NBA data. Please try again later.")
            return
        upcoming = fetch_upcoming_nba_games(days_ahead=3)

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

    # Train models (unchanged)
    with st.spinner("Analyzing recent performance data..."):
        gbr_models, arima_models, team_stats = train_team_models(team_data)
        team_stats_global = team_stats
        results.clear()

        for _, row in upcoming.iterrows():
            home, away = row['home_team'], row['away_team']
            home_pred, _ = predict_team_score(home, gbr_models, arima_models, team_stats, team_data)
            away_pred, _ = predict_team_score(away, gbr_models, arima_models, team_stats, team_data)

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
