import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import nfl_data_py as nfl
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2, TeamGameLog
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import GradientBoostingRegressor
from pmdarima import auto_arima
from pathlib import Path
import requests
import firebase_admin
from firebase_admin import credentials, auth

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
# MODEL TRAINING & PREDICTION (ARIMA + GBR ENSEMBLE)
################################################################################
@st.cache_data(ttl=3600)
def train_team_models(team_data: pd.DataFrame):
    """
    Trains GradientBoostingRegressor + ARIMA for each team's 'score'.
    Returns: gbr_models, arima_models, team_stats
    """
    gbr_models = {}
    arima_models = {}
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

        # Train GBR if enough data
        if len(scores) >= 10:
            X = np.arange(len(scores)).reshape(-1, 1)
            y = scores.values
            gbr = GradientBoostingRegressor(random_state=42)
            gbr.fit(X, y)
            gbr_models[team] = gbr

        # Train ARIMA if enough data
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
    """Predict a team's next-game score by blending ARIMA & GBR outputs."""
    if team not in team_stats:
        return None, (None, None)

    df_team = team_data[team_data['team'] == team]
    data_len = len(df_team)
    gbr_pred = None
    arima_pred = None

    # Gradient Boosting
    if team in gbr_models:
        X_next = np.array([[data_len]])
        gbr_pred = gbr_models[team].predict(X_next)[0]

    # ARIMA
    if team in arima_models:
        forecast = arima_models[team].predict(n_periods=1)
        arima_pred = forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast.iloc[0]

    # Blend
    if gbr_pred is not None and arima_pred is not None:
        ensemble = (gbr_pred + arima_pred) / 2
    elif gbr_pred is not None:
        ensemble = gbr_pred
    elif arima_pred is not None:
        ensemble = arima_pred
    else:
        ensemble = None

    if ensemble is None:
        return None, (None, None)

    mu = team_stats[team]['mean']
    sigma = team_stats[team]['std']
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
    confidence = round(min(99, max(1, 50 + raw_conf * 15)), 2)
    winner = home_team if diff > 0 else away_team

    # Example threshold for NCAAB. Adjust if needed for NBA or NFL.
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

########################################
# UI COMPONENTS (unchanged)
########################################
def generate_writeup(bet):
    # Extract data from the bet dictionary
    home_team = bet.get('home_team', 'Unknown')
    away_team = bet.get('away_team', 'Unknown')
    predicted_winner = bet.get('predicted_winner', 'N/A')
    confidence = bet.get('confidence', 0)
    predicted_diff = bet.get('predicted_diff', 'N/A')
    spread_suggestion = bet.get('spread_suggestion', 'N/A')
    predicted_total = bet.get('predicted_total', 'N/A')
    ou_suggestion = bet.get('ou_suggestion', 'N/A')

    # Fetch team stats globally
    home_stats = team_stats_global.get(home_team, {})
    away_stats = team_stats_global.get(away_team, {})

    home_mean = home_stats.get('mean', 'N/A')
    home_std = home_stats.get('std', 'N/A')
    home_recent = home_stats.get('recent_form', 'N/A')
    away_mean = away_stats.get('mean', 'N/A')
    away_std = away_stats.get('std', 'N/A')
    away_recent = away_stats.get('recent_form', 'N/A')

    # Construct the writeup
    writeup = f"""
    ### Detailed Analysis:

    #### {home_team} Performance:
    - **Average Score:** {home_mean}
    - **Score Standard Deviation:** {home_std}
    - **Recent Form (Last 5 Games):** {home_recent}

    #### {away_team} Performance:
    - **Average Score:** {away_mean}
    - **Score Standard Deviation:** {away_std}
    - **Recent Form (Last 5 Games):** {away_recent}

    #### Prediction Insight:
    - Predicted Winner: **{predicted_winner}**
    - Confidence Level: **{confidence}%**
    - Projected Score Difference: **{predicted_diff} points**
    - Suggested Spread: **{spread_suggestion}**
    - Total Predicted Points: **{predicted_total}**
    - Over/Under Suggestion: **{ou_suggestion}**

    #### Statistical Edge:
    The confidence level of **{confidence}%** reflects the statistical edge derived from the combined performance metrics of both teams.
    """
    
    return writeup


def display_bet_card(bet):
    with st.container():
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 1])

        # Game Info
        with col1:
            st.markdown(f"### {bet['away_team']} @ {bet['home_team']}")
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
    global results, team_stats_global

    st.header(f"Today's {league_choice} Best Bets 🎯")

    if league_choice == "NFL":
        schedule = load_nfl_schedule()
        if schedule.empty:
            st.error("Unable to load NFL schedule. Please try again later.")
            return
        team_data = preprocess_nfl_data(schedule)
        upcoming_games = fetch_upcoming_nfl_games(schedule, days_ahead=7)

    elif league_choice == "NBA":
        team_data = load_nba_data()
        if team_data.empty:
            st.error("Unable to load NBA data. Please try again later.")
            return
        upcoming_games = fetch_upcoming_nba_games(days_ahead=3)

    else:  # NCAAB
        team_data = load_ncaab_data_current_season(season=2025)
        if team_data.empty:
            st.error("Unable to load NCAAB data. Please try again later.")
            return

        with st.spinner("Fetching upcoming NCAAB games..."):
            upcoming_games = fetch_upcoming_ncaab_games()

    if team_data.empty or upcoming_games.empty:
        st.warning(f"No {league_choice} data available for analysis.")
        return

    # Train models and analyze games
    with st.spinner("Analyzing recent performance data..."):
        gbr_models, arima_models, team_stats_global = train_team_models(team_data)
        results.clear()

        for _, game in upcoming_games.iterrows():
            home_team, away_team = game['home_team'], game['away_team']
            
            home_pred, _ = predict_team_score(home_team, gbr_models, arima_models, team_stats_global, team_data, is_home=1)
            away_pred, _ = predict_team_score(away_team, gbr_models, arima_models, team_stats_global, team_data, is_home=0)

            outcome = evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats_global)
            if outcome:
                results.append({
                    'date': game['gameday'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_pred': home_pred,
                    'away_pred': away_pred,
                    'predicted_winner': outcome['predicted_winner'],
                    'predicted_diff': outcome['diff'],
                    'predicted_total': outcome['total_points'],
                    'confidence': outcome['confidence'],
                    'spread_suggestion': outcome['spread_suggestion'],
                    'ou_suggestion': outcome['ou_suggestion']
                })

    # Display bets based on user preference
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
        
        if top_bets:
            for bet in top_bets:
                display_bet_card(bet)
        else:
            st.info("No high-confidence bets found for today.")
    
    else:
        if results:
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

    if not st.session_state.get('logged_in', False):
        login_ui()
    
    else:
        league_choice = navigation_ui()
        run_league_pipeline(league_choice)

if __name__ == "__main__":
    main()
