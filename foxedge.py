#!/usr/bin/env python
"""
FoxEdge.py
----------
Streamlit UI that:
  - Requires user login.
  - Loads pretrained models for NFL, NBA, or NCAAB from the "models/" folder.
  - Displays predictions for upcoming games (including Top Bets, All Games, CSV export, etc.)
  - Retains the same UI/UX as your original unsplit script.
  - No in-app model training occurs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import pytz
import requests
import firebase_admin
from firebase_admin import credentials, auth

# Data fetching libraries
import nfl_data_py
from nba_api.stats.endpoints import TeamGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
import cbbpy.mens_scraper as cbb

##########################################
# 1) Firebase Login Setup (unchanged)
##########################################
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
    st.warning("Firebase secrets not found or incomplete in st.secrets. Verify secrets.toml.")

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
    for key in ["email", "logged_in"]:
        if key in st.session_state:
            del st.session_state[key]

###############################################
# 2) Load Pretrained Models Based on League
###############################################
@st.cache_resource
def load_pretrained_models(league):
    """
    Loads the pretrained models for the specified league.
    Returns (stack_models, arima_models, team_stats) or (None, None, None) if not found.
    """
    if league == "NFL":
        if not os.path.exists("models/stack_models_nfl.pkl"):
            return None, None, None
        stack_models = joblib.load("models/stack_models_nfl.pkl")
        arima_models = joblib.load("models/arima_models_nfl.pkl")
        team_stats = joblib.load("models/team_stats_nfl.pkl")
    elif league == "NBA":
        if not os.path.exists("models/stack_models_nba.pkl"):
            return None, None, None
        stack_models = joblib.load("models/stack_models_nba.pkl")
        arima_models = joblib.load("models/arima_models_nba.pkl")
        team_stats = joblib.load("models/team_stats_nba.pkl")
    elif league == "NCAAB":
        if not os.path.exists("models/stack_models_ncaab.pkl"):
            return None, None, None
        stack_models = joblib.load("models/stack_models_ncaab.pkl")
        arima_models = joblib.load("models/arima_models_ncaab.pkl")
        team_stats = joblib.load("models/team_stats_ncaab.pkl")
    else:
        return None, None, None
    return stack_models, arima_models, team_stats

###################################
# 3) Prediction and Adjustment Logic
###################################
def predict_team_score(team, stack_models, arima_models, team_stats, team_data):
    if team not in team_stats:
        return None, (None, None)
    df_team = team_data[team_data["team"] == team]
    if len(df_team) < 3:
        return None, (None, None)
    last_features = df_team[["rolling_avg", "rolling_std", "weighted_avg"]].tail(1)
    X_next = last_features.values
    stack_pred = None
    if team in stack_models:
        try:
            stack_pred = float(stack_models[team].predict(X_next)[0])
        except Exception as e:
            print(f"Error predicting with stacking for team {team}: {e}")
            stack_pred = None
    arima_pred = None
    if team in arima_models:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0])
        except Exception as e:
            print(f"Error predicting with ARIMA for team {team}: {e}")
            arima_pred = None
    ensemble = None
    if stack_pred is not None and arima_pred is not None:
        mse_stack = team_stats[team].get("mse", 1)
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
    if ensemble is None:
        return None, (None, None)
    if team_stats[team].get("mse", 0) > 150:
        return None, (None, None)
    bias = team_stats[team].get("bias", 0)
    ensemble_calibrated = ensemble + bias
    mu = team_stats[team]["mean"]
    sigma = team_stats[team]["std"]
    pred_score = round_half(ensemble_calibrated)
    conf_low = round_half(mu - 1.96 * sigma)
    conf_high = round_half(mu + 1.96 * sigma)
    return pred_score, (conf_low, conf_high)

def evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats):
    if home_pred is None or away_pred is None:
        return None
    diff = home_pred - away_pred
    total_points = home_pred + away_pred
    home_std = team_stats.get(home_team, {}).get("std", 5)
    away_std = team_stats.get(away_team, {}).get("std", 5)
    combined_std = max(1.0, (home_std + away_std) / 2)
    raw_conf = abs(diff) / combined_std
    confidence = round(min(99, max(1, 50 + raw_conf * 15)), 2)
    penalty = 0
    if team_stats.get(home_team, {}).get("mse", 0) > 120:
        penalty += 10
    if team_stats.get(away_team, {}).get("mse", 0) > 120:
        penalty += 10
    confidence = max(1, min(99, confidence - penalty))
    winner = home_team if diff > 0 else away_team
    ou_threshold = 145
    return {
        "predicted_winner": winner,
        "diff": round_half(diff),
        "total_points": round_half(total_points),
        "confidence": confidence,
        "spread_suggestion": f"Lean {winner} by {round_half(diff):.1f}",
        "ou_suggestion": f"Take the {'Over' if total_points > ou_threshold else 'Under'} {round_half(total_points):.1f}"
    }

def round_half(x):
    return round(x * 2) / 2

##############################################
# 4) Data Fetching Functions
##############################################
def load_nfl_schedule():
    current_year = datetime.now().year
    years = [current_year - i for i in range(5)]
    schedule = nfl_data_py.import_schedules(years)
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce").dt.tz_localize(None)
    schedule.sort_values("gameday", inplace=True)
    return schedule

def preprocess_nfl_data(schedule):
    home_df = schedule[["gameday", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "score", "away_score": "opp_score"}
    )
    away_df = schedule[["gameday", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "score", "home_score": "opp_score"}
    )
    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=["score"], inplace=True)
    data.sort_values("gameday", inplace=True)
    data["rolling_avg"] = data.groupby("team")["score"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data["rolling_std"] = data.groupby("team")["score"].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    data["season_avg"] = data.groupby("team")["score"].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
    data["weighted_avg"] = data["rolling_avg"] * 0.6 + data["season_avg"] * 0.4
    return data

def fetch_upcoming_nfl_games(schedule, days_ahead=7):
    upcoming = schedule[schedule["home_score"].isna() & schedule["away_score"].isna()].copy()
    now = datetime.now()
    filter_date = now + timedelta(days=days_ahead)
    upcoming = upcoming[upcoming["gameday"] <= filter_date].copy()
    upcoming.sort_values("gameday", inplace=True)
    return upcoming[["gameday", "home_team", "away_team"]]

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

def load_ncaab_data_current_season(season=2025):
    info_df, _, _ = cbb.get_games_season(season=season, info=True, box=False, pbp=False)
    if info_df.empty:
        return pd.DataFrame()
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
    data['season_avg'] = data.groupby('team')['score'].transform(lambda x: x.expanding().mean())
    data['weighted_avg'] = data['rolling_avg'] * 0.6 + data['season_avg'] * 0.4
    data.sort_values(['team', 'gameday'], inplace=True)
    data['game_index'] = data.groupby('team').cumcount()
    return data

def fetch_upcoming_ncaab_games():
    timezone = pytz.timezone('America/Los_Angeles')
    current_time = datetime.now(timezone)
    dates = [
        current_time.strftime('%Y%m%d'),
        (current_time + timedelta(days=1)).strftime('%Y%m%d')
    ]
    rows = []
    for date_str in dates:
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

##############################################
# 5) MAIN PIPELINE (NO TRAINING)
##############################################
def run_league_pipeline(league_choice):
    st.header(f"Today's {league_choice} Best Bets 🎯")
    # 1) Load pretrained models for the selected league
    with st.spinner("Loading pretrained models..."):
        stack_models, arima_models, team_stats = load_pretrained_models(league_choice)
        if not stack_models:
            st.error("No pretrained models found. Please run daily training first.")
            return
    # 2) Load league data for upcoming games
    if league_choice == "NFL":
        schedule = load_nfl_schedule()
        team_data = preprocess_nfl_data(schedule)
        upcoming = fetch_upcoming_nfl_games(schedule, days_ahead=7)
    elif league_choice == "NBA":
        team_data = load_nba_data()
        if team_data.empty:
            st.error("Unable to load NBA data.")
            return
        upcoming = fetch_upcoming_nba_games(days_ahead=3)
    elif league_choice == "NCAAB":
        team_data = load_ncaab_data_current_season(season=2025)
        if team_data.empty:
            st.error("Unable to load NCAAB data.")
            return
        upcoming = fetch_upcoming_ncaab_games()
    else:
        st.error("Invalid league selection.")
        return
    if team_data.empty or upcoming.empty:
        st.warning(f"No upcoming {league_choice} data available for analysis.")
        return

    # (Optional: Compute defensive ratings and adjustments as in original code)
    if league_choice == "NBA":
        def_ratings = team_data.groupby('team')['def_rating'].mean().to_dict()
        sorted_def = sorted(def_ratings.items(), key=lambda x: x[1])
        top_10 = set([t for t, r in sorted_def[:10]])
        bottom_10 = set([t for t, r in sorted(def_ratings.items(), key=lambda x: x[1], reverse=True)[:10]])
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
        # Note: Instead of training, we load the pretrained models.
        # For the purposes of dynamic adjustments below, we still need team_stats.
        # Here, team_stats is already loaded above.
        results = []
        for _, row in upcoming.iterrows():
            home, away = row["home_team"], row["away_team"]
            home_pred, _ = predict_team_score(home, stack_models, arima_models, team_stats, team_data)
            away_pred, _ = predict_team_score(away, stack_models, arima_models, team_stats, team_data)

            # --- Dynamic Adjustments (same as original) ---
            row_gameday = to_naive(row["gameday"])
            if league_choice == "NBA" and home_pred is not None and away_pred is not None:
                home_games = team_data[team_data["team"] == home]
                if not home_games.empty:
                    last_game_home = to_naive(home_games["gameday"].max())
                    rest_days_home = (row_gameday - last_game_home).days
                    if rest_days_home == 0:
                        home_pred -= 3
                    elif rest_days_home >= 3:
                        home_pred += 2
                away_games = team_data[team_data["team"] == away]
                if not away_games.empty:
                    last_game_away = to_naive(away_games["gameday"].max())
                    rest_days_away = (row_gameday - last_game_away).days
                    if rest_days_away == 0:
                        away_pred -= 3
                    elif rest_days_away >= 3:
                        away_pred += 2
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
            elif league_choice == "NFL" and home_pred is not None and away_pred is not None:
                home_games = team_data[team_data["team"] == home]
                if not home_games.empty:
                    last_game_home = to_naive(home_games["gameday"].max())
                    rest_days_home = (row_gameday - last_game_home).days
                    if rest_days_home == 0:
                        home_pred -= 2
                    elif rest_days_home >= 3:
                        home_pred += 1
                away_games = team_data[team_data["team"] == away]
                if not away_games.empty:
                    last_game_away = to_naive(away_games["gameday"].max())
                    rest_days_away = (row_gameday - last_game_away).days
                    if rest_days_away == 0:
                        away_pred -= 2
                    elif rest_days_away >= 3:
                        away_pred += 1
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
                home_games = team_data[team_data["team"] == home]
                if not home_games.empty:
                    last_game_home = to_naive(home_games["gameday"].max())
                    rest_days_home = (row_gameday - last_game_home).days
                    if rest_days_home == 0:
                        home_pred -= 3
                    elif rest_days_home >= 3:
                        home_pred += 2
                away_games = team_data[team_data["team"] == away]
                if not away_games.empty:
                    last_game_away = to_naive(away_games["gameday"].max())
                    rest_days_away = (row_gameday - last_game_away).days
                    if rest_days_away == 0:
                        away_pred -= 3
                    elif rest_days_away >= 3:
                        away_pred += 2
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
                    "date": row["gameday"],
                    "league": league_choice,
                    "home_team": home,
                    "away_team": away,
                    "home_pred": home_pred,
                    "away_pred": away_pred,
                    "predicted_winner": outcome["predicted_winner"],
                    "predicted_diff": outcome["diff"],
                    "predicted_total": outcome["total_points"],
                    "confidence": outcome["confidence"],
                    "spread_suggestion": outcome["spread_suggestion"],
                    "ou_suggestion": outcome["ou_suggestion"]
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
        top_bets = None
        if results:
            df_results = pd.DataFrame(results)
            top_bets = df_results[df_results["confidence"] >= conf_threshold].copy()
            top_bets.sort_values("confidence", ascending=False, inplace=True)
        if top_bets is not None and not top_bets.empty:
            st.markdown(f"### 🔥 Top {len(top_bets)} Bets for Today")
            previous_date = None
            for _, bet_row in top_bets.iterrows():
                bet = bet_row.to_dict()
                current_date = bet["date"].date() if isinstance(bet["date"], datetime) else bet["date"]
                if previous_date != current_date:
                    if isinstance(bet["date"], datetime):
                        st.markdown(f"## {bet['date'].strftime('%A, %B %d, %Y')}")
                    else:
                        st.markdown(f"## {bet['date']}")
                    previous_date = current_date
                display_bet_card(bet, team_stats, team_data=team_data)
        else:
            st.info("No high-confidence bets found. Try lowering the threshold.")
    else:
        if results:
            st.markdown("### 📊 All Games Analysis")
            sorted_results = sorted(results, key=lambda x: x["date"])
            previous_date = None
            for bet in sorted_results:
                current_date = bet["date"].date() if isinstance(bet["date"], datetime) else bet["date"]
                if previous_date != current_date:
                    if isinstance(bet["date"], datetime):
                        st.markdown(f"## {bet['date'].strftime('%A, %B %d, %Y')}")
                    else:
                        st.markdown(f"## {bet['date']}")
                    previous_date = current_date
                display_bet_card(bet, team_stats, team_data=team_data)
        else:
            st.info(f"No upcoming {league_choice} games found.")

######################################
# 6) Streamlit Main + Login
######################################
def main():
    st.set_page_config(page_title="FoxEdge Sports Betting Edge", page_icon="🦊", layout="centered")
    st.title("🦊 FoxEdge Sports Betting Insights (No In-App Training)")
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if not st.session_state["logged_in"]:
        col1, col2 = st.columns(2)
        with col1:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                user_data = login_with_rest(email, password)
                if user_data:
                    st.session_state["logged_in"] = True
                    st.session_state["email"] = user_data["email"]
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
    league_choice = st.sidebar.radio("Select League", ["NFL", "NBA", "NCAAB"])
    run_league_pipeline(league_choice)

if __name__ == "__main__":
    main()
