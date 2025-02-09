################################################################################
# FoxEdge.py
# ------------------------------------------------------------------------------
# Streamlit UI that:
#  - Requires user login
#  - Loads pretrained Stack + ARIMA models from 'models/*.pkl' 
#  - Displays predictions for upcoming games 
#  - No model training here.
################################################################################

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

# If needed for fetching upcoming data:
import nfl_data_py
from nba_api.stats.endpoints import TeamGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
import cbbpy.mens_scraper as cbb

##########################################
# 1) FIREBASE LOGIN (same as your original)
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
    """Same as before. Uses Firebase REST API."""
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
    """Same as before."""
    try:
        user = auth.create_user(email=email, password=password)
        st.success(f"User {email} created successfully!")
        return user
    except Exception as e:
        st.error(f"Error: {e}")

def logout_user():
    """Same as before."""
    for key in ["email", "logged_in"]:
        if key in st.session_state:
            del st.session_state[key]


###############################################
# 2) LOAD MODELS FROM DISK (No training here!)
###############################################
@st.cache_resource
def load_pretrained_models():
    """
    Loads the stack_models, arima_models, and team_stats from disk.
    If not found, returns None.
    """
    if not os.path.exists("models/stack_models.pkl"):
        return None, None, None
    
    stack_models = joblib.load("models/stack_models.pkl")
    arima_models = joblib.load("models/arima_models.pkl")
    team_stats   = joblib.load("models/team_stats.pkl")
    return stack_models, arima_models, team_stats


###################################
# 3) PREDICTION + ADJUSTMENT LOGIC
###################################
def predict_team_score(team, stack_models, arima_models, team_stats, team_data):
    """
    EXACT same logic you used before to do final blending, 
    but skip the part where we train. 
    We just use the loaded stack_models + arima_models + team_data.
    """
    if team not in team_stats:
        return None, (None, None)

    df_team = team_data[team_data["team"] == team]
    if len(df_team) < 3:
        return None, (None, None)

    last_features = df_team[["rolling_avg", "rolling_std", "weighted_avg"]].tail(1)
    X_next = last_features.values

    # Pull from loaded models
    stack_pred = None
    if team in stack_models:
        try:
            stack_pred = float(stack_models[team].predict(X_next)[0])
        except:
            stack_pred = None

    arima_pred = None
    if team in arima_models:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0])
        except:
            arima_pred = None

    # Weighted ensemble
    ensemble = None
    if stack_pred is not None and arima_pred is not None:
        mse_stack = team_stats[team].get("mse", 1)
        # approximate ARIMA MSE from residual if desired
        try:
            resid = arima_models[team].resid()
            mse_arima = np.mean(np.square(resid))
        except:
            mse_arima = None
        
        if mse_arima and mse_arima > 0:
            weight_stack = 1 / (mse_stack + 1e-6)
            weight_arima = 1 / (mse_arima + 1e-6)
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
    conf_low   = round_half(mu - 1.96 * sigma)
    conf_high  = round_half(mu + 1.96 * sigma)
    return pred_score, (conf_low, conf_high)

def evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats):
    """
    Same advanced logic for confidence, diff, total points, etc.
    """
    if home_pred is None or away_pred is None:
        return None

    diff = home_pred - away_pred
    total_points = home_pred + away_pred

    home_std = team_stats.get(home_team, {}).get("std", 5)
    away_std = team_stats.get(away_team, {}).get("std", 5)
    combined_std = max(1.0, (home_std + away_std) / 2)

    raw_conf = abs(diff) / combined_std
    confidence = round(min(99, max(1, 50 + raw_conf * 15)), 2)

    # Adjust confidence if MSE is high
    penalty = 0
    if team_stats.get(home_team, {}).get("mse", 0) > 120:
        penalty += 10
    if team_stats.get(away_team, {}).get("mse", 0) > 120:
        penalty += 10
    confidence = max(1, min(99, confidence - penalty))

    winner = home_team if diff > 0 else away_team

    # Over/Under threshold
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
# 4) DATA FETCHING (Up to you, same logic)
##############################################
# For brevity, let's just show an example for NFL upcoming. 
# Replicate your own fetch_upcoming_nfl_games, load_nba_data, etc.

def load_nfl_schedule():
    current_year = datetime.now().year
    years = [current_year - i for i in range(5)]
    schedule = nfl_data_py.import_schedules(years)
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce").dt.tz_localize(None)
    schedule.sort_values("gameday", inplace=True)
    return schedule

def preprocess_nfl_data(schedule):
    home_df = schedule[["gameday","home_team","home_score","away_score"]].rename(
        columns={"home_team":"team","home_score":"score","away_score":"opp_score"}
    )
    away_df = schedule[["gameday","away_team","away_score","home_score"]].rename(
        columns={"away_team":"team","away_score":"score","home_score":"opp_score"}
    )
    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=["score"], inplace=True)
    data.sort_values("gameday", inplace=True)
    data["rolling_avg"] = data.groupby("team")["score"].transform(lambda x: x.rolling(3,min_periods=1).mean())
    data["rolling_std"] = data.groupby("team")["score"].transform(lambda x: x.rolling(3,min_periods=1).std().fillna(0))
    data["season_avg"]  = data.groupby("team")["score"].transform(lambda x: x.expanding().mean())
    data["weighted_avg"]= data["rolling_avg"]*0.6 + data["season_avg"]*0.4
    return data

def fetch_upcoming_nfl_games(schedule, days_ahead=7):
    now = datetime.now()
    future_date = now + timedelta(days=days_ahead)
    upcoming = schedule[
        schedule["home_score"].isna() & schedule["away_score"].isna() &
        (schedule["gameday"] <= future_date)
    ]
    upcoming.sort_values("gameday", inplace=True)
    return upcoming[["gameday","home_team","away_team"]]

##################################################
# 5) MAIN PIPELINE WITHOUT TRAINING
##################################################
def run_league_pipeline(league_choice):
    st.header(f"Today's {league_choice} Best Bets 🎯")

    # 1) Load the pretrained models
    with st.spinner("Analyzing recent performance data..."):
        stack_models, arima_models, team_stats = load_pretrained_models()
        if not stack_models:
            st.error("No pretrained models found. Please run daily training first.")
            return

    # 2) Load the league’s data for upcoming matchups
    if league_choice == "NFL":
        schedule = load_nfl_schedule()
        data = preprocess_nfl_data(schedule)
        upcoming = fetch_upcoming_nfl_games(schedule)
    else:
        st.warning("This demo only shows NFL. Implement NBA/NCAAB similarly.")
        return

    if data.empty or upcoming.empty:
        st.warning("No upcoming data available for analysis.")
        return

    # 3) For each upcoming game, predict
    results = []
    for _, row in upcoming.iterrows():
        home, away = row["home_team"], row["away_team"]
        home_pred, _ = predict_team_score(home, stack_models, arima_models, team_stats, data)
        away_pred, _ = predict_team_score(away, stack_models, arima_models, team_stats, data)

        # If you have dynamic adjustments, apply them here 
        # (rest days, top_10 defense, etc.) 
        # Then call evaluate_matchup
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
                "ou_suggestion": outcome["ou_suggestion"],
            })

    # 4) Display results
    if not results:
        st.info("No matchups found or no predictions available.")
        return

    # Example layout: "All Games"
    st.subheader("All Games Analysis")
    for bet in results:
        st.write("----")
        st.write(f"**{bet['away_team']} @ {bet['home_team']}** - {bet['date']}")
        st.write(f"Predicted Winner: {bet['predicted_winner']}, Confidence: {bet['confidence']}%")
        st.write(f"Spread Suggestion: {bet['spread_suggestion']}")
        st.write(f"Total Suggestion: {bet['ou_suggestion']}")
    # (You can keep your expanders, charts, etc., from the original code.)

######################################
# 6) STREAMLIT MAIN + LOGIN
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
        st.sidebar.write(f"Logged in as {st.session_state.get('email','Unknown')}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.experimental_rerun()

    # Let user pick league
    league_choice = st.sidebar.radio("Select League", ["NFL", "NBA", "NCAAB"])
    run_league_pipeline(league_choice)

if __name__ == "__main__":
    main()
