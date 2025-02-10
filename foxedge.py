#!/usr/bin/env python3
"""
UI Script: Loads cached models and processed data; provides Firebase login/registration;
manages CSV output; and displays predictions (bet cards, detailed insights, top bets, etc.)
using Streamlit. All UI and visual components are exactly the same as your original script.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import joblib
import requests

# Firebase imports and initialization
import firebase_admin
from firebase_admin import credentials, auth

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
    """
    Login user using Firebase REST API.
    """
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
    """
    Sign up a new user using Firebase.
    """
    try:
        user = auth.create_user(email=email, password=password)
        st.success(f"User {email} created successfully!")
        return user
    except Exception as e:
        st.error(f"Error: {e}")


def logout_user():
    """
    Logs out the current user.
    """
    for key in ["email", "logged_in"]:
        if key in st.session_state:
            del st.session_state[key]


# CSV Management Functions
def initialize_csv(csv_file="predictions.csv"):
    """Initialize the CSV file if it doesn't exist."""
    from pathlib import Path
    if not Path(csv_file).exists():
        columns = [
            "date", "league", "home_team", "away_team", "home_pred", "away_pred",
            "predicted_winner", "predicted_diff", "predicted_total",
            "spread_suggestion", "ou_suggestion", "confidence"
        ]
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)


def save_predictions_to_csv(predictions, csv_file="predictions.csv"):
    """Save predictions to a CSV file."""
    df = pd.DataFrame(predictions)
    from pathlib import Path
    if Path(csv_file).exists():
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_file, index=False)
    st.success("Predictions have been saved to CSV!")


# UI Helper Functions (Bet Cards, Writeup, Top Bets)
def generate_writeup(bet, team_stats_global):
    """Generates a detailed analysis writeup for a given bet."""
    home_team = bet["home_team"]
    away_team = bet["away_team"]
    home_pred = bet["home_pred"]
    away_pred = bet["away_pred"]
    predicted_winner = bet["predicted_winner"]
    confidence = bet["confidence"]

    home_stats = team_stats_global.get(home_team, {})
    away_stats = team_stats_global.get(away_team, {})

    home_mean = home_stats.get("mean", "N/A")
    home_std = home_stats.get("std", "N/A")
    home_recent = home_stats.get("recent_form", "N/A")
    away_mean = away_stats.get("mean", "N/A")
    away_std = away_stats.get("std", "N/A")
    away_recent = away_stats.get("recent_form", "N/A")

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
  Based on the statistical analysis, **{predicted_winner}** is predicted to win with a confidence of **{confidence}%**.
  The projected score difference is **{bet['predicted_diff']} points** and the total points are **{bet['predicted_total']}**.
  Betting suggestions: **{bet['spread_suggestion']}** and **{bet['ou_suggestion']}**.
"""
    return writeup


def display_bet_card(bet, team_stats_global, team_data=None):
    """Displays a bet card with summary and expandable detailed insights."""
    conf = bet["confidence"]
    if conf >= 80:
        confidence_color = "green"
    elif conf < 60:
        confidence_color = "red"
    else:
        confidence_color = "orange"

    with st.container():
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"### **{bet['away_team']} @ {bet['home_team']}**")
            date_obj = bet["date"]
            if isinstance(date_obj, datetime):
                st.caption(date_obj.strftime("%A, %B %d - %I:%M %p"))
        with col2:
            if conf >= 80:
                st.markdown("🔥 **High-Confidence Bet** 🔥")
            st.markdown(f"**Spread Suggestion:** {bet['spread_suggestion']}")
            st.markdown(f"**Total Suggestion:** {bet['ou_suggestion']}")
        with col3:
            tooltip_text = "Confidence is derived from the statistical edge based on recent performance metrics."
            st.markdown(f"<h3 style='color:{confidence_color};' title='{tooltip_text}'>{conf:.1f}% Confidence</h3>", unsafe_allow_html=True)

    with st.expander("Detailed Insights", expanded=False):
        st.markdown(f"**Predicted Winner:** {bet['predicted_winner']}")
        st.markdown(f"**Predicted Total Points:** {bet['predicted_total']}")
        st.markdown(f"**Prediction Margin (Diff):** {bet['predicted_diff']}")
    with st.expander("Game Analysis", expanded=False):
        writeup = generate_writeup(bet, team_stats_global)
        st.markdown(writeup)

    if team_data is not None:
        with st.expander("Recent Performance Trends", expanded=False):
            home_team_data = team_data[team_data["team"] == bet["home_team"]].sort_values("gameday")
            if not home_team_data.empty:
                st.markdown(f"**{bet['home_team']} Recent Scores:**")
                home_scores = home_team_data["score"].tail(5).reset_index(drop=True)
                st.line_chart(home_scores)
            away_team_data = team_data[team_data["team"] == bet["away_team"]].sort_values("gameday")
            if not away_team_data.empty:
                st.markdown(f"**{bet['away_team']} Recent Scores:**")
                away_scores = away_team_data["score"].tail(5).reset_index(drop=True)
                st.line_chart(away_scores)


def find_top_bets(matchups, threshold=70.0):
    """Filters matchups and returns those with confidence above the threshold."""
    df = pd.DataFrame(matchups)
    df_top = df[df["confidence"] >= threshold].copy()
    df_top.sort_values("confidence", ascending=False, inplace=True)
    return df_top


# PredictionEngine: Loads cached models and data instead of training on the fly.
class PredictionEngine:
    def __init__(self):
        self.model_dir = Path("models")
        self.data_dir = Path("data")
        try:
            self.team_stats = joblib.load(self.model_dir / "team_stats.pkl")
        except Exception as e:
            st.error(f"Error loading team stats: {e}")
            self.team_stats = {}
        try:
            self.last_update = joblib.load(self.model_dir / "last_update.pkl")
        except Exception as e:
            st.error(f"Error loading last update timestamp: {e}")
            self.last_update = datetime.now()
        self.load_cached_data()

    def load_cached_data(self):
        try:
            self.nfl_data = pd.read_parquet(self.data_dir / "nfl_processed.parquet")
        except Exception as e:
            st.error(f"Error loading NFL data: {e}")
            self.nfl_data = pd.DataFrame()
        try:
            self.nba_data = pd.read_parquet(self.data_dir / "nba_processed.parquet")
        except Exception as e:
            st.error(f"Error loading NBA data: {e}")
            self.nba_data = pd.DataFrame()
        try:
            self.ncaab_data = pd.read_parquet(self.data_dir / "ncaab_processed.parquet")
        except Exception as e:
            st.error(f"Error loading NCAAB data: {e}")
            self.ncaab_data = pd.DataFrame()

    @staticmethod
    def round_half(number):
        """Rounds a number to the nearest 0.5."""
        return round(number * 2) / 2

    def predict_team_score(self, team: str, league: str):
        """
        Predicts the next-game score for a given team using pre‑trained stacking and ARIMA models.
        Returns (prediction, (conf_low, conf_high)) or (None, (None, None)) if unavailable.
        """
        try:
            stack_model = joblib.load(self.model_dir / f"{team}_stack.pkl")
            arima_model = joblib.load(self.model_dir / f"{team}_arima.pkl")
        except Exception as e:
            st.error(f"Error loading models for {team}: {e}")
            return None, (None, None)

        # Choose team data based on league
        if league == "NFL":
            team_data = self.nfl_data[self.nfl_data["team"] == team]
        elif league == "NBA":
            team_data = self.nba_data[self.nba_data["team"] == team]
        else:
            team_data = self.ncaab_data[self.ncaab_data["team"] == team]

        if team_data.empty:
            st.warning(f"No recent data for {team}.")
            return None, (None, None)

        try:
            latest = team_data.sort_values("gameday").iloc[-1]
            features = np.array([[latest["rolling_avg"], latest["rolling_std"], latest["weighted_avg"]]])
        except Exception as e:
            st.error(f"Error preparing features for {team}: {e}")
            return None, (None, None)

        try:
            stack_pred = float(stack_model.predict(features)[0])
        except Exception as e:
            st.error(f"Error predicting with stacking model for {team}: {e}")
            stack_pred = None

        try:
            arima_forecast = arima_model.predict(n_periods=1)
            arima_pred = float(arima_forecast[0] if isinstance(arima_forecast, (list, np.ndarray)) else arima_forecast)
        except Exception as e:
            st.error(f"Error predicting with ARIMA model for {team}: {e}")
            arima_pred = None

        # Ensemble predictions (weighted if possible)
        if stack_pred is not None and arima_pred is not None:
            mse_stack = self.team_stats.get(team, {}).get("mse", 1)
            mse_arima = None
            try:
                resid = arima_model.resid()
                mse_arima = np.mean(np.square(resid))
            except Exception:
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
        else:
            ensemble = None

        # MSE-based filtering: if team mse > 150, no prediction
        if self.team_stats.get(team, {}).get("mse", 0) > 150 or ensemble is None:
            return None, (None, None)

        # Bias calibration using training residual bias
        bias = self.team_stats.get(team, {}).get("bias", 0)
        ensemble_calibrated = ensemble + bias

        mu = self.team_stats.get(team, {}).get("mean", 0)
        sigma = self.team_stats.get(team, {}).get("std", 0)
        if isinstance(mu, (pd.Series, pd.DataFrame, np.ndarray)):
            mu = mu.item()
        if isinstance(sigma, (pd.Series, pd.DataFrame, np.ndarray)):
            sigma = sigma.item()

        conf_low = self.round_half(mu - 1.96 * sigma)
        conf_high = self.round_half(mu + 1.96 * sigma)

        return self.round_half(ensemble_calibrated), (conf_low, conf_high)

    def evaluate_matchup(self, home_team: str, away_team: str, home_pred, away_pred):
        """
        Evaluates a matchup by computing the predicted spread, total points, confidence, and betting suggestions.
        Returns a dictionary with these values.
        """
        if home_pred is None or away_pred is None:
            return None

        diff = home_pred - away_pred
        total_points = home_pred + away_pred
        home_std = self.team_stats.get(home_team, {}).get("std", 5)
        away_std = self.team_stats.get(away_team, {}).get("std", 5)
        combined_std = max(1.0, (home_std + away_std) / 2)
        raw_conf = abs(diff) / combined_std
        confidence = round(min(99, max(1, 50 + raw_conf * 15)), 2)
        penalty = 0
        if self.team_stats.get(home_team, {}).get("mse", 0) > 120:
            penalty += 10
        if self.team_stats.get(away_team, {}).get("mse", 0) > 120:
            penalty += 10
        confidence = max(1, min(99, confidence - penalty))
        winner = home_team if diff > 0 else away_team
        ou_threshold = 145

        return {
            "predicted_winner": winner,
            "diff": self.round_half(diff),
            "total_points": self.round_half(total_points),
            "confidence": confidence,
            "spread_suggestion": f"Lean {winner} by {self.round_half(diff):.1f}",
            "ou_suggestion": f"Take the {'Over' if total_points > ou_threshold else 'Under'} {self.round_half(total_points):.1f}"
        }

    def get_upcoming_games(self, league: str, days_ahead: int = 7) -> pd.DataFrame:
        """
        Retrieves upcoming games. For NFL, uses cached schedule; for NBA and NCAAB, fetches schedule via APIs.
        """
        now = datetime.now()
        end_date = now + timedelta(days=days_ahead)

        if league == "NFL":
            try:
                schedule = pd.read_parquet(self.data_dir / "nfl_schedule.parquet")
            except Exception as e:
                st.error(f"Error loading NFL schedule: {e}")
                return pd.DataFrame()
            upcoming = schedule[
                (pd.to_datetime(schedule["gameday"]) >= now) &
                (pd.to_datetime(schedule["gameday"]) <= end_date) &
                schedule["home_score"].isna()
            ].copy()
            return upcoming

        elif league == "NBA":
            upcoming_rows = []
            days = 3  # next 3 days
            for offset in range(days + 1):
                date_target = now + timedelta(days=offset)
                date_str = date_target.strftime("%Y-%m-%d")
                try:
                    from nba_api.stats.endpoints import ScoreboardV2
                    from nba_api.stats.static import teams as nba_teams
                    scoreboard = ScoreboardV2(game_date=date_str)
                    games = scoreboard.get_data_frames()[0]
                except Exception as e:
                    st.error(f"Error fetching NBA schedule for {date_str}: {e}")
                    continue
                if games.empty:
                    continue
                nba_team_dict = {tm["id"]: tm["abbreviation"] for tm in nba_teams.get_teams()}
                games["HOME_TEAM_ABBREV"] = games["HOME_TEAM_ID"].map(nba_team_dict)
                games["AWAY_TEAM_ABBREV"] = games["VISITOR_TEAM_ID"].map(nba_team_dict)
                upcoming_df = games[~games["GAME_STATUS_TEXT"].str.contains("Final", case=False, na=False)]
                for _, g in upcoming_df.iterrows():
                    upcoming_rows.append({
                        "gameday": pd.to_datetime(date_str),
                        "home_team": g["HOME_TEAM_ABBREV"],
                        "away_team": g["AWAY_TEAM_ABBREV"]
                    })
            if upcoming_rows:
                upcoming = pd.DataFrame(upcoming_rows)
                upcoming.sort_values("gameday", inplace=True)
                return upcoming
            else:
                return pd.DataFrame()

        elif league == "NCAAB":
            upcoming_rows = []
            timezone = pytz.timezone("America/Los_Angeles")
            current_time = datetime.now(timezone)
            dates = [
                current_time.strftime("%Y%m%d"),
                (current_time + timedelta(days=1)).strftime("%Y%m%d")
            ]
            for date_str in dates:
                url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
                params = {"dates": date_str, "groups": "50", "limit": "357"}
                try:
                    response = requests.get(url, params=params)
                    if response.status_code != 200:
                        st.warning(f"ESPN API request failed for date {date_str} with status code {response.status_code}")
                        continue
                    data = response.json()
                except Exception as e:
                    st.error(f"Error fetching NCAAB schedule for {date_str}: {e}")
                    continue
                games = data.get("events", [])
                if not games:
                    continue
                for game in games:
                    game_time_str = game["date"]
                    try:
                        game_time = datetime.fromisoformat(game_time_str[:-1]).astimezone(timezone)
                    except Exception:
                        continue
                    competitors = game["competitions"][0]["competitors"]
                    home_comp = next((c for c in competitors if c["homeAway"] == "home"), None)
                    away_comp = next((c for c in competitors if c["homeAway"] == "away"), None)
                    if not home_comp or not away_comp:
                        continue
                    home_team = home_comp["team"]["displayName"]
                    away_team = away_comp["team"]["displayName"]
                    upcoming_rows.append({
                        "gameday": game_time,
                        "home_team": home_team,
                        "away_team": away_team
                    })
            if upcoming_rows:
                upcoming = pd.DataFrame(upcoming_rows)
                upcoming.sort_values("gameday", inplace=True)
                return upcoming
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()


def main():
    st.set_page_config(
        page_title="FoxEdge Sports Betting Edge",
        page_icon="🦊",
        layout="centered"
    )
    st.title("🦊 FoxEdge Sports Betting Insights")

    # --- User Authentication via Firebase ---
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        col1, col2 = st.columns(2)
        with col1:
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
        st.stop()  # Prevent further UI until logged in
    else:
        st.sidebar.title("Account")
        st.sidebar.write(f"Logged in as: {st.session_state.get('email','Unknown')}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.experimental_rerun()
    # --- End Authentication ---

    # Initialize Prediction Engine
    engine = PredictionEngine()

    # Warn user if models are outdated
    if datetime.now() - engine.last_update > timedelta(hours=6):
        st.warning("Models may be outdated. Consider updating models.")

    # Sidebar: League selection
    league_choice = st.sidebar.radio("Select League", ["NFL", "NBA", "NCAAB"])

    st.header(f"Upcoming {league_choice} Games")
    upcoming_games = engine.get_upcoming_games(league_choice)
    if upcoming_games.empty:
        st.info(f"No upcoming {league_choice} games found.")
        return

    # Global list to hold predictions (for CSV saving, etc.)
    results = []
    team_stats_global = engine.team_stats  # For use in writeups

    # Iterate over upcoming games and display prediction cards
    for game in upcoming_games.itertuples():
        home_team = game.home_team
        away_team = game.away_team

        home_pred, home_conf = engine.predict_team_score(home_team, league_choice)
        away_pred, away_conf = engine.predict_team_score(away_team, league_choice)

        if home_pred is None or away_pred is None:
            st.info(f"Skipping {away_team} @ {home_team} due to insufficient prediction data.")
            continue

        outcome = engine.evaluate_matchup(home_team, away_team, home_pred, away_pred)
        if outcome is None:
            st.info(f"Skipping {away_team} @ {home_team} due to evaluation issues.")
            continue

        # Append prediction details to results list for CSV management
        results.append({
            "date": game.gameday,
            "league": league_choice,
            "home_team": home_team,
            "away_team": away_team,
            "home_pred": home_pred,
            "away_pred": away_pred,
            "predicted_winner": outcome["predicted_winner"],
            "predicted_diff": outcome["diff"],
            "predicted_total": outcome["total_points"],
            "confidence": outcome["confidence"],
            "spread_suggestion": outcome["spread_suggestion"],
            "ou_suggestion": outcome["ou_suggestion"]
        })

        with st.container():
            st.markdown("---")
            st.subheader(f"**{away_team} @ {home_team}**")
            st.write(f"**Predicted Scores:** {home_team}: {home_pred:.1f} | {away_team}: {away_pred:.1f}")
            st.write(f"**Predicted Winner:** {outcome['predicted_winner']}")
            st.write(f"**Margin:** {abs(outcome['diff']):.1f} points")
            st.write(f"**Total Points:** {outcome['total_points']:.1f}")
            st.write(f"**Confidence:** {outcome['confidence']}%")
            st.write(f"**Spread Suggestion:** {outcome['spread_suggestion']}")
            st.write(f"**Over/Under Suggestion:** {outcome['ou_suggestion']}")
            with st.expander("Detailed Insights"):
                st.markdown(generate_writeup({
                    **outcome,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_pred": home_pred,
                    "away_pred": away_pred
                }, team_stats_global))
            with st.expander("Recent Performance Trends"):
                # Display performance trends using cached data
                if league_choice == "NFL":
                    team_data = engine.nfl_data
                elif league_choice == "NBA":
                    team_data = engine.nba_data
                else:
                    team_data = engine.ncaab_data
                display_bet_card({
                    **outcome,
                    "date": game.gameday,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_pred": home_pred,
                    "away_pred": away_pred
                }, team_stats_global, team_data=team_data)

    view_mode = st.radio("View Mode", ["🎯 Top Bets Only", "📊 All Games"], horizontal=True)
    if view_mode == "🎯 Top Bets Only":
        conf_threshold = st.slider("Minimum Confidence Level", 50.0, 99.0, 75.0, 5.0,
                                     help="Only show bets with confidence above this threshold")
        top_bets = find_top_bets(results, threshold=conf_threshold)
        if not top_bets.empty:
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
                display_bet_card(bet, team_stats_global, team_data=team_data)
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
                display_bet_card(bet, team_stats_global, team_data=team_data)
        else:
            st.info(f"No upcoming {league_choice} games found.")

    # CSV download button
    if st.button("Save Predictions to CSV"):
        if results:
            save_predictions_to_csv(results)
            csv_data = pd.DataFrame(results).to_csv(index=False).encode("utf-8")
            st.download_button(label="Download Predictions as CSV", data=csv_data,
                               file_name="predictions.csv", mime="text/csv")
        else:
            st.warning("No predictions to save.")

    st.sidebar.markdown(
        "### About FoxEdge\n"
        "FoxEdge provides data-driven insights for NFL, NBA, and NCAAB games, helping bettors make informed decisions."
    )


if __name__ == "__main__":
    main()
