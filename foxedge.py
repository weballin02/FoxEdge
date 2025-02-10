#!/usr/bin/env python3
"""
UI Script: Loads cached models and data; makes predictions using the pre‑trained stacking and ARIMA models;
evaluates matchups; and displays upcoming game predictions using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import joblib
import requests

# For NBA schedule via nba_api
from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams as nba_teams


class PredictionEngine:
    """
    Loads cached models, team statistics, and processed league data.
    Provides methods to predict team scores and evaluate game matchups.
    """

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
        """Loads cached processed data for NFL, NBA, and NCAAB."""
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
        """
        Rounds a number to the nearest 0.5.
        
        Args:
            number (float): The number to round.
        
        Returns:
            float: The rounded number.
        """
        return round(number * 2) / 2

    def predict_team_score(self, team: str, league: str):
        """
        Predicts the next-game score for a given team by loading its pre‑trained stacking and ARIMA models.
        Returns a tuple (predicted_score, (conf_low, conf_high)) or (None, (None, None)) if prediction fails.
        
        Args:
            team (str): Team abbreviation.
            league (str): League name ("NFL", "NBA", or "NCAAB").
        
        Returns:
            tuple: (calibrated_prediction, (confidence_lower, confidence_upper))
        """
        try:
            stack_model = joblib.load(self.model_dir / f"{team}_stack.pkl")
            arima_model = joblib.load(self.model_dir / f"{team}_arima.pkl")
        except Exception as e:
            st.error(f"Error loading models for {team}: {e}")
            return None, (None, None)

        # Select team data based on league
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

        # Create ensemble prediction (weighted if possible)
        if stack_pred is not None and arima_pred is not None:
            mse_stack = self.team_stats.get(team, {}).get("mse", 1)
            mse_arima = None
            try:
                resid = arima_model.resid()
                mse_arima = np.mean(np.square(resid))
            except Exception as e:
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

        # MSE-based filtering: if team's mse > 150, return no prediction
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
        Evaluates a matchup by computing the predicted spread, total points, and a confidence value.
        Returns a dictionary with the predicted winner, point difference, total points,
        confidence, spread suggestion, and over/under suggestion.
        
        Args:
            home_team (str): Home team abbreviation.
            away_team (str): Away team abbreviation.
            home_pred (float): Predicted score for the home team.
            away_pred (float): Predicted score for the away team.
        
        Returns:
            dict or None: Evaluation dictionary or None if predictions are unavailable.
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
        Retrieves upcoming games. For NFL, uses cached schedule; for NBA and NCAAB,
        fetches schedule via their respective APIs.
        
        Args:
            league (str): League name ("NFL", "NBA", or "NCAAB").
            days_ahead (int): Number of days ahead to look.
        
        Returns:
            pd.DataFrame: DataFrame of upcoming games.
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

    engine = PredictionEngine()

    # Warn if models may be outdated
    if datetime.now() - engine.last_update > timedelta(hours=6):
        st.warning("Models may be outdated. Consider updating models.")

    league_choice = st.sidebar.radio("Select League", ["NFL", "NBA", "NCAAB"])

    st.header(f"Upcoming {league_choice} Games")
    upcoming_games = engine.get_upcoming_games(league_choice)
    if upcoming_games.empty:
        st.info(f"No upcoming {league_choice} games found.")
        return

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

    st.sidebar.markdown(
        "### About FoxEdge\n"
        "FoxEdge provides data-driven insights for NFL, NBA, and NCAAB games, "
        "helping bettors make informed decisions."
    )


if __name__ == "__main__":
    main()
