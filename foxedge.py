import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import xgboost as xgb
from nba_api.stats.endpoints import (
    scoreboard, leaguedashplayerstats, teamgamelogs, boxscorematchupsv3, 
    boxscoreplayertrackv2, leagueplayerondetails, shotchartdetail
)
from scipy.stats import poisson
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------------------------- #
# 🏀 FETCH TODAY'S NBA SCHEDULE
# ---------------------------- #

def fetch_today_games():
    """Fetches today's scheduled NBA games."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    scoreboard_data = scoreboard.Scoreboard(league_id="00", game_date=today)
    
    games = scoreboard_data.get_data_frames()[0]
    
    game_list = []
    for _, game in games.iterrows():
        home_team = game["HOME_TEAM_ABBREVIATION"]
        away_team = game["VISITOR_TEAM_ABBREVIATION"]
        game_list.append(f"{away_team} @ {home_team}")
    
    return game_list

# ---------------------------- #
# 📊 FETCH TEAM & PLAYER STATISTICS
# ---------------------------- #

def fetch_team_data():
    """Fetches NBA team statistics including pace, offensive & defensive ratings."""
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season="2023-24").get_data_frames()[0]
    return stats.set_index("PLAYER_ID")

def fetch_team_gamelogs():
    """Fetches team game logs for recent performance trends."""
    team_logs = teamgamelogs.TeamGameLogs(season_nullable="2023-24").get_data_frames()[0]
    return team_logs.set_index("TEAM_ID")

def fetch_boxscore_matchups(game_id):
    """Fetches box score matchup stats for a given game ID."""
    boxscore = boxscorematchupsv3.BoxScoreMatchupsV3(game_id=game_id).get_data_frames()[0]
    return boxscore

def fetch_boxscore_player_tracking(game_id):
    """Fetches player tracking data (advanced stats) for a given game ID."""
    player_tracking = boxscoreplayertrackv2.BoxScorePlayerTrackV2(game_id=game_id).get_data_frames()[0]
    return player_tracking

def fetch_shot_quality(team_id, season="2023-24"):
    """Fetches shot quality data for a team."""
    shot_data = shotchartdetail.ShotChartDetail(team_id=team_id, season_nullable=season).get_data_frames()[0]
    return shot_data["SHOT_MADE_FLAG"].mean()  # Returns expected FG%

team_stats = fetch_team_data()
team_game_logs = fetch_team_gamelogs()

# ---------------------------- #
# 🤕 FETCH INJURY REPORTS
# ---------------------------- #

def fetch_injury_data():
    """Fetches NBA injury reports from an external API."""
    try:
        url = "https://www.balldontlie.io/api/v1/players"
        response = requests.get(url)
        data = response.json()
        injuries = {player['last_name']: player for player in data['data'] if player.get('status') == 'Out'}
        return injuries
    except:
        return {}

injury_report = fetch_injury_data()

def adjust_for_injuries(team):
    """Reduces offensive rating if a key player is injured."""
    injury_penalty = 2
    key_players = ["James", "Durant", "Curry", "Antetokounmpo"]
    missing_players = [player for player in key_players if player in injury_report]

    if missing_players:
        adjusted_rating = team_stats.loc[team, "OFF_RATING"] - (len(missing_players) * injury_penalty)
        return max(adjusted_rating, 90)
    return team_stats.loc[team, "OFF_RATING"]

# ---------------------------- #
# 🎯 MACHINE LEARNING MODEL
# ---------------------------- #

def train_ml_model():
    """Trains an XGBoost regression model on past game data."""
    df = team_game_logs
    features = ["PTS", "AST", "REB", "TOV", "FG_PCT", "TEAM_ID"]
    X = df[features]
    y = df["PTS"]  # Predicting team points

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Model MAE:", mean_absolute_error(y_test, preds))

    return model

ml_model = train_ml_model()

def predict_points_ml(team_id):
    """Uses trained model to predict team points based on stats."""
    team_data = team_game_logs.loc[team_id, ["PTS", "AST", "REB", "TOV", "FG_PCT"]].values.reshape(1, -1)
    return ml_model.predict(team_data)[0]

# ---------------------------- #
# 📡 REAL-TIME BETTING ODDS
# ---------------------------- #

def fetch_betting_odds():
    """Fetches real-time sportsbook odds from an API."""
    try:
        url = "https://api.sportsbook.com/nba/odds"
        response = requests.get(url)
        data = response.json()
        return data
    except:
        return {}

# ---------------------------- #
# 🎛️ STREAMLIT UI (Dropdown & Predictions)
# ---------------------------- #

st.title("🏀 Advanced NBA Predictor with AI & Betting Insights")

# Fetch today’s games dynamically
games = fetch_today_games()
selected_game = st.selectbox("Select a Game:", games)

if selected_game:
    away_team, home_team = selected_game.split(" @ ")

    st.write(f"🔍 Running AI-powered prediction for: **{away_team} @ {home_team}**")

    # Predict Points
    home_points = predict_points_ml(home_team)
    away_points = predict_points_ml(away_team)

    # Fetch betting odds
    odds = fetch_betting_odds()

    # Display Results
    st.write(f"✅ **Predicted Points:** {home_team}: {home_points:.1f}, {away_team}: {away_points:.1f}")
    st.write(f"✅ **Predicted Spread:** {home_team} by {home_points - away_points:.1f} points")

    # Compare with sportsbook odds
    if odds:
        sportsbook_line = odds.get(f"{away_team} @ {home_team}", {}).get("total", None)
        if sportsbook_line:
            st.write(f"📡 **Sportsbook Total:** {sportsbook_line}")
            if home_points + away_points > sportsbook_line:
                st.write("🚀 **BET RECOMMENDATION:** Bet the OVER")
            else:
                st.write("🚀 **BET RECOMMENDATION:** Bet the UNDER")
