import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import datetime
from datetime import datetime as dt
from pathlib import Path

# NBA API
from nba_api.stats.endpoints import scoreboardv2, playergamelogs, commonteamroster
from nba_api.stats.static import teams

# Modeling
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K  # For clearing Keras sessions

###############################################################################
# 0) SET PAGE CONFIG IMMEDIATELY
###############################################################################
st.set_page_config(page_title="NBA Player Props Predictor", page_icon="🏀", layout="wide")

###############################################################################
# 1) CSV CONFIG & LOADING
###############################################################################

USE_NBA_CSV_DATA = True  # Toggle True to load from CSV first; fallback to API if missing

def load_csv_data_safe(file_path: str) -> pd.DataFrame:
    """Return a DataFrame from CSV, or an empty DataFrame on error."""
    f = Path(file_path)
    if not f.exists():
        print(f"CSV not found at {file_path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=14400)
def load_nba_players_csv() -> pd.DataFrame:
    """
    Attempt to load historical player logs from data/nba_players_all.csv
    with columns like: 'player_id', 'player_name', 'game_date', 'TEAM_ID',
    'MIN', 'PTS', 'AST', 'REB', 'STL', 'BLK', etc.
    """
    if not USE_NBA_CSV_DATA:
        return pd.DataFrame()
    csv_path = "data/nba_players_all.csv"
    df = load_csv_data_safe(csv_path)
    if df.empty:
        return df

    # Basic cleanup
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df.dropna(subset=["game_date"], inplace=True)

    # Ensure numeric columns
    numeric_cols = ["MIN","PTS","AST","REB","STL","BLK"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Sort by player_id, then date
    if "player_id" in df.columns:
        df.sort_values(["player_id","game_date"], inplace=True)
    return df

###############################################################################
# 2) UTILITY FUNCTIONS
###############################################################################

def calculate_rolling_averages(df, columns, windows):
    """Calculate rolling averages for specified columns and windows."""
    result = df.copy()
    for col in columns:
        for window in windows:
            result[f'{col}_{window}_AVG'] = df[col].rolling(window=window, min_periods=1).mean()
    return result

def format_prediction_output(predictions):
    """Format predictions dict for display."""
    return {
        'Points': f"{predictions.get('PTS', 0):.1f}",
        'Assists': f"{predictions.get('AST', 0):.1f}",
        'Rebounds': f"{predictions.get('REB', 0):.1f}",
        'Steals': f"{predictions.get('STL', 0):.1f}",
        'Blocks': f"{predictions.get('BLK', 0):.1f}"
    }

###############################################################################
# 3) DAY-BY-DAY SEARCH FOR NEXT GAME
###############################################################################

def find_next_gameday_with_games(start_date=None, max_days_ahead=14):
    """
    Checks NBA scoreboard from start_date up to max_days_ahead days.
    Returns the first date string (YYYY-MM-DD) that has at least one NBA game.
    If none are found, returns None.
    """
    if start_date is None:
        start_date = dt.now()

    for i in range(max_days_ahead):
        day_to_check = start_date + datetime.timedelta(days=i)
        date_str = day_to_check.strftime('%Y-%m-%d')

        scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
        games_df = scoreboard.get_data_frames()[0]
        if not games_df.empty:
            return date_str
    return None

###############################################################################
# 4) UI COMPONENTS (Player Prop Cards & Social Posts)
###############################################################################

def generate_social_media_post(bet):
    """
    Generate a social media post for a given bet (player prop).
    Expects 'player_name', 'team', 'predictions', 'confidence'.
    """
    conf = bet.get('confidence', 0)
    if conf >= 85:
        tone = "This one’s a sure-fire winner! Don’t miss out!"
    elif conf >= 70:
        tone = "Looks promising – keep an eye on this one…"
    else:
        tone = "A cautious bet worth watching!"
    
    templates = [
        f"🔥 **Bet Alert!** 🔥\n\n"
        f"**Player:** {bet['player_name']} from {bet['team']}\n\n"
        f"**Prediction Highlights:**\n"
        f"• Points: {bet['predictions'].get('PTS', 0):.1f}\n"
        f"• Assists: {bet['predictions'].get('AST', 0):.1f}\n"
        f"• Rebounds: {bet['predictions'].get('REB', 0):.1f}\n\n"
        f"**Confidence:** {bet.get('confidence', 0):.1f}%\n\n"
        f"{tone}\n\n"
        f"👉 **CTA:** {{cta}}\n\n"
        f"🏷️ {{hashtags}}",
        
        f"🚀 **Hot Tip Alert!** 🚀\n\n"
        f"Player: {bet['player_name']} ({bet['team']})\n\n"
        f"• Points: {bet['predictions'].get('PTS', 0):.1f}\n"
        f"• Assists: {bet['predictions'].get('AST', 0):.1f}\n"
        f"• Rebounds: {bet['predictions'].get('REB', 0):.1f}\n"
        f"• Confidence: {bet.get('confidence', 0):.1f}%\n\n"
        f"{tone}\n\n"
        f"👉 **CTA:** {{cta}}\n\n"
        f"🏷️ {{hashtags}}",
        
        f"🎯 **Pro Pick Alert!** 🎯\n\n"
        f"Player: {bet['player_name']} | Team: {bet['team']}\n"
        f"Predicted Stats: {format_prediction_output(bet['predictions'])}\n"
        f"Confidence: {bet.get('confidence', 0):.1f}%\n\n"
        f"{tone}\n\n"
        f"👉 **CTA:** {{cta}}\n\n"
        f"🏷️ {{hashtags}}"
    ]
    
    selected_template = random.choice(templates)
    cta_options = [
        "Comment your prediction below!",
        "Tag a friend who needs this tip!",
        "Download now for real-time insights!",
        "Join the winning team and share your pick!"
    ]
    selected_cta = random.choice(cta_options)
    hashtag_pool = ["#SportsBetting", "#GameDay", "#BetSmart", "#WinningTips", "#Edge", "#BettingCommunity"]
    selected_hashtags = " ".join(random.sample(hashtag_pool, k=3))
    post = selected_template.replace("{cta}", selected_cta).replace("{hashtags}", selected_hashtags)
    return post

def display_prop_card(bet):
    """
    Display a player prop card. Expects a bet dict with:
      - 'player_name', 'team', 'predictions' dict, 'value_bets' list, 'confidence' float.
    """
    confidence = bet.get('confidence', 0)
    if confidence >= 80:
        confidence_color = "green"
    elif confidence < 60:
        confidence_color = "red"
    else:
        confidence_color = "orange"

    with st.container():
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### **{bet['player_name']}**")
            st.caption(f"Team: {bet['team']}")
        with col2:
            st.markdown(
                f"<h3 style='color:{confidence_color};'>{confidence:.1f}% Confidence</h3>",
                unsafe_allow_html=True,
            )
        if 'predictions' in bet:
            formatted_preds = format_prediction_output(bet['predictions'])
            cols = st.columns(len(formatted_preds))
            for c, (prop, value) in zip(cols, formatted_preds.items()):
                c.metric(label=prop, value=value)

    with st.expander("Detailed Insights", expanded=False):
        if bet.get('value_bets'):
            for vb in bet['value_bets']:
                st.markdown(
                    f"**Prop:** {vb.get('prop', '')} | "
                    f"**Suggestion:** {'OVER' if vb.get('value', 0) > 0 else 'UNDER'} "
                    f"| **Confidence:** {vb.get('confidence', 0):.0%}"
                )
        else:
            st.info("No value bets available.")

    with st.expander("Generate Social Media Post", expanded=False):
        if st.button("Generate Post", key=f"post_{bet['player_name']}"):
            post = generate_social_media_post(bet)
            st.code(post, language="markdown")

###############################################################################
# 5) HELPER: Projected Starting Lineups
###############################################################################

def get_projected_starting_players(players_df, data_fetcher, games=3, n=5):
    """
    For each player in players_df, fetch recent game logs, compute average minutes,
    then return top n players by avg minutes.
    """
    players_with_minutes = []
    for _, player in players_df.iterrows():
        stats = data_fetcher.get_player_stats(player['id'], last_n_games=games)
        if stats.empty:
            avg_min = 0
        else:
            stats['MIN'] = pd.to_numeric(stats['MIN'], errors='coerce')
            avg_min = stats['MIN'].mean()
        players_with_minutes.append({
            'id': player['id'],
            'name': player['name'],
            'avg_min': avg_min
        })

    # Handle potential empty list
    if not players_with_minutes:
        return pd.DataFrame(columns=["id", "name", "avg_min"])

    df = pd.DataFrame(players_with_minutes, columns=["id", "name", "avg_min"])
    if not df.empty and 'avg_min' in df.columns:
        df = df.sort_values(by='avg_min', ascending=False)
    return df.head(n)

###############################################################################
# 6) DATA PROCESSOR & FETCHER
###############################################################################

class DataProcessor:
    def __init__(self):
        self.stats_columns = ['PTS','AST','REB','STL','BLK']

    def process_player_stats(self, player_stats_df):
        """
        Process player stats by computing rolling averages for MIN + key stats.
        """
        if player_stats_df.empty:
            return pd.DataFrame()
        base_cols = ['MIN'] + self.stats_columns
        processed_stats = calculate_rolling_averages(player_stats_df, base_cols, [5, 10])
        for s in self.stats_columns:
            processed_stats[f'{s}_PER_MIN'] = (processed_stats[s] / processed_stats['MIN']).fillna(0)
        return processed_stats

    def adjust_for_opponent(self, player_stats, team_stats):
        """(Optional) Adjust stats based on opponent metrics (not used)."""
        return player_stats

class NBADataFetcher:
    """
    Enhanced data fetcher that can read from CSV or fallback to the API.
    """
    players_csv_data = load_nba_players_csv()

    @st.cache_data(ttl=3600)
    def get_team_name(team_id):
        try:
            nba_teams = teams.get_teams()
            info = next((t for t in nba_teams if t['id'] == team_id), None)
            return info['full_name'] if info else f"Team {team_id}"
        except Exception as e:
            st.error(f"Error fetching team name: {e}")
            return f"Team {team_id}"

    @st.cache_data(ttl=3600)
    def get_team_players(team_id):
        from nba_api.stats.endpoints import commonteamroster
        time.sleep(0.6)
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=team_id)
            roster_df = roster.get_data_frames()[0]
            if not roster_df.empty:
                players_data = [{
                    'id': str(row['PLAYER_ID']),
                    'name': str(row['PLAYER'])
                } for _, row in roster_df.iterrows()]
                return pd.DataFrame(players_data)
            return pd.DataFrame(columns=['id','name'])
        except Exception as e:
            st.error(f"Error fetching team players: {e}")
            return pd.DataFrame(columns=['id','name'])

    @st.cache_data(ttl=3600)
    def get_player_stats(player_id, last_n_games=10):
        """
        If CSV data is available for this player, slice last N games from there.
        Otherwise fallback to the NBA API.
        """
        if USE_NBA_CSV_DATA and not NBADataFetcher.players_csv_data.empty:
            # Filter CSV by player_id
            df = NBADataFetcher.players_csv_data[
                NBADataFetcher.players_csv_data['player_id'] == float(player_id)
            ]
            if df.empty:
                return NBADataFetcher.fetch_player_logs_api(player_id, last_n_games)

            # Sort descending by date, take the last N
            df = df.sort_values('game_date', ascending=False).head(last_n_games)
            df = df.rename(columns={'game_date':'GAME_DATE'})
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            return df
        else:
            # Fallback to the API
            return NBADataFetcher.fetch_player_logs_api(player_id, last_n_games)

    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_player_logs_api(player_id, last_n_games=10):
        from nba_api.stats.endpoints import playergamelogs
        time.sleep(0.6)
        try:
            logs = playergamelogs.PlayerGameLogs(
                player_id_nullable=player_id,
                last_n_games_nullable=last_n_games
            )
            df = logs.get_data_frames()[0]
            if df.empty:
                return pd.DataFrame(columns=['GAME_DATE','MIN','PTS','AST','REB','STL','BLK'])
            return df
        except Exception as e:
            st.error(f"Error fetching player stats API: {e}")
            return pd.DataFrame(columns=['GAME_DATE','MIN','PTS','AST','REB','STL','BLK'])


###############################################################################
# 7) PROP PREDICTOR (Ensemble, Value Bets, etc.)
###############################################################################

class PropPredictor:
    def __init__(self):
        self.models = {
            'PTS': RandomForestRegressor(n_estimators=100, random_state=42),
            'AST': RandomForestRegressor(n_estimators=100, random_state=42),
            'REB': RandomForestRegressor(n_estimators=100, random_state=42),
            'STL': RandomForestRegressor(n_estimators=100, random_state=42),
            'BLK': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.ensemble_models = {}

    @staticmethod
    def prepare_features(player_stats):
        feature_cols = [
            'MIN_5_AVG','PTS_5_AVG','AST_5_AVG','REB_5_AVG',
            'MIN_10_AVG','PTS_10_AVG','AST_10_AVG','REB_10_AVG'
        ]
        if player_stats.empty:
            return pd.DataFrame(columns=feature_cols)
        for col in feature_cols:
            if col not in player_stats.columns:
                player_stats[col] = 0
        return player_stats[feature_cols].fillna(0)

    @staticmethod
    @st.cache_data
    def _cached_predict(_player_stats):
        """
        Baseline if no ensemble model is trained:
        Takes mean of last 10-game rolling average + random variation.
        """
        if _player_stats.empty:
            return pd.Series({'PTS':0,'AST':0,'REB':0,'STL':0,'BLK':0})
        predictions = {}
        for prop in ['PTS','AST','REB','STL','BLK']:
            col = f'{prop}_10_AVG'
            base_val = _player_stats[col].mean() if col in _player_stats.columns else 0
            variation = np.random.normal(0, max(1, base_val*0.1))
            predictions[prop] = max(0, base_val + variation)
        return pd.Series(predictions)

    def train_ensemble_models(self, X_train, y_train_dict):
        K.clear_session()
        self.ensemble_models = {}

        for prop in y_train_dict.keys():
            # XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_train, y_train_dict[prop])

            # GradientBoosting
            gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gbm_model.fit(X_train, y_train_dict[prop])

            # Simple NN
            nn_model = Sequential()
            nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
            nn_model.add(Dense(32, activation='relu'))
            nn_model.add(Dense(1, activation='linear'))
            nn_model.compile(loss='mse', optimizer='adam')
            nn_model.fit(X_train, y_train_dict[prop], epochs=50, batch_size=8, verbose=0)

            self.ensemble_models[prop] = {
                'xgb': xgb_model,
                'gbm': gbm_model,
                'nn': nn_model
            }
        st.success("Ensemble models trained successfully on historical data.")

    def predict_ensemble(self, player_stats):
        X = self.prepare_features(player_stats)
        predictions = {}
        for prop in self.ensemble_models.keys():
            xgb_pred = self.ensemble_models[prop]['xgb'].predict(X).mean()
            gbm_pred = self.ensemble_models[prop]['gbm'].predict(X).mean()
            nn_pred = self.ensemble_models[prop]['nn'].predict(X).mean()
            ensemble_pred = np.mean([xgb_pred, gbm_pred, nn_pred])
            predictions[prop] = ensemble_pred
        return pd.Series(predictions)

    def predict_props(self, player_stats):
        if player_stats is None or player_stats.empty:
            return pd.Series({'PTS':0,'AST':0,'REB':0,'STL':0,'BLK':0})
        if self.ensemble_models:
            return self.predict_ensemble(player_stats)
        return self._cached_predict(player_stats)

    def identify_value_bets(self, predictions, sportsbook_lines):
        """
        Mock approach to lines. In real usage, fetch actual lines from e.g. The Odds API.
        """
        value_bets = []
        if not predictions:
            return value_bets
        # Simple random offsets
        mock_lines = {
            'PTS': predictions.get('PTS',0) + np.random.normal(0,2),
            'AST': predictions.get('AST',0) + np.random.normal(0,1),
            'REB': predictions.get('REB',0) + np.random.normal(0,1)
        }
        for prop, pred in predictions.items():
            if prop in mock_lines:
                diff = pred - mock_lines[prop]
                if abs(diff) > 2:  # Arbitrary threshold
                    value_bets.append({
                        'prop': prop,
                        'prediction': pred,
                        'line': mock_lines[prop],
                        'value': diff,
                        'confidence': min(abs(diff)/4,1)
                    })
        return value_bets

    @staticmethod
    def build_live_training_data(players_df, data_fetcher, min_games=10, last_n_games=20):
        """
        Build a training dataset for each player's last N games (CSV or API).
        """
        training_rows = []
        target_cols = ['PTS','AST','REB','STL','BLK']
        for _, player in players_df.iterrows():
            pid = player['id']
            game_logs = data_fetcher.get_player_stats(pid, last_n_games=last_n_games)
            if game_logs.empty or len(game_logs) < min_games:
                continue

            if 'GAME_DATE' not in game_logs.columns:
                game_logs['GAME_DATE'] = pd.NaT
            else:
                game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'], errors='coerce')
            game_logs.sort_values('GAME_DATE', inplace=True)

            # Ensure numeric
            for c in ['MIN','PTS','AST','REB','STL','BLK']:
                if c in game_logs.columns:
                    game_logs[c] = pd.to_numeric(game_logs[c], errors='coerce').fillna(0)

            # Rolling features
            game_logs['MIN_5_AVG']  = game_logs['MIN'].rolling(window=5, min_periods=1).mean().shift(1)
            game_logs['PTS_5_AVG']  = game_logs['PTS'].rolling(window=5, min_periods=1).mean().shift(1)
            game_logs['AST_5_AVG']  = game_logs['AST'].rolling(window=5, min_periods=1).mean().shift(1)
            game_logs['REB_5_AVG']  = game_logs['REB'].rolling(window=5, min_periods=1).mean().shift(1)

            game_logs['MIN_10_AVG'] = game_logs['MIN'].rolling(window=10, min_periods=1).mean().shift(1)
            game_logs['PTS_10_AVG'] = game_logs['PTS'].rolling(window=10, min_periods=1).mean().shift(1)
            game_logs['AST_10_AVG'] = game_logs['AST'].rolling(window=10, min_periods=1).mean().shift(1)
            game_logs['REB_10_AVG'] = game_logs['REB'].rolling(window=10, min_periods=1).mean().shift(1)

            feature_cols = [
                'MIN_5_AVG','PTS_5_AVG','AST_5_AVG','REB_5_AVG',
                'MIN_10_AVG','PTS_10_AVG','AST_10_AVG','REB_10_AVG'
            ]
            game_logs.dropna(subset=feature_cols, inplace=True)

            for _, row in game_logs.iterrows():
                sample = {fc: row[fc] for fc in feature_cols}
                sample['player_id'] = pid
                sample['player_name'] = player['name']
                for t in target_cols:
                    sample[t] = row[t] if t in row else 0
                training_rows.append(sample)

        if not training_rows:
            st.warning("Not enough historical data for training.")
            return pd.DataFrame(), {}
        training_df = pd.DataFrame(training_rows)
        X = training_df[[
            'MIN_5_AVG','PTS_5_AVG','AST_5_AVG','REB_5_AVG',
            'MIN_10_AVG','PTS_10_AVG','AST_10_AVG','REB_10_AVG'
        ]]
        y_train_dict = {t: training_df[t] for t in target_cols}
        return X, y_train_dict

###############################################################################
# 8) PROCESS A SINGLE GAME
###############################################################################

def process_game(game_data, data_fetcher, processor, predictor, only_starting):
    """
    For a given game, fetch rosters, possibly only top N 'starting' players by MIN,
    then compute predictions for each player.
    """
    home_players = data_fetcher.get_team_players(game_data['HOME_TEAM_ID'])
    away_players = data_fetcher.get_team_players(game_data['VISITOR_TEAM_ID'])

    if only_starting:
        home_players = get_projected_starting_players(home_players, data_fetcher, games=3, n=5)
        away_players = get_projected_starting_players(away_players, data_fetcher, games=3, n=5)

    bets = []
    combined_players = pd.concat([home_players, away_players]).drop_duplicates(subset='id')
    for _, p in combined_players.iterrows():
        p_stats = data_fetcher.get_player_stats(p['id'])
        if p_stats.empty:
            continue
        processed_stats = processor.process_player_stats(p_stats)

        # Determine which side (home/away) for labeling
        if p['id'] in home_players['id'].values:
            team_name = game_data['HOME_TEAM_NAME']
        elif p['id'] in away_players['id'].values:
            team_name = game_data['VISITOR_TEAM_NAME']
        else:
            team_name = "Unknown"

        preds = predictor.predict_props(processed_stats)
        value_bets = predictor.identify_value_bets(preds, None)
        conf = np.mean([preds.get('PTS',0), preds.get('AST',0), preds.get('REB',0)])
        
        bet = {
            'player_name': p['name'],
            'team': team_name,
            'predictions': preds,
            'value_bets': value_bets,
            'confidence': conf
        }
        bets.append(bet)
    return bets

###############################################################################
# 9) MAIN STREAMLIT APP
###############################################################################

def main():
    st.title("🏀 NBA Player Props Predictor")

    # 1) Find next date with NBA games (up to 14 days from now):
    next_game_date = find_next_gameday_with_games(max_days_ahead=14)
    if not next_game_date:
        st.warning("No NBA games in the next 14 days. Try again later!")
        return

    st.info(f"Using date: {next_game_date} for predictions.")
    scoreboard = scoreboardv2.ScoreboardV2(game_date=next_game_date)
    games_df = scoreboard.get_data_frames()[0]
    if games_df.empty:
        st.warning("Unexpected: scoreboard is empty for that date.")
        return

    st.markdown("Select a view mode to display player prop predictions.")
    view_mode = st.radio(
        "View Mode:",
        ["Top Props (Daily)", "Props by Game", "Single Game Analysis"],
        horizontal=True
    )

    use_live_training = st.sidebar.checkbox("Train Ensemble Models (Live)", value=True)
    only_starting = st.sidebar.checkbox("Only Predict for Starting Players", value=True)

    data_fetcher = NBADataFetcher
    processor = DataProcessor()
    predictor = PropPredictor()

    # 2) Optionally build ensemble from today's participants
    if use_live_training:
        st.info("Building training dataset from CSV/API logs...")
        # Collect all unique teams from today's scoreboard
        all_teams_today = pd.concat([
            games_df[['HOME_TEAM_ID']].rename(columns={'HOME_TEAM_ID':'TEAM_ID'}),
            games_df[['VISITOR_TEAM_ID']].rename(columns={'VISITOR_TEAM_ID':'TEAM_ID'})
        ]).drop_duplicates()

        all_players_list = []
        for _, row_ in all_teams_today.iterrows():
            team_p = data_fetcher.get_team_players(row_['TEAM_ID'])
            all_players_list.append(team_p)

        all_players_df = pd.concat(all_players_list, ignore_index=True).drop_duplicates(subset='id')
        X_train, y_train_dict = predictor.build_live_training_data(all_players_df, data_fetcher, min_games=5, last_n_games=25)
        if not X_train.empty:
            st.info(f"Training on {len(X_train)} samples from CSV/API logs...")
            predictor.train_ensemble_models(X_train, y_train_dict)
        else:
            st.warning("Not enough data to train ensemble. Using baseline predictions.")

    # 3) Now show predictions in selected view mode
    if view_mode in ["Top Props (Daily)", "Props by Game"]:
        all_bets = []
        for _, g in games_df.iterrows():
            bet_list = process_game(g, data_fetcher, processor, predictor, only_starting)
            all_bets.extend(bet_list)

        if view_mode == "Top Props (Daily)":
            # Sort all bets by confidence
            all_bets.sort(key=lambda x: x['confidence'], reverse=True)
            st.header("🎯 Top Value Props (Daily)")
            for b in all_bets[:5]:
                display_prop_card(b)
        else:  # Props by Game
            unique_game_ids = games_df['GAME_ID'].unique()
            for game_id in unique_game_ids:
                row_ = games_df[games_df['GAME_ID'] == game_id].iloc[0]
                st.subheader(f"{row_['HOME_TEAM_NAME']} vs {row_['VISITOR_TEAM_NAME']}")
                bet_list = process_game(row_, data_fetcher, processor, predictor, only_starting)
                for b in bet_list:
                    display_prop_card(b)
                st.markdown("---")

    else:  # Single Game Analysis
        game_options = [
            f"{row['HOME_TEAM_NAME']} vs {row['VISITOR_TEAM_NAME']}"
            for _, row in games_df.iterrows()
        ]
        selected_game = st.selectbox("Select Game", options=game_options, key="game_selector")
        if selected_game:
            game_idx = game_options.index(selected_game)
            game_data = games_df.iloc[game_idx]
            bet_list = process_game(game_data, data_fetcher, processor, predictor, only_starting)
            st.header("🎯 Player Props for Selected Game")
            for b in bet_list:
                display_prop_card(b)

    st.markdown("---")
    st.markdown("**Data Source**: CSV & NBA API | Built with Streamlit")

if __name__ == "__main__":
    main()
