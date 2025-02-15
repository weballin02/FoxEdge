import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta

# Official endpoints from your list
from nba_api.stats.endpoints import (
    scoreboardv2, 
    playergamelogs, 
    commonteamroster,
    boxscorematchups 
)
from nba_api.stats.static import teams, players

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K  # For clearing session in Keras

################################################################################
# MATCHUP-BASED UTILITY FUNCTIONS
################################################################################

@st.cache_data(ttl=3600)
def fetch_boxscore_matchupsv3_data(game_id):
    """Fetch advanced matchup data from the boxscorematchups endpoint."""
    try:
        time.sleep(0.6)
        boxscore_obj = boxscorematchups.BoxScoreMatchups(game_id=game_id)
        df = boxscore_obj.get_data_frames()[0]  # Usually the first DataFrame is the relevant one
        return df
    except Exception as e:
        st.error(f"Error fetching matchup data for game {game_id}: {e}")
        return pd.DataFrame()

def aggregate_matchup_data(matchup_df):
    """
    Given the raw matchup DataFrame, group by (gameId, personIdOff)
    to get total partialPossessions, potential assists, etc.
    """
    if matchup_df.empty:
        return pd.DataFrame()

    grouped = matchup_df.groupby(['gameId', 'personIdOff'], as_index=False).agg({
        'partialPossessions': 'sum',
        'matchupPotentialAssists': 'sum'
    })
    grouped.rename(columns={
        'personIdOff': 'PLAYER_ID',
        'partialPossessions': 'MATCHUP_PARTIAL_POSSESSIONS',
        'matchupPotentialAssists': 'MATCHUP_POTENTIAL_AST'
    }, inplace=True)
    return grouped

def merge_matchup_stats(player_logs_df, matchup_agg_df):
    """
    Merge aggregated matchup stats into the player's per-game logs
    using [GAME_ID, PLAYER_ID].
    """
    if player_logs_df.empty or matchup_agg_df.empty:
        return player_logs_df
    merged = pd.merge(
        player_logs_df,
        matchup_agg_df,
        how='left',
        left_on=['GAME_ID', 'PLAYER_ID'],
        right_on=['gameId', 'PLAYER_ID']
    )
    merged.drop(columns=['gameId'], errors='ignore', inplace=True)
    merged['MATCHUP_PARTIAL_POSSESSIONS'] = merged['MATCHUP_PARTIAL_POSSESSIONS'].fillna(0)
    merged['MATCHUP_POTENTIAL_AST'] = merged['MATCHUP_POTENTIAL_AST'].fillna(0)
    return merged

################################################################################
# GENERAL UTILITY
################################################################################

def calculate_rolling_averages(df, columns, windows):
    """Calculate rolling averages for specified columns and windows."""
    result = df.copy()
    for col in columns:
        for window in windows:
            result[f'{col}_{window}_AVG'] = df[col].rolling(window=window, min_periods=1).mean()
    return result

def format_prediction_output(predictions):
    """Format predictions for display on the prop cards."""
    return {
        'Points': f"{predictions.get('PTS', 0):.1f}",
        'Assists': f"{predictions.get('AST', 0):.1f}",
        'Rebounds': f"{predictions.get('REB', 0):.1f}",
        'Steals': f"{predictions.get('STL', 0):.1f}",
        'Blocks': f"{predictions.get('BLK', 0):.1f}"
    }

################################################################################
# UI COMPONENTS
################################################################################

def generate_social_media_post(bet):
    """
    Generate a social media post for a given bet (here, a player prop).
    Expects bet to include 'player_name', 'team', 'predictions', and 'confidence'.
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
    Display a player prop card using a design layout with containers, columns, and expanders.
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
            for col, (prop, value) in zip(cols, formatted_preds.items()):
                col.metric(label=prop, value=value)
    with st.expander("Detailed Insights", expanded=False):
        if bet.get('value_bets'):
            for vb in bet['value_bets']:
                st.markdown(
                    f"**Prop:** {vb.get('prop', '')} | **Suggestion:** {'OVER' if vb.get('value', 0) > 0 else 'UNDER'} | **Confidence:** {vb.get('confidence', 0):.0%}"
                )
        else:
            st.info("No value bets available.")
    with st.expander("Generate Social Media Post", expanded=False):
        if st.button("Generate Post", key=f"post_{bet['player_name']}"):
            post = generate_social_media_post(bet)
            st.code(post, language="markdown")

################################################################################
# DATA PROCESSING & PREDICTION FUNCTIONS
################################################################################

def get_projected_starting_players(players_df, data_fetcher, games=3, n=5):
    """
    For each player in players_df, fetch recent game logs and compute average minutes.
    Returns the top n players by average minutes as a guess for likely starters.
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
    df = pd.DataFrame(players_with_minutes)
    df = df.sort_values(by='avg_min', ascending=False)
    return df.head(n)

class DataProcessor:
    def __init__(self):
        self.stats_columns = ['PTS', 'AST', 'REB', 'STL', 'BLK']

    def process_player_stats(self, player_stats_df):
        """
        Process player stats by computing rolling averages for MIN and key stats,
        plus matchup-based columns if they exist.
        """
        if player_stats_df.empty:
            return pd.DataFrame()

        base_columns = ['MIN'] + self.stats_columns
        processed_stats = calculate_rolling_averages(player_stats_df, base_columns, [5, 10])

        # If matchup columns exist, add them to rolling
        if 'MATCHUP_PARTIAL_POSSESSIONS' in player_stats_df.columns:
            processed_stats = calculate_rolling_averages(
                processed_stats, 
                ['MATCHUP_PARTIAL_POSSESSIONS', 'MATCHUP_POTENTIAL_AST'], 
                [5, 10]
            )

        # Per-minute for PTS, AST, etc.
        for stat in self.stats_columns:
            processed_stats[f'{stat}_PER_MIN'] = (
                processed_stats[stat] / processed_stats['MIN']
            ).fillna(0)

        return processed_stats

    def adjust_for_opponent(self, player_stats, team_stats):
        """(Optional) Adjust stats based on opponent metrics (not used)."""
        return player_stats

class NBADataFetcher:
    @st.cache_data(ttl=3600)
    def get_team_name(team_id):
        try:
            nba_teams = teams.get_teams()
            team_info = next((team for team in nba_teams if team['id'] == team_id), None)
            return team_info['full_name'] if team_info else f"Team {team_id}"
        except Exception as e:
            st.error(f"Error fetching team name: {e}")
            return f"Team {team_id}"

    @st.cache_data(ttl=3600)
    def get_team_players(team_id):
        """Fetch a given team's roster from commonteamroster."""
        try:
            time.sleep(0.6)
            roster = commonteamroster.CommonTeamRoster(team_id=team_id)
            roster_df = roster.get_data_frames()[0]
            if not roster_df.empty:
                players_data = [{
                    'id': str(row['PLAYER_ID']),
                    'name': str(row['PLAYER'])
                } for _, row in roster_df.iterrows()]
                return pd.DataFrame(players_data)
            return pd.DataFrame([{'id': '1', 'name': 'Player Data Unavailable'}])
        except Exception as e:
            st.error(f"Error fetching team players: {e}")
            return pd.DataFrame(columns=['id', 'name'])

    @st.cache_data(ttl=3600)
    def get_upcoming_games(next_n_days=7):
        """
        Collect NBA schedules from scoreboardv2 for the next next_n_days (including today).
        Returns a DataFrame with all games for that timespan.
        """
        all_games = []
        for offset in range(next_n_days+1):  # +1 so we include day=7
            query_date = (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")
            try:
                time.sleep(0.6)
                sb = scoreboardv2.ScoreboardV2(game_date=query_date)
                df = sb.get_data_frames()[0]
                if not df.empty:
                    df['HOME_TEAM_NAME'] = df['HOME_TEAM_ID'].apply(NBADataFetcher.get_team_name)
                    df['VISITOR_TEAM_NAME'] = df['VISITOR_TEAM_ID'].apply(NBADataFetcher.get_team_name)
                    df['SCHEDULE_DATE'] = query_date  # Tag with the date for filtering
                    all_games.append(df)
            except Exception as e:
                st.error(f"Error fetching scoreboard for {query_date}: {e}")

        if all_games:
            combined = pd.concat(all_games, ignore_index=True)
            return combined[[
                'GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
                'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME', 'SCHEDULE_DATE'
            ]]
        else:
            return pd.DataFrame(columns=[
                'GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
                'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME', 'SCHEDULE_DATE'
            ])

    @st.cache_data(ttl=3600)
    def get_player_stats(player_id, last_n_games=10):
        """
        Fetch last_n_games for the given player ID.
        Merge advanced matchup data from boxscorematchups for each game.
        """
        try:
            time.sleep(0.6)
            logs = playergamelogs.PlayerGameLogs(
                player_id_nullable=player_id,
                last_n_games_nullable=last_n_games
            ).get_data_frames()[0]

            if logs.empty:
                return pd.DataFrame()

            # Convert numeric columns
            numeric_cols = ['MIN','PTS','AST','REB','STL','BLK']
            for c in numeric_cols:
                if c in logs.columns:
                    logs[c] = pd.to_numeric(logs[c], errors='coerce')

            # For each game in logs, fetch advanced matchups and merge
            unique_games = logs['GAME_ID'].unique()
            all_matchup_aggregations = []
            for g_id in unique_games:
                matchup_raw = fetch_boxscore_matchupsv3_data(g_id)
                if matchup_raw.empty:
                    continue
                matchup_agg = aggregate_matchup_data(matchup_raw)
                all_matchup_aggregations.append(matchup_agg)

            if all_matchup_aggregations:
                combined_matchup_df = pd.concat(all_matchup_aggregations, ignore_index=True)
                merged = merge_matchup_stats(logs, combined_matchup_df)
                return merged

            return logs
        
        except Exception as e:
            st.error(f"Error fetching player stats for player {player_id}: {e}")
            return pd.DataFrame()

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
        """
        Include advanced matchup-based columns if present (rolled).
        """
        base_feats = [
            'MIN_5_AVG', 'PTS_5_AVG', 'AST_5_AVG', 'REB_5_AVG',
            'MIN_10_AVG', 'PTS_10_AVG', 'AST_10_AVG', 'REB_10_AVG'
        ]
        adv_feats = [
            'MATCHUP_PARTIAL_POSSESSIONS_5_AVG', 'MATCHUP_PARTIAL_POSSESSIONS_10_AVG',
            'MATCHUP_POTENTIAL_AST_5_AVG', 'MATCHUP_POTENTIAL_AST_10_AVG'
        ]
        feature_cols = base_feats + adv_feats
        for col in feature_cols:
            if col not in player_stats.columns:
                player_stats[col] = 0
        return player_stats[feature_cols].fillna(0)

    @staticmethod
    @st.cache_data
    def _cached_predict(_player_stats):
        """Fallback if no ensemble model is trained."""
        try:
            if _player_stats.empty:
                return pd.Series({'PTS': 0, 'AST': 0, 'REB': 0, 'STL': 0, 'BLK': 0})
            predictions = {}
            for prop in ['PTS', 'AST', 'REB', 'STL', 'BLK']:
                col_name = f'{prop}_10_AVG'
                base_value = _player_stats[col_name].mean() if col_name in _player_stats.columns else 0
                variation = np.random.normal(0, max(1, base_value * 0.1))
                predictions[prop] = max(0, base_value + variation)
            return pd.Series(predictions)
        except Exception as e:
            st.error(f"Error in fallback predict: {e}")
            return pd.Series({'PTS': 0, 'AST': 0, 'REB': 0, 'STL': 0, 'BLK': 0})

    def train_ensemble_models(self, X_train, y_train_dict):
        K.clear_session()  # Clear previous Keras session.
        self.ensemble_models = {}

        for prop in y_train_dict.keys():
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_train, y_train_dict[prop])
            gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gbm_model.fit(X_train, y_train_dict[prop])
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

        st.success("Ensemble models trained successfully on live data.")

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
            return pd.Series({'PTS': 0, 'AST': 0, 'REB': 0, 'STL': 0, 'BLK': 0})
        if self.ensemble_models:
            return self.predict_ensemble(player_stats)
        return self._cached_predict(player_stats)

    def identify_value_bets(self, predictions, sportsbook_lines):
        """
        Compare predictions to a mock line. If difference > 2, it's a 'value bet.'
        """
        try:
            value_bets = []
            mock_lines = {
                'PTS': predictions.get('PTS', 0) + np.random.normal(0, 2),
                'AST': predictions.get('AST', 0) + np.random.normal(0, 1),
                'REB': predictions.get('REB', 0) + np.random.normal(0, 1)
            }
            for prop, pred in predictions.items():
                if prop in mock_lines:
                    diff = pred - mock_lines[prop]
                    if abs(diff) > 2:
                        value_bets.append({
                            'prop': prop,
                            'prediction': pred,
                            'line': mock_lines[prop],
                            'value': diff,
                            'confidence': min(abs(diff) / 4, 1)
                        })
            return value_bets
        except Exception as e:
            st.error(f"Error identifying value bets: {e}")
            return []

    @staticmethod
    def build_live_training_data(players_df, data_fetcher, min_games=10, last_n_games=20):
        """
        Build training data from the next_n_days pool, 
        including advanced matchup columns if present.
        """
        training_rows = []
        target_cols = ['PTS', 'AST', 'REB', 'STL', 'BLK']

        for _, player in players_df.iterrows():
            p_id = player['id']
            logs = data_fetcher.get_player_stats(p_id, last_n_games=last_n_games)
            if logs.empty or len(logs) < min_games:
                continue

            try:
                logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
                logs = logs.sort_values('GAME_DATE')
            except Exception as e:
                st.error(f"Error processing logs for {player['name']}: {e}")
                continue

            for col in ['MIN','PTS','AST','REB','STL','BLK','MATCHUP_PARTIAL_POSSESSIONS','MATCHUP_POTENTIAL_AST']:
                if col in logs.columns:
                    logs[col] = pd.to_numeric(logs[col], errors='coerce').fillna(0)

            # Rolling
            for base_col in ['MIN','PTS','AST','REB']:
                logs[f'{base_col}_5_AVG']  = logs[base_col].rolling(window=5, min_periods=1).mean().shift(1)
                logs[f'{base_col}_10_AVG'] = logs[base_col].rolling(window=10, min_periods=1).mean().shift(1)

            if 'MATCHUP_PARTIAL_POSSESSIONS' in logs.columns:
                logs['MATCHUP_PARTIAL_POSSESSIONS_5_AVG'] = (
                    logs['MATCHUP_PARTIAL_POSSESSIONS'].rolling(window=5, min_periods=1).mean().shift(1)
                )
                logs['MATCHUP_PARTIAL_POSSESSIONS_10_AVG'] = (
                    logs['MATCHUP_PARTIAL_POSSESSIONS'].rolling(window=10, min_periods=1).mean().shift(1)
                )
                logs['MATCHUP_POTENTIAL_AST_5_AVG'] = (
                    logs['MATCHUP_POTENTIAL_AST'].rolling(window=5, min_periods=1).mean().shift(1)
                )
                logs['MATCHUP_POTENTIAL_AST_10_AVG'] = (
                    logs['MATCHUP_POTENTIAL_AST'].rolling(window=10, min_periods=1).mean().shift(1)
                )

            feature_cols = [
                'MIN_5_AVG','PTS_5_AVG','AST_5_AVG','REB_5_AVG',
                'MIN_10_AVG','PTS_10_AVG','AST_10_AVG','REB_10_AVG',
                'MATCHUP_PARTIAL_POSSESSIONS_5_AVG','MATCHUP_PARTIAL_POSSESSIONS_10_AVG',
                'MATCHUP_POTENTIAL_AST_5_AVG','MATCHUP_POTENTIAL_AST_10_AVG'
            ]
            logs = logs.dropna(subset=feature_cols, how='all')

            for idx, row in logs.iterrows():
                sample = {col: row.get(col, 0) for col in feature_cols}
                sample['player_id'] = p_id
                sample['player_name'] = player['name']
                for t_col in target_cols:
                    sample[t_col] = row.get(t_col, 0)
                training_rows.append(sample)

        if not training_rows:
            st.warning("Not enough historical data for training.")
            return pd.DataFrame(), {}
        training_df = pd.DataFrame(training_rows)
        X = training_df[feature_cols]
        y_train_dict = {t: training_df[t] for t in target_cols}
        return X, y_train_dict

################################################################################
# HELPER FUNCTION TO PROCESS A SINGLE GAME'S PREDICTIONS
################################################################################

def process_game(game_data, data_fetcher, processor, predictor, only_starting):
    """
    For a given game, fetch team rosters (filtered to likely starters if desired)
    and compute predictions for each player.
    """
    home_players = data_fetcher.get_team_players(game_data['HOME_TEAM_ID'])
    away_players = data_fetcher.get_team_players(game_data['VISITOR_TEAM_ID'])

    if only_starting:
        home_players = get_projected_starting_players(home_players, data_fetcher, games=3, n=5)
        away_players = get_projected_starting_players(away_players, data_fetcher, games=3, n=5)

    bets = []
    for player in pd.concat([home_players, away_players]).drop_duplicates(subset='id').itertuples(index=False):
        player_stats = data_fetcher.get_player_stats(player.id)
        if player_stats.empty:
            continue

        processed_stats = processor.process_player_stats(player_stats)
        preds = predictor.predict_props(processed_stats)
        value_bets = predictor.identify_value_bets(preds, None)
        confidence = np.mean([preds.get('PTS', 0), preds.get('AST', 0), preds.get('REB', 0)])

        if player.id in home_players['id'].values:
            team_name = game_data['HOME_TEAM_NAME']
        elif player.id in away_players['id'].values:
            team_name = game_data['VISITOR_TEAM_NAME']
        else:
            team_name = "Unknown"

        bets.append({
            'player_name': player.name,
            'team': team_name,
            'predictions': preds,
            'value_bets': value_bets,
            'confidence': confidence
        })
    return bets

################################################################################
# MAIN STREAMLIT APP
################################################################################

def main():
    st.set_page_config(
        page_title="NBA Player Props Predictor (Next 7 Days)", 
        page_icon="🏀",
        layout="wide"
    )
    st.title("🏀 NBA Player Props Predictor - Next 7 Days")
    st.markdown("This version fetches upcoming NBA schedules for the next 7 days, "\
                "including advanced matchup data from boxscorematchups.")

    data_fetcher = NBADataFetcher
    processor = DataProcessor()
    predictor = PropPredictor()

    # Gather games for next 7 days
    upcoming_games_df = data_fetcher.get_upcoming_games(next_n_days=7)
    if upcoming_games_df.empty:
        st.warning("No games found in the next 7 days or error fetching data.")
        return

    # Let the user pick which date to analyze
    unique_dates = sorted(upcoming_games_df['SCHEDULE_DATE'].unique())
    selected_date = st.selectbox(
        "Select a date from the next 7 days:",
        options=unique_dates
    )

    # Filter games to only those on the chosen date
    date_games = upcoming_games_df[upcoming_games_df['SCHEDULE_DATE'] == selected_date].copy()
    if date_games.empty:
        st.warning(f"No games found for {selected_date}.")
        return

    # Provide same radio-based view modes
    view_mode = st.radio(
        "View Mode:",
        options=["Please select a view option", "Top Props (Daily)", "Props by Game", "Single Game Analysis"],
        horizontal=True
    )
    if view_mode == "Please select a view option":
        st.info("Please select a valid view mode to begin.")
        return

    # Sidebar toggles
    use_live_training = st.sidebar.checkbox("Train Ensemble Models (Live)", value=True)
    only_starting = st.sidebar.checkbox("Only Predict for Likely Starters", value=True)

    # Optionally train ensemble with all players in these date_games
    if use_live_training:
        all_player_ids = []
        for _, game in date_games.iterrows():
            home_df = data_fetcher.get_team_players(game['HOME_TEAM_ID'])
            away_df = data_fetcher.get_team_players(game['VISITOR_TEAM_ID'])
            combined = pd.concat([home_df, away_df]).drop_duplicates(subset='id')
            all_player_ids.append(combined)
        players_df = pd.concat(all_player_ids).drop_duplicates(subset='id')

        X_train, y_train_dict = predictor.build_live_training_data(players_df, data_fetcher)
        if not X_train.empty:
            predictor.train_ensemble_models(X_train, y_train_dict)

    # Now proceed with the user’s chosen view mode
    if view_mode in ["Top Props (Daily)", "Props by Game"]:
        all_bets = []
        for _, game in date_games.iterrows():
            bets = process_game(game, data_fetcher, processor, predictor, only_starting)
            all_bets.extend(bets)

        if view_mode == "Top Props (Daily)":
            all_bets.sort(key=lambda x: x['confidence'], reverse=True)
            st.header(f"🎯 Top Value Props for {selected_date}")
            for bet in all_bets[:5]:
                display_prop_card(bet)

        elif view_mode == "Props by Game":
            for _, game in date_games.iterrows():
                st.subheader(f"{game['HOME_TEAM_NAME']} vs {game['VISITOR_TEAM_NAME']} " \
                             f"on {selected_date}")
                bets = process_game(game, data_fetcher, processor, predictor, only_starting)
                for bet in bets:
                    display_prop_card(bet)
                st.markdown("---")

    elif view_mode == "Single Game Analysis":
        game_options = [f"{row['HOME_TEAM_NAME']} vs {row['VISITOR_TEAM_NAME']}" 
                        for _, row in date_games.iterrows()]
        selected_game = st.selectbox("Select Game", options=game_options, key="game_selector")
        if selected_game:
            game_idx = game_options.index(selected_game)
            game_data = date_games.iloc[game_idx]
            bets = process_game(game_data, data_fetcher, processor, predictor, only_starting)
            st.header(f"🎯 Player Props for {selected_game} on {selected_date}")
            for bet in bets:
                display_prop_card(bet)

    st.markdown("---")
    st.markdown("Data from NBA API | Built with Streamlit | Next 7 Days Schedule")

if __name__ == "__main__":
    main()
