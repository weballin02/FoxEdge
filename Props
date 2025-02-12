import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2, playergamelogs, teamgamelogs, commonteamroster
from nba_api.stats.static import teams, players
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ------------------------------------------------------------------------------
# Helper Function to Determine Projected Starting Players
# ------------------------------------------------------------------------------
def get_projected_starting_players(players_df, data_fetcher, games=3, n=5):
    """
    For each player in players_df, fetch recent game logs (last `games` games)
    and compute average minutes. Then, select the top `n` players with highest
    average minutes.
    
    :param players_df: DataFrame containing team roster (must include 'id' and 'name').
    :param data_fetcher: Reference to NBADataFetcher class.
    :param games: Number of recent games to use.
    :param n: Number of projected starters to return.
    :return: DataFrame limited to top n players.
    """
    players_with_minutes = []
    for _, player in players_df.iterrows():
        stats = data_fetcher.get_player_stats(player['id'], last_n_games=games)
        if stats.empty:
            avg_min = 0
        else:
            # Ensure MIN is numeric and compute average minutes
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

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def calculate_rolling_averages(df, columns, windows):
    """Calculate rolling averages for specified columns and windows."""
    result = df.copy()
    for col in columns:
        for window in windows:
            result[f'{col}_{window}_AVG'] = df[col].rolling(window=window, min_periods=1).mean()
    return result

def format_prediction_output(predictions):
    """Format predictions for display."""
    return {
        'Points': f"{predictions['PTS']:.1f}",
        'Assists': f"{predictions['AST']:.1f}",
        'Rebounds': f"{predictions['REB']:.1f}",
        'Steals': f"{predictions['STL']:.1f}",
        'Blocks': f"{predictions['BLK']:.1f}"
    }

def create_value_bet_message(bet):
    """Create formatted message for value bet display."""
    direction = 'OVER' if bet['value'] > 0 else 'UNDER'
    return (
        f"{bet['prop']} {direction} {bet['line']:.1f}\n"
        f"Prediction: {bet['prediction']:.1f}\n"
        f"Confidence: {bet['confidence']:.2%}"
    )

# ------------------------------------------------------------------------------
# Data Processor Class
# ------------------------------------------------------------------------------
class DataProcessor:
    def __init__(self):
        self.stats_columns = ['PTS', 'AST', 'REB', 'STL', 'BLK']

    def process_player_stats(self, player_stats_df):
        """Process player statistics."""
        if player_stats_df.empty:
            return pd.DataFrame()
        processed_stats = calculate_rolling_averages(player_stats_df, self.stats_columns, [5, 10])
        for stat in self.stats_columns:
            processed_stats[f'{stat}_PER_MIN'] = (processed_stats[stat] / processed_stats['MIN']).fillna(0)
        return processed_stats

    def adjust_for_opponent(self, player_stats, team_stats):
        """Adjust player stats based on team defense rating."""
        if player_stats.empty or team_stats.empty:
            return player_stats
        if 'PTS' in team_stats.columns:
            points_allowed = team_stats['PTS'].mean()
            league_avg_points = 110.0
            defense_factor = league_avg_points / points_allowed if points_allowed > 0 else 1.0
            for stat in self.stats_columns:
                player_stats[f'{stat}_ADJUSTED'] = (player_stats[stat] * defense_factor).fillna(player_stats[stat])
        return player_stats

# ------------------------------------------------------------------------------
# NBA Data Fetcher Class
# ------------------------------------------------------------------------------
class NBADataFetcher:
    @st.cache_data(ttl=3600)
    def get_team_name(team_id):
        """Get team name from team ID."""
        try:
            nba_teams = teams.get_teams()
            team_info = next((team for team in nba_teams if team['id'] == team_id), None)
            return team_info['full_name'] if team_info else f"Team {team_id}"
        except Exception as e:
            st.error(f"Error fetching team name: {str(e)}")
            return f"Team {team_id}"

    @st.cache_data(ttl=3600)
    def get_team_players(team_id):
        """Get active players for a team."""
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
            st.error(f"Error fetching team players: {str(e)}")
            return pd.DataFrame(columns=['id', 'name'])

    @st.cache_data(ttl=3600)
    def get_todays_games():
        """Fetch today's NBA games."""
        try:
            time.sleep(0.6)
            scoreboard = scoreboardv2.ScoreboardV2(game_date=datetime.now().strftime("%Y-%m-%d"))
            games_df = scoreboard.get_data_frames()[0]
            if not games_df.empty:
                games_df['HOME_TEAM_NAME'] = games_df['HOME_TEAM_ID'].apply(NBADataFetcher.get_team_name)
                games_df['VISITOR_TEAM_NAME'] = games_df['VISITOR_TEAM_ID'].apply(NBADataFetcher.get_team_name)
                return games_df[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 
                                 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME']]
            return pd.DataFrame(columns=['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 
                                         'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME'])
        except Exception as e:
            st.error(f"Error fetching games: {str(e)}")
            return pd.DataFrame(columns=['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 
                                         'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME'])

    @st.cache_data(ttl=3600)
    def get_player_stats(player_id, last_n_games=10):
        """Fetch player game logs."""
        try:
            time.sleep(0.6)
            player_logs = playergamelogs.PlayerGameLogs(
                player_id_nullable=player_id,
                last_n_games_nullable=last_n_games
            )
            stats_df = player_logs.get_data_frames()[0]
            if not stats_df.empty:
                return stats_df
            return pd.DataFrame(columns=['GAME_DATE', 'MIN', 'PTS', 'AST', 'REB', 'STL', 'BLK'])
        except Exception as e:
            st.error(f"Error fetching player stats: {str(e)}")
            return pd.DataFrame(columns=['GAME_DATE', 'MIN', 'PTS', 'AST', 'REB', 'STL', 'BLK'])

    @st.cache_data(ttl=3600)
    def get_team_stats(team_id, last_n_games=10):
        """Fetch team game logs."""
        try:
            time.sleep(0.6)
            team_logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                last_n_games_nullable=last_n_games
            )
            team_df = team_logs.get_data_frames()[0]
            if not team_df.empty:
                return team_df
            return pd.DataFrame(columns=['GAME_DATE', 'PTS', 'REB', 'AST'])
        except Exception as e:
            st.error(f"Error fetching team stats: {str(e)}")
            return pd.DataFrame(columns=['GAME_DATE', 'PTS', 'REB', 'AST'])

# ------------------------------------------------------------------------------
# Prop Predictor Class with Ensemble Implementation and Live Training
# ------------------------------------------------------------------------------
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
        Prepare features for prediction.
        Uses a simplified feature set.
        """
        try:
            feature_cols = [
                'MIN_5_AVG', 'PTS_5_AVG', 'AST_5_AVG', 'REB_5_AVG',
                'MIN_10_AVG', 'PTS_10_AVG', 'AST_10_AVG', 'REB_10_AVG'
            ]
            missing_cols = [col for col in feature_cols if col not in player_stats.columns]
            if missing_cols:
                st.warning(f"Missing columns for features: {missing_cols}")
                for col in missing_cols:
                    player_stats[col] = 0
            return player_stats[feature_cols].fillna(0)
        except Exception as e:
            st.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame(columns=feature_cols)

    @staticmethod
    @st.cache_data
    def _cached_predict(_player_stats):
        """Static cached prediction method as a fallback."""
        try:
            if _player_stats.empty:
                return pd.Series({'PTS': 0, 'AST': 0, 'REB': 0, 'STL': 0, 'BLK': 0})
            predictions = {}
            for prop in ['PTS', 'AST', 'REB', 'STL', 'BLK']:
                col_name = f'{prop}_10_AVG'
                base_value = _player_stats[col_name].mean() if col_name in _player_stats.columns else _player_stats[prop].mean() if prop in _player_stats.columns else 0
                variation = np.random.normal(0, max(1, base_value * 0.1))
                predictions[prop] = max(0, base_value + variation)
            return pd.Series(predictions)
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return pd.Series({'PTS': 0, 'AST': 0, 'REB': 0, 'STL': 0, 'BLK': 0})

    def train_ensemble_models(self, X_train, y_train_dict):
        """
        Train ensemble models for each target prop using XGBoost, GradientBoosting, and a Neural Network.
        :param X_train: DataFrame of features.
        :param y_train_dict: Dictionary with target names as keys.
        """
        self.ensemble_models = {}
        for prop in y_train_dict.keys():
            # XGBoost model
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_train, y_train_dict[prop])
            # Gradient Boosting model
            gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gbm_model.fit(X_train, y_train_dict[prop])
            # Neural Network model using Keras
            nn_model = Sequential()
            nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
            nn_model.add(Dense(32, activation='relu'))
            nn_model.add(Dense(1, activation='linear'))
            nn_model.compile(loss='mse', optimizer='adam')
            nn_model.fit(X_train, y_train_dict[prop], epochs=50, batch_size=8, verbose=0)
            self.ensemble_models[prop] = {'xgb': xgb_model, 'gbm': gbm_model, 'nn': nn_model}
        st.success("Ensemble models trained successfully on live data.")

    def predict_ensemble(self, player_stats):
        """
        Predict player props using the ensemble models.
        :param player_stats: DataFrame of player stats.
        :return: pd.Series of ensemble predictions.
        """
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
        """
        Generate predictions for player props.
        Uses ensemble predictions if available; otherwise falls back to a cached method.
        :param player_stats: DataFrame of player stats.
        :return: pd.Series of predictions.
        """
        if player_stats is None or player_stats.empty:
            return pd.Series({'PTS': 0, 'AST': 0, 'REB': 0, 'STL': 0, 'BLK': 0})
        if self.ensemble_models:
            return self.predict_ensemble(player_stats)
        return self._cached_predict(player_stats)

    def identify_value_bets(self, predictions, sportsbook_lines):
        """Compare predictions to sportsbook lines."""
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
            st.error(f"Error identifying value bets: {str(e)}")
            return []

    @staticmethod
    def build_live_training_data(players_df, data_fetcher, min_games=10, last_n_games=20):
        """
        Build training data from live NBA data using historical game logs for players in players_df.
        :param players_df: DataFrame of players (must include 'id' and 'name').
        :param data_fetcher: Reference to NBADataFetcher.
        :param min_games: Minimum games required.
        :param last_n_games: Number of historical games to fetch.
        :return: (X, y_train_dict)
        """
        training_rows = []
        target_cols = ['PTS', 'AST', 'REB', 'STL', 'BLK']
        for _, player in players_df.iterrows():
            player_id = player['id']
            game_logs = data_fetcher.get_player_stats(player_id, last_n_games=last_n_games)
            if game_logs.empty or len(game_logs) < min_games:
                continue
            try:
                game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'])
                game_logs = game_logs.sort_values('GAME_DATE')
            except Exception as e:
                st.error(f"Error processing game logs for {player['name']}: {e}")
                continue
            for col in ['MIN', 'PTS', 'AST', 'REB']:
                game_logs[col] = pd.to_numeric(game_logs[col], errors='coerce')
            game_logs['MIN_5_AVG'] = game_logs['MIN'].rolling(window=5, min_periods=1).mean().shift(1)
            game_logs['PTS_5_AVG'] = game_logs['PTS'].rolling(window=5, min_periods=1).mean().shift(1)
            game_logs['AST_5_AVG'] = game_logs['AST'].rolling(window=5, min_periods=1).mean().shift(1)
            game_logs['REB_5_AVG'] = game_logs['REB'].rolling(window=5, min_periods=1).mean().shift(1)
            game_logs['MIN_10_AVG'] = game_logs['MIN'].rolling(window=10, min_periods=1).mean().shift(1)
            game_logs['PTS_10_AVG'] = game_logs['PTS'].rolling(window=10, min_periods=1).mean().shift(1)
            game_logs['AST_10_AVG'] = game_logs['AST'].rolling(window=10, min_periods=1).mean().shift(1)
            game_logs['REB_10_AVG'] = game_logs['REB'].rolling(window=10, min_periods=1).mean().shift(1)
            feature_cols = ['MIN_5_AVG', 'PTS_5_AVG', 'AST_5_AVG', 'REB_5_AVG',
                            'MIN_10_AVG', 'PTS_10_AVG', 'AST_10_AVG', 'REB_10_AVG']
            game_logs = game_logs.dropna(subset=feature_cols)
            for idx, row in game_logs.iterrows():
                sample = {col: row[col] for col in feature_cols}
                sample['player_id'] = player_id
                sample['player_name'] = player['name']
                for target in target_cols:
                    sample[target] = row[target]
                training_rows.append(sample)
        if not training_rows:
            st.warning("Not enough historical data available for training.")
            return pd.DataFrame(), {}
        training_df = pd.DataFrame(training_rows)
        X = training_df[feature_cols]
        y_train_dict = {target: training_df[target] for target in target_cols}
        return X, y_train_dict

# ------------------------------------------------------------------------------
# Main Streamlit App
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="NBA Player Props Predictor",
        page_icon="🏀",
        layout="wide"
    )

    data_fetcher = NBADataFetcher
    processor = DataProcessor()
    predictor = PropPredictor()

    st.title("🏀 NBA Player Props Predictor")
    st.markdown("Select a game to view player prop predictions and value bets.")

    # Sidebar options for live training and limiting to starting players.
    # Both options are auto-selected upon app load.
    use_live_training = st.sidebar.checkbox("Train Ensemble Models (Live)", value=True)
    only_starting = st.sidebar.checkbox("Only Predict for Starting Players", value=True)

    games_df = data_fetcher.get_todays_games()
    if not games_df.empty:
        game_options = [
            f"{row['HOME_TEAM_NAME']} vs {row['VISITOR_TEAM_NAME']}"
            for _, row in games_df.iterrows()
        ]
        selected_game = st.selectbox("Select Game", options=game_options, key="game_selector")
        if selected_game:
            game_idx = game_options.index(selected_game)
            game_data = games_df.iloc[game_idx]
            home_stats = data_fetcher.get_team_stats(game_data['HOME_TEAM_ID'])
            away_stats = data_fetcher.get_team_stats(game_data['VISITOR_TEAM_ID'])
            home_players = data_fetcher.get_team_players(game_data['HOME_TEAM_ID'])
            away_players = data_fetcher.get_team_players(game_data['VISITOR_TEAM_ID'])

            # If only predicting for projected starters, limit to top 5 by recent minutes.
            if only_starting:
                home_players = get_projected_starting_players(home_players, data_fetcher, games=3, n=5)
                away_players = get_projected_starting_players(away_players, data_fetcher, games=3, n=5)

            # Optionally train ensemble models using live historical data.
            if use_live_training:
                all_players = pd.concat([home_players, away_players]).drop_duplicates(subset='id')
                st.info("Building live training data from historical game logs. This may take a while...")
                X_train, y_train_dict = PropPredictor.build_live_training_data(all_players, data_fetcher, min_games=10, last_n_games=20)
                if not X_train.empty and y_train_dict:
                    predictor.train_ensemble_models(X_train, y_train_dict)
                else:
                    st.error("Live training data could not be built. Using fallback predictions.")

            all_predictions = []
            all_value_bets = []
            # Process home team players.
            for _, player in home_players.iterrows():
                player_stats = data_fetcher.get_player_stats(player['id'])
                if not player_stats.empty:
                    processed_stats = processor.process_player_stats(player_stats)
                    adjusted_stats = processor.adjust_for_opponent(processed_stats, away_stats)
                    predictions = predictor.predict_props(adjusted_stats)
                    value_bets = predictor.identify_value_bets(predictions, None)
                    for bet in value_bets:
                        bet['player_name'] = player['name']
                    all_value_bets.extend(value_bets)
                    all_predictions.append({
                        'player_name': player['name'],
                        'team': game_data['HOME_TEAM_NAME'],
                        'predictions': predictions
                    })
            # Process away team players.
            for _, player in away_players.iterrows():
                player_stats = data_fetcher.get_player_stats(player['id'])
                if not player_stats.empty:
                    processed_stats = processor.process_player_stats(player_stats)
                    adjusted_stats = processor.adjust_for_opponent(processed_stats, home_stats)
                    predictions = predictor.predict_props(adjusted_stats)
                    value_bets = predictor.identify_value_bets(predictions, None)
                    for bet in value_bets:
                        bet['player_name'] = player['name']
                    all_value_bets.extend(value_bets)
                    all_predictions.append({
                        'player_name': player['name'],
                        'team': game_data['VISITOR_TEAM_NAME'],
                        'predictions': predictions
                    })

            all_value_bets.sort(key=lambda x: x['confidence'], reverse=True)
            st.header("🎯 Top Value Props")
            for i, bet in enumerate(all_value_bets[:5]):
                st.markdown(f"""
                <div class='top-prop'>
                    <div class='prop-header'>
                        {i+1}. {bet['player_name']} - {bet['prop']}
                    </div>
                    <div class='value-indicator'>
                        {'OVER' if bet['value'] > 0 else 'UNDER'} {bet['line']:.1f} 
                        (Prediction: {bet['prediction']:.1f})
                    </div>
                    <div>
                        Confidence: {bet['confidence']:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("View All Player Props"):
                home_tab, away_tab = st.tabs([
                    f"🏠 {game_data['HOME_TEAM_NAME']}", 
                    f"🏃 {game_data['VISITOR_TEAM_NAME']}"
                ])
                with home_tab:
                    for pred in [p for p in all_predictions if p['team'] == game_data['HOME_TEAM_NAME']]:
                        st.subheader(pred['player_name'])
                        formatted_preds = format_prediction_output(pred['predictions'])
                        cols = st.columns(len(formatted_preds))
                        for col, (prop, value) in zip(cols, formatted_preds.items()):
                            col.metric(label=prop, value=value)
                with away_tab:
                    for pred in [p for p in all_predictions if p['team'] == game_data['VISITOR_TEAM_NAME']]:
                        st.subheader(pred['player_name'])
                        formatted_preds = format_prediction_output(pred['predictions'])
                        cols = st.columns(len(formatted_preds))
                        for col, (prop, value) in zip(cols, formatted_preds.items()):
                            col.metric(label=prop, value=value)
    else:
        st.warning("No games scheduled for today or error fetching game data.")

    st.markdown("---")
    st.markdown("Data provided by NBA API | Built with Streamlit")

if __name__ == "__main__":
    main()
