#!/usr/bin/env python
"""
train.py
--------
Headless script that fetches sports data, trains hybrid models 
(StackingRegressor + Auto-ARIMA) for NFL, NBA, and NCAAB,
and saves them into the "models/" folder.
No UI or user login code is included.
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

import nfl_data_py
from nba_api.stats.endpoints import TeamGameLog
from nba_api.stats.static import teams as nba_teams
import cbbpy.mens_scraper as cbb

from pmdarima import auto_arima
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

#########################
# HELPER FUNCTIONS
#########################
def round_half(number):
    """Rounds a number to the nearest 0.5."""
    return round(number * 2) / 2

def tune_model(model, param_grid, X_train, y_train):
    """
    Tunes a given model using GridSearchCV with TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

#########################
# DATA LOADING FUNCTIONS
#########################
def load_nfl_data():
    current_year = datetime.now().year
    years = [current_year - i for i in range(12)]
    schedule = nfl_data_py.import_schedules(years)
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    schedule.sort_values("gameday", inplace=True)
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
    data["season_avg"] = data.groupby("team")["score"].transform(lambda x: x.expanding().mean())
    data["weighted_avg"] = data["rolling_avg"] * 0.6 + data["season_avg"] * 0.4
    return data

def load_nba_data():
    nba_teams_list = nba_teams.get_teams()
    seasons = ['2017-18','2018-19','2019-20','2020-21','2021-22','2022-23','2023-24','2024-25']
    all_rows = []
    for season in seasons:
        for team in nba_teams_list:
            team_id = team['id']
            team_abbrev = team.get('abbreviation', str(team_id))
            try:
                gl = TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
                if gl.empty:
                    continue
                gl['GAME_DATE'] = pd.to_datetime(gl['GAME_DATE'])
                gl.sort_values('GAME_DATE', inplace=True)
                needed = ['PTS','FGA','FTA','TOV','OREB','PTS_OPP']
                for c in needed:
                    if c not in gl.columns:
                        gl[c] = 0
                    gl[c] = pd.to_numeric(gl[c], errors='coerce').fillna(0)
                gl['TEAM_POSSESSIONS'] = gl['FGA'] + 0.44 * gl['FTA'] + gl['TOV'] - gl['OREB']
                gl['TEAM_POSSESSIONS'] = gl['TEAM_POSSESSIONS'].apply(lambda x: x if x > 0 else np.nan)
                gl['OFF_RATING'] = np.where(gl['TEAM_POSSESSIONS'] > 0,
                                            (gl['PTS'] / gl['TEAM_POSSESSIONS']) * 100,
                                            np.nan)
                gl['DEF_RATING'] = np.where(gl['TEAM_POSSESSIONS'] > 0,
                                            (gl['PTS_OPP'] / gl['TEAM_POSSESSIONS']) * 100,
                                            np.nan)
                gl['PACE'] = gl['TEAM_POSSESSIONS']
                gl['rolling_avg'] = gl['PTS'].rolling(window=3, min_periods=1).mean()
                gl['rolling_std'] = gl['PTS'].rolling(window=3, min_periods=1).std().fillna(0)
                gl['season_avg'] = gl['PTS'].expanding().mean()
                gl['weighted_avg'] = gl['rolling_avg'] * 0.6 + gl['season_avg'] * 0.4
                for idx, row_ in gl.iterrows():
                    try:
                        all_rows.append({
                            'gameday': row_['GAME_DATE'],
                            'team': team_abbrev,
                            'score': float(row_['PTS']),
                            'off_rating': row_['OFF_RATING'] if pd.notnull(row_['OFF_RATING']) else np.nan,
                            'def_rating': row_['DEF_RATING'] if pd.notnull(row_['DEF_RATING']) else np.nan,
                            'pace': row_['PACE'] if pd.notnull(row_['PACE']) else np.nan,
                            'rolling_avg': row_['rolling_avg'],
                            'rolling_std': row_['rolling_std'],
                            'season_avg': row_['season_avg'],
                            'weighted_avg': row_['weighted_avg']
                        })
                    except Exception as e:
                        print(f"Error processing row for team {team_abbrev}: {e}")
                        continue
            except Exception as e:
                print(f"Error processing team {team_abbrev} for season {season}: {e}")
                continue
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df.dropna(subset=['score'], inplace=True)
    df.sort_values('gameday', inplace=True)
    for col in ['off_rating','def_rating','pace']:
        df[col].fillna(df[col].mean(), inplace=True)
    return df

def load_ncaab_data_current_season(season=2025):
    info_df, _, _ = cbb.get_games_season(season=season, info=True, box=False, pbp=False)
    if info_df.empty:
        return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(info_df["game_day"]):
        info_df["game_day"] = pd.to_datetime(info_df["game_day"], errors="coerce")
    home_df = info_df[['game_day','home_team','home_score','away_score']].rename(columns={
        "game_day": "gameday",
        "home_team": "team",
        "home_score": "score",
        "away_score": "opp_score"
    })
    home_df['is_home'] = 1
    away_df = info_df[['game_day','away_team','away_score','home_score']].rename(columns={
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
    data.sort_values(['team','gameday'], inplace=True)
    data['game_index'] = data.groupby('team').cumcount()
    return data

def fetch_upcoming_nfl_games(schedule, days_ahead=7):
    upcoming = schedule[schedule['home_score'].isna() & schedule['away_score'].isna()].copy()
    now = datetime.now()
    filter_date = now + timedelta(days=days_ahead)
    upcoming = upcoming[upcoming['gameday'] <= filter_date].copy()
    upcoming.sort_values('gameday', inplace=True)
    return upcoming[['gameday','home_team','away_team']]

def fetch_upcoming_nba_games(days_ahead=3):
    now = datetime.now()
    upcoming_rows = []
    for offset in range(days_ahead+1):
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

#########################
# TRAINING FUNCTION
#########################
def train_team_models(team_data: pd.DataFrame):
    """
    Trains a hybrid model (StackingRegressor + Auto-ARIMA) for each team's 'score' using
    time-series cross validation and hyperparameter optimization.
    Returns:
        stack_models: Dictionary of trained StackingRegressors keyed by team.
        arima_models: Dictionary of trained ARIMA models keyed by team.
        team_stats: Dictionary containing team statistics (including MSE and bias).
    """
    if team_data.empty or "team" not in team_data.columns:
        print("[train.py] team_data is empty or missing 'team' column.")
        return {}, {}, {}
    
    stack_models = {}
    arima_models = {}
    team_stats = {}
    all_teams = team_data["team"].unique()
    for team in all_teams:
        df_team = team_data[team_data["team"] == team].copy()
        df_team.sort_values("gameday", inplace=True)
        scores = df_team["score"].reset_index(drop=True)
        if len(scores) < 3:
            continue
        df_team["rolling_avg"] = df_team["score"].rolling(3, min_periods=1).mean()
        df_team["rolling_std"] = df_team["score"].rolling(3, min_periods=1).std().fillna(0)
        df_team["season_avg"] = df_team["score"].expanding().mean()
        df_team["weighted_avg"] = df_team["rolling_avg"] * 0.6 + df_team["season_avg"] * 0.4
        mean_val = round_half(scores.mean())
        std_val = round_half(scores.std())
        team_stats[team] = {
            "mean": mean_val,
            "std": std_val,
            "max": round_half(scores.max()),
            "recent_form": round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
        }
        features = df_team[["rolling_avg", "rolling_std", "weighted_avg"]].fillna(0)
        X = features.values
        y = scores.values
        n = len(X)
        split_index = int(n * 0.8)
        if split_index < 2 or (n - split_index) < 1:
            continue
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        try:
            xgb = XGBRegressor(random_state=42)
            xgb_grid = {"n_estimators": [50, 100], "max_depth": [3, 5]}
            xgb_best = tune_model(xgb, xgb_grid, X_train, y_train)
        except Exception as e:
            print(f"[train.py] Error tuning XGB for team {team}: {e}")
            xgb_best = XGBRegressor(n_estimators=100, random_state=42)
        try:
            lgbm = LGBMRegressor(random_state=42)
            lgbm_grid = {"n_estimators": [50, 100], "max_depth": [None, 5]}
            lgbm_best = tune_model(lgbm, lgbm_grid, X_train, y_train)
        except Exception as e:
            print(f"[train.py] Error tuning LGBM for team {team}: {e}")
            lgbm_best = LGBMRegressor(n_estimators=100, random_state=42)
        try:
            cat = CatBoostRegressor(verbose=0, random_state=42)
            cat_grid = {"iterations": [50, 100], "learning_rate": [0.1, 0.05]}
            cat_best = tune_model(cat, cat_grid, X_train, y_train)
        except Exception as e:
            print(f"[train.py] Error tuning CatBoost for team {team}: {e}")
            cat_best = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)
        estimators = [("xgb", xgb_best), ("lgbm", lgbm_best), ("cat", cat_best)]
        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=LGBMRegressor(),
            passthrough=False,
            cv=3,
        )
        try:
            stack.fit(X_train, y_train)
            preds = stack.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            print(f"[train.py] {team} - Stacking MSE: {mse}")
            stack_models[team] = stack
            team_stats[team]["mse"] = mse
            bias = np.mean(y_train - stack.predict(X_train))
            team_stats[team]["bias"] = bias
        except Exception as e:
            print(f"[train.py] Error training Stacking Regressor for team {team}: {e}")
            continue
        if len(scores) >= 7:
            try:
                arima = auto_arima(
                    scores,
                    seasonal=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    max_p=3,
                    max_q=3,
                )
                arima_models[team] = arima
            except Exception as e:
                print(f"[train.py] Error training ARIMA for team {team}: {e}")
                continue
    return stack_models, arima_models, team_stats

#########################
# MAIN DAILY TRAINING WORKFLOW
#########################
def daily_training_workflow():
    os.makedirs("models", exist_ok=True)
    
    # ----- NFL Training -----
    print("[train.py] Starting NFL training...")
    nfl_data = load_nfl_data()
    stack_models_nfl, arima_models_nfl, team_stats_nfl = train_team_models(nfl_data)
    joblib.dump(stack_models_nfl, "models/stack_models_nfl.pkl")
    joblib.dump(arima_models_nfl, "models/arima_models_nfl.pkl")
    joblib.dump(team_stats_nfl,  "models/team_stats_nfl.pkl")
    print("[train.py] NFL models saved.")
    
    # ----- NBA Training -----
    print("[train.py] Starting NBA training...")
    nba_data = load_nba_data()
    stack_models_nba, arima_models_nba, team_stats_nba = train_team_models(nba_data)
    joblib.dump(stack_models_nba, "models/stack_models_nba.pkl")
    joblib.dump(arima_models_nba, "models/arima_models_nba.pkl")
    joblib.dump(team_stats_nba,  "models/team_stats_nba.pkl")
    print("[train.py] NBA models saved.")
    
    # ----- NCAAB Training -----
    print("[train.py] Starting NCAAB training...")
    ncaab_data = load_ncaab_data_current_season(season=2025)
    stack_models_ncaab, arima_models_ncaab, team_stats_ncaab = train_team_models(ncaab_data)
    joblib.dump(stack_models_ncaab, "models/stack_models_ncaab.pkl")
    joblib.dump(arima_models_ncaab, "models/arima_models_ncaab.pkl")
    joblib.dump(team_stats_ncaab,  "models/team_stats_ncaab.pkl")
    print("[train.py] NCAAB models saved.")
    
    print("[train.py] Daily training complete!")

if __name__ == "__main__":
    daily_training_workflow()
