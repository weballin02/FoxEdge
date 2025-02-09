################################################################################
# train.py
# ------------------------------------------------------------------------------
# Headless script that fetches sports data, trains Stacking + ARIMA models, 
# then saves them to "models/" so the Streamlit app can load them.
# No user login or Streamlit UI is involved here.
################################################################################

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

import nfl_data_py
from nba_api.stats.endpoints import TeamGameLog
import cbbpy.mens_scraper as cbb

from pmdarima import auto_arima
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

#########################
# Shared Helper Methods #
#########################

def round_half(number):
    """Rounds a number to the nearest 0.5."""
    return round(number * 2) / 2

def tune_model(model, param_grid, X_train, y_train):
    """
    Hyperparameter tuning with GridSearchCV + TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(
        model, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def load_nfl_data():
    """
    Example: load + preprocess NFL data for training. 
    You can replicate your actual logic or call your existing code.
    """
    current_year = datetime.now().year
    years = [current_year - i for i in range(12)]  # last 12 years
    schedule = nfl_data_py.import_schedules(years)

    # Basic cleaning / converting
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    schedule.sort_values("gameday", inplace=True)
    
    # Combine home/away for a single 'team' column
    home_df = schedule[
        ["gameday", "home_team", "home_score", "away_score"]
    ].rename(columns={
        "home_team": "team",
        "home_score": "score",
        "away_score": "opp_score"
    })

    away_df = schedule[
        ["gameday", "away_team", "away_score", "home_score"]
    ].rename(columns={
        "away_team": "team",
        "away_score": "score",
        "home_score": "opp_score"
    })

    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=["score"], inplace=True)
    data.sort_values("gameday", inplace=True)

    # Simple rolling features for the team
    data["rolling_avg"] = data.groupby("team")["score"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    data["rolling_std"] = data.groupby("team")["score"].transform(
        lambda x: x.rolling(3, min_periods=1).std().fillna(0)
    )
    data["season_avg"] = data.groupby("team")["score"].transform(
        lambda x: x.expanding().mean()
    )
    data["weighted_avg"] = (data["rolling_avg"] * 0.6) + (data["season_avg"] * 0.4)
    return data

def train_team_models(team_data: pd.DataFrame):
    """
    Trains a hybrid model (StackingRegressor + Auto-ARIMA) for each team's 'score'.
    Identical to your advanced logic from your single file code.
    """
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

        # Feature engineering
        df_team["rolling_avg"] = df_team["score"].rolling(3, min_periods=1).mean()
        df_team["rolling_std"] = df_team["score"].rolling(3, min_periods=1).std().fillna(0)
        df_team["season_avg"] = df_team["score"].expanding().mean()
        df_team["weighted_avg"] = (
            df_team["rolling_avg"] * 0.6 + df_team["season_avg"] * 0.4
        )

        # Basic stats
        mean_ = round_half(scores.mean())
        std_  = round_half(scores.std())
        team_stats[team] = {
            "mean": mean_,
            "std": std_,
            "max":  round_half(scores.max()),
            "recent_form": round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
        }

        features = df_team[["rolling_avg", "rolling_std", "weighted_avg"]].fillna(0)
        X = features.values
        y = scores.values

        # 80/20 time-series split
        n = len(X)
        split_index = int(n * 0.8)
        if split_index < 2 or (n - split_index) < 1:
            continue
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Tune base models
        try:
            xgb = XGBRegressor(random_state=42)
            xgb_grid = {"n_estimators": [50, 100], "max_depth": [3, 5]}
            xgb_best = tune_model(xgb, xgb_grid, X_train, y_train)
        except:
            xgb_best = XGBRegressor(n_estimators=100, random_state=42)

        try:
            lgbm = LGBMRegressor(random_state=42)
            lgbm_grid = {"n_estimators": [50, 100], "max_depth": [None, 5]}
            lgbm_best = tune_model(lgbm, lgbm_grid, X_train, y_train)
        except:
            lgbm_best = LGBMRegressor(n_estimators=100, random_state=42)

        try:
            cat = CatBoostRegressor(verbose=0, random_state=42)
            cat_grid = {"iterations": [50, 100], "learning_rate": [0.1, 0.05]}
            cat_best = tune_model(cat, cat_grid, X_train, y_train)
        except:
            cat_best = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)

        estimators = [
            ("xgb", xgb_best),
            ("lgbm", lgbm_best),
            ("cat", cat_best),
        ]

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
            print(f"[train.py] Error training stack for {team}: {e}")
            continue

        # ARIMA if enough data
        if len(scores) >= 7:
            try:
                arima = auto_arima(
                    scores,
                    seasonal=False,
                    trace=False,
                    error_action="ignore",
                    suppress_warnings=True,
                    max_p=3,
                    max_q=3,
                )
                arima_models[team] = arima
            except Exception as e:
                print(f"[train.py] Error training ARIMA for {team}: {e}")
                continue

    return stack_models, arima_models, team_stats

def daily_training_workflow():
    """
    1) Fetch new NFL data (or any other leagues you want).
    2) Train Stacking + ARIMA.
    3) Save models for the Streamlit app to load.
    """
    os.makedirs("models", exist_ok=True)

    # 1) Example: load NFL data for training
    print("[train.py] Loading NFL data for training...")
    team_data = load_nfl_data()

    # 2) Train the advanced pipeline
    stack_models, arima_models, team_stats = train_team_models(team_data)

    # 3) Save .pkl for the app
    joblib.dump(stack_models, "models/stack_models.pkl")
    joblib.dump(arima_models, "models/arima_models.pkl")
    joblib.dump(team_stats,  "models/team_stats.pkl")

    print("[train.py] Models saved to 'models/' folder. Daily training complete!")

if __name__ == "__main__":
    daily_training_workflow()
