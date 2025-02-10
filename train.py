#!/usr/bin/env python3
"""
Train Script: Updates NFL, NBA, and NCAAB data; trains per‑team models (Stacking Regressor + ARIMA);
and saves team‑specific models and team statistics to disk.
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from pmdarima import auto_arima

import nfl_data_py as nfl
from nba_api.stats.endpoints import TeamGameLog
from nba_api.stats.static import teams as nba_teams
import cbbpy.mens_scraper as cbb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)


class ModelTrainer:
    """
    Class for updating data, training per‑team models (Stacking Regressor + ARIMA),
    and saving trained models and team statistics.
    """

    def __init__(self):
        self.model_dir = Path("models")
        self.data_dir = Path("data")
        self.model_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

    @staticmethod
    def _round_half(number):
        """Rounds a number to the nearest 0.5."""
        return round(number * 2) / 2

    def _tune_model(self, model, param_grid, X_train, y_train):
        """
        Tunes a model using GridSearchCV with a TimeSeriesSplit.
        """
        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(model, param_grid, cv=tscv,
                            scoring="neg_mean_squared_error", n_jobs=-1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_

    def _preprocess_nfl_data(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess NFL schedule data to create a team-level dataset with rolling features.
        """
        home_df = schedule[["gameday", "home_team", "home_score", "away_score"]].rename(
            columns={"home_team": "team", "home_score": "score", "away_score": "opp_score"}
        )
        away_df = schedule[["gameday", "away_team", "away_score", "home_score"]].rename(
            columns={"away_team": "team", "away_score": "score", "home_score": "opp_score"}
        )
        data = pd.concat([home_df, away_df], ignore_index=True)
        data.dropna(subset=["score"], inplace=True)
        data.sort_values("gameday", inplace=True)
        data["rolling_avg"] = data.groupby("team")["score"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        data["rolling_std"] = data.groupby("team")["score"].transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0)
        )
        data["season_avg"] = data.groupby("team", group_keys=False)["score"].apply(
            lambda x: x.expanding().mean()
        )
        data["weighted_avg"] = (data["rolling_avg"] * 0.6) + (data["season_avg"] * 0.4)
        return data

    def _preprocess_ncaab_data(self, info_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess NCAAB game data to generate features.
        """
        if info_df.empty:
            return pd.DataFrame()
        if not pd.api.types.is_datetime64_any_dtype(info_df["game_day"]):
            info_df["game_day"] = pd.to_datetime(info_df["game_day"], errors="coerce")
        home_df = info_df[["game_day", "home_team", "home_score", "away_score"]].rename(
            columns={
                "game_day": "gameday",
                "home_team": "team",
                "home_score": "score",
                "away_score": "opp_score"
            }
        )
        home_df["is_home"] = 1
        away_df = info_df[["game_day", "away_team", "away_score", "home_score"]].rename(
            columns={
                "game_day": "gameday",
                "away_team": "team",
                "away_score": "score",
                "home_score": "opp_score"
            }
        )
        away_df["is_home"] = 0
        data = pd.concat([home_df, away_df], ignore_index=True)
        data.dropna(subset=["score"], inplace=True)
        data.sort_values("gameday", inplace=True)
        data["rolling_avg"] = data.groupby("team")["score"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        data["rolling_std"] = data.groupby("team")["score"].transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0)
        )
        data["season_avg"] = data.groupby("team", group_keys=False)["score"].apply(
            lambda x: x.expanding().mean()
        )
        data["weighted_avg"] = (data["rolling_avg"] * 0.6) + (data["season_avg"] * 0.4)
        data.sort_values(["team", "gameday"], inplace=True)
        data["game_index"] = data.groupby("team").cumcount()
        return data

    def train_team_models(self, team_data: pd.DataFrame):
        """
        Trains a stacking regressor and an ARIMA model for each team.
        Saves the team‑specific models and aggregates team statistics to disk.
        """
        stack_models = {}
        arima_models = {}
        team_stats = {}

        all_teams = team_data["team"].unique()
        for team in all_teams:
            logging.info(f"Training models for team: {team}")
            df_team = team_data[team_data["team"] == team].copy()
            df_team.sort_values("gameday", inplace=True)
            scores = df_team["score"].reset_index(drop=True)

            if len(scores) < 3:
                logging.info(f"Not enough data for team: {team}")
                continue

            # Enhanced Feature Engineering
            df_team["rolling_avg"] = df_team["score"].rolling(window=3, min_periods=1).mean()
            df_team["rolling_std"] = df_team["score"].rolling(window=3, min_periods=1).std().fillna(0)
            df_team["season_avg"] = df_team["score"].expanding().mean()
            df_team["weighted_avg"] = (df_team["rolling_avg"] * 0.6) + (df_team["season_avg"] * 0.4)

            # Save basic team statistics
            team_stats[team] = {
                "mean": self._round_half(scores.mean()),
                "std": self._round_half(scores.std()),
                "max": self._round_half(scores.max()),
                "recent_form": self._round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
            }

            # Prepare features and target
            features = df_team[["rolling_avg", "rolling_std", "weighted_avg"]].fillna(0)
            X = features.values
            y = scores.values

            # Time-series split: 80% training, 20% testing
            n = len(X)
            split_index = int(n * 0.8)
            if split_index < 2 or n - split_index < 1:
                logging.info(f"Not enough training data for team: {team}")
                continue
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # Hyperparameter tuning for base models
            try:
                xgb = XGBRegressor(random_state=42)
                xgb_grid = {"n_estimators": [50, 100], "max_depth": [3, 5]}
                xgb_best = self._tune_model(xgb, xgb_grid, X_train, y_train)
            except Exception as e:
                logging.error(f"Error tuning XGB for team {team}: {e}")
                xgb_best = XGBRegressor(n_estimators=100, random_state=42)

            try:
                lgbm = LGBMRegressor(random_state=42)
                lgbm_grid = {"n_estimators": [50, 100], "max_depth": [None, 5]}
                lgbm_best = self._tune_model(lgbm, lgbm_grid, X_train, y_train)
            except Exception as e:
                logging.error(f"Error tuning LGBM for team {team}: {e}")
                lgbm_best = LGBMRegressor(n_estimators=100, random_state=42)

            try:
                cat = CatBoostRegressor(verbose=0, random_state=42)
                cat_grid = {"iterations": [50, 100], "learning_rate": [0.1, 0.05]}
                cat_best = self._tune_model(cat, cat_grid, X_train, y_train)
            except Exception as e:
                logging.error(f"Error tuning CatBoost for team {team}: {e}")
                cat_best = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)

            estimators = [
                ("xgb", xgb_best),
                ("lgbm", lgbm_best),
                ("cat", cat_best)
            ]

            # Train stacking regressor
            stack = StackingRegressor(
                estimators=estimators,
                final_estimator=LGBMRegressor(),
                passthrough=False,
                cv=3
            )
            try:
                stack.fit(X_train, y_train)
                preds = stack.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                logging.info(f"Team: {team}, Stacking Regressor MSE: {mse}")
                stack_models[team] = stack
                team_stats[team]["mse"] = mse
                # Compute bias from training data for calibration
                bias = np.mean(y_train - stack.predict(X_train))
                team_stats[team]["bias"] = bias
            except Exception as e:
                logging.error(f"Error training stacking regressor for team {team}: {e}")
                continue

            # Train ARIMA model if sufficient data is available
            if len(scores) >= 7:
                try:
                    arima = auto_arima(
                        scores,
                        seasonal=False,
                        trace=False,
                        error_action="ignore",
                        suppress_warnings=True,
                        max_p=3,
                        max_q=3
                    )
                    arima_models[team] = arima
                except Exception as e:
                    logging.error(f"Error training ARIMA for team {team}: {e}")
                    continue

            # Save team‑specific models to disk
            joblib.dump(stack, self.model_dir / f"{team}_stack.pkl")
            if team in arima_models:
                joblib.dump(arima_models[team], self.model_dir / f"{team}_arima.pkl")

        # Save team statistics and update timestamp
        joblib.dump(team_stats, self.model_dir / "team_stats.pkl")
        timestamp = datetime.now()
        joblib.dump(timestamp, self.model_dir / "last_update.pkl")
        logging.info("Team models and statistics saved successfully.")
        return stack_models, arima_models, team_stats

    def update_nfl_data(self) -> pd.DataFrame:
        """
        Updates NFL data by fetching schedules, processing them, and saving both raw and processed data.
        """
        logging.info("Updating NFL data...")
        current_year = datetime.now().year
        years = list(range(current_year - 2, current_year + 1))
        schedule = nfl.import_schedules(years)
        processed_data = self._preprocess_nfl_data(schedule)
        processed_data.to_parquet(self.data_dir / "nfl_processed.parquet")
        schedule.to_parquet(self.data_dir / "nfl_schedule.parquet")
        logging.info("NFL data updated successfully.")
        return processed_data

    def update_nba_data(self) -> pd.DataFrame:
        """
        Updates NBA data by fetching team game logs, processing them, and saving the result.
        """
        logging.info("Updating NBA data...")
        all_data = []
        for team in nba_teams.get_teams():
            try:
                team_id = team["id"]
                logs = TeamGameLog(team_id=team_id, season="2023-24").get_data_frames()[0]
                if not logs.empty:
                    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
                    logs.sort_values("GAME_DATE", inplace=True)
                    all_data.append(logs)
            except Exception as e:
                logging.error(f"Error fetching NBA data for team {team.get('full_name')}: {e}")
        if all_data:
            processed_data = pd.concat(all_data, ignore_index=True)
            processed_data.to_parquet(self.data_dir / "nba_processed.parquet")
            logging.info("NBA data updated successfully.")
            return processed_data
        logging.warning("No NBA data available.")
        return None

    def update_ncaab_data(self) -> pd.DataFrame:
        """
        Updates NCAAB data by fetching season games, processing them, and saving the result.
        """
        logging.info("Updating NCAAB data...")
        current_season = datetime.now().year
        try:
            data, _, _ = cbb.get_games_season(season=current_season, info=True, box=False, pbp=False)
            processed_data = self._preprocess_ncaab_data(data)
            processed_data.to_parquet(self.data_dir / "ncaab_processed.parquet")
            logging.info("NCAAB data updated successfully.")
            return processed_data
        except Exception as e:
            logging.error(f"Error fetching NCAAB data: {e}")
            return None

    def run_full_update(self):
        """
        Runs a complete update: updates data for NFL, NBA, and NCAAB,
        trains models for each league, and saves a timestamp of the update.
        """
        logging.info("Starting full update process...")
        nfl_data = self.update_nfl_data()
        nba_data = self.update_nba_data()
        ncaab_data = self.update_ncaab_data()

        if nfl_data is not None:
            self.train_team_models(nfl_data)
        if nba_data is not None:
            self.train_team_models(nba_data)
        if ncaab_data is not None:
            self.train_team_models(ncaab_data)

        # Save last update timestamp is handled in train_team_models.
        logging.info("Full update process completed successfully.")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_full_update()
