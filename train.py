#!/usr/bin/env python
"""
train.py
--------
Balanced version that optimizes performance while maintaining prediction accuracy
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

import nfl_data_py
from nba_api.stats.endpoints import TeamGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
import cbbpy.mens_scraper as cbb

from pmdarima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#########################
# Balanced Model Configuration
#########################
def configure_model_params(is_small_dataset: bool):
    """
    Configure parameters based on dataset size to balance accuracy and performance
    """
    if is_small_dataset:  # For teams with limited data (< 30 games)
        return {
            'xgb': XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                n_jobs=2
            ),
            'lgbm': LGBMRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=2
            ),
            'cat': CatBoostRegressor(
                iterations=50,
                depth=3,
                learning_rate=0.1,
                verbose=0,
                thread_count=2,
                random_state=42
            ),
            'grid_search': False
        }
    else:  # For teams with sufficient data
        return {
            'xgb': {
                'model': XGBRegressor(random_state=42, n_jobs=2),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
                }
            },
            'lgbm': {
                'model': LGBMRegressor(random_state=42, n_jobs=2),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 7],
                    'learning_rate': [0.01, 0.1]
                }
            },
            'cat': {
                'model': CatBoostRegressor(verbose=0, random_state=42, thread_count=2),
                'param_grid': {
                    'iterations': [100, 200],
                    'depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
                }
            },
            'grid_search': True
        }

#########################
# Enhanced Retry Function
#########################
def retry_call(func, retries=3, initial_delay=1, *args, **kwargs):
    """Retry with exponential backoff and logging"""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                delay = initial_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"All retries failed for {func.__name__}: {str(e)}")
                raise e

#########################
# Balanced Training Function
#########################
def train_team_models(team_data: pd.DataFrame, sport_name: str):
    """
    Balanced training function that adapts to dataset size and available resources
    """
    if team_data.empty or "team" not in team_data.columns:
        logger.error(f"[{sport_name}] team_data is empty or missing 'team' column.")
        return {}, {}, {}
    
    stack_models = {}
    arima_models = {}
    team_stats = {}
    
    # Process teams in manageable batches while maintaining model quality
    batch_size = 10  # Increased from 5 to reduce overhead
    all_teams = team_data["team"].unique()
    total_teams = len(all_teams)
    
    for batch_start in range(0, total_teams, batch_size):
        batch_teams = all_teams[batch_start:batch_start + batch_size]
        logger.info(f"[{sport_name}] Processing batch {batch_start//batch_size + 1} of {(total_teams + batch_size - 1)//batch_size}")
        
        for team in batch_teams:
            try:
                df_team = team_data[team_data["team"] == team].copy()
                if len(df_team) < 10:  # Skip teams with insufficient data
                    logger.warning(f"Skipping {team} due to insufficient data (less than 10 games)")
                    continue
                
                # Determine if this is a small dataset
                is_small_dataset = len(df_team) < 30
                model_configs = configure_model_params(is_small_dataset)
                
                # Calculate features with proper handling of time series nature
                df_team.sort_values("gameday", inplace=True)
                scores = df_team["score"].values
                
                # Enhanced feature engineering
                features = pd.DataFrame({
                    'rolling_avg_3': df_team["score"].rolling(3, min_periods=1).mean(),
                    'rolling_avg_5': df_team["score"].rolling(5, min_periods=1).mean(),
                    'rolling_std': df_team["score"].rolling(3, min_periods=1).std(),
                    'ewm_alpha_03': df_team["score"].ewm(alpha=0.3).mean(),
                    'ewm_alpha_07': df_team["score"].ewm(alpha=0.7).mean()
                }).fillna(method='bfill').fillna(0)
                
                # Use time series cross-validation for more reliable evaluation
                tscv = TimeSeriesSplit(n_splits=3 if is_small_dataset else 5)
                
                # Train stacking model
                if model_configs.get('grid_search', False):
                    estimators = []
                    for name, config in model_configs.items():
                        if name != 'grid_search':
                            from sklearn.model_selection import GridSearchCV
                            grid = GridSearchCV(
                                config['model'],
                                config['param_grid'],
                                cv=tscv,
                                n_jobs=2,
                                scoring='neg_mean_squared_error'
                            )
                            grid.fit(features, scores)
                            estimators.append((name, grid.best_estimator_))
                else:
                    estimators = [
                        (name, model) 
                        for name, model in model_configs.items()
                        if name != 'grid_search'
                    ]
                
                # Create and train stacking model
                stack = StackingRegressor(
                    estimators=estimators,
                    final_estimator=LGBMRegressor(n_estimators=100),
                    cv=tscv
                )
                
                stack.fit(features, scores)
                stack_models[team] = stack
                
                # Calculate comprehensive team stats
                recent_scores = scores[-10:] if len(scores) >= 10 else scores
                team_stats[team] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "recent_mean": np.mean(recent_scores),
                    "recent_std": np.std(recent_scores),
                    "trend": np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                }
                
                # Train ARIMA model with appropriate complexity
                if len(scores) >= 20:  # Only use ARIMA with sufficient data
                    arima = auto_arima(
                        scores,
                        start_p=1, start_q=1,
                        max_p=3, max_q=3,
                        m=1,  # Non-seasonal model
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        max_order=5
                    )
                    arima_models[team] = arima
                
            except Exception as e:
                logger.error(f"Error processing team {team}: {str(e)}")
                continue
        
        # Memory management
        import gc
        gc.collect()
    
    return stack_models, arima_models, team_stats

#########################
# Main Training Workflow
#########################
def daily_training_workflow():
    start_time = time.time()
    logger.info("Starting daily training workflow")
    
    os.makedirs("models", exist_ok=True)
    
    for sport, load_func in [
        ("NFL", load_nfl_data),
        ("NBA", load_nba_data),
        ("NCAAB", lambda: load_ncaab_data_current_season(2025))
    ]:
        try:
            logger.info(f"Loading {sport} data...")
            data = load_func()
            
            logger.info(f"Training {sport} models...")
            stack_models, arima_models, team_stats = train_team_models(data, sport)
            
            # Save models
            sport_lower = sport.lower()
            joblib.dump(stack_models, f"models/stack_models_{sport_lower}.pkl")
            joblib.dump(arima_models, f"models/arima_models_{sport_lower}.pkl")
            joblib.dump(team_stats, f"models/team_stats_{sport_lower}.pkl")
            
            logger.info(f"{sport} models saved successfully")
            
        except Exception as e:
            logger.error(f"Error processing {sport}: {str(e)}")
            continue
    
    duration = time.time() - start_time
    logger.info(f"Daily training completed in {duration:.2f} seconds")

if __name__ == "__main__":
    daily_training_workflow()
