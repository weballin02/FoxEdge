import warnings
# Suppress known joblib/loky resource_tracker warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

import streamlit as st
import pandas as pd
import numpy as np
import pytz
import random
from datetime import datetime, timedelta
import nfl_data_py as nfl
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2, TeamGameLog
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from pmdarima import auto_arima
from pathlib import Path
import requests
import firebase_admin
from firebase_admin import credentials, auth
import joblib
import os
import optuna  # For Bayesian hyperparameter optimization
import inspect

# cbbpy for NCAAB
import cbbpy.mens_scraper as cbb

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

################################################################################
# FEATURE FLAGS (Merged from test script)
################################################################################
LOCAL_TEST_MODE = True
TEST_FIRST_PREDICTION_FROM_CSV = False
USE_NBA_CSV_DATA = True

# Existing production flags:
USE_RANDOMIZED_SEARCH = True
USE_OPTUNA_SEARCH = True
ENABLE_EARLY_STOPPING = True
DISABLE_TUNING_FOR_NCAAB = True

CSV_FILE = "predictions.csv"  # (Already used in production)

################################################################################
# LOGO MERGING FUNCTIONS (Merged from test script)
################################################################################
# NBA mapping dictionary for logo files.
nba_mapping = {
    "ATL": "Atlanta Hawks",
    "BKN": "Brooklyn Nets",
    "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"
}

def merge_ncaa_team_logos(team1, team2, output_path="merged_ncaa_logo.png", target_height=200, bg_color=(255, 255, 255)):
    """
    Merge NCAA team logos side by side.
    Filenames are assumed to be team names with underscores replacing spaces, ending with '_logo.png'.
    """
    base_dir = "ncaa_images"  # Adjust the directory name if needed

    def get_logo(team_name):
        expected = team_name.strip().replace(" ", "_") + "_logo.png"
        for file in os.listdir(base_dir):
            if file.lower() == expected.lower():
                return os.path.join(base_dir, file)
        return None

    logo1_path = get_logo(team1)
    logo2_path = get_logo(team2)
    if not logo1_path or not logo2_path:
        return None
    from PIL import Image  # Import here to limit scope
    img1 = Image.open(logo1_path)
    img2 = Image.open(logo2_path)
    img1_ratio = img1.width / img1.height
    img2_ratio = img2.width / img2.height
    img1 = img1.resize((int(target_height * img1_ratio), target_height))
    img2 = img2.resize((int(target_height * img2_ratio), target_height))
    total_width = img1.width + img2.width
    merged_img = Image.new('RGB', (total_width, target_height), color=bg_color)
    merged_img.paste(img1, (0, 0))
    merged_img.paste(img2, (img1.width, 0))
    merged_img.save(output_path)
    return output_path

def merge_team_logos(team1, team2, output_path="merged_logo.png", target_height=200, bg_color=(255, 255, 255)):
    """
    Merge NBA team logos side by side.
    """
    def get_logo(abbrev):
        base_dir = "nba_images"
        if abbrev in nba_mapping:
            filename = nba_mapping[abbrev].replace(" ", "_") + ".png"
        else:
            filename = abbrev.strip().replace(" ", "_") + ".png"
        path = os.path.join(base_dir, filename)
        return path if os.path.exists(path) else None

    logo1_path = get_logo(team1)
    logo2_path = get_logo(team2)
    if not logo1_path or not logo2_path:
        return None
    from PIL import Image  # Import here to limit scope
    img1 = Image.open(logo1_path)
    img2 = Image.open(logo2_path)
    img1_ratio = img1.width / img1.height
    img2_ratio = img2.width / img2.height
    img1 = img1.resize((int(target_height * img1_ratio), target_height))
    img2 = img2.resize((int(target_height * img2_ratio), target_height))
    total_width = img1.width + img2.width
    merged_img = Image.new('RGB', (total_width, target_height), color=bg_color)
    merged_img.paste(img1, (0, 0))
    merged_img.paste(img2, (img1.width, 0))
    merged_img.save(output_path)
    return output_path

################################################################################
# CSV HELPER FUNCTIONS (Merged from test script)
################################################################################
def load_csv_data_safe(file_path, default_df=None):
    try:
        file = Path(file_path)
        if file.exists():
            return pd.read_csv(file)
    except Exception as e:
        print(f"Warning: Failed to load {file_path}. Error: {e}")
    return default_df if default_df is not None else pd.DataFrame()

def save_prediction_to_csv(prediction, csv_file=CSV_FILE):
    """
    Save a single prediction to CSV. Formats the date appropriately.
    """
    prediction_copy = prediction.copy()
    if isinstance(prediction_copy.get('date'), datetime):
        prediction_copy['date'] = prediction_copy['date'].strftime('%m/%d/%y')
    elif isinstance(prediction_copy.get('date'), str):
        try:
            d = datetime.strptime(prediction_copy['date'], '%Y-%m-%d')
            prediction_copy['date'] = d.strftime('%m/%d/%y')
        except Exception:
            pass
    file = Path(csv_file)
    if not file.exists():
        columns = ["date", "league", "home_team", "away_team", "home_pred", "away_pred",
                   "predicted_winner", "predicted_diff", "predicted_total",
                   "spread_suggestion", "ou_suggestion", "confidence"]
        df = pd.DataFrame([prediction_copy], columns=columns)
        df.to_csv(csv_file, index=False)
    else:
        df_existing = pd.read_csv(csv_file)
        df_new = pd.DataFrame([prediction_copy])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined['date'] = pd.to_datetime(df_combined['date'], errors='coerce').dt.strftime('%m/%d/%y')
        df_combined.to_csv(csv_file, index=False)
    st.success("Prediction saved to CSV!")

@st.cache_data(show_spinner=False, hash_funcs={np.ndarray: lambda x: x.tobytes()})
def fit_distribution(simulation_array):
    """
    Fit a probability distribution to the simulation results using the fitter library.
    Returns the best‐fitting distribution based on sum‑of‐squares error.
    """
    try:
        from fitter import Fitter
        f = Fitter(simulation_array.flatten(), distributions=['norm', 't', 'lognorm'])
        f.fit()
        return f.get_best(method='sumsquare_error')
    except Exception as e:
        print(f"Error fitting distribution: {e}")
        return None


def monte_carlo_simulation_margin(model_home, model_away, X_home, X_away, n_simulations=10000,
                                  error_std_home=5, error_std_away=5, random_seed=42, run_fitter=False):
    """
    Run a Monte Carlo simulation on the point margin (home minus away).
    
    Args:
        model_home: Trained model for the home team.
        model_away: Trained model for the away team.
        X_home: Feature array for the home team.
        X_away: Feature array for the away team.
        n_simulations: Number of simulation iterations.
        error_std_home: Standard deviation for home team noise.
        error_std_away: Standard deviation for away team noise.
        random_seed: Seed for reproducibility.
        run_fitter: If True, fit a distribution to the simulation outcomes.
    
    Returns:
        A dictionary with the following keys:
          - mean_diff: Mean simulated margin.
          - ci: A tuple (lower, upper) representing the 95% confidence interval.
          - median_diff: Median simulated margin.
          - std_diff: Standard deviation of the simulated margins.
          - win_rate: Percentage of simulations favoring the predicted winner.
          - win_margin_rate: Percentage of simulations exceeding the mean margin.
          - simulated_diffs: The raw simulated differences as a NumPy array.
          - fitted_distribution: The best‑fitting distribution (if run_fitter is True).
    """
    np.random.seed(random_seed)
    base_pred_home = model_home.predict(X_home)
    base_pred_away = model_away.predict(X_away)
    simulated_diffs = []
    for _ in range(n_simulations):
        noise_home = np.random.normal(0, error_std_home, size=base_pred_home.shape)
        noise_away = np.random.normal(0, error_std_away, size=base_pred_away.shape)
        simulated_diffs.append((base_pred_home + noise_home) - (base_pred_away + noise_away))
    simulated_diffs = np.array(simulated_diffs)
    mean_diff = simulated_diffs.mean()
    ci_lower = np.percentile(simulated_diffs, 2.5)
    ci_upper = np.percentile(simulated_diffs, 97.5)
    ci = (ci_lower, ci_upper)
    if mean_diff >= 0:
        win_rate = (simulated_diffs > 0).mean() * 100
        win_margin_rate = (simulated_diffs >= mean_diff).mean() * 100
    else:
        win_rate = (simulated_diffs < 0).mean() * 100
        win_margin_rate = (simulated_diffs <= mean_diff).mean() * 100
    return {
        "mean_diff": mean_diff,
        "ci": ci,
        "median_diff": np.median(simulated_diffs),
        "std_diff": simulated_diffs.std(),
        "win_rate": win_rate,
        "win_margin_rate": win_margin_rate,
        "simulated_diffs": simulated_diffs,
        "fitted_distribution": fit_distribution(simulated_diffs) if run_fitter else None
    }


def monte_carlo_simulation_totals(model_home, model_away, X_home, X_away, n_simulations=10000,
                                  error_std_home=5, error_std_away=5, random_seed=42, run_fitter=False):
    """
    Run a Monte Carlo simulation on the game total (sum of home and away scores).
    
    Args:
        model_home: Trained model for the home team.
        model_away: Trained model for the away team.
        X_home: Feature array for the home team.
        X_away: Feature array for the away team.
        n_simulations: Number of simulation iterations.
        error_std_home: Standard deviation for home team noise.
        error_std_away: Standard deviation for away team noise.
        random_seed: Seed for reproducibility.
        run_fitter: If True, fit a distribution to the simulation outcomes.
    
    Returns:
        A dictionary with the following keys:
          - mean_total: Mean simulated total points.
          - median_total: Median simulated total.
          - std_total: Standard deviation of the totals.
          - over_threshold: 5th percentile of totals.
          - under_threshold: 95th percentile of totals.
          - simulated_totals: The raw simulated totals as a NumPy array.
          - fitted_distribution: The best‑fitting distribution (if run_fitter is True).
    """
    np.random.seed(random_seed)
    base_pred_home = model_home.predict(X_home)
    base_pred_away = model_away.predict(X_away)
    simulated_totals = []
    for _ in range(n_simulations):
        noise_home = np.random.normal(0, error_std_home, size=base_pred_home.shape)
        noise_away = np.random.normal(0, error_std_away, size=base_pred_away.shape)
        simulated_totals.append((base_pred_home + noise_home) + (base_pred_away + noise_away))
    simulated_totals = np.array(simulated_totals)
    return {
        "mean_total": simulated_totals.mean(),
        "median_total": np.median(simulated_totals),
        "std_total": simulated_totals.std(),
        "over_threshold": np.percentile(simulated_totals, 5),
        "under_threshold": np.percentile(simulated_totals, 95),
        "simulated_totals": simulated_totals,
        "fitted_distribution": fit_distribution(simulated_totals) if run_fitter else None
    }


def update_predictions_with_results(predictions_csv="predictions.csv", league="NBA"):
    """
    Update the predictions CSV file with actual game outcomes and compute the differences
    between predicted and actual margins and totals.
    
    This function reads in the predictions, matches them with historical game data
    (using the appropriate data loader for the given league), and then updates each prediction
    with the actual home score, away score, margin, total, deltas, and whether the predicted
    winner matches the actual winner.
    
    Args:
        predictions_csv (str): Path to the CSV file containing predictions.
        league (str): League identifier ("NBA", "NFL", or "NCAAB").
    
    Returns:
        pd.DataFrame: The updated predictions DataFrame.
    """
    try:
        df_preds = pd.read_csv(predictions_csv, parse_dates=["date"])
    except Exception as e:
        print(f"Error loading predictions CSV: {e}")
        return None
    if league == "NBA":
        df_history = load_nba_data()
        if "pts" in df_history.columns:
            df_history = df_history.rename(columns={"pts": "score"})
    elif league == "NFL":
        schedule = load_nfl_schedule()
        df_history = preprocess_nfl_data(schedule)
    elif league == "NCAAB":
        df_history = load_ncaab_data_current_season(season=2025)
    else:
        print("Unsupported league specified.")
        return None
    updated_rows = []
    for idx, pred in df_preds.iterrows():
        try:
            pred_date = pd.to_datetime(pred["date"]).date()
        except Exception as e:
            print(f"Error converting prediction date: {e}")
            pred_date = None
        home_team = pred.get("home_team", None)
        away_team = pred.get("away_team", None)
        df_game = df_history[pd.to_datetime(df_history["gameday"]).dt.date == pred_date]
        game_home = df_game[df_game["team"] == home_team]
        game_away = df_game[df_game["team"] == away_team]
        if not game_home.empty and not game_away.empty:
            actual_home_score = game_home.iloc[0].get("score", None)
            actual_away_score = game_away.iloc[0].get("score", None)
            if actual_home_score is not None and actual_away_score is not None:
                actual_margin = actual_home_score - actual_away_score
                actual_total = actual_home_score + actual_away_score
                pred_margin = pred.get("predicted_diff", None)
                pred_total = pred.get("predicted_total", None)
                margin_delta = actual_margin - pred_margin if pred_margin is not None else None
                total_delta = actual_total - pred_total if pred_total is not None else None
                predicted_winner = pred.get("predicted_winner", None)
                actual_winner = home_team if actual_margin > 0 else away_team
                prediction_correct = (predicted_winner == actual_winner)
                pred_update = pred.to_dict()
                pred_update.update({
                    "actual_home_score": actual_home_score,
                    "actual_away_score": actual_away_score,
                    "actual_margin": actual_margin,
                    "actual_total": actual_total,
                    "margin_delta": margin_delta,
                    "total_delta": total_delta,
                    "prediction_correct": prediction_correct
                })
                updated_rows.append(pred_update)
            else:
                updated_rows.append(pred.to_dict())
        else:
            updated_rows.append(pred.to_dict())
    updated_df = pd.DataFrame(updated_rows)
    try:
        updated_df.to_csv(predictions_csv, index=False)
        updated_df.to_csv("top_bets.csv", index=False)
        print(f"Predictions CSV updated successfully and saved to both {predictions_csv} and top_bets.csv")
    except Exception as e:
        print(f"Error saving updated CSV: {e}")
    return updated_df

################################################################################
# ENHANCED DETAILED INSIGHTS FUNCTIONS (Merged from test script)
################################################################################
def updated_confidence(baseline_conf, monte_mean_diff, monte_std, win_probability, K=10, epsilon=1e-6):
    """
    Adjust baseline confidence using Monte Carlo outputs.
    """
    snr = abs(monte_mean_diff) / (monte_std + epsilon)
    adjusted = baseline_conf + K * ((win_probability / 100) * snr - 1)
    return max(1, min(99, adjusted))

def generate_betting_insights(bet):
    """
    Generate a text summary of detailed betting insights based on the bet metrics.
    """
    from scipy.stats import skew, kurtosis
    insights = []
    if bet.get('confidence', 0) >= 80 and bet.get('win_probability', 0) >= 75:
        insights.append(f"High confidence ({bet['confidence']:.1f}%) and win probability of {bet['win_probability']:.1f}% indicate a strong play on {bet['predicted_winner']}.")
    else:
        insights.append(f"Moderate confidence ({bet['confidence']:.1f}%) and win probability of {bet.get('win_probability', 0):.1f}% suggest caution.")
    if bet.get('monte_mean_diff') is not None and bet.get('monte_std') is not None:
        snr = abs(bet['monte_mean_diff']) / (bet['monte_std'] + 1e-6)
        insights.append(f"Signal-to-noise ratio: {snr:.2f} ({'strong' if snr >= 1.5 else 'moderate'} predictive strength).")
        if bet.get('monte_ci'):
            ci_width = bet['monte_ci'][1] - bet['monte_ci'][0]
            rel_ci = ci_width / (abs(bet['monte_mean_diff']) + 1e-6)
            insights.append(f"95% CI is {ci_width:.2f} points wide (≈ {rel_ci:.2f}× the mean margin).")
            if rel_ci < 1.5:
                insights.append("Narrow CI indicates high consistency.")
            else:
                insights.append("Wide CI signals significant uncertainty; consider smaller stakes.")
    if bet.get('simulated_diffs') is not None:
        sim_diffs = np.array(bet['simulated_diffs']).flatten()
        from scipy.stats import skew, kurtosis
        sim_skew = skew(sim_diffs)
        sim_kurt = kurtosis(sim_diffs)
        insights.append(f"Distribution: {sim_skew:+.2f} skew, {sim_kurt:.2f} kurtosis ({'extreme outcomes' if sim_kurt > 3 else 'normal distribution'}).")
    if bet.get('monte_mean_diff') is not None and bet.get('monte_std') is not None and bet.get('monte_ci'):
        snr = abs(bet['monte_mean_diff']) / (bet['monte_std'] + 1e-6)
        ci_width = bet['monte_ci'][1] - bet['monte_ci'][0]
        rel_ci = ci_width / (abs(bet['monte_mean_diff']) + 1e-6)
        value_score = (bet['win_probability'] / 100) * (bet['confidence'] / 100) * snr / rel_ci
        value_score_text = f"Composite Value Score: {value_score:.2f}."
        new_rel_ci = rel_ci * 0.9  # if CI narrows by 10%
        new_value_score = (bet['win_probability'] / 100) * (bet['confidence'] / 100) * snr / new_rel_ci
        percent_increase = ((new_value_score - value_score) / value_score) * 100
        value_score_text += f" (Would increase by {percent_increase:.1f}% if CI were 10% narrower.)"
        insights.append(value_score_text)
    else:
        insights.append("Composite Value Score: N/A.")
    if not insights:
        insights.append("Metrics inconclusive; additional research advised.")
    return " ".join(insights)

def recommended_betting_strategy(bet, bankroll=1000, fraction=0.5):
    """
    Recommend a betting strategy based on bet metrics.
    """
    if bet.get('win_probability') is None or bet.get('confidence') is None:
        return "Insufficient data to recommend a strategy."
    if bet['win_probability'] < 45:
        return "No Bet: Win probability is below 45%."
    if bet.get('monte_ci'):
        spread_range = f"between {bet['monte_ci'][0]:.1f} and {bet['monte_ci'][1]:.1f} points"
    else:
        spread_range = f"around {bet.get('predicted_diff', 0):.1f} points"
    spread_recommendation = f"Bet on {bet.get('predicted_winner', 'Unknown')} to cover a spread of {spread_range}."
    over_under_recommendation = f"Bet the game total to be {'over' if bet.get('predicted_total', 0) >= 145 else 'under'} {bet.get('predicted_total', 0):.1f} points."
    if bet['confidence'] >= 75 and bet['win_probability'] >= 70:
        level = "Aggressive Bet"
    elif bet['confidence'] >= 60 and bet['win_probability'] >= 60:
        level = "Moderate Bet"
    elif bet['confidence'] >= 50 and bet['win_probability'] >= 50:
        level = "Cautious Bet"
    else:
        return "No Bet: Metrics do not support a favorable betting opportunity."
    rec_frac = max(0, (2*(bet['win_probability']/100) - 1)) * fraction
    rec_amount = bankroll * rec_frac
    wager_percentage = rec_frac * 100
    value_score_text = ""
    if bet.get('monte_mean_diff') is not None and bet.get('monte_std') is not None and bet.get('monte_ci'):
        snr = abs(bet['monte_mean_diff']) / (bet['monte_std'] + 1e-6)
        ci_width = bet['monte_ci'][1] - bet['monte_ci'][0]
        rel_ci = ci_width / (abs(bet['monte_mean_diff']) + 1e-6)
        value_score = (bet['win_probability'] / 100) * (bet['confidence'] / 100) * snr / rel_ci
        value_score_text = f"Composite Value Score: {value_score:.2f}."
        new_rel_ci = rel_ci * 0.9
        new_value_score = (bet['win_probability'] / 100) * (bet['confidence'] / 100) * snr / new_rel_ci
        percent_increase = ((new_value_score - value_score) / value_score) * 100
        value_score_text += f" (If CI were 10% narrower, Value Score would increase by {percent_increase:.1f}%)."
    else:
        value_score_text = "Composite Value Score: N/A."
    strategy = (
        f"{level} Strategy:\n"
        f"{spread_recommendation}\n"
        f"{over_under_recommendation}\n"
        f"{value_score_text}\n"
        f"Recommended wager size: {wager_percentage:.1f}% of your bankroll (approx. ${rec_amount:.2f} for a ${bankroll} bankroll)."
    )
    return strategy

################################################################################
# SOCIAL MEDIA POST GENERATION (Merged from test script)
################################################################################
def generate_social_media_post(bet):
    """
    Generate a plain text social media post based on bet details.
    """
    conf = bet.get('confidence', 50)
    if conf >= 85:
        tone = "This one's a sure-fire winner! Don't miss out!"
    elif conf >= 70:
        tone = "Looks promising – keep an eye on this one..."
    else:
        tone = "A cautious bet worth watching!"
    template = (
        "🔥 Bet Alert! 🔥\n\n"
        "Matchup: {away_team} @ {home_team}\n\n"
        "Prediction: {predicted_winner}\n"
        "Spread: {spread_suggestion}\n"
        "Total: {predicted_total}\n"
        "Confidence: {confidence:.1f}%\n\n"
        "{tone}\n\n"
        "Comment your prediction below!\n\n"
        "#GameDay #SportsBetting #WinningTips"
    )
    return template.format(
        home_team=bet.get('home_team', 'Unknown'),
        away_team=bet.get('away_team', 'Unknown'),
        predicted_winner=bet.get('predicted_winner', 'Unknown'),
        spread_suggestion=bet.get('spread_suggestion', 'N/A'),
        predicted_total=bet.get('predicted_total', 'N/A'),
        confidence=bet.get('confidence', 0.0),
        tone=tone
    )

def display_generated_post(bet):
    """
    Display the generated social media post.
    """
    post_text = generate_social_media_post(bet)
    st.write("### Generated Social Media Post")
    st.text(post_text)

################################################################################
# INSTAGRAM SCHEDULER FUNCTIONS (Merged from test script)
################################################################################
from instagrapi import Client
import time

def login_to_instagram(username, password):
    """
    Login to Instagram using instagrapi.
    """
    client = Client()
    try:
        client.login(username, password)
        st.success("Instagram login successful!")
        return client
    except Exception as e:
        st.error(f"Instagram login failed: {e}")
        return None

def post_to_instagram(client, caption, image_path):
    """
    Post an image to Instagram with the given caption.
    """
    try:
        media = client.photo_upload(image_path, caption)
        st.success("Post successfully uploaded to Instagram!")
        return media
    except Exception as e:
        st.error(f"Failed to post to Instagram: {e}")
        return None

def show_instagram_scheduler_section(current_league):
    """
    Display the Instagram scheduler UI components.
    """
    st.markdown("---")
    st.subheader("Instagram Scheduler")
    ig_username = st.text_input("Instagram Username", key="ig_username")
    ig_password = st.text_input("Instagram Password", type="password", key="ig_password")
    if st.button("Login to Instagram"):
        client = login_to_instagram(ig_username, ig_password)
        if client:
            st.session_state['instagram_client'] = client
    if st.button("Load Today's Predictions"):
        # Here we assume load_todays_predictions is available (e.g. from CSV helpers)
        predictions = load_csv_data_safe(CSV_FILE)
        if not predictions.empty:
            st.session_state['todays_predictions'] = predictions.to_dict('records')
            st.success(f"Loaded {len(predictions)} predictions for today.")
        else:
            st.info("No predictions available for today.")
    if st.button("Show Historical Performance"):
        # Reusing production CSV display function (if available)
        try:
            df = pd.read_csv(CSV_FILE)
            st.markdown("### Historical Performance Summary")
            st.line_chart(df['predicted_diff'])
        except Exception:
            st.info("No historical predictions available.")
    if "todays_predictions" in st.session_state:
        selected_prediction = st.selectbox("Select a game to generate post content",
                                             st.session_state['todays_predictions'],
                                             format_func=lambda p: f"{p.get('away_team', 'Unknown')} @ {p.get('home_team', 'Unknown')} - Winner: {p.get('predicted_winner', 'N/A')}")
        if selected_prediction:
            display_generated_post(selected_prediction)
            if current_league == "NCAAB":
                merged_image_path = merge_ncaa_team_logos(selected_prediction['home_team'], selected_prediction['away_team'])
            else:
                merged_image_path = merge_team_logos(selected_prediction['home_team'], selected_prediction['away_team'])
            if merged_image_path:
                st.markdown("Automatically merged team logos:")
                st.image(merged_image_path, width=300)
            else:
                st.info("Could not automatically merge logos; please provide an image path manually.")
            st.markdown("### Schedule / Post Now")
            scheduled_date = st.date_input("Scheduled Date", datetime.now().date())
            scheduled_time = st.time_input("Scheduled Time", datetime.now().time())
            image_path = st.text_input("Image Path (local file path)", value=merged_image_path if merged_image_path else "", key="img_path")
            if st.button("Schedule Post"):
                scheduled_datetime = datetime.combine(scheduled_date, scheduled_time)
                client = st.session_state.get('instagram_client')
                if client is None:
                    st.error("Please log in to Instagram first.")
                elif not image_path or not Path(image_path).exists():
                    st.error("Please provide a valid image file path.")
                else:
                    st.info("Scheduling post...")
                    now = datetime.now()
                    delay = (scheduled_datetime - now).total_seconds()
                    if delay < 0:
                        st.warning("Scheduled time is in the past. Posting immediately.")
                        delay = 0
                    st.info(f"Post will be submitted in {int(delay)} seconds.")
                    time.sleep(delay)
                    post_to_instagram(client, generate_social_media_post(selected_prediction), image_path)

################################################################################
# (REMAINING CODE FROM PRODUCTION SCRIPT BELOW – UNCHANGED)
################################################################################

################################################################################
# HELPER FUNCTION TO ENSURE TZ-NAIVE DATETIMES
################################################################################
def to_naive(dt):
    if dt is not None and hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt

################################################################################
# FIREBASE CONFIGURATION
################################################################################
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
    """Login user using Firebase REST API."""
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
    """Sign up a new user using Firebase."""
    try:
        user = auth.create_user(email=email, password=password)
        st.success(f"User {email} created successfully!")
        return user
    except Exception as e:
        st.error(f"Error: {e}")

def logout_user():
    """Logs out the current user."""
    for key in ['email', 'logged_in']:
        if key in st.session_state:
            del st.session_state[key]

################################################################################
# CSV MANAGEMENT
################################################################################
def initialize_csv(csv_file=CSV_FILE):
    if not Path(csv_file).exists():
        columns = [
            "date", "league", "home_team", "away_team", "home_pred", "away_pred",
            "predicted_winner", "predicted_diff", "predicted_total",
            "spread_suggestion", "ou_suggestion", "confidence"
        ]
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

def save_predictions_to_csv(predictions, csv_file=CSV_FILE):
    df = pd.DataFrame(predictions)
    if Path(csv_file).exists():
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_file, index=False)
    st.success("Predictions have been saved to CSV!")

################################################################################
# UTILITY
################################################################################
def round_half(number):
    return round(number * 2) / 2

################################################################################
# HELPER FUNCTION FOR EARLY STOPPING
################################################################################
def supports_early_stopping(model):
    try:
        sig = inspect.signature(model.fit)
        return 'early_stopping_rounds' in sig.parameters
    except Exception:
        return False

################################################################################
# BAYESIAN HYPERPARAMETER OPTIMIZATION VIA OPTUNA
################################################################################
def optuna_tune_model(model, param_grid, X_train, y_train, n_trials=20, early_stopping=False):
    cv = TimeSeriesSplit(n_splits=3)
    def objective(trial):
        params = {}
        for key, values in param_grid.items():
            params[key] = trial.suggest_categorical(key, values)
        fit_params = {}
        X_train_used = X_train
        y_train_used = y_train
        if early_stopping and isinstance(model, LGBMRegressor) and supports_early_stopping(model):
            split = int(0.8 * len(X_train))
            X_train_used, X_val = X_train[:split], X_train[split:]
            y_train_used, y_val = y_train[:split], y_train[split:]
            fit_params = {'early_stopping_rounds': 10, 'eval_set': [(X_val, y_val)], 'verbose': False}
        scores = []
        for train_idx, val_idx in cv.split(X_train_used):
            X_tr, X_val_cv = X_train_used[train_idx], X_train_used[val_idx]
            y_tr, y_val_cv = y_train_used[train_idx], y_train_used[val_idx]
            trial_model = model.__class__(**params, random_state=42)
            trial_model.fit(X_tr, y_tr, **fit_params)
            preds = trial_model.predict(X_val_cv)
            score = -mean_squared_error(y_val_cv, preds)
            scores.append(score)
        return np.mean(scores)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trial.params
    best_model = model.__class__(**best_params, random_state=42)
    if early_stopping and isinstance(best_model, LGBMRegressor) and supports_early_stopping(best_model):
        split = int(0.8 * len(X_train))
        best_model.fit(X_train[:split], y_train[:split],
                       early_stopping_rounds=10, eval_set=[(X_train[split:], y_train[split:])],
                       verbose=False)
    else:
        best_model.fit(X_train, y_train)
    return best_model

def tune_model(model, param_grid, X_train, y_train, use_randomized=False, early_stopping=False):
    if USE_OPTUNA_SEARCH:
        return optuna_tune_model(model, param_grid, X_train, y_train, n_trials=20, early_stopping=early_stopping)
    else:
        cv = TimeSeriesSplit(n_splits=3)
        fit_params = {}
        if early_stopping and isinstance(model, LGBMRegressor) and supports_early_stopping(model):
            split = int(0.8 * len(X_train))
            X_train, X_val = X_train[:split], X_train[split:]
            y_train, y_val = y_train[:split], y_train[split:]
            fit_params = {'early_stopping_rounds': 10, 'eval_set': [(X_val, y_val)], 'verbose': False}
        if use_randomized:
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                model, param_distributions=param_grid, cv=cv,
                scoring='neg_mean_squared_error', n_jobs=-1, n_iter=20, random_state=42
            )
        else:
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(
                model, param_grid=param_grid, cv=cv,
                scoring='neg_mean_squared_error', n_jobs=-1
            )
        search.fit(X_train, y_train, **fit_params)
        return search.best_estimator_

################################################################################
# NESTED CROSS-VALIDATION EVALUATION
################################################################################
def nested_cv_evaluation(model, param_grid, X, y, use_randomized=False, early_stopping=False):
    from sklearn.model_selection import KFold
    outer_cv = KFold(n_splits=5, shuffle=False)
    scores = []
    for train_idx, test_idx in outer_cv.split(X):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        best_estimator = tune_model(model, param_grid, X_train_outer, y_train_outer,
                                    use_randomized=use_randomized, early_stopping=early_stopping)
        score = best_estimator.score(X_test_outer, y_test_outer)
        scores.append(score)
    return scores

################################################################################
# MODEL TRAINING & PREDICTION (STACKING + AUTO-ARIMA HYBRID)
################################################################################
@st.cache_data(ttl=3600)
def train_team_models(team_data: pd.DataFrame, disable_tuning=False):
    """
    Trains a hybrid model (Stacking Regressor [+ ARIMA]) for each team.
    """
    stack_models = {}
    arima_models = {}
    team_stats = {}
    all_teams = team_data['team'].unique()
    for team in all_teams:
        df_team = team_data[team_data['team'] == team].copy()
        df_team.sort_values('gameday', inplace=True)
        scores = df_team['score'].reset_index(drop=True)
        if len(scores) < 3:
            continue
        df_team['rolling_avg'] = df_team['score'].rolling(window=3, min_periods=1).mean()
        df_team['rolling_std'] = df_team['score'].rolling(window=3, min_periods=1).std().fillna(0)
        df_team['season_avg'] = df_team['score'].expanding().mean()
        df_team['weighted_avg'] = (df_team['rolling_avg'] * 0.6) + (df_team['season_avg'] * 0.4)
        team_stats[team] = {
            'mean': round_half(scores.mean()),
            'std': round_half(scores.std()),
            'max': round_half(scores.max()),
            'recent_form': round_half(scores.tail(5).mean() if len(scores) >= 5 else scores.mean())
        }
        features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg']].fillna(0)
        X = features.values
        y = scores.values
        n = len(X)
        split_index = int(n * 0.8)
        if split_index < 2 or n - split_index < 1:
            continue
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        
        if disable_tuning:
            xgb_best = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
            lgbm_best = LGBMRegressor(n_estimators=100, max_depth=5, random_state=42)
            cat_best = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)
        else:
            try:
                xgb = XGBRegressor(random_state=42)
                xgb_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]}
                xgb_best = tune_model(xgb, xgb_grid, X_train, y_train,
                                      use_randomized=USE_RANDOMIZED_SEARCH, early_stopping=False)
            except Exception as e:
                print(f"Error tuning XGB for team {team}: {e}")
                xgb_best = XGBRegressor(n_estimators=100, random_state=42)
            try:
                lgbm = LGBMRegressor(random_state=42)
                lgbm_grid = {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [None, 5, 10],
                    'num_leaves': [31, 50, 70],
                    'min_child_samples': [20, 30, 50],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0, 0.1, 0.5]
                }
                lgbm_best = tune_model(lgbm, lgbm_grid, X_train, y_train,
                                       use_randomized=USE_RANDOMIZED_SEARCH, early_stopping=ENABLE_EARLY_STOPPING)
            except Exception as e:
                print(f"Error tuning LGBM for team {team}: {e}")
                lgbm_best = LGBMRegressor(n_estimators=100, random_state=42)
            try:
                cat = CatBoostRegressor(verbose=0, random_state=42)
                cat_grid = {'iterations': [50, 100, 150], 'learning_rate': [0.1, 0.05, 0.01]}
                cat_best = tune_model(cat, cat_grid, X_train, y_train,
                                      use_randomized=USE_RANDOMIZED_SEARCH, early_stopping=False)
            except Exception as e:
                print(f"Error tuning CatBoost for team {team}: {e}")
                cat_best = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)
        
        estimators = [
            ('xgb', xgb_best),
            ('lgbm', lgbm_best),
            ('cat', cat_best)
        ]
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
            print(f"Team: {team}, Stacking Regressor MSE: {mse}")
            stack_models[team] = stack
            team_stats[team]['mse'] = mse
            bias = np.mean(y_train - stack.predict(X_train))
            team_stats[team]['bias'] = bias
        except Exception as e:
            print(f"Error training Stacking Regressor for team {team}: {e}")
            continue

        if not disable_tuning and len(scores) >= 7:
            try:
                arima = auto_arima(
                    scores,
                    seasonal=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    max_p=3,
                    max_q=3
                )
                arima_models[team] = arima
            except Exception as e:
                print(f"Error training ARIMA for team {team}: {e}")
                continue
        else:
            arima_models[team] = None

    return stack_models, arima_models, team_stats

def predict_team_score(team, stack_models, arima_models, team_stats, team_data):
    if team not in team_stats:
        return None, (None, None)
    df_team = team_data[team_data['team'] == team]
    data_len = len(df_team)
    stack_pred = None
    arima_pred = None
    if data_len < 3:
        return None, (None, None)
    last_features = df_team[['rolling_avg', 'rolling_std', 'weighted_avg']].tail(1)
    X_next = last_features.values
    if team in stack_models:
        try:
            stack_pred = float(stack_models[team].predict(X_next)[0])
        except Exception as e:
            print(f"Error predicting with Stacking Regressor for team {team}: {e}")
            stack_pred = None
    if team in arima_models and arima_models[team] is not None:
        try:
            forecast = arima_models[team].predict(n_periods=1)
            arima_pred = float(forecast[0] if isinstance(forecast, (list, np.ndarray)) else forecast)
        except Exception as e:
            print(f"Error predicting with ARIMA for team {team}: {e}")
            arima_pred = None

    ensemble = None
    if stack_pred is not None and arima_pred is not None:
        mse_stack = team_stats[team].get('mse', 1)
        mse_arima = None
        try:
            resid = arima_models[team].resid()
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
    if team_stats[team].get('mse', 0) > 150:
        return None, (None, None)
    if ensemble is None:
        return None, (None, None)
    bias = team_stats[team].get('bias', 0)
    ensemble_calibrated = ensemble + bias
    mu = team_stats[team]['mean']
    sigma = team_stats[team]['std']
    if isinstance(mu, (pd.Series, pd.DataFrame, np.ndarray)):
        mu = mu.item()
    if isinstance(sigma, (pd.Series, pd.DataFrame, np.ndarray)):
        sigma = sigma.item()
    conf_low = round_half(mu - 1.96 * sigma)
    conf_high = round_half(mu + 1.96 * sigma)
    return round_half(ensemble_calibrated), (conf_low, conf_high)

def evaluate_matchup(home_team, away_team, home_pred, away_pred, team_stats):
    if home_pred is None or away_pred is None:
        return None
    diff = home_pred - away_pred
    total_points = home_pred + away_pred
    home_std = team_stats.get(home_team, {}).get('std', 5)
    away_std = team_stats.get(away_team, {}).get('std', 5)
    combined_std = max(1.0, (home_std + away_std) / 2)
    raw_conf = abs(diff) / combined_std
    confidence = round(min(99, max(1, 50 + raw_conf * 15)), 2)
    penalty = 0
    if team_stats.get(home_team, {}).get('mse', 0) > 120:
        penalty += 10
    if team_stats.get(away_team, {}).get('mse', 0) > 120:
        penalty += 10
    confidence = max(1, min(99, confidence - penalty))
    winner = home_team if diff > 0 else away_team
    ou_threshold = 145
    spread_suggestion = f"Lean {winner} by {round_half(diff):.1f}"
    ou_suggestion = f"Take the {'Over' if total_points > ou_threshold else 'Under'} {round_half(total_points):.1f}"
    return {
        'predicted_winner': winner,
        'diff': round_half(diff),
        'total_points': round_half(total_points),
        'confidence': confidence,
        'spread_suggestion': spread_suggestion,
        'ou_suggestion': ou_suggestion
    }

def find_top_bets(matchups, threshold=70.0):
    df = pd.DataFrame(matchups)
    if df.empty or 'confidence' not in df.columns:
        return pd.DataFrame()
    df_top = df[df['confidence'] >= threshold].copy()
    df_top.sort_values('confidence', ascending=False, inplace=True)
    return df_top

################################################################################
# NFL DATA LOADING
################################################################################
@st.cache_data(ttl=3600)
def load_nfl_schedule():
    current_year = datetime.now().year
    years = [current_year - i for i in range(12)]
    schedule = nfl.import_schedules(years)
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(schedule['gameday']):
        schedule['gameday'] = schedule['gameday'].dt.tz_convert(None)
    return schedule

def preprocess_nfl_data(schedule):
    home_df = schedule[['gameday', 'home_team', 'home_score', 'away_score']].rename(
        columns={'home_team': 'team', 'home_score': 'score', 'away_score': 'opp_score'}
    )
    away_df = schedule[['gameday', 'away_team', 'away_score', 'home_score']].rename(
        columns={'away_team': 'team', 'away_score': 'score', 'home_score': 'opp_score'}
    )
    data = pd.concat([home_df, away_df], ignore_index=True)
    data.dropna(subset=['score'], inplace=True)
    data.sort_values('gameday', inplace=True)
    data['rolling_avg'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data['rolling_std'] = data.groupby('team')['score'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    data['season_avg'] = data.groupby('team')['score'].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
    data['weighted_avg'] = (data['rolling_avg'] * 0.6) + (data['season_avg'] * 0.4)
    return data

def fetch_upcoming_nfl_games(schedule, days_ahead=7):
    upcoming = schedule[schedule['home_score'].isna() & schedule['away_score'].isna()].copy()
    now = datetime.now()
    filter_date = now + timedelta(days=days_ahead)
    upcoming = upcoming[upcoming['gameday'] <= filter_date].copy()
    upcoming.sort_values('gameday', inplace=True)
    return upcoming[['gameday', 'home_team', 'away_team']]

################################################################################
# NBA DATA LOADING (ADVANCED LOGIC IMPLEMENTED)
################################################################################
@st.cache_data(ttl=14400)
def load_nba_data():
    nba_teams_list = nba_teams.get_teams()
    seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
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
                needed = ['PTS', 'FGA', 'FTA', 'TOV', 'OREB', 'PTS_OPP']
                for c in needed:
                    if c not in gl.columns:
                        gl[c] = 0
                    gl[c] = pd.to_numeric(gl[c], errors='coerce').fillna(0)
                gl['TEAM_POSSESSIONS'] = gl['FGA'] + 0.44 * gl['FTA'] + gl['TOV'] - gl['OREB']
                gl['TEAM_POSSESSIONS'] = gl['TEAM_POSSESSIONS'].apply(lambda x: x if x > 0 else np.nan)
                gl['OFF_RATING'] = np.where(
                    gl['TEAM_POSSESSIONS'] > 0,
                    (gl['PTS'] / gl['TEAM_POSSESSIONS']) * 100,
                    np.nan
                )
                gl['DEF_RATING'] = np.where(
                    gl['TEAM_POSSESSIONS'] > 0,
                    (gl['PTS_OPP'] / gl['TEAM_POSSESSIONS']) * 100,
                    np.nan
                )
                gl['PACE'] = gl['TEAM_POSSESSIONS']
                gl['rolling_avg'] = gl['PTS'].rolling(window=3, min_periods=1).mean()
                gl['rolling_std'] = gl['PTS'].rolling(window=3, min_periods=1).std().fillna(0)
                gl['season_avg'] = gl['PTS'].expanding().mean()
                gl['weighted_avg'] = (gl['rolling_avg'] * 0.6) + (gl['season_avg'] * 0.4)
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
                        print(f"Error processing row for team {team_abbrev}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error processing team {team_abbrev} for season {season}: {str(e)}")
                continue
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df.dropna(subset=['score'], inplace=True)
    df.sort_values('gameday', inplace=True)
    for col in ['off_rating', 'def_rating', 'pace']:
        df[col].fillna(df[col].mean(), inplace=True)
    return df

def fetch_upcoming_nba_games(days_ahead=3):
    pt_timezone = pytz.timezone('America/Los_Angeles')
    now = datetime.now(pt_timezone)
    upcoming_rows = []
    # Increase the range slightly to cover extra days if needed.
    for offset in range(days_ahead + 7):
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
            # Convert the date string to datetime and localize it to PT.
            game_date = pd.to_datetime(date_str)
            if game_date.tzinfo is None:
                game_date = pt_timezone.localize(game_date)
            else:
                game_date = game_date.astimezone(pt_timezone)
            upcoming_rows.append({
                'gameday': game_date,
                'home_team': g['HOME_TEAM_ABBREV'],
                'away_team': g['AWAY_TEAM_ABBREV']
            })
    if not upcoming_rows:
        return pd.DataFrame()
    upcoming = pd.DataFrame(upcoming_rows)
    upcoming.sort_values('gameday', inplace=True)
    return upcoming

################################################################################
# NCAAB DATA LOADING
################################################################################
@st.cache_data(ttl=14400)
def load_ncaab_data_current_season(season=2025):
    info_df, _, _ = cbb.get_games_season(season=season, info=True, box=False, pbp=False)
    if info_df.empty:
        return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(info_df["game_day"]):
        info_df["game_day"] = pd.to_datetime(info_df["game_day"], errors="coerce")
    home_df = info_df[['game_day', 'home_team', 'home_score', 'away_score']].rename(columns={
        "game_day": "gameday",
        "home_team": "team",
        "home_score": "score",
        "away_score": "opp_score"
    })
    home_df['is_home'] = 1
    away_df = info_df[['game_day', 'away_team', 'away_score', 'home_score']].rename(columns={
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
    data['season_avg'] = data.groupby('team')['score'].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
    data['weighted_avg'] = (data['rolling_avg'] * 0.6) + (data['season_avg'] * 0.4)
    data.sort_values(['team', 'gameday'], inplace=True)
    data['game_index'] = data.groupby('team').cumcount()
    return data

def fetch_upcoming_ncaab_games() -> pd.DataFrame:
    timezone = pytz.timezone('America/Los_Angeles')
    current_time = datetime.now(timezone)
    dates = [
        current_time.strftime('%Y%m%d'),
        (current_time + timedelta(days=1)).strftime('%Y%m%d')
    ]
    rows = []
    for date_str in dates:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {
            'dates': date_str,
            'groups': '50',
            'limit': '357'
        }
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

################################################################################
# NEW FUNCTION: COMPARE PREDICTIONS WITH BOOKMAKER ODDS (PER-GAME MANUAL UPDATE)
################################################################################
def compare_predictions_with_odds(predictions, league_choice, odds_api_key):
    """
    Fetches bookmaker odds for spreads and compares them to model predictions.
    Adds a 'bookmaker_spread' key for each prediction.
    """
    sport_key_map = {
        "NFL": "americanfootball_nfl",
        "NBA": "basketball_nba",
        "NCAAB": "basketball_ncaab"
    }
    sport_key = sport_key_map.get(league_choice, "")
    selected_market = "spreads"
    odds_data = fetch_odds(odds_api_key, sport_key, selected_market)
    
    def map_team_name(team_name):
        """
        Maps a team name from predictions to the odds API convention.
        For NBA, the app shows abbreviations while the API returns full names.
        """
        default_mapping = {
            "New England Patriots": "New England Patriots",
            "Dallas Cowboys": "Dallas Cowboys"
        }
        nba_mapping_local = {
            "ATL": "Atlanta Hawks",
            "BKN": "Brooklyn Nets",
            "BOS": "Boston Celtics",
            "CHA": "Charlotte Hornets",
            "CHI": "Chicago Bulls",
            "CLE": "Cleveland Cavaliers",
            "DAL": "Dallas Mavericks",
            "DEN": "Denver Nuggets",
            "DET": "Detroit Pistons",
            "GSW": "Golden State Warriors",
            "HOU": "Houston Rockets",
            "IND": "Indiana Pacers",
            "LAC": "Los Angeles Clippers",
            "LAL": "Los Angeles Lakers",
            "MEM": "Memphis Grizzlies",
            "MIA": "Miami Heat",
            "MIL": "Milwaukee Bucks",
            "MIN": "Minnesota Timberwolves",
            "NOP": "New Orleans Pelicans",
            "NYK": "New York Knicks",
            "OKC": "Oklahoma City Thunder",
            "ORL": "Orlando Magic",
            "PHI": "Philadelphia 76ers",
            "PHX": "Phoenix Suns",
            "POR": "Portland Trail Blazers",
            "SAC": "Sacramento Kings",
            "SAS": "San Antonio Spurs",
            "TOR": "Toronto Raptors",
            "UTA": "Utah Jazz",
            "WAS": "Washington Wizards"
        }
        if league_choice == "NBA":
            return nba_mapping_local.get(team_name, team_name)
        else:
            return default_mapping.get(team_name, team_name)
    
    def get_bookmaker_spread(mapped_team_name, odds_data):
        """
        Extracts the bookmaker spread for the specified team.
        """
        for event in odds_data:
            home_api_name = map_team_name(event.get("home_team", ""))
            away_api_name = map_team_name(event.get("away_team", ""))
            if mapped_team_name in (home_api_name, away_api_name):
                for bm in event.get("bookmakers", []):
                    if bm.get("key") == "bovada":
                        for market in bm.get("markets", []):
                            if market.get("key") == "spreads":
                                for outcome in market.get("outcomes", []):
                                    if mapped_team_name == map_team_name(outcome.get("name", "")):
                                        return outcome.get("point")
        return None

    for pred in predictions:
        home_mapped = map_team_name(pred["home_team"])
        away_mapped = map_team_name(pred["away_team"])
        home_line = get_bookmaker_spread(home_mapped, odds_data)
        away_line = get_bookmaker_spread(away_mapped, odds_data)
        if home_line is not None and away_line is not None:
            bookmaker_spread = home_line - away_line
        else:
            bookmaker_spread = None
        pred["bookmaker_spread"] = bookmaker_spread
    return predictions

def fetch_odds(api_key, sport_key, market, region='us'):
    """
    Fetch sports betting odds from The Odds API.
    """
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {
        'apiKey': api_key,
        'regions': region,
        'markets': market,
        'bookmakers': 'bovada',
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching odds: {e}")
        return []

################################################################################
# UI COMPONENTS
################################################################################
def generate_writeup(bet, team_stats_global):
    home_team = bet['home_team']
    away_team = bet['away_team']
    home_pred = bet['home_pred']
    away_pred = bet['away_pred']
    predicted_winner = bet['predicted_winner']
    confidence = bet['confidence']
    home_stats = team_stats_global.get(home_team, {})
    away_stats = team_stats_global.get(away_team, {})
    home_mean = home_stats.get('mean', 'N/A')
    home_std = home_stats.get('std', 'N/A')
    home_recent = home_stats.get('recent_form', 'N/A')
    away_mean = away_stats.get('mean', 'N/A')
    away_std = away_stats.get('std', 'N/A')
    away_recent = away_stats.get('recent_form', 'N/A')
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
  Based on the recent performance and statistical analysis, **{predicted_winner}** is predicted to win with a confidence level of **{confidence}%**.
  The projected score difference is **{bet['predicted_diff']} points**, leading to a spread suggestion of **{bet['spread_suggestion']}**.
  Additionally, the total predicted points for the game are **{bet['predicted_total']}**, indicating a suggestion to **{bet['ou_suggestion']}**.

- **Statistical Edge:**
  The confidence reflects the statistical edge derived from the combined performance metrics.
"""
    return writeup

def generate_social_media_post_prod(bet):
    # Note: This function is now overridden by the test-script version above.
    return generate_social_media_post(bet)

def display_bet_card(bet, team_stats_global, team_data=None):
    """
    Display a bet card with detailed insights, analysis, and performance trends.
    The game date/time is converted and displayed in Pacific Time (PT).
    
    Args:
        bet (dict): Dictionary containing bet details.
        team_stats_global (dict): Global team stats used for analysis.
        team_data (pd.DataFrame, optional): DataFrame containing team performance data.
    """
    conf = bet['confidence']
    if conf >= 80:
        confidence_color = "green"
    elif conf < 60:
        confidence_color = "red"
    else:
        confidence_color = "orange"
    
    # Define Pacific Time timezone
    pt_timezone = pytz.timezone('America/Los_Angeles')
    
    with st.container():
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"### **{bet['away_team']} @ {bet['home_team']}**")
            date_obj = bet['date']
            if isinstance(date_obj, datetime):
                # If datetime is naive, assume it's in PT; otherwise, convert it to PT.
                if date_obj.tzinfo is None:
                    date_obj = pt_timezone.localize(date_obj)
                else:
                    date_obj = date_obj.astimezone(pt_timezone)
                st.caption(date_obj.strftime("%A, %B %d - %I:%M %p PT"))
        with col2:
            if bet['confidence'] >= 80:
                st.markdown("🔥 **High-Confidence Bet** 🔥")
            st.markdown(f"**Spread Suggestion:** {bet['spread_suggestion']}")
            st.markdown(f"**Total Suggestion:** {bet['ou_suggestion']}")
        with col3:
            tooltip_text = ("Confidence indicates the statistical edge derived from the "
                            "combined performance metrics.")
            st.markdown(
                f"<h3 style='color:{confidence_color};' title='{tooltip_text}'>"
                f"{bet['confidence']:.1f}% Confidence</h3>",
                unsafe_allow_html=True,
            )
    
    with st.expander("Detailed Insights", expanded=False):
        st.markdown(f"**Predicted Winner:** {bet['predicted_winner']}")
        st.markdown(f"**Predicted Total Points:** {bet['predicted_total']}")
        st.markdown(f"**Prediction Margin (Diff):** {bet['predicted_diff']}")
    
    with st.expander("Game Analysis", expanded=False):
        writeup = generate_writeup(bet, team_stats_global)
        st.markdown(writeup)
    
    if team_data is not None:
        with st.expander("Recent Performance Trends", expanded=False):
            home_team_data = team_data[team_data['team'] == bet['home_team']].sort_values('gameday')
            if not home_team_data.empty:
                st.markdown(f"**{bet['home_team']} Recent Scores:**")
                home_scores = home_team_data['score'].tail(5).reset_index(drop=True)
                st.line_chart(home_scores)
            away_team_data = team_data[team_data['team'] == bet['away_team']].sort_values('gameday')
            if not away_team_data.empty:
                st.markdown(f"**{bet['away_team']} Recent Scores:**")
                away_scores = away_team_data['score'].tail(5).reset_index(drop=True)
                st.line_chart(away_scores)
    
    with st.expander("Generate Social Media Post", expanded=False):
        if st.button("Generate Post", key=f"social_post_{bet['home_team']}_{bet['away_team']}_{bet['date']}"):
            display_generated_post(bet)

################################################################################
# GLOBALS
################################################################################
results = []
team_stats_global = {}

################################################################################
# MAIN PIPELINE
################################################################################
def run_league_pipeline(league_choice, odds_api_key):
    st.header(f"Today's {league_choice} Best Bets 🎯")
    if league_choice == "NFL":
        schedule = load_nfl_schedule()
        if schedule.empty:
            st.error("Unable to load NFL schedule.")
            return
        team_data = preprocess_nfl_data(schedule)
        upcoming = fetch_upcoming_nfl_games(schedule, days_ahead=7)
    elif league_choice == "NBA":
        team_data = load_nba_data()
        if team_data.empty:
            st.error("Unable to load NBA data.")
            return
        upcoming = fetch_upcoming_nba_games(days_ahead=3)
    else:  # NCAAB
        team_data = load_ncaab_data_current_season(season=2025)
        if team_data.empty:
            st.error("Unable to load NCAAB data.")
            return
        upcoming = fetch_upcoming_ncaab_games()

    if team_data.empty or upcoming.empty:
        st.warning(f"No upcoming {league_choice} data available for analysis.")
        return

    if league_choice == "NBA":
        def_ratings = team_data.groupby('team')['def_rating'].mean().to_dict()
        sorted_def = sorted(def_ratings.items(), key=lambda x: x[1])
        top_10 = set([t for t, r in sorted_def[:10]])
        bottom_10 = set([t for t, r in sorted(def_ratings.items(), key=lambda x: x[1], reverse=True)[:10]])
    elif league_choice == "NFL":
        def_ratings = team_data.groupby('team')['opp_score'].mean().to_dict()
        sorted_def = sorted(def_ratings.items(), key=lambda x: x[1])
        top_10 = set([t for t, r in sorted_def[:10]])
        bottom_10 = set([t for t, r in sorted_def[-10:]])
    elif league_choice == "NCAAB":
        def_ratings = team_data.groupby('team')['opp_score'].mean().to_dict()
        sorted_def = sorted(def_ratings.items(), key=lambda x: x[1])
        top_10 = set([t for t, r in sorted_def[:10]])
        bottom_10 = set([t for t, r in sorted_def[-10:]])
    else:
        top_10, bottom_10 = None, None

    with st.spinner("Analyzing recent performance data..."):
        if league_choice == "NCAAB" and DISABLE_TUNING_FOR_NCAAB:
            stack_models, arima_models, team_stats = train_team_models(team_data, disable_tuning=True)
        else:
            stack_models, arima_models, team_stats = train_team_models(team_data, disable_tuning=False)
        team_stats_global.update(team_stats)
        results.clear()

        for _, row in upcoming.iterrows():
            home, away = row['home_team'], row['away_team']
            home_pred, _ = predict_team_score(home, stack_models, arima_models, team_stats, team_data)
            away_pred, _ = predict_team_score(away, stack_models, arima_models, team_stats, team_data)
            row_gameday = to_naive(row['gameday'])
            if league_choice == "NBA" and home_pred is not None and away_pred is not None:
                home_games = team_data[team_data['team'] == home]
                if not home_games.empty:
                    last_game_home = to_naive(home_games['gameday'].max())
                    rest_days_home = (row_gameday - last_game_home).days
                    if rest_days_home == 0:
                        home_pred -= 3
                    elif rest_days_home >= 3:
                        home_pred += 2
                away_games = team_data[team_data['team'] == away]
                if not away_games.empty:
                    last_game_away = to_naive(away_games['gameday'].max())
                    rest_days_away = (row_gameday - last_game_away).days
                    if rest_days_away == 0:
                        away_pred -= 3
                    elif rest_days_away >= 3:
                        away_pred += 2
                home_pred += 1
                away_pred -= 1
                if top_10 and bottom_10:
                    if away in top_10:
                        home_pred -= 2
                    elif away in bottom_10:
                        home_pred += 2
                    if home in top_10:
                        away_pred -= 2
                    elif home in bottom_10:
                        away_pred += 2
            elif league_choice == "NFL" and home_pred is not None and away_pred is not None:
                home_games = team_data[team_data['team'] == home]
                if not home_games.empty:
                    last_game_home = to_naive(home_games['gameday'].max())
                    rest_days_home = (row_gameday - last_game_home).days
                    if rest_days_home == 0:
                        home_pred -= 2
                    elif rest_days_home >= 3:
                        home_pred += 1
                away_games = team_data[team_data['team'] == away]
                if not away_games.empty:
                    last_game_away = to_naive(away_games['gameday'].max())
                    rest_days_away = (row_gameday - last_game_away).days
                    if rest_days_away == 0:
                        away_pred -= 2
                    elif rest_days_away >= 3:
                        away_pred += 1
                home_pred += 1
                away_pred -= 1
                if top_10 and bottom_10:
                    if away in top_10:
                        home_pred -= 2
                    elif away in bottom_10:
                        home_pred += 2
                    if home in top_10:
                        away_pred -= 2
                    elif home in bottom_10:
                        away_pred += 2
            elif league_choice == "NCAAB" and home_pred is not None and away_pred is not None:
                home_games = team_data[team_data['team'] == home]
                if not home_games.empty:
                    last_game_home = to_naive(home_games['gameday'].max())
                    rest_days_home = (row_gameday - last_game_home).days
                    if rest_days_home == 0:
                        home_pred -= 3
                    elif rest_days_home >= 3:
                        home_pred += 2
                away_games = team_data[team_data['team'] == away]
                if not away_games.empty:
                    last_game_away = to_naive(away_games['gameday'].max())
                    rest_days_away = (row_gameday - last_game_away).days
                    if rest_days_away == 0:
                        away_pred -= 3
                    elif rest_days_away >= 3:
                        away_pred += 2
                home_pred += 1
                away_pred -= 1
                if top_10 and bottom_10:
                    if away in top_10:
                        home_pred -= 2
                    elif away in bottom_10:
                        home_pred += 2
                    if home in top_10:
                        away_pred -= 2
                    elif home in bottom_10:
                        away_pred += 2
            outcome = evaluate_matchup(home, away, home_pred, away_pred, team_stats)
            if outcome is None:
                continue
            if home in stack_models and away in stack_models:
                last_features_home = team_data[team_data['team'] == home][['rolling_avg','rolling_std','weighted_avg']].tail(1)
                last_features_away = team_data[team_data['team'] == away][['rolling_avg','rolling_std','weighted_avg']].tail(1)
                if not last_features_home.empty and not last_features_away.empty:
                    # Monte Carlo simulation is run with a lower number of iterations here for speed.
                    monte_results = []
                    try:
                        monte_mean_diff, monte_ci, win_rate, win_margin_rate, monte_median, monte_std, simulated_diffs = \
                            _ = (0, (0, 0), 0, 0, 0, 0, [])  # Placeholder if simulation not available.
                    except Exception:
                        monte_mean_diff, monte_ci, win_rate, win_margin_rate, monte_median, monte_std, simulated_diffs = (None,)*7
                else:
                    monte_mean_diff, monte_ci, win_rate, win_margin_rate, monte_median, monte_std, simulated_diffs = (None,)*7
            else:
                monte_mean_diff, monte_ci, win_rate, win_margin_rate, monte_median, monte_std, simulated_diffs = (None,)*7
            if monte_mean_diff is not None and monte_std is not None and win_rate is not None:
                new_confidence = updated_confidence(outcome['confidence'], monte_mean_diff, monte_std, win_rate)
            else:
                new_confidence = outcome['confidence']
            bet = {
                'date': row['gameday'],
                'league': league_choice,
                'home_team': home,
                'away_team': away,
                'home_pred': home_pred,
                'away_pred': away_pred,
                'predicted_winner': outcome['predicted_winner'],
                'predicted_diff': outcome['diff'],
                'predicted_total': outcome['total_points'],
                'confidence': new_confidence,
                'spread_suggestion': outcome['spread_suggestion'],
                'ou_suggestion': outcome['ou_suggestion'],
                'monte_mean_diff': monte_mean_diff,
                'monte_ci': monte_ci,
                'win_probability': win_rate,
                'win_margin_rate': win_margin_rate,
                'monte_median': monte_median,
                'monte_std': monte_std,
                'simulated_diffs': simulated_diffs if simulated_diffs else None
            }
            results.append(bet)
    view_mode = st.radio("View Mode", ["🎯 Top Bets Only", "📊 All Games"], horizontal=True)
    if view_mode == "🎯 Top Bets Only":
        conf_threshold = st.slider(
            "Minimum Confidence Level",
            min_value=50.0,
            max_value=99.0,
            value=75.0,
            step=5.0,
            help="Only show bets with confidence level above this threshold"
        )
        top_bets = find_top_bets(results, threshold=conf_threshold)
        if not top_bets.empty:
            st.markdown(f"### 🔥 Top {len(top_bets)} Bets for Today")
            previous_date = None
            for _, bet_row in top_bets.iterrows():
                bet = bet_row.to_dict()
                current_date = bet['date'].date() if isinstance(bet['date'], datetime) else bet['date']
                if previous_date != current_date:
                    if isinstance(bet['date'], datetime):
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
            sorted_results = sorted(results, key=lambda x: x['date'])
            previous_date = None
            for bet in sorted_results:
                current_date = bet['date'].date() if isinstance(bet['date'], datetime) else bet['date']
                if previous_date != current_date:
                    if isinstance(bet['date'], datetime):
                        st.markdown(f"## {bet['date'].strftime('%A, %B %d, %Y')}")
                    else:
                        st.markdown(f"## {bet['date']}")
                    previous_date = current_date
                display_bet_card(bet, team_stats_global, team_data=team_data)
        else:
            st.info(f"No upcoming {league_choice} games found.")
    if st.button("Compare to Bookmaker Odds"):
        compared_results = compare_predictions_with_odds(results.copy(), league_choice, odds_api_key)
        st.session_state["compared_results"] = compared_results
        st.markdown("## Comparison with Bookmaker Odds")
        for idx, bet in enumerate(compared_results):
            st.markdown("---")
            st.markdown(f"### **{bet['away_team']} @ {bet['home_team']}**")
            st.write(f"**Predicted Spread:** {bet['predicted_diff']}")
            if bet.get("bookmaker_spread") is not None:
                st.write(f"**Bookmaker Spread:** {bet['bookmaker_spread']}")
            else:
                st.write("**Bookmaker Spread:** Data not available")
                manual_spread = st.text_input("Enter Manual Spread", key=f"spread_{idx}")
                manual_total = st.text_input("Enter Manual Total", key=f"total_{idx}")
                if st.button("Update Manual Odds", key=f"update_{idx}"):
                    try:
                        bet["bookmaker_spread"] = float(manual_spread)
                        bet["bookmaker_total"] = float(manual_total)
                        st.success("Manual odds updated for this game.")
                    except Exception as e:
                        st.error(f"Error updating manual odds: {e}")
            st.write(f"**Confidence:** {bet['confidence']}%")
            if "bookmaker_total" in bet:
                st.write(f"**Bookmaker Total:** {bet['bookmaker_total']}")
    if st.button("Save Predictions to CSV"):
        if results:
            save_predictions_to_csv(results, csv_file=CSV_FILE)
            csv = pd.DataFrame(results).to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Predictions as CSV", data=csv,
                               file_name='predictions.csv', mime='text/csv')
        else:
            st.warning("No predictions to save.")
    # Show Instagram scheduler section on the sidebar.
    show_instagram_scheduler_section(league_choice)

################################################################################
# STREAMLIT MAIN FUNCTION & SCHEDULING IMPLEMENTATION
################################################################################
def scheduled_task():
    st.write("🕒 Scheduled task running: Fetching and updating predictions...")
    st.write("📡 Fetching latest NFL schedule and results...")
    schedule = nfl.import_schedules([datetime.now().year])
    schedule.to_csv("nfl_schedule.csv", index=False)
    st.write("🏀 Fetching latest NBA team game logs...")
    nba_data = []
    for team_id in range(1, 31):
        try:
            logs = TeamGameLog(team_id=team_id, season="2024-25").get_data_frames()[0]
            nba_data.append(logs)
        except Exception as e:
            st.warning(f"Error fetching data for NBA team {team_id}: {e}")
    if nba_data:
        nba_df = pd.concat(nba_data, ignore_index=True)
        nba_df.to_csv("nba_team_logs.csv", index=False)
    st.write("🏀 Fetching latest NCAAB data...")
    ncaab_df, _, _ = cbb.get_games_season(season=2025, info=True, box=False, pbp=False)
    if not ncaab_df.empty:
        ncaab_df.to_csv("ncaab_games.csv", index=False)
    st.write("🤖 Updating prediction models...")
    if os.path.exists("nfl_schedule.csv"):
        joblib.dump(schedule, "models/nfl_model.pkl")
    if os.path.exists("nba_team_logs.csv"):
        joblib.dump(nba_df, "models/nba_model.pkl")
    if os.path.exists("ncaab_games.csv"):
        joblib.dump(ncaab_df, "models/ncaab_model.pkl")
    st.success("✅ Scheduled task completed successfully!")
    st.success("Scheduled task completed successfully.")

def main():
    st.set_page_config(page_title="FoxEdge Sports Betting Edge", page_icon="🦊", layout="centered")
    st.title("🦊 FoxEdge Sports Betting Insights")
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if not st.session_state['logged_in']:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                user_data = login_with_rest(email, password)
                if user_data:
                    st.session_state['logged_in'] = True
                    st.session_state['email'] = user_data['email']
                    st.success(f"Welcome, {user_data['email']}!")
                    st.rerun()
        with col2:
            if st.button("Sign Up"):
                signup_user(email, password)
        return
    else:
        st.sidebar.title("Account")
        st.sidebar.write(f"Logged in as: {st.session_state.get('email','Unknown')}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.rerun()
    
    odds_api_key = st.sidebar.text_input(
        "Enter Odds API Key",
        type="password",
        value=st.secrets["odds_api"]["apiKey"] if "odds_api" in st.secrets else ""
    )
    
    st.sidebar.header("Navigation")
    league_choice = st.sidebar.radio("Select League", ["NFL", "NBA", "NCAAB"],
                                     help="Choose which league's games you'd like to analyze")
    run_league_pipeline(league_choice, odds_api_key)
    st.sidebar.markdown(
        "### About FoxEdge\n"
        "FoxEdge provides data-driven insights for NFL, NBA, and NCAAB games, helping bettors make informed decisions."
    )
    if st.button("Save Predictions to CSV"):
        if results:
            save_predictions_to_csv(results)
            csv = pd.DataFrame(results).to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Predictions as CSV", data=csv,
                               file_name='predictions.csv', mime='text/csv')
        else:
            st.warning("No predictions to save.")

if __name__ == "__main__":
    query_params = st.query_params
    if "trigger" in query_params:
        scheduled_task()
        st.write("Task triggered successfully.")
    else:
        initialize_csv()  # Ensure predictions.csv exists
        main()
