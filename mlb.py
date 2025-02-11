import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
import xgboost as xgb
from pybaseball import (
    statcast, batting_stats, pitching_stats, schedule_and_record, standings
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import requests  # For real-time odds

# Streamlit Configurations
st.set_page_config(page_title="MLB AI Betting Predictor", layout="wide")

# Title
st.title("⚾ MLB AI Betting Strategy App")

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Fetch Data", "Train AI Model", "Make Predictions", "AI Betting Strategies"])

# MLB Teams List
MLB_TEAMS = ["ATL", "ARI", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET", "HOU",
             "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK", "PHI", "PIT",
             "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH"]

# Function to fetch data
@st.cache_data
def fetch_mlb_data():
    st.write("📡 Fetching MLB data, please wait...")
    pbp_data = statcast()
    batting_data = batting_stats(1871, 2024)
    pitching_data = pitching_stats(1871, 2024)
    return pbp_data, batting_data, pitching_data

# Function to process data
@st.cache_data
def process_data(batting_data, pitching_data):
    st.write("🔍 Processing & Cleaning Data for AI Training...")
    features = ["OPS", "WAR", "ERA", "FIP", "W-L%", "Run Differential"]
    dataset = batting_data.merge(pitching_data, on="Team", how="left")
    dataset = dataset[features]
    dataset.dropna(inplace=True)
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset)
    return dataset_scaled, scaler

# Function to fetch real-time betting odds (Mock API)
@st.cache_data
def fetch_betting_odds():
    # Ideally, replace this with a real API
    url = "https://example.com/api/mlb/odds"  # Replace with actual API
    try:
        response = requests.get(url)
        data = response.json()
        odds_df = pd.DataFrame(data)
        odds_df.to_csv("mlb_betting_odds.csv", index=False)
    except:
        odds_df = pd.read_csv("mock_mlb_odds.csv")  # Fallback to local mock odds
    return odds_df

# Function to calculate expected value (EV) and Kelly Criterion
def calculate_betting_value(predictions, odds_df):
    st.write("📈 Analyzing AI Predictions & Betting Odds...")

    # Merge AI predictions with betting odds
    merged_df = predictions.merge(odds_df, on="Team", how="left")

    # Calculate **Implied Probability** from odds
    merged_df["Implied_Prob"] = 1 / merged_df["Moneyline_Odds"]

    # Calculate **Expected Value (EV)**
    merged_df["EV"] = (merged_df["Win_Probability"] * merged_df["Moneyline_Odds"]) - (1 - merged_df["Win_Probability"])

    # Calculate **Kelly Criterion (for bankroll management)**
    merged_df["Kelly_Bet"] = (merged_df["Win_Probability"] - merged_df["Implied_Prob"]) / merged_df["Implied_Prob"]

    # Filter for **high-value bets**
    value_bets = merged_df[(merged_df["EV"] > 0) & (merged_df["Kelly_Bet"] > 0)]
    
    return value_bets

# Function to display betting recommendations
def ai_betting_recommendations():
    st.write("## 🎯 AI-Driven Betting Strategies")

    # Load AI predictions
    predictions_df = pd.read_csv("mlb_game_predictions.csv")
    
    # Load betting odds
    odds_df = fetch_betting_odds()

    # Calculate betting value
    value_bets = calculate_betting_value(predictions_df, odds_df)

    # Display Value Bets
    st.write("### 🏆 AI-Recommended Bets")
    st.dataframe(value_bets[["Team", "Win_Probability", "Moneyline_Odds", "EV", "Kelly_Bet"]])

    # Visualization
    st.write("### 📊 Betting Strategy Insights")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Team", y="EV", data=value_bets, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Betting Strategies Page
if page == "AI Betting Strategies":
    ai_betting_recommendations()
