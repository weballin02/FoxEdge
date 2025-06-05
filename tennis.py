import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from bs4 import BeautifulSoup
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# –– ODDS SCRAPING MODULE ––

class OddsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.odds_cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        
    def _normalize_player_name(self, name):
        """Normalize player names for matching"""
        return re.sub(r'[^\w\s]', '', name.lower().strip())
        
    def _is_cache_valid(self, cache_key):
        """Check if cached odds are still valid"""
        if cache_key not in self.odds_cache:
            return False
        return time.time() - self.odds_cache[cache_key]['timestamp'] < self.cache_timeout
        
    def fetch_odds_betexplorer(self, player1, player2):
        """Fetch odds from BetExplorer"""
        try:
            # Search for the match
            search_url = f"https://www.betexplorer.com/tennis/"
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for match links containing both players
            p1_norm = self._normalize_player_name(player1)
            p2_norm = self._normalize_player_name(player2)
            
            # Find match links
            match_links = soup.find_all('a', href=re.compile(r'/tennis/.*?/'))
            for link in match_links:
                link_text = self._normalize_player_name(link.get_text())
                if p1_norm in link_text and p2_norm in link_text:
                    match_url = "https://www.betexplorer.com" + link['href']
                    return self._scrape_betexplorer_match(match_url, player1, player2)
            
            return None
        except Exception as e:
            st.warning(f"BetExplorer scraping failed: {str(e)}")
            return None
        
    def _scrape_betexplorer_match(self, match_url, player1, player2):
        """Scrape odds from a specific BetExplorer match page"""
        try:
            response = self.session.get(match_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            odds_data = {}
            
            # Look for odds table
            odds_table = soup.find('table', {'id': 'odds-data-table'})
            if not odds_table:
                odds_table = soup.find('table', class_=re.compile(r'.*odds.*'))
            
            if odds_table:
                rows = odds_table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:
                        try:
                            bookmaker = cells[0].get_text(strip=True)
                            odds1 = float(cells[1].get_text(strip=True))
                            odds2 = float(cells[2].get_text(strip=True))
                            
                            if bookmaker not in odds_data:
                                odds_data[bookmaker] = {}
                            odds_data[bookmaker] = {
                                'player1_odds': odds1,
                                'player2_odds': odds2,
                                'player1': player1,
                                'player2': player2
                            }
                        except (ValueError, IndexError):
                            continue
            
            return odds_data if odds_data else None
        except Exception as e:
            return None
        
    def fetch_odds_flashscore(self, player1, player2):
        """Fetch odds from Flashscore using Selenium"""
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            driver = webdriver.Chrome(options=options)
            
            # Search for tennis matches
            search_url = f"https://www.flashscore.com/tennis/"
            driver.get(search_url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Look for matches
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            p1_norm = self._normalize_player_name(player1)
            p2_norm = self._normalize_player_name(player2)
            
            # Find match elements
            matches = soup.find_all('div', class_=re.compile(r'.*match.*'))
            
            for match in matches:
                match_text = self._normalize_player_name(match.get_text())
                if p1_norm in match_text and p2_norm in match_text:
                    # Click on match to get odds
                    match_link = match.find('a')
                    if match_link and match_link.get('href'):
                        odds_url = "https://www.flashscore.com" + match_link['href'] + "#odds"
                        driver.get(odds_url)
                        time.sleep(2)
                        
                        # Extract odds
                        odds_soup = BeautifulSoup(driver.page_source, 'html.parser')
                        odds_data = self._parse_flashscore_odds(odds_soup, player1, player2)
                        
                        driver.quit()
                        return odds_data
            
            driver.quit()
            return None
            
        except Exception as e:
            st.warning(f"Flashscore scraping failed: {str(e)}")
            return None
        
    def _parse_flashscore_odds(self, soup, player1, player2):
        """Parse odds from Flashscore page"""
        try:
            odds_data = {}
            
            # Look for odds tables
            odds_sections = soup.find_all('div', class_=re.compile(r'.*odds.*'))
            
            for section in odds_sections:
                rows = section.find_all('div', class_=re.compile(r'.*row.*'))
                
                for row in rows:
                    cells = row.find_all(['div', 'span'])
                    if len(cells) >= 3:
                        try:
                            bookmaker = cells[0].get_text(strip=True)
                            odds1_text = cells[1].get_text(strip=True)
                            odds2_text = cells[2].get_text(strip=True)
                            
                            # Parse odds (handle different formats)
                            odds1 = self._parse_odds_format(odds1_text)
                            odds2 = self._parse_odds_format(odds2_text)
                            
                            if odds1 and odds2:
                                odds_data[bookmaker] = {
                                    'player1_odds': odds1,
                                    'player2_odds': odds2,
                                    'player1': player1,
                                    'player2': player2
                                }
                        except:
                            continue
            
            return odds_data if odds_data else None
        except Exception as e:
            return None
        
    def _parse_odds_format(self, odds_text):
        """Parse different odds formats (decimal, fractional, american)"""
        try:
            # Remove extra characters
            odds_text = re.sub(r'[^\d\.\-\+\/]', '', odds_text)
            
            # Decimal odds
            if '.' in odds_text and not '/' in odds_text:
                return float(odds_text)
            
            # Fractional odds (e.g., "5/2")
            if '/' in odds_text:
                parts = odds_text.split('/')
                if len(parts) == 2:
                    return (float(parts[0]) / float(parts[1])) + 1
            
            # American odds
            if odds_text.startswith(('+', '-')):
                american_odds = int(odds_text)
                if american_odds > 0:
                    return (american_odds / 100) + 1
                else:
                    return (100 / abs(american_odds)) + 1
            
            return None
        except:
            return None
        
    def fetch_aggregated_odds(self, player1, player2):
        """Fetch odds from multiple sources and aggregate"""
        cache_key = f"{self._normalize_player_name(player1)}_{self._normalize_player_name(player2)}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.odds_cache[cache_key]['data']
        
        all_odds = {}
        
        # Fetch from multiple sources concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.fetch_odds_betexplorer, player1, player2): 'BetExplorer',
                executor.submit(self.fetch_odds_flashscore, player1, player2): 'Flashscore',
            }
            
            for future in as_completed(futures, timeout=30):
                source = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_odds[source] = result
                except Exception as e:
                    st.warning(f"Failed to fetch odds from {source}: {str(e)}")
        
        # Aggregate odds
        aggregated = self._aggregate_odds(all_odds, player1, player2)
        
        # Cache the result
        self.odds_cache[cache_key] = {
            'data': aggregated,
            'timestamp': time.time()
        }
        
        return aggregated
        
    def _aggregate_odds(self, all_odds, player1, player2):
        """Aggregate odds from multiple sources"""
        if not all_odds:
            return {
                'best_odds_player1': 2.0,
                'best_odds_player2': 2.0,
                'avg_odds_player1': 2.0,
                'avg_odds_player2': 2.0,
                'bookmaker_count': 0,
                'sources': []
            }
        
        p1_odds = []
        p2_odds = []
        sources = []
        
        for source, bookmakers in all_odds.items():
            sources.append(source)
            for bookmaker, odds in bookmakers.items():
                if isinstance(odds, dict):
                    p1_odds.append(odds.get('player1_odds', 2.0))
                    p2_odds.append(odds.get('player2_odds', 2.0))
        
        if not p1_odds:
            p1_odds = [2.0]
            p2_odds = [2.0]
        
        return {
            'best_odds_player1': max(p1_odds),
            'best_odds_player2': max(p2_odds),
            'avg_odds_player1': sum(p1_odds) / len(p1_odds),
            'avg_odds_player2': sum(p2_odds) / len(p2_odds),
            'bookmaker_count': len(p1_odds),
            'sources': sources,
            'all_odds': all_odds
        }


# –– ATP/WTA Next Matches Scraper ––

def fetch_next_atp_matches(max_days_ahead=7):
    """Finds the next day with main-draw ATP matches on the official ATP site (up to max_days_ahead days ahead)."""
    url = "https://www.atptour.com/en/scores/current"
    for i in range(max_days_ahead):
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        matches = []
        for match in soup.select('.day-table-match'):
            try:
                names = match.select('.day-table-name')
                if len(names) == 2:
                    player1 = names[0].text.strip()
                    player2 = names[1].text.strip()
                    surface = "Hard"
                    matches.append({
                        "Player 1": player1,
                        "Player 2": player2,
                        "Surface": surface,
                    })
            except Exception:
                continue
        if matches:
            # ATP only shows "current", so we can't jump days. Just return if found.
            match_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            return pd.DataFrame(matches), match_date
    return pd.DataFrame(), None

def fetch_next_wta_matches(max_days_ahead=7):
    """Finds the next day with main-draw WTA matches on the official WTA site (up to max_days_ahead days ahead)."""
    url = "https://www.wtatennis.com/scores"
    for i in range(max_days_ahead):
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, 'html.parser')
        matches = []
        for match in soup.select('div.match-card'):
            try:
                names = match.select('.match-card__competitor-name')
                if len(names) == 2:
                    player1 = names[0].text.strip()
                    player2 = names[1].text.strip()
                    surface = "Hard"
                    matches.append({
                        "Player 1": player1,
                        "Player 2": player2,
                        "Surface": surface,
                    })
            except Exception:
                continue
        if matches:
            match_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            return pd.DataFrame(matches), match_date
    return pd.DataFrame(), None

# Page configuration

st.set_page_config(
    page_title="Tennis Betting Predictor Pro",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS

st.markdown("""

<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .value-bet-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        border: 2px solid #ffd700;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .profit-positive {
        color: #28a745;
        font-weight: bold;
    }
    .profit-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .odds-comparison {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>

""", unsafe_allow_html=True)

class TennisPredictor:
    def __init__(self):
            self.model = None
            self.scaler = StandardScaler()
            self.features = []
        
        
    def fetch_atp_rankings(self):
        try:
            url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv"
            rankings = pd.read_csv(url)
            return rankings
        except:
            return pd.DataFrame({
                'ranking_date': [20241202] * 10,
                'rank': range(1, 11),
                'player': ['Novak Djokovic', 'Carlos Alcaraz', 'Daniil Medvedev', 
                          'Jannik Sinner', 'Andrey Rublev', 'Stefanos Tsitsipas',
                          'Holger Rune', 'Casper Ruud', 'Taylor Fritz', 'Alex de Minaur'],
                'points': [9945, 8855, 7600, 7500, 4200, 4000, 3800, 3600, 3400, 3200]
            })
        
    def fetch_wta_rankings(self):
        try:
            url = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_rankings_current.csv"
            rankings = pd.read_csv(url)
            return rankings
        except:
            return pd.DataFrame({
                'ranking_date': [20241202] * 10,
                'rank': range(1, 11),
                'player': ['Iga Swiatek', 'Aryna Sabalenka', 'Coco Gauff', 
                          'Elena Rybakina', 'Jessica Pegula', 'Ons Jabeur',
                          'Karolina Muchova', 'Qinwen Zheng', 'Barbora Krejcikova', 'Danielle Collins'],
                'points': [9665, 8716, 6530, 5471, 4990, 3758, 3348, 3267, 3214, 2854]
            })
        
    def fetch_match_data(self, tour, year):
        """Download match data from Jeff Sackmann's repo for ATP/WTA"""
        try:
            base_url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/{tour}_matches_{year}.csv"
            df = pd.read_csv(base_url)
            df['winner_name'] = df['winner_name'].fillna('')
            df['loser_name'] = df['loser_name'].fillna('')
            return df
        except Exception:
            # Fallback demo data
            return pd.DataFrame({
                'tourney_date': [20240101],
                'winner_name': ['Demo Player1'],
                'loser_name': ['Demo Player2'],
                'surface': ['Hard'],
                'score': ['6-4 6-3'],
            })
        
    def calculate_head_to_head(self, player1, player2, matches_df):
        h2h_matches = matches_df[
            ((matches_df['winner_name'] == player1) & (matches_df['loser_name'] == player2)) |
            ((matches_df['winner_name'] == player2) & (matches_df['loser_name'] == player1))
        ]
        p1_wins = len(h2h_matches[h2h_matches['winner_name'] == player1])
        p2_wins = len(h2h_matches[h2h_matches['winner_name'] == player2])
        return p1_wins, p2_wins
        
    def calculate_surface_stats(self, player, surface, matches_df):
        surface_matches = matches_df[matches_df['surface'] == surface]
        wins = len(surface_matches[surface_matches['winner_name'] == player])
        losses = len(surface_matches[surface_matches['loser_name'] == player])
        total = wins + losses
        win_rate = wins / total if total > 0 else 0.5
        return win_rate, total
        
    def calculate_recent_form(self, player, matches_df, days=30):
        # This is a simplification for the demo. Real logic should parse actual dates.
        wins = len(matches_df[matches_df['winner_name'] == player])
        losses = len(matches_df[matches_df['loser_name'] == player])
        total = wins + losses
        form = wins / total if total > 0 else 0.5
        return form, total
        
    def prepare_features(self, player1, player2, surface, rankings_df, matches_df):
        p1_rank = rankings_df[rankings_df['player'] == player1]['rank'].iloc[0] if len(rankings_df[rankings_df['player'] == player1]) > 0 else 50
        p2_rank = rankings_df[rankings_df['player'] == player2]['rank'].iloc[0] if len(rankings_df[rankings_df['player'] == player2]) > 0 else 50
        p1_points = rankings_df[rankings_df['player'] == player1]['points'].iloc[0] if len(rankings_df[rankings_df['player'] == player1]) > 0 else 1000
        p2_points = rankings_df[rankings_df['player'] == player2]['points'].iloc[0] if len(rankings_df[rankings_df['player'] == player2]) > 0 else 1000
        p1_h2h, p2_h2h = self.calculate_head_to_head(player1, player2, matches_df)
        p1_surface_wr, p1_surface_matches = self.calculate_surface_stats(player1, surface, matches_df)
        p2_surface_wr, p2_surface_matches = self.calculate_surface_stats(player2, surface, matches_df)
        p1_form, p1_recent_matches = self.calculate_recent_form(player1, matches_df)
        p2_form, p2_recent_matches = self.calculate_recent_form(player2, matches_df)
        features = [
            p1_rank, p2_rank,
            p1_points, p2_points,
            p1_h2h, p2_h2h,
            p1_surface_wr, p2_surface_wr,
            p1_surface_matches, p2_surface_matches,
            p1_form, p2_form,
            p1_recent_matches, p2_recent_matches,
            1 if surface == 'Hard' else 0,
            1 if surface == 'Clay' else 0,
            1 if surface == 'Grass' else 0
        ]
        return features
        
    def train_model(self, matches_df, rankings_df):
        X = []
        y = []
        sample_matches = matches_df.sample(min(1000, len(matches_df)))
        for _, match in sample_matches.iterrows():
            try:
                features = self.prepare_features(
                    match['winner_name'], match['loser_name'], 
                    match['surface'], rankings_df, matches_df
                )
                X.append(features)
                y.append(1)
                features_reverse = self.prepare_features(
                    match['loser_name'], match['winner_name'], 
                    match['surface'], rankings_df, matches_df
                )
                X.append(features_reverse)
                y.append(0)
            except Exception:
                continue
        if len(X) == 0:
            return False
        X = np.array(X)
        y = np.array(y)
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]
        y = y[mask]
        if len(X) < 10:
            return False
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        return True
        
    def predict_match(self, player1, player2, surface, rankings_df, matches_df):
        if self.model is None:
            return 0.5, "Model not trained"
        try:
            features = self.prepare_features(player1, player2, surface, rankings_df, matches_df)
            features_scaled = self.scaler.transform([features])
            probability = self.model.predict_proba(features_scaled)[0][1]
            confidence = abs(probability - 0.5) * 2
            return probability, f"Confidence: {confidence:.1%}"
        except Exception as e:
            return 0.5, f"Error: {str(e)}"
        
        
def fetch_all_years_matches(tour, start=2015, end=2025):
    frames = []
    for year in range(start, end + 1):
        try:
            df = TennisPredictor().fetch_match_data(tour, year)
            frames.append(df)
        except Exception:
            continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def calculate_betting_return(probability, odds, stake=100):
    """Calculate expected return with Kelly criterion consideration"""
    if probability > (1 / odds):
        expected_return = (probability * (odds - 1) * stake) - ((1 - probability) * stake)
        # Kelly fraction for optimal bet sizing
        kelly_fraction = (probability * odds - 1) / (odds - 1)
        optimal_stake = max(0, min(kelly_fraction * 100, 25))  # Cap at 25% of bankroll
        return {
            'expected_return': expected_return,
            'kelly_fraction': kelly_fraction,
            'optimal_stake': optimal_stake,
            'is_value_bet': expected_return > 0
        }
    return {
        'expected_return': 0,
        'kelly_fraction': 0,
        'optimal_stake': 0,
        'is_value_bet': False
    }

def project_set_score(prob, is_slam=False):
    if is_slam:
        if prob > 0.8:
            return "3-0"
        elif prob > 0.6:
            return "3-1"
        else:
            return "3-2"
    else:
        if prob > 0.8:
            return "2-0"
        elif prob > 0.6:
            return "2-1"
        else:
            return "2-1"

def get_predictions_for_matches(matches_df, predictor, rankings_df, matches_hist_df, tour_name):
    preds = []
    total_matches = len(matches_df)
    
    if enable_live_odds and total_matches > 0:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for i, row in matches_df.iterrows():
        if "winner_name" in row and "loser_name" in row:
            player1 = row['winner_name']
            player2 = row['loser_name']
            surface = row['surface']
            actual_score = row['score'] if 'score' in row else '-'
        else:
            player1 = row['Player 1']
            player2 = row['Player 2']
            surface = row['Surface']
            actual_score = "-"
        
        if enable_live_odds and total_matches > 0:
            progress = (i + 1) / total_matches
            progress_bar.progress(progress)
            status_text.text(f"Fetching odds for {player1} vs {player2}... ({i+1}/{total_matches})")
        
        odds_data = None
        if enable_live_odds:
            try:
                odds_data = st.session_state.odds_scraper.fetch_aggregated_odds(player1, player2)
            except Exception as e:
                st.warning(f"Failed to fetch odds for {player1} vs {player2}: {str(e)}")
        
        prob, confidence_str = predictor.predict_match(
            player1, player2, surface, rankings_df, matches_hist_df
        )
        
        # Project set score
        is_slam = False  # You can add logic to detect Grand Slam
        proj_score = project_set_score(prob, is_slam)
        
        # Calculate value bet info
        odds = odds_data['best_odds_player1'] if odds_data else 2.0
        value_info = calculate_betting_return(prob, odds)
        
        # Filter by confidence/expected value
        confidence = abs(prob - 0.5) * 2
        if confidence < min_confidence or value_info['expected_return'] < min_expected_value:
            continue
        
        preds.append({
            'player1': player1,
            'player2': player2,
            'surface': surface,
            'probability': prob,
            'confidence': confidence_str,
            'projected_score': proj_score,
            'odds': odds,
            'expected_return': value_info['expected_return'],
            'kelly_fraction': value_info['kelly_fraction'],
            'optimal_stake': value_info['optimal_stake'],
            'is_value_bet': value_info['is_value_bet'],
            'odds_data': odds_data,
            'actual_score': actual_score,
            'tour': tour_name
        })
    return preds

def main():
    st.markdown('<h1 class="main-header">🎾 Tennis Betting Predictor Pro</h1>', unsafe_allow_html=True)
    
    # ... (sidebar and model loading code as above)
    
    st.header("Upcoming ATP Matches")
    atp_matches, atp_date = fetch_next_atp_matches()
    if not atp_matches.empty:
        atp_preds = get_predictions_for_matches(
            atp_matches, st.session_state.atp_predictor,
            st.session_state.atp_rankings, st.session_state.atp_matches, "ATP"
        )
        for pred in atp_preds:
            st.markdown(f"""
            <div class="prediction-box">
                <b>{pred['player1']}</b> vs <b>{pred['player2']}</b> ({pred['surface']})<br>
                <span>Win Probability: <b>{pred['probability']:.1%}</b> | {pred['confidence']}</span><br>
                <span>Projected Score: {pred['projected_score']}</span><br>
                <span>Best Odds: <b>{pred['odds']:.2f}</b></span><br>
                <span>Expected Return: <span class="{ 'profit-positive' if pred['expected_return'] > 0 else 'profit-negative' }">${pred['expected_return']:.2f}</span></span><br>
                <span>Kelly Stake: {pred['optimal_stake']:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
            if pred['is_value_bet']:
                st.markdown('<div class="value-bet-box">🔥 <b>Value Bet Detected!</b></div>', unsafe_allow_html=True)
            if show_all_bookmakers and pred['odds_data']:
                st.markdown('<div class="odds-comparison"><b>Bookmaker Odds:</b><br>', unsafe_allow_html=True)
                for src, bookies in pred['odds_data']['all_odds'].items():
                    st.write(f"{src}:")
                    for bookie, odds in bookies.items():
                        st.write(f"- {bookie}: {odds['player1_odds']} / {odds['player2_odds']}")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No ATP matches found.")
    
    st.header("Upcoming WTA Matches")
    wta_matches, wta_date = fetch_next_wta_matches()
    if not wta_matches.empty:
        wta_preds = get_predictions_for_matches(
            wta_matches, st.session_state.wta_predictor,
            st.session_state.wta_rankings, st.session_state.wta_matches, "WTA"
        )
        for pred in wta_preds:
            st.markdown(f"""
            <div class="prediction-box">
                <b>{pred['player1']}</b> vs <b>{pred['player2']}</b> ({pred['surface']})<br>
                <span>Win Probability: <b>{pred['probability']:.1%}</b> | {pred['confidence']}</span><br>
                <span>Projected Score: {pred['projected_score']}</span><br>
                <span>Best Odds: <b>{pred['odds']:.2f}</b></span><br>
                <span>Expected Return: <span class="{ 'profit-positive' if pred['expected_return'] > 0 else 'profit-negative' }">${pred['expected_return']:.2f}</span></span><br>
                <span>Kelly Stake: {pred['optimal_stake']:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
            if pred['is_value_bet']:
                st.markdown('<div class="value-bet-box">🔥 <b>Value Bet Detected!</b></div>', unsafe_allow_html=True)
            if show_all_bookmakers and pred['odds_data']:
                st.markdown('<div class="odds-comparison"><b>Bookmaker Odds:</b><br>', unsafe_allow_html=True)
                for src, bookies in pred['odds_data']['all_odds'].items():
                    st.write(f"{src}:")
                    for bookie, odds in bookies.items():
                        st.write(f"- {bookie}: {odds['player1_odds']} / {odds['player2_odds']}")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No WTA matches found.")

if __name__ == "__main__":
    main()
