import os
import logging
import matplotlib.pyplot as plt
import math
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import shutil
import sqlite3

import numpy as np
import pandas as pd
import requests
import streamlit as st

# pybaseball: first-use auth cache
from pybaseball import statcast
from pybaseball import cache as pybb_cache
pybb_cache.enable()  # keep responses cached between runs

# --- bet_logs absolute path config (shared with rest of stack) ------------
BETLOGS_ROOT = os.getenv("BETLOGS_ROOT", os.getcwd())
BETLOGS_DB = os.path.join(BETLOGS_ROOT, "bet_logs.db")

def _ensure_parent(path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

# ---------- Logging helpers (schema consumed by proof_maker.py) ----------
def log_blog_pick_to_db(pick_data: dict, db_path: str = BETLOGS_DB):
    _ensure_parent(db_path)
    try:
        with sqlite3.connect(db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS blog_pick_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_date TEXT,
                    matchup TEXT,
                    bet_type TEXT,
                    confidence TEXT,
                    edge_pct REAL,
                    odds REAL,
                    predicted_total REAL,
                    predicted_winner TEXT,
                    predicted_margin REAL,
                    bookmaker_total REAL,
                    analysis TEXT
                );
            """)
            cols = ["log_date","matchup","bet_type","confidence","edge_pct","odds",
                    "predicted_total","predicted_winner","predicted_margin","bookmaker_total","analysis"]
            row = [pick_data.get(k) for k in cols]
            con.execute(f"INSERT INTO blog_pick_logs ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", row)
    except Exception as e:
        logging.error(f"[ump_card.log_blog_pick_to_db] {e}")

def log_bet_card_to_db(card: dict, db_path: str = BETLOGS_DB):
    _ensure_parent(db_path)
    try:
        with sqlite3.connect(db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS bet_card_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    game_time TEXT,
                    combined_runs REAL,
                    delta REAL,
                    total_bet_rec TEXT,
                    ml_bet_rec TEXT,
                    over_edge REAL,
                    under_edge REAL,
                    home_ml_edge REAL,
                    away_ml_edge REAL,
                    home_ensemble_pred REAL,
                    away_ensemble_pred REAL,
                    combined_ensemble_pred REAL,
                    ensemble_confidence TEXT,
                    statcast_home_exit REAL,
                    statcast_home_angle REAL,
                    statcast_away_exit REAL,
                    statcast_away_angle REAL,
                    pitcher_home_exit REAL,
                    pitcher_home_angle REAL,
                    pitcher_away_exit REAL,
                    pitcher_away_angle REAL,
                    weather_home_temp REAL,
                    weather_home_wind REAL,
                    weather_away_temp REAL,
                    weather_away_wind REAL,
                    home_pitcher TEXT,
                    away_pitcher TEXT,
                    bookmaker_line REAL,
                    over_price REAL,
                    under_price REAL,
                    home_ml_book REAL,
                    away_ml_book REAL,
                    log_date TEXT
                );
            """)
            defaults = {
                "game_id": None, "home_team": None, "away_team": None, "game_time": None,
                "combined_runs": None, "delta": None, "total_bet_rec": None, "ml_bet_rec": None,
                "over_edge": None, "under_edge": None, "home_ml_edge": None, "away_ml_edge": None,
                "home_ensemble_pred": None, "away_ensemble_pred": None, "combined_ensemble_pred": None,
                "ensemble_confidence": "ump", "statcast_home_exit": None, "statcast_home_angle": None,
                "statcast_away_exit": None, "statcast_away_angle": None, "pitcher_home_exit": None,
                "pitcher_home_angle": None, "pitcher_away_exit": None, "pitcher_away_angle": None,
                "weather_home_temp": None, "weather_home_wind": None, "weather_away_temp": None,
                "weather_away_wind": None, "home_pitcher": None, "away_pitcher": None, "bookmaker_line": None,
                "over_price": None, "under_price": None, "home_ml_book": None, "away_ml_book": None,
                "log_date": datetime.utcnow().isoformat()
            }
            payload = defaults | {k: card.get(k) for k in defaults.keys()}
            cols = list(defaults.keys())
            row = [payload[k] for k in cols]
            con.execute(f"INSERT INTO bet_card_logs ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", row)
    except Exception as e:
        logging.error(f"[ump_card.log_bet_card_to_db] {e}")

# small local slugifier to avoid cross-dep on later function
def _slug(s: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9@ ]+", "", str(s)).strip()
    s = s.replace(" @ ", "@").replace("  ", " ")
    return s.replace(" ", "_").replace("@", "_AT_")

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="MLB Umpire Report Cards",
    page_icon="⚾",
    layout="wide"
)

# --- Session state bootstrap (persist across reruns) ---
if "ran" not in st.session_state:
    st.session_state.ran = False
if "params" not in st.session_state:
    st.session_state.params = None
for _k in ("per_game","season_ranked","df","df_today","cards_disp"):
    if _k not in st.session_state:
        st.session_state[_k] = None

# -----------------------------
# Helpers
# -----------------------------
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_GAMEFEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

@st.cache_data(show_spinner=False)
def fetch_schedule(start_dt: date, end_dt: date) -> pd.DataFrame:
    """Get MLB schedule with gamePk for date window."""
    params = {
        "sportId": 1,
        "startDate": start_dt.strftime("%Y-%m-%d"),
        "endDate": end_dt.strftime("%Y-%m-%d"),
        "language": "en",
    }
    r = requests.get(MLB_SCHEDULE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    records = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") in {"Final", "Live", "In Progress", "Pre-Game", "Preview"}:
                records.append({
                    "game_pk": g.get("gamePk"),
                    "game_date": d.get("date"),
                    "status": g.get("status", {}).get("detailedState"),
                    "home": g.get("teams", {}).get("home", {}).get("team", {}).get("name"),
                    "away": g.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                    "game_time": g.get("gameDate"),
                    "venue": g.get("venue", {}).get("name"),
                    "probable_home": (g.get("teams", {}).get("home", {}).get("probablePitcher", {}) or {}).get("fullName"),
                    "probable_away": (g.get("teams", {}).get("away", {}).get("probablePitcher", {}) or {}).get("fullName"),
                })
    return pd.DataFrame.from_records(records)

@st.cache_data(show_spinner=False)
def fetch_plate_ump_for_game(game_pk: int) -> Tuple[str, int, str]:
    """Return (umpire_name, umpire_id, crew_chief) for the home plate ump of a game."""
    url = MLB_GAMEFEED_URL.format(game_pk=game_pk)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    officials = data.get("liveData", {}).get("boxscore", {}).get("officials", []) or []
    crew_chief = None
    # Types seen: Home Plate, First Base, Second Base, Third Base, Left Field, Right Field
    for off in officials:
        if off.get("officialType") == "Crew Chief":
            person = off.get("official", {}) or {}
            crew_chief = (person.get("fullName") or person.get("firstLastName") or f"{person.get('firstName','')} {person.get('lastName','')}").strip()
        if off.get("officialType") == "Home Plate":
            person = off.get("official", {}) or off.get("official", {})
            name = (person.get("fullName")
                    or person.get("firstLastName")
                    or f"{person.get('firstName','')} {person.get('lastName','')}".strip())
            ump_id = person.get("id")
            return name, ump_id, crew_chief
    return None, None, crew_chief

def within_zone(plate_x, plate_z, sz_top, sz_bot, half_width_ft) -> bool:
    """Geometric zone check using personalized top/bottom and horizontal half-width (feet)."""
    if pd.isna(plate_x) or pd.isna(plate_z) or pd.isna(sz_top) or pd.isna(sz_bot):
        return False
    if plate_z < min(sz_top, sz_bot) or plate_z > max(sz_top, sz_bot):
        return False
    return abs(plate_x) <= half_width_ft

def is_taken_pitch(description: str) -> bool:
    """Filter to taken pitches only (called strikes/balls)."""
    if not isinstance(description, str):
        return False
    d = description.lower()
    # Statcast descriptions for taken pitches commonly include:
    # 'called_strike', 'ball', 'blocked_ball' (blocked still a ball), 'pitchout' (ball)
    return any(key in d for key in ["called_strike", "ball", "blocked_ball", "pitchout"])

def is_called_strike(description: str) -> bool:
    return isinstance(description, str) and "called_strike" in description.lower()

def is_ball_call(description: str) -> bool:
    if not isinstance(description, str):
        return False
    d = description.lower()
    return ("ball" in d) and ("called_strike" not in d)  # exclude called_strike

def edge_band(plate_x, plate_z, sz_top, sz_bot, half_width_ft, band_in_inches=2.0) -> bool:
    """Is the pitch in a band around the edge (inside or outside)?"""
    if pd.isna(plate_x) or pd.isna(plate_z) or pd.isna(sz_top) or pd.isna(sz_bot):
        return False
    band_ft = band_in_inches / 12.0
    # Vertical edges (top/bottom)
    top_near = abs(plate_z - max(sz_top, sz_bot)) <= band_ft
    bot_near = abs(plate_z - min(sz_top, sz_bot)) <= band_ft
    # Horizontal edge (left/right)
    hor_near = abs(abs(plate_x) - half_width_ft) <= band_ft
    return top_near or bot_near or hor_near

def favor_sign(row) -> int:
    """
    Net favor in 'extra strikes' terms:
    +1 if a miss created a strike for the defense (called strike outside zone),
    -1 if a miss took away a strike from defense (ball inside zone).
    Assign to batting team perspective: +1 benefits the defense -> hurts batting team.
    We'll flip to team names later.
    """
    if row["miss_type"] == "strike_outside":
        return +1
    if row["miss_type"] == "ball_inside":
        return -1
    return 0

# -----------------------------
# Bingo export helpers (schema, IDs, probabilities)
# -----------------------------

# -----------------------------
# Public-facing betting translators
# -----------------------------

def compute_run_boost_index(d_cs_pp: float, d_ball_pp: float, edge_pressure: float) -> float:
    """
    Translate plate tendencies into a single "Run Boost Index" measured in runs per game vs league.
    Heuristic mapping calibrated to be conservative and bounded in small samples:
      • +1.0 pp called-strike accuracy (vs lg) ≈ −0.02 runs
      • −1.0 pp ball accuracy (vs lg) ≈ +0.02 runs
      • Each 10% drop in edge-band accuracy adds +0.03 runs (more borderline misses → base traffic)
    Final index is clipped to ±0.50 runs to avoid overselling.
    NaNs are treated as 0 to avoid leaking bogus values when assignments are TBD.
    """
    try:
        import math as _math
        def _nz(x):
            try:
                x = float(x)
                return 0.0 if _math.isnan(x) else x
            except Exception:
                return 0.0
        k_component = -0.02 * _nz(d_cs_pp)           # higher CS accuracy suppresses runs
        bb_component = +0.02 * _nz(-d_ball_pp)       # lower ball accuracy lifts walks/runs
        edge_component = +0.003 * _nz(edge_pressure) # 10% edge-pressure ≈ +0.03 runs
        run_boost = k_component + bb_component + edge_component
        # Clip and round
        run_boost = max(-0.50, min(0.50, run_boost))
        return float(round(run_boost, 3))
    except Exception:
        return 0.0


def classify_plate_tilt(run_boost_index: float, k_lean_pp: float, bb_lean_pp: float) -> str:
    """Label the public-facing tilt for content: Over, Under, or Neutral."""
    try:
        import math as _math
        if _math.isnan(float(run_boost_index)) or _math.isnan(float(k_lean_pp)) or _math.isnan(float(bb_lean_pp)):
            return "Neutral"
    except Exception:
        return "Neutral"
    # Slightly more sensitive, still conservative
    if run_boost_index >= 0.08 or (k_lean_pp <= -1.0 and bb_lean_pp >= 1.0):
        return "Over Tilt"
    if run_boost_index <= -0.08 or (k_lean_pp >= 1.0 and bb_lean_pp <= -1.0):
        return "Under Tilt"
    return "Neutral"

# Public-friendly accuracy with a softened "shadow" zone (wider horizontally, trimmed vertically)
# This stabilizes public-facing accuracy so you don't show baffling 35% CS rates.

def shadow_correct_pct(taken_df: pd.DataFrame, hmargin_ft: float = 0.83, vband_ft: float = 0.2) -> float:
    if taken_df is None or taken_df.empty:
        return np.nan
    try:
        top = taken_df["sz_top"].median()
        bot = taken_df["sz_bot"].median()
        in_shadow = (
            (taken_df["plate_z"] >= (bot + vband_ft)) &
            (taken_df["plate_z"] <= (top - vband_ft)) &
            (taken_df["plate_x"].abs() <= hmargin_ft)
        )
        correct_shadow = (
            ((in_shadow)  & taken_df["is_called_strike"]) |
            ((~in_shadow) & taken_df["is_ball_call"]))
        return float(correct_shadow.mean() * 100.0)
    except Exception:
        return np.nan


def add_ou_tendency_columns(per_game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a proxy Over tendency per umpire using final game totals vs same-day league mean.
    This avoids reliance on closing lines when none are available.
    Adds columns: over_rate_vs_day_mean, games_n_totals
    """
    if per_game_df is None or per_game_df.empty:
        return per_game_df
    pg = per_game_df.copy()
    if "total_runs" not in pg.columns:
        return pg
    # league mean by day
    day_means = pg.groupby("game_date")["total_runs"].mean().rename("day_mean").reset_index()
    pg = pg.merge(day_means, on="game_date", how="left")
    pg["over_vs_day"] = (pg["total_runs"] > pg["day_mean"]).astype(int)
    agg = pg.groupby("plate_umpire").agg(
        over_rate_vs_day_mean=("over_vs_day", "mean"),
        games_n_totals=("over_vs_day", "count")
    ).reset_index()
    return agg


def generate_plate_tilt_card_png(row: pd.Series, out_path: str):
    """Small, social-ready card summarizing the tilt with 3 numbers and a timestamp."""
    try:
        fig, ax = plt.subplots(figsize=(4.0, 5.2), dpi=300)
        ax.axis('off')
        # Header
        title = f"{row.get('matchup','?')}\nPlate: {row.get('plate_umpire','?')}"
        ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=12, weight='bold')
        # Tilt badge
        tilt = row.get('tilt_label', 'Neutral')
        ax.text(0.5, 0.86, tilt, ha='center', va='center', fontsize=11, bbox=dict(boxstyle='round', alpha=0.15))
        # Core stats
        acc = row.get('accuracy_pct', float('nan'))
        edge = row.get('edge_acc_pct', float('nan'))
        kpp = row.get('K_lean_pp', float('nan'))
        bbpp = row.get('BB_lean_pp', float('nan'))
        rbi = row.get('run_boost_index', 0.0)
        txt = (
            f"Accuracy: {acc:.1f}%\n"
            f"Edge Acc (±2\") : {edge:.1f}%\n"
            f"ΔCS acc: {kpp:+.2f} pp | ΔBall acc: {(-bbpp):+.2f} pp\n"
            f"Run Boost Index: {rbi:+.2f} runs"
        )
        ax.text(0.05, 0.60, txt, ha='left', va='top', fontsize=10)
        # Footer
        ts = row.get('assign_ts', datetime.utcnow().isoformat())
        ax.text(0.5, 0.05, f"Assignments fetched: {ts}", ha='center', va='bottom', fontsize=8, alpha=0.7)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    except Exception as _e:
        logging.error(f"[tilt_card] failed: {_e}")

# -----------------------------
# Social Infographic Composer (Fairness + Heatmap + Overlays)
# -----------------------------

def generate_infographic(fairness_path: str, heatmap_path: str, tilt_path: str, meta: dict, export_dir: str) -> str:
    """
    Compose a shareable infographic by pasting Fairness Meter + Heatmap + Tilt card
    with short overlays. Uses PIL if available; otherwise returns fairness image.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        logging.warning("Pillow not available; returning fairness image as infographic.")
        return fairness_path

    try:
        base = Image.open(fairness_path).convert("RGBA")
        w, h = base.size
        # Place heatmap (top-right) and tilt card (bottom-right)
        try:
            heat = Image.open(heatmap_path).convert("RGBA")
            heat = heat.resize((w//2, int(h*0.48)))
            base.paste(heat, (w - heat.width - 8, 8), heat)
        except Exception as e:
            logging.warning(f"heatmap paste failed: {e}")
        try:
            tilt = Image.open(tilt_path).convert("RGBA")
            tilt = tilt.resize((w//2, int(h*0.45)))
            base.paste(tilt, (w - tilt.width - 8, h - tilt.height - 8), tilt)
        except Exception as e:
            logging.warning(f"tilt paste failed: {e}")

        # Overlays
        draw = ImageDraw.Draw(base)
        def _load_font(size):
            try:
                return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
            except Exception:
                try:
                    return ImageFont.truetype("arial.ttf", size)
                except Exception:
                    return ImageFont.load_default()
        title_font = _load_font(28)
        body_font = _load_font(20)
        hook_font = _load_font(22)

        matchup = meta.get("matchup", "?")
        ump = meta.get("plate_umpire", "?")
        acc = meta.get("accuracy_pct")
        edge = meta.get("edge_acc_pct")
        tilt_lbl = meta.get("tilt_label", "Neutral")
        rbi = meta.get("run_boost_index", 0.0)

        draw.rectangle([8, 8, int(w*0.66), 108], fill=(255,255,255,200))
        draw.text((16, 16), f"{matchup}", font=title_font, fill=(0,0,0))
        draw.text((16, 56), f"Plate: {ump}", font=body_font, fill=(0,0,0))

        hook = f"{tilt_lbl} • RBI {rbi:+.2f}" if isinstance(rbi, (int,float)) else str(tilt_lbl)
        draw.rectangle([8, h-108, int(w*0.66), h-8], fill=(255,255,255,200))
        if isinstance(acc, (int,float)) and isinstance(edge, (int,float)):
            draw.text((16, h-100), f"Accuracy {acc:.1f}% | Edge {edge:.1f}%", font=body_font, fill=(0,0,0))
        draw.text((16, h-62), hook, font=hook_font, fill=(0,0,0))

        out = os.path.join(export_dir, f"infographic_{_slug(matchup)}_{_slug(ump)}.png")
        base.save(out)
        return out
    except Exception as e:
        logging.error(f"[infographic] failed: {e}")
        return fairness_path

def _prob_clip(x: float, lo: float = 0.05, hi: float = 0.95) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return 0.5


def _difficulty_from_p(p: float) -> str:
    if p >= 0.35:
        return "easy"
    if p >= 0.22:
        return "med"
    return "hard"


def _slugify(s: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9@ ]+", "", str(s)).strip()
    s = s.replace(" @ ", "@").replace("  ", " ")
    return s.replace(" ", "_").replace("@", "_AT_")


def build_bingo_events_from_ump(per_game: pd.DataFrame) -> pd.DataFrame:
    """
    Build a machine-readable bingo event catalog from per-game umpire summaries already
    produced by this app. We estimate per-game event probabilities using the ump's
    historical frequency in the selected window, with a league fallback for small samples.

    Output columns: id, label, p, group, difficulty, source, context
    """
    import numpy as _np
    import pandas as _pd

    if per_game is None or per_game.empty:
        return _pd.DataFrame(columns=["id","label","p","group","difficulty","source","context"])

    # Thresholds for settle-able game events
    OUTSIDE_GE = 3      # >=3 outside-zone called strikes
    INSIDE_GE  = 2      # >=2 inside-zone balls
    EDGE_ACC_GE = 60.0  # edge-band accuracy >=60%
    TAKEN_GE   = 140    # total taken pitches >=140
    SWING_GE   = 2      # |home extra strikes - away extra strikes| >= 2

    pg = per_game.copy()

    # Boolean event flags per historical game
    pg = pg.assign(
        evt_outside_ge = (pg["miss_strike_outside"] >= OUTSIDE_GE),
        evt_inside_ge  = (pg["miss_ball_inside"] >= INSIDE_GE),
        evt_edgeacc_ge = (pg["edge_acc_pct"] >= EDGE_ACC_GE),
        evt_taken_ge   = (pg["total_taken"] >= TAKEN_GE),
        evt_swing_ge   = ((pg["net_favor_home_extra_strikes"] - pg["net_favor_away_extra_strikes"]).abs() >= SWING_GE),
        evt_home_favor = (pg["net_favor_home_extra_strikes"] > pg["net_favor_away_extra_strikes"]) ,
    )

    # Per-ump historical rates
    rates = {}
    for ump, g in pg.groupby("plate_umpire"):
        rates[ump] = {
            "outside": float(g["evt_outside_ge"].mean()) if len(g) else _np.nan,
            "inside":  float(g["evt_inside_ge"].mean())  if len(g) else _np.nan,
            "edgeacc": float(g["evt_edgeacc_ge"].mean()) if len(g) else _np.nan,
            "taken":   float(g["evt_taken_ge"].mean())   if len(g) else _np.nan,
            "swing":   float(g["evt_swing_ge"].mean())   if len(g) else _np.nan,
            "homefav": float(g["evt_home_favor"].mean()) if len(g) else _np.nan,
            "n": int(len(g)),
        }

    # League baselines for fallback
    league = {
        "outside": float(pg["evt_outside_ge"].mean()),
        "inside":  float(pg["evt_inside_ge"].mean()),
        "edgeacc": float(pg["evt_edgeacc_ge"].mean()),
        "taken":   float(pg["evt_taken_ge"].mean()),
        "swing":   float(pg["evt_swing_ge"].mean()),
        "homefav": float(pg["evt_home_favor"].mean()),
    }

    def _shrink(p_hat: float, n_games: int, p0: float, m: int = 12) -> float:
        """Empirical-Bayes shrinkage toward league baseline.
        p' = (n*p_hat + m*p0)/(n+m). Use m≈12 by default.
        """
        try:
            if _np.isnan(p_hat):
                return float(p0)
        except Exception:
            pass
        return float((n_games * float(p_hat) + m * float(p0)) / (n_games + m))

    rows = []
    for r in per_game.itertuples():
        matchup = f"{getattr(r,'away','?')} @ {getattr(r,'home','?')}"
        ump = r.plate_umpire
        date_str = str(r.game_date)
        pr = rates.get(ump, {})
        n = pr.get("n", 0)

        def pick(key: str) -> float:
            p_hat = pr.get(key, _np.nan)
            p0 = league.get(key, 0.5)
            p = _shrink(p_hat, n, p0, m=12)
            # cap ump-driven event probs so they don't dominate bingo difficulty
            p = max(0.05, min(0.85, float(p)))
            return p

        items = [
            ("UMP_OUTSIDE_GE3",  f"{ump}: ≥{OUTSIDE_GE} outside-zone called strikes", pick("outside"), {"threshold": OUTSIDE_GE}),
            ("UMP_INSIDE_GE2",   f"{ump}: ≥{INSIDE_GE} inside-zone balls",          pick("inside"),  {"threshold": INSIDE_GE}),
            ("UMP_EDGEACC_GE60", f"{ump}: edge-band accuracy ≥ {EDGE_ACC_GE:.0f}%",  pick("edgeacc"), {"threshold_pct": EDGE_ACC_GE}),
            ("UMP_TAKEN_GE140",  f"{ump}: taken pitches ≥ {TAKEN_GE}",               pick("taken"),   {"threshold": TAKEN_GE}),
            ("UMP_SWING_GE2",    f"{ump}: favor swing |H-A| ≥ {SWING_GE}",         pick("swing"),   {"threshold": SWING_GE}),
            ("UMP_HOME_FAV",     f"{ump}: net favor to HOME defense",                    pick("homefav"), {}),
        ]
        for base_id, label, p, extra in items:
            rows.append({
                "id": f"{base_id}_{_slugify(matchup)}_{date_str}",
                "label": f"{label} in {matchup}",
                "p": float(p),
                "group": "ump",
                "difficulty": _difficulty_from_p(float(p)),
                "source": "ump_card.py",
                "context": {"matchup": matchup, "date": date_str, "umpire": ump, "n": int(n), **extra},
            })

    out = _pd.DataFrame(rows, columns=["id","label","p","group","difficulty","source","context"]).drop_duplicates(subset=["id"])  # stable IDs
    return out

# -----------------------------
# UI
# -----------------------------
st.title("⚾ MLB Umpire Report Cards — Plate Accuracy & Favor (Statcast)")

colA, colB, colC, colD = st.columns([1.1, 1, 1, 1.2])
with colA:
    year = st.number_input("Season", min_value=2015, max_value=date.today().year, value=date.today().year, step=1)
with colB:
    start_date = st.date_input("Start date", value=date(year, 3, 15))
with colC:
    end_date = st.date_input("End date", value=date(year, 11, 15))
with colD:
    zone_margin_in = st.slider("Zone horizontal margin (inches)", min_value=0.0, max_value=3.0, value=0.0, step=0.25,
                               help="Expands plate half-width (0.708 ft) to allow tolerance for tracking/zone uncertainty.")

team_filter = st.text_input("Team filter (optional, substring match; e.g., 'Braves' or 'ATL')", "")

st.caption("Notes: • Uses personalized vertical zone (`sz_bot`, `sz_top`) and plate_x horizontal half-width of 0.708 ft + margin. • Only taken pitches (called strikes/balls).")

st.button("Run Report", type="primary",
          on_click=lambda: st.session_state.update(ran=True))

# Current input params snapshot (used to decide recompute)
cur_params = {
    "year": int(year),
    "start_date": pd.to_datetime(start_date).date(),
    "end_date": pd.to_datetime(end_date).date(),
    "zone_margin_in": float(zone_margin_in),
    "team_filter": (team_filter or "").strip().lower(),
}
recompute = (st.session_state.params != cur_params)

# --- Reset button ---
def _reset_state():
    for k in ("ran","params","per_game","season_ranked","df","df_today","cards_disp"):
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

st.button("Reset Report", on_click=_reset_state)

#
# -----------------------------
# Core Pipeline
# -----------------------------
if st.session_state.ran:
    # If inputs changed or this is the first run, recompute heavy pipeline, else reuse cached dataframes
    if recompute:
        st.session_state.params = cur_params
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        st.stop()

    with st.spinner("Fetching schedule..."):
        sched = fetch_schedule(start_date, end_date)
        if team_filter.strip():
            mask = sched["home"].str.contains(team_filter, case=False, na=False) | \
                   sched["away"].str.contains(team_filter, case=False, na=False)
            sched = sched[mask].copy()
        if sched.empty:
            st.warning("No games found for filters.")
            st.stop()

    st.caption(f"bet_logs target → {BETLOGS_DB}")

    # Pull home plate umpires for each game
    st.write(f"Found **{len(sched)}** games. Getting assigned home-plate umpires…")
    umps = []
    prog = st.progress(0.0)
    for i, row in enumerate(sched.itertuples(index=False)):
        name, uid, crew = fetch_plate_ump_for_game(int(row.game_pk))
        umps.append({
            "game_pk": row.game_pk, "home": row.home, "away": row.away,
            "game_date": row.game_date, "plate_umpire": name, "plate_umpire_id": uid,
            "crew_chief": crew,
        })
        if (i + 1) % 10 == 0 or i == len(sched) - 1:
            prog.progress((i + 1) / len(sched))
        # small polite delay to avoid hammering API
        time.sleep(0.03)
    prog.empty()
    df_umps = pd.DataFrame(umps)
    df_umps = df_umps.dropna(subset=["plate_umpire"])

    if df_umps.empty:
        st.warning("No plate-ump assignments found in the selected window.")
        st.stop()

    st.success(f"Plate Ump assignments found for **{len(df_umps)}** games.")

    # Pull Statcast pitches for the window in day chunks to avoid huge downloads
    st.write("Downloading Statcast pitch-by-pitch… (day-chunked)")
    chunks = []
    days = list(daterange(start_date, end_date))
    prog = st.progress(0.0)
    for i, d in enumerate(days):
        try:
            df = statcast(start_dt=d.strftime("%Y-%m-%d"), end_dt=d.strftime("%Y-%m-%d"))
            if df is not None and not df.empty:
                # Keep only columns we need
                keep_cols = {
                    "game_pk", "game_date", "home_team", "away_team",
                    "description", "type", "pitch_number",
                    "plate_x", "plate_z", "sz_top", "sz_bot",
                    "stand", "p_throws", "inning", "inning_topbot",
                    "events", "home_score", "away_score",
                    "player_name", "batter", "pitcher",
                    "strikes", "balls",
                    "pitch_type",
                }
                df = df[[c for c in df.columns if c in keep_cols]].copy()
                chunks.append(df)
        except Exception as e:
            # Continue on individual day failures
            st.warning(f"Statcast fetch failed for {d}: {e}")
        if (i + 1) % 3 == 0 or i == len(days) - 1:
            prog.progress((i + 1) / len(days))
        time.sleep(0.05)
    prog.empty()

    if not chunks:
        st.error("No Statcast data returned in the selected window.")
        st.stop()

    stat = pd.concat(chunks, ignore_index=True)
    stat["game_pk"] = stat["game_pk"].astype("Int64")

    # Join pitches with plate umpires
    df = stat.merge(df_umps, on="game_pk", how="inner")

    # Normalize duplicate columns from merge (game_date_x/game_date_y) and drop unused dupes
    if "game_date" not in df.columns:
        if "game_date_x" in df.columns:
            df["game_date"] = df["game_date_x"]
        elif "game_date_y" in df.columns:
            df["game_date"] = df["game_date_y"]

    # Ensure game_date is a date/string (not Timestamp with time)
    if "game_date" in df.columns:
        try:
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Drop potential duplicate/suffix columns we don't use downstream
    cols_to_drop = [c for c in ["game_date_x", "game_date_y", "home", "away", "status"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if df.empty:
        st.warning("No overlap between Statcast data and games with plate umpires.")
        st.stop()

    # Flag taken pitches (called strikes/balls) but keep ALL pitches so every plate job is included
    df["taken"] = df["description"].apply(is_taken_pitch)

    # Zone geometry
    # Official half-width = 17 inches / 2 = 8.5 inches = 0.7083 ft
    half_width_ft = 17.0 / 24.0 / 2.0  # 0.708333...
    half_width_ft += (zone_margin_in / 12.0)  # margin in feet

    df["in_zone"] = df.apply(
        lambda r: within_zone(r["plate_x"], r["plate_z"], r["sz_top"], r["sz_bot"], half_width_ft), axis=1
    )
    df["is_called_strike"] = df["description"].apply(is_called_strike)
    df["is_ball_call"] = df["description"].apply(is_ball_call)

    # Miss classification
    df["is_correct"] = (
        (df["in_zone"] & df["is_called_strike"]) |
        (~df["in_zone"] & df["is_ball_call"])
    )
    df["miss_type"] = np.where(
        (~df["in_zone"] & df["is_called_strike"]), "strike_outside",
        np.where((df["in_zone"] & df["is_ball_call"]), "ball_inside", "correct")
    )

    # Edge-zone band accuracy (optional quality signal)
    df["is_edge_band"] = df.apply(
        lambda r: edge_band(r["plate_x"], r["plate_z"], r["sz_top"], r["sz_bot"], half_width_ft, band_in_inches=2.0),
        axis=1
    )

    # Favor calc: +1 if strike_outside (helps defense), -1 if ball_inside (helps offense)
    df["favor_sign_defense"] = df.apply(favor_sign, axis=1)

    # Assign batting team (to flip favor into team names)
    # If inning_topbot == 'Top', the AWAY team bats; else HOME bats.
    df["batting_team"] = np.where(df["inning_topbot"].str.lower() == "top", df["away_team"], df["home_team"])

    # For reporting: join game context
    def summarize_game(game_df: pd.DataFrame) -> Dict:
        # Use only taken pitches (called balls/strikes) for accuracy metrics
        taken_df = game_df[game_df["taken"] == True]

        total_taken = len(taken_df)
        correct = int(taken_df["is_correct"].sum()) if total_taken else 0

        called_strike_acc = (
            taken_df.loc[taken_df["is_called_strike"], "in_zone"].mean()
            if (taken_df["is_called_strike"].sum() > 0) else np.nan
        )
        ball_acc = (
            (~taken_df.loc[taken_df["is_ball_call"], "in_zone"]).mean()
            if (taken_df["is_ball_call"].sum() > 0) else np.nan
        )
        edge_taken = int(taken_df["is_edge_band"].sum())
        edge_correct = int((taken_df["is_edge_band"] & taken_df["is_correct"]).sum())
        edge_acc = (edge_correct / edge_taken) if edge_taken > 0 else np.nan

        # Net favor by team from taken pitches only
        favor_by_team = taken_df.groupby("batting_team")["favor_sign_defense"].sum().to_dict() if total_taken else {}
        teams = {
            "home": game_df["home_team"].iloc[0],
            "away": game_df["away_team"].iloc[0]
        }
        home_batting = favor_by_team.get(teams["home"], 0)
        away_batting = favor_by_team.get(teams["away"], 0)
        net_favor_to_home_defense = -home_batting
        net_favor_to_away_defense = -away_batting

        return {
            "total_taken": total_taken,
            "accuracy_pct": (correct / total_taken) * 100 if total_taken else np.nan,
            "called_strike_acc_pct": (called_strike_acc * 100) if not pd.isna(called_strike_acc) else np.nan,
            "ball_acc_pct": (ball_acc * 100) if not pd.isna(ball_acc) else np.nan,
            "edge_acc_pct": (edge_acc * 100) if not pd.isna(edge_acc) else np.nan,
            "misses": int((~taken_df["is_correct"]).sum()) if total_taken else 0,
            "miss_strike_outside": int((taken_df["miss_type"] == "strike_outside").sum()) if total_taken else 0,
            "miss_ball_inside": int((taken_df["miss_type"] == "ball_inside").sum()) if total_taken else 0,
            "net_favor_home_extra_strikes": int(net_favor_to_home_defense),
            "net_favor_away_extra_strikes": int(net_favor_to_away_defense),
            "home": teams["home"],
            "away": teams["away"],
            "game_date": game_df["game_date"].iloc[0],
            "final_home_runs": int(game_df["home_score"].max()) if not game_df["home_score"].isna().all() else np.nan,
            "final_away_runs": int(game_df["away_score"].max()) if not game_df["away_score"].isna().all() else np.nan,
            "total_runs": int((game_df["home_score"].max() or 0) + (game_df["away_score"].max() or 0)) if not (game_df["home_score"].isna().all() or game_df["away_score"].isna().all()) else np.nan,
            "shadow_correct_pct": shadow_correct_pct(taken_df),
        }

    # Per-game summaries per ump
    game_groups = df.groupby(["plate_umpire", "game_pk"], dropna=True)
    rows = []
    prog = st.progress(0.0)
    for j, (key, gdf) in enumerate(game_groups):
        ump, gpk = key
        summary = summarize_game(gdf)
        summary.update({"plate_umpire": ump, "game_pk": gpk})
        rows.append(summary)
        if (j + 1) % 50 == 0 or j == len(game_groups) - 1:
            prog.progress((j + 1) / len(game_groups))
    prog.empty()
    per_game = pd.DataFrame(rows)

    # Over/Under proxy tendency vs same-day league mean
    try:
        ou_tend = add_ou_tendency_columns(per_game)
        if ou_tend is not None and not ou_tend.empty:
            per_game = per_game.merge(ou_tend, on="plate_umpire", how="left")
    except Exception as _ou_e:
        logging.error(f"[ump_card] OU tendency merge failed: {_ou_e}")

    # Attach crew chief to per_game for crew-level summaries
    try:
        per_game = per_game.merge(df_umps[["game_pk", "crew_chief"]].drop_duplicates(), on="game_pk", how="left")
    except Exception:
        per_game["crew_chief"] = np.nan

    # ---------- New Metrics & Modules ----------
    def wilson_interval(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        phat = k / n
        denom = 1 + z**2 / n
        center = (phat + z**2/(2*n)) / denom
        margin = z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom
        return (max(0.0, center - margin) * 100, min(1.0, center + margin) * 100)

    # Plate ELO (seasonal within selected window)
    def compute_plate_elo(pg: pd.DataFrame, k_factor=20.0):
        if pg.empty:
            return pg.assign(elo=np.nan, elo_delta=np.nan)
        # Baseline expected accuracy = league mean in window
        league_mean = pg["accuracy_pct"].mean()
        df_e = pg.sort_values(["plate_umpire", "game_date"]).copy()
        df_e["elo"] = np.nan
        df_e["elo_delta"] = 0.0
        elo_map = {}
        for idx, row in df_e.iterrows():
            u = row["plate_umpire"]
            cur = elo_map.get(u, 1500.0)
            # Translate accuracy to a pseudo-score vs league (0..1)
            acc = row["accuracy_pct"] if not pd.isna(row["accuracy_pct"]) else league_mean
            actual = 1.0 if acc >= league_mean else 0.0
            # Expected from ELO vs league baseline
            exp = 1.0 / (1.0 + 10 ** (-(cur - 1500.0) / 400.0))
            new = cur + k_factor * (actual - exp)
            df_e.at[idx, "elo"] = new
            df_e.at[idx, "elo_delta"] = new - cur
            elo_map[u] = new
        # Reduce to latest elo per ump
        latest = df_e.sort_values("game_date").groupby("plate_umpire").tail(1)
        return latest[["plate_umpire", "elo"]]

    # Add Wilson intervals to per-game cards
    per_game["acc_low"], per_game["acc_high"] = zip(*per_game.apply(lambda r: wilson_interval(
        int(round(r["accuracy_pct"] * r["total_taken"] / 100)) if not pd.isna(r["accuracy_pct"]) else 0,
        int(r["total_taken"]) if not pd.isna(r["total_taken"]) else 0
    ), axis=1))

    # Season drift detector (rolling 10-game z-score)
    def season_drift(pg: pd.DataFrame):
        out = []
        for u, g in pg.sort_values("game_date").groupby("plate_umpire"):
            g = g.copy()
            g["acc_z"] = (g["accuracy_pct"] - g["accuracy_pct"].rolling(10, min_periods=5).mean()) / (g["accuracy_pct"].rolling(10, min_periods=5).std())
            if not g.empty:
                tail = g.tail(1)
                out.append({
                    "plate_umpire": u,
                    "drift_z": float(tail["acc_z"].iloc[0]) if not pd.isna(tail["acc_z"].iloc[0]) else np.nan
                })
        return pd.DataFrame(out)

    # Compute ELO & drift and merge into season table later

    if per_game.empty:
        st.warning("No taken-pitch data to score. Try widening the date range.")
        st.stop()

    # Season aggregates by umpire
    agg_fields = {
        "total_taken": "sum",
        "accuracy_pct": "mean",
        "called_strike_acc_pct": "mean",
        "ball_acc_pct": "mean",
        "edge_acc_pct": "mean",
        "misses": "sum",
        "miss_strike_outside": "sum",
        "miss_ball_inside": "sum",
        "net_favor_home_extra_strikes": "sum",
        "net_favor_away_extra_strikes": "sum",
        "shadow_correct_pct": "mean",
    }
    season = per_game.groupby("plate_umpire", dropna=True).agg(agg_fields).reset_index()

    # Merge ELO and drift
    elo_latest = compute_plate_elo(per_game)
    drift = season_drift(per_game)
    season = season.merge(elo_latest, on="plate_umpire", how="left").merge(drift, on="plate_umpire", how="left")

    # Uncertainty bands (Wilson) at season level
    season["acc_low"], season["acc_high"] = zip(*season.apply(lambda r: wilson_interval(
        int(round(r["accuracy_pct"] * r["total_taken"] / 100)) if not pd.isna(r["accuracy_pct"]) else 0,
        int(r["total_taken"]) if not pd.isna(r["total_taken"]) else 0
    ), axis=1))

    # Rank by accuracy — include ALL umpires with data
    season_ranked = season.copy()
    season_ranked["rank_accuracy"] = season_ranked["accuracy_pct"].rank(ascending=False, method="min").astype(int)
    season_ranked = season_ranked.sort_values(["rank_accuracy", "plate_umpire"])

    # --- Persist heavy results for dropdown-safe rerenders ---
    st.session_state.per_game = per_game
    st.session_state.season_ranked = season_ranked
    st.session_state.df = df
    # Some sections may not define these if assignments are TBD
    try:
        st.session_state.cards_disp = cards_disp
    except NameError:
        st.session_state.cards_disp = None
    try:
        st.session_state.df_today = df_today
    except NameError:
        st.session_state.df_today = None

    else:
        # Reuse previously computed dataframes from session state
        per_game = st.session_state.per_game
        season_ranked = st.session_state.season_ranked
        df = st.session_state.df
        cards_disp = st.session_state.cards_disp
        df_today = st.session_state.df_today

    # -----------------------------
    # Today's Scheduled Games — Umpire Cards
    # -----------------------------
    with st.expander("Today's Scheduled Games — Umpire Cards", expanded=True):
        try:
            today_dt = date.today()
            today_str = today_dt.strftime("%Y-%m-%d")

            # Get today's schedule and assigned home-plate umpires
            sched_today = fetch_schedule(today_dt, today_dt)
            if sched_today.empty:
                st.info("No MLB games scheduled today.")
            else:
                umps_today = []
                prog_t = st.progress(0.0)
                for i, row in enumerate(sched_today.itertuples(index=False)):
                    try:
                        name, uid, crew = fetch_plate_ump_for_game(int(row.game_pk))
                    except Exception:
                        name, uid, crew = (None, None, None)
                    umps_today.append({
                        "game_pk": row.game_pk,
                        "game_date": row.game_date,
                        "home": row.home,
                        "away": row.away,
                        "plate_umpire": name,
                        "plate_umpire_id": uid,
                        "crew_chief": crew,
                        "assign_ts": datetime.utcnow().isoformat()
                    })
                    if (i + 1) % 10 == 0 or i == len(sched_today) - 1:
                        prog_t.progress((i + 1) / len(sched_today))
                    time.sleep(0.02)
                prog_t.empty()

                df_today = pd.DataFrame(umps_today)

                # Bring in start time, venue, and probable pitchers
                df_today = df_today.merge(
                    sched_today[["game_pk","game_time","venue","probable_home","probable_away"]],
                    on="game_pk",
                    how="left"
                )

                # Split into assigned vs TBD
                assigned = df_today.dropna(subset=["plate_umpire"]).copy()
                tbd = df_today[df_today["plate_umpire"].isna()].copy()

                if not tbd.empty:
                    st.info("Assignments pending for some games (shown below). These rows are excluded from tilt calculations until posted.")
                    st.dataframe(tbd[["game_date","game_time","venue","away","home"]], use_container_width=True, hide_index=True)

                # If assignments are not yet posted, show TBD rows
                if df_today.empty or df_today["plate_umpire"].isna().all():
                    view_cols = ["game_date", "away", "home"]
                    st.warning("Home-plate umpire assignments are not posted yet. Showing today's games only.")
                    st.dataframe(df_today[view_cols] if not df_today.empty else sched_today[view_cols], use_container_width=True)
                else:
                    # Join in season-level and recent-form metrics for the assigned umpires
                    season_pick = season_ranked[
                        ["plate_umpire","total_taken","accuracy_pct","edge_acc_pct",
                         "called_strike_acc_pct","ball_acc_pct","shadow_correct_pct",
                         "elo","drift_z","acc_low","acc_high"]
                    ].copy()

                    # Recent 5 games accuracy per ump (from per_game within current window)
                    recent5 = (
                        per_game.sort_values("game_date")
                        .groupby("plate_umpire")
                        .tail(5)
                        .groupby("plate_umpire")["accuracy_pct"]
                        .mean()
                        .reset_index()
                        .rename(columns={"accuracy_pct":"recent5_acc"})
                    )

                    # League baselines (from current window) and per-ump deltas for K/BB lean
                    taken_all = df[df["taken"] == True].copy() if "taken" in df.columns else pd.DataFrame()
                    if not taken_all.empty:
                        lg_cs_acc = (taken_all.loc[taken_all["is_called_strike"], "in_zone"].mean()) * 100
                        lg_ball_acc = ((~taken_all.loc[taken_all["is_ball_call"], "in_zone"]).mean()) * 100
                        ump_cs = (
                            taken_all[taken_all["is_called_strike"]]
                            .groupby("plate_umpire")["in_zone"]
                            .mean()
                            .mul(100)
                            .rename("u_cs_acc")
                        )
                        ump_ball = (
                            (~taken_all[taken_all["is_ball_call"]]["in_zone"])
                            .groupby(taken_all[taken_all["is_ball_call"]]["plate_umpire"])
                            .mean()
                            .mul(100)
                            .rename("u_ball_acc")
                        )
                        deltas = pd.concat([ump_cs, ump_ball], axis=1).reset_index()
                        deltas["d_cs_pp"] = deltas["u_cs_acc"] - lg_cs_acc
                        deltas["d_ball_pp"] = deltas["u_ball_acc"] - lg_ball_acc
                    else:
                        deltas = pd.DataFrame(columns=["plate_umpire","u_cs_acc","u_ball_acc","d_cs_pp","d_ball_pp"])

                    cards = (
                        assigned.merge(season_pick, on="plate_umpire", how="left")
                                .merge(recent5, on="plate_umpire", how="left")
                                .merge(deltas, on="plate_umpire", how="left")
                    )

                    # Derived public-facing metrics (ordered to avoid KeyErrors)
                    cards["matchup"] = cards["away"] + " @ " + cards["home"]

                    # Define edge pressure first (used downstream)
                    cards["edge_pressure"] = 100 - cards["edge_acc_pct"]

                    # Define lean deltas before any functions consume them
                    cards["K_lean_pp"] = cards["d_cs_pp"]  # +pp favors Ks over baseline
                    cards["BB_lean_pp"] = -cards["d_ball_pp"]  # +pp favors walks if ball accuracy is worse
                    cards["k_flag"] = np.where(cards["K_lean_pp"] >= 0.5, "Over K",
                                               np.where(cards["K_lean_pp"] <= -0.5, "Under K", "Neutral"))
                    cards["bb_flag"] = np.where(cards["BB_lean_pp"] >= 0.5, "Over BB",
                                               np.where(cards["BB_lean_pp"] <= -0.5, "Under BB", "Neutral"))

                    # Run Boost Index uses raw deltas and edge_pressure
                    cards["run_boost_index"] = cards.apply(
                        lambda r: compute_run_boost_index(
                            float(r.get("d_cs_pp", 0.0)),
                            float(r.get("d_ball_pp", 0.0)),
                            float(r.get("edge_pressure", 0.0))
                        ), axis=1
                    )

                    # Tilt label now that leans are defined
                    cards["tilt_label"] = cards.apply(
                        lambda r: classify_plate_tilt(
                            float(r.get("run_boost_index", 0.0)),
                            float(r.get("K_lean_pp", 0.0)),
                            float(r.get("BB_lean_pp", 0.0))
                        ), axis=1
                    )

                    # Simple confluence score placeholder (0-3): ump tilt + recent drift + edge pressure
                    cards["confluence_score"] = (
                        (cards["tilt_label"].isin(["Over Tilt","Under Tilt"]).astype(int)) +
                        (cards["drift_z"].abs() >= 1.0).astype(int) +
                        (cards["edge_pressure"] >= 25).astype(int)
                    )
                    # Games-in-window (sample size proxy)
                    games_in_window = per_game.groupby("plate_umpire")["game_pk"].nunique().rename("games_n").reset_index()
                    cards = cards.merge(games_in_window, on="plate_umpire", how="left")

                    # Ensure optional OU-proxy columns exist even if OU tendency calc had no data
                    if "over_rate_vs_day_mean" not in cards.columns:
                        cards["over_rate_vs_day_mean"] = np.nan
                    if "games_n_totals" not in cards.columns:
                        cards["games_n_totals"] = np.nan

                    # Ensure shadow_correct_pct exists
                    if "shadow_correct_pct" not in cards.columns:
                        cards["shadow_correct_pct"] = np.nan

                    # Short human-readable reason line
                    cards["why"] = cards.apply(
                        lambda r: f"KΔ {float(r.get('K_lean_pp',0.0)):+.2f} pp | Edge {float(r.get('edge_pressure',0.0)):.0f}",
                        axis=1
                    )

                    # Gate tilt labels by minimum sample and friction; otherwise Neutral
                    cards["tilt_label"] = np.where(
                        (cards["games_n"].fillna(0) >= 20) & (cards["edge_pressure"].fillna(0) >= 20),
                        cards["tilt_label"],
                        "Neutral"
                    )

                    cards_disp = cards[[
                        "game_date","game_time","venue","matchup",
                        "probable_away","probable_home",
                        "plate_umpire","crew_chief","assign_ts",
                        "tilt_label","run_boost_index","confluence_score",
                        "accuracy_pct","shadow_correct_pct","edge_acc_pct","edge_pressure",
                        "recent5_acc","elo","drift_z","total_taken","games_n",
                        "K_lean_pp","BB_lean_pp","k_flag","bb_flag",
                        "over_rate_vs_day_mean","games_n_totals",
                        "why"
                    ]].copy()

                    # Formatting
                    for c in ["accuracy_pct","shadow_correct_pct","edge_acc_pct","edge_pressure",
                              "recent5_acc","K_lean_pp","BB_lean_pp"]:
                        if c in cards_disp.columns:
                            cards_disp[c] = cards_disp[c].round(2)
                    if "elo" in cards_disp.columns:
                        cards_disp["elo"] = cards_disp["elo"].round(0)
                    if "drift_z" in cards_disp.columns:
                        cards_disp["drift_z"] = cards_disp["drift_z"].round(2)
                    for c in ["run_boost_index","over_rate_vs_day_mean"]:
                        if c in cards_disp.columns:
                            cards_disp[c] = cards_disp[c].astype(float).round(3)

                    if cards_disp.empty:
                        st.warning("No assigned plate umpires yet for today’s slate after filtering. Showing schedule only.")
                        st.dataframe(df_today[["game_date","game_time","venue","away","home"]], use_container_width=True, hide_index=True)
                    else:
                        # Sort by start time if available; otherwise by matchup
                        if "game_time" in cards_disp.columns and not cards_disp["game_time"].isna().all():
                            try:
                                cards_disp["_t"] = pd.to_datetime(cards_disp["game_time"], errors="coerce")
                                cards_disp = cards_disp.sort_values(["_t","matchup"]).drop(columns=["_t"])
                            except Exception:
                                cards_disp = cards_disp.sort_values(["game_date","matchup"])
                        else:
                            cards_disp = cards_disp.sort_values(["game_date","matchup"])

                        st.caption("Assigned plate umpires with season metrics, recent form, start time, venue, probables, assignment timestamp, and derived leans (K/BB deltas vs league), Run Boost Index, Tilt label, and a simple Confluence score. Edge Pressure = 100 − Edge Acc%.")
                        _show = cards_disp.copy()
                        for _c in ["over_rate_vs_day_mean","games_n_totals"]:
                            try:
                                if _show[_c].isna().all():
                                    _show = _show.drop(columns=[_c])
                            except Exception:
                                pass
                        st.dataframe(_show, use_container_width=True, hide_index=True)

                        # Download
                        st.download_button(
                            "⬇️ Download Today's Umpire Cards (CSV)",
                            data=cards_disp.to_csv(index=False),
                            file_name=f"today_umpire_cards_{today_str}.csv",
                            mime="text/csv"
                        )
        except Exception as _tod_e:
            st.warning(f"Today's umpire cards failed: {_tod_e}")

    # -----------------------------
    # Display
    # -----------------------------
    st.subheader("Season Report Cards — By Umpire")
    st.caption("Accuracy metrics are averages of per-game rates; volume-weighted effects will be close given similar daily loads. • New: Plate ELO, Wilson CIs, Drift flags, Bias matrix, Story Card export, Fairness & Heatmap assets.")

    st.dataframe(
        season_ranked[
            ["rank_accuracy", "plate_umpire", "total_taken", "accuracy_pct",
             "called_strike_acc_pct", "ball_acc_pct", "edge_acc_pct",
             "misses", "miss_strike_outside", "miss_ball_inside",
             "net_favor_home_extra_strikes", "net_favor_away_extra_strikes",
             "elo", "drift_z", "acc_low", "acc_high"]
        ].round(2),
        use_container_width=True
    )
    # --- Log content receipts: top and bottom accuracy umps for the window ---
    try:
        if not season_ranked.empty:
            today_iso = date.today().isoformat()
            top3 = season_ranked.sort_values("accuracy_pct", ascending=False).head(3)
            bot3 = season_ranked.sort_values("accuracy_pct", ascending=True).head(3)
            for label, chunk in (("Top Ump Accuracy", top3), ("Bottom Ump Accuracy", bot3)):
                for r in chunk.itertuples(index=False):
                    log_blog_pick_to_db({
                        "log_date": datetime.utcnow().isoformat(),
                        "matchup": str(getattr(r, "plate_umpire", "")),
                        "bet_type": label,
                        "confidence": None,
                        "edge_pct": None,
                        "odds": None,
                        "predicted_total": None,
                        "predicted_winner": None,
                        "predicted_margin": None,
                        "bookmaker_total": None,
                        "analysis": f"Acc {getattr(r,'accuracy_pct',np.nan):.1f}% | Edge {getattr(r,'edge_acc_pct',np.nan):.1f}% | Taken {int(getattr(r,'total_taken',0))}"
                    })
            st.caption(f"🧾 Logged {len(top3)+len(bot3)} ump content rows to {BETLOGS_DB}")
    except Exception as e:
        logging.error(f"[ump_card] season content logging failed: {e}")

    # -----------------------------
    # Bingo export — Umpire-driven events
    # -----------------------------
    with st.expander("Bingo export — Umpire-driven events", expanded=False):
        try:
            events_df = build_bingo_events_from_ump(per_game)
            if not events_df.empty:
                st.dataframe(events_df.head(50), use_container_width=True, hide_index=True)
                json_blob = events_df.to_json(orient="records", indent=2)
                csv_blob = events_df.to_csv(index=False)
                today_str = date.today().strftime("%Y-%m-%d")
                st.download_button("⬇️ Download edge_bingo_events.json", data=json_blob, file_name=f"{today_str}_edge_bingo_events.json", mime="application/json")
                st.download_button("⬇️ Download edge_bingo_events.csv", data=csv_blob, file_name=f"{today_str}_edge_bingo_events.csv", mime="text/csv")
                st.caption(f"Total events: {len(events_df)} | Umpires covered: {events_df['context'].apply(lambda x: x.get('umpire') if isinstance(x, dict) else '').nunique()}")
            else:
                st.info("No events generated for bingo export (check per-game summaries).")
        except Exception as _be:
            st.warning(f"Bingo export failed: {_be}")

    # --- Crew-Level Drift & Summary ---
    with st.expander("Crew-Level Drift & Summary", expanded=False):
        crew_tbl = per_game.dropna(subset=["crew_chief"]).groupby("crew_chief").agg(
            games=("game_pk", "nunique"),
            total_taken=("total_taken", "sum"),
            acc_pct=("accuracy_pct", "mean"),
            edge_acc_pct=("edge_acc_pct", "mean"),
            misses=("misses", "sum")
        ).reset_index()
        # Simple crew drift: mean of member umps' latest drift_z values
        crew_drift = season_ranked.merge(per_game[["plate_umpire", "crew_chief"]].drop_duplicates(), on="plate_umpire", how="left")
        crew_drift = crew_drift.dropna(subset=["crew_chief"]).groupby("crew_chief")["drift_z"].mean().reset_index().rename(columns={"drift_z":"crew_drift_z"})
        crew_tbl = crew_tbl.merge(crew_drift, on="crew_chief", how="left")
        crew_tbl = crew_tbl.sort_by("crew_drift_z", ascending=False) if hasattr(crew_tbl, 'sort_by') else crew_tbl.sort_values("crew_drift_z", ascending=False)
        st.dataframe(crew_tbl.round(2), use_container_width=True)

    # -----------------------------
    # Batch Social Export — Today (No-UI Assets)
    # -----------------------------
    with st.expander("Batch Social Export — Today (No-UI Assets)", expanded=False):
        export_date = date.today()
        export_dir = os.path.join("exports", export_date.strftime("%Y%m%d"))
        os.makedirs(export_dir, exist_ok=True)

        if "cards_disp" not in locals() or cards_disp is None or cards_disp.empty:
            st.info("No assigned plate umpires available yet. Run the 'Today's Umpire Cards' section after assignments post.")
        else:
            manifests = []
            generated_paths = []
            # Build a quick lookup from per_game for selecting the correct game row by ump and date
            pg_idx = per_game.set_index(["plate_umpire","game_date"]).sort_index()
            for r in cards_disp.itertuples(index=False):
                try:
                    ump = getattr(r, "plate_umpire")
                    gdate = getattr(r, "game_date")
                    matchup = f"{getattr(r,'matchup')}"
                    # Find matching game in per_game; fallback to latest for ump if date-matched not found
                    try:
                        sel = pg_idx.loc[(ump, gdate)].iloc[0] if hasattr(pg_idx.loc[(ump, gdate)], 'iloc') else pg_idx.loc[(ump, gdate)]
                        sel = pd.Series(sel)
                    except Exception:
                        alt = per_game[per_game["plate_umpire"] == ump].sort_values("game_date").tail(1)
                        if alt.empty:
                            logging.warning(f"No per_game row for {ump} {gdate}; skipping assets.")
                            continue
                        sel = alt.iloc[0]

                    # --- Fairness Meter PNG (silent render) ---
                    sel_ump = ump
                    season_row = season_ranked[season_ranked['plate_umpire'] == sel_ump].head(1)
                    elo_v = float(season_row['elo'].iloc[0]) if not season_row.empty and not pd.isna(season_row['elo'].iloc[0]) else np.nan
                    drift_v = float(season_row['drift_z'].iloc[0]) if not season_row.empty and not pd.isna(season_row['drift_z'].iloc[0]) else np.nan
                    acc_low = sel.get('acc_low', np.nan)
                    acc_high = sel.get('acc_high', np.nan)

                    h_x = int(sel['net_favor_home_extra_strikes'])
                    a_x = int(sel['net_favor_away_extra_strikes'])
                    swing = h_x - a_x
                    swing_max = max(5, abs(h_x), abs(a_x), abs(swing)) * 1.25

                    fig, axes = plt.subplots(3, 1, figsize=(6, 7.5), dpi=300, gridspec_kw={'height_ratios':[1.2,1.2,1.0]})
                    fig.suptitle(f"Fairness Meter — {sel_ump} ({sel['game_date']})", fontsize=14, y=0.98)
                    ax = axes[0]; ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.axis('off')
                    ax.barh(0.6, 100, height=0.28, alpha=0.10)
                    if not pd.isna(acc_low) and not pd.isna(acc_high):
                        ax.barh(0.6, max(0, acc_high-acc_low), left=max(0, acc_low), height=0.28, alpha=0.25)
                    ax.barh(0.6, max(0, min(100, float(sel['accuracy_pct']))), height=0.28)
                    ax.text(0, 0.9, "Accuracy", fontsize=11, va='center')
                    ax.text(100, 0.9, f"{sel['accuracy_pct']:.1f}%", ha='right', va='center', fontsize=11)
                    ax.barh(0.15, 100, height=0.22, alpha=0.10)
                    edge_val = float(sel['edge_acc_pct']) if not pd.isna(sel['edge_acc_pct']) else np.nan
                    if not pd.isna(edge_val):
                        ax.barh(0.15, max(0, min(100, edge_val)), height=0.22)
                    ax.text(0, 0.35, "Edge Acc (±2\")", fontsize=11, va='center')
                    ax.text(100, 0.35, f"{edge_val:.1f}%" if not pd.isna(edge_val) else "—", ha='right', va='center', fontsize=11)
                    ax2 = axes[1]; ax2.set_xlim(-swing_max, swing_max); ax2.set_ylim(0, 1); ax2.axvline(0, linestyle='--', alpha=0.5); ax2.axis('off')
                    ax2.barh(0.5, -h_x, height=0.35); ax2.barh(0.5, a_x, height=0.35)
                    ax2.text(-swing_max, 0.85, f"Home: {h_x:+d}", va='center', fontsize=11)
                    ax2.text(swing_max, 0.85, f"Away: {a_x:+d}", va='center', ha='right', fontsize=11)
                    ax2.text(0, 0.1, f"Swing (H−A): {swing:+d}", ha='center', fontsize=11)
                    ax2.set_title("Net Favor (extra strikes)")
                    ax3 = axes[2]; ax3.axis('off')
                    info_lines = [
                        f"Taken Pitches: {int(sel['total_taken'])} | Misses: {int(sel['misses'])}",
                        f"Miss split: Outside→Strike {int(sel['miss_strike_outside'])} | Inside→Ball {int(sel['miss_ball_inside'])}",
                        f"ELO: {elo_v:.0f}" if not pd.isna(elo_v) else "ELO: —",
                        f"Drift z: {drift_v:+.2f}" if not pd.isna(drift_v) else "Drift z: —",
                    ]
                    ax3.text(0.02, 0.8, "\n".join(info_lines), fontsize=10, va='top')
                    ax3.text(0.02, 0.25, matchup, fontsize=11, va='center')
                    fairness_path = os.path.join(export_dir, f"fairness_{sel['game_date']}_{sel_ump.replace(' ','_')}.png")
                    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(fairness_path, bbox_inches='tight'); plt.close(fig)
                    generated_paths.append(fairness_path)

                    # --- Miss heatmaps PNG ---
                    gdf = df[(df["game_pk"] == sel["game_pk"]) & (df["taken"] == True)].copy()
                    def heat(ax, data, title):
                        ax.set_title(title, fontsize=9)
                        ax.set_xlim(-1.5, 1.5); ax.set_ylim(0, 5)
                        ax.axvline(x=-0.7083, linestyle='--'); ax.axvline(x=0.7083, linestyle='--')
                        ax.axhline(y=data["sz_bot"].median(), linestyle='--'); ax.axhline(y=data["sz_top"].median(), linestyle='--')
                        if not data.empty:
                            _ = ax.hexbin(data["plate_x"], data["plate_z"], gridsize=25, mincnt=1)
                    fig2, axs = plt.subplots(2, 2, figsize=(6,6), dpi=300)
                    heat(axs[0,0], gdf[~gdf["is_correct"]], "Misses — Overall")
                    heat(axs[0,1], gdf[gdf["is_edge_band"] & (~gdf["is_correct"])], "Misses — Edge Band")
                    heat(axs[1,0], gdf[(gdf["stand"]=='L') & (~gdf["is_correct"])], "Misses — LHB")
                    heat(axs[1,1], gdf[(gdf["stand"]=='R') & (~gdf["is_correct"])], "Misses — RHB")
                    heatmap_path = os.path.join(export_dir, f"heatmaps_{sel['game_date']}_{sel_ump.replace(' ','_')}.png")
                    fig2.savefig(heatmap_path, bbox_inches='tight'); plt.close(fig2)
                    generated_paths.append(heatmap_path)

                    # --- Tilt Card PNG ---
                    # Build a row using today's cards for this ump
                    _crow = cards_disp.loc[cards_disp["plate_umpire"]==ump].iloc[0]
                    _row = pd.Series({
                        "matchup": matchup,
                        "plate_umpire": ump,
                        "accuracy_pct": float(getattr(_crow, "accuracy_pct")) if not pd.isna(getattr(_crow, "accuracy_pct")) else float('nan'),
                        "edge_acc_pct": float(getattr(_crow, "edge_acc_pct")) if not pd.isna(getattr(_crow, "edge_acc_pct")) else float('nan'),
                        "K_lean_pp": float(getattr(_crow, "K_lean_pp", 0.0)),
                        "BB_lean_pp": float(getattr(_crow, "BB_lean_pp", 0.0)),
                        "run_boost_index": float(getattr(_crow, "run_boost_index", 0.0)),
                        "tilt_label": str(getattr(_crow, "tilt_label", "Neutral")),
                        "assign_ts": str(getattr(_crow, "assign_ts", datetime.utcnow().isoformat()))
                    })
                    tilt_card_path = os.path.join(export_dir, f"tilt_{sel['game_date']}_{sel_ump.replace(' ','_')}.png")
                    generate_plate_tilt_card_png(_row, tilt_card_path)
                    generated_paths.append(tilt_card_path)

                    # --- Composite Infographic ---
                    meta = {
                        "matchup": matchup,
                        "plate_umpire": ump,
                        "accuracy_pct": float(getattr(_crow, "accuracy_pct", np.nan)),
                        "edge_acc_pct": float(getattr(_crow, "edge_acc_pct", np.nan)),
                        "tilt_label": str(getattr(_crow, "tilt_label", "Neutral")),
                        "run_boost_index": float(getattr(_crow, "run_boost_index", 0.0)),
                    }
                    infographic_path = generate_infographic(fairness_path, heatmap_path, tilt_card_path, meta, export_dir)
                    generated_paths.append(infographic_path)

                    # --- One-line caption (txt) ---
                    caption = f"{matchup} — {ump}: {getattr(_crow,'tilt_label','Neutral')} | RBI {getattr(_crow,'run_boost_index',0.0):+.2f}. Why: {getattr(_crow,'why','')}."
                    cap_path = os.path.join(export_dir, f"caption_{sel['game_date']}_{_slug(matchup)}_{_slug(ump)}.txt")
                    with open(cap_path, 'w', encoding='utf-8') as fh:
                        fh.write(caption)
                    generated_paths.append(cap_path)

                    manifests.append({
                        "game_date": sel["game_date"],
                        "matchup": matchup,
                        "plate_umpire": ump,
                        "tilt_label": getattr(_crow, 'tilt_label', 'Neutral'),
                        "run_boost_index": getattr(_crow, 'run_boost_index', np.nan),
                        "fairness_png": fairness_path,
                        "heatmaps_png": heatmap_path,
                        "tilt_png": tilt_card_path,
                        "infographic_png": infographic_path,
                        "caption_txt": cap_path,
                    })
                except Exception as _bx:
                    logging.error(f"[batch export] failed for {getattr(r,'plate_umpire','?')}: {_bx}")

            # Write manifest CSV and thread Markdown
            if manifests:
                man_df = pd.DataFrame(manifests)
                manifest_csv = os.path.join(export_dir, f"today_manifest_{export_date.strftime('%Y%m%d')}.csv")
                man_df.to_csv(manifest_csv, index=False)
                st.success(f"Generated {len(manifests)} game asset packs → {export_dir}")
                st.download_button("⬇️ Download Manifest CSV", data=man_df.to_csv(index=False), file_name=os.path.basename(manifest_csv), mime="text/csv")

                # Thread Markdown: Top 3 by |RBI| then by confluence
                top = cards_disp.copy()
                top["abs_rbi"] = top["run_boost_index"].abs()
                top = top.sort_values(["abs_rbi","confluence_score"], ascending=[False, False]).head(3)
                lines = ["# Tonight's Plate Tilts\n"]
                for rr in top.itertuples(index=False):
                    lines.append(f"**{getattr(rr,'matchup')}** — {getattr(rr,'plate_umpire')} :: {getattr(rr,'tilt_label')} | RBI {getattr(rr,'run_boost_index'):+.2f} \nWhy: {getattr(rr,'why')}\n")
                thread_md = "\n".join(lines)
                st.download_button("⬇️ Download Top-3 Thread (Markdown)", data=thread_md, file_name=f"top3_thread_{export_date.strftime('%Y%m%d')}.md", mime="text/markdown")

                # Zip everything
                zip_base = os.path.join(export_dir, f"ump_assets_{export_date.strftime('%Y%m%d')}")
                zip_path = shutil.make_archive(zip_base, 'zip', root_dir=export_dir)
                with open(zip_path, 'rb') as fh:
                    st.download_button("⬇️ Download All Assets (zip)", data=fh.read(), file_name=os.path.basename(zip_path), mime="application/zip")
            else:
                st.info("No assets generated (no assigned umpires or per-game rows found).")

    with st.expander("Social Graphic Pack & Winible Export", expanded=False):
        export_date = st.date_input("Export Date", value=date.today())
        export_dir = os.path.join("exports", export_date.strftime("%Y%m%d"))
        os.makedirs(export_dir, exist_ok=True)

        # Pick a game for assets
        asset_game = st.selectbox("Pick a game for assets", options=[f"{r.home} vs {r.away} — {r.game_date} — {r.plate_umpire}" for r in per_game.itertuples()], index=0)
        sel_idx = [i for i,_ in enumerate(per_game.itertuples()) if f"{per_game.iloc[i]['home']} vs {per_game.iloc[i]['away']} — {per_game.iloc[i]['game_date']} — {per_game.iloc[i]['plate_umpire']}" == asset_game][0]
        sel = per_game.iloc[sel_idx]

        # Fairness Meter PNG (enhanced)
        # Pull season-level ELO & drift for the selected ump
        sel_ump = sel['plate_umpire']
        season_row = season_ranked[season_ranked['plate_umpire'] == sel_ump].head(1)
        elo_v = float(season_row['elo'].iloc[0]) if not season_row.empty and not pd.isna(season_row['elo'].iloc[0]) else np.nan
        drift_v = float(season_row['drift_z'].iloc[0]) if not season_row.empty and not pd.isna(season_row['drift_z'].iloc[0]) else np.nan
        acc_low = sel.get('acc_low', np.nan)
        acc_high = sel.get('acc_high', np.nan)

        # Compute favor swing (home vs away extra strikes)
        h_x = int(sel['net_favor_home_extra_strikes'])
        a_x = int(sel['net_favor_away_extra_strikes'])
        swing = h_x - a_x  # + favors HOME defense overall
        swing_max = max(5, abs(h_x), abs(a_x), abs(swing)) * 1.25

        fig, axes = plt.subplots(3, 1, figsize=(6, 7.5), dpi=300, gridspec_kw={'height_ratios':[1.2,1.2,1.0]})
        fig.suptitle(f"Fairness Meter — {sel_ump} ({sel['game_date']})", fontsize=14, y=0.98)

        # --- Accuracy bar with CI ---
        ax = axes[0]
        ax.set_xlim(0, 100); ax.set_ylim(0, 1)
        ax.axis('off')
        ax.barh(0.6, 100, height=0.28, alpha=0.10)
        # CI band
        if not pd.isna(acc_low) and not pd.isna(acc_high):
            ax.barh(0.6, max(0, acc_high-acc_low), left=max(0, acc_low), height=0.28, alpha=0.25)
        # Accuracy value
        ax.barh(0.6, max(0, min(100, float(sel['accuracy_pct']))), height=0.28)
        ax.text(0, 0.9, "Accuracy", fontsize=11, va='center')
        ax.text(100, 0.9, f"{sel['accuracy_pct']:.1f}%", ha='right', va='center', fontsize=11)
        # Edge accuracy
        ax.barh(0.15, 100, height=0.22, alpha=0.10)
        edge_val = float(sel['edge_acc_pct']) if not pd.isna(sel['edge_acc_pct']) else np.nan
        if not pd.isna(edge_val):
            ax.barh(0.15, max(0, min(100, edge_val)), height=0.22)
        ax.text(0, 0.35, "Edge Acc (±2\")", fontsize=11, va='center')
        ax.text(100, 0.35, f"{edge_val:.1f}%" if not pd.isna(edge_val) else "—", ha='right', va='center', fontsize=11)

        # --- Net Favor swing (centered) ---
        ax2 = axes[1]
        ax2.set_xlim(-swing_max, swing_max); ax2.set_ylim(0, 1)
        ax2.axvline(0, linestyle='--', alpha=0.5)
        ax2.axis('off')
        # home left, away right visualization
        ax2.barh(0.5, -h_x, height=0.35)  # if h_x negative (more strikes against home while batting) this draws to left
        ax2.barh(0.5, a_x, height=0.35)   # away extra strikes to the right
        ax2.text(-swing_max, 0.85, f"Home: {h_x:+d}", va='center', fontsize=11)
        ax2.text(swing_max, 0.85, f"Away: {a_x:+d}", va='center', ha='right', fontsize=11)
        ax2.text(0, 0.1, f"Swing (H−A): {swing:+d}", ha='center', fontsize=11)
        ax2.set_title("Net Favor (extra strikes)")

        # --- Info panel: volume, misses, crew, ELO, drift ---
        ax3 = axes[2]
        ax3.axis('off')
        info_lines = [
            f"Taken Pitches: {int(sel['total_taken'])} | Misses: {int(sel['misses'])}",
            f"Miss split: Outside→Strike {int(sel['miss_strike_outside'])} | Inside→Ball {int(sel['miss_ball_inside'])}",
            f"Crew Chief: {per_game.loc[per_game['game_pk']==sel['game_pk'], 'crew_chief'].dropna().iloc[0] if 'crew_chief' in per_game.columns and not per_game.loc[per_game['game_pk']==sel['game_pk'], 'crew_chief'].dropna().empty else '—'}",
            f"ELO: {elo_v:.0f}" if not pd.isna(elo_v) else "ELO: —",
            f"Drift z: {drift_v:+.2f}" if not pd.isna(drift_v) else "Drift z: —",
        ]
        ax3.text(0.02, 0.8, "\n".join(info_lines), fontsize=10, va='top')
        # Team labels
        ax3.text(0.02, 0.25, f"{sel['away']} @ {sel['home']}", fontsize=11, va='center')

        fairness_path = os.path.join(export_dir, f"fairness_{sel['game_date']}_{sel_ump.replace(' ','_')}.png")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fairness_path, bbox_inches='tight')
        plt.close(fig)
        st.image(fairness_path, caption="Fairness Meter (enhanced)")

        # Plate Tilt Card (compact, social-ready)
        tilt_card_path = os.path.join(export_dir, f"tilt_{sel['game_date']}_{sel_ump.replace(' ','_')}.png")
        try:
            # Build a minimal row with required fields for the card function
            _row = pd.Series({
                "matchup": f"{sel['away']} @ {sel['home']}",
                "plate_umpire": sel_ump,
                "accuracy_pct": float(sel["accuracy_pct"]) if not pd.isna(sel["accuracy_pct"]) else float('nan'),
                "edge_acc_pct": float(sel["edge_acc_pct"]) if not pd.isna(sel["edge_acc_pct"]) else float('nan'),
                "K_lean_pp": float(cards_disp.loc[cards_disp["plate_umpire"]==sel_ump, "K_lean_pp"].iloc[0]) if "cards_disp" in locals() and not cards_disp.loc[cards_disp["plate_umpire"]==sel_ump].empty else 0.0,
                "BB_lean_pp": float(cards_disp.loc[cards_disp["plate_umpire"]==sel_ump, "BB_lean_pp"].iloc[0]) if "cards_disp" in locals() and not cards_disp.loc[cards_disp["plate_umpire"]==sel_ump].empty else 0.0,
                "run_boost_index": float(cards_disp.loc[cards_disp["plate_umpire"]==sel_ump, "run_boost_index"].iloc[0]) if "cards_disp" in locals() and not cards_disp.loc[cards_disp["plate_umpire"]==sel_ump].empty else 0.0,
                "tilt_label": str(cards_disp.loc[cards_disp["plate_umpire"]==sel_ump, "tilt_label"].iloc[0]) if "cards_disp" in locals() and not cards_disp.loc[cards_disp["plate_umpire"]==sel_ump].empty else "Neutral",
                "assign_ts": str(cards_disp.loc[cards_disp["plate_umpire"]==sel_ump, "assign_ts"].iloc[0]) if "cards_disp" in locals() and not cards_disp.loc[cards_disp["plate_umpire"]==sel_ump].empty else datetime.utcnow().isoformat()
            })
            generate_plate_tilt_card_png(_row, tilt_card_path)
            st.image(tilt_card_path, caption="Plate Tilt Card")
        except Exception as _tc_e:
            logging.error(f"[ump_card] tilt card failed: {_tc_e}")

        # Miss heatmap grid (Overall / Edge-Band / LHB / RHB)
        gdf = df[(df["game_pk"] == sel["game_pk"]) & (df["taken"] == True)].copy()
        def heat(ax, data, title):
            ax.set_title(title, fontsize=9)
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(0, 5)
            ax.axvline(x=-0.7083, linestyle='--'); ax.axvline(x=0.7083, linestyle='--')
            ax.axhline(y=data["sz_bot"].median(), linestyle='--'); ax.axhline(y=data["sz_top"].median(), linestyle='--')
            if not data.empty:
                _ = ax.hexbin(data["plate_x"], data["plate_z"], gridsize=25, mincnt=1)
        fig2, axs = plt.subplots(2, 2, figsize=(6,6), dpi=300)
        heat(axs[0,0], gdf[~gdf["is_correct"]], "Misses — Overall")
        heat(axs[0,1], gdf[gdf["is_edge_band"] & (~gdf["is_correct"])], "Misses — Edge Band")
        heat(axs[1,0], gdf[(gdf["stand"]=='L') & (~gdf["is_correct"])], "Misses — LHB")
        heat(axs[1,1], gdf[(gdf["stand"]=='R') & (~gdf["is_correct"])], "Misses — RHB")
        heatmap_path = os.path.join(export_dir, f"heatmaps_{sel['game_date']}_{sel['plate_umpire'].replace(' ','_')}.png")
        fig2.savefig(heatmap_path, bbox_inches='tight')
        st.image(heatmap_path, caption="Miss Heatmaps")

        # --- Proof logging for selected game card ----------------------------
        try:
            matchup = f"{sel['away']} @ {sel['home']}"
            gid = f"{sel['game_date']}_{_slug(matchup)}_{_slug(sel_ump)}"
            swing = int(sel['net_favor_home_extra_strikes']) - int(sel['net_favor_away_extra_strikes'])
            # Blog-facing summary
            log_blog_pick_to_db({
                "log_date": datetime.utcnow().isoformat(),
                "matchup": f"{matchup} • {sel_ump}",
                "bet_type": "Ump Card",
                "confidence": None,
                "edge_pct": None,
                "odds": None,
                "predicted_total": None,
                "predicted_winner": None,
                "predicted_margin": None,
                "bookmaker_total": None,
                "analysis": f"Acc {sel['accuracy_pct']:.1f}% | Edge {sel['edge_acc_pct']:.1f}% | Swing H−A {swing:+d} | Taken {int(sel['total_taken'])}"
            })
            # Minimal bet-card style row (so proof_maker can include it in summaries if desired)
            log_bet_card_to_db({
                "game_id": gid,
                "home_team": sel['home'],
                "away_team": sel['away'],
                "game_time": sel['game_date'],
                "combined_runs": None,
                "delta": float(swing),
                "total_bet_rec": None,
                "ml_bet_rec": None,
                "ensemble_confidence": "ump:fairness",
                "log_date": datetime.utcnow().isoformat()
            })
            st.caption(f"🧾 Logged Ump Card summary to {BETLOGS_DB}")
        except Exception as e:
            logging.error(f"[ump_card] selected-game logging failed: {e}")

        # Story Card JSON export for Winible
        post_json = {
            "date": export_date.strftime("%Y-%m-%d"),
            "platform": "Winible",
            "post_type": "paid",
            "audience_segment": "members",
            "post_objective": "Deliver plate card & drive subs",
            "urgency_level": "medium",
            "hook_headline": f"{sel['plate_umpire']} ran {sel['accuracy_pct']:.1f}% with {int(sel['net_favor_home_extra_strikes']-sel['net_favor_away_extra_strikes']):+d} extra strikes swing.",
            "copy_blocks": [
                {"section":"Summary","text": f"Accuracy {sel['accuracy_pct']:.1f}% | Edge {sel['edge_acc_pct']:.1f}% | H {sel['net_favor_home_extra_strikes']} vs A {sel['net_favor_away_extra_strikes']} extra strikes."},
                {"section":"Why it matters","text":"Zone profile shifted outcomes on the edges. Use this to size live totals & prop risk."}
            ],
            "cta": "$9 unlocks tonight’s prop build tuned to this zone.",
            "assets": {
                "fairness_png": fairness_path,
                "heatmaps_png": heatmap_path,
                "tilt_png": tilt_card_path,
                "infographic_png": os.path.join(export_dir, f"infographic_{_slug(f"{sel['away']} @ {sel['home']}")}_{_slug(sel_ump)}.png"),
            }
        }
        st.download_button("⬇️ Download Winible Story Card JSON", data=pd.Series(post_json).to_json(), file_name=f"winible_story_{export_date.strftime('%Y%m%d')}.json", mime="application/json")

    csv1 = season_ranked.round(4).to_csv(index=False).encode()
    st.download_button("⬇️ Download Season Report (CSV)", data=csv1, file_name=f"umpire_season_{year}.csv", mime="text/csv")

    # Per-umpire explorer
    st.subheader("Per-Umpire Game Explorer")
    target_ump = st.selectbox("Select an umpire", options=sorted(season["plate_umpire"].dropna().unique().tolist()))
    one = per_game[per_game["plate_umpire"] == target_ump].copy()
    one = one.sort_values("game_date")
    st.dataframe(
        one[["game_date", "home", "away", "total_taken", "accuracy_pct",
             "called_strike_acc_pct", "ball_acc_pct", "edge_acc_pct",
             "misses", "miss_strike_outside", "miss_ball_inside",
             "net_favor_home_extra_strikes", "net_favor_away_extra_strikes",
             "acc_low", "acc_high"
            ]].round(2),
        use_container_width=True
    )
    csv2 = one.round(4).to_csv(index=False).encode()
    st.download_button(f"⬇️ Download {target_ump} Game Cards (CSV)", data=csv2,
                       file_name=f"{target_ump.replace(' ','_')}_games_{year}.csv", mime="text/csv")

    with st.expander("Pitch-Type × Handedness Bias & Zone Profile", expanded=False):
        # Bias matrix
        taken = df[df["taken"] == True].copy()
        if "pitch_type" in taken.columns and not taken["pitch_type"].isna().all():
            # League baselines per pitch_type × stand
            league = taken.groupby(["pitch_type", "stand"]).apply(lambda x: (x["is_correct"].mean()*100)).rename("league_acc").reset_index()
            ump_mat = taken.groupby(["plate_umpire", "pitch_type", "stand"]).apply(lambda x: (x["is_correct"].mean()*100)).rename("ump_acc").reset_index()
            mat = ump_mat.merge(league, on=["pitch_type","stand"], how="left")
            mat["delta"] = mat["ump_acc"] - mat["league_acc"]
            target_ump2 = st.selectbox("Select an umpire for bias view", options=sorted(season["plate_umpire"].dropna().unique().tolist()), key="biasump")
            view = mat[mat["plate_umpire"] == target_ump2]
            st.dataframe(view.sort_values(["stand","pitch_type"]), use_container_width=True)
        else:
            st.info("No pitch_type data available in this window.")

        # Simple zone profile indices
        u_for_zone = st.selectbox("Select an umpire for zone profile", options=sorted(season["plate_umpire"].dropna().unique().tolist()), key="zoneump")
        zdf = taken[taken["plate_umpire"] == u_for_zone].copy()
        if not zdf.empty:
            # Vertical tightness: called-strike rate for inside-zone near top/bottom vs league
            def v_tight(d):
                if d.empty: return np.nan
                top = d["sz_top"].median(); bot = d["sz_bot"].median()
                band = 0.25
                inside_band = d[(d["plate_z"] > bot+band) & (d["plate_z"] < top-band)]
                cs = inside_band[inside_band["is_called_strike"]]["in_zone"].mean()
                return cs*100 if not pd.isna(cs) else np.nan
            vt = v_tight(zdf)
            st.metric("Vertical Tightness (proxy)", f"{vt:.1f}%" if not pd.isna(vt) else "—")
        else:
            st.info("No taken pitches for selected ump in this window.")

    # --- Zone Profile Heatmaps (Called-Strike Probability) ---
    with st.expander("Zone Profile Heatmaps (Called-Strike Probability)", expanded=False):
        z_ump = st.selectbox("Select an umpire for heatmaps", options=sorted(season["plate_umpire"].dropna().unique().tolist()), key="zoneheatump")
        zdf = df[(df["plate_umpire"] == z_ump) & (df["taken"] == True)].copy()
        if zdf.empty:
            st.info("No taken pitches for this ump in the selected window.")
        else:
            # Bin space
            x_edges = np.linspace(-1.5, 1.5, 50)
            z_edges = np.linspace(0.5, 4.2, 50)
            # Called strike probability grid
            strikes = zdf[zdf["is_called_strike"]]
            Hs, _, _ = np.histogram2d(strikes["plate_x"], strikes["plate_z"], bins=[x_edges, z_edges])
            Ht, _, _ = np.histogram2d(zdf["plate_x"], zdf["plate_z"], bins=[x_edges, z_edges])
            P = np.divide(Hs, Ht, out=np.zeros_like(Hs), where=Ht>0)
            figp, axp = plt.subplots(figsize=(6,5), dpi=300)
            im = axp.imshow(P.T, origin='lower', extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]], aspect='auto')
            axp.axvline(-0.7083, linestyle='--'); axp.axvline(0.7083, linestyle='--')
            # Use median top/bot for guide
            axp.axhline(zdf["sz_bot"].median(), linestyle='--'); axp.axhline(zdf["sz_top"].median(), linestyle='--')
            axp.set_title(f"Called-Strike Probability — {z_ump}")
            st.pyplot(figp)

    # --- Bad-Beat Timeline (Heuristic) ---
    with st.expander("Bad-Beat Timeline (Heuristic)", expanded=False):
        # Select game
        game_opts = [f"{r.home} vs {r.away} — {r.game_date} — {r.plate_umpire}" for r in per_game.itertuples()]
        sel_game = st.selectbox("Select game", options=game_opts, key="bbgame")
        idx = [i for i,_ in enumerate(per_game.itertuples()) if f"{per_game.iloc[i]['home']} vs {per_game.iloc[i]['away']} — {per_game.iloc[i]['game_date']} — {per_game.iloc[i]['plate_umpire']}" == sel_game][0]
        gpk = int(per_game.iloc[idx]["game_pk"])
        g = df[(df["game_pk"] == gpk) & (df["taken"] == True)].copy()
        if g.empty:
            st.info("No taken pitches for this game.")
        else:
            # Heuristic leverage score
            score_diff = (g["home_score"] - g["away_score"]).abs().fillna(0)
            late = (g["inning"] >= 7).astype(int)
            full_count = ((g["balls"] == 3) & (g["strikes"] >= 2)).astype(int)
            edge = g["is_edge_band"].astype(int)
            miss = (~g["is_correct"]).astype(int)
            g["leverage_score"] = (1.5*late + 1.2*(score_diff <= 1) + 1.0*full_count + 0.5*edge) * miss
            # Favor team perspective
            g["favor_team"] = np.where(g["miss_type"] == "strike_outside", np.where(g["inning_topbot"].str.lower()=="top", g["home_team"], g["away_team"]),
                                        np.where(g["miss_type"] == "ball_inside", np.where(g["inning_topbot"].str.lower()=="top", g["away_team"], g["home_team"]), ""))
            g = g[g["leverage_score"] > 0].copy()
            g["count"] = g["balls"].astype(str) + "-" + g["strikes"].astype(str)
            cols = ["inning", "inning_topbot", "count", "miss_type", "is_edge_band", "favor_team", "leverage_score"]
            timeline = g.sort_values(["leverage_score", "inning"], ascending=[False, True])[cols].head(12)
            st.dataframe(timeline, use_container_width=True)

    # --- Prop Recommender (Beta) — Ks / Walks ---
    with st.expander("Prop Recommender (Beta) — Ks / Walks", expanded=False):
        # League baselines from taken pitches in window
        taken_all = df[df["taken"] == True]
        if taken_all.empty:
            st.info("No taken-pitch data to derive baselines.")
        else:
            league_cs_acc = (taken_all.loc[taken_all["is_called_strike"], "in_zone"].mean()) * 100
            league_ball_acc = ((~taken_all.loc[taken_all["is_ball_call"], "in_zone"]).mean()) * 100
            game_opt2 = [f"{r.home} vs {r.away} — {r.game_date} — {r.plate_umpire}" for r in per_game.itertuples()]
            pick = st.selectbox("Select game for recommendations", options=game_opt2, key="propgame")
            idx2 = [i for i,_ in enumerate(per_game.itertuples()) if f"{per_game.iloc[i]['home']} vs {per_game.iloc[i]['away']} — {per_game.iloc[i]['game_date']} — {per_game.iloc[i]['plate_umpire']}" == pick][0]
            gp = int(per_game.iloc[idx2]["game_pk"])
            ump = per_game.iloc[idx2]["plate_umpire"]
            gdf = df[(df["game_pk"] == gp) & (df["taken"] == True)].copy()
            if gdf.empty:
                st.info("No taken-pitch data for this game.")
            else:
                udf = df[(df["plate_umpire"] == ump) & (df["taken"] == True)].copy()
                u_cs_acc = (udf.loc[udf["is_called_strike"], "in_zone"].mean()) * 100 if not udf.empty else np.nan
                u_ball_acc = ((~udf.loc[udf["is_ball_call"], "in_zone"]).mean()) * 100 if not udf.empty else np.nan
                d_cs = (u_cs_acc - league_cs_acc) if not pd.isna(u_cs_acc) else 0.0
                d_ball = (u_ball_acc - league_ball_acc) if not pd.isna(u_ball_acc) else 0.0
                # Starter heuristic: first pitcher id per team
                first_by_team = gdf.sort_values("pitch_number").groupby(["home_team", "away_team"]).head(1)
                home = per_game.iloc[idx2]["home"]; away = per_game.iloc[idx2]["away"]
                # Recs
                k_lean = "Over" if d_cs > 0.5 else ("Under" if d_cs < -0.5 else "Neutral")
                bb_lean = "Over" if d_ball < -0.5 else ("Under" if d_ball > 0.5 else "Neutral")
                st.write(f"**Ump Effect — {ump}:** ΔCalled-Strike Acc = {d_cs:.2f} pp vs lg | ΔBall Acc = {d_ball:.2f} pp vs lg")
                st.write(f"**Strikeouts:** {k_lean} (both starters) | **Walks:** {bb_lean}")
                st.caption("Rule of thumb: +0.5 pp or more moves lean to Over (Ks) or Under (BBs) respectively.")

    # Quick charts
    st.subheader("Quick Charts")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.caption("Accuracy vs Volume")
        st.scatter_chart(season_ranked, x="total_taken", y="accuracy_pct", size=None, color="plate_umpire")
    with cc2:
        st.caption("Edge-Band Accuracy (2 in) vs Overall Accuracy")
        st.scatter_chart(season_ranked, x="edge_acc_pct", y="accuracy_pct", size=None, color="plate_umpire")

    st.info(
        "Definitions: "
        "• **Accuracy%** = Correct calls / Taken pitches. "
        "• **Called-Strike Acc%** = Share of called strikes that were inside zone. "
        "• **Ball Acc%** = Share of ball calls that were outside zone. "
        "• **Edge Acc%** = Accuracy for pitches within 2 inches of the zone boundary. "
        "• **Net Favor (extra strikes)** = Sum of missed calls (+1 for outside-zone strikes, −1 for inside-zone balls), "
        "reported by which team was on defense while the batting team experienced those calls."
    )

    st.success("Done. Adjust the margin slider if you want a stricter/looser plate, then re-run.")