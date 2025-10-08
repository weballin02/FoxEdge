# market_mood.py
# Streamlit dashboard for DKNetwork splits with correct pairing, signals, CLV, and sane state handling.

import os
import re
import json
import time
import math
import pytz
import qrcode
import logging
import sqlite3
import requests
import pandas as pd
import altair as alt
from io import StringIO
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_autorefresh import st_autorefresh

# ------------------------------ CONFIG ------------------------------

PAGE_TITLE = "FoxEdge Market Snapshot"
TZ = pytz.timezone("America/Los_Angeles")

BETLOGS_ROOT = os.getenv("BETLOGS_ROOT", os.getcwd())
BETLOGS_DB = os.path.join(BETLOGS_ROOT, "bet_logs.db")
SNAPSHOT_DIR = Path("snapshots")
EXPORTS_DIR = Path("exports")
HISTORY_FILE = Path("mood_history.csv")
SPLITS_HISTORY_FILE = Path("splits_history.csv")

SNAPSHOT_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)

# DK event groups; override via sidebar if needed
EG_MLB = 84240

# Signal thresholds (sidebar-adjustable)
DEFAULT_RLM_BETS_PP = 5.0      # percentage points increase in %bets
DEFAULT_RLM_PROB_PP = -0.5     # implied prob delta must be <= this (i.e., go down) to count as RLM
DEFAULT_STEAM_PROB_PP = 1.0    # absolute change in implied prob (pp) to call steam
DEFAULT_ODDLOT_PP = 10.0       # tickets vs handle divergence (pp) on normalized two-sided share
DEFAULT_EXPOSURE_RATIO = 0.25  # liability proxy advantage vs the other side

GATE_THRESHOLD = 60.0          # UI gate for "irrational" market

# ------------------------------ UTIL ------------------------------

def now_pt():
    return datetime.now(TZ)

def ts_iso_utc():
    return datetime.utcnow().isoformat()

def implied_prob_from_odds(odds: int) -> float:
    """American odds to implied probability in [0,1]."""
    if pd.isna(odds):
        return math.nan
    try:
        o = int(str(odds).replace("−", "-"))
    except Exception:
        return math.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    elif o < 0:
        return abs(o) / (abs(o) + 100.0)
    return 0.5

def payout_multiplier(odds: int) -> float:
    """Expected payout multiple per $1 stake (approx, ignoring hold)."""
    if odds >= 0:
        return odds / 100.0
    return 100.0 / abs(odds)

def safe_int(x, default=None):
    try:
        return int(str(x).replace("−", "-"))
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def parse_total_side(side_text: str):
    """Return ('Over'|'Under', float_total) or (None,None)."""
    s = str(side_text or "").strip()
    m = re.match(r"(?i)^(over|under)\s+([0-9]+(?:\.[0-9])?)", s)
    if not m:
        return None, None
    return m.group(1).title(), float(m.group(2))

# ------------------------------ DATA FETCH ------------------------------

def fetch_dk_splits(event_group: int, date_range: str = "today", timeout=20) -> pd.DataFrame:
    """
    Scrape DKNetwork betting splits pages (handles pagination).
    Returns tidy DataFrame with columns:
      matchup, game_time, market, side, odds, %handle, %bets, update_time
    """
    base = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
    params = {"tb_eg": event_group, "tb_edate": date_range, "tb_emt": "0"}
    first_url = base + "?" + "&".join(f"{k}={v}" for k, v in params.items())

    def get_html(url):
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text

    def discover_pages(html):
        soup = BeautifulSoup(html, "html.parser")
        urls = {first_url}
        pag = soup.select_one("div.tb_pagination")
        if pag:
            for a in pag.find_all("a", href=True):
                if "tb_page=" in a["href"]:
                    urls.add(a["href"])
        # sort by tb_page= if present
        def page_num(u):
            m = re.search(r"tb_page=(\d+)", u)
            return int(m.group(1)) if m else 1
        return [u for u in sorted(urls, key=page_num)]

    def parse_page(html):
        soup = BeautifulSoup(html, "html.parser")
        games = soup.select("div.tb-se")
        out = []
        now = now_pt()
        for g in games:
            title_node = g.select_one("div.tb-se-title h5")
            if not title_node:
                continue
            matchup = title_node.get_text(strip=True)
            # time text not reliably parseable; keep raw
            tnode = g.select_one("div.tb-se-title span")
            gtime = (tnode.get_text(strip=True) if tnode else "").replace("\xa0", " ")
            # market sections
            for section in g.select(".tb-market-wrap > div"):
                head = section.select_one(".tb-se-head > div")
                if not head:
                    continue
                market_name = head.get_text(strip=True)
                # We support Moneyline, Spread, Total; ML/Total are most reliable
                if market_name not in ("Moneyline", "Spread", "Total"):
                    continue
                for row in section.select(".tb-sodd"):
                    side_el = row.select_one(".tb-slipline")
                    odds_el = row.select_one("a.tb-odd-s")
                    if not side_el or not odds_el:
                        continue
                    side_raw = side_el.get_text(strip=True)
                    odds_raw = odds_el.get_text(strip=True)
                    odds_val = safe_int(odds_raw, default=None)
                    # find first two percent texts in the row (%handle then %bets)
                    pct_texts = [s.strip().replace("%", "") for s in row.find_all(string=lambda t: "%" in t)]
                    pct_handle, pct_bets = (pct_texts + ["", ""])[:2]
                    out.append({
                        "matchup": matchup,
                        "game_time": gtime,
                        "market": market_name,
                        "side": side_raw,
                        "odds": odds_val,
                        "%handle": pd.to_numeric(pct_handle, errors="coerce"),
                        "%bets": pd.to_numeric(pct_bets, errors="coerce"),
                        "update_time": now
                    })
        return out

    try:
        first_html = get_html(first_url)
    except Exception as e:
        logging.error(f"fetch_dk_splits first page failed: {e}")
        return pd.DataFrame(columns=["matchup","game_time","market","side","odds","%handle","%bets","update_time"])

    pages = discover_pages(first_html)
    records = []
    for url in pages:
        html = first_html if url == first_url else get_html(url)
        records.extend(parse_page(html))

    df = pd.DataFrame.from_records(records)
    if df.empty and date_range == "today":
        # Try tomorrow (DK sometimes shifts content boundary)
        return fetch_dk_splits(event_group, "tomorrow", timeout=timeout)

    # clean/normalize
    for c in ["%handle", "%bets"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df = df.dropna(subset=["matchup", "market", "side"])
    # normalize keys for grouping
    df["market_norm"] = df["market"].str.strip().str.lower()
    df["side_norm"] = df["side"].str.strip().str.lower()
    return df.reset_index(drop=True)

# ------------------------------ SQLITE ------------------------------

def with_conn(db_path=BETLOGS_DB):
    return sqlite3.connect(db_path)

def ensure_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS line_snapshots (
            game_id TEXT,
            matchup TEXT,
            market TEXT,
            side TEXT,
            book TEXT,
            snapshot_type TEXT,  -- OPEN|CLOSE|YOUR_REC
            odds INTEGER,
            total REAL,
            timestamp_utc TEXT,
            PRIMARY KEY (game_id, market, side, book, snapshot_type)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clv_logs (
            game_id TEXT,
            matchup TEXT,
            market TEXT,
            side TEXT,
            book TEXT,
            entry_odds INTEGER,
            close_odds INTEGER,
            entry_total REAL,
            close_total REAL,
            clv_prob_pp REAL,
            clv_line_move REAL,
            computed_utc TEXT,
            PRIMARY KEY (game_id, market, side, book)
        )
    """)
    conn.execute("""
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
        )
    """)

def mk_game_id(date_iso: str, matchup: str, market: str, side: str) -> str:
    gid = f"{date_iso}_{matchup.replace(' ','_')}_{market}_{side.replace(' ','_')}"
    return gid[:128]

def snapshot_rows(conn, df: pd.DataFrame, snap_type: str, date_iso: str, book="DK"):
    ensure_tables(conn)
    rows = []
    for _, r in df.iterrows():
        market = str(r["market"])
        side = str(r["side"])
        total_val = None
        if market.lower().startswith("total"):
            ou, tot = parse_total_side(side)
            total_val = tot
        gid = mk_game_id(date_iso, r["matchup"], market, side)
        rows.append((
            gid, r["matchup"], market, side, book, snap_type,
            None if pd.isna(r["odds"]) else int(r["odds"]),
            None if total_val is None else float(total_val),
            ts_iso_utc()
        ))
    conn.executemany("""
        INSERT OR IGNORE INTO line_snapshots
        (game_id, matchup, market, side, book, snapshot_type, odds, total, timestamp_utc)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, rows)

def clv_line_move(market: str, side: str, entry_total, close_total):
    if market.lower().startswith("total") and entry_total is not None and close_total is not None:
        move = close_total - entry_total  # positive means market went up
        if side.lower().startswith("over"):
            return move  # positive = good for Over rec
        if side.lower().startswith("under"):
            return -move # positive = good for Under rec
    return None

def compute_and_persist_clv(conn):
    ensure_tables(conn)
    snaps = pd.read_sql_query("SELECT * FROM line_snapshots WHERE book='DK'", conn)
    if snaps.empty:
        return pd.DataFrame(columns=[
            "game_id","matchup","market","side","book",
            "entry_odds","close_odds","entry_total","close_total",
            "clv_prob_pp","clv_line_move","computed_utc"
        ])
    keys = ["game_id","matchup","market","side","book"]
    entry = snaps[snaps["snapshot_type"]=="YOUR_REC"][keys+["odds","total"]].rename(columns={"odds":"entry_odds","total":"entry_total"})
    close = snaps[snaps["snapshot_type"]=="CLOSE"][keys+["odds","total"]].rename(columns={"odds":"close_odds","total":"close_total"})
    merged = entry.merge(close, on=keys, how="inner")
    if merged.empty:
        return pd.DataFrame(columns=[
            "game_id","matchup","market","side","book",
            "entry_odds","close_odds","entry_total","close_total","clv_prob_pp","clv_line_move","computed_utc"
        ])

    def clv_prob_pp(r):
        eo, co = r["entry_odds"], r["close_odds"]
        if pd.isna(eo) or pd.isna(co):
            return None
        pe = implied_prob_from_odds(int(eo)) * 100.0
        pc = implied_prob_from_odds(int(co)) * 100.0
        return round(pc - pe, 3)

    merged["clv_prob_pp"] = merged.apply(clv_prob_pp, axis=1)
    merged["clv_line_move"] = merged.apply(lambda r: clv_line_move(r["market"], r["side"], r["entry_total"], r["close_total"]), axis=1)
    merged["computed_utc"] = ts_iso_utc()

    rows = merged[[
        "game_id","matchup","market","side","book",
        "entry_odds","close_odds","entry_total","close_total","clv_prob_pp","clv_line_move","computed_utc"
    ]].to_records(index=False)
    conn.executemany("""
        INSERT INTO clv_logs (game_id, matchup, market, side, book, entry_odds, close_odds, entry_total, close_total, clv_prob_pp, clv_line_move, computed_utc)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(game_id, market, side, book) DO UPDATE SET
            entry_odds=excluded.entry_odds,
            close_odds=excluded.close_odds,
            entry_total=excluded.entry_total,
            close_total=excluded.close_total,
            clv_prob_pp=excluded.clv_prob_pp,
            clv_line_move=excluded.clv_line_move,
            computed_utc=excluded.computed_utc
    """, list(rows))
    return merged

def log_blog_pick(conn, pick: dict):
    ensure_tables(conn)
    cols = ["log_date","matchup","bet_type","confidence","edge_pct","odds",
            "predicted_total","predicted_winner","predicted_margin","bookmaker_total","analysis"]
    row = [pick.get(k) for k in cols]
    conn.execute(f"INSERT INTO blog_pick_logs ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", row)

# ------------------------------ SIGNALS ------------------------------

def iter_pairs(df: pd.DataFrame):
    """Yield (matchup, market_str, rowA, rowB) for valid two-side markets."""
    for (matchup, market_norm), g in df.groupby(["matchup","market_norm"]):
        gg = g.sort_values("side_norm")
        if len(gg) != 2:
            continue
        yield matchup, gg["market"].iloc[0], gg.iloc[0], gg.iloc[1]

def exposure_proxy(handle_pct, odds):
    return (handle_pct/100.0) * payout_multiplier(int(odds))

def exposure_signal(a, b, min_ratio=DEFAULT_EXPOSURE_RATIO):
    la = exposure_proxy(a["%handle"], a["odds"])
    lb = exposure_proxy(b["%handle"], b["odds"])
    hot = "A" if la > lb else "B"
    diff_ratio = abs(la - lb) / max(la, lb) if max(la, lb) > 0 else 0.0
    return hot, diff_ratio >= min_ratio, la, lb, diff_ratio

def oddlot_divergence(a, b, min_gap_pp=DEFAULT_ODDLOT_PP):
    # normalize each side to percent share among the two sides
    tA = a["%bets"]; tB = b["%bets"]
    hA = a["%handle"]; hB = b["%handle"]
    if any(pd.isna(x) for x in [tA,tB,hA,hB]) or (tA+tB)==0 or (hA+hB)==0:
        return None, 0.0, False
    tA_norm = tA/(tA+tB)*100.0
    hA_norm = hA/(hA+hB)*100.0
    gap = abs(tA_norm - hA_norm)
    side = "A" if (tA_norm - hA_norm) > 0 else "B"
    return side, gap, gap >= min_gap_pp

def rlm_signal(curr, prev, min_bets_pp=DEFAULT_RLM_BETS_PP, max_prob_pp=DEFAULT_RLM_PROB_PP):
    # bets must rise by >= min_bets_pp, implied prob must decline by <= max_prob_pp
    dbets = curr["%bets"] - prev["%bets"]
    p_now = implied_prob_from_odds(int(curr["odds"])) * 100.0 if not pd.isna(curr["odds"]) else math.nan
    p_prev = implied_prob_from_odds(int(prev["odds"])) * 100.0 if not pd.isna(prev["odds"]) else math.nan
    if any(math.isnan(x) for x in [p_now, p_prev]):
        return False, dbets, math.nan
    dprob = p_now - p_prev
    return (dbets >= min_bets_pp and dprob <= max_prob_pp), dbets, dprob

def steam_signal(curr, prev, min_prob_pp=DEFAULT_STEAM_PROB_PP):
    p_now = implied_prob_from_odds(int(curr["odds"])) * 100.0 if not pd.isna(curr["odds"]) else math.nan
    p_prev = implied_prob_from_odds(int(prev["odds"])) * 100.0 if not pd.isna(prev["odds"]) else math.nan
    if any(math.isnan(x) for x in [p_now, p_prev]):
        return False, math.nan
    dprob_abs = abs(p_now - p_prev)
    return dprob_abs >= min_prob_pp, dprob_abs

# ------------------------------ HISTORY ------------------------------

def update_mood_history(mood_score: float) -> pd.DataFrame:
    today = now_pt().strftime("%Y-%m-%d")
    if HISTORY_FILE.exists():
        hist = pd.read_csv(HISTORY_FILE)
    else:
        hist = pd.DataFrame(columns=["date","score"])
    hist = hist[hist["date"] != today]
    hist = pd.concat([hist, pd.DataFrame([{"date": today, "score": mood_score}])], ignore_index=True).tail(7)
    hist.to_csv(HISTORY_FILE, index=False)
    hist["date"] = pd.to_datetime(hist["date"])
    return hist

def update_splits_history(df: pd.DataFrame) -> pd.DataFrame:
    ts = now_pt().strftime("%Y-%m-%d %H:%M")
    df_h = df.copy()
    df_h["snapshot_time"] = ts
    if SPLITS_HISTORY_FILE.exists():
        hist = pd.read_csv(SPLITS_HISTORY_FILE)
    else:
        hist = pd.DataFrame()
    # previous snapshot rows if any
    prev = pd.DataFrame(columns=df_h.columns)
    if not hist.empty:
        last_ts = hist["snapshot_time"].iloc[-1]
        prev = hist[hist["snapshot_time"] == last_ts].copy()
    hist = pd.concat([hist, df_h], ignore_index=True)
    hist.to_csv(SPLITS_HISTORY_FILE, index=False)
    return prev

# ------------------------------ IMAGE ASSETS ------------------------------

def build_mood_ball_overlay(df: pd.DataFrame):
    try:
        base = Image.open("mood_ball.png").convert("RGBA")
    except Exception:
        return None
    irr = (df["%bets"] - df["%handle"]).abs().fillna(0.0)
    rows = df.copy()
    rows["irr"] = irr
    rows = rows.sort_values("irr", ascending=False).head(4)
    gradient = [(0,'#00ff00'),(25,'#ffff00'),(50,'#ffa500'),(75,'#ff0000'),(100,'#ff3c78')]
    def hex2rgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    mask = Image.new("L", base.size, 0)
    draw = ImageDraw.Draw(mask)
    w,h = base.size
    inset = int(min(w,h)*0.08)
    rows_list = rows.to_dict("records")
    for i in range(4):
        if i >= len(rows_list): break
        irr_val = rows_list[i]["irr"]
        stop_idx = min(int(irr_val//25), len(gradient)-1)
        rgb = hex2rgb(gradient[stop_idx][1]) + (150,)
        qmask = Image.new("L", base.size, 0)
        qdraw = ImageDraw.Draw(qmask)
        qdraw.pieslice([(inset,inset),(w-inset,h-inset)], start=i*90, end=(i+1)*90, fill=200)
        quad = Image.new("RGBA", base.size, rgb)
        overlay = Image.composite(quad, overlay, qmask)
    return Image.alpha_composite(overlay, base)

def export_image(img: Image.Image, prefix: str) -> Path:
    ts = now_pt().strftime("%Y%m%d_%H%M%S")
    out = EXPORTS_DIR / f"{prefix}_{ts}.png"
    img.save(out)
    return out

def make_winible_link(campaign: str) -> str:
    return f"https://winible.com/foxedgeai?utm_source=streamlit&utm_campaign={campaign}"

# ------------------------------ STREAMLIT UI ------------------------------

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stMetric {border: 2px solid #ff3c78; border-radius: 8px; padding: 12px; background-color: #1a1a1d;}
#MainMenu, footer {visibility: hidden;}
.streamlit-expanderHeader {background-color: #1f1f23 !important; border-radius: 4px; padding: 5px;}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Select View", ["Overview", "Game Details"])

refresh_interval = st.sidebar.slider("Auto-refresh interval (sec)", 10, 300, 60, 5)
st_autorefresh(interval=refresh_interval*1000, key="auto_refresh")

event_group = st.sidebar.selectbox("League (DK EG code)", options=[EG_MLB], index=0)
rlm_bets_pp  = st.sidebar.slider("RLM min +Δ%bets (pp)", 0.0, 20.0, DEFAULT_RLM_BETS_PP, 0.5)
rlm_prob_pp  = st.sidebar.slider("RLM max Δprob (pp)", -5.0, 0.0, DEFAULT_RLM_PROB_PP, 0.1, format="%.1f")
steam_pp     = st.sidebar.slider("Steam |Δprob| (pp)", 0.5, 5.0, DEFAULT_STEAM_PROB_PP, 0.1, format="%.1f")
oddlot_pp    = st.sidebar.slider("Odd-lot gap (pp)", 0.0, 30.0, DEFAULT_ODDLOT_PP, 0.5)
expos_ratio  = st.sidebar.slider("Exposure ratio", 0.0, 1.0, DEFAULT_EXPOSURE_RATIO, 0.05)

# Stable daily flags
today_key = now_pt().strftime("%Y%m%d")
if "daily_key" not in st.session_state:
    st.session_state["daily_key"] = today_key
for k in ("open_sql_logged","close_sql_logged","clv_sql_logged"):
    st.session_state.setdefault(k, False)

st.title("⚾ MLB Market Mood & Public Fade Dashboard")
st.caption(f"bet_logs → {BETLOGS_DB}")
st.write(f"As of {now_pt().strftime('%b %d, %Y • %I:%M %p PT')}")

# Fetch splits
splits_df = fetch_dk_splits(event_group, "today")
if splits_df.empty:
    st.warning("No splits available for today. Showing tomorrow if present.")
    splits_df = fetch_dk_splits(event_group, "tomorrow")

# Save snapshot CSV
snap_path = SNAPSHOT_DIR / f"splits_snapshot_{today_key}.csv"
splits_df.to_csv(snap_path, index=False)

# Display quick tables
with st.expander("Raw Splits"):
    st.dataframe(splits_df.drop(columns=["market_norm","side_norm"]), use_container_width=True)

# Mood / PFR
splits_df["irr"] = (splits_df["%bets"] - splits_df["%handle"]).abs()
mood_score = float(splits_df["irr"].mean()) if not splits_df.empty else 0.0

hist = update_mood_history(mood_score)
spark = alt.Chart(hist).mark_line(point=True).encode(x='date:T', y='score:Q').properties(width=700, height=120)
st.subheader("Mood Trend (Last 7 days)")
st.altair_chart(spark, use_container_width=True)
st.metric(label="Market Irrationality Index", value=f"{mood_score:.1f}%")

# Pro signals
st.markdown("---")
st.subheader("🔍 Pro Edge Signals")
pro_signals = []

# Previous snapshot map for RLM/Steam (keyed by matchup|market_norm|side_norm)
prev_map = st.session_state.get("prev_map", {})

def key_of(row):
    return f"{row['matchup']}|{row['market_norm']}|{row['side_norm']}"

for matchup, market_str, A, B in iter_pairs(splits_df):
    # Exposure
    hot_side, hit, la, lb, ratio = exposure_signal(A, B, min_ratio=expos_ratio)
    if hit:
        pro_signals.append(f"{matchup} [{market_str}]: Exposure hot side {hot_side} (ratio {ratio:.2f})")

    # Odd-lot
    side_ol, gap, ok = oddlot_divergence(A, B, min_gap_pp=oddlot_pp)
    if ok:
        pro_signals.append(f"{matchup} [{market_str}]: Odd-lot {gap:.1f}pp on side {side_ol}")

    # RLM + Steam per side
    for row in (A, B):
        k = key_of(row)
        prev = prev_map.get(k)
        if prev:
            is_rlm, dbets, dprob = rlm_signal(row, prev, min_bets_pp=rlm_bets_pp, max_prob_pp=rlm_prob_pp)
            if is_rlm:
                pro_signals.append(f"{matchup} [{market_str}] {row['side']}: ⚡ RLM (+{dbets:.1f}pp bets, {dprob:.2f}pp prob)")
            is_steam, dpp = steam_signal(row, prev, min_prob_pp=steam_pp)
            if is_steam:
                pro_signals.append(f"{matchup} [{market_str}] {row['side']}: 💨 Steam (|Δprob| {dpp:.2f}pp)")
        prev_map[k] = {"%bets": row["%bets"], "odds": row["odds"]}

st.session_state["prev_map"] = prev_map

if pro_signals:
    for s in pro_signals:
        st.markdown(f"- {s}")
else:
    st.success("✅ No actionable distortions detected.")

# Public Fade dashboard
st.subheader("📉 Public Fade Ratio")
pub = splits_df.sort_values("%bets", ascending=False)
pub_top = pub[pub["%bets"] > 70]
if pub_top.empty:
    st.write("No heavy public leans > 70% right now.")
else:
    chart = (
        alt.Chart(pub_top)
           .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
           .encode(x=alt.X('side:N', sort='-y'), y=alt.Y('%bets:Q'), color=alt.Color('%bets:Q', scale=alt.Scale(domain=[0,100], range=['#ff3c78','#001f3f'])), tooltip=['matchup','market','side','%bets','%handle'])
           .properties(width=500, height=320)
    )
    st.altair_chart(chart, use_container_width=True)

# Game details page
if page == "Game Details":
    st.header("Game Details")
    for m in splits_df["matchup"].unique():
        sub = splits_df[splits_df["matchup"] == m]
        irr = float((sub["%bets"] - sub["%handle"]).abs().mean())
        with st.expander(f"{m} • Avg Irrationality {irr:.1f}%", expanded=False):
            st.dataframe(sub[["market","side","odds","%bets","%handle","irr"]].rename(columns={"irr":"irrationality"}), use_container_width=True)

# OPEN / CLOSE / CLV automation
date_iso = now_pt().date().isoformat()
with with_conn(BETLOGS_DB) as con:
    ensure_tables(con)
    # Log OPEN once/day
    if not st.session_state["open_sql_logged"]:
        snapshot_rows(con, splits_df, "OPEN", date_iso)
        st.session_state["open_sql_logged"] = True
        st.caption("📼 OPEN snapshots logged.")

    # Manual controls
    with st.sidebar.expander("CLV Tools", expanded=False):
        close_hour = st.slider("Auto CLOSE hour (PT)", 20, 23, 23, 1, key="auto_close_hour")
        if st.button("📥 Log CLOSE now"):
            snapshot_rows(con, splits_df, "CLOSE", date_iso)
            st.session_state["close_sql_logged"] = True
            st.success("CLOSE snapshots recorded.")
        if st.button("🧮 Compute CLV"):
            clv_df = compute_and_persist_clv(con)
            st.session_state["clv_df"] = clv_df
            st.success(f"CLV computed for {0 if clv_df is None else len(clv_df)} picks.")
        st.markdown(f"**Status:** OPEN={'✅' if st.session_state['open_sql_logged'] else '❌'} • CLOSE={'✅' if st.session_state.get('close_sql_logged', False) else '❌'} • CLV={'✅' if st.session_state.get('clv_sql_logged', False) else '❌'}")

    # Auto CLOSE + CLV once per day after hour
    if now_pt().hour >= int(st.session_state.get("auto_close_hour", 23)):
        if not st.session_state.get("close_sql_logged", False):
            with with_conn(BETLOGS_DB) as con2:
                snapshot_rows(con2, splits_df, "CLOSE", date_iso)
            st.session_state["close_sql_logged"] = True
            st.caption("📼 CLOSE snapshots logged automatically.")
        if not st.session_state.get("clv_sql_logged", False) and st.session_state.get("close_sql_logged", False):
            with with_conn(BETLOGS_DB) as con3:
                clv_df_auto = compute_and_persist_clv(con3)
            st.session_state["clv_df"] = clv_df_auto
            st.session_state["clv_sql_logged"] = True
            st.caption(f"🧮 CLV computed automatically for {0 if clv_df_auto is None else len(clv_df_auto)} picks.")

# Movement table vs previous snapshot on disk
prev_splits = update_splits_history(splits_df)
if not prev_splits.empty:
    merged = splits_df.merge(prev_splits[["matchup","market","side","%bets","%handle"]], on=["matchup","market","side"], how="left", suffixes=("","_prev"))
    merged["bets_delta_pp"] = merged["%bets"] - merged["%bets_prev"]
    merged["handle_delta_pp"] = merged["%handle"] - merged["%handle_prev"]
    with st.expander("📊 Split Movements Since Last Run"):
        st.dataframe(merged[["matchup","market","side","%bets","%bets_prev","bets_delta_pp","%handle","%handle_prev","handle_delta_pp"]], use_container_width=True)

# Mood-ball and share assets
tinted = build_mood_ball_overlay(splits_df)
if tinted is not None:
    ball_path = export_image(tinted, "mood_ball")
    st.image(ball_path, caption="Market Mood Ball (Heatmap)", use_container_width=True)
    # QR
    qr_link = make_winible_link("qr_cta")
    qri = qrcode.make(qr_link)
    qr_path = export_image(qri, "qr")
    # Caption
    top_fade = splits_df.sort_values("%bets", ascending=False).head(1)
    if not top_fade.empty:
        cap = f"Market irrationality {mood_score:.1f}%. Top public: {top_fade.iloc[0]['matchup']} {top_fade.iloc[0]['side']} at {top_fade.iloc[0]['%bets']:.0f}% bets. 🔓 $5 unlocks full card."
    else:
        cap = f"Market irrationality {mood_score:.1f}%. 🔓 $5 unlocks full card."
    st.subheader("Suggested Caption")
    st.markdown(
        f"""
        <div style="background:#1a1a1d;border:2px solid #ff3c78;border-radius:8px;padding:12px;color:#fff;">
        {cap}
        </div>
        """, unsafe_allow_html=True
    )
    # Markdown snippet
    st.subheader("Markdown Snippet")
    md = f"![Market Mood Ball]({ball_path})\n\n{cap}\n\n[Unlock the Full Playbook]({make_winible_link('md_cta')})\n\n![Scan to Subscribe]({qr_path})"
    st.code(md)

# CLV table if computed
if isinstance(st.session_state.get("clv_df"), pd.DataFrame) and not st.session_state["clv_df"].empty:
    st.subheader("📈 CLV Results (Your Picks vs Closing Line)")
    show_cols = ["matchup","market","side","entry_odds","close_odds","entry_total","close_total","clv_prob_pp","clv_line_move"]
    st.dataframe(st.session_state["clv_df"][show_cols], use_container_width=True)