import os
import io
import time
import math
import numpy as np
import pandas as pd
import streamlit as st

# --------------- Page config & style ---------------
st.set_page_config(
    page_title="FoxEdge MLB Props",
    page_icon="🦊",
    layout="wide"
)

CUSTOM_CSS = """
<style>
/* Typography */
html, body, [class*="css"] {
  font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
}
/* Cards */
.block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
.stat-card {background: #0f172a; border: 1px solid #1f2937; padding: 14px 16px; border-radius: 12px; color: #e5e7eb;}
.stat-value {font-size: 26px; font-weight: 700; margin-bottom: 2px;}
.stat-label {font-size: 12px; opacity: 0.8; letter-spacing: .02em;}
/* Pills */
.pill {display:inline-block; padding:4px 10px; border-radius:999px; font-weight:600; font-size:12px; border:1px solid #e5e7eb22;}
.pill-green {background:#052e1a; color:#86efac; border-color:#16a34a33;}
.pill-red {background:#2e0b0b; color:#fda4af; border-color:#ef444433;}
.pill-amber {background:#3a2b0a; color:#fcd34d; border-color:#f59e0b33;}
/* Dataframe tweaks */
thead tr th {background:#0b1220 !important; color:#cbd5e1 !important;}
tbody tr td {color:#e2e8f0;}
/* Section headers */
h2, h3 {letter-spacing:.01em}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------- Odds math helpers ---------------

def american_to_prob(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_to_multiplier(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def prob_to_american(p: float):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(p):
        return np.nan
    p = max(min(p, 1 - 1e-6), 1e-6)
    if p >= 0.5:
        return int(round(-(p / (1 - p)) * 100.0))
    return int(round(((1 - p) / p) * 100.0))


def kelly_fraction_decimal(p: float, dec: float) -> float:
    # dec is decimal odds; net payout multiplier b = dec-1
    b = max(dec - 1.0, 1e-9)
    f = (p * b - (1 - p)) / b
    return max(0.0, f)


def remove_vig_two_way(q_over_raw: float, q_under_raw: float) -> tuple[float, float]:
    s = q_over_raw + q_under_raw
    if s <= 0:
        return (float('nan'), float('nan'))
    return (q_over_raw / s, q_under_raw / s)

# --------------- Helpers ---------------
def load_priced_df(uploaded_file: io.BytesIO | None, default_path: str | None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if default_path and os.path.exists(default_path):
        return pd.read_csv(default_path)
    return pd.DataFrame()

def style_recs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    def color_row(row):
        if str(row.get("rec","")).upper() == "BET":
            return ["background-color: rgba(22,163,74,0.10)"] * len(row)
        return [""] * len(row)
    styled = df.style.apply(color_row, axis=1).format({
        "fair_prob": "{:.3f}",
        "edge": "{:.3%}",
        "ev": "{:.3f}",
        "kelly": "{:.3f}",
        "line": "{:.1f}"
    })
    return styled

def df_download_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# --------------- Sidebar ---------------
with st.sidebar:
    st.title("🦊 FoxEdge")
    st.write("Filter and publish MLB player prop recommendations.")
    uploaded = st.file_uploader("Upload priced props CSV", type=["csv"], help="Use the output from mlb_prop_pricer.py (priced_props.csv).")
    default_path = st.text_input("...or path to CSV on disk", value="priced_props.csv")
    st.caption("If both are provided, upload takes precedence.")

    st.markdown("---")
    st.subheader("Filters")
    only_bets = st.toggle("Only show BETs", value=True)
    market_filter = st.multiselect("Markets", options=[], default=None, placeholder="Populates after CSV loads")
    selection_filter = st.multiselect("Sides", options=["Over","Under"], default=None)
    min_ev = st.slider("Minimum EV per $1", min_value=0.0, max_value=0.5, value=0.03, step=0.005)
    min_edge = st.slider("Minimum Edge", min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.2f")
    min_books = st.slider("Min. books at line", min_value=1, max_value=5, value=2, step=1)
    max_abs_price = st.slider("Max |book_price| (juice guard)", min_value=110, max_value=250, value=160, step=5)
    st.markdown("---")
    player_query = st.text_input("Player search", placeholder="e.g., Aaron Judge")
    team_query = st.text_input("Team search", placeholder="e.g., Yankees")
    st.markdown("---")
    sort_by = st.selectbox("Sort by", ["ev","edge","kelly","fair_prob","book_price"])
    sort_desc = st.toggle("Descending", value=True)

    st.markdown("---")
    st.subheader("Top Card")
    card_limit = st.slider("Max bets on card", min_value=50, max_value=800, value=200, step=10)
    card_min_kelly = st.slider("Min Kelly for card", min_value=0.0, max_value=0.05, value=0.01, step=0.001, format="%.3f")
    card_max_per_team_market = st.number_input("Max per team per market", min_value=1, max_value=5, value=2, step=1)
    require_tb_consensus3 = st.toggle("Require 3+ books for Total Bases on card", value=True)

    st.markdown("---")
    st.subheader("Elite Bets (Best of Best)")
    elite_limit = st.slider("Max elite bets", min_value=5, max_value=150, value=50, step=5)
    elite_min_ev = st.slider("Min EV for elite", min_value=0.00, max_value=0.20, value=0.06, step=0.005, format="%.3f")
    elite_min_edge = st.slider("Min edge for elite", min_value=0.00, max_value=0.30, value=0.08, step=0.01, format="%.2f")
    elite_min_kelly = st.slider("Min Kelly for elite", min_value=0.000, max_value=0.050, value=0.015, step=0.001, format="%.3f")
    elite_min_books = st.slider("Min books (elite)", min_value=1, max_value=6, value=3, step=1)
    elite_max_abs_price = st.slider("Max |price| (elite)", min_value=110, max_value=250, value=140, step=5)
    elite_require_tb_books4 = st.toggle("Require 4+ books for TB (elite)", value=True)
    elite_max_per_team_market = st.number_input("Elite: max per team per market", min_value=1, max_value=5, value=1, step=1)
    # Composition caps for Elite
    elite_max_pct_tb = st.slider("Elite: max % Total Bases", min_value=0, max_value=100, value=50, step=5)
    elite_max_pct_per_market = st.slider("Elite: max % per single market", min_value=0, max_value=100, value=60, step=5)
    elite_max_per_game = st.number_input("Elite: max plays per game", min_value=1, max_value=3, value=1, step=1)
    st.markdown("---")
    st.subheader("Pro Bar Thresholds")
    pro_ev_card = st.number_input("Pro bar EV (Card)", min_value=0.00, max_value=0.20, value=0.03, step=0.005, format="%.3f")
    pro_edge_card = st.number_input("Pro bar Edge (Card)", min_value=0.00, max_value=0.30, value=0.05, step=0.01, format="%.2f")
    pro_ev_elite = st.number_input("Pro bar EV (Elite)", min_value=0.00, max_value=0.20, value=0.06, step=0.005, format="%.3f")
    pro_edge_elite = st.number_input("Pro bar Edge (Elite)", min_value=0.00, max_value=0.30, value=0.08, step=0.01, format="%.2f")
    enforce_pro_bar = st.toggle("Enforce pro bar on Card/Elite", value=False)

# --------------- Data load ---------------
df = load_priced_df(uploaded, default_path)

if df.empty:
    st.info("Upload a priced props CSV or point to a valid path in the sidebar.")
    st.stop()

# Normalize expected columns and types
rename_map = {
    "player_name": "player",
    "team": "home_team",  # best-effort; may not exist
}
for c_old, c_new in rename_map.items():
    if c_old in df.columns and c_new not in df.columns:
        df.rename(columns={c_old: c_new}, inplace=True)

# --------------- Fair/edge recomputation (sanity & caps) ---------------
# Ensure basic columns exist
if "selection" in df.columns:
    df["selection"] = df["selection"].astype(str).str.strip().str.title()
if "book_price" in df.columns:
    df["book_price"] = pd.to_numeric(df["book_price"], errors="coerce")
if "fair_prob" in df.columns:
    df["fair_prob"] = pd.to_numeric(df["fair_prob"], errors="coerce")
if "line" in df.columns:
    df["line"] = pd.to_numeric(df["line"], errors="coerce")

# Compute book implied probs
if {"book_price"}.issubset(df.columns):
    df["book_prob_raw"] = df["book_price"].apply(american_to_prob)

# Keys for pairing Over/Under at same line
key_cols = [c for c in ["game_id","market","player","line"] if c in df.columns]
if len(key_cols) < 4:
    # fallback if game_id missing
    key_cols = [c for c in ["home_team","away_team","market","player","line"] if c in df.columns]

if all(col in df.columns for col in key_cols + ["selection","book_prob_raw"]):
    # Build no‑vig priors per key using the median book implied prob on each side
    grp = df.groupby(key_cols)
    priors = []
    for k, g in grp:
        q_over_raw = g.loc[g["selection"]=="Over", "book_prob_raw"].median()
        q_under_raw = g.loc[g["selection"]=="Under", "book_prob_raw"].median()
        if pd.notna(q_over_raw) and pd.notna(q_under_raw):
            q_over, q_under = remove_vig_two_way(float(q_over_raw), float(q_under_raw))
        else:
            q_over, q_under = (np.nan, np.nan)
        # books at this key
        if "n_books" in g.columns:
            n_books = int(g["n_books"].max())
        elif "bookmaker" in g.columns:
            n_books = int(g["bookmaker"].nunique())
        else:
            n_books = 1
        # expand tuple key into columns for a clean merge
        key_map = {}
        if isinstance(k, tuple):
            for i, col in enumerate(key_cols):
                key_map[col] = k[i]
        else:
            key_map[key_cols[0]] = k
        row = {**key_map, "prior_over": q_over, "prior_under": q_under, "n_books_key": n_books}
        priors.append(row)
    priors_df = pd.DataFrame(priors)
    if not priors_df.empty:
        df = df.merge(priors_df, on=key_cols, how="left")

    # Choose prior for the row’s side
    df["prior_prob"] = np.where(df["selection"]=="Over", df["prior_over"], df["prior_under"]).astype(float)

    # Shrink model probability toward prior by market
    shrink_map = {
        "batter_total_bases": 0.25,
        "batter_rbis": 0.25,
        "batter_hits": 0.20,
        "pitcher_strikeouts": 0.15,
        "pitcher_outs": 0.15,
        "pitcher_earned_runs": 0.15,
        "pitcher_walks": 0.15,
        "pitcher_hits_allowed": 0.15,
    }
    w = df["market"].map(shrink_map).fillna(0.20)
    p_model = df.get("fair_prob", pd.Series(np.nan, index=df.index)).astype(float)
    p_prior = df.get("prior_prob", pd.Series(np.nan, index=df.index)).astype(float)
    # If model fair_prob is missing, fall back to prior; otherwise shrink
    p_base = np.where(p_model.notna(), (1 - w) * p_model + w * p_prior, p_prior)
    # Cap deviation from prior to avoid 30%+ edges: 0.18 default, 0.25 if n_books>=3
    n_books_key = df.get("n_books", df.get("n_books_key", pd.Series(1, index=df.index))).fillna(1).astype(int)
    cap = np.where(n_books_key >= 3, 0.25, 0.18)
    # If no prior, just clip to [0.02, 0.98]
    delta = p_base - p_prior
    delta_capped = np.sign(delta) * np.minimum(np.abs(delta), np.where(p_prior.notna(), cap, cap))
    p_final = np.where(p_prior.notna(), p_prior + delta_capped, p_base)
    p_final = np.clip(p_final, 0.02, 0.98)
    # Backfill any remaining NaNs: prior -> book implied -> 0.5
    p_final = np.where(np.isfinite(p_final), p_final, p_prior)
    p_final = np.where(np.isfinite(p_final), p_final, df.get("book_prob_raw", pd.Series(np.nan, index=df.index)))
    p_final = np.where(np.isfinite(p_final), p_final, 0.5)

    # Recompute fair price, EV, edge (vs no‑vig prior), Kelly
    df["fair_prob"] = p_final
    df["fair_price"] = df["fair_prob"].apply(prob_to_american).astype("Int64")
    b = df["book_price"].apply(american_to_multiplier)
    # Push-aware EV computation: use push_prob if present, else default to 0
    p_push = df.get("push_prob", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["ev"] = df["fair_prob"] * b - (1 - df["fair_prob"] - p_push)
    # Edge relative to no‑vig prior (more conservative than vs vigged book)
    df["edge"] = (df["fair_prob"] - df["prior_prob"]).fillna(df["fair_prob"] - df["book_prob_raw"])  # fallback if prior missing
    # Kelly capped at 2%
    dec = 1.0 + b
    df["kelly"] = np.minimum(0.02, df.apply(lambda r: kelly_fraction_decimal(r["fair_prob"], 1.0 + american_to_multiplier(r["book_price"])) if pd.notna(r["book_price"]) else 0.0, axis=1))

    # If n_books missing, add from key-level count
    if "n_books" not in df.columns:
        df["n_books"] = n_books_key

# Required core columns for display; filter to those that exist
cols_priority = [c for c in ["home_team","away_team","market","player","selection","line","book_price","fair_price","fair_prob","edge","ev","kelly","n_books","rec","notes"] if c in df.columns]
# Populate market list
unique_markets = sorted(df["market"].astype(str).str.strip().unique().tolist()) if "market" in df else []
if market_filter == [] or market_filter is None or len(market_filter) == 0:
    market_filter = unique_markets
    # update sidebar multiselect options dynamically
    with st.sidebar:
        st.session_state["__markets_loaded"] = True
# Since we can't programmatically update the options live here, also guard below.

# --------------- KPI header ---------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    n_total = len(df)
    st.markdown(f'<div class="stat-card"><div class="stat-value">{n_total:,}</div><div class="stat-label">props</div></div>', unsafe_allow_html=True)
with c2:
    n_bets = int((df["rec"].astype(str).str.upper() == "BET").sum()) if "rec" in df else 0
    st.markdown(f'<div class="stat-card"><div class="stat-value">{n_bets:,}</div><div class="stat-label">BETs</div></div>', unsafe_allow_html=True)
with c3:
    med_edge = float(df["edge"].median()) if "edge" in df else float("nan")
    st.markdown(f'<div class="stat-card"><div class="stat-value">{med_edge*100:.1f}%</div><div class="stat-label">median edge</div></div>', unsafe_allow_html=True)
with c4:
    med_ev = float(df["ev"].median()) if "ev" in df else float("nan")
    st.markdown(f'<div class="stat-card"><div class="stat-value">{med_ev:.3f}</div><div class="stat-label">median EV</div></div>', unsafe_allow_html=True)

st.markdown("### 🎯 Filtered recommendations")

# --------------- Filtering ---------------
work = df.copy()

# Basic guards
if "ev" in work:
    work = work[work["ev"] >= min_ev]
if "edge" in work:
    work = work[work["edge"] >= min_edge]
if "n_books" in work:
    work = work[(work["n_books"] >= min_books) | (work["edge"] >= 0.12)]
if "book_price" in work:
    work = work[work["book_price"].abs() <= max_abs_price]

# Market and side filters
if "market" in work:
    sel_markets = market_filter if market_filter else unique_markets
    work = work[work["market"].isin(sel_markets)]
if selection_filter and "selection" in work:
    work = work[work["selection"].isin(selection_filter)]

# Query filters
if player_query and "player" in work:
    q = player_query.strip().lower()
    work = work[work["player"].str.lower().str.contains(q, na=False)]
if team_query and "home_team" in work:
    q2 = team_query.strip().lower()
    mask_team = work["home_team"].str.lower().str.contains(q2, na=False) | work["away_team"].str.lower().str.contains(q2, na=False)
    work = work[mask_team]

# Only BETs
if only_bets and "rec" in work:
    work = work[work["rec"].astype(str).str.upper() == "BET"]

# Sorting
if sort_by in work.columns:
    work = work.sort_values(by=[sort_by, "edge", "kelly"], ascending=[not sort_desc, False, False])

# Helper: iteratively trim the weakest tail until medians meet the pro bar (or we run out)

def _enforce_pro_bar(df_in: pd.DataFrame, min_ev_req: float, min_edge_req: float, max_iters: int = 8) -> pd.DataFrame:
    dfw = df_in.copy()
    if dfw.empty or ("ev" not in dfw.columns) or ("edge" not in dfw.columns):
        return dfw
    for _ in range(max_iters):
        med_ev = float(dfw["ev"].median()) if not dfw["ev"].isna().all() else float("nan")
        med_edge = float(dfw["edge"].median()) if not dfw["edge"].isna().all() else float("nan")
        if (not math.isnan(med_ev) and med_ev >= min_ev_req) and (not math.isnan(med_edge) and med_edge >= min_edge_req):
            return dfw
        n = len(dfw)
        if n <= 5:
            break
        drop_n = max(1, n // 10)
        dfw = dfw.sort_values(by=["ev","edge"], ascending=[True, True]).iloc[drop_n:]
    return dfw

# ---------------- Top Card (capped best bets) ----------------
# Start from filtered set and keep only BETs
card_df = work.copy()
if "rec" in card_df:
    card_df = card_df[card_df["rec"].astype(str).str.upper() == "BET"]
# Card-only guards
if "kelly" in card_df:
    card_df = card_df[card_df["kelly"] >= card_min_kelly]
if "n_books" in card_df and require_tb_consensus3:
    card_df = card_df[~((card_df["market"] == "batter_total_bases") & (card_df["n_books"] < 3))]
# Rank by EV, then edge, then Kelly
for col in ["ev","edge","kelly"]:
    if col not in card_df.columns:
        card_df[col] = 0.0
card_df = card_df.sort_values(by=["ev","edge","kelly"], ascending=[False, False, False])

# Exposure guard: max N bets per team per market
def _cap_exposure(df: pd.DataFrame, max_per_team_market: int) -> pd.DataFrame:
    kept_idx = []
    counts = {}
    for idx, r in df.iterrows():
        m = str(r.get("market", ""))
        h = str(r.get("home_team", ""))
        a = str(r.get("away_team", ""))
        key_h = (m, h, "home")
        key_a = (m, a, "away")
        counts.setdefault(key_h, 0)
        counts.setdefault(key_a, 0)
        if counts[key_h] >= card_max_per_team_market or counts[key_a] >= card_max_per_team_market:
            continue
        kept_idx.append(idx)
        counts[key_h] += 1
        counts[key_a] += 1
    return df.loc[kept_idx]

card_df = _cap_exposure(card_df, card_max_per_team_market)
# Enforce hard cap on count
card_df = card_df.head(card_limit)

# Optionally enforce pro bar thresholds by trimming weakest bets
if enforce_pro_bar and not card_df.empty:
    card_df = _enforce_pro_bar(card_df, pro_ev_card, pro_edge_card)

# KPI: medians and pass/fail vs pro bar
med_card_ev = float(card_df["ev"].median()) if (not card_df.empty and "ev" in card_df) else float("nan")
med_card_edge = float(card_df["edge"].median()) if (not card_df.empty and "edge" in card_df) else float("nan")
card_ok = (not math.isnan(med_card_ev) and med_card_ev >= pro_ev_card) and (not math.isnan(med_card_edge) and med_card_edge >= pro_edge_card)

k1, k2 = st.columns(2)
with k1:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{med_card_edge*100:.1f}%</div><div class="stat-label">card median edge {"✅" if card_ok else "❌"} (pro ≥ {pro_edge_card*100:.0f}%)</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{med_card_ev:.3f}</div><div class="stat-label">card median EV {"✅" if card_ok else "❌"} (pro ≥ {pro_ev_card:.3f})</div></div>', unsafe_allow_html=True)

st.markdown("### 📌 Top Best Bets (Card)")
st.caption(f"{len(card_df)} bets selected (cap {card_limit}). Filters: EV≥{min_ev:.3f}, Edge≥{min_edge:.2f}, Kelly≥{card_min_kelly:.3f}, max |price|≤{max_abs_price}, min books≥{min_books}")

# Market composition chips for the card
if not card_df.empty and "market" in card_df:
    mcounts_card = card_df["market"].value_counts().to_dict()
    chips_card = " ".join([f'<span class="pill pill-green">{m}: {n}</span>' for m,n in mcounts_card.items()])
    st.markdown(chips_card, unsafe_allow_html=True)

# Card table and download
if not card_df.empty:
    card_cols = [c for c in ["home_team","away_team","market","player","selection","line","book_price","fair_price","fair_prob","edge","ev","kelly","n_books","rec","notes"] if c in card_df.columns]
    st.dataframe(card_df[card_cols], use_container_width=True, height=420)
    st.download_button("⬇️ Download card CSV", data=df_download_bytes(card_df), file_name="foxedge_card.csv", mime="text/csv")
else:
    st.info("No bets meet the card criteria. Loosen filters or lower Kelly/min books.")

# ---------------- Elite Bets (best-of-best, very strict) ----------------
# Start from the full df rather than the already-filtered work to avoid hiding strong but filtered plays
elite = df.copy()
if "rec" in elite:
    elite = elite[elite["rec"].astype(str).str.upper() == "BET"]
# Hard guards for elite set
if "ev" in elite:
    elite = elite[elite["ev"] >= elite_min_ev]
if "edge" in elite:
    elite = elite[elite["edge"] >= elite_min_edge]
if "kelly" in elite:
    elite = elite[elite["kelly"] >= elite_min_kelly]
if "book_price" in elite:
    elite = elite[elite["book_price"].abs() <= elite_max_abs_price]
if "n_books" in elite:
    elite = elite[elite["n_books"] >= elite_min_books]
if "n_books" in elite and elite_require_tb_books4:
    elite = elite[~((elite["market"] == "batter_total_bases") & (elite["n_books"] < 4))]

# Optional: limit to current market filter if user set one
if market_filter and "market" in elite and len(market_filter) > 0:
    elite = elite[elite["market"].isin(market_filter)]
if selection_filter and "selection" in elite and len(selection_filter) > 0:
    elite = elite[elite["selection"].isin(selection_filter)]

# Rank by EV -> edge -> Kelly
for col in ["ev","edge","kelly"]:
    if col not in elite.columns:
        elite[col] = 0.0
elite = elite.sort_values(by=["ev","edge","kelly"], ascending=[False, False, False])

# Exposure: at most N per team per market and at most 1 per player

def _cap_exposure_elite(df_in: pd.DataFrame, max_per_tm_mkt: int) -> pd.DataFrame:
    kept = []
    tm_counts = {}
    seen_players = set()
    for idx, r in df_in.iterrows():
        m = str(r.get("market",""))
        h = str(r.get("home_team",""))
        a = str(r.get("away_team",""))
        p = str(r.get("player",""))
        if p in seen_players:
            continue
        k_h = (m,h,"home"); k_a = (m,a,"away")
        tm_counts.setdefault(k_h,0); tm_counts.setdefault(k_a,0)
        if tm_counts[k_h] >= elite_max_per_team_market or tm_counts[k_a] >= elite_max_per_team_market:
            continue
        kept.append(idx)
        seen_players.add(p)
        tm_counts[k_h] += 1
        tm_counts[k_a] += 1
    return df_in.loc[kept]

elite = _cap_exposure_elite(elite, elite_max_per_team_market)

# ---- Market/game composition caps for Elite ----
# Convert percentage caps to absolute counts based on target elite_limit
_tb_cap = max(1, int(round((elite_max_pct_tb / 100.0) * elite_limit)))
_mkt_cap = max(1, int(round((elite_max_pct_per_market / 100.0) * elite_limit)))

def _cap_market_game(df_in: pd.DataFrame) -> pd.DataFrame:
    kept = []
    mcounts: dict[str,int] = {}
    gcounts: dict[str,int] = {}
    for idx, r in df_in.iterrows():
        m = str(r.get("market", ""))
        # build a stable game key regardless of order
        h = str(r.get("home_team", ""))
        a = str(r.get("away_team", ""))
        gkey = "|".join(sorted([h, a])) if h or a else ""
        # market caps
        if m == "batter_total_bases" and mcounts.get(m, 0) >= _tb_cap:
            continue
        if mcounts.get(m, 0) >= _mkt_cap:
            continue
        # per-game cap
        if gkey and gcounts.get(gkey, 0) >= elite_max_per_game:
            continue
        kept.append(idx)
        mcounts[m] = mcounts.get(m, 0) + 1
        if gkey:
            gcounts[gkey] = gcounts.get(gkey, 0) + 1
        if len(kept) >= elite_limit:
            break
    return df_in.loc[kept]

elite = _cap_market_game(elite)
# Enforce hard cap on count
elite = elite.head(elite_limit)

# Optionally enforce pro bar thresholds for elite
if enforce_pro_bar and not elite.empty:
    elite = _enforce_pro_bar(elite, pro_ev_elite, pro_edge_elite)

# KPI: medians and pass/fail vs pro bar for elite
med_elite_ev = float(elite["ev"].median()) if (not elite.empty and "ev" in elite) else float("nan")
med_elite_edge = float(elite["edge"].median()) if (not elite.empty and "edge" in elite) else float("nan")
elite_ok = (not math.isnan(med_elite_ev) and med_elite_ev >= pro_ev_elite) and (not math.isnan(med_elite_edge) and med_elite_edge >= pro_edge_elite)

k3, k4 = st.columns(2)
with k3:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{med_elite_edge*100:.1f}%</div><div class="stat-label">elite median edge {"✅" if elite_ok else "❌"} (pro ≥ {pro_edge_elite*100:.0f}%)</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{med_elite_ev:.3f}</div><div class="stat-label">elite median EV {"✅" if elite_ok else "❌"} (pro ≥ {pro_ev_elite:.3f})</div></div>', unsafe_allow_html=True)

st.markdown("### 🏆 Elite Bets (Best of Best)")
st.caption(f"{len(elite)} elite bets • EV≥{elite_min_ev:.3f} • Edge≥{elite_min_edge:.2f} • Kelly≥{elite_min_kelly:.3f} • |price|≤{elite_max_abs_price} • books≥{elite_min_books}{' • TB needs 4+ books' if elite_require_tb_books4 else ''}")

if not elite.empty:
    elite_cols = [c for c in ["home_team","away_team","market","player","selection","line","book_price","fair_price","fair_prob","edge","ev","kelly","n_books","rec","notes"] if c in elite.columns]
    # Chips for elite composition
    if "market" in elite:
        mcounts_elite = elite["market"].value_counts().to_dict()
        chips_elite = " ".join([f'<span class="pill pill-green">{m}: {n}</span>' for m,n in mcounts_elite.items()])
        st.markdown(chips_elite, unsafe_allow_html=True)
    st.dataframe(elite[elite_cols], use_container_width=True, height=360)
    st.download_button("⬇️ Download elite CSV", data=df_download_bytes(elite), file_name="foxedge_elite.csv", mime="text/csv")
else:
    st.info("No elite bets under current thresholds. Loosen slightly or widen markets.")

# --------------- Summary chips ---------------
left, right = st.columns([3,1])
with left:
    mcounts = work["market"].value_counts().to_dict() if "market" in work else {}
    chips = " ".join([f'<span class="pill pill-amber">{m}: {n}</span>' for m,n in mcounts.items()])
    st.markdown(chips, unsafe_allow_html=True)
with right:
    st.download_button("⬇️ Download filtered CSV", data=df_download_bytes(work), file_name="foxedge_filtered.csv", mime="text/csv")

st.markdown("#### Results")
st.dataframe(work[cols_priority], use_container_width=True, height=600)

# --------------- Footer ---------------
st.caption("Tip: set min EV/edge and tighten price & book guards to publish a short card.")
