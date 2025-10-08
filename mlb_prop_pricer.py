
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB Player Props Fair Pricer (Market-Fit Edition)

Supported markets (CSV 'market' column, case-insensitive):
- batter_hits
- batter_rbis
- batter_total_bases
- pitcher_strikeouts
- pitcher_earned_runs
- pitcher_outs
- pitcher_hits_allowed
- pitcher_walks

How it works (no projections required):
- Groups by (player, market).
- Converts prices to de-vig probabilities where both sides exist per line.
- Fits a distribution suited for MLB props:
  * batter_hits: Binomial with AB (n) and p fitted (n softly bounded 3..6)
  * batter_rbis: Poisson (λ)
  * batter_total_bases: Zero-Inflated Poisson (π0, λ)
  * pitcher_strikeouts: Poisson (λ)
  * pitcher_earned_runs: Poisson (λ)
  * pitcher_outs: Poisson (λ)
  * pitcher_hits_allowed: Poisson (λ)
  * pitcher_walks: Poisson (λ)
- When underdetermined (few points), falls back to conservative priors and shrinks to market priors.
- Prices each row: fair_prob, fair_price, EV, Kelly, rec.

CSV schema (identical to your prior file):
game_id,commence_time,in_play,bookmaker,last_update,home_team,away_team,market,label,description,price,point
Example:
...,9/28/2025,FALSE,DraftKings,9/26/2025,SEA Mariners,OAK Athletics,pitcher_strikeouts,Over,Logan Gilbert,-120,6.5
"""

from __future__ import annotations

import sys, math, argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.stats import binom, poisson, norm
from scipy.optimize import minimize

# ---------------- Config ----------------

KELLY_CAP = 0.015
KELLY_FRACTION = 0.4

# Shrink fitted probs toward market priors when both sides exist at a line
SHRINK_TO_PRIOR = 0.18

# Minimal EV thresholds per market (conservative)
EV_MIN = {
    "batter_hits": 0.012,
    "batter_rbis": 0.012,
    "batter_total_bases": 0.018,  # ZIP model still noisy
    "pitcher_strikeouts": 0.012,
    "pitcher_earned_runs": 0.012,
    "pitcher_outs": 0.012,
    "pitcher_hits_allowed": 0.012,
    "pitcher_walks": 0.012,
}

# Consensus and price guards
MIN_BOOKS = 2               # require at least this many books at a line
STRONG_EDGE_SINGLE = 0.12   # if only 1 book, require at least this edge to allow BET
MAX_ABS_PRICE = 160         # avoid extreme juice worse than -160 or longer than +160

REQUIRED_COLS = ["game_id","commence_time","in_play","bookmaker","last_update",
                 "home_team","away_team","market","label","description","price","point"]

MARKETS = set(EV_MIN.keys())

# -------------- Odds utils --------------

def american_to_prob(odds: int | float) -> float:
    odds = float(odds)
    if odds > 0: return 100.0/(odds+100.0)
    return abs(odds)/(abs(odds)+100.0)

def american_to_decimal(odds: int | float) -> float:
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds/100.0
    else:
        return 1.0 + 100.0/abs(odds)

def prob_to_american(p: float) -> int:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return -int(round(p/(1-p)*100)) if p > 0.5 else int(round((1-p)/p*100))

def remove_vig_two_way(p_over_raw: float, p_under_raw: float) -> Tuple[float,float]:
    s = p_over_raw + p_under_raw
    if s <= 0: return 0.5, 0.5
    return p_over_raw/s, p_under_raw/s

def kelly_fraction_decimal(p: float, price_decimal: float, frac: float = KELLY_FRACTION) -> float:
    b = price_decimal - 1.0
    if b <= 0: return 0.0
    f_full = (p * (b + 1) - 1) / b
    return max(0.0, f_full * frac)

# -------------- CSV --------------
# --- replace your existing load_props_csv with this ---
def load_props_csv(path: str, encoding_override: str = "", sep_override: str = "") -> pd.DataFrame:
    import io

    REQUIRED_COLS = ["game_id","commence_time","in_play","bookmaker","last_update",
                     "home_team","away_team","market","label","description","price","point"]

    # if user provided overrides, honor them first
    if encoding_override or sep_override:
        df = pd.read_csv(
            path,
            encoding=(encoding_override or None),
            encoding_errors="replace",
            sep=(sep_override if sep_override else None),
            engine=("python" if not sep_override else None),
        )
    else:
        # try common encodings; then delimiter sniff
        df = None
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc, encoding_errors="replace")
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            # last resort: sniff delimiter with python engine
            with open(path, "rb") as f:
                raw = f.read()
            for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
                try:
                    text = raw.decode(enc, errors="replace")
                    df = pd.read_csv(io.StringIO(text), engine="python", sep=None)
                    break
                except Exception:
                    continue
        if df is None:
            raise ValueError("Could not read CSV with common encodings or delimiter sniffing.")

    # clean BOM in headers
    df.columns = [c.replace("\ufeff","").strip() for c in df.columns]

    # validate schema
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Props CSV missing columns: {missing}. Got: {list(df.columns)}")

    # normalize fields
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["market"] = df["market"].astype(str).str.strip().str.lower()
    df["description"] = df["description"].astype(str).str.strip()
    # Clean and coerce odds and lines
    df["price"] = (
        df["price"].astype(str)
        .str.replace("−","-", regex=False)
        .str.replace("\u2212","-", regex=False)
        .str.replace("+","", regex=False)
        .str.strip()
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("Int64")

    df["point"] = (
        df["point"].astype(str)
        .str.replace(",","", regex=False)
        .str.strip()
    )
    df["point"] = pd.to_numeric(df["point"], errors="coerce")

    # Drop unusable rows
    df = df.dropna(subset=["price","point"]).copy()
    df["price"] = df["price"].astype(int)

    df = df[df["label"].isin(["over","under"])].copy()
    SUPPORTED = {
        "batter_hits","batter_rbis","batter_total_bases",
        "pitcher_strikeouts","pitcher_earned_runs","pitcher_hits_allowed","pitcher_walks","pitcher_outs"
    }
    df = df[df["market"].isin(SUPPORTED)].copy()

    # Exclude in-play
    df["in_play"] = df["in_play"].astype(str).str.lower().str.strip()
    df = df[~df["in_play"].isin(["true","1","yes"])]

    # De-duplicate identical quotes from the same book
    df = df.drop_duplicates(subset=[
        "bookmaker","market","description","label","point","price"
    ]).copy()

    return df



# -------------- Priors --------------

def prior_lambda_poisson(market: str, line: float) -> float:
    # MLB counts priors by market; mildly sub-line to avoid auto-overs
    m = market.lower()
    if m == "pitcher_strikeouts":
        return max(0.5, 0.90 * max(0.5, line))
    if m in ("pitcher_earned_runs","pitcher_walks","pitcher_hits_allowed","batter_rbis"):
        return max(0.2, 0.85 * max(0.5, line))
    if m == "batter_total_bases":
        return max(0.2, 0.80 * max(0.5, line))  # TB is spiky; keep λ modest
    return max(0.2, 0.90 * max(0.5, line))

def prior_binom_pa_range() -> Tuple[float,float]:
    # Typical PA for starters ~3.5–5.5; bound soft-fit inside
    return 3.0, 6.0

# -------------- Fitters --------------

def fit_poisson(points: List[Tuple[float,float]]) -> Tuple[float,str]:
    xs = np.array([x for x,_ in points], dtype=float)
    ys = np.array([y for _,y in points], dtype=float)
    x0 = float(np.median(xs))
    lam0 = max(0.2, x0)
    def loss(log_lam):
        lam = math.exp(log_lam)
        preds = 1.0 - poisson.cdf(np.floor(xs).astype(int), lam)
        return np.mean((preds - ys)**2)
    res = minimize(loss, x0=np.array([math.log(lam0)]), method="L-BFGS-B")
    if not res.success:
        return lam0, "poisson prior"
    return float(math.exp(res.x[0])), "poisson fit"

def fit_binomial(points: List[Tuple[float,float]]) -> Tuple[float,float,str]:
    """
    Fit n (AB) and p (hit prob per AB) to tail probs P(X >= k) where line ~ k-0.5.
    Constrain n in [3,6], p in (0,1).
    """
    xs = np.array([x for x,_ in points], dtype=float)
    ys = np.array([y for _,y in points], dtype=float)
    n_lo, n_hi = prior_binom_pa_range()
    n0 = 4.2
    p0 = 0.22
    def tail_prob(n, p, line_arr):
        # For hits, line 0.5 => k=1; 1.5 => k=2, etc.
        ks = np.floor(line_arr + 0.5).astype(int)
        probs = []
        for k in ks:
            # P[X >= k] = 1 - CDF(k-1)
            probs.append(1.0 - binom.cdf(k-1, int(round(n)), p))
        return np.array(probs, dtype=float)
    def loss(theta):
        t_n, t_logit_p = theta
        n = float(np.clip(t_n, n_lo, n_hi))
        p = 1.0/(1.0 + math.exp(-t_logit_p))
        pred = tail_prob(n, p, xs)
        return np.mean((pred - ys)**2)
    res = minimize(loss, x0=np.array([n0, math.log(p0/(1-p0))]), method="L-BFGS-B")
    if not res.success:
        return n0, p0, "binom prior"
    n_fit = float(np.clip(res.x[0], n_lo, n_hi))
    p_fit = float(1.0/(1.0+math.exp(-res.x[1])))
    return n_fit, p_fit, "binom fit"

def fit_zip(points: List[Tuple[float,float]]) -> Tuple[float,float,str]:
    """
    Zero-Inflated Poisson for total bases: P(X=0)=π0+(1-π0)Poi(0;λ), else (1-π0)Poi(k;λ).
    Optimize π0 in [0,0.9], λ>0 by matching tail probs across lines.
    """
    xs = np.array([x for x,_ in points], dtype=float)
    ys = np.array([y for _,y in points], dtype=float)
    lam0 = max(0.3, np.median(xs))
    pi0_0 = 0.25
    def tail_zip(pi0, lam, k):
        # P(X >= k+0.5) ~ P(X >= ceil(k+0.5)) = 1 - P(X <= ceil(k-0.5))
        thr = int(math.floor(k))  # for lines ending .5, floor is fine
        # CDF for ZIP: π0*I{k>=0} + (1-π0)*PoissonCDF(k;λ) but careful at k=0
        # We want tail: 1 - CDF(thr)
        cdf_poi = poisson.cdf(thr, lam)
        cdf_zip = pi0 + (1 - pi0)*cdf_poi
        return max(0.0, 1.0 - cdf_zip)
    def loss(theta):
        t_logit_pi0, t_log_lam = theta
        pi0 = 1.0/(1.0 + math.exp(-t_logit_pi0))
        pi0 = min(max(pi0, 0.0), 0.9)
        lam = math.exp(t_log_lam)
        preds = np.array([tail_zip(pi0, lam, x) for x in xs])
        return np.mean((preds - ys)**2)
    res = minimize(loss, x0=np.array([math.log(pi0_0/(1-pi0_0)), math.log(lam0)]), method="L-BFGS-B")
    if not res.success:
        return pi0_0, lam0, "zip prior"
    pi0 = float(1.0/(1.0 + math.exp(-res.x[0]))); pi0 = min(max(pi0, 0.0), 0.9)
    lam = float(math.exp(res.x[1]))
    return pi0, lam, "zip fit"

# -------------- Priors at line (market prior) --------------

def line_priors(line_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Compute de-vig priors using all offers at this exact line across books.
    We take the median implied probability for Over and for Under, then renormalize.
    """
    if line_df.empty:
        return None, None
    over_probs = line_df.loc[line_df["label"]=="over", "price"].apply(american_to_prob).to_numpy()
    under_probs = line_df.loc[line_df["label"]=="under", "price"].apply(american_to_prob).to_numpy()
    if over_probs.size == 0 or under_probs.size == 0:
        return None, None
    q_over_raw = float(np.median(over_probs))
    q_under_raw = float(np.median(under_probs))
    q_over, q_under = remove_vig_two_way(q_over_raw, q_under_raw)
    return q_over, q_under

# -------------- Market prob functions --------------

def prob_over_poisson(line: float, lam: float) -> float:
    k = int(math.floor(line))
    return 1.0 - poisson.cdf(k, max(1e-4, lam))

def prob_over_binom_hits(line: float, n_ab: float, p_hit_ab: float) -> float:
    # hits thresholds: 0.5->1, 1.5->2, etc.
    k = int(math.floor(line + 0.5))
    n = int(round(float(np.clip(n_ab, 1.0, 7.0))))
    p = float(np.clip(p_hit_ab, 1e-4, 0.8))
    return 1.0 - binom.cdf(k-1, n, p)

def prob_over_zip_tb(line: float, pi0: float, lam: float) -> float:
    thr = int(math.floor(line))
    cdf_poi = poisson.cdf(thr, max(1e-4, lam))
    cdf_zip = float(np.clip(pi0 + (1 - pi0)*cdf_poi, 0.0, 1.0))
    return max(0.0, 1.0 - cdf_zip)

# -------------- Group pricing --------------
# -------------- Group pricing --------------
# Push-aware helpers
def prob_under_poisson(line: float, lam: float) -> float:
    k = int(math.floor(line))
    if abs(line - k) < 1e-9:   # integer line
        return float(poisson.cdf(k - 1, max(1e-4, lam)))
    return float(poisson.cdf(k, max(1e-4, lam)))

def prob_push_poisson(line: float, lam: float) -> float:
    k = int(math.floor(line))
    if abs(line - k) < 1e-9:
        return float(poisson.pmf(k, max(1e-4, lam)))
    return 0.0

def prob_under_binom_hits(line: float, n_ab: float, p_hit_ab: float) -> float:
    k_int = int(math.floor(line))
    n = int(round(float(np.clip(n_ab, 1.0, 7.0))))
    p = float(np.clip(p_hit_ab, 1e-4, 0.8))
    if abs(line - k_int) < 1e-9:   # integer line: UNDER wins X <= k-1
        return float(binom.cdf(k_int - 1, n, p))
    else:                          # half line: UNDER wins X <= floor(line)
        return float(binom.cdf(k_int, n, p))

def prob_push_binom_hits(line: float, n_ab: float, p_hit_ab: float) -> float:
    k_int = int(math.floor(line))
    if abs(line - k_int) < 1e-9:
        n = int(round(float(np.clip(n_ab, 1.0, 7.0))))
        p = float(np.clip(p_hit_ab, 1e-4, 0.8))
        # P(X = k)
        return float(binom.pmf(k_int, n, p))
    return 0.0

def prob_under_zip_tb(line: float, pi0: float, lam: float) -> float:
    k_int = int(math.floor(line))
    lam = max(1e-4, lam)
    # ZIP CDF at t: π0 + (1-π0)*PoiCDF(t;λ)
    if abs(line - k_int) < 1e-9:   # integer line: UNDER wins X <= k-1
        t = k_int - 1
    else:
        t = k_int
    cdf_poi = float(poisson.cdf(max(t, -1), lam))
    cdf_zip = float(np.clip(pi0 + (1 - pi0) * cdf_poi, 0.0, 1.0))
    return cdf_zip

def prob_push_zip_tb(line: float, pi0: float, lam: float) -> float:
    k_int = int(math.floor(line))
    lam = max(1e-4, lam)
    if abs(line - k_int) < 1e-9:
        # ZIP pmf at k: π0*1{k=0} + (1-π0)*PoiPMF(k;λ)
        pmf = float(poisson.pmf(k_int, lam))
        if k_int == 0:
            return float(np.clip(pi0 + (1 - pi0) * pmf, 0.0, 1.0))
        return float((1 - pi0) * pmf)
    return 0.0

def price_group(group: pd.DataFrame, market: str, ev_min: float) -> pd.DataFrame:
    # Build per-line priors and fit points
    points = []
    line_to_priors: Dict[float, Tuple[Optional[float],Optional[float]]] = {}
    line_to_books: Dict[float, int] = {}
    for line, ldf in group.groupby("point"):
        q_over, q_under = line_priors(ldf)
        fline = float(line)
        line_to_priors[fline] = (q_over, q_under)
        line_to_books[fline] = int(ldf["bookmaker"].nunique())
        if q_over is not None:
            points.append((fline, float(q_over)))

    # Fit model
    fit_note = ""
    params = {}
    m = market.lower()

    if m == "batter_hits":
        n_ab, p_hit_ab, fit_note = fit_binomial(points) if len(points) >= 2 else (4.2, 0.22, "binom prior")
        params = {"n_ab": n_ab, "p_hit_ab": p_hit_ab}
        def p_over_fn(x):  return prob_over_binom_hits(x, n_ab, p_hit_ab)
        def p_under_fn(x): return prob_under_binom_hits(x, n_ab, p_hit_ab)
        def p_push_fn(x):  return prob_push_binom_hits(x, n_ab, p_hit_ab)

    elif m == "batter_total_bases":
        if len(points) >= 2:
            pi0, lam, fit_note = fit_zip(points)
        else:
            pi0, lam, fit_note = 0.30, prior_lambda_poisson(m, float(np.median(group['point']))), "zip prior"
        params = {"pi0": pi0, "lambda": lam}
        def p_over_fn(x):  return prob_over_zip_tb(x, pi0, lam)
        def p_under_fn(x): return prob_under_zip_tb(x, pi0, lam)
        def p_push_fn(x):  return prob_push_zip_tb(x, pi0, lam)

    elif m == "pitcher_outs":
        # Treat outs as Poisson-distributed counts; use a modestly sub-line prior
        if len(points) >= 1:
            lam, fit_note = fit_poisson(points)
        else:
            lam = max(0.5, 0.92 * float(np.median(group['point'])))
            fit_note = "poisson prior"
        params = {"lambda": lam}
        def p_over_fn(x):  return prob_over_poisson(x, lam)
        def p_under_fn(x): return prob_under_poisson(x, lam)
        def p_push_fn(x):  return prob_push_poisson(x, lam)

    else:
        # Poisson markets
        if len(points) >= 1:
            lam, fit_note = fit_poisson(points)
        else:
            lam = prior_lambda_poisson(m, float(np.median(group['point'])))
            fit_note = "poisson prior"
        params = {"lambda": lam}
        def p_over_fn(x):  return prob_over_poisson(x, lam)
        def p_under_fn(x): return prob_under_poisson(x, lam)
        def p_push_fn(x):  return prob_push_poisson(x, lam)

    # Price rows
    rows = []
    for _, r in group.iterrows():
        line = float(r["point"])
        sel = str(r["label"])
        price = int(r["price"])

        p_over_model  = float(np.clip(p_over_fn(line),  1e-4, 1-1e-4))
        p_under_model = float(np.clip(p_under_fn(line), 1e-4, 1-1e-4))
        p_push_model  = float(np.clip(p_push_fn(line),  0.0,   1.0))

        q_over, q_under = line_to_priors.get(line, (None, None))
        if q_over is not None and q_under is not None:
            p_over_star  = (1 - SHRINK_TO_PRIOR) * p_over_model  + SHRINK_TO_PRIOR * q_over
            p_under_star = (1 - SHRINK_TO_PRIOR) * p_under_model + SHRINK_TO_PRIOR * q_under
            fair_prob = p_over_star if sel=="over" else p_under_star
            prior     = q_over if sel=="over" else q_under
            edge      = fair_prob - prior
        else:
            fair_prob = p_over_model if sel=="over" else p_under_model
            edge      = fair_prob - 0.5  # weak proxy without priors

        dec = american_to_decimal(price)
        # push-aware EV: loss excludes push mass
        p_win  = fair_prob
        p_loss = max(0.0, 1.0 - fair_prob - p_push_model)
        ev    = float(p_win*(dec-1.0) - p_loss)
        kelly = min(KELLY_CAP, kelly_fraction_decimal(p_win, dec))
        fair_price = prob_to_american(fair_prob)

        # Consensus and price guards
        books = int(line_to_books.get(line, 1))
        price_ok = (abs(price) <= MAX_ABS_PRICE)
        consensus_ok = (books >= MIN_BOOKS) or (edge >= STRONG_EDGE_SINGLE)
        base_ok = (ev > 0 and edge >= ev_min)
        rec = "BET" if (base_ok and price_ok and consensus_ok) else "PASS"

        guard_notes = []
        if not price_ok: guard_notes.append("price")
        if not consensus_ok: guard_notes.append(f"books={books}")

        note_params = ", ".join([f"{k}={v:.3f}" for k,v in params.items()])
        note_prior = "; prior_line" if q_over is not None else ""
        extra = ("; guard="+"/".join(guard_notes)) if guard_notes else ""
        rows.append({
            "game_id": r["game_id"],
            "commence_time": r["commence_time"],
            "bookmaker": r["bookmaker"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "market": r["market"],
            "player": r["description"],
            "selection": r["label"].capitalize(),
            "line": line,
            "book_price": price,
            "fair_prob": round(float(fair_prob), 4),
            "fair_price": int(fair_price),
            "edge": round(float(edge), 4),
            "ev": round(float(ev), 4),
            "kelly": round(float(kelly), 4),
            "n_books": books,
            "rec": rec,
            "notes": f"{fit_note} [{note_params}]"+note_prior+extra
        })
    return pd.DataFrame(rows)

# -------------- Pipeline --------------

def props_mlb_pipeline(props_csv: str, encoding: str = "", sep: str = "") -> pd.DataFrame:
    df = load_props_csv(props_csv, encoding_override=encoding, sep_override=sep)
    if df.empty:
        return pd.DataFrame()

    out_frames = []
    for (market, player), sub in df.groupby(["market","description"]):
        ev_min = EV_MIN.get(market, 0.012)
        priced = price_group(sub, market, ev_min=ev_min)
        out_frames.append(priced)

    if not out_frames:
        return pd.DataFrame()

    out = pd.concat(out_frames, ignore_index=True)
    out["rec"] = pd.Categorical(out["rec"], categories=["BET","PASS"], ordered=True)
    out = out.sort_values(["rec","ev","edge"], ascending=[True, False, False]).reset_index(drop=True)
    return out

# -------------- CLI --------------

def main():
    ap = argparse.ArgumentParser(description="MLB Player Props fair pricer (market-fit; no projections required)")
    ap.add_argument("--props-csv", required=True, help="CSV of prop odds")
    ap.add_argument("--save", default="", help="Optional output CSV")
    ap.add_argument("--encoding", default="", help="Optional CSV encoding override")
    ap.add_argument("--sep", default="", help="Optional CSV delimiter override (e.g. '\\t')")
    args = ap.parse_args()

    out = props_mlb_pipeline(args.props_csv, encoding=args.encoding, sep=args.sep)
    if out.empty:
        print("No priced props. Check CSV contents or add alt lines.")
        sys.exit(0)

    pd.set_option("display.max_columns", None); pd.set_option("display.width", 220)
    print(out.to_string(index=False))

    if args.save:
        out.to_csv(args.save, index=False)
        print(f"\nSaved: {args.save}")

if __name__ == "__main__":
    main()
