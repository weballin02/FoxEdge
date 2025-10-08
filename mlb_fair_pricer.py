#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB fair pricer v2 — market-anchored, date-corrected, de-duplicated, with sane bet logic.

Usage:
  python mlb_fair_pricer.py --date today
  python mlb_fair_pricer.py --date 2025-09-24 --save out.csv
  python mlb_fair_pricer.py --date 2025-09-24 --csv-backup dk_odds_backup.csv
  python mlb_fair_pricer.py --date today --close-csv closes_YYYYMMDD.csv  # optional CLV merge

Key changes vs v1:
- DK splits respect --date; fallback to 'today'/'tomorrow' only if needed.
- Dedupe mapped odds by (game_id, market, selection, total_line) with a stable best-line policy.
- Totals REC rule enforces direction: Over only if λ > line + delta; Under only if λ < line − delta.
- REC always requires (EV > 0) and (edge ≥ threshold). No more “BET but EV<0.”
- Output sorted with BET first (categorical), then by EV desc.
- Optional CLV join against a closing-odds CSV (game_id,market,selection,close_price).

Dependencies:
  pip install statsapi pandas numpy scipy beautifulsoup4 python-dateutil pytz requests
"""

from __future__ import annotations

import os, sys, re, math, argparse, logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
from dateutil import tz
import pytz

# ---------- Config ----------
LOCAL_TZ = tz.gettz(os.environ.get("LOCAL_TZ", "America/Los_Angeles"))

DEFAULT_EV_THRESHOLDS = {
    "sides_ev_min": 0.018,   # 1.8% min edge vs prior
    "totals_ev_min": 0.020,  # 2.0% min edge vs prior
    "totals_min_delta": 1.0  # require ≥ 1.0 run delta vs line AND correct direction
}

KELLY_CAP = 0.015     # 1.5% bankroll max per play
KELLY_FRACTION = 0.4  # fractional Kelly

# Crude park factors (keep until you wire your full table)
PARK_FACTOR = {
    "BOS": 1.06, "NYY": 1.04, "CHC": 1.05, "COL": 1.12, "CIN": 1.07,
    "LAD": 1.01, "SD": 0.96, "SEA": 0.95, "SF": 0.94, "TB": 0.97,
    "ATL": 1.03, "PHI": 1.04, "TEX": 1.08, "MIL": 1.01, "ARI": 1.02,
    "MIA": 0.95, "BAL": 0.98, "TOR": 1.02, "DET": 0.98, "CLE": 0.98,
    "HOU": 1.03, "KC": 1.01, "MIN": 1.00, "WSH": 0.99, "STL": 1.02,
    "OAK": 0.93, "PIT": 0.98, "LAA": 0.99, "CHW": 1.03, "NYM": 0.99
}

# ---------- statsapi ----------
try:
    import statsapi  # pip install statsapi
except Exception:
    print("Install dependency: pip install statsapi", file=sys.stderr)
    raise

# ---------- Odds/Prob utils ----------
def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return float(abs(odds)) / (abs(odds) + 100.0)

def american_to_decimal(odds: int) -> float:
    if odds >= 100: return 1 + odds / 100.0
    if odds <= -100: return 1 + 100.0 / abs(odds)
    return 1.0

def prob_to_american(p: float) -> int:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return -int(round(p / (1 - p) * 100)) if p > 0.5 else int(round((1 - p) / p * 100))

def remove_vig_two_way(p1_raw: float, p2_raw: float) -> Tuple[float, float]:
    s = p1_raw + p2_raw
    return (p1_raw / s, p2_raw / s) if s > 0 else (0.5, 0.5)

def kelly_fraction_decimal(p: float, price_decimal: float, frac: float = KELLY_FRACTION) -> float:
    b = price_decimal - 1.0
    if b <= 0: return 0.0
    f_full = (p * (b + 1) - 1) / b
    return max(0.0, f_full * frac)

def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))

def logistic(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def clean_odds(x) -> int:
    try:
        return int(str(x).replace("−","-").strip())
    except Exception:
        try:
            return int(round(float(str(x).strip())))
        except Exception:
            return 0

def parse_total_side(side_text: str) -> Tuple[str, Optional[float]]:
    s = (side_text or "").strip().lower()
    m = re.search(r"(over|under)\s*([0-9]+(?:\.\d+)?)", s)
    return (m.group(1), float(m.group(2))) if m else ("", None)

# ---------- Schedule ----------
def mlb_schedule_with_probables(date_iso: str) -> pd.DataFrame:
    """
    DataFrame with: game_pk, home_name, away_name, home_abbr, away_abbr, venue, start_time_utc, home_prob, away_prob
    """
    resp = statsapi.get('schedule', {'sportId': 1, 'date': date_iso, 'hydrate': 'probablePitcher'})
    rows = []
    for day in resp.get('dates', []):
        for g in day.get('games', []):
            rows.append({
                "game_pk": g["gamePk"],
                "home_name": g['teams']['home']['team']['name'],
                "away_name": g['teams']['away']['team']['name'],
                "home_abbr": g['teams']['home']['team'].get('abbreviation', '') or g['teams']['home']['team']['name'][:3].upper(),
                "away_abbr": g['teams']['away']['team'].get('abbreviation', '') or g['teams']['away']['team']['name'][:3].upper(),
                "venue": g.get('venue', {}).get('name', ''),
                "start_time_utc": g['gameDate'],
                "home_prob": g['teams']['home'].get('probablePitcher', {}).get('fullName'),
                "away_prob": g['teams']['away'].get('probablePitcher', {}).get('fullName'),
            })
    return pd.DataFrame(rows)

# ---------- Team recent runs ----------
def team_recent_runs(team_id: int, last_n: int = 14) -> float:
    """
    Average runs per game for the team's most recent `last_n` FINAL games.
    Uses statsapi.schedule over ~60 days; fallback to league baseline if empty.
    """
    if not team_id:
        return 4.35
    today = datetime.now().date()
    start = (today - timedelta(days=60)).strftime("%m/%d/%Y")
    end   = today.strftime("%m/%d/%Y")

    games = statsapi.schedule(start_date=start, end_date=end, team=team_id)
    if not games:
        return 4.35

    finals = [g for g in games if g.get("status") == "Final"]
    finals.sort(key=lambda g: g.get("game_date", ""))
    finals = finals[-last_n:] if len(finals) > last_n else finals
    if not finals:
        return 4.35

    runs = []
    for g in finals:
        if int(g.get("home_id", 0)) == team_id:
            runs.append(int(g.get("home_score", 0)))
        elif int(g.get("away_id", 0)) == team_id:
            runs.append(int(g.get("away_score", 0)))
    return float(sum(runs)) / max(1, len(runs))

# ---------- dknetwork DraftKings betting splits ----------
def _dk_date_param_from_iso(date_iso: str) -> str:
    """
    dknetwork accepts 'today', 'tomorrow', and absolute dates in 'YYYY-MM-DD' (as of recent pages).
    We try the explicit ISO first; on failure, fall back to 'today' then 'tomorrow'.
    """
    return date_iso  # explicit first; fetch() will fallback if empty.

def fetch_dk_splits(event_group: int, date_param: str) -> pd.DataFrame:
    """
    Fetch dknetwork DraftKings betting splits across pagination.
    Attempts Playwright-first for first page, fallback to requests.
    Returns tidy rows with (matchup, game_time, market, side, odds, %handle, %bets, update_time).
    """
    from urllib.parse import urlencode, urlparse, parse_qs
    base = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
    params = {"tb_eg": event_group, "tb_edate": date_param, "tb_emt": "0"}
    first_url = f"{base}?{urlencode(params)}"
    pac = pytz.timezone("America/Los_Angeles"); now = datetime.now(pac)

    def _get_html(url: str) -> str:
        if url == first_url:
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    b = p.chromium.launch(headless=True)
                    page = b.new_page(); page.goto(url, timeout=25000)
                    page.wait_for_selector("div.tb-se", timeout=15000)
                    html = page.content(); b.close(); return html
            except Exception:
                pass
        try:
            resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=25)
            resp.raise_for_status(); return resp.text
        except Exception:
            return ""

    def _discover(html: str) -> list[str]:
        if not html: return [first_url]
        soup = BeautifulSoup(html, "html.parser")
        urls = {first_url}
        pag = soup.select_one("div.tb_pagination")
        if pag:
            for a in pag.find_all("a", href=True):
                if "tb_page=" in a["href"]:
                    urls.add(a["href"])
        def _page_num(u: str):
            try:
                return int(parse_qs(urlparse(u).query).get("tb_page", ["1"])[0])
            except Exception: return 1
        return sorted(list(urls), key=_page_num)

    def _clean(t: str) -> str:
        return re.sub(r"opens?\s+in\s+(?:a\s+)?new\s+tab", "", t or "", flags=re.I).strip()

    def _parse(html: str) -> list[dict]:
        if not html: return []
        soup = BeautifulSoup(html, "html.parser")
        games = soup.select("div.tb-se"); recs=[]
        for game in games:
            title_node = game.select_one("div.tb-se-title h5")
            if not title_node: continue
            title = _clean(title_node.get_text(strip=True))
            time_node = game.select_one("div.tb-se-title span")
            game_time = _clean(time_node.get_text(strip=True)) if time_node else ""
            for section in game.select(".tb-market-wrap > div"):
                head = section.select_one(".tb-se-head > div")
                if not head: continue
                market_name = _clean(head.get_text(strip=True))
                if market_name not in ("Moneyline","Total"): continue
                for row in section.select(".tb-sodd"):
                    side_el = row.select_one(".tb-slipline")
                    odds_el = row.select_one("a.tb-odd-s")
                    if not side_el or not odds_el: continue
                    side_raw = _clean(side_el.get_text(strip=True))
                    odds_val = clean_odds(_clean(odds_el.get_text(strip=True)))
                    pct_texts = [s.strip().replace("%","") for s in row.find_all(string=lambda t: "%" in t)]
                    handle_pct, bets_pct = (pct_texts + ["",""])[:2]
                    try: handle_pct = float(handle_pct or 0)
                    except: handle_pct = 0.0
                    try: bets_pct = float(bets_pct or 0)
                    except: bets_pct = 0.0
                    recs.append({
                        "matchup": title, "game_time": game_time, "market": market_name,
                        "side": side_raw, "odds": odds_val, "%handle": handle_pct,
                        "%bets": bets_pct, "update_time": now,
                    })
        return recs

    first_html = _get_html(first_url)
    urls = _discover(first_html)
    records=[]
    for url in urls:
        html = first_html if url == first_url else _get_html(url)
        records.extend(_parse(html))

    # Hard fallbacks if empty
    if not records and date_param not in ("today","tomorrow"):
        logging.info("dknetwork empty for %s; retrying 'today'", date_param)
        return fetch_dk_splits(event_group, "today")
    if not records and date_param == "today":
        logging.info("dknetwork empty for 'today'; retrying 'tomorrow'")
        return fetch_dk_splits(event_group, "tomorrow")

    return pd.DataFrame.from_records(records, columns=["matchup","game_time","market","side","odds","%handle","%bets","update_time"])

# ---------- Transform splits -> standardized odds ----------
def parse_matchup_to_pair(matchup: str) -> Tuple[str,str]:
    s = (matchup or "").upper().replace("@"," AT ")
    m = re.split(r"\s+(AT|VS)\s+", s)
    if len(m) >= 3:
        left, sep, right = m[0], m[1], m[2]
        away, home = (left, right) if sep == "AT" else (right, left)
        return away.strip(), home.strip()
    toks = [t for t in re.split(r"\s+", s) if t]
    return (toks[0], toks[-1]) if len(toks)>=2 else ("","")

def _resolve_side_to_abbr(side_raw: str, teams_meta: list[dict]) -> str:
    """Best-effort map of DK side text to an MLB team abbreviation.
    Uses full names, nicknames, explicit abbreviations, and common alias rules.
    """
    s = (side_raw or "").upper()
    # Strip odd punctuation but keep spaces for tokenization
    s = re.sub(r"[^A-Z @]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Build lookup maps once per call using provided teams_meta
    name_to_abbr = {}
    nickname_to_abbr = {}
    abbr_set = set()
    for t in teams_meta:
        abbr = t.get('abbreviation', t['name'][:3].upper())
        abbr_set.add(abbr)
        name_to_abbr[t['name'].upper()] = abbr
        # teamName is the nickname (e.g., "Yankees", "Dodgers")
        if t.get('teamName'):
            nickname_to_abbr[t['teamName'].upper()] = abbr
        # also map shortName if present (e.g., "NY Yankees")
        if t.get('shortName'):
            name_to_abbr[t['shortName'].upper()] = abbr

    # 1) Exact full-name match
    if s in name_to_abbr:
        return name_to_abbr[s]

    toks = [tok for tok in re.split(r"\s+", s) if tok]

    # 2) Direct abbreviation token present
    for tok in toks:
        if tok in abbr_set:
            return tok

    # 3) Nickname present (Yankees, Dodgers, Cubs, etc.)
    for tok in reversed(toks):  # prefer later tokens like the nickname
        if tok in nickname_to_abbr:
            return nickname_to_abbr[tok]

    # 4) Heuristics for common ambiguous city codes
    if "YANKEES" in s: return "NYY"
    if "METS" in s: return "NYM"
    if "DODGERS" in s: return "LAD"
    if "ANGELS" in s: return "LAA"
    if "WHITE SOX" in s or "WHITESOX" in s: return "CHW"
    if "CUBS" in s: return "CHC"
    if "RED SOX" in s or "REDSOX" in s: return "BOS"
    if "PADRES" in s: return "SD"
    if "GIANTS" in s and "SAN" in s: return "SF"
    if "CARDINALS" in s or "CARDS" in s: return "STL"
    if "REDS" in s: return "CIN"
    if "GUARDIANS" in s: return "CLE"

    # 5) Last resort: first three non-space letters
    return re.sub(r"\s+", "", s)[:3]

def map_dk_to_schedule(dk_df: pd.DataFrame, sched_df: pd.DataFrame) -> pd.DataFrame:
    """Return standardized odds keyed by MLB game_pk."""
    teams = statsapi.get('teams', {'sportId': 1}).get('teams', [])
    # Normalize market names to be robust to 'Total', 'Totals', 'Game Total', case, etc.
    if dk_df.empty:
        return pd.DataFrame()
    dk_df = dk_df.copy()
    dk_df["market_norm"] = dk_df["market"].astype(str).str.strip().str.lower()

    # Build schedule index
    sched_key = {}
    for _, g in sched_df.iterrows():
        away_full, home_full = g['away_name'].upper(), g['home_name'].upper()
        away_key = None
        home_key = None
        # Use _resolve_side_to_abbr for schedule names as well (robust for city/nickname/abbr)
        away_key = _resolve_side_to_abbr(away_full, teams)
        home_key = _resolve_side_to_abbr(home_full, teams)
        sched_key[(away_key, home_key)] = str(g['game_pk'])

    rows=[]
    for matchup, sub in dk_df.groupby("matchup"):
        a_raw, h_raw = parse_matchup_to_pair(matchup)
        away_key = _resolve_side_to_abbr(a_raw, teams)
        home_key = _resolve_side_to_abbr(h_raw, teams)
        gid = sched_key.get((away_key, home_key))
        if not gid:
            logging.debug("Unmapped matchup '%s' parsed to away='%s', home='%s' (no schedule key match).",
                          matchup, away_key, home_key)
            continue
        start_time = sched_df.loc[sched_df["game_pk"].astype(str).eq(gid), "start_time_utc"]
        start_time = start_time.iloc[0] if not start_time.empty else ""

        # Moneyline
        for _, r in sub[sub["market_norm"].eq("moneyline")].iterrows():
            side = str(r["side"]).strip().lower()
            sel = "HOME" if ("home" in side or home_key.lower() in side) else ("AWAY" if ("away" in side or away_key.lower() in side) else None)
            if sel:
                rows.append({
                    "game_id":gid,"start_time":start_time,"home":home_key,"away":away_key,
                    "market":"moneyline","selection":sel,"price":clean_odds(r["odds"]),
                    "total_line":np.nan,"update_time":r.get("update_time")
                })

        # Totals
        for _, r in sub[sub["market_norm"].str.contains("total", na=False)].iterrows():
            ou, tot = parse_total_side(str(r["side"]))
            if ou and tot is not None:
                rows.append({
                    "game_id":gid,"start_time":start_time,"home":home_key,"away":away_key,
                    "market":"total","selection":ou.upper(),"price":clean_odds(r["odds"]),
                    "total_line":float(tot),"update_time":r.get("update_time")
                })

    out = pd.DataFrame.from_records(rows)
    if out.empty:
        return out

    # Deduping policy:
    # - For moneyline favorites (negative price), keep the MOST negative (best for the favorite backer).
    # - For moneyline dogs (positive price), keep the MOST positive.
    # - For totals, for OVER keep most positive; for UNDER keep least negative (closest to zero) for an overround-consistent best price.
    def _rank_row(r):
        m, sel, price = r["market"], r["selection"], int(r["price"])
        if m == "moneyline":
            return price if price > 0 else -abs(price)
        # totals
        if sel == "OVER":
            return price  # higher positive is better
        else:  # UNDER
            return -abs(price)  # less negative is better
    out["__rank"] = out.apply(_rank_row, axis=1)

    # Keep best row per key, prefer latest update_time on ties
    out = out.sort_values(["game_id","market","selection","total_line","__rank","update_time"], ascending=[True, True, True, True, False, False])
    out = out.drop_duplicates(subset=["game_id","market","selection","total_line"], keep="first").drop(columns="__rank")
    return out.reset_index(drop=True)

# ---------- Pricing ----------
@dataclass
class PriceResult:
    game_id: str; start_time: str; away: str; home: str
    market: str; selection: str; book_price: int
    fair_prob: float; fair_price: int; edge: float; ev: float; kelly: float
    rec: str; notes: str

def pitcher_signal_delta(away_stats: dict, home_stats: dict) -> float:
    def kbb(s): return ((s.get("strikeOuts",0) or 0)+1)/((s.get("baseOnBalls",0) or 0)+1)
    def whip(s):
        ip = float(s.get("inningsPitched",0) or 0)
        return (s.get("hits",0)+s.get("baseOnBalls",0))/max(1.0, ip)
    if not away_stats or not home_stats: return 0.0
    a = np.tanh(0.25*(kbb(away_stats)-kbb(home_stats)))
    b = np.tanh(0.3*(whip(home_stats)-whip(away_stats)))
    return float(np.clip(a+b, -0.8, 0.8))

def get_pitcher_season_stats(name: Optional[str]) -> dict:
    if not name: return {}
    people = statsapi.lookup_player(name)
    if not people: return {}
    pid = people[0]['id']
    year = datetime.now().year
    ps = statsapi.get('people', {'personIds': pid, 'hydrate': f'stats(group=[pitching],type=[season],season={year})'})
    try:
        splits = ps['people'][0]['stats'][0]['splits']
        return splits[0]['stat'] if splits else {}
    except Exception:
        return {}

def price_moneyline(row: pd.Series, sibling_home: pd.Series, sibling_away: pd.Series, pitch_delta: float) -> PriceResult:
    if row["selection"]=="HOME":
        p_this_raw = american_to_prob(int(row["price"]))
        p_sib_raw  = american_to_prob(int(sibling_away["price"]))
        p0,_ = remove_vig_two_way(p_this_raw, p_sib_raw)
        bump = 0.50*pitch_delta  # delta is away-home; home gets negative sign
        p_star = logistic(logit(p0) - bump)
    else:
        p_this_raw = american_to_prob(int(row["price"]))
        p_sib_raw  = american_to_prob(int(sibling_home["price"]))
        p0,_ = remove_vig_two_way(p_this_raw, p_sib_raw)
        bump = 0.50*pitch_delta
        p_star = logistic(logit(p0) + bump)

    fair_prob = float(np.clip(p_star, 0.02, 0.98))
    fair_price = prob_to_american(fair_prob)
    dec = american_to_decimal(int(row["price"]))
    ev = fair_prob*(dec-1.0) - (1 - fair_prob)
    kelly = min(KELLY_CAP, kelly_fraction_decimal(fair_prob, dec)) if ev > 0 else 0.0
    edge = fair_prob - p0
    rec = "BET" if (ev > 0 and edge >= DEFAULT_EV_THRESHOLDS["sides_ev_min"]) else "PASS"
    return PriceResult(
        game_id=str(row["game_id"]), start_time=row["start_time"], away=row["away"], home=row["home"],
        market="ML", selection=row["selection"], book_price=int(row["price"]),
        fair_prob=fair_prob, fair_price=fair_price, edge=edge, ev=ev, kelly=kelly,
        rec=rec, notes=f"Δpitch={pitch_delta:+.2f}; prior={p0:.3f}"
    )

def park_total_multiplier(team_abbr: str) -> float:
    return PARK_FACTOR.get(team_abbr.upper(), 1.00)

def poisson_total_prob_over(total_line: float, lambda_total: float) -> float:
    """
    P(Total > K) for integer K = floor(total_line). Use strict '>' tail (standard sportsbook semantics on .5 lines).
    """
    K = math.floor(total_line)
    return 1.0 - poisson.cdf(K, lambda_total)

# --- Push-aware helpers for totals ---
def poisson_total_prob_under(total_line: float, lambda_total: float) -> float:
    """
    P(Total < line) win probability for UNDER, with push-aware handling on integer lines.
    If the line is integer K, UNDER wins on X <= K-1. If half-line, UNDER wins on X <= floor(line).
    """
    K = math.floor(total_line)
    if abs(total_line - K) < 1e-9:  # integer line
        return float(poisson.cdf(K - 1, lambda_total))
    else:  # half line
        return float(poisson.cdf(K, lambda_total))

def poisson_total_push_prob(total_line: float, lambda_total: float) -> float:
    """
    Push probability for totals: mass at the integer line if the total is an integer; else 0.
    """
    K = math.floor(total_line)
    if abs(total_line - K) < 1e-9:
        return float(poisson.pmf(K, lambda_total))
    return 0.0

def price_total(row: pd.Series, siblings: Dict[Tuple[str,float], pd.Series], away_rr: float, home_rr: float) -> Optional[PriceResult]:
    key_over = ("OVER", float(row["total_line"]))
    key_under = ("UNDER", float(row["total_line"]))
    if key_over not in siblings or key_under not in siblings: return None

    p_over_raw  = american_to_prob(int(siblings[key_over]["price"]))
    p_under_raw = american_to_prob(int(siblings[key_under]["price"]))
    p_over0, p_under0 = remove_vig_two_way(p_over_raw, p_under_raw)

    # base λ from recent runs + park factor and league baseline
    pf = park_total_multiplier(row["home"])
    lambda_total = pf*(0.5*(away_rr+home_rr) + 0.5*8.6)

    line = float(row["total_line"])
    # model win probabilities with push-aware handling
    p_over_model  = poisson_total_prob_over(line, lambda_total)
    p_under_model = poisson_total_prob_under(line, lambda_total)
    p_push_model  = poisson_total_push_prob(line, lambda_total)

    # Blend each side with its own prior (do NOT force complement; keep push mass from model)
    w = 0.45
    p_over_star  = float(np.clip(w * p_over_model  + (1 - w) * p_over0,  0.02, 0.98))
    p_under_star = float(np.clip(w * p_under_model + (1 - w) * p_under0, 0.02, 0.98))

    # Assign fair prob to the correct side
    if row["selection"]=="OVER":
        fair_prob, q_prior = p_over_star, p_over0
    else:
        fair_prob, q_prior = p_under_star, p_under0

    fair_price = prob_to_american(fair_prob)
    dec = american_to_decimal(int(row["price"]))
    # push-aware EV: p_loss excludes push probability
    if row["selection"] == "OVER":
        p_win, p_loss = p_over_star, max(0.0, 1.0 - p_over_star - p_push_model)
    else:
        p_win, p_loss = p_under_star, max(0.0, 1.0 - p_under_star - p_push_model)
    ev = p_win * (dec - 1.0) - p_loss
    edge = fair_prob - q_prior

    # Directional gate: only allow bets consistent with λ vs line disparity
    delta = lambda_total - line
    directional_ok = (row["selection"]=="OVER" and delta >= DEFAULT_EV_THRESHOLDS["totals_min_delta"]) or \
                     (row["selection"]=="UNDER" and -delta >= DEFAULT_EV_THRESHOLDS["totals_min_delta"])

    kelly = min(KELLY_CAP, kelly_fraction_decimal(p_win, dec)) if (ev > 0 and directional_ok) else 0.0
    rec = "BET" if (directional_ok and ev > 0 and edge >= DEFAULT_EV_THRESHOLDS["totals_ev_min"]) else "PASS"

    return PriceResult(
        game_id=str(row["game_id"]), start_time=row["start_time"], away=row["away"], home=row["home"],
        market="TOTAL", selection=row["selection"], book_price=int(row["price"]),
        fair_prob=fair_prob, fair_price=fair_price, edge=edge, ev=ev, kelly=kelly,
        rec=rec, notes=f"λ={lambda_total:.2f}; prior={q_prior:.3f}; pf={pf:.2f}; RR={away_rr:.2f}+{home_rr:.2f}"
    )

def choose_recommendations(priced: List[PriceResult]) -> pd.DataFrame:
    rows=[{
        "game_id":p.game_id,"start_time":p.start_time,"away":p.away,"home":p.home,"market":p.market,
        "selection":p.selection,"book_price":p.book_price,"fair_prob":round(p.fair_prob,4),
        "fair_price":p.fair_price,"edge":round(p.edge,4),"ev":round(p.ev,4),"kelly":round(p.kelly,4),
        "rec":p.rec,"notes":p.notes
    } for p in priced]
    df=pd.DataFrame(rows)
    if df.empty: return df

    # Ensure BET sorts above PASS, then by EV desc
    df["rec"] = pd.Categorical(df["rec"], categories=["BET","PASS"], ordered=True)
    df = df.sort_values(["rec","ev"], ascending=[True, False]).reset_index(drop=True)
    return df

# ---------- CSV backup ----------
def load_odds_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need={"game_id","start_time","home","away","market","selection","price","total_line"}
    missing = need - set(df.columns)
    if missing: raise ValueError(f"CSV missing columns: {missing}")
    # normalize types
    df["price"] = df["price"].apply(clean_odds)
    return df

def implied_prob_from_american(odds: int) -> float:
    return american_to_prob(int(odds))

def clv_cents(open_odds: int, close_odds: int) -> int:
    o, c = int(open_odds), int(close_odds)
    if o <= -100:   # favorite
        return abs(c) - abs(o)   # -120 -> -135 => +15
    else:           # underdog
        return o - c             # +130 -> +115 => +15

def compute_clv(recs_df: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
    """
    close_df must have: game_id, market (ML/TOTAL), selection (HOME/AWAY/OVER/UNDER), close_price
    """
    for col in ["game_id","market","selection"]:
        recs_df[col] = recs_df[col].astype(str)
        close_df[col] = close_df[col].astype(str)

    merged = recs_df.merge(
        close_df[["game_id","market","selection","close_price"]],
        on=["game_id","market","selection"], how="left"
    )
    merged["clv_cents"] = merged.apply(
        lambda r: clv_cents(int(r["book_price"]), int(r["close_price"])) if pd.notnull(r.get("close_price")) else np.nan, axis=1
    )
    merged["open_q"]  = merged["book_price"].apply(lambda x: implied_prob_from_american(int(x)))
    merged["close_q"] = merged["close_price"].apply(lambda x: implied_prob_from_american(int(x)) if pd.notnull(x) else np.nan)
    merged["clv_prob_delta"] = merged["close_q"] - merged["open_q"]
    merged["clv_status"] = np.where(merged["close_price"].notna(),
                                    np.where(merged["clv_cents"] > 0, "beat_close", "lost_close"),
                                    "unknown")
    return merged

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="MLB fair pricer (dknetwork + statsapi) — v2")
    ap.add_argument("--date", default="today", help="YYYY-MM-DD or 'today'/'tomorrow'")
    ap.add_argument("--save", default="", help="Optional CSV output path")
    ap.add_argument("--csv-backup", default="", help="CSV fallback if dknetwork returns nothing")
    ap.add_argument("--close-csv", default="", help="Optional CSV of closing prices for CLV merge")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--event-groups", default="84240,84241,84242,84243",
                    help="Comma-separated DK event_group IDs to union (e.g., '84240,84241').")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve ISO date string
    if args.date == "today":
        date_iso = datetime.now(LOCAL_TZ).date().isoformat()
    elif args.date == "tomorrow":
        date_iso = (datetime.now(LOCAL_TZ).date() + timedelta(days=1)).isoformat()
    else:
        date_iso = args.date

    # Schedule with probables
    sched = mlb_schedule_with_probables(date_iso)
    if sched.empty:
        print(f"No MLB games for {date_iso}")
        sys.exit(1)

    # Fetch dknetwork splits across one or more event groups using the resolved date
    dk_date_param = _dk_date_param_from_iso(date_iso)
    groups = [int(x) for x in str(getattr(args, "event_groups", "84240")).split(",") if     x.strip()]
    frames = []
    for g in groups:
        try:
            df_g = fetch_dk_splits(int(g), dk_date_param)
            if not df_g.empty:
                frames.append(df_g)
            else:
                logging.info("No dknetwork rows for event_group=%s on %s", g,     dk_date_param)
        except Exception as e:
            logging.warning("fetch_dk_splits failed for event_group=%s: %s", g, e)
    dk = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else     pd.DataFrame()

    if dk.empty and args.csv_backup:
        logging.info("dknetwork empty. Using CSV backup: %s", args.csv_backup)
        odds_df = load_odds_csv(args.csv_backup)
    else:
        odds_df = map_dk_to_schedule(dk, sched)

    if odds_df.empty:
        print("No odds available from dknetwork or CSV backup.")
        sys.exit(2)

    # Build per-game context
    priced: List[PriceResult] = []

    # Precompute team IDs and recent run rates for totals
    teams_resp = statsapi.get('teams', {'sportId': 1}).get('teams', [])
    abbr_to_id = {t.get('abbreviation', t['name'][:3].upper()): t['id'] for t in teams_resp}

    # Group by game
    for gid, gsub in odds_df.groupby("game_id"):
        srow = sched[sched["game_pk"].astype(str).eq(gid)]
        if srow.empty:
            continue
        srow = srow.iloc[0]

        # moneyline siblings (after dedupe map)
        ml = gsub[gsub["market"]=="moneyline"].copy()
        sib_home = ml[ml["selection"]=="HOME"].head(1)
        sib_away = ml[ml["selection"]=="AWAY"].head(1)
        sib_home = sib_home.iloc[0] if not sib_home.empty else None
        sib_away = sib_away.iloc[0] if not sib_away.empty else None

        # pitcher stats
        away_prob, home_prob = srow["away_prob"], srow["home_prob"]
        away_stats = get_pitcher_season_stats(away_prob)
        home_stats = get_pitcher_season_stats(home_prob)
        pitch_delta = pitcher_signal_delta(away_stats, home_stats)

        # moneyline pricing
        if sib_home is not None and sib_away is not None:
            for _, r in ml.iterrows():
                priced.append(price_moneyline(r, sib_home, sib_away, pitch_delta))

        # totals pricing
        tot = gsub[gsub["market"]=="total"].copy()
        if not tot.empty:
            ou_sib = {(row["selection"], float(row["total_line"])): row for _, row in tot.iterrows()}
            home_id = abbr_to_id.get(srow["home_abbr"], None)
            away_id = abbr_to_id.get(srow["away_abbr"], None)
            home_rr = team_recent_runs(home_id) if home_id else 4.35
            away_rr = team_recent_runs(away_id) if away_id else 4.35
            for _, r in tot.iterrows():
                res = price_total(r, ou_sib, away_rr, home_rr)
                if res: priced.append(res)

    rec_df = choose_recommendations(priced)
    if rec_df.empty:
        print("No actionable recommendations.")
        sys.exit(0)

    # Optional CLV merge
    if args.close_csv:
        try:
            close_df = pd.read_csv(args.close_csv)
            need = {"game_id","market","selection","close_price"}
            if need.issubset(close_df.columns):
                rec_df = compute_clv(rec_df.copy(), close_df)
            else:
                logging.warning("close-csv missing required columns: %s", need - set(close_df.columns))
        except Exception as e:
            logging.warning("Failed to merge CLV: %s", e)

    # Print
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 180)
    print(rec_df.to_string(index=False))

    # Save
    if args.save:
        rec_df.to_csv(args.save, index=False)
        print(f"\nSaved: {args.save}")

if __name__ == "__main__":
    main()
