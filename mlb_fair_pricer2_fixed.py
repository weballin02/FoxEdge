#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB fair pricer (statsapi schedule + DK Network splits) — corrected and tightened.

Key fixes vs prior:
- Honor --date for BOTH schedule and dk splits (no more hardcoded "today")
- Dedupe DK rows; more robust matchup→schedule mapping
- Totals decision rule requires EV>0 AND edge≥threshold AND λ-direction alignment
- Never output BET rows with negative EV
- Sort recommendations with BET before PASS (categorical)
- Cleaner handling of duplicated/contradictory rows
- Cosmetic: consistent columns; deterministic printing; safer parsing

Usage examples:
  python mlb_fair_pricer.py --date today
  python mlb_fair_pricer.py --date 2025-09-24 --save out.csv
  python mlb_fair_pricer.py --date today --csv-backup dk_odds_backup.csv --verbose
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
    "sides_ev_min": 0.050,   # >= 5% edge baseline
    "sides_ev_min_fav": 0.080,  # stricter for favorites
    "sides_min_improve_cents": 10,  # book vs fair must differ by ≥10 cents
    "totals_ev_min": 0.030,  # >= 3% edge baseline for totals
    "totals_min_delta": 1.2  # λ vs line must differ by ≥1.2 runs
}

KELLY_CAP = 0.015     # 1.5% bankroll cap per play
KELLY_FRACTION = 0.4  # fractional Kelly

# Simple park factors (can be replaced with your table)
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
def american_to_prob(odds: int | float) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def american_to_decimal(odds: int) -> float:
    if odds >= 100: return 1 + odds / 100.0
    if odds <= -100: return 1 + 100.0 / abs(odds)
    return 1.0

def prob_to_american(p: float) -> int:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -int(round(p / (1 - p) * 100)) if p > 0.5 else int(round((1 - p) / p * 100))

def remove_vig_two_way(p1_raw: float, p2_raw: float) -> Tuple[float, float]:
    s = p1_raw + p2_raw
    if s <= 0:
        return 0.5, 0.5
    return p1_raw / s, p2_raw / s

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
        return int(str(x).replace("−","-"))
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0

def parse_total_side(side_text: str) -> Tuple[str, Optional[float]]:
    s = (side_text or "").strip().lower()
    m = re.search(r"(over|under)\s*([0-9]+(?:\.\d+)?)", s)
    return (m.group(1), float(m.group(2))) if m else ("", None)

# ---------- Schedule ----------
def mlb_schedule_with_probables(date_iso: str) -> pd.DataFrame:
    """
    Returns DataFrame with:
      game_pk, home_name, away_name, home_abbr, away_abbr, venue, start_time_utc,
      home_prob, away_prob
    """
    resp = statsapi.get('schedule', {'sportId': 1, 'date': date_iso, 'hydrate': 'probablePitcher'})
    rows = []
    for day in resp.get('dates', []):
        for g in day.get('games', []):
            home_team = g['teams']['home']['team']
            away_team = g['teams']['away']['team']
            rows.append({
                "game_pk": g["gamePk"],
                "home_name": home_team['name'],
                "away_name": away_team['name'],
                "home_abbr": home_team.get('abbreviation', home_team['name'][:3].upper()),
                "away_abbr": away_team.get('abbreviation', away_team['name'][:3].upper()),
                "venue": g.get('venue', {}).get('name', ''),
                "start_time_utc": g['gameDate'],
                "home_prob": g['teams']['home'].get('probablePitcher', {}).get('fullName'),
                "away_prob": g['teams']['away'].get('probablePitcher', {}).get('fullName'),
            })
    return pd.DataFrame(rows)

# Recent runs via team schedule over last ~60d finals, take last N
def team_recent_runs(team_id: int, last_n: int = 14) -> float:
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

# ---------- DK Network DraftKings betting splits ----------
def fetch_dk_splits(event_group: int, date_token: str) -> pd.DataFrame:
    """
    Fetch DK Network splits for given event group and date token.
    `date_token` can be 'today', 'tomorrow', or 'YYYY-MM-DD' (DK supports explicit date).
    """
    from urllib.parse import urlencode, urlparse, parse_qs
    base = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
    params = {"tb_eg": event_group, "tb_edate": date_token, "tb_emt": "0"}
    first_url = f"{base}?{urlencode(params)}"
    pac = pytz.timezone("America/Los_Angeles"); now = datetime.now(pac)

    def _get_html(url: str) -> str:
        # Try Playwright for first page; fallback to requests
        if url == first_url:
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    b = p.chromium.launch(headless=True)
                    page = b.new_page(); page.goto(url, timeout=25000)
                    page.wait_for_selector("div.tb-se", timeout=20000)
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
                if "tb_page=" in a["href"]: urls.add(a["href"])
        def _page_idx(u: str):
            try:
                return int(parse_qs(urlparse(u).query).get("tb_page", ["1"])[0])
            except Exception: return 1
        return sorted(list(urls), key=_page_idx)

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
                if market_name not in ("Moneyline","Total","Totals"): continue
                # normalize plural
                market_name = "Total" if market_name.lower().startswith("total") else market_name
                for row in section.select(".tb-sodd"):
                    side_el = row.select_one(".tb-slipline")
                    odds_el = row.select_one("a.tb-odd-s")
                    if not side_el or not odds_el: continue
                    side_raw = _clean(side_el.get_text(strip=True))
                    odds_val = clean_odds(_clean(odds_el.get_text(strip=True)))
                    # percentages are optional; not used in pricing
                    recs.append({
                        "matchup": title, "game_time": game_time, "market": market_name,
                        "side": side_raw, "odds": odds_val, "update_time": now,
                    })
        return recs

    first_html = _get_html(first_url)
    all_urls = _discover(first_html)
    records=[]
    for url in all_urls:
        html = first_html if url == first_url else _get_html(url)
        records.extend(_parse(html))
    try:
        logging.debug("DK unique matchups scraped: %s", sorted({r["matchup"] for r in records}))
    except Exception:
        pass
    return pd.DataFrame.from_records(records, columns=["matchup","game_time","market","side","odds","update_time"])

def parse_matchup_to_pair(matchup: str) -> Tuple[str,str]:
    s = (matchup or "").upper().replace("@"," AT ")
    # Normalize punctuation around separators
    s = re.sub(r"\s+VS\.?\s+", " VS ", s)
    s = re.sub(r"\s+AT\s+", " AT ", s)
    m = re.split(r"\s+(AT|VS)\s+", s)
    if len(m) >= 3:
        left, sep, right = m[0], m[1], m[2]
        away, home = (left, right) if sep == "AT" else (right, left)
        return away.strip(), home.strip()
    toks = [t for t in re.split(r"\s+", s) if t]
    return (toks[0], toks[-1]) if len(toks)>=2 else ("","")

def map_dk_to_schedule(dk_df: pd.DataFrame, sched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize odds rows keyed by MLB game_pk (string).
    """
    teams = statsapi.get('teams', {'sportId': 1}).get('teams', [])
    name_to_abbr = {t['name'].upper(): t.get('abbreviation', t['name'][:3].upper()) for t in teams}

    abbr_set = set(name_to_abbr.values())
    # Map common nicknames (e.g., "Yankees", "Blue Jays") to abbreviations
    nickname_to_abbr = {}
    for t in teams:
        try:
            nick = t.get("teamName") or t.get("clubName") or ""
            if nick:
                nickname_to_abbr[nick.upper()] = t.get('abbreviation', t['name'][:3].upper())
        except Exception:
            continue

    def to_abbr(raw: str) -> str:
        """Normalize a raw DK team string like 'NY Yankees' -> 'NYY'."""
        s = re.sub(r"[^A-Z ]", "", (raw or "").upper()).strip()
        # 1) direct full-name match
        if s in name_to_abbr:
            return name_to_abbr[s]
        # 2) contains an existing abbr token
        toks = [t for t in s.split() if t]
        for t in toks:
            if t in abbr_set:
                return t
        # 3) try nickname: last two tokens, then last token
        if len(toks) >= 2:
            last2 = " ".join(toks[-2:])
            if last2 in nickname_to_abbr:
                return nickname_to_abbr[last2]
        if toks:
            last1 = toks[-1]
            if last1 in nickname_to_abbr:
                return nickname_to_abbr[last1]
        # 4) common city disambiguations
        if s.startswith("NY "):
            if "YANKEE" in s: return "NYY"
            if "MET" in s:    return "NYM"
        if s.startswith("LA "):
            if "DODGER" in s: return "LAD"
            if "ANGEL" in s:  return "LAA"
        if s.startswith("CHI "):
            if "WHITE" in s or "SOX" in s: return "CHW"
            if "CUB" in s:                 return "CHC"
        if "ST LOUIS" in s or "ST. LOUIS" in s:
            return "STL"
        if "D BACK" in s or "DIAMONDBACK" in s or "D-BACK" in s:
            return "ARI"
        if "BLUE JAY" in s:
            return "TOR"
        # 5) absolute last resort: first three characters
        return s[:3].strip()
    # schedule key for fast lookup
    sched_key = {}
    for _, g in sched_df.iterrows():
        away_key = name_to_abbr.get(g['away_name'].upper(), g['away_name'][:3].upper())
        home_key = name_to_abbr.get(g['home_name'].upper(), g['home_name'][:3].upper())
        sched_key[(away_key, home_key)] = str(g['game_pk'])

    logging.debug("Schedule keys available: %s", sorted(list(sched_key.keys())))

    rows=[]
    unmatched = set()
    for matchup, sub in dk_df.groupby("matchup"):
        a_raw, h_raw = parse_matchup_to_pair(matchup)
        away_key = to_abbr(a_raw)
        home_key = to_abbr(h_raw)
        gid = sched_key.get((away_key, home_key))
        if not gid:
            unmatched.add((matchup, away_key, home_key))
            continue
        start_time = sched_df.loc[sched_df["game_pk"].astype(str).eq(gid), "start_time_utc"]
        start_time = start_time.iloc[0] if not start_time.empty else ""

        # moneyline
        for _, r in sub[sub["market"]=="Moneyline"].iterrows():
            side = str(r["side"]).strip().lower()
            sel = "HOME" if ("home" in side or home_key.lower() in side) else ("AWAY" if ("away" in side or away_key.lower() in side) else None)
            if sel:
                rows.append({"game_id":gid,"start_time":start_time,"home":home_key,"away":away_key,
                             "market":"moneyline","selection":sel,"price":clean_odds(r["odds"]),"total_line":np.nan})

        # totals
        for _, r in sub[sub["market"]=="Total"].iterrows():
            ou, tot = parse_total_side(str(r["side"]))
            if ou and tot is not None:
                rows.append({"game_id":gid,"start_time":start_time,"home":home_key,"away":away_key,
                             "market":"total","selection":ou.upper(),"price":clean_odds(r["odds"]),"total_line":float(tot)})

    if unmatched:
        logging.warning("DK matchups not mapped to schedule: %s", sorted(list(unmatched)))

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        return df

    # dedupe: keep best (most recent) observation per key
    # key includes total_line for totals to avoid cross-line contamination
    df["key"] = df.apply(lambda r: (r["game_id"], r["market"], r["selection"], float(r["total_line"]) if not pd.isna(r["total_line"]) else None, int(r["price"])), axis=1)
    df = df.drop_duplicates(subset="key").drop(columns=["key"]).reset_index(drop=True)
    return df

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
        ip_txt = str(s.get("inningsPitched",0) or "0")
        try:
            # inningsPitched often "123.2": treat .1 as 1/3 inning; .2 as 2/3
            whole, dot, frac = ip_txt.partition(".")
            ip = float(whole) + (1/3 if frac=="1" else 2/3 if frac=="2" else 0.0)
        except Exception:
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
    # de-vig prior from the opposing quoted prices
    if row["selection"]=="HOME":
        p_this_raw = american_to_prob(int(row["price"]))
        p_sib_raw  = american_to_prob(int(sibling_away["price"]))
        p0,_ = remove_vig_two_way(p_this_raw, p_sib_raw)
        bump = 0.50*pitch_delta  # delta is away-home; home gets negative sign in logit
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
    kelly = min(KELLY_CAP, kelly_fraction_decimal(fair_prob, dec))
    edge = fair_prob - p0
    base_thr = DEFAULT_EV_THRESHOLDS["sides_ev_min"]
    fav_thr = DEFAULT_EV_THRESHOLDS.get("sides_ev_min_fav", base_thr)
    is_fav = int(row["price"]) < 0
    thr = fav_thr if is_fav else base_thr
    improve_cents = abs(int(prob_to_american(fair_prob)) - int(row["price"]))
    rec = "BET" if (ev > 0 and edge >= thr and improve_cents >= DEFAULT_EV_THRESHOLDS["sides_min_improve_cents"]) else "PASS"
    return PriceResult(
        game_id=str(row["game_id"]), start_time=row["start_time"], away=row["away"], home=row["home"],
        market="ML", selection=row["selection"], book_price=int(row["price"]),
        fair_prob=fair_prob, fair_price=fair_price, edge=edge, ev=ev, kelly=kelly,
        rec=rec, notes=f"Δpitch={pitch_delta:+.2f}; prior={p0:.3f}"
    )

def park_total_multiplier(team_abbr: str) -> float:
    return PARK_FACTOR.get(team_abbr.upper(), 1.00)

def poisson_total_prob_over(total_line: float, lambda_total: float) -> float:
    K = math.floor(total_line)
    return 1.0 - poisson.cdf(K, lambda_total)

# --- Push-aware helpers for totals ---
def poisson_total_prob_under(total_line: float, lambda_total: float) -> float:
    """
    UNDER win probability, push-aware.
    If the line is integer K, UNDER wins on X <= K-1. If half-line, UNDER wins on X <= floor(line).
    """
    K = math.floor(total_line)
    if abs(total_line - K) < 1e-9:  # integer line
        return float(poisson.cdf(K - 1, lambda_total))
    else:  # half line
        return float(poisson.cdf(K, lambda_total))

def poisson_total_push_prob(total_line: float, lambda_total: float) -> float:
    """
    Push probability mass at the integer line; zero on half-lines.
    """
    K = math.floor(total_line)
    if abs(total_line - K) < 1e-9:
        return float(poisson.pmf(K, lambda_total))
    return 0.0

def price_total(row: pd.Series,
                siblings: Dict[Tuple[str,float], pd.Series],
                away_rr: float, home_rr: float) -> Optional[PriceResult]:
    key_over = ("OVER", float(row["total_line"]))
    key_under = ("UNDER", float(row["total_line"]))
    if key_over not in siblings or key_under not in siblings: return None

    p_over_raw  = american_to_prob(int(siblings[key_over]["price"]))
    p_under_raw = american_to_prob(int(siblings[key_under]["price"]))
    p_over0, p_under0 = remove_vig_two_way(p_over_raw, p_under_raw)

    # model λ: recent runs + league baseline, scaled by park
    pf = park_total_multiplier(row["home"])
    lambda_total = pf*(0.5*(away_rr+home_rr) + 0.5*8.6)

    line = float(row["total_line"])
    # push-aware model win probabilities
    p_over_model  = poisson_total_prob_over(line, lambda_total)
    p_under_model = poisson_total_prob_under(line, lambda_total)
    p_push_model  = poisson_total_push_prob(line, lambda_total)

    # blend each side with its own prior (do not force complement; preserve push mass)
    w = 0.45
    p_over_star  = float(np.clip(w*p_over_model  + (1-w)*p_over0,  0.02, 0.98))
    p_under_star = float(np.clip(w*p_under_model + (1-w)*p_under0, 0.02, 0.98))

    if row["selection"]=="OVER":
        fair_prob, q_prior = p_over_star, p_over0
        directional_ok = (lambda_total >= line + DEFAULT_EV_THRESHOLDS["totals_min_delta"])
        p_win, p_loss = p_over_star, max(0.0, 1.0 - p_over_star - p_push_model)
    else:
        fair_prob, q_prior = p_under_star, p_under0
        directional_ok = (lambda_total <= line - DEFAULT_EV_THRESHOLDS["totals_min_delta"])
        p_win, p_loss = p_under_star, max(0.0, 1.0 - p_under_star - p_push_model)

    fair_price = prob_to_american(fair_prob)
    dec = american_to_decimal(int(row["price"]))
    ev = p_win*(dec-1.0) - p_loss
    kelly = min(KELLY_CAP, kelly_fraction_decimal(p_win, dec))
    edge = fair_prob - q_prior

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

    # Force BET above PASS using categorical ordering, then sort by EV desc
    df["rec"] = pd.Categorical(df["rec"], categories=["BET","PASS"], ordered=True)
    df = df.sort_values(["rec","ev"], ascending=[True, False]).reset_index(drop=True)
    return df

# ---------- CSV backup ----------
def load_odds_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need={"game_id","start_time","home","away","market","selection","price","total_line"}
    missing = need - set(df.columns)
    if missing: raise ValueError(f"CSV missing columns: {missing}")
    return df

# ---------- CLV helpers (optional downstream use) ----------
def implied_prob_from_american(odds: int) -> float:
    return american_to_prob(odds)

def clv_cents(open_odds: int, close_odds: int) -> int:
    o, c = int(open_odds), int(close_odds)
    if o <= -100:   # favorite: more negative is better
        return abs(c) - abs(o)
    else:           # dog: closer to 0 is better
        return o - c

def compute_clv(recs_df: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["game_id","market","selection"]
    for col in key_cols:
        recs_df[col] = recs_df[col].astype(str)
        close_df[col] = close_df[col].astype(str)
    merged = recs_df.merge(
        close_df[["game_id","market","selection","close_price"]],
        on=key_cols, how="left"
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
def resolve_date_token(arg_date: str) -> Tuple[str,str]:
    """
    Returns (iso_date, dk_date_token). DK supports 'today','tomorrow', or explicit ISO.
    We'll pass explicit ISO unless arg_date=='today' or the day is exactly tomorrow.
    """
    now_local = datetime.now(LOCAL_TZ).date()
    if arg_date.lower() == "today":
        return now_local.isoformat(), "today"
    try:
        target = datetime.fromisoformat(arg_date).date()
    except ValueError:
        # fallback to today
        return now_local.isoformat(), "today"
    dk_token = "today" if target == now_local else ("tomorrow" if target == now_local + timedelta(days=1) else target.isoformat())
    return target.isoformat(), dk_token

def main():
    ap = argparse.ArgumentParser(description="MLB fair pricer (DK Network + statsapi) — corrected")
    ap.add_argument("--date", default="today", help="YYYY-MM-DD or 'today'")
    ap.add_argument("--save", default="", help="Optional CSV output path")
    ap.add_argument("--csv-backup", default="", help="CSV fallback if dknetwork returns nothing")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--event-group", type=int, default=84240, help="DK event group id (MLB default 84240)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    date_iso, dk_date_token = resolve_date_token(args.date)
    logging.info(f"Resolved date: schedule={date_iso} dk_date_token={dk_date_token}")

    # Schedule with probables
    sched = mlb_schedule_with_probables(date_iso)
    logging.info("Schedule games fetched: %d", len(sched))
    if sched.empty:
        print(f"No MLB games for {date_iso}")
        sys.exit(1)

    # Fetch dk splits for the same date token; map to schedule
    dk = fetch_dk_splits(args.event_group, dk_date_token)
    logging.info("DK splits rows scraped: %d", len(dk))
    if dk.empty and args.csv_backup:
        logging.info("DK splits empty. Using CSV backup: %s", args.csv_backup)
        odds_df = load_odds_csv(args.csv_backup)
    else:
        odds_df = map_dk_to_schedule(dk, sched)

    if not odds_df.empty:
        logging.info("Mapped odds rows: %d across %d games", len(odds_df), odds_df["game_id"].nunique())
        logging.debug("Game IDs mapped: %s", sorted(odds_df["game_id"].astype(str).unique()))

    if odds_df.empty:
        print("No odds available from DK Network or CSV backup.")
        sys.exit(2)

    # Build per-game context
    priced: List[PriceResult] = []

    # Precompute team ids
    teams_resp = statsapi.get('teams', {'sportId': 1}).get('teams', [])
    abbr_to_id = {t.get('abbreviation', t['name'][:3].upper()): t['id'] for t in teams_resp}

    # Price each game
    logging.info("Pricing %d games", odds_df["game_id"].nunique())
    for gid, gsub in odds_df.groupby("game_id"):
        srow = sched[sched["game_pk"].astype(str).eq(gid)]
        if srow.empty:
            continue
        srow = srow.iloc[0]
        logging.debug("Pricing game_id=%s teams=%s@%s rows=%d", gid, srow["away_abbr"], srow["home_abbr"], len(gsub))
        away_prob, home_prob = srow["away_prob"], srow["home_prob"]
        away_stats = get_pitcher_season_stats(away_prob)
        home_stats = get_pitcher_season_stats(home_prob)
        pitch_delta = pitcher_signal_delta(away_stats, home_stats)

        # Moneylines
        ml = gsub[gsub["market"]=="moneyline"].copy()
        if not ml.empty:
            sib_home = ml[ml["selection"]=="HOME"].head(1)
            sib_away = ml[ml["selection"]=="AWAY"].head(1)
            if not sib_home.empty and not sib_away.empty:
                sib_home = sib_home.iloc[0]
                sib_away = sib_away.iloc[0]
                for _, r in ml.iterrows():
                    res = price_moneyline(r, sib_home, sib_away, pitch_delta)
                    priced.append(res)

        # Totals
        tot = gsub[gsub["market"]=="total"].copy()
        if not tot.empty:
            # siblings keyed by (selection,line)
            ou_sib = {(row["selection"], float(row["total_line"])): row for _, row in tot.iterrows()}
            home_id = abbr_to_id.get(srow["home_abbr"])
            away_id = abbr_to_id.get(srow["away_abbr"])
            home_rr = team_recent_runs(home_id) if home_id else 4.35
            away_rr = team_recent_runs(away_id) if away_id else 4.35
            for _, r in tot.iterrows():
                res = price_total(r, ou_sib, away_rr, home_rr)
                if res:
                    priced.append(res)

    rec_df = choose_recommendations(priced)
    if rec_df.empty:
        print("No actionable recommendations.")
    else:
        pd.set_option("display.max_columns", None); pd.set_option("display.width", 180)
        print(rec_df.to_string(index=False))

    if args.save:
        # Save as printed ordering
        rec_df.to_csv(args.save, index=False)
        print(f"\nSaved: {args.save}")

if __name__ == "__main__":
    main()
