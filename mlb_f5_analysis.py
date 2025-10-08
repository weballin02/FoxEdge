# mlb_f5_analysis_fixed.py
import math
import re
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
import unicodedata

# -------------------- Config --------------------
F5_CSV_PATH = "/Users/matthewfox/Documents/FoxEdgeAI_LOCAL/f5_odds.csv"  # keep
MLB_EVENT_GROUP = 84240  # DK MLB

# Calibration knobs
R_TOTAL_F5 = 0.545       # share of full-game total scored in first 5
SD_F5_DIFF = 1.35        # sd of first-5 run differential in runs
HOME_EDGE_F5 = 0.10      # tiny home edge in runs for first 5

# Park factors (multiplicative on run environment)
PARK_FACTORS = {
    "coors field": 1.15, "great american ball park": 1.10, "oriole park at camden yards": 1.08,
    "globe life field": 1.07, "yankee stadium": 1.05, "citizens bank park": 1.04, "truist park": 1.03,
    "fenway park": 1.02, "american family field": 1.02, "chase field": 1.01, "target field": 1.00,
    "guaranteed rate field": 1.00, "angel stadium": 0.99, "busch stadium": 0.98, "progressive field": 0.97,
    "nationals park": 0.97, "comerica park": 0.96, "minute maid park": 0.96, "rogers centre": 0.95,
    "kauffman stadium": 0.95, "petco park": 0.93, "tropicana field": 0.93, "t-mobile park": 0.92,
    "loandeport park": 0.92, "dodger stadium": 0.92, "pnc park": 0.91, "oakland coliseum": 0.90, "oracle park": 0.88,
}

TEAM_PARKS = {
    "colorado rockies": "coors field", "cincinnati reds": "great american ball park",
    "baltimore orioles": "oriole park at camden yards", "texas rangers": "globe life field",
    "new york yankees": "yankee stadium", "philadelphia phillies": "citizens bank park",
    "atlanta braves": "truist park", "boston red sox": "fenway park",
    "milwaukee brewers": "american family field", "arizona diamondbacks": "chase field",
    "minnesota twins": "target field", "chicago white sox": "guaranteed rate field",
    "los angeles angels": "angel stadium", "st louis cardinals": "busch stadium",
    "cleveland guardians": "progressive field", "washington nationals": "nationals park",
    "detroit tigers": "comerica park", "houston astros": "minute maid park",
    "toronto blue jays": "rogers centre", "kansas city royals": "kauffman stadium",
    "san diego padres": "petco park", "tampa bay rays": "tropicana field",
    "seattle mariners": "t-mobile park", "miami marlins": "loandeport park",
    "los angeles dodgers": "dodger stadium", "pittsburgh pirates": "pnc park",
    "oakland athletics": "oakland coliseum", "san francisco giants": "oracle park",
    "chicago cubs": "wrigley field", "new york mets": "citi field",
}

# -------------------- Utilities --------------------
def clean_odds(odds_str) -> Optional[int]:
    try:
        return int(str(odds_str).replace("−", "-").replace("–", "-").strip())
    except Exception:
        try:
            return int(float(odds_str))
        except Exception:
            return None

def implied_prob_from_odds(odds: Optional[int]) -> float:
    if odds is None or not isinstance(odds, (int, float)):
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        a = abs(odds)
        return a / (a + 100.0)
    return 0.5

def _super_normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r"\s+", " ", text.lower().strip()).replace("\u2212", "-")

def _normalize_team(name: str) -> str:
    if not name:
        return ""
    name = _super_normalize(name)
    m = {
        "nyy": "new york yankees", "ny yankees": "new york yankees",
        "nym": "new york mets", "ny mets": "new york mets",
        "laa": "los angeles angels", "la angels": "los angeles angels",
        "lad": "los angeles dodgers", "la dodgers": "los angeles dodgers",
        "sf giants": "san francisco giants", "sd padres": "san diego padres",
        "chi cubs": "chicago cubs", "chi white sox": "chicago white sox", "chw": "chicago white sox",
        "stl cardinals": "st louis cardinals", "st. louis cardinals": "st louis cardinals",
        "tb rays": "tampa bay rays", "bos red sox": "boston red sox", "bal orioles": "baltimore orioles",
        "kc royals": "kansas city royals", "cle guardians": "cleveland guardians",
        "ari diamondbacks": "arizona diamondbacks", "col rockies": "colorado rockies",
        "mia marlins": "miami marlins", "phi phillies": "philadelphia phillies",
        "was nationals": "washington nationals", "tex rangers": "texas rangers",
        "min twins": "minnesota twins", "mil brewers": "milwaukee brewers",
        "pit pirates": "pittsburgh pirates", "hou astros": "houston astros",
        "sea mariners": "seattle mariners", "tor blue jays": "toronto blue jays",
        "oak athletics": "oakland athletics", "sf": "san francisco giants",
        "sd": "san diego padres",
    }
    return m.get(name, name)

def normalize_matchup(any_str: str) -> str:
    s = _super_normalize(any_str)
    # convert “Away @ Home” -> “Home vs Away”
    if "@" in s:
        away, home = [t.strip() for t in re.split(r"\s*@\s*", s, maxsplit=1)]
        return f"{_normalize_team(home)} vs {_normalize_team(away)}"
    if " vs " in s:
        home, away = [t.strip() for t in re.split(r"\s+vs\s+", s, maxsplit=1)]
        return f"{_normalize_team(home)} vs {_normalize_team(away)}"
    # last resort: best-effort split
    parts = re.split(r"\s*(?:@|vs|at)\s*", s)
    if len(parts) == 2:
        if "@" in s or " at " in s:
            away, home = parts[0], parts[1]
            return f"{_normalize_team(home)} vs {_normalize_team(away)}"
        home, away = parts[0], parts[1]
        return f"{_normalize_team(home)} vs {_normalize_team(away)}"
    return s

def get_park_factor(home_team: str) -> float:
    park = TEAM_PARKS.get(_normalize_team(home_team), "")
    return PARK_FACTORS.get(park, 1.0)

# -------------------- DKNetwork Splits --------------------
def fetch_dk_splits(event_group: int = MLB_EVENT_GROUP, date_range: str = "today") -> pd.DataFrame:
    from urllib.parse import urlencode, urlparse, parse_qs

    base = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
    params = {"tb_eg": event_group, "tb_edate": date_range, "tb_emt": "0"}
    first_url = f"{base}?{urlencode(params)}"

    def clean(t: str) -> str:
        return re.sub(r"opens?\s+in\s+(?:a\s+)?new\s+tab", "", t or "", flags=re.I).strip()

    def _get_html(url: str) -> str:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return ""

    def _discover_page_urls(html: str) -> list[str]:
        if not html:
            return [first_url]
        soup = BeautifulSoup(html, "html.parser")
        urls = {first_url}
        pag = soup.select_one("div.tb_pagination")
        if pag:
            for a in pag.find_all("a", href=True):
                href = a["href"]
                if "tb_page=" in href:
                    urls.add(base + href if not href.startswith("http") else href)
        def pnum(u: str) -> int:
            try:
                return int(parse_qs(urlparse(u).query).get("tb_page", ["1"])[0])
            except Exception:
                return 1
        return sorted(list(urls), key=pnum)

    def _parse_page(html: str) -> list[dict]:
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        games = soup.select("div.tb-se")
        out = []
        now = datetime.now(timezone.utc)
        for g in games:
            t_el = g.select_one("div.tb-se-title h5")
            if not t_el:
                continue
            title = clean(t_el.get_text(strip=True))
            time_el = g.select_one("div.tb-se-title span")
            game_time = clean(time_el.get_text(strip=True)) if time_el else ""
            for section in g.select(".tb-market-wrap > div"):
                head = section.select_one(".tb-se-head > div")
                if not head:
                    continue
                market_name = clean(head.get_text(strip=True))
                if market_name == "Spread":
                    market_name = "Run Line"  # normalize MLB wording
                if market_name not in ("Moneyline", "Total", "Run Line"):
                    continue
                for row in section.select(".tb-sodd"):
                    side_el = row.select_one(".tb-slipline")
                    odds_el = row.select_one("a.tb-odd-s")
                    if not side_el or not odds_el:
                        continue
                    side = clean(side_el.get_text(strip=True))
                    oddstxt = clean(odds_el.get_text(strip=True))
                    odds_val = clean_odds(oddstxt)
                    pct_texts = [s.strip().replace("%", "") for s in row.find_all(string=lambda t: "%" in t)]
                    handle_pct, bets_pct = (pct_texts + ["", ""])[:2]
                    runline_val = None
                    if market_name == "Run Line":
                        parts = side.split()
                        if parts:
                            last = parts[-1].replace("−", "-").replace("–", "-")
                            try:
                                if last.startswith(("+", "-")):
                                    runline_val = float(last)
                            except ValueError:
                                runline_val = None
                    out.append({
                        "matchup": normalize_matchup(title),
                        "game_time": game_time,
                        "market": market_name,
                        "side": side,
                        "odds": odds_val,
                        "runline": runline_val,
                        "%handle": float(handle_pct or 0),
                        "%bets": float(bets_pct or 0),
                        "update_time": now,
                    })
        return out

    first_html = _get_html(first_url)
    urls = _discover_page_urls(first_html)
    records = []
    for u in urls:
        html = first_html if u == first_url else _get_html(u)
        records.extend(_parse_page(html))
    df = pd.DataFrame(records)
    return df

# -------------------- Prob models --------------------
def poisson_cdf(k: int, mu: float) -> float:
    # P(X <= k) for Poisson(mu)
    s = 0.0
    term = math.exp(-mu)
    s += term  # k=0
    for i in range(1, k + 1):
        term *= mu / i
        s += term
    return s

def prob_over_poisson(line: float, mu: float) -> float:
    # Half-points common: Over 4.5 -> 1 - CDF(4; mu)
    if line % 1 == 0.5:
        k = int(math.floor(line))
        return 1.0 - poisson_cdf(k, mu)
    # Integer totals appear occasionally; treat as strict > line
    k = int(line)
    return 1.0 - poisson_cdf(k, mu)

def expected_f5_total(full_total: Optional[float], park_factor: float) -> Optional[float]:
    if full_total is None:
        return None
    return full_total * R_TOTAL_F5 * park_factor

def expected_f5_margin_from_pitchers(home_era: Optional[float], away_era: Optional[float]) -> Optional[float]:
    # Convert ERA to expected runs over 5 innings and take the delta (away - home), add a small home edge
    try:
        if home_era is None or away_era is None:
            return None
        mu_home = (home_era / 9.0) * 5.0
        mu_away = (away_era / 9.0) * 5.0
        # Margin is home runs minus away runs; lower ERA reduces runs allowed
        # Home advantage: tiny positive bump
        return (mu_away - mu_home) + HOME_EDGE_F5
    except Exception:
        return None

def prob_cover_minus_half(mu_margin: float) -> float:
    # P(diff > 0.5) under Normal(mu_margin, SD_F5_DIFF)
    z = (0.5 - mu_margin) / SD_F5_DIFF
    # Normal tail
    return 0.5 * math.erfc(z / math.sqrt(2.0))

# -------------------- Analysis --------------------
def analyze_f5(full_df: pd.DataFrame, f5_df: pd.DataFrame) -> pd.DataFrame:
    full = full_df.copy()
    f5 = f5_df.copy()

    # Normalize matchups
    full["matchup"] = full["matchup"].map(normalize_matchup)
    # Build F5 matchup as home vs away
    f5["matchup"] = (f5["home_team"].map(_normalize_team) + " vs " + f5["away_team"].map(_normalize_team)).map(_super_normalize)

    out_rows = []

    for m in sorted(full["matchup"].unique()):
        # Markets
        fg_runline = full[(full["matchup"] == m) & (full["market"] == "Run Line")]
        fg_total   = full[(full["matchup"] == m) & (full["market"] == "Total")]

        f5_spreads = f5[(f5["matchup"] == m) & (f5["market"] == "spreads_1st_5_innings")]
        f5_totals  = f5[(f5["matchup"] == m) & (f5["market"] == "totals_1st_5_innings")]

        # Parse full total number (take first “Over X.Y” row)
        full_total_num = None
        if not fg_total.empty:
            over_row = fg_total[fg_total["side"].str.lower().str.startswith("over")]
            if not over_row.empty:
                s = str(over_row.iloc[0]["side"])
                mt = re.search(r"(\d+(\.\d+)?)", s)
                if mt:
                    full_total_num = float(mt.group(1))

        # Park factor from home team
        home_team = m.split(" vs ")[0] if " vs " in m else ""
        pf = get_park_factor(home_team)

        # Expected F5 total
        mu_f5_total = expected_f5_total(full_total_num, pf)

        # Spread: derive expected margin from pitchers if present in F5 CSV
        # Expect columns 'home_pitcher_era' and 'away_pitcher_era' if you merged them upstream
        mu_margin = None
        if not f5_spreads.empty and {"home_pitcher_era", "away_pitcher_era"}.issubset(set(f5_spreads.columns)):
            row0 = f5_spreads.iloc[0]
            mu_margin = expected_f5_margin_from_pitchers(
                row0.get("home_pitcher_era"), row0.get("away_pitcher_era")
            )

        # If pitcher ERA missing, fall back: use sign of full-game run line to set margin near 0.35–0.55
        if mu_margin is None and not fg_runline.empty:
            fav_row = fg_runline[fg_runline["runline"] < 0]
            if not fav_row.empty:
                sign = -1.0  # negative runline means home or listed favorite
                mu_margin = 0.40 * (1 if sign < 0 else -1) + HOME_EDGE_F5

        # Evaluate bet edges

        # F5 Spread market (choose most recent, best price on favorite side)
        if not f5_spreads.empty and mu_margin is not None:
            fav_rows = f5_spreads[f5_spreads["point"] < 0]
            if not fav_rows.empty:
                # prefer best (highest) price with newest update
                fav_rows = fav_rows.sort_values(by=["price", "last_update"], ascending=[False, False])
                r = fav_rows.iloc[0]
                line = float(r["point"])  # should be -0.5 typically
                odds = clean_odds(r["price"])
                # Convert μ_margin to cover probability vs -0.5
                p_model = prob_cover_minus_half(mu_margin)
                p_implied = implied_prob_from_odds(odds)
                edge = p_model - p_implied
                out_rows.append({
                    "matchup": m, "bet_type": "F5 Spread", "side": r["label"],
                    "line": line, "odds": odds, "bookmaker": r.get("bookmaker"),
                    "model_p": round(p_model, 4), "implied_p": round(p_implied, 4), "edge": round(edge, 4),
                    "explain": f"μ_margin={mu_margin:.2f}, sd={SD_F5_DIFF}"
                })

        # F5 Total Over side
        if mu_f5_total is not None and not f5_totals.empty:
            over_rows = f5_totals[f5_totals["label"].str.lower() == "over"]
            if not over_rows.empty:
                r = over_rows.sort_values(by=["price", "last_update"], ascending=[False, False]).iloc[0]
                tline = float(r["point"])
                odds = clean_odds(r["price"])
                p_model = prob_over_poisson(tline, mu_f5_total)
                p_implied = implied_prob_from_odds(odds)
                edge = p_model - p_implied
                out_rows.append({
                    "matchup": m, "bet_type": "F5 Total", "side": "Over",
                    "line": tline, "odds": odds, "bookmaker": r.get("bookmaker"),
                    "model_p": round(p_model, 4), "implied_p": round(p_implied, 4), "edge": round(edge, 4),
                    "explain": f"μ_total={mu_f5_total:.2f}, park={pf:.2f}"
                })

    res = pd.DataFrame(out_rows)
    if not res.empty:
        res = res.sort_values("edge", ascending=False).reset_index(drop=True)
    return res

# -------------------- I/O --------------------
def load_f5_csv(path: str = F5_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # canonicalize market tags
    df["market"] = df["market"].str.strip().str.lower()
    # enforce the only valid tags we depend on
    df["market"] = df["market"].replace({
        "spreads_1st_5_innings": "spreads_1st_5_innings",
        "totals_1st_5_innings": "totals_1st_5_innings",
        "totals_1st_1_innings": "totals_1st_5_innings",  # hard fix for bad dumps
    })
    # normalize matchup
    df["home_team"] = df["home_team"].map(_normalize_team)
    df["away_team"] = df["away_team"].map(_normalize_team)
    return df

def main():
    full = fetch_dk_splits(MLB_EVENT_GROUP, "today")
    if full.empty:
        raise RuntimeError("No full-game splits fetched.")
    f5 = load_f5_csv(F5_CSV_PATH)
    if f5.empty:
        raise RuntimeError("F5 CSV is empty or missing required columns.")
    res = analyze_f5(full, f5)
    if res.empty:
        print("No edges found. Tight slate or bad inputs.")
    else:
        print(res.to_string(index=False))

if __name__ == "__main__":
    main()
