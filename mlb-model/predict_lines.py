#!/usr/bin/env python3
"""
predict_lines.py — Multi-market MLB game predictions

Outputs predicted moneyline, run line (±1.5), and O/U for upcoming games
using two models:
  • Totals model  (experiment.py)  → predicted run total
  • Spread model  (model_params.py) → predicted run differential → ML + RL odds

Usage:
    python3 predict_lines.py [--days 3] [--date YYYY-MM-DD]
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE = Path(__file__).parent
CACHE = BASE / "cache"

# ── Constants ─────────────────────────────────────────────────────────────────
SIGMA       = 4.585     # empirical std dev of MLB run differential (2025 season)
SIGMA_SLOPE = 0.15      # σ ± 0.15 per 1.0 FIP above/below replacement
SIGMA_MIN   = 3.8
SIGMA_MAX   = 5.5
REPLACEMENT_FIP = 4.40
REPLACEMENT_ERA = 4.50
REPLACEMENT_BP  = 4.80
FIP_CONSTANT    = 3.15

MLB_API  = "https://statsapi.mlb.com/api/v1"
ODDS_API = "https://api.the-odds-api.com/v4"
SPORT    = "baseball_mlb"

DOME_PARKS = {"TB", "MIA", "HOU", "AZ", "MIL", "SEA", "MIN", "LAA", "ATH", "TEX"}

TEAM_NAME_MAP = {
    "Arizona Diamondbacks":  "AZ",   "Atlanta Braves":       "ATL",
    "Baltimore Orioles":     "BAL",  "Boston Red Sox":       "BOS",
    "Chicago Cubs":          "CHC",  "Chicago White Sox":    "CWS",
    "Cincinnati Reds":       "CIN",  "Cleveland Guardians":  "CLE",
    "Colorado Rockies":      "COL",  "Detroit Tigers":       "DET",
    "Houston Astros":        "HOU",  "Kansas City Royals":   "KC",
    "Los Angeles Angels":    "LAA",  "Los Angeles Dodgers":  "LAD",
    "Miami Marlins":         "MIA",  "Milwaukee Brewers":    "MIL",
    "Minnesota Twins":       "MIN",  "New York Mets":        "NYM",
    "New York Yankees":      "NYY",  "Athletics":            "ATH",
    "Oakland Athletics":     "ATH",  "Sacramento Athletics": "ATH",
    "Philadelphia Phillies": "PHI",  "Pittsburgh Pirates":   "PIT",
    "San Diego Padres":      "SD",   "San Francisco Giants": "SF",
    "Seattle Mariners":      "SEA",  "St. Louis Cardinals":  "STL",
    "Tampa Bay Rays":        "TB",   "Texas Rangers":        "TEX",
    "Toronto Blue Jays":     "TOR",  "Washington Nationals": "WSH",
}

TEAM_ID_MAP = {
    109: "AZ",  144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC", 145: "CWS",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC",
    108: "LAA", 119: "LAD", 146: "MIA", 158: "MIL", 142: "MIN", 121: "NYM",
    147: "NYY", 133: "ATH", 143: "PHI", 134: "PIT", 135: "SD",  137: "SF",
    136: "SEA", 138: "STL", 139: "TB",  140: "TEX", 141: "TOR", 120: "WSH",
}


# ── Math helpers ──────────────────────────────────────────────────────────────
def _ncdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_to_american(p: float) -> str:
    p = max(0.001, min(0.999, p))
    if p >= 0.5:
        return f"{int(-p / (1 - p) * 100)}"
    else:
        return f"+{int((1 - p) / p * 100)}"


def run_diff_to_ml(mu: float, sigma: float = SIGMA) -> tuple[float, float]:
    """P(home wins), P(away wins) from predicted run diff."""
    p_home = _ncdf(mu / sigma)
    return p_home, 1.0 - p_home


def run_diff_to_rl(mu: float, sigma: float = SIGMA) -> tuple[float, float]:
    """P(home covers -1.5), P(away covers +1.5) from predicted run diff."""
    p_home_cover = 1.0 - _ncdf((1.5 - mu) / sigma)
    return p_home_cover, 1.0 - p_home_cover


def game_sigma(home_fip: float, away_fip: float) -> float:
    """Per-game σ: tighter for ace duels, wider for high-FIP matchups."""
    avg_fip = (home_fip + away_fip) / 2.0
    raw = SIGMA + SIGMA_SLOPE * (avg_fip - REPLACEMENT_FIP)
    return round(max(SIGMA_MIN, min(SIGMA_MAX, raw)), 2)


def _local_hour(utc_str: str, tz_name: str) -> int:
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo(tz_name)).hour
    except Exception:
        return 19


def _tailwind_comp(wind_mph: float, wind_dir: float, cf_bearing: int) -> float:
    """Wind component blowing out toward CF. Positive = helps offense."""
    angle = math.radians(wind_dir - (cf_bearing + 180))
    return round(wind_mph * math.cos(angle), 2)


# ── Data loading ──────────────────────────────────────────────────────────────
def _get(url, params=None):
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def load_experiment():
    spec = importlib.util.spec_from_file_location("experiment", BASE / "experiment.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model_params():
    spec = importlib.util.spec_from_file_location("model_params", BASE / "model_params.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_park_factors() -> dict[str, float]:
    p = CACHE / "park_factors.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_stadiums() -> dict:
    spec = importlib.util.spec_from_file_location("fetch_weather", BASE / "fetch_weather.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.STADIUMS


def fetch_forecast_weather(games: list[dict], stadiums: dict) -> dict[int, dict]:
    """Returns {game_pk: {temp_f, tailwind_mph, is_outdoor}} fetched from Open-Meteo forecast."""
    venues_needed = {g["venue"]: stadiums[g["venue"]]
                     for g in games
                     if g.get("venue") in stadiums and stadiums[g["venue"]][3] == 1}

    wx_by_venue: dict[str, dict] = {}
    for venue, (lat, lon, tz, _, _) in venues_needed.items():
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
                "wind_speed_unit": "mph", "temperature_unit": "fahrenheit",
                "timezone": tz, "forecast_days": 7,
            }, timeout=15)
            if r.status_code != 200:
                continue
            h = r.json()["hourly"]
            by_date: dict[str, dict] = {}
            for i, ts in enumerate(h["time"]):
                d, hr = ts[:10], int(ts[11:13])
                by_date.setdefault(d, {})[hr] = {
                    "temp_f":   round(h["temperature_2m"][i],    1),
                    "wind_mph": round(h["wind_speed_10m"][i],    1),
                    "wind_dir": round(h["wind_direction_10m"][i], 1),
                }
            wx_by_venue[venue] = by_date
        except Exception:
            pass

    result: dict[int, dict] = {}
    for g in games:
        pk    = g["game_pk"]
        venue = g.get("venue", "")
        if venue not in stadiums:
            result[pk] = {"temp_f": 72.0, "tailwind_mph": 0.0,
                          "is_outdoor": 0 if g["home"] in DOME_PARKS else 1}
            continue
        _, _, tz, is_outdoor, bearing = stadiums[venue]
        if not is_outdoor:
            result[pk] = {"temp_f": 72.0, "tailwind_mph": 0.0, "is_outdoor": 0}
            continue
        local_hr = _local_hour(g.get("game_time", ""), tz)
        day_wx   = wx_by_venue.get(venue, {}).get(g.get("date", ""), {})
        wx       = day_wx.get(local_hr) or day_wx.get(19) or {}
        if not wx:
            result[pk] = {"temp_f": 72.0, "tailwind_mph": 0.0, "is_outdoor": 1}
            continue
        tailwind = _tailwind_comp(wx["wind_mph"], wx["wind_dir"], bearing)
        result[pk] = {"temp_f": wx["temp_f"], "tailwind_mph": tailwind, "is_outdoor": 1}
    return result


def load_srs(season: int) -> dict[str, float]:
    """Load latest power ratings for the given season.
    Prefers blended_srs (prior + actual) over raw cum_srs when available."""
    f = CACHE / f"blended_srs_{season}.csv"
    col = "blended_srs"
    if not f.exists():
        f = CACHE / f"cum_srs_{season}.csv"
        col = "cum_srs"
    if not f.exists():
        return {}
    latest: dict[str, tuple[str, float]] = {}
    for row in csv.DictReader(open(f)):
        team = row["team"]
        d    = row["date"]
        val  = float(row.get(col, 0))
        if team not in latest or d > latest[team][0]:
            latest[team] = (d, val)
    return {t: v[1] for t, v in latest.items()}


def load_games_2026() -> list[dict]:
    f = CACHE / "games_2026_raw.json"
    return json.loads(f.read_text()) if f.exists() else []


def build_roll10_diff(games: list[dict], before_date: str) -> dict[str, float]:
    """Rolling 10-game avg run differential per team (shift-1, excludes before_date)."""
    history: dict[str, list[float]] = defaultdict(list)
    result:  dict[str, float] = {}
    for g in sorted(games, key=lambda x: x["date"]):
        if g["date"] >= before_date:
            break
        home, away = g["home_abbr"], g["away_abbr"]
        rd = float(g["run_diff"])
        # Record CURRENT value before appending (shift-1 handled by loop order)
        if len(history[home]) >= 3:
            result[home] = sum(history[home][-10:]) / len(history[home][-10:])
        if len(history[away]) >= 3:
            result[away] = sum(history[away][-10:]) / len(history[away][-10:])
        history[home].append(rd)
        history[away].append(-rd)
    return result


def build_rest_map(games: list[dict], before_date: str) -> dict[str, int]:
    """Days rest per team as of before_date, from 2026 completed games."""
    last: dict[str, str] = {}
    for g in sorted(games, key=lambda x: x["date"]):
        if g["date"] >= before_date:
            break
        for abbr in (g["home_abbr"], g["away_abbr"]):
            last[abbr] = g["date"]
    rest: dict[str, int] = {}
    bd = date.fromisoformat(before_date)
    for team, last_date in last.items():
        delta = (bd - date.fromisoformat(last_date)).days
        rest[team] = min(delta, 7)
    return rest


def load_cached_bp_era(season: int) -> dict[str, float]:
    f = CACHE / f"pitcher_starts_{season}.json"
    if not f.exists():
        return {}
    games_f = CACHE / f"games_{season}_raw.json"
    if not games_f.exists():
        return {}
    games_raw = {g["gamePk"]: g for g in json.loads(games_f.read_text())}
    starts = json.loads(f.read_text())
    team_er: dict[str, float] = defaultdict(float)
    team_ip: dict[str, float] = defaultdict(float)
    for game_pk_str, sides in starts.items():
        gm = games_raw.get(int(game_pk_str))
        if not gm:
            continue
        for side in ("home", "away"):
            abbr = gm[f"{side}_abbr"]
            sp   = sides.get(f"{side}_starter", {})
            sp_ip = float(sp.get("ip", 0) or 0)
            game_ip = 9.0
            bp_ip = max(0, game_ip - sp_ip)
            sp_er = float(sp.get("er", 0) or 0)
            game_runs = gm[f"{side}_runs"] if side == "home" else gm["away_runs"]
            bp_er = max(0, game_runs - sp_er)
            team_er[abbr] += bp_er
            team_ip[abbr] += bp_ip
    result = {}
    for abbr in team_er:
        ip = team_ip[abbr]
        result[abbr] = (team_er[abbr] * 9 + 4.8030) / (ip + 30)
    return result


def _parse_ip(ip_str) -> float:
    if not ip_str:
        return 0.0
    parts = str(ip_str).split(".")
    return int(parts[0]) + (int(parts[1]) / 3 if len(parts) > 1 and parts[1] else 0)


def _fip(ip, k, bb, hr, hbp=0) -> float:
    if ip <= 0:
        return REPLACEMENT_FIP
    return min((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + FIP_CONSTANT, 7.5)


def fetch_pitcher_stats(pitcher_id: int, season: int) -> dict:
    if not pitcher_id:
        return dict(cum_fip=REPLACEMENT_FIP, l3_fip=REPLACEMENT_FIP,
                    cum_era=REPLACEMENT_ERA, l3_era=REPLACEMENT_ERA, name="TBD")
    try:
        data = _get(f"{MLB_API}/people/{pitcher_id}/stats",
                    params={"stats": "gameLog", "group": "pitching",
                            "season": season, "gameType": "R"})
        logs = data["stats"][0]["splits"]
        if not logs:
            return dict(cum_fip=REPLACEMENT_FIP, l3_fip=REPLACEMENT_FIP,
                        cum_era=REPLACEMENT_ERA, l3_era=REPLACEMENT_ERA, name="TBD")
        cum_ip, cum_k, cum_bb, cum_hr, cum_er = 0.0, 0, 0, 0, 0
        for g in logs:
            s = g["stat"]
            ip = _parse_ip(s.get("inningsPitched"))
            cum_ip += ip
            cum_k  += int(s.get("strikeOuts", 0))
            cum_bb += int(s.get("baseOnBalls", 0)) + int(s.get("intentionalWalks", 0))
            cum_hr += int(s.get("homeRuns", 0))
            cum_er += int(s.get("earnedRuns", 0))
        cum_fip = _fip(cum_ip, cum_k, cum_bb, cum_hr)
        cum_era = cum_er * 9 / cum_ip if cum_ip > 0 else REPLACEMENT_ERA
        last3 = logs[-3:]
        l3_ip  = sum(_parse_ip(g["stat"].get("inningsPitched")) for g in last3)
        l3_k   = sum(int(g["stat"].get("strikeOuts", 0)) for g in last3)
        l3_bb  = sum(int(g["stat"].get("baseOnBalls", 0)) + int(g["stat"].get("intentionalWalks", 0)) for g in last3)
        l3_hr  = sum(int(g["stat"].get("homeRuns", 0)) for g in last3)
        l3_er  = sum(int(g["stat"].get("earnedRuns", 0)) for g in last3)
        l3_fip = _fip(l3_ip, l3_k, l3_bb, l3_hr) if l3_ip > 0 else cum_fip
        l3_era = l3_er * 9 / l3_ip if l3_ip > 0 else cum_era
        name_data = _get(f"{MLB_API}/people/{pitcher_id}", params={"fields": "people,fullName"})
        name = name_data["people"][0].get("fullName", str(pitcher_id)) if name_data.get("people") else str(pitcher_id)
        return dict(cum_fip=cum_fip, l3_fip=l3_fip, cum_era=cum_era, l3_era=l3_era, name=name)
    except Exception:
        return dict(cum_fip=REPLACEMENT_FIP, l3_fip=REPLACEMENT_FIP,
                    cum_era=REPLACEMENT_ERA, l3_era=REPLACEMENT_ERA, name="TBD")


def fetch_schedule(game_date: str) -> list[dict]:
    data = _get(f"{MLB_API}/schedule", params={
        "sportId": 1, "date": game_date,
        "hydrate": "probablePitcher,team,venue", "gameType": "R",
    })
    games = []
    for db in data.get("dates", []):
        for g in db.get("games", []):
            status = g.get("status", {}).get("abstractGameCode", "")
            if status == "F":
                continue  # skip finished games
            home = TEAM_ID_MAP.get(g["teams"]["home"]["team"]["id"])
            away = TEAM_ID_MAP.get(g["teams"]["away"]["team"]["id"])
            if not home or not away:
                continue
            hp = g["teams"]["home"].get("probablePitcher") or {}
            ap = g["teams"]["away"].get("probablePitcher") or {}
            games.append({
                "game_pk":           g["gamePk"],
                "home":              home,
                "away":              away,
                "home_pitcher_id":   hp.get("id"),
                "home_pitcher_name": hp.get("fullName", "TBD"),
                "away_pitcher_id":   ap.get("id"),
                "away_pitcher_name": ap.get("fullName", "TBD"),
                "game_time":         g.get("gameDate", ""),
                "venue":             g.get("venue", {}).get("name", ""),
                "date":              game_date,
            })
    return games


def fetch_all_odds(api_key: str) -> dict:
    """Fetch h2h + spreads + totals in one call. Returns dict keyed by (home,away)."""
    result: dict[tuple, dict] = {}
    if not api_key:
        return result
    try:
        r = requests.get(f"{ODDS_API}/sports/{SPORT}/odds/",
                         params={"apiKey": api_key, "regions": "us",
                                 "markets": "h2h,spreads,totals",
                                 "oddsFormat": "american"},
                         timeout=20)
        remaining = r.headers.get("x-requests-remaining", "?")
        print(f"  Odds API credits remaining after fetch: {remaining}")
        if r.status_code != 200:
            print(f"  [warn] Odds API HTTP {r.status_code}")
            return result
        for game in r.json():
            home_abbr = TEAM_NAME_MAP.get(game.get("home_team", ""))
            away_abbr = TEAM_NAME_MAP.get(game.get("away_team", ""))
            if not home_abbr or not away_abbr:
                continue
            mkt_data: dict[str, list] = defaultdict(list)
            for bm in game.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    key = mkt["key"]
                    for o in mkt.get("outcomes", []):
                        mkt_data[key].append(o)
            game_date = game.get("commence_time", "")[:10]
            entry: dict = {"date": game_date}
            # Moneyline
            home_ml = [o["price"] for o in mkt_data["h2h"]
                       if TEAM_NAME_MAP.get(o.get("name")) == home_abbr]
            away_ml = [o["price"] for o in mkt_data["h2h"]
                       if TEAM_NAME_MAP.get(o.get("name")) == away_abbr]
            if home_ml: entry["ml_home"] = _median(home_ml)
            if away_ml: entry["ml_away"] = _median(away_ml)
            # Run line (home spread)
            home_rl = [o["point"] for o in mkt_data["spreads"]
                       if TEAM_NAME_MAP.get(o.get("name")) == home_abbr]
            home_rl_price = [o["price"] for o in mkt_data["spreads"]
                             if TEAM_NAME_MAP.get(o.get("name")) == home_abbr]
            away_rl_price = [o["price"] for o in mkt_data["spreads"]
                             if TEAM_NAME_MAP.get(o.get("name")) == away_abbr]
            if home_rl:        entry["rl_point"]      = _median(home_rl)
            if home_rl_price:  entry["rl_price_home"] = _median(home_rl_price)
            if away_rl_price:  entry["rl_price_away"] = _median(away_rl_price)
            # Totals
            ou = [o["point"] for o in mkt_data["totals"] if o.get("name") == "Over"]
            if ou: entry["ou"] = _median(ou)
            result[(home_abbr, away_abbr)] = entry
    except Exception as e:
        print(f"  [warn] Odds API: {e}")
    return result


def _median(vals: list) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def predict_run_diff(home, away, srs, rest_map, roll10, park_factors, mp,
                     home_fip: float = REPLACEMENT_FIP,
                     away_fip: float = REPLACEMENT_FIP) -> float:
    srs_diff    = srs.get(home, 0.0) - srs.get(away, 0.0)
    rest_diff   = rest_map.get(home, 3) - rest_map.get(away, 3)
    r10_diff    = roll10.get(home, 0.0) - roll10.get(away, 0.0)
    sp_fip_diff = away_fip - home_fip   # positive = home starter better
    return (
        mp.srs_weight                          * srs_diff
        + mp.era_weight                        * 0.0
        + mp.rest_weight                       * rest_diff
        + getattr(mp, "roll10_weight",     0.0) * r10_diff
        + getattr(mp, "sp_fip_diff_weight", 0.0) * sp_fip_diff
        + mp.hca
        + mp.intercept
    )


# ── Display ───────────────────────────────────────────────────────────────────
def fmt_american(price: float | None) -> str:
    if price is None:
        return "  N/A"
    p = int(round(price))
    return f"{p:+d}" if p > 0 else str(p)


def fmt_prob_odds(p: float) -> str:
    return prob_to_american(p)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",   default=date.today().isoformat())
    parser.add_argument("--days",   type=int, default=3)
    parser.add_argument("--season", type=int, default=date.today().year)
    args = parser.parse_args()

    load_dotenv(BASE.parent / "opening-lines" / ".env")
    load_dotenv(BASE / ".env")
    api_key = os.getenv("ODDS_API_KEY")

    print("\nMLB Multi-Market Predictor")
    print(f"Dates: {args.date} + {args.days - 1} more day(s)  |  Season: {args.season}\n")

    # Load static data
    print("Loading model and static data…")
    exp        = load_experiment()
    mp         = load_model_params()
    pf         = load_park_factors()
    srs        = load_srs(args.season)
    games_2026 = load_games_2026()
    stadiums   = load_stadiums()
    print(f"  Power ratings loaded for {len(srs)} teams ({args.season} regular season)")

    # Pre-fetch all schedules so we can batch weather + pitcher lookups
    print("\nFetching schedules…")
    schedules: dict[str, list[dict]] = {}
    all_games: list[dict] = []
    for day_offset in range(args.days):
        game_date = (date.fromisoformat(args.date) + timedelta(days=day_offset)).isoformat()
        day_games = fetch_schedule(game_date)
        schedules[game_date] = day_games
        all_games.extend(day_games)
        print(f"  {game_date}: {len(day_games)} game(s)")

    # Fetch all pitcher stats up-front (one API call per unique pitcher)
    print("\nFetching pitcher stats…")
    pitcher_cache: dict[int, dict] = {}
    all_ids = ({g["home_pitcher_id"] for g in all_games} |
               {g["away_pitcher_id"] for g in all_games}) - {None}
    for pid in all_ids:
        pitcher_cache[pid] = fetch_pitcher_stats(pid, args.season)

    # Batch weather for all outdoor games (one Open-Meteo call per unique venue)
    print("\nFetching forecast weather for outdoor venues…")
    wx_map = fetch_forecast_weather(all_games, stadiums)
    outdoor_fetched = sum(1 for v in wx_map.values() if v["is_outdoor"] and v["temp_f"] != 72.0)
    print(f"  {outdoor_fetched} outdoor game(s) with live forecast")

    # Fetch all market odds once (covers all upcoming games)
    print("\nFetching market odds (h2h + spreads + totals)…")
    odds_map = fetch_all_odds(api_key)
    print(f"  {len(odds_map)} games with market odds")

    bp_era = load_cached_bp_era(args.season)

    for day_offset in range(args.days):
        game_date = (date.fromisoformat(args.date) + timedelta(days=day_offset)).isoformat()
        games     = schedules[game_date]

        print(f"\n{'═'*142}")
        print(f"  {game_date}")
        print(f"{'═'*142}")

        if not games:
            print("  No games scheduled.")
            continue

        roll10   = build_roll10_diff(games_2026, game_date)
        rest_map = build_rest_map(games_2026, game_date)

        # Header
        print(f"{'Matchup':<13} {'Away SP / FIP':<22} {'Home SP / FIP':<22} "
              f"{'── Run Diff ──':^14} {'── Moneyline ──':^17} {'── Run Line ──':^17} "
              f"{'── Over/Under ──':^18} {'── Weather ──':^13}")
        print(f"{'':13} {'':22} {'':22} "
              f"{'Pred':>5} {'σ':>4}  "
              f"{'Mdl-H':>6} {'Mkt-H':>6} {'Mkt-A':>6}  "
              f"{'MdlH':>5} {'MktH':>6} {'MktA':>6}  "
              f"{'Pred':>5} {'Mkt':>5} {'Edge':>5}  "
              f"{'Temp':>4} {'Wind':>6}")
        print("─" * 142)

        for g in games:
            home, away = g["home"], g["away"]

            hp = pitcher_cache.get(g["home_pitcher_id"],
                 dict(cum_fip=REPLACEMENT_FIP, l3_fip=REPLACEMENT_FIP,
                      cum_era=REPLACEMENT_ERA, l3_era=REPLACEMENT_ERA, name="TBD"))
            ap = pitcher_cache.get(g["away_pitcher_id"],
                 dict(cum_fip=REPLACEMENT_FIP, l3_fip=REPLACEMENT_FIP,
                      cum_era=REPLACEMENT_ERA, l3_era=REPLACEMENT_ERA, name="TBD"))

            mkt = odds_map.get((home, away), {})
            wx  = wx_map.get(g["game_pk"],
                  {"temp_f": 72.0, "tailwind_mph": 0.0,
                   "is_outdoor": 0 if home in DOME_PARKS else 1})

            sigma = game_sigma(hp["cum_fip"], ap["cum_fip"])

            # Totals prediction with live weather injected
            row = {
                "home_sp_fip": hp["cum_fip"], "away_sp_fip": ap["cum_fip"],
                "home_l3_fip": hp["l3_fip"],  "away_l3_fip": ap["l3_fip"],
                "home_sp_era": hp["cum_era"], "away_sp_era": ap["cum_era"],
                "home_l3_era": hp["l3_era"],  "away_l3_era": ap["l3_era"],
                "home_bp_era": bp_era.get(home, REPLACEMENT_BP),
                "away_bp_era": bp_era.get(away, REPLACEMENT_BP),
                "park_factor": pf.get(home, 1.0),
                "is_outdoor":  wx["is_outdoor"],
                "temp_f":      wx["temp_f"],
                "tailwind_mph": wx["tailwind_mph"],
                "home_roll10": "",  "away_roll10": "",
                "home_ops": "",    "away_ops": "",
                "home_srs": 0.0,   "away_srs": 0.0,
                "home_bp_ip_3d": 0.0, "away_bp_ip_3d": 0.0,
                "open_ou": mkt.get("ou", "") or "",
                "close_ou": mkt.get("ou", "") or "",
            }
            pred_total = exp.predict(row)

            mu_diff = predict_run_diff(home, away, srs, rest_map, roll10, pf, mp,
                                       home_fip=hp["cum_fip"], away_fip=ap["cum_fip"])

            p_home_ml, p_away_ml = run_diff_to_ml(mu_diff, sigma)
            p_home_rl, p_away_rl = run_diff_to_rl(mu_diff, sigma)

            mkt_ml_home = mkt.get("ml_home")
            mkt_ml_away = mkt.get("ml_away")
            mkt_rl_home = mkt.get("rl_price_home")
            mkt_rl_away = mkt.get("rl_price_away")
            mkt_ou      = mkt.get("ou")

            ou_edge = f"{pred_total - mkt_ou:+.1f}" if mkt_ou else "  N/A"

            if wx["is_outdoor"]:
                wx_temp = f"{wx['temp_f']:.0f}°F"
                wx_wind = f"{wx['tailwind_mph']:+.1f}↑"
            else:
                wx_temp = "dome"
                wx_wind = ""

            matchup = f"{away}@{home}"
            away_sp = f"{g['away_pitcher_name'][:14]} {ap['cum_fip']:.2f}"
            home_sp = f"{g['home_pitcher_name'][:14]} {hp['cum_fip']:.2f}"

            print(f"{matchup:<13} {away_sp:<22} {home_sp:<22} "
                  f"{mu_diff:>+5.1f} {sigma:>4.2f}  "
                  f"{prob_to_american(p_home_ml):>6} {fmt_american(mkt_ml_home):>6} {fmt_american(mkt_ml_away):>6}  "
                  f"{prob_to_american(p_home_rl):>5} {fmt_american(mkt_rl_home):>6} {fmt_american(mkt_rl_away):>6}  "
                  f"{pred_total:>5.1f} {mkt_ou or ' N/A':>5} {ou_edge:>5}  "
                  f"{wx_temp:>4} {wx_wind:>6}")

        print(f"\nNOTE: σ is per-game (FIP-adjusted). ML/RL model uses {args.season} regular-season power ratings.")
        print("      TBD pitcher = replacement FIP 4.40. Edge = model O/U minus market O/U.")
        print("      Weather wind = tailwind toward CF (+=helps offense). Temp/wind weights currently 0 in totals model.")


if __name__ == "__main__":
    main()
