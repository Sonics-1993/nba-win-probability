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
SIGMA = 4.585           # empirical std dev of MLB run differential (2025 season)
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


def load_srs(season: int = 2025) -> dict[str, float]:
    """Load latest SRS per team from blended_srs_{season}.csv."""
    f = CACHE / f"blended_srs_{season}.csv"
    if not f.exists():
        f = CACHE / f"cum_srs_{season}.csv"
    if not f.exists():
        return {}
    latest: dict[str, tuple[str, float]] = {}
    for row in csv.DictReader(open(f)):
        team = row["team"]
        d    = row["date"]
        srs  = float(row.get("blended_srs", row.get("cum_srs", 0)))
        if team not in latest or d > latest[team][0]:
            latest[team] = (d, srs)
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
        "hydrate": "probablePitcher,team", "gameType": "R",
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
                "game_pk":          g["gamePk"],
                "home":             home,
                "away":             away,
                "home_pitcher_id":  hp.get("id"),
                "home_pitcher_name": hp.get("fullName", "TBD"),
                "away_pitcher_id":  ap.get("id"),
                "away_pitcher_name": ap.get("fullName", "TBD"),
                "game_time":        g.get("gameDate", ""),
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
    exp   = load_experiment()
    mp    = load_model_params()
    pf    = load_park_factors()
    srs   = load_srs(2025)          # 2025 end-of-season SRS as 2026 proxy
    games_2026 = load_games_2026()
    print(f"  SRS loaded for {len(srs)} teams (end-of-2025 as proxy)")

    # Fetch all market odds once (covers all upcoming games)
    print("\nFetching market odds (h2h + spreads + totals)…")
    odds_map = fetch_all_odds(api_key)
    print(f"  {len(odds_map)} games with market odds\n")

    # Pitcher cache
    pitcher_cache: dict[int, dict] = {}

    for day_offset in range(args.days):
        game_date = (date.fromisoformat(args.date) + timedelta(days=day_offset)).isoformat()

        print(f"\n{'═'*90}")
        print(f"  {game_date}")
        print(f"{'═'*90}")

        # Per-date features
        roll10   = build_roll10_diff(games_2026, game_date)
        rest_map = build_rest_map(games_2026, game_date)
        bp_era   = load_cached_bp_era(args.season)

        print(f"Fetching schedule for {game_date}…")
        games = fetch_schedule(game_date)
        if not games:
            print("  No games scheduled.")
            continue
        print(f"  {len(games)} game(s)\n")

        # Fetch pitcher stats (cached across days)
        new_ids = {g["home_pitcher_id"] for g in games} | {g["away_pitcher_id"] for g in games}
        new_ids -= {None} | set(pitcher_cache.keys())
        for pid in new_ids:
            pitcher_cache[pid] = fetch_pitcher_stats(pid, args.season)

        # Header
        print(f"{'Matchup':<13} {'Away SP / FIP':<22} {'Home SP / FIP':<22} "
              f"{'── Run Diff ──':^14} {'── Moneyline ──':^17} {'── Run Line ──':^17} {'── Over/Under ──':^18}")
        print(f"{'':13} {'':22} {'':22} "
              f"{'Pred':>5} {'σ':>4}  "
              f"{'Mdl-H':>6} {'Mkt-H':>6} {'Mkt-A':>6}  "
              f"{'MdlH':>5} {'MktH':>6} {'MktA':>6}  "
              f"{'Pred':>5} {'Mkt':>5} {'Edge':>5}")
        print("─" * 130)

        for g in games:
            home, away = g["home"], g["away"]

            hp = pitcher_cache.get(g["home_pitcher_id"],
                 dict(cum_fip=REPLACEMENT_FIP, l3_fip=REPLACEMENT_FIP,
                      cum_era=REPLACEMENT_ERA, l3_era=REPLACEMENT_ERA, name="TBD"))
            ap = pitcher_cache.get(g["away_pitcher_id"],
                 dict(cum_fip=REPLACEMENT_FIP, l3_fip=REPLACEMENT_FIP,
                      cum_era=REPLACEMENT_ERA, l3_era=REPLACEMENT_ERA, name="TBD"))

            mkt = odds_map.get((home, away), {})

            # Totals prediction
            row = {
                "home_sp_fip": hp["cum_fip"], "away_sp_fip": ap["cum_fip"],
                "home_l3_fip": hp["l3_fip"],  "away_l3_fip": ap["l3_fip"],
                "home_sp_era": hp["cum_era"], "away_sp_era": ap["cum_era"],
                "home_l3_era": hp["l3_era"],  "away_l3_era": ap["l3_era"],
                "home_bp_era": bp_era.get(home, REPLACEMENT_BP),
                "away_bp_era": bp_era.get(away, REPLACEMENT_BP),
                "park_factor": pf.get(home, 1.0),
                "is_outdoor":  0 if home in DOME_PARKS else 1,
                "temp_f": 72.0, "tailwind_mph": 0.0,
                "home_roll10": "",  "away_roll10": "",
                "home_ops": "",    "away_ops": "",
                "home_srs": 0.0,   "away_srs": 0.0,
                "home_bp_ip_3d": 0.0, "away_bp_ip_3d": 0.0,
                "open_ou": mkt.get("ou", "") or "",
                "close_ou": mkt.get("ou", "") or "",
            }
            pred_total = exp.predict(row)

            # Run differential prediction
            mu_diff = predict_run_diff(home, away, srs, rest_map, roll10, pf, mp,
                                       home_fip=hp["cum_fip"], away_fip=ap["cum_fip"])

            # Probabilities
            p_home_ml, p_away_ml   = run_diff_to_ml(mu_diff)
            p_home_rl, p_away_rl   = run_diff_to_rl(mu_diff)

            # Market odds
            mkt_ml_home = mkt.get("ml_home")
            mkt_ml_away = mkt.get("ml_away")
            mkt_rl_home = mkt.get("rl_price_home")
            mkt_rl_away = mkt.get("rl_price_away")
            mkt_ou      = mkt.get("ou")

            ou_edge = f"{pred_total - mkt_ou:+.1f}" if mkt_ou else "  N/A"

            matchup = f"{away}@{home}"
            away_sp = f"{g['away_pitcher_name'][:14]} {ap['cum_fip']:.2f}"
            home_sp = f"{g['home_pitcher_name'][:14]} {hp['cum_fip']:.2f}"

            print(f"{matchup:<13} {away_sp:<22} {home_sp:<22} "
                  f"{mu_diff:>+5.1f} {SIGMA:>4.1f}  "
                  f"{prob_to_american(p_home_ml):>6} {fmt_american(mkt_ml_home):>6} {fmt_american(mkt_ml_away):>6}  "
                  f"{prob_to_american(p_home_rl):>5} {fmt_american(mkt_rl_home):>6} {fmt_american(mkt_rl_away):>6}  "
                  f"{pred_total:>5.1f} {mkt_ou or ' N/A':>5} {ou_edge:>5}")

        print("\nNOTE: Run diff σ=4.6 (empirical 2025). ML/RL model uses end-2025 SRS.")
        print("      TBD pitcher = replacement FIP 4.40. Edge = model O/U minus market O/U.")


if __name__ == "__main__":
    main()
