"""
live_totals.py — Today's MLB O/U predictions

Fetches today's schedule + probable pitchers from the MLB Stats API, computes
FIP from season game logs, pulls current O/U from the Odds API, and runs
experiment.py predict() on each game.

Usage:
    python3 live_totals.py [--date YYYY-MM-DD] [--season YYYY]
"""

import argparse
import importlib.util
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

BASE = Path(__file__).parent

REPLACEMENT_FIP = 4.40
REPLACEMENT_ERA = 4.50
REPLACEMENT_BP  = 4.80
FIP_CONSTANT    = 3.15

MLB_API  = "https://statsapi.mlb.com/api/v1"
ODDS_API = "https://api.the-odds-api.com/v4"

TEAM_NAME_MAP = {
    "Arizona Diamondbacks":  "AZ",  "Atlanta Braves":         "ATL",
    "Baltimore Orioles":     "BAL", "Boston Red Sox":         "BOS",
    "Chicago Cubs":          "CHC", "Chicago White Sox":      "CWS",
    "Cincinnati Reds":       "CIN", "Cleveland Guardians":    "CLE",
    "Colorado Rockies":      "COL", "Detroit Tigers":         "DET",
    "Houston Astros":        "HOU", "Kansas City Royals":     "KC",
    "Los Angeles Angels":    "LAA", "Los Angeles Dodgers":    "LAD",
    "Miami Marlins":         "MIA", "Milwaukee Brewers":      "MIL",
    "Minnesota Twins":       "MIN", "New York Mets":          "NYM",
    "New York Yankees":      "NYY", "Athletics":              "ATH",
    "Oakland Athletics":     "ATH", "Sacramento Athletics":   "ATH",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates":     "PIT",
    "San Diego Padres":      "SD",  "San Francisco Giants":   "SF",
    "Seattle Mariners":      "SEA", "St. Louis Cardinals":    "STL",
    "Tampa Bay Rays":        "TB",  "Texas Rangers":          "TEX",
    "Toronto Blue Jays":     "TOR", "Washington Nationals":   "WSH",
}

TEAM_ID_MAP = {
    109: "AZ",  144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC", 145: "CWS",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC",
    108: "LAA", 119: "LAD", 146: "MIA", 158: "MIL", 142: "MIN", 121: "NYM",
    147: "NYY", 133: "ATH", 143: "PHI", 134: "PIT", 135: "SD",  137: "SF",
    136: "SEA", 138: "STL", 139: "TB",  140: "TEX", 141: "TOR", 120: "WSH",
}

DOME_PARKS = {"TB", "MIA", "HOU", "AZ", "MIL", "SEA", "MIN", "LAA", "ATH", "TEX"}


def _get(url, params=None, timeout=20):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _parse_ip(ip_str) -> float:
    """'6.2' (6 inn 2 outs) → fractional innings."""
    if not ip_str:
        return 0.0
    parts = str(ip_str).split(".")
    return int(parts[0]) + (int(parts[1]) / 3 if len(parts) > 1 and parts[1] else 0)


def _fip_from_line(ip, k, bb, hr, hbp=0) -> float:
    if ip <= 0:
        return REPLACEMENT_FIP
    return (13 * hr + 3 * (bb + hbp) - 2 * k) / ip + FIP_CONSTANT


def load_experiment():
    spec = importlib.util.spec_from_file_location("experiment", BASE / "experiment.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_park_factors() -> dict:

    pf_file = BASE / "cache" / "park_factors.json"
    if pf_file.exists():
        raw = json.loads(pf_file.read_text())
        return {k: float(v) for k, v in raw.items() if not k.startswith("_")}
    return {}


def fetch_today_games(game_date: str) -> list[dict]:
    """Return list of today's scheduled games with probable pitchers."""
    data = _get(f"{MLB_API}/schedule", params={
        "sportId": 1, "date": game_date,
        "hydrate": "probablePitcher,team",
        "gameType": "R",
    })
    games = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            status = g.get("status", {}).get("abstractGameCode", "")
            if status == "F":
                continue  # skip already-finished games
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            home_abbr = TEAM_NAME_MAP.get(home["team"]["name"])
            away_abbr = TEAM_NAME_MAP.get(away["team"]["name"])
            if not home_abbr or not away_abbr:
                continue
            hp = home.get("probablePitcher", {})
            ap = away.get("probablePitcher", {})
            games.append({
                "game_pk":           g["gamePk"],
                "game_time":         g.get("gameDate", ""),
                "home":              home_abbr,
                "away":              away_abbr,
                "home_team_id":      home["team"]["id"],
                "away_team_id":      away["team"]["id"],
                "home_pitcher_id":   hp.get("id"),
                "away_pitcher_id":   ap.get("id"),
                "home_pitcher_name": hp.get("fullName", "TBD"),
                "away_pitcher_name": ap.get("fullName", "TBD"),
            })
    return games


def fetch_pitcher_stats(pitcher_id, season: int) -> dict:
    """
    Return {cum_fip, cum_era, l3_fip, l3_era} for a pitcher.
    Fetches season stats + game log from MLB Stats API.
    """
    if not pitcher_id:
        return dict(cum_fip=REPLACEMENT_FIP, cum_era=REPLACEMENT_ERA,
                    l3_fip=REPLACEMENT_FIP,  l3_era=REPLACEMENT_ERA)
    try:
        # Season cumulative stats
        s_data   = _get(f"{MLB_API}/people/{pitcher_id}/stats",
                        params={"stats": "season", "season": season, "group": "pitching"})
        s_splits = s_data.get("stats", [{}])[0].get("splits", [])
        cum_fip  = REPLACEMENT_FIP
        cum_era  = REPLACEMENT_ERA
        if s_splits:
            st  = s_splits[0]["stat"]
            ip  = _parse_ip(st.get("inningsPitched", 0))
            cum_fip = _fip_from_line(ip,
                                     int(st.get("strikeOuts",  0)),
                                     int(st.get("baseOnBalls", 0)),
                                     int(st.get("homeRuns",    0)),
                                     int(st.get("hitByPitch",  0)))
            era_str = st.get("era", "")
            if era_str and era_str not in ("-.--", ""):
                cum_era = float(era_str)

        # Game log — last 3 starts
        gl_data   = _get(f"{MLB_API}/people/{pitcher_id}/stats",
                         params={"stats": "gameLog", "season": season, "group": "pitching"})
        gl_splits = gl_data.get("stats", [{}])[0].get("splits", [])
        starts    = [s for s in gl_splits
                     if _parse_ip(s["stat"].get("inningsPitched", 0)) >= 1.0
                     and int(s["stat"].get("gamesStarted", 0)) >= 1]
        l3_fip    = REPLACEMENT_FIP
        l3_era    = REPLACEMENT_ERA
        if starts:
            last3    = starts[-3:]
            tot_ip   = sum(_parse_ip(s["stat"].get("inningsPitched", 0)) for s in last3)
            tot_k    = sum(int(s["stat"].get("strikeOuts",  0)) for s in last3)
            tot_bb   = sum(int(s["stat"].get("baseOnBalls", 0)) for s in last3)
            tot_hr   = sum(int(s["stat"].get("homeRuns",    0)) for s in last3)
            tot_hbp  = sum(int(s["stat"].get("hitByPitch",  0)) for s in last3)
            tot_er   = sum(int(s["stat"].get("earnedRuns",  0)) for s in last3)
            l3_fip   = _fip_from_line(tot_ip, tot_k, tot_bb, tot_hr, tot_hbp)
            l3_era   = (tot_er * 9 / tot_ip) if tot_ip > 0 else REPLACEMENT_ERA

        return dict(cum_fip=cum_fip, cum_era=cum_era, l3_fip=l3_fip, l3_era=l3_era)

    except Exception as e:
        print(f"  [warn] pitcher {pitcher_id}: {e}")
        return dict(cum_fip=REPLACEMENT_FIP, cum_era=REPLACEMENT_ERA,
                    l3_fip=REPLACEMENT_FIP,  l3_era=REPLACEMENT_ERA)


def build_team_roll10(game_date: str, season: int, window: int = 10) -> dict[str, float]:
    """
    Fetch all completed games in the last 35 days and return rolling avg
    runs scored per team (shift-1, up to but not including game_date).
    Single API call covers all teams.
    """
    start = (date.fromisoformat(game_date) - timedelta(days=35)).isoformat()
    data  = _get(f"{MLB_API}/schedule", params={
        "sportId": 1, "startDate": start, "endDate": game_date,
        "hydrate": "linescore", "gameType": "R",
    })
    # Collect per-team run history in date order
    history: dict[int, list[float]] = {}
    for date_block in sorted(data.get("dates", []), key=lambda d: d["date"]):
        if date_block["date"] >= game_date:
            continue
        for g in date_block.get("games", []):
            if g.get("status", {}).get("abstractGameCode") != "F":
                continue
            for side in ("home", "away"):
                t = g["teams"].get(side, {})
                tid   = t.get("team", {}).get("id")
                score = t.get("score")
                if tid and score is not None:
                    history.setdefault(tid, []).append(float(score))

    result = {}
    for tid, scores in history.items():
        abbr = TEAM_ID_MAP.get(tid)
        if abbr and len(scores) >= 3:
            last_n = scores[-window:]
            result[abbr] = sum(last_n) / len(last_n)
    return result


def fetch_current_ou(api_key: str | None) -> dict[tuple[str, str], float]:
    """
    Fetch current Odds API totals lines.
    Returns {(home_abbr, away_abbr): ou_line}.
    """
    if not api_key:
        return {}
    try:
        r = requests.get(f"{ODDS_API}/sports/baseball_mlb/odds/",
                         params={"apiKey": api_key, "regions": "us",
                                 "markets": "totals", "oddsFormat": "american"},
                         timeout=20)
        remaining = r.headers.get("x-requests-remaining", "?")
        print(f"  Odds API credits remaining: {remaining}")
        if r.status_code != 200:
            print(f"  [warn] Odds API HTTP {r.status_code}")
            return {}
        result = {}
        for game in r.json():
            home_abbr = TEAM_NAME_MAP.get(game.get("home_team", ""))
            away_abbr = TEAM_NAME_MAP.get(game.get("away_team", ""))
            if not home_abbr or not away_abbr:
                continue
            lines = []
            for bm in game.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt["key"] != "totals":
                        continue
                    for outcome in mkt.get("outcomes", []):
                        if outcome.get("name") == "Over" and outcome.get("point") is not None:
                            lines.append(float(outcome["point"]))
            if lines:
                result[(home_abbr, away_abbr)] = float(np.median(lines))
        return result
    except Exception as e:
        print(f"  [warn] Odds API: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",   default=date.today().isoformat())
    parser.add_argument("--season", type=int, default=date.today().year)
    args = parser.parse_args()

    load_dotenv(BASE.parent / "opening-lines" / ".env")
    load_dotenv(BASE / ".env")
    api_key = os.getenv("ODDS_API_KEY")

    exp         = load_experiment()
    park_factors = load_park_factors()

    print(f"\nMLB Live Totals — {args.date}  (season {args.season})\n")

    print("Fetching today's schedule…")
    games = fetch_today_games(args.date)
    if not games:
        print("No games scheduled (or all finished).")
        return
    print(f"  {len(games)} game(s) found\n")

    print("Fetching team rolling 10-game runs…")
    roll10 = build_team_roll10(args.date, args.season)
    print(f"  {len(roll10)} teams with run history\n")

    print("Fetching current O/U lines…")
    ou_map = fetch_current_ou(api_key)
    print(f"  {len(ou_map)} games with O/U lines\n")

    print("Fetching pitcher stats…")
    pitcher_cache: dict[int, dict] = {}
    pitcher_ids = set()
    for g in games:
        if g["home_pitcher_id"]: pitcher_ids.add(g["home_pitcher_id"])
        if g["away_pitcher_id"]: pitcher_ids.add(g["away_pitcher_id"])
    for pid in pitcher_ids:
        print(f"  pitcher {pid}…")
        pitcher_cache[pid] = fetch_pitcher_stats(pid, args.season)

    # ── Print predictions ─────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"{'Matchup':<18} {'Away SP':<22} {'Home SP':<22} {'O/U':>5} {'Pred':>5} {'Edge':>6}")
    print(f"{'─'*80}")

    for g in games:
        home = g["home"]
        away = g["away"]

        hp = pitcher_cache.get(g["home_pitcher_id"],
             dict(cum_fip=REPLACEMENT_FIP, l3_fip=REPLACEMENT_FIP,
                  cum_era=REPLACEMENT_ERA, l3_era=REPLACEMENT_ERA))
        ap = pitcher_cache.get(g["away_pitcher_id"],
             dict(cum_fip=REPLACEMENT_FIP, l3_fip=REPLACEMENT_FIP,
                  cum_era=REPLACEMENT_ERA, l3_era=REPLACEMENT_ERA))

        ou = ou_map.get((home, away))

        row = {
            "home_sp_fip":  hp["cum_fip"],
            "away_sp_fip":  ap["cum_fip"],
            "home_l3_fip":  hp["l3_fip"],
            "away_l3_fip":  ap["l3_fip"],
            "home_sp_era":  hp["cum_era"],
            "away_sp_era":  ap["cum_era"],
            "home_l3_era":  hp["l3_era"],
            "away_l3_era":  ap["l3_era"],
            "home_bp_era":  REPLACEMENT_BP,
            "away_bp_era":  REPLACEMENT_BP,
            "park_factor":  park_factors.get(home, 1.0),
            "is_outdoor":   0 if home in DOME_PARKS else 1,
            "temp_f":       72.0,
            "tailwind_mph": 0.0,
            "home_roll10":  roll10.get(home, ""),
            "away_roll10":  roll10.get(away, ""),
            "home_ops":     "",
            "away_ops":     "",
            "home_srs":     0.0,
            "away_srs":     0.0,
            "home_bp_ip_3d": 0.0,
            "away_bp_ip_3d": 0.0,
            "open_ou":      ou or "",
            "close_ou":     ou or "",
        }

        pred = exp.predict(row)
        ou_str = f"{ou:.1f}" if ou else "  N/A"
        edge_str = f"{pred - ou:+.2f}" if ou else "   N/A"

        home_sp_str = f"{g['home_pitcher_name'][:18]} ({hp['cum_fip']:.2f})"
        away_sp_str = f"{g['away_pitcher_name'][:18]} ({ap['cum_fip']:.2f})"
        matchup = f"{away}@{home}"

        print(f"{matchup:<18} {away_sp_str:<22} {home_sp_str:<22} "
              f"{ou_str:>5} {pred:>5.1f} {edge_str:>6}")

    print(f"{'─'*80}")
    print("\nNote: bp_era uses replacement value (4.80) — run full pipeline for live BP stats.")
    print("Edge = model prediction minus current O/U line (positive = lean over).\n")


if __name__ == "__main__":
    main()
