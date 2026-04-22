#!/usr/bin/env python3
"""
Re-parse cached 2024 odds snapshots to extract run line point + price.

Rewrites cache/odds_history_2024.csv with columns:
  date, home, away,
  open_rl_point, open_rl_price_home, open_rl_price_away, open_books,
  close_rl_point, close_rl_price_home, close_rl_price_away, close_books

open_rl_point is the home team's spread point (−1.5 = home fav, +1.5 = home dog).
Prices are median American odds across bookmakers.
"""

import csv
import json
import statistics
from pathlib import Path

BASE       = Path(__file__).parent
CACHE      = BASE / "cache"
ODDS_DIR   = CACHE / "odds_2024"
GAMES_FILE = CACHE / "games_2024_raw.json"
OUT_CSV    = CACHE / "odds_history_2024.csv"

MLB_TEAM_MAP = {
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

FIELDS = [
    "date", "home", "away",
    "open_rl_point", "open_rl_price_home", "open_rl_price_away", "open_books",
    "close_rl_point", "close_rl_price_home", "close_rl_price_away", "close_books",
]


def extract(game: dict, home_abbr: str, away_abbr: str):
    """Return (home_point, home_price, away_price, n_books) using medians."""
    home_pts, home_prs, away_prs = [], [], []
    for bm in game.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market["key"] != "spreads":
                continue
            home_o = next((o for o in market["outcomes"]
                           if MLB_TEAM_MAP.get(o["name"]) == home_abbr), None)
            away_o = next((o for o in market["outcomes"]
                           if MLB_TEAM_MAP.get(o["name"]) == away_abbr), None)
            if not home_o or not away_o:
                continue
            hp = home_o.get("point")
            hpr = home_o.get("price")
            apr = away_o.get("price")
            if hp is not None and hpr is not None and apr is not None:
                home_pts.append(float(hp))
                home_prs.append(float(hpr))
                away_prs.append(float(apr))
    if not home_pts:
        return None, None, None, 0
    return (
        statistics.median(home_pts),
        statistics.median(home_prs),
        statistics.median(away_prs),
        len(home_pts),
    )


def load_snapshot(date_str: str, snap: str) -> dict[tuple[str, str], dict]:
    p = ODDS_DIR / f"snapshot_{date_str}_{snap}.json"
    if not p.exists():
        return {}
    body = json.loads(p.read_text())
    idx = {}
    for g in body.get("data", []):
        h = MLB_TEAM_MAP.get(g.get("home_team", ""))
        a = MLB_TEAM_MAP.get(g.get("away_team", ""))
        if h and a:
            idx[(h, a)] = g
    return idx


def fmt(v) -> str:
    return f"{v:.2f}" if v is not None else ""


def main():
    games_by_date: dict[str, list[dict]] = {}
    for g in json.loads(GAMES_FILE.read_text()):
        games_by_date.setdefault(g["date"], []).append(g)

    rows = []
    for game_date in sorted(games_by_date):
        open_idx  = load_snapshot(game_date, "opening")
        close_idx = load_snapshot(game_date, "closing")
        if not open_idx and not close_idx:
            continue
        for game in games_by_date[game_date]:
            home, away = game["home_abbr"], game["away_abbr"]
            key = (home, away)
            o_pt, o_ph, o_pa, o_n = extract(open_idx.get(key, {}),  home, away)
            c_pt, c_ph, c_pa, c_n = extract(close_idx.get(key, {}), home, away)
            rows.append({
                "date":               game_date,
                "home":               home,
                "away":               away,
                "open_rl_point":      fmt(o_pt),
                "open_rl_price_home": fmt(o_ph),
                "open_rl_price_away": fmt(o_pa),
                "open_books":         o_n,
                "close_rl_point":     fmt(c_pt),
                "close_rl_price_home":fmt(c_ph),
                "close_rl_price_away":fmt(c_pa),
                "close_books":        c_n,
            })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with_price = sum(1 for r in rows if r["open_rl_price_home"])
    print(f"{len(rows)} games written, {with_price} with open price → {OUT_CSV}")


if __name__ == "__main__":
    main()
