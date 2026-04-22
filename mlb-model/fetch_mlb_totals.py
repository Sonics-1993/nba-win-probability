"""
fetch_mlb_totals.py — Historical Over/Under totals odds fetcher

Uses The Odds API historical endpoint with markets=totals.
Snapshots stored separately from run-line cache (cache/totals_{season}/).

Usage:
    python3 fetch_mlb_totals.py --season 2025 --dry-run
    python3 fetch_mlb_totals.py --season 2025
"""

import argparse
import csv
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

BASE      = Path(__file__).parent
ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT     = "baseball_mlb"
REGIONS   = "us"
MARKET    = "totals"
DELAY     = 1.1

MLB_TEAM_MAP = {
    "Arizona Diamondbacks":  "AZ",  "Atlanta Braves":        "ATL",
    "Baltimore Orioles":     "BAL", "Boston Red Sox":        "BOS",
    "Chicago Cubs":          "CHC", "Chicago White Sox":     "CWS",
    "Cincinnati Reds":       "CIN", "Cleveland Guardians":   "CLE",
    "Colorado Rockies":      "COL", "Detroit Tigers":        "DET",
    "Houston Astros":        "HOU", "Kansas City Royals":    "KC",
    "Los Angeles Angels":    "LAA", "Los Angeles Dodgers":   "LAD",
    "Miami Marlins":         "MIA", "Milwaukee Brewers":     "MIL",
    "Minnesota Twins":       "MIN", "New York Mets":         "NYM",
    "New York Yankees":      "NYY", "Athletics":             "ATH",
    "Oakland Athletics":     "ATH", "Sacramento Athletics":  "ATH",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates":    "PIT",
    "San Diego Padres":      "SD",  "San Francisco Giants":  "SF",
    "Seattle Mariners":      "SEA", "St. Louis Cardinals":   "STL",
    "Tampa Bay Rays":        "TB",  "Texas Rangers":         "TEX",
    "Toronto Blue Jays":     "TOR", "Washington Nationals":  "WSH",
}


def snapshot_times(game_date_str: str) -> tuple[str, str]:
    d = date.fromisoformat(game_date_str)
    opening = datetime(d.year, d.month, d.day, 14,  0, 0, tzinfo=timezone.utc)
    closing = datetime(d.year, d.month, d.day, 16, 30, 0, tzinfo=timezone.utc)
    return opening.strftime("%Y-%m-%dT%H:%M:%SZ"), closing.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_snapshot(api_key: str, iso_dt: str) -> tuple[list[dict], str]:
    r = requests.get(
        f"{ODDS_BASE}/historical/sports/{SPORT}/odds/",
        params={"apiKey": api_key, "regions": REGIONS, "markets": MARKET,
                "oddsFormat": "american", "date": iso_dt, "dateFormat": "iso"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("data", []), r.headers.get("x-requests-remaining", "?")


def extract_total(game: dict) -> float | None:
    lines = []
    for bm in game.get("bookmakers", []):
        for mkt in bm.get("markets", []):
            if mkt["key"] != "totals":
                continue
            for outcome in mkt.get("outcomes", []):
                if outcome.get("name") == "Over" and outcome.get("point") is not None:
                    lines.append(float(outcome["point"]))
    return float(np.median(lines)) if lines else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",    type=int, default=2025)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--open-only", action="store_true",
                        help="Fetch opening snapshot only (saves ~half the credits)")
    args = parser.parse_args()
    s = args.season

    load_dotenv(BASE.parent / "opening-lines" / ".env")
    load_dotenv(BASE / ".env")
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not found in .env"); return

    games_file = BASE / "cache" / f"games_{s}_raw.json"
    if not games_file.exists():
        print(f"Run fetch_mlb_games.py --season {s} first."); return

    games   = json.loads(games_file.read_text())
    by_date: dict[str, list] = {}
    for g in games:
        by_date.setdefault(g["date"], []).append(g)
    all_dates = sorted(by_date.keys())

    snap_dir = BASE / "cache" / f"totals_{s}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    snaps = ("opening",) if args.open_only else ("opening", "closing")
    total_slots = len(all_dates) * len(snaps)
    to_fetch = [(d, snap) for d in all_dates for snap in snaps
                if args.force or not (snap_dir / f"snapshot_{d}_{snap}.json").exists()]

    print(f"MLB {s} Totals Odds Fetcher{'  [opening only]' if args.open_only else ''}")
    print(f"  Dates: {all_dates[0]} → {all_dates[-1]}  ({len(all_dates)} days)")
    print(f"  To fetch: {len(to_fetch)}  (cached: {total_slots - len(to_fetch)})")
    print(f"  Est. credits: ~{len(to_fetch) * 10}")

    # Check balance
    r = requests.get(f"{ODDS_BASE}/sports/", params={"apiKey": api_key}, timeout=10)
    print(f"  Credits remaining: {r.headers.get('x-requests-remaining','?')}")

    if args.dry_run:
        print("--dry-run: no calls made."); return

    last_remaining = "?"
    for i, (d, snap) in enumerate(to_fetch):
        open_dt, close_dt = snapshot_times(d)
        iso_dt = open_dt if snap == "opening" else close_dt
        try:
            time.sleep(DELAY)
            data, last_remaining = fetch_snapshot(api_key, iso_dt)
            (snap_dir / f"snapshot_{d}_{snap}.json").write_text(
                json.dumps({"fetched_at": iso_dt, "data": data}))
        except Exception as e:
            print(f"  {d} {snap}: {e} — skipping")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(to_fetch)} done  (remaining: {last_remaining})")

    print(f"\nFetch complete. Credits remaining: {last_remaining}")

    # Build summary CSV
    rows = []
    for game_date in sorted(by_date.keys()):
        def _load(snap: str) -> dict:
            p = snap_dir / f"snapshot_{game_date}_{snap}.json"
            if not p.exists(): return {}
            body = json.loads(p.read_text())
            return {(MLB_TEAM_MAP.get(g.get("home_team","")),
                     MLB_TEAM_MAP.get(g.get("away_team",""))): g
                    for g in body.get("data", [])
                    if MLB_TEAM_MAP.get(g.get("home_team","")) and
                       MLB_TEAM_MAP.get(g.get("away_team",""))}

        open_idx  = _load("opening")
        close_idx = _load("closing")
        for game in by_date[game_date]:
            key = (game["home_abbr"], game["away_abbr"])
            open_ou  = extract_total(open_idx.get(key, {}))
            close_ou = extract_total(close_idx.get(key, {}))
            rows.append({
                "date":     game_date,
                "game_pk":  game["gamePk"],
                "home":     game["home_abbr"],
                "away":     game["away_abbr"],
                "open_ou":  f"{open_ou:.1f}"  if open_ou  is not None else "",
                "close_ou": f"{close_ou:.1f}" if close_ou is not None else "",
            })

    out = BASE / "cache" / f"totals_history_{s}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date","game_pk","home","away","open_ou","close_ou"])
        writer.writeheader()
        writer.writerows(rows)

    both = sum(1 for r in rows if r["open_ou"] and r["close_ou"])
    print(f"Wrote {len(rows)} rows → {out}  ({both} with both lines)")


if __name__ == "__main__":
    main()
