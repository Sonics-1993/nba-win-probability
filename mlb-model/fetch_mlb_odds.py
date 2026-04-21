"""
fetch_mlb_odds.py — Historical Run Line Odds Fetcher

Usage:
    python3 fetch_mlb_odds.py --season 2025 --dry-run
    python3 fetch_mlb_odds.py --season 2024
    python3 fetch_mlb_odds.py --season 2025 --summary-only

Requires ODDS_API_KEY in .env (shared with NBA project).
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
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

BASE      = Path(__file__).parent
ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT     = "baseball_mlb"
REGIONS   = "us"
MARKETS   = "spreads"
REQUEST_DELAY = 1.1

MLB_TEAM_MAP = {
    "Arizona Diamondbacks":    "AZ",
    "Atlanta Braves":          "ATL",
    "Baltimore Orioles":       "BAL",
    "Boston Red Sox":          "BOS",
    "Chicago Cubs":            "CHC",
    "Chicago White Sox":       "CWS",
    "Cincinnati Reds":         "CIN",
    "Cleveland Guardians":     "CLE",
    "Colorado Rockies":        "COL",
    "Detroit Tigers":          "DET",
    "Houston Astros":          "HOU",
    "Kansas City Royals":      "KC",
    "Los Angeles Angels":      "LAA",
    "Los Angeles Dodgers":     "LAD",
    "Miami Marlins":           "MIA",
    "Milwaukee Brewers":       "MIL",
    "Minnesota Twins":         "MIN",
    "New York Mets":           "NYM",
    "New York Yankees":        "NYY",
    "Athletics":               "ATH",
    "Oakland Athletics":       "ATH",
    "Sacramento Athletics":    "ATH",
    "Philadelphia Phillies":   "PHI",
    "Pittsburgh Pirates":      "PIT",
    "San Diego Padres":        "SD",
    "San Francisco Giants":    "SF",
    "Seattle Mariners":        "SEA",
    "St. Louis Cardinals":     "STL",
    "Tampa Bay Rays":          "TB",
    "Texas Rangers":           "TEX",
    "Toronto Blue Jays":       "TOR",
    "Washington Nationals":    "WSH",
}


def cache_dir(season: int) -> Path:
    return BASE / "cache" / f"odds_{season}"


def games_file(season: int) -> Path:
    return BASE / "cache" / f"games_{season}_raw.json"


def summary_file(season: int) -> Path:
    return BASE / "cache" / f"odds_history_{season}.csv"


def snapshot_times(game_date_str: str) -> tuple[str, str]:
    d = date.fromisoformat(game_date_str)
    opening = datetime(d.year, d.month, d.day, 14,  0, 0, tzinfo=timezone.utc)
    closing = datetime(d.year, d.month, d.day, 16, 30, 0, tzinfo=timezone.utc)
    return opening.strftime("%Y-%m-%dT%H:%M:%SZ"), closing.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_snapshot(api_key: str, iso_dt: str) -> tuple[list[dict], str, str]:
    url    = f"{ODDS_BASE}/historical/sports/{SPORT}/odds/"
    params = {"apiKey": api_key, "regions": REGIONS, "markets": MARKETS,
              "oddsFormat": "american", "date": iso_dt, "dateFormat": "iso"}
    resp = requests.get(url, params=params, timeout=20)
    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")
    if resp.status_code == 401:
        raise ValueError("API key rejected (401).")
    if resp.status_code == 422:
        raise ValueError(f"Unprocessable (422) for {iso_dt}.")
    resp.raise_for_status()
    return resp.json().get("data", []), remaining, used


def test_key(api_key: str, console: Console) -> bool:
    resp = requests.get(f"{ODDS_BASE}/sports/", params={"apiKey": api_key}, timeout=10)
    ok = resp.status_code == 200
    if ok:
        console.print(f"  Key OK — remaining={resp.headers.get('x-requests-remaining','?')}")
    else:
        console.print(f"  [red]Key check failed: HTTP {resp.status_code}[/red]")
    return ok


def extract_run_line(game: dict, home_abbr: str) -> tuple[float | None, int]:
    spreads = []
    for bm in game.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market["key"] != "spreads":
                continue
            for outcome in market.get("outcomes", []):
                if MLB_TEAM_MAP.get(outcome["name"]) == home_abbr:
                    pt = outcome.get("point")
                    if pt is not None:
                        spreads.append(-float(pt))
    if not spreads:
        return None, 0
    return float(np.median(spreads)), len(spreads)


def load_games_by_date(season: int) -> dict[str, list[dict]]:
    gf = games_file(season)
    if not gf.exists():
        return {}
    by_date: dict[str, list[dict]] = {}
    for g in json.loads(gf.read_text()):
        by_date.setdefault(g["date"], []).append(g)
    return by_date


def run_fetch(api_key: str, season: int, dates: list[str], force: bool,
              dry_run: bool, console: Console) -> None:
    cd = cache_dir(season)
    cd.mkdir(parents=True, exist_ok=True)

    to_fetch = [(d, s) for d in dates for s in ("opening", "closing")
                if force or not (cd / f"snapshot_{d}_{s}.json").exists()]

    cached = len(dates) * 2 - len(to_fetch)
    console.print(f"  Game dates  : [bold]{len(dates)}[/bold]")
    console.print(f"  To fetch    : [bold]{len(to_fetch)}[/bold]  (cached: {cached})")
    console.print(f"  Est. credits: [bold]~{len(to_fetch)*10}[/bold]")

    if dry_run:
        console.print("\n  [yellow]--dry-run: no API calls made.[/yellow]")
        return
    if not to_fetch:
        console.print("\n  [green]All snapshots cached.[/green]")
        return

    last_remaining = "?"
    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("Fetching…", total=len(to_fetch))
        for d, snap in to_fetch:
            open_dt, close_dt = snapshot_times(d)
            iso_dt = open_dt if snap == "opening" else close_dt
            progress.update(task, description=f"{d} {snap}")
            try:
                time.sleep(REQUEST_DELAY)
                data, remaining, _ = fetch_snapshot(api_key, iso_dt)
                last_remaining = remaining
                (cd / f"snapshot_{d}_{snap}.json").write_text(
                    json.dumps({"fetched_at": iso_dt, "data": data}))
            except ValueError as e:
                console.print(f"\n  [red]Fatal: {e}[/red]")
                return
            except Exception as e:
                console.print(f"\n  [yellow]{d} {snap}: {e} — skipping[/yellow]")
            progress.advance(task)

    console.print(f"\n  Credits remaining: [bold]{last_remaining}[/bold]")


def build_summary(season: int, by_date: dict[str, list[dict]], console: Console) -> None:
    cd = cache_dir(season)
    rows = []
    for game_date in sorted(by_date.keys()):
        def _load(snap: str) -> dict[tuple[str, str], dict]:
            p = cd / f"snapshot_{game_date}_{snap}.json"
            if not p.exists():
                return {}
            body = json.loads(p.read_text())
            return {(MLB_TEAM_MAP.get(g.get("home_team", "")),
                     MLB_TEAM_MAP.get(g.get("away_team", ""))): g
                    for g in body.get("data", [])
                    if MLB_TEAM_MAP.get(g.get("home_team", "")) and
                       MLB_TEAM_MAP.get(g.get("away_team", ""))}

        open_idx  = _load("opening")
        close_idx = _load("closing")
        if not open_idx and not close_idx:
            continue

        for game in by_date[game_date]:
            home, away = game["home_abbr"], game["away_abbr"]
            key = (home, away)
            open_rl,  open_n  = extract_run_line(open_idx.get(key,  {}), home)
            close_rl, close_n = extract_run_line(close_idx.get(key, {}), home)
            rows.append({
                "date":        game_date, "home": home, "away": away,
                "open_rl":     f"{open_rl:.2f}"  if open_rl  is not None else "",
                "open_books":  open_n,
                "close_rl":    f"{close_rl:.2f}" if close_rl is not None else "",
                "close_books": close_n,
            })

    if not rows:
        console.print("  [yellow]No cached snapshots found.[/yellow]")
        return

    sf = summary_file(season)
    with open(sf, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date","home","away","open_rl","open_books","close_rl","close_books"])
        writer.writeheader()
        writer.writerows(rows)

    both = sum(1 for r in rows if r["open_rl"] and r["close_rl"])
    console.print(f"  {len(rows)} games, {both} with both run lines → {sf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",      type=int, default=2025)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--date",        metavar="YYYY-MM-DD")
    parser.add_argument("--summary-only",action="store_true")
    args = parser.parse_args()

    load_dotenv(BASE.parent / "opening-lines" / ".env")
    load_dotenv(BASE / ".env")
    api_key = os.getenv("ODDS_API_KEY")

    console = Console()
    console.print(f"\n[bold]MLB {args.season} Run Line Odds Fetcher[/bold]\n")

    gf = games_file(args.season)
    if not gf.exists():
        console.print(f"[red]Run fetch_mlb_games.py --season {args.season} first.[/red]")
        return

    by_date   = load_games_by_date(args.season)
    all_dates = sorted(by_date.keys())
    dates     = [args.date] if args.date else all_dates

    console.print(f"  Season      : {all_dates[0]} → {all_dates[-1]}")
    console.print(f"  Game dates  : {len(all_dates)}")

    if not args.summary_only:
        if not api_key:
            console.print("[red]ODDS_API_KEY not found in .env[/red]")
            return
        if not args.dry_run:
            console.print("\nVerifying API key…")
            if not test_key(api_key, console):
                return
        run_fetch(api_key, args.season, dates, args.force, args.dry_run, console)

    console.print(f"\nBuilding odds_history_{args.season}.csv…")
    build_summary(args.season, by_date, console)


if __name__ == "__main__":
    main()
