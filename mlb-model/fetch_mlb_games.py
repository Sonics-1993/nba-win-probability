"""
fetch_mlb_games.py — Pull MLB regular season game results

Usage:
    python3 fetch_mlb_games.py              # defaults to 2025
    python3 fetch_mlb_games.py --season 2024
    python3 fetch_mlb_games.py --force
"""

import argparse
import json
import time
from pathlib import Path

import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"
CACHE    = Path(__file__).parent / "cache"


def fetch_team_map(season: int) -> dict[int, str]:
    r = requests.get(f"{BASE_URL}/teams?sportId=1&season={season}", timeout=15)
    r.raise_for_status()
    return {t["id"]: t["abbreviation"] for t in r.json()["teams"]}


def fetch_schedule(season: int) -> list[dict]:
    url = (f"{BASE_URL}/schedule?sportId=1&season={season}&gameType=R"
           f"&startDate={season}-03-01&endDate={season}-11-01"
           f"&hydrate=linescore")
    print(f"Fetching {season} schedule...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    games = [g for d in r.json().get("dates", []) for g in d.get("games", [])]
    print(f"  {len(games)} total games in schedule")
    return games


def fetch_boxscore(game_pk: int) -> dict | None:
    try:
        r = requests.get(f"{BASE_URL}/game/{game_pk}/boxscore", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  Warning: boxscore failed for {game_pk}: {e}")
        return None


def extract_starting_pitcher(bs: dict, side: str) -> dict | None:
    team = bs.get("teams", {}).get(side, {})
    pitcher_ids = team.get("pitchers", [])
    if not pitcher_ids:
        return None
    pid_key = f"ID{pitcher_ids[0]}"
    player = team.get("players", {}).get(pid_key, {})
    return {"id": pitcher_ids[0], "name": player.get("person", {}).get("fullName", "Unknown")}


def build_games(season: int, force: bool = False) -> list[dict]:
    out_games    = CACHE / f"games_{season}_raw.json"
    out_pitchers = CACHE / f"games_{season}_pitchers.json"

    if out_games.exists() and out_pitchers.exists() and not force:
        games = json.loads(out_games.read_text())
        print(f"Cache hit: {out_games}  ({len(games)} games)")
        return games

    team_map = fetch_team_map(season)
    schedule = fetch_schedule(season)

    games_out, pitchers_out = [], {}

    for i, g in enumerate(schedule):
        if g.get("status", {}).get("abstractGameState") != "Final":
            continue

        game_pk  = g["gamePk"]
        date_str = g["gameDate"][:10]
        home     = g["teams"]["home"]["team"]
        away     = g["teams"]["away"]["team"]
        home["abbreviation"] = team_map.get(home["id"], home.get("abbreviation", ""))
        away["abbreviation"] = team_map.get(away["id"], away.get("abbreviation", ""))

        ls        = g.get("linescore", {})
        home_runs = ls.get("teams", {}).get("home", {}).get("runs")
        away_runs = ls.get("teams", {}).get("away", {}).get("runs")
        if home_runs is None or away_runs is None:
            continue

        games_out.append({
            "gamePk":    game_pk,
            "date":      date_str,
            "home_id":   home["id"],
            "home_abbr": home["abbreviation"],
            "home_name": home["name"],
            "away_id":   away["id"],
            "away_abbr": away["abbreviation"],
            "away_name": away["name"],
            "home_runs": int(home_runs),
            "away_runs": int(away_runs),
            "run_diff":  int(home_runs) - int(away_runs),
            "home_win":  int(home_runs) > int(away_runs),
            "venue":     g.get("venue", {}).get("name", ""),
            "status":    "Final",
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(schedule)} processed, {len(games_out)} final...")

        bs = fetch_boxscore(game_pk)
        if bs:
            pitchers_out[str(game_pk)] = {
                "home_sp": extract_starting_pitcher(bs, "home"),
                "away_sp": extract_starting_pitcher(bs, "away"),
            }
        time.sleep(0.05)

    out_games.write_text(json.dumps(games_out, indent=2))
    out_pitchers.write_text(json.dumps(pitchers_out, indent=2))
    print(f"\nWrote {len(games_out)} games → {out_games}")
    print(f"Wrote {len(pitchers_out)} pitcher entries → {out_pitchers}")
    return games_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()
    games = build_games(args.season, force=args.force)
    wins  = sum(1 for g in games if g["home_win"])
    print(f"\nSanity: {len(games)} games, home win rate = {wins/len(games):.3f} (expect ~0.54)")


if __name__ == "__main__":
    main()
