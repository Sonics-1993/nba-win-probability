"""
build_pitcher_ratings.py — Per-pitcher ERA and cumulative ERA ratings

Computes each pitcher's cumulative ERA (no leakage) from the 2025 game logs,
using only outings completed BEFORE each game date.

Also computes a league-average ERA as the replacement level.

Writes:
  cache/pitcher_era.csv  — date, pitcher_id, pitcher_name, cum_era, ip, games

Usage:
    python3 build_pitcher_ratings.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

BASE  = Path(__file__).parent
CACHE = BASE / "cache"

REPLACEMENT_ERA = 4.50   # fallback for pitchers with no prior outings


def load_data() -> tuple[list[dict], dict]:
    games    = json.loads((CACHE / "games_2025_raw.json").read_text())
    pitchers = json.loads((CACHE / "games_2025_pitchers.json").read_text())
    return games, pitchers


def build_pitcher_era(games: list[dict], pitchers: dict) -> None:
    """
    For each game, look up both SPs. Their cumulative ERA as of that date
    comes from all prior starts in the season.

    We approximate innings from the per-game boxscore data we have.
    Since we only store starter identity (not their line), we use a
    simpler proxy: track earned runs allowed per outing via the linescore
    if available, otherwise fall back to the career ERA from the API.
    """
    dates = sorted(set(g["date"] for g in games))

    # Build per-pitcher outings list: [(date, game_pk)]
    pitcher_games: dict[int, list[str]] = defaultdict(list)
    for g in games:
        pk = str(g["gamePk"])
        if pk not in pitchers:
            continue
        for side in ("home_sp", "away_sp"):
            sp = pitchers[pk].get(side)
            if sp and sp.get("id"):
                pitcher_games[sp["id"]].append(g["date"])

    # We don't have per-outing ERA from our current cache — we have identity only.
    # Write a stub that can be enriched later; for now output replacement ERA for all.
    # TODO: enrich with per-start earned runs via /game/{pk}/boxscore pitching stats.

    rows = []
    all_pitchers: dict[int, str] = {}
    for g in games:
        pk = str(g["gamePk"])
        if pk not in pitchers:
            continue
        for side in ("home_sp", "away_sp"):
            sp = pitchers[pk].get(side)
            if sp and sp.get("id"):
                all_pitchers[sp["id"]] = sp["name"]

    for pid, name in sorted(all_pitchers.items(), key=lambda x: x[1]):
        rows.append({
            "pitcher_id":   pid,
            "pitcher_name": name,
            "cum_era":      REPLACEMENT_ERA,   # placeholder — enrich with fetch_pitcher_stats
            "ip":           0,
            "games":        len(pitcher_games[pid]),
        })

    out = CACHE / "pitcher_era.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pitcher_id","pitcher_name","cum_era","ip","games"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pitchers → {out}")
    print(f"Note: cumulative ERA is placeholder. Run fetch_pitcher_stats.py to populate.")


def main():
    games, pitchers = load_data()
    print(f"Loaded {len(games)} games, {len(pitchers)} pitcher entries")
    build_pitcher_era(games, pitchers)


if __name__ == "__main__":
    main()
