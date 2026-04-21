"""
build_bullpen_fatigue.py — Rolling bullpen IP workload per team (last 3 calendar days)

Bullpen IP per game is approximated as max(0, 9 - starter_ip).
Extra-inning games are not captured, but early starter exits (high BP usage)
and back-to-back heavy workloads are the primary fatigue signal.

Usage:
    python3 build_bullpen_fatigue.py --season 2025
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

BASE  = Path(__file__).parent
CACHE = BASE / "cache"
ASSUMED_INNINGS = 9.0
FATIGUE_DAYS    = 3   # look-back window in calendar days


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    s = args.season

    games_file  = CACHE / f"games_{s}_raw.json"
    starts_file = CACHE / f"pitcher_starts_{s}.json"
    for f, script in [(games_file, "fetch_mlb_games.py"),
                      (starts_file, "fetch_pitcher_stats.py")]:
        if not f.exists():
            print(f"Missing {f.name} — run {script} --season {s} first.")
            return

    games  = json.loads(games_file.read_text())
    starts = json.loads(starts_file.read_text())

    # Build per-team, per-date bullpen IP usage: {team: [(date_str, bp_ip)]}
    team_bp: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for g in games:
        pk_str = str(g["gamePk"])
        sp     = starts.get(pk_str, {})
        for side_key, abbr in [("home_sp", g["home_abbr"]), ("away_sp", g["away_abbr"])]:
            entry     = sp.get(side_key, {})
            starter_ip = float(entry.get("ip", 0.0) or 0.0)
            bp_ip      = max(0.0, ASSUMED_INNINGS - starter_ip)
            team_bp[abbr].append((g["date"], bp_ip))

    # For each game, sum bp_ip for each team over the prior FATIGUE_DAYS calendar days
    rows = []
    for g in sorted(games, key=lambda x: (x["date"], x["gamePk"])):
        game_date = date.fromisoformat(g["date"])
        cutoff    = str(game_date - timedelta(days=FATIGUE_DAYS))

        def rolling_bp(abbr: str) -> float:
            return sum(ip for d, ip in team_bp[abbr]
                       if cutoff < d < g["date"])

        rows.append({
            "date":          g["date"],
            "game_pk":       g["gamePk"],
            "home":          g["home_abbr"],
            "away":          g["away_abbr"],
            "home_bp_ip_3d": round(rolling_bp(g["home_abbr"]), 2),
            "away_bp_ip_3d": round(rolling_bp(g["away_abbr"]), 2),
        })

    out = CACHE / f"bullpen_fatigue_{s}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date","game_pk","home","away",
                                                "home_bp_ip_3d","away_bp_ip_3d"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {out}")

    # Sanity: show distribution of combined 3-day BP IP
    combined = [r["home_bp_ip_3d"] + r["away_bp_ip_3d"] for r in rows if r["home_bp_ip_3d"] > 0]
    if combined:
        avg = sum(combined) / len(combined)
        mx  = max(combined)
        print(f"Sanity: avg combined 3-day BP IP = {avg:.2f}  max = {mx:.2f}")
        high = sum(1 for v in combined if v > 12)
        print(f"  Games with combined >12 IP (fatigued): {high} ({high/len(combined):.1%})")


if __name__ == "__main__":
    main()
