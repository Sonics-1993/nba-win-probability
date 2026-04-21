"""
build_rolling_ops.py — Rolling 10-game team OPS (no leakage)

Reads team_batting_{season}.json (written by fetch_pitcher_stats.py).
Writes rolling_team_ops_{season}.csv with columns:
  game_pk, date, team, rolling_ops, rolling_obp, rolling_slg

OBP = (H + BB) / (AB + BB)
SLG = TB / AB
OPS = OBP + SLG

Usage:
    python3 build_rolling_ops.py --season 2025
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

BASE  = Path(__file__).parent
CACHE = BASE / "cache"
WINDOW = 10
MIN_GAMES = 3
LEAGUE_OPS = 0.720   # 2025 MLB average OPS (used as prior / replacement)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    s = args.season

    batting_file = CACHE / f"team_batting_{s}.json"
    games_file   = CACHE / f"games_{s}_raw.json"
    if not batting_file.exists():
        print(f"Missing {batting_file} — run fetch_pitcher_stats.py --season {s} --force first")
        return

    batting = json.loads(batting_file.read_text())
    games   = json.loads(games_file.read_text())

    # Build per-team game log sorted by date
    # Each entry: (date, game_pk, ab, h, bb, hr, tb)
    team_log: dict[str, list] = defaultdict(list)
    for g in sorted(games, key=lambda x: (x["date"], x["gamePk"])):
        pk = str(g["gamePk"])
        bt = batting.get(pk, {})
        for side, abbr in [("home", g["home_abbr"]), ("away", g["away_abbr"])]:
            line = bt.get(side, {})
            if line:
                team_log[abbr].append((
                    g["date"], g["gamePk"],
                    line.get("ab", 0), line.get("h",  0),
                    line.get("bb", 0), line.get("hr", 0),
                    line.get("tb", 0),
                ))

    def ops_from(games_slice):
        ab = sum(x[2] for x in games_slice)
        h  = sum(x[3] for x in games_slice)
        bb = sum(x[4] for x in games_slice)
        tb = sum(x[6] for x in games_slice)
        obp = (h + bb) / (ab + bb) if (ab + bb) > 0 else 0.0
        slg = tb / ab               if ab > 0        else 0.0
        return round(obp + slg, 4), round(obp, 4), round(slg, 4)

    rows = []
    for g in sorted(games, key=lambda x: (x["date"], x["gamePk"])):
        pk = g["gamePk"]
        for side, abbr in [("home", g["home_abbr"]), ("away", g["away_abbr"])]:
            history = [(d, gpk, ab, h, bb, hr, tb)
                       for d, gpk, ab, h, bb, hr, tb in team_log[abbr]
                       if d < g["date"] or (d == g["date"] and gpk < pk)]
            window = history[-WINDOW:]
            if len(window) >= MIN_GAMES:
                roll_ops, roll_obp, roll_slg = ops_from(window)
            else:
                roll_ops, roll_obp, roll_slg = LEAGUE_OPS, 0.315, 0.405

            rows.append({
                "game_pk":    pk,
                "date":       g["date"],
                "team":       abbr,
                "rolling_ops": roll_ops,
                "rolling_obp": roll_obp,
                "rolling_slg": roll_slg,
            })

    out = CACHE / f"rolling_team_ops_{s}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game_pk","date","team","rolling_ops","rolling_obp","rolling_slg"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {out}")

    # Sanity check
    sample = [r for r in rows if r["rolling_ops"] != LEAGUE_OPS]
    if sample:
        avg = sum(r["rolling_ops"] for r in sample) / len(sample)
        print(f"Sanity: avg rolling OPS (non-replacement) = {avg:.3f}")


if __name__ == "__main__":
    main()
