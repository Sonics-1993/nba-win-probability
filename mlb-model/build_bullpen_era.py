"""
build_bullpen_era.py — Cumulative bullpen ERA per team per game (no leakage)

Derives bullpen IP/ER from boxscore starter stats and game runs allowed:
  bullpen_er  = total_runs_allowed - starter_er    (runs allowed proxy)
  bullpen_ip  = assumed_innings - starter_ip       (9 for standard game)

Shrinks toward replacement level (same as starter ERA logic).

Usage:
    python3 build_bullpen_era.py --season 2025
    python3 build_bullpen_era.py --season 2024
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

BASE             = Path(__file__).parent
CACHE            = BASE / "cache"
REPLACEMENT_ERA  = 4.80   # league bullpen avg slightly higher than starter avg
ASSUMED_INNINGS  = 9.0
SHRINK_IP        = 30.0   # more shrinkage than starters — bullpen ERA noisier per game


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

    # Build per-game bullpen outings: (date, team_abbr, bp_ip, bp_er)
    outings: dict[str, list[tuple]] = defaultdict(list)  # team_abbr → [(date, ip, er)]

    for g in sorted(games, key=lambda x: x["date"]):
        pk_str = str(g["gamePk"])
        sp     = starts.get(pk_str, {})

        for side, abbr, opp_runs in [
            ("home_sp", g["home_abbr"], g["away_runs"]),
            ("away_sp", g["away_abbr"], g["home_runs"]),
        ]:
            entry      = sp.get(side, {})
            starter_ip = float(entry.get("ip", 0.0))
            starter_er = float(entry.get("er", 0))
            total_ra   = float(opp_runs)

            bp_ip = max(0.0, ASSUMED_INNINGS - starter_ip)
            bp_er = max(0.0, total_ra - starter_er)
            outings[abbr].append((g["date"], g["gamePk"], bp_ip, bp_er))

    # Build cumulative shrunk bullpen ERA per (game_pk, side), no leakage
    rows = []
    for g in sorted(games, key=lambda x: (x["date"], x["gamePk"])):
        for side, abbr in [("home", g["home_abbr"]), ("away", g["away_abbr"])]:
            history    = [(d, ip, er) for d, pk, ip, er in outings[abbr]
                          if d < g["date"] or (d == g["date"] and pk < g["gamePk"])]
            total_ip   = sum(ip for _, ip, _ in history)
            total_er   = sum(er for _, _, er in history)
            shrunk_era = ((total_er * 9 + REPLACEMENT_ERA * SHRINK_IP)
                          / (total_ip + SHRINK_IP)) if total_ip >= 0 else REPLACEMENT_ERA
            rows.append({
                "date":         g["date"],
                "game_pk":      g["gamePk"],
                "side":         side,
                "cum_bp_era":   round(shrunk_era, 3),
                "prior_bp_ip":  round(total_ip, 1),
                "prior_games":  len(history),
            })

    out = CACHE / f"bullpen_era_{s}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date", "game_pk", "side", "cum_bp_era", "prior_bp_ip", "prior_games"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {out}")
    sample = [r for r in rows if r["prior_games"] >= 10]
    if sample:
        avg = sum(r["cum_bp_era"] for r in sample) / len(sample)
        print(f"Sanity: avg shrunk bullpen ERA (≥10 games) = {avg:.2f}  (expect ~4.2-4.6)")


if __name__ == "__main__":
    main()
