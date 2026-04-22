"""
build_prior.py — Generate preseason_wins_{season}.json for use as a blend prior
                 in build_mlb_srs.py.

Default: seed 2026 from 40% prior-year Vegas preseason + 60% prior-year final SRS,
         then apply mild regression to mean (shrink = 0.85).

Usage:
    python3 build_prior.py --season 2026
    python3 build_prior.py --season 2026 --halflife 40   # adjust blend decay speed
    python3 build_prior.py --season 2026 --print-only     # show without writing

Override individual teams by editing the output JSON directly.
Best practice: replace with actual Vegas opening-day win totals when available.
"""

import argparse
import csv
import json
from pathlib import Path

BASE  = Path(__file__).parent
CACHE = BASE / "cache"

RD_PER_WIN = 0.28
SHRINK     = 0.85   # regression to mean on the blended prior
PRE_WEIGHT = 0.40   # weight on prior-year preseason Vegas lines
ACT_WEIGHT = 0.60   # weight on prior-year actual final SRS


def final_wins_from_srs(season: int) -> dict[str, float]:
    """Convert end-of-season cum_srs to implied win pace."""
    f = CACHE / f"cum_srs_{season}.csv"
    if not f.exists():
        return {}
    rows = list(csv.DictReader(open(f)))
    latest: dict[str, tuple] = {}
    for r in rows:
        t = r["team"]
        if t not in latest or r["date"] > latest[t][0]:
            latest[t] = (r["date"], float(r["cum_srs"]))
    return {t: 81.0 + v / RD_PER_WIN for t, (_, v) in latest.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",     type=int,   default=2026)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    prior_season = args.season - 1

    # Load prior-year preseason wins (Vegas lines)
    pre_file = CACHE / f"preseason_wins_{prior_season}.json"
    if not pre_file.exists():
        print(f"Missing {pre_file.name} — cannot build prior.")
        return
    pre_wins: dict[str, float] = {
        k: v for k, v in json.loads(pre_file.read_text()).items()
        if not k.startswith("_")
    }

    # Load prior-year final actual wins (from SRS)
    act_wins = final_wins_from_srs(prior_season)
    if not act_wins:
        print(f"Missing cum_srs_{prior_season}.csv — using preseason only.")
        act_wins = pre_wins.copy()

    teams = sorted(set(pre_wins) | set(act_wins))
    result: dict[str, float] = {}

    print(f"\nBuilding {args.season} prior from {prior_season} preseason + {prior_season} actual")
    print(f"  Weights: {int(PRE_WEIGHT*100)}% Vegas preseason / {int(ACT_WEIGHT*100)}% actual results")
    print(f"  Regression to mean: shrink={SHRINK}")
    print(f"\n{'Team':<6} {'Pre'+str(prior_season):>8} {'Act'+str(prior_season):>8} {'→ Prior'+str(args.season):>12}")
    print("─" * 40)

    for team in teams:
        pw = pre_wins.get(team, 81.0)
        aw = act_wins.get(team, 81.0)
        raw = PRE_WEIGHT * pw + ACT_WEIGHT * aw
        shrunk = 81.0 + SHRINK * (raw - 81.0)
        result[team] = round(shrunk, 1)
        print(f"  {team:<5}  {pw:>6.1f}   {aw:>6.1f}   {shrunk:>8.1f}")

    output = {
        "_note": (
            f"{args.season} prior: {int(PRE_WEIGHT*100)}% {prior_season} Vegas preseason + "
            f"{int(ACT_WEIGHT*100)}% {prior_season} actual results, shrink={SHRINK}. "
            f"Override with actual {args.season} Vegas win totals for best results."
        ),
        **{t: result[t] for t in sorted(result, key=lambda x: -result[x])},
    }

    if args.print_only:
        print(json.dumps(output, indent=2))
        return

    out_file = CACHE / f"preseason_wins_{args.season}.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nWrote → {out_file.name}")
    print(f"Next: python3 build_mlb_srs.py --season {args.season}  (without --no-blend)")


if __name__ == "__main__":
    main()
