"""
evaluate.py — Score MLB model against run line odds

Usage:
    python3 evaluate.py --season 2025
    python3 evaluate.py --season 2024          # out-of-sample validation
    python3 evaluate.py --season 2025 --by-month
    python3 evaluate.py --season 2025 --worst 20
"""

import argparse
import csv
import importlib
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))


def load_training_data(season: int) -> list[dict]:
    p = BASE / "cache" / f"training_data_{season}.csv"
    if not p.exists():
        print(f"Run build_training_dataset.py --season {season} first.")
        sys.exit(1)
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def predict(row: dict, mp) -> float:
    roll10 = row.get("roll10_diff", "")
    roll10_val = float(roll10) if roll10 != "" else 0.0
    sp_fip = row.get("sp_fip_diff", "")
    sp_fip_val = float(sp_fip) if sp_fip != "" else 0.0
    return (
        mp.srs_weight  * float(row["srs_diff"])
        + mp.era_weight  * float(row["era_diff"])
        + mp.rest_weight * float(row["rest_diff"])
        + getattr(mp, "park_weight",       0.0) * float(row.get("park_adj", 0))
        + getattr(mp, "roll10_weight",     0.0) * roll10_val
        + getattr(mp, "sp_fip_diff_weight", 0.0) * sp_fip_val
        + mp.hca
        + mp.intercept
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",   type=int, default=2025)
    parser.add_argument("--worst",    type=int, default=0)
    parser.add_argument("--by-month", action="store_true")
    args = parser.parse_args()

    import model_params as mp
    importlib.reload(mp)

    rows    = load_training_data(args.season)
    results = []
    skipped = 0

    for r in rows:
        if not r.get("open_rl"):
            skipped += 1
            continue
        pred  = predict(r, mp)
        actual = float(r["run_diff"])
        error  = pred - actual
        results.append({**r, "pred": pred, "error": error, "abs_error": abs(error)})

    if not results:
        print("No games with run line data.")
        return

    mae      = sum(r["abs_error"] for r in results) / len(results)
    line_mae = sum(abs(float(r["open_rl"]) - float(r["run_diff"])) for r in results) / len(results)

    label = "TRAIN" if args.season == 2025 else "HOLDOUT"
    print(f"\n[{label}] Season {args.season} — {len(results)} games (skipped {skipped})")
    print(f"Model MAE       : {mae:.4f}")
    print(f"Run line MAE    : {line_mae:.4f}  (market baseline)")
    print(f"Delta vs market : {mae - line_mae:+.4f}")
    print(f"\nWeights: srs={mp.srs_weight}, era={mp.era_weight}, rest={mp.rest_weight}, "
          f"park={getattr(mp,'park_weight',0.0)}, hca={mp.hca}")

    if args.by_month:
        from collections import defaultdict
        by_month: dict[str, list] = defaultdict(list)
        for r in results:
            by_month[r["date"][:7]].append(r["abs_error"])
        print(f"\n{'Month':<10} {'N':>5} {'MAE':>7}")
        for m in sorted(by_month):
            errs = by_month[m]
            print(f"{m:<10} {len(errs):>5} {sum(errs)/len(errs):>7.4f}")

    if args.worst:
        worst = sorted(results, key=lambda r: -r["abs_error"])[:args.worst]
        print(f"\n{'Date':12} {'Game':14} {'Pred':>6} {'Line':>6} {'Actual':>7} {'Err':>6}")
        for r in worst:
            print(f"{r['date']:12} {r['away']:>3} @ {r['home']:<3}   "
                  f"{r['pred']:>+6.2f}  {float(r['open_rl']):>+6.2f}  "
                  f"{float(r['run_diff']):>+7.1f}  {r['error']:>+6.2f}")


if __name__ == "__main__":
    main()
