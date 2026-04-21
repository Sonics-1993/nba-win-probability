"""
evaluate_totals.py — Score totals model vs market O/U line

Train: 2025-03 through 2025-07  |  Holdout: 2025-08 through 2025-09

Usage:
    python3 evaluate_totals.py
    python3 evaluate_totals.py --by-month
    python3 evaluate_totals.py --worst 20
    python3 evaluate_totals.py --grid-search
"""

import argparse
import csv
import importlib
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

REPLACEMENT_ERA = 4.50
REPLACEMENT_FIP = 4.40


def load_data(season: int = 2025) -> list[dict]:
    p = BASE / "cache" / f"totals_training_{season}.csv"
    if not p.exists():
        print(f"Run build_totals_dataset.py --season {season} first."); sys.exit(1)
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def predict(row: dict, mp) -> float:
    def f(col): return float(row[col]) if row.get(col, "") not in ("", None) else 0.0
    def fb(col, fallback): return float(row[col]) if row.get(col, "") not in ("", None) else fallback

    outdoor = f("is_outdoor") > 0

    # Starter quality: blend cumulative FIP with last-3 FIP for recency
    fip_blend  = getattr(mp, "fip_blend", 0.0)   # weight on last-3 vs cumulative
    home_sp = fb("home_sp_fip", REPLACEMENT_FIP) * (1 - fip_blend) + \
              fb("home_l3_fip", REPLACEMENT_FIP) * fip_blend
    away_sp = fb("away_sp_fip", REPLACEMENT_FIP) * (1 - fip_blend) + \
              fb("away_l3_fip", REPLACEMENT_FIP) * fip_blend

    sp_runs  = (home_sp + away_sp)                        * getattr(mp, "sp_weight",      0.50)
    bp_runs  = (f("home_bp_era") + f("away_bp_era"))      * getattr(mp, "bp_weight",      0.30)
    park_adj = (f("park_factor") - 1.0)                   * getattr(mp, "park_weight",    2.0)
    # Weather only applies to outdoor games
    temp_adj = ((f("temp_f") - 72.0) * getattr(mp, "temp_weight", 0.0)) if outdoor else 0.0
    wind_adj = (f("tailwind_mph")     * getattr(mp, "wind_weight", 0.0)) if outdoor else 0.0
    off_adj  = (f("home_roll10") + f("away_roll10"))       * getattr(mp, "offense_weight", 0.15)
    srs_adj  = (f("home_srs")    + f("away_srs"))          * getattr(mp, "srs_weight",     0.15)
    ops_adj  = (fb("home_ops", 0.720) + fb("away_ops", 0.720) - 1.440) \
                                                            * getattr(mp, "ops_weight",     0.0)
    # Bullpen fatigue: deviation from training-mean combined 3-day BP IP (11.62)
    fat_adj  = (f("home_bp_ip_3d") + f("away_bp_ip_3d") - getattr(mp, "fatigue_center", 11.62)) \
                                                            * getattr(mp, "fatigue_weight", 0.0)

    return (sp_runs + bp_runs + park_adj + temp_adj + wind_adj
            + off_adj + srs_adj + ops_adj + fat_adj + getattr(mp, "intercept", 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--by-month",    action="store_true")
    parser.add_argument("--worst",       type=int, default=0)
    parser.add_argument("--grid-search", action="store_true")
    args = parser.parse_args()

    import totals_params as mp
    importlib.reload(mp)

    rows = [r for r in load_data() if r.get("open_ou")]
    for r in rows:
        r["open_ou"]    = float(r["open_ou"])
        r["total_runs"] = float(r["total_runs"])

    train   = [r for r in rows if r["date"] < "2025-08"]
    holdout = [r for r in rows if r["date"] >= "2025-08"]

    def score(subset, label):
        if not subset: return
        pred_errors   = [abs(predict(r, mp) - r["total_runs"]) for r in subset]
        market_errors = [abs(r["open_ou"] - r["total_runs"]) for r in subset]
        mae  = sum(pred_errors)   / len(pred_errors)
        mmae = sum(market_errors) / len(market_errors)
        print(f"[{label}] N={len(subset)}  Model MAE={mae:.4f}  Market MAE={mmae:.4f}  Delta={mae-mmae:+.4f}")

    score(train,   "TRAIN   Mar-Jul")
    score(holdout, "HOLDOUT Aug-Sep")

    if args.by_month:
        by_month: dict[str, list] = defaultdict(list)
        for r in rows:
            by_month[r["date"][:7]].append(r)
        print(f"\n{'Month':10} {'N':>5} {'MAE':>7} {'Mkt':>7} {'Delta':>7}")
        for m in sorted(by_month):
            sub  = by_month[m]
            mae  = sum(abs(predict(r, mp) - r["total_runs"]) for r in sub) / len(sub)
            mmae = sum(abs(r["open_ou"]   - r["total_runs"]) for r in sub) / len(sub)
            print(f"{m:10} {len(sub):>5} {mae:>7.4f} {mmae:>7.4f} {mae-mmae:>+7.4f}")

    if args.worst:
        results = sorted(rows, key=lambda r: -abs(predict(r, mp) - r["total_runs"]))[:args.worst]
        print(f"\n{'Date':12} {'Game':14} {'Pred':>6} {'Line':>6} {'Act':>5} {'Err':>6}")
        for r in results:
            p = predict(r, mp)
            print(f"{r['date']:12} {r['away']:>3}@{r['home']:<3}  "
                  f"{p:>6.1f}  {r['open_ou']:>6.1f}  {r['total_runs']:>5.0f}  {p - r['total_runs']:>+6.1f}")

    if args.grid_search:
        import numpy as np
        print("\n--- Grid search (on train set) ---")
        best = (9999, {})

        def arr(col, fallback=0.0):
            return np.array([float(r[col]) if r.get(col, "") not in ("", None) else fallback
                             for r in train])

        acts    = arr("total_runs")
        mkt     = arr("open_ou")
        outdoor = arr("is_outdoor")
        sp_cum  = arr("home_sp_fip", REPLACEMENT_FIP) + arr("away_sp_fip", REPLACEMENT_FIP)
        sp_l3   = arr("home_l3_fip", REPLACEMENT_FIP) + arr("away_l3_fip", REPLACEMENT_FIP)
        bp_e    = arr("home_bp_era") + arr("away_bp_era")
        pf      = arr("park_factor") - 1.0
        tf      = arr("temp_f", 72.0)
        tw      = arr("tailwind_mph")
        off     = np.array([(float(r["home_roll10"]) + float(r["away_roll10"]))
                             if r.get("home_roll10") and r.get("away_roll10") else 0.0
                             for r in train])
        srs     = arr("home_srs") + arr("away_srs")
        ops     = (arr("home_ops", 0.720) + arr("away_ops", 0.720)) - 1.440
        mkt_mae = np.mean(np.abs(mkt - acts))

        for fip_blend in [0.0, 0.3, 0.5, 0.7, 1.0]:
            sp_e = sp_cum * (1 - fip_blend) + sp_l3 * fip_blend
            for sp_w in [0.45, 0.50, 0.55, 0.60]:
                for bp_w in [0.25, 0.30, 0.35]:
                    for park_w in [1.5, 2.0, 2.5, 3.0]:
                        for temp_w in [-0.03, -0.02, -0.01, 0.0]:
                            for wind_w in [0.0, 0.03, 0.06]:
                                for off_w in [0.0, 0.10, 0.15, 0.20]:
                                    for srs_w in [0.10, 0.15, 0.20]:
                                        for ops_w in [0.0, 0.5, 1.0, 1.5]:
                                            for intercept in [-0.5, 0.0, 0.5]:
                                                pred = (sp_e * sp_w + bp_e * bp_w
                                                        + park_w * pf
                                                        + temp_w * (tf - 72) * outdoor
                                                        + wind_w * tw * outdoor
                                                        + off_w * off + srs_w * srs
                                                        + ops_w * ops + intercept)
                                                mae = np.mean(np.abs(pred - acts))
                                                if mae < best[0]:
                                                    best = (mae, dict(
                                                        fip_blend=fip_blend,
                                                        sp_w=sp_w, bp_w=bp_w, park_w=park_w,
                                                        temp_w=temp_w, wind_w=wind_w,
                                                        off_w=off_w, srs_w=srs_w,
                                                        ops_w=ops_w, intercept=intercept))

        print(f"Best train MAE: {best[0]:.4f}  (market: {mkt_mae:.4f}  delta: {best[0]-mkt_mae:+.4f})")
        print(f"Weights: {best[1]}")

        # Verify best weights on holdout
        bw = best[1]
        fip_blend = bw["fip_blend"]
        def harr(col, fallback=0.0):
            return np.array([float(r[col]) if r.get(col, "") not in ("", None) else fallback
                             for r in holdout])
        h_acts    = harr("total_runs");  h_mkt = harr("open_ou")
        h_sp_e    = (harr("home_sp_fip", REPLACEMENT_FIP) + harr("away_sp_fip", REPLACEMENT_FIP)) * (1-fip_blend) + \
                    (harr("home_l3_fip", REPLACEMENT_FIP) + harr("away_l3_fip", REPLACEMENT_FIP)) * fip_blend
        h_bp_e    = harr("home_bp_era") + harr("away_bp_era")
        h_pf      = harr("park_factor") - 1.0
        h_tf      = harr("temp_f", 72.0); h_tw = harr("tailwind_mph")
        h_outdoor = harr("is_outdoor")
        h_off     = np.array([(float(r["home_roll10"])+float(r["away_roll10"]))
                               if r.get("home_roll10") and r.get("away_roll10") else 0.0
                               for r in holdout])
        h_srs     = harr("home_srs") + harr("away_srs")
        h_ops     = (harr("home_ops", 0.720) + harr("away_ops", 0.720)) - 1.440
        h_pred    = (h_sp_e * bw["sp_w"] + h_bp_e * bw["bp_w"]
                     + bw["park_w"] * h_pf
                     + bw["temp_w"] * (h_tf - 72) * h_outdoor
                     + bw["wind_w"] * h_tw * h_outdoor
                     + bw["off_w"] * h_off + bw["srs_w"] * h_srs
                     + bw["ops_w"] * h_ops + bw["intercept"])
        h_mae     = np.mean(np.abs(h_pred - h_acts))
        h_mkt_mae = np.mean(np.abs(h_mkt  - h_acts))
        print(f"Holdout check: MAE={h_mae:.4f}  Market={h_mkt_mae:.4f}  Delta={h_mae-h_mkt_mae:+.4f}")


if __name__ == "__main__":
    main()
