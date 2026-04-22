"""
run_experiment.py — Fixed evaluation runner.  DO NOT MODIFY.

Loads experiment.py, evaluates predict() on the 2025 totals dataset,
prints results, and appends a row to results.tsv.

Usage:  python run_experiment.py
"""

import csv
import importlib.util
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
RESULTS = BASE / "results.tsv"

# Baseline numbers from run 155 (autoresearch/apr21, bp_weight=0.29) — new floor to beat
BASELINE_TRAIN_DELTA   = -0.0222
BASELINE_HOLDOUT_DELTA = -0.0104


def load_experiment():
    spec = importlib.util.spec_from_file_location("experiment", BASE / "experiment.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_data(season: int = 2025) -> list[dict]:
    p = BASE / "cache" / f"totals_training_{season}.csv"
    if not p.exists():
        sys.exit(f"Missing {p} — run build_totals_dataset.py --season {season} first.")
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("open_ou")]
    for r in rows:
        r["open_ou"]    = float(r["open_ou"])
        r["total_runs"] = float(r["total_runs"])
    return rows


def score(subset: list[dict], predict) -> tuple[float, float, float]:
    pred_errs   = [abs(predict(r) - r["total_runs"]) for r in subset]
    market_errs = [abs(r["open_ou"] - r["total_runs"]) for r in subset]
    mae  = sum(pred_errs)   / len(pred_errs)
    mmae = sum(market_errs) / len(market_errs)
    return mae, mmae, mae - mmae


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BASE, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def next_run_number() -> int:
    if not RESULTS.exists():
        return 1
    with open(RESULTS) as f:
        data_lines = [l for l in f if l.strip() and not l.startswith("run\t")]
    if not data_lines:
        return 1
    last_run = int(data_lines[-1].split("\t")[0])
    return last_run + 1


def main():
    exp  = load_experiment()
    rows = load_data()

    train   = [r for r in rows if r["date"] < "2025-08"]
    holdout = [r for r in rows if r["date"] >= "2025-08"]

    t_mae, t_mkt, t_delta = score(train,   exp.predict)
    h_mae, h_mkt, h_delta = score(holdout, exp.predict)

    run = next_run_number()
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M")
    gh  = git_hash()

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Run {run}  |  {ts}  |  {gh}")
    print(f"{'─'*60}")
    print(f"TRAIN   N={len(train):4d}  Model={t_mae:.4f}  Market={t_mkt:.4f}  Delta={t_delta:+.4f}",
          f"  (baseline {BASELINE_TRAIN_DELTA:+.4f}  change {t_delta - BASELINE_TRAIN_DELTA:+.4f})")
    print(f"HOLDOUT N={len(holdout):4d}  Model={h_mae:.4f}  Market={h_mkt:.4f}  Delta={h_delta:+.4f}",
          f"  (baseline {BASELINE_HOLDOUT_DELTA:+.4f}  change {h_delta - BASELINE_HOLDOUT_DELTA:+.4f})")
    print(f"{'─'*60}")

    improved = t_delta < BASELINE_TRAIN_DELTA
    print(f"VERDICT: {'✓ KEEP — train improved' if improved else '✗ REVERT — git revert HEAD'}")
    print(f"{'─'*60}\n")

    # ── Append to results.tsv ──────────────────────────────────────────────────
    write_header = not RESULTS.exists() or RESULTS.stat().st_size == 0
    with open(RESULTS, "a") as f:
        if write_header:
            f.write("run\ttimestamp\ttrain_mae\ttrain_delta\tholdout_mae\tholdout_delta\tgit_hash\n")
        f.write(f"{run}\t{ts}\t{t_mae:.4f}\t{t_delta:+.4f}\t{h_mae:.4f}\t{h_delta:+.4f}\t{gh}\n")

    print(f"→ Appended run {run} to results.tsv")


if __name__ == "__main__":
    main()
