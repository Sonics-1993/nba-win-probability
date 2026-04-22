# MLB Totals Autoresearch — Agent Instructions

Adapted from Karpathy's autoresearch framework.
You are an autonomous research agent improving the MLB run-totals prediction model.

---

## Goal

Minimize `train_delta`. Keep if train_delta improves (lower than −0.0202).
Apply holdout guard: revert if holdout worsens >0.005 with only marginal train gain (<0.0003).

**Current state (end of session 3, park2_weight=10.0 active):**
- TRAIN   N=1602  Delta=**−0.0202**
- HOLDOUT N= 781  Delta=**−0.0115**  ← new all-time best since run 11

---

## Current weights (experiment.py)

```
sp_weight      = 0.42   bp_weight      = 0.33   park_weight    = 2.0
park2_weight   = 10.0   offense_weight = 0.19   fatigue_weight = -0.16
era_fip_div_w  = 0.25   gap_weight     = -0.35  model_blend    = 0.34
FIP_CAP        = 5.25   BP_ERA_CAP     = 5.75   fatigue_center = 11.62
ace asymmetry  = 1.7×ace / 0.3×weaker           intercept      = 0.00
```

park2_adj = `(park_factor - 1.0)**2 * park2_weight` (added to predict() alongside linear park_adj)

---

## Files

| File | Role | Modify? |
|---|---|---|
| `experiment.py` | Predict function + weights | **YES — only this file** |
| `run_experiment.py` | Evaluation runner | NO |
| `program.md` | These instructions | NO |
| `results.tsv` | Experiment log | NO |
| `cache/totals_training_2025.csv` | Training data | NO |

---

## The Loop

1. Read program.md, results.tsv, current experiment.py.
2. Pick ONE untested hypothesis.
3. Comment + modify experiment.py. Commit.
4. `python3 run_experiment.py`
5. Keep if improved; else `git revert HEAD --no-edit` + verify baseline.
6. Repeat for 15–20 experiments.

---

## Ideas to Try (session 4)

**High priority — follow session 3 momentum**
- `fatigue_weight = -0.17`: session 3 trend showed signal still active through -0.16.
- `park2_weight = 11.0`: test between 10.0 (kept) and 12.0 (reverted); optimum may sit here.
- `park_weight` reduction: with park2=10.0 handling extremes, linear park_weight may be redundant. Try 1.8, 1.6, 1.5 — if large park2 now covers the variance, linear term could be reduced.
- `bp_weight` around 0.33: try 0.31, 0.32, 0.34 — session 3 showed 0.33 < 0.35; fine-tune.
- `sp_weight` around 0.42: try 0.43, 0.44 — session 3 ceiling was 0.42 (0.43 reverted train), but with new context post-park2 it may read differently.
- `model_blend` around 0.34: try 0.32, 0.33 — session 3 showed 0.33 untried between 0.34 and previous reversal.

**Error-analysis findings — new hypotheses**
- Outdoor intercept: error analysis shows outdoor games are under-predicted by -0.74 runs vs dome -0.11. Try adding `outdoor_intercept * is_outdoor` (a flat positive constant for outdoor games). Try 0.3, 0.5, 0.7. This is distinct from weather signals (ablated) — it's a run-environment floor effect.
- era_fip_div_w = 0.26, 0.27: session 3 showed 0.26 reverted but the direction wasn't exhausted; re-test in new context with park2 active.
- FIP_CAP fine-tune: try 5.1, 5.15, 5.3 — 5.0 and 5.25 tested; narrow the band.
- BP_ERA_CAP fine-tune: try 5.6, 5.65, 5.85.

**Joint search — after individual wins stabilize**
- sp_weight × bp_weight with new optima.
- fatigue_weight × era_fip_div_w joint scan.

---

## Ruled Out — Do Not Re-test

- OPS, SRS, fip_blend>0, cumulative ERA-FIP divergence, separate home/away weights
- Line movement, weather (temp/wind), ERA instead of FIP, per-team intercepts
- model_blend ≥ 0.35, model_blend ≤ 0.33 (re-test 0.33 cautiously — session 3 didn't test it post-park2)
- sp_weight 0.50, 0.38, 0.40 (was 0.42 ceiling in session 3)
- bp_weight 0.30, 0.35, 0.37, 0.40
- park_weight 2.2 (tested earlier); park2_weight 12.0 (reverted)
- era_fip_div_w ≤ 0.05 or ≥ 0.30 (re-test 0.26-0.27 cautiously with new context)
- gap_weight beyond -0.35 or less aggressive than -0.35
- ace asymmetry beyond 1.7× or below 1.5×
- offense_weight 0.17, 0.20 (0.19 is optimum)
- fatigue_weight rollback to -0.14 or above
- FIP_CAP 5.0, 5.5+; BP_ERA_CAP 5.5, 6.0
- Global intercept (median residual near 0; not the same as outdoor_intercept which is conditional)

---

## Quick Reference

```bash
cat results.tsv
python3 run_experiment.py
git revert HEAD --no-edit
git log --oneline -10
```
