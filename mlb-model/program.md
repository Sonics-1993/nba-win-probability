# MLB Totals Autoresearch — Agent Instructions

Adapted from Karpathy's autoresearch framework.
You are an autonomous research agent improving the MLB run-totals prediction model.

---

## Goal

Minimize `train_delta`. Keep if train_delta improves (lower than −0.0231).
Apply holdout guard: revert if holdout worsens >0.005 with only marginal train gain (<0.0003).

**Current state (end of session 6, run 176):**
- TRAIN   N=1602  Delta=**−0.0231**
- HOLDOUT N= 781  Delta=**−0.0104**

Holdout recovered from the session-5 divergence. Zero reverts in session 6 — model is
in a stable improvement region.

---

## Current weights (experiment.py)

```
sp_weight        = 0.42   bp_weight        = 0.27   park_weight      = -3.0
park2_weight     = 18.0   offense_weight   = 0.19   fatigue_weight   = -0.23
era_fip_div_w    = 0.30   gap_weight       = -0.35  model_blend      = 0.28
FIP_CAP          = 5.25   BP_ERA_CAP       = 5.75   fatigue_center   = 11.62
ace asymmetry    = 1.7×ace / 0.3×weaker             intercept        = 0.00
outdoor_intercept = 0.7
```

park formula: `park_adj = (pf - 1.0) * park_weight + (pf - 1.0)**2 * park2_weight`
`outdoor_adj = outdoor_intercept if outdoor else 0.0`

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

## Ideas to Try (session 7)

**PRIORITY 1 — Revert to cleaner optima (session 6 overshot on several)**
Session 6 kept every experiment even when the improvement was marginal or holdout
declined slightly. Several weights drifted past their true optimum:
- `outdoor_intercept = 0.6`: 0.7 tied train with 0.6 but holdout was slightly worse.
  Try reverting to 0.6 — if train holds at -0.0231, it's a cleaner config.
  (This is the one most likely to be worth reverting.)
- `park_weight = -2.4`: was peak train (-0.0227 at the time); -3.0 matched it later
  with worse holdout. Consider testing -2.4 to see if cleaner.
- `model_blend = 0.29`: 0.28 regressed train slightly and holdout was same.
  Reverting would need train to match -0.0231 — check before committing.

**PRIORITY 2 — Continue active trends**
- `outdoor_intercept = 0.8`: if 0.7 is confirmed best, one more step to check ceiling
- `park2_weight = 19.0`: still moving at 18.0 (16→17→18 all kept); try one more
- `bp_weight = 0.26`: 0.27 kept; one more step (floor still unclear)
- `park_weight = -3.3`: -3.0 kept; one more step

**PRIORITY 3 — Fine-tunes not yet exhausted**
- `fatigue_weight = -0.24`: -0.23 kept; -0.22 was also kept; trend still active
- `era_fip_div_w = 0.31`: 0.30 kept; one more step
- `sp_weight = 0.41`: sp has been at 0.42 since session 2; test slightly lower in new context

**PRIORITY 4 — Joint scan (after individual wins stabilize)**
- `outdoor_intercept × park_weight`: both affect outdoor/park run environment;
  joint optimum likely exists

---

## Ruled Out — Do Not Re-test

- OPS, SRS, fip_blend>0, cumulative ERA-FIP divergence, separate home/away weights
- Line movement, weather (temp/wind), ERA instead of FIP, per-team intercepts
- model_blend ≥ 0.35 or ≤ 0.27 (test 0.26 cautiously at most)
- sp_weight 0.43+ (reverted sessions 3 and 5), sp_weight ≤ 0.40
- bp_weight 0.30+ (0.30 was kept session 5 but clearly not optimum now)
- park_weight between -2.7 and 0.4 (swept monotonically)
- park2_weight ≤ 15.0
- era_fip_div_w ≤ 0.25 or ≥ 0.32
- gap_weight beyond -0.35 or less aggressive
- ace asymmetry beyond 1.7× or below 1.5×
- offense_weight ≠ 0.19
- fatigue_weight above -0.19 or below -0.25
- FIP_CAP ≠ 5.25; BP_ERA_CAP ≠ 5.75
- Global intercept (median residual ≈ 0)

---

## Quick Reference

```bash
cat results.tsv
python3 run_experiment.py
git revert HEAD --no-edit
git log --oneline -10
```
