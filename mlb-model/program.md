# MLB Totals Autoresearch — Agent Instructions

Adapted from Karpathy's autoresearch framework.
You are an autonomous research agent improving the MLB run-totals prediction model.

---

## Goal

Minimize `train_delta`. Keep if train_delta improves (lower than −0.0222).
Apply holdout guard: revert if holdout worsens >0.005 with only marginal train gain (<0.0003).

**Current state (end of session 5, run 155):**
- TRAIN   N=1602  Delta=**−0.0222**
- HOLDOUT N= 781  Delta=**−0.0104**

WARNING: Train/holdout divergence is emerging. bp_weight reduction (0.33→0.29) improved
train strongly but hurt holdout from −0.0135 (run 145) to −0.0104 (run 155). The best
holdout this session was −0.0135 at park2_weight=16.0 / bp_weight=0.33. Prioritize
experiments that recover holdout without sacrificing train gains.

---

## Current weights (experiment.py)

```
sp_weight        = 0.42   bp_weight        = 0.29   park_weight      = -1.8
park2_weight     = 16.0   offense_weight   = 0.19   fatigue_weight   = -0.20
era_fip_div_w    = 0.27   gap_weight       = -0.35  model_blend      = 0.30
FIP_CAP          = 5.25   BP_ERA_CAP       = 5.75   fatigue_center   = 11.62
ace asymmetry    = 1.7×ace / 0.3×weaker             intercept        = 0.00
outdoor_intercept = 0.3   (ADDED session 5)
```

park formula: `park_adj = (pf - 1.0) * park_weight + (pf - 1.0)**2 * park2_weight`
`outdoor_adj = outdoor_intercept if outdoor else 0.0` is now included in raw.

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

## Ideas to Try (session 6)

**PRIORITY 1 — Resolve bp_weight train/holdout tension**
bp_weight swept 0.33→0.29, each step improving train but hurting holdout. The sweet spot
may be around 0.31 (held well) or 0.30 (holdout was still ok). Try:
- `bp_weight = 0.31`: was kept in session 5 (holdout −0.0123). Re-test is NOT needed — already in results.
  Instead, think of this as: session 5 kept 0.30 then 0.29 — try `bp_weight = 0.28` to see if
  train continues improving, or the ceiling is 0.29.
- Actually, consider reverting `bp_weight` to 0.31 or 0.30 to check holdout recovery — but
  this would sacrifice train. Instead: test other dimensions first, then revisit bp_weight with
  a joint scan against park2_weight.

**PRIORITY 2 — Park weight / park2 joint area**
- `park_weight = -2.1`: continue trend (−1.8 kept; holdout guard applies)
- `park2_weight = 17.0`: was reverted in session 5 (train hurt). Re-test now with park_weight=−1.8
  (different context from when 17.0 was first tried at park_weight=−0.6).

**PRIORITY 3 — Fine-tunes**
- `outdoor_intercept = 0.4`: 0.3 kept, 0.5 reverted. Midpoint not yet tested.
- `era_fip_div_w = 0.28`: 0.27 kept borderline; continue.
- `fatigue_weight = -0.21`: −0.20 kept; one more step (floor ~−0.22 per program).
- `model_blend = 0.29`: 0.30 kept; one more step (caution — may be floor).
- `sp_weight = 0.43`: consistently reverted in sessions 3 and 5. Do NOT re-test.

**PRIORITY 4 — Joint scans (after individual wins stabilize)**
- `bp_weight × park2_weight`: bp divergence may interact with park2 scaling.
- `fatigue_weight × era_fip_div_w`.

---

## Ruled Out — Do Not Re-test

- OPS, SRS, fip_blend>0, cumulative ERA-FIP divergence, separate home/away weights
- Line movement, weather (temp/wind), ERA instead of FIP, per-team intercepts
- model_blend ≥ 0.35 or ≤ 0.28 (caution below 0.29; test one step at a time)
- sp_weight 0.43+ (reverted sessions 3 and 5), sp_weight ≤ 0.40
- bp_weight 0.35, 0.37, 0.40, 0.30 (kept session 5), 0.31 (kept), 0.32 (kept), 0.33 (prev optimum)
- park_weight between −1.5 and 0.4 (already swept; holdout peaked at 0.4 in session 4)
- park2_weight 17.0 (reverted session 5 — re-test only with new park_weight context as noted above)
- park2_weight 8.0, 12.0 (prev reverts)
- era_fip_div_w ≤ 0.05 or ≥ 0.30
- gap_weight beyond −0.35 or less aggressive than −0.35
- ace asymmetry beyond 1.7× or below 1.5×
- offense_weight 0.17, 0.20 (0.19 is optimum)
- fatigue_weight above −0.16 or below −0.22
- FIP_CAP 5.0, 5.5+; BP_ERA_CAP 5.5, 6.0
- outdoor_intercept 0.5+ (reverted session 5); 0.4 not yet tested
- Global intercept

---

## Quick Reference

```bash
cat results.tsv
python3 run_experiment.py
git revert HEAD --no-edit
git log --oneline -10
```
