# MLB run-differential model weights
# Positive prediction = home team favored (predicted run diff from home perspective)

srs_weight    = 0.35   # blended SRS differential
era_weight    = 0.00   # ERA diff — 0.052 corr even with shrinkage, not worth including
rest_weight   = 0.15   # rest day differential
park_weight   = 0.00   # park adj — minimal signal at current calibration
roll10_weight = 0.10   # rolling 10-game run diff — 0.101 corr, joint grid best
hca           = 0.50   # home field advantage in runs (grid search result)
intercept     = 0.0
