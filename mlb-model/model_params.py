# MLB run-differential model weights
# Positive prediction = home team favored (predicted run diff from home perspective)

srs_weight    = 0.50   # blended SRS differential
era_weight    = 0.00   # ERA diff — noisy until pitcher shrinkage is added (see fetch_pitcher_stats)
rest_weight   = 0.05   # rest day differential
park_weight   = 0.00   # park adj — minimal signal at current calibration
hca           = 0.40   # home field advantage in runs (grid search result)
intercept     = 0.0
