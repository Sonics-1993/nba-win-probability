# MLB totals model weights — tuned 2026-04-21, updated with prior FIP + fatigue
# Run evaluate_totals.py --grid-search to re-tune after data changes.
#
# Prediction (see evaluate_totals.py predict()):
#   starter = cum_fip*(1-fip_blend) + last3_fip*fip_blend
#   (home_sp + away_sp) * sp_weight
#   (home_bp_era + away_bp_era) * bp_weight
#   (park_factor - 1)           * park_weight
#   (temp_f - 72) * temp_weight         [outdoor only — no signal, keep 0]
#   tailwind_mph  * wind_weight          [outdoor only — no signal, keep 0]
#   (home_roll10 + away_roll10) * offense_weight
#   (home_srs + away_srs)       * srs_weight
#   (home_ops + away_ops - 1.44)* ops_weight
#   (home_bp_ip_3d + away_bp_ip_3d - fatigue_center) * fatigue_weight
#   + intercept

fip_blend      = 0.2    # 20% last-3 FIP, 80% cumulative FIP — recency without noise
sp_weight      = 0.50   # FIP × innings fraction (~5.5/9)
bp_weight      = 0.30   # bullpen ERA × innings fraction (~3.5/9)
park_weight    = 2.0    # park factor deviation (COL=1.40 adds +0.80 runs)
temp_weight    = 0.00   # no signal even for outdoor games
wind_weight    = 0.00   # no signal even for outdoor games
offense_weight = 0.15   # rolling 10-game runs scored
srs_weight     = 0.15   # combined team SRS — both good offenses → more runs
ops_weight     = 2.0    # rolling team OPS above/below 1.440 league average
fatigue_weight = -0.05  # negative: high recent BP usage → slight regression toward mean
fatigue_center = 11.62  # training-set mean combined 3-day BP IP (Mar-Jul 2025)
intercept      = 0.00
