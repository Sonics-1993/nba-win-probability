# MLB totals model weights — locked 2026-04-21 from autoresearch run 11 (autoresearch/apr21)
# Beats market on both train (delta -0.0117) and holdout (delta -0.0153).
#
# Prediction (see evaluate_totals.py predict()):
#   home_sp = clamp(cum_fip*(1-fip_blend) + l3_fip*fip_blend, FIP_CAP)
#   sp contribution: 0.6×worse_FIP + 1.4×better_FIP  — ace dominates game total
#   (sp_asymmetric) * sp_weight
#   (clamp(home_bp_era, BP_ERA_CAP) + clamp(away_bp_era, BP_ERA_CAP)) * bp_weight
#   (park_factor - 1)           * park_weight
#   (home_roll10 + away_roll10) * offense_weight
#   ERA-FIP last-3 divergence   * era_fip_div_w   [positive: ERA>FIP → more runs]
#   (home_bp_ip_3d + away_bp_ip_3d - fatigue_center) * fatigue_weight
#   raw_blend = model_blend * raw + (1 - model_blend) * close_ou
#
# Ruled out (ablated): ops_weight, srs_weight, temp_weight, wind_weight, intercept

REPLACEMENT_FIP = 4.40
REPLACEMENT_ERA = 4.50
FIP_CAP    = 5.5   # starter FIP cap (p99=5.56)
BP_ERA_CAP = 6.0   # bullpen ERA cap (p90=5.93)

fip_blend      = 0.0    # pure cumulative FIP
sp_weight      = 0.45
bp_weight      = 0.35
park_weight    = 2.0
temp_weight    = 0.00   # no signal
wind_weight    = 0.00   # no signal
offense_weight = 0.15
srs_weight     = 0.00   # ablated — redundant with roll10
ops_weight     = 0.00   # ablated — collinear with roll10
fatigue_weight = -0.07
fatigue_center = 11.62
era_fip_div_w  = 0.10   # ERA-FIP last-3 divergence
intercept      = 0.00
model_blend    = 0.40   # weight on statistical model; 1-model_blend goes to close_ou
