# MLB totals model weights — locked 2026-04-21 from autoresearch run 155 (session 5)
# Train delta=-0.0222  Holdout delta=-0.0104  (beats market on both)
#
# Prediction (see evaluate_totals.py predict()):
#   home_sp = clamp(cum_fip, FIP_CAP)
#   sp contribution: 0.3×worse_FIP + 1.7×better_FIP  — ace dominates game total
#   (sp_asymmetric) * sp_weight
#   (clamp(home_bp_era, BP_ERA_CAP) + clamp(away_bp_era, BP_ERA_CAP)) * bp_weight
#   (park_factor - 1)           * park_weight          (linear)
#   (park_factor - 1)**2        * park2_weight         (quadratic)
#   abs(home_sp - away_sp)      * gap_weight           (FIP gap compresses total)
#   (home_roll10 + away_roll10) * offense_weight
#   ERA-FIP last-3 divergence   * era_fip_div_w
#   (home_bp_ip_3d + away_bp_ip_3d - fatigue_center) * fatigue_weight
#   outdoor_intercept           if is_outdoor else 0
#   raw_blend = model_blend * raw + (1 - model_blend) * close_ou
#
# Ruled out (ablated): ops_weight, srs_weight, temp_weight, wind_weight, intercept

REPLACEMENT_FIP = 4.40
REPLACEMENT_ERA = 4.50
FIP_CAP    = 5.25  # starter FIP cap
BP_ERA_CAP = 5.75  # bullpen ERA cap

fip_blend         = 0.0
sp_weight         = 0.42
bp_weight         = 0.29
park_weight       = -1.8
park2_weight      = 16.0
offense_weight    = 0.19
fatigue_weight    = -0.20
fatigue_center    = 11.62
era_fip_div_w     = 0.27
gap_weight        = -0.35
model_blend       = 0.30
outdoor_intercept = 0.3
intercept         = 0.00
temp_weight       = 0.00
wind_weight       = 0.00
srs_weight        = 0.00
ops_weight        = 0.00
