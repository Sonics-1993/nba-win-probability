# MLB Totals Research Experiment
# THIS IS THE ONLY FILE YOU SHOULD MODIFY.
# Run:  python run_experiment.py
#
# Experiment 121: park_weight = 1.8 (was 2.0).
# Hypothesis: with park2=10.0 handling extremes, linear park_weight may be redundant. Try reducing to 1.8.

REPLACEMENT_FIP = 4.40
REPLACEMENT_ERA = 4.50
FIP_CAP    = 5.25  # starter FIP cap — EXPERIMENT 28: try 5.25 (was 5.5)
BP_ERA_CAP = 5.75  # bullpen ERA cap — EXPERIMENT 29: try 5.75 (was 6.0)

# --- Weights: tune freely ---
fip_blend      = 0.0    # pure cumulative FIP
sp_weight      = 0.42   # starter FIP contribution — EXP 86: try 0.42 (was 0.41)
bp_weight      = 0.33   # bullpen ERA contribution — EXP 116: restore 0.33 (optimum)
park_weight    = 1.8    # park factor deviation — EXP 121: try 1.8 (was 2.0), park2 handles extremes
temp_weight    = 0.00   # confirmed dead signal
wind_weight    = 0.00   # confirmed dead signal
offense_weight = 0.19   # rolling 10-game runs scored (both teams) — EXP 81: try 0.19 (was 0.18)
srs_weight     = 0.00   # ablated experiment 2
ops_weight     = 0.00   # ablated experiment 1
fatigue_weight = -0.19  # bullpen 3-day IP fatigue — EXP 105: try -0.19 (was -0.18)
fatigue_center = 11.62  # training-mean combined 3-day BP IP
era_fip_div_w  = 0.26   # ERA-FIP last-3 divergence — EXP 116: try 0.26 (was 0.25) in new park2 context
gap_weight     = -0.35  # abs SP FIP gap — EXP 57: try -0.35 (was -0.30)
park2_weight   = 10.0   # EXP 100: park factor quadratic term — try 10.0 (was 8.0)
intercept      = 0.00
model_blend    = 0.31   # EXP 110: try 0.31 (was 0.32) — continue toward market floor


def predict(row: dict) -> float:
    """Return predicted total runs for a game. Modify freely."""

    def f(col):
        return float(row[col]) if row.get(col, "") not in ("", None) else 0.0

    def fb(col, fallback):
        return float(row[col]) if row.get(col, "") not in ("", None) else fallback

    outdoor = f("is_outdoor") > 0

    home_sp = min(fb("home_sp_fip", REPLACEMENT_FIP) * (1 - fip_blend) +
                  fb("home_l3_fip", REPLACEMENT_FIP) * fip_blend, FIP_CAP)
    away_sp = min(fb("away_sp_fip", REPLACEMENT_FIP) * (1 - fip_blend) +
                  fb("away_l3_fip", REPLACEMENT_FIP) * fip_blend, FIP_CAP)

    sp_better = min(home_sp, away_sp)   # lower FIP = ace
    sp_worse  = max(home_sp, away_sp)   # higher FIP = weaker starter
    sp_runs   = (0.3 * sp_worse + 1.7 * sp_better)                          * sp_weight  # EXP 70: 1.7×ace
    bp_runs  = (min(f("home_bp_era"), BP_ERA_CAP) + min(f("away_bp_era"), BP_ERA_CAP)) * bp_weight
    pf       = f("park_factor")
    park_adj = (pf - 1.0) * park_weight + (pf - 1.0) ** 2 * park2_weight
    temp_adj = ((f("temp_f") - 72.0) * temp_weight) if outdoor else 0.0
    wind_adj = (f("tailwind_mph") * wind_weight) if outdoor else 0.0
    off_adj  = (f("home_roll10") + f("away_roll10"))                         * offense_weight
    srs_adj  = (f("home_srs") + f("away_srs"))                              * srs_weight
    ops_adj  = (fb("home_ops", 0.720) + fb("away_ops", 0.720) - 1.440)      * ops_weight
    fat_adj  = (f("home_bp_ip_3d") + f("away_bp_ip_3d") - fatigue_center)   * fatigue_weight
    div_adj  = ((fb("home_l3_era", REPLACEMENT_ERA) - fb("home_l3_fip", REPLACEMENT_FIP)) +
                (fb("away_l3_era", REPLACEMENT_ERA) - fb("away_l3_fip", REPLACEMENT_FIP))) * era_fip_div_w
    gap_adj  = abs(home_sp - away_sp)                                          * gap_weight

    raw = (sp_runs + bp_runs + park_adj + temp_adj + wind_adj
           + off_adj + srs_adj + ops_adj + fat_adj + div_adj + gap_adj + intercept)
    ou  = fb("close_ou", fb("open_ou", raw))  # closing line (sharper); fallback to open, then model
    return model_blend * raw + (1 - model_blend) * ou
