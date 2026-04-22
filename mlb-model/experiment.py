# MLB Totals Research Experiment
# THIS IS THE ONLY FILE YOU SHOULD MODIFY.
# Run:  python run_experiment.py
#
# Experiment 23: Fatigue weight fine-tune — try fatigue_weight = -0.08 (was -0.07).
# Hypothesis: slightly stronger bullpen fatigue penalty may improve predictions;
# a tired bullpen gives up more runs on top of its ERA signal.

REPLACEMENT_FIP = 4.40
REPLACEMENT_ERA = 4.50
FIP_CAP    = 5.5  # starter FIP cap
BP_ERA_CAP = 6.0  # bullpen ERA cap — p90=5.93

# --- Weights: tune freely ---
fip_blend      = 0.0    # pure cumulative FIP
sp_weight      = 0.40   # starter FIP contribution — EXPERIMENT 15: try 0.40 (was 0.45)
bp_weight      = 0.35   # bullpen ERA contribution
park_weight    = 2.0    # park factor deviation
temp_weight    = 0.00   # confirmed dead signal
wind_weight    = 0.00   # confirmed dead signal
offense_weight = 0.15   # rolling 10-game runs scored (both teams)
srs_weight     = 0.00   # ablated experiment 2
ops_weight     = 0.00   # ablated experiment 1
fatigue_weight = -0.08  # bullpen 3-day IP fatigue — EXPERIMENT 23: try -0.08 (was -0.07)
fatigue_center = 11.62  # training-mean combined 3-day BP IP
era_fip_div_w  = 0.25   # ERA-FIP last-3 divergence — EXPERIMENT 21: try 0.25 (was 0.20)
intercept      = 0.00
model_blend    = 0.40   # EXPERIMENT 10: weight on statistical model vs opening O/U line


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
    sp_runs   = (0.6 * sp_worse + 1.4 * sp_better)                          * sp_weight
    bp_runs  = (min(f("home_bp_era"), BP_ERA_CAP) + min(f("away_bp_era"), BP_ERA_CAP)) * bp_weight
    park_adj = (f("park_factor") - 1.0)                                      * park_weight
    temp_adj = ((f("temp_f") - 72.0) * temp_weight) if outdoor else 0.0
    wind_adj = (f("tailwind_mph") * wind_weight) if outdoor else 0.0
    off_adj  = (f("home_roll10") + f("away_roll10"))                         * offense_weight
    srs_adj  = (f("home_srs") + f("away_srs"))                              * srs_weight
    ops_adj  = (fb("home_ops", 0.720) + fb("away_ops", 0.720) - 1.440)      * ops_weight
    fat_adj  = (f("home_bp_ip_3d") + f("away_bp_ip_3d") - fatigue_center)   * fatigue_weight
    div_adj  = ((fb("home_l3_era", REPLACEMENT_ERA) - fb("home_l3_fip", REPLACEMENT_FIP)) +
                (fb("away_l3_era", REPLACEMENT_ERA) - fb("away_l3_fip", REPLACEMENT_FIP))) * era_fip_div_w

    raw = (sp_runs + bp_runs + park_adj + temp_adj + wind_adj
           + off_adj + srs_adj + ops_adj + fat_adj + div_adj + intercept)
    ou  = fb("close_ou", fb("open_ou", raw))  # closing line (sharper); fallback to open, then model
    return model_blend * raw + (1 - model_blend) * ou
