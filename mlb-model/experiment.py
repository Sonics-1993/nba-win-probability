# MLB Totals Research Experiment
# THIS IS THE ONLY FILE YOU SHOULD MODIFY.
# Run:  python run_experiment.py
#
# Hypothesis: OPS (rolling team OPS) is adding noise rather than signal.
# Ablation showed removing ops_weight improves both train (-0.008) and holdout (-0.003).
# OPS is likely collinear with roll10 offense runs — the runs feature already captures output.

REPLACEMENT_FIP = 4.40
REPLACEMENT_ERA = 4.50

# --- Weights: tune freely ---
fip_blend      = 0.2    # 0–1: weight on last-3 FIP vs cumulative FIP
sp_weight      = 0.50   # starter FIP contribution
bp_weight      = 0.30   # bullpen ERA contribution
park_weight    = 2.0    # park factor deviation
temp_weight    = 0.00   # temperature effect (outdoor only) — confirmed dead signal
wind_weight    = 0.00   # tailwind mph (outdoor only) — confirmed dead signal
offense_weight = 0.15   # rolling 10-game runs scored (both teams)
srs_weight     = 0.15   # simple rating system (combined)
ops_weight     = 0.0    # EXPERIMENT 1: ablated — collinear with roll10, hurts both train and holdout
fatigue_weight = -0.05  # bullpen 3-day IP fatigue (negative = regression signal)
fatigue_center = 11.62  # training-mean combined 3-day BP IP
intercept      = 0.00


def predict(row: dict) -> float:
    """Return predicted total runs for a game. Modify freely."""

    def f(col):
        return float(row[col]) if row.get(col, "") not in ("", None) else 0.0

    def fb(col, fallback):
        return float(row[col]) if row.get(col, "") not in ("", None) else fallback

    outdoor = f("is_outdoor") > 0

    home_sp = fb("home_sp_fip", REPLACEMENT_FIP) * (1 - fip_blend) + \
              fb("home_l3_fip", REPLACEMENT_FIP) * fip_blend
    away_sp = fb("away_sp_fip", REPLACEMENT_FIP) * (1 - fip_blend) + \
              fb("away_l3_fip", REPLACEMENT_FIP) * fip_blend

    sp_runs  = (home_sp + away_sp)                                           * sp_weight
    bp_runs  = (f("home_bp_era") + f("away_bp_era"))                        * bp_weight
    park_adj = (f("park_factor") - 1.0)                                      * park_weight
    temp_adj = ((f("temp_f") - 72.0) * temp_weight) if outdoor else 0.0
    wind_adj = (f("tailwind_mph") * wind_weight) if outdoor else 0.0
    off_adj  = (f("home_roll10") + f("away_roll10"))                         * offense_weight
    srs_adj  = (f("home_srs") + f("away_srs"))                              * srs_weight
    ops_adj  = (fb("home_ops", 0.720) + fb("away_ops", 0.720) - 1.440)      * ops_weight
    fat_adj  = (f("home_bp_ip_3d") + f("away_bp_ip_3d") - fatigue_center)   * fatigue_weight

    return (sp_runs + bp_runs + park_adj + temp_adj + wind_adj
            + off_adj + srs_adj + ops_adj + fat_adj + intercept)
