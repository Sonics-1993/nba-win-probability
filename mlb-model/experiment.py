# MLB Totals Research Experiment
# THIS IS THE ONLY FILE YOU SHOULD MODIFY.
# Run:  python run_experiment.py
#
# Experiment 3: grid search on active weights after removing OPS+SRS.
# Grid found: fip_blend=0 (pure cumulative FIP), higher bp/park, slightly lower offense+sp.
# Expected train delta +0.0326 vs current +0.0373.

REPLACEMENT_FIP = 4.40
REPLACEMENT_ERA = 4.50

# --- Weights: tune freely ---
fip_blend      = 0.0    # pure cumulative FIP — last-3 adds noise without OPS/SRS context
sp_weight      = 0.45   # starter FIP contribution
bp_weight      = 0.35   # bullpen ERA contribution (up — BP matters more without SRS)
park_weight    = 3.0    # park factor deviation (up — extreme parks matter more)
temp_weight    = 0.00   # temperature effect (outdoor only) — confirmed dead signal
wind_weight    = 0.00   # tailwind mph (outdoor only) — confirmed dead signal
offense_weight = 0.13   # rolling 10-game runs scored (both teams)
srs_weight     = 0.00   # ablated experiment 2
ops_weight     = 0.00   # ablated experiment 1
fatigue_weight = -0.07  # slightly stronger fatigue regression signal
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
