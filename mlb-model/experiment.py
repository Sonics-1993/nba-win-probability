# MLB Totals Research Experiment
# THIS IS THE ONLY FILE YOU SHOULD MODIFY.
# Run:  python run_experiment.py
#
# Experiment 7: joint re-optimization with FIP cap + asymmetric weighting in place.
# Grid: alpha=0.7 (more extreme ace weighting), sp↑, bp↓, park↓, off↑.

REPLACEMENT_FIP = 4.40
REPLACEMENT_ERA = 4.50
FIP_CAP = 5.5   # cap starter FIP — p99 is 5.56, only ~9 train games exceed this

# --- Weights: tune freely ---
fip_blend      = 0.0    # pure cumulative FIP
sp_weight      = 0.48   # starter FIP contribution (up slightly)
bp_weight      = 0.32   # bullpen ERA contribution (down slightly)
park_weight    = 2.5    # park factor deviation
temp_weight    = 0.00   # confirmed dead signal
wind_weight    = 0.00   # confirmed dead signal
offense_weight = 0.15   # rolling 10-game runs scored (both teams)
srs_weight     = 0.00   # ablated experiment 2
ops_weight     = 0.00   # ablated experiment 1
fatigue_weight = -0.07  # bullpen 3-day IP fatigue
fatigue_center = 11.62  # training-mean combined 3-day BP IP
era_fip_div_w  = 0.10   # ERA-FIP last-3 divergence
intercept      = 0.00


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
    sp_runs   = (0.7 * sp_worse + 1.3 * sp_better)                          * sp_weight
    bp_runs  = (f("home_bp_era") + f("away_bp_era"))                        * bp_weight
    park_adj = (f("park_factor") - 1.0)                                      * park_weight
    temp_adj = ((f("temp_f") - 72.0) * temp_weight) if outdoor else 0.0
    wind_adj = (f("tailwind_mph") * wind_weight) if outdoor else 0.0
    off_adj  = (f("home_roll10") + f("away_roll10"))                         * offense_weight
    srs_adj  = (f("home_srs") + f("away_srs"))                              * srs_weight
    ops_adj  = (fb("home_ops", 0.720) + fb("away_ops", 0.720) - 1.440)      * ops_weight
    fat_adj  = (f("home_bp_ip_3d") + f("away_bp_ip_3d") - fatigue_center)   * fatigue_weight
    div_adj  = ((fb("home_l3_era", REPLACEMENT_ERA) - fb("home_l3_fip", REPLACEMENT_FIP)) +
                (fb("away_l3_era", REPLACEMENT_ERA) - fb("away_l3_fip", REPLACEMENT_FIP))) * era_fip_div_w

    return (sp_runs + bp_runs + park_adj + temp_adj + wind_adj
            + off_adj + srs_adj + ops_adj + fat_adj + div_adj + intercept)
