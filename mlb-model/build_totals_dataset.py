"""
build_totals_dataset.py — Assemble totals (O/U) training data

Usage:
    python3 build_totals_dataset.py --season 2025
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

BASE  = Path(__file__).parent
CACHE = BASE / "cache"
REPLACEMENT_ERA = 4.50
REPLACEMENT_FIP = 4.40
BP_REPLACEMENT  = 4.80
ASSUMED_INNINGS = 9.0
SHRINK_SP       = 20.0
SHRINK_BP       = 30.0
LEAGUE_OPS      = 0.720


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_rolling_runs(games: list[dict], window: int = 10) -> dict:
    """Rolling avg runs scored per team per game (shift-1, no leakage). Key: (game_pk, side)."""
    history: dict[str, list] = defaultdict(list)
    result = {}
    for g in sorted(games, key=lambda x: (x["date"], x["gamePk"])):
        pk = g["gamePk"]
        for side, abbr, runs in [("home", g["home_abbr"], g["home_runs"]),
                                  ("away", g["away_abbr"], g["away_runs"])]:
            hist = history[abbr]
            if len(hist) >= 3:
                result[(pk, side)] = sum(hist[-window:]) / len(hist[-window:])
            hist.append(float(runs))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    s = args.season

    for fname in [f"games_{s}_raw.json", f"totals_history_{s}.csv"]:
        if not (CACHE / fname).exists():
            print(f"Missing {fname}"); return

    games  = json.loads((CACHE / f"games_{s}_raw.json").read_text())
    ou_map = {(r["date"], r["home"], r["away"]): r
              for r in load_csv(CACHE / f"totals_history_{s}.csv")}

    # SRS
    srs_file = CACHE / f"blended_srs_{s}.csv"
    srs_col  = "blended_srs" if srs_file.exists() else "cum_srs"
    if not srs_file.exists(): srs_file = CACHE / f"cum_srs_{s}.csv"
    srs_map  = {(r["date"], r["team"]): float(r[srs_col]) for r in load_csv(srs_file)}

    # Starter ERA + FIP + last-3 (from pitcher_era_{s}.csv)
    era_file = CACHE / f"pitcher_era_{s}.csv"
    era_map:      dict[tuple, float] = {}
    fip_map:      dict[tuple, float] = {}
    last3_era_map: dict[tuple, float] = {}
    last3_fip_map: dict[tuple, float] = {}
    if era_file.exists():
        for r in load_csv(era_file):
            key = (int(r["game_pk"]), r["side"])
            era_map[key]       = float(r["cum_era"])
            fip_map[key]       = float(r["cum_fip"])      if r.get("cum_fip")   else REPLACEMENT_FIP
            last3_era_map[key] = float(r["last3_era"])    if r.get("last3_era") else REPLACEMENT_ERA
            last3_fip_map[key] = float(r["last3_fip"])    if r.get("last3_fip") else REPLACEMENT_FIP

    # Bullpen ERA (proxy from runs allowed − starter ER)
    starts_file = CACHE / f"pitcher_starts_{s}.json"
    bp_map: dict[tuple, float] = {}
    if starts_file.exists():
        starts  = json.loads(starts_file.read_text())
        outings: dict[str, list] = defaultdict(list)
        for g in games:
            pk_str = str(g["gamePk"])
            sp = starts.get(pk_str, {})
            for side_key, abbr, opp_runs in [
                ("home_sp", g["home_abbr"], g["away_runs"]),
                ("away_sp", g["away_abbr"], g["home_runs"]),
            ]:
                entry  = sp.get(side_key, {})
                bp_ip  = max(0.0, ASSUMED_INNINGS - float(entry.get("ip", 0) or 0))
                bp_er  = max(0.0, float(opp_runs) - float(entry.get("er", 0) or 0))
                outings[abbr].append((g["date"], g["gamePk"], bp_ip, bp_er))
        for g in sorted(games, key=lambda x: (x["date"], x["gamePk"])):
            pk = g["gamePk"]
            for side, abbr in [("home", g["home_abbr"]), ("away", g["away_abbr"])]:
                hist     = [(ip, er) for d, gpk, ip, er in outings[abbr]
                            if d < g["date"] or (d == g["date"] and gpk < pk)]
                total_ip = sum(ip for ip, _ in hist)
                total_er = sum(er for _, er in hist)
                bp_map[(pk, side)] = ((total_er * 9 + BP_REPLACEMENT * SHRINK_BP)
                                      / (total_ip + SHRINK_BP))

    # Park factors
    pf_map: dict[str, float] = {}
    pf_file = CACHE / "park_factors.json"
    if pf_file.exists():
        raw    = json.loads(pf_file.read_text())
        pf_map = {k: v for k, v in raw.items() if not k.startswith("_")}

    # Weather
    wx_map = ({int(r["game_pk"]): r for r in load_csv(CACHE / f"weather_{s}.csv")}
              if (CACHE / f"weather_{s}.csv").exists() else {})

    # Rolling runs offense
    roll_off = build_rolling_runs(games)

    # Rolling team OPS
    ops_file = CACHE / f"rolling_team_ops_{s}.csv"
    ops_map: dict[tuple, float] = {}
    if ops_file.exists():
        for r in load_csv(ops_file):
            ops_map[(int(r["game_pk"]), r["team"])] = float(r["rolling_ops"])

    # Bullpen fatigue: rolling IP last 3 days
    fat_file = CACHE / f"bullpen_fatigue_{s}.csv"
    fat_map: dict[int, dict] = {}
    if fat_file.exists():
        for r in load_csv(fat_file):
            fat_map[int(r["game_pk"])] = r

    rows, missing_ou = [], 0
    for g in games:
        date, home, away, pk = g["date"], g["home_abbr"], g["away_abbr"], g["gamePk"]
        ou = ou_map.get((date, home, away))
        if not ou or not ou.get("open_ou"):
            missing_ou += 1

        home_sp_era  = era_map.get((pk, "home"),  REPLACEMENT_ERA)
        away_sp_era  = era_map.get((pk, "away"),  REPLACEMENT_ERA)
        home_sp_fip  = fip_map.get((pk, "home"),  REPLACEMENT_FIP)
        away_sp_fip  = fip_map.get((pk, "away"),  REPLACEMENT_FIP)
        home_l3_era  = last3_era_map.get((pk, "home"), REPLACEMENT_ERA)
        away_l3_era  = last3_era_map.get((pk, "away"), REPLACEMENT_ERA)
        home_l3_fip  = last3_fip_map.get((pk, "home"), REPLACEMENT_FIP)
        away_l3_fip  = last3_fip_map.get((pk, "away"), REPLACEMENT_FIP)
        home_bp_era  = bp_map.get((pk, "home"),   BP_REPLACEMENT)
        away_bp_era  = bp_map.get((pk, "away"),   BP_REPLACEMENT)
        home_srs     = srs_map.get((date, home),  0.0)
        away_srs     = srs_map.get((date, away),  0.0)
        pf           = pf_map.get(home, 1.0)
        wx           = wx_map.get(pk, {})
        home_roll    = roll_off.get((pk, "home"), "")
        away_roll    = roll_off.get((pk, "away"), "")
        home_ops     = ops_map.get((pk, home), "")
        away_ops     = ops_map.get((pk, away), "")
        fat          = fat_map.get(pk, {})
        home_bp_3d   = float(fat.get("home_bp_ip_3d", 0.0))
        away_bp_3d   = float(fat.get("away_bp_ip_3d", 0.0))

        rows.append({
            "date":          date, "game_pk": pk, "home": home, "away": away,
            "total_runs":    g["home_runs"] + g["away_runs"],
            "home_runs":     g["home_runs"], "away_runs": g["away_runs"],
            # Cumulative ERA / FIP (season-to-date, shrunk)
            "home_sp_era":   round(home_sp_era,  3),
            "away_sp_era":   round(away_sp_era,  3),
            "home_sp_fip":   round(home_sp_fip,  3),
            "away_sp_fip":   round(away_sp_fip,  3),
            # Last-3 starts ERA / FIP (recent form)
            "home_l3_era":   round(home_l3_era,  3),
            "away_l3_era":   round(away_l3_era,  3),
            "home_l3_fip":   round(home_l3_fip,  3),
            "away_l3_fip":   round(away_l3_fip,  3),
            # Bullpen / park / weather
            "home_bp_era":   round(home_bp_era,  3),
            "away_bp_era":   round(away_bp_era,  3),
            "home_srs":      round(home_srs,     4),
            "away_srs":      round(away_srs,     4),
            "park_factor":   round(pf,            3),
            "temp_f":        wx.get("temp_f",    ""),
            "wind_mph":      wx.get("wind_mph",  ""),
            "tailwind_mph":  wx.get("tailwind_mph", ""),
            "is_outdoor":    wx.get("is_outdoor", ""),
            # Offense
            "home_roll10":   round(home_roll, 3) if isinstance(home_roll, float) else "",
            "away_roll10":   round(away_roll, 3) if isinstance(away_roll, float) else "",
            "home_ops":      round(home_ops,  4) if isinstance(home_ops,  float) else "",
            "away_ops":      round(away_ops,  4) if isinstance(away_ops,  float) else "",
            # Bullpen fatigue (rolling IP last 3 calendar days)
            "home_bp_ip_3d": round(home_bp_3d, 2),
            "away_bp_ip_3d": round(away_bp_3d, 2),
            # Market line
            "open_ou":       ou.get("open_ou",  "") if ou else "",
            "close_ou":      ou.get("close_ou", "") if ou else "",
        })

    out    = CACHE / f"totals_training_{s}.csv"
    fields = [
        "date","game_pk","home","away","total_runs","home_runs","away_runs",
        "home_sp_era","away_sp_era","home_sp_fip","away_sp_fip",
        "home_l3_era","away_l3_era","home_l3_fip","away_l3_fip",
        "home_bp_era","away_bp_era",
        "home_srs","away_srs","park_factor",
        "temp_f","wind_mph","tailwind_mph","is_outdoor",
        "home_roll10","away_roll10","home_ops","away_ops",
        "home_bp_ip_3d","away_bp_ip_3d",
        "open_ou","close_ou",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    has_fip = sum(1 for r in rows if r["home_sp_fip"] != REPLACEMENT_FIP)
    has_ops = sum(1 for r in rows if r["home_ops"] != "")
    print(f"Wrote {len(rows)} rows → {out}")
    print(f"  Missing O/U:   {missing_ou} ({missing_ou/len(rows):.1%})")
    print(f"  With O/U line: {sum(1 for r in rows if r['open_ou'])}")
    print(f"  With FIP data: {has_fip} ({has_fip/len(rows):.1%})")
    print(f"  With OPS data: {has_ops} ({has_ops/len(rows):.1%})")


if __name__ == "__main__":
    main()
