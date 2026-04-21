"""
build_training_dataset.py — Assemble MLB model training data

Usage:
    python3 build_training_dataset.py --season 2025
    python3 build_training_dataset.py --season 2024
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE  = Path(__file__).parent
CACHE = BASE / "cache"

REPLACEMENT_ERA = 4.50
LEAGUE_AVG_RUNS = 8.8


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_roll10(games: list[dict]) -> dict[tuple, float]:
    """Rolling 10-game avg run diff per team, shift(1) to avoid leakage. Key: (game_pk, side)."""
    team_history: dict[str, list[float]] = defaultdict(list)
    result: dict[tuple, float] = {}
    for g in sorted(games, key=lambda x: (x["date"], x["gamePk"])):
        pk = g["gamePk"]
        home, away = g["home_abbr"], g["away_abbr"]
        h_hist, a_hist = team_history[home], team_history[away]
        if len(h_hist) >= 3:
            result[(pk, "home")] = sum(h_hist[-10:]) / len(h_hist[-10:])
        if len(a_hist) >= 3:
            result[(pk, "away")] = sum(a_hist[-10:]) / len(a_hist[-10:])
        h_hist.append(float(g["run_diff"]))
        a_hist.append(-float(g["run_diff"]))
    return result


def build_rest_days(games: list[dict]) -> dict[tuple, int]:
    last: dict[str, str] = {}
    rest: dict[tuple, int] = {}
    for g in sorted(games, key=lambda x: x["date"]):
        for abbr in (g["home_abbr"], g["away_abbr"]):
            if abbr in last:
                delta = (datetime.fromisoformat(g["date"]) - datetime.fromisoformat(last[abbr])).days
                rest[(g["date"], abbr)] = min(delta, 7)
            else:
                rest[(g["date"], abbr)] = 3
            last[abbr] = g["date"]
    return rest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    s = args.season

    games_file = CACHE / f"games_{s}_raw.json"
    odds_file  = CACHE / f"odds_history_{s}.csv"
    era_file   = CACHE / f"pitcher_era_{s}.csv"

    # SRS: prefer blended, fall back to cumulative
    srs_file = CACHE / f"blended_srs_{s}.csv"
    srs_col  = "blended_srs"
    if not srs_file.exists():
        srs_file = CACHE / f"cum_srs_{s}.csv"
        srs_col  = "cum_srs"

    for f, name in [(games_file, "fetch_mlb_games.py"),
                    (odds_file,  "fetch_mlb_odds.py"),
                    (srs_file,   "build_mlb_srs.py")]:
        if not f.exists():
            print(f"Missing {f.name} — run {name} --season {s} first.")
            return

    games = json.loads(games_file.read_text())
    print(f"Loaded {len(games)} games for {s}")

    odds_map = {(r["date"], r["home"], r["away"]): r for r in load_csv(odds_file)}
    srs_map  = {(r["date"], r["team"]): float(r[srs_col]) for r in load_csv(srs_file)}
    era_map  = ({(int(r["game_pk"]), r["side"]): float(r["cum_era"])
                 for r in load_csv(era_file)} if era_file.exists() else {})

    park_factors: dict[str, float] = {}
    pf_file = CACHE / "park_factors.json"
    if pf_file.exists():
        raw = json.loads(pf_file.read_text())
        park_factors = {k: v for k, v in raw.items() if not k.startswith("_")}

    rest_map   = build_rest_days(games)
    roll10_map = build_roll10(games)
    wx_file    = CACHE / f"weather_{s}.csv"
    wx_map     = ({int(r["game_pk"]): r for r in load_csv(wx_file)}
                  if wx_file.exists() else {})

    rows, missing_odds, missing_srs = [], 0, 0

    for g in games:
        date, home, away, pk = g["date"], g["home_abbr"], g["away_abbr"], g["gamePk"]
        odds = odds_map.get((date, home, away))
        if not odds or not odds.get("open_rl"):
            missing_odds += 1
        home_srs = srs_map.get((date, home), 0.0)
        away_srs = srs_map.get((date, away), 0.0)
        if home_srs == 0.0 and away_srs == 0.0:
            missing_srs += 1
        pf          = park_factors.get(home, 1.0)
        home_roll10 = roll10_map.get((pk, "home"))
        away_roll10 = roll10_map.get((pk, "away"))
        roll10_diff = round(home_roll10 - away_roll10, 4) if (home_roll10 is not None and away_roll10 is not None) else ""
        wx          = wx_map.get(pk, {})
        rows.append({
            "date":         date, "game_pk": pk, "home": home, "away": away,
            "home_runs":    g["home_runs"], "away_runs": g["away_runs"],
            "run_diff":     g["run_diff"],  "home_win":  int(g["home_win"]),
            "home_srs":     round(home_srs, 4),
            "away_srs":     round(away_srs, 4),
            "srs_diff":     round(home_srs - away_srs, 4),
            "home_era":     round(era_map.get((pk, "home"), REPLACEMENT_ERA), 3),
            "away_era":     round(era_map.get((pk, "away"), REPLACEMENT_ERA), 3),
            "era_diff":     round(era_map.get((pk, "away"), REPLACEMENT_ERA)
                                  - era_map.get((pk, "home"), REPLACEMENT_ERA), 3),
            "home_rest":    rest_map.get((date, home), 3),
            "away_rest":    rest_map.get((date, away), 3),
            "rest_diff":    rest_map.get((date, home), 3) - rest_map.get((date, away), 3),
            "park_factor":  round(pf, 3),
            "park_adj":     round((pf - 1.0) * LEAGUE_AVG_RUNS / 2, 3),
            "roll10_diff":  roll10_diff,
            "temp_f":       wx.get("temp_f", ""),
            "wind_mph":     wx.get("wind_mph", ""),
            "tailwind_mph": wx.get("tailwind_mph", ""),
            "is_outdoor":   wx.get("is_outdoor", ""),
            "open_rl":      f"{float(odds['open_rl']):.2f}"  if odds and odds.get("open_rl")  else "",
            "close_rl":     f"{float(odds['close_rl']):.2f}" if odds and odds.get("close_rl") else "",
            "venue":        g.get("venue", ""),
        })

    out = CACHE / f"training_data_{s}.csv"
    fields = ["date","game_pk","home","away","home_runs","away_runs","run_diff","home_win",
              "home_srs","away_srs","srs_diff","home_era","away_era","era_diff",
              "home_rest","away_rest","rest_diff","park_factor","park_adj","roll10_diff",
              "temp_f","wind_mph","tailwind_mph","is_outdoor",
              "open_rl","close_rl","venue"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {out}")
    print(f"  Missing odds: {missing_odds} ({missing_odds/len(rows):.1%})  "
          f"Missing SRS: {missing_srs} ({missing_srs/len(rows):.1%})")
    print(f"  Games with run line: {sum(1 for r in rows if r['open_rl'])}")


if __name__ == "__main__":
    main()
