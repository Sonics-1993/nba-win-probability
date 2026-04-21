"""
build_mlb_srs.py — Cumulative no-leakage SRS for MLB

  blended(t) = α(t) × prior + (1 − α(t)) × cum_srs
  α(t) = exp(−games_played / halflife)

Usage:
    python3 build_mlb_srs.py --season 2025
    python3 build_mlb_srs.py --season 2024 --halflife 40
    python3 build_mlb_srs.py --season 2025 --no-blend
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BASE  = Path(__file__).parent
CACHE = BASE / "cache"

RD_PER_WIN     = 0.28
SRS_ITERS      = 20
BLEND_HALFLIFE = 30


def load_games(season: int) -> list[dict]:
    return json.loads((CACHE / f"games_{season}_raw.json").read_text())


def load_preseason_prior(season: int) -> dict[str, float]:
    p = CACHE / f"preseason_wins_{season}.json"
    if not p.exists():
        print(f"  No preseason_wins_{season}.json — blend disabled.")
        return {}
    wins = json.loads(p.read_text())
    return {t: (w - 81) * RD_PER_WIN for t, w in wins.items() if not t.startswith("_")}


def srs_iteration(teams: list[str], results: list[dict]) -> dict[str, float]:
    rating = {t: 0.0 for t in teams}
    for _ in range(SRS_ITERS):
        new_rating = {}
        for team in teams:
            home_games = [(g["run_diff"],  g["away_abbr"]) for g in results if g["home_abbr"] == team]
            away_games = [(-g["run_diff"], g["home_abbr"]) for g in results if g["away_abbr"] == team]
            all_games  = home_games + away_games
            if not all_games:
                new_rating[team] = 0.0
                continue
            new_rating[team] = (sum(rd for rd, _ in all_games)
                                + sum(rating[opp] for _, opp in all_games)) / len(all_games)
        mean   = np.mean(list(new_rating.values()))
        rating = {t: v - mean for t, v in new_rating.items()}
    return rating


def build_cum_srs(season: int, halflife: float, prior: dict[str, float], no_blend: bool) -> None:
    games = load_games(season)
    teams = sorted(set(g["home_abbr"] for g in games) | set(g["away_abbr"] for g in games))
    dates = sorted(set(g["date"] for g in games))

    cum_rows, blended_rows = [], []

    for date_str in dates:
        past = [g for g in games if g["date"] < date_str]
        srs  = srs_iteration(teams, past) if past else {t: 0.0 for t in teams}
        gp   = {t: 0 for t in teams}
        for g in past:
            gp[g["home_abbr"]] += 1
            gp[g["away_abbr"]] += 1

        for team in teams:
            cum_rows.append({"date": date_str, "team": team,
                             "cum_srs": round(srs.get(team, 0.0), 4),
                             "games_played": gp[team]})
            if not no_blend:
                n       = gp[team]
                alpha   = np.exp(-n / halflife) if prior else 0.0
                blended = alpha * prior.get(team, 0.0) + (1 - alpha) * srs.get(team, 0.0)
                blended_rows.append({"date": date_str, "team": team,
                                     "blended_srs": round(blended, 4),
                                     "alpha": round(alpha, 4),
                                     "games_played": n})

    cum_file = CACHE / f"cum_srs_{season}.csv"
    with open(cum_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date","team","cum_srs","games_played"])
        writer.writeheader()
        writer.writerows(cum_rows)
    print(f"Wrote {len(cum_rows)} rows → {cum_file}")

    if blended_rows:
        blend_file = CACHE / f"blended_srs_{season}.csv"
        with open(blend_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date","team","blended_srs","alpha","games_played"])
            writer.writeheader()
            writer.writerows(blended_rows)
        print(f"Wrote {len(blended_rows)} rows → {blend_file}")
        if prior:
            print(f"  Blend: halflife={halflife} games, {len(prior)} teams with preseason priors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",   type=int,   default=2025)
    parser.add_argument("--halflife", type=float, default=BLEND_HALFLIFE)
    parser.add_argument("--no-blend", action="store_true")
    args = parser.parse_args()

    prior = {} if args.no_blend else load_preseason_prior(args.season)
    if prior:
        print(f"Loaded preseason priors for {len(prior)} teams")
    build_cum_srs(args.season, args.halflife, prior, args.no_blend)


if __name__ == "__main__":
    main()
