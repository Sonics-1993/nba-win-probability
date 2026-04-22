#!/usr/bin/env python3
"""
Run-line underdog backtest — 2024 data.

For each game with a standard ±1.5 run line and a known O/U:
  - Identify the underdog (+1.5 side)
  - Check if they covered (didn't lose by 2+)
  - Compare actual cover rate vs implied probability from the opening price

Output: cover rates and implied probs bucketed by O/U.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

BASE  = Path(__file__).parent
CACHE = BASE / "cache"


def american_to_prob(price: float) -> float:
    """Convert American odds to implied probability (no vig)."""
    if price > 0:
        return 100.0 / (price + 100.0)
    else:
        return -price / (-price + 100.0)


def load_csv(path) -> list[dict]:
    return list(csv.DictReader(open(path)))


def main():
    games_raw = {
        (g["date"], g["home_abbr"], g["away_abbr"]): g
        for g in json.loads((CACHE / "games_2024_raw.json").read_text())
    }

    totals = {
        (r["date"], r["home"], r["away"]): r
        for r in load_csv(CACHE / "totals_history_2024.csv")
    }

    odds = load_csv(CACHE / "odds_history_2024.csv")

    records = []
    for row in odds:
        date, home, away = row["date"], row["home"], row["away"]
        pt_str = row["open_rl_point"]
        ph_str = row["open_rl_price_home"]
        pa_str = row["open_rl_price_away"]

        # only standard ±1.5 run lines with prices
        if pt_str not in ("-1.50", "1.50") or not ph_str or not pa_str:
            continue

        home_pt = float(pt_str)
        home_price = float(ph_str)
        away_price = float(pa_str)

        game = games_raw.get((date, home, away))
        if not game:
            continue

        tot = totals.get((date, home, away), {})
        ou_str = tot.get("open_ou", "")
        if not ou_str:
            continue
        ou = float(ou_str)

        home_runs = game["home_runs"]
        away_runs = game["away_runs"]
        run_diff  = home_runs - away_runs   # positive = home won

        # who is the underdog?
        if home_pt > 0:
            # home is underdog (+1.5), away is favorite (-1.5)
            dog_team    = home
            dog_is_home = True
            dog_price   = home_price
        else:
            # away is underdog (+1.5), home is favorite (-1.5)
            dog_team    = away
            dog_is_home = False
            dog_price   = away_price

        # underdog covers if the game is within 1 run in their favour
        # home dog covers if run_diff >= -1 (not a 2+ run home loss)
        # away dog covers if run_diff <= 1  (not a 2+ run away loss)
        if dog_is_home:
            covered = run_diff >= -1
        else:
            covered = run_diff <= 1

        implied_prob = american_to_prob(dog_price)

        records.append({
            "date":        date,
            "home":        home,
            "away":        away,
            "dog":         dog_team,
            "dog_price":   dog_price,
            "implied_prob":implied_prob,
            "ou":          ou,
            "covered":     covered,
        })

    print(f"\nTotal games with standard RL + O/U: {len(records)}")

    # Overall stats
    n_cover = sum(r["covered"] for r in records)
    avg_impl = sum(r["implied_prob"] for r in records) / len(records)
    print(f"Underdog cover rate (all): {n_cover}/{len(records)} = {n_cover/len(records)*100:.1f}%")
    print(f"Average implied probability: {avg_impl*100:.1f}%")
    print(f"Edge: {(n_cover/len(records) - avg_impl)*100:+.1f} pp\n")

    # Bucket by O/U
    print("O/U bucket  | N games | Cover% | Avg implied | Edge (pp)")
    print("-" * 60)
    buckets = [
        ("≤7.0",  lambda r: r["ou"] <= 7.0),
        ("7.5",   lambda r: r["ou"] == 7.5),
        ("8.0",   lambda r: r["ou"] == 8.0),
        ("8.5",   lambda r: r["ou"] == 8.5),
        ("9.0",   lambda r: r["ou"] == 9.0),
        ("9.5",   lambda r: r["ou"] == 9.5),
        ("≥10.0", lambda r: r["ou"] >= 10.0),
    ]
    for label, fn in buckets:
        sub = [r for r in records if fn(r)]
        if not sub:
            continue
        n = len(sub)
        cov = sum(r["covered"] for r in sub) / n
        impl = sum(r["implied_prob"] for r in sub) / n
        edge = (cov - impl) * 100
        print(f"{label:<12}| {n:>7} | {cov*100:>5.1f}% | {impl*100:>10.1f}% | {edge:>+7.1f} pp")

    # Underdog price buckets for 8.5 O/U games (non-overlapping, bettor-friendly = less negative)
    print("\n--- 8.5 O/U breakdown by underdog price ---")
    sub85 = [r for r in records if r["ou"] == 8.5]
    price_buckets = [
        ("better than -130", lambda r: r["dog_price"] > -130),
        ("-130 to -145",     lambda r: -145 <= r["dog_price"] <= -130),
        ("-146 to -160",     lambda r: -160 <= r["dog_price"] < -145),
        ("worse than -160",  lambda r: r["dog_price"] < -160),
    ]
    for label, fn in price_buckets:
        sub = [r for r in sub85 if fn(r)]
        if not sub:
            continue
        n = len(sub)
        cov = sum(r["covered"] for r in sub) / n
        impl = sum(r["implied_prob"] for r in sub) / n
        avg_price = sum(r["dog_price"] for r in sub) / n
        print(f"  {label:<22} n={n:>3}  cover={cov*100:.1f}%  implied={impl*100:.1f}%  edge={+(cov-impl)*100:+.1f} pp  avg_price={avg_price:.0f}")

    # Median underdog price by O/U bucket
    print("\n--- Median underdog price by O/U ---")
    sorted_ou = sorted(set(r["ou"] for r in records))
    for ou in sorted_ou:
        sub = [r for r in records if r["ou"] == ou]
        prices = sorted(r["dog_price"] for r in sub)
        n = len(prices)
        med = prices[n // 2]
        cov = sum(r["covered"] for r in sub) / n
        impl = american_to_prob(med)
        print(f"  O/U {ou:4.1f}: n={n:>4}  median_price={med:>5.0f}  cover={cov*100:.1f}%  implied={impl*100:.1f}%  edge={+(cov-impl)*100:+.1f} pp")


if __name__ == "__main__":
    main()
