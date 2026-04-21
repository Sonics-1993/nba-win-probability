"""
fetch_pitcher_stats.py — Per-start pitcher stats, cumulative ERA (no leakage)

Usage:
    python3 fetch_pitcher_stats.py --season 2025
    python3 fetch_pitcher_stats.py --season 2024 --force
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import requests

BASE     = Path(__file__).parent
CACHE    = BASE / "cache"
BASE_URL = "https://statsapi.mlb.com/api/v1"
REPLACEMENT_ERA = 4.50


def fetch_pitching_line(game_pk: int, pitcher_id: int, side: str) -> dict | None:
    try:
        r = requests.get(f"{BASE_URL}/game/{game_pk}/boxscore", timeout=15)
        r.raise_for_status()
        bs      = r.json()
        player  = bs["teams"][side]["players"].get(f"ID{pitcher_id}", {})
        stats   = player.get("stats", {}).get("pitching", {})
        ip_str  = str(stats.get("inningsPitched", "0.0"))
        parts   = ip_str.split(".")
        ip      = int(parts[0]) + (int(parts[1]) / 3 if len(parts) > 1 and parts[1] else 0)
        return {"ip": ip, "er": stats.get("earnedRuns", 0)}
    except Exception:
        return None


def build_starts(season: int, games: list[dict], pitchers_raw: dict, force: bool) -> dict:
    out_file = CACHE / f"pitcher_starts_{season}.json"
    if out_file.exists() and not force:
        print(f"Cache hit: {out_file}")
        return json.loads(out_file.read_text())

    starts: dict[str, dict] = {}
    for i, g in enumerate(games):
        pk    = str(g["gamePk"])
        entry = pitchers_raw.get(pk, {})
        starts[pk] = {}
        for side_key, side in [("home_sp", "home"), ("away_sp", "away")]:
            sp = entry.get(side_key)
            if not sp or not sp.get("id"):
                continue
            line = fetch_pitching_line(g["gamePk"], sp["id"], side)
            starts[pk][side_key] = {"id": sp["id"], "name": sp["name"],
                                    "ip": line["ip"] if line else 0.0,
                                    "er": line["er"] if line else 0}
            time.sleep(0.03)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(games)} games...")

    out_file.write_text(json.dumps(starts, indent=2))
    print(f"Wrote pitcher starts → {out_file}")
    return starts


def build_cum_era(season: int, games: list[dict], starts: dict) -> None:
    outings: dict[int, list[tuple]] = defaultdict(list)
    names:   dict[int, str] = {}
    for g in games:
        for side_key in ("home_sp", "away_sp"):
            sp = starts.get(str(g["gamePk"]), {}).get(side_key)
            if sp:
                names[sp["id"]] = sp["name"]
                outings[sp["id"]].append((g["date"], sp.get("ip", 0.0), sp.get("er", 0)))

    rows = []
    for g in sorted(games, key=lambda x: x["date"]):
        for side_key in ("home_sp", "away_sp"):
            sp = starts.get(str(g["gamePk"]), {}).get(side_key)
            if not sp:
                continue
            prior    = [(d, ip, er) for d, ip, er in outings[sp["id"]] if d < g["date"]]
            total_ip = sum(ip for _, ip, _ in prior)
            total_er = sum(er for _, _, er in prior)
            # Shrink toward replacement level when sample is small
            shrink_ip = 20.0
            shrunk_era = ((total_er * 9 + REPLACEMENT_ERA * shrink_ip)
                          / (total_ip + shrink_ip)) if total_ip > 0 else REPLACEMENT_ERA
            rows.append({
                "date":         g["date"],
                "game_pk":      g["gamePk"],
                "side":         side_key.replace("_sp", ""),
                "pitcher_id":   sp["id"],
                "pitcher_name": sp["name"],
                "cum_era":      round(shrunk_era, 3),
                "prior_ip":     round(total_ip, 1),
                "prior_starts": len(prior),
            })

    out = CACHE / f"pitcher_era_{season}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date","game_pk","side","pitcher_id","pitcher_name","cum_era","prior_ip","prior_starts"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {out}")
    sample = [r for r in rows if r["prior_starts"] >= 5]
    if sample:
        print(f"Sanity: avg shrunk ERA (≥5 starts) = {sum(r['cum_era'] for r in sample)/len(sample):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()

    games    = json.loads((CACHE / f"games_{args.season}_raw.json").read_text())
    pitchers = json.loads((CACHE / f"games_{args.season}_pitchers.json").read_text())
    print(f"Loaded {len(games)} games, {len(pitchers)} pitcher entries")
    starts = build_starts(args.season, games, pitchers, args.force)
    build_cum_era(args.season, games, starts)


if __name__ == "__main__":
    main()
