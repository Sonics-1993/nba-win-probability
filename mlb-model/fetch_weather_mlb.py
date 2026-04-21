"""
fetch_weather_mlb.py — Field-level game-day weather from MLB Stats API

One request per game-date fetches all games that day with:
  - condition:  'Roof Closed' | 'In Dome' | 'Clear' | 'Cloudy' | etc.
  - temp:       actual field temperature in °F
  - wind:       field-relative direction string ('12 mph, Out To CF')

Advantages over Open-Meteo:
  - Knows when retractable roofs are closed (no more treating Globe Life
    July games as 97°F outdoor when the stadium is 74°F climate-controlled)
  - Field-level measurement, not a nearby weather-station grid average
  - Wind direction is relative to the diamond — no bearing math needed

Writes weather_{season}.csv (same schema as fetch_weather.py output).

Usage:
    python3 fetch_weather_mlb.py --season 2025
    python3 fetch_weather_mlb.py --season 2025 --force
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import requests

BASE  = Path(__file__).parent
CACHE = BASE / "cache"
BASE_URL = "https://statsapi.mlb.com/api/v1"

# Conditions that mean the park is climate-controlled
DOME_CONDITIONS = {"roof closed", "in dome", "dome", "retractable roof closed"}

# Wind direction → tailwind multiplier (positive = blowing toward CF = more runs)
# Angles are approximate: LF/RF are ~45° off the CF axis
WIND_MULTIPLIERS = {
    "out to cf":  1.00,
    "in from cf": -1.00,
    "out to lf":   0.71,
    "out to rf":   0.71,
    "in from lf": -0.71,
    "in from rf": -0.71,
    "l to r":      0.00,
    "r to l":      0.00,
    "varies":      0.00,
    "calm":        0.00,
    "none":        0.00,
}

WIND_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mph,?\s*(.*)", re.IGNORECASE)


def parse_wind(wind_str: str) -> tuple[float, float]:
    """Return (wind_mph, tailwind_mph). tailwind > 0 = blowing out toward CF."""
    if not wind_str or wind_str.strip().lower() in ("", "calm", "none"):
        return 0.0, 0.0
    m = WIND_RE.match(wind_str.strip())
    if not m:
        return 0.0, 0.0
    speed = float(m.group(1))
    direction = m.group(2).strip().lower()
    multiplier = WIND_MULTIPLIERS.get(direction, 0.0)
    return round(speed, 1), round(speed * multiplier, 2)


def fetch_date(date: str) -> list[dict]:
    """Fetch all games on a date with weather hydration. Returns list of game dicts."""
    r = requests.get(
        f"{BASE_URL}/schedule",
        params={"date": date, "hydrate": "weather,venue", "sportId": 1},
        timeout=15,
    )
    r.raise_for_status()
    games = []
    for d in r.json().get("dates", []):
        games.extend(d.get("games", []))
    return games


def is_outdoor(condition: str) -> int:
    return 0 if condition.strip().lower() in DOME_CONDITIONS else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()
    s = args.season

    games_file = CACHE / f"games_{s}_raw.json"
    if not games_file.exists():
        print(f"Missing {games_file.name}"); return

    games      = json.loads(games_file.read_text())
    all_pks    = {g["gamePk"] for g in games}
    all_dates  = sorted({g["date"] for g in games})

    # Raw cache: {str(gamePk): {condition, temp_f, wind_mph, tailwind_mph, is_outdoor}}
    raw_file = CACHE / f"weather_mlb_raw_{s}.json"
    raw: dict[str, dict] = {}
    if raw_file.exists() and not args.force:
        raw = json.loads(raw_file.read_text())
        print(f"Resuming from cache: {len(raw)}/{len(all_pks)} games already fetched")

    fetched_pks = {int(k) for k in raw}
    dates_needed = [d for d in all_dates
                    if any(g["gamePk"] not in fetched_pks
                           for g in games if g["date"] == d)]

    print(f"Season {s}: {len(all_dates)} dates, {len(dates_needed)} need fetching")

    for i, date in enumerate(dates_needed):
        try:
            day_games = fetch_date(date)
            for g in day_games:
                pk = g.get("gamePk")
                if pk not in all_pks:
                    continue
                wx        = g.get("weather", {})
                condition = wx.get("condition", "")
                temp_str  = wx.get("temp", "")
                wind_str  = wx.get("wind", "")
                temp_f    = float(temp_str) if temp_str and temp_str.isdigit() else 0.0
                wind_mph, tailwind = parse_wind(wind_str)
                outdoor   = is_outdoor(condition)
                raw[str(pk)] = {
                    "condition":    condition,
                    "temp_f":       temp_f,
                    "wind_mph":     wind_mph if outdoor else 0.0,
                    "tailwind_mph": tailwind if outdoor else 0.0,
                    "is_outdoor":   outdoor,
                }
        except Exception as e:
            print(f"  WARN {date}: {e}")

        time.sleep(0.15)

        if (i + 1) % 50 == 0:
            raw_file.write_text(json.dumps(raw))
            print(f"  {i+1}/{len(dates_needed)} dates fetched, {len(raw)} games cached")

    raw_file.write_text(json.dumps(raw))
    print(f"Saved raw cache → {raw_file.name}")

    # Build CSV
    rows = []
    missing = 0
    for g in games:
        pk  = g["gamePk"]
        entry = raw.get(str(pk))
        if not entry:
            missing += 1
            rows.append({
                "game_pk": pk, "date": g["date"],
                "condition": "", "temp_f": "",
                "wind_mph": "", "tailwind_mph": "", "is_outdoor": "",
            })
            continue
        rows.append({
            "game_pk":      pk,
            "date":         g["date"],
            "condition":    entry["condition"],
            "temp_f":       entry["temp_f"],
            "wind_mph":     entry["wind_mph"],
            "tailwind_mph": entry["tailwind_mph"],
            "is_outdoor":   entry["is_outdoor"],
        })

    out = CACHE / f"weather_{s}.csv"
    fields = ["game_pk","date","condition","temp_f","wind_mph","tailwind_mph","is_outdoor"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {out}  (missing: {missing})")

    # Sanity
    outdoor_rows = [r for r in rows if str(r.get("is_outdoor")) == "1" and r.get("temp_f")]
    dome_rows    = [r for r in rows if str(r.get("is_outdoor")) == "0"]
    if outdoor_rows:
        avg_t = sum(float(r["temp_f"]) for r in outdoor_rows) / len(outdoor_rows)
        print(f"Outdoor games: {len(outdoor_rows)}  avg temp={avg_t:.1f}°F")
    print(f"Dome/roof-closed games: {len(dome_rows)}")

    # Show retractable-roof parks breakdown
    training = json.loads(games_file.read_text())
    from collections import defaultdict
    venue_map = {g["gamePk"]: g.get("venue", "") for g in training}
    retractable_venues = ["Globe Life Field", "Chase Field", "loanDepot park",
                          "American Family Field", "T-Mobile Park", "Truist Park"]
    print("\nRetractable roof breakdown:")
    for venue in retractable_venues:
        vgames = [r for r in rows if venue_map.get(r.get("game_pk")) == venue]
        closed = sum(1 for r in vgames if str(r.get("is_outdoor")) == "0")
        print(f"  {venue[:30]:30}  {len(vgames):3} games  {closed:3} roof-closed ({closed/len(vgames):.0%})" if vgames else f"  {venue}: no games")


if __name__ == "__main__":
    main()
