"""
fetch_weather.py — Historical game-time weather per MLB game (Open-Meteo, free)

Fetches one full-season request per stadium (30 total) instead of per-game.
Extracts 19:00 local values (night-game default). Retractable-roof stadiums
treated as outdoor; only fixed domes get is_outdoor=0.

Usage:
    python3 fetch_weather.py --season 2025
    python3 fetch_weather.py --season 2024
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import requests

BASE  = Path(__file__).parent
CACHE = BASE / "cache"

# (lat, lon, timezone, is_outdoor, park_out_bearing)
# park_out_bearing = degrees from N toward CF (0=N, 90=E). Wind FROM opposite = tailwind.
STADIUMS: dict[str, tuple] = {
    "American Family Field":          (43.0280, -87.9712, "America/Chicago",       1, 5),
    "Angel Stadium":                  (33.8003, -117.8827, "America/Los_Angeles",  1, 340),
    "Bristol Motor Speedway":         (36.5153, -82.2567, "America/New_York",      1, 0),
    "Busch Stadium":                  (38.6226, -90.1928, "America/Chicago",       1, 20),
    "Chase Field":                    (33.4455, -112.0667, "America/Phoenix",      1, 345),
    "Citi Field":                     (40.7571, -73.8458, "America/New_York",      1, 5),
    "Citizens Bank Park":             (39.9061, -75.1665, "America/New_York",      1, 10),
    "Comerica Park":                  (42.3390, -83.0485, "America/Detroit",       1, 15),
    "Coors Field":                    (39.7559, -104.9942, "America/Denver",       1, 335),
    "Daikin Park":                    (29.7570, -95.3555, "America/Chicago",       1, 5),
    "Dodger Stadium":                 (34.0739, -118.2400, "America/Los_Angeles",  1, 320),
    "Fenway Park":                    (42.3467, -71.0972, "America/New_York",      1, 95),
    "George M. Steinbrenner Field":   (27.9773, -82.5090, "America/New_York",      1, 355),
    "Globe Life Field":               (32.7473, -97.0827, "America/Chicago",       1, 10),
    "Great American Ball Park":       (39.0979, -84.5082, "America/New_York",      1, 5),
    "Journey Bank Ballpark":          (38.5802, -121.5005, "America/Los_Angeles",  1, 0),
    "Kauffman Stadium":               (39.0517, -94.4803, "America/Chicago",       1, 10),
    "Nationals Park":                 (38.8730, -77.0074, "America/New_York",      1, 5),
    "Oracle Park":                    (37.7786, -122.3893, "America/Los_Angeles",  1, 50),
    "Oriole Park at Camden Yards":    (39.2839, -76.6216, "America/New_York",      1, 30),
    "PNC Park":                       (40.4469, -80.0057, "America/New_York",      1, 350),
    "Petco Park":                     (32.7076, -117.1570, "America/Los_Angeles",  1, 30),
    "Progressive Field":              (41.4962, -81.6852, "America/New_York",      1, 10),
    "Rate Field":                     (41.8300, -87.6338, "America/Chicago",       1, 5),
    "Rogers Centre":                  (43.6414, -79.3894, "America/Toronto",       0, 0),
    "Sutter Health Park":             (38.5802, -121.5005, "America/Los_Angeles",  1, 0),
    "T-Mobile Park":                  (47.5914, -122.3325, "America/Los_Angeles",  1, 330),
    "Target Field":                   (44.9817, -93.2783, "America/Chicago",       1, 5),
    "Tokyo Dome":                     (35.7056, 139.7518,  "Asia/Tokyo",           0, 0),
    "Truist Park":                    (33.8907, -84.4677, "America/New_York",      1, 5),
    "Wrigley Field":                  (41.9484, -87.6553, "America/Chicago",       1, 43),
    "Yankee Stadium":                 (40.8296, -73.9262, "America/New_York",      1, 20),
    "loanDepot park":                 (25.7781, -80.2197, "America/New_York",      1, 350),
    # 2024 venue name aliases
    "Guaranteed Rate Field":          (41.8300, -87.6338, "America/Chicago",       1, 5),
    "Minute Maid Park":               (29.7570, -95.3555, "America/Chicago",       1, 5),
    "Tropicana Field":                (27.7683, -82.6534, "America/New_York",      0, 0),
    "Oakland Coliseum":               (37.7517, -122.2005, "America/Los_Angeles",  1, 15),
    "Rickwood Field":                 (33.5102, -86.8373, "America/Chicago",       1, 0),
    "Estadio Alfredo Harp Helu":      (19.4979, -99.0953, "America/Mexico_City",   1, 0),
    "London Stadium":                 (51.5386, -0.0164,  "Europe/London",         1, 0),
    "Gocheok Sky Dome":               (37.5009, 126.8674, "Asia/Seoul",            0, 0),
}

DEFAULT_HOUR = 19   # fallback if no start time available


def tailwind(wind_speed: float, wind_dir: float, bearing: int) -> float:
    """Component of wind blowing OUT toward CF. Positive = helps offense."""
    angle = math.radians(wind_dir - (bearing + 180))
    return round(wind_speed * math.cos(angle), 2)


def utc_to_local_hour(game_date_utc: str, tz: str) -> int:
    """Convert UTC ISO gameDate string to local hour integer."""
    try:
        from datetime import datetime, timezone as tz_mod
        import zoneinfo
        dt_utc = datetime.fromisoformat(game_date_utc.replace("Z", "+00:00"))
        dt_local = dt_utc.astimezone(zoneinfo.ZoneInfo(tz))
        return dt_local.hour
    except Exception:
        return DEFAULT_HOUR


def fetch_season(lat: float, lon: float, tz: str,
                 start: str, end: str) -> dict[str, dict[int, dict]]:
    """Fetch full season hourly data; return {date: {hour: {temp_f, wind_mph, wind_dir}}}."""
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
            "timezone": tz,
            "wind_speed_unit": "mph",
            "temperature_unit": "fahrenheit",
        },
        timeout=30,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    by_date: dict[str, dict] = {}
    for i, ts in enumerate(h["time"]):
        d, hr = ts[:10], int(ts[11:13])
        by_date.setdefault(d, {})[hr] = {
            "temp_f":   round(h["temperature_2m"][i], 1),
            "wind_mph": round(h["wind_speed_10m"][i], 1),
            "wind_dir": round(h["wind_direction_10m"][i], 1),
        }
    return by_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    s = args.season

    games_file = CACHE / f"games_{s}_raw.json"
    if not games_file.exists():
        print(f"Missing {games_file.name} — run fetch_mlb_games.py --season {s} first.")
        return

    games  = json.loads(games_file.read_text())
    dates  = sorted({g["date"] for g in games})
    start, end = dates[0], dates[-1]

    # Load game start times if available (for correct local hour lookup)
    times_file = CACHE / f"game_times_{s}.json"
    game_times: dict[int, dict] = {}
    if times_file.exists():
        game_times = {int(k): v for k, v in json.loads(times_file.read_text()).items()}
        print(f"Loaded start times for {len(game_times)} games")

    # Find unique venues in this season's games
    venues_used = {g["venue"] for g in games if g["venue"] in STADIUMS}
    venues_skip = {g["venue"] for g in games if g["venue"] not in STADIUMS}
    if venues_skip:
        print(f"Unknown venues (will skip): {venues_skip}")

    print(f"Season {s}: {start} → {end}, {len(venues_used)} venues to fetch")

    # Fetch one request per venue — store all hours
    wx_cache: dict[tuple, dict[int, dict]] = {}  # (venue, date) → {hour → wx}
    for i, venue in enumerate(sorted(venues_used)):
        lat, lon, tz, is_outdoor, bearing = STADIUMS[venue]
        print(f"  [{i+1}/{len(venues_used)}] {venue}...", end=" ", flush=True)
        try:
            by_date = fetch_season(lat, lon, tz, start, end)
            for date, hours in by_date.items():
                wx_cache[(venue, date)] = hours
            print(f"{len(by_date)} days")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)

    # Build output rows — one per game, using actual start hour
    rows = []
    skipped = 0
    for g in games:
        venue = g["venue"]
        key   = (venue, g["date"])
        if key not in wx_cache:
            skipped += 1
            continue
        hours_dict = wx_cache[key]
        lat, lon, tz, is_outdoor, bearing = STADIUMS.get(venue, (0, 0, "", 0, 0))

        # Determine local game hour from start time, fall back to DEFAULT_HOUR
        gt = game_times.get(g["gamePk"], {})
        if gt.get("gameDate") and tz:
            local_hour = utc_to_local_hour(gt["gameDate"], tz)
        else:
            local_hour = DEFAULT_HOUR
        wx = hours_dict.get(local_hour) or hours_dict.get(DEFAULT_HOUR) or next(iter(hours_dict.values()))

        tw = tailwind(wx["wind_mph"], wx["wind_dir"], bearing) if is_outdoor else 0.0
        rows.append({
            "game_pk":      g["gamePk"],
            "venue":        venue,
            "date":         g["date"],
            "game_hour":    local_hour,
            "temp_f":       wx["temp_f"],
            "wind_mph":     wx["wind_mph"] if is_outdoor else 0.0,
            "wind_dir":     wx["wind_dir"] if is_outdoor else 0.0,
            "tailwind_mph": tw,
            "is_outdoor":   is_outdoor,
        })

    out = CACHE / f"weather_{s}.csv"
    fields = ["game_pk","venue","date","game_hour","temp_f","wind_mph","wind_dir","tailwind_mph","is_outdoor"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {out}  (skipped {skipped} games)")
    outdoor = [r for r in rows if r["is_outdoor"]]
    if outdoor:
        avg_t = sum(r["temp_f"] for r in outdoor) / len(outdoor)
        avg_w = sum(r["wind_mph"] for r in outdoor) / len(outdoor)
        print(f"Outdoor sanity: avg temp={avg_t:.1f}°F  avg wind={avg_w:.1f}mph")


if __name__ == "__main__":
    main()
