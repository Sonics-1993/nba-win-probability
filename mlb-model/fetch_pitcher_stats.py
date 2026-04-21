"""
fetch_pitcher_stats.py — Per-start pitcher stats, cumulative ERA/FIP, last-3 form

Fetches boxscore once per game to get:
  - Starter: IP, ER, K, BB, HR  (for ERA and FIP)
  - Team batting: AB, H, BB, HR, TB  (for rolling OPS)

Usage:
    python3 fetch_pitcher_stats.py --season 2025
    python3 fetch_pitcher_stats.py --season 2025 --force
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
REPLACEMENT_ERA  = 4.50
REPLACEMENT_FIP  = 4.40
FIP_CONSTANT     = 3.15   # calibrates league-avg FIP ≈ league-avg ERA
SHRINK_IP        = 20.0
MIN_PRIOR_STARTS = 10    # minimum prior-season starts to trust FIP seed
LEAGUE_FIP       = 4.10   # MLB average FIP (used for year-to-year regression)
YOY_FIP_CORR     = 0.65   # year-to-year FIP correlation for starters


def load_prior_fip(season: int) -> dict[int, float]:
    """
    Return {pitcher_id: regressed prior-season FIP} for use as April shrinkage anchor.

    Applies year-to-year regression toward league mean to account for:
    - Mean reversion between seasons
    - Early-season ramp-up (pitchers not yet at peak form)
    """
    prior_file = CACHE / f"pitcher_era_{season - 1}.csv"
    if not prior_file.exists():
        return {}
    best: dict[int, tuple[int, float]] = {}  # pid -> (prior_starts, cum_fip)
    with open(prior_file, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("cum_fip"):
                continue
            pid     = int(row["pitcher_id"])
            starts  = int(row.get("prior_starts", 0))
            fip_val = float(row["cum_fip"])
            if pid not in best or starts > best[pid][0]:
                best[pid] = (starts, fip_val)
    result = {}
    for pid, (starts, raw_fip) in best.items():
        if starts < MIN_PRIOR_STARTS:
            continue
        # Regress toward league mean: predicted = corr*(prev - mean) + mean
        result[pid] = YOY_FIP_CORR * (raw_fip - LEAGUE_FIP) + LEAGUE_FIP
    print(f"  Prior-season FIP seeds: {len(result)} pitchers from pitcher_era_{season - 1}.csv "
          f"(min {MIN_PRIOR_STARTS} starts, regressed toward {LEAGUE_FIP:.2f})")
    return result


def fetch_game_data(game_pk: int, home_sp_id: int | None, away_sp_id: int | None) -> dict:
    """Single boxscore fetch per game. Returns pitcher lines + team batting."""
    r = requests.get(f"{BASE_URL}/game/{game_pk}/boxscore", timeout=15)
    r.raise_for_status()
    bs = r.json()

    def pitching_line(side: str, pitcher_id: int | None) -> dict:
        if not pitcher_id:
            return {}
        player = bs["teams"][side]["players"].get(f"ID{pitcher_id}", {})
        st = player.get("stats", {}).get("pitching", {})
        ip_str = str(st.get("inningsPitched", "0.0"))
        parts  = ip_str.split(".")
        ip = int(parts[0]) + (int(parts[1]) / 3 if len(parts) > 1 and parts[1] else 0)
        return {
            "ip": round(ip, 2),
            "er": int(st.get("earnedRuns",   0) or 0),
            "k":  int(st.get("strikeOuts",   0) or 0),
            "bb": int(st.get("baseOnBalls",  0) or 0),
            "hr": int(st.get("homeRuns",     0) or 0),
        }

    def batting_line(side: str) -> dict:
        bt = bs["teams"][side].get("teamStats", {}).get("batting", {})
        return {
            "ab": int(bt.get("atBats",      0) or 0),
            "h":  int(bt.get("hits",        0) or 0),
            "bb": int(bt.get("baseOnBalls", 0) or 0),
            "hr": int(bt.get("homeRuns",    0) or 0),
            "tb": int(bt.get("totalBases",  0) or 0),
        }

    return {
        "home_sp":      pitching_line("home", home_sp_id),
        "away_sp":      pitching_line("away", away_sp_id),
        "home_batting": batting_line("home"),
        "away_batting": batting_line("away"),
    }


def _needs_refresh(starts: dict) -> bool:
    """Check whether existing cache is missing K/BB/HR fields."""
    for pk, entry in starts.items():
        for side in ("home_sp", "away_sp"):
            sp = entry.get(side, {})
            if sp and "k" not in sp:
                return True
        break
    return False


def build_starts(season: int, games: list[dict], pitchers_raw: dict, force: bool) -> tuple[dict, dict]:
    """
    Returns (starts, team_batting).
      starts:       {game_pk: {home_sp: {id,name,ip,er,k,bb,hr}, away_sp: ...}}
      team_batting: {game_pk: {home: {ab,h,bb,hr,tb}, away: ...}}
    """
    starts_file  = CACHE / f"pitcher_starts_{season}.json"
    batting_file = CACHE / f"team_batting_{season}.json"

    if starts_file.exists() and batting_file.exists() and not force:
        existing = json.loads(starts_file.read_text())
        if not _needs_refresh(existing):
            print(f"Cache hit: {starts_file} and {batting_file}")
            return existing, json.loads(batting_file.read_text())

    starts:  dict[str, dict] = {}
    batting: dict[str, dict] = {}

    for i, g in enumerate(games):
        pk    = str(g["gamePk"])
        entry = pitchers_raw.get(pk, {})
        home_id = entry.get("home_sp", {}).get("id")
        away_id = entry.get("away_sp", {}).get("id")

        try:
            data = fetch_game_data(g["gamePk"], home_id, away_id)
        except Exception as e:
            print(f"  WARN {pk}: {e}")
            data = {"home_sp": {}, "away_sp": {}, "home_batting": {}, "away_batting": {}}

        starts[pk] = {}
        for side_key, sp_id in [("home_sp", home_id), ("away_sp", away_id)]:
            sp = entry.get(side_key, {})
            line = data.get(side_key, {})
            if sp_id and line:
                starts[pk][side_key] = {
                    "id": sp_id, "name": sp.get("name", ""),
                    **line,
                }

        batting[pk] = {
            "home": data["home_batting"],
            "away": data["away_batting"],
        }

        time.sleep(0.04)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(games)} games fetched...")
            starts_file.write_text(json.dumps(starts, indent=2))
            batting_file.write_text(json.dumps(batting, indent=2))

    starts_file.write_text(json.dumps(starts, indent=2))
    batting_file.write_text(json.dumps(batting, indent=2))
    print(f"Wrote {starts_file.name} and {batting_file.name}")
    return starts, batting


def fip(er_9: float, k: float, bb: float, hr: float, ip: float) -> float:
    if ip <= 0:
        return REPLACEMENT_FIP
    return (13 * hr + 3 * bb - 2 * k) / ip + FIP_CONSTANT


def build_cum_era(season: int, games: list[dict], starts: dict,
                  prior_fip: dict[int, float] | None = None) -> None:
    """Build pitcher_era_{season}.csv with ERA, FIP, last-3 ERA/FIP per start."""
    Outing = tuple  # (date, game_pk, ip, er, k, bb, hr)
    outings:  dict[int, list[Outing]] = defaultdict(list)
    names:    dict[int, str] = {}

    for g in games:
        for side_key in ("home_sp", "away_sp"):
            sp = starts.get(str(g["gamePk"]), {}).get(side_key)
            if sp and sp.get("id"):
                pid = sp["id"]
                names[pid] = sp.get("name", "")
                outings[pid].append((
                    g["date"], g["gamePk"],
                    float(sp.get("ip", 0) or 0),
                    int(sp.get("er", 0) or 0),
                    int(sp.get("k",  0) or 0),
                    int(sp.get("bb", 0) or 0),
                    int(sp.get("hr", 0) or 0),
                ))

    rows = []
    for g in sorted(games, key=lambda x: x["date"]):
        for side_key in ("home_sp", "away_sp"):
            sp = starts.get(str(g["gamePk"]), {}).get(side_key)
            if not sp or not sp.get("id"):
                continue
            pid   = sp["id"]
            prior = [(d, gpk, ip, er, k, bb, hr)
                     for d, gpk, ip, er, k, bb, hr in outings[pid]
                     if d < g["date"]]

            # Cumulative ERA (shrunk)
            tot_ip = sum(o[2] for o in prior)
            tot_er = sum(o[3] for o in prior)
            cum_era = ((tot_er * 9 + REPLACEMENT_ERA * SHRINK_IP)
                       / (tot_ip + SHRINK_IP))

            # Cumulative FIP (shrunk toward prior-season FIP when available, else replacement)
            tot_k  = sum(o[4] for o in prior)
            tot_bb = sum(o[5] for o in prior)
            tot_hr = sum(o[6] for o in prior)
            raw_fip  = fip(0, tot_k, tot_bb, tot_hr, tot_ip) if tot_ip > 0 else REPLACEMENT_FIP
            seed_fip = prior_fip.get(pid, REPLACEMENT_FIP) if prior_fip else REPLACEMENT_FIP
            cum_fip  = ((raw_fip * tot_ip + seed_fip * SHRINK_IP)
                        / (tot_ip + SHRINK_IP))

            # Last-3 starts
            last3 = prior[-3:] if len(prior) >= 3 else []
            if last3:
                l3_ip = sum(o[2] for o in last3)
                l3_er = sum(o[3] for o in last3)
                l3_k  = sum(o[4] for o in last3)
                l3_bb = sum(o[5] for o in last3)
                l3_hr = sum(o[6] for o in last3)
                last3_era = (l3_er * 9 / l3_ip) if l3_ip > 0 else REPLACEMENT_ERA
                last3_fip = fip(0, l3_k, l3_bb, l3_hr, l3_ip) if l3_ip > 0 else REPLACEMENT_FIP
            else:
                last3_era = REPLACEMENT_ERA
                last3_fip = REPLACEMENT_FIP

            rows.append({
                "date":         g["date"],
                "game_pk":      g["gamePk"],
                "side":         side_key.replace("_sp", ""),
                "pitcher_id":   pid,
                "pitcher_name": names.get(pid, ""),
                "cum_era":      round(cum_era,  3),
                "cum_fip":      round(cum_fip,  3),
                "last3_era":    round(last3_era, 3),
                "last3_fip":    round(last3_fip, 3),
                "prior_ip":     round(tot_ip, 1),
                "prior_starts": len(prior),
            })

    out = CACHE / f"pitcher_era_{season}.csv"
    fields = ["date","game_pk","side","pitcher_id","pitcher_name",
              "cum_era","cum_fip","last3_era","last3_fip","prior_ip","prior_starts"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {out}")

    sample = [r for r in rows if r["prior_starts"] >= 5]
    if sample:
        avg_era = sum(r["cum_era"] for r in sample) / len(sample)
        avg_fip = sum(r["cum_fip"] for r in sample) / len(sample)
        print(f"Sanity (≥5 starts): avg ERA={avg_era:.2f}  avg FIP={avg_fip:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()

    games    = json.loads((CACHE / f"games_{args.season}_raw.json").read_text())
    pitchers = json.loads((CACHE / f"games_{args.season}_pitchers.json").read_text())
    print(f"Loaded {len(games)} games, {len(pitchers)} pitcher entries")

    starts, _ = build_starts(args.season, games, pitchers, args.force)
    prior_fip = load_prior_fip(args.season)
    build_cum_era(args.season, games, starts, prior_fip=prior_fip)


if __name__ == "__main__":
    main()
