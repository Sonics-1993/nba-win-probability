"""
NBA Win Probability Charts — Last 30 Days
Fetches all games over the past 30 days and plots home-team win probability.

Folder layout:
  graphs/
    YYYY-MM/
      Overtime/
      Comebacks/      (winner trailed by ≥15 pts at some point, regulation)
      Blowouts/       (final margin ≥20 pts, regulation)
      Close/          (final margin ≤5 pts, regulation)
      Standard/

Win probability is computed from the WinProbabilityPBP stats endpoint when
available, otherwise derived from CDN play-by-play via a logistic margin model.
"""

import os
import re
import time
import math
import requests
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import date, timedelta

warnings.filterwarnings("ignore", category=DeprecationWarning)

from nba_api.stats.endpoints import scoreboardv2, winprobabilitypbp
from nba_api.stats.library.http import STATS_HEADERS

GRAPHS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
DAYS_BACK   = 30
RETRY_WAIT  = 5    # seconds to wait after a rate-limit / transient error
MAX_RETRIES = 4

# Result bucket names (priority order applied in classify_game)
OVERTIME  = "Overtime"
COMEBACK  = "Comebacks"
BLOWOUT   = "Blowouts"
CLOSE     = "Close"
STANDARD  = "Standard"

COMEBACK_THRESHOLD = 15   # points down to count as a comeback
BLOWOUT_THRESHOLD  = 20   # final-margin points for a blowout
CLOSE_THRESHOLD    = 5    # final-margin points for a close game


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def with_retry(fn, *args, label="request", **kwargs):
    """Call fn(*args, **kwargs), retrying up to MAX_RETRIES times on error."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            msg = str(exc)
            # Distinguish rate-limit (429) from other transient errors
            if "429" in msg or "rate" in msg.lower():
                print(f"    Rate-limited on {label}. Waiting {RETRY_WAIT}s… (attempt {attempt}/{MAX_RETRIES})")
            else:
                print(f"    Error on {label}: {exc!r}. Retrying in {RETRY_WAIT}s… (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(RETRY_WAIT)


# ---------------------------------------------------------------------------
# Time / math helpers
# ---------------------------------------------------------------------------

def clock_to_seconds(clock_str):
    """'PT11M44.00S' → seconds remaining in period."""
    m = re.match(r"PT(\d+)M([\d.]+)S", clock_str)
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))


def elapsed_game_seconds(period, clock_str, pd=720, otd=300):
    """Total seconds elapsed in the game."""
    rem = clock_to_seconds(clock_str)
    if period <= 4:
        return (period - 1) * pd + (pd - rem)
    return 4 * pd + (period - 5) * otd + (otd - rem)


def total_game_seconds(max_period, pd=720, otd=300):
    if max_period <= 4:
        return 4 * pd
    return 4 * pd + (max_period - 4) * otd


def win_prob_from_margin(margin, seconds_remaining):
    """
    Logistic model: WP = sigmoid(margin / (0.5458 * sqrt(secs_remaining + 1))).
    Calibrated so a 3-pt lead with 30 s left ≈ 87 % WP.
    """
    if seconds_remaining <= 0:
        return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    sigma = 0.5458 * math.sqrt(seconds_remaining + 1)
    return 1.0 / (1.0 + math.exp(-1.7 * margin / sigma))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_game(final_margin, max_period, max_home_deficit, max_away_deficit):
    """
    Return one of the five bucket names.
    Priority: Overtime > Comeback > Blowout > Close > Standard.
    final_margin = home_score - away_score (positive → home won).
    """
    if max_period > 4:
        return OVERTIME

    home_won = final_margin > 0
    winner_max_deficit = max_home_deficit if home_won else max_away_deficit
    if winner_max_deficit >= COMEBACK_THRESHOLD:
        return COMEBACK

    if abs(final_margin) >= BLOWOUT_THRESHOLD:
        return BLOWOUT

    if abs(final_margin) <= CLOSE_THRESHOLD:
        return CLOSE

    return STANDARD


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _get_game_ids(game_date):
    sb = scoreboardv2.ScoreboardV2(game_date=game_date, timeout=30)
    d  = sb.get_dict()
    gh = d["resultSets"][0]
    headers = gh["headers"]
    games = []
    for row in gh["rowSet"]:
        g = dict(zip(headers, row))
        # Only include completed games (status 3 = Final)
        if g.get("GAME_STATUS_ID") != 3:
            continue
        games.append({
            "game_id":         g["GAME_ID"],
            "gamecode":        g["GAMECODE"],
            "home_team_id":    g["HOME_TEAM_ID"],
            "visitor_team_id": g["VISITOR_TEAM_ID"],
        })
    return games


def get_game_ids(game_date):
    return with_retry(_get_game_ids, game_date, label=f"scoreboard {game_date}")


def _fetch_tricodes(game_id):
    url = (f"https://cdn.nba.com/static/json/liveData/"
           f"boxscore/boxscore_{game_id}.json")
    hdrs = {
        "User-Agent": STATS_HEADERS["User-Agent"],
        "Accept":     "application/json",
        "Referer":    "https://www.nba.com/",
    }
    r = requests.get(url, headers=hdrs, timeout=20)
    r.raise_for_status()
    d = r.json()["game"]
    return d["homeTeam"]["teamTricode"], d["awayTeam"]["teamTricode"]


def get_team_tricodes(game_id):
    try:
        return with_retry(_fetch_tricodes, game_id, label=f"boxscore {game_id}")
    except Exception:
        return "HOME", "AWAY"


def fetch_win_prob_official(game_id):
    """
    Try the official WinProbabilityPBP endpoint.
    Returns (points, max_period, final_margin, max_home_deficit, max_away_deficit)
    or None on failure / empty data.
    No retries — this endpoint is frequently dead; we fall back to CDN quickly.
    """
    def _fetch():
        wp = winprobabilitypbp.WinProbabilityPBP(game_id=game_id, timeout=45)
        d  = wp.get_dict()
        rs = next(r for r in d["resultSets"] if r["name"] == "WinProbPBP")
        return rs["headers"], rs["rowSet"]

    try:
        headers, rows = _fetch()
        if not rows:
            return None

        points, max_period = [], 4
        margins = []
        for row in rows:
            r = dict(zip(headers, row))
            period = int(r["PERIOD"])
            max_period = max(max_period, period)
            rem = float(r["SECONDS_REMAINING"])
            elapsed = (period - 1) * 720 + (720 - rem) if period <= 4 \
                      else 4 * 720 + (period - 5) * 300 + (300 - rem)
            points.append((elapsed, float(r["HOME_PCT"])))
            margins.append(int(r["HOME_SCORE_MARGIN"]))

        points.sort()
        final_margin    = margins[-1] if margins else 0
        max_home_deficit = max((0, max(-m for m in margins)), default=0)
        max_away_deficit = max((0, max( m for m in margins)), default=0)
        return points, max_period, final_margin, max_home_deficit, max_away_deficit

    except Exception:
        return None


def _fetch_cdn_pbp(game_id):
    url = (f"https://cdn.nba.com/static/json/liveData/"
           f"playbyplay/playbyplay_{game_id}.json")
    hdrs = {
        "User-Agent": STATS_HEADERS["User-Agent"],
        "Accept":     "application/json",
        "Referer":    "https://www.nba.com/",
    }
    r = requests.get(url, headers=hdrs, timeout=30)
    r.raise_for_status()
    return r.json()["game"]["actions"]


def fetch_win_prob_from_cdn(game_id):
    """
    Fetch CDN play-by-play and compute win probability from score margin.
    Returns (points, max_period, final_margin, max_home_deficit, max_away_deficit).
    """
    actions = with_retry(_fetch_cdn_pbp, game_id, label=f"CDN PBP {game_id}")

    points, max_period = [], 4
    margins_by_elapsed = {}

    for act in actions:
        period     = int(act.get("period", 0))
        clock      = act.get("clock", "PT00M00.00S")
        score_home = act.get("scoreHome", "")
        score_away = act.get("scoreAway", "")
        if period < 1 or score_home == "" or score_away == "":
            continue
        max_period = max(max_period, period)
        elapsed    = elapsed_game_seconds(period, clock)
        margin     = int(score_home) - int(score_away)
        margins_by_elapsed[round(elapsed, 1)] = margin

    # Recompute WP now that we know max_period (fixes seconds_remaining for OT)
    total = total_game_seconds(max_period)
    seen  = {}
    for elapsed, margin in margins_by_elapsed.items():
        secs_remaining = total - elapsed
        seen[elapsed]  = win_prob_from_margin(margin, secs_remaining)

    points = sorted(seen.items())
    if not points or points[0][0] > 0:
        points.insert(0, (0.0, 0.5))

    all_margins     = list(margins_by_elapsed.values())
    final_margin    = all_margins[-1] if all_margins else 0
    max_home_deficit = max((0, max(-m for m in all_margins)), default=0) if all_margins else 0
    max_away_deficit = max((0, max( m for m in all_margins)), default=0) if all_margins else 0

    return points, max_period, final_margin, max_home_deficit, max_away_deficit


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_xtick_positions(max_period):
    positions, labels = [], []
    for q in range(1, 5):
        positions.append((q - 1) * 720)
        labels.append(f"Q{q}")
    for ot in range(1, max_period - 3):
        positions.append(4 * 720 + (ot - 1) * 300)
        labels.append(f"OT{ot}")
    positions.append(total_game_seconds(max_period))
    labels.append("End")
    return positions, labels


def plot_game(game_id, home_tri, away_tri, game_date, points, max_period, out_path):
    total = total_game_seconds(max_period)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#0e0e0e")
    fig.patch.set_facecolor("#1a1a2e")

    ax.fill_between(xs, ys, 0.5, where=[y > 0.5 for y in ys],
                    alpha=0.35, color="#00a86b", interpolate=True)
    ax.fill_between(xs, ys, 0.5, where=[y < 0.5 for y in ys],
                    alpha=0.35, color="#e63946", interpolate=True)

    ax.plot(xs, ys, color="#f0f0f0", linewidth=1.5, zorder=3)
    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--", zorder=2)

    tick_pos, tick_labels = make_xtick_positions(max_period)
    for pos in tick_pos[1:-1]:
        ax.axvline(pos, color="#444444", linewidth=0.8, linestyle=":", zorder=1)

    ax.set_xlim(0, total)
    ax.set_ylim(0, 1)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, color="#cccccc", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(colors="#cccccc")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444444")

    ax.set_xlabel("Game Time", color="#aaaaaa", fontsize=10, labelpad=8)
    ax.set_ylabel(f"{home_tri} Win Probability", color="#aaaaaa", fontsize=10, labelpad=8)
    ax.set_title(f"{away_tri} @ {home_tri}  —  {game_date}",
                 color="#ffffff", fontsize=13, fontweight="bold", pad=12)

    ax.text(0.01, 0.96, home_tri, transform=ax.transAxes,
            color="#00a86b", fontsize=10, fontweight="bold", va="top")
    ax.text(0.01, 0.04, away_tri, transform=ax.transAxes,
            color="#e63946", fontsize=10, fontweight="bold", va="bottom")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def output_path(game_date_str, bucket, matchup):
    """graphs/YYYY-MM/<Bucket>/<MATCHUP>.png"""
    month_folder = game_date_str[:7]          # "2026-03"
    folder = os.path.join(GRAPHS_DIR, month_folder, bucket)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{matchup}.png")


def main():
    today      = date.today()
    date_range = [today - timedelta(days=d) for d in range(1, DAYS_BACK + 1)]

    total_games = total_saved = total_skipped = total_errors = 0

    for game_date in date_range:
        date_str = game_date.strftime("%Y-%m-%d")
        print(f"\n{'─'*60}")
        print(f"Date: {date_str}")

        try:
            games = get_game_ids(date_str)
        except Exception as exc:
            print(f"  Could not fetch scoreboard: {exc}")
            continue

        if not games:
            print("  No completed games.")
            continue

        print(f"  {len(games)} completed game(s)")
        total_games += len(games)

        for game in games:
            game_id  = game["game_id"]
            gamecode = game["gamecode"]
            matchup  = gamecode.split("/")[-1] if "/" in gamecode else gamecode

            # Check if already processed under any bucket to allow resume
            month_folder = date_str[:7]
            already_done = any(
                os.path.exists(os.path.join(GRAPHS_DIR, month_folder, b, f"{matchup}.png"))
                for b in (OVERTIME, COMEBACK, BLOWOUT, CLOSE, STANDARD)
            )
            if already_done:
                print(f"  [{matchup}] already processed — skipping")
                total_skipped += 1
                continue

            print(f"  Processing {matchup} ({game_id})…", end=" ", flush=True)

            try:
                home_tri, away_tri = get_team_tricodes(game_id)

                # Try official endpoint, fall back to CDN
                result = fetch_win_prob_official(game_id)
                if result:
                    points, max_period, final_margin, mhd, mad = result
                    source = "official"
                else:
                    points, max_period, final_margin, mhd, mad = \
                        fetch_win_prob_from_cdn(game_id)
                    source = "CDN"

                bucket   = classify_game(final_margin, max_period, mhd, mad)
                out_path = output_path(date_str, bucket, matchup)
                plot_game(game_id, home_tri, away_tri, date_str,
                          points, max_period, out_path)

                winner = home_tri if final_margin > 0 else away_tri
                print(f"[{bucket}] margin={final_margin:+d} src={source} → {out_path}")
                total_saved += 1

            except Exception as exc:
                print(f"ERROR — {exc}")
                total_errors += 1

            time.sleep(0.6)   # polite CDN delay between games

    print(f"\n{'='*60}")
    print(f"Done.  Saved: {total_saved}  |  Skipped: {total_skipped}  "
          f"|  Errors: {total_errors}  |  Total games seen: {total_games}")
    print(f"Graphs root: {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
