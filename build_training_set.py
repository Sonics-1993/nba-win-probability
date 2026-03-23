"""
build_training_set.py
─────────────────────────────────────────────────────────────────────────────
1. Re-discovers the 252 processed games and generates partial graph PNGs
   for End-of-Q1, End-of-Q2, End-of-Q3 saved under:
       snapshots/YYYY-MM/<Bucket>/<MATCHUP>_Q{n}.png

2. Uses OpenCV to extract a "visual signature" from each partial graph:
   - Isolates the white WP line by colour thresholding
   - Computes slope, volatility, 50%-crossings, max-excursion, trend-frac

3. Writes training_index.json linking every quarter snapshot to its
   OpenCV features + raw computed features + final game result bucket.

4. Prints a statistical comparison of Comeback vs Blowout patterns,
   highlighting which Q2 signals appear earliest.
"""

import os, re, json, time, math, warnings
import numpy as np
import cv2
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import date, timedelta
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=DeprecationWarning)
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.library.http import STATS_HEADERS

# ── paths & constants ────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR    = os.path.join(BASE_DIR, "graphs")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")
INDEX_PATH    = os.path.join(BASE_DIR, "training_index.json")

DAYS_BACK    = 30
RETRY_WAIT   = 5
MAX_RETRIES  = 4

OVERTIME = "Overtime"
COMEBACK = "Comebacks"
BLOWOUT  = "Blowouts"
CLOSE    = "Close"
STANDARD = "Standard"
ALL_BUCKETS = [OVERTIME, COMEBACK, BLOWOUT, CLOSE, STANDARD]

QUARTERS = {"Q1": 720, "Q2": 1440, "Q3": 2160}


# ── retry wrapper ────────────────────────────────────────────────────────────
def with_retry(fn, *args, label="", **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            is_rate = "429" in str(exc) or "rate" in str(exc).lower()
            tag = "Rate-limited" if is_rate else "Error"
            print(f"    [{label}] {tag}: {exc!r}  → retry in {RETRY_WAIT}s "
                  f"({attempt}/{MAX_RETRIES})")
            time.sleep(RETRY_WAIT)


# ── time / probability helpers ───────────────────────────────────────────────
def clock_to_seconds(s):
    m = re.match(r"PT(\d+)M([\d.]+)S", s)
    return int(m.group(1)) * 60 + float(m.group(2)) if m else 0.0

def elapsed(period, clock, pd=720, otd=300):
    r = clock_to_seconds(clock)
    return (period-1)*pd + pd - r if period <= 4 else 4*pd + (period-5)*otd + otd - r

def total_secs(max_period, pd=720, otd=300):
    return 4*pd if max_period <= 4 else 4*pd + (max_period-4)*otd

def wp_from_margin(margin, secs_rem):
    if secs_rem <= 0:
        return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    sigma = 0.5458 * math.sqrt(secs_rem + 1)
    return 1.0 / (1.0 + math.exp(-1.7 * margin / sigma))


# ── data fetching ────────────────────────────────────────────────────────────
def get_games_for_date(date_str):
    def _f():
        sb = scoreboardv2.ScoreboardV2(game_date=date_str, timeout=30)
        gh = sb.get_dict()["resultSets"][0]
        hdr = gh["headers"]
        out = []
        for row in gh["rowSet"]:
            g = dict(zip(hdr, row))
            if g.get("GAME_STATUS_ID") != 3:
                continue
            code = g["GAMECODE"].split("/")[-1] if "/" in g["GAMECODE"] else g["GAMECODE"]
            out.append({"game_id": g["GAME_ID"], "matchup": code})
        return out
    return with_retry(_f, label=f"scoreboard {date_str}")

def fetch_pbp(game_id):
    def _f():
        url = (f"https://cdn.nba.com/static/json/liveData/"
               f"playbyplay/playbyplay_{game_id}.json")
        r = requests.get(url, headers={
            "User-Agent": STATS_HEADERS["User-Agent"],
            "Accept": "application/json", "Referer": "https://www.nba.com/"
        }, timeout=30)
        r.raise_for_status()
        return r.json()["game"]["actions"]
    return with_retry(_f, label=f"PBP {game_id}")

def fetch_tricodes(game_id):
    def _f():
        url = (f"https://cdn.nba.com/static/json/liveData/"
               f"boxscore/boxscore_{game_id}.json")
        r = requests.get(url, headers={
            "User-Agent": STATS_HEADERS["User-Agent"],
            "Accept": "application/json", "Referer": "https://www.nba.com/"
        }, timeout=20)
        r.raise_for_status()
        d = r.json()["game"]
        return d["homeTeam"]["teamTricode"], d["awayTeam"]["teamTricode"]
    try:
        return with_retry(_f, label=f"boxscore {game_id}")
    except Exception:
        return "HOME", "AWAY"

def pbp_to_series(actions):
    """Return ({elapsed: {margin, wp}}, max_period)."""
    max_period = 4
    raw = {}
    for act in actions:
        p  = int(act.get("period", 0))
        cl = act.get("clock", "PT00M00.00S")
        sh = act.get("scoreHome", "")
        sa = act.get("scoreAway", "")
        if p < 1 or sh == "" or sa == "":
            continue
        max_period = max(max_period, p)
        e = round(elapsed(p, cl), 1)
        raw[e] = int(sh) - int(sa)

    total = total_secs(max_period)
    series = {}
    for e in sorted(raw):
        margin = raw[e]
        series[e] = {"margin": margin, "wp": wp_from_margin(margin, total - e)}
    if not series or min(series) > 0:
        series[0.0] = {"margin": 0, "wp": 0.5}
    return dict(sorted(series.items())), max_period


# ── snapshot generation ───────────────────────────────────────────────────────
def xtick_pos(max_period):
    pos, lbl = [], []
    for q in range(1, 5):
        pos.append((q-1)*720); lbl.append(f"Q{q}")
    for ot in range(1, max_period-3):
        pos.append(4*720 + (ot-1)*300); lbl.append(f"OT{ot}")
    pos.append(total_secs(max_period)); lbl.append("End")
    return pos, lbl

def generate_snapshot(game_id, home_tri, away_tri, game_date,
                       series, max_period, cutoff, out_path):
    """
    Render the WP line up to `cutoff` seconds.
    The x-axis always spans the full game; the greyed zone shows remaining time.
    Saves a fixed 1800×750 px PNG (no tight-crop) so the axes bbox is reliable.
    Returns the axes pixel bbox dict for OpenCV.
    """
    total = total_secs(max_period)
    partial = {e: v for e, v in series.items() if e <= cutoff}
    if partial:
        last_e = max(partial)
        partial[float(cutoff)] = series.get(float(cutoff), partial[last_e])
    xs = sorted(partial)
    ys = [partial[e]["wp"] for e in xs]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#0e0e0e")
    fig.patch.set_facecolor("#1a1a2e")

    ax.fill_between(xs, ys, 0.5, where=[y > 0.5 for y in ys],
                    alpha=0.35, color="#00a86b", interpolate=True)
    ax.fill_between(xs, ys, 0.5, where=[y < 0.5 for y in ys],
                    alpha=0.35, color="#e63946", interpolate=True)
    ax.plot(xs, ys, color="#f0f0f0", linewidth=1.5, zorder=3)
    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--", zorder=2)

    # Shade the future
    ax.axvspan(cutoff, total, color="#2a2a2a", alpha=0.7, zorder=0)
    ax.axvline(cutoff, color="#ffcc00", linewidth=1.4, linestyle="--", zorder=4)

    pos, lbl = xtick_pos(max_period)
    for p in pos[1:-1]:
        ax.axvline(p, color="#444444", linewidth=0.8, linestyle=":", zorder=1)

    ax.set_xlim(0, total)
    ax.set_ylim(0, 1)
    ax.set_xticks(pos)
    ax.set_xticklabels(lbl, color="#cccccc", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(colors="#cccccc")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444444")
    ax.set_xlabel("Game Time", color="#aaaaaa", fontsize=10, labelpad=8)
    ax.set_ylabel(f"{home_tri} Win Probability", color="#aaaaaa", fontsize=10, labelpad=8)

    q_name = {720: "End of Q1", 1440: "End of Q2", 2160: "End of Q3"}.get(cutoff, "")
    ax.set_title(f"{away_tri} @ {home_tri}  —  {game_date}  [{q_name}]",
                 color="#ffffff", fontsize=12, fontweight="bold", pad=12)
    ax.text(0.01, 0.96, home_tri, transform=ax.transAxes,
            color="#00a86b", fontsize=10, fontweight="bold", va="top")
    ax.text(0.01, 0.04, away_tri, transform=ax.transAxes,
            color="#e63946", fontsize=10, fontweight="bold", va="bottom")

    plt.tight_layout()

    # Capture axes bbox BEFORE saving (no bbox_inches crop so coords stay valid)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox     = ax.get_window_extent(renderer=renderer)
    fig_h_px = fig.get_figheight() * fig.dpi   # 750
    ax_bbox  = {
        "x0": int(bbox.x0),
        "y0": int(fig_h_px - bbox.y1),   # flip: renderer y=0 is bottom
        "x1": int(bbox.x1),
        "y1": int(fig_h_px - bbox.y0),
    }

    fig.savefig(out_path, dpi=150)          # fixed 1800×750, no tight-crop
    plt.close(fig)
    return ax_bbox


# ── OpenCV feature extraction ────────────────────────────────────────────────
def extract_cv_features(img_path, ax_bbox, cutoff, total):
    """
    Load the partial-graph PNG, crop to the plot area, isolate the white WP
    line by colour thresholding, and derive visual features.

    Strategy
    --------
    • White line: BGR channels all > 200.
    • For each x-column in the active zone (before the yellow cutoff line),
      take the median y-position of white pixels → convert to WP (0-1).
    • Feature set: slope, volatility, final_wp, mean_wp, 50%-crossings,
      max_excursion, trend_frac, r².
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    h_img, w_img = img.shape[:2]
    x0 = max(0, ax_bbox["x0"])
    y0 = max(0, ax_bbox["y0"])
    x1 = min(w_img, ax_bbox["x1"])
    y1 = min(h_img, ax_bbox["y1"])
    plot = img[y0:y1, x0:x1]
    ph, pw = plot.shape[:2]
    if ph < 10 or pw < 10:
        return None

    # ── isolate white line ──────────────────────────────────────────────────
    b = plot[:, :, 0].astype(np.int32)
    g = plot[:, :, 1].astype(np.int32)
    r = plot[:, :, 2].astype(np.int32)
    white_mask = (r > 200) & (g > 200) & (b > 200)

    # Exclude the greyed-out future region (right of the cutoff line)
    cutoff_col = int(pw * cutoff / total)
    white_mask[:, cutoff_col:] = False

    # ── scan columns → WP values ────────────────────────────────────────────
    wp_values = []
    for col in range(cutoff_col):
        whites = np.where(white_mask[:, col])[0]
        if len(whites) == 0:
            continue
        y_med = float(np.median(whites))
        # y=0 → top of plot → WP=1.0; y=ph → bottom → WP=0.0
        wp = np.clip(1.0 - y_med / ph, 0.0, 1.0)
        wp_values.append(wp)

    if len(wp_values) < 20:     # need a reasonable sample
        return None

    wp  = np.array(wp_values)
    diffs = np.diff(wp)
    xs    = np.linspace(0.0, 1.0, len(wp))

    # Linear regression for slope / R²
    slope, _, r_val, _, _ = sp_stats.linregress(xs, wp)

    # 50% crossings
    signs     = np.sign(wp - 0.5)
    crossings = int(np.sum(np.abs(np.diff(signs)) > 0))

    # Fraction of diffs aligned with overall slope direction
    trend_frac = (float(np.mean(np.sign(diffs) == np.sign(slope)))
                  if slope != 0 else 0.5)

    return {
        "slope":         round(float(slope), 5),
        "volatility":    round(float(np.std(diffs)), 5),
        "final_wp":      round(float(wp[-1]), 3),
        "mean_wp":       round(float(np.mean(wp)), 3),
        "crossings":     crossings,
        "max_excursion": round(float(np.max(np.abs(wp - 0.5))), 3),
        "trend_frac":    round(trend_frac, 3),
        "r_squared":     round(float(r_val ** 2), 3),
        "n_pixels":      len(wp_values),
    }


# ── raw data features (cross-validation baseline) ────────────────────────────
def compute_raw_features(series, q_start, q_end):
    """Compute WP features directly from the raw data series for one quarter."""
    window = {e: v for e, v in series.items() if q_start <= e <= q_end}
    if len(window) < 2:
        return None
    es  = sorted(window)
    wp  = np.array([window[e]["wp"]     for e in es])
    mgn = np.array([window[e]["margin"] for e in es])
    xs  = np.linspace(0.0, 1.0, len(wp))
    slope, _, r_val, _, _ = sp_stats.linregress(xs, wp)
    diffs = np.diff(wp)
    signs = np.sign(wp - 0.5)
    return {
        "slope":         round(float(slope), 5),
        "volatility":    round(float(np.std(diffs)), 5),
        "final_wp":      round(float(wp[-1]), 3),
        "mean_wp":       round(float(np.mean(wp)), 3),
        "crossings":     int(np.sum(np.abs(np.diff(signs)) > 0)),
        "max_excursion": round(float(np.max(np.abs(wp - 0.5))), 3),
        "max_margin":    int(np.max(mgn)),
        "min_margin":    int(np.min(mgn)),
        "r_squared":     round(float(r_val ** 2), 3),
    }


# ── pattern analysis ─────────────────────────────────────────────────────────
def analyze_patterns(index):
    cb = [g for g in index if g["bucket"] == COMEBACK]
    bl = [g for g in index if g["bucket"] == BLOWOUT]

    def qvals(games, quarter, key):
        return [g["quarters"][quarter]["cv"][key]
                for g in games
                if quarter in g.get("quarters", {})
                and g["quarters"][quarter].get("cv")
                and key in g["quarters"][quarter]["cv"]]

    print(f"\n{'═'*66}")
    print(f"  VISUAL PATTERN ANALYSIS  ·  Comebacks ({len(cb)}) vs Blowouts ({len(bl)})")
    print(f"{'═'*66}")

    metrics = [
        ("volatility",    "Volatility",       "std of WP diffs — higher = choppier"),
        ("slope",         "Slope",            "+ve = home gaining, −ve = losing"),
        ("final_wp",      "Final WP",         "home win-prob at end of quarter"),
        ("crossings",     "50% Crossings",    "lead changes"),
        ("max_excursion", "Max Excursion",    "furthest swing from 50%"),
        ("trend_frac",    "Trend Consistency","1.0 = perfectly one-directional"),
        ("r_squared",     "R²",               "how linear the WP curve is"),
    ]

    for q in ("Q1", "Q2", "Q3"):
        print(f"\n  ┌── {q} {'─'*54}")
        print(f"  │  {'Metric':<22} {'Comebacks':>10}  {'Blowouts':>10}  {'Δ (C−B)':>10}")
        print(f"  │  {'─'*22} {'─'*10}  {'─'*10}  {'─'*10}")
        for key, label, note in metrics:
            cv = qvals(cb, q, key)
            bv = qvals(bl, q, key)
            if not cv or not bv:
                continue
            cm, bm = np.mean(cv), np.mean(bv)
            d = cm - bm
            arrow = "▲" if d > 0.0001 else ("▼" if d < -0.0001 else "─")
            print(f"  │  {label:<22} {cm:>10.4f}  {bm:>10.4f}  {arrow}{abs(d):>9.4f}  # {note}")
        print(f"  └{'─'*57}")

    # ── headline findings ────────────────────────────────────────────────────
    print(f"\n  ╔══ EARLY-WARNING SIGNALS AT Q2 {'═'*33}╗")

    findings = []
    for key, label, note in metrics:
        cv = qvals(cb, "Q2", key)
        bv = qvals(bl, "Q2", key)
        if not cv or not bv:
            continue
        cm, bm = np.mean(cv), np.mean(bv)
        d = cm - bm
        pct = d / max(abs(bm), 1e-9) * 100
        direction = "HIGHER" if d > 0 else "lower"
        findings.append((abs(pct), label, cm, bm, pct, note))

    findings.sort(reverse=True)   # biggest relative difference first
    for _, label, cm, bm, pct, note in findings:
        bar = "★" if abs(pct) > 15 else " "
        print(f"  ║ {bar} {label:<20}  CB={cm:.4f}  BL={bm:.4f}  "
              f"Δ={pct:+.1f}%   {note}")

    print(f"  ╚{'═'*63}╝")

    # ── interpretation ───────────────────────────────────────────────────────
    print(f"\n  INTERPRETATION")
    print(f"  {'─'*62}")

    v_cb = qvals(cb, "Q2", "volatility")
    v_bl = qvals(bl, "Q2", "volatility")
    c_cb = qvals(cb, "Q2", "crossings")
    c_bl = qvals(bl, "Q2", "crossings")
    w_cb = qvals(cb, "Q2", "final_wp")
    w_bl = qvals(bl, "Q2", "final_wp")
    s_cb = qvals(cb, "Q2", "slope")
    s_bl = qvals(bl, "Q2", "slope")

    if v_cb and v_bl:
        if np.mean(v_cb) > np.mean(v_bl):
            print("  • Q2 line is CHOPPIER in Comeback games → neither team has")
            print("    control at half-time, keeping a comeback structurally possible.")
        else:
            print("  • Q2 line is SMOOTHER in Comeback games → the trailing team is")
            print("    absorbing a steady deficit before making its late run.")

    if c_cb and c_bl:
        if np.mean(c_cb) > np.mean(c_bl):
            print("  • More 50%-crossings by Q2 in Comebacks → frequent lead changes")
            print("    early; no dominant run has separated the teams yet.")
        else:
            print("  • Fewer 50%-crossings by Q2 in Comebacks → one team leads all")
            print("    first half, then gets overtaken — classic slow-burn comeback.")

    if w_cb and w_bl:
        wm_cb = np.mean(w_cb)
        print(f"  • Average Q2 WP in Comeback games = {wm_cb:.1%}  "
              f"({'home leads at half' if wm_cb > 0.5 else 'home trails at half'}).")
        if wm_cb < 0.45:
            print("    Home teams in Comeback games are typically TRAILING at halftime,")
            print("    then recover in Q3/Q4.")

    if s_cb and s_bl:
        sm_cb, sm_bl = np.mean(s_cb), np.mean(s_bl)
        print(f"  • Q2 slope: Comebacks {sm_cb:+.5f}  Blowouts {sm_bl:+.5f}")
        if abs(sm_cb) < abs(sm_bl):
            print("    Flatter Q2 slope in Comebacks confirms the game is still in")
            print("    flux — no one is running away with it yet.")
        else:
            print("    Steeper Q2 slope in Comebacks indicates a team is already")
            print("    in momentum-building mode before the decisive Q4 push.")

    print(f"\n  (★ = strongest early-warning signal  |  Δ = Comeback minus Blowout)")
    print(f"{'═'*66}\n")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    today      = date.today()
    date_range = [today - timedelta(days=d) for d in range(1, DAYS_BACK + 1)]

    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    index            = []
    n_games          = 0
    n_new_snapshots  = 0
    n_reused         = 0
    n_errors         = 0

    for game_date in date_range:
        ds    = game_date.strftime("%Y-%m-%d")
        month = ds[:7]

        try:
            games = get_games_for_date(ds)
        except Exception as exc:
            print(f"  Skipping {ds}: {exc}")
            continue

        for game in games:
            game_id = game["game_id"]
            matchup = game["matchup"]

            # Find which bucket this game was assigned in the main script
            bucket = next(
                (b for b in ALL_BUCKETS
                 if os.path.exists(
                     os.path.join(GRAPHS_DIR, month, b, f"{matchup}.png"))),
                None,
            )
            if bucket is None:
                continue    # wasn't processed / not a finished game

            n_games += 1
            print(f"  {ds}  {matchup:<8} [{bucket:<8}]", end="  ", flush=True)

            try:
                actions            = fetch_pbp(game_id)
                series, max_period = pbp_to_series(actions)
                home_tri, away_tri = fetch_tricodes(game_id)
            except Exception as exc:
                print(f"SKIP — {exc}")
                n_errors += 1
                continue

            total       = total_secs(max_period)
            final_entry = list(series.values())[-1]

            game_rec = {
                "game_id":      game_id,
                "matchup":      matchup,
                "date":         ds,
                "bucket":       bucket,
                "max_period":   max_period,
                "final_margin": final_entry["margin"],
                "quarters":     {},
            }

            for q_label, cutoff in QUARTERS.items():
                # Skip if the game didn't reach this cutoff meaningfully
                if cutoff > total or max(series) < cutoff * 0.75:
                    continue

                snap_dir  = os.path.join(SNAPSHOTS_DIR, month, bucket)
                os.makedirs(snap_dir, exist_ok=True)
                snap_path = os.path.join(snap_dir, f"{matchup}_{q_label}.png")
                bbox_path = snap_path.replace(".png", "_bbox.json")

                # Generate snapshot (or reload existing)
                if os.path.exists(snap_path) and os.path.exists(bbox_path):
                    with open(bbox_path) as f:
                        ax_bbox = json.load(f)
                    n_reused += 1
                else:
                    ax_bbox = generate_snapshot(
                        game_id, home_tri, away_tri, ds,
                        series, max_period, cutoff, snap_path,
                    )
                    with open(bbox_path, "w") as f:
                        json.dump(ax_bbox, f)
                    n_new_snapshots += 1

                # OpenCV feature extraction from the PNG
                cv_feats  = extract_cv_features(snap_path, ax_bbox, cutoff, total)

                # Raw feature cross-validation (computed directly from data)
                q_start   = cutoff - 720
                raw_feats = compute_raw_features(series, q_start, cutoff)

                game_rec["quarters"][q_label] = {
                    "snapshot": os.path.relpath(snap_path, BASE_DIR),
                    "cv":       cv_feats,
                    "raw":      raw_feats,
                }

            index.append(game_rec)
            quarters_done = ", ".join(game_rec["quarters"].keys())
            print(f"✓  [{quarters_done}]")

            time.sleep(0.35)

    # ── JSON index ────────────────────────────────────────────────────────────
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n{'─'*60}")
    print(f"  Games indexed  : {n_games}")
    print(f"  New snapshots  : {n_new_snapshots}  ({n_new_snapshots} PNGs + bbox JSONs)")
    print(f"  Reused         : {n_reused}")
    print(f"  Errors         : {n_errors}")
    print(f"  Index written  : {INDEX_PATH}")

    # Bucket breakdown
    from collections import Counter
    counts = Counter(g["bucket"] for g in index)
    print(f"\n  Bucket breakdown:")
    for bkt in ALL_BUCKETS:
        print(f"    {bkt:<12} {counts.get(bkt, 0):>3} games")

    # ── pattern analysis ──────────────────────────────────────────────────────
    analyze_patterns(index)


if __name__ == "__main__":
    main()
