#!/usr/bin/env python3
"""
nba_live.py — NBA Live Predictive Alert Dashboard
─────────────────────────────────────────────────────────────────────────────
Usage:
  python3 nba_live.py                    # auto-discover tonight's games
  python3 nba_live.py 0022501038 0022501039 ...  # specify game IDs

Every 5 minutes the dashboard fetches live play-by-play, computes per-quarter
slope / volatility, runs the comeback classifier, and reprints a clean table.
A visible ALERT fires whenever comeback probability ≥ 65 %.
"""

import os, sys, re, math, time, json, pickle, warnings, textwrap
import numpy as np
import requests
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

ET = ZoneInfo("America/New_York")

warnings.filterwarnings("ignore")

from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.library.http import STATS_HEADERS

# ── constants ─────────────────────────────────────────────────────────────────
BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR            = os.path.join(BASE_DIR, "models")
REFRESH_SECS         = 300          # 5 minutes between data fetches
COMEBACK_ALERT_PCTG  = 65           # integer percent threshold

# ── ANSI helpers ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BGYELLOW = "\033[43m\033[30m"
BGRED    = "\033[41m\033[97m"

def clr(text, *codes): return "".join(codes) + str(text) + RESET
def clear_screen():    print("\033[2J\033[H", end="", flush=True)


# ── physics helpers (self-contained, mirrors nba_classifier.py) ────────────────
def _clock_sec(s):
    m = re.match(r"PT(\d+)M([\d.]+)S", s)
    return int(m.group(1)) * 60 + float(m.group(2)) if m else 0.0

def _elapsed(period, clock, pd=720, otd=300):
    r = _clock_sec(clock)
    return (period-1)*pd + pd - r if period <= 4 else 4*pd + (period-5)*otd + otd - r

def _total(max_period, pd=720, otd=300):
    return 4*pd if max_period <= 4 else 4*pd + (max_period-4)*otd

def _wp(margin, secs_rem):
    if secs_rem <= 0:
        return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    sigma = 0.5458 * math.sqrt(secs_rem + 1)
    return 1.0 / (1.0 + math.exp(-1.7 * margin / sigma))


# ── feature extraction ────────────────────────────────────────────────────────
RAW_KEYS = ["slope", "volatility", "final_wp", "mean_wp",
            "crossings", "max_excursion", "max_margin", "min_margin", "r_squared"]

def _raw_features(series, q_start, q_end):
    from scipy import stats as sp_stats
    window = {e: v for e, v in series.items() if q_start <= e <= q_end}
    if len(window) < 3:
        return None
    es  = sorted(window)
    wp  = np.array([window[e]["wp"]     for e in es])
    mgn = np.array([window[e]["margin"] for e in es])
    xs  = np.linspace(0.0, 1.0, len(wp))
    slope, _, r_val, _, _ = sp_stats.linregress(xs, wp)
    diffs = np.diff(wp)
    signs = np.sign(wp - 0.5)
    return {
        "slope":         float(slope),
        "volatility":    float(np.std(diffs)),
        "final_wp":      float(wp[-1]),
        "mean_wp":       float(np.mean(wp)),
        "crossings":     int(np.sum(np.abs(np.diff(signs)) > 0)),
        "max_excursion": float(np.max(np.abs(wp - 0.5))),
        "max_margin":    int(np.max(mgn)),
        "min_margin":    int(np.min(mgn)),
        "r_squared":     float(r_val ** 2),
    }

def _build_fvec(record, up_to_q):
    QUARTERS = ["Q1", "Q2", "Q3"]
    feats = {}
    q_idx = QUARTERS.index(up_to_q)
    prev_wp = 0.5; prev_slope = 0.0; prev_cx = 0
    for q in QUARTERS[:q_idx+1]:
        raw = (record.get("quarters", {}).get(q) or {}).get("raw") or {}
        for k in RAW_KEYS:
            feats[f"{q}_{k}"] = raw.get(k, np.nan)
        cur_wp = raw.get("final_wp", prev_wp)
        cur_slope = raw.get("slope", prev_slope)
        cur_cx = raw.get("crossings", prev_cx)
        if q != "Q1":
            feats[f"{q}_delta_wp"]        = cur_wp    - prev_wp
            feats[f"{q}_delta_slope"]     = cur_slope - prev_slope
            feats[f"{q}_delta_crossings"] = cur_cx    - prev_cx
        prev_wp = cur_wp; prev_slope = cur_slope; prev_cx = cur_cx
    all_mins = [feats.get(f"{q}_min_margin", np.nan) for q in QUARTERS[:q_idx+1]]
    all_maxs = [feats.get(f"{q}_max_margin", np.nan) for q in QUARTERS[:q_idx+1]]
    valid_min = [v for v in all_mins if not np.isnan(v)]
    valid_max = [v for v in all_maxs if not np.isnan(v)]
    feats["game_max_lead"]    = max(valid_max) if valid_max else np.nan
    feats["game_max_deficit"] = min(valid_min) if valid_min else np.nan
    wps = [feats.get(f"{q}_final_wp", np.nan) for q in QUARTERS[:q_idx+1]
           if not np.isnan(feats.get(f"{q}_final_wp", np.nan))]
    feats["game_wp_range"] = max(wps) - min(wps) if len(wps) > 1 else np.nan
    return feats


# ── model loading (cached) ────────────────────────────────────────────────────
_model_cache = {}

def load_model(quarter):
    if quarter not in _model_cache:
        path = os.path.join(MODEL_DIR, f"model_{quarter}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            _model_cache[quarter] = pickle.load(f)
    return _model_cache[quarter]


# ── PBP parsing ───────────────────────────────────────────────────────────────
def parse_pbp(actions):
    """
    Convert CDN PBP actions → {elapsed: {margin, wp}}, max_period,
    current_period, current_clock, home_score, away_score.
    """
    max_period = 1
    current_period = 1
    current_clock  = "PT12M00.00S"
    home_score = away_score = 0
    raw = {}

    for act in actions:
        p  = int(act.get("period", 0))
        cl = act.get("clock", "PT00M00.00S")
        sh = act.get("scoreHome", "")
        sa = act.get("scoreAway", "")
        if p < 1:
            continue
        max_period = max(max_period, p)
        current_period = p
        current_clock  = cl
        if sh != "" and sa != "":
            home_score = int(sh)
            away_score = int(sa)
            e = round(_elapsed(p, cl), 1)
            raw[e] = int(sh) - int(sa)

    if not raw:
        return {}, max_period, current_period, current_clock, 0, 0

    total = _total(max_period)
    series = {0.0: {"margin": 0, "wp": 0.5}}
    for e in sorted(raw):
        m = raw[e]
        series[e] = {"margin": m, "wp": _wp(m, total - e)}

    return series, max_period, current_period, current_clock, home_score, away_score


def quarter_label(period, clock):
    """Format a human-readable period+clock string."""
    secs = _clock_sec(clock)
    mins, s = int(secs // 60), int(secs % 60)
    tag = f"Q{period}" if period <= 4 else f"OT{period-4}"
    return f"{tag} {mins}:{s:02d}"


# ── per-game features ─────────────────────────────────────────────────────────
def compute_game_features(series, max_period, current_period):
    """
    Compute Q1/Q2/Q3 raw features and determine which model to use.
    Returns (record, model_quarter | None).
    """
    record = {"quarters": {}}
    for q_label, q_start, q_end in [("Q1", 0, 720), ("Q2", 720, 1440), ("Q3", 1440, 2160)]:
        raw = _raw_features(series, q_start, q_end)
        if raw:
            record["quarters"][q_label] = {"raw": raw, "cv": None}

    # Use Q1 model once 12 minutes (720 s) of data are available,
    # regardless of whether Q2 has officially started.
    max_elapsed = max(series.keys()) if series else 0
    if current_period >= 4:
        model_q = "Q3"
    elif current_period >= 3:
        model_q = "Q2"
    elif current_period >= 2 or max_elapsed >= 720:
        model_q = "Q1"
    else:
        model_q = None   # less than 12 min played

    return record, model_q


def predict_comeback(record, model_q):
    """Return comeback probability (0-1) or None."""
    if model_q is None:
        return None
    bundle = load_model(model_q)
    if bundle is None:
        return None
    if model_q not in record.get("quarters", {}):
        return None
    try:
        fv    = _build_fvec(record, model_q)
        X     = np.array([[fv.get(k, np.nan) for k in bundle["feat_names"]]])
        prob  = bundle["cb"].predict_proba(X)[0][1]
        return float(prob)
    except Exception:
        return None


# ── data fetching ─────────────────────────────────────────────────────────────
_HDR = {"User-Agent": STATS_HEADERS["User-Agent"],
        "Accept": "application/json", "Referer": "https://www.nba.com/"}

def fetch_tricodes(game_id):
    url = (f"https://cdn.nba.com/static/json/liveData/"
           f"boxscore/boxscore_{game_id}.json")
    r = requests.get(url, headers=_HDR, timeout=15)
    r.raise_for_status()
    g = r.json()["game"]
    return g["homeTeam"]["teamTricode"], g["awayTeam"]["teamTricode"]

def fetch_pbp(game_id):
    url = (f"https://cdn.nba.com/static/json/liveData/"
           f"playbyplay/playbyplay_{game_id}.json")
    r = requests.get(url, headers=_HDR, timeout=20)
    r.raise_for_status()
    return r.json()["game"]["actions"]

def fetch_tonight_games():
    """Return list of {game_id, home_tri, away_tri, tip_off} for today."""
    warnings.filterwarnings("ignore")
    sb = scoreboardv2.ScoreboardV2(
        game_date=date.today().strftime("%Y-%m-%d"), timeout=20)
    d  = sb.get_dict()
    gh = d["resultSets"][0]
    hdr = gh["headers"]
    seen, games = set(), []
    for row in gh["rowSet"]:
        g = dict(zip(hdr, row))
        gid = g["GAME_ID"]
        if gid in seen:
            continue
        seen.add(gid)
        code = g["GAMECODE"].split("/")[-1] if "/" in g["GAMECODE"] else g["GAMECODE"]
        away = code[:3]; home = code[3:]
        games.append({
            "game_id":  gid,
            "home_tri": home,
            "away_tri": away,
            "status_id": g.get("GAME_STATUS_ID", 1),
            "tip_off":  g.get("GAME_STATUS_TEXT", "").strip(),
        })
    return games


# ── game state polling ────────────────────────────────────────────────────────
def refresh_game(info, state):
    """
    Fetch live PBP for one game and update its state dict in place.
    A 403 response means the game hasn't tipped yet — treated as pre-game,
    not an error. Other failures are surfaced in the error column.
    """
    game_id = info["game_id"]
    try:
        actions = fetch_pbp(game_id)
        (series, max_period,
         cur_period, cur_clock,
         home_score, away_score) = parse_pbp(actions)

        record, model_q = compute_game_features(series, max_period, cur_period)
        cb_prob         = predict_comeback(record, model_q)

        # Per-quarter features for display
        q1_raw = (record["quarters"].get("Q1") or {}).get("raw") or {}
        q2_raw = (record["quarters"].get("Q2") or {}).get("raw") or {}

        state.update({
            "home_tri":    info["home_tri"],
            "away_tri":    info["away_tri"],
            "home_score":  home_score,
            "away_score":  away_score,
            "cur_period":  cur_period,
            "cur_clock":   cur_clock,
            "max_period":  max_period,
            "has_data":    bool(series),
            "started":     bool(series and max(series.keys()) > 1),
            "status_id":   info.get("status_id", 1),
            "tip_off":     info.get("tip_off", ""),
            "q1_slope":    q1_raw.get("slope"),
            "q1_vol":      q1_raw.get("volatility"),
            "q2_slope":    q2_raw.get("slope"),
            "q2_vol":      q2_raw.get("volatility"),
            "cb_prob":     cb_prob,
            "model_q":     model_q,
            "error":       None,
        })
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            state["error"] = None   # pre-game: CDN hasn't opened the feed yet
        else:
            state["error"] = f"HTTP {exc.response.status_code}" if exc.response else str(exc)[:50]
    except Exception as exc:
        state["error"] = str(exc)[:60]


# ── rendering ─────────────────────────────────────────────────────────────────
COL_WIDTHS = {
    "matchup": 14, "score": 15, "status": 11,
    "q1_slope": 11, "q2_slope": 11, "cb_prob": 11, "alert": 7,
}
SEP = "─"

def _fmt_slope(val):
    if val is None:
        return clr("  —  ", DIM)
    arrow = "↑" if val > 0.001 else ("↓" if val < -0.001 else "→")
    col   = GREEN if val > 0.001 else (RED if val < -0.001 else RESET)
    return clr(f"{val:+.4f} {arrow}", col)

def _fmt_cb(prob):
    if prob is None:
        return clr("  —  ", DIM)
    pct = int(prob * 100)
    if pct >= COMEBACK_ALERT_PCTG:
        return clr(f"{pct:>3}% ⚡", BOLD + YELLOW)
    elif pct >= 45:
        return clr(f"{pct:>3}%  ", YELLOW)
    else:
        return f"{pct:>3}%  "

def _fmt_status(state):
    if not state.get("started"):
        return clr(state.get("tip_off", "–")[:10], DIM)
    if state.get("status_id") == 3:
        return clr("Final", BOLD)
    return quarter_label(state["cur_period"], state["cur_clock"])

def _fmt_score(state):
    if not state.get("started"):
        return clr("Not started", DIM)
    hs, as_ = state["home_score"], state["away_score"]
    ht, at  = state["home_tri"],  state["away_tri"]
    if hs > as_:
        return f"{clr(ht, GREEN+BOLD)} {hs}–{as_}"
    elif as_ > hs:
        return f"{clr(at, GREEN+BOLD)} {as_}–{hs}"
    else:
        return f"Tied {hs}–{as_}"

def render(states, next_refresh_secs, fetch_ts):
    clear_screen()
    now   = datetime.now().strftime("%I:%M:%S %p")
    today = date.today().strftime("%a %b %d %Y")
    W     = 82

    # ── header ────────────────────────────────────────────────────────────────
    title = f"  🏀  NBA LIVE DASHBOARD  ·  {today}  ·  {now}"
    print(clr("═" * W, CYAN))
    print(clr(title, BOLD + CYAN))
    print(clr("═" * W, CYAN))

    # ── column headers ────────────────────────────────────────────────────────
    hdr = (f"  {'MATCHUP':<14}  {'SCORE':<16}  {'STATUS':<12}"
           f"  {'Q1 SLOPE':<11}  {'Q2 SLOPE':<11}  {'CB PROB':<8}  ALERT")
    print(clr(hdr, BOLD))
    print(clr(SEP * W, DIM))

    alerts = []

    for gid, state in states.items():
        if state.get("error"):
            print(f"  {state.get('away_tri','?')} @ {state.get('home_tri','?')}"
                  f"  [error: {state['error']}]")
            continue

        matchup = f"{state['away_tri']} @ {state['home_tri']}"
        score   = _fmt_score(state)
        status  = _fmt_status(state)
        q1s     = _fmt_slope(state.get("q1_slope"))
        q2s     = _fmt_slope(state.get("q2_slope"))
        cb      = _fmt_cb(state.get("cb_prob"))

        cb_raw = state.get("cb_prob")
        alert_cell = ""
        if cb_raw is not None and int(cb_raw * 100) >= COMEBACK_ALERT_PCTG:
            alert_cell = clr("  ⚡ ", BOLD + YELLOW)
            alerts.append((matchup, int(cb_raw * 100), state["model_q"],
                           state["home_score"], state["away_score"],
                           state["home_tri"], state["away_tri"]))

        # Raw-text width for alignment (strip ANSI codes for padding)
        ansi_strip = re.compile(r"\033\[[0-9;]*m")
        def vis(s): return len(ansi_strip.sub("", str(s)))

        # Fixed-width columns (pad based on visible length)
        def pad(s, w): return str(s) + " " * max(0, w - vis(s))

        line = (f"  {pad(matchup,14)}  {pad(score,16)}  {pad(status,12)}"
                f"  {pad(q1s,13)}  {pad(q2s,13)}  {pad(cb,9)}{alert_cell}")
        print(line)

    # ── alert panel ───────────────────────────────────────────────────────────
    if alerts:
        print(clr(SEP * W, DIM))
        for matchup, pct, model_q, hs, as_, ht, at in alerts:
            margin_str = ""
            if hs != as_:
                leader = ht if hs > as_ else at
                diff   = abs(hs - as_)
                margin_str = f"  ({leader} leads by {diff})"
            msg = (f"  ⚡  COMEBACK ALERT  —  {matchup}  —  "
                   f"{pct}% probability  [{model_q} model]{margin_str}")
            print(clr("█" * W, YELLOW))
            print(clr(msg, BOLD + YELLOW))
            print(clr("█" * W, YELLOW))

    # ── footer ────────────────────────────────────────────────────────────────
    print(clr(SEP * W, DIM))
    mins, secs  = divmod(next_refresh_secs, 60)
    bar_filled  = int((REFRESH_SECS - next_refresh_secs) / REFRESH_SECS * 40)
    bar         = "█" * bar_filled + "░" * (40 - bar_filled)
    print(f"  {clr('Last fetch:', DIM)} {fetch_ts}   "
          f"{clr('Next refresh in', DIM)} {mins}:{secs:02d}  "
          f"[{clr(bar, CYAN)}]")
    print(clr("═" * W, CYAN))
    sys.stdout.flush()


# ── time-window helpers ───────────────────────────────────────────────────────
WINDOW_START_H, WINDOW_START_M = 19,  0   # 7:00 PM ET
WINDOW_END_H,   WINDOW_END_M   =  1, 30   # 1:30 AM ET (next calendar day)

def _tonight_window():
    """Return (window_start, window_end) as timezone-aware ET datetimes."""
    now_et    = datetime.now(ET)
    today_et  = now_et.date()
    tomorrow  = today_et + timedelta(days=1)
    start = datetime(today_et.year,  today_et.month,  today_et.day,
                     WINDOW_START_H, WINDOW_START_M, 0, tzinfo=ET)
    end   = datetime(tomorrow.year,  tomorrow.month,  tomorrow.day,
                     WINDOW_END_H,   WINDOW_END_M,   0, tzinfo=ET)
    return start, end

def wait_for_tipoff(window_start, games):
    """Block until window_start, showing a live countdown."""
    W = 82
    while True:
        now = datetime.now(ET)
        if now >= window_start:
            break
        secs_left = int((window_start - now).total_seconds())
        hrs, rem  = divmod(secs_left, 3600)
        mins, sec = divmod(rem, 60)
        clear_screen()
        date_str = now.strftime("%a %b %d %Y")
        print(clr("═" * W, CYAN))
        print(clr(f"  🏀  NBA LIVE DASHBOARD  ·  {date_str}", BOLD + CYAN))
        print(clr("═" * W, CYAN))
        print()
        print(clr("  ⏳  Waiting for tip-off…  First games at 7:00 PM ET", BOLD))
        print()
        print(f"  Countdown:  {clr(f'{hrs}h {mins:02d}m {sec:02d}s', YELLOW + BOLD)}")
        print()
        if games:
            print(f"  Tonight's slate ({len(games)} game(s)):")
            for g in games:
                tip = g.get("tip_off", "")
                print(f"    {g['away_tri']} @ {g['home_tri']}"
                      + (f"  —  {tip}" if tip else ""))
        print()
        print(clr("═" * W, CYAN))
        sys.stdout.flush()
        time.sleep(1)


# ── main loop ─────────────────────────────────────────────────────────────────
def main():
    # Discover or parse game IDs
    if len(sys.argv) > 1:
        cli_ids = sys.argv[1:]
        games   = []
        print("Resolving tricodes for provided game IDs…")
        for gid in cli_ids:
            try:
                ht, at = fetch_tricodes(gid)
                games.append({"game_id": gid, "home_tri": ht, "away_tri": at,
                               "status_id": 2, "tip_off": "Live"})
            except Exception:
                games.append({"game_id": gid, "home_tri": "???", "away_tri": "???",
                               "status_id": 1, "tip_off": ""})
    else:
        print("Fetching tonight's schedule…")
        games = fetch_tonight_games()

    if not games:
        print("No games found for tonight.")
        sys.exit(0)

    # ── time-window gate ──────────────────────────────────────────────────────
    if len(sys.argv) <= 1:   # only auto-gated in schedule-discovery mode
        window_start, window_end = _tonight_window()
        now_et = datetime.now(ET)
        if now_et > window_end:
            print("Tonight's game window has ended (past 1:30 AM ET). Exiting.")
            sys.exit(0)
        if now_et < window_start:
            wait_for_tipoff(window_start, games)

    print(f"Tracking {len(games)} game(s): "
          f"{', '.join(g['away_tri']+'@'+g['home_tri'] for g in games)}")
    time.sleep(1)

    # Initialise state dict
    states = {g["game_id"]: {
        "home_tri":  g["home_tri"], "away_tri": g["away_tri"],
        "started":   False, "has_data": False,
        "status_id": g.get("status_id", 1),
        "tip_off":   g.get("tip_off", ""),
        "home_score": 0, "away_score": 0,
        "cur_period": 1, "cur_clock": "PT12M00.00S",
        "max_period": 4, "q1_slope": None, "q2_slope": None,
        "q1_vol": None, "q2_vol": None,
        "cb_prob": None, "model_q": None, "error": None,
    } for g in games}

    # Pre-load models
    for q in ("Q1", "Q2", "Q3"):
        load_model(q)

    fetch_ts = "–"

    while True:
        # ── fetch all games ───────────────────────────────────────────────────
        for info in games:
            refresh_game(info, states[info["game_id"]])
        fetch_ts = datetime.now().strftime("%I:%M:%S %p")

        # ── countdown loop ────────────────────────────────────────────────────
        for remaining in range(REFRESH_SECS, 0, -1):
            render(states, remaining, fetch_ts)
            time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
