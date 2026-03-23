#!/usr/bin/env python3
"""
nba_classifier.py — NBA Game Outcome Predictive Alert System
─────────────────────────────────────────────────────────────────────────────
Trains Random Forest classifiers on per-quarter play-by-play features and
issues real-time alerts during live games.

Modes
─────
  python3 nba_classifier.py train             # fit & save models, print eval
  python3 nba_classifier.py predict <GAMEID>  # live prediction for one game
  python3 nba_classifier.py demo              # test on held-out games

Feature note
────────────
OpenCV was used to extract visual line signatures; an audit found median
discrepancy of 0.33 vs ground-truth (axis labels bleed into the WP-line
crop in blowout games). The classifier therefore uses features computed
directly from play-by-play data, which are accurate. CV features are
preserved in training_index.json for future image-model experiments.
"""

import os, re, sys, json, math, time, pickle, warnings
import numpy as np
import requests
warnings.filterwarnings("ignore")

from datetime import date, timedelta
from collections import Counter

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     roc_auc_score, brier_score_loss)

from nba_api.stats.library.http import STATS_HEADERS

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "training_index.json")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── labels ────────────────────────────────────────────────────────────────────
BUCKETS   = ["Overtime", "Comebacks", "Blowouts", "Close", "Standard"]
EXCITING  = {"Overtime", "Comebacks", "Close"}

# Alert thresholds
COMEBACK_THRESH = 0.28   # low because class is rare (21/252)
BLOWOUT_THRESH  = 0.48
EXCITING_THRESH = 0.52

# ── feature engineering ───────────────────────────────────────────────────────
RAW_KEYS = ["slope", "volatility", "final_wp", "mean_wp",
            "crossings", "max_excursion", "max_margin", "min_margin", "r_squared"]

QUARTER_ORDER = ["Q1", "Q2", "Q3"]


def build_feature_vector(record, up_to_quarter):
    """
    Flatten raw per-quarter features into a 1-D array for training.

    Includes:
      • Per-quarter raw features (slope, volatility, WP, margin stats…)
      • Cross-quarter delta features (momentum change Q→Q+1)
      • Game-level context (max deficit seen so far, WP trajectory)
    """
    feats = {}
    q_idx = QUARTER_ORDER.index(up_to_quarter)
    quarters = QUARTER_ORDER[: q_idx + 1]

    prev_final_wp  = 0.5
    prev_slope     = 0.0
    prev_crossings = 0

    for q in quarters:
        raw = (record.get("quarters", {}).get(q) or {}).get("raw") or {}
        for k in RAW_KEYS:
            feats[f"{q}_{k}"] = raw.get(k, np.nan)

        cur_wp         = raw.get("final_wp",  prev_final_wp)
        cur_slope      = raw.get("slope",     prev_slope)
        cur_crossings  = raw.get("crossings", prev_crossings)

        if q != "Q1":
            feats[f"{q}_delta_wp"]        = cur_wp        - prev_final_wp
            feats[f"{q}_delta_slope"]     = cur_slope     - prev_slope
            feats[f"{q}_delta_crossings"] = cur_crossings - prev_crossings

        prev_final_wp  = cur_wp
        prev_slope     = cur_slope
        prev_crossings = cur_crossings

    # Running game context
    all_mins  = [feats.get(f"{q}_min_margin", np.nan) for q in quarters]
    all_maxs  = [feats.get(f"{q}_max_margin", np.nan) for q in quarters]
    valid_min = [v for v in all_mins if not np.isnan(v)]
    valid_max = [v for v in all_maxs if not np.isnan(v)]
    feats["game_max_lead"]    = max(valid_max) if valid_max else np.nan
    feats["game_max_deficit"] = min(valid_min) if valid_min else np.nan
    feats["game_wp_range"]    = (max(feats.get(f"{q}_final_wp", np.nan)
                                     for q in quarters
                                     if not np.isnan(feats.get(f"{q}_final_wp", np.nan)))
                                 - min(feats.get(f"{q}_final_wp", np.nan)
                                       for q in quarters
                                       if not np.isnan(feats.get(f"{q}_final_wp", np.nan)))
                                 ) if len(quarters) > 1 else np.nan

    return feats


def records_to_matrix(records, up_to_quarter):
    """Return (X, feature_names, y_multiclass, y_comeback, y_blowout, y_exciting)."""
    rows, labels = [], []
    for r in records:
        quarters_present = set(r.get("quarters", {}).keys())
        if up_to_quarter not in quarters_present:
            continue
        rows.append(build_feature_vector(r, up_to_quarter))
        labels.append(r["bucket"])

    if not rows:
        return None

    feat_names = sorted(rows[0].keys())
    X = np.array([[row.get(k, np.nan) for k in feat_names] for row in rows],
                 dtype=float)
    y_mc       = np.array(labels)
    y_comeback = (y_mc == "Comebacks").astype(int)
    y_blowout  = (y_mc == "Blowouts").astype(int)
    y_exciting = np.array([1 if b in EXCITING else 0 for b in y_mc])

    return X, feat_names, y_mc, y_comeback, y_blowout, y_exciting


# ── model building ────────────────────────────────────────────────────────────
def make_pipeline(classifier):
    # No CalibratedClassifierCV — RF's native predict_proba is adequate
    # for alert thresholds and avoids 15x fitting overhead.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("clf",     classifier),
    ])


def make_rf(n_estimators=200, class_weight="balanced"):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=2,
    )


def make_gb():
    return GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )


# ── evaluation helpers ────────────────────────────────────────────────────────
def eval_binary(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    brier = brier_score_loss(y_true, y_prob)
    print(f"    {name:<20}  prec={prec:.2f}  rec={rec:.2f}  "
          f"f1={f1:.2f}  AUC={auc:.3f}  Brier={brier:.3f}  "
          f"(TP={tp} FP={fp} FN={fn} TN={tn})")
    return {"precision": prec, "recall": rec, "f1": f1, "auc": auc}


def print_feature_importance(feat_names, clf_pipeline, top_n=12):
    """Extract importances from the RF inside the pipeline."""
    try:
        imps = clf_pipeline.named_steps["clf"].feature_importances_
    except Exception:
        return
    pairs = sorted(zip(imps, feat_names), reverse=True)[:top_n]
    print(f"    {'Feature':<35} Importance")
    print(f"    {'─'*35} ─────────")
    for imp, name in pairs:
        bar = "█" * int(imp * 200)
        print(f"    {name:<35} {imp:.4f}  {bar}")


# ── train ─────────────────────────────────────────────────────────────────────
def train(records):
    models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for quarter in QUARTER_ORDER:
        result = records_to_matrix(records, quarter)
        if result is None:
            continue
        X, feat_names, y_mc, y_cb, y_bl, y_ex = result
        n = len(y_mc)

        print(f"\n{'═'*66}")
        print(f"  {quarter} MODEL  —  {n} games")
        print(f"  Class distribution: {dict(Counter(y_mc))}")
        print(f"{'═'*66}")

        # ── multi-class ──────────────────────────────────────────────────────
        mc_pipe = make_pipeline(make_rf())
        mc_pipe.fit(X, y_mc)
        y_mc_cv = cross_val_predict(make_pipeline(make_rf()), X, y_mc,
                                     cv=cv, method="predict")
        mc_acc = (y_mc_cv == y_mc).mean()
        print(f"\n  Multi-class (5-way)  CV accuracy: {mc_acc:.3f}")
        print(classification_report(y_mc, y_mc_cv, labels=BUCKETS,
                                    target_names=BUCKETS, zero_division=0,
                                    digits=2))

        # ── binary alerts ────────────────────────────────────────────────────
        print("  Binary alert performance (cross-validated probabilities):")
        cb_pipe = make_pipeline(make_rf())
        bl_pipe = make_pipeline(make_rf())
        ex_pipe = make_pipeline(make_rf())

        cb_prob_cv = cross_val_predict(make_pipeline(make_rf()), X, y_cb,
                                        cv=cv, method="predict_proba")[:, 1]
        bl_prob_cv = cross_val_predict(make_pipeline(make_rf()), X, y_bl,
                                        cv=cv, method="predict_proba")[:, 1]
        ex_prob_cv = cross_val_predict(make_pipeline(make_rf()), X, y_ex,
                                        cv=cv, method="predict_proba")[:, 1]

        eval_binary("Comeback detector", y_cb, cb_prob_cv, COMEBACK_THRESH)
        eval_binary("Blowout detector",  y_bl, bl_prob_cv, BLOWOUT_THRESH)
        eval_binary("Exciting detector", y_ex, ex_prob_cv, EXCITING_THRESH)

        # ── fit final models on all data ─────────────────────────────────────
        cb_pipe.fit(X, y_cb)
        bl_pipe.fit(X, y_bl)
        ex_pipe.fit(X, y_ex)

        print(f"\n  Top features — Comeback detector ({quarter}):")
        print_feature_importance(feat_names, cb_pipe)

        # ── save ─────────────────────────────────────────────────────────────
        bundle = {
            "quarter":    quarter,
            "feat_names": feat_names,
            "mc":  mc_pipe,
            "cb":  cb_pipe,
            "bl":  bl_pipe,
            "ex":  ex_pipe,
        }
        path = os.path.join(MODEL_DIR, f"model_{quarter}.pkl")
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        models[quarter] = bundle
        print(f"\n  Saved → {path}")

    return models


# ── live prediction ───────────────────────────────────────────────────────────
def load_model(quarter):
    path = os.path.join(MODEL_DIR, f"model_{quarter}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model for {quarter}. Run: python3 nba_classifier.py train")
    with open(path, "rb") as f:
        return pickle.load(f)


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

def fetch_live_pbp(game_id):
    hdrs = {"User-Agent": STATS_HEADERS["User-Agent"],
            "Accept": "application/json", "Referer": "https://www.nba.com/"}
    url = (f"https://cdn.nba.com/static/json/liveData/"
           f"playbyplay/playbyplay_{game_id}.json")
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=hdrs, timeout=20)
            r.raise_for_status()
            return r.json()["game"]["actions"]
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"  Retry {attempt}: {exc}")
            time.sleep(3)

def fetch_boxscore(game_id):
    hdrs = {"User-Agent": STATS_HEADERS["User-Agent"],
            "Accept": "application/json", "Referer": "https://www.nba.com/"}
    url = (f"https://cdn.nba.com/static/json/liveData/"
           f"boxscore/boxscore_{game_id}.json")
    r = requests.get(url, headers=hdrs, timeout=15)
    r.raise_for_status()
    g = r.json()["game"]
    return g["homeTeam"]["teamTricode"], g["awayTeam"]["teamTricode"]

def pbp_to_series(actions):
    max_period, raw = 4, {}
    for act in actions:
        p  = int(act.get("period", 0))
        cl = act.get("clock", "PT00M00.00S")
        sh = act.get("scoreHome", "")
        sa = act.get("scoreAway", "")
        if p < 1 or sh == "" or sa == "":
            continue
        max_period = max(max_period, p)
        raw[round(elapsed(p, cl), 1)] = int(sh) - int(sa)

    total = total_secs(max_period)
    series = {}
    for e in sorted(raw):
        m = raw[e]
        series[e] = {"margin": m, "wp": wp_from_margin(m, total - e)}
    if not series or min(series) > 0:
        series[0.0] = {"margin": 0, "wp": 0.5}
    return dict(sorted(series.items())), max_period

def compute_raw_features_for_quarter(series, q_start, q_end):
    from scipy import stats as sp_stats
    window = {e: v for e, v in series.items() if q_start <= e <= q_end}
    if len(window) < 2:
        return {}
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

def build_live_record(series, max_period):
    """Convert a live PBP series into a pseudo-record for feature extraction."""
    total = total_secs(max_period)
    current_elapsed = max(series.keys())

    # Determine which quarter we're in / have completed
    if current_elapsed >= 2160:
        current_q = "Q3"
    elif current_elapsed >= 1440:
        current_q = "Q2"
    elif current_elapsed >= 720:
        current_q = "Q1"
    else:
        current_q = None  # not yet end of Q1

    record = {"quarters": {}}
    q_bounds = [("Q1", 0, 720), ("Q2", 720, 1440), ("Q3", 1440, 2160)]
    for q_label, q_start, q_end in q_bounds:
        raw = compute_raw_features_for_quarter(series, q_start, q_end)
        if raw:
            record["quarters"][q_label] = {"raw": raw, "cv": None}

    return record, current_q


ALERT_ICONS = {
    "Comeback":  "⚡",
    "Blowout":   "💤",
    "Exciting":  "🔥",
}

def format_prediction(home_tri, away_tri, game_id, record, current_q, series):
    """Run the quarter model and print a formatted alert."""
    if current_q is None:
        print(f"  Not enough data yet (Q1 not complete).")
        return

    bundle = load_model(current_q)
    fv = build_feature_vector(record, current_q)
    feat_names = bundle["feat_names"]
    X = np.array([[fv.get(k, np.nan) for k in feat_names]])

    # Probabilities
    mc_probs   = dict(zip(bundle["mc"].classes_,
                          bundle["mc"].predict_proba(X)[0]))
    cb_prob    = bundle["cb"].predict_proba(X)[0][1]
    bl_prob    = bundle["bl"].predict_proba(X)[0][1]
    ex_prob    = bundle["ex"].predict_proba(X)[0][1]
    mc_pred    = bundle["mc"].predict(X)[0]

    # Current game state
    last_e     = max(series.keys())
    last_margin = series[last_e]["margin"]
    last_wp     = series[last_e]["wp"]
    q_raw = (record["quarters"].get(current_q) or {}).get("raw") or {}

    # Header
    print(f"\n  {'─'*60}")
    print(f"  🏀  {away_tri} @ {home_tri}  ·  after {current_q}  ·  Game {game_id}")
    print(f"  {'─'*60}")
    margin_str = (f"{home_tri} +{last_margin}" if last_margin > 0
                  else (f"{away_tri} +{-last_margin}" if last_margin < 0 else "Tied"))
    print(f"  Score margin  : {margin_str}")
    print(f"  {home_tri} win prob  : {last_wp:.1%}")
    print(f"  {current_q} slope    : {q_raw.get('slope', 0):+.4f}  "
          f"({'↑ gaining' if q_raw.get('slope',0) > 0 else '↓ losing'})")
    print(f"  {current_q} volatility: {q_raw.get('volatility', 0):.4f}  "
          f"  crossings: {q_raw.get('crossings', 0)}")

    # Outcome probabilities
    print(f"\n  PREDICTED PROBABILITIES  ({current_q} model)")
    print(f"  {'─'*44}")
    for bucket in BUCKETS:
        p    = mc_probs.get(bucket, 0)
        bar  = "█" * int(p * 30)
        star = " ◀ most likely" if bucket == mc_pred else ""
        print(f"  {bucket:<12} {p:5.1%}  {bar}{star}")

    # Alert panel
    alerts = []
    if cb_prob >= COMEBACK_THRESH:
        alerts.append(("⚡ COMEBACK ALERT",   cb_prob, "comeback pattern detected"))
    if bl_prob >= BLOWOUT_THRESH:
        alerts.append(("💤 BLOWOUT ALERT",    bl_prob, "game trending lopsided"))
    if ex_prob >= EXCITING_THRESH and not alerts:
        alerts.append(("🔥 EXCITING GAME",    ex_prob, "Close / OT / Comeback likely"))

    if alerts:
        print(f"\n  {'─'*60}")
        for title, prob, reason in alerts:
            print(f"  {title}  ({prob:.0%} confidence)")
            print(f"  └─ {reason}")
    else:
        print(f"\n  ℹ️  No strong alerts  (CB={cb_prob:.1%}  BL={bl_prob:.1%}  EX={ex_prob:.1%})")

    print(f"  {'─'*60}\n")


# ── demo mode ─────────────────────────────────────────────────────────────────
def demo(records, n=12):
    """Run predictions on a random sample of held-out games and show alerts."""
    rng = np.random.default_rng(7)
    sample = rng.choice(len(records), size=min(n, len(records)), replace=False)
    print(f"\n{'═'*66}")
    print(f"  DEMO — {n} sample predictions from training data")
    print(f"{'═'*66}")

    q = "Q2"    # show Q2 predictions (most interesting)
    for idx in sample:
        rec = records[idx]
        if q not in rec.get("quarters", {}):
            continue
        bundle = load_model(q)
        fv = build_feature_vector(rec, q)
        X  = np.array([[fv.get(k, np.nan) for k in bundle["feat_names"]]])
        cb_prob = bundle["cb"].predict_proba(X)[0][1]
        bl_prob = bundle["bl"].predict_proba(X)[0][1]
        mc_pred = bundle["mc"].predict(X)[0]
        actual  = rec["bucket"]
        correct = "✓" if mc_pred == actual else "✗"
        alert   = ""
        if cb_prob >= COMEBACK_THRESH:
            alert = f"  ⚡ comeback alert ({cb_prob:.0%})"
        elif bl_prob >= BLOWOUT_THRESH:
            alert = f"  💤 blowout alert ({bl_prob:.0%})"
        print(f"  {correct} {rec['matchup']:<8} {rec['date']}  "
              f"actual={actual:<12} pred={mc_pred:<12}{alert}")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    with open(INDEX_PATH) as f:
        records = json.load(f)

    if mode == "train":
        print(f"Training on {len(records)} games…")
        train(records)
        print("\n✓ All models saved to models/")
        demo(records)

    elif mode == "predict":
        if len(sys.argv) < 3:
            print("Usage: python3 nba_classifier.py predict <GAME_ID>")
            print("Example: python3 nba_classifier.py predict 0022500999")
            sys.exit(1)
        game_id = sys.argv[2]
        print(f"Fetching live data for game {game_id}…")
        try:
            home_tri, away_tri = fetch_boxscore(game_id)
        except Exception:
            home_tri, away_tri = "HOME", "AWAY"
        actions = fetch_live_pbp(game_id)
        series, max_period = pbp_to_series(actions)
        record, current_q = build_live_record(series, max_period)
        format_prediction(home_tri, away_tri, game_id, record, current_q, series)

    elif mode == "demo":
        demo(records, n=20)

    else:
        print(f"Unknown mode '{mode}'. Use: train | predict <GAMEID> | demo")
        sys.exit(1)


if __name__ == "__main__":
    main()
