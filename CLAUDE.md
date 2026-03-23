# NBA Win Probability — CLAUDE.md

Project context and standing instructions for Claude Code.

## Project Overview

Python pipeline that fetches NBA play-by-play data, generates win-probability
charts, extracts visual features with OpenCV, and trains a Random Forest
classifier to issue live comeback / blowout alerts.

**Key scripts**
| File | Purpose |
|---|---|
| `nba_win_probability.py` | Fetch last-30-days games, plot WP charts, save PNGs |
| `build_training_set.py` | Generate Q1/Q2/Q3 snapshots, extract CV features, write `training_index.json` |
| `nba_classifier.py` | Train RF models, evaluate, live `predict <GAME_ID>` |
| `nba_live.py` | Live 5-minute dashboard with comeback alerts |

**Generated artefacts** (gitignored, re-creatable)
- `graphs/` — full-game WP PNGs organised by month + result bucket
- `snapshots/` — partial-game (Q1/Q2/Q3) PNGs for training
- `models/` — pickled RF classifiers (`model_Q1.pkl`, `model_Q2.pkl`, `model_Q3.pkl`)
- `training_index.json` — per-game feature index

## Security Rules

- **Never read, edit, or output the content of `.env` files.**
- Never commit `.env`, `.env.*`, or any file containing API keys or credentials.
- Never suggest adding secrets inline in source code; always direct to `.env` + `python-dotenv`.
- The `.claude/` directory contains session data and must never be read or committed.

## Data Sources

- **NBA CDN** (`cdn.nba.com/static/json/liveData/…`) — play-by-play and boxscores
- **nba_api** (`stats.nba.com`) — scoreboard, schedules
  - The `WinProbabilityPBP` stats endpoint currently returns 500 / empty responses;
    all WP is computed from the CDN play-by-play using the logistic margin model.

## Model Notes

- Win probability formula: `WP = sigmoid(margin / (0.5458 × √(seconds_remaining + 1)))`
- Three RF classifiers trained per quarter cutoff (Q1 / Q2 / Q3)
- Comeback alert threshold: **65 %** probability
- OpenCV visual features have a known calibration issue (median error 0.33 vs ground truth);
  classifiers use raw computed features instead.

## Development Notes

- Python 3.12, packages installed `--user --break-system-packages`
- `PYTHONUNBUFFERED=1` needed for live output in background tasks
- CDN returns 403 for pre-game PBP requests — handled silently in `nba_live.py`
