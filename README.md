# WNBA Edge Model

WNBA **research and analytics** pipeline from the Chase Analytics model lab:
game projections, a player watchboard, market snapshots, and a fully graded
prediction record.

> Research software, not betting advice. No output is a wager instruction, and
> nothing here promises profit. See [METHODOLOGY.md](METHODOLOGY.md) for what
> every number means and its known limitations.

**Live dashboard:** https://alphakiller1.github.io/wnba-edge-model/ — rebuilt by
GitHub Pages on every push to `main` from the committed processed data.

## Where this fits in Chase Analytics

- **[mlb-model](https://github.com/Alphakiller1/mlb-model)** — the MLB decision-support
  engine (expected-runs model, promotion gates, paper portfolio).
- **wnba-edge-model (this repo)** — the WNBA product: earlier-stage, same standards:
  de-vigged market probabilities, timestamped + graded predictions, honest empty states.
- **chase-analytics.com** — the consumer MLB research dashboard (MLBMA pipeline).

## What the model does

1. **Scrape + validate** — WNBAnalytics players/teams/games/box scores behind light
   schema contracts that fail loudly on upstream drift; finished box scores are
   cached, never re-downloaded.
2. **Features** — usage/minutes/form/impact signals, z-scored and **shrunk by sample
   size** so 1-minute players cannot top the board; low-sample players are flagged
   and excluded from the public watchboard.
3. **Game projections** — team-efficiency baseline with a **win-rate-blended home
   court** (blowouts do not inflate the home bump) and a **home-win probability
   from the projected margin** so moneyline and spread favorites agree.
4. **Prop pricing** — Normal approximation with **per-market sigma fitted from real
   game logs** (per-player where the sample allows), minutes-adjusted projections,
   **pairwise de-vigged** market probabilities, tiers (Lean / Standard / Strong),
   and a REVIEW flag for implausibly large edges.
5. **Prediction log + grading** — every game run records moneyline, spread, total,
   and rotation player-prop rows with a prediction id, run id, and UTC timestamp;
   `grade-predictions` settles them against game logs and final scores with
   explicit reason codes for anything ungradeable.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Refresh the season (scrape + tables + sigma fit + grading)
wnba-edges refresh-season --season 2026-27

# Features + watchboard (low-sample players excluded by default)
wnba-edges build-features --season 2026-27
wnba-edges edge-board --season 2026-27 --top 25

# Upcoming slate: records moneyline, spread, total, and rotation player props.
wnba-edges build-game-projections --season 2026-27
wnba-edges best-bets --season 2026-27

# Fit per-market prop volatility from real game logs
wnba-edges fit-sigma --season 2026-27

# Optional: pull live player-prop quotes (quota-heavy; requires ODDS_API_KEY)
python -m wnba_edges.market_data --fetch-slate --props
wnba-edges build-prop-projections --season 2026-27

# Price a prop (evaluation is logged for grading; quotes >12h old are refused)
$env:ODDS_API_KEY = "your_key"
python -m wnba_edges.market_data --fetch-game "MIN@PHX" --props
wnba-edges evaluate-player-prop --player "A'ja Wilson" --market player_points `
  --side over --line 24.5 --use-market

# Grade everything that has finished, then view the record
wnba-edges grade-predictions
wnba-edges results

# Build the public dashboard
wnba-edges build-site --season 2026-27
```

## Verify

```bash
ruff check src tests
pytest -q
```

CI runs both plus a dashboard build smoke on every push.

## Outputs

| Path | Contents |
|---|---|
| `data/raw/…` | validated scrape snapshots (JSONL/CSV) |
| `data/processed/…` | season tables, splits, model inputs, `market_sigma_*.csv`, projections |
| `data/predictions/…` | timestamped prop + game prediction logs with grades and reason codes |
| `data/odds/…` | odds snapshots with book attribution and fetch timestamps (not committed) |
| `docs/index.html` | the generated public dashboard |

## Grading rules

- Props grade against the player's first game log in a ±1/+2-day window around the
  slate date; exact-line results are pushes.
- Games grade against final scores matched on date + matchup. Moneyline, spread
  ATS, and total are logged as their own rows; ATS and totals only produce a W-L
  when a book line was captured with the forecast.
- Anything ungradeable is voided with a reason code
  (`player_not_found_in_game_logs`, `void_no_game_within_window`,
  `market_unsupported`, …) — never silently dropped.
- `wnba-edges results` prints W-L-P for moneyline / spread / total / player
  props, winner hit rate, spread/total MAE, and Brier score.

## License

All rights reserved — published for transparency and research review. See
[LICENSE](LICENSE).
