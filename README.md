# WNBA Edge Model

Early WNBA betting/research pipeline focused on the things that usually move props first:

- player usage and minutes stability
- pace and team environment
- advanced impact metrics
- expected minutes / expected points
- injury/news flags
- splits and recent form

The first live scraper targets WNBAnalytics because its rendered app payload currently exposes a rich player board with projected minutes, expected points, RAPM, usage, shot mix, role trust, and confidence fields. The WNBA Stats client is included for official stats endpoints, but those endpoints can rate-limit or timeout, so treat it as a second source and cache aggressively.

Betting logic follows the Chase Analytics Brain / Betting Brain pattern:

- odds -> implied probability
- model probability -> edge
- expected value per unit
- fair odds
- full and quarter Kelly
- confidence tiers: Lean, Standard, Strong
- implausibly large edges become REVIEW, not auto-bets
- odds snapshots can support best-price shopping and closing-line value tracking

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

python -m wnba_edges.cli scrape-wnbanalytics --season 2026-27
python -m wnba_edges.cli build-features --season 2026-27
python -m wnba_edges.cli edge-board --season 2026-27 --top 25

python -m wnba_edges.cli refresh-season --season 2026-27
python -m wnba_edges.cli build-features --season 2026-27

python -m wnba_edges.cli evaluate-player-prop --player "A'ja Wilson" --market player_points --side over --line 24.5 --odds -110
python -m wnba_edges.cli scrape-herhoopstats --research-type player_single_games --min-season 2026 --max-season 2026
python -m wnba_edges.cli scrape-espnanalytics-box --id 20250811-1022500204
```

After editable install, `wnba-edges ...` is also available if your Python Scripts directory is on PATH.

## Odds Snapshots

The odds module mirrors the Betting Brain evaluator approach but uses WNBA markets.

```powershell
$env:ODDS_API_KEY = "your_key"
python -m wnba_edges.market_data --fetch-game "MIN@PHX" --props
python -m wnba_edges.cli evaluate-player-prop --player "A'ja Wilson" --market player_points --side over --line 24.5 --use-market
```

Odds snapshots are stored in:

- `data/odds/odds_latest.csv`
- `data/odds/odds_history.csv`

Outputs land in:

- `data/raw/wnbanalytics_players_2026-27.jsonl`
- `data/raw/wnbanalytics_games_2026-27.jsonl`
- `data/raw/wnbanalytics_game_details_2026-27.jsonl`
- `data/raw/herhoopstats_player_single_games_2026_2026_traditional.csv`
- `data/raw/espnanalytics_YYYYMMDD_GAMEID_player_box.csv`
- `data/raw/espnanalytics_YYYYMMDD_GAMEID_player_actions.csv`
- `data/raw/espnanalytics_YYYYMMDD_GAMEID_team_box.csv`
- `data/raw/espnanalytics_YYYYMMDD_GAMEID_four_factors.csv`
- `data/processed/player_features_2026-27.csv`
- `data/processed/edge_board_2026-27.csv`

Season refresh adds:

- `data/processed/game_results_2026-27.csv`: every completed game outcome, score, pace, total, margin.
- `data/processed/team_game_logs_2026-27.csv`: one row per team-game with box totals and pace context.
- `data/processed/player_game_logs_2026-27.csv`: one row per player-game with minutes, starter flag, box stats, PRA, usage proxy, and fantasy-simple score.
- `data/processed/player_splits_2026-27.csv`: player splits by overall, home/away, W/L, starter/bench, opponent, team, position, month, last 3, last 5, last 10.
- `data/processed/team_splits_2026-27.csv`: team splits by overall, home/away, W/L, opponent, month.
- `data/processed/player_model_inputs_2026-27.csv`: wide modeling table joining season advanced metrics, usage, recent form, splits, volatility, and deltas.

## Source Notes

Suggested source tiers:

1. WNBAnalytics: rich player board, expected stats, RAPM, shot mix, role trust.
2. ESPN Analytics advanced box: Net Points, offensive/defensive usage, WPA, assisted-shot roles, four-factor net points, and player action-level contribution rows. Strong for player role quality and prop-context features.
3. Her Hoop Stats reSEARCH: long-horizon WNBA player/team history, fantasy scoring, advanced criteria, and shot-detail surfaces. Anonymous access appears preview-limited, so use logged-in exports or CSVs for full historical backfill.
4. WNBA Stats: official player/team dashboards and injury report pages.
5. SportsDataverse/wehoop: daily refreshed ESPN/NBA Stats data repositories.
6. Paid or gated sources such as BBall Index and Hoopology: use exports/API access where permitted; do not scrape gated content without permission.

## Model Direction

This scaffold is not a blind betting bot. It is a data engine for surfacing review candidates:

- `minutes_signal`: projected role versus season MPG.
- `usage_signal`: high-usage players with stable minutes.
- `recent_minutes_signal`: last-5 minutes movement from player game logs.
- `recent_usage_signal`: last-5 usage-proxy movement from player game logs.
- `recent_pra_signal`: last-5 PRA movement from player game logs.
- `pace_signal`: team/game environment once schedules and team pace are joined.
- `impact_signal`: RAPM/CVI/BPM blend where available.
- `volatility_penalty`: discounts high-PRA volatility until market-specific errors are calibrated.
- `confidence_penalty`: discounts low-sample or unstable projections.
- `edge_score`: weighted rank for manual odds comparison.
- `value_layer`: Betting Brain-style price check after a market line is supplied.

Next useful additions:

- odds ingestion by book/market/player
- injury report snapshots with timestamps
- game schedule join with opponent pace/defense
- player prop projection model trained on game logs
- closing-line value tracking
