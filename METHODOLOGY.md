# Methodology — WNBA Edge Model

Chase Analytics research software. This document explains what each number on the
dashboard means, how it is produced, and what its known limitations are.

**This is analytics research, not betting advice.** No output is a wager
instruction, and nothing here guarantees an outcome.

## The four layers

1. **Game projections (model layer).** Baseline score, spread, total, and pace from
   team offensive/defensive efficiency ratings and blended pace. Home court is
   estimated empirically from finished games (mean home margin, clamped to 0–4 pts;
   a 1.5-pt prior is used until 20+ games exist). The home-win probability is a
   logistic regression of home wins on the net-rating gap, fit on the season's
   finished games; the dashboard always states the fit basis and sample size.
   *Limitations:* no injury, lineup, or rest adjustments yet; the rating gap uses
   current season-to-date ratings for historical games (mild look-ahead in the fit).

2. **Market snapshot (market layer).** Odds stored from The Odds API with book
   attribution and a `fetched_at` timestamp. Quotes older than 12 hours are refused
   by the evaluator unless explicitly overridden — a fresh model against stale odds
   manufactures phantom edges. When both sides of a line are stored, the implied
   probability is **de-vigged pairwise** (proportional method); a raw single-price
   implied probability is never labeled vig-free.

3. **Edge watchboard (review queue).** A ranked screen over usage, minutes, form,
   and impact signals. Signals are z-scores shrunk toward the league mean by total
   minutes played (prior: 150 minutes), and players under the sample floor
   (fewer than 5 games or under 10 MPG) are flagged `LOW SAMPLE` and excluded from
   the public board. **Watchboard rank is not a bet** — it is a queue of players
   whose role or form is moving, to be priced against a market line.

4. **Graded results (results layer).** Every game run logs **four recorded forecasts**:
   moneyline, spread, total, and the rotation player-prop slate. Each row has a
   prediction id, run id, and UTC timestamp, then is graded against player game logs
   and final scores. Ungradeable predictions carry an explicit reason code and are
   voided — never silently dropped.
   - **Moneyline:** model's favorite (from home-win probability) vs the winner,
     recorded with the captured home/away moneyline when a snapshot exists.
   - **Spread:** ATS vs the captured book home line. A projection without that
     number is not a wager record; direction accuracy is reported separately.
   - **Total:** over/under vs the captured book total, plus total MAE.
   - **Player props:** PTS / REB / AST / 3PM for each rotation player. A W-L is only
     scored when a book line was captured; model-only rows still settle on MAE.

## Prop pricing

- **Projection:** season rate (or the source's projected points where trusted),
  shrunk toward recent form, then adjusted by projected-vs-season minutes when a
  projected role exists. `build-game-projections` auto-builds the rotation slate;
  `evaluate-player-prop` remains available for one-off lines.
- **Distribution:** Normal approximation with a **per-market sigma fitted from real
  player game logs** (pooled within-player game-to-game deviation). Per-player sigma
  is used when the player has 3+ logged games, shrunk toward the league value.
  The evaluator always prints which sigma was used and where it came from.
- **Value:** edge = model probability − (de-vigged) market probability; EV and
  full/quarter Kelly are reported for reference. Unpriced projections are labelled
  model-only and never counted as picks.

## Daily best bets (`hit_likelihood`)

The dashboard's Daily Best Bets list ranks **today's priced sides** (moneyline,
spread, total, and rotation player props) by a single parameter:

```
hit_likelihood = w × shrunk_hist + (1 − w) × model_prob
w = n / (n + 12)
shrunk_hist = (wins + 1) / (n + 2)
```

- **model_prob** is the current projection's probability that the recorded side hits.
- **hist** is the model's graded W-L for that market (latest forecast per finished
  matchup, same uniqueness rule as the public record). When a probability band
  (50–59 / 60–69 / 70%+) has at least 8 samples, that band is used instead of the
  family rate.
- Unpriced rows never enter the list. Tomorrow's games are not mixed into today's
  card. At most four sides per market family, so a strong moneyline record cannot
  occupy every slot.
- This is a research ranking of the model's own record. It is not a wager
  instruction and does not promise that a high `hit_likelihood` will cash.

## Confidence tiers

| Tier | Edge vs market | Suggested review size |
|---|---|---|
| Lean | ≥ 2.0 pts | 0.25–0.5u |
| Standard | ≥ 4.5 pts | 0.5–1.0u |
| Strong | ≥ 8.0 pts | 1.0–2.0u |
| **REVIEW** | ≥ 15 pts | verify inputs — treated as an input error, not a bet |

## Data sources and contracts

WNBAnalytics (player board, games, box scores), ESPN Analytics (advanced box),
Her Hoop Stats (historical research), ESPN scoreboard (upcoming schedule),
The Odds API (market snapshots). Scrapes are validated against light schema
contracts (required fields, minimum row counts) and fail loudly on upstream drift.
Finished-game box scores are cached and never re-downloaded.

## Known limitations

- No injury/news ingestion; a player ruled out after a projection is graded as void
  (`no game within window`), not silently excluded.
- Game projections have no travel/rest/back-to-back adjustments.
- The logistic win-probability fit uses current ratings for past games.
- Prop projections are rate-based; they do not model matchup or usage redistribution.
- Sample sizes are small early in a season; the dashboard labels every fitted
  quantity with its basis and n.
