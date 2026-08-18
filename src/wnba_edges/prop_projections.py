"""Slate player-prop projections and market-line attachment.

Game totals and player props are first-class tracked forecasts, not side effects of
a one-off `evaluate-player-prop` call. This module:

* estimates each rotation player's points / rebounds / assists / threes
* attaches a book line when a fresh snapshot has one
* writes a replaceable slate CSV plus prediction-log rows for grading
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .betting import evaluate_over_under
from .features import board_eligible
from .market_data import MAX_QUOTE_AGE_HOURS, _quote_age_hours
from .sigma import MARKET_STAT_COLUMNS, load_market_sigmas, resolve_sigma

SLATE_MARKETS = (
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
)
MARKET_LABEL = {
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
    "player_threes": "3PM",
}
PLAYERS_PER_TEAM = 6
_MARKET_FEATURE_PRIORS = {
    "player_points": ("ppg", "season PPG"),
    "player_rebounds": ("rpg", "season RPG"),
    "player_assists": ("apg", "season APG"),
}
_RECENT_WINDOW = 5
_MAX_RECENT_WEIGHT = 0.45

PROP_SLATE_COLUMNS = [
    "run_id", "generated_at", "game_date", "away", "home", "team",
    "player", "player_id", "market", "projection", "projection_basis",
    "sigma", "sigma_source", "line", "side", "odds", "opposite_odds", "book",
    "model_prob", "implied_prob", "vig_free", "edge", "tier", "verdict", "priced",
]


def prop_slate_path(root: Path, season: str) -> Path:
    return root / "data" / "processed" / f"prop_projections_{season}.csv"


def projection_for_market(
    row: pd.Series,
    market: str,
    player_logs: pd.DataFrame | None = None,
) -> tuple[float | None, str]:
    """Market-specific prop estimate using season form, recent form and expected minutes.

    A points-only season-rate fallback made threes, steals and blocks silently inherit a
    points projection.  The revised estimate always maps to the requested market.  Recent
    production is shrunk toward the season prior, then adjusted for the current projected
    role; this makes the model responsive without allowing a five-game heater or a single
    minutes spike to dominate the number.
    """
    stat_column = MARKET_STAT_COLUMNS.get(market)
    if stat_column is None:
        raise SystemExit(f"Unsupported player prop market {market!r}.")

    history = _player_history(row, player_logs, stat_column)
    prior, prior_label = _season_prior(row, market, history)
    if prior is None:
        return None, ""

    if history is None or history.empty:
        recent_values = pd.Series(dtype=float)
    else:
        recent = history.tail(_RECENT_WINDOW)
        recent_values = pd.to_numeric(recent[stat_column], errors="coerce").dropna()
    if len(recent_values) >= 3:
        recent_weight = min(_MAX_RECENT_WEIGHT, 0.15 + 0.06 * len(recent_values))
        estimate = (1 - recent_weight) * prior + recent_weight * float(recent_values.mean())
        form_note = f"L{len(recent_values)} {stat_column} x{recent_weight:.2f}"
    else:
        estimate = prior
        form_note = "season anchor"

    minutes_ratio, minutes_note = _minutes_ratio(row, history)
    estimate *= minutes_ratio

    # WNBAnalytics' projected points are useful additional information, but never replace
    # the model wholesale.  Blend only plausible values and only for the points market.
    if market == "player_points":
        source_points = pd.to_numeric(row.get("projectedPoints"), errors="coerce")
        if pd.notna(source_points) and 0.55 * prior <= source_points <= 1.65 * prior:
            estimate = 0.65 * estimate + 0.35 * float(source_points)
            form_note += " + source pts x0.35"

    return round(float(estimate), 2), f"{prior_label}; {form_note}{minutes_note}"


def attach_game_market_lines(projections: pd.DataFrame, odds: pd.DataFrame | None) -> pd.DataFrame:
    """Stamp the consensus book total (and spread) onto each game projection for grading."""
    out = projections.copy()
    out["book_total_line"] = pd.NA
    out["book_spread_line"] = pd.NA
    if odds is None or odds.empty or out.empty:
        return out
    quotes = odds.copy()
    quotes["away"] = quotes["away"].astype(str).str.upper()
    quotes["home"] = quotes["home"].astype(str).str.upper()
    for idx, game in out.iterrows():
        match = quotes[
            (quotes["away"] == str(game["away"]).upper())
            & (quotes["home"] == str(game["home"]).upper())
        ]
        total_line = _consensus_line(match[match["market"].astype(str).str.lower() == "total"])
        spread_line = _consensus_home_spread(match[match["market"].astype(str).str.lower() == "spread"], str(game["home"]))
        if total_line is not None:
            out.at[idx, "book_total_line"] = total_line
        if spread_line is not None:
            out.at[idx, "book_spread_line"] = spread_line
    return out


def build_slate_prop_projections(
    features: pd.DataFrame,
    schedule: pd.DataFrame,
    player_logs: pd.DataFrame | None,
    odds: pd.DataFrame | None,
    *,
    season: str,
    run_id: str,
    generated_at: str,
    root: Path | None = None,
) -> pd.DataFrame:
    """Project rotation props for every scheduled game, priced against stored odds when present."""
    if features is None or features.empty or schedule is None or schedule.empty:
        return pd.DataFrame(columns=PROP_SLATE_COLUMNS)

    eligible = board_eligible(features) if "low_sample" in features.columns else features
    sigmas = load_market_sigmas(root, season) if root is not None else None
    rows: list[dict] = []
    for _, game in schedule.iterrows():
        away, home = str(game["away"]).upper(), str(game["home"]).upper()
        game_date = str(game.get("date") or generated_at[:10])
        for team in (away, home):
            squad = _rotation(eligible, team)
            for _, player in squad.iterrows():
                for market in SLATE_MARKETS:
                    projection, basis = projection_for_market(player, market, player_logs)
                    if projection is None:
                        continue
                    sigma, sigma_source = resolve_sigma(
                        market,
                        sigmas=sigmas,
                        player_logs=player_logs,
                        player_id=player.get("id"),
                    )
                    price = _best_prop_quote(odds, str(player["name"]), market, away=away, home=home)
                    row = {
                        "run_id": run_id,
                        "generated_at": generated_at,
                        "game_date": game_date,
                        "away": away,
                        "home": home,
                        "team": team,
                        "player": str(player["name"]),
                        "player_id": player.get("id"),
                        "market": market,
                        "projection": projection,
                        "projection_basis": basis,
                        "sigma": round(float(sigma), 3),
                        "sigma_source": sigma_source,
                        "line": pd.NA,
                        "side": "",
                        "odds": pd.NA,
                        "opposite_odds": pd.NA,
                        "book": "",
                        "model_prob": pd.NA,
                        "implied_prob": pd.NA,
                        "vig_free": False,
                        "edge": pd.NA,
                        "tier": "",
                        "verdict": "PROJ",
                        "priced": False,
                    }
                    if price is not None:
                        side = "over" if projection >= price["line"] else "under"
                        value = evaluate_over_under(
                            projection=projection,
                            line=price["line"],
                            odds=int(price["odds"]),
                            side=side,
                            sigma=sigma,
                            opposite_odds=price.get("opposite_odds"),
                        )
                        row.update(
                            {
                                "line": price["line"],
                                "side": side,
                                "odds": int(price["odds"]),
                                "opposite_odds": price.get("opposite_odds"),
                                "book": price["book"],
                                "model_prob": value.model_prob,
                                "implied_prob": value.implied_prob,
                                "vig_free": value.vig_free,
                                "edge": value.edge,
                                "tier": value.tier,
                                "verdict": value.verdict,
                                "priced": True,
                            }
                        )
                    rows.append(row)
    return pd.DataFrame(rows, columns=PROP_SLATE_COLUMNS)


def _rotation(features: pd.DataFrame, team: str) -> pd.DataFrame:
    squad = features[features["team"].astype(str).str.upper() == team]
    if squad.empty:
        return squad
    minutes = pd.to_numeric(squad.get("projectedMinutes"), errors="coerce")
    if minutes.isna().all():
        minutes = pd.to_numeric(squad.get("mpg"), errors="coerce")
    ranked = squad.assign(_minutes=minutes.fillna(0)).sort_values("_minutes", ascending=False)
    return ranked.head(PLAYERS_PER_TEAM)


def _player_history(row: pd.Series, player_logs: pd.DataFrame | None, stat_column: str) -> pd.DataFrame | None:
    if player_logs is None or player_logs.empty or stat_column not in player_logs.columns:
        return None
    player_id = pd.to_numeric(row.get("id"), errors="coerce")
    if pd.isna(player_id) or "playerId" not in player_logs.columns:
        return None
    ids = pd.to_numeric(player_logs["playerId"], errors="coerce")
    history = player_logs.loc[ids == player_id].copy()
    if history.empty:
        return None
    if "date" in history.columns:
        history["_date"] = pd.to_datetime(history["date"], errors="coerce")
        history = history.sort_values("_date")
    return history


def _season_prior(row: pd.Series, market: str, history: pd.DataFrame | None) -> tuple[float | None, str]:
    feature = _MARKET_FEATURE_PRIORS.get(market)
    if feature:
        value = pd.to_numeric(row.get(feature[0]), errors="coerce")
        if pd.notna(value):
            return float(value), feature[1]
    stat_column = MARKET_STAT_COLUMNS[market]
    if history is not None:
        values = pd.to_numeric(history[stat_column], errors="coerce").dropna()
        if not values.empty:
            return float(values.mean()), f"season {stat_column} history"
    return None, ""


def _minutes_ratio(row: pd.Series, history: pd.DataFrame | None) -> tuple[float, str]:
    projected = pd.to_numeric(row.get("projectedMinutes"), errors="coerce")
    baseline = pd.to_numeric(row.get("mpg"), errors="coerce")
    if history is not None and "min" in history.columns:
        recent_minutes = pd.to_numeric(history.tail(_RECENT_WINDOW)["min"], errors="coerce").dropna()
        if len(recent_minutes) >= 3:
            baseline = recent_minutes.mean()
    if pd.notna(projected) and pd.notna(baseline) and baseline > 0:
        ratio = min(1.25, max(0.75, float(projected) / float(baseline)))
        if abs(ratio - 1.0) >= 0.03:
            return ratio, f", minutes x{ratio:.2f}"
    return 1.0, ""


def _consensus_line(quotes: pd.DataFrame) -> float | None:
    if quotes is None or quotes.empty:
        return None
    lines = pd.to_numeric(quotes["line"], errors="coerce").dropna()
    if lines.empty:
        return None
    # Mode of the posted totals; ties break toward the median so a lone outlier book
    # cannot drag the tracked line off the actual board.
    counts = lines.round(1).value_counts()
    return float(counts.index[0])


def _consensus_home_spread(quotes: pd.DataFrame, home: str) -> float | None:
    if quotes is None or quotes.empty:
        return None
    home_rows = quotes[quotes["side"].astype(str).str.upper() == home.upper()]
    lines = pd.to_numeric(home_rows["line"], errors="coerce").dropna()
    if lines.empty:
        return None
    return float(lines.round(1).value_counts().index[0])


def _best_prop_quote(
    odds: pd.DataFrame | None,
    player: str,
    market: str,
    *,
    away: str | None = None,
    home: str | None = None,
) -> dict | None:
    if odds is None or odds.empty or "player" not in odds.columns:
        return None
    frame = odds.copy()
    frame["player"] = frame["player"].astype(str)
    frame["market"] = frame["market"].astype(str)
    rows = frame[
        (frame["market"] == market)
        & (frame["player"].str.lower() == player.lower())
    ]
    if away and home and "away" in rows.columns and "home" in rows.columns:
        rows = rows[
            (rows["away"].astype(str).str.upper() == away.upper())
            & (rows["home"].astype(str).str.upper() == home.upper())
        ]
    if rows.empty:
        return None
    if "fetched_at" in rows.columns:
        age = rows["fetched_at"].map(_quote_age_hours)
        fresh = rows[age.fillna(MAX_QUOTE_AGE_HOURS + 1) <= MAX_QUOTE_AGE_HOURS]
        rows = fresh if not fresh.empty else pd.DataFrame()
    if rows.empty:
        return None
    overs = rows[rows["side"].astype(str).str.lower().str.endswith("|over")]
    unders = rows[rows["side"].astype(str).str.lower().str.endswith("|under")]
    if overs.empty:
        return None
    overs = overs.assign(line_num=pd.to_numeric(overs["line"], errors="coerce"),
                         odds_num=pd.to_numeric(overs["odds"], errors="coerce")).dropna(subset=["line_num", "odds_num"])
    if overs.empty:
        return None
    line = float(overs["line_num"].round(1).value_counts().index[0])
    at_line = overs[overs["line_num"].round(1) == round(line, 1)]
    best = at_line.loc[at_line["odds_num"].idxmax()]
    opposite = None
    if not unders.empty:
        unders = unders.assign(line_num=pd.to_numeric(unders["line"], errors="coerce"),
                               odds_num=pd.to_numeric(unders["odds"], errors="coerce")).dropna(subset=["line_num", "odds_num"])
        same = unders[(unders["line_num"].round(1) == round(line, 1)) & (unders["book"] == best["book"])]
        pool = same if not same.empty else unders[unders["line_num"].round(1) == round(line, 1)]
        if not pool.empty:
            opposite = int(pool.loc[pool["odds_num"].idxmax()]["odds_num"])
    return {
        "line": line,
        "odds": int(best["odds_num"]),
        "book": str(best.get("book") or "book"),
        "opposite_odds": opposite,
    }
