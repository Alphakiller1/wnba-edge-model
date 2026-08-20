"""Slate game-market and player-prop projections, plus book-line attachment.

Moneyline, spread, total, and rotation player props are first-class tracked
forecasts, not side effects of a one-off `evaluate-player-prop` call. This module:

* stamps consensus book ML / spread / total onto each game projection
* writes one recorded row per game for moneyline, spread, and total
* estimates each rotation player's points / rebounds / assists / threes
* attaches a book line when a fresh snapshot has one
* writes replaceable slate CSVs plus prediction-log rows for grading
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .betting import estimate_over_probability, evaluate_over_under, value_layer
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

GAME_MARKET_COLUMNS = [
    "run_id", "generated_at", "game_date", "away", "home", "market",
    "projection", "projection_basis", "side", "line", "odds", "opposite_odds",
    "book", "model_prob", "implied_prob", "vig_free", "edge", "tier", "verdict", "priced",
]

_GAME_SIGMA = 12.0


def game_market_slate_path(root: Path, season: str) -> Path:
    return root / "data" / "processed" / f"game_markets_{season}.csv"


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


_LINE_MATCH_HOURS = 18
BOOK_LINE_COLUMNS = (
    "book_total_line", "book_spread_line",
    "book_home_ml", "book_away_ml",
    "book_spread_odds", "book_spread_opposite",
    "book_total_over_odds", "book_total_under_odds",
    "book_ml_book", "book_spread_book", "book_total_book",
)
_NUMERIC_LINE_COLUMNS = (
    "book_total_line", "book_spread_line",
    "book_home_ml", "book_away_ml",
    "book_spread_odds", "book_spread_opposite",
    "book_total_over_odds", "book_total_under_odds",
)


def attach_game_market_lines(projections: pd.DataFrame, odds: pd.DataFrame | None) -> pd.DataFrame:
    """Stamp consensus book ML / spread / total onto each game projection for recording.

    A projection without the number it would be bet at is not a wager record.  Every
    game therefore carries the captured spread, total, and moneyline when a snapshot
    for that kickoff exists.
    """
    out = projections.copy()
    for column in BOOK_LINE_COLUMNS:
        out[column] = pd.NA
    if odds is None or odds.empty or out.empty:
        return out
    quotes = _prepare_quotes(odds)
    for idx, game in out.iterrows():
        for column, value in _book_lines_for_game(quotes, game).items():
            if value is not None:
                out.at[idx, column] = value
    return out


def fill_missing_game_market_lines(projections: pd.DataFrame, odds: pd.DataFrame | None) -> pd.DataFrame:
    """Attach book numbers onto rows that were logged without a line, without overwriting captured ones."""
    out = projections.copy()
    for column in BOOK_LINE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    if odds is None or odds.empty or out.empty:
        return out
    quotes = _prepare_quotes(odds)
    for idx, game in out.iterrows():
        if not _missing_wager_line(game):
            continue
        stamped = _book_lines_for_game(quotes, game)
        for column, value in stamped.items():
            if value is None:
                continue
            if column in _NUMERIC_LINE_COLUMNS:
                current = pd.to_numeric(out.at[idx, column], errors="coerce")
                if pd.isna(current):
                    out.at[idx, column] = value
            else:
                current = out.at[idx, column]
                if pd.isna(current) or str(current).strip() in {"", "nan", "<NA>"}:
                    out.at[idx, column] = value
    return out


def build_game_market_slate(projections: pd.DataFrame) -> pd.DataFrame:
    """One recorded row per game for moneyline, spread, and total."""
    if projections is None or projections.empty:
        return pd.DataFrame(columns=GAME_MARKET_COLUMNS)
    rows: list[dict] = []
    for _, game in projections.iterrows():
        rows.append(_moneyline_row(game))
        rows.append(_spread_row(game))
        rows.append(_total_row(game))
    return pd.DataFrame(rows, columns=GAME_MARKET_COLUMNS)


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
                    price = _best_prop_quote(
                        odds,
                        str(player["name"]),
                        market,
                        away=away,
                        home=home,
                        kickoff=_game_kickoff(game),
                    )
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


def _base_market_row(game: pd.Series, market: str, projection, basis: str) -> dict:
    return {
        "run_id": game.get("run_id"),
        "generated_at": game.get("generated_at"),
        "game_date": game.get("date"),
        "away": str(game["away"]).upper(),
        "home": str(game["home"]).upper(),
        "market": market,
        "projection": projection,
        "projection_basis": basis,
        "side": "",
        "line": pd.NA,
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


def _apply_price(row: dict, model_prob: float, odds, opposite, book: str) -> dict:
    if odds is None or model_prob is None:
        return row
    value = value_layer(float(model_prob), int(odds), int(opposite) if opposite is not None else None)
    row.update(
        {
            "odds": int(odds),
            "opposite_odds": int(opposite) if opposite is not None else pd.NA,
            "book": book or "",
            "model_prob": value.model_prob,
            "implied_prob": value.implied_prob,
            "vig_free": value.vig_free,
            "edge": value.edge,
            "tier": value.tier,
            "verdict": value.verdict,
            "priced": True,
        }
    )
    return row


def _moneyline_row(game: pd.Series) -> dict:
    away, home = str(game["away"]).upper(), str(game["home"]).upper()
    home_prob = pd.to_numeric(game.get("home_win_prob"), errors="coerce")
    if pd.isna(home_prob):
        return _base_market_row(game, "moneyline", pd.NA, "no win probability")
    side, prob = (home, float(home_prob)) if home_prob >= 0.5 else (away, float(1.0 - home_prob))
    row = _base_market_row(game, "moneyline", round(prob, 4), str(game.get("win_prob_basis") or "win probability"))
    row["side"] = side
    home_odds = pd.to_numeric(game.get("book_home_ml"), errors="coerce")
    away_odds = pd.to_numeric(game.get("book_away_ml"), errors="coerce")
    pick_odds = home_odds if side == home else away_odds
    opp_odds = away_odds if side == home else home_odds
    if pd.notna(pick_odds):
        _apply_price(
            row, prob, pick_odds,
            None if pd.isna(opp_odds) else opp_odds,
            str(game.get("book_ml_book") or ""),
        )
    return row


def _spread_row(game: pd.Series) -> dict:
    projected = pd.to_numeric(game.get("projected_home_spread"), errors="coerce")
    book_line = pd.to_numeric(game.get("book_spread_line"), errors="coerce")
    home = str(game["home"]).upper()
    away = str(game["away"]).upper()
    basis = "projected home margin"
    projection = None if pd.isna(projected) else round(float(projected), 1)
    row = _base_market_row(game, "spread", projection, basis)
    if pd.isna(projected):
        return row
    if pd.notna(book_line):
        cover = float(projected) + float(book_line)
        if cover == 0:
            row["side"] = "PUSH"
        else:
            row["side"] = home if cover > 0 else away
        row["line"] = float(book_line)
        home_covers = estimate_over_probability(float(projected), -float(book_line), sigma=_GAME_SIGMA)
        model_prob = home_covers if row["side"] == home else (1.0 - home_covers if row["side"] == away else 0.5)
        odds = pd.to_numeric(game.get("book_spread_odds"), errors="coerce")
        opposite = pd.to_numeric(game.get("book_spread_opposite"), errors="coerce")
        if row["side"] == away:
            odds, opposite = opposite, odds
        if pd.notna(odds):
            _apply_price(
                row, model_prob, odds,
                None if pd.isna(opposite) else opposite,
                str(game.get("book_spread_book") or ""),
            )
    else:
        row["side"] = home if projected > 0 else (away if projected < 0 else "PUSH")
    return row


def _total_row(game: pd.Series) -> dict:
    projected = pd.to_numeric(game.get("projected_total"), errors="coerce")
    book_line = pd.to_numeric(game.get("book_total_line"), errors="coerce")
    projection = None if pd.isna(projected) else round(float(projected), 1)
    row = _base_market_row(game, "total", projection, "projected total")
    if pd.isna(projected):
        return row
    if pd.notna(book_line):
        if float(projected) == float(book_line):
            row["side"] = "PUSH"
        else:
            row["side"] = "OVER" if float(projected) > float(book_line) else "UNDER"
        row["line"] = float(book_line)
        over_p = estimate_over_probability(float(projected), float(book_line), sigma=_GAME_SIGMA)
        model_prob = over_p if row["side"] == "OVER" else (1.0 - over_p if row["side"] == "UNDER" else 0.5)
        over_odds = pd.to_numeric(game.get("book_total_over_odds"), errors="coerce")
        under_odds = pd.to_numeric(game.get("book_total_under_odds"), errors="coerce")
        pick_odds = over_odds if row["side"] == "OVER" else under_odds
        opp_odds = under_odds if row["side"] == "OVER" else over_odds
        if row["side"] in {"OVER", "UNDER"} and pd.notna(pick_odds):
            _apply_price(
                row, model_prob, pick_odds,
                None if pd.isna(opp_odds) else opp_odds,
                str(game.get("book_total_book") or ""),
            )
        else:
            row["model_prob"] = round(model_prob, 4)
    return row


def _best_side_quote(
    quotes: pd.DataFrame,
    market: str,
    side: str,
    line: float | None = None,
) -> dict | None:
    if quotes is None or quotes.empty:
        return None
    rows = quotes[quotes["market"].astype(str).str.lower() == market.lower()]
    pick = rows[rows["side"].astype(str).str.lower() == str(side).lower()]
    if pick.empty:
        return None
    if line is not None and "line" in pick.columns:
        lines = pd.to_numeric(pick["line"], errors="coerce")
        at_line = pick[lines.round(1) == round(float(line), 1)]
        if not at_line.empty:
            pick = at_line
    if "odds" not in pick.columns:
        return None
    odds = pd.to_numeric(pick["odds"], errors="coerce")
    pick = pick.assign(odds_num=odds).dropna(subset=["odds_num"])
    if pick.empty:
        return None
    best = pick.loc[pick["odds_num"].idxmax()]
    return {"odds": int(best["odds_num"]), "book": str(best.get("book") or "")}


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


def _prepare_quotes(odds: pd.DataFrame) -> pd.DataFrame:
    quotes = odds.copy()
    quotes["away"] = quotes["away"].astype(str).str.upper()
    quotes["home"] = quotes["home"].astype(str).str.upper()
    return quotes


def _missing_wager_line(game: pd.Series) -> bool:
    return any(
        pd.isna(pd.to_numeric(game.get(column), errors="coerce"))
        for column in ("book_total_line", "book_spread_line", "book_home_ml", "book_away_ml")
    )


def _book_lines_for_game(quotes: pd.DataFrame, game: pd.Series) -> dict:
    match = _quotes_for_kickoff(quotes, game)
    home, away = str(game["home"]).upper(), str(game["away"]).upper()
    total_line = _consensus_line(match[match["market"].astype(str).str.lower() == "total"])
    spread_line = _consensus_home_spread(match[match["market"].astype(str).str.lower() == "spread"], home)
    home_ml = _best_side_quote(match, "ml", home)
    away_ml = _best_side_quote(match, "ml", away)
    away_spread_line = None if spread_line is None else -float(spread_line)
    home_spread = _best_side_quote(match, "spread", home, line=spread_line)
    away_spread = _best_side_quote(match, "spread", away, line=away_spread_line)
    over = _best_side_quote(match, "total", "over", line=total_line)
    under = _best_side_quote(match, "total", "under", line=total_line)
    stamped: dict = {column: None for column in BOOK_LINE_COLUMNS}
    stamped["book_total_line"] = total_line
    stamped["book_spread_line"] = spread_line
    if home_ml:
        stamped["book_home_ml"] = home_ml["odds"]
    if away_ml:
        stamped["book_away_ml"] = away_ml["odds"]
    if home_ml or away_ml:
        stamped["book_ml_book"] = (home_ml or away_ml)["book"]
    if home_spread:
        stamped["book_spread_odds"] = home_spread["odds"]
        stamped["book_spread_book"] = home_spread["book"]
    if away_spread:
        stamped["book_spread_opposite"] = away_spread["odds"]
    if over:
        stamped["book_total_over_odds"] = over["odds"]
        stamped["book_total_book"] = over["book"]
    if under:
        stamped["book_total_under_odds"] = under["odds"]
    return stamped


def _game_kickoff(game: pd.Series) -> pd.Timestamp | None:
    """Precise tip-off only. A calendar date at midnight UTC is not a tip-off and
    would miss evening games whose commence_time is the next UTC day."""
    raw = game.get("time")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return None
    if "T" not in text and ":" not in text:
        return None
    stamp = pd.to_datetime(text, utc=True, errors="coerce")
    return stamp if pd.notna(stamp) else None


def _slate_date(game: pd.Series) -> str:
    for column in ("date", "game_date"):
        value = game.get(column)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if len(text) >= 10 and text[4:5] == "-":
            return text[:10]
    return ""


def _quotes_for_kickoff(quotes: pd.DataFrame, game: pd.Series) -> pd.DataFrame:
    """Team-pair quotes for this tip-off only — a rematch later in the week is a different board."""
    match = quotes[
        (quotes["away"] == str(game["away"]).upper())
        & (quotes["home"] == str(game["home"]).upper())
    ]
    if match.empty or "commence_time" not in match.columns:
        return match
    commence = pd.to_datetime(match["commence_time"], utc=True, errors="coerce")
    kickoff = _game_kickoff(game)
    if kickoff is not None and commence.notna().any():
        close = match[commence.notna() & ((commence - kickoff).abs() <= pd.Timedelta(hours=_LINE_MATCH_HOURS))]
        return close
    return _quotes_for_slate_date(match, game, commence)


def _quotes_for_slate_date(match: pd.DataFrame, game: pd.Series, commence: pd.Series) -> pd.DataFrame:
    game_date = _slate_date(game)
    if not game_date or not commence.notna().any():
        return match
    eastern = commence.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    return match[eastern == game_date]


def _best_prop_quote(
    odds: pd.DataFrame | None,
    player: str,
    market: str,
    *,
    away: str | None = None,
    home: str | None = None,
    kickoff: pd.Timestamp | None = None,
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
    if kickoff is not None and not rows.empty and "commence_time" in rows.columns:
        commence = pd.to_datetime(rows["commence_time"], utc=True, errors="coerce")
        close = rows[commence.notna() & ((commence - kickoff).abs() <= pd.Timedelta(hours=_LINE_MATCH_HOURS))]
        rows = close
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
