"""Timestamped prediction log + grading loop.

Every evaluated prop and every game-projection run is persisted with a
prediction id, run id, and UTC timestamp, then graded against player game logs
and finished game results. Ungradeable rows carry an explicit reason code —
nothing is silently skipped.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .sigma import MARKET_STAT_COLUMNS

PROP_COLUMNS = [
    "prediction_id", "run_id", "sport", "season", "recorded_at", "game_date",
    "player", "player_id", "market", "side", "line", "odds", "opposite_odds",
    "odds_source", "quote_age_hours", "projection", "projection_basis",
    "sigma", "sigma_source", "model_prob", "implied_prob", "vig_free",
    "edge", "ev_per_unit", "tier", "verdict",
    "settled", "won", "push", "actual", "ungraded_reason", "graded_at",
]

GAME_COLUMNS = [
    "prediction_id", "run_id", "sport", "season", "recorded_at", "date", "away", "home",
    "projected_away_pts", "projected_home_pts", "projected_total",
    "projected_home_spread", "home_win_prob", "win_prob_basis",
    "book_total_line", "book_spread_line", "book_home_ml", "book_away_ml",
    "predicted_ml_side", "predicted_spread_ats", "predicted_total_side",
    "settled", "actual_away_pts", "actual_home_pts", "actual_total", "actual_winner",
    "home_win", "winner_correct", "predicted_spread_side", "spread_side_correct",
    "spread_ats_correct", "total_side_correct",
    "spread_error", "total_error",
    "ungraded_reason", "graded_at",
]

MARKET_COLUMNS = [
    "prediction_id", "run_id", "sport", "season", "recorded_at", "game_date",
    "away", "home", "market", "side", "line", "odds", "opposite_odds",
    "odds_source", "projection", "projection_basis",
    "model_prob", "implied_prob", "vig_free", "edge", "tier", "verdict", "priced",
    "settled", "won", "push", "actual", "ungraded_reason", "graded_at",
]

# Grade a prop against the player's first game in [game_date-1, game_date+2]
# (recorded_at is UTC; slate dates are US-eastern). Void after the window plus
# this many grace days with no matching game log.
PROP_VOID_AFTER_DAYS = 4
GAME_VOID_AFTER_DAYS = 4

REASON_PLAYER_NOT_FOUND = "player_not_found_in_game_logs"
REASON_NO_GAME_IN_WINDOW = "no_game_in_window_yet"
REASON_MARKET_UNSUPPORTED = "market_unsupported"
REASON_RESULT_PENDING = "result_not_available_yet"
REASON_VOID_NO_GAME = "void_no_game_within_window"
REASON_VOID_NO_RESULT = "void_no_result_recorded"


def predictions_dir(root: Path) -> Path:
    return root / "data" / "predictions"


def prop_log_path(root: Path) -> Path:
    return predictions_dir(root) / "prop_predictions.csv"


def game_log_path(root: Path) -> Path:
    return predictions_dir(root) / "game_predictions.csv"


def market_log_path(root: Path) -> Path:
    return predictions_dir(root) / "market_predictions.csv"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns, dtype="object")
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    # Object dtype throughout: the log mixes strings/bools/floats per column and
    # pandas 3 refuses cross-dtype assignment into inferred numeric columns.
    return frame[columns].astype("object")


def _save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def log_prop_prediction(root: Path, row: dict) -> str:
    prediction_id = uuid.uuid4().hex
    record = {column: row.get(column, pd.NA) for column in PROP_COLUMNS}
    record.update(
        prediction_id=prediction_id,
        sport="wnba",
        recorded_at=_now_iso(),
        settled=False,
        ungraded_reason="",
    )
    if not isinstance(record.get("run_id"), str) or not record["run_id"]:
        record["run_id"] = uuid.uuid4().hex[:12]
    if not isinstance(record.get("game_date"), str) or not record["game_date"]:
        record["game_date"] = record["recorded_at"][:10]
    frame = _load(prop_log_path(root), PROP_COLUMNS)
    frame = pd.concat([frame, pd.DataFrame([record])], ignore_index=True)
    _save(frame, prop_log_path(root))
    return prediction_id


def log_prop_predictions_batch(root: Path, slate: pd.DataFrame, season: str) -> int:
    """Persist a slate of prop projections, idempotent on run + game + player + market.

    Priced rows keep their book line for a later W-L grade. Model-only rows still log the
    projection so they can be scored on MAE once the box score arrives — they never invent
    a win/loss against a line that was not captured.
    """
    if slate is None or slate.empty:
        return 0
    frame = _load(prop_log_path(root), PROP_COLUMNS)
    existing = set(
        zip(
            frame["run_id"].astype(str),
            frame["game_date"].astype(str),
            frame["player"].astype(str),
            frame["market"].astype(str),
        )
    )
    added = 0
    rows = []
    recorded_at = _now_iso()
    for _, item in slate.iterrows():
        run_id = str(item.get("run_id") or "")
        key = (run_id, str(item["game_date"]), str(item["player"]), str(item["market"]))
        if key in existing:
            continue
        rows.append(
            {
                "prediction_id": uuid.uuid4().hex,
                "run_id": run_id,
                "sport": "wnba",
                "season": season,
                "recorded_at": item.get("generated_at") or recorded_at,
                "game_date": item["game_date"],
                "player": item["player"],
                "player_id": item.get("player_id"),
                "market": item["market"],
                "side": item.get("side") or "",
                "line": item.get("line"),
                "odds": item.get("odds"),
                "opposite_odds": item.get("opposite_odds"),
                "odds_source": item.get("book") or "",
                "quote_age_hours": pd.NA,
                "projection": item.get("projection"),
                "projection_basis": item.get("projection_basis"),
                "sigma": item.get("sigma"),
                "sigma_source": item.get("sigma_source"),
                "model_prob": item.get("model_prob"),
                "implied_prob": item.get("implied_prob"),
                "vig_free": item.get("vig_free"),
                "edge": item.get("edge"),
                "ev_per_unit": pd.NA,
                "tier": item.get("tier") or "",
                "verdict": item.get("verdict") or "",
                "settled": False,
                "ungraded_reason": "",
            }
        )
        added += 1
    if rows:
        frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
        _save(frame, prop_log_path(root))
    return added


def log_game_projections(root: Path, projections: pd.DataFrame, season: str) -> int:
    """Persist every projection run, idempotently within one run and matchup.

    A projection is an audit event, not merely the final forecast for a game.  Keeping each
    run makes it possible to see how the model moved as fresh data arrived; the dashboard
    reports headline accuracy from the latest pregame run per matchup so repeat refreshes do
    not manufacture extra wins or losses.
    """
    if projections.empty:
        return 0
    frame = _load(game_log_path(root), GAME_COLUMNS)
    existing = set(
        zip(
            frame["run_id"].astype(str), frame["date"].astype(str),
            frame["away"].astype(str), frame["home"].astype(str),
        )
    )
    added = 0
    rows = []
    for _, game in projections.iterrows():
        run_id = str(game.get("run_id", "") or "")
        key = (run_id, str(game["date"]), str(game["away"]), str(game["home"]))
        if key in existing:
            continue
        book_total = _as_float(game.get("book_total_line"))
        book_spread = _as_float(game.get("book_spread_line"))
        book_home_ml = _as_float(game.get("book_home_ml"))
        book_away_ml = _as_float(game.get("book_away_ml"))
        projected_total = _as_float(game.get("projected_total"))
        win_prob = _as_float(game.get("home_win_prob"))
        projected_spread = _as_float(game.get("projected_home_spread"))
        rows.append(
            {
                "prediction_id": uuid.uuid4().hex,
                "run_id": run_id,
                "sport": "wnba",
                "season": season,
                "recorded_at": game.get("generated_at", _now_iso()),
                "date": game["date"],
                "away": game["away"],
                "home": game["home"],
                "projected_away_pts": game["projected_away_pts"],
                "projected_home_pts": game["projected_home_pts"],
                "projected_total": game["projected_total"],
                "projected_home_spread": game["projected_home_spread"],
                "home_win_prob": game.get("home_win_prob", pd.NA),
                "win_prob_basis": game.get("win_prob_basis", ""),
                "book_total_line": book_total if book_total is not None else pd.NA,
                "book_spread_line": book_spread if book_spread is not None else pd.NA,
                "book_home_ml": book_home_ml if book_home_ml is not None else pd.NA,
                "book_away_ml": book_away_ml if book_away_ml is not None else pd.NA,
                "predicted_ml_side": _predicted_ml_side(str(game["home"]), str(game["away"]), win_prob),
                "predicted_spread_ats": _predicted_spread_ats(projected_spread, book_spread),
                "predicted_total_side": _predicted_total_side(projected_total, book_total),
                "settled": False,
                "ungraded_reason": "",
            }
        )
        added += 1
    if rows:
        frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
        _save(frame, game_log_path(root))
    return added


def backfill_logged_game_lines(root: Path, odds: pd.DataFrame | None) -> int:
    """Stamp stored book numbers onto logged forecasts that were saved without a line."""
    from .prop_projections import fill_missing_game_market_lines

    if odds is None or odds.empty:
        return 0
    frame = _load(game_log_path(root), GAME_COLUMNS)
    if frame.empty:
        return 0
    before = frame[["book_total_line", "book_spread_line", "book_home_ml", "book_away_ml"]].copy()
    filled = fill_missing_game_market_lines(frame, odds)
    changed = 0
    for idx, row in filled.iterrows():
        newly = False
        for column in ("book_total_line", "book_spread_line", "book_home_ml", "book_away_ml"):
            old = pd.to_numeric(before.at[idx, column], errors="coerce")
            new = pd.to_numeric(row.get(column), errors="coerce")
            if pd.isna(old) and pd.notna(new):
                newly = True
                break
        if not newly:
            continue
        changed += 1
        book_total = _as_float(row.get("book_total_line"))
        book_spread = _as_float(row.get("book_spread_line"))
        projected_total = _as_float(row.get("projected_total"))
        projected_spread = _as_float(row.get("projected_home_spread"))
        filled.loc[idx, "predicted_total_side"] = _predicted_total_side(projected_total, book_total)
        filled.loc[idx, "predicted_spread_ats"] = _predicted_spread_ats(projected_spread, book_spread)
        if not _audit_text(row.get("predicted_ml_side")).strip():
            filled.loc[idx, "predicted_ml_side"] = _predicted_ml_side(
                str(row["home"]), str(row["away"]), _as_float(row.get("home_win_prob"))
            )
    if changed:
        _save(filled, game_log_path(root))
    return changed


def log_market_predictions_batch(root: Path, slate: pd.DataFrame, season: str) -> int:
    """Persist moneyline, spread, and total rows, idempotent on run + game + market."""
    if slate is None or slate.empty:
        return 0
    frame = _load(market_log_path(root), MARKET_COLUMNS)
    existing = set(
        zip(
            frame["run_id"].astype(str),
            frame["game_date"].astype(str),
            frame["away"].astype(str),
            frame["home"].astype(str),
            frame["market"].astype(str),
        )
    )
    added = 0
    rows = []
    recorded_at = _now_iso()
    for _, item in slate.iterrows():
        run_id = str(item.get("run_id") or "")
        key = (
            run_id, str(item["game_date"]), str(item["away"]), str(item["home"]), str(item["market"]),
        )
        if key in existing:
            continue
        rows.append(
            {
                "prediction_id": uuid.uuid4().hex,
                "run_id": run_id,
                "sport": "wnba",
                "season": season,
                "recorded_at": item.get("generated_at") or recorded_at,
                "game_date": item["game_date"],
                "away": item["away"],
                "home": item["home"],
                "market": item["market"],
                "side": item.get("side") or "",
                "line": item.get("line"),
                "odds": item.get("odds"),
                "opposite_odds": item.get("opposite_odds"),
                "odds_source": item.get("book") or "",
                "projection": item.get("projection"),
                "projection_basis": item.get("projection_basis"),
                "model_prob": item.get("model_prob"),
                "implied_prob": item.get("implied_prob"),
                "vig_free": item.get("vig_free"),
                "edge": item.get("edge"),
                "tier": item.get("tier") or "",
                "verdict": item.get("verdict") or "",
                "priced": item.get("priced"),
                "settled": False,
                "ungraded_reason": "",
            }
        )
        added += 1
    if rows:
        frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
        _save(frame, market_log_path(root))
    return added


def grade_props(root: Path, player_logs: pd.DataFrame, today: datetime | None = None) -> dict:
    """Grade unsettled prop predictions against player game logs."""
    frame = _load(prop_log_path(root), PROP_COLUMNS)
    if frame.empty:
        return {"graded": 0, "voided": 0, "pending": 0}
    today = today or datetime.now(timezone.utc)
    logs = player_logs.copy()
    logs["date"] = pd.to_datetime(logs["date"])
    logs["_name"] = logs["name"].astype(str).str.lower()

    graded = voided = pending = 0
    for idx, row in frame.iterrows():
        if str(row.get("settled")).lower() == "true" or row.get("settled") is True:
            continue
        stat_column = MARKET_STAT_COLUMNS.get(str(row["market"]))
        if stat_column is None or stat_column not in logs.columns:
            frame.loc[idx, ["settled", "ungraded_reason", "graded_at"]] = [
                True, REASON_MARKET_UNSUPPORTED, _now_iso(),
            ]
            voided += 1
            continue
        game_date = pd.to_datetime(str(row["game_date"]))
        window_start = game_date - timedelta(days=1)
        window_end = game_date + timedelta(days=2)
        candidates = logs[
            (logs["_name"] == str(row["player"]).lower())
            & (logs["date"] >= window_start)
            & (logs["date"] <= window_end)
        ].sort_values("date")
        if candidates.empty:
            if today.replace(tzinfo=None) > (window_end + timedelta(days=PROP_VOID_AFTER_DAYS)):
                reason = (
                    REASON_VOID_NO_GAME
                    if (logs["_name"] == str(row["player"]).lower()).any()
                    else REASON_PLAYER_NOT_FOUND
                )
                frame.loc[idx, ["settled", "ungraded_reason", "graded_at"]] = [
                    True, reason, _now_iso(),
                ]
                voided += 1
            else:
                frame.loc[idx, "ungraded_reason"] = REASON_NO_GAME_IN_WINDOW
                pending += 1
            continue
        actual = pd.to_numeric(candidates.iloc[0][stat_column], errors="coerce")
        if pd.isna(actual):
            frame.loc[idx, "ungraded_reason"] = REASON_RESULT_PENDING
            pending += 1
            continue
        line = pd.to_numeric(row["line"], errors="coerce")
        if pd.isna(line):
            # Model-only projection: keep the actual for MAE, never invent a W-L against a
            # line that was not captured with the forecast.
            frame.loc[idx, ["settled", "won", "push", "actual", "ungraded_reason", "graded_at"]] = [
                True, "", False, float(actual), "", _now_iso(),
            ]
            graded += 1
            continue
        push = bool(actual == line)
        over = bool(actual > line)
        want_over = str(row["side"]).lower() == "over"
        won = "" if push else str(over == want_over)
        frame.loc[idx, ["settled", "won", "push", "actual", "ungraded_reason", "graded_at"]] = [
            True, won, push, float(actual), "", _now_iso(),
        ]
        graded += 1
    _save(frame, prop_log_path(root))
    return {"graded": graded, "voided": voided, "pending": pending}


def grade_games(root: Path, game_results: pd.DataFrame, today: datetime | None = None) -> dict:
    frame = _load(game_log_path(root), GAME_COLUMNS)
    if frame.empty:
        return {"graded": 0, "voided": 0, "pending": 0}
    today = today or datetime.now(timezone.utc)
    results = game_results.copy()
    results["date"] = pd.to_datetime(results["date"]).dt.date.astype(str)
    by_key = {
        (row["date"], str(row["away"]).upper(), str(row["home"]).upper()): row
        for _, row in results.iterrows()
    }
    graded = voided = pending = 0
    for idx, row in frame.iterrows():
        key = (str(row["date"]), str(row["away"]).upper(), str(row["home"]).upper())
        outcome = by_key.get(key)
        is_settled = str(row.get("settled")).lower() == "true" or row.get("settled") is True
        reopen_void = (
            is_settled
            and _audit_text(row.get("ungraded_reason")) == REASON_VOID_NO_RESULT
            and outcome is not None
        )
        if is_settled and not reopen_void:
            # Schema upgrades must enrich historic audit rows too.  Existing settled rows
            # already have the final score, so a spread-direction grade can be backfilled
            # without changing their original prediction or outcome.
            if not _audit_text(row.get("spread_side_correct")).strip():
                _backfill_spread_grade(frame, idx, row)
            if not _audit_text(row.get("total_side_correct")).strip():
                _backfill_total_grade(frame, idx, row)
            if not _audit_text(row.get("spread_ats_correct")).strip():
                _backfill_spread_ats_grade(frame, idx, row)
            if not _audit_text(row.get("predicted_ml_side")).strip():
                win_prob = _as_float(row.get("home_win_prob"))
                frame.loc[idx, "predicted_ml_side"] = _predicted_ml_side(
                    str(row["home"]), str(row["away"]), win_prob
                )
            continue
        if outcome is None:
            game_day = pd.to_datetime(str(row["date"]))
            if today.replace(tzinfo=None) > game_day + timedelta(days=GAME_VOID_AFTER_DAYS):
                frame.loc[idx, ["settled", "ungraded_reason", "graded_at"]] = [
                    True, REASON_VOID_NO_RESULT, _now_iso(),
                ]
                voided += 1
            else:
                frame.loc[idx, "ungraded_reason"] = REASON_RESULT_PENDING
                pending += 1
            continue
        away_pts = float(outcome["awayPts"])
        home_pts = float(outcome["homePts"])
        actual_total = home_pts + away_pts
        home_win = home_pts > away_pts
        win_prob = pd.to_numeric(row["home_win_prob"], errors="coerce")
        winner_correct = ""
        if pd.notna(win_prob):
            winner_correct = str((win_prob >= 0.5) == home_win)
        book_total = _as_float(row.get("book_total_line"))
        book_spread = _as_float(row.get("book_spread_line"))
        predicted_total_side = _audit_text(row.get("predicted_total_side")) or _predicted_total_side(
            _as_float(row.get("projected_total")), book_total
        )
        total_side_correct = _total_side_correct(predicted_total_side, book_total, actual_total)
        predicted_ml = _audit_text(row.get("predicted_ml_side")) or _predicted_ml_side(
            str(row["home"]), str(row["away"]), _as_float(row.get("home_win_prob"))
        )
        predicted_spread_ats = _audit_text(row.get("predicted_spread_ats")) or _predicted_spread_ats(
            _as_float(row.get("projected_home_spread")), book_spread
        )
        spread_ats_correct = _spread_ats_correct(predicted_spread_ats, book_spread, home_pts - away_pts)
        frame.loc[
            idx,
            [
                "settled", "actual_away_pts", "actual_home_pts", "actual_total", "actual_winner",
                "home_win", "winner_correct", "predicted_ml_side",
                "predicted_spread_side", "spread_side_correct",
                "predicted_spread_ats", "spread_ats_correct",
                "predicted_total_side", "total_side_correct",
                "spread_error", "total_error",
                "ungraded_reason", "graded_at",
            ],
        ] = [
            True, away_pts, home_pts, actual_total, str(outcome["winner"]),
            home_win, winner_correct, predicted_ml,
            *_spread_grade(float(row["projected_home_spread"]), home_pts - away_pts),
            predicted_spread_ats, spread_ats_correct,
            predicted_total_side, total_side_correct,
            round(float(row["projected_home_spread"]) - (home_pts - away_pts), 1),
            round(float(row["projected_total"]) - actual_total, 1),
            "", _now_iso(),
        ]
        graded += 1
    _save(frame, game_log_path(root))
    return {"graded": graded, "voided": voided, "pending": pending}


def grade_markets(root: Path, game_results: pd.DataFrame, today: datetime | None = None) -> dict:
    """Grade recorded moneyline, spread, and total rows against finished scores."""
    frame = _load(market_log_path(root), MARKET_COLUMNS)
    if frame.empty:
        return {"graded": 0, "voided": 0, "pending": 0}
    today = today or datetime.now(timezone.utc)
    results = game_results.copy()
    results["date"] = pd.to_datetime(results["date"]).dt.date.astype(str)
    by_key = {
        (row["date"], str(row["away"]).upper(), str(row["home"]).upper()): row
        for _, row in results.iterrows()
    }
    graded = voided = pending = 0
    for idx, row in frame.iterrows():
        key = (str(row["game_date"]), str(row["away"]).upper(), str(row["home"]).upper())
        outcome = by_key.get(key)
        is_settled = str(row.get("settled")).lower() == "true" or row.get("settled") is True
        reopen_void = (
            is_settled
            and _audit_text(row.get("ungraded_reason")) == REASON_VOID_NO_RESULT
            and outcome is not None
        )
        if is_settled and not reopen_void:
            continue
        if outcome is None:
            game_day = pd.to_datetime(str(row["game_date"]))
            if today.replace(tzinfo=None) > game_day + timedelta(days=GAME_VOID_AFTER_DAYS):
                frame.loc[idx, ["settled", "ungraded_reason", "graded_at"]] = [
                    True, REASON_VOID_NO_RESULT, _now_iso(),
                ]
                voided += 1
            else:
                frame.loc[idx, "ungraded_reason"] = REASON_RESULT_PENDING
                pending += 1
            continue
        away_pts = float(outcome["awayPts"])
        home_pts = float(outcome["homePts"])
        market = str(row["market"]).lower()
        side = str(row.get("side") or "").upper()
        if market == "moneyline":
            winner = str(outcome["winner"]).upper()
            won = "" if not side else str(side == winner)
            actual = 1.0 if winner == str(row["home"]).upper() else 0.0
            push = False
        elif market == "spread":
            line = _as_float(row.get("line"))
            actual = home_pts - away_pts
            if line is None or not side:
                frame.loc[idx, ["settled", "won", "push", "actual", "ungraded_reason", "graded_at"]] = [
                    True, "", False, actual, "", _now_iso(),
                ]
                graded += 1
                continue
            cover = actual + line
            if cover == 0:
                won, push = "", True
            else:
                covered = str(row["home"]).upper() if cover > 0 else str(row["away"]).upper()
                won, push = str(side == covered), False
        elif market == "total":
            line = _as_float(row.get("line"))
            actual = home_pts + away_pts
            if line is None or not side:
                frame.loc[idx, ["settled", "won", "push", "actual", "ungraded_reason", "graded_at"]] = [
                    True, "", False, actual, "", _now_iso(),
                ]
                graded += 1
                continue
            if actual == line:
                won, push = "", True
            else:
                actual_side = "OVER" if actual > line else "UNDER"
                won, push = str(side == actual_side), False
        else:
            frame.loc[idx, ["settled", "ungraded_reason", "graded_at"]] = [
                True, REASON_MARKET_UNSUPPORTED, _now_iso(),
            ]
            voided += 1
            continue
        frame.loc[idx, ["settled", "won", "push", "actual", "ungraded_reason", "graded_at"]] = [
            True, won, push, actual, "", _now_iso(),
        ]
        graded += 1
    _save(frame, market_log_path(root))
    return {"graded": graded, "voided": voided, "pending": pending}


def _spread_grade(projected_home_spread: float, actual_home_margin: float) -> tuple[str, str]:
    """Grade the model's spread *side* from its projected home margin.

    This is direction accuracy (the model's predicted winner/margin side), not an ATS wager
    result: no sportsbook line is assumed unless one was captured with the prediction.
    """
    if projected_home_spread == 0 or actual_home_margin == 0:
        return "PUSH", "PUSH"
    side = "HOME" if projected_home_spread > 0 else "AWAY"
    correct = (projected_home_spread > 0) == (actual_home_margin > 0)
    return side, str(correct)


def _backfill_spread_grade(frame: pd.DataFrame, idx, row: pd.Series) -> None:
    projected = _as_float(row.get("projected_home_spread"))
    home = _as_float(row.get("actual_home_pts"))
    away = _as_float(row.get("actual_away_pts"))
    if projected is None or home is None or away is None:
        return
    side, correct = _spread_grade(projected, home - away)
    frame.loc[idx, ["predicted_spread_side", "spread_side_correct"]] = [side, correct]


def _predicted_ml_side(home: str, away: str, win_prob: float | None) -> str:
    if win_prob is None:
        return ""
    return str(home).upper() if win_prob >= 0.5 else str(away).upper()


def _predicted_spread_ats(projected_home_spread: float | None, book_line: float | None) -> str:
    """HOME/AWAY/PUSH from the model's margin against the captured home spread."""
    if projected_home_spread is None or book_line is None:
        return ""
    cover = projected_home_spread + book_line
    if cover == 0:
        return "PUSH"
    return "HOME" if cover > 0 else "AWAY"


def _spread_ats_correct(predicted_side: str, book_line: float | None, actual_home_margin: float) -> str:
    if not predicted_side or book_line is None:
        return ""
    cover = actual_home_margin + book_line
    if cover == 0:
        return "PUSH"
    actual_side = "HOME" if cover > 0 else "AWAY"
    if predicted_side == "PUSH":
        return "PUSH" if actual_side == "PUSH" else "False"
    return str(predicted_side == actual_side)


def _backfill_spread_ats_grade(frame: pd.DataFrame, idx, row: pd.Series) -> None:
    book_line = _as_float(row.get("book_spread_line"))
    projected = _as_float(row.get("projected_home_spread"))
    home = _as_float(row.get("actual_home_pts"))
    away = _as_float(row.get("actual_away_pts"))
    if book_line is None or projected is None or home is None or away is None:
        return
    predicted = _audit_text(row.get("predicted_spread_ats")) or _predicted_spread_ats(projected, book_line)
    frame.loc[idx, ["predicted_spread_ats", "spread_ats_correct"]] = [
        predicted, _spread_ats_correct(predicted, book_line, home - away),
    ]


def _predicted_total_side(projected_total: float | None, book_line: float | None) -> str:
    """OVER / UNDER / PUSH from the model's total against the captured book number."""
    if projected_total is None or book_line is None:
        return ""
    if projected_total == book_line:
        return "PUSH"
    return "OVER" if projected_total > book_line else "UNDER"


def _total_side_correct(predicted_side: str, book_line: float | None, actual_total: float) -> str:
    """ATS-style over/under grade. Empty when no book line was captured with the forecast."""
    if not predicted_side or book_line is None:
        return ""
    if actual_total == book_line:
        return "PUSH"
    actual_side = "OVER" if actual_total > book_line else "UNDER"
    if predicted_side == "PUSH":
        return "PUSH" if actual_side == "PUSH" else "False"
    return str(predicted_side == actual_side)


def _backfill_total_grade(frame: pd.DataFrame, idx, row: pd.Series) -> None:
    book_line = _as_float(row.get("book_total_line"))
    projected = _as_float(row.get("projected_total"))
    home = _as_float(row.get("actual_home_pts"))
    away = _as_float(row.get("actual_away_pts"))
    if book_line is None or projected is None or home is None or away is None:
        return
    actual_total = home + away
    predicted = _audit_text(row.get("predicted_total_side")) or _predicted_total_side(projected, book_line)
    frame.loc[idx, ["actual_total", "predicted_total_side", "total_side_correct"]] = [
        actual_total, predicted, _total_side_correct(predicted, book_line, actual_total),
    ]


def results_summary(root: Path) -> dict:
    """Aggregate graded prediction performance for the CLI and dashboard."""
    props = _load(prop_log_path(root), PROP_COLUMNS)
    games = _load(game_log_path(root), GAME_COLUMNS)
    summary: dict = {"props": {}, "games": {}, "markets": {}, "generated_at": _now_iso()}

    settled = props[props["settled"].astype(str).str.lower() == "true"]
    reason = settled["ungraded_reason"].fillna("").astype(str).str.strip()
    tracked = settled[reason.isin(["", "nan", "<NA>"])]
    for market, group in tracked.groupby("market"):
        wins = (group["won"].astype(str) == "True").sum()
        losses = (group["won"].astype(str) == "False").sum()
        pushes = (group["push"].astype(str).str.lower() == "true").sum()
        actuals = pd.to_numeric(group["actual"], errors="coerce")
        projections = pd.to_numeric(group["projection"], errors="coerce")
        abs_err = (actuals - projections).abs()
        mae = round(float(abs_err.mean()), 2) if abs_err.notna().any() else None
        summary["props"][str(market)] = {
            "wins": int(wins),
            "losses": int(losses),
            "pushes": int(pushes),
            "hit_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) else None,
            "mae": mae,
            "n_mae": int(abs_err.notna().sum()),
        }
    summary["props"]["_pending"] = int((props["settled"].astype(str).str.lower() != "true").sum())
    reasons = props[props["ungraded_reason"].astype(str).str.len() > 0]
    summary["props"]["_reasons"] = reasons["ungraded_reason"].value_counts().to_dict()
    summary["props"]["_records"] = _prop_audit_records(props)

    markets = _load(market_log_path(root), MARKET_COLUMNS)
    settled_markets = markets[markets["settled"].astype(str).str.lower() == "true"]
    market_reason = settled_markets["ungraded_reason"].fillna("").astype(str).str.strip()
    tracked_markets = settled_markets[market_reason.isin(["", "nan", "<NA>"])]
    for market, group in tracked_markets.groupby("market"):
        wins = (group["won"].astype(str) == "True").sum()
        losses = (group["won"].astype(str) == "False").sum()
        pushes = (group["push"].astype(str).str.lower() == "true").sum()
        summary["markets"][str(market)] = {
            "wins": int(wins),
            "losses": int(losses),
            "pushes": int(pushes),
            "hit_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) else None,
            "pending": 0,
        }
    summary["markets"]["_pending"] = int((markets["settled"].astype(str).str.lower() != "true").sum())
    summary["markets"]["_logged"] = int(len(markets))
    summary["markets"]["_records"] = _market_audit_records(markets)

    graded_games = games[games["settled"].astype(str).str.lower() == "true"]
    scored = graded_games[graded_games["winner_correct"].astype(str).isin(["True", "False"])].copy()
    if not scored.empty:
        # Preserve every run in the CSV, but score the production record from the most
        # recent forecast available for each actual game.  A game refreshed three times
        # should produce three audit rows, not triple the public W-L record.
        scored["_recorded_sort"] = pd.to_datetime(scored["recorded_at"], errors="coerce", utc=True)
        latest = (
            scored.sort_values(["date", "away", "home", "_recorded_sort"])
            .drop_duplicates(["date", "away", "home"], keep="last")
            .copy()
        )
        latest = latest.sort_values(["date", "_recorded_sort"], ascending=[False, False])
        correct = (latest["winner_correct"].astype(str) == "True").sum()
        win_probs = pd.to_numeric(latest["home_win_prob"], errors="coerce")
        actual = latest["home_win"].astype(str).str.lower() == "true"
        brier = float(((win_probs - actual.astype(float)) ** 2).mean())

        bands = []
        confidence = pd.Series(
            [max(prob, 1 - prob) if pd.notna(prob) else float("nan") for prob in win_probs],
            index=latest.index,
        )
        for label, low, high in (("50–59%", 0.50, 0.60), ("60–69%", 0.60, 0.70), ("70%+", 0.70, 1.01)):
            band = latest[(confidence >= low) & (confidence < high)]
            if band.empty:
                continue
            hits = int((band["winner_correct"].astype(str) == "True").sum())
            bands.append({
                "band": label,
                "n": int(len(band)),
                "correct": hits,
                "hit_rate": round(hits / len(band) * 100, 1),
            })

        recent = []
        for _, row in latest.iterrows():
            probability = _as_float(row.get("home_win_prob"))
            home = str(row["home"])
            away = str(row["away"])
            favorite = home if probability >= 0.5 else away
            favorite_probability = max(probability, 1 - probability) if probability is not None else None
            recent.append({
                "date": str(row["date"]),
                "matchup": f"{away} @ {home}",
                "favorite": favorite,
                "probability": round(favorite_probability * 100, 1) if favorite_probability is not None else None,
                "actual_winner": str(row.get("actual_winner") or "—"),
                "correct": str(row["winner_correct"]).lower() == "true",
                "spread_error": _as_float(row.get("spread_error")),
                "total_error": _as_float(row.get("total_error")),
                "total_side": _audit_text(row.get("predicted_total_side")) or "—",
                "total_status": _audit_text(row.get("total_side_correct")),
                "ml_side": _audit_text(row.get("predicted_ml_side")) or "—",
                "spread_ats": _audit_text(row.get("predicted_spread_ats")) or "—",
                "spread_ats_status": _audit_text(row.get("spread_ats_correct")),
                "spread_side": _audit_text(row.get("predicted_spread_side")) or "—",
                "spread_status": _audit_text(row.get("spread_side_correct")),
                "book_spread_line": _as_float(row.get("book_spread_line")),
                "book_total_line": _as_float(row.get("book_total_line")),
                "book_home_ml": _as_float(row.get("book_home_ml")),
                "book_away_ml": _as_float(row.get("book_away_ml")),
                "projected_home_spread": _as_float(row.get("projected_home_spread")),
                "projected_total": _as_float(row.get("projected_total")),
                "run_id": str(row.get("run_id") or ""),
            })
        summary["games"] = {
            "n": int(len(latest)),
            "correct": int(correct),
            "winner_hit_rate": round(correct / len(latest) * 100, 1),
            "spread_mae": round(pd.to_numeric(latest["spread_error"], errors="coerce").abs().mean(), 2),
            "total_mae": round(pd.to_numeric(latest["total_error"], errors="coerce").abs().mean(), 2),
            "brier": round(brier, 4),
            "audit_runs": int(len(scored)),
            "confidence_bands": bands,
            "recent": recent,
        }
        total_side = latest[latest["total_side_correct"].astype(str).isin(["True", "False"])]
        if not total_side.empty:
            total_hits = int((total_side["total_side_correct"].astype(str) == "True").sum())
            summary["games"]["total_side_n"] = int(len(total_side))
            summary["games"]["total_side_correct"] = total_hits
            summary["games"]["total_side_hit_rate"] = round(total_hits / len(total_side) * 100, 1)
        spread_ats = latest[latest["spread_ats_correct"].astype(str).isin(["True", "False"])]
        if not spread_ats.empty:
            ats_hits = int((spread_ats["spread_ats_correct"].astype(str) == "True").sum())
            summary["games"]["spread_ats_n"] = int(len(spread_ats))
            summary["games"]["spread_ats_correct"] = ats_hits
            summary["games"]["spread_ats_hit_rate"] = round(ats_hits / len(spread_ats) * 100, 1)
        spread_dir = latest[latest["spread_side_correct"].astype(str).isin(["True", "False"])]
        if not spread_dir.empty:
            spread_hits = int((spread_dir["spread_side_correct"].astype(str) == "True").sum())
            summary["games"]["spread_n"] = int(len(spread_dir))
            summary["games"]["spread_correct"] = spread_hits
            summary["games"]["spread_hit_rate"] = round(spread_hits / len(spread_dir) * 100, 1)
        for market, flags in (
            ("moneyline", latest["winner_correct"]),
            ("spread", latest["spread_ats_correct"]),
            ("total", latest["total_side_correct"]),
        ):
            record = _wl_from_flags(flags)
            if record is None:
                continue
            existing = summary["markets"].get(market)
            existing_n = 0
            if isinstance(existing, dict):
                existing_n = int(existing.get("wins", 0)) + int(existing.get("losses", 0)) + int(existing.get("pushes", 0))
            if existing_n == 0:
                summary["markets"][market] = record
    summary["games"]["_pending"] = int((games["settled"].astype(str).str.lower() != "true").sum())
    summary["games"]["_logged"] = int(len(games))
    summary["games"]["_records"] = _game_audit_records(games)
    return summary


def _as_float(value) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else None


def _wl_from_flags(series: pd.Series) -> dict | None:
    flags = series.astype(str)
    tracked = flags.isin(["True", "False"])
    if not tracked.any():
        return None
    wins = int((flags[tracked] == "True").sum())
    losses = int((flags[tracked] == "False").sum())
    pushes = int((flags == "PUSH").sum())
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) else None,
    }


def _audit_text(value) -> str:
    return "" if pd.isna(value) else str(value)


def _audit_status(row: pd.Series, correctness_column: str) -> tuple[str, str]:
    settled = _audit_text(row.get("settled")).lower() == "true"
    reason = _audit_text(row.get("ungraded_reason"))
    if not settled:
        return ("Pending", reason or "awaiting result")
    if reason:
        return ("Voided", reason)
    if _audit_text(row.get("push")).lower() == "true":
        return ("Push", "")
    correct = _audit_text(row.get(correctness_column)).lower()
    if correct == "true":
        return ("Correct", "")
    if correct == "false":
        return ("Miss", "")
    return ("Recorded", "projection only")


def _prop_audit_records(props: pd.DataFrame) -> list[dict]:
    records = []
    ordered = props.copy()
    ordered["_sort"] = pd.to_datetime(ordered["recorded_at"], errors="coerce", utc=True)
    for _, row in ordered.sort_values("_sort", ascending=False).iterrows():
        status, detail = _audit_status(row, "won")
        records.append(
            {
                "recorded_at": _audit_text(row.get("recorded_at")),
                "game_date": _audit_text(row.get("game_date")),
                "player": _audit_text(row.get("player")),
                "market": _audit_text(row.get("market")),
                "side": _audit_text(row.get("side")),
                "line": _as_float(row.get("line")),
                "odds": _as_float(row.get("odds")),
                "projection": _as_float(row.get("projection")),
                "model_prob": _as_float(row.get("model_prob")),
                "edge": _as_float(row.get("edge")),
                "actual": _as_float(row.get("actual")),
                "status": status,
                "status_detail": detail,
                "prediction_id": _audit_text(row.get("prediction_id")),
            }
        )
    return records


def _market_audit_records(markets: pd.DataFrame) -> list[dict]:
    records = []
    ordered = markets.copy()
    if ordered.empty:
        return records
    ordered["_sort"] = pd.to_datetime(ordered["recorded_at"], errors="coerce", utc=True)
    for _, row in ordered.sort_values("_sort", ascending=False).iterrows():
        status, detail = _audit_status(row, "won")
        records.append(
            {
                "recorded_at": _audit_text(row.get("recorded_at")),
                "game_date": _audit_text(row.get("game_date")),
                "matchup": f"{_audit_text(row.get('away'))} @ {_audit_text(row.get('home'))}",
                "market": _audit_text(row.get("market")),
                "side": _audit_text(row.get("side")),
                "line": _as_float(row.get("line")),
                "odds": _as_float(row.get("odds")),
                "projection": _as_float(row.get("projection")),
                "model_prob": _as_float(row.get("model_prob")),
                "edge": _as_float(row.get("edge")),
                "actual": _as_float(row.get("actual")),
                "status": status,
                "status_detail": detail,
                "prediction_id": _audit_text(row.get("prediction_id")),
            }
        )
    return records


def _game_audit_records(games: pd.DataFrame) -> list[dict]:
    records = []
    ordered = games.copy()
    ordered["_sort"] = pd.to_datetime(ordered["recorded_at"], errors="coerce", utc=True)
    for _, row in ordered.sort_values(["date", "_sort"], ascending=[False, False]).iterrows():
        status, detail = _audit_status(row, "winner_correct")
        home_probability = _as_float(row.get("home_win_prob"))
        home = _audit_text(row.get("home"))
        away = _audit_text(row.get("away"))
        favorite = home if home_probability is not None and home_probability >= 0.5 else away
        favorite_probability = (
            max(home_probability, 1 - home_probability) if home_probability is not None else None
        )
        records.append(
            {
                "recorded_at": _audit_text(row.get("recorded_at")),
                "date": _audit_text(row.get("date")),
                "matchup": f"{away} @ {home}",
                "projection": (
                    f"{_as_float(row.get('projected_away_pts')):.1f}–"
                    f"{_as_float(row.get('projected_home_pts')):.1f}"
                    if _as_float(row.get("projected_away_pts")) is not None
                    and _as_float(row.get("projected_home_pts")) is not None
                    else "—"
                ),
                "favorite": favorite,
                "favorite_probability": favorite_probability,
                "actual_winner": _audit_text(row.get("actual_winner")) or "—",
                "spread_error": _as_float(row.get("spread_error")),
                "total_error": _as_float(row.get("total_error")),
                "spread_side": _audit_text(row.get("predicted_spread_side")) or "—",
                "spread_status": _audit_text(row.get("spread_side_correct")),
                "ml_side": _audit_text(row.get("predicted_ml_side")) or "—",
                "spread_ats": _audit_text(row.get("predicted_spread_ats")) or "—",
                "spread_ats_status": _audit_text(row.get("spread_ats_correct")),
                "total_side": _audit_text(row.get("predicted_total_side")) or "—",
                "total_status": _audit_text(row.get("total_side_correct")),
                "book_total_line": _as_float(row.get("book_total_line")),
                "status": status,
                "status_detail": detail,
                "run_id": _audit_text(row.get("run_id")),
                "prediction_id": _audit_text(row.get("prediction_id")),
            }
        )
    return records
