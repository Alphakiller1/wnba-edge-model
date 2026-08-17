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
    "settled", "actual_away_pts", "actual_home_pts", "actual_winner",
    "home_win", "winner_correct", "predicted_spread_side", "spread_side_correct",
    "spread_error", "total_error",
    "ungraded_reason", "graded_at",
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
                "settled": False,
                "ungraded_reason": "",
            }
        )
        added += 1
    if rows:
        frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
        _save(frame, game_log_path(root))
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
        line = pd.to_numeric(row["line"], errors="coerce")
        if pd.isna(actual) or pd.isna(line):
            frame.loc[idx, "ungraded_reason"] = REASON_RESULT_PENDING
            pending += 1
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
        is_settled = str(row.get("settled")).lower() == "true" or row.get("settled") is True
        if is_settled:
            # Schema upgrades must enrich historic audit rows too.  Existing settled rows
            # already have the final score, so a spread-direction grade can be backfilled
            # without changing their original prediction or outcome.
            if not _audit_text(row.get("spread_side_correct")).strip():
                _backfill_spread_grade(frame, idx, row)
            continue
        key = (str(row["date"]), str(row["away"]).upper(), str(row["home"]).upper())
        outcome = by_key.get(key)
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
        home_win = home_pts > away_pts
        win_prob = pd.to_numeric(row["home_win_prob"], errors="coerce")
        winner_correct = ""
        if pd.notna(win_prob):
            winner_correct = str((win_prob >= 0.5) == home_win)
        frame.loc[
            idx,
            [
                "settled", "actual_away_pts", "actual_home_pts", "actual_winner",
                "home_win", "winner_correct", "predicted_spread_side", "spread_side_correct",
                "spread_error", "total_error",
                "ungraded_reason", "graded_at",
            ],
        ] = [
            True, away_pts, home_pts, str(outcome["winner"]),
            home_win, winner_correct, *_spread_grade(float(row["projected_home_spread"]), home_pts - away_pts),
            round(float(row["projected_home_spread"]) - (home_pts - away_pts), 1),
            round(float(row["projected_total"]) - (home_pts + away_pts), 1),
            "", _now_iso(),
        ]
        graded += 1
    _save(frame, game_log_path(root))
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


def results_summary(root: Path) -> dict:
    """Aggregate graded prediction performance for the CLI and dashboard."""
    props = _load(prop_log_path(root), PROP_COLUMNS)
    games = _load(game_log_path(root), GAME_COLUMNS)
    summary: dict = {"props": {}, "games": {}, "generated_at": _now_iso()}

    settled = props[props["settled"].astype(str).str.lower() == "true"]
    graded = settled[settled["won"].astype(str).isin(["True", "False"])]
    for market, group in graded.groupby("market"):
        wins = (group["won"].astype(str) == "True").sum()
        losses = (group["won"].astype(str) == "False").sum()
        pushes = (settled[settled["market"] == market]["push"].astype(str).str.lower() == "true").sum()
        summary["props"][str(market)] = {
            "wins": int(wins),
            "losses": int(losses),
            "pushes": int(pushes),
            "hit_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) else None,
        }
    summary["props"]["_pending"] = int((props["settled"].astype(str).str.lower() != "true").sum())
    reasons = props[props["ungraded_reason"].astype(str).str.len() > 0]
    summary["props"]["_reasons"] = reasons["ungraded_reason"].value_counts().to_dict()
    summary["props"]["_records"] = _prop_audit_records(props)

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
        for _, row in latest.head(12).iterrows():
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
    settled_spread = graded_games[graded_games["spread_side_correct"].astype(str).isin(["True", "False"])]
    if not settled_spread.empty:
        spread_hits = int((settled_spread["spread_side_correct"].astype(str) == "True").sum())
        summary["games"]["spread_n"] = int(len(settled_spread))
        summary["games"]["spread_correct"] = spread_hits
        summary["games"]["spread_hit_rate"] = round(spread_hits / len(settled_spread) * 100, 1)
    summary["games"]["_pending"] = int((games["settled"].astype(str).str.lower() != "true").sum())
    summary["games"]["_logged"] = int(len(games))
    summary["games"]["_records"] = _game_audit_records(games)
    return summary


def _as_float(value) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else None


def _audit_text(value) -> str:
    return "" if pd.isna(value) else str(value)


def _audit_status(row: pd.Series, correctness_column: str) -> tuple[str, str]:
    settled = _audit_text(row.get("settled")).lower() == "true"
    reason = _audit_text(row.get("ungraded_reason"))
    if not settled:
        return ("Pending", reason or "awaiting result")
    if reason:
        return ("Voided", reason)
    correct = _audit_text(row.get(correctness_column)).lower() == "true"
    return ("Correct" if correct else "Miss", "")


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
                "status": status,
                "status_detail": detail,
                "run_id": _audit_text(row.get("run_id")),
                "prediction_id": _audit_text(row.get("prediction_id")),
            }
        )
    return records
