from datetime import datetime, timezone

import pandas as pd

from wnba_edges.predictions import (
    REASON_MARKET_UNSUPPORTED,
    REASON_PLAYER_NOT_FOUND,
    grade_games,
    grade_props,
    log_game_projections,
    log_prop_prediction,
    game_log_path,
    prop_log_path,
    results_summary,
)


def _log_prop(root, **overrides):
    row = {
        "season": "2026-27",
        "game_date": "2026-07-10",
        "player": "A'ja Wilson",
        "player_id": 1,
        "market": "player_points",
        "side": "over",
        "line": 24.5,
        "odds": -110,
        "model_prob": 0.55,
        "tier": "Standard",
        "verdict": "PLAY",
    }
    row.update(overrides)
    return log_prop_prediction(root, row)


def _player_logs():
    return pd.DataFrame(
        [
            {"playerId": 1, "name": "A'ja Wilson", "date": "2026-07-10", "pts": 30, "reb": 9, "ast": 3},
            {"playerId": 1, "name": "A'ja Wilson", "date": "2026-07-12", "pts": 18, "reb": 11, "ast": 2},
        ]
    )


def test_prop_grading_win_push_and_reasons(tmp_path):
    _log_prop(tmp_path)  # over 24.5, actual 30 -> win
    _log_prop(tmp_path, line=30.0)  # push
    _log_prop(tmp_path, side="under")  # under 24.5, actual 30 -> loss
    _log_prop(tmp_path, player="Nobody Real", game_date="2026-07-01")  # -> void, player not found
    _log_prop(tmp_path, market="player_weird_market")  # -> unsupported

    today = datetime(2026, 7, 20, tzinfo=timezone.utc)
    outcome = grade_props(tmp_path, _player_logs(), today=today)
    assert outcome["graded"] == 3
    assert outcome["voided"] == 2

    frame = pd.read_csv(prop_log_path(tmp_path))
    assert (frame["settled"].astype(str).str.lower() == "true").all()
    reasons = set(frame["ungraded_reason"].dropna().astype(str)) - {""}
    assert REASON_PLAYER_NOT_FOUND in reasons
    assert REASON_MARKET_UNSUPPORTED in reasons
    wins = (frame["won"].astype(str) == "True").sum()
    losses = (frame["won"].astype(str) == "False").sum()
    pushes = (frame["push"].astype(str).str.lower() == "true").sum()
    assert (wins, losses, pushes) == (1, 1, 1)


def test_prop_not_voided_before_window_expires(tmp_path):
    _log_prop(tmp_path, game_date="2026-07-19")
    today = datetime(2026, 7, 20, tzinfo=timezone.utc)
    outcome = grade_props(tmp_path, _player_logs(), today=today)
    assert outcome["pending"] == 1
    assert outcome["voided"] == 0


def test_game_grading_and_summary(tmp_path):
    projections = pd.DataFrame(
        [
            {
                "run_id": "r1", "generated_at": "2026-07-09T12:00:00+00:00",
                "date": "2026-07-10", "away": "CHI", "home": "MIN",
                "projected_away_pts": 78.0, "projected_home_pts": 88.0,
                "projected_total": 166.0, "projected_home_spread": 10.0,
                "home_win_prob": 0.75, "win_prob_basis": "logistic fit on 60 games",
            }
        ]
    )
    assert log_game_projections(tmp_path, projections, "2026-27") == 1
    # Idempotent: logging the same slate again adds nothing.
    assert log_game_projections(tmp_path, projections, "2026-27") == 0

    results = pd.DataFrame(
        [{"date": "2026-07-10", "away": "CHI", "home": "MIN", "awayPts": 80, "homePts": 90, "winner": "MIN"}]
    )
    outcome = grade_games(tmp_path, results)
    assert outcome["graded"] == 1

    summary = results_summary(tmp_path)
    assert summary["games"]["n"] == 1
    assert summary["games"]["winner_hit_rate"] == 100.0
    assert summary["games"]["spread_mae"] == 0.0
    assert summary["games"]["brier"] == round((0.75 - 1.0) ** 2, 4)
    assert summary["games"]["correct"] == 1
    assert summary["games"]["recent"][0]["correct"] is True


def test_game_log_keeps_each_unique_projection_run(tmp_path):
    projection = pd.DataFrame(
        [
            {
                "run_id": "morning", "generated_at": "2026-07-09T12:00:00+00:00",
                "date": "2026-07-10", "away": "CHI", "home": "MIN",
                "projected_away_pts": 78.0, "projected_home_pts": 88.0,
                "projected_total": 166.0, "projected_home_spread": 10.0,
                "home_win_prob": 0.75, "win_prob_basis": "test",
            }
        ]
    )
    assert log_game_projections(tmp_path, projection, "2026-27") == 1
    assert log_game_projections(tmp_path, projection, "2026-27") == 0
    afternoon = projection.assign(run_id="afternoon", generated_at="2026-07-09T18:00:00+00:00")
    assert log_game_projections(tmp_path, afternoon, "2026-27") == 1
    assert len(pd.read_csv(game_log_path(tmp_path))) == 2
