from datetime import datetime, timezone

import pandas as pd

from wnba_edges.predictions import (
    REASON_MARKET_UNSUPPORTED,
    REASON_PLAYER_NOT_FOUND,
    game_log_path,
    grade_games,
    grade_props,
    log_game_projections,
    log_prop_prediction,
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
    assert summary["games"]["_records"][0]["status"] == "Correct"
    assert summary["games"]["_records"][0]["matchup"] == "CHI @ MIN"
    assert summary["games"]["spread_correct"] == 1
    assert summary["games"]["spread_hit_rate"] == 100.0
    assert summary["games"]["_records"][0]["spread_side"] == "HOME"
    assert summary["games"]["_records"][0]["spread_status"] == "True"
    assert summary["props"]["_records"] == []
    # No book total was captured with this forecast, so there is no invented O/U record.
    assert "total_side_hit_rate" not in summary["games"]
    assert summary["games"]["_records"][0]["total_side"] == "—"


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


def test_game_total_side_graded_against_captured_book_line(tmp_path):
    projections = pd.DataFrame(
        [
            {
                "run_id": "r1", "generated_at": "2026-07-09T12:00:00+00:00",
                "date": "2026-07-10", "away": "CHI", "home": "MIN",
                "projected_away_pts": 78.0, "projected_home_pts": 88.0,
                "projected_total": 166.0, "projected_home_spread": 10.0,
                "home_win_prob": 0.75, "win_prob_basis": "test",
                "book_total_line": 160.5,
            }
        ]
    )
    assert log_game_projections(tmp_path, projections, "2026-27") == 1
    logged = pd.read_csv(game_log_path(tmp_path))
    assert logged.iloc[0]["predicted_total_side"] == "OVER"

    # Actual 80+90=170, over 160.5 — model over is correct.
    results = pd.DataFrame(
        [{"date": "2026-07-10", "away": "CHI", "home": "MIN", "awayPts": 80, "homePts": 90, "winner": "MIN"}]
    )
    assert grade_games(tmp_path, results)["graded"] == 1
    summary = results_summary(tmp_path)
    assert summary["games"]["total_side_n"] == 1
    assert summary["games"]["total_side_correct"] == 1
    assert summary["games"]["total_side_hit_rate"] == 100.0
    assert summary["games"]["_records"][0]["total_side"] == "OVER"
    assert summary["games"]["_records"][0]["total_status"] == "True"


def test_model_only_prop_settles_for_mae_without_inventing_wl(tmp_path):
    _log_prop(tmp_path, line=None, side="", odds=None, projection=24.0)
    today = datetime(2026, 7, 20, tzinfo=timezone.utc)
    outcome = grade_props(tmp_path, _player_logs(), today=today)
    assert outcome["graded"] == 1
    summary = results_summary(tmp_path)
    record = summary["props"]["player_points"]
    assert record["wins"] == 0
    assert record["losses"] == 0
    assert record["hit_rate"] is None
    assert record["mae"] == 6.0
    assert summary["props"]["_records"][0]["status"] == "Recorded"


def test_recorded_game_markets_grade_ml_spread_ats_and_total(tmp_path):
    from wnba_edges.predictions import grade_markets, log_market_predictions_batch, market_log_path
    from wnba_edges.prop_projections import build_game_market_slate

    game = pd.DataFrame(
        [
            {
                "run_id": "r1", "generated_at": "2026-07-09T12:00:00+00:00",
                "date": "2026-07-10", "away": "CHI", "home": "MIN",
                "projected_away_pts": 78.0, "projected_home_pts": 88.0,
                "projected_total": 166.0, "projected_home_spread": 10.0,
                "home_win_prob": 0.75, "win_prob_basis": "test",
                "book_total_line": 160.5, "book_spread_line": -4.5,
                "book_home_ml": -200, "book_away_ml": 170,
                "book_spread_odds": -110, "book_spread_opposite": -110,
                "book_total_over_odds": -110, "book_total_under_odds": -110,
                "book_ml_book": "draftkings", "book_spread_book": "draftkings",
                "book_total_book": "draftkings",
            }
        ]
    )
    assert log_game_projections(tmp_path, game, "2026-27") == 1
    slate = build_game_market_slate(game)
    assert log_market_predictions_batch(tmp_path, slate, "2026-27") == 3
    assert log_market_predictions_batch(tmp_path, slate, "2026-27") == 0

    results = pd.DataFrame(
        [{"date": "2026-07-10", "away": "CHI", "home": "MIN", "awayPts": 80, "homePts": 90, "winner": "MIN"}]
    )
    assert grade_games(tmp_path, results)["graded"] == 1
    assert grade_markets(tmp_path, results)["graded"] == 3

    summary = results_summary(tmp_path)
    assert summary["games"]["spread_ats_n"] == 1
    assert summary["games"]["spread_ats_correct"] == 1
    assert summary["markets"]["moneyline"]["wins"] == 1
    assert summary["markets"]["spread"]["wins"] == 1
    assert summary["markets"]["total"]["wins"] == 1
    logged = pd.read_csv(market_log_path(tmp_path))
    assert set(logged["market"]) == {"moneyline", "spread", "total"}
    assert (logged["settled"].astype(str).str.lower() == "true").all()


def test_unpriced_markets_settle_without_inventing_wl(tmp_path):
    from wnba_edges.predictions import grade_markets, log_market_predictions_batch
    from wnba_edges.prop_projections import build_game_market_slate

    game = pd.DataFrame(
        [
            {
                "run_id": "r1", "generated_at": "2026-07-09T12:00:00+00:00",
                "date": "2026-07-10", "away": "CHI", "home": "MIN",
                "projected_away_pts": 78.0, "projected_home_pts": 88.0,
                "projected_total": 166.0, "projected_home_spread": 10.0,
                "home_win_prob": 0.75, "win_prob_basis": "test",
            }
        ]
    )
    slate = build_game_market_slate(game)
    assert log_market_predictions_batch(tmp_path, slate, "2026-27") == 3
    results = pd.DataFrame(
        [{"date": "2026-07-10", "away": "CHI", "home": "MIN", "awayPts": 80, "homePts": 90, "winner": "MIN"}]
    )
    assert grade_markets(tmp_path, results)["graded"] == 3
    summary = results_summary(tmp_path)
    # Moneyline still has a recorded favorite even without book odds.
    assert summary["markets"]["moneyline"]["wins"] == 1
    # Spread/total have no captured line, so MAE-style settle with no W-L.
    assert summary["markets"]["spread"]["wins"] == 0
    assert summary["markets"]["spread"]["losses"] == 0
    assert summary["markets"]["spread"]["hit_rate"] is None
    assert summary["markets"]["total"]["wins"] == 0
    assert summary["markets"]["total"]["losses"] == 0
    assert summary["markets"]["total"]["hit_rate"] is None
