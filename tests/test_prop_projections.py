import pandas as pd
import pytest

from wnba_edges.cli import _projection_for_market


def _row(**overrides):
    values = {
        "id": 7,
        "name": "Test Player",
        "ppg": 10.0,
        "rpg": 5.0,
        "apg": 4.0,
        "mpg": 20.0,
        "projectedMinutes": 24.0,
        "projectedPoints": 12.0,
    }
    values.update(overrides)
    return pd.Series(values)


def _logs():
    return pd.DataFrame(
        [
            {"playerId": 7, "date": f"2026-07-{day:02d}", "min": 20, "pts": pts,
             "reb": 5, "ast": 4, "fg3m": threes, "stl": 1, "blk": 1}
            for day, pts, threes in [(1, 10, 2), (3, 12, 1), (5, 14, 2), (7, 16, 3), (9, 18, 2)]
        ]
    )


def test_points_projection_blends_recent_form_minutes_and_source_projection():
    projection, basis = _projection_for_market(_row(), "player_points", _logs())
    # (10 * .55 + 14 * .45) * 1.2, then a conservative 35% source blend at 12.
    assert projection == pytest.approx(13.4)
    assert "L5 pts" in basis
    assert "minutes x1.20" in basis
    assert "source pts x0.35" in basis


def test_non_points_market_uses_its_own_history_not_points_fallback():
    projection, basis = _projection_for_market(_row(), "player_threes", _logs())
    assert projection == pytest.approx(2.4)
    assert "fg3m" in basis


def test_short_recent_sample_stays_at_season_prior():
    logs = _logs().head(2)
    projection, basis = _projection_for_market(_row(projectedMinutes=20), "player_rebounds", logs)
    assert projection == pytest.approx(5.0)
    assert "season anchor" in basis
