from datetime import datetime, timezone

import pandas as pd
import pytest

from wnba_edges.cli import _projection_for_market
from wnba_edges.predictions import log_prop_predictions_batch, prop_log_path
from wnba_edges.prop_projections import (
    attach_game_market_lines,
    build_slate_prop_projections,
    projection_for_market,
)


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


def test_attach_game_market_lines_uses_consensus_total_and_home_spread():
    projections = pd.DataFrame(
        [{"away": "CHI", "home": "MIN", "projected_total": 166.0, "projected_home_spread": 4.0}]
    )
    odds = pd.DataFrame(
        [
            {"away": "CHI", "home": "MIN", "market": "total", "side": "over", "line": 165.5},
            {"away": "CHI", "home": "MIN", "market": "total", "side": "under", "line": 165.5},
            {"away": "CHI", "home": "MIN", "market": "spread", "side": "MIN", "line": -3.5},
            {"away": "CHI", "home": "MIN", "market": "spread", "side": "CHI", "line": 3.5},
        ]
    )
    out = attach_game_market_lines(projections, odds)
    assert out.iloc[0]["book_total_line"] == 165.5
    assert out.iloc[0]["book_spread_line"] == -3.5


def test_attach_game_market_lines_ignores_a_later_rematch():
    projections = pd.DataFrame(
        [
            {
                "away": "ATL", "home": "LAS", "date": "2026-08-20",
                "time": "2026-08-21T02:00:00Z", "projected_total": 178.0,
            },
            {
                "away": "ATL", "home": "LAS", "date": "2026-08-24",
                "time": "2026-08-25T02:00:00Z", "projected_total": 178.0,
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "away": "ATL", "home": "LAS", "market": "total", "side": "over",
                "line": 181.5, "commence_time": "2026-08-21T02:00:00Z",
            },
            {
                "away": "ATL", "home": "LAS", "market": "spread", "side": "LAS",
                "line": -4.5, "commence_time": "2026-08-21T02:00:00Z",
            },
        ]
    )
    out = attach_game_market_lines(projections, odds)
    assert out.iloc[0]["book_total_line"] == 181.5
    assert pd.isna(out.iloc[1]["book_total_line"])
    assert pd.isna(out.iloc[1]["book_spread_line"])


def test_build_slate_prop_projections_prices_when_a_quote_exists():
    features = pd.DataFrame(
        [
            {
                "name": "Star Player", "id": 1, "team": "MIN",
                "ppg": 20.0, "rpg": 8.0, "apg": 3.0, "mpg": 32.0,
                "projectedMinutes": 32.0, "projectedPoints": 20.0, "low_sample": False,
            },
            {
                "name": "Away Guard", "id": 2, "team": "CHI",
                "ppg": 14.0, "rpg": 4.0, "apg": 5.0, "mpg": 28.0,
                "projectedMinutes": 28.0, "projectedPoints": 14.0, "low_sample": False,
            },
        ]
    )
    schedule = pd.DataFrame([{"date": "2026-08-18", "away": "CHI", "home": "MIN"}])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    odds = pd.DataFrame(
        [
            {
                "away": "CHI", "home": "MIN", "player": "Star Player",
                "market": "player_points", "side": "Star Player|over",
                "line": 19.5, "odds": -110, "book": "draftkings", "fetched_at": now,
            },
            {
                "away": "CHI", "home": "MIN", "player": "Star Player",
                "market": "player_points", "side": "Star Player|under",
                "line": 19.5, "odds": -110, "book": "draftkings", "fetched_at": now,
            },
        ]
    )
    slate = build_slate_prop_projections(
        features, schedule, None, odds,
        season="2026-27", run_id="r1", generated_at=now,
    )
    assert not slate.empty
    assert set(slate["market"]) <= {
        "player_points", "player_rebounds", "player_assists", "player_threes",
    }
    star_pts = slate[(slate["player"] == "Star Player") & (slate["market"] == "player_points")].iloc[0]
    assert bool(star_pts["priced"]) is True
    assert star_pts["line"] == 19.5
    assert star_pts["side"] == "over"
    unpriced = slate[(slate["player"] == "Away Guard") & (slate["market"] == "player_points")].iloc[0]
    assert bool(unpriced["priced"]) is False
    assert unpriced["verdict"] == "PROJ"


def test_prop_batch_log_is_idempotent(tmp_path):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    slate = pd.DataFrame(
        [
            {
                "run_id": "r1", "generated_at": now, "game_date": "2026-08-18",
                "away": "CHI", "home": "MIN", "team": "MIN",
                "player": "Star Player", "player_id": 1, "market": "player_points",
                "projection": 20.0, "projection_basis": "season PPG",
                "sigma": 6.0, "sigma_source": "league", "line": 19.5, "side": "over",
                "odds": -110, "opposite_odds": -110, "book": "draftkings",
                "model_prob": 0.55, "implied_prob": 0.5, "vig_free": True,
                "edge": 0.05, "tier": "Lean", "verdict": "PLAY", "priced": True,
            }
        ]
    )
    assert log_prop_predictions_batch(tmp_path, slate, "2026-27") == 1
    assert log_prop_predictions_batch(tmp_path, slate, "2026-27") == 0
    assert len(pd.read_csv(prop_log_path(tmp_path))) == 1
    # projection_for_market remains the shared estimator the CLI wraps.
    assert projection_for_market(_row(projectedMinutes=20), "player_assists")[0] == pytest.approx(4.0)
