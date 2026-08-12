import math

import pandas as pd
import pytest

from wnba_edges.projections import (
    UnknownTeamsError,
    build_game_projections,
    estimate_home_court,
    fit_win_probability,
    win_probability,
)


def _teams():
    return pd.DataFrame(
        [
            {"abbr": "MIN", "ortg": 110.0, "drtg": 100.0, "net": 10.0, "pace": 82.0},
            {"abbr": "LVA", "ortg": 106.0, "drtg": 102.0, "net": 4.0, "pace": 80.0},
            {"abbr": "CHI", "ortg": 98.0, "drtg": 106.0, "net": -8.0, "pace": 78.0},
        ]
    )


def _results(n_per_gap: int = 30):
    """Synthetic finished games: big favorites win more often."""
    rows = []
    for i in range(n_per_gap):
        # MIN (net +10) hosts CHI (net -8): gap 18, home wins 90%.
        rows.append(
            {"date": "2026-06-01", "away": "CHI", "home": "MIN",
             "winner": "MIN" if i % 10 else "CHI", "home_margin": 12}
        )
        # LVA (net +4) hosts MIN (net +10): gap -6, home wins 40%.
        rows.append(
            {"date": "2026-06-02", "away": "MIN", "home": "LVA",
             "winner": "LVA" if i % 5 < 2 else "MIN", "home_margin": -3}
        )
    return pd.DataFrame(rows)


def test_home_court_empirical_and_prior():
    prior, basis = estimate_home_court(None)
    assert prior == 1.5 and basis == "prior"
    margins = pd.DataFrame({"home_margin": [2.0] * 30})
    value, basis = estimate_home_court(margins)
    assert value == pytest.approx(2.0)
    assert basis.startswith("empirical")


def test_win_probability_fit_is_directional():
    b0, b1, n = fit_win_probability(_results(), _teams())
    assert n == 60
    assert b1 > 0
    assert win_probability(18, b0, b1) > win_probability(0, b0, b1) > win_probability(-18, b0, b1)
    for gap in (-20, 0, 20):
        p = win_probability(gap, b0, b1)
        assert 0.0 < p < 1.0 and math.isfinite(p)


def test_projections_include_win_prob_and_metadata():
    schedule = pd.DataFrame([{"date": "2026-07-20", "time": "19:00", "away": "CHI", "home": "MIN"}])
    out = build_game_projections(_teams(), schedule, _results())
    assert len(out) == 1
    row = out.iloc[0]
    assert 0.5 < row["home_win_prob"] < 1.0
    assert "logistic fit" in row["win_prob_basis"]
    assert row["run_id"] and row["generated_at"]


def test_unknown_team_only_slate_raises():
    schedule = pd.DataFrame([{"date": "2026-07-20", "time": "19:00", "away": "XXX", "home": "YYY"}])
    with pytest.raises(UnknownTeamsError, match="XXX"):
        build_game_projections(_teams(), schedule, None)


def test_unknown_exhibition_skipped_when_club_games_exist():
    schedule = pd.DataFrame(
        [
            {"date": "2026-07-20", "time": "19:00", "away": "CHI", "home": "MIN"},
            {"date": "2026-07-20", "time": "15:00", "away": "TEA", "home": "TEA"},
        ]
    )
    out = build_game_projections(_teams(), schedule, _results())
    assert len(out) == 1
    assert out.iloc[0]["away"] == "CHI"
    assert out.iloc[0]["home"] == "MIN"
