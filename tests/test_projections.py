import math

import pandas as pd
import pytest

from wnba_edges.projections import (
    GAME_MARGIN_SIGMA,
    HOME_COURT_POINTS_PRIOR,
    UnknownTeamsError,
    build_game_projections,
    estimate_home_court,
    win_probability_from_margin,
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
    assert prior == HOME_COURT_POINTS_PRIOR and basis == "prior"
    margins = pd.DataFrame({"home_margin": [2.0] * 30})
    value, basis = estimate_home_court(margins)
    assert HOME_COURT_POINTS_PRIOR < value <= 2.0
    assert "shrunk mean" in basis


def test_home_court_ignores_blowout_mean_when_home_wins_are_even():
    """A 4-pt mean margin with a 50/50 home-win rate must not become a 4-pt HCA."""
    rows = []
    for i in range(40):
        if i % 2 == 0:
            rows.append({"away": "CHI", "home": "MIN", "winner": "MIN", "home_margin": 12.0})
        else:
            rows.append({"away": "CHI", "home": "MIN", "winner": "CHI", "home_margin": -4.0})
    value, basis = estimate_home_court(pd.DataFrame(rows))
    assert value < 2.0
    assert "win-rate blend" in basis
    assert "50%" in basis


def test_win_probability_follows_projected_margin():
    assert win_probability_from_margin(0) == pytest.approx(0.5)
    assert win_probability_from_margin(12) > win_probability_from_margin(0) > win_probability_from_margin(-12)
    for margin in (-20, 0, 20):
        p = win_probability_from_margin(margin)
        assert 0.0 < p < 1.0 and math.isfinite(p)
    # Even teams with a 1.2-pt home bump should sit near the league home-win rate.
    even = win_probability_from_margin(1.2)
    assert 0.52 < even < 0.56


def test_projections_include_win_prob_and_metadata():
    schedule = pd.DataFrame([{"date": "2026-07-20", "time": "19:00", "away": "CHI", "home": "MIN"}])
    out = build_game_projections(_teams(), schedule, _results())
    assert len(out) == 1
    row = out.iloc[0]
    assert 0.5 < row["home_win_prob"] < 1.0
    assert "Normal CDF" in row["win_prob_basis"]
    assert row["run_id"] and row["generated_at"]
    # Moneyline favorite and spread favorite must agree on the posted numbers.
    assert (row["projected_home_spread"] > 0) == (row["home_win_prob"] > 0.5)
    assert row["projected_home_spread"] == pytest.approx(
        row["projected_home_pts"] - row["projected_away_pts"]
    )


def test_toss_up_does_not_flip_to_home_on_spread_only():
    """Road talent of ~2 pts should stay a road favorite after a ~1.2-pt HCA."""
    teams = pd.DataFrame(
        [
            {"abbr": "IND", "ortg": 108.0, "drtg": 100.0, "net": 8.0, "pace": 80.0},
            {"abbr": "NYL", "ortg": 106.0, "drtg": 102.0, "net": 4.0, "pace": 80.0},
        ]
    )
    # 50/50 home wins, modest margins — HCA near the 1.2 prior.
    results = pd.DataFrame(
        [
            {"away": "IND", "home": "NYL", "winner": "NYL" if i % 2 == 0 else "IND",
             "home_margin": 2.0 if i % 2 == 0 else -2.0}
            for i in range(40)
        ]
    )
    schedule = pd.DataFrame([{"date": "2026-08-22", "time": "19:00", "away": "IND", "home": "NYL"}])
    row = build_game_projections(teams, schedule, results).iloc[0]
    assert row["projected_home_spread"] < 0
    assert row["home_win_prob"] < 0.5


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
    assert GAME_MARGIN_SIGMA == 12.0
