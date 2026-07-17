import pytest

from wnba_edges.betting import (
    american_to_decimal,
    american_to_implied,
    devig_pair,
    estimate_over_probability,
    prob_to_american,
    sigma_for_market,
    value_layer,
)


def test_american_round_trips():
    assert american_to_implied(-110) == pytest.approx(110 / 210)
    assert american_to_implied(150) == pytest.approx(100 / 250)
    assert american_to_decimal(-110) == pytest.approx(1 + 100 / 110)
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert prob_to_american(0.5238) in (-110, -109, -111)
    assert prob_to_american(0.4) == 150


def test_devig_pair_removes_overround():
    p_over, p_under = devig_pair(-110, -110)
    assert p_over == pytest.approx(0.5)
    assert p_over + p_under == pytest.approx(1.0)
    p_fav, p_dog = devig_pair(-150, 130)
    assert p_fav + p_dog == pytest.approx(1.0)
    assert p_fav > 0.5


def test_value_layer_vig_flag_and_edge():
    raw = value_layer(0.55, -110)
    assert raw.vig_free is False
    devig = value_layer(0.55, -110, opposite_odds=-110)
    assert devig.vig_free is True
    # De-vigged implied (0.500) is below raw implied (0.524): edge grows.
    assert devig.edge > raw.edge


def test_tiers_and_review_threshold():
    assert value_layer(0.55, -110, opposite_odds=-110).tier == "Standard"
    assert value_layer(0.53, -110, opposite_odds=-110).tier == "Lean"
    assert value_layer(0.60, -110, opposite_odds=-110).tier == "Strong"
    review = value_layer(0.70, -110, opposite_odds=-110)
    assert review.verdict == "REVIEW"
    assert review.implausible


def test_estimate_over_probability_symmetry():
    p = estimate_over_probability(projection=20.0, line=20.0, sigma=5.0)
    assert p == pytest.approx(0.5)
    hi = estimate_over_probability(projection=25.0, line=20.0, sigma=5.0)
    lo = estimate_over_probability(projection=15.0, line=20.0, sigma=5.0)
    assert hi + lo == pytest.approx(1.0)


def test_sigma_fallbacks_differ_by_market():
    assert sigma_for_market("player_assists") < sigma_for_market("player_points")
    assert sigma_for_market("unknown_market") == 6.5
