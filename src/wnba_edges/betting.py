from __future__ import annotations

import math
from dataclasses import dataclass

IMPLAUSIBLE_EDGE = 0.15

CONFIDENCE_TIERS = [
    (0.080, "Strong", "1.0-2.0u"),
    (0.045, "Standard", "0.5-1.0u"),
    (0.020, "Lean", "0.25-0.5u"),
    (-1.0, "Pass", "0u"),
]

# Fallback per-market game-to-game standard deviations, used only when no fitted
# sigma table exists (`wnba-edges fit-sigma` writes market_sigma_<season>.csv from
# real player game logs). One global sigma is never applied across markets: assists
# and rebounds vary far less game-to-game than points, and pricing them with a
# points-sized sigma collapses every probability toward 50%.
DEFAULT_SIGMA_BY_MARKET = {
    "player_points": 6.0,
    "player_rebounds": 3.2,
    "player_assists": 2.4,
    "player_threes": 1.4,
    "player_steals": 1.2,
    "player_blocks": 1.1,
    "player_pra": 8.0,
}
LEGACY_DEFAULT_SIGMA = 6.5


@dataclass(frozen=True)
class ValueResult:
    model_prob: float
    implied_prob: float
    decimal_odds: float
    edge: float
    ev_per_unit: float
    fair_odds: int
    kelly_full: float
    kelly_quarter: float
    tier: str
    units: str
    verdict: str
    implausible: bool
    vig_free: bool = False


def american_to_decimal(odds: int) -> float:
    return 1 + (odds / 100.0) if odds > 0 else 1 + (100.0 / -odds)


def american_to_implied(odds: int) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / (-odds + 100.0)


def prob_to_american(probability: float) -> int:
    p = min(max(probability, 1e-4), 1 - 1e-4)
    if p >= 0.5:
        return -round(100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def devig_pair(odds_side: int, odds_opposite: int) -> tuple[float, float]:
    """Proportionally de-vig a two-way market.

    Returns (true_prob_side, true_prob_opposite). Raw implied probabilities from a
    single price include the book's margin and must never be labeled vig-free.
    """
    raw_side = american_to_implied(odds_side)
    raw_opp = american_to_implied(odds_opposite)
    overround = raw_side + raw_opp
    if overround <= 0:
        raise ValueError("Invalid odds pair: implied probabilities sum to zero.")
    return raw_side / overround, raw_opp / overround


def value_layer(model_prob: float, odds: int, opposite_odds: int | None = None) -> ValueResult:
    """Price a model probability against a market price.

    When `opposite_odds` (the other side of the same line) is supplied, the implied
    probability is de-vigged pairwise and the edge is measured against the true
    market probability. Without it the edge includes the book's vig and the result
    is flagged `vig_free=False`.
    """
    model_prob = min(max(model_prob, 0.001), 0.999)
    decimal = american_to_decimal(odds)
    if opposite_odds is not None:
        implied, _ = devig_pair(odds, opposite_odds)
        vig_free = True
    else:
        implied = american_to_implied(odds)
        vig_free = False
    edge = model_prob - implied
    ev = model_prob * (decimal - 1) - (1 - model_prob)
    b = decimal - 1
    kelly = (b * model_prob - (1 - model_prob)) / b if b > 0 else 0.0
    quarter_kelly = max(0.0, kelly / 4)

    tier, units = "Pass", "0u"
    for edge_min, label, unit_range in CONFIDENCE_TIERS:
        if edge >= edge_min:
            tier, units = label, unit_range
            break

    implausible = edge >= IMPLAUSIBLE_EDGE
    if implausible:
        verdict = "REVIEW"
        tier = "Review"
        units = "verify inputs"
    elif edge >= 0.020 and ev > 0:
        verdict = "PLAY"
    else:
        verdict = "PASS"

    return ValueResult(
        model_prob=round(model_prob, 4),
        implied_prob=round(implied, 4),
        decimal_odds=round(decimal, 3),
        edge=round(edge, 4),
        ev_per_unit=round(ev, 4),
        fair_odds=prob_to_american(model_prob),
        kelly_full=round(kelly, 4),
        kelly_quarter=round(quarter_kelly, 4),
        tier=tier,
        units=units,
        verdict=verdict,
        implausible=implausible,
        vig_free=vig_free,
    )


def sigma_for_market(market: str) -> float:
    """Fallback sigma when no fitted table is available."""
    return DEFAULT_SIGMA_BY_MARKET.get(market, LEGACY_DEFAULT_SIGMA)


def estimate_over_probability(projection: float, line: float, sigma: float = LEGACY_DEFAULT_SIGMA) -> float:
    """Normal approximation for player props.

    Sigma should come from the fitted per-market table (see `wnba_edges.sigma`);
    the default exists only for backward compatibility.
    """
    if sigma <= 0:
        sigma = LEGACY_DEFAULT_SIGMA
    z = (line - projection) / sigma
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return 1 - cdf


def evaluate_over_under(
    projection: float,
    line: float,
    odds: int,
    side: str,
    sigma: float = LEGACY_DEFAULT_SIGMA,
    opposite_odds: int | None = None,
) -> ValueResult:
    p_over = estimate_over_probability(projection=projection, line=line, sigma=sigma)
    model_prob = p_over if side.lower() == "over" else 1 - p_over
    return value_layer(model_prob=model_prob, odds=odds, opposite_odds=opposite_odds)
