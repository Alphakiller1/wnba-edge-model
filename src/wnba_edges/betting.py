from __future__ import annotations

from dataclasses import dataclass


IMPLAUSIBLE_EDGE = 0.15

CONFIDENCE_TIERS = [
    (0.080, "Strong", "1.0-2.0u"),
    (0.045, "Standard", "0.5-1.0u"),
    (0.020, "Lean", "0.25-0.5u"),
    (-1.0, "Pass", "0u"),
]


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


def american_to_decimal(odds: int) -> float:
    return 1 + (odds / 100.0) if odds > 0 else 1 + (100.0 / -odds)


def american_to_implied(odds: int) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / (-odds + 100.0)


def prob_to_american(probability: float) -> int:
    p = min(max(probability, 1e-4), 1 - 1e-4)
    if p >= 0.5:
        return -round(100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def value_layer(model_prob: float, odds: int) -> ValueResult:
    model_prob = min(max(model_prob, 0.001), 0.999)
    implied = american_to_implied(odds)
    decimal = american_to_decimal(odds)
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
    )


def estimate_over_probability(projection: float, line: float, sigma: float = 6.5) -> float:
    """Normal approximation for player points/rebounds/assists props.

    This is a transparent prior. Replace sigma by market/stat-specific historical
    error once game logs and settled props are available.
    """
    import math

    if sigma <= 0:
        sigma = 6.5
    z = (line - projection) / sigma
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return 1 - cdf


def evaluate_over_under(projection: float, line: float, odds: int, side: str, sigma: float = 6.5) -> ValueResult:
    p_over = estimate_over_probability(projection=projection, line=line, sigma=sigma)
    model_prob = p_over if side.lower() == "over" else 1 - p_over
    return value_layer(model_prob=model_prob, odds=odds)
