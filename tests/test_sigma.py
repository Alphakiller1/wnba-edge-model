import random

import pandas as pd

from wnba_edges.sigma import fit_market_sigmas, player_sigma, resolve_sigma


def _logs(sigma_pts: float = 6.0, sigma_ast: float = 2.0, players: int = 20, games: int = 20):
    random.seed(7)
    rows = []
    for player in range(players):
        base_pts = random.uniform(8, 22)
        base_ast = random.uniform(1, 6)
        for _ in range(games):
            rows.append(
                {
                    "playerId": player,
                    "pts": random.gauss(base_pts, sigma_pts),
                    "ast": random.gauss(base_ast, sigma_ast),
                    "reb": random.gauss(5, 3.0),
                    "fg3m": random.gauss(1.5, 1.3),
                    "stl": random.gauss(1, 1.0),
                    "blk": random.gauss(0.7, 0.9),
                    "pra": random.gauss(base_pts + base_ast + 5, 8.0),
                }
            )
    return pd.DataFrame(rows)


def test_fitted_sigmas_recover_generating_values():
    sigmas = fit_market_sigmas(_logs())
    by_market = dict(zip(sigmas["market"], sigmas["sigma"]))
    assert abs(by_market["player_points"] - 6.0) < 1.0
    assert abs(by_market["player_assists"] - 2.0) < 0.6
    assert by_market["player_assists"] < by_market["player_points"]


def test_player_sigma_shrinks_toward_league():
    logs = _logs()
    sigma, n = player_sigma(logs, 0, "player_points", league_sigma=6.0)
    assert n == 20
    assert 3.0 < sigma < 9.0


def test_resolve_sigma_sources():
    logs = _logs()
    sigmas = fit_market_sigmas(logs)
    sigma, source = resolve_sigma("player_points", sigmas=sigmas, player_logs=logs, player_id=0)
    assert source.startswith("player-fitted")
    sigma, source = resolve_sigma("player_points", sigmas=sigmas)
    assert source == "league-fitted"
    sigma, source = resolve_sigma("player_points", sigmas=None)
    assert "fallback prior" in source
    assert sigma == 6.0
