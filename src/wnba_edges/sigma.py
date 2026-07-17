"""Per-market prop volatility fitted from real player game logs.

`fit_market_sigmas` estimates the game-to-game standard deviation of each prop
stat within players (deviation from the player's own season mean), which is the
sigma a Normal over/under approximation actually needs. A single global sigma
misprices low-variance markets (assists, rebounds) badly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MARKET_STAT_COLUMNS = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
    "player_threes": "fg3m",
    "player_steals": "stl",
    "player_blocks": "blk",
    "player_pra": "pra",
}

MIN_GAMES_PER_PLAYER = 8
# Shrinkage prior weight (in games) for per-player sigma toward the league sigma.
PLAYER_SHRINK_GAMES = 10


def market_sigma_path(root: Path, season: str) -> Path:
    return root / "data" / "processed" / f"market_sigma_{season}.csv"


def fit_market_sigmas(player_logs: pd.DataFrame) -> pd.DataFrame:
    """League-level within-player sigma per market, from player game logs."""
    rows: list[dict] = []
    for market, column in MARKET_STAT_COLUMNS.items():
        if column not in player_logs.columns:
            continue
        frame = player_logs[["playerId", column]].copy()
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=[column])
        counts = frame.groupby("playerId")[column].transform("count")
        frame = frame[counts >= MIN_GAMES_PER_PLAYER]
        if frame.empty:
            continue
        deviations = frame[column] - frame.groupby("playerId")[column].transform("mean")
        n = len(deviations)
        players = frame["playerId"].nunique()
        # Pooled within-player variance; ddof accounts for one mean per player.
        variance = float((deviations**2).sum()) / max(n - players, 1)
        rows.append(
            {
                "market": market,
                "stat": column,
                "sigma": round(variance**0.5, 3),
                "n_games": n,
                "n_players": players,
            }
        )
    return pd.DataFrame(rows)


def player_sigma(
    player_logs: pd.DataFrame,
    player_id,
    market: str,
    league_sigma: float,
) -> tuple[float, int]:
    """Per-player sigma shrunk toward the league sigma; returns (sigma, n_games)."""
    column = MARKET_STAT_COLUMNS.get(market)
    if column is None or column not in player_logs.columns:
        return league_sigma, 0
    values = pd.to_numeric(
        player_logs.loc[player_logs["playerId"] == player_id, column], errors="coerce"
    ).dropna()
    n = len(values)
    if n < 3:
        return league_sigma, n
    var_player = float(values.var(ddof=1))
    var_league = league_sigma**2
    var = ((n - 1) * var_player + PLAYER_SHRINK_GAMES * var_league) / (n - 1 + PLAYER_SHRINK_GAMES)
    return var**0.5, n


def load_market_sigmas(root: Path, season: str) -> pd.DataFrame | None:
    path = market_sigma_path(root, season)
    if not path.exists():
        return None
    return pd.read_csv(path)


def resolve_sigma(
    market: str,
    *,
    sigmas: pd.DataFrame | None = None,
    player_logs: pd.DataFrame | None = None,
    player_id=None,
) -> tuple[float, str]:
    """Best available sigma for a market: player-fitted > league-fitted > fallback prior."""
    from .betting import sigma_for_market

    league_sigma: float | None = None
    if sigmas is not None and not sigmas.empty:
        match = sigmas[sigmas["market"] == market]
        if not match.empty:
            league_sigma = float(match.iloc[0]["sigma"])

    if league_sigma is not None and player_logs is not None and player_id is not None:
        sigma, n = player_sigma(player_logs, player_id, market, league_sigma)
        if n >= 3:
            return sigma, f"player-fitted (n={n}, shrunk to league)"
        return league_sigma, "league-fitted"
    if league_sigma is not None:
        return league_sigma, "league-fitted"
    return sigma_for_market(market), "fallback prior (run fit-sigma to fit from logs)"
