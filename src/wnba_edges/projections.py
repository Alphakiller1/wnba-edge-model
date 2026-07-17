from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .teams import KNOWN_ABBRS

# Prior used only until enough finished games exist to estimate home court
# empirically (see `estimate_home_court`).
HOME_COURT_POINTS_PRIOR = 1.5
MIN_GAMES_FOR_HOME_COURT = 20
MIN_GAMES_FOR_WIN_FIT = 40


class UnknownTeamsError(ValueError):
    """Raised instead of silently dropping games whose team code is unrecognized."""

    def __init__(self, games: list[str]):
        self.games = games
        super().__init__(
            "Unknown team code(s) in schedule — refusing to silently drop games: "
            + ", ".join(games)
        )


def estimate_home_court(game_results: pd.DataFrame | None) -> tuple[float, str]:
    """Mean home margin from finished games; falls back to the prior when thin."""
    if game_results is None or game_results.empty or "home_margin" not in game_results.columns:
        return HOME_COURT_POINTS_PRIOR, "prior"
    margins = pd.to_numeric(game_results["home_margin"], errors="coerce").dropna()
    if len(margins) < MIN_GAMES_FOR_HOME_COURT:
        return HOME_COURT_POINTS_PRIOR, "prior"
    value = float(margins.mean())
    # Clamp to a plausible band; a wild empirical value signals data problems.
    return min(max(value, 0.0), 4.0), f"empirical (n={len(margins)})"


def fit_win_probability(
    game_results: pd.DataFrame | None,
    teams: pd.DataFrame,
) -> tuple[float, float, int]:
    """Logistic fit of P(home win) on net-rating gap over finished games.

    Returns (intercept, coefficient, n). Uses current season-to-date net ratings
    as the gap for each historical game — a simplification, but calibrated
    against real outcomes instead of the old hand-tuned clamp formula.
    """
    default = (0.35, 0.13, 0)  # gentle prior: ~59% at gap 0 (home court), +gap slope
    if game_results is None or game_results.empty:
        return default
    net_by_team = {
        str(row["abbr"]).upper(): float(row["net"])
        for _, row in teams.iterrows()
        if pd.notna(row.get("net"))
    }
    xs: list[float] = []
    ys: list[int] = []
    for _, game in game_results.iterrows():
        home = str(game.get("home", "")).upper()
        away = str(game.get("away", "")).upper()
        winner = str(game.get("winner", "")).upper()
        if home not in net_by_team or away not in net_by_team or winner not in {home, away}:
            continue
        xs.append(net_by_team[home] - net_by_team[away])
        ys.append(1 if winner == home else 0)
    n = len(xs)
    if n < MIN_GAMES_FOR_WIN_FIT:
        return default
    b0, b1 = _logistic_fit(xs, ys)
    # Guard against degenerate fits on odd data; keep the prior instead.
    if not (math.isfinite(b0) and math.isfinite(b1)) or b1 < 0 or b1 > 1.0:
        return default
    return b0, b1, n


def _logistic_fit(xs: list[float], ys: list[int], iterations: int = 200, lr: float = 0.01) -> tuple[float, float]:
    """Two-parameter logistic regression via plain gradient descent (no new deps)."""
    b0, b1 = 0.0, 0.05
    n = len(xs)
    for _ in range(iterations):
        g0 = g1 = 0.0
        for x, y in zip(xs, ys):
            p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
            g0 += p - y
            g1 += (p - y) * x
        b0 -= lr * g0 / n
        b1 -= lr * g1 / n
    return b0, b1


def win_probability(rating_gap: float, b0: float, b1: float) -> float:
    return 1.0 / (1.0 + math.exp(-(b0 + b1 * rating_gap)))


def build_game_projections(
    teams: pd.DataFrame,
    schedule: pd.DataFrame,
    game_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Project each scheduled game; raises on unknown teams instead of dropping them."""
    team_index = teams.set_index("abbr").to_dict(orient="index")
    home_court, home_court_basis = estimate_home_court(game_results)
    b0, b1, fit_n = fit_win_probability(game_results, teams)
    win_basis = f"logistic fit on {fit_n} games" if fit_n else "heuristic prior (insufficient finished games)"
    run_id = uuid.uuid4().hex[:12]
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    unknown: list[str] = []
    rows: list[dict] = []
    for _, game in schedule.iterrows():
        away = str(game["away"]).strip().upper()
        home = str(game["home"]).strip().upper()
        missing = [t for t in (away, home) if t not in team_index or t not in KNOWN_ABBRS]
        if missing:
            unknown.append(f"{away}@{home} ({', '.join(missing)})")
            continue
        away_team = team_index[away]
        home_team = team_index[home]
        pace = (float(away_team["pace"]) + float(home_team["pace"])) / 2
        away_eff = (float(away_team["ortg"]) + float(home_team["drtg"])) / 2
        home_eff = (float(home_team["ortg"]) + float(away_team["drtg"])) / 2
        away_pts = away_eff * pace / 100 - home_court / 2
        home_pts = home_eff * pace / 100 + home_court / 2
        spread_home = home_pts - away_pts
        total = away_pts + home_pts
        rating_gap = float(home_team["net"]) - float(away_team["net"])
        home_win = win_probability(rating_gap, b0, b1)
        rows.append(
            {
                "run_id": run_id,
                "generated_at": generated_at,
                "date": game.get("date", ""),
                "time": game.get("time", ""),
                "away": away,
                "home": home,
                "projected_away_pts": round(away_pts, 1),
                "projected_home_pts": round(home_pts, 1),
                "projected_total": round(total, 1),
                "projected_home_spread": round(spread_home, 1),
                "projected_pace": round(pace, 1),
                "away_ortg": round(float(away_team["ortg"]), 1),
                "away_drtg": round(float(away_team["drtg"]), 1),
                "home_ortg": round(float(home_team["ortg"]), 1),
                "home_drtg": round(float(home_team["drtg"]), 1),
                "away_net": round(float(away_team["net"]), 1),
                "home_net": round(float(home_team["net"]), 1),
                "home_win_prob": round(home_win, 4),
                "win_prob_basis": win_basis,
                "home_court_pts": round(home_court, 2),
                "home_court_basis": home_court_basis,
                "model_note": "Team ORtg/DRtg + blended pace baseline; injury/odds adjustments pending.",
            }
        )
    if unknown:
        raise UnknownTeamsError(unknown)
    return pd.DataFrame(rows)


def load_schedule(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
