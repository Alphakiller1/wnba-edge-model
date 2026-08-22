from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import pandas as pd

from .betting import estimate_over_probability
from .teams import KNOWN_ABBRS

# Same Normal sigma used to price game spread/total sides. Home court is the
# point margin that reproduces the observed home-win rate under this noise model.
GAME_MARGIN_SIGMA = 12.0
# Φ^{-1}(0.54) * 12 ≈ 1.2 — a typical WNBA home-win rate, used until the season
# sample is large enough to estimate from finished games.
HOME_COURT_POINTS_PRIOR = 1.2
MIN_GAMES_FOR_HOME_COURT = 20
HOME_COURT_SHRINK_K = 40
HOME_COURT_MAX = 3.0

_UNIT_NORMAL = NormalDist()


class UnknownTeamsError(ValueError):
    """Raised instead of silently dropping games whose team code is unrecognized."""

    def __init__(self, games: list[str]):
        self.games = games
        super().__init__(
            "Unknown team code(s) in schedule — refusing to silently drop games: "
            + ", ".join(games)
        )


def estimate_home_court(game_results: pd.DataFrame | None) -> tuple[float, str]:
    """Home-court points consistent with how often home actually wins.

    Raw mean home margin is pulled up by blowouts. Using that mean as home court,
    then converting the projected margin through ``GAME_MARGIN_SIGMA``, overstates
    home favorites relative to the league home-win rate. This estimator maps the
    observed home-win rate through the same sigma used to price covers, blends in
    the mean margin, and shrinks toward the 1.2-pt prior.
    """
    if game_results is None or game_results.empty:
        return HOME_COURT_POINTS_PRIOR, "prior"
    frame = game_results.copy()
    if "home_margin" not in frame.columns:
        return HOME_COURT_POINTS_PRIOR, "prior"
    frame["_margin"] = pd.to_numeric(frame["home_margin"], errors="coerce")
    frame = frame.dropna(subset=["_margin"])
    n = len(frame)
    if n < MIN_GAMES_FOR_HOME_COURT:
        return HOME_COURT_POINTS_PRIOR, "prior"

    margin_hca = float(frame["_margin"].mean())
    winrate_hca: float | None = None
    p_home: float | None = None
    if "winner" in frame.columns and "home" in frame.columns:
        home = frame["home"].astype(str).str.upper()
        winner = frame["winner"].astype(str).str.upper()
        decided = winner.ne("") & winner.ne("NAN") & home.ne("")
        if int(decided.sum()) >= MIN_GAMES_FOR_HOME_COURT:
            p_home = float((winner[decided] == home[decided]).mean())
            winrate_hca = GAME_MARGIN_SIGMA * _inv_phi(p_home)

    if winrate_hca is None:
        blended = margin_hca
        basis = f"shrunk mean margin (n={n})"
    else:
        # Two-thirds weight on the win-rate mapping so moneyline and spread do
        # not inherit blowout-inflated home court.
        blended = (margin_hca + 2.0 * winrate_hca) / 3.0
        basis = (
            f"win-rate blend (n={n}, home win {p_home:.0%}, "
            f"mean margin {margin_hca:.1f})"
        )

    value = (n * blended + HOME_COURT_SHRINK_K * HOME_COURT_POINTS_PRIOR) / (
        n + HOME_COURT_SHRINK_K
    )
    return min(max(value, 0.0), HOME_COURT_MAX), basis


def _inv_phi(probability: float) -> float:
    clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return float(_UNIT_NORMAL.inv_cdf(clipped))


def win_probability_from_margin(margin: float, sigma: float = GAME_MARGIN_SIGMA) -> float:
    """P(home wins) from the projected home margin under the game-level Normal."""
    return float(estimate_over_probability(margin, 0.0, sigma=sigma))


def build_game_projections(
    teams: pd.DataFrame,
    schedule: pd.DataFrame,
    game_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Project each scheduled WNBA club game.

    Exhibition / All-Star / unknown codes (e.g. TEA@TEA) are skipped with a
    printed warning — same posture as the MLB model's All-Star skip — instead
    of aborting the whole slate.
    """
    team_index = teams.set_index("abbr").to_dict(orient="index")
    home_court, home_court_basis = estimate_home_court(game_results)
    win_basis = (
        f"Normal CDF of projected margin (σ={GAME_MARGIN_SIGMA:g})"
    )
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
        away_pts = round(away_eff * pace / 100 - home_court / 2, 1)
        home_pts = round(home_eff * pace / 100 + home_court / 2, 1)
        spread_home = round(home_pts - away_pts, 1)
        total = round(away_pts + home_pts, 1)
        # Price the posted (rounded) margin so moneyline and spread cannot disagree.
        home_win = win_probability_from_margin(spread_home)
        rows.append(
            {
                "run_id": run_id,
                "generated_at": generated_at,
                "date": game.get("date", ""),
                "time": game.get("time", ""),
                "away": away,
                "home": home,
                "projected_away_pts": away_pts,
                "projected_home_pts": home_pts,
                "projected_total": total,
                "projected_home_spread": spread_home,
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
        print(
            f"skipped {len(unknown)} non-club schedule row(s): "
            + "; ".join(unknown)
        )
    if not rows and not schedule.empty:
        # Still hard-fail if *every* row was unrecognized — that is data drift.
        raise UnknownTeamsError(unknown or ["(empty projection after filters)"])
    return pd.DataFrame(rows)


def load_schedule(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
