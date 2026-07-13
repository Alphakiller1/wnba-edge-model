from __future__ import annotations

from pathlib import Path

import pandas as pd


HOME_COURT_POINTS = 1.5


def build_game_projections(teams: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    team_index = teams.set_index("abbr").to_dict(orient="index")
    rows: list[dict] = []
    for _, game in schedule.iterrows():
        away = str(game["away"]).strip().upper()
        home = str(game["home"]).strip().upper()
        if away not in team_index or home not in team_index:
            continue
        away_team = team_index[away]
        home_team = team_index[home]
        pace = (float(away_team["pace"]) + float(home_team["pace"])) / 2
        away_eff = (float(away_team["ortg"]) + float(home_team["drtg"])) / 2
        home_eff = (float(home_team["ortg"]) + float(away_team["drtg"])) / 2
        away_pts = away_eff * pace / 100 - HOME_COURT_POINTS / 2
        home_pts = home_eff * pace / 100 + HOME_COURT_POINTS / 2
        spread_home = home_pts - away_pts
        total = away_pts + home_pts
        rating_gap = float(home_team["net"]) - float(away_team["net"])
        confidence = min(88, max(42, 58 + abs(rating_gap) * 1.7))
        rows.append(
            {
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
                "confidence": round(confidence, 1),
                "model_note": "Team ORtg/DRtg + blended pace baseline; injury/odds adjustments pending.",
            }
        )
    return pd.DataFrame(rows)


def load_schedule(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_default_schedule(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("2026-07-13", "19:00 ET", "LAS", "ATL"),
        ("2026-07-13", "21:00 ET", "PHX", "MIN"),
        ("2026-07-14", "11:00 ET", "PDX", "CON"),
        ("2026-07-14", "19:00 ET", "WAS", "TOR"),
        ("2026-07-15", "12:00 ET", "SEA", "CHI"),
        ("2026-07-15", "13:00 ET", "LAS", "MIN"),
        ("2026-07-15", "20:00 ET", "GSV", "IND"),
        ("2026-07-16", "19:00 ET", "PDX", "WAS"),
        ("2026-07-16", "21:00 ET", "NYL", "DAL"),
        ("2026-07-17", "19:30 ET", "SEA", "IND"),
        ("2026-07-17", "19:30 ET", "LAS", "CHI"),
        ("2026-07-17", "19:30 ET", "ATL", "TOR"),
        ("2026-07-17", "22:00 ET", "CON", "PHX"),
        ("2026-07-18", "20:00 ET", "NYL", "IND"),
        ("2026-07-18", "20:00 ET", "PDX", "MIN"),
        ("2026-07-18", "20:30 ET", "WAS", "GSV"),
        ("2026-07-19", "13:00 ET", "LAS", "DAL"),
        ("2026-07-19", "16:00 ET", "CHI", "ATL"),
        ("2026-07-19", "19:00 ET", "CON", "PHX"),
    ]
    pd.DataFrame(rows, columns=["date", "time", "away", "home"]).to_csv(path, index=False)
