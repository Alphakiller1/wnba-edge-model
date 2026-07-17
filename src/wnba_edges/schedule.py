"""Upcoming-slate schedule from ESPN's public scoreboard API.

Replaces the old hardcoded fallback schedule, which silently projected a stale
week of games whenever the schedule CSV was missing.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .http import HttpClient
from .teams import team_abbr

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"


def fetch_upcoming_schedule(
    days: int = 10,
    start: date | None = None,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """Pre-game events for the next `days` days: date, time, away, home."""
    client = client or HttpClient(timeout=30, retries=3, pause_seconds=1.0)
    start = start or date.today()
    rows: list[dict[str, Any]] = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        payload = client.get_json(f"{SCOREBOARD_URL}?dates={day.strftime('%Y%m%d')}")
        rows.extend(parse_scoreboard(payload, day))
    frame = pd.DataFrame(rows, columns=["date", "time", "away", "home"])
    return frame.drop_duplicates(subset=["date", "away", "home"]).reset_index(drop=True)


def parse_scoreboard(payload: dict, day: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        status = ((event.get("status") or {}).get("type") or {}).get("state", "")
        if status != "pre":
            continue
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        away = home = None
        for competitor in competitions[0].get("competitors", []):
            team = competitor.get("team") or {}
            name = team.get("displayName") or team.get("abbreviation") or ""
            abbr = team_abbr(name)
            if competitor.get("homeAway") == "home":
                home = abbr
            elif competitor.get("homeAway") == "away":
                away = abbr
        if not away or not home:
            continue
        start_time = event.get("date", "")
        rows.append(
            {
                "date": day.isoformat(),
                "time": start_time,
                "away": away,
                "home": home,
            }
        )
    return rows


def write_schedule(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
