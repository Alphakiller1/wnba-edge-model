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

# ESPN serves the SAME scoreboard payload from several hosts, and they do not fail
# together. As of 2026-08-12 `site.api.espn.com` returns 403 to this client (even
# with a browser User-Agent) while `site.web.api.espn.com` returns 200 on the
# identical path. Ordered by preference; `fetch_upcoming_schedule` falls through
# the list rather than dying on the first host that refuses us.
SCOREBOARD_HOSTS = [
    "https://site.web.api.espn.com",
    "https://site.api.espn.com",
]
SCOREBOARD_PATH = "/apis/site/v2/sports/basketball/wnba/scoreboard"

# Back-compat: callers/tests that imported the single URL still work.
SCOREBOARD_URL = SCOREBOARD_HOSTS[0] + SCOREBOARD_PATH


def _scoreboard_urls() -> list[str]:
    return [h + SCOREBOARD_PATH for h in SCOREBOARD_HOSTS]


def fetch_upcoming_schedule(
    days: int = 10,
    start: date | None = None,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """Pre-game events for the next `days` days: date, time, away, home."""
    client = client or HttpClient(timeout=30, retries=3, pause_seconds=1.0)
    start = start or date.today()
    urls = _scoreboard_urls()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        stamp = day.strftime("%Y%m%d")
        payload = None
        for base in urls:
            try:
                payload = client.get_json(f"{base}?dates={stamp}")
                break
            except Exception as exc:            # try the next host, do not abort
                failures.append(f"{base}: {type(exc).__name__}")
        if payload is None:
            continue
        rows.extend(parse_scoreboard(payload, day))
    if not rows and failures:
        raise RuntimeError(
            "no schedule rows; every ESPN host failed. Tried: "
            + "; ".join(dict.fromkeys(failures))
        )
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
