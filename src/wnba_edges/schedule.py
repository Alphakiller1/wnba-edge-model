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


def _event_sides(event: dict) -> tuple[str | None, str | None, str | None, str | None]:
    competitions = event.get("competitions") or []
    if not competitions:
        return None, None, None, None
    away = home = away_score = home_score = None
    for competitor in competitions[0].get("competitors", []):
        team = competitor.get("team") or {}
        name = team.get("displayName") or team.get("abbreviation") or ""
        abbr = team_abbr(name)
        score = competitor.get("score")
        if competitor.get("homeAway") == "home":
            home = abbr
            home_score = score
        elif competitor.get("homeAway") == "away":
            away = abbr
            away_score = score
    return away, home, away_score, home_score


def parse_scoreboard(payload: dict, day: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        status = ((event.get("status") or {}).get("type") or {}).get("state", "")
        if status != "pre":
            continue
        away, home, _, _ = _event_sides(event)
        if not away or not home:
            continue
        rows.append(
            {
                "date": day.isoformat(),
                "time": event.get("date", ""),
                "away": away,
                "home": home,
            }
        )
    return rows


def parse_finished_scoreboard(payload: dict, day: date) -> list[dict[str, Any]]:
    """Final scores from an ESPN scoreboard payload (WNBAnalytics often lags a day)."""
    rows: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        status = ((event.get("status") or {}).get("type") or {}).get("state", "")
        if status != "post":
            continue
        away, home, away_score, home_score = _event_sides(event)
        if not away or not home:
            continue
        try:
            away_pts = int(float(away_score))
            home_pts = int(float(home_score))
        except (TypeError, ValueError):
            continue
        if away_pts <= 0 and home_pts <= 0:
            continue
        winner = home if home_pts > away_pts else away
        rows.append(
            {
                "date": day.isoformat(),
                "time": event.get("date", ""),
                "away": away,
                "home": home,
                "awayPts": away_pts,
                "homePts": home_pts,
                "winner": winner,
                "source": "espn_scoreboard",
            }
        )
    return rows


def fetch_finished_scores(
    days_back: int = 4,
    start: date | None = None,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """Recent ESPN finals, including last night when the box-score source has not caught up."""
    client = client or HttpClient(timeout=30, retries=3, pause_seconds=1.0)
    start = start or date.today()
    urls = _scoreboard_urls()
    rows: list[dict[str, Any]] = []
    for offset in range(days_back):
        day = start - timedelta(days=offset)
        stamp = day.strftime("%Y%m%d")
        payload = None
        for base in urls:
            try:
                payload = client.get_json(f"{base}?dates={stamp}")
                break
            except Exception:
                continue
        if payload is None:
            continue
        rows.extend(parse_finished_scoreboard(payload, day))
    columns = ["date", "time", "away", "home", "awayPts", "homePts", "winner", "source"]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.drop_duplicates(subset=["date", "away", "home"]).reset_index(drop=True)


def apply_finished_scores(
    game_results: pd.DataFrame,
    finished: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    """Fill missing finals from ESPN without overwriting scored WNBAnalytics boxes."""
    if finished is None or finished.empty:
        return game_results
    out = game_results.copy()
    if out.empty:
        out = pd.DataFrame(columns=["date", "away", "home", "awayPts", "homePts", "winner", "season"])
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out["away"] = out["away"].astype(str).str.upper()
    out["home"] = out["home"].astype(str).str.upper()
    added: list[dict[str, Any]] = []
    for _, row in finished.iterrows():
        day = str(row["date"])[:10]
        away = str(row["away"]).upper()
        home = str(row["home"]).upper()
        away_pts = int(row["awayPts"])
        home_pts = int(row["homePts"])
        winner = str(row["winner"]).upper()
        mask = (out["date"] == day) & (out["away"] == away) & (out["home"] == home)
        if mask.any():
            existing_total = (
                pd.to_numeric(out.loc[mask, "awayPts"], errors="coerce").fillna(0)
                + pd.to_numeric(out.loc[mask, "homePts"], errors="coerce").fillna(0)
            )
            if (existing_total > 0).any():
                continue
            out.loc[mask, "awayPts"] = away_pts
            out.loc[mask, "homePts"] = home_pts
            out.loc[mask, "winner"] = winner
            out.loc[mask, "total"] = away_pts + home_pts
            out.loc[mask, "home_margin"] = home_pts - away_pts
            out.loc[mask, "away_result"] = "W" if winner == away else "L"
            out.loc[mask, "home_result"] = "W" if winner == home else "L"
            continue
        added.append(
            {
                "season": season,
                "date": day,
                "away": away,
                "home": home,
                "awayPts": away_pts,
                "homePts": home_pts,
                "winner": winner,
                "loser": away if winner == home else home,
                "total": away_pts + home_pts,
                "home_margin": home_pts - away_pts,
                "away_result": "W" if winner == away else "L",
                "home_result": "W" if winner == home else "L",
                "margin": abs(home_pts - away_pts),
                "source": "espn_scoreboard",
            }
        )
    if added:
        out = pd.concat([out, pd.DataFrame(added)], ignore_index=True)
    return out.sort_values(["date", "away", "home"]).reset_index(drop=True)


def write_schedule(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
