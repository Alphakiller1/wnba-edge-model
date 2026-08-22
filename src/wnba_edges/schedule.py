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
SUMMARY_PATH = "/apis/site/v2/sports/basketball/wnba/summary"

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


def _int_stat(value) -> int | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not text or text in {";", "-"}:
        return None
    if "-" in text and not text.startswith("-"):
        text = text.split("-", 1)[0]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _minutes_stat(value) -> float | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        try:
            return round(int(minutes) + int(seconds) / 60.0, 1)
        except ValueError:
            return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_espn_player_box(payload: dict, day: date) -> list[dict[str, Any]]:
    """Player lines from an ESPN game summary. DNP rows are skipped."""
    status = ((payload.get("header") or {}).get("competitions") or [{}])[0]
    state = ((status.get("status") or {}).get("type") or {}).get("state", "")
    if state and state != "post":
        return []
    away = home = None
    for competitor in status.get("competitors") or []:
        team = competitor.get("team") or {}
        abbr = team_abbr(team.get("displayName") or team.get("abbreviation") or "")
        if competitor.get("homeAway") == "home":
            home = abbr
        elif competitor.get("homeAway") == "away":
            away = abbr
    if not away or not home:
        return []
    event_id = str((payload.get("header") or {}).get("id") or "")
    rows: list[dict[str, Any]] = []
    for block in (payload.get("boxscore") or {}).get("players") or []:
        team = team_abbr(((block.get("team") or {}).get("displayName") or (block.get("team") or {}).get("abbreviation") or ""))
        opponent = away if team == home else home
        home_away = "home" if team == home else "away"
        groups = block.get("statistics") or []
        if not groups:
            continue
        keys = [str(key) for key in (groups[0].get("keys") or [])]
        for athlete in groups[0].get("athletes") or []:
            if athlete.get("didNotPlay"):
                continue
            person = athlete.get("athlete") or {}
            name = str(person.get("displayName") or "").strip()
            if not name:
                continue
            stats = dict(zip(keys, athlete.get("stats") or []))
            pts = _int_stat(stats.get("points"))
            reb = _int_stat(stats.get("rebounds"))
            ast = _int_stat(stats.get("assists"))
            stl = _int_stat(stats.get("steals"))
            blk = _int_stat(stats.get("blocks"))
            fg3m = _int_stat(stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
            minutes = _minutes_stat(stats.get("minutes"))
            if pts is None and reb is None and ast is None:
                continue
            pts = pts or 0
            reb = reb or 0
            ast = ast or 0
            rows.append(
                {
                    "date": day.isoformat(),
                    "game_id": event_id,
                    "name": name,
                    "team": team,
                    "opponent": opponent,
                    "home_away": home_away,
                    "min": minutes,
                    "pts": pts,
                    "reb": reb,
                    "ast": ast,
                    "stl": stl or 0,
                    "blk": blk or 0,
                    "fg3m": fg3m or 0,
                    "pra": pts + reb + ast,
                    "starter": bool(athlete.get("starter")),
                    "source": "espn_scoreboard",
                }
            )
    return rows


def fetch_finished_player_logs(
    days_back: int = 4,
    start: date | None = None,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """ESPN player boxes for recent finals, used when WNBAnalytics boxes have not landed."""
    client = client or HttpClient(timeout=30, retries=3, pause_seconds=0.6)
    start = start or date.today()
    urls = _scoreboard_urls()
    summary_urls = [host + SUMMARY_PATH for host in SCOREBOARD_HOSTS]
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
        for event in payload.get("events") or []:
            state = ((event.get("status") or {}).get("type") or {}).get("state", "")
            if state != "post":
                continue
            event_id = str(event.get("id") or "")
            if not event_id:
                continue
            summary = None
            for host in summary_urls:
                try:
                    summary = client.get_json(f"{host}?event={event_id}")
                    break
                except Exception:
                    continue
            if summary is None:
                continue
            rows.extend(parse_espn_player_box(summary, day))
    columns = [
        "date", "game_id", "name", "team", "opponent", "home_away",
        "min", "pts", "reb", "ast", "stl", "blk", "fg3m", "pra", "starter", "source",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.drop_duplicates(subset=["date", "name", "team"]).reset_index(drop=True)


def apply_espn_player_logs(player_logs: pd.DataFrame, espn_logs: pd.DataFrame, season: str) -> pd.DataFrame:
    """Append ESPN player lines that WNBAnalytics has not stored yet."""
    if espn_logs is None or espn_logs.empty:
        return player_logs
    out = player_logs.copy() if player_logs is not None else pd.DataFrame()
    if out.empty:
        added = espn_logs.copy()
        added["season"] = season
        return added.reset_index(drop=True)
    existing = set(
        zip(
            pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d"),
            out["name"].astype(str).str.lower(),
        )
    )
    extra = []
    for _, row in espn_logs.iterrows():
        key = (str(row["date"])[:10], str(row["name"]).lower())
        if key in existing:
            continue
        item = row.to_dict()
        item["season"] = season
        extra.append(item)
        existing.add(key)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
    return out.reset_index(drop=True)


def write_schedule(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
