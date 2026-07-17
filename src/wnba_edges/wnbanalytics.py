from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .http import HttpClient

BASE_URL = "https://wnbanalytics.com"


def scrape_players(season: str = "2026-27", client: HttpClient | None = None) -> list[dict[str, Any]]:
    """Scrape the WNBAnalytics player board."""
    client = client or HttpClient()
    players = client.get_json(f"{BASE_URL}/api/players?season={season}")
    if season:
        players = [row for row in players if row.get("season") == season]
    return players


def players_frame(season: str = "2026-27", client: HttpClient | None = None) -> pd.DataFrame:
    rows = scrape_players(season=season, client=client)
    return pd.DataFrame(rows)


def scrape_games(season: str = "2026-27", client: HttpClient | None = None) -> list[dict[str, Any]]:
    client = client or HttpClient()
    games = client.get_json(f"{BASE_URL}/api/games?season={season}")
    for row in games:
        row.setdefault("season", season)
    return games


def scrape_teams(season: str = "2026-27", client: HttpClient | None = None) -> list[dict[str, Any]]:
    client = client or HttpClient()
    teams = client.get_json(f"{BASE_URL}/api/teams?season={season}")
    for row in teams:
        row.setdefault("season", season)
    return teams


def scrape_game_detail(game_id: int, client: HttpClient | None = None) -> dict[str, Any]:
    client = client or HttpClient()
    return client.get_json(f"{BASE_URL}/api/games/{game_id}")


def scrape_player_detail(slug: str, season: str, player_id: int, client: HttpClient | None = None) -> dict[str, Any]:
    client = client or HttpClient()
    return client.get_json(f"{BASE_URL}/api/players/{slug}?season={season}&playerId={player_id}")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Atomic write: a mid-run failure must never leave a half-written snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def extract_react_payload(html: str, key: str) -> Any:
    decoded = html.encode("utf-8").decode("unicode_escape")
    markers = (f'"{key}":', f'{key}":')
    array_start = -1
    marker = markers[0]
    for candidate in markers:
        array_start = decoded.find(candidate)
        if array_start != -1:
            marker = candidate
            break
    if array_start == -1:
        raise ValueError(f"Could not decode {key!r} payload")

    raw = decoded[array_start + len(marker) :]
    payload_text = _balanced_json_value(raw)
    normalized = payload_text.replace('"$undefined"', "null")
    return json.loads(normalized)


def _balanced_json_value(text: str) -> str:
    stripped = text.lstrip()
    if not stripped:
        raise ValueError("Empty payload")
    pairs = {"[": "]", "{": "}"}
    if stripped[0] not in pairs:
        raise ValueError(f"Payload does not start with JSON object/array: {stripped[:20]!r}")
    stack: list[str] = []
    in_string = False
    escaped = False

    for index, char in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return stripped[: index + 1]

    raise ValueError("Could not find the end of the embedded JSON value")


_extract_react_array = extract_react_payload
