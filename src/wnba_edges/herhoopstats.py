from __future__ import annotations

from pathlib import Path
import html as html_lib
import re
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from .http import HttpClient


BASE_URL = "https://herhoopstats.com"


def fetch_research_table(
    research_type: str,
    min_season: int,
    max_season: int,
    stats_to_show: str = "traditional",
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """Fetch a public Her Hoop Stats WNBA reSEARCH table.

    Supported useful types include:
    - player_single_games
    - player_single_seasons
    - team_single_games
    - team_single_seasons

    HHS is form-driven; this uses the same GET parameters as the site UI.
    """
    client = client or HttpClient(timeout=60, retries=2, pause_seconds=2)
    url = f"{BASE_URL}/stats/wnba/research/{research_type}/"
    params = {
        "min_season": min_season,
        "max_season": max_season,
        "game_types": "reg",
        "result": "both",
        "location": "both",
        "stats_to_show": stats_to_show,
        "submit": "true",
    }
    # Use requests through the shared client behavior without adding a second
    # query builder dependency.
    import requests

    response = requests.get(url, params=params, headers={"User-Agent": client_headers_user_agent()}, timeout=client.timeout)
    response.raise_for_status()
    table = parse_first_research_table(response.text)
    table.insert(0, "source", "herhoopstats")
    table.insert(1, "research_type", research_type)
    return table


def parse_first_research_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.research") or soup.find("table")
    if table is None:
        raise ValueError("No Her Hoop Stats research table found")

    headers = [cell.get_text(" ", strip=True) for cell in table.select("thead th")]
    rows: list[list[str]] = []
    for tr in table.select("tbody tr"):
        cells = [
            html_lib.unescape(value)
            for value in re.findall(r'sorttable_customkey="([^"]*)"', str(tr))
        ]
        if not cells:
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    if not rows:
        raise ValueError("Her Hoop Stats research table had no rows")

    if not headers:
        headers = [f"col_{index + 1}" for index in range(max(len(row) for row in rows))]
    normalized_rows = [row + [""] * (len(headers) - len(row)) for row in rows]
    frame = pd.DataFrame(normalized_rows, columns=headers[: len(normalized_rows[0])])
    return clean_columns(frame)


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        str(column)
        .strip()
        .replace("%", "pct")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("+", "plus")
        .lower()
        for column in out.columns
    ]
    for column in out.columns:
        if column in {"player", "team", "opponent", "source", "research_type", "date"}:
            continue
        converted = pd.to_numeric(out[column].astype(str).str.replace(",", "", regex=False), errors="coerce")
        if converted.notna().any():
            out[column] = converted
    if "player" in out.columns:
        out["player"] = out["player"].map(normalize_player_name)
    if "mp" in out.columns:
        out["mp_decimal"] = out["mp"].map(minutes_sort_key_to_decimal)
    return out


def normalize_player_name(name: str) -> str:
    text = str(name).strip()
    if "," not in text:
        return text
    last, first = [part.strip() for part in text.split(",", 1)]
    return f"{first} {last}".strip()


def minutes_sort_key_to_decimal(value: Any) -> float | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        raw = int(float(text))
    except ValueError:
        return None
    minutes = raw // 100
    seconds = raw % 100
    return round(minutes + seconds / 60, 3)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def client_headers_user_agent() -> str:
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
