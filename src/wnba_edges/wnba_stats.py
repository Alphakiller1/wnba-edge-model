from __future__ import annotations

from urllib.parse import urlencode

import pandas as pd

from .http import HttpClient

BASE_URL = "https://stats.wnba.com/stats"

STATS_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.wnba.com",
    "Referer": "https://www.wnba.com/",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def league_dash_player_stats(
    season: int = 2026,
    measure_type: str = "Advanced",
    per_mode: str = "PerGame",
    season_type: str = "Regular Season",
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """Fetch the official WNBA Stats league player dashboard.

    The WNBA Stats API is useful but can be picky about headers and network timing.
    Cache results and expect occasional retries.
    """
    client = client or HttpClient(timeout=45, retries=3, pause_seconds=2.0)
    params = {
        "College": "",
        "Conference": "",
        "Country": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "DraftPick": "",
        "DraftYear": "",
        "GameScope": "",
        "GameSegment": "",
        "Height": "",
        "LastNGames": 0,
        "LeagueID": "10",
        "Location": "",
        "MeasureType": measure_type,
        "Month": 0,
        "OpponentTeamID": 0,
        "Outcome": "",
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": per_mode,
        "Period": 0,
        "PlayerExperience": "",
        "PlayerPosition": "",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": season,
        "SeasonSegment": "",
        "SeasonType": season_type,
        "ShotClockRange": "",
        "StarterBench": "",
        "TeamID": 0,
        "VsConference": "",
        "VsDivision": "",
        "Weight": "",
    }
    payload = client.get_json(f"{BASE_URL}/leaguedashplayerstats?{urlencode(params)}", headers=STATS_HEADERS)
    result_set = payload["resultSets"][0]
    return pd.DataFrame(result_set["rowSet"], columns=result_set["headers"])
