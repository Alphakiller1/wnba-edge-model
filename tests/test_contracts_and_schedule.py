from datetime import date

import pandas as pd
import pytest

from wnba_edges.contracts import ContractError, validate_frame, validate_rows
from wnba_edges.schedule import parse_scoreboard
from wnba_edges.teams import team_abbr


def test_contract_missing_field_names_the_field():
    rows = [{"id": 1, "name": "x"}] * 120
    with pytest.raises(ContractError, match="team"):
        validate_rows(rows, "wnbanalytics_players")


def test_contract_min_rows():
    with pytest.raises(ContractError, match="at least"):
        validate_rows([], "wnbanalytics_games")


def test_contract_frame_passes():
    frame = pd.DataFrame(
        [{"abbr": "MIN", "name": "Minnesota Lynx", "ortg": 1, "drtg": 1, "net": 0, "pace": 80}] * 13
    )
    assert validate_frame(frame, "wnbanalytics_teams") is frame


def test_team_abbr_normalizes_foreign_codes():
    assert team_abbr("Las Vegas Aces") == "LVA"
    assert team_abbr("LV") == "LVA"
    assert team_abbr("CONN") == "CON"
    assert team_abbr("GS") == "GSV"
    assert team_abbr("MIN") == "MIN"


def test_parse_scoreboard_pre_games_only():
    payload = {
        "events": [
            {
                "date": "2026-07-20T23:00Z",
                "status": {"type": {"state": "pre"}},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"displayName": "Minnesota Lynx"}},
                            {"homeAway": "away", "team": {"displayName": "Las Vegas Aces"}},
                        ]
                    }
                ],
            },
            {
                "date": "2026-07-20T01:00Z",
                "status": {"type": {"state": "post"}},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"displayName": "Chicago Sky"}},
                            {"homeAway": "away", "team": {"displayName": "Indiana Fever"}},
                        ]
                    }
                ],
            },
        ]
    }
    rows = parse_scoreboard(payload, date(2026, 7, 20))
    assert rows == [
        {"date": "2026-07-20", "time": "2026-07-20T23:00Z", "away": "LVA", "home": "MIN"}
    ]
