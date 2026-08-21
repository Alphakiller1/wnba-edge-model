from datetime import date

import pandas as pd
import pytest

from wnba_edges.contracts import ContractError, validate_frame, validate_rows
from wnba_edges.schedule import apply_finished_scores, parse_finished_scoreboard, parse_scoreboard
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


def test_parse_finished_scoreboard_reads_finals():
    payload = {
        "events": [
            {
                "date": "2026-08-21T00:00Z",
                "status": {"type": {"state": "post"}},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "DAL"}, "score": "91"},
                            {"homeAway": "away", "team": {"abbreviation": "IND"}, "score": "85"},
                        ]
                    }
                ],
            },
            {
                "date": "2026-08-21T23:30Z",
                "status": {"type": {"state": "pre"}},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "WSH"}, "score": "0"},
                            {"homeAway": "away", "team": {"abbreviation": "MIN"}, "score": "0"},
                        ]
                    }
                ],
            },
        ]
    }
    rows = parse_finished_scoreboard(payload, date(2026, 8, 20))
    assert rows == [
        {
            "date": "2026-08-20",
            "time": "2026-08-21T00:00Z",
            "away": "IND",
            "home": "DAL",
            "awayPts": 85,
            "homePts": 91,
            "winner": "DAL",
            "source": "espn_scoreboard",
        }
    ]


def test_apply_finished_scores_appends_missing_and_skips_scored():
    existing = pd.DataFrame(
        [
            {
                "season": "2026-27",
                "date": "2026-08-19",
                "away": "MIN",
                "home": "GSV",
                "awayPts": 77,
                "homePts": 66,
                "winner": "MIN",
                "total": 143,
                "home_margin": -11,
            }
        ]
    )
    finished = pd.DataFrame(
        [
            {
                "date": "2026-08-19",
                "away": "MIN",
                "home": "GSV",
                "awayPts": 99,
                "homePts": 90,
                "winner": "MIN",
            },
            {
                "date": "2026-08-20",
                "away": "IND",
                "home": "DAL",
                "awayPts": 85,
                "homePts": 91,
                "winner": "DAL",
            },
        ]
    )
    out = apply_finished_scores(existing, finished, "2026-27")
    min_gsv = out[(out["away"] == "MIN") & (out["home"] == "GSV")].iloc[0]
    assert int(min_gsv["awayPts"]) == 77
    ind_dal = out[(out["away"] == "IND") & (out["home"] == "DAL")].iloc[0]
    assert int(ind_dal["homePts"]) == 91
    assert ind_dal["winner"] == "DAL"
    assert int(ind_dal["total"]) == 176


def test_apply_finished_scores_fills_zero_zero_placeholders():
    existing = pd.DataFrame(
        [
            {
                "season": "2026-27",
                "date": "2026-08-20",
                "away": "ATL",
                "home": "LAS",
                "awayPts": 0,
                "homePts": 0,
                "winner": "LAS",
                "total": 0,
                "home_margin": 0,
            }
        ]
    )
    finished = pd.DataFrame(
        [
            {
                "date": "2026-08-20",
                "away": "ATL",
                "home": "LAS",
                "awayPts": 124,
                "homePts": 88,
                "winner": "ATL",
            }
        ]
    )
    out = apply_finished_scores(existing, finished, "2026-27")
    row = out.iloc[0]
    assert int(row["awayPts"]) == 124
    assert int(row["homePts"]) == 88
    assert row["winner"] == "ATL"
    assert row["away_result"] == "W"
