from datetime import date

import pandas as pd
import pytest

from wnba_edges.contracts import ContractError, validate_frame, validate_rows
from wnba_edges.schedule import (
    apply_espn_player_logs,
    apply_finished_scores,
    parse_espn_player_box,
    parse_finished_scoreboard,
    parse_scoreboard,
)
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


def test_parse_espn_player_box_skips_dnp_and_reads_threes():
    payload = {
        "header": {
            "id": "401857161",
            "competitions": [
                {
                    "status": {"type": {"state": "post"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"abbreviation": "WSH"}},
                        {"homeAway": "away", "team": {"abbreviation": "MIN"}},
                    ],
                }
            ],
        },
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "MIN", "displayName": "Minnesota Lynx"},
                    "statistics": [
                        {
                            "keys": [
                                "minutes", "points",
                                "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
                                "rebounds", "assists", "steals", "blocks",
                            ],
                            "athletes": [
                                {
                                    "starter": True,
                                    "didNotPlay": False,
                                    "athlete": {"displayName": "Napheesa Collier"},
                                    "stats": ["33", "24", "3-7", "11", "4", "2", "1"],
                                },
                                {
                                    "didNotPlay": True,
                                    "athlete": {"displayName": "Injured Player"},
                                    "stats": [],
                                },
                            ],
                        }
                    ],
                }
            ]
        },
    }
    rows = parse_espn_player_box(payload, date(2026, 8, 21))
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "Napheesa Collier"
    assert row["team"] == "MIN"
    assert row["pts"] == 24
    assert row["fg3m"] == 3
    assert row["pra"] == 39


def test_apply_espn_player_logs_does_not_overwrite_existing():
    existing = pd.DataFrame(
        [{"date": "2026-08-21", "name": "Napheesa Collier", "pts": 24, "team": "MIN"}]
    )
    espn = pd.DataFrame(
        [
            {"date": "2026-08-21", "name": "Napheesa Collier", "pts": 99, "team": "MIN"},
            {"date": "2026-08-21", "name": "Kayla McBride", "pts": 18, "team": "MIN", "fg3m": 4},
        ]
    )
    out = apply_espn_player_logs(existing, espn, "2026-27")
    collier = out[out["name"] == "Napheesa Collier"].iloc[0]
    assert int(collier["pts"]) == 24
    assert "Kayla McBride" in set(out["name"])
