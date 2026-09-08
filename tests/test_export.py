import json

import pandas as pd

from wnba_edges import export
from wnba_edges.report import build_site


def _seed(root):
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    (root / "data" / "odds").mkdir(parents=True)
    (root / "data" / "predictions").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "r1",
                "generated_at": "2026-07-16T12:00:00+00:00",
                "date": "2026-07-20",
                "time": "19:00 ET",
                "away": "CHI",
                "home": "MIN",
                "projected_away_pts": 78.0,
                "projected_home_pts": 88.0,
                "projected_total": 166.0,
                "projected_home_spread": 10.0,
                "projected_pace": 80.0,
                "away_net": -8.0,
                "home_net": 10.0,
                "home_win_prob": 0.78,
                "win_prob_basis": "logistic fit on 60 games",
                "book_total_line": 164.5,
                "book_spread_line": -9.5,
                "book_home_ml": -420,
                "book_away_ml": 330,
                "book_ml_book": "draftkings",
            }
        ]
    ).to_csv(processed / "game_projections_2026-27.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "r1",
                "generated_at": "2026-07-16T12:00:00+00:00",
                "game_date": "2026-07-20",
                "away": "CHI",
                "home": "MIN",
                "team": "MIN",
                "player": "Napheesa Collier",
                "player_id": "1001",
                "market": "player_points",
                "projection": 22.4,
                "line": 21.5,
                "side": "over",
                "odds": -110,
                "book": "draftkings",
                "model_prob": 0.54,
                "edge": 0.03,
                "verdict": "Lean",
                "priced": True,
            }
        ]
    ).to_csv(processed / "prop_projections_2026-27.csv", index=False)
    pd.DataFrame(
        [{"date": "2026-07-15", "away": "NYL", "home": "CON", "awayPts": 80, "homePts": 88}]
    ).to_csv(processed / "game_results_2026-27.csv", index=False)
    (root / "data" / "odds" / "odds_status.json").write_text(
        json.dumps({
            "requested_bookmakers": ["draftkings"],
            "game": {
                "fetched_at": "2026-07-16T12:00:00+00:00",
                "events": 1,
                "quotes": 6,
                "api_remaining": "40",
            },
        })
        + "\n",
        encoding="utf-8",
    )


def test_payload_carries_authority_and_projected_vs_book(tmp_path):
    _seed(tmp_path)
    payload = export.payload(tmp_path, "2026-27")
    assert payload["schema"] == export.SCHEMA
    assert payload["authority"] == "RESEARCH_ONLY"
    assert payload["may_bet"] is False
    assert payload["unmet_gates"] == []
    game = payload["games"][0]
    assert game["projected_total"] == 166.0
    assert game["book"]["total"] == 164.5
    assert game["kickoff"]
    assert "kickoff_utc" in game
    prop = payload["player_projections"][0]
    assert prop["player_id"] == "1001"
    assert prop["priced"] is True
    assert prop["action"] == "MONITOR"


def test_write_bundle_matches_nfl_field_contract(tmp_path):
    _seed(tmp_path)
    paths = export.write_bundle(tmp_path, tmp_path / "_site", "2026-27")
    board = json.loads(paths["board"].read_text(encoding="utf-8"))
    build = json.loads(paths["build"].read_text(encoding="utf-8"))
    record = json.loads(paths["record"].read_text(encoding="utf-8"))
    assert board["may_bet"] is False
    assert build["state"] in {"fresh", "degraded"}
    assert "generated_at" in build
    assert "sources" in build
    assert "odds" in build
    assert "issues" in build
    assert record["authority"] == "shadow_only"
    for key in (
        "games_graded",
        "pending_snapshots",
        "model_mae",
        "consensus_mae",
        "book_mae",
        "ats",
        "totals",
    ):
        assert key in record
    assert set(record["ats"]) == {"win", "loss", "push"}


def test_empty_projections_still_emit_explicit_absences(tmp_path):
    (tmp_path / "data" / "processed").mkdir(parents=True)
    pd.DataFrame(columns=["book_total_line", "book_spread_line"]).to_csv(
        tmp_path / "data" / "processed" / "game_projections_2026-27.csv",
        index=False,
    )
    payload = export.payload(tmp_path, "2026-27")
    assert payload["games"] == []
    assert payload["may_bet"] is False
    health = export.health(tmp_path, "2026-27")
    assert health["state"] == "degraded"
    assert any("game_projections" in issue for issue in health["issues"])


def test_build_site_writes_json_beside_html(tmp_path):
    _seed(tmp_path)
    out = build_site(tmp_path, season="2026-27", out=tmp_path / "_site" / "index.html")
    assert out.exists()
    assert (tmp_path / "_site" / "board.json").exists()
    board = json.loads((tmp_path / "_site" / "board.json").read_text(encoding="utf-8"))
    assert board["schema"] == export.SCHEMA
    assert "BET" not in json.dumps({k: board[k] for k in ("authority", "may_bet", "schema")})
