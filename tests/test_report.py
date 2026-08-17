import pandas as pd

from wnba_edges.board_wnba import _full_game_group, _matchup_drivers
from wnba_edges.report import build_site


def _seed(root):
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "r1", "generated_at": "2026-07-16T12:00:00+00:00",
                "date": "2026-07-20", "time": "19:00 ET", "away": "CHI", "home": "MIN",
                "projected_away_pts": 78.0, "projected_home_pts": 88.0,
                "projected_total": 166.0, "projected_home_spread": 10.0,
                "projected_pace": 80.0, "away_net": -8.0, "home_net": 10.0,
                "home_win_prob": 0.78, "win_prob_basis": "logistic fit on 60 games",
                "home_court_pts": 1.9, "home_court_basis": "empirical (n=160)",
            }
        ]
    ).to_csv(processed / "game_projections_2026-27.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": "Good Sample", "team": "MIN", "pos": "G", "edge_score": 3.2,
                "watch_reason": "usage/creation", "ppg": 18.0, "mpg": 30.0,
                "gp": 20, "low_sample": False,
            },
            {
                "name": "Tiny Sample", "team": "ATL", "pos": "F", "edge_score": 9.9,
                "watch_reason": "LOW SAMPLE", "ppg": 1.0, "mpg": 1.0,
                "gp": 2, "low_sample": True,
            },
        ]
    ).to_csv(processed / "player_features_2026-27.csv", index=False)


def test_build_site_layers_disclaimer_and_low_sample_gate(tmp_path):
    _seed(tmp_path)
    out = build_site(tmp_path, season="2026-27", out=tmp_path / "docs" / "index.html")
    html = out.read_text(encoding="utf-8")
    for marker in (
        "Game Projections",
        "Market Snapshot",
        "Stale Anchor Board",
        "Graded Results",
        "How to read this page",
        "not betting advice",
        "1-800-GAMBLER",
        "logistic fit on 60 games",
    ):
        assert marker in html, marker
    # Low-sample player is excluded from the public board.
    assert "Tiny Sample" not in html
    assert "Good Sample" in html
    # Empty states render for layers without data (odds + results).
    assert "No odds snapshot stored" in html
    assert "No graded predictions yet" in html


def test_build_site_handles_missing_everything(tmp_path):
    out = build_site(tmp_path, season="2026-27", out=tmp_path / "docs" / "index.html")
    html = out.read_text(encoding="utf-8")
    assert "No game projections yet" in html
    assert "No feature board yet" in html


def test_matchup_breakdown_explains_the_projection_inputs():
    group = _matchup_drivers(
        pd.Series(
            {
                "away": "CHI", "home": "MIN", "home_win_prob": 0.72,
                "away_ortg": 104.0, "away_drtg": 109.0,
                "home_ortg": 112.0, "home_drtg": 102.0,
                "away_net": -5.0, "home_net": 10.0,
                "projected_pace": 82.0, "home_court_pts": 1.9,
            }
        ),
        pace_reference=80.0,
    )
    assert group is not None
    assert group.label == "Why MIN"
    tiles = {tile.label: tile for tile in group.tiles}
    assert tiles["CHI scoring"].state == "vs MIN 102.0 DRtg"
    assert tiles["MIN scoring"].state == "vs CHI 109.0 DRtg"
    assert tiles["Net edge"].value == "MIN +15.0"
    assert tiles["Home boost"].value == "+1.9"


def test_full_game_group_uses_normalized_odds_snapshot_markets():
    game = pd.Series(
        {
            "away": "CHI", "home": "MIN", "home_win_prob": 0.60,
            "projected_home_spread": 4.0, "projected_total": 166.0,
        }
    )
    quotes = pd.DataFrame(
        [
            {"market": "ml", "side": "MIN", "odds": -120},
            {"market": "ml", "side": "CHI", "odds": 100},
            {"market": "spread", "side": "MIN", "line": -3.5, "odds": -110},
            {"market": "spread", "side": "CHI", "line": 3.5, "odds": -110},
            {"market": "total", "side": "over", "line": 165.5, "odds": -110},
            {"market": "total", "side": "under", "line": 165.5, "odds": -110},
        ]
    )
    group = _full_game_group(game, quotes)
    assert group.priced == 3
    assert all(tile.is_priced for tile in group.tiles)
    moneyline = next(tile for tile in group.tiles if tile.label == "Moneyline")
    spread = next(tile for tile in group.tiles if tile.label == "Spread")
    assert moneyline.value == "Model MIN 60%"
    assert "Best book: MIN ML @ -120" in moneyline.state
    assert spread.value == "Model MIN -4.0"
    assert "Best book: MIN -3.5 @ -110" in spread.state
