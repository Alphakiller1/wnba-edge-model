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
        "Daily Best Bets",
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
    # Matchup drivers sit on one even row, not a leftover five-tile grid.
    assert "Ratings · pace · venue" not in html
    assert "Net rating" in html
    assert "Why MIN" in html
    # Empty states render for layers without data (odds + results).
    assert "No odds snapshot stored" in html
    assert "No graded predictions yet" in html
    assert "Prop projections" in html


def test_build_site_handles_missing_everything(tmp_path):
    out = build_site(tmp_path, season="2026-27", out=tmp_path / "docs" / "index.html")
    html = out.read_text(encoding="utf-8")
    assert "No priced sides on today's slate" in html
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
    assert group.state == ""
    assert len(group.tiles) == 3
    tiles = {tile.label: tile for tile in group.tiles}
    assert tiles["CHI scoring"].value == "104.0"
    assert tiles["CHI scoring"].state == "ORtg vs MIN 102.0 DRtg"
    assert tiles["MIN scoring"].value == "112.0"
    assert tiles["MIN scoring"].state == "ORtg vs CHI 109.0 DRtg"
    assert tiles["Net rating"].value == "MIN +15.0"
    assert "82.0 pace" in tiles["Net rating"].state
    assert "MIN +1.9 home" in tiles["Net rating"].state


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
    assert moneyline.value == "MIN 60%"
    assert moneyline.state.startswith("Book · -120")
    assert "edge" in moneyline.state
    assert spread.value == "MIN -4.0"
    assert "MIN -3.5" in spread.state
    assert "-110" in spread.state


def test_site_renders_prop_slate_and_player_props_filter(tmp_path):
    _seed(tmp_path)
    processed = tmp_path / "data" / "processed"
    pd.DataFrame(
        [
            {
                "run_id": "r1", "generated_at": "2026-07-16T12:00:00+00:00",
                "game_date": "2026-07-20", "away": "CHI", "home": "MIN", "team": "MIN",
                "player": "Napheesa Collier", "player_id": 9, "market": "player_points",
                "projection": 22.4, "projection_basis": "season PPG", "sigma": 6.0,
                "sigma_source": "league", "line": 21.5, "side": "over", "odds": -110,
                "opposite_odds": -110, "book": "draftkings", "model_prob": 0.58,
                "implied_prob": 0.5, "vig_free": True, "edge": 0.08, "tier": "Standard",
                "verdict": "PLAY", "priced": True,
            },
            {
                "run_id": "r1", "generated_at": "2026-07-16T12:00:00+00:00",
                "game_date": "2026-07-20", "away": "CHI", "home": "MIN", "team": "MIN",
                "player": "Napheesa Collier", "player_id": 9, "market": "player_rebounds",
                "projection": 9.1, "projection_basis": "season RPG", "sigma": 3.0,
                "sigma_source": "league", "line": "", "side": "", "odds": "",
                "opposite_odds": "", "book": "", "model_prob": "",
                "implied_prob": "", "vig_free": False, "edge": "", "tier": "",
                "verdict": "PROJ", "priced": False,
            },
        ]
    ).to_csv(processed / "prop_projections_2026-27.csv", index=False)
    html = build_site(tmp_path, season="2026-27", out=tmp_path / "docs" / "index.html").read_text(encoding="utf-8")
    assert "Player props" in html
    assert "Slate player-prop projections" in html
    assert "Napheesa Collier" in html
    assert "Collier PTS" in html


def test_site_renders_recorded_game_markets(tmp_path):
    _seed(tmp_path)
    processed = tmp_path / "data" / "processed"
    pd.DataFrame(
        [
            {
                "run_id": "r1", "generated_at": "2026-07-16T12:00:00+00:00",
                "game_date": "2026-07-20", "away": "CHI", "home": "MIN",
                "market": "moneyline", "projection": 0.78, "projection_basis": "logistic",
                "side": "MIN", "line": "", "odds": -150, "opposite_odds": 130,
                "book": "draftkings", "model_prob": 0.78, "implied_prob": 0.58,
                "vig_free": True, "edge": 0.20, "tier": "Standard", "verdict": "PLAY",
                "priced": True,
            },
            {
                "run_id": "r1", "generated_at": "2026-07-16T12:00:00+00:00",
                "game_date": "2026-07-20", "away": "CHI", "home": "MIN",
                "market": "spread", "projection": 10.0, "projection_basis": "projected home margin",
                "side": "MIN", "line": -4.5, "odds": -110, "opposite_odds": -110,
                "book": "draftkings", "model_prob": 0.70, "implied_prob": 0.52,
                "vig_free": True, "edge": 0.18, "tier": "Standard", "verdict": "PLAY",
                "priced": True,
            },
            {
                "run_id": "r1", "generated_at": "2026-07-16T12:00:00+00:00",
                "game_date": "2026-07-20", "away": "CHI", "home": "MIN",
                "market": "total", "projection": 166.0, "projection_basis": "projected total",
                "side": "OVER", "line": 160.5, "odds": -110, "opposite_odds": -110,
                "book": "draftkings", "model_prob": 0.68, "implied_prob": 0.52,
                "vig_free": True, "edge": 0.16, "tier": "Standard", "verdict": "PLAY",
                "priced": True,
            },
        ]
    ).to_csv(processed / "game_markets_2026-27.csv", index=False)
    html = build_site(tmp_path, season="2026-27", out=tmp_path / "docs" / "index.html").read_text(encoding="utf-8")
    assert "Recorded moneyline, spread and total" in html
    assert "Moneyline" in html
    assert "MIN vs -4.5" in html
    assert "OVER 160.5" in html


def test_site_hides_game_projection_audit_and_shows_lines(tmp_path):
    _seed(tmp_path)
    predictions = tmp_path / "data" / "predictions"
    predictions.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "prediction_id": "p1", "run_id": "r1", "sport": "wnba", "season": "2026-27",
                "recorded_at": "2026-07-16T12:00:00+00:00", "date": "2026-07-20",
                "away": "CHI", "home": "MIN",
                "projected_away_pts": 78.0, "projected_home_pts": 88.0,
                "projected_total": 166.0, "projected_home_spread": 10.0,
                "home_win_prob": 0.78, "win_prob_basis": "test",
                "book_total_line": 160.5, "book_spread_line": -4.5,
                "book_home_ml": -150, "book_away_ml": 130,
                "predicted_ml_side": "MIN", "predicted_spread_ats": "HOME",
                "predicted_total_side": "OVER",
                "settled": True, "actual_away_pts": 80, "actual_home_pts": 90,
                "actual_total": 170, "actual_winner": "MIN", "home_win": True,
                "winner_correct": True, "predicted_spread_side": "HOME",
                "spread_side_correct": True, "spread_ats_correct": True,
                "total_side_correct": True, "spread_error": 0.0, "total_error": -4.0,
                "ungraded_reason": "", "graded_at": "2026-07-21T00:00:00+00:00",
            }
        ]
    ).to_csv(predictions / "game_predictions.csv", index=False)
    html = build_site(tmp_path, season="2026-27", out=tmp_path / "docs" / "index.html").read_text(encoding="utf-8")
    assert "All game-projection audit rows" not in html
    assert "All moneyline / spread / total audit rows" not in html
    assert "All player-prop audit rows" not in html
    assert "Graded game calls" in html
    assert "166.0 vs 160.5" in html
    assert "+10.0 vs -4.5" in html
    assert "MIN 78% @ -150" in html
