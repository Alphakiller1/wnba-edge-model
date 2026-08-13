import pandas as pd

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
