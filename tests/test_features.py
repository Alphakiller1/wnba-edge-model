import pandas as pd

from wnba_edges.features import board_eligible, build_player_features, zscore


def _players():
    rows = []
    # 12 established rotation players.
    for i in range(12):
        rows.append(
            {
                "id": i,
                "name": f"Starter {i}",
                "team": "MIN",
                "pos": "G",
                "season": "2026-27",
                "gp": 20,
                "mpg": 28.0,
                "ppg": 12.0 + i * 0.5,
                "rpg": 5.0,
                "apg": 3.0,
                "usg": 20.0 + i * 0.4,
                "projectedMinutes": 28.0,
                "projectedPoints": 12.0 + i * 0.5,
                "projectionConfidence": 80,
            }
        )
    # A 1-minute-per-game player with an absurd small-sample usage rate.
    rows.append(
        {
            "id": 99,
            "name": "Tiny Sample",
            "team": "ATL",
            "pos": "F",
            "season": "2026-27",
            "gp": 2,
            "mpg": 1.0,
            "ppg": 1.0,
            "rpg": 1.0,
            "apg": 0.0,
            "usg": 65.0,
            "projectedMinutes": 2.0,
            "projectedPoints": 1.0,
            "projectionConfidence": 20,
        }
    )
    return pd.DataFrame(rows)


def test_low_sample_player_flagged_and_not_top():
    out = build_player_features(_players())
    tiny = out[out["name"] == "Tiny Sample"].iloc[0]
    assert bool(tiny["low_sample"]) is True
    assert "LOW SAMPLE" in tiny["watch_reason"]
    # Shrinkage keeps the noise rate out of the top of the board.
    top10 = out.head(10)["name"].tolist()
    assert "Tiny Sample" not in top10


def test_board_eligible_excludes_low_sample():
    out = build_player_features(_players())
    eligible = board_eligible(out)
    assert "Tiny Sample" not in eligible["name"].tolist()
    assert len(eligible) == len(out) - 1


def test_zscore_degenerate_series_is_zero():
    flat = pd.Series([3.0, 3.0, 3.0])
    assert zscore(flat).abs().sum() == 0
    empty = pd.Series([None, None], dtype="float64")
    assert zscore(empty).abs().sum() == 0
