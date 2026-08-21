from datetime import date

import pandas as pd

from wnba_edges.best_bets import (
    build_market_history,
    hit_likelihood,
    probability_band,
    rank_best_bets,
    resolve_slate_date,
)


def test_hit_likelihood_shrinks_tiny_samples_toward_the_model():
    # 1-0 must not print as a lock; the 12-game prior keeps history from dominating.
    one_and_oh = hit_likelihood(0.60, wins=1, losses=0)
    assert 0.55 < one_and_oh < 0.70
    # A long 80% record pulls an 80% model number toward ~80%, not 50%.
    long = hit_likelihood(0.80, wins=40, losses=10)
    assert 0.76 < long < 0.82
    # Laplace 0-2 is below 50% after shrink, so a 70% model is pulled down.
    cold = hit_likelihood(0.70, wins=2, losses=8)
    assert cold < 0.70


def test_probability_band_edges():
    assert probability_band(0.55) == "50–59%"
    assert probability_band(0.60) == "60–69%"
    assert probability_band(0.72) == "70%+"


def test_resolve_slate_date_prefers_today_then_next():
    markets = pd.DataFrame(
        {
            "priced": [True, True, False],
            "game_date": ["2026-08-21", "2026-08-22", "2026-08-21"],
        }
    )
    props = pd.DataFrame(columns=["priced", "game_date"])
    assert resolve_slate_date(markets, props, today=date(2026, 8, 21)) == "2026-08-21"
    assert resolve_slate_date(markets, props, today=date(2026, 8, 20)) == "2026-08-21"
    assert resolve_slate_date(markets, props, today=date(2026, 8, 23)) == "2026-08-22"


def _game_row(**overrides):
    row = {
        "run_id": "r1",
        "generated_at": "2026-08-21T16:00:00+00:00",
        "game_date": "2026-08-21",
        "away": "MIN",
        "home": "WAS",
        "market": "moneyline",
        "projection": 0.80,
        "projection_basis": "test",
        "side": "MIN",
        "line": "",
        "odds": -135,
        "opposite_odds": 185,
        "book": "draftkings",
        "model_prob": 0.80,
        "implied_prob": 0.62,
        "vig_free": True,
        "edge": 0.18,
        "tier": "Strong",
        "verdict": "PLAY",
        "priced": True,
    }
    row.update(overrides)
    return row


def _prop_row(**overrides):
    row = {
        "run_id": "r1",
        "generated_at": "2026-08-21T16:00:00+00:00",
        "game_date": "2026-08-21",
        "away": "MIN",
        "home": "WAS",
        "team": "MIN",
        "player": "Napheesa Collier",
        "player_id": 1,
        "market": "player_points",
        "projection": 24.0,
        "projection_basis": "test",
        "sigma": 5.0,
        "sigma_source": "test",
        "line": 22.5,
        "side": "over",
        "odds": -110,
        "opposite_odds": -110,
        "book": "fanduel",
        "model_prob": 0.58,
        "implied_prob": 0.50,
        "vig_free": True,
        "edge": 0.08,
        "tier": "Strong",
        "verdict": "PLAY",
        "priced": True,
    }
    row.update(overrides)
    return row


def _graded_game(**overrides):
    row = {
        "prediction_id": "g1",
        "run_id": "old",
        "sport": "wnba",
        "season": "2026-27",
        "recorded_at": "2026-08-20T16:00:00+00:00",
        "date": "2026-08-20",
        "away": "IND",
        "home": "DAL",
        "projected_away_pts": 88,
        "projected_home_pts": 90,
        "projected_total": 178,
        "projected_home_spread": 2,
        "home_win_prob": 0.22,
        "win_prob_basis": "test",
        "book_total_line": 184.5,
        "book_spread_line": 3.0,
        "book_home_ml": 120,
        "book_away_ml": -140,
        "predicted_ml_side": "IND",
        "predicted_spread_ats": "DAL",
        "predicted_total_side": "UNDER",
        "settled": True,
        "actual_away_pts": 85,
        "actual_home_pts": 91,
        "actual_total": 176,
        "actual_winner": "DAL",
        "home_win": True,
        "winner_correct": True,
        "predicted_spread_side": "DAL",
        "spread_side_correct": True,
        "spread_ats_correct": True,
        "total_side_correct": True,
        "spread_error": 1.0,
        "total_error": 2.0,
        "ungraded_reason": "",
        "graded_at": "2026-08-21T00:00:00+00:00",
    }
    row.update(overrides)
    return row


def _graded_prop(**overrides):
    row = {
        "prediction_id": "p1",
        "run_id": "old",
        "sport": "wnba",
        "season": "2026-27",
        "recorded_at": "2026-08-18T16:00:00+00:00",
        "game_date": "2026-08-18",
        "player": "A'ja Wilson",
        "player_id": 2,
        "market": "player_points",
        "side": "over",
        "line": 24.5,
        "odds": -110,
        "opposite_odds": -110,
        "odds_source": "dk",
        "quote_age_hours": 1,
        "projection": 26.0,
        "projection_basis": "test",
        "sigma": 5,
        "sigma_source": "test",
        "model_prob": 0.58,
        "implied_prob": 0.5,
        "vig_free": True,
        "edge": 0.08,
        "ev_per_unit": 0.05,
        "tier": "Strong",
        "verdict": "PLAY",
        "settled": True,
        "won": False,
        "push": False,
        "actual": 20,
        "ungraded_reason": "",
        "graded_at": "2026-08-19T00:00:00+00:00",
    }
    row.update(overrides)
    return row


def test_rank_prefers_historically_strong_markets_and_excludes_unpriced():
    markets = pd.DataFrame(
        [
            _game_row(market="moneyline", side="MIN", model_prob=0.80, edge=0.18),
            _game_row(
                market="spread", side="MIN", line=2.5, model_prob=0.80, edge=0.18,
                away="MIN", home="WAS",
            ),
            _game_row(
                market="moneyline", side="DAL", model_prob=0.80, priced=False,
                away="IND", home="DAL", game_date="2026-08-21",
            ),
            _game_row(
                market="moneyline", side="GSV", model_prob=0.78, away="GSV", home="CHI",
            ),
        ]
    )
    props = pd.DataFrame(
        [
            _prop_row(model_prob=0.90, edge=0.20),
            _prop_row(
                player="Arike Ogunbowale", market="player_points", model_prob=0.51,
                line=18.5,
            ),
        ]
    )
    games_log = pd.DataFrame(
        [_graded_game(date=f"2026-08-{day:02d}", away="AAA", home=f"H{day}") for day in range(1, 21)]
        + [
            _graded_game(
                date=f"2026-07-{day:02d}", away="BBB", home=f"S{day}",
                spread_ats_correct=False, winner_correct=True,
            )
            for day in range(1, 6)
        ]
    )
    # Points props: historically 2-8, so a 90% model_prob still ranks below ML.
    props_log = pd.DataFrame(
        [_graded_prop(player=f"P{i}", won=(i < 2), game_date=f"2026-08-{i + 1:02d}") for i in range(10)]
    )
    ranked = rank_best_bets(
        markets, props, games_log, props_log, slate_date="2026-08-21", top=10,
    )
    assert list(ranked["selection"])[0].startswith("MIN ML") or list(ranked["selection"])[0].startswith("GSV ML")
    assert "DAL ML" not in set(ranked["selection"])
    # 0.51 model_prob is below the 0.52 floor.
    assert not ranked["selection"].str.contains("Arike").any()
    # Moneyline should outrank the historically cold points prop even at 90% model_prob.
    families = list(ranked["family"])
    assert families[0] == "moneyline"
    assert ranked.iloc[0]["hit_likelihood"] >= ranked.loc[ranked["family"] == "prop", "hit_likelihood"].max()


def test_tomorrow_is_not_mixed_into_today():
    markets = pd.DataFrame(
        [
            _game_row(game_date="2026-08-21", side="MIN", model_prob=0.70),
            _game_row(
                game_date="2026-08-22", away="ATL", home="PHX", side="ATL",
                model_prob=0.99, edge=0.40,
            ),
        ]
    )
    ranked = rank_best_bets(
        markets, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        slate_date="2026-08-21", top=10,
    )
    assert set(ranked["game_date"]) == {"2026-08-21"}
    assert "ATL ML" not in set(ranked["selection"])


def test_diversity_cap_limits_one_family():
    markets = pd.DataFrame(
        [
            _game_row(away="A", home=f"H{i}", side="A", model_prob=0.80 + i / 100)
            for i in range(8)
        ]
    )
    ranked = rank_best_bets(
        markets, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        slate_date="2026-08-21", top=10,
    )
    assert len(ranked) == 4
    assert set(ranked["family"]) == {"moneyline"}


def test_history_uses_latest_run_per_matchup():
    games = pd.DataFrame(
        [
            _graded_game(run_id="old", recorded_at="2026-08-19T10:00:00+00:00", winner_correct=False),
            _graded_game(run_id="new", recorded_at="2026-08-20T10:00:00+00:00", winner_correct=True),
        ]
    )
    history = build_market_history(games, pd.DataFrame())
    stats = history.lookup("moneyline", 0.78)
    assert stats["wins"] == 1
    assert stats["losses"] == 0
    assert stats["n"] == 1
