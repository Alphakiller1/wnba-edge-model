import pandas as pd

from wnba_edges import market_data
from wnba_edges.market_data import COLUMNS, filter_odds_to_requested_books


def _quotes():
    return pd.DataFrame(
        [
            {"book": "fanatics", "market": "ml", "odds": -140},
            {"book": "fanduel", "market": "ml", "odds": -150},
            {"book": "draftkings", "market": "ml", "odds": -145},
        ]
    )


def test_filter_odds_passes_through_when_unlocked(monkeypatch):
    monkeypatch.delenv("ODDS_BOOKMAKERS", raising=False)
    out = filter_odds_to_requested_books(_quotes())
    assert len(out) == 3


def test_filter_odds_keeps_only_requested_book(monkeypatch):
    monkeypatch.setenv("ODDS_BOOKMAKERS", "fanatics")
    out = filter_odds_to_requested_books(_quotes())
    assert list(out["book"]) == ["fanatics"]


def test_filter_odds_does_not_fall_back_to_other_books(monkeypatch):
    monkeypatch.setenv("ODDS_BOOKMAKERS", "fanatics")
    other = pd.DataFrame([{"book": "fanduel", "market": "ml", "odds": -110}])
    out = filter_odds_to_requested_books(other)
    assert out.empty


def test_prop_slate_fetch_requests_only_modeled_prop_markets(monkeypatch):
    calls = []
    monkeypatch.setattr(market_data, "list_events", lambda: {("A", "B"): "event-1"})
    monkeypatch.setattr(market_data, "store", lambda rows, **kwargs: calls.append((rows, kwargs)))
    monkeypatch.setattr(
        market_data,
        "fetch_event_odds",
        lambda event_id, **kwargs: [
            {**dict.fromkeys(COLUMNS, ""), "event_id": event_id, "market": "player_points"}
        ] if kwargs == {"props_only": True} else [],
    )

    rows = market_data.fetch_slate(props=True)

    assert len(rows) == 1
    assert calls[0][1]["replace_markets"] == {
        "player_points", "player_rebounds", "player_assists", "player_threes",
    }


def test_prop_store_preserves_game_lines(monkeypatch, tmp_path):
    latest = tmp_path / "odds_latest.csv"
    history = tmp_path / "odds_history.csv"
    monkeypatch.setattr(market_data, "ODDS_LATEST_CSV", latest)
    monkeypatch.setattr(market_data, "ODDS_HISTORY_CSV", history)
    monkeypatch.setattr(market_data, "ODDS_DIR", tmp_path)
    base = dict.fromkeys(COLUMNS, "")
    pd.DataFrame(
        [
            {**base, "away": "NYL", "home": "SEA", "market": "ml", "side": "SEA"},
            {**base, "away": "NYL", "home": "SEA", "market": "player_points", "player": "Old"},
        ],
        columns=COLUMNS,
    ).to_csv(latest, index=False)

    market_data.store(
        [{**base, "away": "NYL", "home": "SEA", "market": "player_points", "player": "New"}],
        replace_markets={"player_points", "player_rebounds", "player_assists", "player_threes"},
    )

    stored = pd.read_csv(latest, dtype=str).fillna("")
    assert list(stored["market"]) == ["ml", "player_points"]
    assert list(stored["player"]) == ["", "New"]
