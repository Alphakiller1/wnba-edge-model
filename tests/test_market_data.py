import pandas as pd

from wnba_edges.market_data import filter_odds_to_requested_books, odds_scope_params


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


def test_a_book_locked_pull_asks_for_the_book_instead_of_the_region(monkeypatch):
    """The Odds API bills markets x regions, and bookmakers stands in for regions.

    Sending both would charge the full region rate for quotes we then filter away.
    """
    monkeypatch.setenv("ODDS_BOOKMAKERS", "fanduel")
    assert odds_scope_params() == {"bookmakers": "fanduel"}


def test_an_unlocked_pull_still_asks_by_region(monkeypatch):
    monkeypatch.delenv("ODDS_BOOKMAKERS", raising=False)
    assert set(odds_scope_params()) == {"regions"}
