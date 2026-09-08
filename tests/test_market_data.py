import json

import pandas as pd
import pytest

from wnba_edges import cli, market_data
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
    monkeypatch.setattr(market_data, "_record_fetch_status", lambda *args, **kwargs: None)
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


def _valid_fanatics_rows(now):
    base = dict.fromkeys(COLUMNS, "")
    common = {
        **base, "fetched_at": now, "event_id": "evt", "away": "MIN",
        "home": "ATL", "book": "fanatics",
    }
    return [
        {**common, "market": "ml", "side": "MIN", "odds": "-110"},
        {**common, "market": "ml", "side": "ATL", "odds": "-110"},
        {**common, "market": "spread", "side": "MIN", "line": "2.5", "odds": "-110"},
        {**common, "market": "spread", "side": "ATL", "line": "-2.5", "odds": "-110"},
        {**common, "market": "total", "side": "over", "line": "165.5", "odds": "-110"},
        {**common, "market": "total", "side": "under", "line": "165.5", "odds": "-110"},
    ]


def test_validate_latest_snapshot_accepts_exact_paired_book(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    latest = tmp_path / "latest.csv"
    monkeypatch.setattr(market_data, "ODDS_LATEST_CSV", latest)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pd.DataFrame(_valid_fanatics_rows(now), columns=COLUMNS).to_csv(latest, index=False)

    summary = market_data.validate_latest_snapshot("fanatics")

    assert summary == {"rows": 6, "events": 1, "game_rows": 6, "prop_rows": 0}


def test_validate_latest_snapshot_rejects_empty_locked_board(monkeypatch, tmp_path):
    latest = tmp_path / "latest.csv"
    monkeypatch.setattr(market_data, "ODDS_LATEST_CSV", latest)
    monkeypatch.setattr(market_data, "ODDS_STATUS_JSON", tmp_path / "odds_status.json")
    pd.DataFrame(columns=COLUMNS).to_csv(latest, index=False)

    with pytest.raises(SystemExit, match="snapshot is empty"):
        market_data.validate_latest_snapshot("fanatics")


def test_odds_quota_error_is_recoverable():
    err = market_data._odds_http_error(
        401,
        '{"message":"Usage quota has been reached.","error_code":"OUT_OF_USAGE_CREDITS"}',
    )
    assert isinstance(err, market_data.OddsQuotaError)
    other = market_data._odds_http_error(401, '{"message":"Invalid API key"}')
    assert isinstance(other, SystemExit)


def test_fetch_slate_quota_miss_clears_locked_snapshot(monkeypatch, tmp_path):
    latest = tmp_path / "odds_latest.csv"
    status = tmp_path / "odds_status.json"
    monkeypatch.setattr(market_data, "ODDS_LATEST_CSV", latest)
    monkeypatch.setattr(market_data, "ODDS_HISTORY_CSV", tmp_path / "odds_history.csv")
    monkeypatch.setattr(market_data, "ODDS_DIR", tmp_path)
    monkeypatch.setattr(market_data, "ODDS_STATUS_JSON", status)
    monkeypatch.setenv("ODDS_BOOKMAKERS", "draftkings")
    pd.DataFrame(
        [{**dict.fromkeys(COLUMNS, ""), "book": "draftkings", "market": "ml", "odds": "-110"}],
        columns=COLUMNS,
    ).to_csv(latest, index=False)

    def _boom(path, params):
        market_data._LAST_USAGE["quota_exhausted"] = "1"
        raise market_data.OddsQuotaError("quota")

    monkeypatch.setattr(market_data, "_get", _boom)
    rows = market_data.fetch_slate(props=False)
    assert rows == []
    stored = pd.read_csv(latest)
    assert stored.empty
    recorded = json.loads(status.read_text())
    assert recorded["game"]["quota_exhausted"] is True


def test_validate_latest_allows_empty_after_quota_miss(monkeypatch, tmp_path):
    latest = tmp_path / "latest.csv"
    status = tmp_path / "odds_status.json"
    monkeypatch.setattr(market_data, "ODDS_LATEST_CSV", latest)
    monkeypatch.setattr(market_data, "ODDS_STATUS_JSON", status)
    pd.DataFrame(columns=COLUMNS).to_csv(latest, index=False)
    status.write_text('{"game": {"quota_exhausted": true}}', encoding="utf-8")
    summary = market_data.validate_latest_snapshot("draftkings", allow_quota_miss=True)
    assert summary["rows"] == 0


def test_validate_latest_snapshot_rejects_unpaired_line(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    latest = tmp_path / "latest.csv"
    monkeypatch.setattr(market_data, "ODDS_LATEST_CSV", latest)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = _valid_fanatics_rows(now)
    rows[-1]["line"] = "166.5"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(latest, index=False)

    with pytest.raises(SystemExit, match="total sides/lines do not pair"):
        market_data.validate_latest_snapshot("fanatics")


def test_projection_loader_uses_exact_latest_snapshot_not_history(monkeypatch, tmp_path):
    latest = tmp_path / "latest.csv"
    history = tmp_path / "history.csv"
    monkeypatch.setattr(market_data, "ODDS_LATEST_CSV", latest)
    monkeypatch.setattr(market_data, "ODDS_HISTORY_CSV", history)
    monkeypatch.setenv("ODDS_BOOKMAKERS", "draftkings")
    pd.DataFrame([{"book": "draftkings", "market": "total", "line": 177.5}]).to_csv(latest, index=False)
    pd.DataFrame([{"book": "draftkings", "market": "total", "line": 175.5}]).to_csv(history, index=False)

    current = cli._load_odds()
    grading = cli._load_odds(latest_only=False)

    assert list(current["line"]) == [177.5]
    assert sorted(grading["line"].tolist()) == [175.5, 177.5]
