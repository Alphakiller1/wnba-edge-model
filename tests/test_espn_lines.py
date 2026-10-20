from wnba_edges import espn_lines, market_data


def _espn_event(state="pre", provider="DraftKings"):
    """WNBA Finals G3 as ESPN served it on 2026-10-20."""
    return {
        "id": "401918307",
        "date": "2026-10-22T00:00Z",
        "status": {"type": {"state": state}},
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"displayName": "Atlanta Dream"}},
                {"homeAway": "away", "team": {"displayName": "Golden State Valkyries"}},
            ],
            "odds": [{
                "provider": {"name": provider},
                "moneyline": {"home": {"close": {"odds": "-162"}, "open": {"odds": "-148"}},
                              "away": {"close": {"odds": "+136"}}},
                "pointSpread": {"home": {"close": {"line": "-3.5", "odds": "-110"}},
                                "away": {"close": {"line": "+3.5", "odds": "-110"}}},
                "total": {"over": {"close": {"line": "o151.5", "odds": "-108"}},
                          "under": {"close": {"line": "u151.5", "odds": "-112"}}},
            }],
        }],
    }


def test_espn_draftkings_odds_become_odds_api_rows():
    event = espn_lines.event_from_espn(_espn_event())
    rows = market_data._normalize_event(event, "2026-10-20T22:00:00+00:00")
    by = {(r["market"], r["side"]): r for r in rows}
    assert by[("ml", "ATL")]["odds"] == -162 and by[("ml", "GSV")]["odds"] == 136
    assert by[("spread", "ATL")]["line"] == -3.5 and by[("spread", "GSV")]["line"] == 3.5
    assert by[("total", "over")]["line"] == 151.5 and by[("total", "under")]["odds"] == -112
    assert {r["book"] for r in rows} == {"draftkings"}
    assert {(r["away"], r["home"]) for r in rows} == {("GSV", "ATL")}


def test_started_games_and_other_books_are_skipped():
    assert espn_lines.event_from_espn(_espn_event(state="in")) is None
    assert espn_lines.event_from_espn(_espn_event(provider="ESPN BET")) is None


def test_espn_fills_a_quota_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(market_data, "ODDS_DIR", tmp_path)
    monkeypatch.setattr(market_data, "ODDS_STATUS_JSON", tmp_path / "odds_status.json")
    monkeypatch.setattr(market_data, "requested_bookmakers", lambda: {"draftkings"})
    stored = []
    monkeypatch.setattr(market_data, "store", lambda rows, **kw: stored.append(rows))
    monkeypatch.setattr(espn_lines, "fetch_events",
                        lambda **kw: [espn_lines.event_from_espn(_espn_event())])

    def quota(*_a, **_k):
        raise market_data.OddsQuotaError("out of credits")
    monkeypatch.setattr(market_data, "_get", quota)
    rows = market_data.fetch_slate(props=False)
    assert len(rows) == 6 and stored and stored[0] == rows
    status = (tmp_path / "odds_status.json").read_text(encoding="utf-8")
    assert '"source": "espn"' in status and '"quotes": 6' in status


def test_a_board_locked_to_another_book_does_not_take_espn(monkeypatch):
    monkeypatch.setattr(market_data, "requested_bookmakers", lambda: {"fanduel"})
    assert market_data._espn_game_rows("2026-10-20T22:00:00+00:00") == []
