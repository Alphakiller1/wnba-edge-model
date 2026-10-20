"""DraftKings game lines from ESPN's public scoreboard, shaped as Odds API events.

The Odds API key this model shares with the NFL, CFB and MLB models runs out
mid-month; on 2026-10-20 it had 2 credits, so the WNBA Finals went unpriced
while ESPN's scoreboard carried DraftKings' moneyline, spread and total for
every game. The other three models already read ESPN first; this gives the
WNBA model the same free source for game markets (player props have none).

Events come back in The Odds API's shape, so ``market_data._normalize_event``
turns them into the same rows a paid fetch would.
"""
from __future__ import annotations

from datetime import date, timedelta

from .http import HttpClient
from .schedule import _scoreboard_urls

BOOK_KEY = "draftkings"


def _price(value) -> int | None:
    text = str(value or "").strip().upper()
    if text == "EVEN":
        return 100
    try:
        return int(float(text.replace("+", "")))
    except ValueError:
        return None


def _point(value) -> float | None:
    text = str(value or "").strip().lower().lstrip("ou")
    try:
        return float(text)
    except ValueError:
        return None


def _close(block: dict | None) -> dict:
    block = block or {}
    return block.get("close") or block.get("open") or {}


def event_from_espn(event: dict) -> dict | None:
    """One ESPN scoreboard event -> an Odds API event with DraftKings markets, or
    None when the game has started or carries no DraftKings odds."""
    if ((event.get("status") or {}).get("type") or {}).get("state") != "pre":
        return None
    comp = (event.get("competitions") or [{}])[0]
    odds = next((o for o in comp.get("odds") or []
                 if str((o.get("provider") or {}).get("name") or "").lower() == BOOK_KEY), None)
    if not odds:
        return None
    teams = {c.get("homeAway"): (c.get("team") or {}).get("displayName") for c in comp.get("competitors") or []}
    away, home = teams.get("away"), teams.get("home")
    if not away or not home:
        return None
    markets = []
    moneyline = odds.get("moneyline") or {}
    prices = [(home, _price(_close(moneyline.get("home")).get("odds"))),
              (away, _price(_close(moneyline.get("away")).get("odds")))]
    if all(price is not None for _name, price in prices):
        markets.append({"key": "h2h", "outcomes": [{"name": n, "price": p} for n, p in prices]})
    spread = odds.get("pointSpread") or {}
    sides = [(home, _close(spread.get("home"))), (away, _close(spread.get("away")))]
    if all(_point(q.get("line")) is not None and _price(q.get("odds")) is not None for _n, q in sides):
        markets.append({"key": "spreads", "outcomes": [
            {"name": n, "price": _price(q.get("odds")), "point": _point(q.get("line"))} for n, q in sides]})
    total = odds.get("total") or {}
    sides = [("Over", _close(total.get("over"))), ("Under", _close(total.get("under")))]
    if all(_point(q.get("line")) is not None and _price(q.get("odds")) is not None for _n, q in sides):
        markets.append({"key": "totals", "outcomes": [
            {"name": n, "price": _price(q.get("odds")), "point": _point(q.get("line"))} for n, q in sides]})
    if not markets:
        return None
    return {
        "id": f"espn-{event.get('id')}",
        "commence_time": event.get("date", ""),
        "away_team": away,
        "home_team": home,
        "bookmakers": [{"key": BOOK_KEY, "markets": markets}],
    }


def fetch_events(days: int = 3, start: date | None = None,
                 client: HttpClient | None = None) -> list[dict]:
    """DraftKings-priced pre-game events for the next ``days`` days."""
    client = client or HttpClient(timeout=30, retries=3, pause_seconds=1.0)
    start = start or date.today()
    events: dict[str, dict] = {}
    for offset in range(days):
        stamp = (start + timedelta(days=offset)).strftime("%Y%m%d")
        for url in _scoreboard_urls():
            try:
                payload = client.get_json(f"{url}?dates={stamp}")
            except Exception:
                continue
            for raw in payload.get("events") or []:
                shaped = event_from_espn(raw)
                if shaped:
                    events[shaped["id"]] = shaped
            break
    return list(events.values())
