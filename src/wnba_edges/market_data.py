from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .paths import repository_root
from .teams import TEAM_NAME_TO_ABBR, team_abbr  # noqa: F401 — re-exported for compat

ROOT = repository_root(__file__)
DATA_DIR = ROOT / "data"
ODDS_DIR = DATA_DIR / "odds"
ODDS_LATEST_CSV = ODDS_DIR / "odds_latest.csv"
ODDS_HISTORY_CSV = ODDS_DIR / "odds_history.csv"
ODDS_STATUS_JSON = ODDS_DIR / "odds_status.json"

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_SPORT_KEY = os.getenv("WNBA_ODDS_SPORT_KEY", "basketball_wnba")
ODDS_REGIONS = os.getenv("ODDS_REGIONS", "us")
ODDS_FORMAT = "american"


def _bookmakers() -> str:
    """Live ODDS_BOOKMAKERS env (e.g. fanatics). Empty means any book."""
    return os.getenv("ODDS_BOOKMAKERS", "").strip()


def requested_bookmakers() -> set[str]:
    """Odds API book keys the caller locked this run to (empty = any book)."""
    return {part.strip().lower() for part in _bookmakers().split(",") if part.strip()}


def filter_odds_to_requested_books(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Drop quotes from books that were not requested.

    A Fanatics-only pull must not be priced against leftover DraftKings / FanDuel
    rows sitting in odds_history.csv. Empty after the filter means unpriced — we
    do not fall back to other books.
    """
    if frame is None or frame.empty:
        return frame
    allowed = requested_bookmakers()
    if not allowed or "book" not in frame.columns:
        return frame
    filtered = frame[frame["book"].astype(str).str.lower().isin(allowed)].copy()
    return filtered
ODDS_GAME_MARKETS = "h2h,spreads,totals"
ODDS_PROP_MARKETS = os.getenv(
    "WNBA_PROP_MARKETS",
    "player_points,player_rebounds,player_assists,player_threes",
)

# Quotes older than this are refused unless the caller explicitly allows stale
# prices; a fresh model against days-old odds manufactures phantom edges.
MAX_QUOTE_AGE_HOURS = 12.0

COLUMNS = [
    "fetched_at",
    "commence_time",
    "event_id",
    "away",
    "home",
    "book",
    "market",
    "side",
    "line",
    "odds",
    "player",
]

_LAST_USAGE: dict[str, str] = {}


class OddsQuotaError(RuntimeError):
    """The Odds API key has no remaining usage credits."""


def _odds_http_error(code: int, body: str) -> Exception:
    """Map Odds API HTTP failures. Quota misses are recoverable; other errors abort."""
    if code in {401, 429} and "OUT_OF_USAGE_CREDITS" in body:
        return OddsQuotaError(f"Odds API quota exhausted ({code}): {body}")
    return SystemExit(f"Odds API error {code}: {body}")


def _get(path: str, params: dict[str, Any]) -> Any:
    api_key = os.getenv("ODDS_API_KEY", "") or ODDS_API_KEY
    if not api_key:
        raise SystemExit(
            "No ODDS_API_KEY set. Get a key from The Odds API, then set "
            '$env:ODDS_API_KEY = "your_key" in PowerShell.'
        )
    params = {**params, "apiKey": api_key}
    url = f"{ODDS_API_BASE}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            _LAST_USAGE["remaining"] = response.headers.get("x-requests-remaining", "?")
            _LAST_USAGE["used"] = response.headers.get("x-requests-used", "?")
            _LAST_USAGE.pop("quota_exhausted", None)
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        error = _odds_http_error(exc.code, body)
        if isinstance(error, OddsQuotaError):
            _LAST_USAGE["quota_exhausted"] = "1"
        raise error from exc


def list_events() -> dict[tuple[str, str], str]:
    data = _get(f"/sports/{ODDS_SPORT_KEY}/events", {})
    out: dict[tuple[str, str], str] = {}
    for event in data:
        away = team_abbr(event.get("away_team", ""))
        home = team_abbr(event.get("home_team", ""))
        out[(away, home)] = event.get("id", "")
    return out


def fetch_event_odds(
    event_id: str,
    props: bool = False,
    *,
    props_only: bool = False,
) -> list[dict]:
    """Fetch one event without paying twice for markets already in the snapshot."""
    markets = ODDS_PROP_MARKETS if props_only else ODDS_GAME_MARKETS + ("," + ODDS_PROP_MARKETS if props else "")
    params = {"regions": ODDS_REGIONS, "markets": markets, "oddsFormat": ODDS_FORMAT}
    if _bookmakers():
        params["bookmakers"] = _bookmakers()
    event = _get(f"/sports/{ODDS_SPORT_KEY}/events/{event_id}/odds", params)
    return _normalize_event(event, datetime.now(timezone.utc).isoformat(timespec="seconds"))


def fetch_slate(*, props: bool = False) -> list[dict]:
    """Pull every live WNBA board line in one snapshot.

    Game markets (ML / spread / total) come from the bulk odds endpoint — one request
    for the whole slate. Player props are per-event and optional because they burn
    quota much faster.
    """
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if props:
        try:
            events = list_events()
        except OddsQuotaError as exc:
            print(f"WARNING: {exc}")
            return _quota_miss_snapshot("props", fetched_at)
        rows: list[dict] = []
        failed_events = 0
        for event_id in events.values():
            try:
                # Player props are billed per event and market. Game lines came from
                # the bulk endpoint already, so requesting them here wastes quota.
                rows.extend(fetch_event_odds(event_id, props_only=True))
            except OddsQuotaError as exc:
                failed_events += 1
                print(f"WARNING: prop fetch stopped after quota miss: {exc}")
                break
            except SystemExit as exc:
                # Keep quotes from successful calls if quota/network failure happens
                # part-way through the slate. The old implementation discarded them.
                failed_events += 1
                print(f"WARNING: prop fetch stopped after a failed event: {exc}")
                break
        locked = requested_bookmakers()
        _record_fetch_status("props", fetched_at, len(events), rows)
        if locked:
            books = sorted({str(row.get("book") or "").lower() for row in rows if row.get("book")})
            print(
                f"book-locked fetch ({', '.join(sorted(locked))}): "
                f"{len(events)} event(s), {len(rows)} quote(s), books={books or ['(none)']}"
            )
        if not rows:
            print("No live WNBA lines returned.")
            if _LAST_USAGE:
                print(f"API quota: used {_LAST_USAGE.get('used')}, remaining {_LAST_USAGE.get('remaining')}.")
            return rows
        store(rows, replace_markets=_market_names(ODDS_PROP_MARKETS))
        prop_games = len({row.get("event_id") for row in rows if str(row.get("market", "")).startswith("player_")})
        print(f"player-prop coverage: {prop_games}/{len(events)} event(s)" + (f"; {failed_events} failed" if failed_events else ""))
    else:
        params = {"regions": ODDS_REGIONS, "markets": ODDS_GAME_MARKETS, "oddsFormat": ODDS_FORMAT}
        if _bookmakers():
            params["bookmakers"] = _bookmakers()
        try:
            payload = _get(f"/sports/{ODDS_SPORT_KEY}/odds", params)
        except OddsQuotaError as exc:
            print(f"WARNING: {exc}")
            return _quota_miss_snapshot("game", fetched_at)
        if not isinstance(payload, list):
            raise SystemExit(f"Unexpected Odds API payload: {type(payload).__name__}")
        rows = []
        for event in payload:
            rows.extend(_normalize_event(event, fetched_at))
        _record_fetch_status("game", fetched_at, len(payload), rows)
        locked = requested_bookmakers()
        if locked:
            books = sorted({str(row.get("book") or "").lower() for row in rows if row.get("book")})
            print(
                f"book-locked fetch ({', '.join(sorted(locked))}): "
                f"{len(payload)} event(s), {len(rows)} quote(s), books={books or ['(none)']}"
            )
        if not rows:
            print("No live WNBA lines returned.")
            # A locked-book miss must not keep yesterday's other-book snapshot as "latest".
            if locked:
                store(rows, replace_latest=True)
            return rows
        store(rows, replace_latest=True)
    if _LAST_USAGE:
        print(f"API quota: used {_LAST_USAGE.get('used')}, remaining {_LAST_USAGE.get('remaining')}.")
    return rows


def _quota_exhausted() -> bool:
    return _LAST_USAGE.get("quota_exhausted") == "1"


def quota_miss_recorded() -> bool:
    """True when the latest fetch sidecar says The Odds API was out of credits."""
    try:
        status = json.loads(ODDS_STATUS_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    for scope in ("game", "props"):
        block = status.get(scope) or {}
        if block.get("quota_exhausted"):
            return True
    return False


def _quota_miss_snapshot(scope: str, fetched_at: str) -> list[dict]:
    """Empty locked board: do not keep yesterday's quotes or mix other books."""
    print(
        "Odds API quota exhausted. Publishing an unpriced locked-book snapshot "
        "instead of aborting grading or mixing other books."
    )
    rows: list[dict] = []
    _record_fetch_status(scope, fetched_at, 0, rows)
    if requested_bookmakers():
        store(rows, replace_latest=True)
    return rows


def _record_fetch_status(scope: str, fetched_at: str, events: int, rows: list[dict]) -> None:
    """Persist the requested book even when it returns zero quotes.

    An empty CSV cannot say whether no fetch ran or a locked sportsbook returned
    no board. This sidecar lets the dashboard state the latter precisely.
    """
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        status = json.loads(ODDS_STATUS_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        status = {}
    status["requested_bookmakers"] = sorted(requested_bookmakers())
    status[scope] = {
        "fetched_at": fetched_at,
        "events": int(events),
        "quotes": len(rows),
        "returned_books": sorted({str(row.get("book") or "").lower() for row in rows if row.get("book")}),
        "api_used": _LAST_USAGE.get("used"),
        "api_remaining": _LAST_USAGE.get("remaining"),
        "quota_exhausted": _quota_exhausted(),
    }
    ODDS_STATUS_JSON.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def fetch_game(away: str, home: str, props: bool = False) -> list[dict]:
    away, home = away.upper(), home.upper()
    event_id = list_events().get((away, home))
    if not event_id:
        print(f"{away}@{home} not on the live WNBA board.")
        return []
    rows = fetch_event_odds(event_id, props=props)
    store(rows)
    if _LAST_USAGE:
        print(f"API quota: used {_LAST_USAGE.get('used')}, remaining {_LAST_USAGE.get('remaining')}.")
    return rows


def _normalize_event(event: dict, fetched_at: str) -> list[dict]:
    away = team_abbr(event.get("away_team", ""))
    home = team_abbr(event.get("home_team", ""))
    base = {
        "fetched_at": fetched_at,
        "commence_time": event.get("commence_time", ""),
        "event_id": event.get("id", ""),
        "away": away,
        "home": home,
    }
    rows: list[dict] = []
    for book in event.get("bookmakers", []):
        book_key = book.get("key", "")
        for market in book.get("markets", []):
            market_key = market.get("key", "")
            for outcome in market.get("outcomes", []):
                rows.append(_normalize_outcome(base, book_key, market_key, outcome))
    return rows


def _normalize_outcome(base: dict, book: str, market_key: str, outcome: dict) -> dict:
    name = str(outcome.get("name", ""))
    description = str(outcome.get("description", ""))
    line = outcome.get("point", "")
    odds = outcome.get("price", "")
    player = ""

    if market_key == "h2h":
        market, side = "ml", team_abbr(name)
    elif market_key == "spreads":
        market, side = "spread", team_abbr(name)
    elif market_key == "totals":
        market, side = "total", name.lower()
    elif market_key.startswith("player_"):
        market = market_key
        player = description
        side = f"{description}|{name.lower()}"
    else:
        market, side = market_key, name

    return {
        **base,
        "book": book,
        "market": market,
        "side": side,
        "line": line,
        "odds": odds,
        "player": player,
    }


def _market_names(markets: str) -> set[str]:
    return {market.strip() for market in markets.split(",") if market.strip()}


def store(
    rows: list[dict],
    *,
    replace_latest: bool = False,
    replace_markets: set[str] | None = None,
) -> None:
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if rows:
        if ODDS_HISTORY_CSV.exists():
            frame.to_csv(ODDS_HISTORY_CSV, mode="a", header=False, index=False)
        else:
            frame.to_csv(ODDS_HISTORY_CSV, index=False)
    elif not replace_latest:
        return

    if replace_latest or not ODDS_LATEST_CSV.exists():
        latest = frame.astype(str)
    else:
        fetched_games = set(zip(frame["away"], frame["home"]))
        previous = pd.read_csv(ODDS_LATEST_CSV, dtype=str).fillna("")
        same_game = previous.apply(lambda row: (row["away"], row["home"]) in fetched_games, axis=1)
        if replace_markets:
            replaced_market = previous["market"].isin(replace_markets)
            keep = previous[~(same_game & replaced_market)]
        else:
            keep = previous[~same_game]
        latest = pd.concat([keep, frame.astype(str)], ignore_index=True)
    latest.to_csv(ODDS_LATEST_CSV, index=False)
    games = 0 if frame.empty or "away" not in frame.columns else int(frame.groupby(["away", "home"]).ngroups)
    print(f"Stored {len(frame)} WNBA odds rows across {games} game(s).")


def validate_latest_snapshot(
    book: str,
    *,
    max_age_hours: float = MAX_QUOTE_AGE_HOURS,
    allow_quota_miss: bool = False,
) -> dict[str, int]:
    """Fail closed unless the latest snapshot is a fresh, internally paired book board.

    This validates the exact rows the projection engine will load. It prevents a
    book-locked run from silently publishing an empty board, another book's
    leftovers, one-sided prices, or spread/total sides quoted at different lines.
    """
    expected = book.strip().lower()
    if not expected:
        raise SystemExit("Snapshot validation requires one explicit bookmaker key.")
    if not ODDS_LATEST_CSV.exists():
        if allow_quota_miss and quota_miss_recorded():
            print(f"validated empty {expected} snapshot after Odds API quota miss")
            return {"rows": 0, "events": 0, "game_rows": 0, "prop_rows": 0}
        raise SystemExit(f"No latest odds snapshot exists for {expected}.")
    frame = pd.read_csv(ODDS_LATEST_CSV, dtype=str).fillna("")
    if frame.empty:
        if allow_quota_miss and quota_miss_recorded():
            print(f"validated empty {expected} snapshot after Odds API quota miss")
            return {"rows": 0, "events": 0, "game_rows": 0, "prop_rows": 0}
        raise SystemExit(f"Book-lock validation failed: the {expected} snapshot is empty.")
    books = {value.strip().lower() for value in frame["book"] if value.strip()}
    if books != {expected}:
        raise SystemExit(
            f"Book-lock validation failed: expected only {expected}, found {sorted(books)}."
        )
    numeric_odds = pd.to_numeric(frame["odds"], errors="coerce")
    if numeric_odds.isna().any():
        raise SystemExit("Odds snapshot contains a non-numeric price.")
    ages = frame["fetched_at"].map(_quote_age_hours)
    if ages.isna().any() or (ages > max_age_hours).any():
        raise SystemExit(f"Odds snapshot contains a quote older than {max_age_hours:g} hours.")

    errors: list[str] = []
    for (event_id, market), rows in frame.groupby(["event_id", "market"], dropna=False):
        if market == "ml":
            expected_sides = {rows.iloc[0]["away"], rows.iloc[0]["home"]}
            if len(rows) != 2 or set(rows["side"]) != expected_sides:
                errors.append(f"{event_id} ml is not a paired away/home quote")
        elif market == "spread":
            lines = pd.to_numeric(rows["line"], errors="coerce")
            expected_sides = {rows.iloc[0]["away"], rows.iloc[0]["home"]}
            if (len(rows) != 2 or set(rows["side"]) != expected_sides
                    or lines.isna().any() or abs(float(lines.sum())) > 1e-6):
                errors.append(f"{event_id} spread sides/lines do not pair")
        elif market == "total":
            lines = pd.to_numeric(rows["line"], errors="coerce")
            if (len(rows) != 2 or set(rows["side"].str.lower()) != {"over", "under"}
                    or lines.isna().any() or lines.nunique() != 1):
                errors.append(f"{event_id} total sides/lines do not pair")
        elif str(market).startswith("player_"):
            for (player, line), pair in rows.groupby(["player", "line"], dropna=False):
                sides = {str(value).rsplit("|", 1)[-1].lower() for value in pair["side"]}
                if len(pair) != 2 or sides != {"over", "under"}:
                    errors.append(f"{event_id} {market} {player} {line} is not paired")
    if errors:
        preview = "; ".join(errors[:5])
        raise SystemExit(f"Odds snapshot integrity failed ({len(errors)} issue(s)): {preview}")
    return {
        "rows": len(frame),
        "events": int(frame["event_id"].nunique()),
        "game_rows": int(frame["market"].isin(["ml", "spread", "total"]).sum()),
        "prop_rows": int(frame["market"].str.startswith("player_").sum()),
    }


def _quote_age_hours(fetched_at: str) -> float | None:
    try:
        fetched = datetime.fromisoformat(str(fetched_at))
    except (TypeError, ValueError):
        return None
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - fetched).total_seconds() / 3600.0


def _prop_rows(frame: pd.DataFrame, player: str, market: str, side: str, line: float | None) -> pd.DataFrame:
    rows = frame[
        (frame["market"] == market)
        & (frame["player"].str.lower() == player.lower())
        & (frame["side"].str.lower().str.endswith(f"|{side.lower()}"))
    ]
    if line is not None:
        rows = rows[rows["line"].apply(lambda value: _same_line(value, line))]
    return rows.assign(odds_num=pd.to_numeric(rows["odds"], errors="coerce")).dropna(subset=["odds_num"])


def best_price_player_prop(
    player: str,
    market: str,
    side: str,
    line: float | None = None,
    *,
    max_age_hours: float = MAX_QUOTE_AGE_HOURS,
    allow_stale: bool = False,
) -> dict | None:
    """Best stored price for a prop, with freshness enforced and a paired de-vig quote.

    Returns odds/book plus `age_hours` and, when the opposite side of the same
    line is stored for the same book, `opposite_odds` so the caller can compute
    a vig-free implied probability. Raises SystemExit on stale quotes unless
    `allow_stale` is set — a fresh model against old odds is a phantom edge.
    """
    if not ODDS_LATEST_CSV.exists():
        return None
    frame = pd.read_csv(ODDS_LATEST_CSV, dtype=str).fillna("")
    frame = filter_odds_to_requested_books(frame)
    if frame is None or frame.empty:
        return None
    rows = _prop_rows(frame, player, market, side, line)
    if rows.empty:
        return None
    best = rows.loc[rows["odds_num"].idxmax()]

    age = _quote_age_hours(best["fetched_at"])
    if age is None or age > max_age_hours:
        age_label = f"{age:.1f}h old" if age is not None else "of unknown age"
        if not allow_stale:
            raise SystemExit(
                f"Stored quote for {player} {market} {side} is {age_label} "
                f"(max {max_age_hours:.0f}h). Re-fetch odds or pass --allow-stale."
            )

    opposite_side = "under" if side.lower() == "over" else "over"
    used_line = float(best["line"]) if best["line"] != "" else line
    opposite_odds = None
    opposite = _prop_rows(frame, player, market, opposite_side, used_line)
    same_book = opposite[opposite["book"] == best["book"]]
    if not same_book.empty:
        opposite_odds = int(same_book.iloc[0]["odds_num"])
    elif not opposite.empty:
        opposite_odds = int(opposite.loc[opposite["odds_num"].idxmax()]["odds_num"])

    return {
        "odds": int(best["odds_num"]),
        "book": best["book"],
        "line": used_line,
        "away": best["away"],
        "home": best["home"],
        "n_books": int(rows["book"].nunique()),
        "age_hours": round(age, 2) if age is not None else None,
        "fetched_at": best["fetched_at"],
        "opposite_odds": opposite_odds,
    }


def _same_line(value: str, line: float) -> bool:
    try:
        return abs(float(value) - float(line)) < 1e-6
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch WNBA odds snapshots.")
    parser.add_argument("--fetch-game", metavar="AWAY@HOME")
    parser.add_argument(
        "--fetch-slate",
        action="store_true",
        help="Pull ML/spread/total for every live WNBA game (one API call).",
    )
    parser.add_argument(
        "--props",
        action="store_true",
        help="Pull only the modeled player props for each event, preserving stored game lines.",
    )
    parser.add_argument(
        "--bookmakers",
        default=None,
        help="Comma-separated Odds API book keys, e.g. fanatics. Overrides ODDS_BOOKMAKERS.",
    )
    parser.add_argument(
        "--validate-latest",
        action="store_true",
        help="Require a fresh, single-book snapshot with correctly paired market sides.",
    )
    parser.add_argument(
        "--allow-quota-miss",
        action="store_true",
        help="With --validate-latest, accept an empty snapshot when The Odds API was out of credits.",
    )
    args = parser.parse_args()
    if args.bookmakers:
        os.environ["ODDS_BOOKMAKERS"] = args.bookmakers

    if args.validate_latest:
        books = requested_bookmakers()
        if len(books) != 1:
            raise SystemExit("--validate-latest requires exactly one --bookmakers key.")
        summary = validate_latest_snapshot(
            next(iter(books)),
            allow_quota_miss=args.allow_quota_miss,
        )
        print(
            "validated latest snapshot: "
            f"{summary['rows']} rows, {summary['events']} events, "
            f"{summary['game_rows']} game rows, {summary['prop_rows']} prop rows"
        )
    elif args.fetch_slate:
        fetch_slate(props=args.props)
    elif args.fetch_game:
        away, home = (part.strip().upper() for part in args.fetch_game.split("@", 1))
        fetch_game(away, home, props=args.props)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
