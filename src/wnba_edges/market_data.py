from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ODDS_DIR = DATA_DIR / "odds"
ODDS_LATEST_CSV = ODDS_DIR / "odds_latest.csv"
ODDS_HISTORY_CSV = ODDS_DIR / "odds_history.csv"

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_SPORT_KEY = os.getenv("WNBA_ODDS_SPORT_KEY", "basketball_wnba")
ODDS_REGIONS = os.getenv("ODDS_REGIONS", "us")
ODDS_FORMAT = "american"
ODDS_BOOKMAKERS = os.getenv("ODDS_BOOKMAKERS", "")
ODDS_GAME_MARKETS = "h2h,spreads,totals"
ODDS_PROP_MARKETS = os.getenv(
    "WNBA_PROP_MARKETS",
    "player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals",
)

TEAM_NAME_TO_ABBR = {
    "Atlanta Dream": "ATL",
    "Chicago Sky": "CHI",
    "Connecticut Sun": "CON",
    "Dallas Wings": "DAL",
    "Golden State Valkyries": "GSV",
    "Indiana Fever": "IND",
    "Las Vegas Aces": "LVA",
    "Los Angeles Sparks": "LAS",
    "Minnesota Lynx": "MIN",
    "New York Liberty": "NYL",
    "Phoenix Mercury": "PHX",
    "Portland Fire": "PDX",
    "Seattle Storm": "SEA",
    "Toronto Tempo": "TOR",
    "Washington Mystics": "WAS",
}

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


def team_abbr(name: str) -> str:
    n = str(name).strip()
    if n.upper() in set(TEAM_NAME_TO_ABBR.values()):
        return n.upper()
    return TEAM_NAME_TO_ABBR.get(n, n.upper()[:3])


def _get(path: str, params: dict[str, Any]) -> Any:
    if not ODDS_API_KEY:
        raise SystemExit(
            "No ODDS_API_KEY set. Get a key from The Odds API, then set "
            '$env:ODDS_API_KEY = "your_key" in PowerShell.'
        )
    params = {**params, "apiKey": ODDS_API_KEY}
    url = f"{ODDS_API_BASE}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            _LAST_USAGE["remaining"] = response.headers.get("x-requests-remaining", "?")
            _LAST_USAGE["used"] = response.headers.get("x-requests-used", "?")
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        raise SystemExit(f"Odds API error {exc.code}: {body}") from exc


def list_events() -> dict[tuple[str, str], str]:
    data = _get(f"/sports/{ODDS_SPORT_KEY}/events", {})
    out: dict[tuple[str, str], str] = {}
    for event in data:
        away = team_abbr(event.get("away_team", ""))
        home = team_abbr(event.get("home_team", ""))
        out[(away, home)] = event.get("id", "")
    return out


def fetch_event_odds(event_id: str, props: bool = False) -> list[dict]:
    markets = ODDS_GAME_MARKETS + ("," + ODDS_PROP_MARKETS if props else "")
    params = {"regions": ODDS_REGIONS, "markets": markets, "oddsFormat": ODDS_FORMAT}
    if ODDS_BOOKMAKERS:
        params["bookmakers"] = ODDS_BOOKMAKERS
    event = _get(f"/sports/{ODDS_SPORT_KEY}/events/{event_id}/odds", params)
    return _normalize_event(event, datetime.now(timezone.utc).isoformat(timespec="seconds"))


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


def store(rows: list[dict]) -> None:
    if not rows:
        return
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=COLUMNS)

    if ODDS_HISTORY_CSV.exists():
        frame.to_csv(ODDS_HISTORY_CSV, mode="a", header=False, index=False)
    else:
        frame.to_csv(ODDS_HISTORY_CSV, index=False)

    fetched_games = set(zip(frame["away"], frame["home"]))
    if ODDS_LATEST_CSV.exists():
        previous = pd.read_csv(ODDS_LATEST_CSV, dtype=str).fillna("")
        keep = previous[~previous.apply(lambda row: (row["away"], row["home"]) in fetched_games, axis=1)]
        latest = pd.concat([keep, frame.astype(str)], ignore_index=True)
    else:
        latest = frame.astype(str)
    latest.to_csv(ODDS_LATEST_CSV, index=False)
    print(f"Stored {len(frame)} WNBA odds rows across {frame.groupby(['away', 'home']).ngroups} game(s).")


def best_price_player_prop(player: str, market: str, side: str, line: float | None = None) -> dict | None:
    if not ODDS_LATEST_CSV.exists():
        return None
    frame = pd.read_csv(ODDS_LATEST_CSV, dtype=str).fillna("")
    rows = frame[
        (frame["market"] == market)
        & (frame["player"].str.lower() == player.lower())
        & (frame["side"].str.lower().str.endswith(f"|{side.lower()}"))
    ]
    if line is not None:
        rows = rows[rows["line"].apply(lambda value: _same_line(value, line))]
    rows = rows.assign(odds_num=pd.to_numeric(rows["odds"], errors="coerce")).dropna(subset=["odds_num"])
    if rows.empty:
        return None
    best = rows.loc[rows["odds_num"].idxmax()]
    return {
        "odds": int(best["odds_num"]),
        "book": best["book"],
        "line": float(best["line"]) if best["line"] != "" else None,
        "away": best["away"],
        "home": best["home"],
        "n_books": int(rows["book"].nunique()),
    }


def _same_line(value: str, line: float) -> bool:
    try:
        return abs(float(value) - float(line)) < 1e-6
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch WNBA odds snapshots.")
    parser.add_argument("--fetch-game", metavar="AWAY@HOME")
    parser.add_argument("--props", action="store_true")
    args = parser.parse_args()

    if args.fetch_game:
        away, home = (part.strip().upper() for part in args.fetch_game.split("@", 1))
        fetch_game(away, home, props=args.props)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
