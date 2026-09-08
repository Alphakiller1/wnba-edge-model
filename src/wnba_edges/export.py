"""The JSON contract downstream systems read.

The dashboard in `report.py` is a rendered page. This module is the one place that
decides what a WNBA board looks like as data, so the content engine does not scrape
markup. Shape is a promise, same as `nfl-model/src/nflmodel/export.py`.

Two rules follow from that:

* **The authority travels with the numbers.** Every payload carries the published
  research posture (`may_bet`, unmet gates, evidence). A consumer that reads a
  margin without reading the permission has to do so deliberately.
* **Absences are explicit.** A game the model cannot price is emitted with null
  book fields, never dropped. A silently shorter list is indistinguishable from
  a shorter slate.

`schema` is versioned; add fields freely, rename or remove them only with a
version bump.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .paths import repository_root
from .predictions import results_summary
from .prop_projections import prop_slate_path

SCHEMA = "wnba-edge-model/board/1"

# Publication posture already stated in METHODOLOGY.md. This repo has no
# promotion-gate registry; it does not invent NFL/CFB production gates.
AUTHORITY_LEVEL = "RESEARCH_ONLY"
MAY_BET = False
UNMET_GATES: tuple[str, ...] = ()
EVIDENCE = (
    "METHODOLOGY.md — analytics research, not betting advice. "
    "No output is a wager instruction."
)


def _round(value, places: int = 3):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return round(number, places)


def _text(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return None
    return text


def _bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()
    return frame if frame is not None else pd.DataFrame()


def _kickoff(row: pd.Series) -> tuple[str | None, str | None]:
    """Human tip-off plus UTC ISO when the CSV carries a parseable stamp."""
    raw_time = _text(row.get("time"))
    raw_date = _text(row.get("date") or row.get("game_date"))
    generated = _text(row.get("generated_at"))
    kickoff = " ".join(part for part in (raw_date, raw_time) if part) or None
    for candidate in (generated,):
        if not candidate:
            continue
        try:
            stamp = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            return kickoff, stamp.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        except ValueError:
            continue
    return kickoff, None


def _game(row: pd.Series) -> dict | None:
    away = _text(row.get("away"))
    home = _text(row.get("home"))
    if not away or not home:
        return None
    kickoff, kickoff_utc = _kickoff(row)
    book_total = _round(row.get("book_total_line"), 2)
    book_spread = _round(row.get("book_spread_line"), 2)
    book_home_ml = _round(row.get("book_home_ml"), 0)
    book_away_ml = _round(row.get("book_away_ml"), 0)
    has_book = any(value is not None for value in (book_total, book_spread, book_home_ml, book_away_ml))
    book_name = (
        _text(row.get("book_ml_book"))
        or _text(row.get("book_spread_book"))
        or _text(row.get("book_total_book"))
    )
    return {
        "key": f"{away} @ {home}",
        "date": _text(row.get("date") or row.get("game_date")),
        "away": away,
        "home": home,
        "kickoff": kickoff,
        "kickoff_utc": kickoff_utc,
        "projected_away_pts": _round(row.get("projected_away_pts"), 1),
        "projected_home_pts": _round(row.get("projected_home_pts"), 1),
        "projected_total": _round(row.get("projected_total"), 2),
        "projected_home_spread": _round(row.get("projected_home_spread"), 2),
        "home_win_prob": _round(row.get("home_win_prob"), 4),
        "win_prob_basis": _text(row.get("win_prob_basis")),
        "home_court_pts": _round(row.get("home_court_pts"), 2),
        "action": "MONITOR" if has_book else "AVOID",
        "authority": AUTHORITY_LEVEL,
        "book": {
            "name": book_name,
            "total": book_total,
            "spread": book_spread,
            "home_moneyline": book_home_ml,
            "away_moneyline": book_away_ml,
        } if has_book else None,
    }


def _player_projection(row: pd.Series) -> dict | None:
    player_id = _text(row.get("player_id"))
    player = _text(row.get("player"))
    if not player_id and not player:
        return None
    priced = _bool(row.get("priced"))
    return {
        "game_date": _text(row.get("game_date")),
        "away": _text(row.get("away")),
        "home": _text(row.get("home")),
        "team": _text(row.get("team")),
        "player": player,
        "player_id": player_id,
        "market": _text(row.get("market")),
        "projection": _round(row.get("projection"), 2),
        "line": _round(row.get("line"), 2),
        "side": _text(row.get("side")),
        "odds": _round(row.get("odds"), 0),
        "book": _text(row.get("book")),
        "model_prob": _round(row.get("model_prob"), 4),
        "edge": _round(row.get("edge"), 4) if priced else None,
        "verdict": _text(row.get("verdict")),
        "priced": priced,
        "authority": AUTHORITY_LEVEL,
        "action": "MONITOR" if priced else "AVOID",
    }


def _source_rows(root: Path, season: str) -> list[dict]:
    processed = root / "data" / "processed"
    files = (
        ("game_projections", processed / f"game_projections_{season}.csv"),
        ("prop_projections", prop_slate_path(root, season)),
        ("game_results", processed / f"game_results_{season}.csv"),
        ("game_markets", processed / f"game_markets_{season}.csv"),
        ("player_features", processed / f"player_features_{season}.csv"),
    )
    rows = []
    for name, path in files:
        frame = _read_csv(path)
        rows.append({
            "name": name,
            "path": str(path.relative_to(root)) if path.exists() else str(path),
            "exists": path.exists(),
            "rows": int(len(frame)),
            "state": "fresh" if path.exists() else "missing",
        })
    return rows


def _odds_block(root: Path) -> dict:
    status_path = root / "data" / "odds" / "odds_status.json"
    latest = root / "data" / "odds" / "odds_latest.csv"
    quotes = _read_csv(latest)
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        status = {}
    game = status.get("game") or {}
    requested = status.get("requested_bookmakers") or []
    book = requested[0] if requested else None
    remaining = game.get("api_remaining")
    try:
        remaining_n = int(remaining) if remaining is not None else None
    except (TypeError, ValueError):
        remaining_n = None
    fetched = game.get("fetched_at")
    age = None
    if fetched:
        try:
            stamp = datetime.fromisoformat(str(fetched).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            age = int((datetime.now(UTC) - stamp.astimezone(UTC)).total_seconds())
        except ValueError:
            age = None
    matched = int(len(quotes)) if not quotes.empty else 0
    events = int(game.get("events") or 0)
    state = "fresh" if fetched else ("cached" if matched else "missing")
    return {
        "requested_book": book,
        "state": state,
        "fetched_at": fetched,
        "events": events,
        "matched": int(game.get("quotes") or matched),
        "slate_games": events,
        "slate_matched": events if matched else 0,
        "remaining": remaining_n,
        "age_seconds": age,
        "quotes_csv_rows": matched,
    }


def payload(root: Path | None = None, season: str = "2026-27") -> dict:
    """The slate: authority, games (projected_* vs book_*), and player props."""
    root = Path(root) if root is not None else repository_root(__file__)
    processed = root / "data" / "processed"
    games = [
        game
        for _, row in _read_csv(processed / f"game_projections_{season}.csv").iterrows()
        if (game := _game(row)) is not None
    ]
    props = [
        prop
        for _, row in _read_csv(prop_slate_path(root, season)).iterrows()
        if (prop := _player_projection(row)) is not None
    ]
    stamp = datetime.now(UTC).replace(microsecond=0)
    return {
        "schema": SCHEMA,
        "sport": "wnba",
        "generated_at_utc": stamp.isoformat().replace("+00:00", "Z"),
        "season": season,
        "authority": AUTHORITY_LEVEL,
        "may_bet": MAY_BET,
        "unmet_gates": list(UNMET_GATES),
        "evidence": EVIDENCE,
        "note": (
            "Research software. priced player props carry a matched book line; "
            "unpriced rows are model-only and never a pick."
        ),
        "games": games,
        "player_projections": props,
    }


def health(root: Path | None = None, season: str = "2026-27") -> dict:
    """DataStatus-shaped manifest (NFL/CFB `build.json` field contract)."""
    root = Path(root) if root is not None else repository_root(__file__)
    sources = _source_rows(root, season)
    odds = _odds_block(root)
    issues: list[str] = []
    projections = next((row for row in sources if row["name"] == "game_projections"), None)
    if projections and projections["rows"] == 0:
        issues.append("game_projections CSV has no slate rows")
    if not (root / "data" / "odds" / "odds_status.json").exists():
        issues.append("odds_status.json is missing")
    elif odds.get("state") == "missing":
        issues.append("odds snapshot has no fetch timestamp")
    remaining = odds.get("remaining")
    if remaining is not None and remaining < 20:
        issues.append(f"Odds API remaining credits are {remaining}")
    stamp = datetime.now(UTC).replace(microsecond=0)
    return {
        "state": "fresh" if not issues else "degraded",
        "generated_at": stamp.strftime("%Y-%m-%d %H:%M UTC"),
        "season": season,
        "sources": sources,
        "odds": odds,
        "issues": issues,
    }


def _ats_totals(games: dict) -> tuple[dict, dict]:
    def counts(n: int | None, hits: int | None) -> dict:
        if not n:
            return {"win": 0, "loss": 0, "push": 0}
        wins = int(hits or 0)
        return {"win": wins, "loss": max(int(n) - wins, 0), "push": 0}

    ats = counts(games.get("spread_ats_n"), games.get("spread_ats_correct"))
    totals = counts(games.get("total_side_n"), games.get("total_side_correct"))
    return ats, totals


def record(root: Path | None = None, season: str = "2026-27") -> dict:
    """ModelStatus / results ledger (NFL/CFB `record.json` field contract)."""
    root = Path(root) if root is not None else repository_root(__file__)
    summary = results_summary(root)
    games = summary.get("games") if isinstance(summary.get("games"), dict) else {}
    ats, totals = _ats_totals(games)
    return {
        "scope": "latest pregame forecast per game from data/predictions/game_predictions.csv",
        "authority": "shadow_only",
        "may_bet": MAY_BET,
        "unmet_gates": list(UNMET_GATES),
        "season": season,
        "games_graded": int(games.get("n") or 0),
        "pending_snapshots": int(games.get("_pending") or 0),
        "model_mae": _round(games.get("spread_mae"), 3),
        "consensus_mae": None,
        "book_mae": None,
        "ats": ats,
        "model_total_mae": _round(games.get("total_mae"), 3),
        "book_total_mae": None,
        "totals": totals,
        "winner_hit_rate": _round(games.get("winner_hit_rate"), 1),
        "brier": _round(games.get("brier"), 4),
    }


def write(root: Path, destination: str | Path, season: str = "2026-27") -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload(root, season), indent=2) + "\n", encoding="utf-8")
    return path


def write_bundle(root: Path, site_dir: str | Path, season: str = "2026-27") -> dict[str, Path]:
    """Write board.json, build.json, and record.json beside the HTML page."""
    site = Path(site_dir)
    site.mkdir(parents=True, exist_ok=True)
    board = write(root, site / "board.json", season)
    build = site / "build.json"
    ledger = site / "record.json"
    build.write_text(json.dumps(health(root, season), indent=2) + "\n", encoding="utf-8")
    ledger.write_text(json.dumps(record(root, season), indent=2) + "\n", encoding="utf-8")
    return {"board": board, "build": build, "record": ledger}
