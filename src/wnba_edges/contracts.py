"""Light schema contracts for scraped payloads.

The model reads from unofficial rendered-app payloads; silent schema drift
(a renamed field upstream) must fail the scrape loudly instead of quietly
degrading every downstream signal to NaN.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


class ContractError(RuntimeError):
    pass


CONTRACTS: dict[str, dict] = {
    "wnbanalytics_players": {
        "required": {"id", "name", "team", "gp", "mpg", "ppg", "rpg", "apg", "usg"},
        "min_rows": 100,
    },
    "wnbanalytics_teams": {
        "required": {"abbr", "name", "ortg", "drtg", "net", "pace"},
        "min_rows": 12,
    },
    "wnbanalytics_games": {
        "required": {"id", "date", "away", "home", "awayPts", "homePts", "winner"},
        "min_rows": 1,
    },
    "wnbanalytics_game_detail": {
        "required": {"game", "homeBox", "awayBox"},
        "min_rows": 1,
    },
}


def validate_rows(rows: list[dict[str, Any]], contract: str) -> list[dict[str, Any]]:
    """Validate a list-of-dicts payload against a named contract; returns it unchanged."""
    spec = CONTRACTS[contract]
    if len(rows) < spec["min_rows"]:
        raise ContractError(
            f"{contract}: expected at least {spec['min_rows']} rows, got {len(rows)} — "
            "source may have changed or the fetch failed."
        )
    sample = rows[0]
    missing = sorted(spec["required"] - set(sample.keys()))
    if missing:
        raise ContractError(
            f"{contract}: payload is missing required field(s) {missing}. "
            f"Present fields: {sorted(sample.keys())[:30]}. The upstream schema likely changed."
        )
    return rows


def validate_frame(frame: pd.DataFrame, contract: str) -> pd.DataFrame:
    spec = CONTRACTS[contract]
    if len(frame) < spec["min_rows"]:
        raise ContractError(
            f"{contract}: expected at least {spec['min_rows']} rows, got {len(frame)}."
        )
    missing = sorted(spec["required"] - set(frame.columns))
    if missing:
        raise ContractError(
            f"{contract}: frame is missing required column(s) {missing}."
        )
    return frame
