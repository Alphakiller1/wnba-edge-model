from __future__ import annotations

from pathlib import Path

import pandas as pd

CORE_COLUMNS = [
    "id",
    "name",
    "team",
    "pos",
    "season",
    "gp",
    "mpg",
    "projectedMinutes",
    "expectedMinutes",
    "projectedPoints",
    "expectedPoints",
    "projectedPlayProbability",
    "projectedStarterProbability",
    "projectionConfidence",
    "ppg",
    "rpg",
    "apg",
    "usg",
    "ortg",
    "drtg",
    "netRtg",
    "rapm",
    "rapmConfidence",
    "cvi",
    "cviConfidence",
    "bpm",
    "roleTrustScore",
    "minutesRoleGap",
    "creationLoad",
    "rimPressureIndex",
    "shotQualityIndex",
    "defensiveActivityIndex",
]


# Low-sample gate: players below either floor are flagged and excluded from the
# default edge board. Rate stats on a handful of minutes z-score into huge fake
# signals (a 65% usage rate over 1 minute of floor time is noise, not an edge).
MIN_GAMES_FOR_BOARD = 5
MIN_MPG_FOR_BOARD = 10.0
# Shrinkage prior in total minutes: sample_weight = minutes / (minutes + this).
SHRINK_PRIOR_MINUTES = 150.0


def build_player_features(players: pd.DataFrame) -> pd.DataFrame:
    frame = players.copy()
    if "playerId" in frame.columns and "id" not in frame.columns:
        frame["id"] = frame["playerId"]
    for column in CORE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    numeric_cols = [column for column in CORE_COLUMNS if column not in {"name", "team", "pos", "season"}]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    out = frame[CORE_COLUMNS].copy()

    # Empirical-Bayes-style shrinkage toward the league mean (0 in z-space),
    # weighted by total minutes played, so tiny samples cannot dominate the board.
    total_minutes = (out["gp"].fillna(0) * out["mpg"].fillna(0)).clip(lower=0)
    sample_weight = total_minutes / (total_minutes + SHRINK_PRIOR_MINUTES)
    out["sample_weight"] = sample_weight.round(4)
    out["low_sample"] = (out["gp"].fillna(0) < MIN_GAMES_FOR_BOARD) | (
        out["mpg"].fillna(0) < MIN_MPG_FOR_BOARD
    )

    out["minutes_signal"] = (
        zscore(out["projectedMinutes"].fillna(out["mpg"]) - out["mpg"].fillna(0)) * sample_weight
    )
    out["usage_signal"] = (zscore(out["usg"]) + zscore(out["creationLoad"])) * sample_weight
    out["impact_signal"] = (
        zscore(out["rapm"]) + 0.5 * zscore(out["cvi"]) + 0.25 * zscore(out["bpm"])
    ) * sample_weight
    out["scoring_signal"] = (
        zscore(out["projectedPoints"].fillna(out["ppg"])) + 0.35 * zscore(out["rimPressureIndex"])
    ) * sample_weight
    out["role_signal"] = zscore(out["roleTrustScore"]) + 0.5 * zscore(out["projectionConfidence"])
    out["recent_minutes_signal"] = zscore(_optional(frame, "last5_min_delta")) * sample_weight
    out["recent_usage_signal"] = zscore(_optional(frame, "last5_usage_proxy_delta")) * sample_weight
    out["recent_scoring_signal"] = zscore(_optional(frame, "last5_pts_delta")) * sample_weight
    out["recent_pra_signal"] = zscore(_optional(frame, "last5_pra_delta")) * sample_weight
    out["volatility_penalty"] = zscore(_optional(frame, "pra_std")).clip(lower=0)
    out["confidence_penalty"] = (100 - out["projectionConfidence"].fillna(50)).clip(lower=0) / 100

    # ── the anchor gap ────────────────────────────────────────────────────────
    # A prop line is anchored on a season baseline. Edge therefore lives where a player's
    # CURRENT role has moved away from that baseline — not where the player is good.
    #
    # The previous score summed level terms (usage, scoring, impact) with change terms, and
    # the level terms are mutually correlated, so the ranking collapsed onto "who is a star":
    # measured against the committed season it correlated +0.76 with raw PPG and +0.71 with
    # MPG. Those are the most heavily bet, most efficiently priced players on the slate, so
    # the board was ranking by where an edge is *least* likely to survive the closing line.
    #
    # `anchor_gap` is signed: positive means the player's live role sits above the season
    # anchor, so a season-anchored line is more likely to be set too low.
    out["anchor_gap"] = (
        1.20 * out["minutes_signal"].fillna(0)
        + 1.00 * out["recent_minutes_signal"].fillna(0)
        + 0.85 * out["recent_usage_signal"].fillna(0)
        + 0.85 * out["recent_pra_signal"].fillna(0)
        + 0.40 * out["recent_scoring_signal"].fillna(0)
    )

    # Level is treated as an ATTENTION proxy and penalised, not rewarded. Only positive
    # attention is penalised, so demoting stars does not silently promote deep-bench players
    # whose lines are thin or unposted.
    out["market_attention"] = (
        zscore(out["ppg"]) + zscore(out["usg"]) + zscore(out["mpg"])
    ) / 3
    out["attention_penalty"] = out["market_attention"].fillna(0).clip(lower=0)

    out["edge_score"] = (
        out["anchor_gap"].abs()
        - 0.50 * out["attention_penalty"]
        - 0.40 * out["confidence_penalty"].fillna(0.5)
        - 0.25 * out["volatility_penalty"].fillna(0)
    )
    out["lean"] = out["anchor_gap"].apply(
        lambda gap: "OVER" if gap > 0.25 else ("UNDER" if gap < -0.25 else "—")
    )
    out["watch_reason"] = out.apply(_watch_reason, axis=1)
    return out.sort_values("edge_score", ascending=False).reset_index(drop=True)


def board_eligible(features: pd.DataFrame) -> pd.DataFrame:
    """Rows that pass the low-sample gate — what the public edge board shows."""
    if "low_sample" not in features.columns:
        return features
    mask = features["low_sample"].astype(str).str.lower().isin({"true", "1"}) | (
        features["low_sample"] == True  # noqa: E712 — CSV round-trips booleans as strings
    )
    return features[~mask]


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(0, index=series.index, dtype="float64")
    return (numeric - numeric.mean(skipna=True)) / std


def _optional(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(0, index=frame.index, dtype="float64")


def _watch_reason(row: pd.Series) -> str:
    """Why the season anchor may be stale for this player — stated in market terms.

    These read as a case against the line, not as a description of the player. "Scoring
    pressure" on a star is not a reason to bet anything; "minutes stepped up, season line
    lags" is.
    """
    reasons: list[str] = []

    def moved(key: str, up: str, down: str, threshold: float = 0.75) -> None:
        value = row.get(key, 0) or 0
        if value >= threshold:
            reasons.append(up)
        elif value <= -threshold:
            reasons.append(down)

    moved("minutes_signal", "role expanding", "role shrinking")
    moved("recent_minutes_signal", "minutes up L5", "minutes down L5")
    moved("recent_usage_signal", "usage up L5", "usage down L5")
    moved("recent_pra_signal", "PRA above anchor", "PRA below anchor")
    moved("recent_scoring_signal", "scoring up L5", "scoring cooled L5")

    if (row.get("attention_penalty", 0) or 0) >= 0.75:
        reasons.append("heavily bet — priced efficiently")
    if row.get("confidence_penalty", 1) >= 0.55:
        reasons.append("low projection confidence")
    if (row.get("volatility_penalty", 0) or 0) >= 1.0:
        reasons.append("volatile game-to-game")
    if row.get("low_sample"):
        reasons.append("LOW SAMPLE")
    return ", ".join(reasons) if reasons else "no gap to the season anchor"
