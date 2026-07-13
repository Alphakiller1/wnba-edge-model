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
    out["minutes_signal"] = zscore(out["projectedMinutes"].fillna(out["mpg"]) - out["mpg"].fillna(0))
    out["usage_signal"] = zscore(out["usg"]) + zscore(out["creationLoad"])
    out["impact_signal"] = zscore(out["rapm"]) + 0.5 * zscore(out["cvi"]) + 0.25 * zscore(out["bpm"])
    out["scoring_signal"] = zscore(out["projectedPoints"].fillna(out["ppg"])) + 0.35 * zscore(out["rimPressureIndex"])
    out["role_signal"] = zscore(out["roleTrustScore"]) + 0.5 * zscore(out["projectionConfidence"])
    out["recent_minutes_signal"] = zscore(_optional(frame, "last5_min_delta"))
    out["recent_usage_signal"] = zscore(_optional(frame, "last5_usage_proxy_delta"))
    out["recent_scoring_signal"] = zscore(_optional(frame, "last5_pts_delta"))
    out["recent_pra_signal"] = zscore(_optional(frame, "last5_pra_delta"))
    out["volatility_penalty"] = zscore(_optional(frame, "pra_std")).clip(lower=0)
    out["confidence_penalty"] = (100 - out["projectionConfidence"].fillna(50)).clip(lower=0) / 100

    out["edge_score"] = (
        1.30 * out["minutes_signal"].fillna(0)
        + 1.10 * out["usage_signal"].fillna(0)
        + 0.90 * out["scoring_signal"].fillna(0)
        + 0.75 * out["impact_signal"].fillna(0)
        + 0.60 * out["role_signal"].fillna(0)
        + 0.70 * out["recent_minutes_signal"].fillna(0)
        + 0.55 * out["recent_usage_signal"].fillna(0)
        + 0.45 * out["recent_pra_signal"].fillna(0)
        - 0.80 * out["confidence_penalty"].fillna(0.5)
        - 0.25 * out["volatility_penalty"].fillna(0)
    )
    out["watch_reason"] = out.apply(_watch_reason, axis=1)
    return out.sort_values("edge_score", ascending=False).reset_index(drop=True)


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
    reasons: list[str] = []
    if row.get("minutes_signal", 0) >= 0.75:
        reasons.append("minutes up")
    if row.get("usage_signal", 0) >= 1.0:
        reasons.append("usage/creation")
    if row.get("scoring_signal", 0) >= 1.0:
        reasons.append("scoring pressure")
    if row.get("impact_signal", 0) >= 1.0:
        reasons.append("impact")
    if row.get("recent_minutes_signal", 0) >= 0.75:
        reasons.append("recent minutes")
    if row.get("recent_usage_signal", 0) >= 0.75:
        reasons.append("recent usage")
    if row.get("recent_pra_signal", 0) >= 0.75:
        reasons.append("recent PRA")
    if row.get("confidence_penalty", 1) >= 0.55:
        reasons.append("low confidence")
    return ", ".join(reasons) if reasons else "baseline profile"
