from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REGION = "us-east-1"
IDENTITY_POOL_ID = "us-east-1:7b073343-561b-4a8f-bf2a-765958c3aaaa"
BUCKET = "espnsportsanalytics.com"


@dataclass(frozen=True)
class EspnAnalyticsBox:
    game_id: str
    game_code: str
    date: str
    season: int
    four_factors: pd.DataFrame
    player_box: pd.DataFrame
    team_box: pd.DataFrame
    player_actions: pd.DataFrame


def parse_box_id(box_id: str) -> tuple[str, str, int, str]:
    """Parse ids like 20250811-1022500204 into game_code/date/season/game_id."""
    if "-" not in box_id:
        raise ValueError("ESPN Analytics box id must look like YYYYMMDD-GAMEID")
    game_code, game_id = box_id.split("-", 1)
    date = f"{game_code[:4]}-{game_code[4:6]}-{game_code[6:8]}"
    year = int(date[:4])
    month = int(date[5:7])
    season = year if month >= 2 else year - 1
    return game_code, date, season, game_id


def fetch_box(box_id: str) -> EspnAnalyticsBox:
    game_code, date, season, game_id = parse_box_id(box_id)
    client = _s3_client()
    main = _get_json(client, f"WNBA/netpts/{season}/{date}.json")
    try:
        actions = _get_json(client, f"WNBA/netpts/{season}/{date}_player.json")
    except Exception:
        actions = []

    four_factors = pd.DataFrame(main.get("four_factors", []))
    player_box = pd.DataFrame(main.get("player_box", []))
    team_box = pd.DataFrame(main.get("team_box", []))
    player_actions = pd.DataFrame(actions)

    for frame in (four_factors, player_box, team_box, player_actions):
        if not frame.empty:
            frame.insert(0, "source", "espnanalytics")
            frame.insert(1, "box_id", box_id)
            frame.insert(2, "date", date)

    player_box = player_box[player_box.get("gmId", "").astype(str) == game_id].copy() if not player_box.empty else player_box
    team_box = team_box[team_box.get("gameId", "").astype(str) == game_id].copy() if not team_box.empty else team_box
    four_factors = four_factors[four_factors.get("gameId", "").astype(str) == game_id].copy() if not four_factors.empty else four_factors
    player_actions = player_actions[player_actions.get("gmID", "").astype(str) == game_id].copy() if not player_actions.empty else player_actions

    if "minutes_played" in player_box.columns:
        player_box["minutes_decimal"] = player_box["minutes_played"].map(minutes_to_decimal)
    if "minutes_played" in team_box.columns:
        team_box["minutes_decimal"] = team_box["minutes_played"].map(minutes_to_decimal)

    return EspnAnalyticsBox(
        game_id=game_id,
        game_code=game_code,
        date=date,
        season=season,
        four_factors=four_factors,
        player_box=player_box,
        team_box=team_box,
        player_actions=player_actions,
    )


def write_box(box: EspnAnalyticsBox, root: Path) -> dict[str, Path]:
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"espnanalytics_{box.game_code}_{box.game_id}"
    paths = {
        "four_factors": raw_dir / f"{prefix}_four_factors.csv",
        "player_box": raw_dir / f"{prefix}_player_box.csv",
        "team_box": raw_dir / f"{prefix}_team_box.csv",
        "player_actions": raw_dir / f"{prefix}_player_actions.csv",
    }
    box.four_factors.to_csv(paths["four_factors"], index=False)
    box.player_box.to_csv(paths["player_box"], index=False)
    box.team_box.to_csv(paths["team_box"], index=False)
    box.player_actions.to_csv(paths["player_actions"], index=False)
    return paths


def minutes_to_decimal(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if ":" not in text:
        return None
    minutes, seconds = text.split(":", 1)
    try:
        return round(int(minutes) + int(seconds) / 60, 3)
    except ValueError:
        return None


def _s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install optional dependency boto3 to use ESPN Analytics S3 data: pip install boto3") from exc

    cognito = boto3.client("cognito-identity", region_name=REGION)
    identity_id = cognito.get_id(IdentityPoolId=IDENTITY_POOL_ID)["IdentityId"]
    creds = cognito.get_credentials_for_identity(IdentityId=identity_id)["Credentials"]
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretKey"],
        aws_session_token=creds["SessionToken"],
    )


def _get_json(client, key: str):
    obj = client.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))
