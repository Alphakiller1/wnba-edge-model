from __future__ import annotations

import json
import re
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd

from .http import HttpClient
from .wnbanalytics import scrape_game_detail, scrape_games, scrape_players, scrape_teams, write_jsonl

ROLLING_WINDOWS = (3, 5, 10)


def slugify_player(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug.replace("a-ja", "aja")


def scrape_season_snapshot(root: Path, season: str = "2026-27", pause_seconds: float = 0.5) -> dict[str, Path]:
    """Fetch season-level players, teams, games, and detailed box scores.

    Incremental: finished games whose detail is already stored are not
    re-downloaded — completed box scores never change, and re-fetching the whole
    season every refresh hammers the source for nothing.
    """
    from .contracts import validate_rows

    client = HttpClient(timeout=45, retries=3, pause_seconds=1.5)
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    players = validate_rows(scrape_players(season=season, client=client), "wnbanalytics_players")
    teams = validate_rows(scrape_teams(season=season, client=client), "wnbanalytics_teams")
    games = validate_rows(scrape_games(season=season, client=client), "wnbanalytics_games")

    paths = {
        "players": raw_dir / f"wnbanalytics_players_{season}.jsonl",
        "teams": raw_dir / f"wnbanalytics_teams_{season}.jsonl",
        "games": raw_dir / f"wnbanalytics_games_{season}.jsonl",
        "game_details": raw_dir / f"wnbanalytics_game_details_{season}.jsonl",
    }
    write_jsonl(players, paths["players"])
    write_jsonl(teams, paths["teams"])
    write_jsonl(games, paths["games"])

    existing: dict[int, dict[str, Any]] = {}
    if paths["game_details"].exists():
        for detail in load_raw_jsonl(paths["game_details"]):
            game_id = ((detail.get("game") or {}).get("id"))
            if game_id is not None:
                existing[int(game_id)] = detail

    details: list[dict[str, Any]] = []
    fetched = 0
    for game in games:
        game_id = int(game["id"])
        cached = existing.get(game_id)
        if cached is not None and game.get("winner"):
            details.append(cached)
            continue
        if fetched and pause_seconds:
            sleep(pause_seconds)
        detail = scrape_game_detail(game_id, client=client)
        detail["season"] = season
        details.append(detail)
        fetched += 1
    print(f"game details: {fetched} fetched, {len(details) - fetched} reused from cache")
    write_jsonl(details, paths["game_details"])
    return paths


def load_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_season_tables(root: Path, season: str = "2026-27") -> dict[str, Path]:
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    games = pd.read_json(raw_dir / f"wnbanalytics_games_{season}.jsonl", lines=True)
    players = pd.read_json(raw_dir / f"wnbanalytics_players_{season}.jsonl", lines=True)
    teams = pd.read_json(raw_dir / f"wnbanalytics_teams_{season}.jsonl", lines=True)
    details = load_raw_jsonl(raw_dir / f"wnbanalytics_game_details_{season}.jsonl")

    game_results = normalize_game_results(games, season)
    player_logs = normalize_player_game_logs(details, players, season)
    team_logs = normalize_team_game_logs(details, season)
    player_splits = build_player_splits(player_logs)
    team_splits = build_team_splits(team_logs)
    player_model_inputs = build_player_model_inputs(players, player_splits, player_logs)

    paths = {
        "game_results": processed_dir / f"game_results_{season}.csv",
        "player_game_logs": processed_dir / f"player_game_logs_{season}.csv",
        "team_game_logs": processed_dir / f"team_game_logs_{season}.csv",
        "player_splits": processed_dir / f"player_splits_{season}.csv",
        "team_splits": processed_dir / f"team_splits_{season}.csv",
        "players": processed_dir / f"players_season_{season}.csv",
        "teams": processed_dir / f"teams_season_{season}.csv",
        "player_model_inputs": processed_dir / f"player_model_inputs_{season}.csv",
    }
    game_results.to_csv(paths["game_results"], index=False)
    player_logs.to_csv(paths["player_game_logs"], index=False)
    team_logs.to_csv(paths["team_game_logs"], index=False)
    player_splits.to_csv(paths["player_splits"], index=False)
    team_splits.to_csv(paths["team_splits"], index=False)
    players.to_csv(paths["players"], index=False)
    teams.to_csv(paths["teams"], index=False)
    player_model_inputs.to_csv(paths["player_model_inputs"], index=False)
    return paths


def normalize_game_results(games: pd.DataFrame, season: str) -> pd.DataFrame:
    out = games.copy()
    if "season" not in out.columns:
        out.insert(0, "season", season)
    else:
        out["season"] = out["season"].fillna(season)
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out["total"] = pd.to_numeric(out["awayPts"], errors="coerce") + pd.to_numeric(out["homePts"], errors="coerce")
    out["home_margin"] = pd.to_numeric(out["homePts"], errors="coerce") - pd.to_numeric(out["awayPts"], errors="coerce")
    out["away_result"] = (out["winner"] == out["away"]).map({True: "W", False: "L"})
    out["home_result"] = (out["winner"] == out["home"]).map({True: "W", False: "L"})
    return out.sort_values(["date", "id"]).reset_index(drop=True)


def normalize_player_game_logs(details: list[dict[str, Any]], players: pd.DataFrame, season: str) -> pd.DataFrame:
    player_context = players.set_index("id").to_dict(orient="index") if "id" in players.columns else {}
    rows: list[dict[str, Any]] = []
    for detail in details:
        game = detail["game"]
        for side, box_key in (("home", "homeBox"), ("away", "awayBox")):
            team = game[side]
            opponent = game["away"] if side == "home" else game["home"]
            result = "W" if game["winner"] == team else "L"
            for row in detail.get(box_key, []):
                ctx = player_context.get(row.get("playerId"), {})
                base = {
                    "season": season,
                    "game_id": game["id"],
                    "date": game["date"],
                    "month": game.get("month"),
                    "team": team,
                    "opponent": opponent,
                    "home_away": side,
                    "result": result,
                    "team_pts": game[f"{side}Pts"],
                    "opp_pts": game["awayPts"] if side == "home" else game["homePts"],
                    "pace": game.get("pace"),
                    "margin": game[f"{side}Pts"] - (game["awayPts"] if side == "home" else game["homePts"]),
                    "game_total": game["awayPts"] + game["homePts"],
                }
                enriched = {**base, **row}
                enriched["season_mpg"] = ctx.get("mpg")
                enriched["season_usg"] = ctx.get("usg")
                enriched["season_ppg"] = ctx.get("ppg")
                enriched["season_rpg"] = ctx.get("rpg")
                enriched["season_apg"] = ctx.get("apg")
                rows.append(enriched)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    numeric = ["min", "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "plusMinus", "pace"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["pra"] = frame["pts"] + frame["reb"] + frame["ast"]
    frame["stocks"] = frame["stl"] + frame["blk"]
    frame["fantasy_simple"] = frame["pts"] + 1.2 * frame["reb"] + 1.5 * frame["ast"] + 3 * frame["stocks"] - frame["tov"]
    frame["usage_proxy"] = (frame["fga"] + 0.44 * frame["fta"] + frame["tov"]) / frame["min"].replace(0, pd.NA)
    frame["shot_attempt_rate"] = frame["fga"] / frame["min"].replace(0, pd.NA)
    frame["ft_rate_game"] = frame["fta"] / frame["fga"].replace(0, pd.NA)
    frame["three_rate_game"] = frame["fg3a"] / frame["fga"].replace(0, pd.NA)
    frame["starter"] = frame["starter"].astype(bool)
    return frame.sort_values(["playerId", "date"]).reset_index(drop=True)


def normalize_team_game_logs(details: list[dict[str, Any]], season: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for detail in details:
        game = detail["game"]
        for side, box_key in (("home", "homeBox"), ("away", "awayBox")):
            team = game[side]
            opponent = game["away"] if side == "home" else game["home"]
            box = pd.DataFrame(detail.get(box_key, []))
            totals = box[["min", "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]].sum(numeric_only=True).to_dict() if not box.empty else {}
            rows.append(
                {
                    "season": season,
                    "game_id": game["id"],
                    "date": game["date"],
                    "month": game.get("month"),
                    "team": team,
                    "opponent": opponent,
                    "home_away": side,
                    "result": "W" if game["winner"] == team else "L",
                    "team_pts": game[f"{side}Pts"],
                    "opp_pts": game["awayPts"] if side == "home" else game["homePts"],
                    "margin": game[f"{side}Pts"] - (game["awayPts"] if side == "home" else game["homePts"]),
                    "pace": game.get("pace"),
                    "game_total": game["awayPts"] + game["homePts"],
                    "players_used": len(box),
                    **totals,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    frame["efg"] = (frame["fgm"] + 0.5 * frame["fg3m"]) / frame["fga"].replace(0, pd.NA)
    frame["tov_rate_proxy"] = frame["tov"] / (frame["fga"] + 0.44 * frame["fta"] + frame["tov"]).replace(0, pd.NA)
    frame["three_rate"] = frame["fg3a"] / frame["fga"].replace(0, pd.NA)
    frame["ft_rate"] = frame["fta"] / frame["fga"].replace(0, pd.NA)
    return frame.sort_values(["team", "date"]).reset_index(drop=True)


def build_player_splits(logs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if logs.empty:
        return pd.DataFrame()
    dimensions = [
        ("overall", lambda df: pd.Series("all", index=df.index)),
        ("home_away", lambda df: df["home_away"]),
        ("result", lambda df: df["result"]),
        ("starter", lambda df: df["starter"].map({True: "starter", False: "bench"})),
        ("opponent", lambda df: df["opponent"]),
        ("team", lambda df: df["team"]),
        ("pos", lambda df: df["pos"].fillna("")),
        ("month", lambda df: df["month"]),
    ]
    for split_type, values_fn in dimensions:
        temp = logs.copy()
        temp["split_value"] = values_fn(temp)
        rows.extend(_aggregate_player_split(temp, split_type))

    for window in ROLLING_WINDOWS:
        recent = logs.sort_values("date").groupby("playerId", group_keys=False).tail(window).copy()
        recent["split_value"] = f"last_{window}"
        rows.extend(_aggregate_player_split(recent, "recent"))
    return pd.DataFrame(rows).sort_values(["player", "split_type", "split_value"]).reset_index(drop=True)


def _aggregate_player_split(frame: pd.DataFrame, split_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = ["min", "pts", "reb", "ast", "pra", "stl", "blk", "stocks", "tov", "fga", "fg3a", "fta", "usage_proxy", "fantasy_simple", "plusMinus"]
    for (player_id, player, team, split_value), group in frame.groupby(["playerId", "name", "team", "split_value"], dropna=False):
        row = {"playerId": player_id, "player": player, "team": team, "split_type": split_type, "split_value": split_value, "games": len(group)}
        for metric in metrics:
            row[f"{metric}_avg"] = group[metric].mean()
        row["starter_rate"] = group["starter"].mean()
        row["win_rate"] = (group["result"] == "W").mean()
        rows.append(row)
    return rows


def build_team_splits(logs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if logs.empty:
        return pd.DataFrame()
    dimensions = [
        ("overall", lambda df: pd.Series("all", index=df.index)),
        ("home_away", lambda df: df["home_away"]),
        ("result", lambda df: df["result"]),
        ("opponent", lambda df: df["opponent"]),
        ("month", lambda df: df["month"]),
    ]
    for split_type, values_fn in dimensions:
        temp = logs.copy()
        temp["split_value"] = values_fn(temp)
        for (team, split_value), group in temp.groupby(["team", "split_value"], dropna=False):
            rows.append(
                {
                    "team": team,
                    "split_type": split_type,
                    "split_value": split_value,
                    "games": len(group),
                    "pts_avg": group["team_pts"].mean(),
                    "opp_pts_avg": group["opp_pts"].mean(),
                    "margin_avg": group["margin"].mean(),
                    "pace_avg": group["pace"].mean(),
                    "total_avg": group["game_total"].mean(),
                    "win_rate": (group["result"] == "W").mean(),
                    "efg_avg": group["efg"].mean(),
                    "tov_rate_proxy_avg": group["tov_rate_proxy"].mean(),
                    "three_rate_avg": group["three_rate"].mean(),
                    "ft_rate_avg": group["ft_rate"].mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["team", "split_type", "split_value"]).reset_index(drop=True)


def build_player_model_inputs(players: pd.DataFrame, splits: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Wide player table for modeling props/edges from season, split, and recent data."""
    base = players.copy()
    if "id" in base.columns:
        base = base.rename(columns={"id": "playerId"})

    useful_metrics = [
        "min_avg",
        "pts_avg",
        "reb_avg",
        "ast_avg",
        "pra_avg",
        "stocks_avg",
        "tov_avg",
        "fga_avg",
        "fg3a_avg",
        "fta_avg",
        "usage_proxy_avg",
        "fantasy_simple_avg",
        "plusMinus_avg",
        "starter_rate",
        "win_rate",
    ]
    wide = base
    split_specs = [
        ("overall", "all", "overall"),
        ("recent", "last_3", "last3"),
        ("recent", "last_5", "last5"),
        ("recent", "last_10", "last10"),
        ("home_away", "home", "home"),
        ("home_away", "away", "away"),
        ("starter", "starter", "starter"),
        ("starter", "bench", "bench"),
        ("result", "W", "wins"),
        ("result", "L", "losses"),
    ]
    for split_type, split_value, prefix in split_specs:
        sub = splits[(splits["split_type"] == split_type) & (splits["split_value"].astype(str) == split_value)].copy()
        keep = ["playerId"] + [column for column in useful_metrics if column in sub.columns]
        if sub.empty:
            continue
        sub = sub[keep].groupby("playerId", as_index=False).mean(numeric_only=True)
        sub = sub.rename(columns={column: f"{prefix}_{column}" for column in sub.columns if column != "playerId"})
        wide = wide.merge(sub, on="playerId", how="left")

    if not logs.empty:
        volatility = (
            logs.groupby("playerId")
            .agg(
                pts_std=("pts", "std"),
                reb_std=("reb", "std"),
                ast_std=("ast", "std"),
                pra_std=("pra", "std"),
                min_std=("min", "std"),
                games_logged=("game_id", "nunique"),
            )
            .reset_index()
        )
        wide = wide.merge(volatility, on="playerId", how="left")

    for stat in ("pts", "reb", "ast", "pra", "min", "usage_proxy"):
        season_col = {"pts": "ppg", "reb": "rpg", "ast": "apg", "min": "mpg"}.get(stat, f"overall_{stat}_avg")
        recent_col = f"last5_{stat}_avg"
        if season_col in wide.columns and recent_col in wide.columns:
            wide[f"last5_{stat}_delta"] = pd.to_numeric(wide[recent_col], errors="coerce") - pd.to_numeric(wide[season_col], errors="coerce")

    return wide
