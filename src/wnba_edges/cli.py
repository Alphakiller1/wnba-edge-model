from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .betting import evaluate_over_under
from .espnanalytics import fetch_box, write_box
from .features import build_player_features, load_jsonl
from .herhoopstats import fetch_research_table, write_table
from .market_data import best_price_player_prop
from .season import build_season_tables, scrape_season_snapshot
from .wnbanalytics import scrape_players, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(prog="wnba-edges")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape = subparsers.add_parser("scrape-wnbanalytics")
    scrape.add_argument("--season", default="2026-27")

    features = subparsers.add_parser("build-features")
    features.add_argument("--season", default="2026-27")

    board = subparsers.add_parser("edge-board")
    board.add_argument("--season", default="2026-27")
    board.add_argument("--top", type=int, default=25)

    prop = subparsers.add_parser("evaluate-player-prop")
    prop.add_argument("--season", default="2026-27")
    prop.add_argument("--player", required=True)
    prop.add_argument("--market", default="player_points")
    prop.add_argument("--side", choices=["over", "under"], required=True)
    prop.add_argument("--line", type=float, required=True)
    prop.add_argument("--odds", type=int, default=None)
    prop.add_argument("--sigma", type=float, default=6.5)
    prop.add_argument("--use-market", action="store_true")

    season_scrape = subparsers.add_parser("scrape-season")
    season_scrape.add_argument("--season", default="2026-27")
    season_scrape.add_argument("--pause-seconds", type=float, default=0.05)

    season_build = subparsers.add_parser("build-season-tables")
    season_build.add_argument("--season", default="2026-27")

    refresh = subparsers.add_parser("refresh-season")
    refresh.add_argument("--season", default="2026-27")
    refresh.add_argument("--pause-seconds", type=float, default=0.05)

    hhs = subparsers.add_parser("scrape-herhoopstats")
    hhs.add_argument("--research-type", default="player_single_games")
    hhs.add_argument("--min-season", type=int, default=2026)
    hhs.add_argument("--max-season", type=int, default=2026)
    hhs.add_argument("--stats-to-show", choices=["traditional", "advanced", "jump_ball"], default="traditional")

    espn = subparsers.add_parser("scrape-espnanalytics-box")
    espn.add_argument("--id", required=True, help="ESPN Analytics box id, e.g. 20250811-1022500204")

    args = parser.parse_args()

    if args.command == "scrape-wnbanalytics":
        rows = scrape_players(season=args.season)
        path = DATA / "raw" / f"wnbanalytics_players_{args.season}.jsonl"
        write_jsonl(rows, path)
        print(f"wrote {len(rows)} player rows to {path}")

    elif args.command == "build-features":
        raw_path = DATA / "raw" / f"wnbanalytics_players_{args.season}.jsonl"
        features_path = DATA / "processed" / f"player_features_{args.season}.csv"
        model_inputs_path = DATA / "processed" / f"player_model_inputs_{args.season}.csv"
        if model_inputs_path.exists():
            players = pd.read_csv(model_inputs_path)
        else:
            players = load_jsonl(raw_path)
        feature_frame = build_player_features(players)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        feature_frame.to_csv(features_path, index=False)
        print(f"wrote {len(feature_frame)} feature rows to {features_path}")

    elif args.command == "edge-board":
        features_path = DATA / "processed" / f"player_features_{args.season}.csv"
        board_path = DATA / "processed" / f"edge_board_{args.season}.csv"
        feature_frame = load_features_csv(features_path).head(args.top)
        feature_frame.to_csv(board_path, index=False)
        print(feature_frame[["name", "team", "edge_score", "watch_reason"]].to_string(index=False))
        print(f"wrote top {len(feature_frame)} rows to {board_path}")

    elif args.command == "evaluate-player-prop":
        features_path = DATA / "processed" / f"player_features_{args.season}.csv"
        feature_frame = load_features_csv(features_path)
        row = _match_player(feature_frame, args.player)
        odds = args.odds
        market_note = "manual odds"
        if args.use_market:
            price = best_price_player_prop(args.player, args.market, args.side, args.line)
            if price is None and odds is None:
                raise SystemExit("No stored market price found. Pass --odds or fetch odds first.")
            if price is not None:
                odds = price["odds"]
                market_note = f"{price['book']} best price across {price['n_books']} book(s)"
        if odds is None:
            raise SystemExit("Pass --odds, or use --use-market after fetching odds.")

        projection = _projection_for_market(row, args.market)
        value = evaluate_over_under(
            projection=projection,
            line=args.line,
            odds=odds,
            side=args.side,
            sigma=args.sigma,
        )
        print(f"{row['name']} {args.market} {args.side.title()} {args.line} @ {odds:+d}")
        print(f"projection: {projection:.2f} | source: {market_note}")
        print(f"model prob: {value.model_prob * 100:.1f}% | implied: {value.implied_prob * 100:.1f}%")
        print(f"edge: {value.edge * 100:+.1f} pts | EV/unit: {value.ev_per_unit:+.3f}")
        print(f"fair odds: {value.fair_odds:+d} | tier: {value.tier} | units: {value.units}")
        print(f"verdict: {value.verdict}")

    elif args.command == "scrape-season":
        paths = scrape_season_snapshot(ROOT, season=args.season, pause_seconds=args.pause_seconds)
        for name, path in paths.items():
            print(f"{name}: {path}")

    elif args.command == "build-season-tables":
        paths = build_season_tables(ROOT, season=args.season)
        for name, path in paths.items():
            print(f"{name}: {path}")

    elif args.command == "refresh-season":
        scrape_paths = scrape_season_snapshot(ROOT, season=args.season, pause_seconds=args.pause_seconds)
        table_paths = build_season_tables(ROOT, season=args.season)
        for name, path in {**scrape_paths, **table_paths}.items():
            print(f"{name}: {path}")

    elif args.command == "scrape-herhoopstats":
        frame = fetch_research_table(
            research_type=args.research_type,
            min_season=args.min_season,
            max_season=args.max_season,
            stats_to_show=args.stats_to_show,
        )
        path = DATA / "raw" / f"herhoopstats_{args.research_type}_{args.min_season}_{args.max_season}_{args.stats_to_show}.csv"
        write_table(frame, path)
        print(f"wrote {len(frame)} rows to {path}")

    elif args.command == "scrape-espnanalytics-box":
        box = fetch_box(args.id)
        paths = write_box(box, ROOT)
        print(f"fetched ESPN Analytics box {args.id}:")
        print(f"player_box rows: {len(box.player_box)}")
        print(f"team_box rows: {len(box.team_box)}")
        print(f"four_factor rows: {len(box.four_factors)}")
        print(f"player_action rows: {len(box.player_actions)}")
        for name, path in paths.items():
            print(f"{name}: {path}")


def load_features_csv(path: Path):
    return pd.read_csv(path).sort_values("edge_score", ascending=False)


def _match_player(frame: pd.DataFrame, name: str) -> pd.Series:
    exact = frame[frame["name"].str.lower() == name.lower()]
    if not exact.empty:
        return exact.iloc[0]
    partial = frame[frame["name"].str.lower().str.contains(name.lower(), regex=False)]
    if partial.empty:
        raise SystemExit(f"No player matched {name!r}.")
    if len(partial) > 1:
        names = ", ".join(partial["name"].head(10).tolist())
        raise SystemExit(f"Multiple players matched {name!r}: {names}")
    return partial.iloc[0]


def _projection_for_market(row: pd.Series, market: str) -> float:
    if market == "player_points":
        ppg = pd.to_numeric(row.get("ppg"), errors="coerce")
        projected = pd.to_numeric(row.get("projectedPoints"), errors="coerce")
        if pd.notna(ppg) and pd.notna(projected) and projected >= 0.6 * ppg:
            return float(projected)
        if pd.notna(ppg):
            return float(ppg)

    mapping = {
        "player_rebounds": ("rpg",),
        "player_assists": ("apg",),
    }
    for column in mapping.get(market, ("projectedPoints", "ppg")):
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            return float(value)
    raise SystemExit(f"No usable projection for {row['name']} market {market}.")


if __name__ == "__main__":
    main()
