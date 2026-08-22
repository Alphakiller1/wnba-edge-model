from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .best_bets import TOP_N, build_daily_best_bets, write_daily_best_bets
from .betting import evaluate_over_under
from .espnanalytics import fetch_box, write_box
from .features import MIN_GAMES_FOR_BOARD, MIN_MPG_FOR_BOARD, board_eligible, build_player_features, load_jsonl
from .herhoopstats import fetch_research_table, write_table
from .market_data import MAX_QUOTE_AGE_HOURS, best_price_player_prop
from .predictions import (
    backfill_logged_game_lines,
    grade_games,
    grade_markets,
    grade_props,
    log_game_projections,
    log_market_predictions_batch,
    log_prop_prediction,
    log_prop_predictions_batch,
    results_summary,
)
from .projections import UnknownTeamsError, build_game_projections, load_schedule
from .prop_projections import (
    attach_game_market_lines,
    build_game_market_slate,
    build_slate_prop_projections,
    game_market_slate_path,
    projection_for_market,
    prop_slate_path,
)
from .schedule import fetch_upcoming_schedule, write_schedule
from .season import build_season_tables, overlay_espn_finals, overlay_espn_player_logs, scrape_season_snapshot
from .sigma import fit_market_sigmas, load_market_sigmas, market_sigma_path, resolve_sigma
from .wnbanalytics import scrape_players, write_jsonl

# Repo root when running from a source checkout (editable install); for a
# regular install __file__ resolves into site-packages, so fall back to the
# working directory — CI installs with `pip install .` and runs from the repo.
_PKG_ROOT = Path(__file__).resolve().parents[2]
ROOT = _PKG_ROOT if (_PKG_ROOT / "data").exists() else Path.cwd()
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
    board.add_argument(
        "--include-low-sample",
        action="store_true",
        help=f"Include players under the sample floor (GP<{MIN_GAMES_FOR_BOARD} or MPG<{MIN_MPG_FOR_BOARD:g}).",
    )

    prop = subparsers.add_parser("evaluate-player-prop")
    prop.add_argument("--season", default="2026-27")
    prop.add_argument("--player", required=True)
    prop.add_argument("--market", default="player_points")
    prop.add_argument("--side", choices=["over", "under"], required=True)
    prop.add_argument("--line", type=float, required=True)
    prop.add_argument("--odds", type=int, default=None)
    prop.add_argument("--opposite-odds", type=int, default=None,
                      help="Other side of the same line, for a vig-free implied probability.")
    prop.add_argument("--sigma", type=float, default=None,
                      help="Override the fitted sigma (default: per-market fitted value).")
    prop.add_argument("--use-market", action="store_true")
    prop.add_argument("--allow-stale", action="store_true",
                      help=f"Accept stored quotes older than {MAX_QUOTE_AGE_HOURS:g}h.")
    prop.add_argument("--game-date", default=None,
                      help="Slate date (YYYY-MM-DD) the prop belongs to; defaults to today (UTC).")
    prop.add_argument("--no-log", action="store_true",
                      help="Do not persist this evaluation to the prediction log.")

    season_scrape = subparsers.add_parser("scrape-season")
    season_scrape.add_argument("--season", default="2026-27")
    season_scrape.add_argument("--pause-seconds", type=float, default=0.5)

    season_build = subparsers.add_parser("build-season-tables")
    season_build.add_argument("--season", default="2026-27")

    refresh = subparsers.add_parser("refresh-season")
    refresh.add_argument("--season", default="2026-27")
    refresh.add_argument("--pause-seconds", type=float, default=0.5)

    hhs = subparsers.add_parser("scrape-herhoopstats")
    hhs.add_argument("--research-type", default="player_single_games")
    hhs.add_argument("--min-season", type=int, default=2026)
    hhs.add_argument("--max-season", type=int, default=2026)
    hhs.add_argument("--stats-to-show", choices=["traditional", "advanced", "jump_ball"], default="traditional")

    espn = subparsers.add_parser("scrape-espnanalytics-box")
    espn.add_argument("--id", required=True, help="ESPN Analytics box id, e.g. 20250811-1022500204")

    game_proj = subparsers.add_parser("build-game-projections")
    game_proj.add_argument("--season", default="2026-27")
    game_proj.add_argument("--schedule", default=None)
    game_proj.add_argument("--days", type=int, default=10,
                           help="Days ahead to fetch when no schedule file is given.")

    prop_proj = subparsers.add_parser(
        "build-prop-projections",
        help="Rebuild rotation prop projections for the stored upcoming schedule.",
    )
    prop_proj.add_argument("--season", default="2026-27")

    sched = subparsers.add_parser("fetch-schedule")
    sched.add_argument("--season", default="2026-27")
    sched.add_argument("--days", type=int, default=10)

    sigma_fit = subparsers.add_parser("fit-sigma")
    sigma_fit.add_argument("--season", default="2026-27")

    grade = subparsers.add_parser("grade-predictions")
    grade.add_argument("--season", default="2026-27")

    subparsers.add_parser("results")

    bets = subparsers.add_parser(
        "best-bets",
        help="Rank today's priced sides by calibrated hit likelihood.",
    )
    bets.add_argument("--season", default="2026-27")
    bets.add_argument("--top", type=int, default=TOP_N)

    site = subparsers.add_parser("build-site")
    site.add_argument("--season", default="2026-27")
    site.add_argument("--out", default=None, help="Output HTML path (default docs/index.html).")

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
        low = int(feature_frame["low_sample"].sum())
        print(f"wrote {len(feature_frame)} feature rows to {features_path} ({low} flagged LOW SAMPLE)")

    elif args.command == "edge-board":
        features_path = DATA / "processed" / f"player_features_{args.season}.csv"
        board_path = DATA / "processed" / f"edge_board_{args.season}.csv"
        feature_frame = load_features_csv(features_path)
        if not args.include_low_sample:
            before = len(feature_frame)
            feature_frame = board_eligible(feature_frame)
            excluded = before - len(feature_frame)
            if excluded:
                print(f"excluded {excluded} low-sample players (use --include-low-sample to show them)")
        feature_frame = feature_frame.head(args.top)
        feature_frame.to_csv(board_path, index=False)
        print(feature_frame[["name", "team", "edge_score", "watch_reason"]].to_string(index=False))
        print(f"wrote top {len(feature_frame)} rows to {board_path}")

    elif args.command == "evaluate-player-prop":
        _evaluate_player_prop(args)

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
        overlay_espn_finals(ROOT, args.season)
        overlay_espn_player_logs(ROOT, args.season)
        _fit_sigma(args.season)
        _grade(args.season)

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

    elif args.command == "fetch-schedule":
        frame = fetch_upcoming_schedule(days=args.days)
        path = DATA / "raw" / f"upcoming_schedule_{args.season}.csv"
        write_schedule(frame, path)
        print(f"wrote {len(frame)} upcoming games to {path}")

    elif args.command == "build-game-projections":
        _build_game_projections(args)

    elif args.command == "build-prop-projections":
        _build_prop_projections(args)

    elif args.command == "fit-sigma":
        _fit_sigma(args.season)

    elif args.command == "grade-predictions":
        _grade(args.season)

    elif args.command == "results":
        summary = results_summary(ROOT)
        _print_results(summary)

    elif args.command == "best-bets":
        _write_best_bets(args.season, top=args.top)

    elif args.command == "build-site":
        from .report import build_site

        out = Path(args.out) if args.out else ROOT / "docs" / "index.html"
        path = build_site(ROOT, season=args.season, out=out)
        print(f"wrote dashboard to {path}")


def _load_odds() -> pd.DataFrame | None:
    from .market_data import ODDS_HISTORY_CSV, ODDS_LATEST_CSV

    frames = []
    for path in (ODDS_LATEST_CSV, ODDS_HISTORY_CSV):
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _build_game_projections(args) -> None:
    teams_path = DATA / "processed" / f"teams_season_{args.season}.csv"
    results_path = DATA / "processed" / f"game_results_{args.season}.csv"
    if args.schedule:
        schedule = load_schedule(Path(args.schedule))
    else:
        schedule_path = DATA / "raw" / f"upcoming_schedule_{args.season}.csv"
        print(f"no --schedule given; fetching the next {args.days} days from ESPN")
        schedule = fetch_upcoming_schedule(days=args.days)
        write_schedule(schedule, schedule_path)
    game_results = pd.read_csv(results_path) if results_path.exists() else None
    try:
        projections = build_game_projections(pd.read_csv(teams_path), schedule, game_results)
    except UnknownTeamsError as exc:
        raise SystemExit(f"ERROR: {exc}")
    odds = _load_odds()
    projections = attach_game_market_lines(projections, odds)
    out_path = DATA / "processed" / f"game_projections_{args.season}.csv"
    projections.to_csv(out_path, index=False)
    logged = log_game_projections(ROOT, projections, args.season)
    if not projections.empty:
        basis = projections.iloc[0]["win_prob_basis"]
        home_court = projections.iloc[0]["home_court_pts"]
        with_total = int(pd.to_numeric(projections["book_total_line"], errors="coerce").notna().sum())
        with_spread = int(pd.to_numeric(projections["book_spread_line"], errors="coerce").notna().sum())
        with_ml = int(pd.to_numeric(projections["book_home_ml"], errors="coerce").notna().sum())
        print(f"win prob: {basis} | home court: {home_court} pts")
        print(f"book lines captured: ML {with_ml}/{len(projections)} · spread {with_spread}/{len(projections)} · total {with_total}/{len(projections)}")
    print(f"wrote {len(projections)} game projections to {out_path} ({logged} new logged for grading)")
    _write_game_markets(args.season, projections)
    _write_prop_slate(args.season, schedule, projections, odds)
    _write_best_bets(args.season)


def _build_prop_projections(args) -> None:
    schedule_path = DATA / "raw" / f"upcoming_schedule_{args.season}.csv"
    if not schedule_path.exists():
        raise SystemExit("No upcoming schedule stored. Run build-game-projections first.")
    schedule = load_schedule(schedule_path)
    projections_path = DATA / "processed" / f"game_projections_{args.season}.csv"
    projections = pd.read_csv(projections_path) if projections_path.exists() else pd.DataFrame()
    _write_prop_slate(args.season, schedule, projections, _load_odds())
    _write_best_bets(args.season)


def _write_game_markets(season: str, projections: pd.DataFrame) -> None:
    markets = build_game_market_slate(projections)
    out_path = game_market_slate_path(ROOT, season)
    if markets.empty:
        print("no moneyline/spread/total rows to record")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    markets.to_csv(out_path, index=False)
    logged = log_market_predictions_batch(ROOT, markets, season)
    priced = int(markets["priced"].astype(str).str.lower().isin(["true", "1"]).sum())
    print(
        f"wrote {len(markets)} game-market projections to {out_path} "
        f"({priced} priced against a stored line, {logged} new logged for grading)"
    )


def _write_prop_slate(
    season: str,
    schedule: pd.DataFrame,
    projections: pd.DataFrame,
    odds: pd.DataFrame | None,
) -> None:
    features_path = DATA / "processed" / f"player_features_{season}.csv"
    logs_path = DATA / "processed" / f"player_game_logs_{season}.csv"
    if not features_path.exists():
        print("skipping player-prop projections: player features not built")
        return
    run_id = ""
    generated_at = ""
    if projections is not None and not projections.empty:
        run_id = str(projections.iloc[0].get("run_id") or "")
        generated_at = str(projections.iloc[0].get("generated_at") or "")
    if not run_id:
        run_id = uuid.uuid4().hex[:12]
    if not generated_at:
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    player_logs = pd.read_csv(logs_path) if logs_path.exists() else None
    props = build_slate_prop_projections(
        pd.read_csv(features_path),
        schedule,
        player_logs,
        odds,
        season=season,
        run_id=run_id,
        generated_at=generated_at,
        root=ROOT,
    )
    out_path = prop_slate_path(ROOT, season)
    if props.empty:
        print("no player-prop projections produced (empty rotation or missing priors)")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    props.to_csv(out_path, index=False)
    logged = log_prop_predictions_batch(ROOT, props, season)
    priced = int(props["priced"].astype(str).str.lower().isin(["true", "1"]).sum())
    print(
        f"wrote {len(props)} player-prop projections to {out_path} "
        f"({priced} priced against a stored line, {logged} new logged for grading)"
    )


def _fit_sigma(season: str) -> None:
    logs_path = DATA / "processed" / f"player_game_logs_{season}.csv"
    if not logs_path.exists():
        print(f"skipping fit-sigma: {logs_path} not found (run refresh-season first)")
        return
    sigmas = fit_market_sigmas(pd.read_csv(logs_path))
    out = market_sigma_path(ROOT, season)
    sigmas.to_csv(out, index=False)
    print(f"wrote {len(sigmas)} fitted market sigmas to {out}")
    if not sigmas.empty:
        print(sigmas.to_string(index=False))


def _grade(season: str) -> None:
    logs_path = DATA / "processed" / f"player_game_logs_{season}.csv"
    results_path = DATA / "processed" / f"game_results_{season}.csv"
    odds = _load_odds()
    if odds is not None:
        filled = backfill_logged_game_lines(ROOT, odds)
        if filled:
            print(f"backfilled book lines on {filled} logged game forecast(s)")
    overlay_espn_finals(ROOT, season)
    overlay_espn_player_logs(ROOT, season)
    if logs_path.exists():
        results = pd.read_csv(results_path) if results_path.exists() else None
        prop_result = grade_props(ROOT, pd.read_csv(logs_path), game_results=results)
        print(f"props: {prop_result['graded']} graded, {prop_result['voided']} voided, "
              f"{prop_result['pending']} pending")
    else:
        print("props: skipped (no player game logs)")
    if results_path.exists():
        game_result = grade_games(ROOT, pd.read_csv(results_path))
        print(f"games: {game_result['graded']} graded, {game_result['voided']} voided, "
              f"{game_result['pending']} pending")
        market_result = grade_markets(ROOT, pd.read_csv(results_path))
        print(f"markets: {market_result['graded']} graded, {market_result['voided']} voided, "
              f"{market_result['pending']} pending")
    else:
        print("games: skipped (no game results)")


def _print_results(summary: dict) -> None:
    props = summary.get("props", {})
    print("== Prop record ==")
    for market, record in sorted(props.items()):
        if market.startswith("_"):
            continue
        rate = f"{record['hit_rate']}%" if record.get("hit_rate") is not None else "-"
        mae = f" | MAE {record['mae']}" if record.get("mae") is not None else ""
        print(f"  {market}: {record['wins']}-{record['losses']}-{record['pushes']} ({rate}){mae}")
    print(f"  pending: {props.get('_pending', 0)}")
    reasons = props.get("_reasons") or {}
    for reason, count in reasons.items():
        print(f"    {reason}: {count}")
    games = summary.get("games", {})
    print("== Game projections ==")
    if games.get("n"):
        print(f"  correct={games.get('correct', 0)}/{games['n']} | winner hit rate {games['winner_hit_rate']}% | "
              f"spread MAE {games['spread_mae']} | total MAE {games['total_mae']} | "
              f"Brier {games['brier']}")
        if games.get("spread_n"):
            print(f"  spread sides: {games['spread_correct']}/{games['spread_n']} "
                  f"({games['spread_hit_rate']}%) — model direction, not ATS grading")
        if games.get("total_side_n"):
            print(f"  total sides: {games['total_side_correct']}/{games['total_side_n']} "
                  f"({games['total_side_hit_rate']}%) — vs captured book total")
        print(f"  audit: {games.get('audit_runs', games['n'])} graded run(s), "
              f"{games.get('_logged', games['n'])} total logged prediction(s)")
    print(f"  pending: {games.get('_pending', 0)}")
    markets = summary.get("markets", {})
    print("== Recorded game markets ==")
    for market, record in sorted(markets.items()):
        if market.startswith("_"):
            continue
        rate = f"{record['hit_rate']}%" if record.get("hit_rate") is not None else "-"
        print(f"  {market}: {record['wins']}-{record['losses']}-{record['pushes']} ({rate})")
    print(f"  pending: {markets.get('_pending', 0)} | logged: {markets.get('_logged', 0)}")
    if games.get("spread_ats_n"):
        print(f"  spread ATS: {games['spread_ats_correct']}/{games['spread_ats_n']} "
              f"({games['spread_ats_hit_rate']}%) — vs captured book spread")


def _write_best_bets(season: str, top: int = TOP_N) -> None:
    ranked = build_daily_best_bets(ROOT, season, top=top)
    path = write_daily_best_bets(ROOT, season, ranked)
    if ranked.empty:
        print("daily best bets: no priced sides on today's slate")
        return
    print(f"daily best bets: {len(ranked)} sides for {ranked.iloc[0]['slate_date']} -> {path}")
    _print_best_bets(ranked, len(ranked))


def _print_best_bets(ranked: pd.DataFrame, top: int) -> None:
    print("== Daily best bets ==")
    print("  hit_likelihood blends today's model_prob with shrunk historical hit rate for that market.")
    print("  Research ranking, not betting advice. Unpriced rows are excluded.")
    if ranked is None or ranked.empty:
        print("  (no priced sides on the current slate)")
        return
    show = ranked.head(top).copy()
    show["hit"] = (show["hit_likelihood"] * 100).map(lambda value: f"{value:.1f}%")
    show["model"] = (show["model_prob"] * 100).map(lambda value: f"{value:.1f}%")
    show["hist"] = show.apply(
        lambda row: (
            f"{int(row['hist_wins'])}-{int(row['hist_losses'])}"
            + (f" ({row['hist_hit_rate']}%)" if pd.notna(row["hist_hit_rate"]) else "")
        ),
        axis=1,
    )
    print(show[["rank", "selection", "matchup", "hit", "model", "hist", "book"]].to_string(index=False))


def _evaluate_player_prop(args) -> None:
    features_path = DATA / "processed" / f"player_features_{args.season}.csv"
    feature_frame = load_features_csv(features_path)
    row = _match_player(feature_frame, args.player)
    odds = args.odds
    opposite_odds = args.opposite_odds
    market_note = "manual odds"
    quote_age = None
    if args.use_market:
        price = best_price_player_prop(
            args.player, args.market, args.side, args.line, allow_stale=args.allow_stale
        )
        if price is None and odds is None:
            raise SystemExit("No stored market price found. Pass --odds or fetch odds first.")
        if price is not None:
            odds = price["odds"]
            quote_age = price.get("age_hours")
            if opposite_odds is None:
                opposite_odds = price.get("opposite_odds")
            age_note = f", {quote_age}h old" if quote_age is not None else ""
            market_note = f"{price['book']} best price across {price['n_books']} book(s){age_note}"
    if odds is None:
        raise SystemExit("Pass --odds, or use --use-market after fetching odds.")

    logs_path = DATA / "processed" / f"player_game_logs_{args.season}.csv"
    player_logs = pd.read_csv(logs_path) if logs_path.exists() else None
    projection, projection_basis = _projection_for_market(row, args.market, player_logs)
    if args.sigma is not None:
        sigma, sigma_source = args.sigma, "manual override"
    else:
        sigma, sigma_source = resolve_sigma(
            args.market,
            sigmas=load_market_sigmas(ROOT, args.season),
            player_logs=player_logs,
            player_id=row.get("id"),
        )

    value = evaluate_over_under(
        projection=projection,
        line=args.line,
        odds=odds,
        side=args.side,
        sigma=sigma,
        opposite_odds=opposite_odds,
    )
    implied_label = "implied (vig-free)" if value.vig_free else "implied (incl. vig)"
    print(f"{row['name']} {args.market} {args.side.title()} {args.line} @ {odds:+d}")
    print(f"projection: {projection:.2f} | basis: {projection_basis} | source: {market_note}")
    print(f"sigma: {sigma:.2f} ({sigma_source})")
    print(f"model prob: {value.model_prob * 100:.1f}% | {implied_label}: {value.implied_prob * 100:.1f}%")
    print(f"edge: {value.edge * 100:+.1f} pts | EV/unit: {value.ev_per_unit:+.3f}")
    print(f"fair odds: {value.fair_odds:+d} | tier: {value.tier} | units: {value.units}")
    print(f"verdict: {value.verdict}")
    if bool(row.get("low_sample")):
        print("NOTE: LOW SAMPLE — this player is under the minutes/games floor; treat with caution.")

    if not args.no_log:
        prediction_id = log_prop_prediction(
            ROOT,
            {
                "season": args.season,
                "game_date": args.game_date,
                "player": row["name"],
                "player_id": row.get("id"),
                "market": args.market,
                "side": args.side,
                "line": args.line,
                "odds": odds,
                "opposite_odds": opposite_odds,
                "odds_source": market_note,
                "quote_age_hours": quote_age,
                "projection": round(projection, 2),
                "projection_basis": projection_basis,
                "sigma": round(sigma, 3),
                "sigma_source": sigma_source,
                "model_prob": value.model_prob,
                "implied_prob": value.implied_prob,
                "vig_free": value.vig_free,
                "edge": value.edge,
                "ev_per_unit": value.ev_per_unit,
                "tier": value.tier,
                "verdict": value.verdict,
            },
        )
        print(f"logged prediction {prediction_id[:12]} (grade later with `wnba-edges grade-predictions`)")


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


def _projection_for_market(
    row: pd.Series,
    market: str,
    player_logs: pd.DataFrame | None = None,
) -> tuple[float, str]:
    """CLI wrapper: same estimator as the slate builder, but missing priors are fatal here."""
    projection, basis = projection_for_market(row, market, player_logs)
    if projection is None:
        raise SystemExit(f"No usable {market} projection for {row['name']}.")
    return projection, basis


if __name__ == "__main__":
    main()
