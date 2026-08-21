"""Daily best bets: today's priced sides ranked by calibrated hit likelihood.

``hit_likelihood`` is the ranking parameter. It blends the current model's
probability that a side hits with how similar graded forecasts have actually
scored, shrunk so a 1-0 market does not look like a lock.

This is a research ranking of the current slate, not a wager instruction.
Unpriced rows never enter the list: a projection without a captured book line
is not a bet.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .betting import estimate_over_probability
from .predictions import GAME_COLUMNS, PROP_COLUMNS, game_log_path, prop_log_path
from .prop_projections import MARKET_LABEL, game_market_slate_path, prop_slate_path

TOP_N = 10
PRIOR_N = 12.0
MIN_MODEL_PROB = 0.52
MIN_BAND_N = 8
MAX_PER_FAMILY = 4
GAME_SIGMA = 12.0  # same Normal sigma used to price game spread/total sides
PROB_BANDS = (("50–59%", 0.50, 0.60), ("60–69%", 0.60, 0.70), ("70%+", 0.70, 1.01))

BEST_BET_COLUMNS = [
    "rank", "slate_date", "family", "market", "game_date", "away", "home",
    "matchup", "selection", "player", "side", "line", "odds", "book",
    "model_prob", "hist_n", "hist_wins", "hist_losses", "hist_hit_rate",
    "hist_band", "hit_likelihood", "edge", "tier", "verdict",
]


def best_bets_path(root: Path, season: str) -> Path:
    return root / "data" / "processed" / f"daily_best_bets_{season}.csv"


def hit_likelihood(model_prob: float, wins: int, losses: int, prior_n: float = PRIOR_N) -> float:
    """Calibrated P(hit) = blend of today's model_prob and shrunk historical rate.

    Historical rate uses Laplace smoothing ``(wins + 1) / (n + 2)`` so a tiny
    sample cannot print 100%. Weight on history is ``n / (n + prior_n)``: with
    few graded bets the current projection dominates; with a long record the
    model's observed hit rate for that market pulls the number back.
    """
    prob = min(max(float(model_prob), 0.001), 0.999)
    n = max(int(wins) + int(losses), 0)
    shrunk = (int(wins) + 1) / (n + 2)
    weight = n / (n + prior_n)
    return round(weight * shrunk + (1 - weight) * prob, 4)


def probability_band(model_prob: float) -> str:
    for label, low, high in PROB_BANDS:
        if low <= model_prob < high:
            return label
    return "50–59%" if model_prob < 0.50 else "70%+"


def resolve_slate_date(
    game_markets: pd.DataFrame,
    props: pd.DataFrame,
    today: date | None = None,
) -> str | None:
    """Today if the priced slate has games; otherwise the next priced date."""
    today = today or date.today()
    dates: list[str] = []
    for frame in (game_markets, props):
        if frame is None or frame.empty or "game_date" not in frame.columns:
            continue
        priced = _priced(frame)
        dates.extend(str(value)[:10] for value in priced["game_date"].tolist())
    unique = sorted({day for day in dates if day})
    if not unique:
        return None
    stamp = today.isoformat()
    if stamp in unique:
        return stamp
    later = [day for day in unique if day >= stamp]
    return later[0] if later else unique[-1]


def rank_best_bets(
    game_markets: pd.DataFrame,
    props: pd.DataFrame,
    games_log: pd.DataFrame,
    props_log: pd.DataFrame,
    *,
    slate_date: str,
    top: int = TOP_N,
) -> pd.DataFrame:
    """Return the top ``top`` priced sides for ``slate_date``, ranked by hit_likelihood."""
    history = build_market_history(games_log, props_log)
    pieces = [
        frame for frame in (
            _game_candidates(game_markets, slate_date),
            _prop_candidates(props, slate_date),
        )
        if frame is not None and not frame.empty
    ]
    if not pieces:
        return pd.DataFrame(columns=BEST_BET_COLUMNS)
    candidates = pd.concat(pieces, ignore_index=True)

    rows = []
    for _, item in candidates.iterrows():
        model_prob = float(item["model_prob"])
        stats = history.lookup(str(item["market"]), model_prob)
        likelihood = hit_likelihood(model_prob, stats["wins"], stats["losses"])
        rows.append(
            {
                **item.to_dict(),
                "hist_n": stats["n"],
                "hist_wins": stats["wins"],
                "hist_losses": stats["losses"],
                "hist_hit_rate": stats["hit_rate"],
                "hist_band": stats["band"],
                "hit_likelihood": likelihood,
            }
        )
    ranked = pd.DataFrame(rows)
    ranked["edge"] = pd.to_numeric(ranked["edge"], errors="coerce").fillna(0)
    ranked = ranked.sort_values(
        ["hit_likelihood", "model_prob", "edge"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    picked = _diversity_cap(ranked, top=top)
    picked.insert(0, "rank", range(1, len(picked) + 1))
    picked["slate_date"] = slate_date
    return picked[[column for column in BEST_BET_COLUMNS if column in picked.columns]]


def build_daily_best_bets(
    root: Path,
    season: str = "2026-27",
    *,
    today: date | None = None,
    top: int = TOP_N,
) -> pd.DataFrame:
    markets = _read(game_market_slate_path(root, season))
    props = _read(prop_slate_path(root, season))
    slate_date = resolve_slate_date(markets, props, today=today)
    if slate_date is None:
        return pd.DataFrame(columns=BEST_BET_COLUMNS)
    return rank_best_bets(
        markets,
        props,
        _read(game_log_path(root), GAME_COLUMNS),
        _read(prop_log_path(root), PROP_COLUMNS),
        slate_date=slate_date,
        top=top,
    )


def write_daily_best_bets(root: Path, season: str, frame: pd.DataFrame | None = None, **kwargs) -> Path:
    path = best_bets_path(root, season)
    ranked = frame if frame is not None else build_daily_best_bets(root, season, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(path, index=False)
    return path


class MarketHistory:
    """Per-market (and probability-band) graded W-L for calibrating today's slate."""

    def __init__(self, tables: dict[str, dict]):
        self.tables = tables

    def lookup(self, market: str, model_prob: float) -> dict:
        family = self.tables.get(market) or _empty_stats()
        band_label = probability_band(model_prob)
        band = family.get("bands", {}).get(band_label)
        if band and band["n"] >= MIN_BAND_N:
            return {**band, "band": band_label}
        return {**{k: family[k] for k in ("wins", "losses", "n", "hit_rate")}, "band": "all"}


def build_market_history(games_log: pd.DataFrame, props_log: pd.DataFrame) -> MarketHistory:
    tables: dict[str, dict] = {}
    latest = _latest_games(games_log)
    tables["moneyline"] = _flag_stats(latest, "winner_correct", _ml_model_prob)
    tables["spread"] = _flag_stats(latest, "spread_ats_correct", _spread_model_prob)
    tables["total"] = _flag_stats(latest, "total_side_correct", _total_model_prob)

    props = _tracked_props(props_log)
    if not props.empty:
        props = (
            props.sort_values(["game_date", "player", "market", "_recorded_sort"])
            .drop_duplicates(["game_date", "player", "market"], keep="last")
        )
        for market, group in props.groupby("market"):
            tables[str(market)] = _won_stats(group)
    return MarketHistory(tables)


def _latest_games(games: pd.DataFrame) -> pd.DataFrame:
    if games is None or games.empty:
        return pd.DataFrame()
    frame = games.copy()
    settled = frame["settled"].astype(str).str.lower() == "true"
    scored = frame[settled & frame["winner_correct"].astype(str).isin(["True", "False"])].copy()
    if scored.empty:
        return scored
    scored["_recorded_sort"] = pd.to_datetime(scored["recorded_at"], errors="coerce", utc=True)
    return (
        scored.sort_values(["date", "away", "home", "_recorded_sort"])
        .drop_duplicates(["date", "away", "home"], keep="last")
        .copy()
    )


def _tracked_props(props: pd.DataFrame) -> pd.DataFrame:
    if props is None or props.empty:
        return pd.DataFrame()
    frame = props.copy()
    settled = frame["settled"].astype(str).str.lower() == "true"
    reason = frame["ungraded_reason"].fillna("").astype(str).str.strip()
    won = frame["won"].astype(str).isin(["True", "False"])
    tracked = frame[settled & reason.isin(["", "nan", "<NA>"]) & won].copy()
    if tracked.empty:
        return tracked
    tracked["_recorded_sort"] = pd.to_datetime(tracked["recorded_at"], errors="coerce", utc=True)
    tracked["_model_prob"] = pd.to_numeric(tracked["model_prob"], errors="coerce")
    return tracked


def _flag_stats(frame: pd.DataFrame, column: str, prob_fn) -> dict:
    if frame is None or frame.empty or column not in frame.columns:
        return _empty_stats()
    usable = frame[frame[column].astype(str).isin(["True", "False"])].copy()
    if usable.empty:
        return _empty_stats()
    usable["_model_prob"] = usable.apply(prob_fn, axis=1)
    wins = int((usable[column].astype(str) == "True").sum())
    losses = int((usable[column].astype(str) == "False").sum())
    return {
        "wins": wins,
        "losses": losses,
        "n": wins + losses,
        "hit_rate": _rate(wins, losses),
        "bands": _band_table(usable, column),
    }


def _won_stats(group: pd.DataFrame) -> dict:
    wins = int((group["won"].astype(str) == "True").sum())
    losses = int((group["won"].astype(str) == "False").sum())
    return {
        "wins": wins,
        "losses": losses,
        "n": wins + losses,
        "hit_rate": _rate(wins, losses),
        "bands": _band_table(group, "won"),
    }


def _band_table(frame: pd.DataFrame, flag_column: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    probs = pd.to_numeric(frame["_model_prob"], errors="coerce")
    for label, low, high in PROB_BANDS:
        band = frame[(probs >= low) & (probs < high)]
        if band.empty:
            continue
        wins = int((band[flag_column].astype(str) == "True").sum())
        losses = int((band[flag_column].astype(str) == "False").sum())
        out[label] = {
            "wins": wins,
            "losses": losses,
            "n": wins + losses,
            "hit_rate": _rate(wins, losses),
        }
    return out


def _ml_model_prob(row: pd.Series) -> float | None:
    probability = pd.to_numeric(row.get("home_win_prob"), errors="coerce")
    if pd.isna(probability):
        return None
    return float(max(probability, 1 - probability))


def _spread_model_prob(row: pd.Series) -> float | None:
    projected = pd.to_numeric(row.get("projected_home_spread"), errors="coerce")
    book = pd.to_numeric(row.get("book_spread_line"), errors="coerce")
    if pd.isna(projected) or pd.isna(book):
        return None
    home_covers = estimate_over_probability(float(projected), -float(book), sigma=GAME_SIGMA)
    cover = float(projected) + float(book)
    if cover == 0:
        return 0.5
    return float(home_covers if cover > 0 else 1.0 - home_covers)


def _total_model_prob(row: pd.Series) -> float | None:
    projected = pd.to_numeric(row.get("projected_total"), errors="coerce")
    book = pd.to_numeric(row.get("book_total_line"), errors="coerce")
    if pd.isna(projected) or pd.isna(book):
        return None
    over_p = estimate_over_probability(float(projected), float(book), sigma=GAME_SIGMA)
    if float(projected) == float(book):
        return 0.5
    return float(over_p if projected > book else 1.0 - over_p)


def _game_candidates(markets: pd.DataFrame, slate_date: str) -> pd.DataFrame:
    if markets is None or markets.empty:
        return pd.DataFrame()
    frame = _priced(markets)
    frame = frame[frame["game_date"].astype(str).str[:10] == slate_date]
    side = frame["side"].astype(str)
    frame = frame[side.str.len().gt(0) & ~side.str.upper().isin(["PUSH", "NAN", "NONE"])]
    frame = frame.copy()
    frame["model_prob"] = pd.to_numeric(frame["model_prob"], errors="coerce")
    frame = frame[frame["model_prob"] >= MIN_MODEL_PROB]
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "family": str(row["market"]),
                "market": str(row["market"]),
                "game_date": str(row["game_date"])[:10],
                "away": str(row["away"]).upper(),
                "home": str(row["home"]).upper(),
                "matchup": f"{str(row['away']).upper()} @ {str(row['home']).upper()}",
                "selection": _game_selection(row),
                "player": "",
                "side": str(row["side"]),
                "line": pd.to_numeric(row.get("line"), errors="coerce"),
                "odds": pd.to_numeric(row.get("odds"), errors="coerce"),
                "book": str(row.get("book") or ""),
                "model_prob": float(row["model_prob"]),
                "edge": pd.to_numeric(row.get("edge"), errors="coerce"),
                "tier": str(row.get("tier") or ""),
                "verdict": str(row.get("verdict") or ""),
            }
        )
    return pd.DataFrame(rows)


def _prop_candidates(props: pd.DataFrame, slate_date: str) -> pd.DataFrame:
    if props is None or props.empty:
        return pd.DataFrame()
    frame = _priced(props)
    frame = frame[frame["game_date"].astype(str).str[:10] == slate_date]
    frame = frame.copy()
    frame["model_prob"] = pd.to_numeric(frame["model_prob"], errors="coerce")
    frame = frame[frame["model_prob"] >= MIN_MODEL_PROB]
    side = frame["side"].astype(str).str.lower()
    frame = frame[side.isin(["over", "under"])]
    rows = []
    for _, row in frame.iterrows():
        market = str(row["market"])
        label = MARKET_LABEL.get(market, market)
        line = pd.to_numeric(row.get("line"), errors="coerce")
        player = str(row.get("player") or "")
        side_label = str(row["side"]).upper()
        line_txt = f"{float(line):g}" if pd.notna(line) else ""
        rows.append(
            {
                "family": "prop",
                "market": market,
                "game_date": str(row["game_date"])[:10],
                "away": str(row["away"]).upper(),
                "home": str(row["home"]).upper(),
                "matchup": f"{str(row['away']).upper()} @ {str(row['home']).upper()}",
                "selection": f"{player} {side_label} {line_txt} {label}".strip(),
                "player": player,
                "side": side_label,
                "line": line,
                "odds": pd.to_numeric(row.get("odds"), errors="coerce"),
                "book": str(row.get("book") or ""),
                "model_prob": float(row["model_prob"]),
                "edge": pd.to_numeric(row.get("edge"), errors="coerce"),
                "tier": str(row.get("tier") or ""),
                "verdict": str(row.get("verdict") or ""),
            }
        )
    return pd.DataFrame(rows)


def _game_selection(row: pd.Series) -> str:
    market = str(row["market"])
    side = str(row["side"])
    line = pd.to_numeric(row.get("line"), errors="coerce")
    home = str(row["home"]).upper()
    away = str(row["away"]).upper()
    if market == "moneyline":
        return f"{side} ML"
    if market == "spread" and pd.notna(line):
        pick_line = float(line) if side == home else -float(line)
        return f"{side} {pick_line:+.1f}"
    if market == "total" and pd.notna(line):
        return f"{side.title()} {float(line):g}"
    if market == "spread":
        return f"{side} spread"
    return f"{away} @ {home} {market}"


def _diversity_cap(ranked: pd.DataFrame, top: int) -> pd.DataFrame:
    counts: dict[str, int] = {}
    kept = []
    for idx, row in ranked.iterrows():
        family = str(row["family"])
        if counts.get(family, 0) >= MAX_PER_FAMILY:
            continue
        counts[family] = counts.get(family, 0) + 1
        kept.append(idx)
        if len(kept) >= top:
            break
    return ranked.loc[kept].reset_index(drop=True)


def _priced(frame: pd.DataFrame) -> pd.DataFrame:
    if "priced" not in frame.columns:
        return frame.iloc[0:0]
    mask = frame["priced"].astype(str).str.lower().isin(["true", "1"])
    return frame.loc[mask].copy()


def _rate(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n <= 0:
        return None
    return round(wins / n * 100, 1)


def _empty_stats() -> dict:
    return {"wins": 0, "losses": 0, "n": 0, "hit_rate": None, "bands": {}}


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    frame = pd.read_csv(path)
    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        return frame[columns]
    return frame
