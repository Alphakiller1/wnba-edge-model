"""
WNBA adapter for the shared Board kernel (``board.py``).

The kernel is vendored byte-identical from mlb-model and owns the card anatomy, filters,
counters and empty states. This module owns the only basketball-specific decisions:

* **principals** are each side's biggest role change against the season anchor — the player
  whose line is most likely to be stale. This slot used to hold the usage leader, which is
  the most heavily bet and most efficiently priced name in the game and therefore the least
  useful thing to point a reader at
* **groups** are Full Game, plus a non-market matchup breakdown that makes the efficiency,
  tempo and home-court inputs behind the projection visible on every card
* **scores** are projected points, which the model already emits per side

Markets are priced against stored odds when a snapshot exists and fall back to model-only
tiles otherwise, so an empty odds file produces an honest board rather than a fake one.
"""
from __future__ import annotations

import html
import math
import re

import pandas as pd

from .betting import value_layer
from .board import GEM_EDGE_PTS, GEM_STATES, Board, Card, Group, Principal, Side, Tile
from .prop_projections import MARKET_LABEL

# Kernel gem thresholds are expressed in percentage points; betting.value_layer returns
# edge as a fraction. Converting here keeps one definition of "gem" across all sports.
_GEM_EDGE_FRACTION = GEM_EDGE_PTS / 100.0

_VERDICT_TONE = {
    "Strong": "pos",
    "Standard": "pos",
    "Lean": "warnc",
    "Pass": "mut",
    "REVIEW": "warnc",
}

# Odds API book keys are lowercase slugs; the tile footer only has room for a short mark.
_BOOK_SHORT = {
    "draftkings": "DK",
    "fanduel": "FD",
    "betmgm": "MGM",
    "betrivers": "BR",
    "caesars": "CZR",
    "pointsbetus": "PB",
    "pointsbet": "PB",
    "bovada": "Bovada",
    "betonlineag": "BOL",
    "lowvig": "LowVig",
    "williamhill_us": "WH",
    "wynnbet": "Wynn",
    "unibet_us": "Unibet",
    "fanatics": "Fanatics",
}


# ESPN serves WNBA marks under the internal abbreviation, with two exceptions where their
# slug differs from ours. Falls back to a lettered badge if the image 404s.
_ESPN_SLUG = {"GSV": "gs", "PDX": "por", "LVA": "lv"}


def _logo(abbr: str) -> str:
    slug = _ESPN_SLUG.get(abbr, abbr.lower())
    safe = html.escape(abbr)
    return (
        f'<img src="https://a.espncdn.com/i/teamlogos/wnba/500/{slug}.png" alt="" '
        f'aria-hidden="true" loading="lazy" '
        f'onerror="this.outerHTML=&quot;<span class=bd-side__fallback>{safe}</span>&quot;">'
    )


def _num(value):
    number = pd.to_numeric(value, errors="coerce")
    if number is None or (isinstance(number, float) and math.isnan(number)):
        return None
    return float(number)


def _tip_off(value) -> str:
    """Render the stored UTC tip-off in Eastern time, the league's publishing timezone."""
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    if stamp is None or pd.isna(stamp):
        return ""
    # "%-I" is a glibc extension and raises on Windows, where this is also built locally.
    eastern = stamp.tz_convert("America/New_York")
    return f"{eastern.strftime('%I').lstrip('0') or '12'}:{eastern.strftime('%M %p')} ET"


# ── market pricing ───────────────────────────────────────────────────────────


def _quotes_for(odds: pd.DataFrame | None, away: str, home: str) -> pd.DataFrame | None:
    if odds is None or odds.empty:
        return None
    match = odds[(odds["away"] == away) & (odds["home"] == home)]
    return match if not match.empty else None


def _best_pair(quotes: pd.DataFrame, market: str, side: str, opposite: str):
    """Best available price for ``side`` plus its paired opposite, for de-vigging.

    ``market_data`` deliberately stores provider-independent labels (``ml``, ``spread``,
    ``total``), while the first board implementation still queried The Odds API's raw
    labels.  Accept both forms here so a stored snapshot actually reaches the cards.
    """
    normalized_market = {"h2h": "ml", "spreads": "spread", "totals": "total"}.get(market, market)
    rows = quotes[quotes["market"].astype(str).str.lower() == normalized_market]
    if rows.empty:
        return None, None, None, None
    sides = rows["side"].astype(str).str.lower()
    pick = rows[sides == str(side).lower()]
    other = rows[sides == str(opposite).lower()]
    if pick.empty:
        return None, None, None, None
    best = pick.loc[pd.to_numeric(pick["odds"], errors="coerce").idxmax()]
    opposite_odds = None
    if not other.empty:
        opposite_odds = _num(other.loc[pd.to_numeric(other["odds"], errors="coerce").idxmax()]["odds"])
    return _num(best["odds"]), opposite_odds, _num(best.get("line")), str(best.get("book") or "book")


def _book_short(book: str) -> str:
    key = str(book or "book").strip().lower()
    if key in _BOOK_SHORT:
        return _BOOK_SHORT[key]
    return key.replace("_", " ").title() or "Book"


def _priced_tile(
    label: str,
    model_prob: float,
    odds,
    opposite_odds,
    *,
    model_display: str,
    book_display: str,
    book: str,
) -> Tile:
    result = value_layer(model_prob, int(odds), int(opposite_odds) if opposite_odds else None)
    # Group is already "Model vs market" — repeating "Model" on every value just crowds the
    # number. Keep "edge +X.Y pts" in state so `_tile_edge_value` can still rank headlines.
    market_read = " · ".join(
        part for part in (
            " ".join(piece for piece in (_book_short(book), book_display) if piece),
            f"{int(odds):+d}",
            f"edge {result.edge * 100:+.1f} pts",
        )
        if part
    )
    if result.implausible:
        # An edge this large is treated as an input error, never as a bet.
        return Tile(
            label=label,
            value=model_display,
            state=f"{market_read} · REVIEW",
            tone="warnc",
            note="Edge implausibly large — inputs suspected.",
            priced=True,
        )
    verdict = result.tier
    return Tile(
        label=label,
        value=model_display,
        state=f"{market_read} · {verdict}".strip(" ·"),
        tone=_VERDICT_TONE.get(verdict, "mut"),
        note=f"Fair {result.fair_odds:+d} · {'de-vigged' if result.vig_free else 'raw hold'}",
        gem=result.edge >= _GEM_EDGE_FRACTION and verdict.upper() in {
            state.upper() for state in GEM_STATES
        } | {"STRONG", "STANDARD"},
        priced=True,
    )


def _model_tile(label: str, value: str, detail: str) -> Tile:
    """No matched price: show the model's own number, never counted as a pick."""
    return Tile(
        label=label,
        value=value,
        state=f"{detail} · model only",
        tone="mut",
        note="No stored book price for this market.",
    )


def _full_game_group(game: pd.Series, quotes: pd.DataFrame | None) -> Group:
    away, home = str(game["away"]), str(game["home"])
    win = _num(game.get("home_win_prob"))
    spread = _num(game.get("projected_home_spread"))
    total = _num(game.get("projected_total"))

    favored, favored_prob = (home, win) if (win or 0) >= 0.5 else (away, 1.0 - (win or 0.5))

    tiles: list[Tile] = []

    # Moneyline
    ml_tile = None
    if quotes is not None and win is not None:
        odds, opposite, _, book = _best_pair(quotes, "h2h", favored, away if favored == home else home)
        if odds:
            ml_tile = _priced_tile(
                "Moneyline", favored_prob, odds, opposite,
                model_display=f"{favored} {favored_prob * 100:.0f}%",
                book_display="", book=book,
            )
    tiles.append(
        ml_tile
        or _model_tile(
            "Moneyline",
            f"{favored} {favored_prob * 100:.0f}%" if win is not None else "—",
            "win probability",
        )
    )

    # Spread
    # Unpriced spread reads as a betting line, not a raw margin: the stored column is the
    # projected HOME margin, so a favourite must be shown as a negative number.
    if spread is None:
        spread_model = ("—", "projected")
    elif spread >= 0:
        spread_model = (f"{home} -{spread:.1f}", "projected line")
    else:
        spread_model = (f"{away} -{abs(spread):.1f}", "projected line")
    spread_tile = None
    if quotes is not None and spread is not None:
        odds, opposite, line, book = _best_pair(quotes, "spreads", home, away)
        if odds and line is not None:
            # Model cover probability from the projected margin against the posted line.
            cover = _cover_probability(spread, line)
            spread_tile = _priced_tile(
                "Spread", cover, odds, opposite,
                model_display=spread_model[0], book_display=f"{home} {line:+g}", book=book,
            )
    tiles.append(spread_tile or _model_tile("Spread", *spread_model))

    # Total
    total_tile = None
    if quotes is not None and total is not None:
        odds, opposite, line, book = _best_pair(quotes, "totals", "Over", "Under")
        if odds and line is not None:
            from .betting import estimate_over_probability

            over = estimate_over_probability(total, line, sigma=_TOTAL_SIGMA)
            side, probability = ("Over", over) if over >= 0.5 else ("Under", 1 - over)
            if side == "Under":
                odds, opposite, _, book = _best_pair(quotes, "totals", "Under", "Over")
            if odds:
                total_tile = _priced_tile(
                    "Total", probability, odds, opposite,
                    model_display=f"{total:.1f}", book_display=f"{side} {line:g}", book=book,
                )
    tiles.append(
        total_tile
        or _model_tile("Total", f"{total:.1f}" if total is not None else "—", "projected")
    )

    priced = sum(1 for tile in tiles if tile.is_priced)
    return Group(
        label="Model vs market",
        tiles=tuple(tiles),
        tag="fullgame",
        state="" if priced else "Model baseline",
    )


# Game-total spread is far wider than a player prop; this is the team-total scale used
# when converting a projected total into an over probability.
_TOTAL_SIGMA = 12.0


def _tile_edge_value(tile: Tile) -> float:
    """Recover the labeled market edge for headline ranking without parsing display value."""
    match = re.search(r"edge\s+([+-]?\d+(?:\.\d+)?)\s+pts", tile.state)
    return float(match.group(1)) if match else 0.0


def _cover_probability(projected_home_margin: float, posted_home_line: float) -> float:
    """P(home covers) from the projected margin, on the game-margin scale."""
    from .betting import estimate_over_probability

    # Home covers when margin > -line; reuse the same normal model as totals.
    return estimate_over_probability(projected_home_margin, -posted_home_line, sigma=_TOTAL_SIGMA)


# ── principals ───────────────────────────────────────────────────────────────


def _edge_movers(features: pd.DataFrame | None, away: str, home: str) -> tuple[Principal, ...]:
    """Each side's largest gap between live role and season anchor.

    This slot used to carry the usage leader, which is the same mistake the watchboard made:
    the highest-usage player is the most heavily bet and most efficiently priced name in the
    game, so pointing at her tells a reader nothing they can act on. The player whose minutes
    or usage have moved away from the season baseline is where a season-anchored line is
    most likely to be stale — which is the thing this card exists to point at.
    """
    if features is None or features.empty:
        return ()
    principals = []
    for team in (away, home):
        squad = features[features["team"] == team]
        if "anchor_gap" in squad.columns:
            eligible = squad[~squad["low_sample"].astype(str).str.lower().isin({"true", "1"})]
            squad = eligible if not eligible.empty else squad
        if squad.empty:
            principals.append(Principal(name="—", team=team, stat="no player data"))
            continue
        if "anchor_gap" in squad.columns:
            gaps = pd.to_numeric(squad["anchor_gap"], errors="coerce").abs()
            if gaps.notna().any():
                mover = squad.loc[gaps.idxmax()]
                gap = _num(mover.get("anchor_gap")) or 0.0
                lean = str(mover.get("lean") or "").strip()
                direction = "role up" if gap > 0 else "role down"
                stat = f"{direction} {gap:+.1f}"
                if lean in {"OVER", "UNDER"}:
                    stat += f" · {lean}"
                principals.append(
                    Principal(name=str(mover["name"]), team=team, stat=stat)
                )
                continue
        # No anchor-gap column (older feature file): fall back to the minutes leader rather
        # than rendering an empty slot.
        minutes = pd.to_numeric(squad.get("mpg"), errors="coerce")
        leader = squad.loc[minutes.fillna(0).idxmax()]
        principals.append(
            Principal(name=str(leader["name"]), team=team, stat="no anchor gap computed")
        )
    return tuple(principals)


def _matchup_drivers(game: pd.Series, pace_reference: float | None) -> Group | None:
    """Show the observable inputs that are driving a game's projection.

    Three tiles, same grid as moneyline / spread / total: each side's offense-versus-defense
    matchup, then the net-rating gap with pace and home court as supporting context. Five
    mini-cards in that same grid left a hole and made ratings look like extra betting markets.
    These are not markets and never count as Picks.
    """
    tiles: list[Tile] = []
    away, home = str(game["away"]), str(game["home"])

    win = _num(game.get("home_win_prob"))
    favored = home if win is None or win >= 0.5 else away

    away_ortg, home_drtg = _num(game.get("away_ortg")), _num(game.get("home_drtg"))
    if away_ortg is not None and home_drtg is not None:
        tiles.append(Tile(
            label=f"{away} scoring",
            value=f"{away_ortg:.1f}",
            state=f"ORtg vs {home} {home_drtg:.1f} DRtg",
            tone="mut",
            note="Season offensive rating versus opponent defensive rating; both are points per 100 possessions.",
        ))

    home_ortg, away_drtg = _num(game.get("home_ortg")), _num(game.get("away_drtg"))
    if home_ortg is not None and away_drtg is not None:
        tiles.append(Tile(
            label=f"{home} scoring",
            value=f"{home_ortg:.1f}",
            state=f"ORtg vs {away} {away_drtg:.1f} DRtg",
            tone="mut",
            note="Season offensive rating versus opponent defensive rating; both are points per 100 possessions.",
        ))

    # Pace and home court are supporting context for the net-rating gap, not their own
    # markets. Folding them here keeps the shelf on one even row.
    context: list[str] = []
    pace = _num(game.get("projected_pace"))
    if pace is not None:
        if pace_reference and abs(pace - pace_reference) >= 1.0:
            delta = pace - pace_reference
            direction = "faster" if delta > 0 else "slower"
            context.append(f"{pace:.1f} pace, {abs(delta):.1f} {direction}")
        else:
            context.append(f"{pace:.1f} pace")
    hca = _num(game.get("home_court_pts"))
    if hca is not None:
        context.append(f"{home} +{hca:.1f} home")

    home_net, away_net = _num(game.get("home_net")), _num(game.get("away_net"))
    if home_net is not None and away_net is not None:
        gap = home_net - away_net
        stronger = home if gap >= 0 else away
        tiles.append(Tile(
            label="Net rating",
            value=f"{stronger} +{abs(gap):.1f}",
            state=" · ".join(context) or "per 100 possessions",
            tone="side",
            note="Net rating is offensive rating minus defensive rating, expressed per 100 possessions.",
        ))
    elif context:
        tiles.append(Tile(
            label="Setup",
            value=context[0].split(",")[0],
            state=" · ".join(context[1:]) or "game environment",
            tone="mut",
        ))

    if not tiles:
        return None
    return Group(
        label=f"Why {favored}",
        tiles=tuple(tiles),
        tag="",
        state="",
        market=False,
    )


_PROP_TILES_PER_CARD = 3


def _is_priced_row(row: pd.Series) -> bool:
    return str(row.get("priced", "")).lower() in {"true", "1"}


def _prop_group(game: pd.Series, props: pd.DataFrame | None) -> Group | None:
    """Up to three rotation props for this matchup — priced when a book line exists."""
    if props is None or props.empty:
        return None
    away, home = str(game["away"]).upper(), str(game["home"]).upper()
    match = props[
        (props["away"].astype(str).str.upper() == away)
        & (props["home"].astype(str).str.upper() == home)
    ]
    if match.empty:
        return None
    ranked = match.copy()
    ranked["_priced"] = ranked.apply(_is_priced_row, axis=1)
    ranked["_edge"] = pd.to_numeric(ranked.get("edge"), errors="coerce").abs().fillna(-1)
    ranked = ranked.sort_values(["_priced", "_edge"], ascending=[False, False])

    tiles: list[Tile] = []
    seen_players: set[str] = set()
    for _, row in ranked.iterrows():
        player = str(row["player"])
        if player in seen_players:
            continue
        seen_players.add(player)
        market = str(row["market"])
        market_label = MARKET_LABEL.get(market, market.replace("player_", "").upper())
        last_name = player.split()[-1] if player.strip() else player
        label = f"{last_name} {market_label}"
        projection = _num(row.get("projection"))
        if projection is None:
            continue
        if row["_priced"]:
            model_prob = _num(row.get("model_prob"))
            odds = _num(row.get("odds"))
            line = _num(row.get("line"))
            side = str(row.get("side") or "").title()
            if model_prob is not None and odds is not None:
                tiles.append(
                    _priced_tile(
                        label,
                        model_prob,
                        odds,
                        _num(row.get("opposite_odds")),
                        model_display=f"{projection:.1f}",
                        book_display=f"{side} {line:g}" if line is not None else side,
                        book=str(row.get("book") or "book"),
                    )
                )
            else:
                tiles.append(_model_tile(label, f"{projection:.1f}", f"{market_label.lower()} proj"))
        else:
            tiles.append(_model_tile(label, f"{projection:.1f}", f"{market_label.lower()} proj"))
        if len(tiles) >= _PROP_TILES_PER_CARD:
            break
    if not tiles:
        return None
    priced = sum(1 for tile in tiles if tile.is_priced)
    return Group(
        label="Player props",
        tiles=tuple(tiles),
        tag="props",
        state="" if priced else "Model baseline",
    )


# ── assembly ─────────────────────────────────────────────────────────────────


def build_card(
    game: pd.Series,
    features,
    odds,
    pace_reference: float | None = None,
    props: pd.DataFrame | None = None,
) -> Card:
    away, home = str(game["away"]), str(game["home"])
    win = _num(game.get("home_win_prob"))
    home_fav = (win or 0.5) >= 0.5
    quotes = _quotes_for(odds, away, home)
    group = _full_game_group(game, quotes)
    drivers = _matchup_drivers(game, pace_reference)
    prop_group = _prop_group(game, props)

    def side(abbr: str, points, net, probability) -> Side:
        # Win% only. Net rating was truncating the line at card width, and a clipped
        # "net …" is worse than not showing it — the full ratings live in the matchup page.
        detail = f"{probability * 100:.0f}% win" if probability is not None else ""
        points_value = _num(points)
        return Side(
            abbr=abbr,
            score=f"{points_value:.0f}" if points_value is not None else "—",
            detail=detail,
            logo_html=_logo(abbr),
            favored=(abbr == home) == home_fav,
        )

    priced = [tile for tile in group.tiles if tile.is_priced]
    if priced:
        top = max(priced, key=_tile_edge_value)
        headline = f"{top.label} {top.state}"
        tone = "pos" if top.gem else "side"
    else:
        headline = "Model projection only — no stored price"
        tone = "mut"

    return Card(
        key=f"{away}@{home}",
        league="WNBA",
        start_text=_tip_off(game.get("time")),
        away=side(away, game.get("projected_away_pts"), game.get("away_net"),
                  (1 - win) if win is not None else None),
        home=side(home, game.get("projected_home_pts"), game.get("home_net"), win),
        headline=headline,
        headline_tone=tone,
        principals=_edge_movers(features, away, home),
        principal_label="Biggest role change vs season anchor",
        groups=tuple(g for g in (group, prop_group, drivers) if g is not None),
        action_label="Methodology",
        action_js="location.hash='methodology'",
        footer_label="How this game is modelled",
        footer_js="location.hash='methodology'",
        note="No priced markets — model projections only.",
    )


def build_board(
    projections,
    features,
    odds,
    *,
    data_date: str | None = None,
    props: pd.DataFrame | None = None,
) -> Board:
    cards = []
    if projections is not None and not projections.empty:
        # Slate-mean pace, so "fast game" is stated relative to tonight rather than to a
        # hardcoded league constant that drifts across seasons.
        pace_series = pd.to_numeric(projections.get("projected_pace"), errors="coerce")
        pace_reference = float(pace_series.mean()) if pace_series.notna().any() else None
        cards = [
            build_card(game, features, odds, pace_reference, props)
            for _, game in projections.iterrows()
        ]

    basis = ""
    if projections is not None and not projections.empty:
        basis = str(projections.iloc[0].get("win_prob_basis", "") or "")

    meta = [f"{len(cards)} games", str(data_date or "slate pending")]
    if basis:
        meta.append(basis)
    if props is not None and not props.empty:
        priced = (
            int(props["priced"].astype(str).str.lower().isin(["true", "1"]).sum())
            if "priced" in props.columns else 0
        )
        meta.append(f"{len(props)} prop projections" + (f" · {priced} priced" if priced else ""))

    filters = [("all", f"All {len(cards)}"), ("gems", "◆ Gems"), ("fullgame", "Full game")]
    if any("props" in card.tags for card in cards):
        filters.append(("props", "Player props"))

    return Board(
        sport="WNBA",
        cards=cards,
        date_label=str(data_date or ""),
        meta=meta,
        filters=filters,
        sorts=[("start", "Start time"), ("picks", "Priced markets"), ("gems", "Gems")],
        empty_text=(
            "No game projections yet. Run `wnba-edges build-game-projections` after a "
            "season refresh and the board fills in."
        ),
    )
