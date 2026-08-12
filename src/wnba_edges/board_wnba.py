"""
WNBA adapter for the shared Board kernel (``board.py``).

The kernel is vendored byte-identical from mlb-model and owns the card anatomy, filters,
counters and empty states. This module owns the only basketball-specific decisions:

* **principals** are each side's projected usage leader — the WNBA analogue of MLB's
  starting pitchers: the player the game is most likely to be decided through
* **groups** are Full Game only; the WNBA board offers no first-half split today
* **scores** are projected points, which the model already emits per side

Markets are priced against stored odds when a snapshot exists and fall back to model-only
tiles otherwise, so an empty odds file produces an honest board rather than a fake one.
"""
from __future__ import annotations

import html
import math

import pandas as pd

from .betting import value_layer
from .board import GEM_EDGE_PTS, GEM_STATES, Board, Card, Group, Principal, Side, Tile

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
    """Best available price for `side` plus its paired opposite, for de-vigging."""
    rows = quotes[quotes["market"] == market]
    if rows.empty:
        return None, None, None
    pick = rows[rows["side"] == side]
    other = rows[rows["side"] == opposite]
    if pick.empty:
        return None, None, None
    best = pick.loc[pd.to_numeric(pick["odds"], errors="coerce").idxmax()]
    opposite_odds = None
    if not other.empty:
        opposite_odds = _num(other.loc[pd.to_numeric(other["odds"], errors="coerce").idxmax()]["odds"])
    return _num(best["odds"]), opposite_odds, _num(best.get("line"))


def _priced_tile(label: str, model_prob: float, odds, opposite_odds, detail: str) -> Tile:
    result = value_layer(model_prob, int(odds), int(opposite_odds) if opposite_odds else None)
    if result.implausible:
        # An edge this large is treated as an input error, never as a bet.
        return Tile(
            label=label,
            value=f"{result.edge * 100:+.1f}",
            state=f"{detail} · REVIEW",
            tone="warnc",
            note="Edge implausibly large — inputs suspected.",
            priced=True,
        )
    verdict = result.tier
    return Tile(
        label=label,
        value=f"{result.edge * 100:+.1f}",
        state=f"{detail} · {verdict}".strip(" ·"),
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
        odds, opposite, _ = _best_pair(quotes, "h2h", favored, away if favored == home else home)
        if odds:
            ml_tile = _priced_tile("Moneyline", favored_prob, odds, opposite, favored)
    tiles.append(
        ml_tile
        or _model_tile(
            "Moneyline",
            f"{favored_prob * 100:.0f}%" if win is not None else "—",
            favored,
        )
    )

    # Spread
    spread_tile = None
    if quotes is not None and spread is not None:
        odds, opposite, line = _best_pair(quotes, "spreads", home, away)
        if odds and line is not None:
            # Model cover probability from the projected margin against the posted line.
            cover = _cover_probability(spread, line)
            spread_tile = _priced_tile("Spread", cover, odds, opposite, f"{home} {line:+g}")
    # Unpriced spread reads as a betting line, not a raw margin: the stored column is the
    # projected HOME margin, so a favourite must be shown as a negative number.
    if spread is None:
        spread_model = ("—", "projected")
    elif spread >= 0:
        spread_model = (f"{home} -{spread:.1f}", "projected line")
    else:
        spread_model = (f"{away} -{abs(spread):.1f}", "projected line")
    tiles.append(spread_tile or _model_tile("Spread", *spread_model))

    # Total
    total_tile = None
    if quotes is not None and total is not None:
        odds, opposite, line = _best_pair(quotes, "totals", "Over", "Under")
        if odds and line is not None:
            from .betting import estimate_over_probability

            over = estimate_over_probability(total, line, sigma=_TOTAL_SIGMA)
            side, probability = ("Over", over) if over >= 0.5 else ("Under", 1 - over)
            if side == "Under":
                odds, opposite, _ = _best_pair(quotes, "totals", "Under", "Over")
            if odds:
                total_tile = _priced_tile("Total", probability, odds, opposite, f"{side} {line:g}")
    tiles.append(
        total_tile
        or _model_tile("Total", f"{total:.1f}" if total is not None else "—", "projected")
    )

    priced = sum(1 for tile in tiles if tile.is_priced)
    return Group(
        label="Full game",
        tiles=tuple(tiles),
        tag="fullgame",
        state="Priced" if priced else "No price",
    )


# Game-total spread is far wider than a player prop; this is the team-total scale used
# when converting a projected total into an over probability.
_TOTAL_SIGMA = 12.0


def _cover_probability(projected_home_margin: float, posted_home_line: float) -> float:
    """P(home covers) from the projected margin, on the game-margin scale."""
    from .betting import estimate_over_probability

    # Home covers when margin > -line; reuse the same normal model as totals.
    return estimate_over_probability(projected_home_margin, -posted_home_line, sigma=_TOTAL_SIGMA)


# ── principals ───────────────────────────────────────────────────────────────


def _usage_leaders(features: pd.DataFrame | None, away: str, home: str) -> tuple[Principal, ...]:
    if features is None or features.empty:
        return ()
    principals = []
    for team in (away, home):
        squad = features[features["team"] == team]
        if squad.empty:
            principals.append(Principal(name="—", team=team, stat="no player data"))
            continue
        minutes = pd.to_numeric(squad.get("mpg"), errors="coerce")
        points = pd.to_numeric(squad.get("ppg"), errors="coerce")
        leader = squad.loc[(minutes.fillna(0) * 1.0 + points.fillna(0) * 2.0).idxmax()]
        bits = []
        ppg, usage = _num(leader.get("ppg")), _num(leader.get("usg"))
        if ppg is not None:
            bits.append(f"{ppg:.1f} PPG")
        if usage is not None:
            bits.append(f"{usage:.1f} USG")
        principals.append(
            Principal(name=str(leader["name"]), team=team, stat=" · ".join(bits))
        )
    return tuple(principals)


# ── assembly ─────────────────────────────────────────────────────────────────


def build_card(game: pd.Series, features, odds) -> Card:
    away, home = str(game["away"]), str(game["home"])
    win = _num(game.get("home_win_prob"))
    home_fav = (win or 0.5) >= 0.5
    quotes = _quotes_for(odds, away, home)
    group = _full_game_group(game, quotes)

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
        top = max(priced, key=lambda tile: float(tile.value.rstrip("%") or 0))
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
        principals=_usage_leaders(features, away, home),
        principal_label="Usage leaders",
        groups=(group,),
        action_label="Methodology",
        action_js="location.hash='methodology'",
        footer_label="How this game is modelled",
        footer_js="location.hash='methodology'",
        note="No priced markets — model projections only.",
    )


def build_board(projections, features, odds, *, data_date: str | None = None) -> Board:
    cards = []
    if projections is not None and not projections.empty:
        cards = [build_card(game, features, odds) for _, game in projections.iterrows()]

    basis = ""
    if projections is not None and not projections.empty:
        basis = str(projections.iloc[0].get("win_prob_basis", "") or "")

    meta = [f"{len(cards)} games", str(data_date or "slate pending")]
    if basis:
        meta.append(basis)

    return Board(
        sport="WNBA",
        cards=cards,
        date_label=str(data_date or ""),
        meta=meta,
        filters=[("all", f"All {len(cards)}"), ("gems", "◆ Gems"), ("fullgame", "Full game")],
        sorts=[("start", "Start time"), ("picks", "Priced markets"), ("gems", "Gems")],
        empty_text=(
            "No game projections yet. Run `wnba-edges build-game-projections` after a "
            "season refresh and the board fills in."
        ),
    )
