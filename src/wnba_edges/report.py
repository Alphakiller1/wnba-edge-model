"""Self-contained static dashboard for GitHub Pages — Chase Analytics design contract.

Visual identity is vendored verbatim from the Chase Analytics design system
(mlbma-pipeline / mlb-model `chase_tokens.css`): deep-navy surfaces, violet brand
family, gold eyebrow labels, DM Sans body + Roboto Condensed display + Oswald
wordmark, glassy boards with violet glow. Do not invent new token values here.

Layers are explicitly separated so a visitor can always tell what they are
looking at: (1) model projections, (2) market snapshot, (3) edge watchboard
(a ranked review queue, not priced bets), and (4) graded results. Every layer
renders an honest empty state when its data does not exist yet.
"""
from __future__ import annotations

import html
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .board import BOARD_JS, board_html
from .board_wnba import build_board
from .features import MIN_GAMES_FOR_BOARD, MIN_MPG_FOR_BOARD, board_eligible
from .predictions import results_summary
from .sigma import load_market_sigmas

_STATIC = Path(__file__).resolve().parent / "static"

# One font load for the whole document. Identical to mlb-model's chase_theme._FONT_IMPORT.
_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800&"
    "family=Oswald:ital,wght@0,600;0,700;0,900;1,600;1,700;1,900&"
    "family=Roboto+Condensed:wght@400;500;600;700;800&display=swap');"
)

# Names this page used before the tokens were vendored. They alias onto the canonical
# tokens rather than carrying their own values — a second literal is how a product drifts
# off-brand. Never give one of these a colour of its own.
_LOCAL_ALIASES = """
:root{
  --sans:var(--font-primary);
  --display:var(--font-display);
  --wordmark:var(--font-wordmark);
  --card-shadow:var(--ca-card-shadow);
  --glow:var(--ca-glow-violet);
  --board-top:var(--ca-board-top);
  --board-bottom:var(--ca-board-bottom);
}
"""


def brand_css() -> str:
    """Fonts + chase_tokens.css + the board kernel — the shared Chase identity.

    chase_tokens.css and board.css are vendored byte-identical from mlb-model; a drift test
    (tests/test_board_contract.py) fails the build if they diverge. That is what keeps the
    MLB, WNBA and NFL products looking like one brand.
    """
    tokens = (_STATIC / "chase_tokens.css").read_text(encoding="utf-8")
    board = (_STATIC / "board.css").read_text(encoding="utf-8")
    return _FONT_IMPORT + tokens + _LOCAL_ALIASES + board


def esc(value) -> str:
    return html.escape(str(value))


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame if not frame.empty else None


def build_site(root: Path, season: str, out: Path) -> Path:
    processed = root / "data" / "processed"
    projections = _read_csv(processed / f"game_projections_{season}.csv")
    features = _read_csv(processed / f"player_features_{season}.csv")
    game_results = _read_csv(processed / f"game_results_{season}.csv")
    player_logs = _read_csv(processed / f"player_game_logs_{season}.csv")
    odds = _read_csv(root / "data" / "odds" / "odds_latest.csv")
    sigmas = load_market_sigmas(root, season)
    summary = results_summary(root)

    data_date, stale_days = _freshness(game_results, projections)
    built_at = datetime.now(timezone.utc).strftime("%b %d · %H:%M UTC")
    fresh = stale_days is None or stale_days <= 1

    body = f"""
{_nav(fresh, data_date)}
<header class="hero">
  <div class="wrap">
    <div class="hero-eyebrow"><span class="hero-eyebrow-dot"></span>CHASE ANALYTICS&ensp;|&ensp;WNBA INTELLIGENCE</div>
    <h1 class="hero-title">The WNBA slate,<br>modeled and graded.</h1>
    <p class="hero-sub">Game projections, a player watchboard, market snapshots, and a fully
    graded prediction record &mdash; fit on real outcomes, honest about uncertainty.
    Research software, not betting advice.</p>
    <div class="hero-meta">
      <span class="pill">data through {esc(data_date or "n/a")}</span>
      {_stale_badge(stale_days)}
      <span class="pill pill-dim">built {esc(built_at)}</span>
    </div>
    {_summary_tiles(features, game_results, player_logs, projections)}
  </div>
</header>
<main class="wrap">
  {_projections_section(projections, features, odds, data_date)}
  {_market_section(odds)}
  {_board_section(features)}
  {_results_section(summary)}
  {_methodology_section(sigmas)}
</main>
<footer>
  <div class="wrap">
    <p><b>Chase Analytics &mdash; WNBA Edge Model</b> is research and analytics software.
    It does not provide betting advice, does not guarantee outcomes, and no output is a wager
    instruction. Model limitations are documented in the
    <a href="https://github.com/Alphakiller1/wnba-edge-model/blob/main/METHODOLOGY.md">methodology</a>.
    If you or someone you know has a gambling problem, call 1-800-GAMBLER.</p>
    <p class="foot-links">
      <a href="https://github.com/Alphakiller1/wnba-edge-model">Source</a><span>&middot;</span>
      <a href="https://alphakiller1.github.io/mlb-model/">MLB Model</a><span>&middot;</span>
      <a href="https://chase-analytics.com/">chase-analytics.com</a>
    </p>
  </div>
</footer>
"""
    document = (_TEMPLATE.replace("__TOKENS__", brand_css())
            .replace("__BODY__", body)
            .replace("__SCRIPT__", BOARD_JS))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out


# ── chrome ────────────────────────────────────────────────────────────────────

_LOGO_SVG = (
    '<svg viewBox="0 0 36 36" width="30" height="30" aria-hidden="true">'
    '<path d="M18 5 C21 13 24 20 33 31 L3 31 C12 20 15 13 18 5 Z" fill="#7C4DFF"/>'
    "</svg>"
)


def _nav(fresh: bool, data_date: str | None) -> str:
    dot = "dot-ok" if fresh else "dot-warn"
    label = esc(data_date or "no data")
    links = "".join(
        f'<a class="nav-link" href="#{anchor}">{text}</a>'
        for anchor, text in (
            ("projections", "Projections"),
            ("market", "Market"),
            ("watchboard", "Watchboard"),
            ("results", "Results"),
            ("methodology", "Methodology"),
        )
    )
    return f"""
<header class="chase-header">
  <nav class="chase-nav wrap">
    <a href="https://chase-analytics.com" class="chase-logo" title="Chase Analytics">
      {_LOGO_SVG}
      <span class="chase-wordmark">CHASE&nbsp;<em>ANALYTICS</em></span>
    </a>
    <div class="nav-links">{links}</div>
    <div class="chase-status">
      <span class="product-tag">WNBA MODEL</span>
      <span class="chase-timestamp"><span class="pipeline-dot {dot}"></span>{label}</span>
    </div>
  </nav>
</header>"""


def _freshness(game_results: pd.DataFrame | None, projections: pd.DataFrame | None):
    dates = []
    if game_results is not None and "date" in game_results.columns:
        dates.append(str(game_results["date"].max())[:10])
    if projections is not None and "generated_at" in projections.columns:
        dates.append(str(projections["generated_at"].max())[:10])
    if not dates:
        return None, None
    latest = max(dates)
    try:
        stale_days = (date.today() - date.fromisoformat(latest)).days
    except ValueError:
        stale_days = None
    return latest, stale_days


def _stale_badge(stale_days) -> str:
    if stale_days is None or stale_days <= 1:
        return '<span class="pill pill-ok">fresh</span>'
    return f'<span class="pill pill-warn">STALE &mdash; last refresh {stale_days} days ago</span>'


def _summary_tiles(features, game_results, player_logs, projections) -> str:
    tiles = [
        ("Players modeled", len(features) if features is not None else 0),
        ("Finished games", len(game_results) if game_results is not None else 0),
        ("Player game logs", len(player_logs) if player_logs is not None else 0),
        ("Games projected", len(projections) if projections is not None else 0),
    ]
    cells = "".join(
        f'<div class="tile"><span class="tile-v">{value:,}</span><span class="tile-l">{esc(label)}</span></div>'
        for label, value in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _section_head(anchor: str, number: str, title: str, kicker: str, blurb: str) -> str:
    return f"""
<div class="sec-head" id="{anchor}">
  <div class="sec-eyebrow">{esc(kicker)} &middot; {esc(number)} / 4</div>
  <h2 class="sec-title">{esc(title)}</h2>
  <p class="sec-blurb">{blurb}</p>
</div>"""


# ── layer 1 · projections ────────────────────────────────────────────────────

def _projections_section(projections, features=None, odds=None, data_date=None) -> str:
    """Layer 1 — the slate board.

    Renders through the shared Board kernel, so a WNBA game card has the same anatomy as an
    MLB or NFL one: status, expected score, principals, then the market groups it prices.
    """
    head = _section_head(
        "projections", "1", "Game Projections", "Model layer",
        "Baseline score, spread, total and pace from team efficiency ratings. "
        "The home-win probability is a logistic fit on this season's finished games "
        "&mdash; the basis and sample size are always shown.",
    )
    if projections is None:
        return f"""<section>{head}<div class="empty">No game projections yet. Run
        <code>wnba-edges build-game-projections</code> after a season refresh.</div></section>"""
    hc = esc(projections.iloc[0].get("home_court_pts", ""))
    board = build_board(projections, features, odds, data_date=data_date)
    return f"""<section>{head}
<p class="basis-note">Home court: {hc} pts &nbsp;&middot;&nbsp; a market tile only counts as a
pick when a stored book price backs it; everything else is labelled model&nbsp;only.</p>
{board_html(board)}</section>"""


def _projection_card(game: pd.Series) -> str:
    win = pd.to_numeric(game.get("home_win_prob"), errors="coerce")
    home_fav = pd.notna(win) and win >= 0.5
    win_pct = float(win) * 100 if pd.notna(win) else None
    away_pct = 100 - win_pct if win_pct is not None else None
    spread = pd.to_numeric(game.get("projected_home_spread"), errors="coerce")
    spread_txt = f"{spread:+.1f}" if pd.notna(spread) else "&ndash;"
    time_txt = _tipoff(str(game.get("time", "")))
    fav_side = "home" if home_fav else "away"
    prob_label = (
        f'{esc(game["home" if home_fav else "away"])} {(win_pct if home_fav else away_pct):.0f}%'
        if win_pct is not None else "&ndash;"
    )
    bar = ""
    if win_pct is not None:
        bar = f"""
  <div class="wp-row" title="Home win probability {win_pct:.0f}%">
    <span class="wp-team{' is-fav' if not home_fav else ''}">{esc(game["away"])}</span>
    <div class="wp-track"><i class="wp-fill" style="width:{win_pct:.1f}%"></i><s class="wp-mid"></s></div>
    <span class="wp-team{' is-fav' if home_fav else ''}">{esc(game["home"])}</span>
  </div>"""
    return f"""
<article class="mx-card">
  <div class="mx-top"><span class="mx-date">{esc(game.get("date", ""))}</span><span class="mx-time">{esc(time_txt)}</span></div>
  <div class="mx-teams">
    <div class="mx-team{' is-fav' if not home_fav else ''}">
      <span class="mx-abbr">{esc(game["away"])}</span>
      <span class="mx-pts">{esc(game["projected_away_pts"])}</span>
      <span class="mx-net">net {esc(game.get("away_net", ""))}</span>
    </div>
    <div class="mx-at">@</div>
    <div class="mx-team{' is-fav' if home_fav else ''}">
      <span class="mx-abbr">{esc(game["home"])}</span>
      <span class="mx-pts">{esc(game["projected_home_pts"])}</span>
      <span class="mx-net">net {esc(game.get("home_net", ""))}</span>
    </div>
  </div>
  {bar}
  <div class="mx-stats">
    <div class="mx-stat mx-stat--{fav_side}"><i>Win</i><b>{prob_label}</b></div>
    <div class="mx-stat"><i>Spread</i><b>{spread_txt}</b></div>
    <div class="mx-stat"><i>Total</i><b>{esc(game["projected_total"])}</b></div>
    <div class="mx-stat"><i>Pace</i><b>{esc(game["projected_pace"])}</b></div>
  </div>
</article>"""


def _tipoff(raw: str) -> str:
    """Compact tip-off label from either an ISO instant or a preformatted string."""
    raw = raw.strip()
    if "T" in raw:
        try:
            when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return when.strftime("%H:%M UTC")
        except ValueError:
            pass
    return raw[:16]


# ── layer 2 · market ─────────────────────────────────────────────────────────

def _market_section(odds: pd.DataFrame | None) -> str:
    head = _section_head(
        "market", "2", "Market Snapshot", "Market layer",
        "Stored odds with book attribution and quote timestamps. Edges are only ever priced "
        "against de-vigged market probabilities from these snapshots &mdash; a model number "
        "without a market price is a projection, not an edge.",
    )
    if odds is None:
        return f"""<section>{head}<div class="empty"><b>No odds snapshot stored.</b><br>
        Fetch one with <code>python -m wnba_edges.market_data --fetch-game AWY@HOM --props</code>
        (requires <code>ODDS_API_KEY</code>). Nothing on this page pretends to be an edge
        without a price.</div></section>"""
    fetched = odds["fetched_at"].astype(str).max()
    by_market = odds.groupby("market").size().sort_values(ascending=False)
    rows = "".join(
        f'<tr><td>{esc(market)}</td><td class="num">{count}</td></tr>'
        for market, count in by_market.items()
    )
    books = odds["book"].nunique()
    return f"""<section>{head}
<p class="basis-note">Latest snapshot {esc(fetched)} &nbsp;&middot;&nbsp; {books} book(s) &nbsp;&middot;&nbsp; {len(odds)} quotes</p>
<div class="board"><div class="tablewrap"><table>
<thead><tr><th>Market</th><th class="num">Quotes</th></tr></thead>
<tbody>{rows}</tbody></table></div></div></section>"""


# ── layer 3 · watchboard ─────────────────────────────────────────────────────

def _board_section(features: pd.DataFrame | None) -> str:
    head = _section_head(
        "watchboard", "3", "Stale Anchor Board", "Review queue",
        "Players whose <b>current role has moved away from the season baseline a prop line "
        "is set on</b> &mdash; ranked by the size of that gap, not by how good the player is. "
        "High-usage stars are penalised here, not rewarded: they carry the most betting "
        "attention and are the most efficiently priced names on the slate. Research "
        "candidates to price against a market line, never bets by themselves. Players under "
        f"the sample floor (GP &lt; {MIN_GAMES_FOR_BOARD} or MPG &lt; {MIN_MPG_FOR_BOARD:g}) "
        "are excluded.",
    )
    if features is None:
        return f"""<section>{head}<div class="empty">No feature board yet. Run
        <code>wnba-edges build-features</code> after a season refresh.</div></section>"""
    total = len(features)
    eligible = board_eligible(features)
    excluded = total - len(eligible)
    top = eligible.head(15)
    max_score = max(
        (float(row["edge_score"]) for _, row in top.iterrows() if pd.notna(row["edge_score"])),
        default=1.0,
    ) or 1.0
    rows = []
    for i, (_, row) in enumerate(top.iterrows()):
        score = float(row["edge_score"])
        width = max(4.0, score / max_score * 100)
        gap = pd.to_numeric(row.get("anchor_gap"), errors="coerce")
        gap_cell = f"{gap:+.1f}" if pd.notna(gap) else "&ndash;"
        lean = str(row.get("lean") or "").strip()
        if lean == "OVER":
            lean_cell = '<span class="lean lean-over">&#9650; Over</span>'
        elif lean == "UNDER":
            lean_cell = '<span class="lean lean-under">&#9660; Under</span>'
        else:
            lean_cell = '<span class="dim">&ndash;</span>'
        rows.append(f"""<tr>
<td class="rank">{i + 1:02d}</td>
<td class="player">{esc(row["name"])}<span class="player-sub">{esc(row["team"])} &middot; {esc(row.get("pos", "-"))}</span></td>
<td class="scorebar"><div class="sb-track"><i style="width:{width:.1f}%"></i></div></td>
<td class="num score">{score:.2f}</td>
<td class="num">{gap_cell}</td>
<td>{lean_cell}</td>
<td class="num dim">{esc(row.get("ppg", ""))}</td>
<td class="num dim">{esc(row.get("mpg", ""))}</td>
<td class="reason">{_reason_chips(str(row["watch_reason"]))}</td>
</tr>""")
    return f"""<section>{head}
<p class="basis-note">{len(eligible)} of {total} players pass the sample floor &nbsp;&middot;&nbsp; {excluded} excluded as low-sample
&nbsp;&middot;&nbsp; <b>Gap</b> is the signed distance between live role and season anchor; positive means the
season-anchored line is more likely to be set too low.</p>
<div class="board"><div class="tablewrap"><table class="wb">
<thead><tr><th></th><th>Player</th><th>Anchor gap</th><th class="num">Score</th>
<th class="num">Gap</th><th>Lean</th>
<th class="num">PPG</th><th class="num">MPG</th><th>Why the anchor may be stale</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div></div></section>"""


def _reason_chips(reason: str) -> str:
    parts = [p.strip() for p in reason.split(",") if p.strip()]
    return "".join(
        f'<span class="rchip{" rchip-warn" if p.lower() in {"low confidence", "low sample"} else ""}">{esc(p)}</span>'
        for p in parts[:4]
    )


# ── layer 4 · results ────────────────────────────────────────────────────────

def _results_section(summary: dict) -> str:
    head = _section_head(
        "results", "4", "Graded Results", "Results layer",
        "Every logged prediction is graded against finished games; ungradeable rows carry an "
        "explicit reason code. An empty record means nothing has resolved yet &mdash; never "
        "hidden losses.",
    )
    props = {k: v for k, v in summary.get("props", {}).items() if not k.startswith("_")}
    games = summary.get("games", {})
    prop_records = summary.get("props", {}).get("_records", [])
    game_records = games.get("_records", [])
    has_games = bool(games.get("n"))
    if not props and not has_games and not prop_records and not game_records:
        pending = summary.get("props", {}).get("_pending", 0) + games.get("_pending", 0)
        pending_note = (
            f'<span class="pill pill-dim">{pending} prediction(s) awaiting grading</span>'
            if pending else ""
        )
        return f"""<section>{head}<div class="empty"><b>No graded predictions yet.</b> {pending_note}<br>
        Predictions logged by <code>evaluate-player-prop</code> and <code>build-game-projections</code>
        are graded by <code>wnba-edges grade-predictions</code> once games finish.</div></section>"""
    prop_rows = "".join(
        f"""<tr><td>{esc(market)}</td>
<td class="num">{record["wins"]}-{record["losses"]}-{record["pushes"]}</td>
<td class="num">{esc(record["hit_rate"]) if record["hit_rate"] is not None else "&ndash;"}%</td></tr>"""
        for market, record in sorted(props.items())
    )
    prop_table = (
        f"""<div class="board"><div class="tablewrap"><table>
<thead><tr><th>Market</th><th class="num">W-L-P</th><th class="num">Hit rate</th></tr></thead>
<tbody>{prop_rows}</tbody></table></div></div>"""
        if props else ""
    )
    game_block = ""
    if has_games:
        pending = int(games.get("_pending", 0))
        audit_runs = int(games.get("audit_runs", games["n"]))
        logged = int(games.get("_logged", games["n"]))
        bands = games.get("confidence_bands", [])
        band_rows = "".join(
            f"<tr><td>{esc(band['band'])}</td><td class=\"num\">{band['correct']}-{band['n'] - band['correct']}</td>"
            f"<td class=\"num\">{band['hit_rate']}%</td></tr>"
            for band in bands
        )
        calibration = (
            f"""<div class=\"board\"><div class=\"tablewrap\"><table>
<thead><tr><th>Model confidence</th><th class=\"num\">W-L</th><th class=\"num\">Hit rate</th></tr></thead>
<tbody>{band_rows}</tbody></table></div></div>"""
            if band_rows else ""
        )
        recent_rows = []
        for record in games.get("recent", []):
            status = "✓ Correct" if record["correct"] else "× Miss"
            row_class = "result-hit" if record["correct"] else "result-miss"
            confidence = f"{record['favorite']} {record['probability']:.0f}%" if record["probability"] is not None else "—"
            spread_error = f"{abs(record['spread_error']):.1f}" if record["spread_error"] is not None else "—"
            total_error = f"{abs(record['total_error']):.1f}" if record["total_error"] is not None else "—"
            recent_rows.append(
                f"<tr class=\"{row_class}\"><td>{esc(record['date'])}</td><td>{esc(record['matchup'])}</td>"
                f"<td>{esc(confidence)}</td><td>{esc(record['actual_winner'])}</td>"
                f"<td>{status}</td><td class=\"num\">{spread_error}</td><td class=\"num\">{total_error}</td></tr>"
            )
        recent_table = (
            f"""<div class=\"board\"><div class=\"tablewrap\"><table>
<thead><tr><th>Date</th><th>Matchup</th><th>Model call</th><th>Winner</th><th>Result</th>
<th class=\"num\">Spread error</th><th class=\"num\">Total error</th></tr></thead>
<tbody>{''.join(recent_rows)}</tbody></table></div></div>"""
            if recent_rows else ""
        )
        game_block = f"""
<div class="tiles tiles-results">
  <div class="tile"><span class="tile-v">{games["correct"]}/{games["n"]}</span><span class="tile-l">Correct winner calls</span></div>
  <div class="tile"><span class="tile-v">{games["winner_hit_rate"]}%</span><span class="tile-l">Winner hit rate</span></div>
  <div class="tile"><span class="tile-v">{games.get("spread_correct", 0)}/{games.get("spread_n", 0)}</span><span class="tile-l">Correct spread sides</span></div>
  <div class="tile"><span class="tile-v">{games.get("spread_hit_rate", "—")}%</span><span class="tile-l">Spread-side hit rate</span></div>
  <div class="tile"><span class="tile-v">{games["spread_mae"]}</span><span class="tile-l">Spread MAE</span></div>
  <div class="tile"><span class="tile-v">{games["total_mae"]}</span><span class="tile-l">Total MAE</span></div>
  <div class="tile"><span class="tile-v">{games["brier"]}</span><span class="tile-l">Brier score</span></div>
</div>"""
        game_block += f"""
<p class="basis-note">Headline record uses the latest forecast for each matchup: {games["n"]} graded games from
{audit_runs} graded projection run(s) &middot; {logged} total predictions logged &middot; {pending} awaiting grading.</p>
<h3 class="results-subtitle">Where the model is succeeding</h3>{calibration}
<h3 class="results-subtitle">Recent graded game calls</h3>{recent_table}"""
    return f"""<section>{head}{game_block}{prop_table}
{_game_audit_table(game_records)}{_prop_audit_table(prop_records)}</section>"""


def _audit_status_cell(record: dict) -> str:
    tone = {"Correct": "result-hit", "Miss": "result-miss", "Pending": "result-pending", "Voided": "result-void"}.get(
        record.get("status"), ""
    )
    detail = f'<span class="player-sub">{esc(record["status_detail"])}</span>' if record.get("status_detail") else ""
    return f'<span class="{tone}">{esc(record.get("status", "—"))}</span>{detail}'


def _audit_number(value, digits: int = 1, *, suffix: str = "") -> str:
    return f"{float(value):.{digits}f}{suffix}" if value is not None else "—"


def _game_audit_table(records: list[dict]) -> str:
    if not records:
        return ""
    rows = "".join(
        f"<tr><td>{esc(record['date'])}<span class=\"player-sub\">{esc(record['recorded_at'])}</span></td>"
        f"<td>{esc(record['matchup'])}</td><td>{esc(record['projection'])}</td>"
        f"<td>{esc(record['favorite'])} {_audit_number(record['favorite_probability'] * 100 if record['favorite_probability'] is not None else None, 0, suffix='%')}</td>"
        f"<td>{esc(record['actual_winner'])}</td><td>{_audit_status_cell(record)}</td>"
        f"<td>{esc(record['spread_side'])} {_spread_status_label(record['spread_status'])}</td>"
        f"<td class=\"num\">{_audit_number(abs(record['spread_error']) if record['spread_error'] is not None else None)}</td>"
        f"<td class=\"num\">{_audit_number(abs(record['total_error']) if record['total_error'] is not None else None)}</td>"
        f"<td class=\"dim\">{esc(record['run_id'][:10])}</td></tr>"
        for record in records
    )
    return f"""<h3 class="results-subtitle">All game-projection audit rows ({len(records)})</h3>
<p class="basis-note">Every generated forecast is retained, including multiple runs for the same matchup.</p>
<div class="board"><div class="tablewrap"><table>
<thead><tr><th>Game date / logged</th><th>Matchup</th><th>Projected score</th><th>Model call</th><th>Winner</th>
<th>Grade</th><th>Spread side</th><th class="num">Spread error</th><th class="num">Total error</th><th>Run</th></tr></thead>
<tbody>{rows}</tbody></table></div></div>"""


def _spread_status_label(status: str) -> str:
    if status == "True":
        return '<span class="result-hit">✓</span>'
    if status == "False":
        return '<span class="result-miss">×</span>'
    if status == "PUSH":
        return '<span class="result-void">push</span>'
    return '<span class="result-pending">pending</span>'


def _prop_audit_table(records: list[dict]) -> str:
    if not records:
        return """<h3 class="results-subtitle">All player-prop audit rows (0)</h3>
<div class="empty">No player-prop predictions have been logged yet. Every evaluation made with
<code>evaluate-player-prop</code> will appear here with its line, model probability, edge, outcome, and grade.</div>"""
    rows = "".join(
        f"<tr><td>{esc(record['game_date'])}<span class=\"player-sub\">{esc(record['recorded_at'])}</span></td>"
        f"<td>{esc(record['player'])}</td><td>{esc(record['market'])}</td>"
        f"<td>{esc(record['side'])} {_audit_number(record['line'])} @ {_audit_number(record['odds'], 0)}</td>"
        f"<td class=\"num\">{_audit_number(record['projection'])}</td>"
        f"<td class=\"num\">{_audit_number(record['model_prob'] * 100 if record['model_prob'] is not None else None, 1, suffix='%')}</td>"
        f"<td class=\"num\">{_audit_number(record['edge'] * 100 if record['edge'] is not None else None, 1, suffix='%')}</td>"
        f"<td class=\"num\">{_audit_number(record['actual'])}</td><td>{_audit_status_cell(record)}</td></tr>"
        for record in records
    )
    return f"""<h3 class="results-subtitle">All player-prop audit rows ({len(records)})</h3>
<div class="board"><div class="tablewrap"><table>
<thead><tr><th>Game date / logged</th><th>Player</th><th>Market</th><th>Pick / price</th><th class="num">Projection</th>
<th class="num">Model prob.</th><th class="num">Edge</th><th class="num">Actual</th><th>Grade</th></tr></thead>
<tbody>{rows}</tbody></table></div></div>"""


# ── methodology ──────────────────────────────────────────────────────────────

def _methodology_section(sigmas: pd.DataFrame | None) -> str:
    if sigmas is None or sigmas.empty:
        sigma_note = (
            "Per-market sigmas not yet fitted &mdash; evaluations fall back to labeled priors "
            "until <code>wnba-edges fit-sigma</code> runs."
        )
    else:
        parts = " &nbsp;".join(
            f'<span class="sigchip">{esc(row["market"]).replace("player_", "")}'
            f' <b>{float(row["sigma"]):.2f}</b></span>'
            for _, row in sigmas.iterrows()
        )
        n = int(sigmas.iloc[0].get("n_games", 0) or 0)
        sigma_note = f"Per-market game-to-game sigma, fitted on {n:,} player-games: {parts}"
    return f"""
<section>
  <div class="sec-head" id="methodology">
    <div class="sec-eyebrow">Methodology</div>
    <h2 class="sec-title">How to read this page</h2>
  </div>
  <div class="board prose">
    <p><b>Tiers.</b> Lean (edge &ge; 2 pts) &middot; Standard (&ge; 4.5 pts) &middot; Strong (&ge; 8 pts).
    Edges &ge; 15 pts flag <b class="warn">REVIEW</b> &mdash; implausibly large edges are treated as
    input errors, never as bets. Edges are measured against de-vigged market probabilities
    whenever both sides of a line are stored.</p>
    <p><b>Volatility.</b> {sigma_note}</p>
    <p><b>Known limitations.</b> Projections are team-efficiency baselines without injury or
    lineup adjustments; player projections are season rates with a minutes adjustment; the
    watchboard is a screening tool. Full details:
    <a href="https://github.com/Alphakiller1/wnba-edge-model/blob/main/METHODOLOGY.md">METHODOLOGY.md</a>.</p>
  </div>
</section>"""


# ── template ─────────────────────────────────────────────────────────────────
# Tokens vendored verbatim from the Chase Analytics design system (chase_tokens.css).

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WNBA Edge Model — Chase Analytics</title>
<meta name="description" content="WNBA research dashboard: game projections, player watchboard, market snapshots, and a graded prediction record.">
<style>
__TOKENS__
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;scroll-padding-top:76px}
body{background:var(--bg);color:var(--text);font:15px/1.55 var(--sans);
background-image:radial-gradient(1100px 480px at 78% -12%,rgba(124,77,255,.16),transparent 62%),
radial-gradient(900px 420px at 8% 4%,rgba(91,43,224,.10),transparent 55%),
linear-gradient(rgba(255,255,255,.022) 1px,transparent 1px),
linear-gradient(90deg,rgba(255,255,255,.022) 1px,transparent 1px);
background-size:auto,auto,26px 26px,26px 26px}
a{color:var(--v-light);text-decoration:none}a:hover{text-decoration:underline}
b{font-weight:700}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px}
.num{font-variant-numeric:tabular-nums;text-align:right}
.dim{color:var(--text-2)}
/* ── nav ── */
.chase-header{position:sticky;top:0;z-index:50;background:rgba(8,9,15,.82);backdrop-filter:blur(14px);
border-bottom:1px solid var(--border-soft)}
.chase-nav{display:flex;align-items:center;gap:26px;height:64px}
.chase-logo{display:flex;align-items:center;gap:10px;flex:0 0 auto}
.chase-wordmark{font-family:var(--wordmark);font-weight:700;font-size:19px;letter-spacing:.04em;color:var(--text);font-style:italic}
.chase-wordmark em{font-style:italic;background:linear-gradient(90deg,var(--v-light),var(--ca-purple));
-webkit-background-clip:text;background-clip:text;color:transparent}
.nav-links{display:flex;gap:4px;flex:1;justify-content:center}
.nav-link{padding:7px 13px;border-radius:9px;color:var(--text-2);font-size:13.5px;font-weight:600}
.nav-link:hover{color:var(--text);background:rgba(124,77,255,.12);text-decoration:none}
.chase-status{display:flex;align-items:center;gap:12px;flex:0 0 auto}
.product-tag{font-family:var(--display);font-weight:700;font-size:11px;letter-spacing:.14em;color:var(--v-light);
border:1px solid var(--border-violet);border-radius:999px;padding:4px 11px}
.chase-timestamp{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--text-2)}
.pipeline-dot{width:8px;height:8px;border-radius:50%}
.dot-ok{background:var(--green);box-shadow:0 0 8px rgba(60,203,127,.8)}
.dot-warn{background:var(--gold);box-shadow:0 0 8px rgba(232,194,74,.8)}
/* ── hero ── */
.hero{padding:58px 0 44px;border-bottom:1px solid var(--border-soft)}
.hero-eyebrow{display:inline-flex;align-items:center;gap:9px;color:var(--gold);font-family:var(--display);
font-weight:700;font-size:12px;letter-spacing:.2em;border:1px solid var(--border-2);
border-radius:999px;padding:7px 15px;background:rgba(14,16,24,.6)}
.hero-eyebrow-dot{width:7px;height:7px;border-radius:50%;background:var(--v-mid);box-shadow:0 0 10px var(--v-mid)}
.hero-title{font-family:var(--display);font-weight:800;font-size:clamp(38px,6vw,64px);line-height:1.02;
letter-spacing:-.015em;text-transform:uppercase;margin:22px 0 14px;max-width:820px}
.hero-sub{color:var(--text-2);max-width:640px;font-size:16px}
.hero-meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.pill{border:1px solid var(--border-2);border-radius:999px;padding:4px 13px;font-size:12.5px;background:var(--bg-2);color:var(--text-2)}
.pill-ok{color:var(--green);border-color:rgba(60,203,127,.35)}
.pill-warn{color:#0B0C12;background:var(--gold);border-color:var(--gold);font-weight:700}
.pill-dim{color:var(--text-3)}
/* ── stat tiles ── */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:28px}
.tile{position:relative;overflow:hidden;background:linear-gradient(180deg,var(--board-top),var(--board-bottom));
border:1px solid var(--border-soft);border-radius:16px;padding:18px 18px 15px;box-shadow:var(--card-shadow)}
.tile::before{content:"";position:absolute;inset:0 0 auto 0;height:2px;background:var(--v-grad);opacity:.75}
.tile-v{display:block;font-family:var(--display);font-weight:800;font-size:34px;line-height:1;letter-spacing:-.01em;font-variant-numeric:tabular-nums}
.tile-l{display:block;color:var(--text-3);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;margin-top:7px}
.tiles-results{margin:4px 0 16px}
/* ── sections ── */
main{padding:14px 0 40px}
section{margin-top:52px}
.sec-eyebrow{color:var(--gold);font-family:var(--display);font-weight:700;font-size:12px;letter-spacing:.2em;text-transform:uppercase}
.sec-title{font-family:var(--display);font-weight:800;font-size:clamp(26px,3.4vw,36px);text-transform:uppercase;
letter-spacing:-.01em;line-height:1.05;margin:8px 0 10px}
.sec-blurb{color:var(--text-2);max-width:700px;font-size:14.5px}
.basis-note{color:var(--text-3);font-size:13px;margin:14px 0 2px;font-variant-numeric:tabular-nums}
.empty{background:linear-gradient(180deg,var(--board-top),var(--board-bottom));border:1px dashed var(--border-2);
border-radius:16px;padding:22px;color:var(--text-2);margin-top:16px;font-size:14px;line-height:1.7}
code{background:var(--bg-4);border:1px solid var(--border-soft);border-radius:6px;padding:1.5px 7px;
font:12.5px var(--display);color:var(--v-light)}
/* ── projection matrix ── */
.matrix{display:grid;grid-template-columns:repeat(auto-fill,minmax(272px,1fr));gap:14px;margin-top:14px}
.mx-card{position:relative;overflow:hidden;background:linear-gradient(180deg,var(--board-top),var(--board-bottom));
border:1px solid var(--border-soft);border-radius:16px;padding:15px 16px 13px;box-shadow:var(--card-shadow);
transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease}
.mx-card:hover{transform:translateY(-3px);border-color:var(--border-violet);box-shadow:var(--glow)}
.mx-top{display:flex;justify-content:space-between;font-size:11.5px;color:var(--text-3);
font-family:var(--display);letter-spacing:.06em;text-transform:uppercase;margin-bottom:12px;font-variant-numeric:tabular-nums}
.mx-teams{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:6px;text-align:center}
.mx-abbr{display:block;font-family:var(--display);font-weight:700;font-size:14px;letter-spacing:.08em;color:var(--text-2)}
.mx-pts{display:block;font-family:var(--display);font-weight:800;font-size:33px;line-height:1.06;
letter-spacing:-.01em;font-variant-numeric:tabular-nums}
.mx-team.is-fav .mx-pts{color:var(--v-light)}
.mx-team.is-fav .mx-abbr{color:var(--v-light)}
.mx-net{display:block;font-size:10.5px;color:var(--text-3);margin-top:2px;font-variant-numeric:tabular-nums}
.mx-at{color:var(--text-3);font-size:12px}
.wp-row{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:8px;margin:12px 0 2px}
.wp-team{font-family:var(--display);font-weight:700;font-size:10.5px;letter-spacing:.08em;color:var(--text-3)}
.wp-team.is-fav{color:var(--v-light)}
.wp-track{position:relative;height:7px;border-radius:4px;background:var(--bg-4);overflow:hidden;direction:rtl}
.wp-fill{position:absolute;right:0;top:0;height:100%;background:var(--v-grad);border-radius:4px}
.wp-mid{position:absolute;left:50%;top:-1px;bottom:-1px;width:2px;background:rgba(245,246,250,.28)}
.mx-stats{display:grid;grid-template-columns:1.35fr 1fr 1fr 1fr;gap:4px;margin-top:12px;
border-top:1px solid var(--border-soft);padding-top:11px}
.mx-stat i{display:block;font-style:normal;font-size:9.5px;font-weight:700;color:var(--text-3);
text-transform:uppercase;letter-spacing:.1em}
.mx-stat b{display:block;font-family:var(--display);font-weight:700;font-size:14.5px;margin-top:2px;font-variant-numeric:tabular-nums}
.mx-stat--home b,.mx-stat--away b{color:var(--v-light)}
/* ── boards & tables ── */
.board{background:linear-gradient(180deg,var(--board-top),var(--board-bottom));border:1px solid var(--border-soft);
border-radius:16px;box-shadow:var(--card-shadow);margin-top:14px;overflow:hidden}
.tablewrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:14px}
thead th{font-family:var(--display);font-weight:700;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;
color:var(--text-3);text-align:left;padding:12px 14px 9px;border-bottom:1px solid var(--border-soft);white-space:nowrap}
thead th.num{text-align:right}
tbody td{padding:10px 14px;border-bottom:1px solid var(--border-soft);white-space:nowrap;font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:rgba(124,77,255,.05)}
.wb .rank{font-family:var(--display);color:var(--text-3);width:34px;font-size:12.5px}
.wb .player{font-weight:600}
.player-sub{display:block;font-size:11px;color:var(--text-3);font-weight:500;letter-spacing:.04em}
.wb .scorebar{width:170px;min-width:120px}
.sb-track{height:7px;border-radius:4px;background:var(--bg-4);overflow:hidden}
.sb-track i{display:block;height:100%;background:var(--v-grad);border-radius:4px}
.wb .score{font-family:var(--display);font-weight:800;font-size:15px;color:var(--v-light)}
.reason{white-space:normal;min-width:220px}
.results-subtitle{font-family:var(--display);font-size:16px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;
margin:26px 0 9px;color:var(--text)}
.result-hit{color:var(--green);font-weight:700}
.result-miss{color:var(--red);font-weight:700}
.result-pending{color:var(--gold);font-weight:700}
.result-void{color:var(--text-3);font-weight:700}
.result-hit td:nth-child(5){color:var(--green);font-weight:700}
.result-miss td:nth-child(5){color:var(--red);font-weight:700}
.rchip{display:inline-block;font-size:10.5px;font-weight:600;color:var(--text-2);background:var(--bg-4);
border:1px solid var(--border-soft);border-radius:999px;padding:2px 9px;margin:2px 3px 2px 0;letter-spacing:.02em}
.rchip-warn{color:var(--gold);border-color:rgba(232,194,74,.3)}
.lean{font-family:var(--display);font-size:11.5px;font-weight:700;letter-spacing:.04em;white-space:nowrap}
.lean-over{color:var(--ca-green)}
.lean-under{color:var(--ca-red)}
.sigchip{display:inline-block;background:var(--bg-4);border:1px solid var(--border-soft);border-radius:8px;
padding:3px 9px;font-size:12.5px;color:var(--text-2);margin:2px 2px 2px 0}
.sigchip b{color:var(--v-light);font-family:var(--display)}
/* Matchup cards: three even columns, readable type, ratings as a fact strip
   rather than a second row of betting tiles. Overrides the shared board kernel
   without editing vendored board.css. */
.bd-group__tiles{grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.bd-tile{padding:10px 11px 9px}
.bd-tile__value{font-size:18px;letter-spacing:-.01em}
.bd-tile__state{text-transform:none;letter-spacing:.015em;font-size:11px;font-weight:500;
color:var(--text-3);line-height:1.4}
.bd-tile.is-idle{background:transparent;border-color:transparent;padding:8px 6px 4px}
.bd-tile.is-idle .bd-tile__label{color:var(--text-3)}
.bd-tile.is-idle .bd-tile__value{color:var(--text)}
.bd-tile.is-idle.is-side .bd-tile__value{color:var(--side)}
.bd-tile.is-idle .bd-tile__state{color:var(--text-3)}
.bd-group:not(:has(.bd-group__count)){padding-top:10px}
.bd-group:not(:has(.bd-group__count)) .bd-group__tiles{gap:0 16px}
.bd-group:not(:has(.bd-group__count)) .bd-tile+.bd-tile{border-left:1px solid var(--border-soft);
border-radius:0;padding-left:14px}
/* ── prose / footer ── */
.prose{padding:20px 22px;font-size:14px;color:var(--text-2);line-height:1.7}
.prose p+p{margin-top:12px}
.prose b{color:var(--text)}
.warn{color:var(--gold)}
footer{border-top:1px solid var(--border-soft);margin-top:56px;padding:26px 0 40px;font-size:13px;color:var(--text-3);line-height:1.7}
footer b{color:var(--text-2)}
footer p+p{margin-top:10px}
.foot-links{display:flex;gap:10px}
.foot-links span{color:var(--text-3)}
/* ── responsive ── */
@media (max-width:860px){.nav-links{display:none}.chase-nav{justify-content:space-between}}
@media (max-width:480px){.chase-timestamp{display:none}}
@media (max-width:600px){
  .hero{padding:40px 0 32px}
  .wb .scorebar{min-width:90px;width:90px}
  .mx-stats{grid-template-columns:repeat(2,1fr);row-gap:10px}
  .tile-v{font-size:28px}
  .bd-group__tiles{grid-template-columns:1fr}
  .bd-group:not(:has(.bd-group__count)) .bd-tile+.bd-tile{border-left:0;border-top:1px solid var(--border-soft);
  padding-left:6px;padding-top:10px;margin-top:4px}
}
</style>
</head>
<body>
__BODY__
<script>__SCRIPT__</script>
</body>
</html>
"""
