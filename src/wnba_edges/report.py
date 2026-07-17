"""Self-contained static dashboard for GitHub Pages.

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

from .features import MIN_GAMES_FOR_BOARD, MIN_MPG_FOR_BOARD, board_eligible
from .predictions import results_summary
from .sigma import load_market_sigmas


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
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body = f"""
<header class="hero">
  <div class="wrap">
    <div class="eyebrow">Chase Analytics &middot; Model Lab</div>
    <h1>WNBA Edge Model</h1>
    <p class="sub">Research dashboard: game projections, a player watchboard, market snapshots,
    and a graded prediction record. Analytics research &mdash; not betting advice.</p>
    <div class="meta">
      <span class="pill">data through {esc(data_date or "n/a")}</span>
      {_stale_badge(stale_days)}
      <span class="pill dim">built {esc(built_at)}</span>
    </div>
    {_summary_tiles(features, game_results, player_logs, projections)}
  </div>
</header>
<main class="wrap">
  {_projections_section(projections)}
  {_market_section(odds)}
  {_board_section(features)}
  {_results_section(summary)}
  {_methodology_section(sigmas)}
</main>
<footer>
  <div class="wrap">
    <p><strong>Chase Analytics &mdash; WNBA Edge Model</strong> is research and analytics software.
    It does not provide betting advice, does not guarantee outcomes, and no output is a wager
    instruction. Model limitations are documented in the
    <a href="https://github.com/Alphakiller1/wnba-edge-model/blob/main/METHODOLOGY.md">methodology</a>.
    If you or someone you know has a gambling problem, call 1-800-GAMBLER.</p>
    <p class="dim">Source: <a href="https://github.com/Alphakiller1/wnba-edge-model">github.com/Alphakiller1/wnba-edge-model</a></p>
  </div>
</footer>
"""
    document = _TEMPLATE.replace("__BODY__", body)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out


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
        return '<span class="pill ok">fresh</span>'
    return f'<span class="pill warn">STALE &mdash; last refresh {stale_days} days ago</span>'


def _summary_tiles(features, game_results, player_logs, projections) -> str:
    tiles = [
        ("Players", len(features) if features is not None else 0),
        ("Finished games", len(game_results) if game_results is not None else 0),
        ("Player game logs", len(player_logs) if player_logs is not None else 0),
        ("Projected games", len(projections) if projections is not None else 0),
    ]
    cells = "".join(
        f'<div class="tile"><div class="tile-v">{value:,}</div><div class="tile-l">{esc(label)}</div></div>'
        for label, value in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _projections_section(projections: pd.DataFrame | None) -> str:
    head = _section_head(
        "1", "Game Projections", "Model layer",
        "Baseline score, spread, total and pace from team efficiency ratings; "
        "home-win probability is fit on finished games.",
    )
    if projections is None:
        return f"""{head}<div class="empty">No game projections yet. Run
        <code>wnba-edges build-game-projections</code> after a season refresh.</div>"""
    basis = esc(projections.iloc[0].get("win_prob_basis", ""))
    hc = esc(projections.iloc[0].get("home_court_pts", ""))
    cards = "".join(_projection_card(game) for _, game in projections.iterrows())
    return f"""{head}
<p class="note">Win probability: {basis} &middot; home court: {hc} pts</p>
<div class="cards">{cards}</div>"""


def _projection_card(game: pd.Series) -> str:
    win = pd.to_numeric(game.get("home_win_prob"), errors="coerce")
    win_txt = f"{win * 100:.0f}%" if pd.notna(win) else "&ndash;"
    fav_home = pd.notna(win) and win >= 0.5
    spread = pd.to_numeric(game.get("projected_home_spread"), errors="coerce")
    spread_txt = f"{spread:+.1f}" if pd.notna(spread) else "&ndash;"
    return f"""
<article class="card">
  <div class="card-top"><span>{esc(game.get("date", ""))}</span><span class="dim">{esc(str(game.get("time", ""))[:16])}</span></div>
  <div class="teams">
    <div class="team {'is-fav' if not fav_home else ''}">
      <div class="abbr">{esc(game["away"])}</div>
      <div class="pts">{esc(game["projected_away_pts"])}</div>
      <div class="net dim">net {esc(game.get("away_net", ""))}</div>
    </div>
    <div class="at">@</div>
    <div class="team {'is-fav' if fav_home else ''}">
      <div class="abbr">{esc(game["home"])}</div>
      <div class="pts">{esc(game["projected_home_pts"])}</div>
      <div class="net dim">net {esc(game.get("home_net", ""))}</div>
    </div>
  </div>
  <div class="statrow">
    <div><span class="lbl">Home win</span><span class="val">{win_txt}</span></div>
    <div><span class="lbl">Spread</span><span class="val">{spread_txt}</span></div>
    <div><span class="lbl">Total</span><span class="val">{esc(game["projected_total"])}</span></div>
    <div><span class="lbl">Pace</span><span class="val">{esc(game["projected_pace"])}</span></div>
  </div>
</article>"""


def _market_section(odds: pd.DataFrame | None) -> str:
    head = _section_head(
        "2", "Market Snapshot", "Market layer",
        "Stored odds with book attribution and quote timestamps. Edges are only priced "
        "against de-vigged market probabilities from these snapshots.",
    )
    if odds is None:
        return f"""{head}<div class="empty">No odds snapshot stored. Fetch one with
        <code>python -m wnba_edges.market_data --fetch-game AWY@HOM --props</code>
        (requires <code>ODDS_API_KEY</code>). Model probabilities without a market
        price are projections, not edges &mdash; nothing here pretends otherwise.</div>"""
    fetched = odds["fetched_at"].astype(str).max()
    by_market = odds.groupby("market").size().sort_values(ascending=False)
    rows = "".join(
        f"<tr><td>{esc(market)}</td><td class='num'>{count}</td></tr>"
        for market, count in by_market.items()
    )
    books = odds["book"].nunique()
    return f"""{head}
<p class="note">Latest snapshot: {esc(fetched)} &middot; {books} book(s) &middot; {len(odds)} quotes</p>
<div class="tablewrap"><table>
<thead><tr><th>Market</th><th class="num">Quotes</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""


def _board_section(features: pd.DataFrame | None) -> str:
    head = _section_head(
        "3", "Edge Watchboard", "Review queue",
        "A ranked review queue of usage, minutes, and form signals &mdash; these are "
        "research candidates to price against a market, not bets. Players under the "
        f"sample floor (GP &lt; {MIN_GAMES_FOR_BOARD} or MPG &lt; {MIN_MPG_FOR_BOARD:g}) are excluded.",
    )
    if features is None:
        return f"""{head}<div class="empty">No feature board yet. Run
        <code>wnba-edges build-features</code> after a season refresh.</div>"""
    total = len(features)
    eligible = board_eligible(features)
    excluded = total - len(eligible)
    top = eligible.head(15)
    rows = "".join(
        f"""<tr><td class="rank">{i + 1}</td><td>{esc(row["name"])}</td>
<td class="dim">{esc(row["team"])}</td><td class="dim">{esc(row.get("pos", "-"))}</td>
<td class="num">{float(row["edge_score"]):.2f}</td>
<td class="num dim">{esc(row.get("ppg", ""))}</td><td class="num dim">{esc(row.get("mpg", ""))}</td>
<td class="reason">{esc(row["watch_reason"])}</td></tr>"""
        for i, (_, row) in enumerate(top.iterrows())
    )
    return f"""{head}
<p class="note">{len(eligible)} of {total} players pass the sample floor ({excluded} excluded as low-sample).</p>
<div class="tablewrap"><table>
<thead><tr><th></th><th>Player</th><th>Team</th><th>Pos</th><th class="num">Score</th>
<th class="num">PPG</th><th class="num">MPG</th><th>Watch reason</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""


def _results_section(summary: dict) -> str:
    head = _section_head(
        "4", "Graded Results", "Results layer",
        "Every logged prediction is graded against finished games; ungradeable rows carry "
        "an explicit reason code. An empty record means nothing has been predicted yet — "
        "not hidden losses.",
    )
    props = {k: v for k, v in summary.get("props", {}).items() if not k.startswith("_")}
    games = summary.get("games", {})
    has_games = bool(games.get("n"))
    if not props and not has_games:
        pending = summary.get("props", {}).get("_pending", 0) + games.get("_pending", 0)
        pending_note = f" {pending} prediction(s) await grading." if pending else ""
        return f"""{head}<div class="empty">No graded predictions yet.{pending_note}
        Predictions logged by <code>evaluate-player-prop</code> and
        <code>build-game-projections</code> are graded by <code>wnba-edges grade-predictions</code>
        once games finish.</div>"""
    prop_rows = "".join(
        f"""<tr><td>{esc(market)}</td>
<td class="num">{record["wins"]}-{record["losses"]}-{record["pushes"]}</td>
<td class="num">{esc(record["hit_rate"]) if record["hit_rate"] is not None else "&ndash;"}%</td></tr>"""
        for market, record in sorted(props.items())
    )
    prop_table = (
        f"""<div class="tablewrap"><table>
<thead><tr><th>Market</th><th class="num">W-L-P</th><th class="num">Hit rate</th></tr></thead>
<tbody>{prop_rows}</tbody></table></div>"""
        if props
        else '<p class="note">No graded props yet.</p>'
    )
    game_block = ""
    if has_games:
        game_block = f"""<p class="note">Game projections: n={games["n"]} &middot;
winner hit rate {games["winner_hit_rate"]}% &middot; spread MAE {games["spread_mae"]} &middot;
total MAE {games["total_mae"]} &middot; Brier {games["brier"]}</p>"""
    return f"{head}{prop_table}{game_block}"


def _methodology_section(sigmas: pd.DataFrame | None) -> str:
    if sigmas is None or sigmas.empty:
        sigma_note = (
            "Per-market sigmas not yet fitted — evaluations fall back to labeled priors "
            "until <code>wnba-edges fit-sigma</code> runs."
        )
    else:
        parts = ", ".join(
            f"{esc(row['market']).replace('player_', '')} {float(row['sigma']):.1f}"
            for _, row in sigmas.iterrows()
        )
        sigma_note = f"Fitted per-market game-to-game sigma: {parts}."
    return f"""
<section>
  <div class="sec-head"><div><span class="kicker">Methodology</span>
  <h2>How to read this page</h2></div></div>
  <div class="prose">
    <p><strong>Tiers:</strong> Lean (edge &ge; 2 pts), Standard (&ge; 4.5 pts), Strong (&ge; 8 pts).
    Edges &ge; 15 pts are flagged <strong>REVIEW</strong> — implausibly large edges are treated as
    input errors, never as bets. Edges are measured against de-vigged (vig-free) market
    probabilities whenever both sides of a line are stored.</p>
    <p>{sigma_note}</p>
    <p><strong>Known limitations:</strong> projections are team-efficiency baselines without
    injury or lineup adjustments; player projections are season rates with a minutes adjustment;
    the watchboard is a screening tool. Full details:
    <a href="https://github.com/Alphakiller1/wnba-edge-model/blob/main/METHODOLOGY.md">METHODOLOGY.md</a>.</p>
  </div>
</section>"""


def _section_head(number: str, title: str, kicker: str, blurb: str) -> str:
    return f"""
<section>
<div class="sec-head">
  <div><span class="kicker">{esc(kicker)} &middot; {esc(number)}/4</span>
  <h2>{esc(title)}</h2>
  <p class="blurb">{blurb}</p></div>
</div>
</section>"""


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WNBA Edge Model — Chase Analytics</title>
<meta name="description" content="WNBA research dashboard: game projections, player watchboard, market snapshots, and a graded prediction record.">
<style>
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--line:#2a3240;--text:#e6e9ee;--dim:#8b96a5;
--gold:#e8b45a;--blue:#5aa2e8;--green:#4cc38a;--red:#e5645f;--amber:#e8c55a;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--blue)}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
.hero{background:linear-gradient(180deg,#11161f,#0d1117);border-bottom:1px solid var(--line);padding:40px 0 28px}
.eyebrow{color:var(--gold);font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px}
h1{font-size:34px;letter-spacing:-.01em}
.sub{color:var(--dim);max-width:640px;margin-top:8px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.pill{border:1px solid var(--line);border-radius:999px;padding:3px 12px;font-size:12px;background:var(--panel)}
.pill.ok{color:var(--green);border-color:#2b4a3c}
.pill.warn{color:#161b22;background:var(--amber);border-color:var(--amber);font-weight:700}
.pill.dim{color:var(--dim)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:20px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.tile-v{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums}
.tile-l{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.08em;margin-top:2px}
main{padding:10px 0 30px}
section{margin-top:34px}
.kicker{color:var(--gold);font-size:11px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
h2{font-size:22px;margin-top:4px}
.blurb{color:var(--dim);max-width:680px;margin-top:6px;font-size:14px}
.note{color:var(--dim);font-size:13px;margin:10px 0 4px}
.empty{background:var(--panel);border:1px dashed var(--line);border-radius:10px;padding:18px;color:var(--dim);margin-top:12px;font-size:14px}
code{background:var(--panel2);border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:12.5px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;margin-top:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card-top{display:flex;justify-content:space-between;font-size:12px;color:var(--dim);margin-bottom:10px}
.teams{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:8px;text-align:center}
.abbr{font-weight:700;font-size:15px}
.pts{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}
.team.is-fav .pts{color:var(--gold)}
.net{font-size:11px}
.at{color:var(--dim)}
.statrow{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
.statrow .lbl{display:block;font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em}
.statrow .val{font-weight:600;font-variant-numeric:tabular-nums;font-size:14px}
.tablewrap{overflow-x:auto;margin-top:12px;border:1px solid var(--line);border-radius:10px}
table{width:100%;border-collapse:collapse;background:var(--panel);font-size:14px}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.08em}
tbody tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.rank{color:var(--dim)}
td.reason{white-space:normal;min-width:200px;color:var(--dim);font-size:13px}
.dim{color:var(--dim)}
.prose{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-top:12px;font-size:14px}
.prose p+p{margin-top:10px}
footer{border-top:1px solid var(--line);margin-top:40px;padding:22px 0 34px;font-size:13px;color:var(--dim)}
footer p+p{margin-top:8px}
@media (max-width:600px){h1{font-size:26px}.statrow{grid-template-columns:repeat(2,1fr);row-gap:10px}}
</style>
</head>
<body>
__BODY__
</body>
</html>
"""
