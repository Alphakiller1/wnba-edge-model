import { dashboardData } from "./model-data";

const formatPlus = (value: number) => `${value > 0 ? "+" : ""}${value.toFixed(1)}`;

export default function Home() {
  const { summary, edgeBoard, gameProjections, teams, recentGames, leaders } = dashboardData;
  const topEdge = edgeBoard[0];

  return (
    <main className="min-h-screen bg-[#f5f7f2] text-[#18211f]">
      <section className="border-b border-[#cbd5c7] bg-[#101816] text-white">
        <div className="mx-auto grid max-w-7xl gap-8 px-5 py-7 lg:grid-cols-[1.1fr_0.9fr] lg:px-8">
          <div>
            <div className="mb-4 flex flex-wrap items-center gap-2 text-xs font-semibold uppercase tracking-wide text-[#b7c8bd]">
              <span>WNBA Edge Model</span>
              <span className="h-1 w-1 rounded-full bg-[#f48b55]" />
              <span>{summary.lastUpdated}</span>
            </div>
            <h1 className="max-w-3xl text-4xl font-semibold tracking-normal md:text-6xl">
              Player prop research board
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-[#d7e1dc]">
              A deployable view of the WNBA model layer: season outcomes, player logs,
              advanced usage signals, recent-form deltas, and Betting Brain-style edge ranking.
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <Metric label="Players" value={summary.players.toString()} />
            <Metric label="Games" value={summary.games.toString()} />
            <Metric label="Projected" value={summary.gameProjections.toString()} />
            <Metric label="Player Logs" value={summary.playerLogs.toLocaleString()} />
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-5 py-6 lg:px-8">
        <div className="rounded-md border border-[#cbd5c7] bg-white">
          <div className="flex flex-wrap items-end justify-between gap-3 border-b border-[#d8dfd5] px-4 py-3">
            <div>
              <h2 className="text-lg font-semibold">Game Projections</h2>
              <p className="text-sm text-[#66756d]">Baseline projected score, spread, total, pace, and confidence.</p>
            </div>
            <div className="flex flex-wrap gap-2 text-xs">
              <Legend color="bg-[#dff3e5] text-[#17643a]" label="Strong / positive" />
              <Legend color="bg-[#fff1c7] text-[#7c5a00]" label="Review" />
              <Legend color="bg-[#ffe1dd] text-[#9d2f25]" label="Risk / weak" />
            </div>
          </div>
          <div className="grid gap-0 md:grid-cols-2 xl:grid-cols-3">
            {gameProjections.map((game) => (
              <ProjectionCard key={`${game.date}-${game.away}-${game.home}`} game={game} />
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-5 px-5 pb-6 lg:grid-cols-[1.5fr_0.8fr] lg:px-8">
        <div className="rounded-md border border-[#cbd5c7] bg-white">
          <div className="flex flex-wrap items-end justify-between gap-3 border-b border-[#d8dfd5] px-4 py-3">
            <div>
              <h2 className="text-lg font-semibold">Edge Board</h2>
              <p className="text-sm text-[#66756d]">Ranked review candidates, not auto-bets.</p>
            </div>
            <div className="text-right font-mono text-sm text-[#66756d]">
              Leader: <span className="text-[#18211f]">{topEdge.name}</span> {formatPlus(topEdge.edgeScore)}
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[920px] border-collapse text-sm">
              <thead className="bg-[#eef2eb] text-left text-xs uppercase text-[#596760]">
                <tr>
                  <th className="px-4 py-3">Player</th>
                  <th className="px-3 py-3">Team</th>
                  <th className="px-3 py-3 text-right">Score</th>
                  <th className="px-3 py-3 text-right">MPG</th>
                  <th className="px-3 py-3 text-right">USG</th>
                  <th className="px-3 py-3 text-right">P/R/A</th>
                  <th className="px-3 py-3 text-right">Recent PRA</th>
                  <th className="px-4 py-3">Watch Reason</th>
                </tr>
              </thead>
              <tbody>
                {edgeBoard.map((row, index) => (
                  <tr key={`${row.name}-${row.team}`} className="border-t border-[#edf1e9]">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        <span className="w-6 font-mono text-xs text-[#7b877f]">{index + 1}</span>
                        <div>
                          <div className="font-semibold">{row.name}</div>
                          <div className="text-xs text-[#6b7870]">{row.pos}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-3 font-mono">{row.team}</td>
                    <td className="px-3 py-3 text-right font-mono font-semibold text-[#b94f2d]">
                      {formatPlus(row.edgeScore)}
                    </td>
                    <td className="px-3 py-3 text-right font-mono">{row.mpg.toFixed(1)}</td>
                    <td className="px-3 py-3 text-right font-mono">{row.usg.toFixed(1)}</td>
                    <td className="px-3 py-3 text-right font-mono">
                      {row.ppg.toFixed(1)}/{row.rpg.toFixed(1)}/{row.apg.toFixed(1)}
                    </td>
                    <td className="px-3 py-3 text-right font-mono">{formatPlus(row.recentPraSignal)}</td>
                    <td className="max-w-[330px] px-4 py-3 text-[#526158]">{row.watchReason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <aside className="space-y-5">
          <div className="rounded-md border border-[#cbd5c7] bg-white p-4">
            <h2 className="mb-3 text-lg font-semibold">Source Stack</h2>
            <div className="space-y-2">
              {summary.sources.map((source) => (
                <div key={source} className="flex items-center justify-between gap-3 rounded border border-[#edf1e9] px-3 py-2">
                  <span className="text-sm">{source}</span>
                  <span className="h-2 w-2 rounded-full bg-[#3f8f63]" />
                </div>
              ))}
            </div>
          </div>

          <LeaderPanel title="Usage Leaders" rows={leaders.usage} suffix="%" />
          <LeaderPanel title="Last 5 PRA" rows={leaders.last5PRA} />
          <LeaderPanel title="Volatility Watch" rows={leaders.volatility} />
        </aside>
      </section>

      <section className="mx-auto grid max-w-7xl gap-5 px-5 pb-8 lg:grid-cols-[1fr_1fr] lg:px-8">
        <div className="rounded-md border border-[#cbd5c7] bg-white">
          <div className="border-b border-[#d8dfd5] px-4 py-3">
            <h2 className="text-lg font-semibold">Team Context</h2>
            <p className="text-sm text-[#66756d]">Net rating and pace shape prop environment.</p>
          </div>
          <div className="grid gap-0 sm:grid-cols-2">
            {teams.slice(0, 10).map((team) => (
              <div key={team.abbr} className="border-b border-r border-[#edf1e9] p-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="font-semibold">{team.abbr}</div>
                    <div className="text-xs text-[#66756d]">{team.name}</div>
                  </div>
                  <div className="font-mono text-sm">{team.w}-{team.l}</div>
                </div>
                <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                  <Mini label="NET" value={formatPlus(team.net)} />
                  <Mini label="PACE" value={team.pace.toFixed(1)} />
                  <Mini label="PO" value={`${team.playoffOdds.toFixed(0)}%`} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-md border border-[#cbd5c7] bg-white">
          <div className="border-b border-[#d8dfd5] px-4 py-3">
            <h2 className="text-lg font-semibold">Recent Outcomes</h2>
            <p className="text-sm text-[#66756d]">Latest completed games from the season table.</p>
          </div>
          <div className="divide-y divide-[#edf1e9]">
            {recentGames.map((game) => (
              <div key={`${game.date}-${game.away}-${game.home}`} className="grid grid-cols-[84px_1fr_70px] gap-3 px-4 py-3 text-sm">
                <div className="font-mono text-xs text-[#6b7870]">{game.date.slice(5)}</div>
                <div>
                  <div className="font-semibold">
                    {game.away} {game.awayPts} @ {game.home} {game.homePts}
                  </div>
                  <div className="text-xs text-[#66756d]">{game.topPerf}</div>
                </div>
                <div className="text-right font-mono text-xs">
                  <div>{game.total} total</div>
                  <div className="text-[#66756d]">{game.pace.toFixed(1)} pace</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="border-t border-[#cbd5c7] bg-[#e9eee5]">
        <div className="mx-auto max-w-7xl px-5 py-5 text-sm leading-6 text-[#56645c] lg:px-8">
          This model is a research and projection surface. It highlights price-sensitive review
          candidates using usage, minutes, recent form, team context, advanced box inputs, and
          Betting Brain value math. It still needs closing-line and settled-prop calibration before
          it should be treated as an automated betting system.
        </div>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-white/15 bg-white/8 p-4">
      <div className="text-xs uppercase text-[#b7c8bd]">{label}</div>
      <div className="mt-2 font-mono text-3xl font-semibold">{value}</div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return <span className={`rounded px-2 py-1 font-medium ${color}`}>{label}</span>;
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[#7b877f]">{label}</div>
      <div className="font-mono font-semibold">{value}</div>
    </div>
  );
}

function ProjectionCard({
  game,
}: {
  game: {
    date: string;
    time: string;
    away: string;
    home: string;
    awayPts: number;
    homePts: number;
    total: number;
    homeSpread: number;
    pace: number;
    confidence: number;
    favorite: string;
    awayNet: number;
    homeNet: number;
    note: string;
  };
}) {
  const confidenceTone =
    game.confidence >= 75
      ? "bg-[#dff3e5] text-[#17643a] border-[#9ccfae]"
      : game.confidence >= 62
        ? "bg-[#fff1c7] text-[#7c5a00] border-[#e4c66a]"
        : "bg-[#ffe1dd] text-[#9d2f25] border-[#efaaa1]";
  const totalTone =
    game.total >= 174
      ? "bg-[#dff3e5] text-[#17643a]"
      : game.total <= 168
        ? "bg-[#e7ecf7] text-[#314a73]"
        : "bg-[#f1f4ee] text-[#526158]";
  const paceTone =
    game.pace >= 82
      ? "text-[#17643a]"
      : game.pace <= 79
        ? "text-[#314a73]"
        : "text-[#526158]";
  const spreadAbs = Math.abs(game.homeSpread);
  const spreadTone =
    spreadAbs >= 6
      ? "bg-[#dff3e5] text-[#17643a]"
      : spreadAbs <= 2
        ? "bg-[#fff1c7] text-[#7c5a00]"
        : "bg-[#f1f4ee] text-[#526158]";
  const homeFavored = game.homeSpread >= 0;
  const spreadLabel = homeFavored
    ? `${game.home} -${spreadAbs.toFixed(1)}`
    : `${game.away} -${spreadAbs.toFixed(1)}`;

  return (
    <article className="border-b border-r border-[#edf1e9] p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="font-mono text-xs text-[#6b7870]">
          {game.date.slice(5)} · {game.time}
        </div>
        <span className={`rounded border px-2 py-1 text-xs font-semibold ${confidenceTone}`}>
          {game.confidence.toFixed(0)} conf
        </span>
      </div>
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
        <TeamProjection abbr={game.away} points={game.awayPts} net={game.awayNet} active={game.favorite === game.away} />
        <div className="text-center text-xs uppercase text-[#7b877f]">@</div>
        <TeamProjection abbr={game.home} points={game.homePts} net={game.homeNet} active={game.favorite === game.home} align="right" />
      </div>
      <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
        <div className={`rounded px-2 py-2 ${spreadTone}`}>
          <div className="uppercase opacity-70">Spread</div>
          <div className="font-mono font-semibold">{spreadLabel}</div>
        </div>
        <div className={`rounded px-2 py-2 ${totalTone}`}>
          <div className="uppercase opacity-70">Total</div>
          <div className="font-mono font-semibold">{game.total.toFixed(1)}</div>
        </div>
        <div className="rounded bg-[#f1f4ee] px-2 py-2">
          <div className="uppercase text-[#7b877f]">Pace</div>
          <div className={`font-mono font-semibold ${paceTone}`}>{game.pace.toFixed(1)}</div>
        </div>
      </div>
    </article>
  );
}

function TeamProjection({
  abbr,
  points,
  net,
  active,
  align = "left",
}: {
  abbr: string;
  points: number;
  net: number;
  active: boolean;
  align?: "left" | "right";
}) {
  const netTone = net >= 4 ? "text-[#17643a]" : net <= -4 ? "text-[#9d2f25]" : "text-[#7c5a00]";
  return (
    <div className={align === "right" ? "text-right" : "text-left"}>
      <div className={`text-xl font-semibold ${active ? "text-[#17643a]" : "text-[#18211f]"}`}>{abbr}</div>
      <div className="font-mono text-3xl font-semibold">{points.toFixed(1)}</div>
      <div className={`font-mono text-xs ${netTone}`}>{formatPlus(net)} net</div>
    </div>
  );
}

function LeaderPanel({ title, rows, suffix = "" }: { title: string; rows: readonly { name: string; team: string; value: number }[]; suffix?: string }) {
  return (
    <div className="rounded-md border border-[#cbd5c7] bg-white p-4">
      <h2 className="mb-3 text-lg font-semibold">{title}</h2>
      <div className="space-y-2">
        {rows.map((row) => (
          <div key={`${title}-${row.name}`} className="grid grid-cols-[1fr_48px_64px] items-center gap-2 text-sm">
            <span className="truncate font-medium">{row.name}</span>
            <span className="font-mono text-xs text-[#6b7870]">{row.team}</span>
            <span className="text-right font-mono font-semibold">
              {row.value.toFixed(1)}
              {suffix}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
