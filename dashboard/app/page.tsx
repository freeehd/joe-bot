import { getJson, emptyCommand } from "@/lib/api";
import { Empty, Metric, Panel, Status } from "@/components/UI";

export default async function CommandCenter() {
  const d = await getJson("/api/command-center", emptyCommand);
  const equity = Number(d.account.equity ?? 0), pnl = Number(d.performance.daily_pnl ?? 0), dd = Number(d.performance.drawdown ?? 0);
  return <><div className="pageHead"><div><small>OPERATOR / COMMAND</small><h1>Command Center</h1></div><div className="mode"><Status value={d.mode}/><span>Entries {d.risk.entries_enabled ? "enabled" : "disabled"}</span></div></div>
    <div className="metrics"><Metric label="Account Equity" value={`$${equity.toLocaleString(undefined,{maximumFractionDigits:2})}`}/><Metric label="Daily P/L" value={`${pnl>=0?"+":""}$${pnl.toFixed(2)}`}/><Metric label="Drawdown" value={`${(dd*100).toFixed(2)}%`}/><Metric label="Gross Exposure" value={`${(Number(d.portfolio.gross_exposure ?? 0)*100).toFixed(1)}%`}/><Metric label="Risk Remaining" value={`${(Number(d.portfolio.risk_remaining ?? 1)*100).toFixed(0)}%`}/></div>
    <Panel title="Market Regime" aside={<Status value={String(d.regime.label)}/>}><div className="regime"><strong>{(Number(d.regime.confidence)*100).toFixed(0)}%</strong><span>SPY {(Number(d.regime.spy_return)*100).toFixed(2)}%</span><span>QQQ {(Number(d.regime.qqq_return)*100).toFixed(2)}%</span><span>Breadth {(Number(d.regime.breadth)*100).toFixed(0)}%</span></div></Panel>
    <div className="grid2"><Panel title="Top Opportunities">{d.scanner.length ? <pre>{JSON.stringify(d.scanner.slice(0,6),null,2)}</pre> : <Empty>No promoted runtime scanner data yet.</Empty>}</Panel><Panel title="Open Positions">{d.positions.length ? <pre>{JSON.stringify(d.positions,null,2)}</pre> : <Empty>No open positions.</Empty>}</Panel></div>
    <Panel title="Active Risk State" aside={<Status value={d.risk.kill_switches.length ? "DEGRADED":"UP"}/>}>{d.risk.kill_switches.length ? <div className="chips">{d.risk.kill_switches.map((x:string)=><span key={x}>{x}</span>)}</div> : <p className="good">No active kill switches.</p>}</Panel>
  </>;
}
