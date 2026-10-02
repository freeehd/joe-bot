import Link from "next/link";

const nav = [
  ["/", "Command Center"], ["/scanner", "Live Scanner"], ["/portfolio", "Portfolio"],
  ["/positions", "Positions"], ["/models", "Models"], ["/laya", "Laya"], ["/research", "Research"], ["/backtests", "Backtests"], ["/execution", "Execution"], ["/replay", "Replay"], ["/risk", "Risk Center"], ["/system", "System Health"],
];
export function Nav() {
  return <aside className="nav"><div className="brand"><span className="brandMark">A</span><div><b>ARGUS</b><small>JOE BOT</small></div></div>
    <nav>{nav.map(([href,label]) => <Link key={href} href={href}>{label}</Link>)}</nav>
    <div className="navFoot"><span className="safeDot"/>LIVE CAPITAL LOCKED<small>Operator OS v0.1</small></div>
  </aside>;
}
