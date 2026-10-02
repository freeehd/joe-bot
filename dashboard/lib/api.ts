const API = process.env.NEXT_PUBLIC_ARGUS_API ?? "http://127.0.0.1:8000";

export async function getJson<T>(path: string, fallback: T): Promise<T> {
  try {
    const response = await fetch(`${API}${path}`, { cache: "no-store" });
    if (!response.ok) return fallback;
    return await response.json() as T;
  } catch {
    return fallback;
  }
}

export const emptyCommand = {
  mode: "OFFLINE",
  account: { equity: 0, cash: 0, buying_power: 0 },
  performance: { daily_pnl: 0, drawdown: 0 },
  regime: { label: "UNKNOWN", confidence: 0, spy_return: 0, qqq_return: 0, breadth: 0 },
  scanner: [] as Record<string, unknown>[], positions: [] as Record<string, unknown>[],
  portfolio: { gross_exposure: 0, net_exposure: 0, risk_remaining: 1 },
  risk: { entries_enabled: false, kill_switches: ["offline"] }, alerts: [] as Record<string, unknown>[],
  updated_at: "", version: 0,
};
