// Shape of the snapshot published by scripts/dashboard_exporter.py.
// Keep in sync with that script's build_snapshot().

export interface Confidence {
  score: number; // 0..1
  label: "low" | "medium" | "high";
}

export interface BotMeta {
  mode: "paper" | "live";
  halal: boolean;
  running: boolean;
  symbols: string[];
  exit_rule: string | null;
}

export interface Summary {
  bankroll_usd: number;
  equity_usd: number;
  realized_pnl_usd: number;
  unrealized_pnl_usd: number;
  total_pnl_usd: number;
  win_rate: number; // 0..1
  n_open: number;
  n_settled: number;
  total_fees_usd: number;
  avg_win_usd: number;
  avg_loss_usd: number;
}

export interface OpenPosition {
  position_id: string;
  symbol: string;
  side: "long" | "short";
  entry_price: number;
  mark_price: number | null;
  base_size: number;
  notional_usd: number;
  unrealized_pnl_usd: number;
  opened_at_utc: string;
  confidence: Confidence | null;
  regime_label: string | null;
}

export interface ClosedPosition {
  position_id: string;
  symbol: string;
  side: "long" | "short";
  entry_price: number;
  exit_price: number;
  realized_pnl_usd: number;
  fees_usd: number;
  reason: "won" | "lost" | string;
  opened_at_utc: string;
  closed_at_utc: string | null;
  confidence: Confidence | null;
  regime_label: string | null;
}

export interface EquityPoint {
  t: string;
  equity: number;
}

export interface PerSymbol {
  symbol: string;
  n_open: number;
  n_closed: number;
  realized_pnl_usd: number;
  win_rate: number | null;
  open_notional_usd: number;
}

export interface Snapshot {
  schema_version: number;
  generated_at_utc: string;
  window_days: number;
  bot: BotMeta;
  summary: Summary;
  open_positions: OpenPosition[];
  closed_positions: ClosedPosition[];
  equity_curve: EquityPoint[];
  exit_mix: Record<string, number>;
  per_symbol: PerSymbol[];
}

// What /api/state returns to the client.
export interface StateResponse {
  snapshot: Snapshot | null;
  stale: boolean;
  age_seconds: number | null;
  server_time_utc: string;
}
