import type { Summary } from "@/lib/types";
import { fmtUsd, fmtPct, signClass } from "@/lib/format";

export default function Kpis({ s }: { s: Summary }) {
  return (
    <section className="kpis">
      <div className="kpi k-equity">
        <div className="label">Equity</div>
        <div className="value mono">{fmtUsd(s.equity_usd)}</div>
        <div className="meta">bankroll {fmtUsd(s.bankroll_usd)}</div>
      </div>
      <div className="kpi k-realized">
        <div className="label">Total P/L</div>
        <div className={"value mono " + signClass(s.total_pnl_usd)}>
          {fmtUsd(s.total_pnl_usd, true)}
        </div>
        <div className="meta">
          realized {fmtUsd(s.realized_pnl_usd, true)} · unreal {fmtUsd(s.unrealized_pnl_usd, true)}
        </div>
      </div>
      <div className="kpi k-winrate">
        <div className="label">Win rate</div>
        <div className="value mono">{fmtPct(s.win_rate)}</div>
        <div className="meta">
          avg win {fmtUsd(s.avg_win_usd, true)} · loss {fmtUsd(s.avg_loss_usd, true)}
        </div>
      </div>
      <div className="kpi k-counts">
        <div className="label">Open / Settled</div>
        <div className="value mono">
          {s.n_open} / {s.n_settled}
        </div>
        <div className="meta">fees {fmtUsd(s.total_fees_usd)}</div>
      </div>
    </section>
  );
}
