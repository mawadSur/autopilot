import type { PerSymbol as PerSymbolRow } from "@/lib/types";
import { fmtUsd, fmtPct, signClass } from "@/lib/format";

export default function PerSymbol({ rows }: { rows: PerSymbolRow[] }) {
  return (
    <div className="panel">
      <h2>Per symbol</h2>
      {rows.length === 0 ? (
        <div className="empty">No symbol activity yet.</div>
      ) : (
        <table className="sym-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Open</th>
              <th>Closed</th>
              <th>Win %</th>
              <th>Realized</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.symbol}>
                <td>{r.symbol}</td>
                <td className="mono">{r.n_open}</td>
                <td className="mono">{r.n_closed}</td>
                <td className="mono">{r.win_rate == null ? "—" : fmtPct(r.win_rate)}</td>
                <td className={"mono " + signClass(r.realized_pnl_usd)}>
                  {fmtUsd(r.realized_pnl_usd, true)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
