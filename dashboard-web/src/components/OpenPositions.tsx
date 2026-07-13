import type { OpenPosition } from "@/lib/types";
import { fmtUsd, fmtPrice, signClass, shortTime } from "@/lib/format";

function confPct(p: OpenPosition): number {
  const s = p.confidence?.score;
  return s == null ? 0 : Math.max(0, Math.min(100, s * 100));
}

export default function OpenPositions({ positions }: { positions: OpenPosition[] }) {
  const symbols = new Set(positions.map((p) => p.symbol));
  return (
    <div className="panel">
      <h2>
        Open positions
        <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className="count">{symbols.size} sym</span>
          <span className="count">{positions.length}</span>
        </span>
      </h2>
      {positions.length === 0 ? (
        <div className="empty">No open positions.</div>
      ) : (
        <div className="pos-list">
          {positions.map((p) => {
            const label = p.confidence?.label ?? "low";
            const upnl = p.unrealized_pnl_usd;
            return (
              <div className="pos-card" key={p.position_id}>
                <div>
                  <span className="pc-sym">{p.symbol}</span>
                  <span className={"pc-side " + p.side}>{p.side}</span>
                </div>
                <div className={"pc-upnl mono " + signClass(upnl)}>
                  {p.mark_price != null ? fmtUsd(upnl, true) : "—"}
                </div>
                <div className="pc-meta">
                  <span>entry {fmtPrice(p.entry_price)}</span>
                  {p.mark_price != null && <span>mark {fmtPrice(p.mark_price)}</span>}
                  <span>{fmtUsd(p.notional_usd)}</span>
                  {p.confidence && <span>conf {p.confidence.score.toFixed(2)}</span>}
                  {p.regime_label && <span className="regime">{p.regime_label}</span>}
                  <span className="regime">{shortTime(p.opened_at_utc)}</span>
                </div>
                <div className="pc-conf-track" title={`entry confidence ${p.confidence?.score ?? "—"}`}>
                  <span className={"pc-conf-fill " + label} style={{ width: confPct(p) + "%" }} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
