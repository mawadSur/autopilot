import type { ClosedPosition } from "@/lib/types";
import { fmtUsd, fmtPrice, signClass, shortTime } from "@/lib/format";

export default function ExitFeed({ closed }: { closed: ClosedPosition[] }) {
  return (
    <div className="panel">
      <h2>
        Recent exits <span className="count">{closed.length}</span>
      </h2>
      {closed.length === 0 ? (
        <div className="empty">No closed positions yet.</div>
      ) : (
        <div className="feed">
          {closed.map((p) => {
            const pnl = p.realized_pnl_usd;
            const won = pnl > 0;
            return (
              <div className="feed-item" key={p.position_id}>
                <span className={"chip " + (won ? "won" : "lost")}>{won ? "WON" : "LOST"}</span>
                <span className="fi-title" title={`${p.symbol} ${p.side}`}>
                  {p.symbol} <span className="fi-sub">{p.side}</span>
                  <span className="fi-sub">
                    {" · "}
                    {fmtPrice(p.entry_price)}→{fmtPrice(p.exit_price)} · {shortTime(p.closed_at_utc)}
                  </span>
                </span>
                <span className={"fi-pnl mono " + signClass(pnl)}>{fmtUsd(pnl, true)}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
