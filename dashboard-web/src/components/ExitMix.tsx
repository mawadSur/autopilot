const ORDER = ["won", "lost"];

export default function ExitMix({ mix }: { mix: Record<string, number> }) {
  const keys = Object.keys(mix || {});
  const ordered = ORDER.filter((k) => k in mix).concat(keys.filter((k) => !ORDER.includes(k)));
  const total = ordered.reduce((a, k) => a + (mix[k] || 0), 0);
  const maxV = Math.max(...ordered.map((k) => mix[k] || 0), 1);

  return (
    <div className="panel">
      <h2>Win / loss mix</h2>
      {total === 0 ? (
        <div className="empty">Nothing settled yet.</div>
      ) : (
        <div className="mix">
          {ordered.map((k) => {
            const v = mix[k] || 0;
            const cls = k === "won" || k === "lost" ? k : "other";
            return (
              <div className="mix-row" key={k}>
                <span className="name">{k}</span>
                <span className="track">
                  <span className={"fill " + cls} style={{ width: (v / maxV) * 100 + "%" }} />
                </span>
                <span className="num mono">{v}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
