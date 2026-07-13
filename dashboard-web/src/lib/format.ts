// Formatting helpers shared across dashboard components.

export function fmtUsd(v: number | null | undefined, signed = false): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "$—";
  const sign = signed && v >= 0 ? "+" : v < 0 ? "-" : "";
  return (
    sign +
    "$" +
    Math.abs(v).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })
  );
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—%";
  return (v * 100).toFixed(1) + "%";
}

export function fmtPrice(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  // crypto prices span 6-figure BTC to sub-dollar; adapt precision.
  const abs = Math.abs(v);
  const digits = abs >= 1000 ? 2 : abs >= 1 ? 3 : 5;
  return "$" + v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: digits });
}

export function signClass(v: number): string {
  return v >= 0 ? "pos" : "neg";
}

export function shortTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const secs = Math.max(0, (Date.now() - d.getTime()) / 1000);
  if (secs < 60) return `${Math.floor(secs)}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}
