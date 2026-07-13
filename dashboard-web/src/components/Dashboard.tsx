"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { StateResponse } from "@/lib/types";
import Header from "./Header";
import Kpis from "./Kpis";
import OpenPositions from "./OpenPositions";
import EquityChart from "./EquityChart";
import ExitFeed from "./ExitFeed";
import ExitMix from "./ExitMix";
import PerSymbol from "./PerSymbol";

const POLL_MS = 15000;

export default function Dashboard({ userName }: { userName: string | null }) {
  const [state, setState] = useState<StateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updatedLabel, setUpdatedLabel] = useState("—");
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const tick = useCallback(async () => {
    try {
      const res = await fetch("/api/state", { cache: "no-store" });
      if (res.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = (await res.json()) as StateResponse;
      setState(data);
      setError(null);
      setUpdatedLabel(new Date().toLocaleTimeString());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    tick();
    timer.current = setInterval(tick, POLL_MS);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [tick]);

  const snap = state?.snapshot ?? null;
  const stale = state?.stale ?? false;
  const age = state?.age_seconds ?? null;

  return (
    <div className="wrap">
      <Header bot={snap?.bot ?? null} updatedLabel={updatedLabel} userName={userName} />

      {error && (
        <div className="banner error">
          <span className="dot" />
          Couldn&apos;t reach the dashboard API: {error}
        </div>
      )}
      {!error && !snap && (
        <div className="banner">
          <span className="dot" />
          Waiting for the first snapshot from the bot. Make sure the local exporter
          (scripts/dashboard_exporter.py) is running and pointed at this dashboard.
        </div>
      )}
      {!error && snap && stale && (
        <div className="banner">
          <span className="dot" />
          Data is stale{age != null ? ` (last update ${Math.round(age)}s ago)` : ""} — the exporter
          or the bot may have stopped. Showing the last known state.
        </div>
      )}

      {snap && (
        <>
          <Kpis s={snap.summary} />
          <div className="grid">
            <OpenPositions positions={snap.open_positions} />
            <div className="stack">
              <EquityChart curve={snap.equity_curve} />
              <PerSymbol rows={snap.per_symbol} />
              <ExitMix mix={snap.exit_mix} />
              <ExitFeed closed={snap.closed_positions} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
