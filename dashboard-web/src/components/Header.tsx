"use client";

import { useRouter } from "next/navigation";
import type { BotMeta } from "@/lib/types";

export default function Header({
  bot,
  updatedLabel,
  userName,
}: {
  bot: BotMeta | null;
  updatedLabel: string;
  userName: string | null;
}) {
  const router = useRouter();

  async function logout() {
    try {
      await fetch("/api/logout", { method: "POST" });
    } finally {
      router.push("/login");
      router.refresh();
    }
  }

  const symbols = bot?.symbols?.length ? bot.symbols.join(" · ") : "—";
  const mode = bot?.mode ?? "paper";
  const running = bot?.running;

  return (
    <header>
      <div className="brand">
        <div>
          <h1>
            Autopilot <span className="accent">Halal Bot</span>
          </h1>
          <div className="sub">
            {symbols}
            {bot?.exit_rule ? ` · exit ${bot.exit_rule}` : ""} · polls every 15s
          </div>
        </div>
      </div>
      <div className="header-right">
        {bot?.halal !== false && (
          <span className="badge halal">
            <span className="dot" />
            HALAL
          </span>
        )}
        <span className={"badge mode" + (mode === "live" ? " live" : "")}>
          <span className="dot" />
          {mode === "live" ? "LIVE" : "PAPER"}
          {running === false ? " · offline" : ""}
        </span>
        <span className="updated">
          updated <b>{updatedLabel}</b>
        </span>
        {userName && <span className="updated">· {userName}</span>}
        <button className="linklike" onClick={logout}>
          Sign out
        </button>
      </div>
    </header>
  );
}
