import { NextResponse } from "next/server";
import { kv, SNAPSHOT_KEY, SNAPSHOT_META_KEY } from "@/lib/kv";
import type { Snapshot, StateResponse } from "@/lib/types";

// Session-gated by middleware. Returns the latest published snapshot plus a
// staleness flag so the UI can warn when the exporter has stopped.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const ttl = Number(process.env.SNAPSHOT_TTL_SECONDS || "120");
  const now = new Date();

  let snapshot: Snapshot | null = null;
  let receivedAt: string | null = null;
  try {
    const client = kv();
    const [rawSnap, rawMeta] = await Promise.all([
      client.get<Snapshot | string>(SNAPSHOT_KEY),
      client.get<string>(SNAPSHOT_META_KEY),
    ]);
    if (rawSnap) {
      snapshot =
        typeof rawSnap === "string" ? (JSON.parse(rawSnap) as Snapshot) : (rawSnap as Snapshot);
    }
    receivedAt = rawMeta ?? null;
  } catch (err) {
    console.error("state: KV read failed", err);
    return NextResponse.json(
      { snapshot: null, stale: true, age_seconds: null, server_time_utc: now.toISOString(), error: "storage read failed" },
      { status: 200 },
    );
  }

  let ageSeconds: number | null = null;
  const stampIso = receivedAt || snapshot?.generated_at_utc || null;
  if (stampIso) {
    const stamp = new Date(stampIso);
    if (!Number.isNaN(stamp.getTime())) {
      ageSeconds = Math.max(0, (now.getTime() - stamp.getTime()) / 1000);
    }
  }
  const stale = ageSeconds === null ? true : ageSeconds > ttl;

  const body: StateResponse = {
    snapshot,
    stale,
    age_seconds: ageSeconds,
    server_time_utc: now.toISOString(),
  };
  return NextResponse.json(body);
}
