import { NextRequest, NextResponse } from "next/server";
import { createHash, timingSafeEqual } from "crypto";
import { kv, SNAPSHOT_KEY, SNAPSHOT_META_KEY } from "@/lib/kv";

// Machine-to-machine endpoint: the local dashboard_exporter.py POSTs snapshots
// here with a shared secret. Node runtime for crypto.timingSafeEqual + KV.
export const runtime = "nodejs";

function secretsMatch(provided: string, expected: string): boolean {
  // Hash both to fixed length so timingSafeEqual never throws on length diff.
  const a = createHash("sha256").update(provided).digest();
  const b = createHash("sha256").update(expected).digest();
  return timingSafeEqual(a, b);
}

export async function POST(req: NextRequest) {
  const expected = process.env.EXPORTER_SECRET || "";
  if (!expected) {
    return NextResponse.json(
      { error: "EXPORTER_SECRET is not configured on the server" },
      { status: 500 },
    );
  }

  const provided = req.headers.get("x-exporter-secret") || "";
  if (!provided || !secretsMatch(provided, expected)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  let snapshot: unknown;
  try {
    snapshot = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON body" }, { status: 400 });
  }

  // Minimal shape guard — reject obvious garbage so the UI never renders junk.
  if (
    !snapshot ||
    typeof snapshot !== "object" ||
    !("summary" in (snapshot as Record<string, unknown>)) ||
    !("schema_version" in (snapshot as Record<string, unknown>))
  ) {
    return NextResponse.json(
      { error: "payload does not look like a dashboard snapshot" },
      { status: 422 },
    );
  }

  try {
    const client = kv();
    const receivedAt = new Date().toISOString();
    await Promise.all([
      client.set(SNAPSHOT_KEY, JSON.stringify(snapshot)),
      client.set(SNAPSHOT_META_KEY, receivedAt),
    ]);
    return NextResponse.json({ ok: true, received_at: receivedAt });
  } catch (err) {
    console.error("ingest: KV write failed", err);
    return NextResponse.json({ error: "storage write failed" }, { status: 502 });
  }
}
