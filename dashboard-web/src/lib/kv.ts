import { Redis } from "@upstash/redis";

// KV abstraction over Upstash / Vercel KV.
//
// In production the app reads either the Vercel-KV env pair
// (KV_REST_API_URL / KV_REST_API_TOKEN, injected automatically when you add a
// Vercel KV store) or the raw Upstash pair (UPSTASH_REDIS_REST_URL / ..._TOKEN).
//
// For LOCAL DEV ONLY, when no cloud creds are present, we fall back to an
// in-process Map so the whole flow (login → ingest → state → render) can be
// exercised with `next dev` and no external account. `next dev` runs a single
// long-lived Node process, so writes from /api/ingest are visible to
// /api/state. This fallback NEVER activates in production because Vercel always
// injects the KV creds. A one-time warning is logged when it's used.

const url = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL || "";
const token = process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN || "";

export interface KvClient {
  get<T = unknown>(key: string): Promise<T | null>;
  set(key: string, value: string): Promise<unknown>;
}

let _redis: KvClient | null = null;
let _warnedDev = false;

// process-wide in-memory store for the dev fallback. Lives on globalThis so it
// is shared across route bundles + survives HMR within one `next dev` process
// (module-level state alone is NOT shared across route handlers — the same
// reason production requires real KV).
const _g = globalThis as unknown as { __autopilot_kv?: Map<string, string> };
const _mem = _g.__autopilot_kv ?? (_g.__autopilot_kv = new Map<string, string>());
const _memClient: KvClient = {
  async get<T = unknown>(key: string): Promise<T | null> {
    const v = _mem.get(key);
    return (v === undefined ? null : (v as unknown as T)) ?? null;
  },
  async set(key: string, value: string): Promise<unknown> {
    _mem.set(key, value);
    return "OK";
  },
};

export function kvConfigured(): boolean {
  return Boolean(url && token);
}

export function kv(): KvClient {
  if (kvConfigured()) {
    if (!_redis) _redis = new Redis({ url, token }) as unknown as KvClient;
    return _redis;
  }
  // No cloud creds → dev fallback (only reachable when running without KV env).
  if (process.env.NODE_ENV === "production") {
    throw new Error(
      "KV is not configured in production: set KV_REST_API_URL + KV_REST_API_TOKEN " +
        "(or the UPSTASH_REDIS_REST_* pair).",
    );
  }
  if (!_warnedDev) {
    console.warn(
      "[kv] No KV credentials found — using an in-memory dev store. " +
        "Data will not persist across restarts. Set KV_REST_API_URL/TOKEN for real KV.",
    );
    _warnedDev = true;
  }
  return _memClient;
}

// Key names used across the app.
export const SNAPSHOT_KEY = "dashboard:snapshot";
export const SNAPSHOT_META_KEY = "dashboard:snapshot:received_at"; // ISO string
export const userKey = (email: string) => `user:${email.toLowerCase().trim()}`;
