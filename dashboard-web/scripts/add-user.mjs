#!/usr/bin/env node
// Create or update a dashboard viewer account (invite-only; no public signup).
//
// Usage:
//   node scripts/add-user.mjs <email> <password> ["Full Name"]
//
// Reads KV creds from the environment (or dashboard-web/.env.local). Set:
//   KV_REST_API_URL + KV_REST_API_TOKEN   (Vercel KV), or
//   UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN   (Upstash direct)
//
// After deploying, pull the prod env locally with:  vercel env pull .env.local

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import bcrypt from "bcryptjs";
import { Redis } from "@upstash/redis";

const __dirname = dirname(fileURLToPath(import.meta.url));

// --- lightweight .env.local loader (does not override real env) ---
function loadEnvLocal() {
  const candidates = [join(__dirname, "..", ".env.local"), join(__dirname, "..", ".env")];
  for (const path of candidates) {
    try {
      const text = readFileSync(path, "utf8");
      for (const line of text.split("\n")) {
        const t = line.trim();
        if (!t || t.startsWith("#") || !t.includes("=")) continue;
        const idx = t.indexOf("=");
        const key = t.slice(0, idx).trim();
        let val = t.slice(idx + 1).trim();
        if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
          val = val.slice(1, -1);
        }
        if (!(key in process.env)) process.env[key] = val;
      }
    } catch {
      /* file may not exist — fine */
    }
  }
}

loadEnvLocal();

const [, , email, password, name] = process.argv;
if (!email || !password) {
  console.error('Usage: node scripts/add-user.mjs <email> <password> ["Full Name"]');
  process.exit(1);
}

const url = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL;
const token = process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN;
if (!url || !token) {
  console.error(
    "Missing KV credentials. Set KV_REST_API_URL + KV_REST_API_TOKEN " +
      "(or the UPSTASH_REDIS_REST_* pair) in the environment or dashboard-web/.env.local.\n" +
      "Tip: after deploying, run  vercel env pull .env.local",
  );
  process.exit(1);
}

const redis = new Redis({ url, token });
const key = `user:${email.toLowerCase().trim()}`;

const record = {
  email: email.toLowerCase().trim(),
  name: name || undefined,
  passwordHash: await bcrypt.hash(password, 10),
  createdAt: new Date().toISOString(),
};

const existing = await redis.get(key);
await redis.set(key, JSON.stringify(record));
console.log(`${existing ? "Updated" : "Created"} viewer account: ${record.email}${name ? ` (${name})` : ""}`);
