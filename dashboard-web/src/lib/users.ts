import bcrypt from "bcryptjs";
import { kv, kvConfigured, userKey } from "./kv";

// Viewer accounts live in KV under `user:{email}`. Passwords are bcrypt-hashed.
// Accounts are created out-of-band via scripts/add-user.mjs (admin action) —
// there is no public sign-up.

export interface UserRecord {
  email: string;
  name?: string;
  passwordHash: string;
  createdAt: string;
}

// Dev-only login: when KV is not configured (local `next dev` fallback), allow
// a single account defined via DEV_LOGIN_EMAIL/DEV_LOGIN_PASSWORD so the flow
// can be exercised without provisioning KV. Never active in production (KV is
// always configured there).
let _devUserCache: UserRecord | null = null;
async function devUser(email: string): Promise<UserRecord | null> {
  const devEmail = process.env.DEV_LOGIN_EMAIL?.toLowerCase().trim();
  const devPass = process.env.DEV_LOGIN_PASSWORD;
  if (kvConfigured() || !devEmail || !devPass) return null;
  if (email.toLowerCase().trim() !== devEmail) return null;
  if (!_devUserCache) {
    _devUserCache = {
      email: devEmail,
      name: process.env.DEV_LOGIN_NAME || "Dev",
      passwordHash: await bcrypt.hash(devPass, 10),
      createdAt: new Date().toISOString(),
    };
  }
  return _devUserCache;
}

export async function getUser(email: string): Promise<UserRecord | null> {
  const dev = await devUser(email);
  if (dev) return dev;

  const raw = await kv().get<UserRecord | string>(userKey(email));
  if (!raw) return null;
  // Upstash auto-deserializes JSON, but tolerate a stringified blob too.
  if (typeof raw === "string") {
    try {
      return JSON.parse(raw) as UserRecord;
    } catch {
      return null;
    }
  }
  return raw as UserRecord;
}

export async function verifyCredentials(
  email: string,
  password: string,
): Promise<UserRecord | null> {
  const user = await getUser(email);
  if (!user) {
    // Guard against user-enumeration timing: run a dummy compare.
    await bcrypt.compare(password, "$2a$10$invalidinvalidinvalidinvalidinvalidinv");
    return null;
  }
  const ok = await bcrypt.compare(password, user.passwordHash);
  return ok ? user : null;
}

export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, 10);
}
