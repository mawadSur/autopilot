import { SignJWT, jwtVerify, type JWTPayload } from "jose";

// Session = a short-ish-lived JWT stored in an HttpOnly, Secure cookie.
// Signed/verified with jose (edge-safe, so middleware can verify without a DB).

export const SESSION_COOKIE = "autopilot_session";
const SESSION_TTL_SECONDS = 60 * 60 * 24 * 7; // 7 days

function secretKey(): Uint8Array {
  const secret = process.env.SESSION_SECRET;
  if (!secret || secret.length < 16) {
    throw new Error(
      "SESSION_SECRET is missing or too short (need >= 16 chars). " +
        "Generate one with: openssl rand -base64 48",
    );
  }
  return new TextEncoder().encode(secret);
}

export interface SessionPayload extends JWTPayload {
  email: string;
  name?: string;
}

export async function createSessionToken(payload: {
  email: string;
  name?: string;
}): Promise<string> {
  return new SignJWT({ email: payload.email, name: payload.name })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime(`${SESSION_TTL_SECONDS}s`)
    .sign(secretKey());
}

export async function verifySessionToken(
  token: string | undefined | null,
): Promise<SessionPayload | null> {
  if (!token) return null;
  try {
    const { payload } = await jwtVerify(token, secretKey());
    if (typeof payload.email !== "string") return null;
    return payload as SessionPayload;
  } catch {
    return null;
  }
}

export const sessionCookieOptions = {
  httpOnly: true,
  secure: process.env.NODE_ENV === "production",
  sameSite: "lax" as const,
  path: "/",
  maxAge: SESSION_TTL_SECONDS,
};
