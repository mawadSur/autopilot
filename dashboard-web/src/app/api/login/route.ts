import { NextRequest, NextResponse } from "next/server";
import { verifyCredentials } from "@/lib/users";
import {
  SESSION_COOKIE,
  createSessionToken,
  sessionCookieOptions,
} from "@/lib/session";

// Runs in the Node runtime (bcrypt is not edge-friendly).
export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  let body: { email?: string; password?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid body" }, { status: 400 });
  }

  const email = (body.email || "").toString().trim().toLowerCase();
  const password = (body.password || "").toString();
  if (!email || !password) {
    return NextResponse.json(
      { error: "email and password are required" },
      { status: 400 },
    );
  }

  let user;
  try {
    user = await verifyCredentials(email, password);
  } catch (err) {
    // KV/config errors shouldn't leak details to the client.
    console.error("login: verifyCredentials failed", err);
    return NextResponse.json({ error: "server error" }, { status: 500 });
  }

  if (!user) {
    return NextResponse.json(
      { error: "invalid email or password" },
      { status: 401 },
    );
  }

  const token = await createSessionToken({ email: user.email, name: user.name });
  const res = NextResponse.json({ ok: true, email: user.email, name: user.name ?? null });
  res.cookies.set(SESSION_COOKIE, token, sessionCookieOptions);
  return res;
}
