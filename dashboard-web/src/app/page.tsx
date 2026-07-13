import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { SESSION_COOKIE, verifySessionToken } from "@/lib/session";
import Dashboard from "@/components/Dashboard";

// Server component: middleware already gates access, but we also read the
// session here to greet the viewer by name (and redirect defensively).
export const dynamic = "force-dynamic";

export default async function Page() {
  const store = await cookies();
  const session = await verifySessionToken(store.get(SESSION_COOKIE)?.value);
  if (!session) redirect("/login");

  const userName = session.name || session.email || null;
  return <Dashboard userName={userName} />;
}
