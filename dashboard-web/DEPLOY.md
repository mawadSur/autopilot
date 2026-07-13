# Deploying the Autopilot Dashboard to Vercel

This walks you from zero to a live, login-gated dashboard that shows the bot's
trading in near-real-time. ~15 minutes. Everything below runs from the
`dashboard-web/` directory unless noted.

## What you'll end up with

```
Your Mac:  live_supervisor (bot) -> local Redis
           dashboard_exporter.py  --every 30s-->  https://<you>.vercel.app/api/ingest
Vercel:    Next.js dashboard  <- reads snapshot from Vercel KV  <- viewers log in
```

The bot and its Redis never leave your machine. Only a computed snapshot
(no keys, no secrets) is pushed out.

---

## 1. Prerequisites

- A **Vercel account** (free): https://vercel.com/signup
- The Vercel CLI (already installed here): `vercel --version`
- Log in once (interactive — opens a browser):

  ```bash
  vercel login
  ```

## 2. Create the project

From `dashboard-web/`:

```bash
vercel link          # create/link a Vercel project (accept the prompts)
```

When asked for the root directory, it's the current dir (`dashboard-web`).

## 3. Add a KV (Redis) store

The dashboard stores the latest snapshot in Vercel KV (Upstash Redis).

1. Open the project on vercel.com → **Storage** → **Create Database** →
   **Upstash for Redis** (a.k.a. "KV"). Accept the free plan.
2. **Connect** it to this project. Vercel injects `KV_REST_API_URL` and
   `KV_REST_API_TOKEN` into the project's environment automatically.

(Alternatively, provision an Upstash Redis directly at upstash.com and set
`UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN` yourself — the app reads
either pair.)

## 4. Set the app secrets

Generate two secrets and add them to the project (all environments):

```bash
vercel env add SESSION_SECRET production     # paste: openssl rand -base64 48
vercel env add SESSION_SECRET preview        # (same value is fine)
vercel env add SESSION_SECRET development

vercel env add EXPORTER_SECRET production    # paste: openssl rand -base64 32
vercel env add EXPORTER_SECRET preview
vercel env add EXPORTER_SECRET development
```

Keep the `EXPORTER_SECRET` value handy — the local exporter needs the same one.

## 5. Deploy

```bash
vercel --prod
```

Note the production URL it prints, e.g. `https://autopilot-dashboard.vercel.app`.

## 6. Create viewer accounts

Pull the prod env locally so the admin script can reach KV, then add users:

```bash
vercel env pull .env.local
npm install                       # if you haven't already
npm run add-user -- alice@example.com "a-strong-password" "Alice"
npm run add-user -- bob@example.com   "another-password"   "Bob"
```

Each person signs in at the production URL with their email + password.
Re-running `add-user` for an existing email updates the password. To revoke
someone, delete their `user:<email>` key in the Upstash console.

## 7. Start the exporter (feeds the dashboard)

From the **repo root** (not `dashboard-web/`), with the venv:

```bash
DASHBOARD_INGEST_URL="https://<your-app>.vercel.app/api/ingest" \
DASHBOARD_EXPORTER_SECRET="<the EXPORTER_SECRET from step 4>" \
DASHBOARD_BANKROLL_USD=10000 \
PYTHONPATH=src ./.venv/bin/python scripts/dashboard_exporter.py --interval 30
```

You should see `published equity=$… open=… settled=…` lines every 30s, and the
dashboard will show live data within a minute.

### Keep it running (launchd)

To publish continuously without a terminal open, create
`~/Library/LaunchAgents/com.autopilot.dashboard-exporter.plist` (model it on the
existing `launchd/com.autopilot.*.plist` files), pointing at
`scripts/dashboard_exporter.py` with the two env vars above, then:

```bash
launchctl load ~/Library/LaunchAgents/com.autopilot.dashboard-exporter.plist
```

The exporter only needs the bot's local Redis + outbound HTTPS; it will keep the
dashboard fresh as long as your Mac is on. If the exporter stops, the dashboard
shows a "data is stale" banner rather than wrong numbers.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard says "Waiting for the first snapshot" | The exporter hasn't published yet. Check step 7 output for errors; confirm `DASHBOARD_INGEST_URL` ends in `/api/ingest`. |
| Exporter logs `HTTP 401` | `EXPORTER_SECRET` mismatch between the exporter and Vercel. |
| Exporter logs `HTTP 500 EXPORTER_SECRET not configured` | You didn't set `EXPORTER_SECRET` in Vercel (step 4) or didn't redeploy after. |
| Login always fails | The account doesn't exist in KV — run `add-user` (step 6). Confirm `.env.local` has the KV creds (`vercel env pull`). |
| "data is stale" banner | The exporter or the bot stopped. Restart the exporter; check the bot is running. |
| Numbers look wrong on first deploy | The exporter walks the last 14 days of closed trades from local Redis; that's expected history. Use `--days N` to change the window. |

## Security notes

- Viewers are read-only; there is no control surface — the dashboard cannot
  place, stop, or modify trades. It only displays a snapshot.
- The snapshot contains position/PnL data, never API keys or Redis credentials.
- `/api/ingest` is protected by a timing-safe secret check; all viewer pages and
  `/api/state` require a signed session cookie.
