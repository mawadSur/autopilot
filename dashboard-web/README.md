# Autopilot Dashboard

A login-gated web dashboard (Next.js on Vercel) that shows how the Autopilot
**halal crypto trading bot** is doing — equity, PnL, win rate, open positions,
recent exits, and per-symbol breakdown. Paper-trading only.

## Architecture

The bot runs **locally** and writes positions to **local Redis**. A Vercel
function can't reach local Redis, so a small local exporter bridges the gap:

```
live_supervisor  ->  local Redis (PositionStore)            [bot, unchanged]
scripts/dashboard_exporter.py  --HTTPS-->  /api/ingest  ->  Vercel KV (snapshot)
Vercel dashboard  <--reads KV--  behind per-user login
```

- `scripts/dashboard_exporter.py` (in the repo root, not here) reads the bot's
  Redis every ~30s, computes the snapshot, and POSTs it to `/api/ingest` with a
  shared secret.
- The Next.js app stores the latest snapshot in Vercel KV and serves it to
  logged-in viewers who poll `/api/state` every 15s.
- Auth: individual accounts (email + bcrypt password), JWT session cookie,
  edge middleware gate. Accounts are created with `npm run add-user`.

## Local development

```bash
cd dashboard-web
npm install
cp .env.example .env.local     # fill in the values (see below)
npm run dev                    # http://localhost:3000
```

You need a KV store even for local dev. Either create a Vercel KV store and
`vercel env pull .env.local`, or provision a free Upstash Redis and paste its
REST URL + token.

Create a login:

```bash
npm run add-user -- you@example.com "your-password" "Your Name"
```

Feed it live data from the running bot (from the repo root):

```bash
DASHBOARD_INGEST_URL="http://localhost:3000/api/ingest" \
DASHBOARD_EXPORTER_SECRET="<same as EXPORTER_SECRET in .env.local>" \
PYTHONPATH=src ./.venv/bin/python scripts/dashboard_exporter.py --interval 30
```

## Environment variables

| Var | Purpose |
|-----|---------|
| `SESSION_SECRET` | Signs viewer session JWTs. `openssl rand -base64 48` |
| `EXPORTER_SECRET` | Shared secret the exporter sends to `/api/ingest`. `openssl rand -base64 32` |
| `KV_REST_API_URL` / `KV_REST_API_TOKEN` | Vercel KV (auto-injected when you add a KV store) |
| `UPSTASH_REDIS_REST_URL` / `..._TOKEN` | Alternative if you use Upstash directly |
| `SNAPSHOT_TTL_SECONDS` | Age (s) after which the UI flags data as stale (default 120) |

## Deploy

See [`DEPLOY.md`](./DEPLOY.md) for the full Vercel + KV walkthrough.
