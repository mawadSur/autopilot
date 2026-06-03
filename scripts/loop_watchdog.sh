#!/usr/bin/env bash
# SHADOW-loop watchdog — keep the paper-trading loops + dashboard alive.
#
# Idempotent: only (re)starts a process whose recorded PID is not currently
# running, so it is safe to run repeatedly (cron every few minutes) and safe to
# run while the loops are already up. It NEVER places orders — it only relaunches
# the existing SHADOW-only loops with their canonical config (the single source
# of truth lives here).
#
# Install (auto-restart on crash; survives as long as cron runs):
#   crontab -l 2>/dev/null | grep -q loop_watchdog || \
#     ( crontab -l 2>/dev/null; echo "*/5 * * * * cd '$PWD' && ./scripts/loop_watchdog.sh >> runs/watchdog.cron.log 2>&1" ) | crontab -
#
# Caveat: this auto-restarts on process crash and keeps things up while the WSL
# instance is running. A full Windows reboot stops WSL (and cron) until WSL is
# started again — for true reboot-survival, launch this watchdog (or wsl) from
# Windows Task Scheduler at logon. Documented honestly rather than pretended.
set -u

cd "$(dirname "$0")/.." || exit 1
PY="./.venv/bin/python"
LOG="runs/watchdog.log"
mkdir -p runs

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

# ensure <name> <pidfile> <command...>
ensure() {
  local name="$1" pidfile="$2"; shift 2
  local pid; pid="$(cat "$pidfile" 2>/dev/null || true)"
  if alive "$pid"; then
    return 0
  fi
  echo "$(ts) [$name] not running (pid='${pid:-none}') -> restarting" >> "$LOG"
  nohup "$@" > "runs/${name}.out" 2>&1 &
  echo $! > "$pidfile"
  echo "$(ts) [$name] started pid $(cat "$pidfile")" >> "$LOG"
}

# (B) the complete honest whale-follow pipeline — filters + book-aware
# entry/exit + risk gates. Fresh ledger forward-test.
ensure whale_honest_loop runs/whale_honest_loop.pid \
  $PY -u src/whale_follow_runner.py \
    --roster-source leaderboard --leaderboard-window all --leaderboard-limit 100 \
    --interval 1800 --rank-refresh-scans 48 --min-convergence 3 --min-confidence 0.5 \
    --stop-loss-pct 0.20 --take-profit-pct 0.30 --take-profit-price 0.95 \
    --min-entry-price 0.15 --max-entry-price 0.85 --max-book-frac 0.05 --depth-band 0.05 \
    --max-exposure 2000 --daily-loss-limit 200 \
    --size 100 --bankroll 1000 --discord --ledger-path runs/whale_honest_ledger.jsonl

# (T9) the market-neutral arbitrage hedge — YES+NO < $1 intra-market, shadow only.
ensure arb_shadow_loop runs/arb_shadow_loop.pid \
  $PY -u src/arb_shadow_runner.py \
    --interval 300 --max-markets 200 --min-edge 0.005 \
    --size 100 --bankroll 1000 --discord --ledger-path runs/arb_shadow_ledger.jsonl

# Read-only dashboard for the honest ledger. Checked by PORT (it's a singleton
# bound to :8888) rather than a PID file — more robust than pid tracking on the
# WSL /mnt/c filesystem, and avoids ever spawning a second server that can't bind.
if curl -s -o /dev/null --max-time 4 "http://127.0.0.1:8888/" 2>/dev/null; then
  : # dashboard already serving
else
  echo "$(ts) [dashboard] :8888 not responding -> restarting" >> "$LOG"
  nohup $PY src/dashboard/server.py --port 8888 \
    --ledger-path runs/whale_honest_ledger.jsonl --bankroll 1000 \
    > runs/dashboard.out 2>&1 &
  echo $! > runs/dashboard_8888.pid
  echo "$(ts) [dashboard] started pid $(cat runs/dashboard_8888.pid)" >> "$LOG"
fi
