#!/usr/bin/env bash
# SHADOW-loop watchdog — keep the funding-carry shadow runner alive.
#
# Idempotent: only (re)starts a process whose recorded PID is dead, so it is safe
# to run repeatedly (cron every few minutes) and safe to run while the runner is
# already up. It NEVER places orders — it relaunches the existing SHADOW-only
# runner with its canonical config (the single source of truth lives here). The
# carry runner is restart-safe (it folds its append-only ledger on start), so a
# crash + relaunch resumes the same positions and cumulative carry.
#
# Install (auto-restart on crash; survives as long as cron runs):
#   crontab -l 2>/dev/null | grep -q loop_watchdog || \
#     ( crontab -l 2>/dev/null; echo "*/5 * * * * cd '$PWD' && ./scripts/loop_watchdog.sh >> runs/watchdog.cron.log 2>&1" ) | crontab -
#
# Caveat: auto-restarts on process crash + keeps things up while the WSL instance
# runs. A full Windows reboot stops WSL (and cron) until WSL is started again —
# for true reboot-survival, launch this (or `wsl`) from Windows Task Scheduler.
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

# Stablecoin YIELD shadow runner — hourly accrual of a Kraken-earn-style yield +
# live peg monitor on USDC/USDT/DAI, with Discord updates. SHADOW-only (public
# ticker reads + arithmetic; no orders/keys/custody). --apy is operator-supplied
# (verify on Kraken Earn). Supersedes the parked funding-carry runner.
ensure yield_shadow_runner runs/yield_shadow_runner.pid \
  $PY -u src/yield_shadow_runner.py \
    --stablecoin USDC --stablecoin USDT --stablecoin DAI \
    --apy 0.045 --notional 10000 --interval 3600 --depeg-threshold 0.005 \
    --discord --ledger-path runs/yield_ledger.jsonl
