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

# Crypto funding-carry SHADOW runner — hourly accrual of the top net carries.
ensure funding_carry_runner runs/funding_carry_runner.pid \
  $PY -u src/funding_carry_runner.py \
    --exchange hyperliquid --period-hours 1 --interval 3600 \
    --top 8 --min-net 0.15 --notional 1000 \
    --ledger-path runs/funding_carry_ledger.jsonl
