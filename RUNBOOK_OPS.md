# Autopilot Crypto Bot — Operations Runbook

Last updated: 2026-05-18 (Sprint 1 Wave 2).

This runbook is the single-page reference an operator uses when something
goes wrong with the live or paper supervisor. It assumes you already have
`.venv/` populated, Redis reachable at `$REDIS_URL` (or
`redis://localhost:6379/0` by default), and any required exchange/API env
vars in `.env`.

For background on the crypto supervisor design, see `Readme.md` § "Run
the multi-symbol crypto supervisor". For the prediction-market stack,
see `CLAUDE.md`.

## 1. Start / stop the supervisor

### Live + paper supervisor (`src/live_supervisor.py`)

This is the canonical entry point for crypto trading. Paper mode is the
default; live mode requires per-symbol shakedown gates (≥14 days clean
PnL) to be unlocked.

Start a paper-mode session with all three crypto symbols, multiprocessing
per symbol, logs to a timestamped `runs/<ts>_*/supervisor.log`:

```bash
./.venv/bin/python src/live_supervisor.py \
    --mode paper \
    --symbols ETH/USDT,BTC/USDT,SOL/USDT \
    --workers \
    --log-dir ./runs \
    --interval 5
```

Useful flags (full list via `--help`):

| flag | meaning |
|------|---------|
| `--mode {paper,live}` | Operator intent. `live` requires shakedown unlock. |
| `--symbols a,b,c` | Comma-separated symbols. |
| `--interval N` | Tick interval in seconds (default 5). |
| `--bankroll N` | Starting equity in USD (default 10000). |
| `--risk-pct N` | Fraction of bankroll per trade (default 0.005). |
| `--min-confidence N` | Model confidence floor (default 0.6). |
| `--once` | Single tick then exit (cron-friendly). |
| `--workers` | One child process per tradeable (Lane D D3 multiprocessing). |
| `--log-dir DIR` | Write `supervisor.log` to a timestamped subdir. |
| `--auto-pause-enabled` | Opt-in daily-loss + confidence-shift auto-pause gate. |

Stop the supervisor cleanly via `Ctrl-C` (it catches `SIGINT` and
terminates child workers via the shutdown flag). For multiprocessing
runs, give it ~7s to drain children before re-issuing the signal.

### Legacy paper-trade entry (`src/paper_trade.py`)

This is the older single-file forward-only paper backtester. It does NOT
share Redis state with `live_supervisor.py` and is kept around for
1-minute ETH model bake-offs. Drive it via the `TradingConfig` settings
in `src/config.py` (`THR_LONG`, `CAPITAL`, etc.) — there is no
`argparse` surface:

```bash
./.venv/bin/python src/paper_trade.py
```

Output is the trade blotter + `$`-formatted summary on stdout. If you
want supervisor-style Redis-backed paper trading, use `live_supervisor.py
--mode paper` instead.

## 2. Force-flat paper positions

`scripts/force_flat_paper.py` mirrors `LiveSupervisor._paper_force_flat`:
closes every paper-tagged open position at the current Coinbase ticker
mid ± 5 bps slippage, records realized PnL via `position_store.record_close`,
and prints a per-position blotter.

This is the SAFE path — positions show up in `list_closed_today` and
`daily_realized_pnl_usd`.

```bash
./.venv/bin/python scripts/force_flat_paper.py
```

Live-tagged positions are skipped (the script's `if position.exchange !=
"coinbase-paper"` guard); use the standard supervisor force-flat or the
exchange UI for those.

## 3. Cleanup zombie paper positions

`scripts/cleanup_zombies.py` is the BLUNT-FORCE tool for the case where
a paper position is stuck open with no realistic chance of closing
cleanly (stale entry against bad ticker data, supervisor crash before
fill confirmation, position outlived a supervisor restart). It removes
the position from `open_set` and deletes the blob outright — NO realized
PnL is recorded, NO closed-set entry is created. The position vanishes
from the store.

Dry-run by default — `--write` must be passed explicitly. Live-tagged
positions are NEVER touched.

```bash
# Dry-run: see what would be purged. Default age threshold = 24h.
./.venv/bin/python scripts/cleanup_zombies.py

# Customize the threshold (e.g. 6h).
./.venv/bin/python scripts/cleanup_zombies.py --hours 6

# Actually delete.
./.venv/bin/python scripts/cleanup_zombies.py --hours 24 --write

# Non-default Redis target.
./.venv/bin/python scripts/cleanup_zombies.py \
    --redis-url redis://localhost:6380/0 --namespace autopilot-staging
```

Prefer `force_flat_paper.py` first — only fall back to `cleanup_zombies.py`
when force-flat itself fails (e.g. ticker fetch keeps timing out, or
the position blob is in a state record_close can't process).

## 4. `.kill_switch` semantics

The kill-switch is a file on disk at `$KILL_SWITCH_FILE`. Its presence
trips the supervisor's highest-priority circuit breaker
(`src/risk/circuit_breakers.py` `is_kill_switch_tripped`).

### Trip behaviour (edge-triggered, commit `aacc427`)

* **False → True edge** (operator drops the file): the supervisor
  increments `_kill_switch_trips_today`, calls `_force_flat_all(reason=
  "kill_switch_file_present")`, and fires a single notifier alert.
* **Held tick** (file still present, latch active): subsequent ticks
  return `action_taken="force_flatted"` with `notes="kill_switch_held"`
  — no re-count, no re-alert, no re-flatten attempt.
* **True → False edge** (operator removes the file): the latch clears,
  a LOG line records "kill switch cleared", and ticks resume normally.

If the supervisor crashed mid-trip (or DNS-flake prevented force-flatten
from running), the latch in memory is gone but the file may still be
present. On restart the trip path re-runs once because the latch starts
clean.

### Verify it's absent

```bash
# Absent path = supervisor will run normally.
[ -e "$KILL_SWITCH_FILE" ] && echo "PRESENT — supervisor will refuse new entries" || echo "absent"
```

### Clear the held latch

If the file is present and you want the supervisor to resume:

```bash
rm "$KILL_SWITCH_FILE"
```

The next tick will log "kill switch cleared (file removed); resuming
normal ticks" and the latch clears. If you want to STAY halted but
silence the held-tick log churn, leave the file present — held ticks
are quiet by design (no alerts after the first edge).

## 5. Coinbase REST DNS-flake recovery

Symptoms (per the 2026-05-16 incident memory): `get_ticker` calls
intermittently timing out, supervisor reports "consecutive_errors"
breaker trip, kill-switch held with no fills, `paper.log` shows
`ExchangeError` clusters.

### Mitigations (in order)

1. **Verify DNS resolves** to `api.coinbase.com` /
   `api.exchange.coinbase.com`:

   ```bash
   dig +short api.coinbase.com
   ```

   Empty / SERVFAIL = local resolver problem; restart `mDNSResponder` on
   macOS (`sudo killall -HUP mDNSResponder`) or your `systemd-resolved`
   on Linux.

2. **Bump the REST timeout** in `CoinbaseExchange` if the failure mode
   is slow-DNS rather than no-DNS. The default request timeout is 10s;
   move it to 30s as a one-line patch in
   `src/exchanges/coinbase.py` if the operator network is consistently
   slow (typical on coffee-shop / hotel WiFi).

3. **If kill-switch tripped from consecutive errors**, the latch behaviour
   from § 4 applies: clear the kill-switch file, let the supervisor
   re-fetch on the next tick. If DNS recovered, you'll see the first
   successful `get_ticker` reset `_consecutive_errors`.

4. **If positions are still open from before the flake**, run
   `scripts/force_flat_paper.py` (it uses the same Coinbase
   `get_ticker` call, so it relies on DNS having recovered) OR — if
   the position is paper and you accept losing the realized-PnL
   record — `scripts/cleanup_zombies.py --write` to drop them outright.

5. **Geo-block detector**: if `dig` works but every REST call returns
   401 / 403 / "service not available in your region", you're hitting a
   Coinbase geo-block (the prediction-market Polymarket stack has the
   same failure mode against a different geo). VPN, or move the
   supervisor to a different egress IP.

## 6. Breaker cap recovery

When the daily loss limit trips (`DAILY_LOSS_LIMIT_USD` env, or its
default), the supervisor's circuit breaker emits
`recommended_action="halt_new_entries"` for the rest of the UTC day.
The supervisor will tick (read tickers, capture snapshots) but will not
place new orders. This is by design.

### Resume earlier than UTC midnight

* The breaker rolls automatically at UTC midnight when `daily_close()`
  runs — no operator action needed for the standard recovery path.
* If the operator needs to resume early (e.g. the daily-loss limit was
  hit because of a real fill, not a runaway loop), the cleanest path is
  to **stop the supervisor, clear the day's closed-set in Redis,**
  then restart:

  ```bash
  # Verify which closed set you'd be wiping — replace YYYY-MM-DD with today UTC.
  redis-cli -u "$REDIS_URL" smembers "autopilot:closed:$(date -u +%Y-%m-%d)"

  # Wipe the day's realized-PnL set (this resets daily_realized_pnl_usd
  # to zero, which clears the daily-loss breaker on the next tick).
  redis-cli -u "$REDIS_URL" del "autopilot:closed:$(date -u +%Y-%m-%d)"
  ```

  ⚠ This DROPS the audit trail for today's closed positions in Redis
  (the JSON files under `runs/<ts>_*/` and the per-position blob keys
  still exist). Only do this if you understand you're trading
  auditability for resume-now.

* If the breaker was tripped because positions were force-flatted with
  large losses, also clear the `.kill_switch` file if present (§ 4)
  AND check for `~/.autopilot_auto_paused` (the auto-pause marker):

  ```bash
  ls -la ~/.autopilot_auto_paused
  rm ~/.autopilot_auto_paused   # only after operator review
  ```

## 7. Log locations

| log | written by | purpose |
|-----|------------|---------|
| `runs/<UTC-ts>_<symbols>/supervisor.log` | `live_supervisor.py` (when `--log-dir runs`) | Per-run rotating LOGGER output. The canonical place to look first. |
| `paper.log` | legacy paper-trade harness | Stdout capture from older `paper_trade.py` runs. |
| `paper_xgb.log` | legacy XGBoost paper bake-off | Stdout capture from the XGBoost variant. |
| `logs/discord_paper_bridge.log` | `scripts/discord_paper_bridge.py` | Discord notifier bridge. |
| `logs/telegram_control_bot.log` | `scripts/telegram_control_bot.py` | Telegram bot. |
| `logs/eth_paper_multiday*/` | older paper bake-off runs | Multi-day ETH paper runs (legacy). |
| `logs/retrain_*.log` | `scripts/retrain_all_crypto_models.sh` | Retrain output (BTC/ETH/SOL XGBoost). |

For real-time tailing of the current supervisor run, prefer
`src/paper_session_monitor.py runs/<UTC-ts>_*/supervisor.log --follow` —
it parses the structured log lines and prints rolling per-symbol stats
rather than raw log spam.

## 8. Sprint 1 exit policy

Sprint 1 Wave 2 wires `src/exit_policy.py` into the supervisor's tick loop
so every open position gets evaluated for stop-loss, take-profit,
trailing-stop, time-stop, and (optionally) signal-reversal **before** any
new entry is considered on the same tick. The supervisor's
``never-reopens-on-same-tick`` invariant guarantees a position closed on
tick N cannot be reopened by an entry signal on tick N — capital
preservation first. Per-tag close dispatch matches commit ``2b62a7d``:
paper-tagged positions fall back to ``_paper_force_flat`` on a transient
``ExchangeError``; live-tagged positions surface the error so the
consecutive-error / kill-switch breakers can react.

Master switches live in ``src/config.py`` and are env-overridable:
``EXIT_POLICY_ENABLED`` (default True, set to ``0`` to reproduce the
legacy "no exit policy" path — see the
``autopilot-no-exit-policy-blocker-2026-05-16`` memory for the symptoms),
``KELLY_SIZING_ENABLED`` (default True; pairs Kelly-fraction-from-regime
with ``KELLY_FLOOR_PCT`` / ``KELLY_CAP_PCT`` to clip the predictor's
resolved sizing), plus the per-reason thresholds ``STOP_LOSS_PCT``,
``TAKE_PROFIT_PCT``, ``TIME_STOP_BARS``, ``TRAILING_STOP_PCT``, and
``EXIT_SIGNAL_REVERSAL``. The supervisor exposes
``exits_by_reason_total{reason=...}`` counters,
``open_positions_count``, and ``oldest_open_position_age_s`` gauges
alongside the existing tick metrics — operators can plot these via the
Prometheus dashboard to see which exit reasons are firing and to spot
positions stuck open before the breaker cap intervenes.

## 9. Lane E nightly digest (Sprint 2)

`scripts/run_postmortem.py` runs the five Lane E forensic specialists
(Signal / Execution / Sizing / Context / Process) over every position
closed during a given UTC day, lets `LossPostmortemSynthesizer` classify
each trade's root cause + write the per-trade JSON+MD under
`runs/postmortems/`, and rolls everything into a single daily digest at
`docs/digests/YYYY-MM-DD.md`. A one-line summary is posted to Discord +
Telegram (whichever channels are configured via `DISCORD_WEBHOOK_URL` /
`TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`).

The launchd plist `launchd/com.autopilot.loss-postmortem.plist` schedules
this to run at **00:05 UTC** daily. The default `--date` is the previous
UTC day, so the 00:05 wake-up lines up with the just-closed UTC day.

### Install / start the job

```bash
# Load the plist (the -w flag enables it across reboots).
launchctl load -w /Users/mawad/Desktop/autopilot/launchd/com.autopilot.loss-postmortem.plist

# Verify it's registered. Should show the label and an exit status column.
launchctl list | grep loss-postmortem

# Fire it manually (handy after installing, or for a one-off ad-hoc run).
launchctl start com.autopilot.loss-postmortem

# Unload (e.g. to update the plist).
launchctl unload -w /Users/mawad/Desktop/autopilot/launchd/com.autopilot.loss-postmortem.plist
```

### Ad-hoc invocation

The launchd job hides flags from operators on purpose; for manual runs
the script accepts the full CLI surface:

```bash
# Default: yesterday in UTC, posts to configured channels.
env PYTHONPATH=src ./.venv/bin/python scripts/run_postmortem.py

# Specific date, no notifier dispatch (write the digest only).
env PYTHONPATH=src ./.venv/bin/python scripts/run_postmortem.py \
    --date 2026-05-18 --no-publish

# Dry-run (print to stdout, write nothing, post nothing). Use this when
# validating the digest format before letting launchd post anything.
env PYTHONPATH=src ./.venv/bin/python scripts/run_postmortem.py \
    --date 2026-05-18 --dry-run
```

### Where to look

| artefact | path |
|----------|------|
| Daily digest markdown | `docs/digests/YYYY-MM-DD.md` |
| Per-trade postmortem JSON + MD | `runs/postmortems/{trade_id}.{json,md}` |
| launchd stdout + stderr | `logs/postmortem.log` |
| Retrain queue (Signal-cluster feedback) | `runs/retrain_queue.jsonl` |
| Risk recommendations (Sizing-cluster feedback) | `runs/risk_recommendations.jsonl` |

If Redis is unreachable when the job fires, the digest is still written
with a "Redis unreachable, no data analyzed today" header and the
process exits 0 — launchd does NOT retry. Re-run manually via
`launchctl start com.autopilot.loss-postmortem` once Redis is back.

## 10. Calibration drift monitor (Sprint 2)

`scripts/diagnose_calibration_drift.py` is the daily reliability check
for closed trades. It buckets each closed position by its entry-time
model confidence, compares the realised winrate in each bucket against
the bucket midpoint, fits a weighted least-squares line, and alerts
when the slope drifts more than ±10% from 1.0. Sibling to
`scripts/diagnose_live_features.py`; intended to run on the same cron,
right after it.

Recommended cadence: daily at 00:10 UTC (5 minutes after the Lane E
nightly digest, so the just-closed UTC day's data is settled).

```bash
# Today (UTC) only, with Telegram + Discord alert on verdict=ALERT.
./.venv/bin/python scripts/diagnose_calibration_drift.py \
    --date $(date -u +%F) --alert

# Past week, all symbols, stdout only.
./.venv/bin/python scripts/diagnose_calibration_drift.py --window-days 7

# Single symbol + JSON sidecar for downstream consumption.
./.venv/bin/python scripts/diagnose_calibration_drift.py \
    --date 2026-05-18 --symbol ETH/USD \
    --out runs/calibration/2026-05-18.json
```

### Verdict states

| verdict | meaning | exit |
|---------|---------|------|
| `OK` | slope ∈ [0.90, 1.10] -- model is well-calibrated within tolerance | 0 |
| `ALERT` | slope outside the tolerance band -- entry-conf calibration has drifted | 1 only when `--alert` set, else 0 |
| `NO_DATA` | empty window, or no bucket meets `--min-n-per-bucket` (default 5) -- too few trades to assess | 0 |

If Redis is unreachable when the job fires, the script prints
`Redis unreachable, skipping today's calibration check` and exits 0 --
the cron does not need to retry. Non-connection `RedisError`s exit 2
and log a traceback for operator triage.

## 11. Quick triage flowchart

```
supervisor not placing trades?
├── is $KILL_SWITCH_FILE present?            → § 4
├── is ~/.autopilot_auto_paused present?     → § 6 (auto-pause marker)
├── is daily_realized_pnl_usd ≤ -$LIMIT?     → § 6 (daily-loss breaker)
├── is shakedown unlock missing for live?    → review .shakedown.json
├── are positions stuck open?                → § 2 then § 3
├── are tickers timing out / DNS flaky?      → § 5
└── otherwise                                → check runs/<ts>/supervisor.log
```

## 12. OutcomeAdjuster daily (Sprint 2 #6)

`scripts/run_outcome_adjuster.py` walks the previous UTC day's closed
positions, recomputes per-regime streaks, and nudges the
`optimal_threshold` returned by `RegimeLookup` on a per-regime basis.
The adjustment is BOUNDED (default ±0.05) so a bad streak cannot
silently muzzle the bot — and it RELAXES toward zero on consecutive
winning trades.

State lives in a single Redis hash:

```bash
# Inspect current adjustments (no TTL — slow-moving operator state).
redis-cli HGETALL autopilot:regime_outcome_adjustment
# 1) "trend_down"
# 2) "+0.020000"
# 3) "chop"
# 4) "-0.010000"
```

Manual reset (e.g. after a model retrain invalidates prior streaks):

```bash
# Clear ONE label.
./.venv/bin/python scripts/run_outcome_adjuster.py --reset chop

# Clear the entire hash.
./.venv/bin/python scripts/run_outcome_adjuster.py --reset all

# Equivalent raw Redis:
redis-cli HDEL autopilot:regime_outcome_adjustment chop
redis-cli DEL  autopilot:regime_outcome_adjustment
```

Ad-hoc dry-run (no writes — useful before a manual operator decision):

```bash
env PYTHONPATH=src ./.venv/bin/python \
    scripts/run_outcome_adjuster.py --date 2026-05-18 --dry-run
```

### Install / start the job

```bash
cp launchd/com.autopilot.outcome-adjuster.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.autopilot.outcome-adjuster.plist
launchctl start com.autopilot.outcome-adjuster
```

Runs at 00:10 UTC daily (5 minutes after the Lane E digest at 00:05).
Logs to `logs/outcome_adjuster.log`. Exit code 0 even on Redis-down so
launchd does not retry — re-run manually via `launchctl start
com.autopilot.outcome-adjuster` once Redis is back.

### How the adjustments influence live trading

Each tick the live predictor calls `RegimeLookup.resolve_params(...)`.
When an `OutcomeAdjuster` is wired (the predictor wires one automatically
when `REDIS_URL` is set), the lookup adds the current adjustment for the
closest neighbor's regime label to the resolved `optimal_threshold`,
clipped to `[0, 1]`. The resolved dict gains two observable fields
(`_outcome_adjustment_delta`, `_regime_label`) so the supervisor and
diagnostic scripts can see exactly which regime + delta steered the
threshold. With no adjuster wired, behavior is identical to today.
