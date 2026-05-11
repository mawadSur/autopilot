# Autopilot launchd schedules (macOS)

Per-job plists for the nightly + weekly jobs the operator runs locally.
These are templates — the paths are hard-coded to `/Users/mawad/Desktop/autopilot`;
copy + edit before installing on another machine.

## Jobs

| Plist | Cadence (UTC) | What it runs |
|---|---|---|
| `com.autopilot.alpha-lab.plist` | Daily 05:00 | `alpha_lab.nightly_runner` (cross-asset rank-IC miner) |
| `com.autopilot.reconciliation.plist` | Daily 03:00 | `ops.reconciliation_cli` (Coinbase ↔ position-store balance check) |
| `com.autopilot.e3-weekly.plist` | Sunday 06:00 | `llm_strategy_gen.weekly_job` (LLM strategy-gen pipeline, dry-run by default) |

All three plists set `TZ=UTC` in `EnvironmentVariables` so the `Hour` /
`Weekday` fields are interpreted in UTC. Without the override, launchd
runs `StartCalendarInterval` in the user's local timezone.

## Install

```bash
mkdir -p ~/Library/LaunchAgents ~/Library/Logs/autopilot

# Install all three
for f in ops/launchd/com.autopilot.*.plist; do
  cp "$f" ~/Library/LaunchAgents/
  launchctl load -w ~/Library/LaunchAgents/$(basename "$f")
done
```

## Verify

```bash
launchctl list | grep com.autopilot
```

Each loaded job appears as `<PID-or-dash> <last-exit> com.autopilot.<name>`.

## Uninstall

```bash
for f in ops/launchd/com.autopilot.*.plist; do
  name=$(basename "$f")
  launchctl unload -w ~/Library/LaunchAgents/"$name"
  rm ~/Library/LaunchAgents/"$name"
done
```

## Pre-flight checklist before enabling

- **alpha-lab**: harmless on first run — `build_default_feature_sources()`
  returns `[]` by default, so the runner logs "no sources wired" and exits 0.
  Wire production sources by overriding the factory in a private deploy module.
- **reconciliation**: needs `REDIS_URL`, `COINBASE_API_KEY`, `COINBASE_API_SECRET`
  in the launchd environment (add to `EnvironmentVariables` in the plist
  or load from a shell profile via a wrapper script).
- **e3-weekly**: stays in dry-run until **all** of: Docker sandboxing for
  LLM proposals (roadmap #8), real walk-forward backtest (#9), and a wired
  `gh_runner` via `make_subprocess_gh_runner()`. Without these, the job
  logs "would have done X" but never writes files or opens PRs.

## Logs

`StandardOutPath` and `StandardErrorPath` point at `~/Library/Logs/autopilot/`.
`ls ~/Library/Logs/autopilot/` to inspect. The logs are append-only — rotate
via `logrotate` or a periodic `truncate -s 0` if they grow.
