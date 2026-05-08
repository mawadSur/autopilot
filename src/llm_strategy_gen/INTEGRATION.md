# LLM Strategy-Gen Loop — INTEGRATION

CEO Plan Phase 5 / E3 — `llm_strategy_gen` package. **SKELETON.** Most
of the production wiring is deferred to follow-up PRs; this file
documents what *is* implemented, what *isn't*, and the operational
guardrails that must be in place before you ever flip
`--no-dry-run`.

## Pipeline at a glance

```
runs/postmortems/*.json
runs/performance_audit.json     ──► OutcomeAnalyst.analyze()        ──► [OutcomePattern, ...]
                                                                                  │
                                                                                  ▼
                                     FeatureProposalGenerator.propose()  ──► [FeatureProposal, ...]
                                                                                  │
                                                                                  ▼
                                     BacktestRunner.run()                 ──► BacktestResult
                                     (Sharpe-gate; STUB)                          │
                                                                                  ▼
                                     PROpener.open_pr()                   ──► gh pr create --draft
```

Top-level orchestrator: `WeeklyJob.run()` → `Dict[str, int]`.

## Required environment

| Variable                          | Required for          | Default                | Notes |
|-----------------------------------|-----------------------|------------------------|-------|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Real LLM calls       | _none_ (skipped)       | Skeleton degrades to no-op without it. |
| `GEMINI_MODEL`                    | Optional              | `gemini-2.5-flash`     | Passed through to LLM caller. |
| `GH_TOKEN`                        | `--no-dry-run`        | _none_                 | Used by `gh pr create`. |
| `AUTOPILOT_LLM_STRATEGY_GATE`     | Optional              | `0.2`                  | Sharpe-lift gate. CLI `--gate-threshold` wins. |

The skeleton's default wiring constructs `FeatureProposalGenerator`
with `llm_caller=None`, so a no-config invocation produces zero
proposals (and therefore zero PRs). Production deployment must wire a
real Gemini caller in — see "Wiring the LLM" below.

## Cron schedule

Recommended: weekly, Sunday 6am UTC.

```
0 6 * * 0  cd /opt/autopilot && \
           AUTOPILOT_LLM_STRATEGY_GATE=0.2 \
           ./.venv/bin/python -m llm_strategy_gen.weekly_job \
                --repo-root /opt/autopilot \
                --window-days 7 \
                --dry-run \
                >> /var/log/autopilot/llm_strategy_gen.log 2>&1
```

Switch `--dry-run` → `--no-dry-run` ONLY after every item under
"Pre-flight before live mode" is checked off.

## Pre-flight before live mode

1. **Real sandboxing for proposal code execution is in place.** See
   "Security caveats" below. The skeleton's `_validate_proposal_safety`
   is **not** a sandbox.
2. **The stub backtest is replaced with real retraining.** Today
   `BacktestRunner` returns a deterministic hash, not a Sharpe lift.
   Until that's real, every "passed_gate=True" is meaningless.
3. **A second human reviewer is configured on the GitHub repo.** Even
   with `--draft`, `--no-dry-run` lets a Claude run create branches and
   open PRs autonomously. The CODEOWNERS file should require a manual
   approval before merge.
4. **A kill switch exists.** Production deployment must check a
   filesystem flag (`/etc/autopilot/llm_strategy_gen.disabled`) at the
   top of `WeeklyJob.run()` and bail early if present, so an operator
   can stop the loop without touching cron.
5. **Audit logging is wired to a durable store**, not just stdout —
   every proposal text + Sharpe lift + PR URL needs to land in
   `runs/llm_strategy_gen/<utc-date>.jsonl` for post-hoc review.

## Operator runbook

* PRs opened by the loop are titled `WIP: LLM-proposed feature <name>`,
  draft-mode, and contain a CAVEATS section. **Never auto-merge.**
* The proposed feature lands at
  `src/llm_strategy_gen/proposed_features/<slug>.py` with a header
  comment quoting the LLM's expected lift, the stub backtest result,
  and the security caveat.
* The stub test file at
  `tests/prediction_market_scanner/test_llm_proposed_<slug>.py` calls
  `self.skipTest(...)`. **Do not delete the skipTest** until the human
  reviewer has rewritten it as real assertions and verified the
  feature behaves as advertised.
* If a proposal's code looks malicious, do not merge — file a bug
  documenting the prompt-injection or jailbreak vector and tighten
  `_FORBIDDEN_MODULES` / `_FORBIDDEN_CALLS` in `backtest_runner.py`.

## Security caveats

The skeleton's `_validate_proposal_safety` is **AST-only and brittle**:

* Rejects: imports of `os`, `subprocess`, `shutil`, `socket`, `urllib`,
  `requests`, `pickle`, `ctypes`, `sys`, `pathlib`, `builtins`,
  `importlib`, `marshal`, `shelve`, `tempfile`, `io`, `asyncio`,
  `multiprocessing`, `threading`. Calls of `__import__`, `compile`,
  `exec`, `eval`, `open`, `getattr`, `setattr`, `delattr`, `globals`,
  `locals`, `vars`, `breakpoint`, `help`, `input`. Attribute access
  rooted at any forbidden module name.
* **Does not** stop: type confusion, descriptor abuse, format-string
  attacks via `__class__.__mro__`, encoded-import strings (`getattr`
  is forbidden but the proposal could pull a module name from a
  string literal and use `__class__.__bases__[0].__subclasses__()`).
* **Does not** stop: anything once you actually `exec` the code, since
  Python's stdlib has many ways to escape an AST allowlist at runtime.

The skeleton **never `exec`s proposal code** in the default
codepath — `BacktestRunner` uses a deterministic hash stub. Once real
retraining is wired:

* Run the proposal code in an ephemeral Docker container with
  `--network=none`, `--read-only`, `--memory=2g`, `--cpus=2`,
  `--cap-drop=ALL`, and a tmpfs-backed scratch dir.
* Or use RestrictedPython with an explicit builtins allowlist.
* Or run on a VM with a strict snapshot/rollback workflow.

Until then, treat every `passed_gate=True` proposal as a starting
point for human review, not a candidate for auto-merge.

## Wiring the LLM

The skeleton's `OutcomeAnalyst` and `FeatureProposalGenerator`
accept an injected `llm_caller`. Production wiring should be a thin
adapter around `src/llm_judge.py`'s `_request_gemini_json` —
construct the prompt, call Gemini, run the response through
`utils.extract_json_object`, return the dict. The shared timeout and
retry behavior in `llm_judge.py` (4 retries with exponential backoff,
30s default timeout) should be reused as-is.

## What the skeleton does not implement (follow-up PRs)

| Concern                           | Where it lives today                                  | Replacement plan |
|-----------------------------------|-------------------------------------------------------|------------------|
| Real Sharpe lift                  | `BacktestRunner._stub_sharpe_with_proposal`           | Real retraining + walk-forward backtest |
| Sandboxed code execution          | `_validate_proposal_safety` (AST-only)                | Docker isolation or RestrictedPython |
| Worktree creation                 | not implemented                                       | Use the harness `using-git-worktrees` skill |
| Real `gh pr create` invocation    | command string built, never shelled out (skeleton)    | `subprocess.run` adapter; capture URL from stdout |
| Audit log persistence             | logger only                                           | `runs/llm_strategy_gen/<date>.jsonl` |
| Kill switch                       | not implemented                                       | Filesystem-flag check at `WeeklyJob.run()` entry |
| LLM caller                        | `None` by default                                     | Adapter around `llm_judge._request_gemini_json` |
| Proposal de-duplication           | not implemented                                       | Hash-based skip if same proposal already opened |

## Tests

Skeleton tests live under
`tests/prediction_market_scanner/test_llm_strategy_gen_*.py` and run
with the standard pre-flight invocation:

```bash
env PYTHONPATH=src ./.venv/bin/python -m unittest discover \
    -s tests/prediction_market_scanner \
    -p "test_llm_strategy_gen_*.py"
```

Every test mocks the LLM caller. There is **no** test path that calls
out to the live Gemini API, and there should never be one — the cost
+ rate-limit risk would be unacceptable.
