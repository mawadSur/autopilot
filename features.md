# Features Status

Tracks the local-first prediction-market research + paper-trading bot spec against what currently exists in this repo.

**Legend:** ✅ done · 🟡 partial · ❌ missing · ⚠️ deviation from spec

**Important:** This repo is `autopilot/`, not `prediction_market_bot/`. The directory layout is flat under `src/` rather than `app/agents/ + app/services/ + app/models/`. The functional parts of the spec largely exist; the **toolchain, packaging, and storage layer** are the main gaps.

---

## Constraints (research / paper-trading only)

| Item | Status | Notes |
|---|---|---|
| Research, backtest, paper-trading only | ✅ | No real-money execution path is wired in the prediction-market stack. |
| No real-money trades | ✅ | `src/orchestrator.py` writes `trade_execution_*.json` decisions; no broker connector exists for prediction markets. The legacy crypto stack (`src/live_trader.py`) is separate and not exercised by the prediction-market pipeline. |
| No guaranteed-profit claims | ✅ | README and skills frame everything as research/forecasting. |
| Execution disabled by default | 🟡 | True for prediction-market side; legacy `src/live_trader.py` exists but is not invoked by the new pipeline. Needs explicit README disclaimer (see task 4). |
| All outputs auditable + logged | 🟡 | `output/scan_*.json`, `trade_execution_*.json`, `performance_audit.json` capture decisions. No central structured log file. SQLite layer would tighten this. |

---

## Architecture (7 components)

### 1. Market Scanner Agent — ✅ done

| Spec | Status | Where |
|---|---|---|
| Input: list of active markets | ✅ | `src/fetcher.py` (Polymarket Gamma API) |
| Filter by liquidity / volume / spread / time-to-resolution | ✅ | `src/analyzer.py`, `src/ranker.py` |
| Flag abnormal price moves, volume spikes, wide spreads | 🟡 | `analyzer.py` does liquidity + spread filtering; explicit *anomaly* flagging is partial. |
| Output ranked markets with `research_priority` score | ✅ | `src/ranker.py` produces `priority_score`; CLI prints ranked table. |

### 2. Research Agents — 🟡 partial

| Spec | Status | Where |
|---|---|---|
| Twitter / X research adapter | ❌ | Not implemented. No `twitter_research_agent/` module. |
| Reddit research adapter | ✅ | `src/reddit_research_agent/` (PRAW + Gemini) |
| RSS research adapter | ✅ | `src/news_research_agent/` (Google News RSS + Gemini) |
| Mock data by default | ❌ | All agents hit real APIs. No mock-mode flag. |
| Output: bullish / bearish thesis, evidence quality, misinformation risk, sentiment, sources, summary | ❌ | **Material schema gaps.** Reddit: `pro_argument`/`anti_argument` (renamed from bullish/bearish), `evidence_quality_score` ✅; **missing** `misinformation_risk_score`, `sentiment_score`, `key_sources`, `summary`. News: only `summary` and `source_quality_score` (renamed); **missing** `bullish_thesis`, `bearish_thesis`, `misinformation_risk_score`, `sentiment_score`, `key_sources`. |
| No illegal scraping / rule bypassing | ✅ | Uses official APIs only (PRAW, Google News RSS). |

### 3. Synthesis Agent — 🟡 partial (verdict semantics differ from spec)

| Spec | Status | Where |
|---|---|---|
| Compare research narrative to market-implied probability | ✅ | `src/synthesis_agent/analyzer.py` |
| Verdict: stale / efficient / overreactive / unclear | ❌ | Code has `verdict: Literal["no edge", "possible edge", "strong research edge"]` — a *trade-conviction* categorical. Spec wants a *market-efficiency* categorical. Different semantic axis entirely. |
| Structured research summary | ✅ | `SynthesisReport` pydantic model. |

### 4. Probability Calibration Agent — 🟡 partial (XGBoost is a MOCK)

| Spec | Status | Where |
|---|---|---|
| XGBoost baseline | ❌ | `src/calibration_agent/ml_service.py:21` — `get_xgboost_probability` is a TODO mock that returns `market.implied_prob` plus `random.uniform(...)` jitter. Not a real model. **Calibration is currently garbage-in/garbage-out.** |
| LLM-style qualitative adjustment | ✅ | `src/calibration_agent/analyzer.py` (Gemini-backed) |
| Returns: xgboost_prob, llm_adjustment, calibrated_true_prob, confidence_score, edge_vs_market, action | ✅ | `CalibrationReport` carries all six fields; `llm_adjustment` is named `llm_adjustment_pct_points` in code. |
| Action: pass / monitor / paper_trade_candidate | 🟡 | Present, but the value is `"paper-trade candidate"` (with hyphen + space) rather than spec's `paper_trade_candidate`. Cosmetic mismatch. |

### 5. Risk Management Agent — ✅ done

| Spec | Status | Where |
|---|---|---|
| Simulate position sizing only | ✅ | `src/risk_management_agent/risk_engine.py` (Kelly + penalties) |
| Bankroll, confidence, liquidity, spread, correlation, downside | ✅ | `risk_engine` applies all six. |
| Block trades that are too risky | ✅ | `RiskAssessment.allow_trade` |
| Returns: allow_trade, simulated_position_size_pct, max_loss, expected_value, risk_reasons, final_recommendation | ✅ | `src/risk_management_agent/models.py: RiskAssessment` covers all six fields. |

### 6. Paper Trade Tracker — 🟡 partial (volume blocker RESOLVED)

| Spec | Status | Where |
|---|---|---|
| Track simulated entries, exits, PnL, thesis, confidence, timestamps | 🟡 | `trade_execution_*.json` records the *decision*; explicit entry/exit price + PnL fields are NOT carried (orchestrator's `event_payload` lacks them — see B3 work). `mark_trade_settled.py` flips status + final_outcome but doesn't compute PnL. |
| No real trading API connection | ✅ | None wired. |
| TODO comment for future execution | ❌ | No explicit TODO marker in code. Worth adding to make the disabled-by-default contract obvious. |
| Sufficient labeled volume to train calibration | ✅ **RESOLVED** | Previously a hard blocker (a single orchestrator run produces ~1 trade decision; XGBoost training needs hundreds). Two new writers unblock this: `src/calibration_agent/shadow_capture.py` polls every active market on each interval (`source="shadow"`, full fidelity) and `src/calibration_agent/backfill_from_polymarket.py` one-shots all resolved markets (`source="backfill"`, degraded fidelity, smoke-test only). See "Data Sources" below. |

### 7. Post-Mortem Agents — 🟡 partial (5 expected, 2 done as Python, 3 covered by skills/Iterative Improver)

| Spec agent | Status | Where |
|---|---|---|
| Outcome Review Agent | ✅ | `src/outcome_review_agent/` — 4-quadrant matrix, runs in batch via `PerformanceTracker` |
| Data Quality Failure Agent | ✅ | `src/data_quality_auditor/` — 7 failure modes, batch + 3 modular methods |
| Model Error Agent | 🟡 | `src/iterative_improver/` covers this conditionally (only fires on "Good Failure" trades), produces 3 feature recommendations. Spec wants it to run on every settled trade. |
| Execution Review Agent | 🟡 | Lives as **interactive skill** (`.claude/skills/execution-reviewer/`), not a Python agent. Reads `src/simulator.py` directly. |
| Learning Loop Agent | 🟡 | Lives as **interactive skill** (`.claude/skills/post-mortem-auditor/`) — produces a Changelog Entry + suggested code change. |
| 4-quadrant classification (good/bad process × good/bad outcome) | ✅ | `OutcomeReview.matrix_classification` enforces it as a `Literal` |
| Save lessons to SQLite | ❌ | All audits are JSON-stored (`performance_audit.json`); no SQLite layer. |

### 8. Orchestrator — ✅ done

| Spec | Status | Where |
|---|---|---|
| Master pipeline controller | ✅ | `src/orchestrator.py` (batch) + `.claude/skills/pipeline-orchestrator/` (interactive) |
| Coordinates all 7 stages | ✅ | Stages 1-6 in batch, stage 7 via `PerformanceTracker` consuming the JSON output. |

---

## API Endpoints (FastAPI)

Spec wants these endpoints. The legacy `src/main.py` FastAPI app is for crypto trading, not the prediction-market scanner. **None of the spec endpoints exist for the prediction-market pipeline today.**

| Spec endpoint | Status | Notes |
|---|---|---|
| `GET /health` | 🟡 | Exists in legacy `src/main.py` and `src/dashboard_server.py` (`/api/health`). Not for the prediction-market scanner. |
| `POST /scan` | ❌ | Scanner is CLI-only (`main.py --top 20`). |
| `POST /research` | ❌ | Research is invoked via `src/orchestrator.py` CLI. |
| `POST /predict` | 🟡 | Exists in legacy crypto API (calls SageMaker). Not wired to `calibration_agent`. |
| `POST /risk` | ❌ | `risk_management_agent` is library-only. |
| `POST /paper-trade` | 🟡 | Legacy `/paper-trade/start` triggers a *crypto* paper trade. No prediction-market paper trade endpoint. |
| `POST /settle` | ❌ | `mark_trade_settled.py` is CLI-only. |
| `GET /trades` | ❌ | Trade logs are filesystem JSON. |
| `GET /postmortems` | ❌ | `performance_audit.json` is filesystem JSON. |

**Required:** A new FastAPI app for the prediction-market pipeline (separate from the legacy crypto API) wired to the existing agents.

---

## Data Sources

Three writers produce `trade_execution_<id>.json` (or shadow-prefixed) logs that share the canonical schema. They differ on **how features were captured**, which is the only thing the calibration training pipeline cares about.

| Source | Generator | Fidelity | Volume | Use case |
|---|---|---|---|---|
| `orchestrator` | `src/orchestrator.py` (real decisions) | high | low (per-trade) | production |
| `shadow` | `src/calibration_agent/shadow_capture.py` | high | high (every active market per scan) | bulk dataset accumulation |
| `backfill` | `src/calibration_agent/backfill_from_polymarket.py` | **DEGRADED** (post-resolution feature snapshots) | one-shot, hundreds | smoke-testing the training pipeline only |

**Filtering rule (`src/calibration_agent/build_dataset.py`):** by default `assemble_dataset` keeps only `FULL_FIDELITY_SOURCES = ("orchestrator", "shadow")`. Backfill rows are skipped and counted under the `backfill_excluded` skip-reason bucket. Pass `include_backfill=True` (Python) or `--include-backfill` (CLI) to opt them in. Pre-Pass-1 trade logs (no `source` field) are treated as `"orchestrator"` for back-compat and a single aggregated WARNING is emitted per assemble call.

**CLI examples:**

```bash
# Shadow capture: one-shot (writes one JSON per active market into ./shadow_logs)
./.venv/bin/python src/calibration_agent/shadow_capture.py \
    --output-dir ./shadow_logs --once

# Shadow capture: long-running daemon (sleep 1h between scans)
./.venv/bin/python src/calibration_agent/shadow_capture.py \
    --output-dir ./shadow_logs --interval-seconds 3600

# Backfill (one-shot; uses degraded post-resolution features)
./.venv/bin/python src/calibration_agent/backfill_from_polymarket.py \
    --output-dir ./backfill --limit 200 --min-volume 5000 --days-back 90

# Build dataset (default: orchestrator + shadow only)
./.venv/bin/python src/calibration_agent/build_dataset.py ./trade_logs \
    --output ./datasets/calibration.parquet

# Build dataset (smoke-test only — includes degraded backfill rows)
./.venv/bin/python src/calibration_agent/build_dataset.py ./trade_logs \
    --output ./datasets/calibration_smoke.parquet --include-backfill
```

---

## Storage

| Spec | Status | Notes |
|---|---|---|
| SQLite for local storage | ❌ | Currently filesystem JSON: `output/scan_*.json`, `trade_execution_*.json`, `performance_audit.json`. Migrating to SQLite would centralize audit + enable `GET /trades` / `GET /postmortems` cleanly. |

---

## Schemas — ✅ done

All agents already use Pydantic. Schemas live alongside agents:
- `src/calibration_agent/models.py: CalibrationReport`
- `src/risk_management_agent/models.py: RiskAssessment, RiskMetrics`
- `src/synthesis_agent/models.py: SynthesisReport`
- `src/news_research_agent/models.py: NewsResearchReport`
- `src/reddit_research_agent/models.py: RedditResearchReport`
- `src/outcome_review_agent/models.py: OutcomeReview`
- `src/data_quality_auditor/models.py: DataQualityAudit, FocusedAuditFinding, FailureModeFinding`
- `src/iterative_improver/models.py: RetrainingRecommendation, FeatureRecommendation`

---

## Tests

| Spec | Status | Notes |
|---|---|---|
| pytest coverage for market filtering | ✅ | `tests/prediction_market_scanner/test_main.py`, `test_market.py`, `test_ranker.py` (uses `unittest`, runs the same way) |
| Probability calibration tests | ✅ | `test_calibration_agent_*.py` |
| Risk blocking tests | ✅ | `test_risk_management_agent_*.py` |
| Paper-trade creation tests | 🟡 | Trade execution log writing is exercised in `test_orchestrator.py`; PnL tracking isn't covered (because PnL isn't tracked yet). |
| Post-mortem classification tests | ✅ | `test_outcome_review_agent.py`, `test_data_quality_auditor.py`, `test_iterative_improver.py`, `test_outcome_review_e2e.py`, `test_analytics_dashboard.py`, `test_mark_trade_settled.py` |

⚠️ **Deviation:** Spec asks for `pytest`; the prediction-market suite uses `unittest`. Both work; tests are runnable via `python -m unittest discover`. No conversion needed for correctness — only convention.

**Total:** 156 tests pass across the prediction-market suite.

---

## Toolchain

| Spec | Status | Notes |
|---|---|---|
| Python 3.11+ | 🟡 | README says 3.10 is the safest assumption; Dockerfile pins 3.10. Tests pass on 3.11. Worth bumping to 3.11 in the README + Dockerfile. |
| `pyproject.toml` | ❌ | None. Project uses `requirements.txt` only. |
| `.env.example` | ❌ | README documents env vars; no `.env.example` template. |
| `ruff` for linting | ❌ | No linter configured. CLAUDE.md notes "no formatter, linter, pre-commit hook, or CI workflow." |
| `pytest` | 🟡 | Installed (`tests/test_core.py` is pytest); the prediction-market suite uses `unittest`. |
| Pydantic for schemas | ✅ | Used everywhere. |
| pandas / numpy | ✅ | Used in legacy stack and ranker. |
| scikit-learn / xgboost | ✅ | XGBoost in `calibration_agent/ml_baseline.py`. |
| LLM provider abstraction | ❌ | Hard-coded to Gemini for prediction-market agents; OpenAI for `social-narrative-agent/`. No abstraction layer. |

---

## Documentation

| Spec | Status | Notes |
|---|---|---|
| README setup instructions | ✅ | `Readme.md` covers venv + pip install. |
| README test instructions | ✅ | Covered. |
| README how-to-start-API | 🟡 | Covers legacy `src/main.py` only. No prediction-market API exists yet. |
| README example API calls | ❌ | Currently shows CLI examples only. |
| Paper-trading-only disclaimer | ❌ | Not present in README. **Required.** |
| System architecture diagram (text) | 🟡 | `Readme.md` describes components in prose; no explicit diagram. |
| Safety notes around prediction markets + risk | ❌ | Not present. **Required.** |

---

## Summary

### What's done
- All 7 architectural components exist as Python modules.
- All agents use Pydantic for schemas.
- 156 tests cover the prediction-market suite.
- Outcome review (4-quadrant matrix) + Data Quality Auditor + Iterative Improver are wired through `PerformanceTracker`.
- Five interactive Claude Code skills cover the operator-driven path: `pipeline-orchestrator`, `narrative-calibrator`, `risk-gatekeeper`, `execution-reviewer`, `post-mortem-auditor`.

### What's partial — needs filling in
- **Twitter research agent** is missing entirely.
- **Mock-data mode** for research agents is not implemented.
- **PnL tracking** in paper trades — orchestrator writes decisions but doesn't carry entry/exit prices or P&L.
- **Anomaly flagging** in scanner (price spikes, volume spikes) needs explicit fields.
- **Synthesis verdict** isn't a strict 4-way Literal (stale/efficient/overreactive/unclear).
- **Calibration agent** doesn't surface `xgboost_prob` and `llm_adjustment` separately.
- **Model Error Agent** (Iterative Improver) is conditional, not unconditional.
- **Execution Review** + **Learning Loop** live as Claude Code skills, not Python agents.

### What's missing
- **FastAPI app for the prediction-market pipeline** — none of `/scan`, `/research`, `/risk`, `/paper-trade`, `/settle`, `/trades`, `/postmortems` exist.
- **SQLite storage layer** — everything is filesystem JSON.
- **`pyproject.toml`** — no modern Python packaging.
- **`.env.example`** — env vars are documented but no template.
- **`ruff`** — no linter configured.
- **LLM provider abstraction** — Gemini / OpenAI are hard-coded per agent.
- **Paper-trading-only disclaimer + safety notes** in README.
- **Architecture diagram** in README.

### Findings from parallel code review (2026-04-25)

Four review agents inspected schemas, tests, training infra, and production-readiness in parallel. Updates above incorporate their findings. Additional flags worth surfacing:

- **Severity 1 / blocker:** XGBoost calibration baseline is mock (`src/calibration_agent/ml_service.py:21`). All current "calibrated" probabilities are jittered market-implied probs.
- **Severity 2:** `src/orchestrator.py:593-595` swallows per-market exceptions silently during multi-market runs. Failed markets disappear from the completed list with no log.
- **Severity 3:** `features_window` and `model_meta` slots are `None` by default; consumer agents handle this with sentinel fallbacks, but predicate functions in `conditional_review_agents` should `.get()` defensively (not enforced).
- **Test coverage**: 156 tests pass, schema-breaking changes WILL be caught (Pydantic `extra="forbid"` everywhere), but cross-stage integration is weakly tested. Specifically untested: kill-switch boundaries (0.01/0.99), XGBoost↔calibration constraint (`calibrated_true_prob >= xgboost_prob` for non-negative LLM adjustment), end-to-end with real-shaped payloads.
- **Legacy ETH model**: artifacts are 1 day old (last train: 2026-04-24 15:48); ready to retrain on demand.

### Recommended next steps (in priority order)

1. **README disclaimer + safety notes** (15 min). Required by spec; trivially small change.
2. **`.env.example`** (10 min). Mirror what's in `Readme.md`.
3. **`pyproject.toml`** (30 min). Add minimal config so tooling (ruff, pytest, mypy) has something to read.
4. **`ruff` config** (15 min). Hook into pyproject.
5. **FastAPI app for the prediction-market pipeline** (~1 day). Wire `/scan`, `/research`, `/predict`, `/risk`, `/paper-trade`, `/settle`, `/trades`, `/postmortems` to the existing agents.
6. **SQLite storage layer** (~1-2 days). Migrate `trade_execution_*.json` + `performance_audit.json` to SQLite tables; expose via the API endpoints above.
7. **PnL tracking** + entry/exit price slots in `event_payload` (~half day).
8. **Twitter research agent** (~half day, depends on API access).
9. **Mock-data mode** for research agents (~half day).
10. **LLM provider abstraction** (~1 day, larger refactor).
