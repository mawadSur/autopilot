# TODOS — Autopilot backlog

Backlog of follow-on work, organized by category. Items are P0/P1/P2/P3 by priority. Closed items keep a ✅ DONE marker with the commit hash so the audit trail stays intact across the full project history.

This file is human-mutable; agents read it but should not mutate it without operator authorization. See **How to use this file** at the bottom.

Last updated: 2026-05-08 (post-Wave-1 + Lane D in flight + E4 regime memory partial).

---

## Lane D follow-ups (multi-symbol multiprocess supervisor)

- **D2 Polymarket tradeable + supervisor wiring** — ✅ DONE (`0ac448d` adapter + `live_supervisor.py` wiring; 25 unit tests pass).
  - Brief: `autopilot_lane_d_launch_briefs_2026_05_08.md` ("Sub-agent D2").
  - Adds `PolymarketTradeable` adapter + `SupervisorConfig.tradeables` field so Polymarket binary markets tick alongside crypto symbols in the same loop.
  - Closed 2026-05-30: adapter + `SupervisorConfig.tradeables` + `--polymarket-markets` CLI flag + the symbols-OR-tradeables config validation all landed and pass `tests.prediction_market_scanner.test_polymarket_tradeable` (25 tests OK). The 3 broker follow-ups in the adapter header (`get_balances` / limit+cancel orders / single-market `get_ticker`) are demoted to the P2 broker backlog — they are NOT part of D2 closure. D3 (multiprocessing-per-symbol) remains the open Lane D item.

- **D3 multiprocessing-per-symbol supervisor refactor** — P1, ~3 engineering days, pending D2.
  - Spawn one child process per Tradeable via `mp.get_context("spawn")`. Daily-close leader election via Redis SETNX. Crash respawn with 5s backoff, 3/hr cap.
  - Needs the 60-second multi-process smoke test + Redis-shared shakedown verification.
  - Highest-risk sub-agent of Lane D — must follow with a real-Redis multi-process smoke before push.

- **CoinbaseExchange.get_product() helper** — P2.
  - Today the CoinbaseTradeable adapter probes a few fee-tier paths and falls back to a 60bps taker assumption. Plumb a real `get_product(symbol)` method through to ccxt and surface tick_size + min_size + the live taker tier.
  - Suggested next action: 1-day brief that adds `get_product()` to `src/exchanges/coinbase.py` and wires the adapter to it.

- **Coinbase fee-tier query API** — P2.
  - Same shape as above — Coinbase exposes a /fees endpoint that returns the operator's current 30-day-volume tier. Adapter currently hardcodes the public-tier 60bps taker; a query would let live PnL math reflect lower tiers automatically.

- **Hyperliquid margin-tier estimator** — P2.
  - HyperliquidTradeable.risk_attributes returns a placeholder margin_used_usd today. Pull the live tier from `clearinghouseState` so liquidation_price + margin_used reflect the actual maintenance margin.

---

## CEO roadmap (Phases 3-5)

- **E2 cross-asset alpha lab** — P1, ~12-15d, needs eng-review.
  - Walk-forward backtest harness across crypto + prediction-market datasets so new alpha hypotheses can be A/B tested cheaply.
  - Suggested next action: write the eng-review brief; covers data sourcing, equity-curve aggregation, leakage isolation, and the minimum tests before live.

- **E3 LLM strategy generation loop** — P1, ~10d.
  - Build a feedback loop where Gemini proposes new feature definitions / position sizing tweaks based on the postmortem corpus, then those proposals are fed into E2's harness for backtesting before any human review.
  - Adversarial guard: the LLM must propose with a structured "hypothesis + null + falsifier" frame, not raw "increase XYZ by N%".

- **E4 regime memory** — ✅ DONE (predictor integration `54e3c1d`).
  - `src/regime_memory/{store,lookup,backfill}.py` landed at `45d642c` / `107e96e` / `98d425d`. Integration into `predictor.py` landed at `54e3c1d` (see Closed note below). See `src/regime_memory/INTEGRATION.md`.
  - Closed 2026-05-30: the real public API is `RegimeLookup.resolve_params()` (not `resolve()`), wired into `XGBoostPredictor._resolve_threshold()` — consulted from `predict_full()` and applied only when `_regime_confidence >= 0.5` — with a 232-line test file (`tests/prediction_market_scanner/test_predictor.py`; 51 predictor tests pass, regime-override + low-confidence/broken-store/mid-predict-raise fallbacks all covered). `MultiSymbolXGBoostPredictor.predict_full()` routes to the per-symbol predictor, so it inherits the integration. Remaining open sub-items are doc caveats only (encoder VERSION stamp in the `.npz` + RiskCalculator-side override per `INTEGRATION.md`); the path is inert until `REGIME_STORE_PATH` is set, and `optimal_threshold`/`regime_label` are still v0 synthetic heuristics — do not enable on weak evidence without a real per-symbol threshold sweep.

- **P3 stocks adapter (Alpaca / IBKR)** — P2, depends on Lane D D1 + D2.
  - Lane D ships the Tradeable Protocol; once that's stable, a Stocks adapter is a 2-day brief instead of a 2-week port. Lower priority than crypto + Polymarket because the legal review is heavier.

---

## Forensics swarm improvements

- **A1 Mahalanobis live on real bundles** — P1.
  - Trainer writes `feature_means` + `feature_stds` per fold (commit `906f843`) but the shipped `model_crypto/*/meta.json` files were trained before that change and lack the stats. Until a re-run lands, A1 still skips Mahalanobis on production bundles.
  - Action: run `scripts/retrain_all_crypto_models.sh` (this commit). Validate the new meta.jsons populate the new fields. Re-spot-check A1's verdicts on a few real losses.

- **Verdict-ladder cluster-tier promotion** — ✅ DONE (`c2fb9a4` + `c91282b`).
  - A5 ProcessIntegrity promotes contributing → primary when the error counter ≥ 15.
  - A4 ContextForensics promotes contributing → primary when ≥ 10 in-window news headlines cluster.

- **Twitter/X sentiment** — P2, scaffold landed at `317de79`, needs API keys + wiring.
  - The scaffold is a no-op fetcher with the documented `X_API_BEARER_TOKEN` env var. Once an account is provisioned, add the live fetch path + tests + a single context-forensics integration test that proves the wiring.

- **Position metadata structuring** — ✅ DONE (`af9cdbe`).
  - `Position.partial_fills` / `rejection_reason` / `stop_trigger_price` populated by the supervisor. A2 ExecutionForensics prefers structured fields, falls back to notes scan for legacy snapshots.

- **Breaker snapshot canonical fields** — ✅ DONE (`8d8fd53`).
  - `kill_switch_reason` / `stop_loss_trigger_price` / `breaker_decision` populated. A5 prefers structured, falls back to notes scan.

- **Predictor surface extension** — ✅ DONE (`9518192`).
  - `PredictorResult` now exposes `feature_buffer` + `model_probs`; `predict_full()` is the rich path; `model_meta` accessor unblocks Mahalanobis once retrain lands.

---

## Operational hardening

- **Real-Redis multiprocessing smoke test** — P1, needs real infra.
  - Today's D3 acceptance test uses `fakeredis`. A 60-second smoke against a real Redis (and a real Coinbase REST stub) is the gate before D3 lands on `main`.
  - Suggested next action: ops-time brief — spin up local Redis, run 3 children for 60s, assert 15 ticks recorded + clean shutdown + shakedown counter increments exactly once per symbol.

- **Real training run on 3 crypto datasets** — P1.
  - See `scripts/retrain_all_crypto_models.sh` (this commit). Estimated runtime ~30-60 min on developer machine.
  - Action: operator runs `chmod +x scripts/retrain_all_crypto_models.sh && bash scripts/retrain_all_crypto_models.sh`. Validates AUC + reliability_slope + feature_means/feature_stds populated y/n. Commit the regenerated `model_crypto/*/meta.json` if numbers look sane.

- **Grafana dashboard JSON deploy** — P2.
  - See `observability/grafana_dashboard.json` (this commit). v0 — 8 panels covering tick duration, model confidence, order latency, daily PnL by symbol, shakedown clean-days, kill switch / auto-trip / auto-pause, reconciliation health, open positions.
  - Action: operator imports via `grafana-cli` or the dashboard import UI. Adjust target queries / data source after import (the dashboard ships with `prometheus` as default but is environment-agnostic).

- **Operator dry-run with `--auto-pause-enabled`** — P1, needs real session.
  - Required pre-flight before live promotion. Run the supervisor in paper mode for ≥1 UTC day with `--auto-pause-enabled --auto-pause-loss-pct 0.02 --auto-pause-confidence-window 200 --auto-pause-confidence-sigma 2.0`. Confirm the marker file appears + clears as expected.

- **Audit-trail wart on `5ffbb91`** — P3.
  - The commit message of `5ffbb91` says "Lane E: loss-postmortem integration tests" but the actual diff is W1C Task 1 — auto-pause + confidence-history CLI wiring (3 files, +318 lines on `live_supervisor.py` and 2 test files). Cause: a staging race during W1A's first commit attempt.
  - The actual Lane E integration tests + 5 simulated-loss fixtures landed at `b8ed10b`.
  - Cannot be cleanly rebased per harness restriction (no `git rebase -i`); branch is already pushed. Documented in this round's README "Commit history notes" subsection.

---

## Documentation

- **README mention of retrain script** — ✅ DONE (this commit, "Operational tools" section).
- **README mention of Grafana dashboard import path** — ✅ DONE (this commit, "Operational tools" section).
- **README "Commit history notes" subsection** — ✅ DONE (this commit, points archaeologists at `5ffbb91` vs `b8ed10b`).

---

## Profitability Pivot (CEO review 2026-05-31)

Strategic record: `docs/PROFITABILITY_PIVOT.md`. Crypto kill arithmetic: `docs/CRYPTO_1M_KILL.md`.
Decision: **kill the crypto 1m directional stack** (predicted edge +10–20bps < ~120bps Coinbase
round-trip — unprofitable by arithmetic), **freeze ~80% of the infra**, and **lead with model-free
Polymarket arbitrage**, proven in shadow before any capital.

Phase 0 — foundation (the rung-0 validate/kill tools):
- **T1 honest fees in simulator** — ✅ DONE (`5f1db49`). `from_coinbase_fees()` wires 60/40bps.
- **T2 backtest gate persists verdict** — ✅ DONE (`5f1db49`). `profit_report.json` now carries `gate_passed`/`gate_verdict`.
- **T3 shadow PnL ledger** — ✅ DONE (`94eda08`). `src/state/pnl_ledger.py`, append-only, no-look-ahead guard.
- **T4 documented crypto kill** — ✅ DONE (`5f1db49`). Deterministic test + `docs/CRYPTO_1M_KILL.md`.
- **T5 freeze + governance norm** — ✅ DONE. "No new subsystem without a validated edge it serves." Frozen: `loss_postmortem`, `regime_memory`, `llm_strategy_gen`, extra adapters, D3, Grafana.

Phase 1 — model-free arbitrage edge (SHADOW only, no execution):
- **T6 honest directional EV helper** — ✅ DONE (`42f3ed2`). Warns: `fair_prob` is a MOCK until a real forecaster is validated.
- **T7 intra-market arb detector** — ✅ DONE (`42f3ed2`). `src/arb_detector.py` YES+NO<$1 net of fee/gas, logs shadow candidates to the ledger.
- **T8 read-only Kalshi feed** — ✅ DONE (`94eda08`). `src/exchanges/kalshi_market_data.py` for cross-venue gaps. Base URL/auth need live verification.
- **T9 run the closed-loop shadow ledger 2–4 weeks** — ⏳ P1, in progress. Prove arb survives real depth/latency/slippage before any capital. Feed built (`src/exchanges/polymarket_market_data.py` + `src/arb_shadow_runner.py`, shadow-only). FOLLOW-UP to make live scans productive: `fetcher.fetch_active_markets()` returns `models.Market` without `clobTokenIds`, so the default scan skips every market — wire a Gamma-dict fetch or extend `Market` to retain `clobTokenIds`. Verify CLOB `/book` shape + clobTokenIds ordering live (the official `polymarket-cli clob book`/`clob midpoint` corroborate the endpoints).

Candidate (UNDER REVIEW — added from external research 2026-05-31, decide if worth keeping):
- **T13 smart-money wallet follower (shadow)** — P2 CANDIDATE. A second model-free edge, distinct from arbitrage: don't predict events, predict *which traders win*. Method (sourced from a Polymarket-bot writeup): rank wallets on public on-chain trade history (filter ≥100 trades AND win_rate > 70%, sort by total profit → top ~50 target list), then fire a **whale-convergence** signal when ≥3 target wallets simultaneously hold the same side of a market. Shadow-log candidates to the PnL ledger and score Brier/realized-PnL net of the 2% fee BEFORE any capital — same discipline as the arb path.
  - Reusable repos named in the writeup: `github.com/warproxxx/poly_data` (86M public Polymarket trades — the wallet-ranking data source), `github.com/Polymarket/polymarket-cli` (official Rust CLI; also validates our T9 `clob book`/`clob midpoint` endpoints), `github.com/Polymarket/agents` (official Python agent framework — useful for T10 execution), `github.com/dylanpersonguy/Polymarket-Trading-Bot` (exit/strategy reference).
  - Honest caveats (why "review if worth keeping"): the source is a copy-trading promo (kreo.app) with **unaudited** results ($200→$14.3k/74% win/Sharpe 2.47 in 27d — treat as a lead, not evidence). Real risks: survivorship bias (wallets picked on PAST profit may not persist), **late-copy** (price moves before you mirror, eroding edge), and category drift (the writeup itself found you must copy crypto-specialists only for crypto, never average across categories). Its own hard-won lessons worth stealing regardless: $50K+ market-depth minimum, volume-exit ~73c (never hold to settlement), 4–48h resolution window.
  - Decision gate: build it shadow-only, run alongside the arb shadow loop, and KEEP only if following the top wallets beats the market net of fees in our own ledger over a few weeks.

Phase 2 — execution (🔒 GATED: real money, explicit deliberate opt-in per Constitution; NOT built autonomously):
- **T10 real Polymarket CLOB execution** — 🔒 P2. `polymarket_tradeable.py` is a stub; needs `py-clob-client` + signed orders + `get_balances` + settlement reader. Only after T9 shows positive net-of-fee edge.
- **T11 strict size cap + self-slippage guard** — 🔒 P2. Reuse `risk_engine` Kelly/fee haircut; cap to ~$5–15k depth.
- **T12 $50–100 live pilot** — 🔒 P2. Smallest real-money confirmation that fills match the shadow ledger. Scale only if they do.

Explicitly deferred: LLM-forecasting edge (revisit only on thin/news-latency markets after arb pays — build the shadow loop so it's measurable); crypto funding-rate/basis arb on Hyperliquid (Phase 3 option).

### Make the whale-follow edge real (CEO review 2026-06-01)

CEO review verdict: the whale-convergence signal is **−EV at resolution** — `whale_leaderboard_ledger.jsonl` shows **165 settled / 13.3% win / −$6,232 realized** (hold-to-resolution). The only positive number (`whale_optimized` +$299 / 33 settled) comes ENTIRELY from the SL/TP exit overlay AND assumes you can sell at the current mark — an unmodeled, likely-false liquidity assumption on thin books. Operator chose to KEEP whale-follow but make the edge legitimate. The work below is the honest test; W4's gate decides keep-or-kill. **All SHADOW; no live capital until the gate passes net of realistic cost.**

- **W1 book-aware EXIT pricing** — ✅ DONE (`28d3b3e`). `polymarket_market_data.vwap_sell_price(bids, units)` walks the bid book best-first (partial-fill aware); the CLOB outcome token id (`asset`) is threaded through both convergence builders into the notes; `make_whale_fill_fn` parses it + walks that token's book; `exit_rules.apply_exit_rules(fill_fn=...)` DECIDES on the mark but BOOKS realized P/L + recorded exit_price at the book fill (falls back to mark for pre-W1 records / empty book). 18 tests. LIVE-VERIFIED: CLOB `/book?token_id=` returns `{bids,asks}` (prices [0,1]); on a deep market a 171-unit sell filled at best-bid with 0 bps slippage — on thin sports markets the VWAP will sit well below the mark, which is the point. LIMITATION: the unfillable tail on a too-thin book isn't modeled (optimistic there). Loop restarted with W1 active (new records carry `asset=`).
- **W2 book-aware ENTRY + per-market depth cap** — ✅ DONE (`W2 commit`). `polymarket_market_data.vwap_buy_price(asks, units)` walks the ASK side cheapest-first; book-aware entry prices off the ask VWAP for size/best_ask units (`--no-book-entry` to disable, default on), falling back to the /trades mark on any error; per-market DEPTH CAP (`--max-book-frac 0.05`, `--depth-band 0.05`) skips a market whose near-mid ask depth can't absorb our size at ≤5% (marked 'thin_book', FAILS OPEN on a read error). Adversarially verified (holds=true). 13 tests.
- **W3 enforced exposure cap + daily-loss kill switch** — ✅ DONE (`W3 commit`, built in an isolated worktree in parallel with W2, hand-merged). EXPOSURE CAP `--max-exposure 2000` refuses an entry that would push Σ open notional past the cap. DAILY-LOSS KILL SWITCH `--daily-loss-limit 200` refuses ALL new entries for the scan once today's (UTC) realized loss across settled records hits the limit (prints a loud trip line). Additive risk gates; function-default-OFF / CLI-default-ON. 8 tests. (Note: kill switch resets at UTC midnight automatically; there is no manual `--resume` — re-tripping each scan is the intended behavior.)
- **W4 fresh re-run + KEEP/KILL gate** — P1. New ledger, W1+W2+W3 active, run 2–4 weeks. KEEP only if realized clears net of the 2% fee AND modeled slippage. If it doesn't, treat −$6,232 as the verdict and kill whale-follow (fold back to the arb edge). This is T13's original decision gate, finally with honest fills.

Note: **T9 (the chosen market-neutral arbitrage edge) was never actually run** — `arb_detector.py`/`arb_shadow_runner.py` exist but no arb ledger is on disk; blocked by `fetch_active_markets()` returning markets without `clobTokenIds`. It remains the bounded-downside fallback if W4 kills whale-follow.

---

## How to use this file

- **Humans mutate; agents read.** Agents may reference this file for context but should not edit without operator authorization (call it out in the brief if a backlog item is being closed).
- **Closed items stay** with a ✅ DONE marker + commit hash. We don't delete them — the audit trail is more useful than a clean file.
- **Priorities** are P0 (drop everything) / P1 (next sprint) / P2 (this quarter) / P3 (eventually). Re-rank during the weekly retro; don't bikeshed mid-week.
- **Each entry** carries a "what / why / suggested next action" body. If you can't write the next action concisely, the item probably needs a brief, not a TODO.
- **When in doubt, link the brief.** Memory files under `~/.claude/projects/-Users-mawad-Desktop-autopilot/memory/` are the source of truth for round-by-round briefs; this file is the rolling backlog index.
