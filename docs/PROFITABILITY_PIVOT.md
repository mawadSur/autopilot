# Profitability Pivot — 2026-05-31

Outcome of a CEO/quant review of the whole project. This is the strategic record;
`docs/CRYPTO_1M_KILL.md` holds the crypto arithmetic; `TODOS.md` holds the task backlog.

## Verdict (one line)

Freeze ~80% of the infra, build the cost-aware backtest + PnL ledger, kill the crypto
1m directional stack on documented evidence, and prove a **model-free Polymarket
arbitrage** edge in shadow before spending a dollar.

## Why the crypto 1m stack was killed (arithmetic, not tuning)

- Models predict a **+10–20bps** move. The repo's own Coinbase adapter charges
  **60bps taker / 40bps maker = ~120bps round-trip** (`coinbase_tradeable.py` FeeModel).
  The predicted edge is **6–12x smaller than the cost to capture it.** No model quality
  fixes an edge-smaller-than-cost inequality.
- The simulator was hiding this: it charged ~8–16bps while the real cost is ~120bps —
  a ~7.5x understatement, with the maker path unwired. Fixed in
  `trading/simulator.py` (`from_coinbase_fees()`), proven in
  `tests/prediction_market_scanner/test_fee_honesty_kill.py` (a perfect +20bps winner
  nets **−$100** at real fees, **+$4** at the old fake fees).
- No `profit_report.json` has ever existed → no backtest ever produced a profitable,
  validated model. The whole trading record is 3 SOL paper trades, all losses.

## Where the money is (2026 landscape, researched)

- **Crypto:** not directional ML. Market-neutral funding-rate/basis arb (~10–30% APY)
  and maker-rebate market-making on low-fee venues (Hyperliquid 0.01% maker). Deferred.
- **Prediction markets (the chosen arena):** model-free **cross-venue / intra-market
  arbitrage** (YES+NO < $1; Polymarket-vs-Kalshi gaps) — doesn't require beating the
  crowd. Plus liquidity-reward farming. Capacity-bounded (~$5–15k depth), which suits a
  solo operator. The existing `risk_engine` already does cost-aware sizing honestly.

## The ladder (where we are → where we go)

```
  rung 0: measure edge net of cost   <-- WE WERE HERE (no tool, no ledger)
  rung 1: one backtested edge          <-- building the tooling now
  rung 2: shadow track record beats market net of fees
  rung 3: tiny live pilot ($50-100)
  rung 4: scale within capacity
```

## What shipped this session (Phase 0 + buildable Phase 1, all SHADOW/no real money)

| Task | What | Status |
|------|------|--------|
| T1 | Wire real Coinbase fees into the simulator | ✅ done |
| T2 | Backtest persists gate verdict to profit_report.json | ✅ done |
| T3 | Shadow PnL ledger (`src/state/pnl_ledger.py`, no-look-ahead guard) | ✅ done |
| T4 | Documented crypto 1m kill (deterministic test + `docs/CRYPTO_1M_KILL.md`) | ✅ done |
| T5 | Freeze + governance norm (this doc) | ✅ done |
| T6 | Honest directional EV helper (warns: fair_prob is a MOCK today) | ✅ done |
| T7 | Intra-market arb detector (`src/arb_detector.py`, shadow, logs to ledger) | ✅ done |
| T8 | Read-only Kalshi market-data client (`src/exchanges/kalshi_market_data.py`) | ✅ done |
| T9 | Run the closed-loop shadow ledger for 2–4 weeks (prove arb survives slippage) | ⏳ ongoing |
| T10 | Real Polymarket CLOB execution (py-clob-client + signing) | 🔒 gated: explicit opt-in |
| T11 | Strict size cap + self-slippage guard | 🔒 gated |
| T12 | $50–100 live pilot | 🔒 gated: real money, Constitution opt-in |

**Hard line:** T10–T12 move real money. Per the Constitution (paper default, live = explicit
deliberate opt-in), they were NOT built autonomously and require a deliberate human go.

## Frozen until rung 1 (one validated edge)

`loss_postmortem` (forensics swarm), `regime_memory`, `llm_strategy_gen`, the multi-asset
adapters beyond Polymarket/Kalshi, the D3 multiprocessing-supervisor refactor, and the
Grafana dashboard. They are rung-3-to-5 tooling for an edge that does not exist yet.

## Governance norm (working rule, not a Constitution amendment)

**No new subsystem without a validated edge it serves.** New strategy code must clear the
cost-aware backtest / shadow ledger before it earns more infrastructure. This concentrates
the scarce resource — operator/agent attention — on finding the first profitable dollar.
(This adds discipline; it does not change the locked Constitution standards, so it needs no
`GETTINGAJET`.)
