# Strategy & Profitability Research — 2026-06-04

> Method: hybrid research run (103 agents). 6 read-only internal audits of every
> strategy this repo has tried (checked against the actual code + ledgers) run
> concurrently with 6 web-research angles. 69 external claims were extracted; the
> 30 most decision-relevant went through 3-vote adversarial refutation — **9 survived,
> 21 were refuted** (listed as myths in §9). Scope: crypto + prediction markets,
> research-only (no capital committed). Honors the Constitution: paper-first, honest
> reporting, edge must beat cost.

## 1. TL;DR / The one recommendation

**Run a small, delta-neutral BTC/ETH funding/basis carry on a low-fee perp venue as a YIELD play — not as alpha — and treat it as the crypto sibling of the already-live stablecoin yield runner. But gate it behind a paper run that proves net edge > cost, because our own shadow test of this exact idea already lost money (-0.17% on $8,000).**

This is a deliberately unexciting recommendation, and that is the point. Every directional/edge strategy this project has tried died on the same wall: **net edge below total cost.** Crypto 1m XGBoost (10-20 bps edge vs 120 bps round-trip), whale-follow (-$5,516 / 22% win across 347 trades), funding carry (-$13.64), intra-market arb (0 executable arbs in 400 markets). The verified external research confirms there is no retail-accessible high-Sharpe edge hiding here: maker-rebate market-making is a latency game retail loses (confidence 0.82, [2]); cross-venue prediction-market arb carries settlement-divergence total-loss risk (confidence 0.85, [3]); Kalshi/Polymarket reward pools are explicitly tiny ($10-$1,000/market/day, [4]).

So the honest answer is: **there is no high-EV, low-risk directional path for a solo operator in our universe right now.** The least-bad path that has any positive expected value net of cost is a *yield* posture — park excess stablecoins (already live, ~4.5%) and, only if it clears a paper-validated cost gate, add a tiny BTC/ETH carry sleeve expecting **~5-10% net APY** ([1], BIS WP 1087). Do not chase the headline 20-40% carry numbers — those are gross, crash-predictive, and concentrated in illiquid names retail cannot hedge ([1]).

**If you insist on a directional edge: do not deploy capital until a paper backtest shows positive expectancy AFTER fees+slippage+spread.** The win-rate/payoff/sizing methodology for that case is fully specified in §§5-7, but it is conditional on an edge we do not currently have.

---

## 2. What we currently have

| Strategy | Status | Realized P&L | Win rate | Verdict |
|---|---|---|---|---|
| Crypto 1m directional XGBoost (Coinbase) | **Killed** | n/a (never deployed) | n/a | Edge 10-20 bps << 120 bps round-trip; mathematically unprofitable. Kill is sound. |
| Polymarket whale-follow (4 variants, 347 trades) | **Killed** | **-$5,516** | **22%** | Copy lag + stale REST `/positions` (~2s). Only "Honest" variant marginal (+$631, 60% WR, N=35, p=0.23 — luck). |
| Crypto funding-rate carry (8 positions, $8k) | **Killed** | **-$13.64 (-0.17%)** | n/a | Snapshot carry is a ~10x mirage; rates decay 85-90% within 7h. |
| Polymarket intra-market YES+NO arb | **Shelved** | n/a (0 captured) | n/a | Edges real but transient (<30s); 40%+ CLOB timeouts; unexecutable at retail latency. |
| Stablecoin yield accrual + peg monitor | **Alive** | +$2.16 (14h, $30k shadow) | n/a | Mechanically sound; 4.5% APY ~45-75 bps **below** 1y T-Bills (~4.95%). Floor, not edge. |
| Reusable risk/sizing/backtest infra | **Alive** | n/a | n/a | Production-ready: fee-honest Kelly, 5 circuit breakers, no-look-ahead ledger, cost-aware backtest. |

**The through-line (the lesson we have paid for four times):** *net edge must exceed total cost (fees + slippage + spread + borrow/basis).* This single inequality killed every strategy above. It **rules out**: (a) any short-horizon directional crypto strategy on Coinbase-class fees (120 bps round-trip eats any 1m-scale move); (b) any prediction-market strategy that depends on sub-second execution against bots (whale-follow, intra-market arb); (c) chasing fat headline carries in illiquid alt names (decay + borrow + basis drift). It **does not** rule out: low-frequency, market-neutral yield where the edge is structural and the cost is amortized over a long hold.

---

## 3. What can be done better (concrete, per viable path)

**A. Stablecoin yield (alive):** The runner is sound but the *rate* is wrong. At 4.5% it loses to T-Bills. Concretely:
- Add an **APY-compression alert**: redeploy trigger when the live rate drops below 4.0% (lever already scoped in audit).
- Diversify custody: split notional CeFi (Kraken 4.5%) vs DeFi (Aave/Morpho 4-8%, higher smart-contract risk) so no single platform failure wipes the sleeve.
- Only justified as a *parking sink for idle operational stablecoins*, OR if T-Bills fall below 4.5%. It is not worth capital expansion at current rates.

**B. BTC/ETH funding/basis carry (the recommended add):** The prior carry run died because it ranked by *snapshot* rate and chased illiquid alts (HYPE/ZRO/APEX) where decay is 85-90% and the hedge leg is unlistable. Fix the design:
- **Restrict universe to BTC and ETH only.** Tier-1 funding is lower (0.5-2%/yr typical) but decays slower and has a clean, liquid spot/perp hedge ([1]).
- **Rank by *realized persistence*, not snapshot.** Backtest the actual paid funding history; require the rate to have held for N hours, not a one-print spike.
- **Decay-triggered early exit:** close when realized rate drops below the cost hurdle.
- **Low-fee venue** (Hyperliquid base 0.015% maker / 0.045% taker, [2]) — but pay the *base maker fee as a cost*; the negative rebate is unreachable at our size ([2]).

**C. Directional crypto (killed) — only revivable if:** rescale labels to a **multi-hour/daily** horizon whose gross edge comfortably exceeds 120 bps (need >150 bps gross to have margin), AND/OR move to a venue where round-trip cost is ~20-40 bps. Both require full retrain + replacing the simulator's optimistic same-bar maker-fill assumption (`simulator.py:469-479`) with a realistic resting-limit model. This is a *new strategy*, not a tune.

**D. Prediction-market arb (shelved) — only revivable** with WebSocket CLOB (sub-second) + full-depth VWAP slippage modeling + a legging guard ($5k+ depth both legs). Even then, settlement-divergence risk on any cross-venue hedge is a **total-loss tail** ([3]). Not worth it at current latency.

---

## 4. The most profitable path (recommended): small BTC/ETH delta-neutral carry as yield

**Why it wins on EV net of cost (conditionally).** The mechanism is structural, not predictive: short perp + long spot (when funding positive) earns the funding spread with no directional exposure. Unlike a 1m directional bet, the edge is *not* a tiny move that fees swallow — it accrues continuously over a multi-day hold, so a single 120-bps-class round-trip is amortized across many funding intervals. BIS WP 1087 finds crypto carry "averages above 10% annually" gross ([1]); the realized, hedgeable, after-cost number for a retail operator on liquid names is **~5-10% net APY** ([1]). That is positive expectancy net of cost — the bar nothing else here clears.

**Why it wins on survivability.** It is delta-neutral: a sharp BTC up-move that would liquidate a naked short is offset by the long spot leg. There is no directional drawdown. The realized variance is dominated by basis drift and funding flips, both bounded and monitorable, not by price direction.

**Capacity.** On BTC/ETH the order books are deep enough that a small operator's size ($5k-$50k) has negligible market impact — capacity is **not** the binding constraint at our scale (the binding constraint is per-unit net edge, since the deep liquidity that gives capacity is what compresses the spread). This is the inverse of the stablecoin yield runner, which *is* capacity-capped by per-user Earn limits.

**The main risk and how to bound it.** The dominant risk is **exchange/liquidation/de-peg blowup** (custody failure, the short-perp leg getting liquidated in a violent up-move before the hedge rebalances), NOT funding turning negative ([1]). Bound it by: (1) margin the short leg conservatively (low leverage, large maintenance buffer — see §7); (2) rotate to flat the moment funding turns negative; (3) cap the sleeve at a small fraction of total capital; (4) use the existing per-symbol notional circuit breaker and kill-switch file. **Marginal honesty:** the net edge of this sleeve *over the already-live ~4.5% stablecoin yield is thin* once execution and exchange risk are priced ([1]). It is worth running only as a paper-validated diversifier, not as a primary money-maker.

---

## 5. Win rate & expectancy — the real math

**High win rate is not the goal. Positive expectancy net of cost is.** Win rate alone carries zero information about win/loss *magnitude* and cannot determine the sign of profit ([5], confidence 0.97). The canonical identity:

> **Expectancy = (WinRate × AvgWin) − (LossRate × AvgLoss)**, where LossRate = 1 − WinRate ([5]).

A strategy is profitable *before costs* iff Expectancy > 0; costs (fees + slippage + spread) subtract directly, so the real test is Expectancy − Costs > 0.

**Breakeven win rate is a deterministic function of reward-to-risk R (= AvgWin/AvgLoss):**

> **WR_breakeven = 1 / (1 + R) = risk / (risk + reward)** ([6], confidence 0.95).

| R (reward:risk) | Breakeven WR (before cost) |
|---|---|
| 0.5 | 66.7% |
| 1.0 (1:1) | 50.0% |
| 2.0 (1:2) | 33.3% |
| 3.0 (1:3) | 25.0% |
| 4.0 (1:4) | 20.0% |
| 5.0 (1:5) | 16.7% |

**Worked example showing why high WR is a trap:** A 40% win rate at 2.5:1 (win $250, loss $100) yields **+$40/trade** = 0.40 × $250 − 0.60 × $100 ([7], confidence 0.92). At 3:1 (win $300, loss $100), 40% WR yields **+$60/trade** — because breakeven at R=3 is only 25%, a 40% hit rate has a *large* margin. Meanwhile an 80% WR system with win $50 / loss $300 nets **−$20/trade**. The high win rate loses money. *(Note: the conceptual thesis is verified; the cited source's exact figures were partly misattributed — the principle holds, the specific quotes do not all appear in the source.)*

**Target combo IF directional:** Aim **low-win-rate / high-payoff**: target **R = 3:1**, require realized calibrated WR to clear **25% + a cost buffer**. Costs shift the required WR *up* — budget the buffer explicitly. For a Polymarket-class 200 bps fee or a Coinbase 120 bps round-trip, add roughly the cost-as-fraction-of-avg-loss to the breakeven WR. **A high-WR path is only acceptable if the rare tail loss is hard-capped** (it is not, in any path we've tested). Caveat ([7] corrections): the *ideal* R compresses in practice — slippage on stops and early profit-taking can drop +0.60R ideal to ~+0.25R realized. Still positive, but plan for the thinner number.

---

## 6. Take-profit & stop-loss strategy (concrete spec)

**Where TP/SL does NOT apply (state plainly):** The recommended path (delta-neutral carry) and the live stablecoin yield runner are **non-directional** — there is no "stop" on a hedged position; the exit triggers are *funding-flip* and *decay-below-hurdle*, not price stops. Pure arb (YES+NO<$1) also has no TP/SL — it's held to resolution. **TP/SL applies only if a directional/edge path is revived** (multi-hour crypto directional, or a single-leg prediction-market bet). For that case, the concrete parameterized spec below:

**Stop-loss (volatility-scaled, not fixed %):**
- **Initial stop = entry − (k_sl × ATR(14))**, starting **k_sl = 2.0** ATR. Evidence: the existing simulator supports ATR-based stops (`simulator.py:744-760`); ATR scaling adapts the stop to current regime so you aren't stopped out by normal noise. For prediction markets where ATR is ill-defined, use a fixed fractional stop at **−0.33R to −0.50R** of cost basis.
- Stop is checked **first**, before TP (capital preservation — mirrors `exit_rules.py`, stop-loss evaluated before take-profit).

**Take-profit (R-multiple):**
- **TP = entry + (R_target × stop_distance)**, starting **R_target = 3.0** (i.e., 3:1). This makes breakeven WR = 25% (§5), giving a wide margin.
- Optionally scale out: take 50% at +2R, let the rest run with a trailing stop.

**Trailing rule:**
- Once price reaches **+1R**, move stop to breakeven (eliminate downside). Beyond that, trail at **k_trail × ATR**, starting **k_trail = 1.5** ATR (tighter than the initial 2.0 ATR to lock gains). The simulator supports trailing stops.

**Time stop:**
- Exit any directional position that has not reached +1R within **T_max** bars — start **T_max = 4× the label horizon** (e.g., for a 4h-horizon model, 16h). Rationale: our prior 1m runs that *did* fill exited on a 5-min time stop; a time stop prevents capital from being tied up in dead trades and bounds exposure.

**Validation requirement (mandatory before any of these go live):** Use `src/exit_backtest.py` `grid_search()` — it walks price paths chronologically (no look-ahead), tests every (stop_loss, tp_pct, tp_price) combo with honest 200 bps fees, and reports a `worst_return` floor. Pick the **risk-adjusted** combo (worst_return ≥ −60%), not the max-mean combo. Starting parameters above are priors to seed the grid, **not** validated values.

---

## 7. Position sizing & risk control

**Fractional Kelly, never full Kelly.** The growth-optimal fraction is **f\* = p − q/b = (bp − q)/b** = edge/odds ([8], confidence 0.96). But full Kelly on an *estimated* or *gross-of-fee* p overbets and can drive realized growth negative ([8]); beyond ~2× Kelly, geometric growth turns **negative despite positive per-trade EV** ([9], confidence 0.90 — the exact 2× point holds in the log-normal limit; "roughly" 2× for discrete bets). Therefore:

- **Size at k × f\*, with k = 0.25** (quarter-Kelly). This is exactly what `risk_management_agent/risk_engine.py` already does — it feeds a **fee-adjusted** edge into Kelly, applies the 0.25 fractional factor, then layers liquidity (0.5× if spread >5% or volume <10k) and correlation (30% per same-category position) penalties.
- **Feed Kelly the NET edge** (after fees+slippage+spread), never the raw calibrated probability ([8]). The risk engine deducts the 200 bps Polymarket platform fee *before* sizing.
- **Fixed-risk-per-trade alternative** (for the directional case): risk a constant **0.5-1.0% of bankroll per trade**, position size = risk_amount / stop_distance (the simulator's `max_risk_per_trade` sizing). This caps per-trade loss regardless of conviction.

**Hard kills (already built — `src/risk/circuit_breakers.py`, five breakers, severity-ordered):**
- **Kill-switch file** → force_flat (manual abort).
- **Daily realized loss limit** → halt_new_entries. Start: **−2% of bankroll/day**.
- **Max drawdown %** → halt_new_entries. Start: **−10% from equity peak** (matches the backtest gate's DD ≤ 10%).
- **Total notional cap** and **per-symbol notional cap** → halt. For the carry sleeve, cap total at a small fraction of capital and per-symbol so BTC and ETH are bounded independently.

**Leverage limits (carry-specific):** Keep the short-perp leg at **low effective leverage** (target ≤ 2-3×, large maintenance buffer) so a sharp up-move cannot liquidate the hedge before rebalance. Note: circuit_breakers does not currently gate max leverage — add `max_leverage_ratio` if the carry sleeve goes live (scoped lever in the infra audit).

**Backtest gate (already wired — `src/backtest.py:509-535`):** No model/strategy is promoted unless **profit_factor ≥ 1.8 AND max_drawdown ≤ 10%**, with the verdict persisted to `profit_report.json`. Honor this for any new edge.

---

## 8. Concrete next steps — the ladder

Governing rule (project constitution + infra audit): **no new subsystem without a validated edge; paper-first; the edge must clear the cost gate before any real money.**

**Rung 0 — Keep alive (do now):** Stablecoin yield SHADOW runner stays running as the cost-of-carry floor and idle-cash sink. Add the APY-compression alert (<4.0% → redeploy). No capital change.

**Rung 1 — Build & paper-test the BTC/ETH carry sleeve (the one new build):**
1. Retarget `funding_carry_scanner.py` to **BTC/ETH only**, rank by **realized persistence** (replay paid funding history), not snapshot.
2. Add **decay-triggered exit** and **funding-flip flat** to `funding_carry_runner.py`.
3. Run SHADOW for **≥30 days**, logging to `runs/funding_carry_ledger.jsonl` via the existing no-look-ahead ledger.
4. **GATE:** promote to live ONLY if net realized APY (after real Hyperliquid fees + basis drift + slippage) is **> 5% AND > the live stablecoin yield + a risk premium**. If it nets near-zero or negative (as the *first* carry run did, -0.17%), **kill it again** — that is a valid, honest outcome.

**Rung 2 — Live, tiny (only if Rung 1 gate passes):** Deploy a small sleeve with all circuit breakers armed (§7). Re-evaluate weekly against the live ledger.

**Retire / do NOT rebuild:**
- Crypto 1m directional XGBoost — keep the killed code as reference; do not revive without multi-hour labels showing >150 bps gross.
- Whale-follow — shelved permanently unless WebSocket CLOB entry-detection is built (out of scope).
- Intra-market arb — shelved; needs sub-second infra we don't have.

**Keep as reusable infra (do not rebuild):** `risk_engine.py`, `circuit_breakers.py`, `pnl_ledger.py`, `exit_rules.py`, `exit_backtest.py`, `trading/simulator.py`, `auto_pause.py` — all production-ready, fee-honest, no-look-ahead. Any new edge plugs into these.

---

## 9. Risks, caveats & what we DON'T know

**Where the evidence is thin (stated plainly):**
- The "~5-10% net carry APY" figure is a *range estimate* from BIS WP 1087 ([1]), **not** a number we have realized. Our only live carry data point lost money (-$13.64). We do **not** know the after-cost BTC/ETH-only carry until Rung 1 produces 30 days of honest shadow data. Treat the recommendation as *conditional on that test*.
- BIS carry figures are **gross**, and high carry is **crash-predictive** — it is compensation for tail risk, not free money ([1]). The fat 20-40% carries sit in illiquid names retail cannot cleanly hedge.
- The "Honest" whale variant's +$631/60% WR is **statistically inconclusive** (N=35, p=0.23) — likely luck, not edge. Do not resurrect it on this basis.

**Refuted myths — do NOT rely on these (they failed 2-3/3 adversarial votes):**
- ❌ "Cash-and-carry basis structurally collapsed to ~4.5%, 93% of days below breakeven." **Refuted 3/3.**
- ❌ "Perp funding is structurally pinned near the 0.01%/8h baseline, positive >92% of the time, Ethena floors spikes." **Refuted 3/3.**
- ❌ "A 2025 academic backtest showed only 40% of funding-arb opportunities positive after costs; CEX negative Sharpe." **Refuted 3/3.**
- ❌ "Ethena USDe realized ~11% avg delta-neutral carry." **Refuted 3/3** — do not use as a benchmark.
- ❌ "Cross-venue Polymarket-vs-Kalshi arb is geo-impossible for a US solo operator." **Refuted 3/3** (the *real* killer is settlement-divergence risk [3], not geo-fencing).
- ❌ "Polymarket pays makers a daily rebate / runs a low-variance resting-order yield program." **Refuted 3/3.**
- ❌ "The single biggest profit driver is expectancy × frequency × fractional sizing." **Refuted 3/3** as *worded* — but the underlying expectancy/Kelly math (§§5, 7) is from separately-verified claims [5][6][8].
- ❌ "Turtle-style trend followers run 30-40% WR at 3-5:1." **Refuted 3/3** as a sourced fact — the *general* low-WR/high-payoff principle is still valid via [6][7], just don't cite the Turtles.

**Known unknowns:** real BTC/ETH funding persistence at our hold horizon; actual Hyperliquid fill realism for the hedge legs; whether the carry's net edge survives a real de-peg/liquidation event; whether T-Bill rates fall below 4.5% (which would shift the stablecoin sleeve from "skip" to "worthwhile").

---

## 10. Citations

1. BIS Working Paper No. 1087 — *Crypto carry* (Apr 2023, rev. Oct 2025): https://www.bis.org/publ/work1087.htm
2. Hyperliquid fee schedule & maker-rebate program: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees
3. DeFi Rate — prediction-market contract settlement / divergence risk: https://defirate.com/prediction-markets/how-contracts-settle/
4. Kalshi Liquidity Incentive Program: https://help.kalshi.com/en/articles/13823851-liquidity-incentive-program
5. Trader's Second Brain — expectancy formula: https://traderssecondbrain.com/guides/expectancy-formula
6. LuxAlgo — win rate and risk/reward connection: https://www.luxalgo.com/blog/win-rate-and-riskreward-connection-explained/
7. HeyGoTrade — what is expectancy in trading: https://www.heygotrade.com/en/blog/what-is-expectancy-in-trading/
8. Wikipedia — Kelly criterion: https://en.wikipedia.org/wiki/Kelly_criterion
9. Wikipedia — Kelly criterion (overbetting / ~2× zero-growth): https://en.wikipedia.org/wiki/Kelly_criterion
