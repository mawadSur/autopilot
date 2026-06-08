# Stocks adapter — Alpaca (paper)

P3 backlog item from `TODOS.md`: "Once Lane D ships the Tradeable
Protocol, a Stocks adapter is a 2-day brief instead of a 2-week port."
This adapter wires US equities into the live supervisor on top of Lane D
D1 (`Tradeable` protocol) + D2 (adapter scaffolding) so the supervisor's
tick loop can iterate Alpaca symbols alongside the existing crypto +
Polymarket tradeables.

> Constitution alignment (`CLAUDE.md`): Article 0 mission explicitly
> includes stocks. Paper-mode default per Article 2 — live execution is
> 🔒 GATED behind `ALPACA_TRADING_ENABLED=true` at the connector layer,
> never on by default.

## What it does

| Surface | File | Purpose |
| --- | --- | --- |
| HTTP client | `src/exchanges/alpaca.py` | Thin `requests`-based wrapper. Read paths (`get_account`, `get_clock`, `get_calendar`, `get_asset`, `get_ticker`, `get_open_orders`, `get_position`) are live; write paths (`place_market_order`, `place_limit_order`, `cancel_order`) are 🔒 flag-gated. |
| Protocol adapter | `src/exchanges/adapters/alpaca_tradeable.py` | `AlpacaTradeable` — bind one `AlpacaExchange` + one ticker to the `Tradeable` protocol. Asset class `SPOT_EQUITY`. |
| Supervisor wiring | `src/live_supervisor.py` | `--alpaca-symbols AAPL,MSFT` CLI flag, paper-default `AlpacaExchange` construction from env, dispatch routing for `spot_equity` in `_dispatch_tick` / `_tick_spot_equity`. |

`AlpacaTradeable.symbol` returns `"alpaca:AAPL"` so the supervisor's
heterogeneous-symbol routing can disambiguate venues from a single
string — same pattern as `polymarket:<id>`.

### Conventions baked in

- **Asset class**: `AssetClass.SPOT_EQUITY` (added to the enum in this work; backward-compatible).
- **Fee model default**: `FeeModel(maker=0.0, taker=0.0, settlement_fee_bps=0)`. Alpaca is commission-free for retail US equities; sub-bp SEC + TAF settlement fees exist but are typically absorbed by the broker. Override via the `fee_model=` kwarg for paid SIP / non-retail tiers.
- **Tick size**: `0.01` (Reg NMS Rule 612 — US equities above $1 trade in cent ticks).
- **Min size**: `1.0` (whole share). Pass `min_size=0.0001` for fractional-share symbols.
- **Risk attributes**: `kelly_divisor=1.0`, `liquidation_price=None`, `margin_used_usd=None`. Spot equity has no implied probability and no implicit leverage; margin accounts on Alpaca exist but per-position margin pre-trade requires symbol-level lookups the venue does not expose today (documented TODO).

## Paper-mode safety story

Three layered gates ensure a fresh checkout cannot accidentally place a
real order:

1. **Default base URL is paper**. `AlpacaExchange(api_key, api_secret)` with no `paper=` override routes every trading-API call to `https://paper-api.alpaca.markets/v2`. Live writes require `paper=False` AND a custom `--base-url`-equivalent env var (`ALPACA_BASE_URL`).
2. **Write paths refuse without the env flag**. `place_market_order`, `place_limit_order`, and `cancel_order` all read `ALPACA_TRADING_ENABLED` on every call (not at construction — so toggling the flag flips behaviour without restarting). When unset/falsy, the call raises `NotImplementedError` with a pointer to the flag. The adapter does **not** re-check the flag; it merely delegates so a single source-of-truth governs the gate.
3. **Supervisor `--mode live` is independent**. Even when `--mode live` is set, the 14-day paper-trade shakedown still has to clear before live order routing is honoured on any symbol. Alpaca's flag is additional, not alternative.

Read paths (account snapshot, clock, calendar, latest quote/bar, open
orders, current position) are **not** gated — they're necessary for the
supervisor to observe market state in paper mode.

## Environment variables

| Name | Required | Default | Purpose |
| --- | --- | --- | --- |
| `ALPACA_API_KEY` | yes | _(empty)_ | Alpaca API key id. Both paper and live keys live in the operator's Alpaca dashboard. |
| `ALPACA_API_SECRET` | yes | _(empty)_ | Alpaca API secret. **Never commit.** |
| `ALPACA_PAPER` | no | `true` | Set to `false` to construct a live `AlpacaExchange`. Recommended workflow: leave `true` until you've validated the paper-trading bot for ≥14 calendar days. |
| `ALPACA_BASE_URL` | no | _(auto)_ | Override trading API base URL. Defaults to the paper or live URL per `ALPACA_PAPER`. Useful for an internal proxy. |
| `ALPACA_TRADING_ENABLED` | for writes | unset | 🔒 Must be `true` for any order placement / cancel. Defaults to unset → writes raise `NotImplementedError`. |

The supervisor's `main()` already loads `.env` from repo root and
`src/.env` (`python-dotenv`) before reading these.

## Example supervisor invocation

Paper-mode tick loop on Apple + Microsoft, exiting after one pass:

```bash
ALPACA_API_KEY=PK… ALPACA_API_SECRET=SK… \
  ./.venv/bin/python -m src.live_supervisor \
    --alpaca-symbols AAPL,MSFT \
    --once \
    --shakedown-state-path ./.shakedown.json
```

Heterogeneous tick (crypto + Polymarket + Alpaca in a single supervisor
process):

```bash
./.venv/bin/python -m src.live_supervisor \
  --symbols ETH/USDT,BTC/USDT \
  --polymarket-markets 0x123abc \
  --alpaca-symbols AAPL,MSFT \
  --interval 5 \
  --mode paper
```

Live writes (🔒 — only after a clean 14-day paper run + manual review):

```bash
ALPACA_API_KEY=… ALPACA_API_SECRET=… \
ALPACA_PAPER=false \
ALPACA_TRADING_ENABLED=true \
  ./.venv/bin/python -m src.live_supervisor \
    --alpaca-symbols AAPL \
    --mode live
```

Even with all three set, `--mode live` only takes effect after the
shakedown gate unlocks each symbol's `paper_days_clean >=
shakedown_min_days` (default 14).

## Test status

Two hermetic suites cover the adapter; both run offline (every HTTP call
is patched at `exchanges.alpaca.requests.{get,post,delete}`):

```bash
# Stocks-stack suite (new)
env PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests/stocks
# 41 tests, all passing.

# Prediction-market suite (pre-existing, still green)
env PYTHONPATH=src ./.venv/bin/python -m unittest \
  tests.prediction_market_scanner.test_alpaca_tradeable \
  tests.prediction_market_scanner.test_alpaca_exchange
# 45 tests, all passing.
```

Coverage:

- Protocol conformance (`isinstance(adapter, Tradeable)`).
- Static metadata (symbol namespacing, tick_size, min_size, asset_class).
- Fee model default + override.
- Paper-mode safety: connector defaults to paper base URL, `ALPACA_TRADING_ENABLED` truthy/falsy parsing, write paths refuse when unset, adapter propagates `NotImplementedError`.
- HTTP-mocked ticker fetch (combined quote + bar, fallback to mid on bar 404).
- Order placement happy path (market + cancel, with flag set + HTTP mocked).
- Risk attributes (cash equity, no leverage).
- Supervisor registration: CLI parses `--alpaca-symbols`, `SupervisorConfig` accepts tradeables-only, `main()` rejects the empty case, `_dispatch_tick` routes `spot_equity` to the adapter-driven handler (never to `self.exchange.get_ticker`), kill-switch force-flats, ticker errors are caught and recorded.

## Dependencies

The adapter uses only the existing `requests` dependency — no
`alpaca-py` SDK. This was a deliberate choice: the environment this
worktree runs in does not have PyPI access at boot, and Alpaca's REST
API is small enough that a hand-rolled client (under 850 lines incl.
docstrings + tests) is honest about its surface area. The fallback HTTP
client lives at `src/exchanges/alpaca.py`. Should an operator want the
official SDK, the adapter can be retargeted in a follow-up PR — the
`Tradeable` protocol shape is the public contract, not the connector.

## Known follow-ups

- **Real order routing** (🔒): writes are wired but every supervisor tick today only reads ticker data. A follow-up needs to (a) build a `predict_fn` for equities and (b) thread it through the supervisor's risk gate the same way crypto does. The Alpaca write paths themselves are flag-gated and tested.
- **Margin sizing**: `risk_attributes.margin_used_usd` returns `None`. Wiring real margin requires per-symbol initial-margin lookups Alpaca does not expose pre-trade.
- **Market-hours awareness**: `AlpacaExchange.get_clock()` is exposed but the supervisor's tick loop does not yet skip ticks outside RTH. Easy follow-up once a predictor lands.
- **Short-borrow checks**: `AlpacaAsset.shortable` / `easy_to_borrow` are surfaced on the asset metadata but not consulted before submitting a short.
