"""Tests for src/whale_follow_runner.py (SHADOW-ONLY).

No network. A fake data-api client returns canned /holders payloads (the EXACT
live-verified shape). We assert:
  * find_convergence flags an outcome only when >= N DISTINCT target wallets
    hold it, ignoring non-target wallets and below-threshold counts;
  * run_once logs exactly the expected SHADOW open records to a tempfile
    PnlLedger, places NO order (the fake exposes only read methods), and the
    records carry the shadow flag + contributing wallets.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any, Dict, List

from state.pnl_ledger import PnlLedger, TradeRecord
from exchanges.polymarket_data_api import PolymarketDataAPIError

import whale_follow_runner as runner
from whale_follow_runner import (
    STRATEGY,
    convergence_from_positions,
    find_convergence,
    make_whale_price_fn,
    run_once,
    _latest_price_for_outcome,
)


def _holder(wallet: str, outcome_index: int, name: str) -> Dict[str, Any]:
    """A /holders[].holders[] shaped dict with the fields the runner reads."""
    return {
        "proxyWallet": wallet,
        "asset": "111" if outcome_index == 0 else "222",
        "amount": 100.0,
        "outcomeIndex": outcome_index,
        "name": name,
        "pseudonym": wallet[-3:],
        "verified": True,
    }


def _holders_groups(yes_wallets: List[str], no_wallets: List[str]) -> List[Dict[str, Any]]:
    """Build the per-token /holders payload for a binary market."""
    return [
        {
            "token": "111",
            "holders": [_holder(w, 0, "Yes") for w in yes_wallets],
        },
        {
            "token": "222",
            "holders": [_holder(w, 1, "No") for w in no_wallets],
        },
    ]


class _FakeClient:
    """Read-only fake exposing ONLY get_holders — no order/execute surface."""

    def __init__(self, holders_by_market: Dict[str, List[Dict[str, Any]]]) -> None:
        self._holders_by_market = holders_by_market
        self.holders_calls: List[str] = []

    def get_holders(self, market_condition_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        self.holders_calls.append(market_condition_id)
        return self._holders_by_market.get(market_condition_id, [])


def _trade(outcome_index: int, price: float, timestamp: int) -> Dict[str, Any]:
    """A /trades[] shaped dict with the fields _latest_price_for_outcome reads."""
    return {
        "proxyWallet": "0xWHALE",
        "side": "BUY",
        "conditionId": "0xCOND",
        "price": price,
        "outcome": "Yes" if outcome_index == 0 else "No",
        "outcomeIndex": outcome_index,
        "timestamp": timestamp,
    }


class _PricedFakeClient(_FakeClient):
    """Fake exposing get_holders AND get_trades (current price), still read-only.

    ``get_trades`` returns the per-market canned trade list and records each
    call so tests can assert whether the price path was exercised.
    """

    def __init__(
        self,
        holders_by_market: Dict[str, List[Dict[str, Any]]],
        trades_by_market: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        super().__init__(holders_by_market)
        self._trades_by_market = trades_by_market
        self.trades_calls: List[Dict[str, Any]] = []

    def get_trades(
        self,
        user: Any = None,
        market: Any = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        self.trades_calls.append({"market": market, "limit": limit})
        return self._trades_by_market.get(market, [])


class _RaisingTradesClient(_FakeClient):
    """Fake whose get_trades blows up — proves the no-price path is robust AND
    that mark_entry=False never calls get_trades at all (the test fails loudly
    if it does)."""

    def __init__(
        self,
        holders_by_market: Dict[str, List[Dict[str, Any]]],
        *,
        error: Exception = None,
    ) -> None:
        super().__init__(holders_by_market)
        self._error = error or AssertionError(
            "get_trades must not be called when mark_entry is False"
        )
        self.trades_calls = 0

    def get_trades(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        self.trades_calls += 1
        raise self._error


TARGETS = ["0xT1", "0xT2", "0xT3", "0xT4"]


# ---------------------------------------------------------------------------
# find_convergence
# ---------------------------------------------------------------------------


class FindConvergenceTest(unittest.TestCase):
    def test_flags_outcome_with_enough_target_holders(self) -> None:
        # 3 targets on YES (idx 0), plus an untracked wallet that must be ignored.
        groups = _holders_groups(
            yes_wallets=["0xT1", "0xT2", "0xT3", "0xUNTRACKED"],
            no_wallets=["0xT4"],
        )
        client = _FakeClient({"0xCOND": groups})

        cands = find_convergence(
            client, TARGETS, markets_condition_ids=["0xCOND"], min_convergence=3
        )

        self.assertEqual(len(cands), 1)
        cand = cands[0]
        self.assertEqual(cand["conditionId"], "0xCOND")
        self.assertEqual(cand["outcomeIndex"], 0)
        self.assertEqual(cand["outcome"], "Yes")
        self.assertEqual(cand["n_target_holders"], 3)
        self.assertEqual(set(cand["wallets"]), {"0xT1", "0xT2", "0xT3"})
        # The untracked wallet was not counted.
        self.assertNotIn("0xUNTRACKED", cand["wallets"])

    def test_below_threshold_not_flagged(self) -> None:
        groups = _holders_groups(yes_wallets=["0xT1", "0xT2"], no_wallets=["0xT3"])
        client = _FakeClient({"0xCOND": groups})
        cands = find_convergence(
            client, TARGETS, markets_condition_ids=["0xCOND"], min_convergence=3
        )
        self.assertEqual(cands, [])

    def test_non_target_wallets_ignored(self) -> None:
        # Three non-target wallets on YES; zero targets converge.
        groups = _holders_groups(
            yes_wallets=["0xX", "0xY", "0xZ"], no_wallets=[]
        )
        client = _FakeClient({"0xCOND": groups})
        cands = find_convergence(
            client, TARGETS, markets_condition_ids=["0xCOND"], min_convergence=3
        )
        self.assertEqual(cands, [])

    def test_duplicate_wallet_counted_once(self) -> None:
        # 0xT1 appears twice on YES; counts as a single distinct holder.
        groups = [
            {
                "token": "111",
                "holders": [
                    _holder("0xT1", 0, "Yes"),
                    _holder("0xT1", 0, "Yes"),  # duplicate row
                    _holder("0xT2", 0, "Yes"),
                ],
            }
        ]
        client = _FakeClient({"0xCOND": groups})
        cands = find_convergence(
            client, TARGETS, markets_condition_ids=["0xCOND"], min_convergence=3
        )
        # Only 2 distinct targets -> below the threshold of 3.
        self.assertEqual(cands, [])

    def test_multiple_markets_scanned(self) -> None:
        client = _FakeClient(
            {
                "0xCOND1": _holders_groups(["0xT1", "0xT2", "0xT3"], []),
                "0xCOND2": _holders_groups([], ["0xT1", "0xT2", "0xT4"]),
            }
        )
        cands = find_convergence(
            client,
            TARGETS,
            markets_condition_ids=["0xCOND1", "0xCOND2"],
            min_convergence=3,
        )
        by_market = {c["conditionId"]: c for c in cands}
        self.assertEqual(set(by_market), {"0xCOND1", "0xCOND2"})
        self.assertEqual(by_market["0xCOND1"]["outcomeIndex"], 0)
        self.assertEqual(by_market["0xCOND2"]["outcomeIndex"], 1)


# ---------------------------------------------------------------------------
# run_once: logs SHADOW records, places NO orders
# ---------------------------------------------------------------------------


class RunOnceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "pnl_ledger.jsonl")
        self.ledger = PnlLedger(self.path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_logs_exactly_expected_shadow_records(self) -> None:
        client = _FakeClient(
            {
                "0xCOND1": _holders_groups(["0xT1", "0xT2", "0xT3"], []),
                "0xCOND2": _holders_groups([], ["0xT1", "0xT2", "0xT4"]),
            }
        )

        cands = run_once(
            ledger=self.ledger,
            client=client,
            target_wallets=TARGETS,
            markets_condition_ids=["0xCOND1", "0xCOND2"],
            min_convergence=3,
        )
        self.assertEqual(len(cands), 2)

        records = self.ledger.all_records()
        self.assertEqual(len(records), 2)
        by_market = {r.market_id: r for r in records}
        self.assertEqual(set(by_market), {"0xCOND1", "0xCOND2"})

        rec1 = by_market["0xCOND1"]
        self.assertEqual(rec1.venue, "polymarket")
        self.assertEqual(rec1.strategy, STRATEGY)
        self.assertEqual(rec1.side, "Yes")
        self.assertEqual(rec1.status, "open")
        self.assertEqual(rec1.entry_price, 0.0)
        # Shadow flag + contributing wallets recorded in notes.
        self.assertIn("SHADOW MODE - NO ORDERS", rec1.notes)
        for w in ("0xT1", "0xT2", "0xT3"):
            self.assertIn(w, rec1.notes)

        rec2 = by_market["0xCOND2"]
        self.assertEqual(rec2.side, "No")

    def test_no_convergence_logs_nothing(self) -> None:
        client = _FakeClient({"0xCOND1": _holders_groups(["0xT1"], ["0xT2"])})
        cands = run_once(
            ledger=self.ledger,
            client=client,
            target_wallets=TARGETS,
            markets_condition_ids=["0xCOND1"],
            min_convergence=3,
        )
        self.assertEqual(cands, [])
        self.assertEqual(self.ledger.all_records(), [])

    def test_client_has_no_order_method(self) -> None:
        client = _FakeClient({})
        # The runner only ever calls read methods; assert the fake exposes
        # nothing that could place/sign an order.
        for forbidden in (
            "place_order",
            "create_order",
            "submit_order",
            "sign",
            "post",
        ):
            self.assertFalse(hasattr(client, forbidden))

    def test_none_client_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_once(
                ledger=self.ledger,
                client=None,
                target_wallets=TARGETS,
                markets_condition_ids=["0xCOND1"],
            )

    def test_records_use_size_usd_label(self) -> None:
        client = _FakeClient({"0xCOND1": _holders_groups(["0xT1", "0xT2", "0xT3"], [])})
        run_once(
            ledger=self.ledger,
            client=client,
            target_wallets=TARGETS,
            markets_condition_ids=["0xCOND1"],
            min_convergence=3,
            size_usd=0.0,
        )
        rec = self.ledger.all_records()[0]
        self.assertEqual(rec.size, 0.0)


# ---------------------------------------------------------------------------
# _latest_price_for_outcome: current decision-time entry price
# ---------------------------------------------------------------------------


class LatestPriceForOutcomeTest(unittest.TestCase):
    def test_picks_max_timestamp_matching_outcome(self) -> None:
        client = _PricedFakeClient(
            {},
            {
                "0xCOND": [
                    _trade(0, 0.40, timestamp=100),  # older YES
                    _trade(0, 0.55, timestamp=300),  # newest YES -> wins
                    _trade(0, 0.50, timestamp=200),  # middle YES
                    _trade(1, 0.99, timestamp=999),  # NO: ignored (wrong outcome)
                ]
            },
        )
        price = _latest_price_for_outcome(client, "0xCOND", 0)
        self.assertEqual(price, 0.55)
        # It queried /trades for the right market.
        self.assertEqual(client.trades_calls[-1]["market"], "0xCOND")

    def test_no_matching_outcome_returns_none(self) -> None:
        client = _PricedFakeClient(
            {}, {"0xCOND": [_trade(1, 0.30, timestamp=100)]}
        )
        # Asking for outcomeIndex 0 when only outcome 1 traded -> None.
        self.assertIsNone(_latest_price_for_outcome(client, "0xCOND", 0))

    def test_empty_trades_returns_none(self) -> None:
        client = _PricedFakeClient({}, {"0xCOND": []})
        self.assertIsNone(_latest_price_for_outcome(client, "0xCOND", 0))

    def test_data_api_error_returns_none(self) -> None:
        client = _RaisingTradesClient(
            {}, error=PolymarketDataAPIError("boom")
        )
        self.assertIsNone(_latest_price_for_outcome(client, "0xCOND", 0))

    def test_arbitrary_exception_returns_none(self) -> None:
        client = _RaisingTradesClient({}, error=RuntimeError("network down"))
        self.assertIsNone(_latest_price_for_outcome(client, "0xCOND", 0))

    def test_out_of_range_price_skipped(self) -> None:
        client = _PricedFakeClient(
            {},
            {
                "0xCOND": [
                    _trade(0, 1.50, timestamp=300),  # invalid (>1), newest -> skipped
                    _trade(0, 0.42, timestamp=200),  # valid -> wins
                ]
            },
        )
        self.assertEqual(_latest_price_for_outcome(client, "0xCOND", 0), 0.42)


# ---------------------------------------------------------------------------
# run_once mark_entry: real entry price, and default preserves behavior
# ---------------------------------------------------------------------------


class RunOnceMarkEntryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "pnl_ledger.jsonl")
        self.ledger = PnlLedger(self.path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_mark_entry_true_records_latest_matching_price(self) -> None:
        client = _PricedFakeClient(
            {"0xCOND": _holders_groups(["0xT1", "0xT2", "0xT3"], [])},
            {
                "0xCOND": [
                    _trade(0, 0.40, timestamp=100),
                    _trade(0, 0.62, timestamp=500),  # newest YES -> entry price
                    _trade(1, 0.90, timestamp=900),  # NO: ignored
                ]
            },
        )
        run_once(
            ledger=self.ledger,
            client=client,
            target_wallets=TARGETS,
            markets_condition_ids=["0xCOND"],
            min_convergence=3,
            mark_entry=True,
        )
        rec = self.ledger.all_records()[0]
        self.assertEqual(rec.entry_price, 0.62)
        # Notes reflect a real mark, NOT the legacy "entry_price=0.0" claim.
        self.assertIn("entry_price=0.6200", rec.notes)
        self.assertNotIn("entry_price=0.0 (holders endpoint", rec.notes)
        # outcomeIndex marker preserved for later re-pricing.
        self.assertIn("outcomeIndex=0", rec.notes)
        self.assertTrue(client.trades_calls)

    def test_mark_entry_true_no_price_falls_back_to_zero(self) -> None:
        # Holders converge on YES (idx 0) but only NO (idx 1) has traded.
        client = _PricedFakeClient(
            {"0xCOND": _holders_groups(["0xT1", "0xT2", "0xT3"], [])},
            {"0xCOND": [_trade(1, 0.30, timestamp=100)]},
        )
        run_once(
            ledger=self.ledger,
            client=client,
            target_wallets=TARGETS,
            markets_condition_ids=["0xCOND"],
            min_convergence=3,
            mark_entry=True,
        )
        rec = self.ledger.all_records()[0]
        self.assertEqual(rec.entry_price, 0.0)
        self.assertIn("entry_price=0.0 (holders endpoint carries no price)", rec.notes)

    def test_mark_entry_false_records_zero_and_makes_no_trades_call(self) -> None:
        # If get_trades is called the fake raises AssertionError -> test fails.
        client = _RaisingTradesClient(
            {"0xCOND": _holders_groups(["0xT1", "0xT2", "0xT3"], [])}
        )
        run_once(
            ledger=self.ledger,
            client=client,
            target_wallets=TARGETS,
            markets_condition_ids=["0xCOND"],
            min_convergence=3,
            # mark_entry defaults to False
        )
        rec = self.ledger.all_records()[0]
        self.assertEqual(rec.entry_price, 0.0)
        self.assertIn("entry_price=0.0 (holders endpoint carries no price)", rec.notes)
        self.assertEqual(client.trades_calls, 0)


# ---------------------------------------------------------------------------
# convergence_from_positions: leaderboard-mode convergence from CURRENT holdings
# ---------------------------------------------------------------------------


def _position(
    *,
    condition_id: str,
    outcome_index: int,
    outcome: str = "Yes",
    size: float = 100.0,
    cur_price: float = 0.55,
    redeemable: bool = False,
    realized_pnl: float = 0.0,
) -> Dict[str, Any]:
    """A /positions-shaped dict with the fields convergence_from_positions reads.

    Defaults describe a CURRENT OPEN holding (size>0, not redeemable,
    0<curPrice<1). Override to make it resolved/settled.
    """
    return {
        "proxyWallet": "0xW",
        "conditionId": condition_id,
        "outcomeIndex": outcome_index,
        "outcome": outcome,
        "size": size,
        "curPrice": cur_price,
        "redeemable": redeemable,
        "realizedPnl": realized_pnl,
    }


class _PositionsFakeClient:
    """Read-only fake exposing ONLY get_positions(user=...).

    Returns canned positions per wallet; a wallet listed in ``raises_for`` makes
    get_positions raise (to exercise the per-wallet skip path). No order surface.
    """

    def __init__(
        self,
        positions_by_wallet: Dict[str, List[Dict[str, Any]]],
        raises_for: Any = None,
    ) -> None:
        self._positions_by_wallet = positions_by_wallet
        self._raises_for = set(raises_for or ())
        self.positions_calls: List[str] = []

    def get_positions(self, user: str, limit: int = 500) -> List[Dict[str, Any]]:
        self.positions_calls.append(user)
        if user in self._raises_for:
            raise PolymarketDataAPIError(f"boom for {user}")
        return self._positions_by_wallet.get(user, [])


class ConvergenceFromPositionsTest(unittest.TestCase):
    def test_emits_one_candidate_for_shared_current_holding(self) -> None:
        # T1 and T2 currently hold the SAME (cond,outcome); T3 holds elsewhere.
        client = _PositionsFakeClient(
            {
                "0xT1": [_position(condition_id="0xCONV", outcome_index=0)],
                "0xT2": [_position(condition_id="0xCONV", outcome_index=0)],
                "0xT3": [_position(condition_id="0xOTHER", outcome_index=1,
                                   outcome="No")],
            }
        )
        cands = convergence_from_positions(
            client, ["0xT1", "0xT2", "0xT3"], min_convergence=2
        )
        self.assertEqual(len(cands), 1)
        cand = cands[0]
        self.assertEqual(cand["conditionId"], "0xCONV")
        self.assertEqual(cand["outcomeIndex"], 0)
        self.assertEqual(cand["outcome"], "Yes")
        self.assertEqual(cand["n_target_holders"], 2)
        self.assertEqual(set(cand["wallets"]), {"0xT1", "0xT2"})
        # Candidate shape is the find_convergence shape plus ``title`` (for the
        # do-not-trade blocklist) and ``asset`` (the CLOB outcome token id, for
        # book-aware exit fills), both carried from /positions.
        self.assertEqual(
            set(cand),
            {"conditionId", "outcomeIndex", "outcome", "title", "asset",
             "n_target_holders", "wallets"},
        )

    def test_resolved_and_redeemable_positions_excluded(self) -> None:
        # Both wallets "hold" 0xRES idx 0 but each holding is settled, so it
        # must NOT converge. T1 redeemable; T2 curPrice at the 1.0 boundary.
        client = _PositionsFakeClient(
            {
                "0xT1": [_position(condition_id="0xRES", outcome_index=0,
                                   redeemable=True)],
                "0xT2": [_position(condition_id="0xRES", outcome_index=0,
                                   cur_price=1.0)],
            }
        )
        cands = convergence_from_positions(
            client, ["0xT1", "0xT2"], min_convergence=2
        )
        self.assertEqual(cands, [])

    def test_zero_price_and_zero_size_excluded(self) -> None:
        client = _PositionsFakeClient(
            {
                "0xT1": [_position(condition_id="0xX", outcome_index=0,
                                   cur_price=0.0)],   # resolved-to-0
                "0xT2": [_position(condition_id="0xX", outcome_index=0,
                                   size=0.0)],        # nothing held
            }
        )
        cands = convergence_from_positions(
            client, ["0xT1", "0xT2"], min_convergence=2
        )
        self.assertEqual(cands, [])

    def test_errored_wallet_skipped_others_still_count(self) -> None:
        # T2 raises; T1 and T3 still converge on the shared holding.
        client = _PositionsFakeClient(
            {
                "0xT1": [_position(condition_id="0xCONV", outcome_index=1,
                                   outcome="No")],
                "0xT3": [_position(condition_id="0xCONV", outcome_index=1,
                                   outcome="No")],
            },
            raises_for=["0xT2"],
        )
        cands = convergence_from_positions(
            client, ["0xT1", "0xT2", "0xT3"], min_convergence=2
        )
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["outcomeIndex"], 1)
        self.assertEqual(set(cands[0]["wallets"]), {"0xT1", "0xT3"})
        # The bad wallet WAS attempted (proving the skip, not a silent omission).
        self.assertIn("0xT2", client.positions_calls)

    def test_distinct_wallet_counted_once_per_market_outcome(self) -> None:
        # T1 holds the same (cond,outcome) across two position rows -> count 1.
        client = _PositionsFakeClient(
            {
                "0xT1": [
                    _position(condition_id="0xCONV", outcome_index=0),
                    _position(condition_id="0xCONV", outcome_index=0),
                ],
                "0xT2": [_position(condition_id="0xCONV", outcome_index=0)],
            }
        )
        cands = convergence_from_positions(
            client, ["0xT1", "0xT2"], min_convergence=2
        )
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["n_target_holders"], 2)

    def test_below_threshold_not_flagged(self) -> None:
        client = _PositionsFakeClient(
            {"0xT1": [_position(condition_id="0xCONV", outcome_index=0)]}
        )
        cands = convergence_from_positions(
            client, ["0xT1"], min_convergence=2
        )
        self.assertEqual(cands, [])

    def test_return_stats_yields_winrates_from_same_positions(self) -> None:
        # T1: 1 settled win + the open converging holding -> win_rate 1.0.
        # T2: 1 settled loss + the open converging holding -> win_rate 0.0.
        client = _PositionsFakeClient(
            {
                "0xT1": [
                    _position(condition_id="0xSETTLED", outcome_index=0,
                              redeemable=True, cur_price=1.0, realized_pnl=50.0),
                    _position(condition_id="0xCONV", outcome_index=0),
                ],
                "0xT2": [
                    _position(condition_id="0xSETTLED2", outcome_index=1,
                              redeemable=True, cur_price=0.0, realized_pnl=-20.0),
                    _position(condition_id="0xCONV", outcome_index=0),
                ],
            }
        )
        cands, stats = convergence_from_positions(
            client, ["0xT1", "0xT2"], min_convergence=2, return_stats=True
        )
        self.assertEqual(len(cands), 1)
        self.assertEqual(set(stats), {"0xT1", "0xT2"})
        self.assertEqual(stats["0xT1"].win_rate, 1.0)
        self.assertEqual(stats["0xT2"].win_rate, 0.0)
        # Built from the SAME fetch — exactly one /positions call per wallet.
        self.assertEqual(sorted(client.positions_calls), ["0xT1", "0xT2"])

    def test_client_has_no_order_method(self) -> None:
        client = _PositionsFakeClient({})
        for forbidden in ("place_order", "create_order", "submit_order", "sign", "post"):
            self.assertFalse(hasattr(client, forbidden))


# ---------------------------------------------------------------------------
# make_whale_price_fn: re-price an open position from its notes marker
# ---------------------------------------------------------------------------


class MakeWhalePriceFnTest(unittest.TestCase):
    def _record(self, notes: str, market_id: str = "0xCOND") -> TradeRecord:
        return TradeRecord(
            trade_id="whale-test",
            ts_utc="2026-06-01T00:00:00+00:00",
            venue="polymarket",
            market_id=market_id,
            side="Yes",
            entry_price=0.5,
            size=0.0,
            fees_usd=0.0,
            slippage_bps=0.0,
            strategy=STRATEGY,
            status="open",
            notes=notes,
        )

    def test_parses_outcome_index_and_returns_fetched_price(self) -> None:
        client = _PricedFakeClient(
            {},
            {
                "0xCOND": [
                    _trade(1, 0.10, timestamp=100),  # NO: ignored
                    _trade(1, 0.73, timestamp=400),  # newest NO -> returned
                ]
            },
        )
        price_fn = make_whale_price_fn(client)
        record = self._record(
            "SHADOW MODE - NO ORDERS; whale_convergence n=3 outcomeIndex=1; "
            "entry_price=0.0 (holders endpoint carries no price); wallets=0xT1,0xT2,0xT3"
        )
        self.assertEqual(price_fn(record), 0.73)
        self.assertEqual(client.trades_calls[-1]["market"], "0xCOND")

    def test_returns_none_when_notes_have_no_outcome_index(self) -> None:
        client = _PricedFakeClient(
            {}, {"0xCOND": [_trade(0, 0.50, timestamp=100)]}
        )
        price_fn = make_whale_price_fn(client)
        record = self._record("no marker here at all")
        self.assertIsNone(price_fn(record))
        # Without an outcomeIndex we must not even hit /trades.
        self.assertEqual(client.trades_calls, [])

    def test_returns_none_when_fetch_fails(self) -> None:
        client = _RaisingTradesClient({}, error=PolymarketDataAPIError("boom"))
        price_fn = make_whale_price_fn(client)
        record = self._record("whale_convergence outcomeIndex=0;")
        self.assertIsNone(price_fn(record))


# ---------------------------------------------------------------------------
# Module-level safety: no order/execution symbols exported
# ---------------------------------------------------------------------------


class ModuleSafetyTest(unittest.TestCase):
    def test_runner_module_has_no_order_helpers(self) -> None:
        for forbidden in ("place_order", "create_order", "submit_order", "sign_order"):
            self.assertFalse(hasattr(runner, forbidden))


# ---------------------------------------------------------------------------
# main(): --roster-source leaderboard vs live, single-pass, injected fakes
# ---------------------------------------------------------------------------


class _MainFakeClient:
    """Read-only fake covering BOTH main() paths.

    leaderboard path uses get_profit_leaderboard + get_positions; live path uses
    get_trades + get_positions + get_holders. NO order/execute surface.
    """

    def __init__(
        self,
        *,
        leaderboard: List[Dict[str, Any]] = None,
        positions_by_wallet: Dict[str, List[Dict[str, Any]]] = None,
        trades: List[Dict[str, Any]] = None,
        holders_by_market: Dict[str, List[Dict[str, Any]]] = None,
        market_trades: Dict[str, List[Dict[str, Any]]] = None,
    ) -> None:
        self._leaderboard = leaderboard or []
        self._positions_by_wallet = positions_by_wallet or {}
        self._trades = trades or []
        self._holders_by_market = holders_by_market or {}
        self._market_trades = market_trades or {}

    def get_profit_leaderboard(self, window: str = "all", limit: int = 100):
        return self._leaderboard

    def get_positions(self, user: str, limit: int = 500):
        return self._positions_by_wallet.get(user, [])

    def get_trades(self, user: Any = None, market: Any = None,
                   limit: int = 100, offset: int = 0):
        if market is not None:
            return self._market_trades.get(market, [])
        # Global feed paged by offset (live discovery walks pages).
        return self._trades if offset == 0 else []

    def get_holders(self, market_condition_id: str, limit: int = 100):
        return self._holders_by_market.get(market_condition_id, [])


class MainTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "pnl_ledger.jsonl")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_main(self, fake: "_MainFakeClient", argv: List[str]) -> int:
        """Run main() with PolymarketDataAPIClient swapped for ``fake``."""
        import exchanges.polymarket_data_api as data_api_mod
        original = data_api_mod.PolymarketDataAPIClient
        data_api_mod.PolymarketDataAPIClient = lambda *a, **k: fake
        try:
            return runner.main(argv + ["--ledger-path", self.path])
        finally:
            data_api_mod.PolymarketDataAPIClient = original

    def test_leaderboard_mode_logs_candidate_with_confidence(self) -> None:
        # Two leaderboard winners both CURRENTLY hold 0xCONV idx 0 -> converge.
        # A current /trades mark on that outcome gives a real entry price.
        fake = _MainFakeClient(
            leaderboard=[
                {"proxyWallet": "0xW1", "amount": 9e6, "name": "Theo4"},
                {"proxyWallet": "0xW2", "amount": 5e5, "name": "WhaleTwo"},
            ],
            positions_by_wallet={
                "0xW1": [
                    _position(condition_id="0xCONV", outcome_index=0,
                              outcome="Yes"),
                    # a settled win so confidence sees a real win-rate
                    _position(condition_id="0xSET", outcome_index=0,
                              redeemable=True, cur_price=1.0, realized_pnl=10.0),
                ],
                "0xW2": [
                    _position(condition_id="0xCONV", outcome_index=0,
                              outcome="Yes"),
                ],
            },
            market_trades={
                "0xCONV": [_trade(0, 0.61, timestamp=500)],
            },
        )

        rc = self._run_main(
            fake,
            ["--roster-source", "leaderboard", "--min-convergence", "2",
             "--leaderboard-limit", "10"],
        )
        self.assertEqual(rc, 0)

        ledger = PnlLedger(self.path)
        records = ledger.all_records()
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.market_id, "0xCONV")
        self.assertEqual(rec.side, "Yes")
        self.assertEqual(rec.strategy, STRATEGY)
        self.assertEqual(rec.status, "open")
        # Real decision-time entry mark from /trades.
        self.assertEqual(rec.entry_price, 0.61)
        # Confidence marker present in notes (the leaderboard path scores it).
        self.assertRegex(rec.notes, r"confidence=[0-9.]+ \((low|medium|high)\)")
        self.assertIn("SHADOW MODE - NO ORDERS", rec.notes)
        for w in ("0xW1", "0xW2"):
            self.assertIn(w, rec.notes)

    def test_leaderboard_mode_no_convergence_logs_nothing(self) -> None:
        fake = _MainFakeClient(
            leaderboard=[{"proxyWallet": "0xW1", "amount": 1.0}],
            positions_by_wallet={
                "0xW1": [_position(condition_id="0xCONV", outcome_index=0)],
            },
        )
        rc = self._run_main(
            fake, ["--roster-source", "leaderboard", "--min-convergence", "2"]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(PnlLedger(self.path).all_records(), [])

    def test_settlement_on_by_default_closes_resolved_position(self) -> None:
        # Seed an OPEN whale position whose market has since RESOLVED (won).
        # On the next main() run (settlement ON by default) it must be settled.
        seed = PnlLedger(self.path)
        seed.append(
            TradeRecord(
                trade_id="whale-seed",
                ts_utc="2026-06-01T00:00:00+00:00",
                venue="polymarket",
                market_id="0xRESOLVED",
                side="Yes",
                entry_price=0.50,
                size=100.0,
                fees_usd=0.0,
                slippage_bps=0.0,
                strategy=STRATEGY,
                status="open",
                notes="SHADOW MODE - NO ORDERS; whale_convergence n=3 "
                "outcomeIndex=0; entry_price=0.5000; wallets=0xT1,0xT2,0xT3",
            )
        )

        # No convergence this run (empty leaderboard) — we only care about the
        # settlement sweep firing.
        fake = _MainFakeClient(leaderboard=[])

        import exchanges.polymarket_market_data as mkt_mod
        original = mkt_mod.get_market_resolution
        mkt_mod.get_market_resolution = lambda cid: (
            {"closed": True,
             "tokens": [
                 {"outcome": "Yes", "price": 1.0, "winner": True},
                 {"outcome": "No", "price": 0.0, "winner": False},
             ]}
            if cid == "0xRESOLVED"
            else None
        )
        try:
            rc = self._run_main(fake, ["--roster-source", "leaderboard"])
        finally:
            mkt_mod.get_market_resolution = original
        self.assertEqual(rc, 0)

        ledger = PnlLedger(self.path)
        settled = ledger.settled()
        self.assertEqual(len(settled), 1)
        rec = settled[0]
        self.assertEqual(rec.trade_id, "whale-seed")
        self.assertEqual(rec.exit_price, 1.0)
        self.assertEqual(rec.market_outcome, "won:Yes")
        self.assertAlmostEqual(rec.realized_pnl_usd, 96.0, places=4)
        self.assertEqual(ledger.open_positions(), [])

    def test_no_settle_flag_leaves_resolved_position_open(self) -> None:
        # Same seed, but --no-settle: the resolver must never be consulted and
        # the position stays open.
        seed = PnlLedger(self.path)
        seed.append(
            TradeRecord(
                trade_id="whale-seed",
                ts_utc="2026-06-01T00:00:00+00:00",
                venue="polymarket",
                market_id="0xRESOLVED",
                side="Yes",
                entry_price=0.50,
                size=100.0,
                fees_usd=0.0,
                slippage_bps=0.0,
                strategy=STRATEGY,
                status="open",
                notes="SHADOW MODE - NO ORDERS; whale_convergence n=3 "
                "outcomeIndex=0; entry_price=0.5000; wallets=0xT1",
            )
        )
        fake = _MainFakeClient(leaderboard=[])

        import exchanges.polymarket_market_data as mkt_mod
        calls: List[str] = []
        original = mkt_mod.get_market_resolution

        def _spy(cid: str):
            calls.append(cid)
            return {"closed": True, "tokens": [{"winner": True}, {"winner": False}]}

        mkt_mod.get_market_resolution = _spy
        try:
            rc = self._run_main(
                fake, ["--roster-source", "leaderboard", "--no-settle"]
            )
        finally:
            mkt_mod.get_market_resolution = original
        self.assertEqual(rc, 0)

        # --no-settle: resolver untouched, position still open.
        self.assertEqual(calls, [])
        ledger = PnlLedger(self.path)
        self.assertEqual(len(ledger.open_positions()), 1)
        self.assertEqual(ledger.settled(), [])

    def test_live_mode_still_works_as_before(self) -> None:
        # Live path: discover wallets from /trades, rank by /positions, derive
        # hot markets, flag convergence on /holders. Two ranked wallets converge.
        ranked_positions = [
            _position(condition_id=f"0xH{i}", outcome_index=0, outcome="Yes",
                      redeemable=True, cur_price=1.0, realized_pnl=5.0)
            for i in range(20)
        ]
        fake = _MainFakeClient(
            trades=[
                {"proxyWallet": "0xA", "conditionId": "0xHOT"},
                {"proxyWallet": "0xB", "conditionId": "0xHOT"},
                {"proxyWallet": "0xC", "conditionId": "0xHOT"},
            ],
            positions_by_wallet={
                "0xA": ranked_positions,
                "0xB": ranked_positions,
                "0xC": ranked_positions,
            },
            holders_by_market={
                "0xHOT": _holders_groups(["0xA", "0xB", "0xC"], []),
            },
            market_trades={"0xHOT": [_trade(0, 0.44, timestamp=300)]},
        )

        rc = self._run_main(
            fake,
            ["--roster-source", "live", "--min-convergence", "3",
             "--min-settled", "20", "--min-win-rate", "0.60",
             "--discover-pages", "1", "--max-markets", "5"],
        )
        self.assertEqual(rc, 0)

        records = PnlLedger(self.path).all_records()
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.market_id, "0xHOT")
        self.assertEqual(rec.side, "Yes")
        self.assertEqual(rec.entry_price, 0.44)
        self.assertEqual(rec.strategy, STRATEGY)


if __name__ == "__main__":
    unittest.main()
