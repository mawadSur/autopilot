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


if __name__ == "__main__":
    unittest.main()
