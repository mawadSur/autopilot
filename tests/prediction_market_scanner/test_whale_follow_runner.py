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

from state.pnl_ledger import PnlLedger

import whale_follow_runner as runner
from whale_follow_runner import STRATEGY, find_convergence, run_once


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
# Module-level safety: no order/execution symbols exported
# ---------------------------------------------------------------------------


class ModuleSafetyTest(unittest.TestCase):
    def test_runner_module_has_no_order_helpers(self) -> None:
        for forbidden in ("place_order", "create_order", "submit_order", "sign_order"):
            self.assertFalse(hasattr(runner, forbidden))


if __name__ == "__main__":
    unittest.main()
