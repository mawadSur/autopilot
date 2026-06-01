"""Tests for src/arb_shadow_runner.py.

Fully offline: a fake ``fetch_markets_fn`` returns synthetic markets and a
fake ``market_data_client`` returns crafted YES/NO best asks. One market's
asks sum to < 1 (an arb), one sums to >= 1 (no arb), and one has unavailable
asks (must be skipped without crashing). We run ``run_once`` against a
tempfile :class:`PnlLedger` and assert the arb is found, exactly the expected
number of shadow records were logged with ``status='open'``, and — the safety
invariant — that the client only ever has read methods called (no
order/execution surface exists or is touched).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Sequence, Tuple

from arb_shadow_runner import build_market_rows, run_once
from state.pnl_ledger import PnlLedger


# ---------------------------------------------------------------------------
# Fakes (no network, READ-ONLY surface only)
# ---------------------------------------------------------------------------


class _FakeMarketDataClient:
    """Read-only stand-in for ``exchanges.polymarket_market_data``.

    Maps a market id -> crafted ``(yes_ask, no_ask)`` (or ``None`` to signal
    "asks unavailable"). Records every read so tests can prove the runner only
    ever READS. There is deliberately NO order/place/sign/submit method here —
    if the runner ever tried to call one, the test would ``AttributeError``.
    """

    def __init__(self, asks_by_id: Dict[str, Optional[Tuple[float, float]]]) -> None:
        self._asks_by_id = asks_by_id
        self.read_calls: List[str] = []

    def get_yes_no_best_asks(
        self,
        market: Any,
        session: Any = None,
    ) -> Optional[Tuple[float, float]]:
        market_id = str(market.get("id"))
        self.read_calls.append(market_id)
        return self._asks_by_id.get(market_id)


def _markets() -> List[Dict[str, Any]]:
    return [
        {"id": "ARB", "title": "Arb market (asks sum < 1)"},
        {"id": "NOARB", "title": "No-arb market (asks sum >= 1)"},
        {"id": "MISSING", "title": "Asks unavailable (must be skipped)"},
    ]


def _asks_map() -> Dict[str, Optional[Tuple[float, float]]]:
    return {
        # 0.45 + 0.45 = 0.90 -> gross edge 0.10, easily an arb after 200bps fee.
        "ARB": (0.45, 0.45),
        # 0.55 + 0.50 = 1.05 -> no gross edge, no arb.
        "NOARB": (0.55, 0.50),
        # None -> asks unavailable; the runner must skip this row.
        "MISSING": None,
    }


class ArbShadowRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "runs", "pnl_ledger.jsonl")
        self.ledger = PnlLedger(self.path)
        self.client = _FakeMarketDataClient(_asks_map())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _fetch(self) -> Sequence[Dict[str, Any]]:
        return _markets()

    # ------------------------------------------------------------------
    # build_market_rows
    # ------------------------------------------------------------------

    def test_build_rows_skips_unavailable_market(self) -> None:
        rows = build_market_rows(_markets(), market_data_client=self.client)
        ids = {row["id"] for row in rows}
        self.assertIn("ARB", ids)
        self.assertIn("NOARB", ids)
        self.assertNotIn("MISSING", ids)  # skipped, no crash
        self.assertEqual(len(rows), 2)
        # Rows carry the canonical arb-input fields.
        arb_row = next(r for r in rows if r["id"] == "ARB")
        self.assertEqual(arb_row["yes_ask"], 0.45)
        self.assertEqual(arb_row["no_ask"], 0.45)
        self.assertEqual(arb_row["title"], "Arb market (asks sum < 1)")

    def test_build_rows_max_markets_caps_scan(self) -> None:
        rows = build_market_rows(_markets(), market_data_client=self.client, max_markets=1)
        # Only the first market is read at all.
        self.assertEqual(self.client.read_calls, ["ARB"])
        self.assertEqual(len(rows), 1)

    def test_build_rows_resilient_to_per_market_read_error(self) -> None:
        class _BoomClient:
            def __init__(self) -> None:
                self.read_calls: List[str] = []

            def get_yes_no_best_asks(self, market: Any, session: Any = None):
                mid = str(market.get("id"))
                self.read_calls.append(mid)
                if mid == "NOARB":
                    raise RuntimeError("transient book read failure")
                return _asks_map().get(mid)

        client = _BoomClient()
        rows = build_market_rows(_markets(), market_data_client=client)
        # NOARB blew up but the scan continued; ARB still made it through.
        ids = {row["id"] for row in rows}
        self.assertEqual(ids, {"ARB"})
        self.assertEqual(client.read_calls, ["ARB", "NOARB", "MISSING"])

    # ------------------------------------------------------------------
    # run_once
    # ------------------------------------------------------------------

    def test_run_once_finds_arb_and_logs_one_shadow_record(self) -> None:
        opportunities = run_once(
            ledger=self.ledger,
            fetch_markets_fn=self._fetch,
            market_data_client=self.client,
            size_usd=100.0,
        )

        # Exactly one arb (ARB); NOARB rejected by the identity, MISSING skipped.
        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0]["market_id"], "ARB")
        self.assertGreater(opportunities[0]["net_edge_pct"], 0.0)

        # Exactly one shadow record, status='open', no exit/settlement.
        records = self.ledger.all_records()
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.status, "open")
        self.assertEqual(rec.market_id, "ARB")
        self.assertEqual(rec.venue, "polymarket")
        self.assertEqual(rec.side, "YES+NO")
        self.assertIsNone(rec.exit_price)
        self.assertIsNone(rec.realized_pnl_usd)
        self.assertIn("SHADOW", rec.notes)

        # open_positions agrees; nothing settled.
        self.assertEqual(len(self.ledger.open_positions()), 1)
        self.assertEqual(len(self.ledger.settled()), 0)

    def test_run_once_logs_nothing_when_no_arb(self) -> None:
        # All markets non-arb / unavailable.
        client = _FakeMarketDataClient(
            {
                "ARB": (0.60, 0.55),  # 1.15, no arb now
                "NOARB": (0.55, 0.50),
                "MISSING": None,
            }
        )
        opportunities = run_once(
            ledger=self.ledger,
            fetch_markets_fn=self._fetch,
            market_data_client=client,
        )
        self.assertEqual(opportunities, [])
        self.assertEqual(self.ledger.all_records(), [])

    def test_run_once_only_calls_read_methods_no_execution(self) -> None:
        run_once(
            ledger=self.ledger,
            fetch_markets_fn=self._fetch,
            market_data_client=self.client,
        )
        # The fake client exposes ONLY get_yes_no_best_asks; assert that is the
        # entire surface the runner used and that no order/execution-shaped
        # attribute even exists to be called.
        self.assertEqual(self.client.read_calls, ["ARB", "NOARB", "MISSING"])
        for forbidden in (
            "place_order",
            "submit_order",
            "create_order",
            "post_order",
            "sign",
            "sign_order",
            "execute",
            "buy",
            "sell",
            "send_transaction",
        ):
            self.assertFalse(
                hasattr(self.client, forbidden),
                msg=f"shadow client must not expose execution method {forbidden!r}",
            )

    def test_run_once_handles_empty_market_fetch(self) -> None:
        opportunities = run_once(
            ledger=self.ledger,
            fetch_markets_fn=lambda: [],
            market_data_client=self.client,
        )
        self.assertEqual(opportunities, [])
        self.assertEqual(self.ledger.all_records(), [])
        self.assertEqual(self.client.read_calls, [])


if __name__ == "__main__":
    unittest.main()
