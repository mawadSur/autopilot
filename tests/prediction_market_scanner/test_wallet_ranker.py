"""Tests for src/wallet_ranker.py.

No network: a tiny fake data-api client returns canned /trades and /positions
payloads (the EXACT live-verified shapes). We pin down the no-look-ahead rule
(only settled positions count), the win-rate / realized-PnL math, discovery
de-duplication + page bounds, and the rank_wallets filter + sort.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

from wallet_ranker import (
    WalletStats,
    discover_active_wallets,
    rank_wallets,
    stats_from_positions,
)


def _position(
    *,
    condition_id: str = "0xCOND",
    realized_pnl: float = 0.0,
    redeemable: bool = True,
) -> Dict[str, Any]:
    """A /positions-shaped dict with the fields the ranker reads."""
    return {
        "proxyWallet": "0xW",
        "asset": "111",
        "conditionId": condition_id,
        "size": 100.0,
        "avgPrice": 0.5,
        "initialValue": 50.0,
        "currentValue": 50.0 + realized_pnl,
        "cashPnl": realized_pnl,
        "percentPnl": 0.0,
        "totalBought": 50.0,
        "realizedPnl": realized_pnl,
        "percentRealizedPnl": 0.0,
        "curPrice": 1.0 if realized_pnl > 0 else 0.0,
        "redeemable": redeemable,
        "title": "t",
        "outcome": "Yes",
        "outcomeIndex": 0,
        "oppositeOutcome": "No",
        "oppositeAsset": "222",
        "endDate": "2026-05-01T00:00:00Z",
        "negativeRisk": False,
    }


class _FakeClient:
    """Read-only fake exposing only get_trades / get_positions.

    Deliberately exposes NO order/execution method so a test can assert the
    ranker never reaches for one.
    """

    def __init__(
        self,
        trades_pages: Optional[List[List[Dict[str, Any]]]] = None,
        positions_by_wallet: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        self._trades_pages = trades_pages or []
        self._positions_by_wallet = positions_by_wallet or {}
        self.positions_calls: List[tuple] = []

    def get_trades(self, *, limit: int = 100, offset: int = 0, **_kw: Any) -> List[Dict[str, Any]]:
        page_index = offset // limit if limit else 0
        if 0 <= page_index < len(self._trades_pages):
            return self._trades_pages[page_index]
        return []

    def get_positions(self, user: str, limit: int = 500) -> List[Dict[str, Any]]:
        self.positions_calls.append((user, limit))
        return self._positions_by_wallet.get(user, [])


# ---------------------------------------------------------------------------
# stats_from_positions: no look-ahead + math
# ---------------------------------------------------------------------------


class StatsFromPositionsTest(unittest.TestCase):
    def test_only_settled_positions_count(self) -> None:
        positions = [
            _position(realized_pnl=10.0, redeemable=True),   # settled win
            _position(realized_pnl=-5.0, redeemable=True),   # settled loss
            _position(realized_pnl=999.0, redeemable=False),  # OPEN -> ignored
        ]
        stats = stats_from_positions("0xW", positions)
        # Only the two settled positions feed the record.
        self.assertEqual(stats.n_settled, 2)
        self.assertEqual(stats.n_wins, 1)
        self.assertAlmostEqual(stats.win_rate, 0.5)
        # 10 + (-5) = 5; the unsettled +999 must NOT leak in.
        self.assertAlmostEqual(stats.realized_pnl_usd, 5.0)
        self.assertEqual(stats.sampled_positions, 3)

    def test_unsettled_position_excluded_from_winrate(self) -> None:
        # A single, unsettled, would-be huge winner must not count at all.
        stats = stats_from_positions(
            "0xW", [_position(realized_pnl=1000.0, redeemable=False)]
        )
        self.assertEqual(stats.n_settled, 0)
        self.assertEqual(stats.n_wins, 0)
        self.assertEqual(stats.win_rate, 0.0)
        self.assertEqual(stats.realized_pnl_usd, 0.0)

    def test_flat_position_is_settled_but_not_a_win(self) -> None:
        stats = stats_from_positions(
            "0xW", [_position(realized_pnl=0.0, redeemable=True)]
        )
        self.assertEqual(stats.n_settled, 1)
        self.assertEqual(stats.n_wins, 0)  # exactly flat is not > 0
        self.assertEqual(stats.win_rate, 0.0)

    def test_empty_positions(self) -> None:
        stats = stats_from_positions("0xW", [])
        self.assertEqual(stats, WalletStats("0xW", 0, 0, 0.0, 0.0, 0))


# ---------------------------------------------------------------------------
# discover_active_wallets: distinct + bounds
# ---------------------------------------------------------------------------


class DiscoverWalletsTest(unittest.TestCase):
    def test_distinct_wallets_first_seen_order(self) -> None:
        page0 = [
            {"proxyWallet": "0xA"},
            {"proxyWallet": "0xB"},
            {"proxyWallet": "0xA"},  # dup within a page
        ]
        page1 = [
            {"proxyWallet": "0xB"},  # dup across pages
            {"proxyWallet": "0xC"},
        ]
        client = _FakeClient(trades_pages=[page0, page1])
        wallets = discover_active_wallets(client, pages=3, page_size=3)
        self.assertEqual(wallets, ["0xA", "0xB", "0xC"])

    def test_stops_on_short_page(self) -> None:
        # Page 0 is short (1 < page_size 3) -> feed exhausted, stop.
        client = _FakeClient(trades_pages=[[{"proxyWallet": "0xA"}]])
        wallets = discover_active_wallets(client, pages=5, page_size=3)
        self.assertEqual(wallets, ["0xA"])

    def test_page_bound_respected(self) -> None:
        # Three FULL pages available but pages=2 -> only first two consulted.
        full = lambda tag: [{"proxyWallet": f"0x{tag}{i}"} for i in range(3)]
        client = _FakeClient(trades_pages=[full("A"), full("B"), full("C")])
        wallets = discover_active_wallets(client, pages=2, page_size=3)
        self.assertNotIn("0xC0", wallets)
        self.assertEqual(len(wallets), 6)


# ---------------------------------------------------------------------------
# rank_wallets: filter + sort
# ---------------------------------------------------------------------------


class RankWalletsTest(unittest.TestCase):
    def _wallet_positions(self, n_settled: int, n_wins: int, pnl: float) -> List[Dict[str, Any]]:
        """Build a positions list with an exact settled count / win count / total PnL.

        Each loss carries a fixed small negative; every win carries a fixed
        positive EXCEPT the first, which absorbs the remainder so the summed
        realized PnL equals ``pnl`` exactly (no float-split drift).
        """
        positions: List[Dict[str, Any]] = []
        n_losses = n_settled - n_wins
        loss_each = -1.0
        win_each = 1.0
        # first_win absorbs: target_total - (other_wins * win_each) - (losses * loss_each)
        first_win = pnl - (n_wins - 1) * win_each - n_losses * loss_each
        for i in range(n_wins):
            value = first_win if i == 0 else win_each
            positions.append(_position(realized_pnl=value, redeemable=True))
        for _ in range(n_losses):
            positions.append(_position(realized_pnl=loss_each, redeemable=True))
        return positions

    def test_filters_then_sorts_by_pnl_desc(self) -> None:
        positions_by_wallet = {
            # passes: 25 settled, 20 wins (0.80 >= 0.60), pnl 500
            "0xHIGH": self._wallet_positions(25, 20, 500.0),
            # passes: 25 settled, 18 wins (0.72), pnl 900 -> ranks ABOVE 0xHIGH
            "0xTOP": self._wallet_positions(25, 18, 900.0),
            # fails win_rate: 25 settled but only 10 wins (0.40)
            "0xLOWWR": self._wallet_positions(25, 10, 50.0),
            # fails min_settled: only 5 settled
            "0xTHIN": self._wallet_positions(5, 5, 1000.0),
        }
        client = _FakeClient(positions_by_wallet=positions_by_wallet)

        ranked = rank_wallets(
            client,
            candidate_wallets=list(positions_by_wallet.keys()),
            min_settled=20,
            min_win_rate=0.60,
            top_n=50,
        )

        # Only the two qualifying wallets survive, sorted by realized PnL desc.
        self.assertEqual([s.wallet for s in ranked], ["0xTOP", "0xHIGH"])
        self.assertAlmostEqual(ranked[0].realized_pnl_usd, 900.0, places=2)
        # Each survivor fetched exactly once (N+1-style, one /positions per wallet).
        self.assertEqual(len(client.positions_calls), 4)

    def test_top_n_truncates(self) -> None:
        positions_by_wallet = {
            "0xA": self._wallet_positions(25, 20, 300.0),
            "0xB": self._wallet_positions(25, 20, 200.0),
            "0xC": self._wallet_positions(25, 20, 100.0),
        }
        client = _FakeClient(positions_by_wallet=positions_by_wallet)
        ranked = rank_wallets(
            client,
            candidate_wallets=list(positions_by_wallet.keys()),
            min_settled=20,
            min_win_rate=0.60,
            top_n=2,
        )
        self.assertEqual([s.wallet for s in ranked], ["0xA", "0xB"])

    def test_discovers_when_no_candidates_given(self) -> None:
        trades_pages = [[{"proxyWallet": "0xA"}, {"proxyWallet": "0xB"}]]
        positions_by_wallet = {
            "0xA": self._wallet_positions(25, 20, 500.0),
            "0xB": self._wallet_positions(25, 5, 50.0),  # 0.20 win-rate -> filtered
        }
        client = _FakeClient(
            trades_pages=trades_pages, positions_by_wallet=positions_by_wallet
        )
        ranked = rank_wallets(client, min_settled=20, min_win_rate=0.60)
        self.assertEqual([s.wallet for s in ranked], ["0xA"])

    def test_client_exposes_no_order_method(self) -> None:
        # The fake client used throughout has no order/execute/sign surface.
        client = _FakeClient()
        for forbidden in ("place_order", "create_order", "sign", "submit_order"):
            self.assertFalse(hasattr(client, forbidden))


if __name__ == "__main__":
    unittest.main()
