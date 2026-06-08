import unittest
from unittest.mock import patch

import requests

from fetcher import _market_from_gamma_payload, fetch_active_markets


class FakeResponse:
    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []
        self.closed = False

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        if not self.responses:
            raise AssertionError("No queued response left for FakeSession.get")
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class FetcherTests(unittest.TestCase):
    def test_market_from_gamma_payload_parses_market_fields(self):
        event_payload = {
            "title": "US Election",
            "description": "Event-level rules.",
            "resolutionSource": "Event source.",
            "openInterest": 125000.5,
            "endDate": "2026-11-05T00:00:00Z",
            "tags": [{"label": "Politics"}],
            "volume24hr": 8000.0,
        }
        market_payload = {
            "id": "mkt-123",
            "question": "Will candidate X win?",
            "description": "Market-level rules.",
            "bestBid": 0.41,
            "bestAsk": 0.43,
            "volume24hr": None,
            "oneHourPriceChange": 0.02,
            "oneHourVolumeChange": 0.07,
            "oneDayPriceChange": -0.03,
            "volume1wk": 56000.0,
            "endDate": "2026-11-04T23:00:00Z",
        }

        market = _market_from_gamma_payload(market_payload, event_payload)

        self.assertEqual(market.market_id, "mkt-123")
        self.assertEqual(market.title, "Will candidate X win?")
        self.assertEqual(market.category, "Politics")
        self.assertAlmostEqual(market.implied_prob, 0.42)
        self.assertAlmostEqual(market.volume_24h, 8000.0)
        self.assertEqual(market.price_history, {"1h": 0.02, "6h": 0.0, "24h": -0.03})
        self.assertAlmostEqual(market.open_interest, 125000.5)
        self.assertAlmostEqual(market.avg_volume_7d, 8000.0)
        self.assertAlmostEqual(market.volume_change_1h, 0.07)
        self.assertIn("Market-level rules.", market.rules_text)
        self.assertIn("Event-level rules.", market.rules_text)
        self.assertIn("Event source.", market.rules_text)

    @patch("fetcher.random.uniform", return_value=0.0)
    @patch("fetcher.time.sleep")
    def test_fetch_active_markets_handles_rate_limits_and_pagination(self, sleep_mock, _uniform_mock):
        page_one = [
            {
                "id": "m1",
                "question": "Market 1",
                "description": "Rules 1",
                "active": True,
                "closed": False,
                "bestBid": 0.60,
                "bestAsk": 0.62,
                "volume24hr": 9000.0,
                "oneHourPriceChange": 0.01,
                "oneDayPriceChange": 0.05,
                "endDate": "2026-06-01T00:00:00Z",
                "events": [{"openInterest": 50000.0, "description": "Event rules 1"}],
            },
            {
                "id": "m2",
                "question": "Market 2",
                "description": "Rules 2",
                "active": True,
                "closed": False,
                "bestBid": 0.40,
                "bestAsk": 0.42,
                "volume24hr": 3000.0,
                "endDate": "2026-06-01T00:00:00Z",
                "events": [{"openInterest": 40000.0, "description": "Event rules 2"}],
            },
        ]
        page_two = [
            {
                "id": "m3",
                "question": "Market 3",
                "description": "Rules 3",
                "active": True,
                "closed": False,
                "bestBid": 0.20,
                "bestAsk": 0.21,
                "volume24hr": 7500.0,
                "endDate": "2026-06-01T00:00:00Z",
                "events": [{"openInterest": 30000.0, "description": "Event rules 3"}],
            }
        ]
        session = FakeSession(
            [
                FakeResponse(429, [], headers={"Retry-After": "1"}),
                FakeResponse(200, page_one),
                FakeResponse(200, page_two),
            ]
        )

        markets = fetch_active_markets(min_volume_24h=5000.0, page_size=2, session=session)

        self.assertEqual([market.market_id for market in markets], ["m1", "m3"])
        self.assertEqual(session.calls[0]["params"]["offset"], 0)
        self.assertEqual(session.calls[1]["params"]["offset"], 0)
        self.assertEqual(session.calls[2]["params"]["offset"], 2)
        self.assertEqual(session.headers["Accept"], "application/json")
        self.assertEqual(session.headers["User-Agent"], "autopilot-polymarket-scanner/1.0")
        sleep_mock.assert_called_once_with(1.0)
        self.assertFalse(session.closed)


if __name__ == "__main__":
    unittest.main()
