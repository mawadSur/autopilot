"""End-to-end test: orchestrator-shaped event_payload → mark_settled → PerformanceTracker → analytics_dashboard."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from analytics_dashboard import _extract_trades, summarize_trades
from mark_trade_settled import mark_settled
from outcome_review_agent.analyzer import OutcomeReviewAgent
from outcome_review_agent.logger import PerformanceTracker


def _sample_review_dict() -> dict:
    return {
        "matrix_classification": "Deserved Success",
        "thesis_held": True,
        "unknown_at_entry": False,
        "calibration_reasonable": True,
        "resulting_detected": False,
        "research_module_flaw": False,
        "risk_module_flaw": False,
        "key_takeaways": [],
        "reasoning": "Test entry.",
    }


class FakeResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class _FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.response


class FakeClient:
    def __init__(self, response):
        self.models = _FakeModels(response)


class FakeTypesModule:
    def HttpOptions(self, **kwargs):
        return type("HttpOptions", (), {"kwargs": kwargs})()

    def GenerateContentConfig(self, **kwargs):
        return type("GenerateContentConfig", (), {"kwargs": kwargs})()


def _orchestrator_event_payload(market_id: str) -> dict:
    """Mimics the slot shape produced by orchestrator.run_final_risk_gate."""
    return {
        "event_id": market_id,
        "trade_id": market_id,
        "status": "open",
        "created_at_utc": "2026-04-25T00:00:00+00:00",
        "settled_at": None,
        "final_outcome": None,
        "post_settlement_news": None,
        "scanner": {"price": 0.42, "volume_24h": 12000},
        "calibration": {"calibrated_true_prob": 0.55},
        "risk": {"risk_assessment": {"allow_trade": True}},
    }


class OutcomeReviewEndToEndTests(unittest.TestCase):
    def test_open_trade_is_invisible_until_marked_settled(self):
        """The orchestrator writes status=open; tracker should ignore until settled."""
        agent = OutcomeReviewAgent(
            api_key="test-key",
            client=FakeClient(FakeResponse(parsed=_sample_review_dict())),
            types_module=FakeTypesModule(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            log_path = store / "trade_execution_mkt-7.json"
            log_path.write_text(json.dumps(_orchestrator_event_payload("mkt-7")), encoding="utf-8")

            tracker = PerformanceTracker(trade_store_dir=store, outcome_review_agent=agent)
            first = tracker.process_settled_trades()
            self.assertEqual(first["new_reviews"], 0)
            self.assertEqual(len(agent.client.models.calls), 0)

            mark_settled(log_path, final_outcome=True, post_settlement_news="Resolved.")

            second = tracker.process_settled_trades()
            self.assertEqual(second["new_reviews"], 1)
            self.assertEqual(len(agent.client.models.calls), 1)

    def test_full_chain_produces_dashboard_summary(self):
        """orchestrator payloads → mark_settled (varied outcomes) → tracker → dashboard summary."""
        # Different review dicts for the four quadrants so the dashboard sees variety.
        reviews_by_market = {
            "mkt-A": {**_sample_review_dict(), "matrix_classification": "Deserved Success"},
            "mkt-B": {**_sample_review_dict(), "matrix_classification": "Good Failure"},
            "mkt-C": {**_sample_review_dict(), "matrix_classification": "Dumb Luck"},
            "mkt-D": {**_sample_review_dict(), "matrix_classification": "Poetic Justice"},
        }

        # Stateful fake client: returns the next queued review per call.
        class StatefulModels:
            def __init__(self, queue):
                self.queue = list(queue)
                self.calls = []

            def generate_content(self, *, model, contents, config):
                self.calls.append({"model": model, "contents": contents, "config": config})
                return FakeResponse(parsed=self.queue.pop(0))

        class StatefulClient:
            def __init__(self, queue):
                self.models = StatefulModels(queue)

        outcomes = {"mkt-A": True, "mkt-B": False, "mkt-C": True, "mkt-D": False}

        # Order matters — PerformanceTracker iterates trade files sorted by name.
        sorted_ids = sorted(reviews_by_market)
        agent = OutcomeReviewAgent(
            api_key="test-key",
            client=StatefulClient([reviews_by_market[mid] for mid in sorted_ids]),
            types_module=FakeTypesModule(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            for market_id in sorted_ids:
                log = store / f"trade_execution_{market_id}.json"
                log.write_text(json.dumps(_orchestrator_event_payload(market_id)), encoding="utf-8")
                mark_settled(log, final_outcome=outcomes[market_id], post_settlement_news=f"News for {market_id}")

            tracker = PerformanceTracker(trade_store_dir=store, outcome_review_agent=agent)
            result = tracker.process_settled_trades()
            self.assertEqual(result["new_reviews"], 4)

            audit = json.loads((store / "performance_audit.json").read_text(encoding="utf-8"))

            # Dashboard consumes the audit straight from disk.
            trades = _extract_trades(audit)
            buf = io.StringIO()
            with redirect_stdout(buf):
                summarize_trades(trades)
            output = buf.getvalue()

            self.assertIn("Total Trades: 4", output)
            self.assertIn("Win Rate: 50.00%", output)             # mkt-A + mkt-C won
            self.assertIn("Process Integrity Score: 50.00%", output)  # Deserved + Good Failure
            self.assertIn("Deserved Success: 1", output)
            self.assertIn("Good Failure: 1", output)
            self.assertIn("Dumb Luck: 1", output)
            self.assertIn("Poetic Justice: 1", output)


if __name__ == "__main__":
    unittest.main()
