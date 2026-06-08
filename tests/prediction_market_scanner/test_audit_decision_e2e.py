"""End-to-end audit-and-decision pipeline test.

Walks a single market through the full chain:

    orchestrator.run_final_risk_gate
        -> writes trade_execution_<id>.json (status=open)
    mark_trade_settled.mark_settled
        -> flips status to "settled" + fills final_outcome / settled_at
    PerformanceTracker.process_settled_trades
        -> dispatches OutcomeReviewAgent + DataQualityAuditor (FakeClient)
        -> writes performance_audit.json
    analytics_dashboard._extract_trades + summarize_trades
        -> parses the audit and prints a summary

All LLM calls are stubbed via the FakeClient/FakeResponse pattern that the
existing outcome-review and data-quality tests already use. The trade store is
a ``tempfile.TemporaryDirectory`` so we don't pollute the repo. The
orchestrator's hard-coded REPO_ROOT write target is monkey-patched into the
tempdir for the duration of the test.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import orchestrator
from analytics_dashboard import _extract_trades, summarize_trades
from calibration_agent.models import CalibrationReport
from data_quality_auditor.analyzer import DataQualityAuditor
from mark_trade_settled import mark_settled
from models import Market
from orchestrator import run_final_risk_gate
from outcome_review_agent.analyzer import OutcomeReviewAgent
from outcome_review_agent.logger import PerformanceTracker
from risk_management_agent.risk_engine import RiskCalculator


# ---- Fake LLM client wiring ------------------------------------------------


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


# ---- Sample agent responses ------------------------------------------------


def _outcome_review_dict() -> dict:
    return {
        "matrix_classification": "Deserved Success",
        "thesis_held": True,
        "unknown_at_entry": False,
        "calibration_reasonable": True,
        "resulting_detected": False,
        "research_module_flaw": False,
        "risk_module_flaw": False,
        "key_takeaways": ["Process and outcome both lined up."],
        "reasoning": "Calibration was reasonable and the outcome matched the thesis.",
    }


def _clean_data_quality_audit_dict() -> dict:
    finding = {"detected": False, "evidence": "No anomalies observed."}
    return {
        "stale_data": finding,
        "duplicate_sources": finding,
        "missing_primary_sources": finding,
        "misleading_sentiment": finding,
        "scraping_gaps": finding,
        "timestamp_mismatches": finding,
        "incorrect_market_metadata": finding,
        "data_failure": False,
        "failure_mode": None,
        "severity": None,
        "recommended_fix": "No action needed.",
    }


# ---- Test fixtures ---------------------------------------------------------


def _build_market() -> Market:
    return Market(
        market_id="mkt-audit-1",
        title="Will Candidate X Win Audit-Test?",
        category="Politics",
        implied_prob=0.42,
        bid_price=0.41,
        ask_price=0.43,
        volume_24h=25000.0,
        price_history={"1h": 0.005, "6h": 0.01, "24h": 0.02},
        open_interest=40000.0,
        resolution_date=datetime(2026, 11, 5, tzinfo=timezone.utc),
        rules_text="Resolves to YES if Candidate X wins.",
    )


def _build_calibration() -> CalibrationReport:
    return CalibrationReport(
        xgboost_prob=0.45,
        llm_adjustment_pct_points=2.0,
        calibrated_true_prob=0.55,
        confidence_score=82,
        key_drivers=["Reddit surfaced a credible underpricing signal."],
        key_uncertainties=["Sample size of fresh evidence is small."],
        edge_vs_market=0.08,
        action="paper-trade candidate",
        reasoning="Modest upward adjustment supported by qualitative evidence.",
    )


# ---- The end-to-end test ---------------------------------------------------


class AuditDecisionEndToEndTests(unittest.TestCase):
    """Test 3 — orchestrator decision -> mark_settled -> audit -> dashboard summary."""

    def test_full_chain_orchestrator_to_dashboard_summary(self):
        market = _build_market()
        calibration = _build_calibration()
        scanner_row = {
            "market_id": market.market_id,
            "title": market.title,
            "category": market.category,
            "implied_prob": market.implied_prob,
            "research_priority": 99,
        }

        # Stub the OutcomeReviewAgent + DataQualityAuditor with FakeClients so
        # PerformanceTracker can dispatch both without ever touching Gemini.
        outcome_agent = OutcomeReviewAgent(
            api_key="test-key",
            client=FakeClient(FakeResponse(parsed=_outcome_review_dict())),
            types_module=FakeTypesModule(),
        )
        data_quality_auditor = DataQualityAuditor(
            api_key="test-key",
            client=FakeClient(FakeResponse(parsed=_clean_data_quality_audit_dict())),
            types_module=FakeTypesModule(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)

            # The orchestrator's `_write_trade_execution_log` writes to
            # REPO_ROOT by default. Redirect that into our tempdir so the
            # test doesn't leak files into the repo.
            def _write_into_tempdir(*, event_payload, market_id):
                output_path = store / f"trade_execution_{market_id}.json"
                output_path.write_text(
                    json.dumps(
                        orchestrator._to_serializable(event_payload),
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                return output_path

            with patch.object(
                orchestrator, "_write_trade_execution_log", side_effect=_write_into_tempdir
            ):
                execution = run_final_risk_gate(
                    calibration=calibration,
                    market=market,
                    scanner_row=scanner_row,
                    reddit_report={"summary": "Bullish primary-source evidence."},
                    news_report={"summary": "Coverage supportive."},
                    bankroll=10_000.0,
                    risk_calculator=RiskCalculator(),
                )

            log_path = execution["log_path"]
            self.assertTrue(log_path.exists(), msg=f"orchestrator did not write {log_path}")
            self.assertEqual(log_path.parent, store)

            # Sanity check the file shape the orchestrator emitted.
            written = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(written["status"], "open")
            self.assertIsNone(written["final_outcome"])
            self.assertEqual(written["trade_id"], market.market_id)

            # Step 2: flip to settled (round-trip through mark_settled).
            settled_payload = mark_settled(
                log_path,
                final_outcome=True,
                post_settlement_news="Election called for Candidate X by AP.",
            )
            self.assertEqual(settled_payload["status"], "settled")
            self.assertTrue(settled_payload["final_outcome"])
            self.assertIsNotNone(settled_payload["settled_at"])
            # Round-tripped log on disk reflects the same.
            on_disk = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["status"], "settled")
            self.assertTrue(on_disk["final_outcome"])

            # Step 3: PerformanceTracker dispatches BOTH the primary outcome
            # review and the additional data-quality auditor, writing one
            # composite entry into performance_audit.json.
            tracker = PerformanceTracker(
                trade_store_dir=store,
                outcome_review_agent=outcome_agent,
                additional_review_agents={"data_quality": data_quality_auditor},
            )
            tracker_result = tracker.process_settled_trades()
            self.assertEqual(tracker_result["new_reviews"], 1)
            self.assertEqual(tracker_result["total_reviews"], 1)

            audit_path = store / "performance_audit.json"
            self.assertTrue(audit_path.exists())
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertIn("reviews", audit)
            self.assertEqual(len(audit["reviews"]), 1)

            entry = audit["reviews"][0]
            # Both agents wrote into the audit entry.
            self.assertIn("outcome_review", entry)
            self.assertIn("data_quality_review", entry)
            self.assertEqual(
                entry["outcome_review"]["matrix_classification"], "Deserved Success"
            )
            self.assertFalse(entry["data_quality_review"]["data_failure"])
            self.assertEqual(entry["trade_id"], market.market_id)
            self.assertTrue(entry["final_outcome"])

            # Step 4: dashboard reads the audit and prints a summary.
            trades = _extract_trades(audit)
            self.assertEqual(len(trades), 1)

            buf = io.StringIO()
            with redirect_stdout(buf):
                summarize_trades(trades)
            output = buf.getvalue()

            # Summary should reflect the matrix classification and the win.
            self.assertIn("Total Trades: 1", output)
            self.assertIn("Win Rate: 100.00%", output)
            self.assertIn("Deserved Success: 1", output)
            # Process Integrity Score should be 100% (Deserved Success counts
            # as good process under the dashboard's mapping).
            self.assertIn("Process Integrity Score: 100.00%", output)

            # The dashboard's `_extract_trades` lifts outcome_review fields
            # onto the trade dict, so the matrix_classification (and trade_id
            # for traceability) must be present after flattening.
            flattened = trades[0]
            self.assertEqual(
                flattened.get("matrix_classification"), "Deserved Success"
            )
            self.assertEqual(flattened.get("trade_id"), market.market_id)


if __name__ == "__main__":
    unittest.main()
