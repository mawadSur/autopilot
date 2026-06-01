"""Tests for ``calibration_agent.build_dataset``."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from calibration_agent.build_dataset import (
    OUTPUT_COLUMNS,
    SKIP_BACKFILL_EXCLUDED,
    SKIP_INVALID_FEATURES,
    SKIP_MISSING_FEATURE_COLUMNS,
    SKIP_NO_FEATURES_WINDOW,
    SKIP_NO_MARKET_OUTCOME,
    SKIP_NOT_SETTLED,
    assemble_dataset,
)
from calibration_agent.ml_service import ALL_FEATURE_COLUMNS


def _full_features_window(*, captured: str = "2026-04-01T00:00:00+00:00") -> Dict[str, Any]:
    """A flat features_window dict with all required numeric columns."""

    base = {col: 0.0 for col in ALL_FEATURE_COLUMNS}
    # Diversify a few values so test assertions can spot mismatches.
    base["implied_prob"] = 0.55
    base["spread"] = 0.02
    base["volume_24h"] = 12345.0
    base["news_sentiment_score"] = 12.5
    base["captured_at_utc"] = captured
    return base


_SENTINEL_NO_SOURCE = object()


def _trade_payload(
    *,
    trade_id: str,
    status: str = "settled",
    market_outcome: Optional[bool] = True,
    final_outcome: Optional[bool] = True,
    features_window: Optional[Dict[str, Any]] = None,
    settled_at: Optional[str] = "2026-04-02T00:00:00+00:00",
    source: Any = "orchestrator",
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "trade_id": trade_id,
        "event_id": trade_id,
        "status": status,
        "settled_at": settled_at,
        "final_outcome": final_outcome,
        "market_outcome": market_outcome,
        "features_window": features_window,
    }
    # Mirror the canonical schema: source/notes are present on every
    # post-Pass-1 payload; tests that want to exercise the back-compat path
    # for pre-schema artifacts can pass ``source=_SENTINEL_NO_SOURCE``.
    if source is not _SENTINEL_NO_SOURCE:
        payload["source"] = source
        payload["notes"] = notes
    return payload


def _write_payload(directory: Path, name: str, payload: Dict[str, Any]) -> Path:
    file_path = directory / f"trade_execution_{name}.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    return file_path


def _capture_stderr(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        result = func(*args, **kwargs)
    return result, buffer.getvalue()


class AssembleDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

        # 1. Settled + labeled + full features (qualifies).
        _write_payload(
            self.tmp_path,
            "qualifies",
            _trade_payload(
                trade_id="qualifies",
                features_window=_full_features_window(),
                market_outcome=True,
            ),
        )
        # 2. Settled but market_outcome is None.
        _write_payload(
            self.tmp_path,
            "no_outcome",
            _trade_payload(
                trade_id="no_outcome",
                features_window=_full_features_window(),
                market_outcome=None,
            ),
        )
        # 3. Settled but features_window is None.
        _write_payload(
            self.tmp_path,
            "no_features",
            _trade_payload(
                trade_id="no_features",
                features_window=None,
                market_outcome=True,
            ),
        )
        # 4. Open status (skipped by default).
        _write_payload(
            self.tmp_path,
            "open_trade",
            _trade_payload(
                trade_id="open_trade",
                status="open",
                features_window=_full_features_window(),
                market_outcome=None,
                final_outcome=None,
                settled_at=None,
            ),
        )
        # 5. Settled but missing one feature column.
        broken_features = _full_features_window()
        del broken_features["news_sentiment_score"]
        _write_payload(
            self.tmp_path,
            "missing_col",
            _trade_payload(
                trade_id="missing_col",
                features_window=broken_features,
                market_outcome=True,
            ),
        )

    def test_returns_single_qualifying_row(self) -> None:
        df, stderr = _capture_stderr(assemble_dataset, self.tmp_path)

        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), list(OUTPUT_COLUMNS))
        self.assertIn("source", df.columns)
        row = df.iloc[0]
        self.assertEqual(row["trade_id"], "qualifies")
        self.assertEqual(row["source"], "orchestrator")
        self.assertEqual(int(row["market_outcome"]), 1)
        self.assertEqual(int(row["final_outcome"]), 1)
        self.assertAlmostEqual(float(row["implied_prob"]), 0.55)
        self.assertAlmostEqual(float(row["news_sentiment_score"]), 12.5)
        self.assertIn("Assembled 1 rows from 5 files", stderr)

    def test_each_skip_reason_counted(self) -> None:
        _, stderr = _capture_stderr(assemble_dataset, self.tmp_path)

        for reason in (
            SKIP_NOT_SETTLED,
            SKIP_NO_FEATURES_WINDOW,
            SKIP_MISSING_FEATURE_COLUMNS,
            SKIP_NO_MARKET_OUTCOME,
        ):
            self.assertIn(f"{reason}=1", stderr, msg=f"missing skip reason {reason}: {stderr}")
        self.assertIn("skipped 4", stderr)

    def test_writes_parquet_when_extension_is_parquet(self) -> None:
        output = self.tmp_path / "out.parquet"
        _capture_stderr(
            assemble_dataset, self.tmp_path, output_path=output
        )

        self.assertTrue(output.exists())
        loaded = pd.read_parquet(output)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(list(loaded.columns), list(OUTPUT_COLUMNS))
        self.assertEqual(loaded.iloc[0]["trade_id"], "qualifies")

    def test_writes_csv_when_extension_is_csv(self) -> None:
        output = self.tmp_path / "out.csv"
        _capture_stderr(
            assemble_dataset, self.tmp_path, output_path=output
        )

        self.assertTrue(output.exists())
        loaded = pd.read_csv(output)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(list(loaded.columns), list(OUTPUT_COLUMNS))
        self.assertEqual(loaded.iloc[0]["trade_id"], "qualifies")
        self.assertEqual(int(loaded.iloc[0]["market_outcome"]), 1)

    def test_include_unlabeled_keeps_open_trade(self) -> None:
        df, stderr = _capture_stderr(
            assemble_dataset, self.tmp_path, include_unlabeled=True
        )

        # Open trade is *still* skipped because market_outcome is None — but the
        # skip reason must shift from not_settled to no_market_outcome, so the
        # not_settled bucket should now be zero.
        self.assertNotIn(f"{SKIP_NOT_SETTLED}=", stderr)
        self.assertIn(f"{SKIP_NO_MARKET_OUTCOME}=2", stderr)
        # Still only one fully-qualified row.
        self.assertEqual(len(df), 1)

    def test_empty_dir_returns_empty_dataframe(self) -> None:
        with tempfile.TemporaryDirectory() as empty_dir:
            df, stderr = _capture_stderr(assemble_dataset, empty_dir)

        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), list(OUTPUT_COLUMNS))
        self.assertIn("Assembled 0 rows from 0 files", stderr)

    def test_missing_dir_returns_empty_dataframe(self) -> None:
        df, _ = _capture_stderr(
            assemble_dataset, self.tmp_path / "does_not_exist"
        )
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), list(OUTPUT_COLUMNS))


class SourceFilteringTests(unittest.TestCase):
    """Tests for the ``source`` filter / back-compat behavior."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def test_default_excludes_backfill_rows(self) -> None:
        _write_payload(
            self.tmp_path,
            "orchestrator_row",
            _trade_payload(
                trade_id="orchestrator_row",
                features_window=_full_features_window(),
                source="orchestrator",
            ),
        )
        _write_payload(
            self.tmp_path,
            "backfill_row",
            _trade_payload(
                trade_id="backfill_row",
                features_window=_full_features_window(),
                source="backfill",
                notes="backfilled — degraded fidelity",
            ),
        )

        df, stderr = _capture_stderr(assemble_dataset, self.tmp_path)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["trade_id"], "orchestrator_row")
        self.assertEqual(df.iloc[0]["source"], "orchestrator")
        self.assertIn(f"{SKIP_BACKFILL_EXCLUDED}=1", stderr)

    def test_include_backfill_keeps_them(self) -> None:
        _write_payload(
            self.tmp_path,
            "orchestrator_row",
            _trade_payload(
                trade_id="orchestrator_row",
                features_window=_full_features_window(),
                source="orchestrator",
            ),
        )
        _write_payload(
            self.tmp_path,
            "backfill_row",
            _trade_payload(
                trade_id="backfill_row",
                features_window=_full_features_window(),
                source="backfill",
                notes="backfilled — degraded fidelity",
            ),
        )

        df, stderr = _capture_stderr(
            assemble_dataset, self.tmp_path, include_backfill=True
        )

        self.assertEqual(len(df), 2)
        sources = sorted(df["source"].tolist())
        self.assertEqual(sources, ["backfill", "orchestrator"])
        self.assertNotIn(f"{SKIP_BACKFILL_EXCLUDED}=", stderr)

    def test_shadow_source_is_full_fidelity_by_default(self) -> None:
        _write_payload(
            self.tmp_path,
            "shadow_row",
            _trade_payload(
                trade_id="shadow_row",
                features_window=_full_features_window(),
                source="shadow",
            ),
        )

        df, stderr = _capture_stderr(assemble_dataset, self.tmp_path)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["trade_id"], "shadow_row")
        self.assertEqual(df.iloc[0]["source"], "shadow")
        self.assertNotIn(f"{SKIP_BACKFILL_EXCLUDED}=", stderr)

    def test_missing_source_treated_as_orchestrator(self) -> None:
        _write_payload(
            self.tmp_path,
            "no_source_row",
            _trade_payload(
                trade_id="no_source_row",
                features_window=_full_features_window(),
                source=_SENTINEL_NO_SOURCE,
            ),
        )

        with self.assertLogs(
            "calibration_agent.build_dataset", level="WARNING"
        ) as captured:
            df, _ = _capture_stderr(assemble_dataset, self.tmp_path)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["trade_id"], "no_source_row")
        self.assertEqual(df.iloc[0]["source"], "orchestrator")
        joined = "\n".join(captured.output)
        self.assertIn("without explicit 'source'", joined)
        self.assertIn("treating as 'orchestrator'", joined)
        self.assertIn("1 trade log", joined)


if __name__ == "__main__":
    unittest.main()
