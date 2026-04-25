from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TradeRecord:
    """Normalized representation of a single trade execution payload."""

    trade_id: str
    payload: Dict[str, Any]
    source_file: str


class PerformanceTracker:
    """Tracks historical trade quality and outcome statistics.

    Responsibilities:
    1) Reads ``trade_execution_[ID].json`` files from a JSON store.
    2) Calls an ``OutcomeReviewAgent`` once the trade is marked ``Settled``.
    3) Appends generated reviews to a master ``performance_audit.json`` file.
    4) Computes aggregate Process Health vs Win Rate.
    """

    def __init__(
        self,
        *,
        trade_store_dir: str | Path,
        outcome_review_agent: Any,
        audit_file: str | Path | None = None,
        trade_file_glob: str = "trade_execution_*.json",
        additional_review_agents: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.trade_store_dir = Path(trade_store_dir)
        self.outcome_review_agent = outcome_review_agent
        self.trade_file_glob = trade_file_glob
        self.audit_file = Path(audit_file) if audit_file else self.trade_store_dir / "performance_audit.json"
        # Each entry runs alongside the primary outcome review and lands in the audit
        # entry under "<name>_review" (e.g. "data_quality" → "data_quality_review").
        self.additional_review_agents: Dict[str, Any] = dict(additional_review_agents or {})

    def process_settled_trades(self) -> Dict[str, Any]:
        """Review any new settled trades and update the performance audit."""
        audit = self._load_audit()
        reviewed_keys = {entry.get("trade_key") for entry in audit.get("reviews", []) if entry.get("trade_key")}

        new_reviews: List[Dict[str, Any]] = []
        settled_trades = self._iter_settled_trades()
        for trade in settled_trades:
            trade_key = self._trade_key(trade.trade_id, trade.source_file)
            if trade_key in reviewed_keys:
                continue

            outcome_review = self._call_review_agent(self.outcome_review_agent, trade.payload)
            audit_entry = {
                "trade_id": trade.trade_id,
                "source_file": trade.source_file,
                "trade_key": trade_key,
                "settled_at": self._extract_settlement_time(trade.payload),
                "reviewed_at": self._utc_now_iso(),
                "outcome_review": outcome_review,
                "final_outcome": trade.payload.get("final_outcome"),
            }
            for agent_name, agent in self.additional_review_agents.items():
                audit_entry[f"{agent_name}_review"] = self._call_review_agent(agent, trade.payload)
            new_reviews.append(audit_entry)
            reviewed_keys.add(trade_key)

        if new_reviews:
            audit.setdefault("reviews", []).extend(new_reviews)

        metrics = self.calculate_aggregate_metrics(audit.get("reviews", []))
        audit["aggregates"] = metrics
        audit["last_updated_at"] = self._utc_now_iso()
        self._write_audit(audit)

        return {
            "new_reviews": len(new_reviews),
            "total_reviews": len(audit.get("reviews", [])),
            "aggregates": metrics,
        }

    def calculate_aggregate_metrics(self, reviews: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute Process Health and Win Rate from all stored OutcomeReviews."""
        if not reviews:
            return {
                "review_count": 0,
                "process_health_pct": 0.0,
                "win_rate_pct": 0.0,
            }

        good_process_count = 0
        good_outcome_count = 0
        for review_entry in reviews:
            outcome_review = review_entry.get("outcome_review", {})
            if self._is_good_process(outcome_review):
                good_process_count += 1
            if self._is_good_outcome(outcome_review):
                good_outcome_count += 1

        total = len(reviews)
        return {
            "review_count": total,
            "good_process_count": good_process_count,
            "good_outcome_count": good_outcome_count,
            "process_health_pct": round((good_process_count / total) * 100, 2),
            "win_rate_pct": round((good_outcome_count / total) * 100, 2),
        }

    def _iter_settled_trades(self) -> Iterable[TradeRecord]:
        for file_path in sorted(self.trade_store_dir.glob(self.trade_file_glob)):
            payload = self._load_json_file(file_path)
            if payload is None:
                continue

            for trade_payload in self._expand_trade_payload(payload):
                if not self._is_settled(trade_payload):
                    continue

                trade_id = self._extract_trade_id(trade_payload, fallback_file=file_path)
                yield TradeRecord(
                    trade_id=trade_id,
                    payload=trade_payload,
                    source_file=file_path.name,
                )

    def _call_review_agent(self, agent: Any, trade_payload: Dict[str, Any]) -> Dict[str, Any]:
        method_candidates = (
            "review_trade",
            "analyze_trade",
            "review",
            "run",
            "__call__",
        )

        for method_name in method_candidates:
            method = getattr(agent, method_name, None)
            if not callable(method):
                continue

            if inspect.iscoroutinefunction(method):
                raise RuntimeError(
                    "PerformanceTracker expects a synchronous review-agent method. "
                    "Provide a sync adapter for async agents."
                )

            try:
                review = method(trade_payload=trade_payload)
            except TypeError:
                try:
                    review = method(trade_payload)
                except TypeError:
                    continue

            if isinstance(review, dict):
                return review
            return {"review": review}

        raise RuntimeError(
            "Review agent must implement one of: "
            "review_trade(trade_payload), analyze_trade(trade_payload), review(trade_payload), run(trade_payload), "
            "or be directly callable."
        )

    def _load_audit(self) -> Dict[str, Any]:
        if not self.audit_file.exists():
            return {
                "reviews": [],
                "aggregates": {
                    "review_count": 0,
                    "process_health_pct": 0.0,
                    "win_rate_pct": 0.0,
                },
            }

        existing = self._load_json_file(self.audit_file)
        if not isinstance(existing, dict):
            return {"reviews": []}
        existing.setdefault("reviews", [])
        return existing

    def _write_audit(self, audit_payload: Dict[str, Any]) -> None:
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_file.open("w", encoding="utf-8") as handle:
            json.dump(audit_payload, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_json_file(path: Path) -> Optional[Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _expand_trade_payload(payload: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(payload, dict):
            if isinstance(payload.get("trades"), list):
                for item in payload["trades"]:
                    if isinstance(item, dict):
                        yield item
                return
            yield payload
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item

    @staticmethod
    def _is_settled(trade_payload: Dict[str, Any]) -> bool:
        status_candidates = (
            trade_payload.get("status"),
            trade_payload.get("trade_status"),
            trade_payload.get("state"),
        )
        return any(str(status).strip().lower() == "settled" for status in status_candidates if status is not None)

    @staticmethod
    def _extract_trade_id(trade_payload: Dict[str, Any], *, fallback_file: Path) -> str:
        for key in ("trade_id", "id", "execution_id", "order_id"):
            value = trade_payload.get(key)
            if value is not None and str(value).strip():
                return str(value)

        stem = fallback_file.stem
        if stem.startswith("trade_execution_"):
            return stem.replace("trade_execution_", "", 1)
        return stem

    @staticmethod
    def _extract_settlement_time(trade_payload: Dict[str, Any]) -> Optional[str]:
        for key in ("settled_at", "settlement_time", "resolved_at", "updated_at"):
            value = trade_payload.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return None

    @staticmethod
    def _trade_key(trade_id: str, source_file: str) -> str:
        return f"{source_file}:{trade_id}"

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @staticmethod
    def _is_good_process(outcome_review: Dict[str, Any]) -> bool:
        if not isinstance(outcome_review, dict):
            return False

        if isinstance(outcome_review.get("good_process"), bool):
            return outcome_review["good_process"]

        quadrant_tokens = PerformanceTracker._quadrant_tokens(outcome_review)
        return "good process" in quadrant_tokens

    @staticmethod
    def _is_good_outcome(outcome_review: Dict[str, Any]) -> bool:
        if not isinstance(outcome_review, dict):
            return False

        if isinstance(outcome_review.get("good_outcome"), bool):
            return outcome_review["good_outcome"]

        quadrant_tokens = PerformanceTracker._quadrant_tokens(outcome_review)
        return "good outcome" in quadrant_tokens

    @staticmethod
    def _quadrant_tokens(outcome_review: Dict[str, Any]) -> Tuple[str, ...]:
        candidate_strings = (
            outcome_review.get("quadrant"),
            outcome_review.get("classification"),
            outcome_review.get("label"),
        )
        tokens: List[str] = []
        for value in candidate_strings:
            if value is None:
                continue
            lowered = str(value).strip().lower()
            if lowered:
                tokens.append(lowered)
        return tuple(tokens)
