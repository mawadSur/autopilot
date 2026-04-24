from __future__ import annotations

from typing import Any, Mapping, Sequence

from models import Market
from risk_management_agent.models import RiskMetrics


def _normalize_category(value: str) -> str:
    return str(value or "").strip().lower()


def _extract_position_category(position: Any) -> str:
    if isinstance(position, Market):
        return _normalize_category(position.category)
    if isinstance(position, Mapping):
        return _normalize_category(position.get("category"))
    return _normalize_category(getattr(position, "category", ""))


class RiskCalculator:
    def __init__(
        self,
        *,
        fractional_kelly: float = 0.25,
        liquidity_spread_threshold: float = 0.05,
        liquidity_volume_threshold: float = 10_000.0,
        correlation_penalty_per_position: float = 0.30,
    ) -> None:
        if not 0.0 <= float(fractional_kelly) <= 1.0:
            raise ValueError("fractional_kelly must be between 0.0 and 1.0")
        if float(liquidity_spread_threshold) < 0.0:
            raise ValueError("liquidity_spread_threshold must be non-negative")
        if float(liquidity_volume_threshold) < 0.0:
            raise ValueError("liquidity_volume_threshold must be non-negative")
        if float(correlation_penalty_per_position) < 0.0:
            raise ValueError("correlation_penalty_per_position must be non-negative")

        self.fractional_kelly = float(fractional_kelly)
        self.liquidity_spread_threshold = float(liquidity_spread_threshold)
        self.liquidity_volume_threshold = float(liquidity_volume_threshold)
        self.correlation_penalty_per_position = float(correlation_penalty_per_position)

    def calculate_base_metrics(
        self,
        *,
        market: Market,
        calibrated_true_prob: float,
        bankroll: float,
        market_price: float | None = None,
        existing_open_positions: Sequence[Any] | None = None,
    ) -> RiskMetrics:
        if not isinstance(market, Market):
            raise TypeError("market must be a Market instance")

        market.refresh_derived_fields()
        probability = float(calibrated_true_prob)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("calibrated_true_prob must be between 0.0 and 1.0")

        bankroll_value = float(bankroll)
        if bankroll_value < 0.0:
            raise ValueError("bankroll must be non-negative")

        price = float(market.implied_prob if market_price is None else market_price)
        if not 0.0 < price < 1.0:
            raise ValueError("market_price must be between 0.0 and 1.0 exclusive")

        edge = probability - price
        raw_kelly_fraction = max(0.0, edge / (1.0 - price))
        raw_kelly_size_pct = raw_kelly_fraction * 100.0
        fractional_kelly_size_pct = raw_kelly_size_pct * self.fractional_kelly

        liquidity_penalty_applied = (
            float(market.spread) > self.liquidity_spread_threshold
            or float(market.volume_24h) < self.liquidity_volume_threshold
        )
        liquidity_penalty_multiplier = 0.5 if liquidity_penalty_applied else 1.0

        same_category = _normalize_category(market.category)
        open_positions = list(existing_open_positions or [])
        same_category_open_positions = sum(
            1 for position in open_positions if _extract_position_category(position) == same_category
        )
        correlation_penalty_multiplier = max(
            0.0,
            1.0 - self.correlation_penalty_per_position * same_category_open_positions,
        )
        correlation_penalty_applied = same_category_open_positions > 0

        adjusted_position_size_pct = (
            fractional_kelly_size_pct * liquidity_penalty_multiplier * correlation_penalty_multiplier
        )
        position_notional = bankroll_value * adjusted_position_size_pct / 100.0
        max_loss_if_wrong = position_notional
        expected_value_estimate = 0.0
        if position_notional > 0.0:
            expected_value_estimate = position_notional * (probability - price) / price

        return RiskMetrics(
            market_price=price,
            calibrated_true_prob=probability,
            bankroll=bankroll_value,
            raw_kelly_size_pct=raw_kelly_size_pct,
            fractional_kelly_size_pct=fractional_kelly_size_pct,
            liquidity_penalty_multiplier=liquidity_penalty_multiplier,
            correlation_penalty_multiplier=correlation_penalty_multiplier,
            same_category_open_positions=same_category_open_positions,
            liquidity_penalty_applied=liquidity_penalty_applied,
            correlation_penalty_applied=correlation_penalty_applied,
            adjusted_position_size_pct=adjusted_position_size_pct,
            max_loss_if_wrong=max_loss_if_wrong,
            expected_value_estimate=expected_value_estimate,
        )
