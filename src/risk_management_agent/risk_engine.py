from __future__ import annotations

from typing import Any, Mapping, Sequence

from config import POLYMARKET_FEE_BPS
from models import Market
from risk_management_agent.models import RiskMetrics


# Tighter than the original 0.01/0.99 — Kelly explodes faster than people
# expect at p > 0.95, so the platform-level extreme-price filter rejects
# anything outside [0.02, 0.98] *before* sizing is even computed.
DEFAULT_PRICE_FLOOR: float = 0.02
DEFAULT_PRICE_CEIL: float = 0.98


def _normalize_category(value: str) -> str:
    return str(value or "").strip().lower()


def _extract_position_category(position: Any) -> str:
    if isinstance(position, Market):
        return _normalize_category(position.category)
    if isinstance(position, Mapping):
        return _normalize_category(position.get("category"))
    return _normalize_category(getattr(position, "category", ""))


def apply_polymarket_fees(
    true_prob: float,
    market_price: float,
    polymarket_fee_bps: int = POLYMARKET_FEE_BPS,
) -> float:
    """Return the fee-adjusted *probability* used downstream by Kelly sizing.

    We model Polymarket fees as a haircut on the realized edge: the trader's
    calibrated probability minus the market price is the gross edge, and the
    platform takes ``polymarket_fee_bps`` of it. The post-fee probability is
    what Kelly should size against, so:

        edge_gross = p_true - p_market
        edge_net   = edge_gross * (1 - fee_frac)
        return p_market + edge_net

    Negative gross edges are passed through unchanged — fees only erode profit
    when the trade *would* have been profitable. (We do not amplify losses on
    a wrong-side bet, because the LP doesn't refund.)
    """
    fee_frac = max(0.0, float(polymarket_fee_bps)) / 10_000.0
    edge_gross = float(true_prob) - float(market_price)
    if edge_gross <= 0.0:
        return float(true_prob)
    edge_net = edge_gross * (1.0 - fee_frac)
    return float(market_price) + edge_net


class RiskCalculator:
    def __init__(
        self,
        *,
        fractional_kelly: float = 0.25,
        liquidity_spread_threshold: float = 0.05,
        liquidity_volume_threshold: float = 10_000.0,
        correlation_penalty_per_position: float = 0.30,
        polymarket_fee_bps: int = POLYMARKET_FEE_BPS,
        price_floor: float = DEFAULT_PRICE_FLOOR,
        price_ceil: float = DEFAULT_PRICE_CEIL,
    ) -> None:
        if not 0.0 <= float(fractional_kelly) <= 1.0:
            raise ValueError("fractional_kelly must be between 0.0 and 1.0")
        if float(liquidity_spread_threshold) < 0.0:
            raise ValueError("liquidity_spread_threshold must be non-negative")
        if float(liquidity_volume_threshold) < 0.0:
            raise ValueError("liquidity_volume_threshold must be non-negative")
        if float(correlation_penalty_per_position) < 0.0:
            raise ValueError("correlation_penalty_per_position must be non-negative")
        if float(polymarket_fee_bps) < 0.0:
            raise ValueError("polymarket_fee_bps must be non-negative")
        if not 0.0 <= float(price_floor) < float(price_ceil) <= 1.0:
            raise ValueError("price_floor must be < price_ceil and both within [0,1]")

        self.fractional_kelly = float(fractional_kelly)
        self.liquidity_spread_threshold = float(liquidity_spread_threshold)
        self.liquidity_volume_threshold = float(liquidity_volume_threshold)
        self.correlation_penalty_per_position = float(correlation_penalty_per_position)
        self.polymarket_fee_bps = int(polymarket_fee_bps)
        self.price_floor = float(price_floor)
        self.price_ceil = float(price_ceil)

    def passes_market_filters(
        self,
        price: float,
        *,
        p_min: float | None = None,
        p_max: float | None = None,
    ) -> bool:
        """Reject markets whose price is in the extreme-price danger zone.

        Kelly sizing explodes faster than people expect at p > 0.95, so we
        gate the entire risk pipeline behind a price band ``[p_min, p_max]``.
        Default band is ``[0.02, 0.98]`` (tighter than the older 0.01/0.99).
        Centralizing this here lets ``orchestrator`` drop its duplicated
        kill-switch check.
        """
        floor = self.price_floor if p_min is None else float(p_min)
        ceil = self.price_ceil if p_max is None else float(p_max)
        try:
            value = float(price)
        except (TypeError, ValueError):
            return False
        return floor <= value <= ceil

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

        # Apply Polymarket fees BEFORE Kelly. The pre-fee edge is the input to
        # ``apply_polymarket_fees``; the post-fee probability then feeds Kelly
        # sizing AND the EV estimate. This is the behaviour change called out
        # in the eng review (P1 #9).
        fee_adjusted_prob = apply_polymarket_fees(
            probability, price, polymarket_fee_bps=self.polymarket_fee_bps
        )
        edge = fee_adjusted_prob - price
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
            # EV uses the *fee-adjusted* edge so it reflects what the trader
            # actually expects to keep, not the gross alpha the model claims.
            expected_value_estimate = position_notional * (fee_adjusted_prob - price) / price

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
