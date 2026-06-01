from datetime import datetime, timezone

from models import Market


def test_market_computes_spread_and_days_to_resolution():
    market = Market(
        market_id="mkt-1",
        title="Will event happen?",
        category="politics",
        implied_prob=0.42,
        bid_price=0.41,
        ask_price=0.45,
        volume_24h=15000.0,
        price_history={"1h": 0.01, "6h": -0.02, "24h": 0.05},
        open_interest=22000.0,
        resolution_date=datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc),
        rules_text="Resolves Yes if the named event occurs before the deadline.",
    )

    market.refresh_derived_fields(now=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc))

    assert market.spread == 0.04
    assert market.days_to_resolution == 5.0


def test_market_normalizes_price_history_and_resolution_date_string():
    market = Market(
        market_id="mkt-2",
        title="Will another event happen?",
        category="macro",
        implied_prob=0.55,
        bid_price=0.54,
        ask_price=0.56,
        volume_24h=9000.0,
        price_history={"1h": "0.01", "24h": 0.03},
        open_interest=12000.0,
        resolution_date="2026-04-22T00:00:00Z",
        rules_text="Resolves Yes based on the listed source.",
    )

    assert market.price_history == {"1h": 0.01, "6h": 0.0, "24h": 0.03}
    assert market.resolution_date.tzinfo == timezone.utc
    assert market.spread == 0.02
