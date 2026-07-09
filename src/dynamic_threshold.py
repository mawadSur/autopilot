"""Cost-aware dynamic entry threshold (volatility + liquidity).

The entry gate ``P(long) >= thr_long`` used a static (or per-regime) threshold.
This raises the required model conviction when the *expected execution cost* of
a trade is high, and relaxes it when conditions are cheap — keeping the bar for
entry tied to what a fill actually costs, which is the whole post-fee-expectancy
theme.

Two drivers, both read from the per-bar feature row (stateless, like
:mod:`regime_router`):

* **Volatility** — ``atrp_14`` (ATR / price). High vol means larger adverse
  slippage on a market fill (the simulator itself charges ``0.5 * ATR``), so a
  high-vol bar should demand a higher threshold. Measured as a *relative*
  deviation from a reference vol so it is unit-free and symbol-portable.
* **Liquidity** — spread + book thinness / adverse imbalance. A wide spread or
  thin book raises the round-trip cost, so it raises the threshold.

    thr = clip(base_thr + s_vol * vol_signal(row) + s_liq * liq_signal(row),
               thr_min, thr_max)

Both signals are 0 at "reference" conditions, so at reference the threshold is
exactly ``base_thr`` (identity) — the dynamic layer only ever *deviates* from
the base a regime router (or config) already chose.

DATA REALITY: the liquidity columns (``spread_pct``, ``*_depth_*``,
``l2_imbalance_*``) are identically 0 in the current datasets (the L2 book was
never backfilled), so :func:`liq_signal` returns 0 and only the volatility term
moves. Once the book is populated the liquidity term activates with no code
change. The volatility term is fully live today (``atrp_14`` is populated).

Sign note: ``s_vol > 0`` (default) is the cost/noise-aware direction — trade
*less* when vol is high. The opposite view (high vol => bigger moves clear fees
=> trade *more*) is a sign flip; the branch audit sweeps it so the direction is
decided by post-fee expectancy, not assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

__all__ = ["DynamicThresholdConfig", "DynamicThreshold", "vol_signal", "liq_signal"]


def _get(row: Any, key: str, default: float = float("nan")) -> float:
    try:
        val = row.get(key, default) if hasattr(row, "get") else row[key]
    except (KeyError, IndexError, TypeError):
        return float(default)
    try:
        out = float(val)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


@dataclass
class DynamicThresholdConfig:
    ref_atrp: float = 0.0008        # reference volatility (ATR/price fraction)
    s_vol: float = 0.06             # thr bump per +100% relative volatility
    vol_clip: float = 3.0           # clip vol_signal to [-1, vol_clip]
    ref_spread_bps: float = 2.0     # reference spread for liquidity normalization
    s_liq: float = 0.04             # thr bump per unit liquidity stress
    liq_clip: float = 5.0           # clip liq_signal to [0, liq_clip]
    thr_min: float = 0.30
    thr_max: float = 0.90


def vol_signal(row: Any, *, ref_atrp: float, vol_clip: float) -> float:
    """Relative volatility deviation: 0 at reference vol, >0 when elevated.

    ``(atrp_14 / ref_atrp) - 1`` clipped to ``[-1, vol_clip]``. Returns 0 on
    missing/invalid data (no adjustment) rather than a spurious push."""
    atrp = _get(row, "atrp_14")
    if not np.isfinite(atrp) or ref_atrp <= 0:
        return 0.0
    return float(np.clip(atrp / float(ref_atrp) - 1.0, -1.0, float(vol_clip)))


def liq_signal(row: Any, *, ref_spread_bps: float, liq_clip: float) -> float:
    """Liquidity stress: 0 when cheap/absent, >0 when spread wide / book thin.

    Only spread and top-of-book depth ratio are used; both are 0 in the current
    datasets, so this returns 0 (inert) until the L2 book is backfilled. It never
    goes negative — good liquidity relaxes only back to the base threshold, it
    does not *lower* it below base (that lever is the model's job, not the book)."""
    stress = 0.0
    spread_bps = _get(row, "spread_pct", 0.0) * 1e4
    if np.isfinite(spread_bps) and spread_bps > 0.0 and ref_spread_bps > 0.0:
        stress += float(np.clip(spread_bps / float(ref_spread_bps) - 1.0, 0.0, float(liq_clip)))
    # Book thinness: depth_ratio_5 < 1 means asks thinner than bids for a buy.
    depth_ratio = _get(row, "depth_ratio_5", 0.0)
    if np.isfinite(depth_ratio) and depth_ratio > 0.0:
        stress += float(np.clip(1.0 / depth_ratio - 1.0, 0.0, float(liq_clip)))
    return stress


@dataclass
class DynamicThreshold:
    cfg: DynamicThresholdConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.cfg is None:
            self.cfg = DynamicThresholdConfig()

    def adjust(self, base_thr: float, row: Any) -> float:
        """Return the cost-adjusted long threshold for one feature row."""
        c = self.cfg
        vs = vol_signal(row, ref_atrp=c.ref_atrp, vol_clip=c.vol_clip)
        ls = liq_signal(row, ref_spread_bps=c.ref_spread_bps, liq_clip=c.liq_clip)
        thr = float(base_thr) + c.s_vol * vs + c.s_liq * ls
        return float(np.clip(thr, c.thr_min, c.thr_max))

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "DynamicThreshold":
        base = DynamicThresholdConfig()
        if not cfg:
            return cls(base)
        fields = DynamicThresholdConfig.__dataclass_fields__
        kwargs = {k: float(cfg[k]) for k in fields if k in cfg}
        return cls(DynamicThresholdConfig(**{**base.__dict__, **kwargs}))
