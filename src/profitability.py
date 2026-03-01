from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_pct_change(arr: np.ndarray) -> np.ndarray:
    if arr.size < 2:
        return np.zeros(0, dtype=float)
    prev = arr[:-1]
    curr = arr[1:]
    denom = np.where(prev == 0, 1e-12, prev)
    return (curr - prev) / denom


def compute_profitability_metrics(
    report: Dict,
    equity_curve: pd.DataFrame,
    trade_log: List[dict],
    *,
    bars_per_year: int = 525_600,
) -> Dict:
    portfolio = (report or {}).get("portfolio", {}) or {}
    start_capital = float(portfolio.get("start_capital", 0.0) or 0.0)
    end_equity = float(portfolio.get("end_equity", 0.0) or 0.0)
    net_profit = end_equity - start_capital
    net_profit_pct = (end_equity / max(1e-12, start_capital) - 1.0) * 100.0 if start_capital else 0.0
    max_dd_pct = float(portfolio.get("max_drawdown", 0.0) or 0.0) * 100.0

    trade_returns = np.array(
        [t.get("ret") for t in trade_log if t.get("action") == "exit" and np.isfinite(t.get("ret", np.nan))],
        dtype=float,
    )
    trade_pnls = np.array(
        [t.get("pnl") for t in trade_log if t.get("action") == "exit" and np.isfinite(t.get("pnl", np.nan))],
        dtype=float,
    )
    gain_sum = float(np.sum(trade_pnls[trade_pnls > 0])) if trade_pnls.size else 0.0
    loss_sum = float(np.sum(np.abs(trade_pnls[trade_pnls < 0]))) if trade_pnls.size else 0.0
    if loss_sum > 0:
        profit_factor = gain_sum / loss_sum
    else:
        profit_factor = float("inf") if gain_sum > 0 else 0.0

    expectancy = float(np.mean(trade_returns)) if trade_returns.size else 0.0
    if trade_returns.size > 1:
        std_r = float(np.std(trade_returns, ddof=1))
        sqn = float((expectancy / max(1e-12, std_r)) * np.sqrt(trade_returns.size)) if std_r > 0 else 0.0
    else:
        sqn = 0.0

    if equity_curve is not None and "equity" in equity_curve.columns and len(equity_curve) > 1:
        eq = equity_curve["equity"].to_numpy(dtype=float)
        rets = _safe_pct_change(eq)
        if rets.size > 1:
            mean_r = float(np.mean(rets))
            std_r = float(np.std(rets, ddof=1))
            sharpe = float((mean_r / max(1e-12, std_r)) * np.sqrt(bars_per_year)) if std_r > 0 else 0.0
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    if max_dd_pct > 0:
        recovery = net_profit_pct / max_dd_pct
    else:
        recovery = float("inf") if net_profit_pct > 0 else 0.0

    return {
        "net_profit": float(net_profit),
        "net_profit_pct": float(net_profit_pct),
        "profit_factor": float(profit_factor),
        "sharpe_annualized": float(sharpe),
        "max_drawdown_pct": float(max_dd_pct),
        "expectancy": float(expectancy),
        "sqn": float(sqn),
        "recovery_factor": float(recovery),
        "trade_returns": trade_returns,
    }


def monte_carlo_permutation_test(
    trade_returns: Iterable[float],
    *,
    runs: int = 500,
    seed: int = 42,
) -> Dict:
    arr = np.array(list(trade_returns), dtype=float)
    if arr.size < 2:
        return {"runs": 0, "p_value": 1.0, "original_net_profit": 0.0, "p05": 0.0}

    rng = np.random.default_rng(seed)
    perm_net = np.zeros(runs, dtype=float)
    for i in range(runs):
        perm = rng.permutation(arr)
        perm_net[i] = float((np.prod(1.0 + perm) - 1.0))

    p05 = float(np.percentile(perm_net, 5))
    p_value = float(np.mean(perm_net >= float((np.prod(1.0 + arr) - 1.0))))
    return {
        "runs": int(runs),
        "p_value": p_value,
        "p05": p05,
        "perm_mean": float(np.mean(perm_net)),
        "perm_std": float(np.std(perm_net, ddof=1)) if runs > 1 else 0.0,
    }


def kelly_fraction(returns: np.ndarray, *, cap: float = 2.0) -> float:
    if returns.size < 2:
        return 0.0
    mean_r = float(np.mean(returns))
    var_r = float(np.var(returns, ddof=1))
    if var_r <= 0:
        return 0.0
    k = mean_r / var_r
    return float(np.clip(k, 0.0, cap))


def vol_target_leverage(equity_curve: pd.DataFrame, *, target_vol: float = 0.20) -> float:
    if equity_curve is None or "equity" not in equity_curve.columns or len(equity_curve) < 2:
        return 1.0
    eq = equity_curve["equity"].to_numpy(dtype=float)
    rets = _safe_pct_change(eq)
    if rets.size < 2:
        return 1.0
    vol = float(np.std(rets, ddof=1) * np.sqrt(525_600))
    if vol <= 0:
        return 1.0
    return float(np.clip(target_vol / vol, 0.1, 3.0))


def profitability_report(metrics: Dict, mc: Dict) -> str:
    net_profit = metrics.get("net_profit", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    sharpe = metrics.get("sharpe_annualized", 0.0)
    max_dd = metrics.get("max_drawdown_pct", 0.0)
    exp = metrics.get("expectancy", 0.0)
    sqn = metrics.get("sqn", 0.0)
    p05 = mc.get("p05", 0.0)
    robust = "ROBUST" if p05 > 0 else "NOT ROBUST"
    return (
        "=== PROFITABILITY REPORT ===\n"
        f"Net Profit: ${net_profit:,.0f}   |   Profit Factor: {pf:.2f}   |   Sharpe: {sharpe:.2f}\n"
        f"Max DD: {max_dd:.1f}%   |   Expectancy: {exp:.2f}R   |   SQN: {sqn:.2f}\n"
        f"Monte-Carlo 5th percentile profit: ${p05*100:,.0f}  →  {robust}"
    )
