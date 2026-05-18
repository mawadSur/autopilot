"""Cost-aware threshold re-selection for vol-normalized crypto models.

Re-picks the operating threshold for an XGBoost model bundle by maximizing
*expected net P&L per trade* instead of raw winrate. Reads the existing
sweep that ``src/crypto_training/train_xgboost.py`` (or sibling sweep
scripts under ``scripts/``) wrote into ``meta.json`` and does NOT retrain
anything.

Why
---

Lifting the threshold raises win-rate but shrinks ``n``, and a +1pp
winrate at thr=0.60 can still net less per-trade than a slightly worse
winrate at thr=0.55 once you pay slippage + commissions on every
round-trip. The script computes:

    net_pnl_per_trade
        = wr * avg_win
        - (1 - wr) * abs(avg_loss)
        - slippage_per_round_trip
        - commission_per_round_trip

where slippage and commission are bps-per-side * notional * 2 (entry +
exit). The picked threshold is the one with the highest net_pnl among
candidates passing the ``--min-n`` floor (statistical-significance
guardrail; default 100). Ties are broken by Sharpe (higher is better),
then by lower threshold (more trades = more revenue at parity).

Sharpe approximation
--------------------

If the sweep entry carries a ``sharpe`` field (newer ``threshold_metrics``
dict written by ``train_xgboost.py``), we surface it as-is. Otherwise we
approximate per-trade Sharpe analytically from a Bernoulli model::

    mean    = wr * avg_win + (1 - wr) * avg_loss            # (avg_loss < 0)
    var     = wr * (avg_win - mean)^2 + (1 - wr) * (avg_loss - mean)^2
    sharpe  = (mean - cost_per_trade) / sqrt(var)           # 0 if var == 0

This is a per-trade Sharpe (no annualization), useful only for
*relative* comparison across thresholds on the same model. Costs are
folded into the numerator so the metric matches what net_pnl actually
captures.

Meta shapes supported
---------------------

Two on-disk shapes are tolerated (the trainer's output drifted across
sprints):

1. **Rich dict** (``meta["threshold_metrics"]``, written by
   ``train_xgboost.py``): keys are stringified thresholds (e.g.
   ``"0.5000"``), values are dicts with ``n_trades, win_rate, avg_win,
   avg_loss, sharpe, max_drawdown``. Used directly.

2. **Test precision curve** (``meta["test_precision_curve"]``, written
   by the sibling sweep scripts under ``scripts/``): list of dicts
   ``{thr, n_trades, win_rate}``. Lacks per-trade payoff magnitudes, so
   we fall back to ``--target-move-bps`` (symmetric assumption:
   ``avg_win = -avg_loss = target_move_bps * notional / 10_000``). The
   table prints ``(symmetric-payoff approx)`` when this path is taken.

Exit codes
----------

* ``0`` — at least one threshold passed the n-floor and a winner was
  picked (even if net_pnl is negative, we still report the least-bad).
* ``1`` — no sweep data on disk (both fields empty/missing).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Resolve canonical model paths for the v2 crypto bundles. Relative to
# the repo root so the script works regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
SYMBOL_META_PATHS: Dict[str, Path] = {
    "eth": _REPO_ROOT / "model_crypto" / "eth_usd_voln_v2_blend09_alt" / "meta.json",
    "btc": _REPO_ROOT / "model_crypto" / "btc_usd_voln_v2" / "meta.json",
    "sol": _REPO_ROOT / "model_crypto" / "sol_usd_voln_v2" / "meta.json",
}


@dataclass
class Candidate:
    """One row of the sweep table after cost adjustment."""

    thr: float
    n: int
    wr: float
    avg_win: float        # dollars per winning trade (positive)
    avg_loss: float       # dollars per losing trade  (negative)
    net_pnl: float        # expected dollars per trade after costs
    sharpe: float         # per-trade Sharpe (relative comparison only)
    approx: bool          # True if avg_win/avg_loss came from --target-move-bps

    def as_table_row(self) -> str:
        win_str = f"{self.avg_win:.4f}"
        loss_str = f"{self.avg_loss:.4f}"
        return (
            f"| {self.thr:.4f} | {self.n:5d} | {self.wr:.4f} "
            f"| {win_str:>8} | {loss_str:>9} "
            f"| {self.net_pnl:+.4f} | {self.sharpe:+.4f} |"
        )


def _resolve_meta_path(args: argparse.Namespace) -> Path:
    if args.meta:
        return Path(args.meta).expanduser().resolve()
    sym = (args.symbol or "").lower()
    if sym not in SYMBOL_META_PATHS:
        raise SystemExit(
            f"--symbol {args.symbol!r} not in {sorted(SYMBOL_META_PATHS)}"
        )
    return SYMBOL_META_PATHS[sym]


def _parse_threshold_metrics_dict(
    tm: Dict[str, Dict[str, Any]],
    *,
    notional: float,
) -> List[Tuple[float, int, float, float, float, Optional[float], bool]]:
    """Parse rich dict form: keyed by str(thr), entries carry payoffs.

    ``avg_win`` / ``avg_loss`` in the simulator are in *position-size
    units* (the trainer uses ``position_size=1.0``), so we scale them
    to dollars by ``notional``.

    Returns a list of tuples ``(thr, n, wr, avg_win$, avg_loss$, sharpe_or_None, approx=False)``.
    """
    out: List[Tuple[float, int, float, float, float, Optional[float], bool]] = []
    for key, entry in tm.items():
        try:
            thr = float(key)
        except (TypeError, ValueError):
            # Numeric thr stored as float keys after json round-trip
            # would be impossible (json keys are strings), but defend
            # anyway and skip malformed keys.
            continue
        n = int(entry.get("n_trades", 0) or 0)
        wr = float(entry.get("win_rate", 0.0) or 0.0)
        # The simulator returns avg_win/avg_loss as per-trade pnl in
        # position-size units; scale to dollars via notional.
        raw_win = float(entry.get("avg_win", 0.0) or 0.0)
        raw_loss = float(entry.get("avg_loss", 0.0) or 0.0)
        avg_win_dollars = raw_win * notional
        avg_loss_dollars = raw_loss * notional
        sharpe_field = entry.get("sharpe")
        sharpe = float(sharpe_field) if sharpe_field is not None else None
        out.append((thr, n, wr, avg_win_dollars, avg_loss_dollars, sharpe, False))
    return out


def _parse_test_precision_curve(
    curve: List[Dict[str, Any]],
    *,
    notional: float,
    target_move_bps: float,
) -> List[Tuple[float, int, float, float, float, Optional[float], bool]]:
    """Parse list form: thr/n_trades/win_rate only, no per-trade payoffs.

    Assumes symmetric payoff at ``target_move_bps`` (the labeling
    target move on the vol-normalized dataset). avg_win = -avg_loss =
    target_move_bps * notional / 10_000.
    """
    out: List[Tuple[float, int, float, float, float, Optional[float], bool]] = []
    move_dollars = (target_move_bps / 10_000.0) * notional
    for entry in curve:
        thr = float(entry.get("thr", 0.0))
        n = int(entry.get("n_trades", 0) or 0)
        wr = float(entry.get("win_rate", 0.0) or 0.0)
        avg_win = move_dollars
        avg_loss = -move_dollars
        out.append((thr, n, wr, avg_win, avg_loss, None, True))
    return out


def _extract_candidates(
    meta: Dict[str, Any],
    *,
    notional: float,
    target_move_bps: float,
) -> Tuple[List[Tuple[float, int, float, float, float, Optional[float], bool]], str]:
    """Return ``(rows, source_label)`` from whichever sweep field is populated.

    Prefers the rich ``threshold_metrics`` dict; falls back to
    ``test_precision_curve``. Returns empty list if both are missing.
    """
    tm = meta.get("threshold_metrics")
    if isinstance(tm, dict) and tm:
        return (
            _parse_threshold_metrics_dict(tm, notional=notional),
            "threshold_metrics (rich)",
        )
    curve = meta.get("test_precision_curve")
    if isinstance(curve, list) and curve:
        return (
            _parse_test_precision_curve(
                curve, notional=notional, target_move_bps=target_move_bps
            ),
            f"test_precision_curve (symmetric-payoff approx @ {target_move_bps:g} bps)",
        )
    return [], "none"


def _approx_per_trade_sharpe(
    wr: float, avg_win: float, avg_loss: float, cost_per_trade: float
) -> float:
    """Per-trade Sharpe under Bernoulli outcomes.

    See module docstring for the formula. Returns 0.0 when variance
    rounds to 0 (degenerate single-outcome cases) so the comparison
    function never sees NaN/Inf.
    """
    mean = wr * avg_win + (1.0 - wr) * avg_loss
    var = (
        wr * (avg_win - mean) ** 2
        + (1.0 - wr) * (avg_loss - mean) ** 2
    )
    if var <= 0.0:
        return 0.0
    std = math.sqrt(var)
    return (mean - cost_per_trade) / std


def compute_candidates(
    rows: List[Tuple[float, int, float, float, float, Optional[float], bool]],
    *,
    slippage_bps: float,
    commission_bps: float,
    notional: float,
    min_n: int,
) -> Tuple[List[Candidate], List[Candidate]]:
    """Return ``(all_rows_sorted_by_thr, eligible_rows_sorted_by_thr)``.

    ``eligible`` is the subset where ``n >= min_n``; ``all_rows`` is
    every candidate (useful for the printed table so the operator can
    see what got filtered).
    """
    # Round-trip costs: slippage AND commission are bps-per-side,
    # paid on entry + exit, so each is doubled.
    cost_per_trade = (
        (slippage_bps + commission_bps) * 2.0 / 10_000.0
    ) * notional

    all_rows: List[Candidate] = []
    for thr, n, wr, avg_win, avg_loss, sharpe_field, approx in rows:
        net = (
            wr * avg_win
            - (1.0 - wr) * abs(avg_loss)
            - cost_per_trade
        )
        if sharpe_field is None:
            sharpe = _approx_per_trade_sharpe(wr, avg_win, avg_loss, cost_per_trade)
        else:
            sharpe = sharpe_field
        all_rows.append(
            Candidate(
                thr=thr,
                n=n,
                wr=wr,
                avg_win=avg_win,
                avg_loss=avg_loss,
                net_pnl=net,
                sharpe=sharpe,
                approx=approx,
            )
        )
    all_rows.sort(key=lambda c: c.thr)
    eligible = [c for c in all_rows if c.n >= min_n]
    return all_rows, eligible


def _pick_winner(eligible: List[Candidate]) -> Optional[Candidate]:
    """Highest net_pnl wins; ties broken by higher sharpe, then lower thr."""
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda c: (c.net_pnl, c.sharpe, -c.thr),
    )


def _print_table(
    *,
    all_rows: List[Candidate],
    winner: Optional[Candidate],
    current_thr: Optional[float],
    source_label: str,
    min_n: int,
    slippage_bps: float,
    commission_bps: float,
    notional: float,
    meta_path: Path,
    file=None,
) -> None:
    # Resolve sys.stdout at CALL time so contextlib.redirect_stdout in
    # tests captures the output. Binding ``file=sys.stdout`` at def-time
    # would point at the original stdout forever.
    if file is None:
        file = sys.stdout
    print(f"meta: {meta_path}", file=file)
    print(f"source: {source_label}", file=file)
    print(
        f"params: slippage_bps={slippage_bps:g} commission_bps={commission_bps:g} "
        f"notional=${notional:g} min_n={min_n}",
        file=file,
    )
    cost_per_trade = (
        (slippage_bps + commission_bps) * 2.0 / 10_000.0
    ) * notional
    print(f"round-trip cost per trade: ${cost_per_trade:.4f}", file=file)
    print("", file=file)
    print(
        "| thr    | n     | wr     | avg_win  | avg_loss  | net_pnl  | sharpe   | tag",
        file=file,
    )
    print(
        "|--------|-------|--------|----------|-----------|----------|----------|-----",
        file=file,
    )
    for c in all_rows:
        tags: List[str] = []
        if winner is not None and c is winner:
            tags.append("<-- cost-aware pick")
        if current_thr is not None and abs(c.thr - current_thr) < 1e-9:
            tags.append("<-- current")
        if c.n < min_n:
            tags.append(f"(filtered: n<{min_n})")
        tag = " ".join(tags)
        print(f"{c.as_table_row()} {tag}".rstrip(), file=file)
    print("", file=file)
    if winner is None:
        print(
            f"NO WINNER: no threshold had n >= {min_n} "
            f"(or sweep was empty). Nothing to write.",
            file=file,
        )
    else:
        approx_note = " (symmetric-payoff approx)" if winner.approx else ""
        print(
            f"WINNER: thr={winner.thr:.4f} n={winner.n} wr={winner.wr:.4f} "
            f"net_pnl=${winner.net_pnl:+.4f}/trade sharpe={winner.sharpe:+.4f}"
            f"{approx_note}",
            file=file,
        )
        if winner.net_pnl < 0:
            print(
                "WARNING: best candidate has NEGATIVE expected net P&L. "
                "This model is unprofitable at the current cost assumptions; "
                "the winner is the least-bad option, not a trade-it-now signal.",
                file=file,
            )


def _atomic_write_meta(meta_path: Path, meta: Dict[str, Any]) -> None:
    """Write meta.json via tmp + rename so a crash can't leave it half-written."""
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp, meta_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-pick the operating threshold of a crypto XGBoost model "
            "by maximizing expected net P&L per trade (winrate * avg_win "
            "- (1-winrate) * |avg_loss| - slippage - commission). "
            "Dry-run by default; pass --write to persist into meta.json."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--meta",
        type=str,
        help="Path to meta.json. Mutually exclusive with --symbol.",
    )
    src.add_argument(
        "--symbol",
        choices=sorted(SYMBOL_META_PATHS),
        help="Canonical symbol shortcut (eth/btc/sol).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Slippage cost in bps PER SIDE (paid on entry AND exit). Default 5.0.",
    )
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=5.0,
        help="Commission in bps PER SIDE (paid on entry AND exit). Default 5.0.",
    )
    parser.add_argument(
        "--notional-usd",
        type=float,
        default=50.0,
        help="Per-trade notional USD used to dollarize bps costs. Default $50.",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=100,
        help=(
            "Minimum trade count for a threshold to be eligible. "
            "Sparse thresholds get filtered. Default 100."
        ),
    )
    parser.add_argument(
        "--target-move-bps",
        type=float,
        default=20.0,
        help=(
            "Assumed symmetric per-trade payoff in bps when meta only "
            "has test_precision_curve (no avg_win/avg_loss). Default 20 bps."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Persist winner into meta.json as meta.cost_aware_threshold "
            "(does NOT touch optimal_threshold). Default: dry-run."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    meta_path = _resolve_meta_path(args)

    if not meta_path.exists():
        print(f"ERROR: meta.json not found at {meta_path}", file=sys.stderr)
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    rows, source_label = _extract_candidates(
        meta,
        notional=args.notional_usd,
        target_move_bps=args.target_move_bps,
    )
    if not rows:
        print(
            f"ERROR: no threshold sweep data on disk at {meta_path}: both "
            f"threshold_metrics and test_precision_curve are missing/empty.",
            file=sys.stderr,
        )
        return 1

    all_rows, eligible = compute_candidates(
        rows,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        notional=args.notional_usd,
        min_n=args.min_n,
    )
    winner = _pick_winner(eligible)
    current_thr = meta.get("optimal_threshold")

    _print_table(
        all_rows=all_rows,
        winner=winner,
        current_thr=current_thr,
        source_label=source_label,
        min_n=args.min_n,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        notional=args.notional_usd,
        meta_path=meta_path,
    )

    if args.write:
        if winner is None:
            print(
                "--write requested but no eligible winner; refusing to "
                "scribble a None threshold into meta.json.",
                file=sys.stderr,
            )
            return 1
        meta["cost_aware_threshold"] = float(winner.thr)
        meta["cost_aware_threshold_at"] = datetime.now(timezone.utc).isoformat()
        meta["cost_aware_threshold_params"] = {
            "slippage_bps": float(args.slippage_bps),
            "commission_bps": float(args.commission_bps),
            "notional_usd": float(args.notional_usd),
            "min_n": int(args.min_n),
            "target_move_bps": float(args.target_move_bps),
            "source": source_label,
        }
        _atomic_write_meta(meta_path, meta)
        print(
            f"\nWROTE: meta.cost_aware_threshold = {winner.thr:.4f} -> {meta_path}",
            file=sys.stdout,
        )
        print(
            "(meta.optimal_threshold preserved; predictor adapters that "
            "want the cost-aware value must opt in.)",
            file=sys.stdout,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
