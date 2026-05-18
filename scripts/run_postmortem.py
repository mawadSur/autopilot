"""Lane E nightly postmortem orchestrator.

Runs the five forensic specialists (Signal / Execution / Sizing / Context /
Process) over every position closed on a given UTC date via
:class:`LossPostmortemSynthesizer`, aggregates the per-trade postmortems
into a single markdown digest at ``docs/digests/YYYY-MM-DD.md``, and posts
a one-line summary to the configured notification channels.

The specialists themselves live under ``src/loss_postmortem/`` and are NOT
modified by this script — every per-agent crash, timeout, or missing
snapshot is already converted to ``verdict="unknown"`` by
:class:`BaseForensicsAgent.safe_investigate`. The runner's job is the
date-scoped fan-out, the digest write, and the notifier dispatch.

Flags
-----
``--date YYYY-MM-DD``
    UTC date whose closed positions should be analysed. Defaults to
    yesterday in UTC (the launchd plist fires at 00:05 UTC, so the
    "yesterday" default lines up with the just-completed UTC day).
``--position-store-url URL``
    Redis URL backing the position store. Defaults to ``$REDIS_URL`` or
    ``redis://localhost:6379/0`` (matches :class:`PositionStore`'s default).
``--out-dir DIR``
    Where the daily digest markdown is written. Defaults to
    ``docs/digests``; created if missing.
``--no-publish``
    Run specialists + synthesizer + write the markdown, but skip the
    notifier dispatch. Useful for ad-hoc operator-driven runs.
``--dry-run``
    Run specialists + synthesizer, print the digest to stdout, write
    nothing to disk, post nothing. Pairs with the validation step in the
    deliverable check-list.

Exit semantics
--------------
The script exits 0 even when Redis is unreachable so launchd does not
keep retrying — it writes an empty digest with a "Redis unreachable"
header and moves on. Per-specialist failures land in their own digest
section (driven by :meth:`Synthesizer.process_one`'s safety wrappers).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
for _p in (SRC, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import redis  # noqa: E402  (sys.path bootstrap above)

from alerts.notifier import Notifier  # noqa: E402
from loss_postmortem.synthesizer import (  # noqa: E402
    LossPostmortemSynthesizer,
    PostmortemReport,
)
from state.position_store import Position, PositionStore  # noqa: E402

LOGGER = logging.getLogger("run_postmortem")

DEFAULT_OUT_DIR = REPO / "docs" / "digests"
DEFAULT_RUNS_DIR = REPO / "runs"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Lane E forensic swarm over a UTC day's closed "
            "positions and emit a markdown digest."
        ),
    )
    parser.add_argument(
        "--date",
        default=None,
        help="UTC date YYYY-MM-DD to analyse (defaults to yesterday UTC).",
    )
    parser.add_argument(
        "--position-store-url",
        default=None,
        help=(
            "Redis URL backing the position store (default: $REDIS_URL or "
            "redis://localhost:6379/0)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for the daily digest markdown (default: docs/digests).",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Skip notifier dispatch. Still writes the digest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print to stdout only. Write nothing, post nothing.",
    )
    return parser.parse_args(argv)


def _resolve_target_date(raw: Optional[str]) -> _dt.date:
    if raw is None:
        now = _dt.datetime.now(_dt.timezone.utc)
        return (now - _dt.timedelta(days=1)).date()
    try:
        return _dt.datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"--date must be YYYY-MM-DD, got {raw!r} ({exc})")


def _date_to_utc_midnight(target: _dt.date) -> _dt.datetime:
    return _dt.datetime(
        target.year, target.month, target.day, 0, 0, 0, tzinfo=_dt.timezone.utc
    )


def _short_id(trade_id: str) -> str:
    if len(trade_id) <= 12:
        return trade_id
    return trade_id[:8] + "…"


def _read_closed_positions(
    *, store: PositionStore, target: _dt.date
) -> List[Position]:
    """Return the day's closed positions. Raises redis.exceptions.ConnectionError on Redis-down."""
    # Anchor on the middle of the UTC day so the date arithmetic inside
    # ``_closed_set_key`` lands on ``target`` regardless of microsecond drift.
    when = _date_to_utc_midnight(target) + _dt.timedelta(hours=12)
    return list(store.list_closed_today(now_utc=when))


def _format_empty_digest(*, target: _dt.date, reason: str) -> str:
    lines = [
        f"# Loss Postmortem Digest — {target.isoformat()}",
        "",
        f"_{reason}_",
        "",
        "No closed positions were analysed.",
        "",
    ]
    return "\n".join(lines)


def _format_digest(
    *,
    target: _dt.date,
    positions: List[Position],
    reports: List[PostmortemReport],
    errors: List[str],
) -> str:
    """Build the daily digest markdown for ``target``."""
    rc_counts: dict = {}
    losses_total = 0.0
    losing_count = 0
    for r in reports:
        rc = r.root_cause or "Unknown"
        rc_counts[rc] = rc_counts.get(rc, 0) + 1
        if r.loss_usd is not None and r.loss_usd < 0:
            losing_count += 1
            losses_total += float(r.loss_usd)

    lines: List[str] = []
    lines.append(f"# Loss Postmortem Digest — {target.isoformat()}")
    lines.append("")
    lines.append(f"- Closed positions inspected: {len(positions)}")
    lines.append(f"- Postmortems produced: {len(reports)}")
    lines.append(
        f"- Losing positions: {losing_count} (total ${losses_total:,.2f})"
    )
    if rc_counts:
        dist = ", ".join(f"{k}={v}" for k, v in sorted(rc_counts.items()))
        lines.append(f"- Root-cause distribution: {dist}")
    else:
        lines.append("- Root-cause distribution: (none)")
    lines.append("")

    if errors:
        for msg in errors:
            lines.append(f"## Specialist errored: {msg}")
            lines.append("")

    if not reports:
        lines.append("_No postmortems to report._")
        lines.append("")
        return "\n".join(lines)

    # Per-trade section, sorted by loss_usd (most negative first), then by trade_id.
    def _sort_key(rep: PostmortemReport):
        loss = rep.loss_usd if rep.loss_usd is not None else 0.0
        return (loss, rep.trade_id)

    lines.append("## Per-trade findings")
    lines.append("")
    for rep in sorted(reports, key=_sort_key):
        sym = rep.symbol or "?"
        loss_str = (
            f"${rep.loss_usd:,.2f}" if rep.loss_usd is not None else "n/a"
        )
        lines.append(
            f"### `{_short_id(rep.trade_id)}` — {sym} — {loss_str} — "
            f"{rep.root_cause}"
        )
        lines.append(f"_{rep.summary}_")
        lines.append("")
        for f in rep.findings:
            err_tag = f" (error: `{f.error}`)" if f.error else ""
            lines.append(
                f"- **{f.agent}**: {f.verdict} (confidence "
                f"{f.confidence:.2f}){err_tag}"
            )
            for bullet in (f.evidence or [])[:3]:
                lines.append(f"  - {bullet}")
        if rep.actions:
            lines.append("")
            lines.append("Suggested actions:")
            for a in rep.actions:
                lines.append(f"- `{a}`")
        lines.append("")
    return "\n".join(lines)


def _one_line_summary(
    *,
    target: _dt.date,
    positions_count: int,
    reports_count: int,
    losing_count: int,
    losses_total: float,
    digest_path: Path,
) -> str:
    return (
        f"Postmortem digest {target.isoformat()}: "
        f"closed={positions_count} reports={reports_count} "
        f"losing={losing_count} pnl=${losses_total:,.2f} — {digest_path}"
    )


def run(
    *,
    target: _dt.date,
    position_store_url: Optional[str],
    out_dir: Path,
    dry_run: bool,
    no_publish: bool,
    position_store: Optional[PositionStore] = None,
    synthesizer: Optional[LossPostmortemSynthesizer] = None,
    notifier: Optional[Notifier] = None,
) -> int:
    """Programmatic entrypoint. Returns the intended process exit code (0).

    All injectable seams are optional kwargs so tests can substitute
    fakeredis-backed stores, in-process synthesizers, and stub notifiers.
    """
    LOGGER.info(
        "lane_e_postmortem start target=%s dry_run=%s no_publish=%s",
        target.isoformat(),
        dry_run,
        no_publish,
    )

    # Resolve the position store. Tests inject a pre-built one; production
    # builds one against the configured Redis URL.
    if position_store is None:
        position_store = PositionStore(redis_url=position_store_url)

    # Read closed positions for the target day. The one named exception
    # this runner catches at this seam is redis.exceptions.ConnectionError
    # per the spec — write an empty-data digest and exit 0 so launchd
    # doesn't keep retrying.
    try:
        positions = _read_closed_positions(store=position_store, target=target)
    except redis.exceptions.ConnectionError as exc:
        LOGGER.warning("redis unreachable while reading closed positions: %r", exc)
        digest = _format_empty_digest(
            target=target,
            reason="Redis unreachable, no data analyzed today",
        )
        if dry_run:
            print(digest)
            return 0
        _write_digest(out_dir=out_dir, target=target, digest=digest)
        return 0

    if synthesizer is None:
        synthesizer = LossPostmortemSynthesizer(
            runs_dir=DEFAULT_RUNS_DIR,
            position_store=position_store,
            worker_config={
                "redis_url": position_store_url
                or os.environ.get("REDIS_URL")
                or "redis://localhost:6379/0",
            },
        )

    reports: List[PostmortemReport] = []
    specialist_errors: List[str] = []
    for pos in positions:
        try:
            report = synthesizer.process_one(pos.position_id)
        except redis.exceptions.ConnectionError as exc:
            # Mid-run Redis loss: record and continue so a flaky network
            # doesn't strand the whole digest.
            LOGGER.warning(
                "redis unreachable mid-run for %s: %r", pos.position_id, exc
            )
            specialist_errors.append(
                f"redis_disconnect:{_short_id(pos.position_id)}: {exc!r}"
            )
            continue
        reports.append(report)

    losing_count = sum(
        1 for r in reports if r.loss_usd is not None and r.loss_usd < 0
    )
    losses_total = sum(
        float(r.loss_usd)
        for r in reports
        if r.loss_usd is not None and r.loss_usd < 0
    )

    digest = _format_digest(
        target=target,
        positions=positions,
        reports=reports,
        errors=specialist_errors,
    )

    if dry_run:
        print(digest)
        LOGGER.info(
            "lane_e_postmortem dry_run done positions=%d reports=%d",
            len(positions),
            len(reports),
        )
        return 0

    digest_path = _write_digest(out_dir=out_dir, target=target, digest=digest)

    if no_publish:
        LOGGER.info(
            "lane_e_postmortem skipped publish (--no-publish) digest=%s",
            digest_path,
        )
        return 0

    summary_line = _one_line_summary(
        target=target,
        positions_count=len(positions),
        reports_count=len(reports),
        losing_count=losing_count,
        losses_total=losses_total,
        digest_path=digest_path,
    )
    _dispatch(
        notifier=notifier,
        summary_line=summary_line,
        digest_path=digest_path,
    )
    LOGGER.info("lane_e_postmortem done digest=%s", digest_path)
    return 0


def _write_digest(*, out_dir: Path, target: _dt.date, digest: str) -> Path:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.warning("could not create digest dir %s: %r", out_dir, exc)
    path = out_dir / f"{target.isoformat()}.md"
    path.write_text(digest, encoding="utf-8")
    LOGGER.info("wrote digest %s", path)
    return path


def _dispatch(
    *,
    notifier: Optional[Notifier],
    summary_line: str,
    digest_path: Path,
) -> None:
    """Post the one-line summary via the notifier. Quiet when unconfigured."""
    if notifier is None:
        notifier = Notifier()
    cfg = notifier.is_configured()
    if not (cfg.get("discord") or cfg.get("telegram")):
        LOGGER.info("no notifier channel configured; skipping publish")
        return
    fields = {"Digest": str(digest_path)}
    # ``alert`` is the action-required surface; the digest is operationally
    # interesting even on quiet days.
    posted = notifier.alert(
        summary_line, severity="info", fields=fields
    )
    if not posted:
        LOGGER.warning("notifier.alert returned False for digest dispatch")


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args(argv)
    target = _resolve_target_date(args.date)
    return run(
        target=target,
        position_store_url=args.position_store_url,
        out_dir=Path(args.out_dir),
        dry_run=bool(args.dry_run),
        no_publish=bool(args.no_publish),
    )


if __name__ == "__main__":
    raise SystemExit(main())
