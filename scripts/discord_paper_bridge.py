"""Tail-bridge from live_supervisor.py paper logs to a Discord webhook.

Watches the latest ``eth_voln_*.log`` under each provided log dir and posts a
condensed event stream to Discord. Designed for the parallel v1/v2 paper
sessions but symbol/tag agnostic.

Events surfaced:
  * paper fill drained -- aggregated per minute per tag (avoids 10x/min spam)
  * log rotation -- per occurrence (informational)

The supervisor's own ``Notifier`` already posts kill_switch_tripped,
auto_pause, daily_close, and live-fill events to the same webhook -- the
bridge stays out of those lanes to avoid duplicate pings.

Reads ``DISCORD_WEBHOOK_URL`` from environment or from ``src/.env``. Persists
per-file byte offsets to ``logs/.discord_bridge_offsets.json`` so restarts
resume cleanly.

Run::

    ./.venv/bin/python scripts/discord_paper_bridge.py \\
        --watch v1=logs/eth_paper_multiday \\
        --watch v2=logs/eth_paper_multiday_v2

Or use the defaults (same as above)::

    ./.venv/bin/python scripts/discord_paper_bridge.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

LOGGER = logging.getLogger("discord_paper_bridge")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OFFSETS = REPO_ROOT / "logs" / ".discord_bridge_offsets.json"
MUTE_FLAG_PATH = REPO_ROOT / "logs" / ".bridge_muted"

FILL_RE = re.compile(
    r"paper fill: drained pending (?P<sym>\S+) side=(?P<side>\w+) "
    r"at next-tick price (?P<price>[\d.]+)"
)


def _load_dotenv_webhook() -> Optional[str]:
    """Read DISCORD_WEBHOOK_URL from src/.env if not in os.environ."""

    val = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if val:
        return val
    env_path = REPO_ROOT / "src" / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        if key.strip() == "DISCORD_WEBHOOK_URL":
            return raw.strip().strip("'\"") or None
    return None


@dataclass
class FillBucket:
    """Per-tag, per-minute fill aggregation."""

    tag: str
    minute_key: str
    buys: int = 0
    sells: int = 0
    prices: List[float] = field(default_factory=list)

    def add(self, side: str, price: float) -> None:
        if side == "buy":
            self.buys += 1
        elif side == "sell":
            self.sells += 1
        self.prices.append(price)

    def render(self) -> str:
        avg = sum(self.prices) / len(self.prices) if self.prices else 0.0
        lo = min(self.prices) if self.prices else 0.0
        hi = max(self.prices) if self.prices else 0.0
        parts = []
        if self.buys:
            parts.append(f"{self.buys} BUY")
        if self.sells:
            parts.append(f"{self.sells} SELL")
        return (
            f"`[{self.tag}]` {' / '.join(parts)} fills @ "
            f"avg ${avg:,.2f} (range ${lo:,.2f}-${hi:,.2f}) "
            f"-- minute {self.minute_key}Z"
        )


def _latest_log(log_dir: Path) -> Optional[Path]:
    """Return the most recently modified ``eth_voln_*.log`` in ``log_dir``."""

    if not log_dir.exists():
        return None
    cands = sorted(
        log_dir.glob("eth_voln_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


class DiscordPoster:
    """Best-effort Discord webhook poster with a soft rate limit."""

    def __init__(self, webhook_url: str, min_interval_s: float = 1.2) -> None:
        self.webhook_url = webhook_url
        self.min_interval_s = min_interval_s
        self._last_post_t = 0.0

    def post(self, message: str) -> bool:
        now = time.monotonic()
        gap = now - self._last_post_t
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)
        try:
            resp = requests.post(
                self.webhook_url,
                json={"content": message[:1900]},
                timeout=5.0,
            )
            self._last_post_t = time.monotonic()
            if resp.status_code >= 300:
                LOGGER.warning(
                    "discord post returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
            return True
        except Exception as exc:  # noqa: BLE001 -- best-effort
            LOGGER.warning("discord post raised: %s", exc)
            return False


class Bridge:
    def __init__(
        self,
        *,
        watches: List[Tuple[str, Path]],
        poster: DiscordPoster,
        offsets_path: Path,
        flush_seconds: int = 30,
    ) -> None:
        self.watches = watches
        self.poster = poster
        self.offsets_path = offsets_path
        self.flush_seconds = flush_seconds
        self.offsets: Dict[str, int] = self._load_offsets()
        self.buckets: Dict[str, FillBucket] = {}
        self.current_files: Dict[str, Path] = {}

    def _load_offsets(self) -> Dict[str, int]:
        if not self.offsets_path.exists():
            return {}
        try:
            return json.loads(self.offsets_path.read_text())
        except Exception:  # noqa: BLE001
            return {}

    def _save_offsets(self) -> None:
        self.offsets_path.parent.mkdir(parents=True, exist_ok=True)
        self.offsets_path.write_text(json.dumps(self.offsets, indent=2))

    def _resolve_log(self, log_dir: Path) -> Optional[Path]:
        return _latest_log(log_dir)

    def _read_new_lines(self, tag: str, log_path: Path) -> List[str]:
        key = str(log_path)
        last_offset = self.offsets.get(key, -1)
        size = log_path.stat().st_size
        if last_offset < 0:
            # First time seeing this file -- start at EOF, don't replay history.
            self.offsets[key] = size
            self._save_offsets()
            self.current_files[tag] = log_path
            return []
        if size < last_offset:
            # File was rotated/truncated.
            last_offset = 0
        with log_path.open("rb") as f:
            f.seek(last_offset)
            chunk = f.read().decode("utf-8", errors="replace")
        self.offsets[key] = size
        self._save_offsets()
        self.current_files[tag] = log_path
        return chunk.splitlines()

    def _flush_bucket(self, key: str) -> None:
        bucket = self.buckets.pop(key, None)
        if bucket is None or not bucket.prices:
            return
        # Honour the telegram /mute flag: drop the bucket silently.
        if MUTE_FLAG_PATH.exists():
            return
        self.poster.post(bucket.render())

    def _process_line(self, tag: str, line: str) -> None:
        # Extract leading timestamp "YYYY-MM-DD HH:MM:SS" for minute keying.
        ts_match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}):\d{2}", line)
        minute_key = ts_match.group(1) if ts_match else "unknown"

        m = FILL_RE.search(line)
        if m:
            side = m.group("side")
            try:
                price = float(m.group("price"))
            except ValueError:
                price = 0.0
            bucket_key = f"{tag}:{minute_key}"
            bucket = self.buckets.get(bucket_key)
            if bucket is None:
                bucket = FillBucket(tag=tag, minute_key=minute_key)
                self.buckets[bucket_key] = bucket
            bucket.add(side, price)

    def _maybe_flush_stale_buckets(self) -> None:
        # Flush minute buckets older than the current wall-clock minute.
        now_minute = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
        stale = [k for k, b in self.buckets.items() if b.minute_key < now_minute]
        for k in stale:
            self._flush_bucket(k)

    def run(self) -> None:
        LOGGER.info(
            "bridge starting: watches=%s offsets_path=%s",
            [(t, str(p)) for t, p in self.watches],
            self.offsets_path,
        )
        # Post a startup ping per watched tag.
        for tag, log_dir in self.watches:
            log_path = self._resolve_log(log_dir)
            if log_path is None:
                self.poster.post(
                    f":warning: `[{tag}]` no log yet under {log_dir} (will retry)"
                )
            else:
                self.poster.post(
                    f":satellite: `[{tag}]` bridge online, tailing `{log_path.name}`"
                )

        while True:
            for tag, log_dir in self.watches:
                log_path = self._resolve_log(log_dir)
                if log_path is None:
                    continue
                # Detect log rotation: latest file changed for this tag.
                prev = self.current_files.get(tag)
                if prev is not None and prev != log_path:
                    self.poster.post(
                        f":arrows_counterclockwise: `[{tag}]` log rotated to "
                        f"`{log_path.name}`"
                    )
                lines = self._read_new_lines(tag, log_path)
                for line in lines:
                    self._process_line(tag, line)
            self._maybe_flush_stale_buckets()
            time.sleep(self.flush_seconds)


def _parse_watch(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"watch spec must be TAG=PATH, got {spec!r}"
        )
    tag, _, path = spec.partition("=")
    return tag.strip(), Path(path.strip())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watch",
        action="append",
        type=_parse_watch,
        help="TAG=PATH log directory to watch; repeatable",
    )
    parser.add_argument(
        "--offsets",
        type=Path,
        default=DEFAULT_OFFSETS,
        help=f"offsets state file (default {DEFAULT_OFFSETS})",
    )
    parser.add_argument(
        "--flush-seconds",
        type=int,
        default=30,
        help="poll interval and bucket flush cadence (default 30s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level (default INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    watches = args.watch or [
        ("ETH v1", REPO_ROOT / "logs" / "eth_paper_multiday"),
        ("ETH v2", REPO_ROOT / "logs" / "eth_paper_multiday_v2"),
    ]

    webhook = _load_dotenv_webhook()
    if not webhook:
        LOGGER.error(
            "DISCORD_WEBHOOK_URL not set (checked env + src/.env). Exiting."
        )
        return 2

    poster = DiscordPoster(webhook)
    bridge = Bridge(
        watches=watches,
        poster=poster,
        offsets_path=args.offsets,
        flush_seconds=args.flush_seconds,
    )
    try:
        bridge.run()
    except KeyboardInterrupt:
        LOGGER.info("bridge interrupted; flushing buckets")
        for k in list(bridge.buckets.keys()):
            bridge._flush_bucket(k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
