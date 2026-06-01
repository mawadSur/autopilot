"""Telegram control bot for the live_supervisor paper sessions.

Long-polls Telegram's ``getUpdates`` endpoint and dispatches a small whitelist
of commands against the ETH v1 / v2 supervisors. Reads
``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` from environment or
``src/.env`` (same vars the notifier already uses). Only messages from the
configured chat are honored; everything else is ignored silently.

Commands::

    /status                 -- one-line health on v1, v2, bridge
    /stop v1 | /stop v2     -- SIGINT the named supervisor
    /start v1 | /start v2   -- spawn a new supervisor with the canonical args
    /mute                   -- silence the Discord paper-fill bridge
    /unmute                 -- resume the Discord paper-fill bridge
    /help                   -- list commands

State persisted at ``logs/.telegram_bot_offset`` (update_id watermark) and
``logs/.bridge_muted`` (mute flag the Discord bridge polls). Run as a
nohup daemon::

    ./.venv/bin/python scripts/telegram_control_bot.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

LOGGER = logging.getLogger("telegram_control_bot")

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
OFFSET_PATH = REPO_ROOT / "logs" / ".telegram_bot_offset"
MUTE_FLAG_PATH = REPO_ROOT / "logs" / ".bridge_muted"

API_BASE = "https://api.telegram.org/bot{token}"
LONG_POLL_TIMEOUT_S = 30


@dataclass(frozen=True)
class SupervisorVariant:
    """Static launch config for one paper-supervisor variant."""

    tag: str
    model_dir: str
    threshold: float
    log_dir: Path

    def crypto_model_map(self) -> str:
        return f"ETH/USD={self.model_dir}:{self.threshold}"

    def launch_argv(self) -> List[str]:
        return [
            str(VENV_PY),
            "src/live_supervisor.py",
            "--symbols", "ETH/USD",
            "--mode", "paper",
            "--bankroll", "10000",
            "--min-confidence", "0.51",
            "--interval", "5",
            "--log-dir", str(self.log_dir),
        ]


VARIANTS: Dict[str, SupervisorVariant] = {
    "v1": SupervisorVariant(
        tag="v1",
        model_dir="model_crypto/eth_usd_voln_v1",
        threshold=0.55,
        log_dir=REPO_ROOT / "logs" / "eth_paper_multiday",
    ),
    "v2": SupervisorVariant(
        tag="v2",
        model_dir="model_crypto/eth_usd_voln_v2",
        threshold=0.57,
        log_dir=REPO_ROOT / "logs" / "eth_paper_multiday_v2",
    ),
}


def _load_dotenv(key: str) -> Optional[str]:
    """Read ``key`` from os.environ, falling back to src/.env."""

    val = os.environ.get(key, "").strip()
    if val:
        return val
    env_path = REPO_ROOT / "src" / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, raw = line.partition("=")
        if k.strip() == key:
            return raw.strip().strip("'\"") or None
    return None


_LOG_DIR_TOKEN_RE = re.compile(r"--log-dir\s+(\S+)")


def _find_supervisor_pid(variant: SupervisorVariant) -> Optional[int]:
    """Locate the supervisor PID for ``variant`` by matching its log dir.

    Walks ``ps -ax -o pid,command`` (macOS-compatible) and matches the
    ``--log-dir <DIR>`` token exactly -- substring matching would let
    ``logs/eth_paper_multiday`` collide with ``logs/eth_paper_multiday_v2``.
    Accepts both absolute and relative paths that resolve to ``variant.log_dir``.
    """
    target = variant.log_dir.resolve()
    try:
        out = subprocess.run(
            ["ps", "-ax", "-o", "pid,command"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    for line in out.stdout.splitlines():
        if "live_supervisor.py" not in line:
            continue
        m = _LOG_DIR_TOKEN_RE.search(line)
        if not m:
            continue
        raw_dir = m.group(1)
        try:
            # Resolve relative to repo root for relative paths.
            resolved = (REPO_ROOT / raw_dir).resolve() if not Path(raw_dir).is_absolute() else Path(raw_dir).resolve()
        except OSError:
            continue
        if resolved != target:
            continue
        try:
            return int(line.split(None, 1)[0])
        except (ValueError, IndexError):
            continue
    return None


def _supervisor_proc_stats(pid: int) -> Optional[Dict[str, str]]:
    """Return etime + rss for ``pid`` or None if ``pid`` is gone."""

    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime=,rss="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    raw = out.stdout.strip()
    if not raw:
        return None
    parts = raw.split()
    if len(parts) < 2:
        return None
    return {"etime": parts[0], "rss_kb": parts[1]}


def _latest_log(log_dir: Path) -> Optional[Path]:
    if not log_dir.exists():
        return None
    cands = sorted(
        log_dir.glob("eth_voln_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _count_paper_fills(log_path: Optional[Path]) -> int:
    if log_path is None or not log_path.exists():
        return 0
    needle = b"paper fill: drained pending"
    n = 0
    try:
        with log_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                n += chunk.count(needle)
    except OSError:
        return 0
    return n


def _last_predictor_line(log_path: Optional[Path]) -> Optional[str]:
    """Grab the most recent ``xgb predictor:`` line for current P(long)."""

    if log_path is None or not log_path.exists():
        return None
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    last = None
    for line in tail.splitlines():
        if "xgb predictor:" in line:
            last = line
    return last


class TelegramClient:
    def __init__(self, token: str, chat_id: str) -> None:
        self.base = API_BASE.format(token=token)
        self.chat_id = chat_id
        self._session = requests.Session()

    def send(self, text: str) -> bool:
        try:
            r = self._session.post(
                self.base + "/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text[:4000],
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if r.status_code >= 300:
                LOGGER.warning("telegram send %d: %s", r.status_code, r.text[:200])
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("telegram send raised: %s", exc)
            return False

    def poll_updates(self, offset: int) -> Tuple[List[dict], int]:
        """Long-poll getUpdates and return (updates, new_offset)."""

        try:
            r = self._session.get(
                self.base + "/getUpdates",
                params={
                    "offset": offset,
                    "timeout": LONG_POLL_TIMEOUT_S,
                    "allowed_updates": json.dumps(["message"]),
                },
                timeout=LONG_POLL_TIMEOUT_S + 10,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("telegram getUpdates raised: %s", exc)
            time.sleep(2)
            return [], offset
        if r.status_code >= 300:
            LOGGER.warning("telegram getUpdates %d: %s", r.status_code, r.text[:200])
            time.sleep(2)
            return [], offset
        payload = r.json()
        if not payload.get("ok"):
            return [], offset
        results = payload.get("result", [])
        new_offset = offset
        for u in results:
            uid = int(u.get("update_id", 0))
            if uid >= new_offset:
                new_offset = uid + 1
        return results, new_offset


def _load_offset() -> int:
    if not OFFSET_PATH.exists():
        return 0
    try:
        return int(OFFSET_PATH.read_text().strip() or "0")
    except (ValueError, OSError):
        return 0


def _save_offset(offset: int) -> None:
    OFFSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    OFFSET_PATH.write_text(str(offset))


# ----------------------------------------------------------------------
# Command handlers
# ----------------------------------------------------------------------
def cmd_help() -> str:
    return (
        "*Autopilot control*\n"
        "/status -- v1/v2 supervisor + bridge health\n"
        "/stop v1 | /stop v2 -- SIGINT named supervisor\n"
        "/start v1 | /start v2 -- spawn supervisor\n"
        "/mute -- silence Discord fill summaries\n"
        "/unmute -- resume Discord fill summaries\n"
        "/help -- this message"
    )


def cmd_status() -> str:
    lines = ["*Status*"]
    for tag, var in VARIANTS.items():
        pid = _find_supervisor_pid(var)
        if pid is None:
            lines.append(f"`{tag}` -- stopped")
            continue
        stats = _supervisor_proc_stats(pid) or {}
        log_path = _latest_log(var.log_dir)
        fills = _count_paper_fills(log_path)
        last_pred = _last_predictor_line(log_path) or ""
        # Extract just "P(long)=X.XXX" for compactness.
        m = re.search(r"P\(long\)=([\d.]+)", last_pred)
        plong = f" P(long)={m.group(1)}" if m else ""
        lines.append(
            f"`{tag}` -- pid {pid}, up {stats.get('etime','?')}, "
            f"rss {stats.get('rss_kb','?')}KB, "
            f"fills {fills},{plong}"
        )
    muted = MUTE_FLAG_PATH.exists()
    lines.append(f"_bridge_: {'muted' if muted else 'live'}")
    return "\n".join(lines)


def cmd_stop(tag: str) -> str:
    var = VARIANTS.get(tag)
    if var is None:
        return f"unknown variant `{tag}`. use v1 or v2."
    pid = _find_supervisor_pid(var)
    if pid is None:
        return f"`{tag}` not running"
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return f"`{tag}` already gone (pid {pid})"
    except PermissionError:
        return f"permission denied sending SIGINT to {pid}"
    return f":stop_sign: `{tag}` SIGINT sent to pid {pid}"


def cmd_start(tag: str) -> str:
    var = VARIANTS.get(tag)
    if var is None:
        return f"unknown variant `{tag}`. use v1 or v2."
    if _find_supervisor_pid(var) is not None:
        return f"`{tag}` already running"
    var.log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_file = var.log_dir / f"eth_voln_{var.tag}_{ts}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["CRYPTO_MODEL_MAP"] = var.crypto_model_map()
    try:
        proc = subprocess.Popen(  # noqa: S603 -- argv is hardcoded
            var.launch_argv(),
            stdout=open(log_file, "ab"),
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(REPO_ROOT),
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        return f"start `{tag}` failed: {exc}"
    return f":rocket: `{tag}` started pid {proc.pid}, log {log_file.name}"


def cmd_mute() -> str:
    MUTE_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MUTE_FLAG_PATH.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return ":mute: Discord bridge muted (kill-switch alerts via supervisor still fire)"


def cmd_unmute() -> str:
    try:
        MUTE_FLAG_PATH.unlink(missing_ok=True)
    except OSError as exc:
        return f"unmute failed: {exc}"
    return ":loud_sound: Discord bridge unmuted"


def dispatch(text: str) -> Optional[str]:
    """Parse ``text`` and return a reply, or None to silently ignore."""

    text = text.strip()
    if not text.startswith("/"):
        return None
    # Strip /command@botname suffix that telegram appends in groups.
    head, _, rest = text.partition(" ")
    head = head.split("@", 1)[0].lower()
    arg = rest.strip().lower()

    if head == "/help":
        return cmd_help()
    if head == "/status":
        return cmd_status()
    if head == "/stop":
        return cmd_stop(arg or "")
    if head == "/start":
        return cmd_start(arg or "")
    if head == "/mute":
        return cmd_mute()
    if head == "/unmute":
        return cmd_unmute()
    return None  # unknown command -- silent


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    token = _load_dotenv("TELEGRAM_BOT_TOKEN")
    chat_id = _load_dotenv("TELEGRAM_CHAT_ID")
    if not token:
        LOGGER.error("TELEGRAM_BOT_TOKEN not set (env + src/.env). Exiting.")
        return 2
    if not chat_id:
        LOGGER.error("TELEGRAM_CHAT_ID not set; refusing to start without whitelist.")
        return 2

    client = TelegramClient(token, chat_id)
    client.send(":satellite: Control bot online. /help for commands.")

    offset = _load_offset()
    LOGGER.info("control bot starting from update offset %d", offset)

    while True:
        updates, new_offset = client.poll_updates(offset)
        if new_offset != offset:
            offset = new_offset
            _save_offset(offset)
        for u in updates:
            msg = u.get("message") or {}
            chat = msg.get("chat") or {}
            # Whitelist: only honor commands from the configured chat.
            if str(chat.get("id")) != str(chat_id):
                LOGGER.info(
                    "ignoring message from unauthorized chat %s",
                    chat.get("id"),
                )
                continue
            text = msg.get("text") or ""
            reply = dispatch(text)
            if reply is not None:
                LOGGER.info("cmd %r -> reply len=%d", text[:60], len(reply))
                client.send(reply)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOGGER.info("control bot interrupted; exiting")
        sys.exit(0)
