"""Alerts pipeline — best-effort Discord (info/log) and Telegram (action-required)
notifications so the live trader can run unattended.

Design tenets:
    * **Best-effort, never raises.** Every public method swallows transport
      errors and returns ``False`` on failure. The trader cannot crash because
      a webhook is down.
    * **Graceful degradation.** With no channels configured the public methods
      no-op and emit a *single* WARNING log so noisy log files don't drown
      operators.
    * **Hermetic by construction.** All HTTP goes through ``self._session``
      which tests substitute with a stub.

Channel routing:
    * ``info`` / ``daily_summary`` / ``fill_event``  -> Discord only.
    * ``alert`` / ``kill_switch_tripped``            -> Discord + Telegram.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Literal, Optional

import requests

logger = logging.getLogger(__name__)

AlertSeverity = Literal["debug", "info", "warning", "alert", "critical"]

# Discord embed colors (decimal). Picked to match common operator dashboards:
#   info     -> blue   (0x3498DB)
#   warning  -> yellow (0xF1C40F)
#   alert    -> orange (0xE67E22)
#   critical -> red    (0xE74C3C)
#   daily    -> green  (0x2ECC71)
#   debug    -> grey   (0x95A5A6)
_SEVERITY_COLORS: Dict[str, int] = {
    "debug": 0x95A5A6,
    "info": 0x3498DB,
    "warning": 0xF1C40F,
    "alert": 0xE67E22,
    "critical": 0xE74C3C,
    "daily": 0x2ECC71,
}

# Per Telegram Bot API docs, MarkdownV2 reserves these as control characters
# anywhere in the message and they MUST be backslash-escaped:
#     _ * [ ] ( ) ~ ` > # + - = | { } . !
# https://core.telegram.org/bots/api#markdownv2-style
_TELEGRAM_MD_V2_SPECIAL = r"_*[]()~`>#+-=|{}.!"

_DISCORD_TITLE_CAP = 256
_DISCORD_EMBED_DESC_CAP = 4096
_DISCORD_FIELD_VALUE_CAP = 1024
_TELEGRAM_TEXT_CAP = 4096
_USER_AGENT = "autopilot-notifier/0.1"


def _escape_telegram_md(text: str) -> str:
    """Escape Telegram MarkdownV2 reserved characters with a backslash.

    The Telegram Bot API rejects messages with unescaped reserved characters
    when ``parse_mode=MarkdownV2``. This helper is conservative: it escapes
    every reserved char even inside what looks like intentional formatting,
    because alert payloads are short and we'd rather be safe than 400.
    """
    if not text:
        return ""
    out = []
    for ch in text:
        if ch in _TELEGRAM_MD_V2_SPECIAL:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_signed_usd(value: float) -> str:
    """Format a USD amount with an explicit sign, e.g. +$12.34 / -$5.00."""
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    if cap <= 1:
        return text[:cap]
    return text[: cap - 1] + "…"


class Notifier:
    """Best-effort Discord + Telegram notifier."""

    def __init__(
        self,
        *,
        discord_webhook_url: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        timeout_s: float = 5.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.discord_webhook_url = discord_webhook_url or os.environ.get(
            "DISCORD_WEBHOOK_URL"
        ) or None
        self.telegram_bot_token = telegram_bot_token or os.environ.get(
            "TELEGRAM_BOT_TOKEN"
        ) or None
        self.telegram_chat_id = telegram_chat_id or os.environ.get(
            "TELEGRAM_CHAT_ID"
        ) or None
        # Treat empty strings as unset (env files often leave blanks).
        if self.discord_webhook_url == "":
            self.discord_webhook_url = None
        if self.telegram_bot_token == "":
            self.telegram_bot_token = None
        if self.telegram_chat_id == "":
            self.telegram_chat_id = None

        self.timeout_s = timeout_s
        if session is not None:
            self._session = session
        else:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": _USER_AGENT})
        # One-time warning latch so unconfigured deployments don't spam logs.
        self._unconfigured_warning_emitted = False

    # ------------------------------------------------------------------
    # Configuration introspection
    # ------------------------------------------------------------------
    def is_configured(self) -> Dict[str, bool]:
        """Return a per-channel configuration flag.

        Telegram requires both a bot token and a chat id to be useful.
        """
        return {
            "discord": bool(self.discord_webhook_url),
            "telegram": bool(self.telegram_bot_token and self.telegram_chat_id),
        }

    def _any_channel_configured(self) -> bool:
        cfg = self.is_configured()
        return cfg["discord"] or cfg["telegram"]

    def _warn_unconfigured_once(self) -> None:
        if not self._unconfigured_warning_emitted:
            logger.warning(
                "Notifier called but no channel configured "
                "(DISCORD_WEBHOOK_URL / TELEGRAM_BOT_TOKEN+CHAT_ID unset). "
                "Notifications will be dropped."
            )
            self._unconfigured_warning_emitted = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def info(
        self,
        message: str,
        *,
        fields: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Discord-only informational message.

        Returns True on success, False if the channel is not configured or
        the HTTP call failed.
        """
        if not self._any_channel_configured():
            self._warn_unconfigured_once()
            return False
        if not self.discord_webhook_url:
            # Telegram-only deployments still receive alerts; info is a no-op.
            return False
        return self._post_discord(
            message=message, severity="info", fields=fields
        )

    def alert(
        self,
        message: str,
        *,
        severity: AlertSeverity = "alert",
        fields: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Action-required alert. Sends to BOTH Discord and Telegram.

        Returns True if at least one channel succeeded.
        """
        if not self._any_channel_configured():
            self._warn_unconfigured_once()
            return False

        discord_ok = False
        telegram_ok = False
        if self.discord_webhook_url:
            discord_ok = self._post_discord(
                message=message, severity=severity, fields=fields
            )
        if self.telegram_bot_token and self.telegram_chat_id:
            telegram_ok = self._post_telegram(
                message=message, severity=severity, fields=fields
            )
        return discord_ok or telegram_ok

    def daily_summary(
        self,
        *,
        equity_usd: float,
        daily_pnl_usd: float,
        open_positions: int,
        closed_today: int,
        win_rate_pct: Optional[float] = None,
    ) -> bool:
        """Discord-only formatted daily PnL summary."""
        if not self._any_channel_configured():
            self._warn_unconfigured_once()
            return False
        if not self.discord_webhook_url:
            return False
        fields: Dict[str, str] = {
            "Equity": f"${equity_usd:,.2f}",
            "Daily PnL": _format_signed_usd(daily_pnl_usd),
            "Open Positions": str(open_positions),
            "Closed Today": str(closed_today),
        }
        if win_rate_pct is not None:
            fields["Win Rate"] = f"{win_rate_pct:.1f}%"
        return self._post_discord(
            message="Daily Summary",
            severity="daily",
            fields=fields,
        )

    def fill_event(
        self,
        *,
        symbol: str,
        side: str,
        fill_price: float,
        fill_size: float,
        fees_usd: float,
    ) -> bool:
        """Discord-only short fill notification."""
        if not self._any_channel_configured():
            self._warn_unconfigured_once()
            return False
        if not self.discord_webhook_url:
            return False
        fields = {
            "Symbol": symbol,
            "Side": side.upper(),
            "Price": f"${fill_price:,.4f}",
            "Size": f"{fill_size:g}",
            "Fees": f"${fees_usd:,.4f}",
        }
        message = f"Fill: {side.upper()} {fill_size:g} {symbol} @ ${fill_price:,.4f}"
        return self._post_discord(
            message=message, severity="info", fields=fields
        )

    def kill_switch_tripped(self, reason: str) -> bool:
        """Critical-severity alert sent to BOTH channels."""
        return self.alert(
            f"KILL SWITCH TRIPPED: {reason}",
            severity="critical",
            fields={"Reason": reason},
        )

    # ------------------------------------------------------------------
    # Discord transport
    # ------------------------------------------------------------------
    def _build_discord_payload(
        self,
        *,
        message: str,
        severity: str,
        fields: Optional[Dict[str, str]],
    ) -> Dict[str, object]:
        title = _truncate(message, _DISCORD_TITLE_CAP)
        embed: Dict[str, object] = {
            "title": title,
            "color": _SEVERITY_COLORS.get(severity, _SEVERITY_COLORS["info"]),
            "timestamp": _utcnow_iso(),
        }
        if len(message) > _DISCORD_TITLE_CAP:
            embed["description"] = _truncate(message, _DISCORD_EMBED_DESC_CAP)
        if fields:
            embed["fields"] = [
                {
                    "name": str(k)[:256],
                    "value": _truncate(str(v), _DISCORD_FIELD_VALUE_CAP),
                    "inline": True,
                }
                for k, v in fields.items()
            ]
        return {"username": "autopilot", "embeds": [embed]}

    def _post_discord(
        self,
        *,
        message: str,
        severity: str,
        fields: Optional[Dict[str, str]],
    ) -> bool:
        if not self.discord_webhook_url:
            return False
        payload = self._build_discord_payload(
            message=message, severity=severity, fields=fields
        )
        try:
            resp = self._session.post(
                self.discord_webhook_url,
                json=payload,
                timeout=self.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort transport
            logger.warning("Discord notify failed (network): %s", exc)
            return False
        status = getattr(resp, "status_code", 0)
        if 200 <= status < 300:
            return True
        logger.warning(
            "Discord notify failed (HTTP %s): %s",
            status,
            getattr(resp, "text", ""),
        )
        return False

    # ------------------------------------------------------------------
    # Telegram transport
    # ------------------------------------------------------------------
    def _build_telegram_text(
        self,
        *,
        message: str,
        severity: str,
        fields: Optional[Dict[str, str]],
    ) -> str:
        # Severity badge sits at the top in bold (escaped per MarkdownV2 rules).
        badge = f"*\\[{_escape_telegram_md(severity.upper())}\\]*"
        body = _escape_telegram_md(message)
        lines = [f"{badge} {body}"]
        if fields:
            for k, v in fields.items():
                lines.append(
                    f"_{_escape_telegram_md(str(k))}_: "
                    f"{_escape_telegram_md(str(v))}"
                )
        text = "\n".join(lines)
        return _truncate(text, _TELEGRAM_TEXT_CAP)

    def _post_telegram(
        self,
        *,
        message: str,
        severity: str,
        fields: Optional[Dict[str, str]],
    ) -> bool:
        if not (self.telegram_bot_token and self.telegram_chat_id):
            return False
        url = (
            f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        )
        text = self._build_telegram_text(
            message=message, severity=severity, fields=fields
        )
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
        }
        try:
            resp = self._session.post(
                url,
                json=payload,
                timeout=self.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort transport
            logger.warning("Telegram notify failed (network): %s", exc)
            return False
        status = getattr(resp, "status_code", 0)
        if 200 <= status < 300:
            return True
        # Telegram returns a JSON body with a `description` field on failures.
        body_text = getattr(resp, "text", "")
        try:
            body_json = resp.json() if hasattr(resp, "json") else {}
            description = body_json.get("description") if isinstance(
                body_json, dict
            ) else None
        except (ValueError, json.JSONDecodeError):
            description = None
        logger.warning(
            "Telegram notify failed (HTTP %s): %s",
            status,
            description or body_text,
        )
        return False
