"""Tests for src/alerts/notifier.py.

The notifier is best-effort: it must never raise, must degrade gracefully when
no channel is configured, and must never make a real HTTP call from tests.
We feed every test a stub ``Session`` whose ``.post`` is recorded so we can
assert the request shape (URL, JSON body, timeout).
"""
from __future__ import annotations

import logging
import os
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from alerts import AlertSeverity, Notifier
from alerts import notifier as notifier_module


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 204,
        text: str = "",
        json_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._json = json_body or {}

    def json(self) -> Dict[str, Any]:
        return self._json


class _FakeSession:
    """Records every .post() and returns scripted responses."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.responses: List[Any] = []  # _FakeResponse OR Exception
        self.headers: Dict[str, str] = {}

    def queue(self, *responses: Any) -> None:
        self.responses.extend(responses)

    def post(self, url: str, json: Optional[Dict[str, Any]] = None,
             timeout: Optional[float] = None, **_: Any) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if not self.responses:
            return _FakeResponse(status_code=204)
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _clear_alert_env() -> Dict[str, Optional[str]]:
    """Strip the three alert env vars; return originals for restoration."""
    keys = ("DISCORD_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    return saved


def _restore_env(saved: Dict[str, Optional[str]]) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _EnvIsolatedTestCase(unittest.TestCase):
    """Base class that scrubs alert env vars per-test."""

    def setUp(self) -> None:
        self._saved_env = _clear_alert_env()

    def tearDown(self) -> None:
        _restore_env(self._saved_env)


class InitTests(_EnvIsolatedTestCase):
    def test_init_reads_env_vars_when_args_omitted(self) -> None:
        os.environ["DISCORD_WEBHOOK_URL"] = "https://discord.example/wh"
        os.environ["TELEGRAM_BOT_TOKEN"] = "bot:token"
        os.environ["TELEGRAM_CHAT_ID"] = "12345"
        n = Notifier(session=_FakeSession())
        self.assertEqual(n.discord_webhook_url, "https://discord.example/wh")
        self.assertEqual(n.telegram_bot_token, "bot:token")
        self.assertEqual(n.telegram_chat_id, "12345")
        self.assertEqual(
            n.is_configured(), {"discord": True, "telegram": True}
        )


class UnconfiguredTests(_EnvIsolatedTestCase):
    def test_no_channels_configured_returns_false_and_logs_once(self) -> None:
        n = Notifier(session=_FakeSession())
        with self.assertLogs(notifier_module.logger, level="WARNING") as cap:
            self.assertFalse(n.info("hello"))
            self.assertFalse(n.alert("uh oh"))
            self.assertFalse(n.info("again"))
        # Exactly one WARNING about being unconfigured even though we called
        # the API three times.
        unconfigured_warnings = [
            r for r in cap.records if "no channel configured" in r.getMessage()
        ]
        self.assertEqual(len(unconfigured_warnings), 1)


class DiscordPathTests(_EnvIsolatedTestCase):
    def _notifier(self, session: _FakeSession) -> Notifier:
        return Notifier(
            discord_webhook_url="https://discord.example/wh",
            session=session,
        )

    def test_info_posts_to_discord(self) -> None:
        s = _FakeSession()
        s.queue(_FakeResponse(204))
        n = self._notifier(s)
        ok = n.info("scanner started", fields={"top": "5"})
        self.assertTrue(ok)
        self.assertEqual(len(s.calls), 1)
        call = s.calls[0]
        self.assertEqual(call["url"], "https://discord.example/wh")
        self.assertEqual(call["timeout"], 5.0)
        body = call["json"]
        self.assertEqual(body["username"], "autopilot")
        embeds = body["embeds"]
        self.assertEqual(len(embeds), 1)
        embed = embeds[0]
        self.assertEqual(embed["title"], "scanner started")
        # info severity color is blue (0x3498DB).
        self.assertEqual(embed["color"], 0x3498DB)
        # Fields preserved.
        self.assertEqual(
            embed["fields"],
            [{"name": "top", "value": "5", "inline": True}],
        )
        self.assertIn("timestamp", embed)

    def test_info_returns_false_on_discord_http_error(self) -> None:
        s = _FakeSession()
        s.queue(_FakeResponse(status_code=400, text="Bad Request"))
        n = self._notifier(s)
        with self.assertLogs(notifier_module.logger, level="WARNING"):
            self.assertFalse(n.info("oops"))

    def test_info_returns_false_on_network_exception(self) -> None:
        s = _FakeSession()
        s.queue(ConnectionError("dns fail"))
        n = self._notifier(s)
        with self.assertLogs(notifier_module.logger, level="WARNING"):
            self.assertFalse(n.info("oops"))

    def test_daily_summary_formats_pnl_with_sign(self) -> None:
        s = _FakeSession()
        s.queue(_FakeResponse(204), _FakeResponse(204))
        n = self._notifier(s)

        self.assertTrue(n.daily_summary(
            equity_usd=10_000.0,
            daily_pnl_usd=123.45,
            open_positions=2,
            closed_today=4,
            win_rate_pct=60.0,
        ))
        self.assertTrue(n.daily_summary(
            equity_usd=10_000.0,
            daily_pnl_usd=-50.0,
            open_positions=0,
            closed_today=1,
        ))
        self.assertEqual(len(s.calls), 2)
        positive_fields = {
            f["name"]: f["value"]
            for f in s.calls[0]["json"]["embeds"][0]["fields"]
        }
        negative_fields = {
            f["name"]: f["value"]
            for f in s.calls[1]["json"]["embeds"][0]["fields"]
        }
        self.assertEqual(positive_fields["Daily PnL"], "+$123.45")
        self.assertEqual(negative_fields["Daily PnL"], "-$50.00")
        self.assertIn("Win Rate", positive_fields)
        self.assertNotIn("Win Rate", negative_fields)
        # Daily summary uses the green "daily" color.
        self.assertEqual(
            s.calls[0]["json"]["embeds"][0]["color"], 0x2ECC71
        )

    def test_fill_event_includes_symbol_side_price_size(self) -> None:
        s = _FakeSession()
        s.queue(_FakeResponse(204))
        n = self._notifier(s)
        self.assertTrue(n.fill_event(
            symbol="ETH/USDT",
            side="buy",
            fill_price=3210.5,
            fill_size=0.05,
            fees_usd=0.1605,
        ))
        embed = s.calls[0]["json"]["embeds"][0]
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        self.assertEqual(fields["Symbol"], "ETH/USDT")
        self.assertEqual(fields["Side"], "BUY")
        self.assertIn("3,210.5", fields["Price"])
        self.assertEqual(fields["Size"], "0.05")
        self.assertIn("ETH/USDT", embed["title"])
        self.assertIn("BUY", embed["title"])


class AlertRoutingTests(_EnvIsolatedTestCase):
    def _both_channel_notifier(self, session: _FakeSession) -> Notifier:
        return Notifier(
            discord_webhook_url="https://discord.example/wh",
            telegram_bot_token="bot:tok",
            telegram_chat_id="42",
            session=session,
        )

    def test_alert_sends_to_both_channels(self) -> None:
        s = _FakeSession()
        s.queue(_FakeResponse(204), _FakeResponse(200, json_body={"ok": True}))
        n = self._both_channel_notifier(s)
        self.assertTrue(n.alert("market halted"))
        self.assertEqual(len(s.calls), 2)
        urls = [c["url"] for c in s.calls]
        self.assertIn("https://discord.example/wh", urls)
        self.assertTrue(any("api.telegram.org/botbot:tok/sendMessage" in u
                            for u in urls))

    def test_alert_returns_true_when_at_least_one_succeeds(self) -> None:
        s = _FakeSession()
        # Discord OK, Telegram fails.
        s.queue(
            _FakeResponse(204),
            _FakeResponse(status_code=400, text="Bad", json_body={
                "ok": False, "description": "chat not found"
            }),
        )
        n = self._both_channel_notifier(s)
        with self.assertLogs(notifier_module.logger, level="WARNING"):
            self.assertTrue(n.alert("partial outage"))

    def test_alert_returns_false_when_both_fail(self) -> None:
        s = _FakeSession()
        s.queue(
            _FakeResponse(status_code=500, text="discord down"),
            ConnectionError("telegram unreachable"),
        )
        n = self._both_channel_notifier(s)
        with self.assertLogs(notifier_module.logger, level="WARNING"):
            self.assertFalse(n.alert("everything is on fire"))

    def test_alert_severity_badge_in_discord_embed(self) -> None:
        # Run the same alert at four different severities and confirm the
        # embed color tracks the severity map exactly.
        expected: Dict[AlertSeverity, int] = {
            "info": 0x3498DB,
            "warning": 0xF1C40F,
            "alert": 0xE67E22,
            "critical": 0xE74C3C,
        }
        for severity, color in expected.items():
            s = _FakeSession()
            # Queue Discord response only; telegram unconfigured here.
            s.queue(_FakeResponse(204))
            n = Notifier(
                discord_webhook_url="https://discord.example/wh",
                session=s,
            )
            self.assertTrue(n.alert("issue", severity=severity))
            embed = s.calls[0]["json"]["embeds"][0]
            self.assertEqual(
                embed["color"], color,
                msg=f"severity={severity}",
            )

    def test_kill_switch_tripped_uses_critical_severity_in_both_channels(
        self,
    ) -> None:
        s = _FakeSession()
        s.queue(_FakeResponse(204), _FakeResponse(200, json_body={"ok": True}))
        n = self._both_channel_notifier(s)
        self.assertTrue(n.kill_switch_tripped("daily loss limit breached"))
        self.assertEqual(len(s.calls), 2)
        # Discord call -> critical color (red).
        discord_call = next(
            c for c in s.calls
            if c["url"] == "https://discord.example/wh"
        )
        self.assertEqual(
            discord_call["json"]["embeds"][0]["color"], 0xE74C3C,
        )
        # Telegram call -> CRITICAL appears in bracketed badge.
        tg_call = next(
            c for c in s.calls if "api.telegram.org" in c["url"]
        )
        self.assertIn("CRITICAL", tg_call["json"]["text"])
        self.assertIn("KILL SWITCH TRIPPED", tg_call["json"]["text"])


class TelegramFormatTests(_EnvIsolatedTestCase):
    def test_telegram_markdown_v2_escapes_special_chars(self) -> None:
        s = _FakeSession()
        s.queue(_FakeResponse(200, json_body={"ok": True}))
        n = Notifier(
            telegram_bot_token="bot:tok",
            telegram_chat_id="42",
            session=s,
        )
        # Message contains *, _, and a period — all reserved in MarkdownV2.
        self.assertTrue(n.alert("price *moved* _fast_ today."))
        body = s.calls[0]["json"]
        self.assertEqual(body["parse_mode"], "MarkdownV2")
        text = body["text"]
        # The literal asterisks and underscores in the message body must be
        # backslash-escaped — but the badge formatting `*[...]* ` we add
        # ourselves stays intact.
        self.assertIn("\\*moved\\*", text)
        self.assertIn("\\_fast\\_", text)
        self.assertIn("today\\.", text)
        # chat_id is forwarded verbatim.
        self.assertEqual(body["chat_id"], "42")


class IsConfiguredTests(_EnvIsolatedTestCase):
    def test_is_configured_reports_per_channel_status(self) -> None:
        # Nothing set.
        self.assertEqual(
            Notifier(session=_FakeSession()).is_configured(),
            {"discord": False, "telegram": False},
        )
        # Discord only.
        self.assertEqual(
            Notifier(
                discord_webhook_url="https://discord.example/wh",
                session=_FakeSession(),
            ).is_configured(),
            {"discord": True, "telegram": False},
        )
        # Telegram requires BOTH token and chat_id.
        self.assertEqual(
            Notifier(
                telegram_bot_token="bot:tok",
                session=_FakeSession(),
            ).is_configured(),
            {"discord": False, "telegram": False},
        )
        self.assertEqual(
            Notifier(
                telegram_bot_token="bot:tok",
                telegram_chat_id="42",
                session=_FakeSession(),
            ).is_configured(),
            {"discord": False, "telegram": True},
        )
        # Both.
        self.assertEqual(
            Notifier(
                discord_webhook_url="https://discord.example/wh",
                telegram_bot_token="bot:tok",
                telegram_chat_id="42",
                session=_FakeSession(),
            ).is_configured(),
            {"discord": True, "telegram": True},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
