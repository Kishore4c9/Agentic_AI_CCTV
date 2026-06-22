"""Unit tests for the AlertSystem, PushNotificationChannel, and WebhookChannel."""

from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from agentic_cctv.alert_system import (
    AlertSystem,
    PushNotificationChannel,
    WebhookChannel,
)
from agentic_cctv.models import AlertPayload, CooldownConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_payload(
    camera_id: str = "cam-01",
    alert_type: str = "intrusion",
    alert_id: str = "alert-001",
) -> AlertPayload:
    return AlertPayload(
        alert_id=alert_id,
        event_id="evt-001",
        camera_id=camera_id,
        tenant_id="tenant-a",
        site_id="site-1",
        timestamp=datetime(2025, 1, 15, 14, 30, 0),
        alert_type=alert_type,
        description="Test alert",
        threat_level="medium",
        frame_crop_url=None,
    )


# ---------------------------------------------------------------------------
# PushNotificationChannel tests
# ---------------------------------------------------------------------------


class TestPushNotificationChannel:
    @pytest.mark.asyncio
    async def test_deliver_returns_true(self) -> None:
        channel = PushNotificationChannel()
        result = await channel.deliver(_make_payload())
        assert result is True


# ---------------------------------------------------------------------------
# WebhookChannel tests
# ---------------------------------------------------------------------------


class TestWebhookChannel:
    @pytest.mark.asyncio
    async def test_deliver_returns_true(self) -> None:
        channel = WebhookChannel(webhook_url="https://hooks.example.com/cctv")
        result = await channel.deliver(_make_payload())
        assert result is True

    def test_stores_webhook_url(self) -> None:
        url = "https://hooks.example.com/cctv"
        channel = WebhookChannel(webhook_url=url)
        assert channel.webhook_url == url


# ---------------------------------------------------------------------------
# AlertSystem — basic delivery
# ---------------------------------------------------------------------------


class TestAlertSystemDelivery:
    @pytest.mark.asyncio
    async def test_delivers_to_all_channels(self) -> None:
        push = PushNotificationChannel()
        webhook = WebhookChannel(webhook_url="https://example.com")
        system = AlertSystem(
            channels=[push, webhook],
            cooldown_config=CooldownConfig(default_seconds=60),
        )

        result = await system.send_alert(_make_payload())

        assert result.delivered is True
        assert "PushNotificationChannel" in result.channels
        assert "WebhookChannel" in result.channels
        assert result.suppressed is False
        assert result.suppressed_count == 0

    @pytest.mark.asyncio
    async def test_no_channels_returns_not_delivered(self) -> None:
        system = AlertSystem(channels=[], cooldown_config=CooldownConfig())
        result = await system.send_alert(_make_payload())

        assert result.delivered is False
        assert result.channels == []
        assert result.suppressed is False

    @pytest.mark.asyncio
    async def test_channel_failure_is_handled_gracefully(self) -> None:
        """A channel that raises should not prevent other channels from delivering."""
        failing_channel = AsyncMock(side_effect=RuntimeError("boom"))
        push = PushNotificationChannel()
        system = AlertSystem(
            channels=[failing_channel, push],
            cooldown_config=CooldownConfig(default_seconds=60),
        )

        result = await system.send_alert(_make_payload())

        assert result.delivered is True
        assert "PushNotificationChannel" in result.channels


# ---------------------------------------------------------------------------
# AlertSystem — cooldown deduplication
# ---------------------------------------------------------------------------


class TestAlertSystemCooldown:
    @pytest.mark.asyncio
    async def test_first_alert_is_delivered(self) -> None:
        system = AlertSystem(
            channels=[PushNotificationChannel()],
            cooldown_config=CooldownConfig(default_seconds=60),
        )
        result = await system.send_alert(_make_payload())
        assert result.delivered is True
        assert result.suppressed is False

    @pytest.mark.asyncio
    async def test_duplicate_within_cooldown_is_suppressed(self) -> None:
        system = AlertSystem(
            channels=[PushNotificationChannel()],
            cooldown_config=CooldownConfig(default_seconds=60),
        )
        await system.send_alert(_make_payload())

        # Second alert for same (camera_id, alert_type) should be suppressed
        result = await system.send_alert(_make_payload())
        assert result.delivered is False
        assert result.suppressed is True
        assert result.suppressed_count == 1

    @pytest.mark.asyncio
    async def test_suppressed_count_increments(self) -> None:
        system = AlertSystem(
            channels=[PushNotificationChannel()],
            cooldown_config=CooldownConfig(default_seconds=60),
        )
        await system.send_alert(_make_payload())

        # Send 3 more duplicates
        for i in range(1, 4):
            result = await system.send_alert(_make_payload())
            assert result.suppressed is True
            assert result.suppressed_count == i

    @pytest.mark.asyncio
    async def test_different_event_types_not_suppressed(self) -> None:
        system = AlertSystem(
            channels=[PushNotificationChannel()],
            cooldown_config=CooldownConfig(default_seconds=60),
        )
        await system.send_alert(_make_payload(alert_type="intrusion"))

        # Different event type should NOT be suppressed
        result = await system.send_alert(_make_payload(alert_type="fire"))
        assert result.delivered is True
        assert result.suppressed is False

    @pytest.mark.asyncio
    async def test_different_cameras_not_suppressed(self) -> None:
        system = AlertSystem(
            channels=[PushNotificationChannel()],
            cooldown_config=CooldownConfig(default_seconds=60),
        )
        await system.send_alert(_make_payload(camera_id="cam-01"))

        # Different camera should NOT be suppressed
        result = await system.send_alert(_make_payload(camera_id="cam-02"))
        assert result.delivered is True
        assert result.suppressed is False

    @pytest.mark.asyncio
    async def test_cooldown_expires_allows_new_alert(self) -> None:
        """After the cooldown period elapses, a new alert should be delivered."""
        system = AlertSystem(
            channels=[PushNotificationChannel()],
            cooldown_config=CooldownConfig(default_seconds=60),
        )
        await system.send_alert(_make_payload())

        # Simulate time passing beyond cooldown by patching time.monotonic
        original_monotonic = time.monotonic
        with patch("agentic_cctv.alert_system.time.monotonic") as mock_mono:
            mock_mono.return_value = original_monotonic() + 61
            result = await system.send_alert(_make_payload())

        assert result.delivered is True
        assert result.suppressed is False

    @pytest.mark.asyncio
    async def test_per_type_override_cooldown(self) -> None:
        """Per-type cooldown overrides should take precedence over default."""
        system = AlertSystem(
            channels=[PushNotificationChannel()],
            cooldown_config=CooldownConfig(
                default_seconds=60,
                per_type_overrides={"fire": 10},
            ),
        )
        await system.send_alert(_make_payload(alert_type="fire"))

        # Within 10s cooldown for fire — should be suppressed
        result = await system.send_alert(_make_payload(alert_type="fire"))
        assert result.suppressed is True

        # After 10s but before 60s — fire should be delivered
        original_monotonic = time.monotonic
        with patch("agentic_cctv.alert_system.time.monotonic") as mock_mono:
            mock_mono.return_value = original_monotonic() + 11
            result = await system.send_alert(_make_payload(alert_type="fire"))
        assert result.delivered is True
        assert result.suppressed is False

    @pytest.mark.asyncio
    async def test_exact_cooldown_boundary(self) -> None:
        """At exactly the cooldown boundary, the alert should still be in cooldown
        (elapsed < cooldown_seconds uses strict less-than)."""
        system = AlertSystem(
            channels=[PushNotificationChannel()],
            cooldown_config=CooldownConfig(default_seconds=60),
        )

        # Record the time of the first alert
        base_time = time.monotonic()
        with patch("agentic_cctv.alert_system.time.monotonic", return_value=base_time):
            await system.send_alert(_make_payload())

        # At exactly 60s elapsed — still in cooldown (< not <=)
        with patch(
            "agentic_cctv.alert_system.time.monotonic",
            return_value=base_time + 60,
        ):
            assert system.is_in_cooldown("cam-01", "intrusion") is False

        # At 59.999s — still in cooldown
        with patch(
            "agentic_cctv.alert_system.time.monotonic",
            return_value=base_time + 59.999,
        ):
            assert system.is_in_cooldown("cam-01", "intrusion") is True


# ---------------------------------------------------------------------------
# AlertSystem — is_in_cooldown
# ---------------------------------------------------------------------------


class TestIsInCooldown:
    def test_no_prior_alert_returns_false(self) -> None:
        system = AlertSystem(
            channels=[], cooldown_config=CooldownConfig(default_seconds=60)
        )
        assert system.is_in_cooldown("cam-01", "intrusion") is False

    @pytest.mark.asyncio
    async def test_after_alert_returns_true(self) -> None:
        system = AlertSystem(
            channels=[PushNotificationChannel()],
            cooldown_config=CooldownConfig(default_seconds=60),
        )
        await system.send_alert(_make_payload())
        assert system.is_in_cooldown("cam-01", "intrusion") is True
