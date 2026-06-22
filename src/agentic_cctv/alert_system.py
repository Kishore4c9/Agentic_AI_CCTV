"""Alert System with pluggable channels and cooldown deduplication.

Delivers alerts via configurable channels (push notification, webhook, etc.)
and suppresses duplicate alerts within a configurable cooldown period per
(camera_id, event_type) combination.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol, Tuple

from agentic_cctv.models import AlertPayload, AlertResult, CooldownConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AlertChannel protocol and concrete channel implementations
# ---------------------------------------------------------------------------


class AlertChannel(Protocol):
    """Protocol for pluggable alert delivery channels."""

    async def deliver(self, payload: AlertPayload) -> bool:
        """Deliver an alert payload. Return True on success."""
        ...


class PushNotificationChannel:
    """Stub push notification channel that logs the alert.

    Real push notification integration is deferred to a later phase.
    """

    async def deliver(self, payload: AlertPayload) -> bool:
        logger.info(
            "PushNotification: alert_id=%s camera=%s type=%s threat=%s — %s",
            payload.alert_id,
            payload.camera_id,
            payload.alert_type,
            payload.threat_level,
            payload.description,
        )
        return True


class WebhookChannel:
    """Stub webhook channel that logs the alert.

    Real webhook HTTP delivery is deferred to a later phase.
    """

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    async def deliver(self, payload: AlertPayload) -> bool:
        logger.info(
            "Webhook [%s]: alert_id=%s camera=%s type=%s threat=%s — %s",
            self.webhook_url,
            payload.alert_id,
            payload.camera_id,
            payload.alert_type,
            payload.threat_level,
            payload.description,
        )
        return True


class MobilePushChannel:
    """Mobile push notification channel that delivers alerts to registered devices.

    Stub implementation for v1 that logs the alert delivery.  Real push
    notification integration (APNs for iOS, FCM for Android) is deferred
    to a later phase.

    Parameters
    ----------
    device_token:
        The device push notification token.
    platform:
        The mobile platform (``"ios"`` or ``"android"``).
    """

    def __init__(self, device_token: str, platform: str) -> None:
        self.device_token = device_token
        self.platform = platform

    async def deliver(self, payload: AlertPayload) -> bool:
        """Deliver an alert as a mobile push notification (stub).

        Logs the delivery attempt.  In production this would call APNs
        or FCM depending on :attr:`platform`.
        """
        logger.info(
            "MobilePush [%s/%s]: alert_id=%s camera=%s type=%s threat=%s — %s",
            self.platform,
            self.device_token[:8],
            payload.alert_id,
            payload.camera_id,
            payload.alert_type,
            payload.threat_level,
            payload.description,
        )
        return True


# ---------------------------------------------------------------------------
# AlertSystem with cooldown deduplication
# ---------------------------------------------------------------------------


class AlertSystem:
    """Delivers alerts through configured channels with cooldown deduplication.

    Cooldown deduplication prevents alert fatigue by suppressing duplicate
    alerts for the same ``(camera_id, event_type)`` within a configurable
    cooldown window.  Suppressed alerts increment a counter on the original
    alert rather than being delivered again.

    Parameters
    ----------
    channels:
        Ordered list of :class:`AlertChannel` implementations to deliver
        alerts through.
    cooldown_config:
        Cooldown timing configuration (default period and per-type overrides).
    """

    def __init__(
        self,
        channels: list[AlertChannel],
        cooldown_config: CooldownConfig,
    ) -> None:
        self.channels = channels
        self.cooldown_config = cooldown_config

        # In-memory cooldown map:
        #   (camera_id, event_type) → (last_alert_monotonic_time, suppressed_count)
        self._cooldown_map: dict[Tuple[str, str], Tuple[float, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_alert(self, payload: AlertPayload) -> AlertResult:
        """Send an alert through all configured channels.

        If the ``(camera_id, alert_type)`` combination is currently in
        cooldown the alert is suppressed and the suppressed counter is
        incremented.  Otherwise the alert is delivered via every channel
        and the cooldown timer is recorded.
        """
        key = (payload.camera_id, payload.alert_type)

        if self.is_in_cooldown(payload.camera_id, payload.alert_type):
            # Suppress: increment counter on the original alert entry
            last_time, count = self._cooldown_map[key]
            self._cooldown_map[key] = (last_time, count + 1)
            logger.debug(
                "Alert suppressed (cooldown): camera=%s type=%s suppressed_count=%d",
                payload.camera_id,
                payload.alert_type,
                count + 1,
            )
            return AlertResult(
                delivered=False,
                channels=[],
                suppressed=True,
                suppressed_count=count + 1,
            )

        # Deliver through all channels
        delivered_channels: list[str] = []
        for channel in self.channels:
            try:
                success = await channel.deliver(payload)
                if success:
                    delivered_channels.append(type(channel).__name__)
            except Exception:
                logger.exception(
                    "Channel %s failed for alert %s",
                    type(channel).__name__,
                    payload.alert_id,
                )

        # Record cooldown timestamp (monotonic clock)
        self._cooldown_map[key] = (time.monotonic(), 0)

        delivered = len(delivered_channels) > 0
        return AlertResult(
            delivered=delivered,
            channels=delivered_channels,
            suppressed=False,
            suppressed_count=0,
        )

    def is_in_cooldown(self, camera_id: str, event_type: str) -> bool:
        """Return ``True`` if the last alert for *(camera_id, event_type)*
        was within the cooldown period."""
        key = (camera_id, event_type)
        entry = self._cooldown_map.get(key)
        if entry is None:
            return False

        last_time, _ = entry
        cooldown_seconds = self.cooldown_config.per_type_overrides.get(
            event_type, self.cooldown_config.default_seconds
        )
        elapsed = time.monotonic() - last_time
        return elapsed < cooldown_seconds
