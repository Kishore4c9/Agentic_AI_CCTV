"""Property-based test for Alert Cooldown Deduplication.

# Feature: agentic-ai-cctv-monitoring, Property 9: Alert Cooldown Deduplication

**Validates: Requirements 8.3, 8.4**

For random alert sequences with the same (camera_id, event_type), only the first
alert is delivered; subsequent alerts within the cooldown period are suppressed;
the suppressed count equals the number of suppressed duplicates.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_cctv.alert_system import AlertSystem, PushNotificationChannel
from agentic_cctv.models import AlertPayload, CooldownConfig

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_num_duplicates_strategy = st.integers(min_value=2, max_value=20)

_camera_id_strategy = st.from_regex(r"cam-[a-z0-9]{3,10}", fullmatch=True)

_alert_type_strategy = st.sampled_from(
    ["intrusion", "fire", "loitering", "vehicle", "package", "person"]
)

_cooldown_seconds_strategy = st.integers(min_value=10, max_value=300)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_payload(camera_id: str, alert_type: str) -> AlertPayload:
    """Create an AlertPayload with the given camera_id and alert_type."""
    return AlertPayload(
        alert_id=f"alert-{uuid.uuid4().hex[:8]}",
        event_id=f"evt-{uuid.uuid4().hex[:8]}",
        camera_id=camera_id,
        tenant_id="tenant-test",
        site_id="site-test",
        timestamp=datetime(2025, 1, 15, 14, 30, 0),
        alert_type=alert_type,
        description="Property test alert",
        threat_level="medium",
        frame_crop_url=None,
    )


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestAlertCooldownDeduplication:
    """Property 9: Alert Cooldown Deduplication.

    **Validates: Requirements 8.3, 8.4**
    """

    @given(
        num_alerts=_num_duplicates_strategy,
        camera_id=_camera_id_strategy,
        alert_type=_alert_type_strategy,
        cooldown_seconds=_cooldown_seconds_strategy,
    )
    @settings(max_examples=20)
    def test_cooldown_deduplication(
        self,
        num_alerts: int,
        camera_id: str,
        alert_type: str,
        cooldown_seconds: int,
    ) -> None:
        """For any random number of duplicate alerts (2-20) with the same
        (camera_id, alert_type) and a random cooldown period (10-300s):

        1. Create an AlertSystem with a PushNotificationChannel and the
           random cooldown.
        2. Send the first alert — assert it is delivered (not suppressed).
        3. Send N-1 more alerts with the same (camera_id, alert_type) —
           assert all are suppressed.
        4. Assert the final suppressed_count equals N-1.
        """
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                self._run_deduplication_scenario(
                    num_alerts, camera_id, alert_type, cooldown_seconds
                )
            )
        finally:
            loop.close()

    async def _run_deduplication_scenario(
        self,
        num_alerts: int,
        camera_id: str,
        alert_type: str,
        cooldown_seconds: int,
    ) -> None:
        # 1. Create AlertSystem with PushNotificationChannel and random cooldown
        system = AlertSystem(
            channels=[PushNotificationChannel()],
            cooldown_config=CooldownConfig(default_seconds=cooldown_seconds),
        )

        # 2. Send the first alert — must be delivered, not suppressed
        first_payload = _make_payload(camera_id, alert_type)
        first_result = await system.send_alert(first_payload)

        assert first_result.delivered is True, (
            f"First alert should be delivered, got delivered={first_result.delivered}"
        )
        assert first_result.suppressed is False, (
            f"First alert should not be suppressed, got suppressed={first_result.suppressed}"
        )
        assert first_result.suppressed_count == 0, (
            f"First alert suppressed_count should be 0, got {first_result.suppressed_count}"
        )

        # 3. Send N-1 more alerts — all should be suppressed
        last_result = None
        for i in range(1, num_alerts):
            payload = _make_payload(camera_id, alert_type)
            result = await system.send_alert(payload)

            assert result.delivered is False, (
                f"Alert {i+1}/{num_alerts} should not be delivered, "
                f"got delivered={result.delivered}"
            )
            assert result.suppressed is True, (
                f"Alert {i+1}/{num_alerts} should be suppressed, "
                f"got suppressed={result.suppressed}"
            )
            assert result.suppressed_count == i, (
                f"Alert {i+1}/{num_alerts} suppressed_count should be {i}, "
                f"got {result.suppressed_count}"
            )
            last_result = result

        # 4. Final suppressed_count equals N-1
        expected_suppressed = num_alerts - 1
        assert last_result is not None, "Should have sent at least one duplicate"
        assert last_result.suppressed_count == expected_suppressed, (
            f"After sending {num_alerts} total alerts, "
            f"suppressed_count should be {expected_suppressed}, "
            f"got {last_result.suppressed_count}"
        )
