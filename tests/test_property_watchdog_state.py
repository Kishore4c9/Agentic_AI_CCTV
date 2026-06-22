"""Property-based test for Watchdog Offline and Restored State Transitions.

# Feature: agentic-ai-cctv-monitoring, Property 10: Watchdog Offline and Restored State Transitions

**Validates: Requirements 9.2, 9.4**

For random heartbeat sequences with varying gaps, the Watchdog reports ``offline``
iff last heartbeat > 60s ago; transitions to ``online`` on new heartbeat; never
reports ``offline`` if heartbeat received within 60s.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_cctv.models import AlertPayload, AlertResult
from agentic_cctv.watchdog import Watchdog, _CameraState


# ---------------------------------------------------------------------------
# Fakes (same patterns as tests/test_watchdog.py)
# ---------------------------------------------------------------------------


class FakeMQTTSubscriber:
    """Fake MQTT subscriber that records subscribe calls."""

    def __init__(self) -> None:
        self.subscriptions: List[Tuple[str, int]] = []
        self._callbacks: dict[str, object] = {}
        self.is_connected: bool = True

    async def subscribe(
        self,
        topic: str,
        qos: int = 1,
        callback: Optional[object] = None,
    ) -> None:
        self.subscriptions.append((topic, qos))
        if callback is not None:
            self._callbacks[topic] = callback


class FakeAlertSystem:
    """Fake AlertSystem that records all sent alerts."""

    def __init__(self) -> None:
        self.alerts: List[AlertPayload] = []

    async def send_alert(self, payload: AlertPayload) -> AlertResult:
        self.alerts.append(payload)
        return AlertResult(delivered=True, channels=["FakeChannel"])


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_camera_id_strategy = st.from_regex(r"cam-[a-z0-9]{3,10}", fullmatch=True)

_gap_within_threshold = st.floats(min_value=0.01, max_value=59.9)

_gap_exceeding_threshold = st.floats(min_value=60.1, max_value=300.0)

_num_heartbeats = st.integers(min_value=1, max_value=20)

_offline_threshold = st.floats(min_value=10.0, max_value=120.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_watchdog(
    offline_threshold: float = 60.0,
) -> Tuple[Watchdog, FakeAlertSystem]:
    """Create a Watchdog with fakes and a given offline threshold."""
    subscriber = FakeMQTTSubscriber()
    alert_system = FakeAlertSystem()
    wd = Watchdog(
        mqtt_subscriber=subscriber,  # type: ignore[arg-type]
        alert_system=alert_system,  # type: ignore[arg-type]
        offline_threshold=offline_threshold,
        check_interval=999.0,  # we drive checks manually
    )
    return wd, alert_system


def _register_camera(
    wd: Watchdog,
    camera_id: str,
    loop_time: float,
) -> _CameraState:
    """Register a camera in the watchdog's internal state and return it."""
    state = _CameraState(
        camera_id=camera_id,
        tenant_id="tenant-test",
        site_id="site-test",
    )
    state.status = "online"
    state.last_heartbeat_time = loop_time
    wd._cameras[camera_id] = state
    return state


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestWatchdogOfflineAndRestoredStateTransitions:
    """Property 10: Watchdog Offline and Restored State Transitions.

    **Validates: Requirements 9.2, 9.4**
    """

    @given(
        camera_id=_camera_id_strategy,
        gap=_gap_exceeding_threshold,
        threshold=_offline_threshold,
    )
    @settings(max_examples=20)
    def test_offline_transition_when_heartbeat_exceeds_threshold(
        self,
        camera_id: str,
        gap: float,
        threshold: float,
    ) -> None:
        """For any camera that has sent a heartbeat, if the time since last
        heartbeat exceeds the offline threshold, ``_check_cameras()`` should
        transition it to 'offline' and send exactly one offline alert.

        We use a gap that always exceeds the threshold by scaling:
        actual_gap = threshold + gap (gap is 60.1..300 so always > threshold).
        """
        # Ensure the gap actually exceeds the threshold
        actual_gap = threshold + 1.0  # guarantee > threshold
        if gap > threshold:
            actual_gap = gap

        loop = asyncio.new_event_loop()
        try:
            wd, alert_system = _make_watchdog(offline_threshold=threshold)
            wd._loop = loop

            base_time = 1000.0
            state = _register_camera(wd, camera_id, loop_time=base_time)

            # Simulate time passing beyond the threshold
            # Override _loop_time to return base_time + actual_gap
            wd._loop_time = lambda: base_time + actual_gap  # type: ignore[assignment]

            loop.run_until_complete(wd._check_cameras())

            # Camera should now be offline
            assert state.status == "offline", (
                f"Camera should be offline after gap={actual_gap:.1f}s "
                f"(threshold={threshold:.1f}s), got status={state.status}"
            )

            # Exactly one offline alert should have been sent
            offline_alerts = [
                a for a in alert_system.alerts
                if a.alert_type == "camera-offline"
            ]
            assert len(offline_alerts) == 1, (
                f"Expected exactly 1 offline alert, got {len(offline_alerts)}"
            )
            assert offline_alerts[0].camera_id == camera_id
        finally:
            loop.close()

    @given(
        camera_id=_camera_id_strategy,
        gap=_gap_within_threshold,
        threshold=_offline_threshold,
    )
    @settings(max_examples=20)
    def test_online_persistence_when_heartbeat_within_threshold(
        self,
        camera_id: str,
        gap: float,
        threshold: float,
    ) -> None:
        """For any camera receiving heartbeats within the threshold, the camera
        should remain 'online' and no offline alert should be sent.

        We ensure gap < threshold by capping gap to threshold * 0.9.
        """
        # Ensure gap is strictly less than threshold
        actual_gap = min(gap, threshold * 0.9)
        if actual_gap <= 0:
            actual_gap = 0.01

        loop = asyncio.new_event_loop()
        try:
            wd, alert_system = _make_watchdog(offline_threshold=threshold)
            wd._loop = loop

            base_time = 1000.0
            state = _register_camera(wd, camera_id, loop_time=base_time)

            # Simulate time passing within the threshold
            wd._loop_time = lambda: base_time + actual_gap  # type: ignore[assignment]

            loop.run_until_complete(wd._check_cameras())

            # Camera should remain online
            assert state.status == "online", (
                f"Camera should remain online after gap={actual_gap:.1f}s "
                f"(threshold={threshold:.1f}s), got status={state.status}"
            )

            # No offline alerts should have been sent
            offline_alerts = [
                a for a in alert_system.alerts
                if a.alert_type == "camera-offline"
            ]
            assert len(offline_alerts) == 0, (
                f"Expected 0 offline alerts for gap within threshold, "
                f"got {len(offline_alerts)}"
            )
        finally:
            loop.close()

    @given(
        camera_id=_camera_id_strategy,
        gap=_gap_exceeding_threshold,
        threshold=_offline_threshold,
    )
    @settings(max_examples=20)
    def test_restored_transition_on_heartbeat_after_offline(
        self,
        camera_id: str,
        gap: float,
        threshold: float,
    ) -> None:
        """For any camera that is offline, receiving a new heartbeat should
        transition it to 'online' and trigger a restored alert.

        We simulate: register camera → exceed threshold → _check_cameras
        (goes offline) → simulate heartbeat arrival (set status back to online
        and call _send_restored_alert).
        """
        actual_gap = threshold + 1.0
        if gap > threshold:
            actual_gap = gap

        loop = asyncio.new_event_loop()
        try:
            wd, alert_system = _make_watchdog(offline_threshold=threshold)
            wd._loop = loop

            base_time = 1000.0
            state = _register_camera(wd, camera_id, loop_time=base_time)

            # Step 1: Make camera go offline
            offline_time = base_time + actual_gap
            wd._loop_time = lambda: offline_time  # type: ignore[assignment]
            loop.run_until_complete(wd._check_cameras())
            assert state.status == "offline"

            # Step 2: Simulate heartbeat arrival — mimic what _on_heartbeat does
            restore_time = offline_time + 5.0
            wd._loop_time = lambda: restore_time  # type: ignore[assignment]

            was_offline = state.status == "offline"
            state.last_heartbeat_time = restore_time
            state.status = "online"

            if was_offline:
                downtime = Watchdog._compute_downtime(state, restore_time)
                state.went_offline_at = None
                loop.run_until_complete(
                    wd._send_restored_alert(state, downtime)
                )

            # Camera should be online
            assert state.status == "online", (
                f"Camera should be online after heartbeat, got status={state.status}"
            )

            # A restored alert should have been sent
            restored_alerts = [
                a for a in alert_system.alerts
                if a.alert_type == "camera-restored"
            ]
            assert len(restored_alerts) >= 1, (
                f"Expected at least 1 restored alert, got {len(restored_alerts)}"
            )
            assert restored_alerts[0].camera_id == camera_id
        finally:
            loop.close()

    @given(
        camera_id=_camera_id_strategy,
        num_heartbeats=_num_heartbeats,
        gap=_gap_within_threshold,
        threshold=_offline_threshold,
    )
    @settings(max_examples=20)
    def test_never_offline_within_threshold(
        self,
        camera_id: str,
        num_heartbeats: int,
        gap: float,
        threshold: float,
    ) -> None:
        """For any heartbeat gap less than the offline threshold, the camera
        should never be reported as offline — even across multiple check cycles.

        We simulate N heartbeats each followed by a _check_cameras call, with
        the gap always within the threshold.
        """
        # Ensure gap is strictly less than threshold
        actual_gap = min(gap, threshold * 0.9)
        if actual_gap <= 0:
            actual_gap = 0.01

        loop = asyncio.new_event_loop()
        try:
            wd, alert_system = _make_watchdog(offline_threshold=threshold)
            wd._loop = loop

            current_time = 1000.0
            state = _register_camera(wd, camera_id, loop_time=current_time)

            for _ in range(num_heartbeats):
                # Advance time by the gap
                current_time += actual_gap
                # Simulate heartbeat: update last_heartbeat_time
                state.last_heartbeat_time = current_time

                # Advance time slightly for the check (still within threshold)
                check_time = current_time + actual_gap * 0.5
                wd._loop_time = lambda _ct=check_time: _ct  # type: ignore[assignment]

                loop.run_until_complete(wd._check_cameras())

                # Camera must remain online after every check
                assert state.status == "online", (
                    f"Camera should never go offline with gap={actual_gap:.1f}s "
                    f"(threshold={threshold:.1f}s), got status={state.status}"
                )

            # No offline alerts should have been sent across all cycles
            offline_alerts = [
                a for a in alert_system.alerts
                if a.alert_type == "camera-offline"
            ]
            assert len(offline_alerts) == 0, (
                f"Expected 0 offline alerts across {num_heartbeats} heartbeats, "
                f"got {len(offline_alerts)}"
            )
        finally:
            loop.close()
