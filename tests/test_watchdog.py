"""Tests for the Watchdog health monitoring module."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_cctv.alert_system import AlertSystem
from agentic_cctv.models import (
    AlertPayload,
    AlertResult,
    CooldownConfig,
    DeviceHealth,
    HeartbeatMessage,
)
from agentic_cctv.watchdog import Watchdog, _CameraState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_heartbeat_payload(
    camera_id: str = "cam-01",
    tenant_id: str = "tenant-a",
    site_id: str = "site-1",
    timestamp: Optional[str] = None,
) -> bytes:
    """Build a heartbeat JSON payload as bytes."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    data = {
        "camera_id": camera_id,
        "tenant_id": tenant_id,
        "site_id": site_id,
        "timestamp": timestamp,
        "cpu_percent": 45.2,
        "memory_percent": 62.1,
        "temperature_celsius": 68.5,
        "inference_latency_ms": 35.2,
        "gpu_utilization_percent": 78.0,
    }
    return json.dumps(data).encode("utf-8")


class FakeMQTTSubscriber:
    """Fake MQTT subscriber that records subscribe calls and allows manual
    message injection."""

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

    def inject_message(self, topic: str, payload: bytes, qos: int = 1) -> None:
        """Simulate an incoming MQTT message by calling the registered callback."""
        for pattern, cb in self._callbacks.items():
            cb(topic, payload, qos)  # type: ignore[operator]


class FakeAlertSystem:
    """Fake AlertSystem that records all sent alerts."""

    def __init__(self) -> None:
        self.alerts: List[AlertPayload] = []

    async def send_alert(self, payload: AlertPayload) -> AlertResult:
        self.alerts.append(payload)
        return AlertResult(delivered=True, channels=["FakeChannel"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_subscriber() -> FakeMQTTSubscriber:
    return FakeMQTTSubscriber()


@pytest.fixture
def fake_alert_system() -> FakeAlertSystem:
    return FakeAlertSystem()


@pytest.fixture
def watchdog(
    fake_subscriber: FakeMQTTSubscriber,
    fake_alert_system: FakeAlertSystem,
) -> Watchdog:
    return Watchdog(
        mqtt_subscriber=fake_subscriber,  # type: ignore[arg-type]
        alert_system=fake_alert_system,  # type: ignore[arg-type]
        offline_threshold=0.3,  # 300ms for fast tests
        check_interval=0.1,  # 100ms for fast tests
    )


# ---------------------------------------------------------------------------
# Tests: start / stop lifecycle
# ---------------------------------------------------------------------------


class TestWatchdogLifecycle:
    @pytest.mark.asyncio
    async def test_start_subscribes_to_health_topic(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
    ) -> None:
        await watchdog.start()
        try:
            assert len(fake_subscriber.subscriptions) == 1
            topic, qos = fake_subscriber.subscriptions[0]
            assert topic == "+/+/+/health"
            assert qos == 1
        finally:
            await watchdog.stop()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
    ) -> None:
        await watchdog.start()
        await watchdog.start()  # second call should be a no-op
        try:
            assert len(fake_subscriber.subscriptions) == 1
        finally:
            await watchdog.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self, watchdog: Watchdog) -> None:
        await watchdog.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_stop_cancels_check_loop(self, watchdog: Watchdog) -> None:
        await watchdog.start()
        assert watchdog._running is True
        await watchdog.stop()
        assert watchdog._running is False
        assert watchdog._check_task is None


# ---------------------------------------------------------------------------
# Tests: heartbeat processing
# ---------------------------------------------------------------------------


class TestHeartbeatProcessing:
    @pytest.mark.asyncio
    async def test_heartbeat_registers_new_camera(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
    ) -> None:
        await watchdog.start()
        try:
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-01/health",
                _make_heartbeat_payload(camera_id="cam-01"),
            )
            status = watchdog.get_device_status("cam-01")
            assert status.camera_id == "cam-01"
            assert status.status == "online"
            assert status.last_heartbeat is not None
            assert status.metrics is not None
        finally:
            await watchdog.stop()

    @pytest.mark.asyncio
    async def test_heartbeat_updates_metrics(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
    ) -> None:
        await watchdog.start()
        try:
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-01/health",
                _make_heartbeat_payload(camera_id="cam-01"),
            )
            status = watchdog.get_device_status("cam-01")
            assert status.metrics is not None
            assert status.metrics.cpu_percent == 45.2
            assert status.metrics.memory_percent == 62.1
            assert status.metrics.temperature_celsius == 68.5
        finally:
            await watchdog.stop()

    @pytest.mark.asyncio
    async def test_invalid_heartbeat_json_is_ignored(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
    ) -> None:
        await watchdog.start()
        try:
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-01/health",
                b"not valid json",
            )
            # Camera should not be registered
            status = watchdog.get_device_status("cam-01")
            assert status.status == "offline"
            assert status.last_heartbeat is None
        finally:
            await watchdog.stop()

    @pytest.mark.asyncio
    async def test_heartbeat_with_z_timestamp(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
    ) -> None:
        """Heartbeat with trailing Z in timestamp should be parsed correctly."""
        await watchdog.start()
        try:
            data = {
                "camera_id": "cam-01",
                "tenant_id": "tenant-a",
                "site_id": "site-1",
                "timestamp": "2025-01-15T14:30:00Z",
                "cpu_percent": 10.0,
                "memory_percent": 20.0,
                "temperature_celsius": None,
                "inference_latency_ms": 0.0,
                "gpu_utilization_percent": None,
            }
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-01/health",
                json.dumps(data).encode("utf-8"),
            )
            status = watchdog.get_device_status("cam-01")
            assert status.status == "online"
            assert status.metrics is not None
            assert status.metrics.temperature_celsius is None
            assert status.metrics.gpu_utilization_percent is None
        finally:
            await watchdog.stop()


# ---------------------------------------------------------------------------
# Tests: get_device_status / get_all_device_status
# ---------------------------------------------------------------------------


class TestDeviceStatus:
    @pytest.mark.asyncio
    async def test_unknown_camera_returns_offline(self, watchdog: Watchdog) -> None:
        status = watchdog.get_device_status("unknown-cam")
        assert status.camera_id == "unknown-cam"
        assert status.status == "offline"
        assert status.last_heartbeat is None
        assert status.metrics is None

    @pytest.mark.asyncio
    async def test_get_all_device_status_empty(self, watchdog: Watchdog) -> None:
        result = watchdog.get_all_device_status()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_device_status_multiple_cameras(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
    ) -> None:
        await watchdog.start()
        try:
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-01/health",
                _make_heartbeat_payload(camera_id="cam-01"),
            )
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-02/health",
                _make_heartbeat_payload(camera_id="cam-02"),
            )
            all_status = watchdog.get_all_device_status()
            assert len(all_status) == 2
            camera_ids = {s.camera_id for s in all_status}
            assert camera_ids == {"cam-01", "cam-02"}
        finally:
            await watchdog.stop()


# ---------------------------------------------------------------------------
# Tests: offline detection
# ---------------------------------------------------------------------------


class TestOfflineDetection:
    @pytest.mark.asyncio
    async def test_camera_goes_offline_after_threshold(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
        fake_alert_system: FakeAlertSystem,
    ) -> None:
        """Camera should transition to offline when no heartbeat for > threshold."""
        await watchdog.start()
        try:
            # Send initial heartbeat
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-01/health",
                _make_heartbeat_payload(camera_id="cam-01"),
            )
            assert watchdog.get_device_status("cam-01").status == "online"

            # Wait for the offline threshold + check interval to pass
            await asyncio.sleep(0.6)

            # Camera should now be offline
            status = watchdog.get_device_status("cam-01")
            assert status.status == "offline"

            # An offline alert should have been sent
            offline_alerts = [
                a for a in fake_alert_system.alerts
                if a.alert_type == "camera-offline"
            ]
            assert len(offline_alerts) >= 1
            alert = offline_alerts[0]
            assert alert.camera_id == "cam-01"
            assert alert.threat_level == "high"
        finally:
            await watchdog.stop()

    @pytest.mark.asyncio
    async def test_camera_stays_online_with_regular_heartbeats(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
        fake_alert_system: FakeAlertSystem,
    ) -> None:
        """Camera should stay online if heartbeats arrive within threshold."""
        await watchdog.start()
        try:
            # Send heartbeats faster than the offline threshold
            for _ in range(5):
                fake_subscriber.inject_message(
                    "tenant-a/site-1/cam-01/health",
                    _make_heartbeat_payload(camera_id="cam-01"),
                )
                await asyncio.sleep(0.1)

            assert watchdog.get_device_status("cam-01").status == "online"

            # No offline alerts should have been sent
            offline_alerts = [
                a for a in fake_alert_system.alerts
                if a.alert_type == "camera-offline"
            ]
            assert len(offline_alerts) == 0
        finally:
            await watchdog.stop()

    @pytest.mark.asyncio
    async def test_offline_alert_not_sent_twice(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
        fake_alert_system: FakeAlertSystem,
    ) -> None:
        """Once a camera is offline, the alert should not be re-sent on each check."""
        await watchdog.start()
        try:
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-01/health",
                _make_heartbeat_payload(camera_id="cam-01"),
            )

            # Wait for offline + multiple check cycles
            await asyncio.sleep(0.8)

            offline_alerts = [
                a for a in fake_alert_system.alerts
                if a.alert_type == "camera-offline" and a.camera_id == "cam-01"
            ]
            # Should only have one offline alert, not multiple
            assert len(offline_alerts) == 1
        finally:
            await watchdog.stop()


# ---------------------------------------------------------------------------
# Tests: camera restored
# ---------------------------------------------------------------------------


class TestCameraRestored:
    @pytest.mark.asyncio
    async def test_camera_restored_after_offline(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
        fake_alert_system: FakeAlertSystem,
    ) -> None:
        """Camera should transition back to online and send restored alert."""
        await watchdog.start()
        try:
            # Send initial heartbeat
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-01/health",
                _make_heartbeat_payload(camera_id="cam-01"),
            )

            # Wait for offline
            await asyncio.sleep(0.6)
            assert watchdog.get_device_status("cam-01").status == "offline"

            # Send a new heartbeat to restore
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-01/health",
                _make_heartbeat_payload(camera_id="cam-01"),
            )

            # Camera should be back online immediately
            assert watchdog.get_device_status("cam-01").status == "online"

            # Give time for the async restored alert to be processed
            await asyncio.sleep(0.2)

            # A restored alert should have been sent
            restored_alerts = [
                a for a in fake_alert_system.alerts
                if a.alert_type == "camera-restored"
            ]
            assert len(restored_alerts) >= 1
            alert = restored_alerts[0]
            assert alert.camera_id == "cam-01"
            assert alert.threat_level == "low"
            assert "back online" in alert.description
        finally:
            await watchdog.stop()

    @pytest.mark.asyncio
    async def test_restored_alert_includes_downtime_duration(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
        fake_alert_system: FakeAlertSystem,
    ) -> None:
        """Restored alert description should mention downtime duration."""
        await watchdog.start()
        try:
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-01/health",
                _make_heartbeat_payload(camera_id="cam-01"),
            )

            # Wait for offline
            await asyncio.sleep(0.6)

            # Restore
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-01/health",
                _make_heartbeat_payload(camera_id="cam-01"),
            )

            await asyncio.sleep(0.2)

            restored_alerts = [
                a for a in fake_alert_system.alerts
                if a.alert_type == "camera-restored"
            ]
            assert len(restored_alerts) >= 1
            # Description should contain "seconds of downtime"
            assert "seconds of downtime" in restored_alerts[0].description
        finally:
            await watchdog.stop()


# ---------------------------------------------------------------------------
# Tests: multiple cameras
# ---------------------------------------------------------------------------


class TestMultipleCameras:
    @pytest.mark.asyncio
    async def test_independent_tracking_per_camera(
        self,
        watchdog: Watchdog,
        fake_subscriber: FakeMQTTSubscriber,
        fake_alert_system: FakeAlertSystem,
    ) -> None:
        """Each camera should be tracked independently."""
        await watchdog.start()
        try:
            # cam-01 sends heartbeats, cam-02 does not after initial
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-01/health",
                _make_heartbeat_payload(camera_id="cam-01"),
            )
            fake_subscriber.inject_message(
                "tenant-a/site-1/cam-02/health",
                _make_heartbeat_payload(camera_id="cam-02"),
            )

            # Keep cam-01 alive, let cam-02 go offline
            for _ in range(5):
                fake_subscriber.inject_message(
                    "tenant-a/site-1/cam-01/health",
                    _make_heartbeat_payload(camera_id="cam-01"),
                )
                await asyncio.sleep(0.1)

            assert watchdog.get_device_status("cam-01").status == "online"
            assert watchdog.get_device_status("cam-02").status == "offline"

            # Only cam-02 should have an offline alert
            offline_alerts = [
                a for a in fake_alert_system.alerts
                if a.alert_type == "camera-offline"
            ]
            cam_ids = {a.camera_id for a in offline_alerts}
            assert "cam-02" in cam_ids
            assert "cam-01" not in cam_ids
        finally:
            await watchdog.stop()


# ---------------------------------------------------------------------------
# Tests: parse heartbeat edge cases
# ---------------------------------------------------------------------------


class TestParseHeartbeat:
    def test_parse_heartbeat_with_all_fields(self) -> None:
        data = {
            "camera_id": "cam-01",
            "tenant_id": "tenant-a",
            "site_id": "site-1",
            "timestamp": "2025-01-15T14:30:00+00:00",
            "cpu_percent": 45.2,
            "memory_percent": 62.1,
            "temperature_celsius": 68.5,
            "inference_latency_ms": 35.2,
            "gpu_utilization_percent": 78.0,
        }
        hb = Watchdog._parse_heartbeat(data)
        assert hb.camera_id == "cam-01"
        assert hb.cpu_percent == 45.2
        assert hb.temperature_celsius == 68.5
        assert hb.gpu_utilization_percent == 78.0

    def test_parse_heartbeat_with_none_optional_fields(self) -> None:
        data = {
            "camera_id": "cam-01",
            "tenant_id": "tenant-a",
            "site_id": "site-1",
            "timestamp": "2025-01-15T14:30:00+00:00",
            "cpu_percent": 10.0,
            "memory_percent": 20.0,
            "temperature_celsius": None,
            "inference_latency_ms": 0.0,
            "gpu_utilization_percent": None,
        }
        hb = Watchdog._parse_heartbeat(data)
        assert hb.temperature_celsius is None
        assert hb.gpu_utilization_percent is None

    def test_parse_heartbeat_with_z_timestamp(self) -> None:
        data = {
            "camera_id": "cam-01",
            "tenant_id": "tenant-a",
            "site_id": "site-1",
            "timestamp": "2025-01-15T14:30:00Z",
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
        }
        hb = Watchdog._parse_heartbeat(data)
        assert hb.timestamp.year == 2025

    def test_parse_heartbeat_missing_required_field_raises(self) -> None:
        data = {
            "tenant_id": "tenant-a",
            "site_id": "site-1",
            "timestamp": "2025-01-15T14:30:00+00:00",
        }
        with pytest.raises(KeyError):
            Watchdog._parse_heartbeat(data)

    def test_parse_heartbeat_defaults_missing_optional_metrics(self) -> None:
        """Missing optional metric fields should default to 0.0 or None."""
        data = {
            "camera_id": "cam-01",
            "tenant_id": "tenant-a",
            "site_id": "site-1",
            "timestamp": "2025-01-15T14:30:00+00:00",
        }
        hb = Watchdog._parse_heartbeat(data)
        assert hb.cpu_percent == 0.0
        assert hb.memory_percent == 0.0
        assert hb.inference_latency_ms == 0.0
        assert hb.temperature_celsius is None
        assert hb.gpu_utilization_percent is None


# ---------------------------------------------------------------------------
# Tests: exact 60-second boundary behaviour (time-manipulated)
# ---------------------------------------------------------------------------


class TestExact60SecondBoundary:
    """Tests that exercise the real 60-second offline threshold using direct
    time manipulation (``_loop_time`` override + ``_check_cameras()`` calls)
    rather than ``asyncio.sleep``.

    **Validates: Requirements 9.2, 9.4**
    """

    @staticmethod
    def _make_watchdog_60s() -> Tuple[Watchdog, FakeAlertSystem]:
        """Create a Watchdog with the real 60s threshold and manual check control."""
        subscriber = FakeMQTTSubscriber()
        alert_system = FakeAlertSystem()
        wd = Watchdog(
            mqtt_subscriber=subscriber,  # type: ignore[arg-type]
            alert_system=alert_system,  # type: ignore[arg-type]
            offline_threshold=60.0,
            check_interval=999.0,  # we drive checks manually
        )
        return wd, alert_system

    @staticmethod
    def _register_camera(
        wd: Watchdog,
        camera_id: str,
        loop_time: float,
    ) -> _CameraState:
        """Register a camera in the watchdog's internal state."""
        state = _CameraState(
            camera_id=camera_id,
            tenant_id="tenant-test",
            site_id="site-test",
        )
        state.status = "online"
        state.last_heartbeat_time = loop_time
        wd._cameras[camera_id] = state
        return state

    # 1. Exactly at 60s — should stay online (uses strict >)
    @pytest.mark.asyncio
    async def test_exactly_at_60s_stays_online(self) -> None:
        """With default 60s threshold, a camera whose last heartbeat was
        exactly 60s ago should remain online because the check uses
        ``elapsed > threshold`` (strict greater-than, not >=)."""
        wd, alert_system = self._make_watchdog_60s()
        loop = asyncio.get_event_loop()
        wd._loop = loop

        base_time = 1000.0
        state = self._register_camera(wd, "cam-boundary", loop_time=base_time)

        # Time is exactly at the 60s mark
        wd._loop_time = lambda: base_time + 60.0  # type: ignore[assignment]
        await wd._check_cameras()

        assert state.status == "online", (
            "Camera should stay online at exactly 60s (strict > comparison)"
        )
        offline_alerts = [
            a for a in alert_system.alerts if a.alert_type == "camera-offline"
        ]
        assert len(offline_alerts) == 0

    # 2. Just over 60s — should go offline
    @pytest.mark.asyncio
    async def test_just_over_60s_goes_offline(self) -> None:
        """With default 60s threshold, a camera whose last heartbeat was
        60.001s ago should transition to offline."""
        wd, alert_system = self._make_watchdog_60s()
        loop = asyncio.get_event_loop()
        wd._loop = loop

        base_time = 1000.0
        state = self._register_camera(wd, "cam-boundary", loop_time=base_time)

        wd._loop_time = lambda: base_time + 60.001  # type: ignore[assignment]
        await wd._check_cameras()

        assert state.status == "offline", (
            "Camera should go offline at 60.001s (just over threshold)"
        )
        offline_alerts = [
            a for a in alert_system.alerts if a.alert_type == "camera-offline"
        ]
        assert len(offline_alerts) == 1
        assert offline_alerts[0].camera_id == "cam-boundary"
        assert offline_alerts[0].threat_level == "high"

    # 3. Just under 60s — should stay online
    @pytest.mark.asyncio
    async def test_just_under_60s_stays_online(self) -> None:
        """With default 60s threshold, a camera whose last heartbeat was
        59.999s ago should remain online."""
        wd, alert_system = self._make_watchdog_60s()
        loop = asyncio.get_event_loop()
        wd._loop = loop

        base_time = 1000.0
        state = self._register_camera(wd, "cam-boundary", loop_time=base_time)

        wd._loop_time = lambda: base_time + 59.999  # type: ignore[assignment]
        await wd._check_cameras()

        assert state.status == "online", (
            "Camera should stay online at 59.999s (under threshold)"
        )
        offline_alerts = [
            a for a in alert_system.alerts if a.alert_type == "camera-offline"
        ]
        assert len(offline_alerts) == 0

    # 4. Rapid online → offline → online cycle
    @pytest.mark.asyncio
    async def test_rapid_online_offline_online_cycle(self) -> None:
        """Camera goes online → offline → online rapidly.  Verify the correct
        alert sequence: exactly 1 camera-offline + 1 camera-restored."""
        wd, alert_system = self._make_watchdog_60s()
        loop = asyncio.get_event_loop()
        wd._loop = loop

        base_time = 1000.0
        state = self._register_camera(wd, "cam-rapid", loop_time=base_time)

        # Step 1: Camera is online (just registered)
        assert state.status == "online"

        # Step 2: Time jumps past threshold → camera goes offline
        offline_time = base_time + 61.0
        wd._loop_time = lambda: offline_time  # type: ignore[assignment]
        await wd._check_cameras()
        assert state.status == "offline"

        # Step 3: Simulate heartbeat arrival → camera restored
        restore_time = offline_time + 5.0
        wd._loop_time = lambda: restore_time  # type: ignore[assignment]

        # Mimic what _on_heartbeat does for a restored camera
        was_offline = state.status == "offline"
        state.last_heartbeat_time = restore_time
        state.status = "online"
        if was_offline:
            downtime = Watchdog._compute_downtime(state, restore_time)
            state.went_offline_at = None
            await wd._send_restored_alert(state, downtime)

        assert state.status == "online"

        # Verify alert sequence
        offline_alerts = [
            a for a in alert_system.alerts if a.alert_type == "camera-offline"
        ]
        restored_alerts = [
            a for a in alert_system.alerts if a.alert_type == "camera-restored"
        ]
        assert len(offline_alerts) == 1, (
            f"Expected 1 offline alert, got {len(offline_alerts)}"
        )
        assert len(restored_alerts) == 1, (
            f"Expected 1 restored alert, got {len(restored_alerts)}"
        )
        assert offline_alerts[0].camera_id == "cam-rapid"
        assert restored_alerts[0].camera_id == "cam-rapid"

    # 5. Downtime duration accuracy
    @pytest.mark.asyncio
    async def test_downtime_duration_accuracy(self) -> None:
        """Verify the downtime duration in the restored alert description is
        approximately correct (within 1s tolerance)."""
        wd, alert_system = self._make_watchdog_60s()
        loop = asyncio.get_event_loop()
        wd._loop = loop

        base_time = 1000.0
        state = self._register_camera(wd, "cam-downtime", loop_time=base_time)

        # Step 1: Camera goes offline at base_time + 61
        offline_time = base_time + 61.0
        wd._loop_time = lambda: offline_time  # type: ignore[assignment]
        await wd._check_cameras()
        assert state.status == "offline"
        assert state.went_offline_at == offline_time

        # Step 2: Camera restored 120s after going offline
        expected_downtime = 120.0
        restore_time = offline_time + expected_downtime
        wd._loop_time = lambda: restore_time  # type: ignore[assignment]

        was_offline = state.status == "offline"
        state.last_heartbeat_time = restore_time
        state.status = "online"
        if was_offline:
            downtime = Watchdog._compute_downtime(state, restore_time)
            state.went_offline_at = None
            await wd._send_restored_alert(state, downtime)

        # Verify the restored alert contains the correct downtime
        restored_alerts = [
            a for a in alert_system.alerts if a.alert_type == "camera-restored"
        ]
        assert len(restored_alerts) == 1
        desc = restored_alerts[0].description

        # Description format: "Camera cam-downtime is back online after 120.0 seconds of downtime."
        assert "seconds of downtime" in desc
        assert "back online" in desc

        # Extract the numeric downtime from the description
        import re
        match = re.search(r"after\s+([\d.]+)\s+seconds", desc)
        assert match is not None, f"Could not parse downtime from description: {desc}"
        reported_downtime = float(match.group(1))
        assert abs(reported_downtime - expected_downtime) < 1.0, (
            f"Reported downtime {reported_downtime}s differs from expected "
            f"{expected_downtime}s by more than 1s"
        )
