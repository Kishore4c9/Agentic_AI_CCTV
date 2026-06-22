"""Tests for the HeartbeatPublisher module."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_cctv.heartbeat_publisher import (
    HeartbeatPublisher,
    _get_cpu_percent,
    _get_gpu_utilization,
    _get_memory_percent,
    _get_temperature,
    _heartbeat_to_dict,
)
from agentic_cctv.models import CameraConfig, HeartbeatMessage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def camera_config() -> CameraConfig:
    return CameraConfig(
        camera_id="cam-lobby-01",
        uri="rtsp://192.168.1.100:554/stream1",
        tenant_id="tenant-acme",
        site_id="site-hq",
        confidence_threshold=0.7,
        monitored_classes=["person", "vehicle"],
    )


class FakeMQTTPublisher:
    """Records all publish calls for assertion."""

    def __init__(self) -> None:
        self.messages: List[Tuple[str, bytes, int, bool]] = []

    async def publish(
        self, topic: str, payload: bytes, qos: int = 1, retain: bool = False
    ) -> None:
        self.messages.append((topic, payload, qos, retain))


@pytest.fixture
def fake_publisher() -> FakeMQTTPublisher:
    return FakeMQTTPublisher()


# ---------------------------------------------------------------------------
# Metric helper tests
# ---------------------------------------------------------------------------


class TestGetCpuPercent:
    def test_returns_float_with_psutil(self) -> None:
        with patch("agentic_cctv.heartbeat_publisher._HAS_PSUTIL", True), \
             patch("agentic_cctv.heartbeat_publisher.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 42.5
            assert _get_cpu_percent() == 42.5

    def test_returns_zero_without_psutil(self) -> None:
        with patch("agentic_cctv.heartbeat_publisher._HAS_PSUTIL", False):
            assert _get_cpu_percent() == 0.0

    def test_returns_zero_on_exception(self) -> None:
        with patch("agentic_cctv.heartbeat_publisher._HAS_PSUTIL", True), \
             patch("agentic_cctv.heartbeat_publisher.psutil") as mock_psutil:
            mock_psutil.cpu_percent.side_effect = RuntimeError("fail")
            assert _get_cpu_percent() == 0.0


class TestGetMemoryPercent:
    def test_returns_float_with_psutil(self) -> None:
        with patch("agentic_cctv.heartbeat_publisher._HAS_PSUTIL", True), \
             patch("agentic_cctv.heartbeat_publisher.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(percent=62.1)
            assert _get_memory_percent() == 62.1

    def test_returns_zero_without_psutil(self) -> None:
        with patch("agentic_cctv.heartbeat_publisher._HAS_PSUTIL", False):
            assert _get_memory_percent() == 0.0


class TestGetTemperature:
    def test_returns_temperature_with_psutil(self) -> None:
        fake_entry = MagicMock(current=68.5)
        with patch("agentic_cctv.heartbeat_publisher._HAS_PSUTIL", True), \
             patch("agentic_cctv.heartbeat_publisher.psutil") as mock_psutil:
            mock_psutil.sensors_temperatures.return_value = {
                "coretemp": [fake_entry]
            }
            assert _get_temperature() == 68.5

    def test_returns_none_when_no_sensors(self) -> None:
        with patch("agentic_cctv.heartbeat_publisher._HAS_PSUTIL", True), \
             patch("agentic_cctv.heartbeat_publisher.psutil") as mock_psutil:
            mock_psutil.sensors_temperatures.return_value = {}
            assert _get_temperature() is None

    def test_returns_none_without_psutil(self) -> None:
        with patch("agentic_cctv.heartbeat_publisher._HAS_PSUTIL", False):
            assert _get_temperature() is None

    def test_returns_none_on_exception(self) -> None:
        with patch("agentic_cctv.heartbeat_publisher._HAS_PSUTIL", True), \
             patch("agentic_cctv.heartbeat_publisher.psutil") as mock_psutil:
            mock_psutil.sensors_temperatures.side_effect = AttributeError("no sensors")
            assert _get_temperature() is None


class TestGetGpuUtilization:
    def test_returns_float_from_nvidia_smi(self) -> None:
        mock_result = MagicMock(returncode=0, stdout="78\n")
        with patch("agentic_cctv.heartbeat_publisher.subprocess.run", return_value=mock_result):
            assert _get_gpu_utilization() == 78.0

    def test_returns_none_when_nvidia_smi_not_found(self) -> None:
        with patch(
            "agentic_cctv.heartbeat_publisher.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert _get_gpu_utilization() is None

    def test_returns_none_on_timeout(self) -> None:
        import subprocess

        with patch(
            "agentic_cctv.heartbeat_publisher.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5),
        ):
            assert _get_gpu_utilization() is None

    def test_returns_none_on_nonzero_exit(self) -> None:
        mock_result = MagicMock(returncode=1, stdout="")
        with patch("agentic_cctv.heartbeat_publisher.subprocess.run", return_value=mock_result):
            assert _get_gpu_utilization() is None


# ---------------------------------------------------------------------------
# HeartbeatMessage serialisation
# ---------------------------------------------------------------------------


class TestHeartbeatToDict:
    def test_serialises_all_fields(self) -> None:
        ts = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        msg = HeartbeatMessage(
            camera_id="cam-01",
            tenant_id="tenant-a",
            site_id="site-1",
            timestamp=ts,
            cpu_percent=45.2,
            memory_percent=62.1,
            temperature_celsius=68.5,
            inference_latency_ms=35.2,
            gpu_utilization_percent=78.0,
        )
        d = _heartbeat_to_dict(msg)
        assert d["camera_id"] == "cam-01"
        assert d["tenant_id"] == "tenant-a"
        assert d["site_id"] == "site-1"
        assert d["timestamp"] == "2025-01-15T14:30:00+00:00"
        assert d["cpu_percent"] == 45.2
        assert d["memory_percent"] == 62.1
        assert d["temperature_celsius"] == 68.5
        assert d["inference_latency_ms"] == 35.2
        assert d["gpu_utilization_percent"] == 78.0

    def test_serialises_none_fields(self) -> None:
        ts = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        msg = HeartbeatMessage(
            camera_id="cam-01",
            tenant_id="tenant-a",
            site_id="site-1",
            timestamp=ts,
            cpu_percent=0.0,
            memory_percent=0.0,
            temperature_celsius=None,
            inference_latency_ms=0.0,
            gpu_utilization_percent=None,
        )
        d = _heartbeat_to_dict(msg)
        assert d["temperature_celsius"] is None
        assert d["gpu_utilization_percent"] is None


# ---------------------------------------------------------------------------
# HeartbeatPublisher tests
# ---------------------------------------------------------------------------


class TestHeartbeatPublisher:
    @pytest.mark.asyncio
    async def test_publish_heartbeat_sends_to_correct_topic(
        self, camera_config: CameraConfig, fake_publisher: FakeMQTTPublisher
    ) -> None:
        publisher = HeartbeatPublisher(camera_config, fake_publisher)

        with patch("agentic_cctv.heartbeat_publisher._get_cpu_percent", return_value=10.0), \
             patch("agentic_cctv.heartbeat_publisher._get_memory_percent", return_value=20.0), \
             patch("agentic_cctv.heartbeat_publisher._get_temperature", return_value=None), \
             patch("agentic_cctv.heartbeat_publisher._get_gpu_utilization", return_value=None):
            await publisher.publish_heartbeat()

        assert len(fake_publisher.messages) == 1
        topic, payload, qos, retain = fake_publisher.messages[0]
        assert topic == "tenant-acme/site-hq/cam-lobby-01/health"
        assert qos == 1
        assert retain is True

    @pytest.mark.asyncio
    async def test_publish_heartbeat_payload_is_valid_json(
        self, camera_config: CameraConfig, fake_publisher: FakeMQTTPublisher
    ) -> None:
        publisher = HeartbeatPublisher(camera_config, fake_publisher)

        with patch("agentic_cctv.heartbeat_publisher._get_cpu_percent", return_value=45.2), \
             patch("agentic_cctv.heartbeat_publisher._get_memory_percent", return_value=62.1), \
             patch("agentic_cctv.heartbeat_publisher._get_temperature", return_value=68.5), \
             patch("agentic_cctv.heartbeat_publisher._get_gpu_utilization", return_value=78.0):
            await publisher.publish_heartbeat()

        _, payload_bytes, _, _ = fake_publisher.messages[0]
        data = json.loads(payload_bytes.decode("utf-8"))
        assert data["camera_id"] == "cam-lobby-01"
        assert data["tenant_id"] == "tenant-acme"
        assert data["site_id"] == "site-hq"
        assert data["cpu_percent"] == 45.2
        assert data["memory_percent"] == 62.1
        assert data["temperature_celsius"] == 68.5
        assert data["inference_latency_ms"] == 0.0
        assert data["gpu_utilization_percent"] == 78.0
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_publish_heartbeat_with_inference_latency_provider(
        self, camera_config: CameraConfig, fake_publisher: FakeMQTTPublisher
    ) -> None:
        latency_fn = lambda: 35.2
        publisher = HeartbeatPublisher(
            camera_config, fake_publisher, inference_latency_provider=latency_fn
        )

        with patch("agentic_cctv.heartbeat_publisher._get_cpu_percent", return_value=0.0), \
             patch("agentic_cctv.heartbeat_publisher._get_memory_percent", return_value=0.0), \
             patch("agentic_cctv.heartbeat_publisher._get_temperature", return_value=None), \
             patch("agentic_cctv.heartbeat_publisher._get_gpu_utilization", return_value=None):
            await publisher.publish_heartbeat()

        _, payload_bytes, _, _ = fake_publisher.messages[0]
        data = json.loads(payload_bytes.decode("utf-8"))
        assert data["inference_latency_ms"] == 35.2

    @pytest.mark.asyncio
    async def test_start_and_stop_lifecycle(
        self, camera_config: CameraConfig, fake_publisher: FakeMQTTPublisher
    ) -> None:
        publisher = HeartbeatPublisher(
            camera_config, fake_publisher, interval_seconds=0.05
        )

        with patch("agentic_cctv.heartbeat_publisher._get_cpu_percent", return_value=0.0), \
             patch("agentic_cctv.heartbeat_publisher._get_memory_percent", return_value=0.0), \
             patch("agentic_cctv.heartbeat_publisher._get_temperature", return_value=None), \
             patch("agentic_cctv.heartbeat_publisher._get_gpu_utilization", return_value=None):
            await publisher.start()
            # Let a couple of heartbeats fire
            await asyncio.sleep(0.15)
            await publisher.stop()

        # Should have published at least 2 heartbeats
        assert len(fake_publisher.messages) >= 2
        # All should be to the health topic with retain=True
        for topic, _, qos, retain in fake_publisher.messages:
            assert topic == "tenant-acme/site-hq/cam-lobby-01/health"
            assert qos == 1
            assert retain is True

    @pytest.mark.asyncio
    async def test_start_is_idempotent(
        self, camera_config: CameraConfig, fake_publisher: FakeMQTTPublisher
    ) -> None:
        publisher = HeartbeatPublisher(
            camera_config, fake_publisher, interval_seconds=10.0
        )

        with patch("agentic_cctv.heartbeat_publisher._get_cpu_percent", return_value=0.0), \
             patch("agentic_cctv.heartbeat_publisher._get_memory_percent", return_value=0.0), \
             patch("agentic_cctv.heartbeat_publisher._get_temperature", return_value=None), \
             patch("agentic_cctv.heartbeat_publisher._get_gpu_utilization", return_value=None):
            await publisher.start()
            await publisher.start()  # second call should be a no-op
            await publisher.stop()

    @pytest.mark.asyncio
    async def test_publish_heartbeat_handles_mqtt_error(
        self, camera_config: CameraConfig
    ) -> None:
        """Publisher should not raise if MQTT publish fails."""
        failing_publisher = AsyncMock()
        failing_publisher.publish = AsyncMock(side_effect=ConnectionError("broker down"))

        publisher = HeartbeatPublisher(camera_config, failing_publisher)

        with patch("agentic_cctv.heartbeat_publisher._get_cpu_percent", return_value=0.0), \
             patch("agentic_cctv.heartbeat_publisher._get_memory_percent", return_value=0.0), \
             patch("agentic_cctv.heartbeat_publisher._get_temperature", return_value=None), \
             patch("agentic_cctv.heartbeat_publisher._get_gpu_utilization", return_value=None):
            # Should not raise
            await publisher.publish_heartbeat()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(
        self, camera_config: CameraConfig, fake_publisher: FakeMQTTPublisher
    ) -> None:
        publisher = HeartbeatPublisher(camera_config, fake_publisher)
        await publisher.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_default_interval_is_30_seconds(
        self, camera_config: CameraConfig, fake_publisher: FakeMQTTPublisher
    ) -> None:
        publisher = HeartbeatPublisher(camera_config, fake_publisher)
        assert publisher._interval == 30.0

    @pytest.mark.asyncio
    async def test_heartbeat_payload_contains_all_required_fields(
        self, camera_config: CameraConfig, fake_publisher: FakeMQTTPublisher
    ) -> None:
        publisher = HeartbeatPublisher(camera_config, fake_publisher)

        with patch("agentic_cctv.heartbeat_publisher._get_cpu_percent", return_value=1.0), \
             patch("agentic_cctv.heartbeat_publisher._get_memory_percent", return_value=2.0), \
             patch("agentic_cctv.heartbeat_publisher._get_temperature", return_value=50.0), \
             patch("agentic_cctv.heartbeat_publisher._get_gpu_utilization", return_value=30.0):
            await publisher.publish_heartbeat()

        _, payload_bytes, _, _ = fake_publisher.messages[0]
        data = json.loads(payload_bytes.decode("utf-8"))
        required_fields = {
            "camera_id",
            "tenant_id",
            "site_id",
            "timestamp",
            "cpu_percent",
            "memory_percent",
            "temperature_celsius",
            "inference_latency_ms",
            "gpu_utilization_percent",
        }
        assert required_fields.issubset(data.keys())
