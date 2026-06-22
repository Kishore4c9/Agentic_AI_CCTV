"""Tests for the main application entry point.

Validates CameraPipeline construction, alert system factory, MQTT connection
handling (graceful fallback), and the shutdown signal mechanism.
"""

from __future__ import annotations

import pytest

from agentic_cctv.alert_system import (
    AlertSystem,
    PushNotificationChannel,
    WebhookChannel,
)
from agentic_cctv.main import (
    CameraPipeline,
    _build_alert_system,
    _connect_mqtt_publisher,
    _connect_mqtt_subscriber,
    run_application,
)
from agentic_cctv.models import (
    AlertConfig,
    BrokerConfig,
    CameraConfig,
    CooldownConfig,
    SystemConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera_config(**overrides) -> CameraConfig:
    """Create a CameraConfig with sensible defaults for testing."""
    defaults = dict(
        camera_id="cam-test-01",
        uri="0",
        tenant_id="tenant-test",
        site_id="site-test",
        confidence_threshold=0.5,
        monitored_classes=["person"],
        inference_runtime="pytorch",
        model_path="./models/yolov8n.pt",
        tracker_algorithm="deepsort",
        frame_skip=3,
    )
    defaults.update(overrides)
    return CameraConfig(**defaults)


# ---------------------------------------------------------------------------
# CameraPipeline tests
# ---------------------------------------------------------------------------


class TestCameraPipeline:
    """Tests for the CameraPipeline class."""

    def test_pipeline_construction(self) -> None:
        """Pipeline should instantiate all sub-components from a CameraConfig."""
        cam = _make_camera_config()
        pipeline = CameraPipeline(cam, mqtt_publisher=None)

        assert pipeline.camera_config is cam
        assert pipeline.video_feeder is not None
        assert pipeline.detection_engine is not None
        assert pipeline.tracker is not None
        assert pipeline.event_encoder is not None
        assert pipeline.heartbeat_publisher is not None
        assert pipeline.runtime is not None
        assert pipeline.is_running is False

    def test_pipeline_construction_with_bytetrack(self) -> None:
        """Pipeline should accept bytetrack as tracker algorithm."""
        cam = _make_camera_config(tracker_algorithm="bytetrack")
        pipeline = CameraPipeline(cam, mqtt_publisher=None)
        assert pipeline.tracker is not None

    @pytest.mark.asyncio
    async def test_process_frame_no_frame_available(self) -> None:
        """process_frame should return gracefully when no frame is available."""
        cam = _make_camera_config()
        pipeline = CameraPipeline(cam, mqtt_publisher=None)
        pipeline._running = True
        # VideoFeeder not started, so get_frame returns None
        await pipeline.process_frame()  # Should not raise

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self) -> None:
        """stop() should set is_running to False."""
        cam = _make_camera_config()
        pipeline = CameraPipeline(cam, mqtt_publisher=None)
        pipeline._running = True
        await pipeline.stop()
        assert pipeline.is_running is False


# ---------------------------------------------------------------------------
# Alert system factory tests
# ---------------------------------------------------------------------------


class TestBuildAlertSystem:
    """Tests for the _build_alert_system factory function."""

    def test_push_and_webhook_channels(self) -> None:
        """Should create both push and webhook channels."""
        config = AlertConfig(
            channels=["push", "webhook"],
            webhook_url="https://example.com/hook",
            cooldown=CooldownConfig(default_seconds=30),
        )
        system = _build_alert_system(config)
        assert isinstance(system, AlertSystem)
        assert len(system.channels) == 2
        assert isinstance(system.channels[0], PushNotificationChannel)
        assert isinstance(system.channels[1], WebhookChannel)

    def test_push_only(self) -> None:
        """Should create only push channel when webhook not configured."""
        config = AlertConfig(
            channels=["push"],
            cooldown=CooldownConfig(),
        )
        system = _build_alert_system(config)
        assert len(system.channels) == 1
        assert isinstance(system.channels[0], PushNotificationChannel)

    def test_unknown_channel_skipped(self) -> None:
        """Unknown channel names should be skipped with a warning."""
        config = AlertConfig(
            channels=["push", "sms", "unknown"],
            cooldown=CooldownConfig(),
        )
        system = _build_alert_system(config)
        # Only push is recognised; sms and unknown are skipped
        assert len(system.channels) == 1

    def test_empty_channels(self) -> None:
        """Empty channel list should produce an AlertSystem with no channels."""
        config = AlertConfig(channels=[], cooldown=CooldownConfig())
        system = _build_alert_system(config)
        assert len(system.channels) == 0

    def test_default_cooldown(self) -> None:
        """Should use default CooldownConfig when none provided."""
        config = AlertConfig(channels=["push"])
        system = _build_alert_system(config)
        assert system.cooldown_config.default_seconds == 60


# ---------------------------------------------------------------------------
# MQTT connection tests
# ---------------------------------------------------------------------------


class TestMQTTConnection:
    """Tests for MQTT connection helper functions."""

    @pytest.mark.asyncio
    async def test_publisher_returns_none_on_failure(self) -> None:
        """_connect_mqtt_publisher should return None when broker is unavailable."""
        broker = BrokerConfig(host="192.0.2.1", port=9999)
        result = await _connect_mqtt_publisher(broker)
        assert result is None

    @pytest.mark.asyncio
    async def test_subscriber_returns_none_on_failure(self) -> None:
        """_connect_mqtt_subscriber should return None when broker is unavailable."""
        broker = BrokerConfig(host="192.0.2.1", port=9999)
        result = await _connect_mqtt_subscriber(broker)
        assert result is None


# ---------------------------------------------------------------------------
# run_application tests
# ---------------------------------------------------------------------------


class TestRunApplication:
    """Tests for the run_application entry point."""

    @pytest.mark.asyncio
    async def test_exits_gracefully_with_no_cameras(self, tmp_path) -> None:
        """Should exit cleanly when config has no cameras."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "deployment_profile: single-machine\n"
            "mqtt:\n"
            "  host: localhost\n"
            "  port: 1883\n"
            "cameras: []\n"
        )
        # Should return without error (no cameras = nothing to do)
        await run_application(str(config_file))

    @pytest.mark.asyncio
    async def test_generates_default_config_when_missing(self, tmp_path) -> None:
        """Should generate a default config file when the path doesn't exist."""
        config_file = tmp_path / "subdir" / "config.yaml"
        assert not config_file.exists()

        # Use ConfigManager directly to verify default generation
        from agentic_cctv.config_manager import ConfigManager

        mgr = ConfigManager(str(config_file))
        cfg = mgr.load()
        assert config_file.exists()
        assert len(cfg.cameras) >= 1
