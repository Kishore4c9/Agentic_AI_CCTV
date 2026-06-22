"""Unit tests for ConfigManager."""

from __future__ import annotations

import os
import textwrap

import pytest
import yaml

from agentic_cctv.config_manager import ConfigManager, ValidationError
from agentic_cctv.models import (
    AlertConfig,
    BrokerConfig,
    CameraConfig,
    CooldownConfig,
    SecurityConfig,
    StorageConfig,
    SystemConfig,
    VLMConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: str, data: dict) -> None:
    """Write a dict as YAML to *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)


def _valid_config_dict() -> dict:
    """Return a minimal valid configuration dict."""
    return {
        "deployment_profile": "single-machine",
        "mqtt": {"host": "localhost", "port": 1883, "use_tls": False},
        "vlm": {
            "backend": "cosmos",
            "api_key": "nvapi-real-key-12345",
            "endpoint": "https://integrate.api.nvidia.com/v1",
            "timeout_seconds": 30,
        },
        "cameras": [
            {
                "camera_id": "cam-01",
                "uri": "rtsp://192.168.1.100:554/stream1",
                "tenant_id": "tenant-acme",
                "site_id": "site-hq",
                "confidence_threshold": 0.7,
                "monitored_classes": ["person", "vehicle"],
                "inference_runtime": "pytorch",
                "model_path": "./models/yolov8n.pt",
                "tracker_algorithm": "deepsort",
                "frame_skip": 3,
            }
        ],
        "storage": {
            "timeseries_db": "sqlite",
            "timeseries_path": "./data/events.db",
            "vector_db": "chromadb",
            "vector_path": "./data/vectors",
            "frame_crop_path": "./data/crops",
            "frame_crop_encryption_key": "",
            "retention": {
                "raw_events_days": 90,
                "aggregated_events_days": 365,
                "frame_crops_hours": 72,
            },
        },
        "alerts": {
            "channels": ["push", "webhook"],
            "webhook_url": "https://hooks.example.com/cctv",
            "cooldown": {
                "default_seconds": 60,
                "per_type_overrides": {"intrusion": 30, "fire": 10},
            },
        },
        "security": {
            "tls_enabled": False,
            "oauth_enabled": False,
            "tenant_isolation": True,
        },
    }


# ---------------------------------------------------------------------------
# Tests: Default config generation
# ---------------------------------------------------------------------------


class TestGenerateDefault:
    """Tests for ConfigManager.generate_default()."""

    def test_generates_file_when_missing(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        mgr = ConfigManager(config_path=cfg_path)
        mgr.generate_default()
        assert os.path.isfile(cfg_path)

    def test_generated_file_is_valid_yaml(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        mgr = ConfigManager(config_path=cfg_path)
        mgr.generate_default()
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict)
        assert "deployment_profile" in data
        assert "mqtt" in data
        assert "vlm" in data
        assert "cameras" in data

    def test_generated_config_has_placeholder_api_key(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        mgr = ConfigManager(config_path=cfg_path)
        mgr.generate_default()
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data["vlm"]["api_key"] == "YOUR_API_KEY_HERE"

    def test_creates_intermediate_directories(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "subdir", "nested", "config.yaml")
        mgr = ConfigManager(config_path=cfg_path)
        mgr.generate_default()
        assert os.path.isfile(cfg_path)


# ---------------------------------------------------------------------------
# Tests: Load
# ---------------------------------------------------------------------------


class TestLoad:
    """Tests for ConfigManager.load()."""

    def test_load_valid_config(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        _write_yaml(cfg_path, _valid_config_dict())
        mgr = ConfigManager(config_path=cfg_path)
        config = mgr.load()
        assert isinstance(config, SystemConfig)
        assert config.deployment_profile == "single-machine"
        assert config.vlm.backend == "cosmos"
        assert len(config.cameras) == 1
        assert config.cameras[0].camera_id == "cam-01"

    def test_load_generates_default_when_missing(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        mgr = ConfigManager(config_path=cfg_path)
        config = mgr.load()
        assert isinstance(config, SystemConfig)
        assert os.path.isfile(cfg_path)

    def test_load_parses_nested_dataclasses(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        _write_yaml(cfg_path, _valid_config_dict())
        mgr = ConfigManager(config_path=cfg_path)
        config = mgr.load()
        assert isinstance(config.mqtt, BrokerConfig)
        assert isinstance(config.vlm, VLMConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.alerts, AlertConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.alerts.cooldown, CooldownConfig)

    def test_load_empty_yaml(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            fh.write("")
        mgr = ConfigManager(config_path=cfg_path)
        config = mgr.load()
        assert isinstance(config, SystemConfig)
        assert config.deployment_profile == "single-machine"

    def test_load_invalid_yaml_raises(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            fh.write("deployment_profile: [\ninvalid yaml")
        mgr = ConfigManager(config_path=cfg_path)
        with pytest.raises(yaml.YAMLError):
            mgr.load()


# ---------------------------------------------------------------------------
# Tests: Validate
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for ConfigManager.validate()."""

    def test_valid_config_no_errors(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        _write_yaml(cfg_path, _valid_config_dict())
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        assert errors == []

    def test_validate_without_load(self) -> None:
        mgr = ConfigManager(config_path="nonexistent.yaml")
        errors = mgr.validate()
        assert len(errors) == 1
        assert "No configuration loaded" in errors[0].message

    def test_invalid_deployment_profile(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["deployment_profile"] = "invalid-profile"
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "deployment_profile" in field_paths

    def test_invalid_vlm_backend(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["vlm"]["backend"] = "unknown-backend"
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "vlm.backend" in field_paths

    def test_empty_vlm_api_key(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["vlm"]["api_key"] = ""
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "vlm.api_key" in field_paths

    def test_placeholder_vlm_api_key(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["vlm"]["api_key"] = "YOUR_API_KEY_HERE"
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "vlm.api_key" in field_paths

    def test_empty_camera_id(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["camera_id"] = ""
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "cameras[0].camera_id" in field_paths

    def test_empty_camera_uri(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["uri"] = ""
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "cameras[0].uri" in field_paths

    def test_empty_tenant_id(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["tenant_id"] = ""
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "cameras[0].tenant_id" in field_paths

    def test_empty_site_id(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["site_id"] = ""
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "cameras[0].site_id" in field_paths

    def test_confidence_threshold_out_of_range(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["confidence_threshold"] = 1.5
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "cameras[0].confidence_threshold" in field_paths

    def test_confidence_threshold_negative(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["confidence_threshold"] = -0.1
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "cameras[0].confidence_threshold" in field_paths

    def test_invalid_inference_runtime(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["inference_runtime"] = "onnx"
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "cameras[0].inference_runtime" in field_paths

    def test_invalid_tracker_algorithm(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["tracker_algorithm"] = "sort"
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "cameras[0].tracker_algorithm" in field_paths

    def test_invalid_mqtt_port_zero(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["mqtt"]["port"] = 0
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "mqtt.port" in field_paths

    def test_invalid_mqtt_port_negative(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["mqtt"]["port"] = -1
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "mqtt.port" in field_paths

    def test_invalid_timeseries_db(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["storage"]["timeseries_db"] = "postgres"
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "storage.timeseries_db" in field_paths

    def test_invalid_vector_db(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["storage"]["vector_db"] = "milvus"
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        field_paths = [e.field_path for e in errors]
        assert "storage.vector_db" in field_paths

    def test_multiple_errors_reported(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["deployment_profile"] = "bad"
        data["vlm"]["backend"] = "bad"
        data["vlm"]["api_key"] = ""
        _write_yaml(cfg_path, data)
        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()
        assert len(errors) >= 3


# ---------------------------------------------------------------------------
# Tests: YAML round-trip (to_yaml / from_yaml)
# ---------------------------------------------------------------------------


class TestYamlRoundTrip:
    """Tests for to_yaml() and from_yaml()."""

    def test_round_trip_preserves_all_fields(self) -> None:
        cam = CameraConfig(
            camera_id="cam-01",
            uri="rtsp://192.168.1.100:554/stream1",
            tenant_id="tenant-acme",
            site_id="site-hq",
            confidence_threshold=0.7,
            monitored_classes=["person", "vehicle"],
            inference_runtime="pytorch",
            model_path="./models/yolov8n.pt",
            tracker_algorithm="deepsort",
            frame_skip=3,
        )
        original = SystemConfig(
            deployment_profile="single-machine",
            mqtt=BrokerConfig(host="localhost", port=1883, use_tls=False),
            vlm=VLMConfig(
                backend="cosmos",
                api_key="nvapi-test-key",
                endpoint="https://integrate.api.nvidia.com/v1",
                timeout_seconds=30,
            ),
            cameras=[cam],
            storage=StorageConfig(
                timeseries_db="sqlite",
                timeseries_path="./data/events.db",
                vector_db="chromadb",
                vector_path="./data/vectors",
                frame_crop_path="./data/crops",
                frame_crop_encryption_key="test-key",
                retention={
                    "raw_events_days": 90,
                    "aggregated_events_days": 365,
                    "frame_crops_hours": 72,
                },
            ),
            alerts=AlertConfig(
                channels=["push", "webhook"],
                webhook_url="https://hooks.example.com/cctv",
                cooldown=CooldownConfig(
                    default_seconds=60,
                    per_type_overrides={"intrusion": 30},
                ),
            ),
            security=SecurityConfig(
                tls_enabled=False, oauth_enabled=False, tenant_isolation=True
            ),
        )

        mgr = ConfigManager()
        yaml_str = mgr.to_yaml(original)
        restored = mgr.from_yaml(yaml_str)

        assert restored.deployment_profile == original.deployment_profile
        assert restored.mqtt.host == original.mqtt.host
        assert restored.mqtt.port == original.mqtt.port
        assert restored.vlm.backend == original.vlm.backend
        assert restored.vlm.api_key == original.vlm.api_key
        assert restored.vlm.endpoint == original.vlm.endpoint
        assert len(restored.cameras) == len(original.cameras)
        assert restored.cameras[0].camera_id == original.cameras[0].camera_id
        assert restored.cameras[0].uri == original.cameras[0].uri
        assert restored.cameras[0].confidence_threshold == original.cameras[0].confidence_threshold
        assert restored.cameras[0].monitored_classes == original.cameras[0].monitored_classes
        assert restored.storage.timeseries_db == original.storage.timeseries_db
        assert restored.storage.vector_db == original.storage.vector_db
        assert restored.storage.retention == original.storage.retention
        assert restored.alerts.channels == original.alerts.channels
        assert restored.alerts.cooldown.default_seconds == original.alerts.cooldown.default_seconds
        assert restored.security.tenant_isolation == original.security.tenant_isolation

    def test_from_yaml_empty_string(self) -> None:
        mgr = ConfigManager()
        config = mgr.from_yaml("")
        assert isinstance(config, SystemConfig)
        assert config.deployment_profile == "single-machine"

    def test_to_yaml_produces_valid_yaml(self) -> None:
        config = SystemConfig()
        mgr = ConfigManager()
        yaml_str = mgr.to_yaml(config)
        data = yaml.safe_load(yaml_str)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Tests: Windows path handling
# ---------------------------------------------------------------------------


class TestWindowsPaths:
    """Tests for Windows-compatible file path handling."""

    def test_load_with_backslash_path(self, tmp_path: object) -> None:
        # Simulate a Windows-style path (works on all platforms via os.path)
        cfg_path = os.path.join(str(tmp_path), "subdir", "config.yaml")
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        _write_yaml(cfg_path, _valid_config_dict())
        mgr = ConfigManager(config_path=cfg_path)
        config = mgr.load()
        assert isinstance(config, SystemConfig)

    def test_generate_default_with_nested_path(self, tmp_path: object) -> None:
        cfg_path = os.path.join(str(tmp_path), "deep", "nested", "dir", "config.yaml")
        mgr = ConfigManager(config_path=cfg_path)
        mgr.generate_default()
        assert os.path.isfile(cfg_path)


# ---------------------------------------------------------------------------
# Tests: VLM Video Snippet config fields (Task 1.7)
# ---------------------------------------------------------------------------


class TestVLMVideoSnippetConfigFields:
    """Unit tests for vlm_input_mode and vlm_video_duration_seconds fields.

    Requirements: 1.2, 1.3, 1.5, 1.6, 1.7, 7.1, 7.2, 7.3
    """

    def test_defaults_applied_when_fields_absent(self, tmp_path: object) -> None:
        """When vlm_input_mode and vlm_video_duration_seconds are absent from
        the YAML, defaults of 'image' and 10 are applied."""
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        # Ensure the new fields are NOT present
        assert "vlm_input_mode" not in data["cameras"][0]
        assert "vlm_video_duration_seconds" not in data["cameras"][0]
        _write_yaml(cfg_path, data)

        mgr = ConfigManager(config_path=cfg_path)
        config = mgr.load()

        assert config.cameras[0].vlm_input_mode == "image"
        assert config.cameras[0].vlm_video_duration_seconds == 10

    def test_existing_config_example_validates_cleanly(self) -> None:
        """The existing config.example.yaml should produce zero validation
        errors related to the new VLM fields (backward compatibility)."""
        if not os.path.isfile("config.example.yaml"):
            pytest.skip("config.example.yaml not found")

        mgr = ConfigManager(config_path="config.example.yaml")
        mgr.load()
        errors = mgr.validate()

        # Filter to only errors about the new fields
        vlm_field_errors = [
            e for e in errors
            if "vlm_input_mode" in e.field_path
            or "vlm_video_duration_seconds" in e.field_path
        ]
        assert vlm_field_errors == [], (
            f"Expected no validation errors for new VLM fields, got: {vlm_field_errors}"
        )

    def test_invalid_vlm_input_mode_rejected(self, tmp_path: object) -> None:
        """An invalid vlm_input_mode value is rejected with a descriptive error."""
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["vlm_input_mode"] = "stream"
        _write_yaml(cfg_path, data)

        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()

        field_paths = [e.field_path for e in errors]
        assert "cameras[0].vlm_input_mode" in field_paths
        # Check the error message mentions allowed values
        vlm_errors = [e for e in errors if "vlm_input_mode" in e.field_path]
        assert any("image" in e.message and "video" in e.message for e in vlm_errors)

    def test_vlm_video_duration_zero_rejected(self, tmp_path: object) -> None:
        """vlm_video_duration_seconds = 0 is rejected."""
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["vlm_video_duration_seconds"] = 0
        _write_yaml(cfg_path, data)

        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()

        field_paths = [e.field_path for e in errors]
        assert "cameras[0].vlm_video_duration_seconds" in field_paths

    def test_vlm_video_duration_61_rejected(self, tmp_path: object) -> None:
        """vlm_video_duration_seconds = 61 is rejected."""
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["vlm_video_duration_seconds"] = 61
        _write_yaml(cfg_path, data)

        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()

        field_paths = [e.field_path for e in errors]
        assert "cameras[0].vlm_video_duration_seconds" in field_paths

    def test_vlm_video_duration_negative_rejected(self, tmp_path: object) -> None:
        """vlm_video_duration_seconds = -1 is rejected."""
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["vlm_video_duration_seconds"] = -1
        _write_yaml(cfg_path, data)

        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()

        field_paths = [e.field_path for e in errors]
        assert "cameras[0].vlm_video_duration_seconds" in field_paths

    def test_vlm_input_mode_image_accepted_without_duration(self, tmp_path: object) -> None:
        """vlm_input_mode='image' is accepted regardless of whether
        vlm_video_duration_seconds is present."""
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["vlm_input_mode"] = "image"
        # Explicitly omit vlm_video_duration_seconds
        _write_yaml(cfg_path, data)

        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()

        vlm_field_errors = [
            e for e in errors
            if "vlm_input_mode" in e.field_path
            or "vlm_video_duration_seconds" in e.field_path
        ]
        assert vlm_field_errors == []

    def test_vlm_input_mode_video_with_valid_duration_accepted(self, tmp_path: object) -> None:
        """vlm_input_mode='video' with vlm_video_duration_seconds=30 is accepted."""
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["vlm_input_mode"] = "video"
        data["cameras"][0]["vlm_video_duration_seconds"] = 30
        _write_yaml(cfg_path, data)

        mgr = ConfigManager(config_path=cfg_path)
        config = mgr.load()
        errors = mgr.validate()

        vlm_field_errors = [
            e for e in errors
            if "vlm_input_mode" in e.field_path
            or "vlm_video_duration_seconds" in e.field_path
        ]
        assert vlm_field_errors == []
        assert config.cameras[0].vlm_input_mode == "video"
        assert config.cameras[0].vlm_video_duration_seconds == 30

    def test_vlm_video_duration_boundary_1_accepted(self, tmp_path: object) -> None:
        """vlm_video_duration_seconds = 1 (lower boundary) is accepted."""
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["vlm_video_duration_seconds"] = 1
        _write_yaml(cfg_path, data)

        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()

        duration_errors = [
            e for e in errors if "vlm_video_duration_seconds" in e.field_path
        ]
        assert duration_errors == []

    def test_vlm_video_duration_boundary_60_accepted(self, tmp_path: object) -> None:
        """vlm_video_duration_seconds = 60 (upper boundary) is accepted."""
        cfg_path = os.path.join(str(tmp_path), "config.yaml")
        data = _valid_config_dict()
        data["cameras"][0]["vlm_video_duration_seconds"] = 60
        _write_yaml(cfg_path, data)

        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()
        errors = mgr.validate()

        duration_errors = [
            e for e in errors if "vlm_video_duration_seconds" in e.field_path
        ]
        assert duration_errors == []

    def test_roundtrip_preserves_new_fields(self) -> None:
        """YAML round-trip preserves vlm_input_mode and vlm_video_duration_seconds."""
        cam = CameraConfig(
            camera_id="cam-01",
            uri="rtsp://192.168.1.100:554/stream1",
            tenant_id="tenant-acme",
            site_id="site-hq",
            confidence_threshold=0.7,
            vlm_input_mode="video",
            vlm_video_duration_seconds=25,
        )
        original = SystemConfig(
            cameras=[cam],
            vlm=VLMConfig(api_key="nvapi-test-key"),
        )

        mgr = ConfigManager()
        yaml_str = mgr.to_yaml(original)
        restored = mgr.from_yaml(yaml_str)

        assert restored.cameras[0].vlm_input_mode == "video"
        assert restored.cameras[0].vlm_video_duration_seconds == 25
