"""Configuration Manager for the Agentic AI CCTV Monitoring Framework.

Loads, validates, and manages the system configuration file (config.yaml).
Supports generating default configs, round-trip YAML serialization, and
descriptive validation error reporting.

Uses ``from __future__ import annotations`` for PEP 604 union-type syntax
on Python 3.9+.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation Error
# ---------------------------------------------------------------------------


@dataclass
class ValidationError:
    """Describes a single configuration validation error."""

    field_path: str
    message: str


# ---------------------------------------------------------------------------
# Valid value sets
# ---------------------------------------------------------------------------

VALID_DEPLOYMENT_PROFILES = {"single-machine", "multi-machine", "edge-cloud-hybrid"}
VALID_VLM_BACKENDS = {"cosmos", "gpt4o", "claude3", "gemini15"}
VALID_VLM_INPUT_MODES = {"image", "video"}
VALID_INFERENCE_RUNTIMES = {"pytorch", "tensorrt"}
VALID_TRACKER_ALGORITHMS = {"deepsort", "bytetrack"}
VALID_TIMESERIES_DBS = {"sqlite", "influxdb", "timescaledb"}
VALID_VECTOR_DBS = {"chromadb", "pinecone", "weaviate"}


# ---------------------------------------------------------------------------
# Default config template (as a Python dict)
# ---------------------------------------------------------------------------


def _default_config_dict() -> dict[str, Any]:
    """Return the default configuration as a plain dict for YAML output."""
    return {
        "deployment_profile": "single-machine",
        "mqtt": {
            "host": "localhost",
            "port": 1883,
            "use_tls": False,
            "ca_cert": None,
            "client_cert": None,
            "client_key": None,
            "username": None,
            "password": None,
        },
        "vlm": {
            "backend": "cosmos",
            "api_key": "YOUR_API_KEY_HERE",
            "endpoint": "https://integrate.api.nvidia.com/v1",
            "timeout_seconds": 30,
        },
        "cameras": [
            {
                "camera_id": "cam-lobby-01",
                "uri": "rtsp://192.168.1.100:554/stream1",
                "tenant_id": "tenant-default",
                "site_id": "site-default",
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
                "per_type_overrides": {
                    "intrusion": 30,
                    "fire": 10,
                },
            },
        },
        "security": {
            "tls_enabled": False,
            "oauth_enabled": False,
            "tenant_isolation": True,
        },
    }


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------


def _build_broker_config(data: dict[str, Any] | None) -> BrokerConfig:
    """Build a BrokerConfig from a raw dict, using defaults for missing keys."""
    if not data:
        return BrokerConfig()
    return BrokerConfig(
        host=data.get("host", "localhost"),
        port=data.get("port", 1883),
        use_tls=data.get("use_tls", False),
        ca_cert=data.get("ca_cert"),
        client_cert=data.get("client_cert"),
        client_key=data.get("client_key"),
        username=data.get("username"),
        password=data.get("password"),
    )


def _build_vlm_config(data: dict[str, Any] | None) -> VLMConfig:
    """Build a VLMConfig from a raw dict."""
    if not data:
        return VLMConfig()
    return VLMConfig(
        backend=data.get("backend", "cosmos"),
        api_key=data.get("api_key", ""),
        endpoint=data.get("endpoint"),
        timeout_seconds=data.get("timeout_seconds", 30),
    )


def _build_camera_config(data: dict[str, Any]) -> CameraConfig:
    """Build a CameraConfig from a raw dict."""
    return CameraConfig(
        camera_id=data.get("camera_id", ""),
        uri=data.get("uri", ""),
        tenant_id=data.get("tenant_id", ""),
        site_id=data.get("site_id", ""),
        confidence_threshold=float(data.get("confidence_threshold", 0.5)),
        monitored_classes=data.get("monitored_classes", []),
        inference_runtime=data.get("inference_runtime", "pytorch"),
        model_path=data.get("model_path", "./models/yolov8n.pt"),
        tracker_algorithm=data.get("tracker_algorithm", "deepsort"),
        frame_skip=data.get("frame_skip", 3),
        vlm_input_mode=data.get("vlm_input_mode", "image"),
        vlm_video_duration_seconds=data.get("vlm_video_duration_seconds", 10),
    )


def _build_cooldown_config(data: dict[str, Any] | None) -> CooldownConfig:
    """Build a CooldownConfig from a raw dict."""
    if not data:
        return CooldownConfig()
    return CooldownConfig(
        default_seconds=data.get("default_seconds", 60),
        per_type_overrides=data.get("per_type_overrides", {}),
    )


def _build_storage_config(data: dict[str, Any] | None) -> StorageConfig:
    """Build a StorageConfig from a raw dict."""
    if not data:
        return StorageConfig()
    return StorageConfig(
        timeseries_db=data.get("timeseries_db", "sqlite"),
        timeseries_path=data.get("timeseries_path", "./data/events.db"),
        vector_db=data.get("vector_db", "chromadb"),
        vector_path=data.get("vector_path", "./data/vectors"),
        frame_crop_path=data.get("frame_crop_path", "./data/crops"),
        frame_crop_encryption_key=data.get("frame_crop_encryption_key", ""),
        retention=data.get(
            "retention",
            {
                "raw_events_days": 90,
                "aggregated_events_days": 365,
                "frame_crops_hours": 72,
            },
        ),
    )


def _build_alert_config(data: dict[str, Any] | None) -> AlertConfig:
    """Build an AlertConfig from a raw dict."""
    if not data:
        return AlertConfig()
    return AlertConfig(
        channels=data.get("channels", ["push", "webhook"]),
        webhook_url=data.get("webhook_url"),
        cooldown=_build_cooldown_config(data.get("cooldown")),
    )


def _build_security_config(data: dict[str, Any] | None) -> SecurityConfig:
    """Build a SecurityConfig from a raw dict."""
    if not data:
        return SecurityConfig()
    return SecurityConfig(
        tls_enabled=data.get("tls_enabled", False),
        oauth_enabled=data.get("oauth_enabled", False),
        tenant_isolation=data.get("tenant_isolation", True),
    )


def _build_system_config(data: dict[str, Any]) -> SystemConfig:
    """Build a full SystemConfig from a raw dict parsed from YAML."""
    cameras_raw = data.get("cameras", [])
    cameras = [_build_camera_config(c) for c in cameras_raw] if cameras_raw else []

    return SystemConfig(
        deployment_profile=data.get("deployment_profile", "single-machine"),
        mqtt=_build_broker_config(data.get("mqtt")),
        vlm=_build_vlm_config(data.get("vlm")),
        cameras=cameras,
        storage=_build_storage_config(data.get("storage")),
        alerts=_build_alert_config(data.get("alerts")),
        security=_build_security_config(data.get("security")),
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _system_config_to_dict(config: SystemConfig) -> dict[str, Any]:
    """Convert a SystemConfig to a plain dict suitable for YAML serialization.

    Uses ``dataclasses.asdict`` but cleans up ``None`` values for readability.
    """
    return asdict(config)


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------


class ConfigManager:
    """Loads, validates, and manages the system configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.  Defaults to ``"config.yaml"``.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config_path = config_path
        self._config: SystemConfig | None = None

    # -- public API ---------------------------------------------------------

    def load(self) -> SystemConfig:
        """Read and parse the configuration file.

        If the file does not exist, :meth:`generate_default` is called first
        and a warning is logged before loading.

        Returns
        -------
        SystemConfig
            The parsed system configuration.

        Raises
        ------
        yaml.YAMLError
            If the file contains invalid YAML syntax.
        """
        if not os.path.isfile(self.config_path):
            logger.warning(
                "Configuration file not found at '%s'. "
                "Generating default configuration.",
                self.config_path,
            )
            self.generate_default()

        with open(self.config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if raw is None:
            raw = {}

        self._config = _build_system_config(raw)
        return self._config

    def validate(self) -> list[ValidationError]:
        """Validate the currently loaded configuration.

        Must be called after :meth:`load`.

        Returns
        -------
        list[ValidationError]
            A (possibly empty) list of validation errors.
        """
        if self._config is None:
            return [
                ValidationError(
                    field_path="(root)",
                    message="No configuration loaded. Call load() first.",
                )
            ]

        errors: list[ValidationError] = []
        cfg = self._config

        # -- deployment_profile ------------------------------------------------
        if cfg.deployment_profile not in VALID_DEPLOYMENT_PROFILES:
            errors.append(
                ValidationError(
                    field_path="deployment_profile",
                    message=(
                        f"Invalid deployment_profile '{cfg.deployment_profile}'. "
                        f"Must be one of: {sorted(VALID_DEPLOYMENT_PROFILES)}"
                    ),
                )
            )

        # -- vlm ---------------------------------------------------------------
        if cfg.vlm.backend not in VALID_VLM_BACKENDS:
            errors.append(
                ValidationError(
                    field_path="vlm.backend",
                    message=(
                        f"Invalid vlm.backend '{cfg.vlm.backend}'. "
                        f"Must be one of: {sorted(VALID_VLM_BACKENDS)}"
                    ),
                )
            )

        if not cfg.vlm.api_key or cfg.vlm.api_key in (
            "YOUR_API_KEY_HERE",
            "",
        ):
            errors.append(
                ValidationError(
                    field_path="vlm.api_key",
                    message=(
                        "vlm.api_key must not be empty or a placeholder value. "
                        "Please provide a valid API key."
                    ),
                )
            )

        # -- cameras -----------------------------------------------------------
        for idx, cam in enumerate(cfg.cameras):
            prefix = f"cameras[{idx}]"

            if not cam.camera_id:
                errors.append(
                    ValidationError(
                        field_path=f"{prefix}.camera_id",
                        message="camera_id must not be empty.",
                    )
                )

            if not cam.uri:
                errors.append(
                    ValidationError(
                        field_path=f"{prefix}.uri",
                        message="uri must not be empty.",
                    )
                )

            if not cam.tenant_id:
                errors.append(
                    ValidationError(
                        field_path=f"{prefix}.tenant_id",
                        message="tenant_id must not be empty.",
                    )
                )

            if not cam.site_id:
                errors.append(
                    ValidationError(
                        field_path=f"{prefix}.site_id",
                        message="site_id must not be empty.",
                    )
                )

            if not (0.0 <= cam.confidence_threshold <= 1.0):
                errors.append(
                    ValidationError(
                        field_path=f"{prefix}.confidence_threshold",
                        message=(
                            f"confidence_threshold must be in [0.0, 1.0], "
                            f"got {cam.confidence_threshold}."
                        ),
                    )
                )

            if cam.inference_runtime not in VALID_INFERENCE_RUNTIMES:
                errors.append(
                    ValidationError(
                        field_path=f"{prefix}.inference_runtime",
                        message=(
                            f"Invalid inference_runtime '{cam.inference_runtime}'. "
                            f"Must be one of: {sorted(VALID_INFERENCE_RUNTIMES)}"
                        ),
                    )
                )

            if cam.tracker_algorithm not in VALID_TRACKER_ALGORITHMS:
                errors.append(
                    ValidationError(
                        field_path=f"{prefix}.tracker_algorithm",
                        message=(
                            f"Invalid tracker_algorithm '{cam.tracker_algorithm}'. "
                            f"Must be one of: {sorted(VALID_TRACKER_ALGORITHMS)}"
                        ),
                    )
                )

            if cam.vlm_input_mode not in VALID_VLM_INPUT_MODES:
                errors.append(
                    ValidationError(
                        field_path=f"{prefix}.vlm_input_mode",
                        message=(
                            f"Invalid vlm_input_mode '{cam.vlm_input_mode}'. "
                            f"Must be one of: {sorted(VALID_VLM_INPUT_MODES)}"
                        ),
                    )
                )

            if not (1 <= cam.vlm_video_duration_seconds <= 60):
                errors.append(
                    ValidationError(
                        field_path=f"{prefix}.vlm_video_duration_seconds",
                        message=(
                            f"vlm_video_duration_seconds must be between 1 and 60 inclusive, "
                            f"got {cam.vlm_video_duration_seconds}."
                        ),
                    )
                )

        # -- mqtt --------------------------------------------------------------
        if not isinstance(cfg.mqtt.port, int) or cfg.mqtt.port <= 0:
            errors.append(
                ValidationError(
                    field_path="mqtt.port",
                    message=(
                        f"mqtt.port must be a positive integer, got {cfg.mqtt.port}."
                    ),
                )
            )

        # -- storage -----------------------------------------------------------
        if cfg.storage.timeseries_db not in VALID_TIMESERIES_DBS:
            errors.append(
                ValidationError(
                    field_path="storage.timeseries_db",
                    message=(
                        f"Invalid storage.timeseries_db '{cfg.storage.timeseries_db}'. "
                        f"Must be one of: {sorted(VALID_TIMESERIES_DBS)}"
                    ),
                )
            )

        if cfg.storage.vector_db not in VALID_VECTOR_DBS:
            errors.append(
                ValidationError(
                    field_path="storage.vector_db",
                    message=(
                        f"Invalid storage.vector_db '{cfg.storage.vector_db}'. "
                        f"Must be one of: {sorted(VALID_VECTOR_DBS)}"
                    ),
                )
            )

        return errors

    def generate_default(self) -> None:
        """Write a default configuration file with placeholder values.

        Creates any intermediate directories as needed.
        """
        dir_name = os.path.dirname(self.config_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        default = _default_config_dict()
        with open(self.config_path, "w", encoding="utf-8") as fh:
            yaml.dump(
                default,
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def to_yaml(self, config: SystemConfig) -> str:
        """Serialize a :class:`SystemConfig` to a YAML string.

        Parameters
        ----------
        config:
            The configuration to serialize.

        Returns
        -------
        str
            YAML representation of the configuration.
        """
        data = _system_config_to_dict(config)
        return yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    def load_template(
        self, template_name: str, **overrides: Any
    ) -> SystemConfig:
        """Generate a ``SystemConfig`` from a pre-built environment template.

        Parameters
        ----------
        template_name:
            Name of the template (e.g. ``"home"``, ``"farm"``).
        **overrides:
            Keyword arguments forwarded to the template's
            ``to_system_config`` method.

        Returns
        -------
        SystemConfig
            The generated configuration.
        """
        from agentic_cctv.environment_templates import generate_config_from_template

        config = generate_config_from_template(template_name, **overrides)
        self._config = config
        return config

    def from_yaml(self, yaml_str: str) -> SystemConfig:
        """Deserialize a YAML string into a :class:`SystemConfig`.

        Parameters
        ----------
        yaml_str:
            YAML content to parse.

        Returns
        -------
        SystemConfig
            The parsed configuration.

        Raises
        ------
        yaml.YAMLError
            If the string contains invalid YAML.
        """
        raw = yaml.safe_load(yaml_str)
        if raw is None:
            raw = {}
        return _build_system_config(raw)
