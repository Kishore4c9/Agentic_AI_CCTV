"""Pre-built environment templates for the Agentic AI CCTV Monitoring Framework.

Each template produces a valid ``SystemConfig`` with appropriate settings for
a specific deployment environment (home, farm, forest, mall, port, GPU desktop).

Templates specify default ``inference_runtime``, ``deployment_profile``,
power profile, and hardware class.  Operators can override any setting via
keyword arguments passed to ``generate_config_from_template``.

Uses ``from __future__ import annotations`` for PEP 604 union-type syntax
on Python 3.9+.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class EnvironmentTemplate:
    """Describes a pre-built deployment environment template.

    Attributes
    ----------
    name:
        Short identifier (e.g. ``"home"``, ``"farm"``).
    description:
        Human-readable description of the environment.
    deployment_profile:
        One of ``"single-machine"``, ``"multi-machine"``, ``"edge-cloud-hybrid"``.
    inference_runtime:
        Default inference runtime — ``"pytorch"`` or ``"tensorrt"``.
    power_profile:
        Descriptive power profile (e.g. ``"mains"``, ``"solar_battery"``, ``"battery_lora"``).
    hardware_class:
        Target hardware (e.g. ``"jetson_nano"``, ``"coral_dev_board_micro"``).
    """

    name: str
    description: str
    deployment_profile: str
    inference_runtime: str
    power_profile: str
    hardware_class: str

    def to_system_config(self, **overrides: Any) -> SystemConfig:
        """Generate a valid ``SystemConfig`` from this template.

        Parameters
        ----------
        **overrides:
            Keyword arguments that override template defaults.  Supported keys:

            * ``inference_runtime`` — override the default runtime
            * ``deployment_profile`` — override the deployment profile
            * ``vlm_api_key`` — set the VLM API key
            * ``vlm_backend`` — set the VLM backend
            * ``mqtt_host`` — MQTT broker host
            * ``mqtt_port`` — MQTT broker port
            * ``camera_id`` — placeholder camera ID
            * ``camera_uri`` — placeholder camera URI
            * ``tenant_id`` — tenant identifier
            * ``site_id`` — site identifier

        Returns
        -------
        SystemConfig
            A fully populated configuration.
        """
        runtime = overrides.get("inference_runtime", self.inference_runtime)
        profile = overrides.get("deployment_profile", self.deployment_profile)

        mqtt = BrokerConfig(
            host=overrides.get("mqtt_host", "localhost"),
            port=overrides.get("mqtt_port", 1883),
        )

        vlm = VLMConfig(
            backend=overrides.get("vlm_backend", "cosmos"),
            api_key=overrides.get("vlm_api_key", "YOUR_API_KEY_HERE"),
            endpoint=overrides.get(
                "vlm_endpoint", "https://integrate.api.nvidia.com/v1"
            ),
            timeout_seconds=overrides.get("vlm_timeout_seconds", 30),
        )

        camera = CameraConfig(
            camera_id=overrides.get("camera_id", f"cam-{self.name}-01"),
            uri=overrides.get("camera_uri", "rtsp://192.168.1.100:554/stream1"),
            tenant_id=overrides.get("tenant_id", "tenant-default"),
            site_id=overrides.get("site_id", "site-default"),
            confidence_threshold=overrides.get("confidence_threshold", 0.7),
            monitored_classes=overrides.get(
                "monitored_classes", ["person", "vehicle"]
            ),
            inference_runtime=runtime,
            model_path=overrides.get("model_path", "./models/yolov8n.pt"),
            tracker_algorithm=overrides.get("tracker_algorithm", "deepsort"),
            frame_skip=overrides.get("frame_skip", 3),
        )

        storage = StorageConfig(
            timeseries_db=overrides.get("timeseries_db", "sqlite"),
            timeseries_path=overrides.get("timeseries_path", "./data/events.db"),
            vector_db=overrides.get("vector_db", "chromadb"),
            vector_path=overrides.get("vector_path", "./data/vectors"),
            frame_crop_path=overrides.get("frame_crop_path", "./data/crops"),
            frame_crop_encryption_key=overrides.get(
                "frame_crop_encryption_key", ""
            ),
            retention=overrides.get(
                "retention",
                {
                    "raw_events_days": 90,
                    "aggregated_events_days": 365,
                    "frame_crops_hours": 72,
                },
            ),
        )

        alerts = AlertConfig(
            channels=overrides.get("alert_channels", ["push", "webhook"]),
            webhook_url=overrides.get(
                "webhook_url", "https://hooks.example.com/cctv"
            ),
            cooldown=CooldownConfig(
                default_seconds=overrides.get("cooldown_seconds", 60),
                per_type_overrides=overrides.get(
                    "cooldown_overrides", {"intrusion": 30, "fire": 10}
                ),
            ),
        )

        security = SecurityConfig(
            tls_enabled=overrides.get("tls_enabled", False),
            oauth_enabled=overrides.get("oauth_enabled", False),
            tenant_isolation=overrides.get("tenant_isolation", True),
        )

        return SystemConfig(
            deployment_profile=profile,
            mqtt=mqtt,
            vlm=vlm,
            cameras=[camera],
            storage=storage,
            alerts=alerts,
            security=security,
        )


# ---------------------------------------------------------------------------
# Pre-built templates
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, EnvironmentTemplate] = {}


def _register(template: EnvironmentTemplate) -> EnvironmentTemplate:
    """Register a template in the global registry."""
    _TEMPLATES[template.name] = template
    return template


HOME_TEMPLATE = _register(
    EnvironmentTemplate(
        name="home",
        description=(
            "Home / apartment deployment — mains power, low-latency local "
            "processing on Jetson Nano or Raspberry Pi 5."
        ),
        deployment_profile="single-machine",
        inference_runtime="pytorch",
        power_profile="mains",
        hardware_class="jetson_nano",
    )
)

FARM_TEMPLATE = _register(
    EnvironmentTemplate(
        name="farm",
        description=(
            "Farm / rural deployment — solar/battery power, power-optimised "
            "inference on Jetson Nano or Coral Dev Board."
        ),
        deployment_profile="single-machine",
        inference_runtime="tensorrt",
        power_profile="solar_battery",
        hardware_class="jetson_nano",
    )
)

FOREST_TEMPLATE = _register(
    EnvironmentTemplate(
        name="forest",
        description=(
            "Forest / wildlife reserve deployment — battery/LoRa connectivity, "
            "minimal power consumption on Coral Dev Board Micro."
        ),
        deployment_profile="single-machine",
        inference_runtime="tensorrt",
        power_profile="battery_lora",
        hardware_class="coral_dev_board_micro",
    )
)

MALL_TEMPLATE = _register(
    EnvironmentTemplate(
        name="mall",
        description=(
            "Shopping mall deployment — mains power, high camera density "
            "support on Jetson Orin NX."
        ),
        deployment_profile="single-machine",
        inference_runtime="tensorrt",
        power_profile="mains",
        hardware_class="jetson_orin_nx",
    )
)

PORT_TEMPLATE = _register(
    EnvironmentTemplate(
        name="port",
        description=(
            "Port / logistics / industrial facility deployment — mains power, "
            "24/7 high-reliability operation on Jetson Orin NX."
        ),
        deployment_profile="single-machine",
        inference_runtime="tensorrt",
        power_profile="mains",
        hardware_class="jetson_orin_nx",
    )
)

GPU_DESKTOP_TEMPLATE = _register(
    EnvironmentTemplate(
        name="gpu_desktop",
        description=(
            "GPU desktop / server deployment — mains power, high-throughput "
            "inference using PyTorch or TensorRT on NVIDIA GPU (RTX 3060+)."
        ),
        deployment_profile="single-machine",
        inference_runtime="pytorch",
        power_profile="mains",
        hardware_class="nvidia_gpu_desktop",
    )
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_template(name: str) -> EnvironmentTemplate:
    """Return the pre-built template with the given *name*.

    Parameters
    ----------
    name:
        Template identifier (e.g. ``"home"``, ``"farm"``).

    Returns
    -------
    EnvironmentTemplate

    Raises
    ------
    KeyError
        If no template with *name* exists.
    """
    if name not in _TEMPLATES:
        available = ", ".join(sorted(_TEMPLATES))
        raise KeyError(
            f"Unknown environment template '{name}'. "
            f"Available templates: {available}"
        )
    return _TEMPLATES[name]


def list_templates() -> list[EnvironmentTemplate]:
    """Return all pre-built environment templates.

    Returns
    -------
    list[EnvironmentTemplate]
        A list of all registered templates, sorted by name.
    """
    return sorted(_TEMPLATES.values(), key=lambda t: t.name)


def generate_config_from_template(
    template_name: str, **overrides: Any
) -> SystemConfig:
    """Generate a ``SystemConfig`` from a named template with optional overrides.

    This is a convenience wrapper around
    ``get_template(name).to_system_config(**overrides)``.

    Parameters
    ----------
    template_name:
        Name of the pre-built template.
    **overrides:
        Keyword arguments forwarded to
        :meth:`EnvironmentTemplate.to_system_config`.

    Returns
    -------
    SystemConfig
    """
    template = get_template(template_name)
    return template.to_system_config(**overrides)
