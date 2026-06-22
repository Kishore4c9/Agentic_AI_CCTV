"""Shared data models for the Agentic AI CCTV Monitoring Framework.

All models are defined as Python dataclasses. Uses ``from __future__ import annotations``
for PEP 604 union-type syntax on Python 3.9+.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Edge Layer Models
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """A single decoded video frame from a camera source."""

    camera_id: str
    timestamp: datetime
    image: Any  # numpy ndarray (BGR); typed as Any to avoid numpy import issues
    frame_number: int


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int


@dataclass
class RawDetection:
    """Raw output from the inference runtime before gate filtering."""

    object_type: str
    bounding_box: BoundingBox
    confidence: float


@dataclass
class Detection:
    """A detection result after the Detection Gate has been applied."""

    object_type: str
    bounding_box: BoundingBox
    confidence: float
    passed_gate: bool  # True iff confidence >= threshold AND object_type in monitored classes


@dataclass
class Track:
    """A tracked object with a persistent identity across frames."""

    track_id: str
    object_type: str
    bounding_box: BoundingBox
    confidence: float
    age: int  # frames since first seen
    is_new: bool  # True on first appearance


# ---------------------------------------------------------------------------
# Event / MQTT Payload Models
# ---------------------------------------------------------------------------


@dataclass
class StructuredEvent:
    """JSON event emitted by the Edge Node and published to MQTT."""

    event_id: str  # UUID
    camera_id: str
    tenant_id: str
    site_id: str
    timestamp: datetime
    object_type: str
    track_id: str
    bounding_box: BoundingBox
    confidence: float
    frame_crop: str  # base64-encoded JPEG
    video_snippet: Optional[str] = None  # base64-encoded MP4 (when video mode)
    media_type: str = "image"  # "image" or "video"


@dataclass
class HeartbeatMessage:
    """Health heartbeat published as an MQTT retained message."""

    camera_id: str
    tenant_id: str
    site_id: str
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    temperature_celsius: Optional[float]
    inference_latency_ms: float
    gpu_utilization_percent: Optional[float]


# ---------------------------------------------------------------------------
# VLM / Scene Understanding Models
# ---------------------------------------------------------------------------


@dataclass
class IdentifiedObject:
    """An object identified by the VLM in a scene."""

    type: str
    action: str
    location: str


@dataclass
class SceneUnderstanding:
    """Structured output from the VLM Reasoner."""

    event_id: str
    scene_description: str
    threat_level: str  # "none", "low", "medium", "high", "critical"
    objects_identified: list[IdentifiedObject] = field(default_factory=list)
    recommended_action: str = "log"  # "alert", "log", "summarise", "escalate"
    confidence: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Alert Models
# ---------------------------------------------------------------------------


@dataclass
class AlertPayload:
    """Payload delivered through alert channels."""

    alert_id: str
    event_id: str
    camera_id: str
    tenant_id: str
    site_id: str
    timestamp: datetime
    alert_type: str
    description: str
    threat_level: str
    frame_crop_url: Optional[str]  # pre-signed URL
    scene_understanding: Optional[SceneUnderstanding] = None


@dataclass
class AlertResult:
    """Result of an alert delivery attempt."""

    delivered: bool
    channels: list[str] = field(default_factory=list)
    suppressed: bool = False
    suppressed_count: int = 0


@dataclass
class CooldownConfig:
    """Configuration for alert cooldown deduplication."""

    default_seconds: int = 60
    per_type_overrides: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestration Models
# ---------------------------------------------------------------------------


@dataclass
class ActionResult:
    """Result of the Orchestration Agent's action decision."""

    action: str  # "alert", "log", "summarise", "escalate"
    alert_payload: Optional[AlertPayload] = None
    context_update: Optional[dict[str, Any]] = None
    cross_camera_refs: list[str] = field(default_factory=list)  # related event_ids


# ---------------------------------------------------------------------------
# Rule / Context Filter Models
# ---------------------------------------------------------------------------


@dataclass
class TimeWindow:
    """A time-of-day window for rule evaluation."""

    start: str  # HH:MM format
    end: str  # HH:MM format


@dataclass
class Zone:
    """A polygonal zone defined by a list of (x, y) vertices."""

    polygon: list[list[int]] = field(default_factory=list)


@dataclass
class SuppressCondition:
    """Condition under which a matching rule should suppress the event."""

    object_type: Optional[str] = None
    time_window: Optional[TimeWindow] = None


@dataclass
class CompoundCondition:
    """A compound condition combining multiple sub-conditions."""

    operator: str = "and"  # "and" or "or"
    conditions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Rule:
    """A single rule within a RuleSet."""

    rule_id: str
    object_type: Optional[str] = None
    min_confidence: Optional[float] = None
    time_window: Optional[TimeWindow] = None
    zone: Optional[Zone] = None
    suppress_if: Optional[SuppressCondition] = None
    compound: Optional[CompoundCondition] = None


@dataclass
class RuleSet:
    """A versioned collection of rules for a specific camera."""

    version_id: str
    camera_id: str
    rules: list[Rule] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RuleSetVersion:
    """Metadata for a single version in the rule-set history."""

    version_id: str
    camera_id: str
    created_at: datetime
    is_active: bool = False


@dataclass
class CompiledRuleSet:
    """Output of the Prompt Compiler: a RuleSet with explanation metadata."""

    ruleset: RuleSet
    original_prompt: str
    explanation: str  # human-readable explanation of compiled rules
    confidence: float


@dataclass
class FilterResult:
    """Result of evaluating a StructuredEvent against a RuleSet."""

    passed: bool
    matched_rules: list[str] = field(default_factory=list)  # IDs of matching rules
    suppressed_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompt Compiler Models
# ---------------------------------------------------------------------------


@dataclass
class PromptScope:
    """Scope for a natural-language prompt (camera, group, or site)."""

    scope_type: str  # "camera", "camera_group", "site"
    target_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Health / Watchdog Models
# ---------------------------------------------------------------------------


@dataclass
class DeviceHealth:
    """Current health status of a camera / Edge Node."""

    camera_id: str
    status: str  # "online", "offline", "degraded"
    last_heartbeat: Optional[datetime] = None
    metrics: Optional[HeartbeatMessage] = None


# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------


@dataclass
class BrokerConfig:
    """MQTT broker connection configuration."""

    host: str = "localhost"
    port: int = 1883
    use_tls: bool = False
    ca_cert: Optional[str] = None
    client_cert: Optional[str] = None
    client_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    tls_insecure: bool = False
    connect_timeout: int = 30


@dataclass
class CameraConfig:
    """Per-camera configuration."""

    camera_id: str
    uri: str  # RTSP URL, USB index, or file path
    tenant_id: str
    site_id: str
    confidence_threshold: float
    monitored_classes: list[str] = field(default_factory=list)
    inference_runtime: str = "pytorch"  # "pytorch" or "tensorrt"
    model_path: str = "./models/yolov8n.pt"
    tracker_algorithm: str = "deepsort"  # "deepsort" or "bytetrack"
    frame_skip: int = 3
    vlm_input_mode: str = "image"  # "image" or "video"
    vlm_video_duration_seconds: int = 10  # 1–60 seconds


@dataclass
class VLMConfig:
    """VLM backend configuration."""

    backend: str = "cosmos"  # "cosmos", "gpt4o", "claude3", "gemini15"
    api_key: str = ""
    endpoint: Optional[str] = None
    timeout_seconds: int = 30


@dataclass
class StorageConfig:
    """Storage layer configuration."""

    timeseries_db: str = "sqlite"
    timeseries_path: str = "./data/events.db"
    vector_db: str = "chromadb"
    vector_path: str = "./data/vectors"
    frame_crop_path: str = "./data/crops"
    frame_crop_encryption_key: str = ""
    retention: dict[str, int] = field(
        default_factory=lambda: {
            "raw_events_days": 90,
            "aggregated_events_days": 365,
            "frame_crops_hours": 72,
        }
    )


@dataclass
class AlertConfig:
    """Alert delivery configuration."""

    channels: list[str] = field(default_factory=lambda: ["push", "webhook"])
    webhook_url: Optional[str] = None
    cooldown: CooldownConfig = field(default_factory=CooldownConfig)


@dataclass
class SecurityConfig:
    """Security and multi-tenancy configuration."""

    tls_enabled: bool = False
    oauth_enabled: bool = False
    tenant_isolation: bool = True


@dataclass
class SystemConfig:
    """Top-level system configuration loaded from config.yaml."""

    deployment_profile: str = "single-machine"  # "single-machine", "multi-machine", "edge-cloud-hybrid"
    mqtt: BrokerConfig = field(default_factory=BrokerConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    storage: StorageConfig = field(default_factory=StorageConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
