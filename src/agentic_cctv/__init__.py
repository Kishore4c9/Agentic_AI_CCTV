"""Agentic AI CCTV Monitoring Framework."""

__version__ = "0.1.0"

from agentic_cctv.environment_templates import (
    EnvironmentTemplate,
    get_template,
    list_templates,
    generate_config_from_template,
    HOME_TEMPLATE,
    FARM_TEMPLATE,
    FOREST_TEMPLATE,
    MALL_TEMPLATE,
    PORT_TEMPLATE,
    GPU_DESKTOP_TEMPLATE,
)
from agentic_cctv.event_summarizer import EventSummarizer
from agentic_cctv.vlm_backends import (
    VLMBackend,
    CosmosBackend,
    GPT4oBackend,
    Claude3Backend,
    Gemini15Backend,
)
from agentic_cctv.vlm_reasoner import VLMReasoner, validate_vlm_response
from agentic_cctv.orchestration_agent import (
    OrchestrationAgent,
    AlertTool,
    LogTool,
    VectorSearchTool,
    MCPContextTool,
    A2ACommTool,
    ToolResult,
)
from agentic_cctv.mcp_server import MCPContextServer, ContextEntry
from agentic_cctv.a2a_comm import A2ACommHub, A2AMessage
from agentic_cctv.frame_crop_store import (
    FrameCropStore,
    FrameCropStoreTool,
    PreSignedURL,
    encrypt_crop,
    decrypt_crop,
    generate_presigned_url,
    validate_presigned_url,
)
from agentic_cctv.phase2_pipeline import ContextFilterSubscriber
from agentic_cctv.prompt_compiler import LLMClient, PromptCompiler, HistoryTestResult
from agentic_cctv.vector_db import VectorDB
from agentic_cctv.alert_system import (
    AlertChannel,
    AlertSystem,
    PushNotificationChannel,
    WebhookChannel,
)
from agentic_cctv.config_manager import ConfigManager, ValidationError
from agentic_cctv.context_filter import ContextFilter
from agentic_cctv.heartbeat_publisher import HeartbeatPublisher
from agentic_cctv.detection_engine import DetectionEngine, apply_detection_gate
from agentic_cctv.event_encoder import EventEncoder, MQTTPublisherProtocol
from agentic_cctv.mqtt_client import MQTTPublisher, MQTTSubscriber, build_topic
from agentic_cctv.rule_store import RuleStore
from agentic_cctv.runtimes import InferenceRuntime, PyTorchRuntime, TensorRTRuntime
from agentic_cctv.store_and_forward import StoreAndForwardQueue
from agentic_cctv.timeseries_db import TimeSeriesDB, TimeSeriesDBSubscriber
from agentic_cctv.tracker import Tracker
from agentic_cctv.video_feeder import HealthMetrics, VideoFeeder
from agentic_cctv.watchdog import Watchdog
from agentic_cctv.health_api import create_health_app, start_health_server
from agentic_cctv.models import (
    ActionResult,
    AlertConfig,
    AlertPayload,
    AlertResult,
    BoundingBox,
    BrokerConfig,
    CameraConfig,
    CompiledRuleSet,
    CompoundCondition,
    CooldownConfig,
    Detection,
    DeviceHealth,
    FilterResult,
    Frame,
    HeartbeatMessage,
    IdentifiedObject,
    PromptScope,
    RawDetection,
    Rule,
    RuleSet,
    RuleSetVersion,
    SceneUnderstanding,
    SecurityConfig,
    StorageConfig,
    StructuredEvent,
    SuppressCondition,
    SystemConfig,
    TimeWindow,
    Track,
    VLMConfig,
    Zone,
)

__all__ = [
    "A2ACommHub",
    "A2ACommTool",
    "A2AMessage",
    "AlertChannel",
    "AlertSystem",
    "AlertTool",
    "Claude3Backend",
    "ConfigManager",
    "ContextFilter",
    "ContextFilterSubscriber",
    "CosmosBackend",
    "DetectionEngine",
    "EnvironmentTemplate",
    "EventEncoder",
    "EventSummarizer",
    "FARM_TEMPLATE",
    "FOREST_TEMPLATE",
    "FrameCropStore",
    "FrameCropStoreTool",
    "PreSignedURL",
    "encrypt_crop",
    "decrypt_crop",
    "generate_config_from_template",
    "generate_presigned_url",
    "get_template",
    "GPU_DESKTOP_TEMPLATE",
    "GPT4oBackend",
    "Gemini15Backend",
    "HealthMetrics",
    "HeartbeatPublisher",
    "HOME_TEMPLATE",
    "InferenceRuntime",
    "LLMClient",
    "list_templates",
    "LogTool",
    "MALL_TEMPLATE",
    "MCPContextTool",
    "MCPContextServer",
    "ContextEntry",
    "MQTTPublisher",
    "MQTTPublisherProtocol",
    "MQTTSubscriber",
    "OrchestrationAgent",
    "PORT_TEMPLATE",
    "PromptCompiler",
    "PushNotificationChannel",
    "PyTorchRuntime",
    "RuleStore",
    "StoreAndForwardQueue",
    "TensorRTRuntime",
    "HistoryTestResult",
    "TimeSeriesDB",
    "TimeSeriesDBSubscriber",
    "ToolResult",
    "Tracker",
    "VLMBackend",
    "VLMReasoner",
    "ValidationError",
    "validate_presigned_url",
    "VectorDB",
    "VectorSearchTool",
    "VideoFeeder",
    "Watchdog",
    "WebhookChannel",
    "create_health_app",
    "start_health_server",
    "apply_detection_gate",
    "build_topic",
    "validate_vlm_response",
    "ActionResult",
    "AlertConfig",
    "AlertPayload",
    "AlertResult",
    "BoundingBox",
    "BrokerConfig",
    "CameraConfig",
    "CompiledRuleSet",
    "CompoundCondition",
    "CooldownConfig",
    "Detection",
    "DeviceHealth",
    "FilterResult",
    "Frame",
    "HeartbeatMessage",
    "IdentifiedObject",
    "PromptScope",
    "RawDetection",
    "Rule",
    "RuleSet",
    "RuleSetVersion",
    "SceneUnderstanding",
    "SecurityConfig",
    "StorageConfig",
    "StructuredEvent",
    "SuppressCondition",
    "SystemConfig",
    "TimeWindow",
    "Track",
    "VLMConfig",
    "Zone",
]
