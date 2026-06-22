"""Main application entry point for the Agentic AI CCTV Monitoring Framework.

Loads configuration, instantiates per-camera pipelines (VideoFeeder →
DetectionEngine → Tracker → EventEncoder → MQTTPublisher → TimeSeriesDB →
AlertSystem), starts Phase 2 components (ContextFilter → VLMReasoner →
OrchestrationAgent → AlertSystem), starts Phase 3 components (Watchdog,
shared MCPContextServer, A2ACommHub, RetentionScheduler, frame crop
retention), starts Phase 4 components (Dashboard, Mobile API, PromptCompiler),
and runs the async processing loop with graceful shutdown on SIGINT/SIGTERM
(Windows compatible).

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import Any, List, Optional

from aiohttp import web

from agentic_cctv.a2a_comm import A2ACommHub
from agentic_cctv.alert_system import (
    AlertSystem,
    PushNotificationChannel,
    WebhookChannel,
)
from agentic_cctv.config_manager import ConfigManager
from agentic_cctv.dashboard import start_dashboard_server
from agentic_cctv.environment_templates import generate_config_from_template
from agentic_cctv.health_api import start_health_server
from agentic_cctv.context_filter import ContextFilter
from agentic_cctv.detection_engine import DetectionEngine
from agentic_cctv.event_encoder import EventEncoder
from agentic_cctv.event_summarizer import EventSummarizer
from agentic_cctv.heartbeat_publisher import HeartbeatPublisher
from agentic_cctv.mcp_server import MCPContextServer
from agentic_cctv.mobile_api import start_mobile_server
from agentic_cctv.models import (
    AlertConfig,
    BrokerConfig,
    CameraConfig,
    CooldownConfig,
    SystemConfig,
    VLMConfig,
)
from agentic_cctv.mqtt_client import MQTTPublisher, MQTTSubscriber
from agentic_cctv.orchestration_agent import (
    AlertTool,
    A2ACommTool,
    LogTool,
    MCPContextTool,
    OrchestrationAgent,
    VectorSearchTool,
)
from agentic_cctv.frame_crop_store import FrameCropStore, FrameCropStoreTool
from agentic_cctv.phase2_pipeline import ContextFilterSubscriber
from agentic_cctv.prompt_compiler import PromptCompiler
from agentic_cctv.retention_scheduler import RetentionScheduler
from agentic_cctv.rule_store import RuleStore
from agentic_cctv.runtimes import PyTorchRuntime
from agentic_cctv.timeseries_db import TimeSeriesDB, TimeSeriesDBSubscriber
from agentic_cctv.tracker import Tracker
from agentic_cctv.video_feeder import VideoFeeder
from agentic_cctv.vlm_reasoner import VLMReasoner
from agentic_cctv.watchdog import Watchdog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Set up root logger with a sensible default format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# VLM Backend → LLMClient Adapter
# ---------------------------------------------------------------------------


class VLMBackendLLMAdapter:
    """Adapts a VLM backend (``analyze(image_b64, event_context)``) to the
    :class:`~agentic_cctv.prompt_compiler.LLMClient` protocol
    (``generate(system_prompt, user_prompt) -> str``).

    The adapter passes the system and user prompts as event context fields
    to the VLM backend's ``analyze`` method and extracts the text response
    from the returned dict.

    Parameters
    ----------
    vlm_backend:
        Any object with an ``async analyze(image_b64, event_context)`` method.
    """

    def __init__(self, vlm_backend: Any) -> None:
        self._backend = vlm_backend

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a text response by delegating to the VLM backend.

        Parameters
        ----------
        system_prompt:
            Instructions for the LLM.
        user_prompt:
            The user's prompt text.

        Returns
        -------
        str
            The text response extracted from the VLM backend result.
        """
        event_context = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        result = await self._backend.analyze("", event_context)

        # Extract text from the VLM response dict
        if isinstance(result, dict):
            # Try common response fields
            for key in ("scene_description", "response", "text", "content"):
                if key in result and result[key]:
                    return str(result[key])
            # Fall back to JSON serialisation of the full result
            import json
            return json.dumps(result)
        return str(result)


# ---------------------------------------------------------------------------
# CameraPipeline
# ---------------------------------------------------------------------------


class CameraPipeline:
    """Encapsulates the per-camera processing pipeline.

    Each pipeline owns a VideoFeeder, DetectionEngine, Tracker,
    EventEncoder, and HeartbeatPublisher for a single camera.

    Parameters
    ----------
    camera_config:
        Per-camera configuration.
    mqtt_publisher:
        Shared MQTT publisher (may be ``None`` if broker is unavailable).
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        mqtt_publisher: Optional[MQTTPublisher],
    ) -> None:
        self.camera_config = camera_config
        self.video_feeder = VideoFeeder(camera_config)

        # Inference runtime
        self.runtime = PyTorchRuntime()

        self.detection_engine = DetectionEngine(camera_config, self.runtime)
        self.tracker = Tracker(
            algorithm=camera_config.tracker_algorithm,
            max_age=30,
        )
        self.event_encoder = EventEncoder(
            camera_config=camera_config,
            mqtt_publisher=mqtt_publisher,
        )
        self.heartbeat_publisher = HeartbeatPublisher(
            camera_config=camera_config,
            mqtt_publisher=mqtt_publisher,  # type: ignore[arg-type]
        )
        self._running = False

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start the video feeder and heartbeat publisher."""
        logger.info(
            "Starting pipeline for camera '%s' (uri=%s).",
            self.camera_config.camera_id,
            self.camera_config.uri,
        )
        try:
            self.runtime.load_model(self.camera_config.model_path)
            logger.info(
                "Model loaded for camera '%s': %s",
                self.camera_config.camera_id,
                self.camera_config.model_path,
            )
        except Exception:
            logger.exception(
                "Failed to load model for camera '%s'. "
                "Detection will not work until the model is available.",
                self.camera_config.camera_id,
            )

        await self.video_feeder.start()
        await self.heartbeat_publisher.start()
        self._running = True

    async def stop(self) -> None:
        """Stop the video feeder and heartbeat publisher."""
        self._running = False
        await self.heartbeat_publisher.stop()
        await self.video_feeder.stop()
        logger.info(
            "Pipeline stopped for camera '%s'.",
            self.camera_config.camera_id,
        )

    # -- processing ---------------------------------------------------------

    async def process_frame(self) -> None:
        """Read one frame, run detection → tracking → encoding pipeline.

        If no frame is available (camera not ready or between frames),
        this method returns immediately.
        """
        frame = self.video_feeder.get_frame()
        if frame is None:
            return

        try:
            detections = self.detection_engine.detect(frame)
            tracks = self.tracker.update(detections, frame)

            for track in tracks:
                if track.is_new:
                    await self.event_encoder.encode_and_publish(track, frame)
        except Exception:
            logger.exception(
                "Error processing frame for camera '%s'.",
                self.camera_config.camera_id,
            )

    @property
    def is_running(self) -> bool:
        """Return ``True`` if the pipeline is currently running."""
        return self._running


# ---------------------------------------------------------------------------
# Alert system factory
# ---------------------------------------------------------------------------


def _build_alert_system(alert_config: AlertConfig) -> AlertSystem:
    """Create an AlertSystem from the alert configuration."""
    channels: list = []

    configured = alert_config.channels if alert_config.channels else []
    for ch_name in configured:
        lower = ch_name.lower()
        if lower == "push":
            channels.append(PushNotificationChannel())
        elif lower == "webhook":
            url = alert_config.webhook_url or ""
            channels.append(WebhookChannel(webhook_url=url))
        else:
            logger.warning("Unknown alert channel '%s', skipping.", ch_name)

    cooldown = alert_config.cooldown if alert_config.cooldown else CooldownConfig()
    return AlertSystem(channels=channels, cooldown_config=cooldown)


# ---------------------------------------------------------------------------
# VLM backend factory
# ---------------------------------------------------------------------------


def _build_vlm_backend(vlm_config: VLMConfig) -> object:
    """Create a VLM backend from the VLM configuration.

    Returns the appropriate backend instance based on ``vlm_config.backend``.
    Falls back to CosmosBackend if the backend name is unrecognised.
    """
    from agentic_cctv.vlm_backends import (
        Claude3Backend,
        CosmosBackend,
        Gemini15Backend,
        GPT4oBackend,
    )

    backend_name = vlm_config.backend.lower()
    if backend_name == "cosmos":
        return CosmosBackend(vlm_config)
    elif backend_name == "gpt4o":
        return GPT4oBackend()
    elif backend_name == "claude3":
        return Claude3Backend()
    elif backend_name == "gemini15":
        return Gemini15Backend()
    else:
        logger.warning(
            "Unknown VLM backend '%s', defaulting to CosmosBackend.",
            vlm_config.backend,
        )
        return CosmosBackend(vlm_config)


def _build_vector_db(storage_path: str) -> Optional[object]:
    """Attempt to create a VectorDB instance.  Returns ``None`` if chromadb
    is not installed."""
    try:
        from agentic_cctv.vector_db import VectorDB

        return VectorDB(storage_path)
    except ImportError:
        logger.warning(
            "chromadb not installed — VectorDB disabled. "
            "Install with: pip install chromadb==1.0.7"
        )
        return None
    except Exception:
        logger.exception("Failed to initialise VectorDB at '%s'.", storage_path)
        return None


# ---------------------------------------------------------------------------
# MQTT connection helpers
# ---------------------------------------------------------------------------


async def _connect_mqtt_publisher(
    broker_config: BrokerConfig,
) -> Optional[MQTTPublisher]:
    """Attempt to connect an MQTTPublisher. Returns ``None`` on failure."""
    tls_status = "TLS enabled" if broker_config.use_tls else "TLS disabled"
    logger.info(
        "Connecting MQTT publisher to %s:%s (%s).",
        broker_config.host,
        broker_config.port,
        tls_status,
    )
    publisher = MQTTPublisher(broker_config)
    try:
        await publisher.connect()
        logger.info(
            "MQTT publisher connected to %s:%s (%s).",
            broker_config.host,
            broker_config.port,
            tls_status,
        )
        return publisher
    except (ConnectionError, OSError) as exc:
        logger.warning(
            "MQTT broker not available at %s:%s — continuing without MQTT. (%s)",
            broker_config.host,
            broker_config.port,
            exc,
        )
        return None


async def _connect_mqtt_subscriber(
    broker_config: BrokerConfig,
) -> Optional[MQTTSubscriber]:
    """Attempt to connect an MQTTSubscriber. Returns ``None`` on failure."""
    tls_status = "TLS enabled" if broker_config.use_tls else "TLS disabled"
    logger.info(
        "Connecting MQTT subscriber to %s:%s (%s).",
        broker_config.host,
        broker_config.port,
        tls_status,
    )
    subscriber = MQTTSubscriber(broker_config)
    try:
        await subscriber.connect()
        logger.info(
            "MQTT subscriber connected to %s:%s (%s).",
            broker_config.host,
            broker_config.port,
            tls_status,
        )
        return subscriber
    except (ConnectionError, OSError) as exc:
        logger.warning(
            "MQTT subscriber could not connect to %s:%s — "
            "TimeSeriesDB event persistence via MQTT disabled. (%s)",
            broker_config.host,
            broker_config.port,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Frame crop retention background task
# ---------------------------------------------------------------------------


async def _frame_crop_purge_loop(
    frame_crop_store: FrameCropStore,
    interval_seconds: int = 3600,
) -> None:
    """Periodically purge expired frame crops.

    Runs as a background asyncio task, calling
    :meth:`FrameCropStore.purge_expired` at the configured interval.

    Parameters
    ----------
    frame_crop_store:
        The :class:`FrameCropStore` instance to purge.
    interval_seconds:
        How often to run the purge (default 3600 = 1 hour).
    """
    try:
        while True:
            try:
                purged = frame_crop_store.purge_expired()
                if purged > 0:
                    logger.info(
                        "Frame crop purge: removed %d expired crops.", purged
                    )
            except Exception:
                logger.exception("Error during frame crop purge.")
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


async def run_application(config_path: str = "config.yaml") -> None:
    """Main async entry point.

    1. Load and validate configuration.
    2. Create shared components (MQTT, TimeSeriesDB, AlertSystem).
    3. Create per-camera pipelines.
    4. Run the processing loop until shutdown is requested.
    5. Clean up on exit.
    """
    # -- Load config --------------------------------------------------------
    _configure_logging()
    logger.info("Starting Agentic AI CCTV Monitoring Framework…")

    config_manager = ConfigManager(config_path)
    try:
        config: SystemConfig = config_manager.load()
    except Exception:
        logger.exception("Failed to load configuration from '%s'.", config_path)
        return

    # -- Environment template support (Phase 4) ----------------------------
    # If the raw config specifies an environment_template field, generate a
    # config from that template and apply any overrides from the config file.
    try:
        import yaml as _yaml

        with open(config_path, "r", encoding="utf-8") as _fh:
            _raw_config = _yaml.safe_load(_fh) or {}
        _template_name = _raw_config.get("environment_template")
        if _template_name:
            logger.info(
                "Environment template '%s' specified — generating config from template.",
                _template_name,
            )
            # Collect overrides from the raw config (everything except the template key)
            _overrides = {k: v for k, v in _raw_config.items() if k != "environment_template"}
            try:
                config = generate_config_from_template(_template_name, **_overrides)
                logger.info(
                    "Config generated from environment template '%s'.",
                    _template_name,
                )
            except Exception:
                logger.exception(
                    "Failed to generate config from template '%s'. "
                    "Falling back to standard config.",
                    _template_name,
                )
    except Exception:
        # If we can't re-read the config file for template detection, just
        # continue with the already-loaded config.
        pass

    errors = config_manager.validate()
    if errors:
        for err in errors:
            logger.error("Config error [%s]: %s", err.field_path, err.message)
        logger.warning(
            "Configuration has %d validation error(s). "
            "The system will attempt to start, but some features may not work.",
            len(errors),
        )

    if not config.cameras:
        logger.error("No cameras configured. Nothing to do.")
        return

    # -- Shared components --------------------------------------------------
    mqtt_publisher = await _connect_mqtt_publisher(config.mqtt)
    mqtt_subscriber = await _connect_mqtt_subscriber(config.mqtt)

    # TimeSeriesDB
    ts_db_path = config.storage.timeseries_path
    ts_db_dir = os.path.dirname(ts_db_path)
    if ts_db_dir:
        os.makedirs(ts_db_dir, exist_ok=True)
    timeseries_db = TimeSeriesDB(ts_db_path)
    logger.info("TimeSeriesDB opened at '%s'.", ts_db_path)

    # Subscribe TimeSeriesDB to events topic via MQTT
    if mqtt_subscriber is not None:
        ts_subscriber = TimeSeriesDBSubscriber(timeseries_db)
        try:
            await mqtt_subscriber.subscribe(
                topic="+/+/+/events",
                qos=1,
                callback=ts_subscriber,
            )
            logger.info("TimeSeriesDB subscribed to +/+/+/events.")
        except Exception:
            logger.exception("Failed to subscribe TimeSeriesDB to events topic.")

    # AlertSystem
    alert_system = _build_alert_system(config.alerts)
    logger.info("AlertSystem initialised with channels: %s.", config.alerts.channels)

    # -- Phase 2 components -------------------------------------------------

    # RuleStore (shares the same SQLite path as TimeSeriesDB for simplicity,
    # but uses its own table)
    rule_store_path = config.storage.timeseries_path
    rule_store = RuleStore(rule_store_path)
    logger.info("RuleStore opened at '%s'.", rule_store_path)

    # ContextFilter
    context_filter = ContextFilter(
        rule_store=rule_store,
        timeseries_db=timeseries_db,
    )
    logger.info("ContextFilter initialised.")

    # VectorDB (optional — requires chromadb)
    vector_db = _build_vector_db(config.storage.vector_path)

    # VLM Backend + Reasoner
    vlm_backend = _build_vlm_backend(config.vlm)
    vlm_reasoner = VLMReasoner(
        backend=vlm_backend,
        vector_db=vector_db,
        timeseries_db=timeseries_db,
    )
    logger.info("VLMReasoner initialised with backend '%s'.", config.vlm.backend)

    # OrchestrationAgent with tools
    # -- Phase 3: Shared MCP server and A2A hub ----------------------------
    mcp_server = MCPContextServer()
    logger.info("MCPContextServer initialised (in-process mode).")

    a2a_hub = A2ACommHub()
    logger.info("A2ACommHub initialised (in-process mode).")

    # Register each camera as an agent in the A2A hub
    for cam_config in config.cameras:
        a2a_hub.register_agent(cam_config.camera_id)
    logger.info(
        "Registered %d camera agent(s) in A2ACommHub.", len(config.cameras)
    )

    tools = [
        AlertTool(alert_system),
        LogTool(timeseries_db),
        MCPContextTool(server=mcp_server),
        A2ACommTool(hub=a2a_hub, agent_id="orchestrator"),
    ]
    if vector_db is not None:
        tools.insert(2, VectorSearchTool(vector_db))

    # Frame Crop Store (optional — requires valid encryption key)
    frame_crop_store: Optional[FrameCropStore] = None
    enc_key = config.storage.frame_crop_encryption_key
    if enc_key:
        try:
            retention_hours = config.storage.retention.get("frame_crops_hours", 72)
            frame_crop_store = FrameCropStore(
                crop_path=config.storage.frame_crop_path,
                encryption_key_hex=enc_key,
                retention_hours=retention_hours,
            )
            tools.append(FrameCropStoreTool(frame_crop_store))
            logger.info(
                "FrameCropStore initialised at '%s' (retention=%dh).",
                config.storage.frame_crop_path,
                retention_hours,
            )
        except ValueError as exc:
            logger.error(
                "FrameCropStore disabled — invalid encryption key: %s", exc
            )
    else:
        logger.warning(
            "FrameCropStore disabled — no encryption key configured. "
            "Set 'storage.frame_crop_encryption_key' in config.yaml."
        )

    orchestration_agent = OrchestrationAgent(tools=tools)
    logger.info("OrchestrationAgent initialised with %d tools.", len(tools))

    # Subscribe ContextFilterSubscriber to events topic via MQTT
    phase2_subscriber: Optional[MQTTSubscriber] = None
    if mqtt_subscriber is not None:
        # Get the running event loop for async bridging in MQTT callbacks
        loop = asyncio.get_running_loop()
        cf_subscriber = ContextFilterSubscriber(
            context_filter=context_filter,
            vlm_reasoner=vlm_reasoner,
            orchestration_agent=orchestration_agent,
            alert_system=alert_system,
            loop=loop,
        )
        try:
            await mqtt_subscriber.subscribe(
                topic="+/+/+/events",
                qos=1,
                callback=cf_subscriber,
            )
            logger.info(
                "Phase 2 pipeline subscribed to +/+/+/events "
                "(ContextFilter → VLMReasoner → OrchestrationAgent → AlertSystem)."
            )
        except Exception:
            logger.exception(
                "Failed to subscribe Phase 2 pipeline to events topic."
            )

    # -- Per-camera pipelines -----------------------------------------------
    pipelines: List[CameraPipeline] = []
    for cam_config in config.cameras:
        pipeline = CameraPipeline(
            camera_config=cam_config,
            mqtt_publisher=mqtt_publisher,
        )
        pipelines.append(pipeline)

    # Start all pipelines
    for pipeline in pipelines:
        await pipeline.start()

    logger.info(
        "All %d camera pipeline(s) started. Entering main loop.",
        len(pipelines),
    )

    # -- Phase 3: Watchdog, RetentionScheduler, Frame Crop Purge -----------
    watchdog: Optional[Watchdog] = None
    if mqtt_subscriber is not None:
        watchdog = Watchdog(
            mqtt_subscriber=mqtt_subscriber,
            alert_system=alert_system,
        )
        await watchdog.start()
        logger.info("Watchdog started (health monitoring active).")
    else:
        logger.warning(
            "Watchdog not started — MQTT subscriber unavailable."
        )

    # -- Health API server (Requirement 9.3, 17.1) -------------------------
    health_runner: Optional[web.AppRunner] = None
    if watchdog is not None:
        try:
            health_runner, _health_site = await start_health_server(
                watchdog=watchdog,
                host="0.0.0.0",
                port=8080,
            )
            logger.info("Health API dashboard available at http://0.0.0.0:8080/api/health/devices")
        except Exception:
            logger.exception("Failed to start Health API server.")
    else:
        logger.warning(
            "Health API server not started — Watchdog unavailable."
        )

    # RetentionScheduler for TimeSeriesDB + VectorDB
    retention_config = config.storage.retention
    retention_scheduler = RetentionScheduler(
        timeseries_db=timeseries_db,
        vector_db=vector_db,
        raw_events_days=retention_config.get("raw_events_days", 90),
        aggregated_events_days=retention_config.get("aggregated_events_days", 365),
    )
    await retention_scheduler.start()
    logger.info(
        "RetentionScheduler started (raw=%dd, aggregated=%dd).",
        retention_config.get("raw_events_days", 90),
        retention_config.get("aggregated_events_days", 365),
    )

    # Frame crop retention purge loop
    frame_crop_purge_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
    if frame_crop_store is not None:
        frame_crop_purge_task = asyncio.create_task(
            _frame_crop_purge_loop(frame_crop_store, interval_seconds=3600)
        )
        logger.info(
            "Frame crop purge loop started (interval=3600s, retention=%dh).",
            frame_crop_store.retention_hours,
        )

    # -- Event Summarizer (hourly + daily summaries) -----------------------
    # Determine default tenant_id and site_id from the first camera config
    default_tenant = "default"
    default_site = "default"
    if config.cameras:
        default_tenant = config.cameras[0].tenant_id
        default_site = config.cameras[0].site_id

    event_summarizer = EventSummarizer(
        timeseries_db=timeseries_db,
        alert_system=alert_system,
        llm_backend=vlm_backend,
        tenant_id=default_tenant,
        site_id=default_site,
    )
    await event_summarizer.start_scheduler()
    logger.info(
        "EventSummarizer started (hourly + daily summaries, tenant=%s).",
        default_tenant,
    )

    # -- Phase 4: PromptCompiler, Dashboard, Mobile API --------------------

    # PromptCompiler — adapts VLM backend to LLMClient protocol
    llm_adapter = VLMBackendLLMAdapter(vlm_backend)
    prompt_compiler = PromptCompiler(
        llm_client=llm_adapter,
        rule_store=rule_store,
        context_filter=context_filter,
        timeseries_db=timeseries_db,
    )
    logger.info("PromptCompiler initialised (VLM backend adapted to LLMClient).")

    # Dashboard server (port 8081) — requires Watchdog
    dashboard_runner: Optional[web.AppRunner] = None
    if watchdog is not None:
        try:
            dashboard_runner, _dashboard_site = await start_dashboard_server(
                watchdog=watchdog,
                timeseries_db=timeseries_db,
                alert_system=alert_system,
                rule_store=rule_store,
                prompt_compiler=prompt_compiler,
                context_filter=context_filter,
                host="0.0.0.0",
                port=8081,
            )
            logger.info(
                "Dashboard server available at http://0.0.0.0:8081/dashboard"
            )
        except Exception:
            logger.exception("Failed to start Dashboard server.")
    else:
        logger.warning(
            "Dashboard server not started — Watchdog unavailable."
        )

    # Mobile API server (port 8082) — requires Watchdog
    mobile_runner: Optional[web.AppRunner] = None
    if watchdog is not None:
        try:
            mobile_runner, _mobile_site = await start_mobile_server(
                watchdog=watchdog,
                timeseries_db=timeseries_db,
                alert_system=alert_system,
                rule_store=rule_store,
                prompt_compiler=prompt_compiler,
                context_filter=context_filter,
                host="0.0.0.0",
                port=8082,
            )
            logger.info(
                "Mobile API server available at http://0.0.0.0:8082/api/mobile"
            )
        except Exception:
            logger.exception("Failed to start Mobile API server.")
    else:
        logger.warning(
            "Mobile API server not started — Watchdog unavailable."
        )

    # -- Shutdown signal handling -------------------------------------------
    shutdown_event = asyncio.Event()

    def _request_shutdown(*args: object) -> None:
        """Signal handler that sets the shutdown event."""
        logger.info("Shutdown signal received. Stopping…")
        shutdown_event.set()

    # Register signal handlers (Windows-compatible)
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _request_shutdown)
        loop.add_signal_handler(signal.SIGTERM, _request_shutdown)
    except NotImplementedError:
        # Windows does not support add_signal_handler on ProactorEventLoop.
        # Fall back to signal.signal for SIGINT. SIGTERM is not reliably
        # available on Windows.
        signal.signal(signal.SIGINT, _request_shutdown)
        try:
            signal.signal(signal.SIGTERM, _request_shutdown)
        except (OSError, ValueError):
            pass  # SIGTERM not available on this platform

    # -- Main processing loop -----------------------------------------------
    try:
        while not shutdown_event.is_set():
            for pipeline in pipelines:
                if pipeline.is_running:
                    await pipeline.process_frame()
            # Yield control briefly to avoid busy-waiting
            await asyncio.sleep(0.01)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Interrupted. Shutting down…")

    # -- Graceful shutdown --------------------------------------------------
    logger.info("Shutting down…")

    # Stop Health API server
    if health_runner is not None:
        await health_runner.cleanup()
        logger.info("Health API server stopped.")

    # Stop Dashboard server
    if dashboard_runner is not None:
        await dashboard_runner.cleanup()
        logger.info("Dashboard server stopped.")

    # Stop Mobile API server
    if mobile_runner is not None:
        await mobile_runner.cleanup()
        logger.info("Mobile API server stopped.")

    # Stop event summarizer
    await event_summarizer.stop_scheduler()

    # Stop retention schedulers first
    if frame_crop_purge_task is not None:
        frame_crop_purge_task.cancel()
        try:
            await frame_crop_purge_task
        except asyncio.CancelledError:
            pass
        logger.info("Frame crop purge loop stopped.")

    await retention_scheduler.stop()

    # Stop Watchdog
    if watchdog is not None:
        await watchdog.stop()

    # Stop camera pipelines
    logger.info("Shutting down pipelines…")
    for pipeline in pipelines:
        await pipeline.stop()

    # Disconnect MQTT
    if mqtt_publisher is not None:
        await mqtt_publisher.disconnect()
    if mqtt_subscriber is not None:
        await mqtt_subscriber.disconnect()

    # Close databases and stores
    rule_store.close()
    timeseries_db.close()
    if frame_crop_store is not None:
        frame_crop_store.close()
    logger.info("Agentic AI CCTV Monitoring Framework stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Synchronous entry point that delegates to :func:`run_application`."""
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    asyncio.run(run_application(config_path))


if __name__ == "__main__":
    main()
