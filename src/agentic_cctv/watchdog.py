"""Watchdog health monitoring for the Agentic AI CCTV Monitoring Framework.

Subscribes to ``+/+/+/health`` MQTT topic, tracks last heartbeat per camera,
raises camera-offline alerts when no heartbeat is received within 60 seconds,
and sends camera-restored notifications when heartbeats resume after an offline
period.  Logs downtime duration for each offline→online transition.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from agentic_cctv.alert_system import AlertSystem
from agentic_cctv.models import AlertPayload, DeviceHealth, HeartbeatMessage
from agentic_cctv.mqtt_client import MQTTSubscriber

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEALTH_TOPIC_PATTERN = "+/+/+/health"
OFFLINE_THRESHOLD_SECONDS = 60.0
CHECK_INTERVAL_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Internal tracking state
# ---------------------------------------------------------------------------


class _CameraState:
    """Internal mutable state for a single tracked camera."""

    __slots__ = (
        "camera_id",
        "tenant_id",
        "site_id",
        "status",
        "last_heartbeat",
        "last_heartbeat_time",
        "metrics",
        "went_offline_at",
    )

    def __init__(self, camera_id: str, tenant_id: str, site_id: str) -> None:
        self.camera_id = camera_id
        self.tenant_id = tenant_id
        self.site_id = site_id
        self.status: str = "online"
        self.last_heartbeat: Optional[datetime] = None
        self.last_heartbeat_time: Optional[float] = None  # asyncio loop time
        self.metrics: Optional[HeartbeatMessage] = None
        self.went_offline_at: Optional[float] = None  # asyncio loop time


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


class Watchdog:
    """Monitors camera health via MQTT heartbeat messages.

    Subscribes to the ``+/+/+/health`` wildcard topic and tracks the last
    heartbeat received from each camera.  A background check loop runs every
    ~10 seconds and transitions cameras to ``"offline"`` status when no
    heartbeat has been received within 60 seconds, raising a camera-offline
    alert via the :class:`AlertSystem`.  When a heartbeat resumes for an
    offline camera, the Watchdog transitions it back to ``"online"`` and
    sends a camera-restored notification.

    Parameters
    ----------
    mqtt_subscriber:
        An :class:`MQTTSubscriber` instance used to subscribe to health topics.
    alert_system:
        An :class:`AlertSystem` instance used to send offline/restored alerts.
    offline_threshold:
        Seconds without a heartbeat before a camera is considered offline.
        Defaults to 60.
    check_interval:
        Seconds between periodic health check sweeps.  Defaults to 10.
    """

    def __init__(
        self,
        mqtt_subscriber: MQTTSubscriber,
        alert_system: AlertSystem,
        offline_threshold: float = OFFLINE_THRESHOLD_SECONDS,
        check_interval: float = CHECK_INTERVAL_SECONDS,
    ) -> None:
        self._subscriber = mqtt_subscriber
        self._alert_system = alert_system
        self._offline_threshold = offline_threshold
        self._check_interval = check_interval

        # camera_id → _CameraState
        self._cameras: Dict[str, _CameraState] = {}
        self._check_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to the health topic and start the periodic check loop."""
        if self._running:
            return

        self._running = True
        self._loop = asyncio.get_running_loop()

        await self._subscriber.subscribe(
            HEALTH_TOPIC_PATTERN,
            qos=1,
            callback=self._on_heartbeat,
        )

        self._check_task = asyncio.ensure_future(self._check_loop())
        logger.info(
            "Watchdog started (offline_threshold=%.0fs, check_interval=%.0fs)",
            self._offline_threshold,
            self._check_interval,
        )

    async def stop(self) -> None:
        """Stop the periodic check loop and clean up."""
        self._running = False
        if self._check_task is not None:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None
        logger.info("Watchdog stopped")

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_device_status(self, camera_id: str) -> DeviceHealth:
        """Return the current health status for a single camera.

        Parameters
        ----------
        camera_id:
            The camera identifier to query.

        Returns
        -------
        DeviceHealth
            Current status, last heartbeat time, and latest metrics.
            If the camera has never been seen, returns status ``"offline"``
            with no heartbeat or metrics.
        """
        state = self._cameras.get(camera_id)
        if state is None:
            return DeviceHealth(
                camera_id=camera_id,
                status="offline",
                last_heartbeat=None,
                metrics=None,
            )
        return DeviceHealth(
            camera_id=state.camera_id,
            status=state.status,
            last_heartbeat=state.last_heartbeat,
            metrics=state.metrics,
        )

    def get_all_device_status(self) -> list[DeviceHealth]:
        """Return the current health status for all tracked cameras.

        Returns
        -------
        list[DeviceHealth]
            A list of :class:`DeviceHealth` for every camera that has sent
            at least one heartbeat.
        """
        return [
            DeviceHealth(
                camera_id=s.camera_id,
                status=s.status,
                last_heartbeat=s.last_heartbeat,
                metrics=s.metrics,
            )
            for s in self._cameras.values()
        ]

    # ------------------------------------------------------------------
    # MQTT heartbeat callback
    # ------------------------------------------------------------------

    def _on_heartbeat(self, topic: str, payload: bytes, qos: int) -> None:
        """Handle an incoming heartbeat message from the MQTT broker.

        This callback is invoked by the ``MQTTSubscriber`` on the paho-mqtt
        network thread.  It parses the heartbeat JSON, updates internal state,
        and schedules an async restored-alert if the camera was offline.
        """
        try:
            data = json.loads(payload)
            heartbeat = self._parse_heartbeat(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.error("Failed to parse heartbeat from topic %s: %s", topic, exc)
            return

        camera_id = heartbeat.camera_id
        now_loop = self._loop_time()

        state = self._cameras.get(camera_id)
        if state is None:
            state = _CameraState(
                camera_id=camera_id,
                tenant_id=heartbeat.tenant_id,
                site_id=heartbeat.site_id,
            )
            self._cameras[camera_id] = state
            logger.info("Watchdog: new camera registered — %s", camera_id)

        was_offline = state.status == "offline"

        # Update state
        state.last_heartbeat = heartbeat.timestamp
        state.last_heartbeat_time = now_loop
        state.metrics = heartbeat
        state.tenant_id = heartbeat.tenant_id
        state.site_id = heartbeat.site_id
        state.status = "online"

        if was_offline:
            downtime = self._compute_downtime(state, now_loop)
            logger.info(
                "Watchdog: camera %s restored after %.1fs downtime",
                camera_id,
                downtime,
            )
            state.went_offline_at = None
            # Schedule the restored alert on the event loop
            if self._loop is not None and self._running:
                asyncio.run_coroutine_threadsafe(
                    self._send_restored_alert(state, downtime),
                    self._loop,
                )

        logger.debug("Watchdog: heartbeat received from %s", camera_id)

    # ------------------------------------------------------------------
    # Periodic check loop
    # ------------------------------------------------------------------

    async def _check_loop(self) -> None:
        """Background loop that checks for offline cameras."""
        while self._running:
            try:
                await self._check_cameras()
            except Exception:
                logger.exception("Watchdog check loop error")
            try:
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break

    async def _check_cameras(self) -> None:
        """Sweep all tracked cameras and transition stale ones to offline."""
        now = self._loop_time()
        for state in list(self._cameras.values()):
            if state.last_heartbeat_time is None:
                continue

            elapsed = now - state.last_heartbeat_time
            if elapsed > self._offline_threshold and state.status != "offline":
                state.status = "offline"
                state.went_offline_at = now
                logger.warning(
                    "Watchdog: camera %s is OFFLINE (no heartbeat for %.1fs)",
                    state.camera_id,
                    elapsed,
                )
                await self._send_offline_alert(state)

    # ------------------------------------------------------------------
    # Alert helpers
    # ------------------------------------------------------------------

    async def _send_offline_alert(self, state: _CameraState) -> None:
        """Send a camera-offline alert via the AlertSystem."""
        payload = AlertPayload(
            alert_id=f"alert-offline-{uuid.uuid4().hex[:12]}",
            event_id=f"health-{state.camera_id}",
            camera_id=state.camera_id,
            tenant_id=state.tenant_id,
            site_id=state.site_id,
            timestamp=datetime.now(timezone.utc),
            alert_type="camera-offline",
            description=(
                f"Camera {state.camera_id} has not sent a heartbeat in over "
                f"{self._offline_threshold:.0f} seconds and is considered offline."
            ),
            threat_level="high",
            frame_crop_url=None,
        )
        try:
            result = await self._alert_system.send_alert(payload)
            logger.info(
                "Watchdog: camera-offline alert sent for %s (delivered=%s)",
                state.camera_id,
                result.delivered,
            )
        except Exception:
            logger.exception(
                "Watchdog: failed to send camera-offline alert for %s",
                state.camera_id,
            )

    async def _send_restored_alert(
        self, state: _CameraState, downtime: float
    ) -> None:
        """Send a camera-restored notification via the AlertSystem."""
        payload = AlertPayload(
            alert_id=f"alert-restored-{uuid.uuid4().hex[:12]}",
            event_id=f"health-{state.camera_id}",
            camera_id=state.camera_id,
            tenant_id=state.tenant_id,
            site_id=state.site_id,
            timestamp=datetime.now(timezone.utc),
            alert_type="camera-restored",
            description=(
                f"Camera {state.camera_id} is back online after "
                f"{downtime:.1f} seconds of downtime."
            ),
            threat_level="low",
            frame_crop_url=None,
        )
        try:
            result = await self._alert_system.send_alert(payload)
            logger.info(
                "Watchdog: camera-restored alert sent for %s "
                "(downtime=%.1fs, delivered=%s)",
                state.camera_id,
                downtime,
                result.delivered,
            )
        except Exception:
            logger.exception(
                "Watchdog: failed to send camera-restored alert for %s",
                state.camera_id,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_heartbeat(data: dict) -> HeartbeatMessage:
        """Parse a raw JSON dict into a :class:`HeartbeatMessage`."""
        timestamp_str = data["timestamp"]
        if isinstance(timestamp_str, str):
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1] + "+00:00"
            timestamp = datetime.fromisoformat(timestamp_str)
        else:
            timestamp = timestamp_str

        return HeartbeatMessage(
            camera_id=data["camera_id"],
            tenant_id=data["tenant_id"],
            site_id=data["site_id"],
            timestamp=timestamp,
            cpu_percent=float(data.get("cpu_percent", 0.0)),
            memory_percent=float(data.get("memory_percent", 0.0)),
            temperature_celsius=(
                float(data["temperature_celsius"])
                if data.get("temperature_celsius") is not None
                else None
            ),
            inference_latency_ms=float(data.get("inference_latency_ms", 0.0)),
            gpu_utilization_percent=(
                float(data["gpu_utilization_percent"])
                if data.get("gpu_utilization_percent") is not None
                else None
            ),
        )

    def _loop_time(self) -> float:
        """Return the current event loop time, or fall back to a monotonic-like value."""
        if self._loop is not None:
            try:
                return self._loop.time()
            except Exception:
                pass
        import time
        return time.monotonic()

    @staticmethod
    def _compute_downtime(state: _CameraState, now: float) -> float:
        """Compute downtime duration in seconds."""
        if state.went_offline_at is not None:
            return now - state.went_offline_at
        return 0.0
