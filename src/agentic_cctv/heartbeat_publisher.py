"""Heartbeat Publisher for the Agentic AI CCTV Monitoring Framework.

Publishes ``HeartbeatMessage`` JSON to ``{tenant_id}/{site_id}/{camera_id}/health``
every 30 seconds as an MQTT retained message, including CPU, memory, temperature,
inference latency, and GPU utilisation metrics.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timezone
from typing import Optional

from agentic_cctv.event_encoder import MQTTPublisherProtocol
from agentic_cctv.models import CameraConfig, HeartbeatMessage
from agentic_cctv.mqtt_client import build_topic

logger = logging.getLogger(__name__)

# Try to import psutil; fall back gracefully if unavailable.
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Metric collection helpers
# ---------------------------------------------------------------------------


def _get_cpu_percent() -> float:
    """Return current CPU utilisation percentage, or 0.0 if unavailable."""
    if _HAS_PSUTIL:
        try:
            return psutil.cpu_percent(interval=None)
        except Exception:
            logger.debug("Failed to read CPU percent", exc_info=True)
    return 0.0


def _get_memory_percent() -> float:
    """Return current memory utilisation percentage, or 0.0 if unavailable."""
    if _HAS_PSUTIL:
        try:
            return psutil.virtual_memory().percent
        except Exception:
            logger.debug("Failed to read memory percent", exc_info=True)
    return 0.0


def _get_temperature() -> Optional[float]:
    """Return CPU/SoC temperature in Celsius, or ``None`` if unavailable."""
    if _HAS_PSUTIL:
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Pick the first available sensor group and its first reading.
                for _name, entries in temps.items():
                    if entries:
                        return entries[0].current
        except Exception:
            logger.debug("Failed to read temperature", exc_info=True)
    return None


def _get_gpu_utilization() -> Optional[float]:
    """Return GPU utilisation percentage via ``nvidia-smi``, or ``None``."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi may return multiple GPUs; take the first.
            first_line = result.stdout.strip().splitlines()[0].strip()
            return float(first_line)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return None


def _heartbeat_to_dict(msg: HeartbeatMessage) -> dict:
    """Serialise a ``HeartbeatMessage`` to a JSON-compatible dict."""
    return {
        "camera_id": msg.camera_id,
        "tenant_id": msg.tenant_id,
        "site_id": msg.site_id,
        "timestamp": msg.timestamp.isoformat(),
        "cpu_percent": msg.cpu_percent,
        "memory_percent": msg.memory_percent,
        "temperature_celsius": msg.temperature_celsius,
        "inference_latency_ms": msg.inference_latency_ms,
        "gpu_utilization_percent": msg.gpu_utilization_percent,
    }


# ---------------------------------------------------------------------------
# HeartbeatPublisher
# ---------------------------------------------------------------------------


class HeartbeatPublisher:
    """Periodically publishes ``HeartbeatMessage`` to the MQTT health topic.

    Parameters
    ----------
    camera_config:
        Camera configuration providing ``camera_id``, ``tenant_id``,
        and ``site_id`` for the heartbeat topic.
    mqtt_publisher:
        An MQTT publisher satisfying :class:`MQTTPublisherProtocol`.
    interval_seconds:
        Interval between heartbeat publications (default 30 s).
    inference_latency_provider:
        Optional callable returning the latest inference latency in
        milliseconds.  When ``None``, the heartbeat reports ``0.0``.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        mqtt_publisher: MQTTPublisherProtocol,
        interval_seconds: float = 30.0,
        inference_latency_provider: Optional[object] = None,
    ) -> None:
        self._config = camera_config
        self._mqtt_publisher = mqtt_publisher
        self._interval = interval_seconds
        self._inference_latency_provider = inference_latency_provider
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running = False

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start the background heartbeat publishing loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._loop())
        logger.info(
            "HeartbeatPublisher started for camera '%s' (interval=%.1fs).",
            self._config.camera_id,
            self._interval,
        )

    async def stop(self) -> None:
        """Stop the background heartbeat publishing loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info(
            "HeartbeatPublisher stopped for camera '%s'.",
            self._config.camera_id,
        )

    # -- publishing ---------------------------------------------------------

    async def publish_heartbeat(self) -> None:
        """Collect system metrics and publish a single heartbeat message."""
        inference_latency = 0.0
        if self._inference_latency_provider is not None:
            try:
                inference_latency = float(self._inference_latency_provider())  # type: ignore[operator]
            except Exception:
                logger.debug("Failed to get inference latency", exc_info=True)

        heartbeat = HeartbeatMessage(
            camera_id=self._config.camera_id,
            tenant_id=self._config.tenant_id,
            site_id=self._config.site_id,
            timestamp=datetime.now(timezone.utc),
            cpu_percent=_get_cpu_percent(),
            memory_percent=_get_memory_percent(),
            temperature_celsius=_get_temperature(),
            inference_latency_ms=inference_latency,
            gpu_utilization_percent=_get_gpu_utilization(),
        )

        topic = build_topic(
            self._config.tenant_id,
            self._config.site_id,
            self._config.camera_id,
            "health",
        )
        payload = json.dumps(_heartbeat_to_dict(heartbeat)).encode("utf-8")

        try:
            await self._mqtt_publisher.publish(
                topic, payload, qos=1, retain=True,
            )
            logger.debug(
                "Heartbeat published for camera '%s' to %s",
                self._config.camera_id,
                topic,
            )
        except Exception:
            logger.exception(
                "Failed to publish heartbeat for camera '%s'",
                self._config.camera_id,
            )

    # -- internal -----------------------------------------------------------

    async def _loop(self) -> None:
        """Background loop that publishes heartbeats at the configured interval."""
        while self._running:
            await self.publish_heartbeat()
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break
