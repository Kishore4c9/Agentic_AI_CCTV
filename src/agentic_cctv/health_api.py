"""REST API endpoint for real-time device health dashboard.

Exposes ``Watchdog.get_device_status`` and ``Watchdog.get_all_device_status``
as JSON endpoints via an aiohttp web server.

Endpoints
---------
``GET /api/health/devices``
    Returns current status, last heartbeat, and device metrics for **all**
    tracked cameras.

``GET /api/health/devices/{camera_id}``
    Returns current status, last heartbeat, and device metrics for a
    **single** camera identified by ``camera_id``.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from aiohttp import web

from agentic_cctv.models import DeviceHealth, HeartbeatMessage
from agentic_cctv.watchdog import Watchdog

logger = logging.getLogger(__name__)

# Typed application key for the Watchdog instance
_watchdog_key = web.AppKey("watchdog", Watchdog)


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _serialize_datetime(dt: Optional[datetime]) -> Optional[str]:
    """Convert a datetime to ISO-8601 string, or ``None``."""
    if dt is None:
        return None
    return dt.isoformat()


def _serialize_heartbeat(hb: Optional[HeartbeatMessage]) -> Optional[dict[str, Any]]:
    """Convert a :class:`HeartbeatMessage` to a JSON-safe dict, or ``None``."""
    if hb is None:
        return None
    return {
        "camera_id": hb.camera_id,
        "tenant_id": hb.tenant_id,
        "site_id": hb.site_id,
        "timestamp": _serialize_datetime(hb.timestamp),
        "cpu_percent": hb.cpu_percent,
        "memory_percent": hb.memory_percent,
        "temperature_celsius": hb.temperature_celsius,
        "inference_latency_ms": hb.inference_latency_ms,
        "gpu_utilization_percent": hb.gpu_utilization_percent,
    }


def _serialize_device_health(dh: DeviceHealth) -> dict[str, Any]:
    """Convert a :class:`DeviceHealth` to a JSON-safe dict."""
    return {
        "camera_id": dh.camera_id,
        "status": dh.status,
        "last_heartbeat": _serialize_datetime(dh.last_heartbeat),
        "metrics": _serialize_heartbeat(dh.metrics),
    }


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------


async def get_all_devices(request: web.Request) -> web.Response:
    """Return health status for all tracked cameras.

    ``GET /api/health/devices``

    Response JSON::

        {
            "devices": [
                {
                    "camera_id": "cam-01",
                    "status": "online",
                    "last_heartbeat": "2025-01-15T14:30:00+00:00",
                    "metrics": { ... }
                },
                ...
            ]
        }
    """
    watchdog: Watchdog = request.app[_watchdog_key]
    devices = watchdog.get_all_device_status()
    payload = {
        "devices": [_serialize_device_health(d) for d in devices],
    }
    return web.json_response(payload)


async def get_device(request: web.Request) -> web.Response:
    """Return health status for a single camera.

    ``GET /api/health/devices/{camera_id}``

    Response JSON::

        {
            "camera_id": "cam-01",
            "status": "online",
            "last_heartbeat": "2025-01-15T14:30:00+00:00",
            "metrics": { ... }
        }
    """
    watchdog: Watchdog = request.app[_watchdog_key]
    camera_id = request.match_info["camera_id"]
    device = watchdog.get_device_status(camera_id)
    return web.json_response(_serialize_device_health(device))


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_health_app(watchdog: Watchdog) -> web.Application:
    """Create an aiohttp :class:`web.Application` with health API routes.

    Parameters
    ----------
    watchdog:
        The :class:`Watchdog` instance whose device status will be exposed.

    Returns
    -------
    web.Application
        An aiohttp application ready to be run or composed into a larger app.
    """
    app = web.Application()
    app[_watchdog_key] = watchdog
    app.router.add_get("/api/health/devices", get_all_devices)
    app.router.add_get("/api/health/devices/{camera_id}", get_device)
    return app


async def start_health_server(
    watchdog: Watchdog,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> tuple[web.AppRunner, web.TCPSite]:
    """Start the health API server as a background aiohttp site.

    Parameters
    ----------
    watchdog:
        The :class:`Watchdog` instance to expose.
    host:
        Bind address.  Defaults to ``"0.0.0.0"``.
    port:
        Bind port.  Defaults to ``8080``.

    Returns
    -------
    tuple[web.AppRunner, web.TCPSite]
        The runner and site, which the caller should clean up on shutdown
        via ``await runner.cleanup()``.
    """
    app = create_health_app(watchdog)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Health API server started on http://%s:%d", host, port)
    return runner, site
