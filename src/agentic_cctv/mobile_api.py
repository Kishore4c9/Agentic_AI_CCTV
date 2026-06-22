"""Mobile-compatible REST API for the Agentic AI CCTV Monitoring Framework.

Provides mobile-optimised endpoints for alert management, prompt configuration,
push notification device registration, and a compact dashboard summary.

Endpoints
---------
**Alert Management:**

``GET /api/mobile/alerts``
    List alerts with pagination (query params: ``camera_id``, ``tenant_id``,
    ``status``, ``limit``, ``offset``).

``GET /api/mobile/alerts/{alert_id}``
    Get single alert details.

``POST /api/mobile/alerts/{alert_id}/acknowledge``
    Acknowledge an alert.

``POST /api/mobile/alerts/{alert_id}/dismiss``
    Dismiss an alert.

``POST /api/mobile/alerts/{alert_id}/escalate``
    Escalate an alert.

**Prompt Configuration:**

``POST /api/mobile/prompt/compile``
    Compile a natural language prompt into rules.

``POST /api/mobile/prompt/activate``
    Activate compiled rules.

``GET /api/mobile/rules/{camera_id}``
    Get active ruleset for a camera.

**Push Notification Registration:**

``POST /api/mobile/push/register``
    Register a device for push notifications.

``DELETE /api/mobile/push/unregister``
    Unregister a device.

``GET /api/mobile/push/devices``
    List registered devices (query param: ``tenant_id``).

**Dashboard Summary (mobile-optimised):**

``GET /api/mobile/summary``
    Compact overview (camera counts, alert counts, recent critical alerts).

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from aiohttp import web

from agentic_cctv.dashboard import (
    _get_alert_by_id,
    _migrate_alert_status,
    _serialize_datetime,
    _update_alert_status,
)
from agentic_cctv.models import AlertPayload, PromptScope
from agentic_cctv.rule_store import RuleStore
from agentic_cctv.timeseries_db import TimeSeriesDB
from agentic_cctv.watchdog import Watchdog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typed application keys
# ---------------------------------------------------------------------------

_watchdog_key = web.AppKey("mobile_watchdog", Watchdog)
_tsdb_key = web.AppKey("mobile_tsdb", TimeSeriesDB)
_alert_system_key = web.AppKey("mobile_alert_system", object)
_rule_store_key = web.AppKey("mobile_rule_store", RuleStore)
_prompt_compiler_key = web.AppKey("mobile_prompt_compiler", object)
_context_filter_key = web.AppKey("mobile_context_filter", object)
_push_devices_key = web.AppKey("mobile_push_devices", dict)


# ---------------------------------------------------------------------------
# Device registration dataclass
# ---------------------------------------------------------------------------


def _make_device_entry(
    device_token: str, platform: str, tenant_id: str,
) -> dict[str, Any]:
    """Create a device registration dict."""
    return {
        "device_token": device_token,
        "platform": platform,
        "tenant_id": tenant_id,
        "registered_at": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------


@web.middleware
async def cors_middleware(
    request: web.Request, handler: Any,
) -> web.Response:
    """Add CORS headers to every response for mobile app access.

    Handles preflight ``OPTIONS`` requests and adds permissive CORS headers
    to all responses.
    """
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            response = exc

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, DELETE, OPTIONS"
    )
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization"
    )
    response.headers["Access-Control-Max-Age"] = "3600"
    return response


# ---------------------------------------------------------------------------
# Request handlers — Alert Management
# ---------------------------------------------------------------------------


async def get_alerts(request: web.Request) -> web.Response:
    """Return alerts with pagination and optional filters.

    ``GET /api/mobile/alerts``

    Query params: ``camera_id``, ``tenant_id``, ``status``, ``limit``
    (default 20), ``offset`` (default 0).
    """
    tsdb: TimeSeriesDB = request.app[_tsdb_key]
    camera_id = request.query.get("camera_id")
    tenant_id = request.query.get("tenant_id")
    status_filter = request.query.get("status")

    try:
        limit = int(request.query.get("limit", "20"))
    except (ValueError, TypeError):
        limit = 20
    limit = max(1, min(limit, 100))

    try:
        offset = int(request.query.get("offset", "0"))
    except (ValueError, TypeError):
        offset = 0
    offset = max(0, offset)

    # Fetch more than needed so we can apply offset and status filter
    fetch_limit = limit + offset + 200  # buffer for status filtering
    alerts = tsdb.get_alerts(
        camera_id=camera_id, tenant_id=tenant_id, limit=fetch_limit,
    )

    # Apply status filter if provided
    if status_filter:
        alerts = [
            a for a in alerts if a.get("status", "active") == status_filter
        ]

    total = len(alerts)
    paginated = alerts[offset : offset + limit]

    return web.json_response({
        "alerts": paginated,
        "total": total,
        "limit": limit,
        "offset": offset,
    })


async def get_alert(request: web.Request) -> web.Response:
    """Return a single alert by ID.

    ``GET /api/mobile/alerts/{alert_id}``
    """
    tsdb: TimeSeriesDB = request.app[_tsdb_key]
    alert_id = request.match_info["alert_id"]
    alert = _get_alert_by_id(tsdb, alert_id)
    if alert is None:
        return web.json_response({"error": "Alert not found"}, status=404)
    return web.json_response(alert)


async def acknowledge_alert(request: web.Request) -> web.Response:
    """Acknowledge an alert.

    ``POST /api/mobile/alerts/{alert_id}/acknowledge``
    """
    tsdb: TimeSeriesDB = request.app[_tsdb_key]
    alert_id = request.match_info["alert_id"]

    alert = _get_alert_by_id(tsdb, alert_id)
    if alert is None:
        return web.json_response({"error": "Alert not found"}, status=404)

    _update_alert_status(tsdb, alert_id, "acknowledged")
    logger.info("Mobile: Alert %s acknowledged", alert_id)
    return web.json_response({
        "alert_id": alert_id,
        "status": "acknowledged",
    })


async def dismiss_alert(request: web.Request) -> web.Response:
    """Dismiss an alert.

    ``POST /api/mobile/alerts/{alert_id}/dismiss``
    """
    tsdb: TimeSeriesDB = request.app[_tsdb_key]
    alert_id = request.match_info["alert_id"]

    alert = _get_alert_by_id(tsdb, alert_id)
    if alert is None:
        return web.json_response({"error": "Alert not found"}, status=404)

    _update_alert_status(tsdb, alert_id, "dismissed")
    logger.info("Mobile: Alert %s dismissed", alert_id)
    return web.json_response({
        "alert_id": alert_id,
        "status": "dismissed",
    })


async def escalate_alert(request: web.Request) -> web.Response:
    """Escalate an alert (re-send with higher threat level).

    ``POST /api/mobile/alerts/{alert_id}/escalate``
    """
    tsdb: TimeSeriesDB = request.app[_tsdb_key]
    alert_id = request.match_info["alert_id"]

    alert = _get_alert_by_id(tsdb, alert_id)
    if alert is None:
        return web.json_response({"error": "Alert not found"}, status=404)

    _update_alert_status(tsdb, alert_id, "escalated")

    # Attempt to re-send via alert system with escalated threat level
    alert_system = request.app.get(_alert_system_key)
    if alert_system is not None:
        try:
            escalated_payload = AlertPayload(
                alert_id=f"{alert_id}-escalated",
                event_id=alert.get("event_id", ""),
                camera_id=alert.get("camera_id", ""),
                tenant_id=alert.get("tenant_id", ""),
                site_id="",
                timestamp=datetime.utcnow(),
                alert_type=alert.get("alert_type", "escalated"),
                description=f"ESCALATED: {alert.get('description', '')}",
                threat_level="critical",
                frame_crop_url=None,
            )
            await alert_system.send_alert(escalated_payload)
        except Exception:
            logger.exception(
                "Failed to send escalated alert for %s", alert_id,
            )

    logger.info("Mobile: Alert %s escalated", alert_id)
    return web.json_response({
        "alert_id": alert_id,
        "status": "escalated",
    })


# ---------------------------------------------------------------------------
# Request handlers — Prompt Configuration
# ---------------------------------------------------------------------------


async def compile_prompt(request: web.Request) -> web.Response:
    """Compile a natural language prompt into a ruleset.

    ``POST /api/mobile/prompt/compile``

    Request JSON::

        {
            "prompt": "...",
            "scope_type": "camera",
            "target_ids": ["cam-01"]
        }
    """
    prompt_compiler = request.app.get(_prompt_compiler_key)
    if prompt_compiler is None:
        return web.json_response(
            {"error": "Prompt compiler not configured"}, status=501,
        )

    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    prompt_text = body.get("prompt", "")
    scope_type = body.get("scope_type", "camera")
    target_ids = body.get("target_ids", [])

    if not prompt_text:
        return web.json_response(
            {"error": "prompt field is required"}, status=400,
        )
    if not target_ids:
        return web.json_response(
            {"error": "target_ids field is required"}, status=400,
        )

    scope = PromptScope(scope_type=scope_type, target_ids=target_ids)

    try:
        compiled = await prompt_compiler.compile(prompt_text, scope)
    except Exception as exc:
        logger.error("Mobile: Prompt compilation failed: %s", exc)
        return web.json_response(
            {"error": f"Compilation failed: {exc}"}, status=500,
        )

    return web.json_response({
        "status": "compiled",
        "explanation": compiled.explanation,
        "confidence": compiled.confidence,
        "rules_count": len(compiled.ruleset.rules),
        "ruleset": {
            "version_id": compiled.ruleset.version_id,
            "camera_id": compiled.ruleset.camera_id,
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "object_type": r.object_type,
                    "min_confidence": r.min_confidence,
                }
                for r in compiled.ruleset.rules
            ],
        },
    })


async def activate_prompt(request: web.Request) -> web.Response:
    """Activate a compiled ruleset.

    ``POST /api/mobile/prompt/activate``

    Request JSON::

        {
            "prompt": "...",
            "scope_type": "camera",
            "target_ids": ["cam-01"]
        }
    """
    prompt_compiler = request.app.get(_prompt_compiler_key)
    if prompt_compiler is None:
        return web.json_response(
            {"error": "Prompt compiler not configured"}, status=501,
        )

    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    scope_type = body.get("scope_type", "camera")
    target_ids = body.get("target_ids", [])
    if not target_ids:
        return web.json_response(
            {"error": "target_ids field is required"}, status=400,
        )

    prompt_text = body.get("prompt", "")
    if not prompt_text:
        return web.json_response(
            {"error": "prompt field is required for activation"}, status=400,
        )

    scope = PromptScope(scope_type=scope_type, target_ids=target_ids)

    try:
        compiled = await prompt_compiler.compile(prompt_text, scope)
        version_ids = await prompt_compiler.confirm_and_activate(
            compiled, scope,
        )
    except Exception as exc:
        logger.error("Mobile: Prompt activation failed: %s", exc)
        return web.json_response(
            {"error": f"Activation failed: {exc}"}, status=500,
        )

    return web.json_response({
        "status": "activated",
        "version_ids": version_ids,
    })


async def get_rules(request: web.Request) -> web.Response:
    """Return the active ruleset for a camera.

    ``GET /api/mobile/rules/{camera_id}``
    """
    rule_store = request.app.get(_rule_store_key)
    if rule_store is None:
        return web.json_response(
            {"error": "Rule store not configured"}, status=501,
        )

    camera_id = request.match_info["camera_id"]
    active = rule_store.get_active_ruleset(camera_id)
    if active is None:
        return web.json_response(
            {"error": f"No active ruleset for camera {camera_id}"}, status=404,
        )

    return web.json_response({
        "version_id": active.version_id,
        "camera_id": active.camera_id,
        "created_at": _serialize_datetime(active.created_at),
        "rules": [
            {
                "rule_id": r.rule_id,
                "object_type": r.object_type,
                "min_confidence": r.min_confidence,
            }
            for r in active.rules
        ],
    })


# ---------------------------------------------------------------------------
# Request handlers — Push Notification Registration
# ---------------------------------------------------------------------------


async def register_device(request: web.Request) -> web.Response:
    """Register a device for push notifications.

    ``POST /api/mobile/push/register``

    Request JSON::

        {
            "device_token": "abc123...",
            "platform": "ios",
            "tenant_id": "t1"
        }
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    device_token = body.get("device_token", "")
    platform = body.get("platform", "")
    tenant_id = body.get("tenant_id", "")

    if not device_token:
        return web.json_response(
            {"error": "device_token is required"}, status=400,
        )
    if platform not in ("ios", "android"):
        return web.json_response(
            {"error": "platform must be 'ios' or 'android'"}, status=400,
        )
    if not tenant_id:
        return web.json_response(
            {"error": "tenant_id is required"}, status=400,
        )

    devices: dict[str, dict[str, Any]] = request.app[_push_devices_key]
    entry = _make_device_entry(device_token, platform, tenant_id)
    devices[device_token] = entry

    logger.info(
        "Mobile: Registered device %s (%s) for tenant %s",
        device_token[:8],
        platform,
        tenant_id,
    )
    return web.json_response({
        "status": "registered",
        "device_token": device_token,
        "platform": platform,
        "tenant_id": tenant_id,
    })


async def unregister_device(request: web.Request) -> web.Response:
    """Unregister a device from push notifications.

    ``DELETE /api/mobile/push/unregister``

    Request JSON::

        {
            "device_token": "abc123..."
        }
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    device_token = body.get("device_token", "")
    if not device_token:
        return web.json_response(
            {"error": "device_token is required"}, status=400,
        )

    devices: dict[str, dict[str, Any]] = request.app[_push_devices_key]
    if device_token not in devices:
        return web.json_response(
            {"error": "Device not found"}, status=404,
        )

    del devices[device_token]
    logger.info("Mobile: Unregistered device %s", device_token[:8])
    return web.json_response({
        "status": "unregistered",
        "device_token": device_token,
    })


async def list_devices(request: web.Request) -> web.Response:
    """List registered push notification devices.

    ``GET /api/mobile/push/devices``

    Query param: ``tenant_id`` (optional filter).
    """
    devices: dict[str, dict[str, Any]] = request.app[_push_devices_key]
    tenant_id = request.query.get("tenant_id")

    device_list = list(devices.values())
    if tenant_id:
        device_list = [d for d in device_list if d["tenant_id"] == tenant_id]

    return web.json_response({"devices": device_list})


# ---------------------------------------------------------------------------
# Request handlers — Mobile Summary
# ---------------------------------------------------------------------------


async def get_summary(request: web.Request) -> web.Response:
    """Return a compact mobile-optimised dashboard summary.

    ``GET /api/mobile/summary``

    Response JSON::

        {
            "total_cameras": 10,
            "online_cameras": 8,
            "offline_cameras": 2,
            "total_alerts": 25,
            "active_alerts": 5,
            "recent_critical_alerts": [...]
        }
    """
    watchdog: Watchdog = request.app[_watchdog_key]
    tsdb: TimeSeriesDB = request.app[_tsdb_key]

    devices = watchdog.get_all_device_status()
    total_cameras = len(devices)
    online_count = sum(1 for d in devices if d.status == "online")
    offline_count = total_cameras - online_count

    # Get recent alerts
    recent_alerts = tsdb.get_alerts(limit=100)
    total_alerts = len(recent_alerts)

    # Count active alerts (not acknowledged/dismissed)
    active_alerts = sum(
        1
        for a in recent_alerts
        if a.get("status", "active") == "active"
    )

    # Get recent critical/high threat alerts for the summary
    critical_alerts = [
        a
        for a in recent_alerts
        if a.get("threat_level") in ("critical", "high")
    ][:5]

    return web.json_response({
        "total_cameras": total_cameras,
        "online_cameras": online_count,
        "offline_cameras": offline_count,
        "total_alerts": total_alerts,
        "active_alerts": active_alerts,
        "recent_critical_alerts": critical_alerts,
    })


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_mobile_app(
    watchdog: Watchdog,
    timeseries_db: TimeSeriesDB,
    alert_system: Optional[object] = None,
    rule_store: Optional[RuleStore] = None,
    prompt_compiler: Optional[object] = None,
    context_filter: Optional[object] = None,
) -> web.Application:
    """Create an aiohttp Application with all mobile API routes.

    Parameters
    ----------
    watchdog:
        The Watchdog instance for camera health data.
    timeseries_db:
        The TimeSeriesDB instance for events and alerts.
    alert_system:
        Optional AlertSystem instance for escalation.
    rule_store:
        Optional RuleStore instance for rule set management.
    prompt_compiler:
        Optional PromptCompiler instance for natural language prompts.
    context_filter:
        Optional ContextFilter instance for rule reload.

    Returns
    -------
    web.Application
        An aiohttp application ready to be run or composed.
    """
    # Run DB migration for alert status column
    _migrate_alert_status(timeseries_db)

    app = web.Application(middlewares=[cors_middleware])

    # Store dependencies
    app[_watchdog_key] = watchdog
    app[_tsdb_key] = timeseries_db
    app[_push_devices_key] = {}  # in-memory device registry

    if alert_system is not None:
        app[_alert_system_key] = alert_system
    if rule_store is not None:
        app[_rule_store_key] = rule_store
    if prompt_compiler is not None:
        app[_prompt_compiler_key] = prompt_compiler
    if context_filter is not None:
        app[_context_filter_key] = context_filter

    # Alert management
    app.router.add_get("/api/mobile/alerts", get_alerts)
    app.router.add_get("/api/mobile/alerts/{alert_id}", get_alert)
    app.router.add_post(
        "/api/mobile/alerts/{alert_id}/acknowledge", acknowledge_alert,
    )
    app.router.add_post(
        "/api/mobile/alerts/{alert_id}/dismiss", dismiss_alert,
    )
    app.router.add_post(
        "/api/mobile/alerts/{alert_id}/escalate", escalate_alert,
    )

    # Prompt configuration
    app.router.add_post("/api/mobile/prompt/compile", compile_prompt)
    app.router.add_post("/api/mobile/prompt/activate", activate_prompt)
    app.router.add_get("/api/mobile/rules/{camera_id}", get_rules)

    # Push notification registration
    app.router.add_post("/api/mobile/push/register", register_device)
    app.router.add_delete("/api/mobile/push/unregister", unregister_device)
    app.router.add_get("/api/mobile/push/devices", list_devices)

    # Mobile summary
    app.router.add_get("/api/mobile/summary", get_summary)

    return app


async def start_mobile_server(
    watchdog: Watchdog,
    timeseries_db: TimeSeriesDB,
    alert_system: Optional[object] = None,
    rule_store: Optional[RuleStore] = None,
    prompt_compiler: Optional[object] = None,
    context_filter: Optional[object] = None,
    host: str = "0.0.0.0",
    port: int = 8082,
) -> tuple[web.AppRunner, web.TCPSite]:
    """Start the mobile API server as a background aiohttp site.

    Parameters
    ----------
    watchdog:
        The Watchdog instance for camera health data.
    timeseries_db:
        The TimeSeriesDB instance for events and alerts.
    alert_system:
        Optional AlertSystem instance.
    rule_store:
        Optional RuleStore instance.
    prompt_compiler:
        Optional PromptCompiler instance.
    context_filter:
        Optional ContextFilter instance.
    host:
        Bind address.  Defaults to ``"0.0.0.0"``.
    port:
        Bind port.  Defaults to ``8082``.

    Returns
    -------
    tuple[web.AppRunner, web.TCPSite]
        The runner and site for cleanup via ``await runner.cleanup()``.
    """
    app = create_mobile_app(
        watchdog=watchdog,
        timeseries_db=timeseries_db,
        alert_system=alert_system,
        rule_store=rule_store,
        prompt_compiler=prompt_compiler,
        context_filter=context_filter,
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Mobile API server started on http://%s:%d", host, port)
    return runner, site
