"""Operator Dashboard for the Agentic AI CCTV Monitoring Framework.

Provides a web-based operator dashboard with:
- REST API endpoints for camera health, events, alerts, rules, and prompt config
- Alert management (acknowledge, dismiss, escalate) with DB-backed status
- A single-page HTML/JS dashboard served inline
- Integration with Watchdog, TimeSeriesDB, AlertSystem, RuleStore, PromptCompiler,
  and ContextFilter

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from aiohttp import web

from agentic_cctv.models import (
    AlertPayload,
    DeviceHealth,
    HeartbeatMessage,
    PromptScope,
    RuleSet,
)
from agentic_cctv.rule_store import RuleStore
from agentic_cctv.timeseries_db import TimeSeriesDB
from agentic_cctv.watchdog import Watchdog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typed application keys
# ---------------------------------------------------------------------------

_watchdog_key = web.AppKey("dashboard_watchdog", Watchdog)
_tsdb_key = web.AppKey("dashboard_tsdb", TimeSeriesDB)
_alert_system_key = web.AppKey("dashboard_alert_system", object)
_rule_store_key = web.AppKey("dashboard_rule_store", RuleStore)
_prompt_compiler_key = web.AppKey("dashboard_prompt_compiler", object)
_context_filter_key = web.AppKey("dashboard_context_filter", object)


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _serialize_datetime(dt: Optional[datetime]) -> Optional[str]:
    """Convert a datetime to ISO-8601 string, or ``None``."""
    if dt is None:
        return None
    return dt.isoformat()


def _serialize_heartbeat(hb: Optional[HeartbeatMessage]) -> Optional[dict[str, Any]]:
    """Convert a HeartbeatMessage to a JSON-safe dict, or ``None``."""
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
    """Convert a DeviceHealth to a JSON-safe dict."""
    return {
        "camera_id": dh.camera_id,
        "status": dh.status,
        "last_heartbeat": _serialize_datetime(dh.last_heartbeat),
        "metrics": _serialize_heartbeat(dh.metrics),
    }


# ---------------------------------------------------------------------------
# DB migration helper for alert status
# ---------------------------------------------------------------------------


def _migrate_alert_status(tsdb: TimeSeriesDB) -> None:
    """Add ``status`` column to the alerts table if not present.

    Status values: ``active``, ``acknowledged``, ``dismissed``, ``escalated``.
    """
    cursor = tsdb._conn.execute("PRAGMA table_info(alerts)")
    columns = {row[1] for row in cursor.fetchall()}
    if "status" not in columns:
        tsdb._conn.execute(
            "ALTER TABLE alerts ADD COLUMN status TEXT DEFAULT 'active'"
        )
        tsdb._conn.commit()
        logger.info("Migrated alerts table: added status column")


def _update_alert_status(tsdb: TimeSeriesDB, alert_id: str, status: str) -> bool:
    """Update the status of an alert. Returns True if the alert existed."""
    cursor = tsdb._conn.execute(
        "UPDATE alerts SET status = ? WHERE alert_id = ?",
        (status, alert_id),
    )
    tsdb._conn.commit()
    return cursor.rowcount > 0


def _get_alert_by_id(tsdb: TimeSeriesDB, alert_id: str) -> Optional[dict]:
    """Get a single alert by ID."""
    cursor = tsdb._conn.execute(
        "SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return dict(row)


def _get_event_by_id(tsdb: TimeSeriesDB, event_id: str) -> Optional[dict]:
    """Get a single event by ID."""
    cursor = tsdb._conn.execute(
        "SELECT * FROM events WHERE event_id = ?", (event_id,)
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# Request handlers — Camera Health
# ---------------------------------------------------------------------------


async def get_cameras(request: web.Request) -> web.Response:
    """Return health status for all tracked cameras.

    ``GET /api/dashboard/cameras``
    """
    watchdog: Watchdog = request.app[_watchdog_key]
    devices = watchdog.get_all_device_status()
    return web.json_response({
        "cameras": [_serialize_device_health(d) for d in devices],
    })


async def get_camera(request: web.Request) -> web.Response:
    """Return health status for a single camera.

    ``GET /api/dashboard/cameras/{camera_id}``
    """
    watchdog: Watchdog = request.app[_watchdog_key]
    camera_id = request.match_info["camera_id"]
    device = watchdog.get_device_status(camera_id)
    return web.json_response(_serialize_device_health(device))


# ---------------------------------------------------------------------------
# Request handlers — Events
# ---------------------------------------------------------------------------


async def get_events(request: web.Request) -> web.Response:
    """Return recent events with optional filters.

    ``GET /api/dashboard/events``

    Query params: ``camera_id``, ``tenant_id``, ``limit`` (default 50).
    """
    tsdb: TimeSeriesDB = request.app[_tsdb_key]
    camera_id = request.query.get("camera_id")
    tenant_id = request.query.get("tenant_id")
    try:
        limit = int(request.query.get("limit", "50"))
    except (ValueError, TypeError):
        limit = 50
    limit = max(1, min(limit, 1000))

    events = tsdb.get_events(camera_id=camera_id, tenant_id=tenant_id, limit=limit)
    return web.json_response({"events": events})


async def get_event(request: web.Request) -> web.Response:
    """Return a single event by ID.

    ``GET /api/dashboard/events/{event_id}``
    """
    tsdb: TimeSeriesDB = request.app[_tsdb_key]
    event_id = request.match_info["event_id"]
    event = _get_event_by_id(tsdb, event_id)
    if event is None:
        return web.json_response({"error": "Event not found"}, status=404)
    return web.json_response(event)


# ---------------------------------------------------------------------------
# Request handlers — Alerts
# ---------------------------------------------------------------------------


async def get_alerts(request: web.Request) -> web.Response:
    """Return recent alerts with optional filters.

    ``GET /api/dashboard/alerts``

    Query params: ``camera_id``, ``tenant_id``, ``limit`` (default 50).
    """
    tsdb: TimeSeriesDB = request.app[_tsdb_key]
    camera_id = request.query.get("camera_id")
    tenant_id = request.query.get("tenant_id")
    try:
        limit = int(request.query.get("limit", "50"))
    except (ValueError, TypeError):
        limit = 50
    limit = max(1, min(limit, 1000))

    alerts = tsdb.get_alerts(camera_id=camera_id, tenant_id=tenant_id, limit=limit)
    return web.json_response({"alerts": alerts})


async def acknowledge_alert(request: web.Request) -> web.Response:
    """Acknowledge an alert.

    ``POST /api/dashboard/alerts/{alert_id}/acknowledge``
    """
    tsdb: TimeSeriesDB = request.app[_tsdb_key]
    alert_id = request.match_info["alert_id"]

    alert = _get_alert_by_id(tsdb, alert_id)
    if alert is None:
        return web.json_response({"error": "Alert not found"}, status=404)

    _update_alert_status(tsdb, alert_id, "acknowledged")
    logger.info("Alert %s acknowledged", alert_id)
    return web.json_response({
        "alert_id": alert_id,
        "status": "acknowledged",
    })


async def dismiss_alert(request: web.Request) -> web.Response:
    """Dismiss an alert.

    ``POST /api/dashboard/alerts/{alert_id}/dismiss``
    """
    tsdb: TimeSeriesDB = request.app[_tsdb_key]
    alert_id = request.match_info["alert_id"]

    alert = _get_alert_by_id(tsdb, alert_id)
    if alert is None:
        return web.json_response({"error": "Alert not found"}, status=404)

    _update_alert_status(tsdb, alert_id, "dismissed")
    logger.info("Alert %s dismissed", alert_id)
    return web.json_response({
        "alert_id": alert_id,
        "status": "dismissed",
    })


async def escalate_alert(request: web.Request) -> web.Response:
    """Escalate an alert (re-send with higher threat level).

    ``POST /api/dashboard/alerts/{alert_id}/escalate``
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
            logger.exception("Failed to send escalated alert for %s", alert_id)

    logger.info("Alert %s escalated", alert_id)
    return web.json_response({
        "alert_id": alert_id,
        "status": "escalated",
    })


# ---------------------------------------------------------------------------
# Request handlers — Rule Sets
# ---------------------------------------------------------------------------


async def get_rules(request: web.Request) -> web.Response:
    """Return the active ruleset for a camera.

    ``GET /api/dashboard/rules/{camera_id}``
    """
    rule_store: RuleStore = request.app[_rule_store_key]
    camera_id = request.match_info["camera_id"]

    active = rule_store.get_active_ruleset(camera_id)
    if active is None:
        return web.json_response(
            {"error": f"No active ruleset for camera {camera_id}"}, status=404
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


async def get_rules_history(request: web.Request) -> web.Response:
    """Return version history for a camera's rulesets.

    ``GET /api/dashboard/rules/{camera_id}/history``
    """
    rule_store: RuleStore = request.app[_rule_store_key]
    camera_id = request.match_info["camera_id"]

    versions = rule_store.get_version_history(camera_id)
    return web.json_response({
        "camera_id": camera_id,
        "versions": [
            {
                "version_id": v.version_id,
                "camera_id": v.camera_id,
                "created_at": _serialize_datetime(v.created_at),
                "is_active": v.is_active,
            }
            for v in versions
        ],
    })


async def rollback_rules(request: web.Request) -> web.Response:
    """Rollback a camera's ruleset to a previous version.

    ``POST /api/dashboard/rules/{camera_id}/rollback``

    Request JSON: ``{"version_id": "..."}``
    """
    rule_store: RuleStore = request.app[_rule_store_key]
    camera_id = request.match_info["camera_id"]

    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    version_id = body.get("version_id")
    if not version_id:
        return web.json_response(
            {"error": "version_id is required"}, status=400
        )

    try:
        new_ruleset = rule_store.rollback(camera_id, version_id)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=404)

    # Trigger context filter reload if available
    context_filter = request.app.get(_context_filter_key)
    if context_filter is not None:
        try:
            context_filter.reload_rules(camera_id)
        except Exception:
            logger.exception("Failed to reload rules for %s", camera_id)

    return web.json_response({
        "camera_id": camera_id,
        "new_version_id": new_ruleset.version_id,
        "status": "ok",
    })


# ---------------------------------------------------------------------------
# Request handlers — Prompt Configuration
# ---------------------------------------------------------------------------


async def compile_prompt(request: web.Request) -> web.Response:
    """Compile a natural language prompt into a ruleset.

    ``POST /api/dashboard/prompt/compile``

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
            {"error": "Prompt compiler not configured"}, status=501
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
            {"error": "prompt field is required"}, status=400
        )
    if not target_ids:
        return web.json_response(
            {"error": "target_ids field is required"}, status=400
        )

    scope = PromptScope(scope_type=scope_type, target_ids=target_ids)

    try:
        compiled = await prompt_compiler.compile(prompt_text, scope)
    except Exception as exc:
        logger.error("Prompt compilation failed: %s", exc)
        return web.json_response(
            {"error": f"Compilation failed: {exc}"}, status=500
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

    ``POST /api/dashboard/prompt/activate``

    Request JSON: compiled ruleset data from the compile endpoint.
    """
    prompt_compiler = request.app.get(_prompt_compiler_key)
    if prompt_compiler is None:
        return web.json_response(
            {"error": "Prompt compiler not configured"}, status=501
        )

    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    # Extract scope and compiled data from the request
    scope_type = body.get("scope_type", "camera")
    target_ids = body.get("target_ids", [])
    if not target_ids:
        return web.json_response(
            {"error": "target_ids field is required"}, status=400
        )

    scope = PromptScope(scope_type=scope_type, target_ids=target_ids)

    try:
        # Re-compile if prompt is provided, or use provided ruleset data
        prompt_text = body.get("prompt", "")
        if prompt_text:
            compiled = await prompt_compiler.compile(prompt_text, scope)
            version_ids = await prompt_compiler.confirm_and_activate(compiled, scope)
        else:
            return web.json_response(
                {"error": "prompt field is required for activation"}, status=400
            )
    except Exception as exc:
        logger.error("Prompt activation failed: %s", exc)
        return web.json_response(
            {"error": f"Activation failed: {exc}"}, status=500
        )

    return web.json_response({
        "status": "activated",
        "version_ids": version_ids,
    })


# ---------------------------------------------------------------------------
# Request handlers — Dashboard Overview
# ---------------------------------------------------------------------------


async def get_overview(request: web.Request) -> web.Response:
    """Return summary stats for the dashboard.

    ``GET /api/dashboard/overview``
    """
    watchdog: Watchdog = request.app[_watchdog_key]
    tsdb: TimeSeriesDB = request.app[_tsdb_key]

    devices = watchdog.get_all_device_status()
    total_cameras = len(devices)
    online_count = sum(1 for d in devices if d.status == "online")
    offline_count = total_cameras - online_count

    # Get recent alert and event counts
    recent_alerts = tsdb.get_alerts(limit=1000)
    recent_events = tsdb.get_events(limit=1000)

    return web.json_response({
        "total_cameras": total_cameras,
        "online_cameras": online_count,
        "offline_cameras": offline_count,
        "recent_alert_count": len(recent_alerts),
        "recent_event_count": len(recent_events),
    })


# ---------------------------------------------------------------------------
# Inline HTML Dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CCTV Monitoring Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; padding: 16px; }
  h1 { color: #00d4ff; margin-bottom: 16px; font-size: 1.5rem; }
  h2 { color: #00d4ff; margin-bottom: 8px; font-size: 1.1rem; }
  .stats { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  .stat-card { background: #16213e; border-radius: 8px; padding: 16px; min-width: 140px;
               text-align: center; border: 1px solid #0f3460; }
  .stat-card .value { font-size: 2rem; font-weight: bold; color: #00d4ff; }
  .stat-card .label { font-size: 0.8rem; color: #a0a0a0; margin-top: 4px; }
  .section { background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 16px;
             border: 1px solid #0f3460; }
  .camera-grid { display: flex; gap: 8px; flex-wrap: wrap; }
  .camera-card { background: #0f3460; border-radius: 6px; padding: 10px; min-width: 160px; }
  .camera-card .status-dot { display: inline-block; width: 10px; height: 10px;
                              border-radius: 50%; margin-right: 6px; }
  .camera-card .status-dot.online { background: #00e676; }
  .camera-card .status-dot.offline { background: #ff1744; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.85rem; }
  th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #0f3460; }
  th { color: #00d4ff; font-weight: 600; }
  button { background: #0f3460; color: #e0e0e0; border: 1px solid #00d4ff; border-radius: 4px;
           padding: 4px 10px; cursor: pointer; font-size: 0.75rem; margin: 0 2px; }
  button:hover { background: #00d4ff; color: #1a1a2e; }
  .btn-ack { border-color: #00e676; }
  .btn-dismiss { border-color: #ffc107; }
  .btn-escalate { border-color: #ff1744; }
  .prompt-form { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
  .prompt-form input, .prompt-form select { background: #0f3460; color: #e0e0e0;
    border: 1px solid #00d4ff; border-radius: 4px; padding: 6px 10px; }
  .prompt-form input[type=text] { flex: 1; min-width: 200px; }
  #prompt-result { margin-top: 8px; font-size: 0.85rem; color: #a0a0a0; }
  .rule-viewer { margin-top: 8px; }
  .rule-viewer select { background: #0f3460; color: #e0e0e0; border: 1px solid #00d4ff;
    border-radius: 4px; padding: 4px 8px; margin-bottom: 8px; }
  #rule-content { background: #0a0a1a; padding: 10px; border-radius: 4px;
    font-family: monospace; font-size: 0.8rem; white-space: pre-wrap; min-height: 60px; }
</style>
</head>
<body>
<h1>CCTV Monitoring Dashboard</h1>

<div class="stats" id="overview-stats">
  <div class="stat-card"><div class="value" id="stat-total">-</div><div class="label">Total Cameras</div></div>
  <div class="stat-card"><div class="value" id="stat-online">-</div><div class="label">Online</div></div>
  <div class="stat-card"><div class="value" id="stat-offline">-</div><div class="label">Offline</div></div>
  <div class="stat-card"><div class="value" id="stat-alerts">-</div><div class="label">Recent Alerts</div></div>
  <div class="stat-card"><div class="value" id="stat-events">-</div><div class="label">Recent Events</div></div>
</div>

<div class="section">
  <h2>Camera Health</h2>
  <div class="camera-grid" id="camera-grid"></div>
</div>

<div class="section">
  <h2>Live Event Feed</h2>
  <table><thead><tr><th>Event ID</th><th>Camera</th><th>Type</th><th>Confidence</th><th>Time</th></tr></thead>
  <tbody id="event-tbody"></tbody></table>
</div>

<div class="section">
  <h2>Alerts</h2>
  <table><thead><tr><th>Alert ID</th><th>Camera</th><th>Type</th><th>Threat</th><th>Status</th><th>Actions</th></tr></thead>
  <tbody id="alert-tbody"></tbody></table>
</div>

<div class="section">
  <h2>Rule Set Viewer</h2>
  <div class="rule-viewer">
    <label>Camera: <select id="rule-camera-select"><option value="">Select camera...</option></select></label>
    <div id="rule-content">Select a camera to view its active ruleset.</div>
  </div>
</div>

<div class="section">
  <h2>Prompt Configuration</h2>
  <div class="prompt-form">
    <input type="text" id="prompt-input" placeholder="e.g. Alert me when a person is detected after 10pm">
    <input type="text" id="prompt-cameras" placeholder="Camera IDs (comma-separated)">
    <button onclick="compilePrompt()">Compile</button>
  </div>
  <div id="prompt-result"></div>
</div>

<script>
const API = '/api/dashboard';

async function fetchJSON(url) {
  const r = await fetch(url);
  return r.json();
}

async function postJSON(url, data) {
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  return r.json();
}

async function refreshOverview() {
  try {
    const d = await fetchJSON(API + '/overview');
    document.getElementById('stat-total').textContent = d.total_cameras;
    document.getElementById('stat-online').textContent = d.online_cameras;
    document.getElementById('stat-offline').textContent = d.offline_cameras;
    document.getElementById('stat-alerts').textContent = d.recent_alert_count;
    document.getElementById('stat-events').textContent = d.recent_event_count;
  } catch(e) { console.error('Overview fetch failed', e); }
}

async function refreshCameras() {
  try {
    const d = await fetchJSON(API + '/cameras');
    const grid = document.getElementById('camera-grid');
    grid.innerHTML = '';
    const sel = document.getElementById('rule-camera-select');
    sel.innerHTML = '<option value="">Select camera...</option>';
    d.cameras.forEach(c => {
      const card = document.createElement('div');
      card.className = 'camera-card';
      card.innerHTML = '<span class="status-dot ' + c.status + '"></span><strong>' + c.camera_id + '</strong><br><small>' + c.status + '</small>';
      grid.appendChild(card);
      const opt = document.createElement('option');
      opt.value = c.camera_id; opt.textContent = c.camera_id;
      sel.appendChild(opt);
    });
  } catch(e) { console.error('Camera fetch failed', e); }
}

async function refreshEvents() {
  try {
    const d = await fetchJSON(API + '/events?limit=20');
    const tbody = document.getElementById('event-tbody');
    tbody.innerHTML = '';
    d.events.forEach(ev => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>' + (ev.event_id||'').substring(0,12) + '</td><td>' + (ev.camera_id||'') +
        '</td><td>' + (ev.object_type||'') + '</td><td>' + (ev.confidence||0).toFixed(2) +
        '</td><td>' + (ev.timestamp||'') + '</td>';
      tbody.appendChild(tr);
    });
  } catch(e) { console.error('Event fetch failed', e); }
}

async function refreshAlerts() {
  try {
    const d = await fetchJSON(API + '/alerts?limit=20');
    const tbody = document.getElementById('alert-tbody');
    tbody.innerHTML = '';
    d.alerts.forEach(a => {
      const tr = document.createElement('tr');
      const status = a.status || 'active';
      tr.innerHTML = '<td>' + (a.alert_id||'').substring(0,12) + '</td><td>' + (a.camera_id||'') +
        '</td><td>' + (a.alert_type||'') + '</td><td>' + (a.threat_level||'') +
        '</td><td>' + status + '</td><td>' +
        '<button class="btn-ack" onclick="alertAction(\\'' + a.alert_id + '\\',\\'acknowledge\\')">Ack</button>' +
        '<button class="btn-dismiss" onclick="alertAction(\\'' + a.alert_id + '\\',\\'dismiss\\')">Dismiss</button>' +
        '<button class="btn-escalate" onclick="alertAction(\\'' + a.alert_id + '\\',\\'escalate\\')">Escalate</button>' +
        '</td>';
      tbody.appendChild(tr);
    });
  } catch(e) { console.error('Alert fetch failed', e); }
}

async function alertAction(alertId, action) {
  try {
    await postJSON(API + '/alerts/' + alertId + '/' + action, {});
    refreshAlerts();
  } catch(e) { console.error('Alert action failed', e); }
}

async function loadRules() {
  const camId = document.getElementById('rule-camera-select').value;
  const el = document.getElementById('rule-content');
  if (!camId) { el.textContent = 'Select a camera to view its active ruleset.'; return; }
  try {
    const d = await fetchJSON(API + '/rules/' + camId);
    el.textContent = JSON.stringify(d, null, 2);
  } catch(e) { el.textContent = 'No active ruleset found.'; }
}

document.getElementById('rule-camera-select').addEventListener('change', loadRules);

async function compilePrompt() {
  const prompt = document.getElementById('prompt-input').value;
  const cams = document.getElementById('prompt-cameras').value.split(',').map(s => s.trim()).filter(Boolean);
  const el = document.getElementById('prompt-result');
  if (!prompt || !cams.length) { el.textContent = 'Please enter a prompt and camera IDs.'; return; }
  try {
    const d = await postJSON(API + '/prompt/compile', {prompt: prompt, scope_type: 'camera', target_ids: cams});
    el.textContent = JSON.stringify(d, null, 2);
  } catch(e) { el.textContent = 'Compilation failed: ' + e.message; }
}

function refreshAll() {
  refreshOverview();
  refreshCameras();
  refreshEvents();
  refreshAlerts();
}

refreshAll();
setInterval(refreshAll, 5000);
</script>
</body>
</html>
"""


async def serve_dashboard(request: web.Request) -> web.Response:
    """Serve the inline HTML dashboard page.

    ``GET /dashboard``
    """
    return web.Response(text=_DASHBOARD_HTML, content_type="text/html")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_dashboard_app(
    watchdog: Watchdog,
    timeseries_db: TimeSeriesDB,
    alert_system: Optional[object] = None,
    rule_store: Optional[RuleStore] = None,
    prompt_compiler: Optional[object] = None,
    context_filter: Optional[object] = None,
) -> web.Application:
    """Create an aiohttp Application with all dashboard routes.

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
        Optional ContextFilter instance for rule reload on rollback.

    Returns
    -------
    web.Application
        An aiohttp application ready to be run or composed.
    """
    # Run DB migration for alert status column
    _migrate_alert_status(timeseries_db)

    app = web.Application()

    # Store dependencies
    app[_watchdog_key] = watchdog
    app[_tsdb_key] = timeseries_db
    if alert_system is not None:
        app[_alert_system_key] = alert_system
    if rule_store is not None:
        app[_rule_store_key] = rule_store
    if prompt_compiler is not None:
        app[_prompt_compiler_key] = prompt_compiler
    if context_filter is not None:
        app[_context_filter_key] = context_filter

    # Dashboard HTML page
    app.router.add_get("/dashboard", serve_dashboard)

    # Camera health
    app.router.add_get("/api/dashboard/cameras", get_cameras)
    app.router.add_get("/api/dashboard/cameras/{camera_id}", get_camera)

    # Events
    app.router.add_get("/api/dashboard/events", get_events)
    app.router.add_get("/api/dashboard/events/{event_id}", get_event)

    # Alerts
    app.router.add_get("/api/dashboard/alerts", get_alerts)
    app.router.add_post("/api/dashboard/alerts/{alert_id}/acknowledge", acknowledge_alert)
    app.router.add_post("/api/dashboard/alerts/{alert_id}/dismiss", dismiss_alert)
    app.router.add_post("/api/dashboard/alerts/{alert_id}/escalate", escalate_alert)

    # Rule sets
    if rule_store is not None:
        app.router.add_get("/api/dashboard/rules/{camera_id}", get_rules)
        app.router.add_get("/api/dashboard/rules/{camera_id}/history", get_rules_history)
        app.router.add_post("/api/dashboard/rules/{camera_id}/rollback", rollback_rules)

    # Prompt configuration
    app.router.add_post("/api/dashboard/prompt/compile", compile_prompt)
    app.router.add_post("/api/dashboard/prompt/activate", activate_prompt)

    # Overview
    app.router.add_get("/api/dashboard/overview", get_overview)

    return app


async def start_dashboard_server(
    watchdog: Watchdog,
    timeseries_db: TimeSeriesDB,
    alert_system: Optional[object] = None,
    rule_store: Optional[RuleStore] = None,
    prompt_compiler: Optional[object] = None,
    context_filter: Optional[object] = None,
    host: str = "0.0.0.0",
    port: int = 8081,
) -> tuple[web.AppRunner, web.TCPSite]:
    """Start the dashboard server as a background aiohttp site.

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
        Bind address. Defaults to ``"0.0.0.0"``.
    port:
        Bind port. Defaults to ``8081``.

    Returns
    -------
    tuple[web.AppRunner, web.TCPSite]
        The runner and site for cleanup via ``await runner.cleanup()``.
    """
    app = create_dashboard_app(
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
    logger.info("Dashboard server started on http://%s:%d", host, port)
    return runner, site
