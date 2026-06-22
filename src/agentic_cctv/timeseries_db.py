"""Time Series DB writer for the Agentic AI CCTV Monitoring Framework.

Provides SQLite-backed persistence for events, alerts, heartbeats, and rule sets.
Includes an MQTT subscriber callback that persists every ``StructuredEvent``
to the ``events`` table independently of the agent pipeline.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Optional

from agentic_cctv.models import (
    AlertPayload,
    BoundingBox,
    HeartbeatMessage,
    StructuredEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    object_type TEXT NOT NULL,
    track_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    bounding_box TEXT NOT NULL,
    detection_gate_passed BOOLEAN NOT NULL,
    context_gate_passed BOOLEAN NOT NULL,
    vlm_invoked BOOLEAN DEFAULT FALSE,
    action_taken TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_ALERTS_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(event_id),
    camera_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    threat_level TEXT NOT NULL,
    description TEXT,
    delivered_channels TEXT,
    cooldown_suppressed_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_HEARTBEATS_TABLE = """
CREATE TABLE IF NOT EXISTS heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    cpu_percent REAL,
    memory_percent REAL,
    temperature_celsius REAL,
    inference_latency_ms REAL,
    gpu_utilization_percent REAL
);
"""

_CREATE_RULE_SETS_TABLE = """
CREATE TABLE IF NOT EXISTS rule_sets (
    version_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    rules TEXT NOT NULL,
    original_prompt TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE
);
"""

_CREATE_AGGREGATED_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS aggregated_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    object_type TEXT NOT NULL,
    event_count INTEGER NOT NULL,
    avg_confidence REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, camera_id, tenant_id, site_id, object_type)
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_camera_time ON events(camera_id, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_events_tenant ON events(tenant_id);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_camera_time ON alerts(camera_id, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_heartbeats_camera_time ON heartbeats(camera_id, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_rule_sets_camera ON rule_sets(camera_id, is_active);",
    "CREATE INDEX IF NOT EXISTS idx_aggregated_events_date ON aggregated_events(date);",
    "CREATE INDEX IF NOT EXISTS idx_aggregated_events_tenant ON aggregated_events(tenant_id);",
]


# ---------------------------------------------------------------------------
# TimeSeriesDB
# ---------------------------------------------------------------------------


class TimeSeriesDB:
    """SQLite-backed time series database for events, alerts, heartbeats, and rule sets.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for in-memory databases.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._create_schema()

    # -- schema setup -------------------------------------------------------

    def _create_schema(self) -> None:
        """Create all tables and indexes if they do not already exist."""
        cursor = self._conn.cursor()
        cursor.execute(_CREATE_EVENTS_TABLE)
        cursor.execute(_CREATE_ALERTS_TABLE)
        cursor.execute(_CREATE_HEARTBEATS_TABLE)
        cursor.execute(_CREATE_RULE_SETS_TABLE)
        cursor.execute(_CREATE_AGGREGATED_EVENTS_TABLE)
        for idx_sql in _CREATE_INDEXES:
            cursor.execute(idx_sql)
        self._conn.commit()

    # -- insert methods -----------------------------------------------------

    def insert_event(
        self,
        event: StructuredEvent,
        detection_gate_passed: bool = True,
        context_gate_passed: bool = False,
    ) -> None:
        """Persist a :class:`StructuredEvent` to the ``events`` table.

        Parameters
        ----------
        event:
            The structured event to persist.
        detection_gate_passed:
            Whether the event passed the detection gate.
        context_gate_passed:
            Whether the event passed the context gate.
        """
        bbox_json = json.dumps(
            {
                "x": event.bounding_box.x,
                "y": event.bounding_box.y,
                "width": event.bounding_box.width,
                "height": event.bounding_box.height,
            }
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO events
                (event_id, camera_id, tenant_id, site_id, timestamp,
                 object_type, track_id, confidence, bounding_box,
                 detection_gate_passed, context_gate_passed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.camera_id,
                event.tenant_id,
                event.site_id,
                event.timestamp.isoformat(),
                event.object_type,
                event.track_id,
                event.confidence,
                bbox_json,
                detection_gate_passed,
                context_gate_passed,
            ),
        )
        self._conn.commit()

    def insert_alert(
        self,
        alert: AlertPayload,
        delivered_channels: list[str],
        cooldown_suppressed_count: int = 0,
    ) -> None:
        """Persist an :class:`AlertPayload` to the ``alerts`` table.

        Parameters
        ----------
        alert:
            The alert payload to persist.
        delivered_channels:
            List of channel names the alert was delivered through.
        cooldown_suppressed_count:
            Number of duplicate alerts suppressed by cooldown.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO alerts
                (alert_id, event_id, camera_id, tenant_id,
                 alert_type, threat_level, description,
                 delivered_channels, cooldown_suppressed_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert.alert_id,
                alert.event_id,
                alert.camera_id,
                alert.tenant_id,
                alert.alert_type,
                alert.threat_level,
                alert.description,
                json.dumps(delivered_channels),
                cooldown_suppressed_count,
            ),
        )
        self._conn.commit()

    def insert_heartbeat(self, heartbeat: HeartbeatMessage) -> None:
        """Persist a :class:`HeartbeatMessage` to the ``heartbeats`` table.

        Parameters
        ----------
        heartbeat:
            The heartbeat message to persist.
        """
        self._conn.execute(
            """
            INSERT INTO heartbeats
                (camera_id, tenant_id, site_id, timestamp,
                 cpu_percent, memory_percent, temperature_celsius,
                 inference_latency_ms, gpu_utilization_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                heartbeat.camera_id,
                heartbeat.tenant_id,
                heartbeat.site_id,
                heartbeat.timestamp.isoformat(),
                heartbeat.cpu_percent,
                heartbeat.memory_percent,
                heartbeat.temperature_celsius,
                heartbeat.inference_latency_ms,
                heartbeat.gpu_utilization_percent,
            ),
        )
        self._conn.commit()

    # -- query methods ------------------------------------------------------

    def get_events(
        self,
        camera_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query events with optional camera and tenant filters.

        Parameters
        ----------
        camera_id:
            Filter by camera ID.  ``None`` returns all cameras.
        tenant_id:
            Filter by tenant ID.  ``None`` returns all tenants.
        limit:
            Maximum number of rows to return.

        Returns
        -------
        list[dict]
            List of event rows as dictionaries.
        """
        query = "SELECT * FROM events"
        conditions: list[str] = []
        params: list[object] = []

        if camera_id is not None:
            conditions.append("camera_id = ?")
            params.append(camera_id)
        if tenant_id is not None:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_events_by_time_range(
        self,
        camera_id: str,
        start_iso: str,
        end_iso: str,
        limit: int = 10000,
    ) -> list[dict]:
        """Query events for a specific camera within a time range.

        Parameters
        ----------
        camera_id:
            The camera ID to filter by.
        start_iso:
            ISO-8601 datetime string for the start of the range (inclusive).
        end_iso:
            ISO-8601 datetime string for the end of the range (inclusive).
        limit:
            Maximum number of rows to return.

        Returns
        -------
        list[dict]
            List of event rows as dictionaries.
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM events
            WHERE camera_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (camera_id, start_iso, end_iso, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_alerts(
        self,
        camera_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query alerts with optional camera and tenant filters.

        Parameters
        ----------
        camera_id:
            Filter by camera ID.  ``None`` returns all cameras.
        tenant_id:
            Filter by tenant ID.  ``None`` returns all tenants.
        limit:
            Maximum number of rows to return.

        Returns
        -------
        list[dict]
            List of alert rows as dictionaries.
        """
        query = "SELECT * FROM alerts"
        conditions: list[str] = []
        params: list[object] = []

        if camera_id is not None:
            conditions.append("camera_id = ?")
            params.append(camera_id)
        if tenant_id is not None:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_heartbeats(
        self,
        tenant_id: str,
        camera_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query heartbeats scoped by tenant with optional camera filter.

        Parameters
        ----------
        tenant_id:
            **Required.** Filter by tenant ID to enforce tenant isolation.
        camera_id:
            Filter by camera ID.  ``None`` returns all cameras for the tenant.
        limit:
            Maximum number of rows to return.

        Returns
        -------
        list[dict]
            List of heartbeat rows as dictionaries.
        """
        query = "SELECT * FROM heartbeats WHERE tenant_id = ?"
        params: list[object] = [tenant_id]

        if camera_id is not None:
            query += " AND camera_id = ?"
            params.append(camera_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # -- retention methods -------------------------------------------------

    def aggregate_events(self, cutoff_iso: str) -> int:
        """Aggregate raw events older than *cutoff_iso* into daily summaries.

        Creates or updates rows in the ``aggregated_events`` table with daily
        counts and average confidence grouped by camera, tenant, site, and
        object type.

        Parameters
        ----------
        cutoff_iso:
            ISO-8601 datetime string.  Events with ``timestamp < cutoff_iso``
            are aggregated.

        Returns
        -------
        int
            Number of aggregation rows upserted.
        """
        cursor = self._conn.execute(
            """
            SELECT DATE(timestamp) AS date,
                   camera_id,
                   tenant_id,
                   site_id,
                   object_type,
                   COUNT(*) AS event_count,
                   AVG(confidence) AS avg_confidence
            FROM events
            WHERE timestamp < ?
            GROUP BY DATE(timestamp), camera_id, tenant_id, site_id, object_type
            """,
            (cutoff_iso,),
        )
        rows = cursor.fetchall()
        count = 0
        for row in rows:
            self._conn.execute(
                """
                INSERT INTO aggregated_events
                    (date, camera_id, tenant_id, site_id, object_type,
                     event_count, avg_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, camera_id, tenant_id, site_id, object_type)
                DO UPDATE SET
                    event_count = event_count + excluded.event_count,
                    avg_confidence = (
                        (avg_confidence * event_count + excluded.avg_confidence * excluded.event_count)
                        / (event_count + excluded.event_count)
                    )
                """,
                (
                    row[0],  # date
                    row[1],  # camera_id
                    row[2],  # tenant_id
                    row[3],  # site_id
                    row[4],  # object_type
                    row[5],  # event_count
                    row[6],  # avg_confidence
                ),
            )
            count += 1
        self._conn.commit()
        if count > 0:
            logger.info("Aggregated %d groups of raw events.", count)
        return count

    def purge_raw_events(self, cutoff_iso: str) -> int:
        """Delete raw events older than *cutoff_iso*.

        Also deletes alerts that reference purged events and heartbeats
        older than the same cutoff.

        Parameters
        ----------
        cutoff_iso:
            ISO-8601 datetime string.  Events with ``timestamp < cutoff_iso``
            are deleted.

        Returns
        -------
        int
            Number of raw events deleted.
        """
        # Delete alerts referencing events that will be purged
        self._conn.execute(
            """
            DELETE FROM alerts
            WHERE event_id IN (
                SELECT event_id FROM events WHERE timestamp < ?
            )
            """,
            (cutoff_iso,),
        )
        # Delete old raw events
        cursor = self._conn.execute(
            "DELETE FROM events WHERE timestamp < ?",
            (cutoff_iso,),
        )
        event_count = cursor.rowcount

        # Delete old heartbeats with the same cutoff
        self._conn.execute(
            "DELETE FROM heartbeats WHERE timestamp < ?",
            (cutoff_iso,),
        )
        self._conn.commit()

        if event_count > 0:
            logger.info("Purged %d raw events older than %s.", event_count, cutoff_iso)
        return event_count

    def purge_aggregated_events(self, cutoff_iso: str) -> int:
        """Delete aggregated event rows older than *cutoff_iso*.

        Parameters
        ----------
        cutoff_iso:
            ISO-8601 date string (``YYYY-MM-DD``).  Aggregated rows with
            ``date < cutoff_iso`` are deleted.

        Returns
        -------
        int
            Number of aggregated rows deleted.
        """
        cursor = self._conn.execute(
            "DELETE FROM aggregated_events WHERE date < ?",
            (cutoff_iso,),
        )
        count = cursor.rowcount
        self._conn.commit()
        if count > 0:
            logger.info(
                "Purged %d aggregated event rows older than %s.", count, cutoff_iso
            )
        return count

    def get_all_event_ids(self) -> set[str]:
        """Return the set of all event IDs currently in the events table.

        Used by VectorDB retention to identify orphaned embeddings.

        Returns
        -------
        set[str]
            Set of event_id strings.
        """
        cursor = self._conn.execute("SELECT event_id FROM events")
        return {row[0] for row in cursor.fetchall()}

    def get_aggregated_events(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query aggregated events with optional tenant filter.

        Parameters
        ----------
        tenant_id:
            Filter by tenant ID.  ``None`` returns all tenants.
        limit:
            Maximum number of rows to return.

        Returns
        -------
        list[dict]
            List of aggregated event rows as dictionaries.
        """
        query = "SELECT * FROM aggregated_events"
        params: list[object] = []

        if tenant_id is not None:
            query += " WHERE tenant_id = ?"
            params.append(tenant_id)

        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# TimeSeriesDBSubscriber — MQTT callback for independent event persistence
# ---------------------------------------------------------------------------


class TimeSeriesDBSubscriber:
    """MQTT subscriber callback that persists every ``StructuredEvent`` to the
    ``events`` table independently of the agent pipeline.

    Designed to be used as a callback with :class:`~agentic_cctv.mqtt_client.MQTTSubscriber`.

    Parameters
    ----------
    db:
        The :class:`TimeSeriesDB` instance to write events to.
    """

    def __init__(self, db: TimeSeriesDB) -> None:
        self._db = db

    def __call__(self, topic: str, payload: bytes, qos: int) -> None:
        """Handle an incoming MQTT message by parsing and persisting the event.

        Parameters
        ----------
        topic:
            The MQTT topic the message was received on.
        payload:
            The raw message payload (expected to be JSON-encoded ``StructuredEvent``).
        qos:
            The QoS level of the received message.
        """
        try:
            data = json.loads(payload)
            bbox_data = data.get("bounding_box", {})
            bounding_box = BoundingBox(
                x=int(bbox_data.get("x", 0)),
                y=int(bbox_data.get("y", 0)),
                width=int(bbox_data.get("width", 0)),
                height=int(bbox_data.get("height", 0)),
            )

            from datetime import datetime

            timestamp_str = data.get("timestamp", "")
            # Handle ISO format timestamps with or without trailing Z
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1] + "+00:00"
            timestamp = datetime.fromisoformat(timestamp_str)

            event = StructuredEvent(
                event_id=data["event_id"],
                camera_id=data["camera_id"],
                tenant_id=data["tenant_id"],
                site_id=data["site_id"],
                timestamp=timestamp,
                object_type=data["object_type"],
                track_id=data["track_id"],
                bounding_box=bounding_box,
                confidence=float(data["confidence"]),
                frame_crop=data.get("frame_crop", ""),
            )

            self._db.insert_event(event)
            logger.debug(
                "Persisted event %s from topic %s", event.event_id, topic
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.error(
                "Failed to parse/persist event from topic %s: %s", topic, exc
            )
