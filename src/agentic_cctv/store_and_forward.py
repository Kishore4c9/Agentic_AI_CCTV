"""SQLite-backed store-and-forward queue for offline MQTT message buffering.

Every MQTT publish first writes to a local SQLite-backed queue (the "outbox").
On successful delivery (PUBACK/PUBCOMP), the message is removed.  On connection
loss, messages accumulate (up to ``max_age_seconds`` of buffering).  On
reconnection the queue drains in FIFO order before new messages are published.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_cctv.mqtt_client import MQTTPublisher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_CREATE_OUTBOX_TABLE = """\
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    payload BLOB NOT NULL,
    qos INTEGER NOT NULL,
    retain BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    attempts INTEGER DEFAULT 0
);
"""

_CREATE_OUTBOX_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_outbox_created ON outbox(created_at);
"""

_INSERT_MESSAGE = """\
INSERT INTO outbox (topic, payload, qos, retain) VALUES (?, ?, ?, ?);
"""

_SELECT_ALL_FIFO = """\
SELECT id, topic, payload, qos, retain FROM outbox ORDER BY id ASC;
"""

_DELETE_BY_ID = """\
DELETE FROM outbox WHERE id = ?;
"""

_COUNT_MESSAGES = """\
SELECT COUNT(*) FROM outbox;
"""

_INCREMENT_ATTEMPTS = """\
UPDATE outbox SET attempts = attempts + 1 WHERE id = ?;
"""


# ---------------------------------------------------------------------------
# StoreAndForwardQueue
# ---------------------------------------------------------------------------


class StoreAndForwardQueue:
    """SQLite-backed persistent queue for offline MQTT message buffering.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for tests.
    max_age_seconds:
        Maximum age (in seconds) for buffered messages.  Messages older than
        this are eligible for pruning.  Defaults to 300 (5 minutes).
    """

    def __init__(self, db_path: str, max_age_seconds: int = 300) -> None:
        self._db_path = db_path
        self._max_age_seconds = max_age_seconds
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_OUTBOX_TABLE)
        self._conn.execute(_CREATE_OUTBOX_INDEX)
        self._conn.commit()

    # -- public API ---------------------------------------------------------

    def enqueue(
        self,
        topic: str,
        payload: bytes,
        qos: int,
        retain: bool = False,
    ) -> None:
        """Insert a message into the outbox table.

        Parameters
        ----------
        topic:
            MQTT topic string.
        payload:
            Message payload as bytes.
        qos:
            Quality of Service level (0, 1, or 2).
        retain:
            Whether the broker should retain this message.
        """
        self._conn.execute(_INSERT_MESSAGE, (topic, payload, qos, retain))
        self._conn.commit()
        logger.debug("Enqueued message to topic %s (qos=%d)", topic, qos)

    def dequeue(self, message_id: int) -> None:
        """Remove a specific message from the outbox after successful delivery.

        Parameters
        ----------
        message_id:
            The ``id`` column value of the message to remove.
        """
        self._conn.execute(_DELETE_BY_ID, (message_id,))
        self._conn.commit()

    def drain(self, publisher: MQTTPublisher) -> int:
        """Drain all queued messages in FIFO order via *publisher*.

        For each message the method attempts to publish through the
        ``MQTTPublisher``.  Successfully published messages are removed from
        the queue.  Messages that fail to publish remain in the queue with
        their ``attempts`` counter incremented.

        Because ``MQTTPublisher.publish`` is async, this method obtains (or
        creates) an event loop and runs each publish call synchronously.

        Parameters
        ----------
        publisher:
            An ``MQTTPublisher`` instance to publish through.

        Returns
        -------
        int
            The number of messages successfully drained (published and removed).
        """
        cursor = self._conn.execute(_SELECT_ALL_FIFO)
        rows = cursor.fetchall()

        if not rows:
            return 0

        drained = 0
        loop = _get_or_create_event_loop()

        for row in rows:
            msg_id, topic, payload, qos, retain = row
            try:
                loop.run_until_complete(
                    publisher.publish(
                        topic,
                        bytes(payload),
                        qos=qos,
                        retain=bool(retain),
                    )
                )
                self.dequeue(msg_id)
                drained += 1
                logger.debug("Drained message %d to topic %s", msg_id, topic)
            except Exception:
                # Increment attempts counter but leave message in queue
                self._conn.execute(_INCREMENT_ATTEMPTS, (msg_id,))
                self._conn.commit()
                logger.warning(
                    "Failed to drain message %d to topic %s", msg_id, topic,
                    exc_info=True,
                )

        return drained

    def size(self) -> int:
        """Return the number of messages currently in the queue."""
        cursor = self._conn.execute(_COUNT_MESSAGES)
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Return the running event loop or create a new one.

    This is needed because ``drain`` is synchronous but calls async
    ``MQTTPublisher.publish``.  If there is already a running loop (e.g.
    during tests with ``pytest-asyncio``), we create a *new* loop to avoid
    "cannot run nested event loop" errors.
    """
    try:
        loop = asyncio.get_running_loop()
        # If we're inside a running loop we can't use run_until_complete on it.
        # Create a fresh loop for synchronous draining.
        return asyncio.new_event_loop()
    except RuntimeError:
        # No running loop — safe to get/create one.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
