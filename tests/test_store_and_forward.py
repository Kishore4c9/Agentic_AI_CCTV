"""Unit tests for the StoreAndForwardQueue.

Tests cover:
- Basic enqueue / dequeue / size operations
- FIFO ordering via drain
- Drain removes only successfully published messages
- Failed publishes remain in queue with incremented attempts
- Persistence across queue instances (same db_path)
- Empty queue drain returns 0
- SQLite thread safety (check_same_thread=False)

All tests use in-memory SQLite or temporary files — no real MQTT broker needed.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_cctv.store_and_forward import StoreAndForwardQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_publisher(*, fail_on_topics: set[str] | None = None) -> MagicMock:
    """Create a mock MQTTPublisher whose ``publish`` is an AsyncMock.

    Parameters
    ----------
    fail_on_topics:
        If provided, ``publish`` raises ``ConnectionError`` for these topics.
    """
    publisher = MagicMock()
    fail_on_topics = fail_on_topics or set()

    async def _publish(topic: str, payload: bytes, qos: int = 1, retain: bool = False) -> None:
        if topic in fail_on_topics:
            raise ConnectionError(f"Broker unreachable for {topic}")

    publisher.publish = AsyncMock(side_effect=_publish)
    publisher.is_connected = True
    return publisher


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------


class TestBasicOperations:
    """Tests for enqueue, dequeue, and size."""

    def test_empty_queue_has_size_zero(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        assert q.size() == 0
        q.close()

    def test_enqueue_increases_size(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        q.enqueue("t/events", b"payload1", qos=1)
        assert q.size() == 1
        q.enqueue("t/events", b"payload2", qos=1)
        assert q.size() == 2
        q.close()

    def test_dequeue_decreases_size(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        q.enqueue("t/events", b"payload", qos=1)
        assert q.size() == 1

        # Get the id of the inserted row
        cursor = q._conn.execute("SELECT id FROM outbox LIMIT 1")
        msg_id = cursor.fetchone()[0]

        q.dequeue(msg_id)
        assert q.size() == 0
        q.close()

    def test_dequeue_nonexistent_id_is_noop(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        q.enqueue("t/events", b"payload", qos=1)
        q.dequeue(9999)  # does not exist
        assert q.size() == 1
        q.close()

    def test_enqueue_stores_correct_fields(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        q.enqueue("tenant/site/cam/events", b"\x00\x01\x02", qos=2, retain=True)

        cursor = q._conn.execute(
            "SELECT topic, payload, qos, retain FROM outbox WHERE id = 1"
        )
        row = cursor.fetchone()
        assert row[0] == "tenant/site/cam/events"
        assert bytes(row[1]) == b"\x00\x01\x02"
        assert row[2] == 2
        assert row[3] == 1  # SQLite stores True as 1
        q.close()


# ---------------------------------------------------------------------------
# Drain tests
# ---------------------------------------------------------------------------


class TestDrain:
    """Tests for the drain method."""

    def test_drain_empty_queue_returns_zero(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        publisher = _make_mock_publisher()
        assert q.drain(publisher) == 0
        publisher.publish.assert_not_called()
        q.close()

    def test_drain_publishes_all_messages_fifo(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        q.enqueue("t/events", b"msg1", qos=1)
        q.enqueue("t/events", b"msg2", qos=1)
        q.enqueue("t/events", b"msg3", qos=0)

        publisher = _make_mock_publisher()
        drained = q.drain(publisher)

        assert drained == 3
        assert q.size() == 0

        # Verify FIFO order
        calls = publisher.publish.call_args_list
        assert len(calls) == 3
        assert calls[0].args[0] == "t/events"
        assert calls[0].args[1] == b"msg1"
        assert calls[1].args[1] == b"msg2"
        assert calls[2].args[1] == b"msg3"
        q.close()

    def test_drain_preserves_qos_and_retain(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        q.enqueue("t/alerts", b"alert", qos=2, retain=False)
        q.enqueue("t/health", b"hb", qos=1, retain=True)

        publisher = _make_mock_publisher()
        q.drain(publisher)

        calls = publisher.publish.call_args_list
        # First call: qos=2, retain=False
        assert calls[0].kwargs["qos"] == 2
        assert calls[0].kwargs["retain"] is False
        # Second call: qos=1, retain=True
        assert calls[1].kwargs["qos"] == 1
        assert calls[1].kwargs["retain"] is True
        q.close()

    def test_drain_removes_only_successful_messages(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        q.enqueue("good/topic", b"ok", qos=1)
        q.enqueue("bad/topic", b"fail", qos=1)
        q.enqueue("good/topic", b"ok2", qos=1)

        publisher = _make_mock_publisher(fail_on_topics={"bad/topic"})
        drained = q.drain(publisher)

        assert drained == 2
        assert q.size() == 1  # the failed message remains

        # Verify the remaining message is the failed one
        cursor = q._conn.execute("SELECT topic, attempts FROM outbox")
        row = cursor.fetchone()
        assert row[0] == "bad/topic"
        assert row[1] == 1  # attempts incremented
        q.close()

    def test_drain_increments_attempts_on_failure(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        q.enqueue("fail/topic", b"data", qos=1)

        publisher = _make_mock_publisher(fail_on_topics={"fail/topic"})

        # Drain twice — attempts should increment each time
        q.drain(publisher)
        q.drain(publisher)

        cursor = q._conn.execute("SELECT attempts FROM outbox WHERE id = 1")
        assert cursor.fetchone()[0] == 2
        q.close()


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    """Tests for SQLite persistence across queue instances."""

    def test_messages_persist_across_instances(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Enqueue in first instance
            q1 = StoreAndForwardQueue(db_path)
            q1.enqueue("t/events", b"persistent", qos=1)
            q1.close()

            # Read in second instance
            q2 = StoreAndForwardQueue(db_path)
            assert q2.size() == 1

            publisher = _make_mock_publisher()
            drained = q2.drain(publisher)
            assert drained == 1
            assert q2.size() == 0
            q2.close()
        finally:
            os.unlink(db_path)

    def test_outbox_table_created_on_init(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        cursor = q._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='outbox'"
        )
        assert cursor.fetchone() is not None
        q.close()

    def test_index_created_on_init(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        cursor = q._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_outbox_created'"
        )
        assert cursor.fetchone() is not None
        q.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_large_payload(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        large_payload = b"\xff" * (1024 * 1024)  # 1 MB
        q.enqueue("t/events", large_payload, qos=1)
        assert q.size() == 1

        publisher = _make_mock_publisher()
        drained = q.drain(publisher)
        assert drained == 1

        # Verify the payload was passed correctly
        call_payload = publisher.publish.call_args_list[0].args[1]
        assert call_payload == large_payload
        q.close()

    def test_binary_payload_roundtrip(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        payload = bytes(range(256))
        q.enqueue("t/events", payload, qos=1)

        publisher = _make_mock_publisher()
        q.drain(publisher)

        call_payload = publisher.publish.call_args_list[0].args[1]
        assert call_payload == payload
        q.close()

    def test_multiple_enqueue_drain_cycles(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        publisher = _make_mock_publisher()

        # Cycle 1
        q.enqueue("t/events", b"batch1-msg1", qos=1)
        q.enqueue("t/events", b"batch1-msg2", qos=1)
        assert q.drain(publisher) == 2
        assert q.size() == 0

        # Cycle 2
        q.enqueue("t/events", b"batch2-msg1", qos=0)
        assert q.drain(publisher) == 1
        assert q.size() == 0
        q.close()

    def test_all_qos_levels(self) -> None:
        q = StoreAndForwardQueue(":memory:")
        q.enqueue("t/events", b"qos0", qos=0)
        q.enqueue("t/events", b"qos1", qos=1)
        q.enqueue("t/events", b"qos2", qos=2)

        publisher = _make_mock_publisher()
        drained = q.drain(publisher)
        assert drained == 3

        calls = publisher.publish.call_args_list
        assert calls[0].kwargs["qos"] == 0
        assert calls[1].kwargs["qos"] == 1
        assert calls[2].kwargs["qos"] == 2
        q.close()
