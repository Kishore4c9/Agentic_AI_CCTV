"""Property-based test for Store-and-Forward Queue message ordering and completeness.

# Feature: agentic-ai-cctv-monitoring, Property 4: Store-and-Forward Queue Preserves All Messages in Order

**Validates: Requirements 3.7, 11.4**

For random sequences of (topic, payload, qos) tuples, drain returns all messages
in FIFO order with no loss or duplication.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_cctv.store_and_forward import StoreAndForwardQueue

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_topic_strategy = st.from_regex(r"[a-zA-Z0-9\-/]+", fullmatch=True).filter(
    lambda s: len(s) >= 1
)

_payload_strategy = st.binary(min_size=1, max_size=256)

_qos_strategy = st.sampled_from([0, 1, 2])

_message_strategy = st.tuples(_topic_strategy, _payload_strategy, _qos_strategy)

_message_list_strategy = st.lists(_message_strategy, min_size=0, max_size=50)


# ---------------------------------------------------------------------------
# Mock publisher that records all publish calls
# ---------------------------------------------------------------------------


@dataclass
class _RecordingPublisher:
    """A mock MQTTPublisher that records all publish calls in order."""

    published: List[Tuple[str, bytes, int, bool]] = field(default_factory=list)

    async def publish(
        self,
        topic: str,
        payload: bytes,
        qos: int = 1,
        retain: bool = False,
    ) -> None:
        self.published.append((topic, payload, qos, retain))


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestStoreAndForwardQueueOrdering:
    """Property 4: Store-and-Forward Queue Preserves All Messages in Order.

    **Validates: Requirements 3.7, 11.4**
    """

    @given(messages=_message_list_strategy)
    @settings(max_examples=20)
    def test_drain_returns_all_messages_in_fifo_order(
        self,
        messages: list,
    ) -> None:
        """For any random sequence of (topic, payload, qos) tuples:

        1. Create a StoreAndForwardQueue with ``:memory:`` db.
        2. Enqueue all messages.
        3. Create a mock publisher that records all publish calls.
        4. Drain the queue.
        5. Assert: drain count equals number of enqueued messages.
        6. Assert: published messages are in the same FIFO order as enqueued.
        7. Assert: queue size is 0 after drain.
        8. Assert: no message loss or duplication.
        """
        # 1. Create queue
        queue = StoreAndForwardQueue(":memory:")

        try:
            # 2. Enqueue all messages
            for topic, payload, qos in messages:
                queue.enqueue(topic, payload, qos)

            # Verify queue size matches enqueued count
            assert queue.size() == len(messages), (
                f"Queue size {queue.size()} != enqueued count {len(messages)}"
            )

            # 3. Create mock publisher
            publisher = _RecordingPublisher()

            # 4. Drain the queue
            drained = queue.drain(publisher)

            # 5. Drain count equals number of enqueued messages
            assert drained == len(messages), (
                f"Drained {drained} messages, expected {len(messages)}"
            )

            # 6. Published messages are in the same FIFO order as enqueued
            assert len(publisher.published) == len(messages), (
                f"Publisher received {len(publisher.published)} messages, "
                f"expected {len(messages)}"
            )

            for i, (topic, payload, qos) in enumerate(messages):
                pub_topic, pub_payload, pub_qos, _pub_retain = publisher.published[i]
                assert pub_topic == topic, (
                    f"Message {i}: expected topic {topic!r}, got {pub_topic!r}"
                )
                assert pub_payload == payload, (
                    f"Message {i}: payload mismatch"
                )
                assert pub_qos == qos, (
                    f"Message {i}: expected qos {qos}, got {pub_qos}"
                )

            # 7. Queue size is 0 after drain
            assert queue.size() == 0, (
                f"Queue size after drain is {queue.size()}, expected 0"
            )

            # 8. No message loss or duplication (already verified by count
            #    and order checks above, but explicitly assert set equality
            #    of payloads for clarity)
            enqueued_topics = [t for t, _, _ in messages]
            published_topics = [t for t, _, _, _ in publisher.published]
            assert enqueued_topics == published_topics, (
                "Message topics differ between enqueued and published"
            )

        finally:
            queue.close()
