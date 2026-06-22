"""A2A Communication Hub for the Agentic AI CCTV Monitoring Framework.

Implements a lightweight, in-process Agent-to-Agent (A2A) communication hub
for v1 single-machine mode.  The hub maintains a thread-safe in-memory
message bus that allows registered agents (identified by ``agent_id``,
typically the ``camera_id`` they manage) to exchange messages for
multi-camera coordination.

The interface is designed to be compatible with a future ``a2a-sdk``
integration for distributed multi-machine deployments — the local hub
can be swapped for a remote A2A client without changing the tool layer.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class A2AMessage:
    """A single inter-agent message.

    Attributes
    ----------
    message_id:
        Unique identifier for this message (UUID).
    from_agent_id:
        The agent that sent this message.
    to_agent_id:
        The target agent, or ``None`` for broadcast messages.
    message_data:
        Arbitrary message payload (scene summary, coordination data, etc.).
    timestamp:
        When this message was created.
    """

    message_id: str
    from_agent_id: str
    to_agent_id: Optional[str]
    message_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class A2ACommHub:
    """In-process A2A communication hub for inter-agent message exchange.

    Maintains a thread-safe in-memory message bus where each registered
    agent has its own bounded message queue.  For v1 single-machine mode
    this runs locally; the interface is designed so it can be replaced by
    a remote ``a2a-sdk`` client in future phases.

    All public methods are thread-safe.

    Parameters
    ----------
    max_queue_size:
        Maximum number of pending messages per agent.  When the queue is
        full, the oldest message is discarded to make room.  Defaults to
        1000.
    """

    def __init__(self, max_queue_size: int = 1000) -> None:
        self._max_queue_size = max_queue_size
        self._agents: Set[str] = set()
        self._queues: Dict[str, Deque[A2AMessage]] = {}
        self._lock = threading.Lock()
        logger.info(
            "A2ACommHub initialised (in-process mode, max_queue_size=%d)",
            max_queue_size,
        )

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str) -> None:
        """Register an agent in the hub.

        If the agent is already registered this is a no-op.

        Parameters
        ----------
        agent_id:
            Unique identifier for the agent (typically the camera_id).
        """
        with self._lock:
            if agent_id not in self._agents:
                self._agents.add(agent_id)
                self._queues[agent_id] = deque(maxlen=self._max_queue_size)
                logger.debug("A2ACommHub: registered agent %s", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the hub and discard its pending messages.

        If the agent is not registered this is a no-op.

        Parameters
        ----------
        agent_id:
            The agent identifier to remove.
        """
        with self._lock:
            self._agents.discard(agent_id)
            self._queues.pop(agent_id, None)
            logger.debug("A2ACommHub: unregistered agent %s", agent_id)

    def list_agents(self) -> List[str]:
        """Return a sorted list of all registered agent IDs.

        Returns
        -------
        List[str]
            Sorted list of registered agent identifiers.
        """
        with self._lock:
            return sorted(self._agents)

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send_message(
        self,
        from_agent_id: str,
        to_agent_id: str,
        message_data: Dict[str, Any],
    ) -> A2AMessage:
        """Send a message from one agent to another.

        The target agent must be registered; if not, the message is
        silently dropped and a warning is logged.

        Parameters
        ----------
        from_agent_id:
            The sending agent's identifier.
        to_agent_id:
            The target agent's identifier.
        message_data:
            Arbitrary payload to deliver.

        Returns
        -------
        A2AMessage
            The constructed message (regardless of delivery success).
        """
        msg = A2AMessage(
            message_id=uuid.uuid4().hex,
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            message_data=message_data,
        )
        with self._lock:
            queue = self._queues.get(to_agent_id)
            if queue is not None:
                queue.append(msg)
                logger.debug(
                    "A2ACommHub: message %s from %s → %s",
                    msg.message_id,
                    from_agent_id,
                    to_agent_id,
                )
            else:
                logger.warning(
                    "A2ACommHub: target agent %s not registered; "
                    "message %s dropped",
                    to_agent_id,
                    msg.message_id,
                )
        return msg

    def broadcast_message(
        self,
        from_agent_id: str,
        message_data: Dict[str, Any],
    ) -> A2AMessage:
        """Broadcast a message to all registered agents except the sender.

        Parameters
        ----------
        from_agent_id:
            The sending agent's identifier.
        message_data:
            Arbitrary payload to deliver.

        Returns
        -------
        A2AMessage
            The constructed broadcast message.
        """
        msg = A2AMessage(
            message_id=uuid.uuid4().hex,
            from_agent_id=from_agent_id,
            to_agent_id=None,
            message_data=message_data,
        )
        with self._lock:
            for agent_id, queue in self._queues.items():
                if agent_id != from_agent_id:
                    queue.append(msg)
            recipient_count = sum(
                1 for a in self._agents if a != from_agent_id
            )
        logger.debug(
            "A2ACommHub: broadcast %s from %s to %d agents",
            msg.message_id,
            from_agent_id,
            recipient_count,
        )
        return msg

    def receive_messages(self, agent_id: str) -> List[A2AMessage]:
        """Retrieve and clear all pending messages for an agent.

        Parameters
        ----------
        agent_id:
            The agent whose messages to retrieve.

        Returns
        -------
        List[A2AMessage]
            All pending messages (oldest first).  The agent's queue is
            cleared after retrieval.  Returns an empty list if the agent
            is not registered.
        """
        with self._lock:
            queue = self._queues.get(agent_id)
            if queue is None:
                return []
            messages = list(queue)
            queue.clear()
        logger.debug(
            "A2ACommHub: agent %s received %d messages",
            agent_id,
            len(messages),
        )
        return messages
