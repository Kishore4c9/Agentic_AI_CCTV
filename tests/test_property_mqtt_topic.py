"""Property-based test for MQTT Topic Hierarchy Formatting.

# Feature: agentic-ai-cctv-monitoring, Property 3: MQTT Topic Hierarchy Formatting

**Validates: Requirements 3.4**

For random alphanumeric+hyphen tenant/site/camera IDs and valid suffix,
the topic matches ``{tenant}/{site}/{camera}/{suffix}`` with no empty segments.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_cctv.mqtt_client import build_topic

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Alphanumeric + hyphen identifiers (at least 1 character)
_id_strategy = st.from_regex(r"[a-zA-Z0-9\-]+", fullmatch=True).filter(
    lambda s: len(s) >= 1
)

_suffix_strategy = st.sampled_from(["events", "alerts", "health"])


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestMQTTTopicHierarchyFormatting:
    """Property 3: MQTT Topic Hierarchy Formatting.

    **Validates: Requirements 3.4**
    """

    @given(
        tenant_id=_id_strategy,
        site_id=_id_strategy,
        camera_id=_id_strategy,
        suffix=_suffix_strategy,
    )
    @settings(max_examples=20)
    def test_topic_matches_hierarchy_with_no_empty_segments(
        self,
        tenant_id: str,
        site_id: str,
        camera_id: str,
        suffix: str,
    ) -> None:
        """For any random alphanumeric+hyphen tenant_id, site_id, camera_id
        and a valid suffix, ``build_topic()`` returns a topic that:

        1. Matches ``{tenant_id}/{site_id}/{camera_id}/{suffix}`` exactly.
        2. Has exactly 4 segments when split by ``/``.
        3. Contains no empty segments.
        """
        topic = build_topic(tenant_id, site_id, camera_id, suffix)

        # 1. Result matches the expected format
        expected = f"{tenant_id}/{site_id}/{camera_id}/{suffix}"
        assert topic == expected, (
            f"Expected topic {expected!r}, got {topic!r}"
        )

        # 2. Exactly 4 segments
        segments = topic.split("/")
        assert len(segments) == 4, (
            f"Expected 4 segments, got {len(segments)}: {segments}"
        )

        # 3. No segment is empty
        for i, segment in enumerate(segments):
            assert len(segment) > 0, (
                f"Segment {i} is empty in topic {topic!r}"
            )
