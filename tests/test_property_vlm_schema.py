"""Property test for VLM Response Schema Validation.

# Feature: agentic-ai-cctv-monitoring, Property 6: VLM Response Schema Validation

**Validates: Requirements 5.5**

For random dicts with valid/invalid field combinations, the validator returns
valid iff all required fields are present with correct types:
- ``scene_description`` as string
- ``threat_level`` in ["none", "low", "medium", "high", "critical"]
- ``objects_identified`` as list
- ``recommended_action`` in ["alert", "log", "summarise", "escalate"]
- ``confidence`` as float in [0, 1]
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agentic_cctv.vlm_reasoner import validate_vlm_response


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

VALID_THREAT_LEVELS = ["none", "low", "medium", "high", "critical"]
VALID_ACTIONS = ["alert", "log", "summarise", "escalate"]


def valid_vlm_response_strategy():
    """Strategy that generates valid VLM response dicts."""
    return st.fixed_dictionaries(
        {
            "scene_description": st.text(min_size=0, max_size=200),
            "threat_level": st.sampled_from(VALID_THREAT_LEVELS),
            "objects_identified": st.lists(
                st.fixed_dictionaries(
                    {
                        "type": st.text(min_size=1, max_size=20),
                        "action": st.text(min_size=1, max_size=20),
                        "location": st.text(min_size=1, max_size=20),
                    }
                ),
                max_size=5,
            ),
            "recommended_action": st.sampled_from(VALID_ACTIONS),
            "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        }
    )


# Strategy for arbitrary values that are NOT valid for specific fields
_non_string = st.one_of(
    st.integers(),
    st.floats(allow_nan=False),
    st.lists(st.integers(), max_size=3),
    st.none(),
    st.booleans(),
)

_invalid_threat_level = st.text(min_size=1, max_size=30).filter(
    lambda s: s not in VALID_THREAT_LEVELS
)

_non_list = st.one_of(
    st.text(max_size=20),
    st.integers(),
    st.floats(allow_nan=False),
    st.none(),
    st.booleans(),
)

_invalid_action = st.text(min_size=1, max_size=30).filter(
    lambda s: s not in VALID_ACTIONS
)

_invalid_confidence = st.one_of(
    st.floats(max_value=-0.001, allow_nan=False),
    st.floats(min_value=1.001, allow_nan=False),
    st.text(max_size=10),
    st.none(),
    st.booleans(),
    st.lists(st.integers(), max_size=2),
)


# ---------------------------------------------------------------------------
# Property 6: VLM Response Schema Validation
# ---------------------------------------------------------------------------


@given(response=valid_vlm_response_strategy())
@settings(max_examples=20)
def test_valid_response_passes_validation(response: dict) -> None:
    """A response with all required fields and correct types is valid."""
    assert validate_vlm_response(response) is True


@given(
    base=valid_vlm_response_strategy(),
    bad_scene_desc=_non_string,
)
@settings(max_examples=20)
def test_invalid_scene_description_fails(base: dict, bad_scene_desc) -> None:
    """Replacing scene_description with a non-string makes the response invalid."""
    response = {**base, "scene_description": bad_scene_desc}
    assert validate_vlm_response(response) is False


@given(
    base=valid_vlm_response_strategy(),
    bad_threat=_invalid_threat_level,
)
@settings(max_examples=20)
def test_invalid_threat_level_fails(base: dict, bad_threat: str) -> None:
    """A threat_level not in the allowed set makes the response invalid."""
    response = {**base, "threat_level": bad_threat}
    assert validate_vlm_response(response) is False


@given(
    base=valid_vlm_response_strategy(),
    bad_objects=_non_list,
)
@settings(max_examples=20)
def test_invalid_objects_identified_fails(base: dict, bad_objects) -> None:
    """Replacing objects_identified with a non-list makes the response invalid."""
    response = {**base, "objects_identified": bad_objects}
    assert validate_vlm_response(response) is False


@given(
    base=valid_vlm_response_strategy(),
    bad_action=_invalid_action,
)
@settings(max_examples=20)
def test_invalid_recommended_action_fails(base: dict, bad_action: str) -> None:
    """A recommended_action not in the allowed set makes the response invalid."""
    response = {**base, "recommended_action": bad_action}
    assert validate_vlm_response(response) is False


@given(
    base=valid_vlm_response_strategy(),
    bad_confidence=_invalid_confidence,
)
@settings(max_examples=20)
def test_invalid_confidence_fails(base: dict, bad_confidence) -> None:
    """A confidence that is not a float in [0, 1] makes the response invalid."""
    response = {**base, "confidence": bad_confidence}
    assert validate_vlm_response(response) is False


@given(
    base=valid_vlm_response_strategy(),
    field_to_remove=st.sampled_from(
        [
            "scene_description",
            "threat_level",
            "objects_identified",
            "recommended_action",
            "confidence",
        ]
    ),
)
@settings(max_examples=20)
def test_missing_required_field_fails(base: dict, field_to_remove: str) -> None:
    """Removing any required field makes the response invalid."""
    response = {k: v for k, v in base.items() if k != field_to_remove}
    assert validate_vlm_response(response) is False
