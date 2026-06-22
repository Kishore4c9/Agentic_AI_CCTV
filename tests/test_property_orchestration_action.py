# Feature: agentic-ai-cctv-monitoring, Property 7: Orchestration Action Output Invariant
"""Property test: Orchestration Action Output Invariant.

For random ``SceneUnderstanding`` inputs, the agent's action decision is
always one of ``{"alert", "log", "summarise", "escalate"}``.

**Validates: Requirements 6.1**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_cctv.models import IdentifiedObject, SceneUnderstanding
from agentic_cctv.orchestration_agent import VALID_ACTIONS, OrchestrationAgent


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_THREAT_LEVELS = st.sampled_from(["none", "low", "medium", "high", "critical"])
_RECOMMENDED_ACTIONS = st.sampled_from(["alert", "log", "summarise", "escalate"])

# Also test with arbitrary strings to ensure the agent never returns an
# invalid action even when the SceneUnderstanding contains unexpected values.
_ANY_THREAT_LEVEL = st.one_of(
    _THREAT_LEVELS,
    st.text(min_size=0, max_size=20),
)
_ANY_RECOMMENDED_ACTION = st.one_of(
    _RECOMMENDED_ACTIONS,
    st.text(min_size=0, max_size=20),
)

_identified_object = st.builds(
    IdentifiedObject,
    type=st.text(min_size=1, max_size=20),
    action=st.text(min_size=1, max_size=20),
    location=st.text(min_size=1, max_size=30),
)

_scene_understanding = st.builds(
    SceneUnderstanding,
    event_id=st.uuids().map(str),
    scene_description=st.text(min_size=0, max_size=200),
    threat_level=_ANY_THREAT_LEVEL,
    objects_identified=st.lists(_identified_object, min_size=0, max_size=5),
    recommended_action=_ANY_RECOMMENDED_ACTION,
    confidence=st.floats(min_value=0.0, max_value=1.0),
    raw_response=st.just({}),
    embedding=st.none(),
)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=20)
@given(scene=_scene_understanding)
def test_action_decision_always_in_valid_set(scene: SceneUnderstanding) -> None:
    """Property 7: For random SceneUnderstanding inputs, the agent's action
    decision is always one of {"alert", "log", "summarise", "escalate"}.

    **Validates: Requirements 6.1**
    """
    agent = OrchestrationAgent()
    action = agent.decide_action(scene)
    assert action in VALID_ACTIONS, (
        f"decide_action returned '{action}' which is not in {VALID_ACTIONS}; "
        f"threat_level='{scene.threat_level}', "
        f"recommended_action='{scene.recommended_action}'"
    )
