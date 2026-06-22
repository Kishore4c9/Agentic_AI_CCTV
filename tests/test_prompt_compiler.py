"""Unit tests for the Prompt Compiler.

Tests prompt-to-ruleset conversion with mocked LLM, scope application
(single camera, camera group, site-wide), rule confirmation flow
(confirm_and_activate persists to RuleStore and reloads ContextFilter),
and error handling (invalid LLM response, LLM timeout/failure).

Requirements: 7.1, 7.2, 7.5
"""

from __future__ import annotations

import json
import pytest

from agentic_cctv.models import (
    CompiledRuleSet,
    PromptScope,
    Rule,
    RuleSet,
    TimeWindow,
)
from agentic_cctv.prompt_compiler import (
    LLMClient,
    PromptCompiler,
    HistoryTestResult,
    _parse_llm_response,
)
from agentic_cctv.rule_store import RuleStore
from agentic_cctv.context_filter import ContextFilter
from agentic_cctv.timeseries_db import TimeSeriesDB


# ---------------------------------------------------------------------------
# Mock LLM Client
# ---------------------------------------------------------------------------


class MockLLMClient:
    """A mock LLM client that returns a pre-configured response."""

    def __init__(self, response: str | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_llm_response(
    rules: list[dict] | None = None,
    explanation: str = "Alert on persons at night",
    confidence: float = 0.9,
) -> str:
    """Build a valid JSON string mimicking an LLM response."""
    if rules is None:
        rules = [
            {
                "rule_id": "rule-001",
                "object_type": "person",
                "min_confidence": 0.8,
                "time_window": {"start": "22:00", "end": "06:00"},
                "zone": None,
                "suppress_if": None,
                "compound": None,
            }
        ]
    return json.dumps(
        {"rules": rules, "explanation": explanation, "confidence": confidence}
    )


def _make_rule_store() -> RuleStore:
    """Create an in-memory RuleStore for testing."""
    return RuleStore(":memory:")


# ---------------------------------------------------------------------------
# Tests: Prompt-to-RuleSet conversion (Req 7.1)
# ---------------------------------------------------------------------------


class TestCompilePrompt:
    """Tests for PromptCompiler.compile()."""

    @pytest.mark.asyncio
    async def test_basic_compile(self) -> None:
        """A valid LLM response produces a CompiledRuleSet with correct fields."""
        response = _make_valid_llm_response()
        client = MockLLMClient(response=response)
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        result = await compiler.compile("Alert on persons at night", scope)

        assert isinstance(result, CompiledRuleSet)
        assert result.original_prompt == "Alert on persons at night"
        assert result.explanation == "Alert on persons at night"
        assert result.confidence == pytest.approx(0.9)
        assert len(result.ruleset.rules) == 1
        assert result.ruleset.rules[0].object_type == "person"
        assert result.ruleset.rules[0].min_confidence == pytest.approx(0.8)
        assert result.ruleset.rules[0].time_window is not None
        assert result.ruleset.rules[0].time_window.start == "22:00"
        assert result.ruleset.rules[0].time_window.end == "06:00"

    @pytest.mark.asyncio
    async def test_compile_multiple_rules(self) -> None:
        """LLM response with multiple rules produces a multi-rule RuleSet."""
        rules = [
            {
                "rule_id": "rule-001",
                "object_type": "person",
                "min_confidence": 0.7,
            },
            {
                "rule_id": "rule-002",
                "object_type": "vehicle",
                "min_confidence": 0.6,
                "suppress_if": {
                    "object_type": "vehicle",
                    "time_window": {"start": "08:00", "end": "18:00"},
                },
            },
        ]
        response = _make_valid_llm_response(rules=rules)
        client = MockLLMClient(response=response)
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        result = await compiler.compile("Alert on people and vehicles", scope)

        assert len(result.ruleset.rules) == 2
        assert result.ruleset.rules[0].object_type == "person"
        assert result.ruleset.rules[1].object_type == "vehicle"
        assert result.ruleset.rules[1].suppress_if is not None
        assert result.ruleset.rules[1].suppress_if.object_type == "vehicle"

    @pytest.mark.asyncio
    async def test_compile_with_compound_condition(self) -> None:
        """LLM response with compound conditions is parsed correctly."""
        rules = [
            {
                "rule_id": "rule-001",
                "compound": {
                    "operator": "or",
                    "conditions": [
                        {"object_type": "person", "min_confidence": 0.8},
                        {"object_type": "vehicle", "min_confidence": 0.7},
                    ],
                },
            }
        ]
        response = _make_valid_llm_response(rules=rules)
        client = MockLLMClient(response=response)
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        result = await compiler.compile("Alert on people or vehicles", scope)

        rule = result.ruleset.rules[0]
        assert rule.compound is not None
        assert rule.compound.operator == "or"
        assert len(rule.compound.conditions) == 2

    @pytest.mark.asyncio
    async def test_compile_passes_prompt_to_llm(self) -> None:
        """The user prompt and scope are forwarded to the LLM."""
        response = _make_valid_llm_response()
        client = MockLLMClient(response=response)
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="site", target_ids=["cam-01", "cam-02"])
        await compiler.compile("Watch for intruders", scope)

        assert len(client.calls) == 1
        _, user_prompt = client.calls[0]
        assert "Watch for intruders" in user_prompt
        assert "site" in user_prompt

    @pytest.mark.asyncio
    async def test_compile_camera_id_from_scope(self) -> None:
        """The compiled ruleset uses the first target_id as camera_id."""
        response = _make_valid_llm_response()
        client = MockLLMClient(response=response)
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="camera", target_ids=["cam-lobby-01"])
        result = await compiler.compile("Alert on persons", scope)

        assert result.ruleset.camera_id == "cam-lobby-01"

    @pytest.mark.asyncio
    async def test_compile_empty_target_ids(self) -> None:
        """When target_ids is empty, camera_id defaults to 'unscoped'."""
        response = _make_valid_llm_response()
        client = MockLLMClient(response=response)
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="site", target_ids=[])
        result = await compiler.compile("Alert on persons", scope)

        assert result.ruleset.camera_id == "unscoped"


# ---------------------------------------------------------------------------
# Tests: Scope application (Req 7.5)
# ---------------------------------------------------------------------------


class TestScopeApplication:
    """Tests for scope-based rule application."""

    @pytest.mark.asyncio
    async def test_single_camera_scope(self) -> None:
        """Scope 'camera' with one target applies to that single camera."""
        response = _make_valid_llm_response()
        client = MockLLMClient(response=response)
        store = _make_rule_store()
        compiler = PromptCompiler(
            llm_client=client, rule_store=store
        )

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        compiled = await compiler.compile("Alert on persons", scope)
        version_ids = await compiler.confirm_and_activate(compiled, scope)

        assert len(version_ids) == 1
        active = store.get_active_ruleset("cam-01")
        assert active is not None
        assert len(active.rules) == 1

    @pytest.mark.asyncio
    async def test_camera_group_scope(self) -> None:
        """Scope 'camera_group' applies rules to all cameras in the group."""
        response = _make_valid_llm_response()
        client = MockLLMClient(response=response)
        store = _make_rule_store()
        compiler = PromptCompiler(
            llm_client=client, rule_store=store
        )

        scope = PromptScope(
            scope_type="camera_group",
            target_ids=["cam-01", "cam-02", "cam-03"],
        )
        compiled = await compiler.compile("Alert on vehicles", scope)
        version_ids = await compiler.confirm_and_activate(compiled, scope)

        assert len(version_ids) == 3
        for cam_id in ["cam-01", "cam-02", "cam-03"]:
            active = store.get_active_ruleset(cam_id)
            assert active is not None
            assert len(active.rules) == 1

    @pytest.mark.asyncio
    async def test_site_scope(self) -> None:
        """Scope 'site' applies rules to all cameras in target_ids."""
        response = _make_valid_llm_response()
        client = MockLLMClient(response=response)
        store = _make_rule_store()
        compiler = PromptCompiler(
            llm_client=client, rule_store=store
        )

        scope = PromptScope(
            scope_type="site",
            target_ids=["cam-a", "cam-b"],
        )
        compiled = await compiler.compile("Alert on animals", scope)
        version_ids = await compiler.confirm_and_activate(compiled, scope)

        assert len(version_ids) == 2
        for cam_id in ["cam-a", "cam-b"]:
            active = store.get_active_ruleset(cam_id)
            assert active is not None


# ---------------------------------------------------------------------------
# Tests: Confirmation flow (Req 7.2, 7.3)
# ---------------------------------------------------------------------------


class TestConfirmAndActivate:
    """Tests for confirm_and_activate persisting to RuleStore and reloading ContextFilter."""

    @pytest.mark.asyncio
    async def test_persists_to_rule_store(self) -> None:
        """confirm_and_activate saves the ruleset to the RuleStore."""
        response = _make_valid_llm_response()
        client = MockLLMClient(response=response)
        store = _make_rule_store()
        compiler = PromptCompiler(llm_client=client, rule_store=store)

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        compiled = await compiler.compile("Alert on persons", scope)
        version_ids = await compiler.confirm_and_activate(compiled, scope)

        assert len(version_ids) == 1
        active = store.get_active_ruleset("cam-01")
        assert active is not None
        assert active.version_id == version_ids[0]

    @pytest.mark.asyncio
    async def test_reloads_context_filter(self) -> None:
        """confirm_and_activate triggers ContextFilter.reload_rules for each camera."""
        response = _make_valid_llm_response()
        client = MockLLMClient(response=response)
        store = _make_rule_store()
        context_filter = ContextFilter(rule_store=store)

        compiler = PromptCompiler(
            llm_client=client,
            rule_store=store,
            context_filter=context_filter,
        )

        scope = PromptScope(
            scope_type="camera_group", target_ids=["cam-01", "cam-02"]
        )
        compiled = await compiler.compile("Alert on persons", scope)

        # Before activation, no active rulesets
        assert store.get_active_ruleset("cam-01") is None
        assert store.get_active_ruleset("cam-02") is None

        await compiler.confirm_and_activate(compiled, scope)

        # After activation, both cameras have active rulesets
        assert store.get_active_ruleset("cam-01") is not None
        assert store.get_active_ruleset("cam-02") is not None

    @pytest.mark.asyncio
    async def test_no_rule_store_raises(self) -> None:
        """confirm_and_activate raises RuntimeError when no RuleStore is configured."""
        response = _make_valid_llm_response()
        client = MockLLMClient(response=response)
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        compiled = await compiler.compile("Alert on persons", scope)

        with pytest.raises(RuntimeError, match="no RuleStore configured"):
            await compiler.confirm_and_activate(compiled, scope)

    @pytest.mark.asyncio
    async def test_each_camera_gets_unique_version_id(self) -> None:
        """Each camera in a group scope gets a unique version_id."""
        response = _make_valid_llm_response()
        client = MockLLMClient(response=response)
        store = _make_rule_store()
        compiler = PromptCompiler(llm_client=client, rule_store=store)

        scope = PromptScope(
            scope_type="camera_group", target_ids=["cam-01", "cam-02"]
        )
        compiled = await compiler.compile("Alert on persons", scope)
        version_ids = await compiler.confirm_and_activate(compiled, scope)

        assert len(version_ids) == 2
        assert version_ids[0] != version_ids[1]


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in PromptCompiler."""

    @pytest.mark.asyncio
    async def test_llm_timeout(self) -> None:
        """LLM timeout raises RuntimeError."""
        client = MockLLMClient(error=TimeoutError("LLM timed out"))
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        with pytest.raises(RuntimeError, match="LLM call failed"):
            await compiler.compile("Alert on persons", scope)

    @pytest.mark.asyncio
    async def test_llm_network_error(self) -> None:
        """LLM network error raises RuntimeError."""
        client = MockLLMClient(error=ConnectionError("Network unreachable"))
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        with pytest.raises(RuntimeError, match="LLM call failed"):
            await compiler.compile("Alert on persons", scope)

    @pytest.mark.asyncio
    async def test_invalid_json_response(self) -> None:
        """LLM returning non-JSON raises ValueError."""
        client = MockLLMClient(response="This is not JSON at all")
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        with pytest.raises(ValueError, match="not valid JSON"):
            await compiler.compile("Alert on persons", scope)

    @pytest.mark.asyncio
    async def test_missing_rules_field(self) -> None:
        """LLM response missing 'rules' field raises ValueError."""
        client = MockLLMClient(
            response=json.dumps({"explanation": "test", "confidence": 0.9})
        )
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        with pytest.raises(ValueError, match="missing required 'rules' field"):
            await compiler.compile("Alert on persons", scope)

    @pytest.mark.asyncio
    async def test_rules_not_a_list(self) -> None:
        """LLM response with 'rules' as non-list raises ValueError."""
        client = MockLLMClient(
            response=json.dumps({"rules": "not a list"})
        )
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        with pytest.raises(ValueError, match="'rules' field must be a list"):
            await compiler.compile("Alert on persons", scope)

    @pytest.mark.asyncio
    async def test_invalid_rule_in_list(self) -> None:
        """LLM response with an invalid rule entry raises ValueError."""
        client = MockLLMClient(
            response=json.dumps({"rules": ["not a dict"]})
        )
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        with pytest.raises(ValueError, match="Invalid rule at index 0"):
            await compiler.compile("Alert on persons", scope)

    @pytest.mark.asyncio
    async def test_markdown_fenced_json(self) -> None:
        """LLM response wrapped in markdown code fences is handled correctly."""
        inner = _make_valid_llm_response()
        fenced = f"```json\n{inner}\n```"
        client = MockLLMClient(response=fenced)
        compiler = PromptCompiler(llm_client=client)

        scope = PromptScope(scope_type="camera", target_ids=["cam-01"])
        result = await compiler.compile("Alert on persons", scope)

        assert len(result.ruleset.rules) == 1
        assert result.ruleset.rules[0].object_type == "person"


# ---------------------------------------------------------------------------
# Tests: _parse_llm_response edge cases
# ---------------------------------------------------------------------------


class TestParseLLMResponse:
    """Tests for the _parse_llm_response helper."""

    def test_defaults_for_missing_optional_fields(self) -> None:
        """Missing explanation and confidence get sensible defaults."""
        raw = json.dumps({"rules": [{"rule_id": "rule-001"}]})
        result = _parse_llm_response(raw)

        assert result["explanation"] == "No explanation provided"
        assert result["confidence"] == pytest.approx(0.5)

    def test_confidence_clamped_to_0_1(self) -> None:
        """Confidence values outside [0, 1] are clamped."""
        raw = json.dumps(
            {"rules": [], "explanation": "test", "confidence": 1.5}
        )
        result = _parse_llm_response(raw)
        assert result["confidence"] == pytest.approx(1.0)

        raw = json.dumps(
            {"rules": [], "explanation": "test", "confidence": -0.5}
        )
        result = _parse_llm_response(raw)
        assert result["confidence"] == pytest.approx(0.0)

    def test_non_dict_response_raises(self) -> None:
        """A JSON array (not object) raises ValueError."""
        with pytest.raises(ValueError, match="must be a JSON object"):
            _parse_llm_response("[1, 2, 3]")

    def test_rule_with_zone(self) -> None:
        """A rule with a zone polygon is parsed correctly."""
        raw = json.dumps({
            "rules": [
                {
                    "rule_id": "rule-zone",
                    "zone": {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
                }
            ],
            "explanation": "Zone rule",
            "confidence": 0.8,
        })
        result = _parse_llm_response(raw)
        rule = result["rules"][0]
        assert rule.zone is not None
        assert len(rule.zone.polygon) == 4


# ---------------------------------------------------------------------------
# Tests: test_against_history stub
# ---------------------------------------------------------------------------


class TestTestAgainstHistory:
    """Tests for the test_against_history stub."""

    @pytest.mark.asyncio
    async def test_stub_returns_test_result(self) -> None:
        """The stub returns a TestResult with zero counts when DB is empty."""
        client = MockLLMClient(response=_make_valid_llm_response())
        db = TimeSeriesDB(":memory:")
        compiler = PromptCompiler(llm_client=client, timeseries_db=db)

        ruleset = RuleSet(
            version_id="rs-test",
            camera_id="cam-01",
            rules=[],
        )
        result = await compiler.test_against_history(ruleset, "cam-01", days=7)

        assert isinstance(result, HistoryTestResult)
        assert result.camera_id == "cam-01"
        assert result.days_tested == 7
        assert result.total_events == 0
        assert result.matched_events == 0
        assert result.expected_alert_rate == pytest.approx(0.0)
