from contextlib import contextmanager
from dataclasses import dataclass

import pytest

from codeevolver_benchmarks import alfworld_vision_program as alfworld_vision_program_module
from codeevolver_benchmarks.alfworld_vision_program import AlfWorldVisionProgram
from codeevolver_benchmarks.alfworld_runtime import AlfWorldServerPool
from codeevolver_benchmarks import _tracing
from codeevolver_benchmarks.common import ModelResponse, strip_code_fence
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_strip_code_fence() -> None:
    assert strip_code_fence("```python\nprint('ok')\n```") == "print('ok')"


def test_model_response_usage_defaults_to_zero() -> None:
    response = ModelResponse(text="ok", reasoning="")
    assert response.total_tokens == 0


def test_alfworld_select_action_prefers_exact_candidate() -> None:
    actions = ["look", "go to fridge 1", "open fridge 1"]
    action, match_method = AlfWorldVisionProgram._select_action(
        '{"action":"open fridge 1"}', actions
    )
    assert action == "open fridge 1"
    assert match_method == "json_exact"


def test_alfworld_select_action_falls_back_to_look() -> None:
    action, match_method = AlfWorldVisionProgram._select_action(
        "not a valid command", ["look", "inventory"]
    )
    assert action == "look"
    assert match_method == "fallback_noop"


def test_alfworld_select_action_handles_empty_model_content() -> None:
    action, match_method = AlfWorldVisionProgram._select_action("", ["look", "inventory"])
    assert action == "look"
    assert match_method == "empty_response"


def test_alfworld_select_action_can_extract_candidate_from_reasoning() -> None:
    actions = ["look", "open fridge 1"]
    reasoning = "I should inspect the contents, so use open fridge 1 next."
    action, match_method = AlfWorldVisionProgram._select_action(reasoning, actions)
    assert action == "open fridge 1"
    assert match_method == "line_substring"


def test_alfworld_select_action_rejects_empty_environment_actions() -> None:
    with pytest.raises(RuntimeError, match="no admissible actions"):
        AlfWorldVisionProgram._select_action("", [])


def test_alfworld_select_action_rejects_negated_action_in_favor_of_final_choice() -> None:
    # "Not go to garbagecan 1" must not win just because it is the longer
    # match: the model explicitly rejected it in favor of the action named
    # later in the same line.
    actions = ["go to garbagecan 1", "open fridge 1", "look"]
    reasoning = "Not go to garbagecan 1; instead open fridge 1"
    action, match_method = AlfWorldVisionProgram._select_action(reasoning, actions)
    assert action == "open fridge 1"
    assert match_method == "line_substring"


def test_alfworld_select_action_uses_last_line_of_multiline_reasoning() -> None:
    # Both the first and last line independently match a *different*
    # admissible action, so this only passes if "last" genuinely outranks
    # "first" -- a mutation swapping that priority would select
    # "go to fridge 1" instead and fail the assertion below.
    actions = ["look", "open fridge 1", "go to fridge 1"]
    reasoning = (
        "go to fridge 1\n"
        "Reconsidering: the image shows the fridge door is already open.\n"
        "open fridge 1"
    )
    action, match_method = AlfWorldVisionProgram._select_action(reasoning, actions)
    assert action == "open fridge 1"
    assert match_method == "line_exact"


def test_alfworld_select_action_extracts_json_embedded_in_prose() -> None:
    actions = ["look", "go to fridge 1", "open fridge 1"]
    text = (
        'Based on the camera image, I will proceed with {"action": "go to fridge 1"} '
        "to check inventory."
    )
    action, match_method = AlfWorldVisionProgram._select_action(text, actions)
    assert action == "go to fridge 1"
    assert match_method == "json_exact"


def test_alfworld_select_action_prefers_last_of_multiple_json_objects() -> None:
    # Regression for the leftmost-JSON bug: reasoning that explores and
    # rejects a candidate before committing must not have the rejected,
    # earlier JSON object win just because it appears first in the text.
    actions = ["look", "go to cabinet 1", "take mug 1 from countertop 1"]
    text = (
        'I considered {"action": "go to cabinet 1"} but the cabinet is closed '
        '... so instead I will do {"action": "take mug 1 from countertop 1"}.'
    )
    action, match_method = AlfWorldVisionProgram._select_action(text, actions)
    assert action == "take mug 1 from countertop 1"
    assert match_method == "json_exact"


def test_alfworld_select_action_skips_malformed_last_json_object() -> None:
    # The last JSON-looking fragment is unparsable (unterminated string); an
    # earlier, valid JSON object must still be found rather than aborting
    # extraction entirely.
    actions = ["look", "go to cabinet 1", "take mug 1 from countertop 1"]
    text = (
        'First I noted {"action": "go to cabinet 1"} as a good option, then in '
        'hindsight wrote {"action": "take mug 1 from countertop 1} which is malformed json.'
    )
    action, match_method = AlfWorldVisionProgram._select_action(text, actions)
    assert action == "go to cabinet 1"
    assert match_method == "json_exact"


def test_alfworld_select_action_reports_action_line_match_method() -> None:
    actions = ["look", "go to fridge 1", "open fridge 1"]
    text = "action: go to fridge 1."
    action, match_method = AlfWorldVisionProgram._select_action(text, actions)
    assert action == "go to fridge 1"
    assert match_method == "action_line_exact"


def test_alfworld_select_action_reports_prefix_match_method() -> None:
    actions = ["look", "open fridge 1"]
    text = "open fridge 1 to check for milk"
    action, match_method = AlfWorldVisionProgram._select_action(text, actions)
    assert action == "open fridge 1"
    assert match_method == "line_prefix"


def test_alfworld_pool_defaults_to_two_workers(monkeypatch) -> None:
    monkeypatch.delenv("ALFWORLD_SERVER_COUNT", raising=False)
    pool = AlfWorldServerPool()
    assert len(pool.servers) == 2


@dataclass
class _StubResponseWithFinishReason:
    """A model response that (unlike ModelResponse) carries finish_reason."""

    text: str
    reasoning: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None


class _StubModel:
    """Replays a fixed queue of responses, one per __call__ step."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)

    def complete(self, messages) -> object:
        return self._responses.pop(0)


class _StubWorld:
    """Replays a fixed queue of post-step states, ignoring the action taken."""

    def __init__(self, step_states: list[dict]) -> None:
        self._step_states = list(step_states)

    def step(self, action: str) -> dict:
        return self._step_states.pop(0)


class _StubAlfworldPool:
    def __init__(self, initial_state: dict, world: _StubWorld) -> None:
        self._initial_state = initial_state
        self._world = world

    @contextmanager
    def acquire(self, game_file: str):
        yield self._world, self._initial_state


def _alfworld_info(*, won: bool, goal_condition: float) -> dict:
    return {
        "admissible_commands": [["look", "go to fridge 1"]],
        "won": [won],
        "goal_condition_success_rate": [goal_condition],
    }


def test_alfworld_program_call_records_match_method_and_parse_failures(monkeypatch) -> None:
    # Drives __call__ end to end against a stubbed model and a stubbed
    # alfworld pool -- no real environment or game file is started. Covers
    # the previously-untested wiring: match_method/finish_reason landing on
    # each trace step, and parse_failures counting exactly the
    # fallback_noop/empty_response steps.
    program = AlfWorldVisionProgram.__new__(AlfWorldVisionProgram)
    program.config = {"max_steps": 3}
    program.model = _StubModel(
        [
            # Step 1: empty text AND reasoning -> "empty_response" parse
            # failure. ModelResponse has no finish_reason attribute at all,
            # which exercises the defensive getattr not crashing when it's
            # absent.
            ModelResponse(text="", reasoning=""),
            # Step 2: a clean JSON match, and this response DOES carry
            # finish_reason -- must be recorded on the trace step.
            _StubResponseWithFinishReason(
                text='{"action": "go to fridge 1"}',
                reasoning="going to the fridge",
                finish_reason="stop",
            ),
            # Step 3: matches no admissible action -> "fallback_noop" parse
            # failure.
            ModelResponse(text="entirely unrelated gibberish", reasoning="entirely unrelated gibberish"),
        ]
    )

    world = _StubWorld(
        [
            {
                "info": _alfworld_info(won=False, goal_condition=0.0),
                "observation": "obs after step 1",
                "frame": "frame-1",
                "done": False,
            },
            {
                "info": _alfworld_info(won=False, goal_condition=0.5),
                "observation": "obs after step 2",
                "frame": "frame-2",
                "done": False,
            },
            {
                "info": _alfworld_info(won=True, goal_condition=1.0),
                "observation": "obs after step 3",
                "frame": "frame-3",
                "done": True,
            },
        ]
    )
    initial_state = {
        "info": _alfworld_info(won=False, goal_condition=0.0),
        "observation": "initial obs",
        "frame": "frame-0",
    }
    monkeypatch.setattr(
        alfworld_vision_program_module,
        "alfworld_pool",
        _StubAlfworldPool(initial_state, world),
    )

    result = program("fake_game_file.tw-pddl")

    assert result["success"] is True
    assert result["goal_condition_success_rate"] == 1.0
    assert len(result["trace"]) == 3
    assert all("match_method" in step for step in result["trace"])

    match_methods = [step["match_method"] for step in result["trace"]]
    assert match_methods == ["empty_response", "json_exact", "fallback_noop"]

    # Exactly the empty_response/fallback_noop steps count as parse failures.
    assert result["parse_failures"] == 2

    # finish_reason recorded when the response carries it, defaults to None
    # (no crash) when the attribute is absent.
    assert result["trace"][0]["finish_reason"] is None
    assert result["trace"][1]["finish_reason"] == "stop"
    assert result["trace"][2]["finish_reason"] is None


def test_traceable_emits_bounded_codeevolver_span(monkeypatch) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_tracing.trace, "get_tracer", provider.get_tracer)

    @_tracing.traceable("tool", name="seed.test", max_attr_chars=5)
    def sample(value: str) -> str:
        return value.upper()

    assert sample("abcdefgh") == "ABCDEFGH"
    span = exporter.get_finished_spans()[0]
    assert span.name == "seed.test"
    assert span.attributes["ce.span_kind"] == "tool"
    assert span.attributes["ce.inputs.value"] == "abcde"
    assert span.attributes["ce.output"] == "ABCDE"
