import pytest

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
    assert AlfWorldVisionProgram._select_action('{"action":"open fridge 1"}', actions) == "open fridge 1"


def test_alfworld_select_action_falls_back_to_look() -> None:
    assert AlfWorldVisionProgram._select_action("not a valid command", ["look", "inventory"]) == "look"


def test_alfworld_select_action_handles_empty_model_content() -> None:
    assert AlfWorldVisionProgram._select_action("", ["look", "inventory"]) == "look"


def test_alfworld_select_action_can_extract_candidate_from_reasoning() -> None:
    actions = ["look", "open fridge 1"]
    reasoning = "I should inspect the contents, so use open fridge 1 next."
    assert AlfWorldVisionProgram._select_action(reasoning, actions) == "open fridge 1"


def test_alfworld_select_action_rejects_empty_environment_actions() -> None:
    with pytest.raises(RuntimeError, match="no admissible actions"):
        AlfWorldVisionProgram._select_action("", [])


def test_alfworld_pool_defaults_to_two_workers(monkeypatch) -> None:
    monkeypatch.delenv("ALFWORLD_SERVER_COUNT", raising=False)
    pool = AlfWorldServerPool()
    assert len(pool.servers) == 2


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
