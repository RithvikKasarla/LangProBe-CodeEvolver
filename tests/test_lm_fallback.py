"""Provider fallback for the task LM: routing, divert rules, circuit breaker.

No network: every test drives `OpenAICompatibleModel` with fake clients whose
`chat.completions.create` is a scripted function.
"""

from __future__ import annotations

import httpx
import openai
import pytest

from codeevolver_benchmarks import common
from codeevolver_benchmarks.common import (
    ModelResponse,
    OpenAICompatibleModel,
    ProviderBreaker,
    Route,
    breaker_for,
    reset_breakers,
    should_fallback,
)


ROUTES = {
    "gmi": {
        "api_key_env": "GMI_API_KEY",
        "base_url": "https://api.gmi-serving.com/v1",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    },
    "deepinfra": {
        "api_key_env": "DEEPINFRA_API_KEY",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
    },
}

CONFIG = {
    "provider": "gmi",
    "provider_preference": ["gmi", "deepseek", "deepinfra"],
    "routes": ROUTES,
    "reasoning_effort": "high",
    "max_tokens": 4096,
    "temperature": 0,
    "seed": 42,
    "max_steps": 40,
}

LEGACY_CONFIG = {
    "provider": "openrouter",
    "api_key_env": "OPENROUTER_API_KEY",
    "base_url": "https://openrouter.ai/api/v1",
    "model": "qwen/qwen3.6-35b-a3b",
    "reasoning_effort": "medium",
    "max_tokens": 8192,
    "max_steps": 40,
}


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content
        self.reasoning_content = "thought"


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Usage:
    prompt_tokens = 3
    completion_tokens = 4
    total_tokens = 7


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage()


def api_error(status: int, message: str = "boom") -> openai.APIStatusError:
    """A real openai error object for `status`, built without a network call."""
    request = httpx.Request("POST", "https://example.invalid/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": message}})
    return openai.APIStatusError(message, response=response, body=None)


def bad_request(message: str) -> openai.BadRequestError:
    request = httpx.Request("POST", "https://example.invalid/chat/completions")
    response = httpx.Response(400, request=request, json={"error": {"message": message}})
    return openai.BadRequestError(message, response=response, body=None)


class _FakeClient:
    """Stands in for `openai.OpenAI`, recording every call it is given."""

    def __init__(self, base_url: str, **_: object) -> None:
        self.base_url = base_url
        self.calls: list[dict[str, object]] = []
        self.behaviour = None  # set by the fixture

        outer = self

        class _Completions:
            def create(self, **kwargs: object) -> _Response:
                outer.calls.append(kwargs)
                return outer.behaviour(len(outer.calls), kwargs)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


@pytest.fixture
def fake_openai(monkeypatch):
    """Replace `OpenAI` with fakes, one per base_url, and clear breaker state."""
    reset_breakers()
    clients: dict[str, _FakeClient] = {}

    def factory(*, base_url: str, **kwargs: object) -> _FakeClient:
        client = clients.get(base_url)
        if client is None:
            client = _FakeClient(base_url, **kwargs)
            client.behaviour = lambda _n, _kw: _Response("ok")
            clients[base_url] = client
        return client

    monkeypatch.setattr(common, "OpenAI", factory)
    for name in ("GMI_API_KEY", "DEEPSEEK_API_KEY", "DEEPINFRA_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(name, f"test-{name}")
    for name in (
        "LM_PROVIDER",
        "LM_FALLBACK",
        "LM_BREAKER",
        "LM_BREAKER_FAILURES",
        "LM_BREAKER_COOLDOWN",
    ):
        monkeypatch.delenv(name, raising=False)
    yield clients
    reset_breakers()


@pytest.fixture
def model(monkeypatch, fake_openai):
    monkeypatch.setattr(common, "load_runtime_config", lambda _name: dict(CONFIG))
    return lambda: OpenAICompatibleModel("appworld")


def by_provider(clients: dict[str, _FakeClient], provider: str) -> _FakeClient:
    return clients[ROUTES[provider]["base_url"]]


# --- routing ---------------------------------------------------------------


def test_default_primary_and_cover_come_from_the_preference_order(model):
    lm = model()
    assert lm.route.provider == "gmi"
    assert lm.fallback_route.provider == "deepseek"


def test_every_route_serves_the_same_model_family(model):
    lm = model()
    assert lm.route.model == "deepseek-ai/DeepSeek-V4-Flash"
    assert lm.fallback_route.model == "deepseek-v4-flash"


def test_lm_provider_env_repoints_the_primary(monkeypatch, model):
    monkeypatch.setenv("LM_PROVIDER", "deepseek")
    lm = model()
    assert lm.route.provider == "deepseek"
    assert lm.fallback_route.provider == "deepinfra"


def test_cover_wraps_around_the_preference_order(monkeypatch, model):
    monkeypatch.setenv("LM_PROVIDER", "deepinfra")
    assert model().fallback_route.provider == "gmi"


def test_lm_fallback_can_name_an_explicit_cover(monkeypatch, model):
    monkeypatch.setenv("LM_FALLBACK", "deepinfra")
    assert model().fallback_route.provider == "deepinfra"


def test_lm_fallback_off_disarms_the_divert(monkeypatch, model):
    monkeypatch.setenv("LM_FALLBACK", "0")
    assert model().fallback_route is None


def test_unknown_provider_fails_loudly(monkeypatch, model):
    monkeypatch.setenv("LM_PROVIDER", "typo")
    with pytest.raises(RuntimeError, match="typo"):
        model()


def test_cover_equal_to_primary_fails_loudly(monkeypatch, model):
    monkeypatch.setenv("LM_FALLBACK", "gmi")
    with pytest.raises(RuntimeError, match="also the primary"):
        model()


def test_unparseable_boolean_fails_loudly(monkeypatch, model):
    monkeypatch.setenv("LM_BREAKER", "maybe")
    with pytest.raises(RuntimeError, match="boolean"):
        model()


def test_missing_primary_key_still_raises(monkeypatch, model):
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GMI_API_KEY"):
        model()


def test_missing_cover_key_degrades_to_no_cover(monkeypatch, model):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert model().fallback_route is None


def test_single_route_config_is_unchanged(monkeypatch, fake_openai):
    monkeypatch.setattr(common, "load_runtime_config", lambda _n: dict(LEGACY_CONFIG))
    lm = OpenAICompatibleModel("alfworld_vision")
    assert lm.route.provider == "openrouter"
    assert lm.fallback_route is None


# --- retry budget (matched to the cover) -----------------------------------


def test_primary_gets_one_retry_when_a_cover_exists(monkeypatch, model):
    captured: list[int] = []
    real = common.OpenAI
    monkeypatch.setattr(
        common,
        "OpenAI",
        lambda **kw: (captured.append(kw["max_retries"]), real(**kw))[1],
    )
    model()
    assert captured == [1, 1]


def test_full_retry_budget_when_there_is_no_cover(monkeypatch, model):
    monkeypatch.setenv("LM_FALLBACK", "0")
    captured: list[int] = []
    real = common.OpenAI
    monkeypatch.setattr(
        common,
        "OpenAI",
        lambda **kw: (captured.append(kw["max_retries"]), real(**kw))[1],
    )
    model()
    assert captured == [5]


# --- divert rules ----------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        api_error(402, "Insufficient balance"),
        api_error(401),
        api_error(403),
        api_error(429),
        api_error(500),
        api_error(503),
        openai.APIConnectionError(request=httpx.Request("POST", "https://x.invalid")),
        bad_request("provider is on fire"),
    ],
)
def test_provider_faults_divert(exc):
    assert should_fallback(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        bad_request("This model's maximum context length is 128000 tokens"),
        bad_request("context_length_exceeded"),
        bad_request("Unsupported parameter: 'seed'"),
        ValueError("the program itself is wrong"),
        KeyError("missing"),
    ],
)
def test_request_faults_and_program_errors_do_not_divert(exc):
    assert should_fallback(exc) is False


def test_a_failed_call_is_reissued_on_the_cover(model, fake_openai):
    lm = model()
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        api_error(402, "Insufficient balance")
    )
    assert lm.complete([{"role": "user", "content": "hi"}]).text == "ok"
    assert len(by_provider(fake_openai, "deepseek").calls) == 1


def test_the_cover_gets_the_identical_request_and_its_own_model_id(model, fake_openai):
    lm = model()
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        api_error(500)
    )
    messages = [{"role": "user", "content": "hi"}]
    lm.complete(messages)
    primary = by_provider(fake_openai, "gmi").calls[0]
    cover = by_provider(fake_openai, "deepseek").calls[0]
    assert primary["messages"] == cover["messages"] == messages
    for key in ("reasoning_effort", "max_tokens", "temperature", "seed"):
        assert primary[key] == cover[key]
    assert primary["model"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert cover["model"] == "deepseek-v4-flash"


def test_a_program_error_is_raised_not_diverted(model, fake_openai):
    lm = model()
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        ValueError("bad program")
    )
    with pytest.raises(ValueError):
        lm.complete([{"role": "user", "content": "hi"}])
    assert by_provider(fake_openai, "deepseek").calls == []


def test_context_window_error_is_raised_not_diverted(model, fake_openai):
    lm = model()
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        bad_request("maximum context length is 128000 tokens")
    )
    with pytest.raises(openai.BadRequestError):
        lm.complete([{"role": "user", "content": "hi"}])
    assert by_provider(fake_openai, "deepseek").calls == []


def test_cover_failure_propagates(model, fake_openai):
    lm = model()
    for provider in ("gmi", "deepseek"):
        by_provider(fake_openai, provider).behaviour = lambda _n, _kw: (
            _ for _ in ()
        ).throw(api_error(500))
    with pytest.raises(openai.APIStatusError):
        lm.complete([{"role": "user", "content": "hi"}])


def test_no_divert_when_fallback_is_disarmed(monkeypatch, model, fake_openai):
    monkeypatch.setenv("LM_FALLBACK", "0")
    lm = model()
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        api_error(500)
    )
    with pytest.raises(openai.APIStatusError):
        lm.complete([{"role": "user", "content": "hi"}])


def test_response_parsing_is_unchanged(model, fake_openai):
    result = model().complete([{"role": "user", "content": "hi"}])
    assert isinstance(result, ModelResponse)
    assert (result.text, result.reasoning, result.total_tokens) == ("ok", "thought", 7)


# --- circuit breaker -------------------------------------------------------


def test_a_stray_failure_followed_by_success_does_not_open_the_breaker(
    model, fake_openai
):
    lm = model()
    gmi = by_provider(fake_openai, "gmi")

    def flaky(n, _kw):
        if n == 1:
            raise api_error(402)
        return _Response("ok")

    gmi.behaviour = flaky
    lm.complete([{"role": "user", "content": "hi"}])
    lm.complete([{"role": "user", "content": "hi"}])
    assert breaker_for("gmi").state == "closed"


def test_sustained_outage_stops_costing_requests(model, fake_openai):
    lm = model()
    gmi = by_provider(fake_openai, "gmi")
    gmi.behaviour = lambda _n, _kw: (_ for _ in ()).throw(api_error(402))
    for _ in range(10):
        lm.complete([{"role": "user", "content": "hi"}])
    # Three attempts open it; the remaining seven calls never touch gmi at all.
    assert len(gmi.calls) == 3
    assert breaker_for("gmi").state == "open"
    assert len(by_provider(fake_openai, "deepseek").calls) == 10


def test_probe_after_cooldown_closes_a_recovered_provider(monkeypatch, model, fake_openai):
    monkeypatch.setenv("LM_BREAKER_COOLDOWN", "0")
    lm = model()
    gmi = by_provider(fake_openai, "gmi")
    gmi.behaviour = lambda n, _kw: (_ for _ in ()).throw(api_error(500)) if n <= 3 else _Response("ok")
    for _ in range(3):
        lm.complete([{"role": "user", "content": "hi"}])
    assert breaker_for("gmi").state == "open"
    lm.complete([{"role": "user", "content": "hi"}])  # the probe succeeds
    assert breaker_for("gmi").state == "closed"


def test_a_failed_probe_reopens_the_breaker(monkeypatch, model, fake_openai):
    monkeypatch.setenv("LM_BREAKER_COOLDOWN", "0")
    lm = model()
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        api_error(500)
    )
    for _ in range(4):
        lm.complete([{"role": "user", "content": "hi"}])
    assert breaker_for("gmi").state == "open"


def test_the_last_route_is_never_skipped(model, fake_openai):
    """An open breaker on the cover must not zero out the row."""
    lm = model()
    breaker = breaker_for("deepseek")
    for _ in range(breaker.failure_threshold):
        breaker.record_failure()
    assert breaker.state == "open"
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        api_error(500)
    )
    assert lm.complete([{"role": "user", "content": "hi"}]).text == "ok"


def test_non_divertible_errors_do_not_count_against_provider_health(model, fake_openai):
    lm = model()
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        bad_request("context_length_exceeded")
    )
    for _ in range(5):
        with pytest.raises(openai.BadRequestError):
            lm.complete([{"role": "user", "content": "hi"}])
    assert breaker_for("gmi").state == "closed"


def test_breaker_can_be_disabled(monkeypatch, model, fake_openai):
    monkeypatch.setenv("LM_BREAKER", "0")
    lm = model()
    gmi = by_provider(fake_openai, "gmi")
    gmi.behaviour = lambda _n, _kw: (_ for _ in ()).throw(api_error(500))
    for _ in range(6):
        lm.complete([{"role": "user", "content": "hi"}])
    assert len(gmi.calls) == 6  # every call re-tries the primary


def test_breaker_state_is_shared_across_model_instances(model, fake_openai):
    first, second = model(), model()
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        api_error(500)
    )
    for lm in (first, second, first):
        lm.complete([{"role": "user", "content": "hi"}])
    assert breaker_for("gmi").state == "open"


def test_breaker_threshold_is_configurable(monkeypatch, model, fake_openai):
    monkeypatch.setenv("LM_BREAKER_FAILURES", "1")
    lm = model()
    gmi = by_provider(fake_openai, "gmi")
    gmi.behaviour = lambda _n, _kw: (_ for _ in ()).throw(api_error(500))
    for _ in range(4):
        lm.complete([{"role": "user", "content": "hi"}])
    assert len(gmi.calls) == 1


def test_breaker_is_threadsafe_under_concurrent_failures():
    import threading

    breaker = ProviderBreaker("x", failure_threshold=50, cooldown_seconds=60)
    threads = [
        threading.Thread(target=lambda: [breaker.record_failure() for _ in range(10)])
        for _ in range(10)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert breaker.state == "open"


def test_route_is_hashable_and_frozen():
    route = Route("gmi", "m", "https://x", "K")
    with pytest.raises(Exception):
        route.model = "other"  # type: ignore[misc]
