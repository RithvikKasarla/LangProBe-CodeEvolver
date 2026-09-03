"""Provider fallback for the task LM: routing, divert rules, circuit breaker.

No network: every test drives `OpenAICompatibleModel` with fake clients whose
`chat.completions.create` is a scripted function.
"""

from __future__ import annotations

import dataclasses
import json

import httpx
import openai
import pytest

from codeevolver_benchmarks import common
from codeevolver_benchmarks.common import (
    BreakerDecision,
    ModelResponse,
    OpenAICompatibleModel,
    ProviderAPIError,
    ProviderBreaker,
    ProviderResponseError,
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


class _EmptyChoicesResponse:
    """A "successful" HTTP call whose body has nothing usable in it."""

    choices: list = []
    usage = _Usage()


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
    # A cover now means the whole three-way preference ring is reachable
    # (Fix 4), so all three routes get a client built at init, each with the
    # cover-sized retry budget.
    captured: list[int] = []
    real = common.OpenAI
    monkeypatch.setattr(
        common,
        "OpenAI",
        lambda **kw: (captured.append(kw["max_retries"]), real(**kw))[1],
    )
    model()
    assert captured == [1, 1, 1]


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
        # A malformed/empty response is the PROVIDER's fault, not the
        # program's -- and json.JSONDecodeError is a ValueError subclass, so
        # this also pins that should_fallback special-cases it before the
        # generic "not an APIError" early return would otherwise say False.
        json.JSONDecodeError("boom", "doc", 0),
        ProviderResponseError("empty choices"),
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
    # Neither route is OpenRouter and neither has provider_preferences
    # configured, so extra_body must be built per-route, not inherited from
    # some other route's config (Fix 4's companion fix).
    assert "extra_body" not in primary
    assert "extra_body" not in cover


ROUTES_WITH_PROVIDER_PREFERENCES = {
    "gmi": {
        "api_key_env": "GMI_API_KEY",
        "base_url": "https://api.gmi-serving.com/v1",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "provider_preferences": {"order": ["gmi-only"], "allow_fallbacks": False},
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        # Deliberately no provider_preferences of its own.
    },
}

CONFIG_WITH_PROVIDER_PREFERENCES = {
    "provider": "gmi",
    "provider_preference": ["gmi", "deepseek"],
    "routes": ROUTES_WITH_PROVIDER_PREFERENCES,
    "reasoning_effort": "high",
    "max_tokens": 4096,
    "temperature": 0,
    "seed": 42,
    "max_steps": 40,
}


def test_provider_preferences_are_read_per_route_not_from_the_config(
    monkeypatch, fake_openai
):
    """Each attempt's `extra_body` must come from THAT route's own
    `provider_preferences`, never from `self.config` and never leaked from a
    sibling route. No prior fixture put `provider_preferences` on an
    individual route inside a multi-route block, so a regression that reads
    `self.config.get("provider_preferences")` (absent here) instead of
    `route.provider_preferences` passed the whole suite -- the old
    `assert "extra_body" not in ...` held trivially under both correct and
    broken code. This pins it: gmi carries its own preferences, deepseek
    (its sibling in the same ring) carries none, and both must be exact.
    """
    monkeypatch.setattr(
        common, "load_runtime_config", lambda _n: dict(CONFIG_WITH_PROVIDER_PREFERENCES)
    )
    lm = OpenAICompatibleModel("appworld")
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        api_error(500)
    )
    lm.complete([{"role": "user", "content": "hi"}])
    gmi_request = by_provider(fake_openai, "gmi").calls[0]
    deepseek_request = by_provider(fake_openai, "deepseek").calls[0]
    assert gmi_request["extra_body"] == {
        "provider": {"order": ["gmi-only"], "allow_fallbacks": False}
    }
    assert "extra_body" not in deepseek_request


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
    """Every route in the (now three-deep, Fix 4) ring failing must be
    classifiable as infrastructure, not laundered as a raw provider
    exception the engine's classifier would book as a program bug.
    """
    lm = model()
    for provider in ("gmi", "deepseek", "deepinfra"):
        by_provider(fake_openai, provider).behaviour = lambda _n, _kw: (
            _ for _ in ()
        ).throw(api_error(500))
    with pytest.raises(ProviderAPIError):
        lm.complete([{"role": "user", "content": "hi"}])


def test_no_divert_when_fallback_is_disarmed(monkeypatch, model, fake_openai):
    monkeypatch.setenv("LM_FALLBACK", "0")
    lm = model()
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        api_error(500)
    )
    with pytest.raises(ProviderAPIError):
        lm.complete([{"role": "user", "content": "hi"}])


def test_response_parsing_is_unchanged(model, fake_openai):
    result = model().complete([{"role": "user", "content": "hi"}])
    assert isinstance(result, ModelResponse)
    assert (result.text, result.reasoning, result.total_tokens) == ("ok", "thought", 7)


def test_empty_choices_is_a_provider_fault_that_diverts(model, fake_openai):
    """`response.choices[0]` on `choices: []` used to raise IndexError
    *after* the primary's breaker was already marked healthy, with no
    divert. It must instead be caught inside `_send`, before
    `record_success`, and treated like any other provider fault.
    """
    lm = model()
    lm.config["response_json_decode_retries"] = 0  # skip _send's own backoff
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: _EmptyChoicesResponse()
    assert lm.complete([{"role": "user", "content": "hi"}]).text == "ok"
    assert len(by_provider(fake_openai, "deepseek").calls) == 1
    assert breaker_for("gmi").state == "closed"  # no false record_success


_ENGINE_CLASSIFIER_GATE_1 = (
    "litellm",
    "apierror",
    "apistatuserror",
    "apiconnectionerror",
    "serviceunavailable",
    "internalservererror",
    "ratelimiterror",
    "timeout",
)
_ENGINE_CLASSIFIER_GATE_2 = (
    "error code: 402",
    "insufficient balance",
    "model_access_denied",
    "connection reset",
    "connection aborted",
    "remote end closed connection",
    "temporarily unavailable",
)


def test_provider_api_error_satisfies_the_engines_row_failure_classifier():
    """Pins the exact sentinel the engine's row-failure classifier looks for
    (both gates, verified against the engine source) so a future edit that
    changes ProviderAPIError's name or message fails this test loudly instead
    of silently misclassifying an exhausted provider as a program bug.

    The cause is deliberately `ProviderResponseError`, not an
    `openai.APIStatusError`: the latter's own class name embeds the GATE-1
    token "apistatuserror", so a summary built from ANY outer exception
    wrapping one satisfies GATE 1 by accident -- renaming the outer
    `ProviderAPIError` class would still pass. `ProviderResponseError`'s name
    contains none of the GATE-1 tokens (asserted below), so GATE 1 here can
    only be satisfied by `ProviderAPIError`'s own name.
    """
    cause = ProviderResponseError("gmi returned a response with no choices")
    assert not any(
        pattern in type(cause).__name__.lower() for pattern in _ENGINE_CLASSIFIER_GATE_1
    )
    exc = ProviderAPIError("gmi", cause)
    assert type(exc).__name__ == "ProviderAPIError"
    summary = f"{type(exc).__name__}: {exc}".lower()
    assert any(pattern in summary for pattern in _ENGINE_CLASSIFIER_GATE_1)
    assert any(pattern in summary for pattern in _ENGINE_CLASSIFIER_GATE_2)
    # DO NOT use "insufficient balance" as the signature: it also occurs in
    # AppWorld's simulated payment-app output and would collide with genuine
    # program text.
    assert "insufficient balance" not in summary


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


def test_a_non_divertible_probe_failure_does_not_strand_the_breaker(
    monkeypatch, model, fake_openai
):
    """A non-divertible failure on a probe call is zero evidence about
    provider health -- it's the request's fault and would fail identically
    on any provider. It must not get stuck in "probing" forever (`allow()`
    then returns False forever, retiring the provider for the whole
    process); it must fall back to "open" so the next call re-probes.
    """
    monkeypatch.setenv("LM_BREAKER_COOLDOWN", "0")
    lm = model()
    breaker = breaker_for("gmi")
    for _ in range(breaker.failure_threshold):
        breaker.record_failure()
    assert breaker.state == "open"

    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        bad_request("context_length_exceeded")
    )
    with pytest.raises(openai.BadRequestError):
        lm.complete([{"role": "user", "content": "hi"}])
    # Today: stuck reporting "probing" forever. Wanted: back to "open" so a
    # later, unrelated call still gets a chance to re-probe after cooldown.
    assert breaker.state == "open"


def test_abort_probe_cannot_cancel_a_different_callers_probe():
    """`abort_probe()` must only release the probe promotion the CALLING
    attempt actually holds -- not just clear the shared `_probing` flag
    unconditionally.

    Before the fix, `abort_probe()` took no argument and always cleared
    `_probing`, so an unrelated non-divertible failure (e.g. one from a
    route that bypassed `allow()` entirely, like the ring's last route)
    could cancel a DIFFERENT, still-unresolved attempt's probe -- and a
    second `allow()` would then return True while the first probe hadn't
    reported back, admitting two simultaneous requests to a still-down
    provider. This reproduces the review's exact repro sequence.
    """
    breaker = ProviderBreaker("x", failure_threshold=3, cooldown_seconds=0)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == "open"

    first = breaker.allow()  # THE sanctioned probe (call it E)
    assert first == BreakerDecision(allowed=True, is_probe=True)

    # An unrelated caller that never held the probe (is_probe=False, e.g.
    # the last route's allow()-bypassing attempt) must not release E's
    # promotion.
    breaker.abort_probe(False)
    assert breaker.state == "probing"

    second = breaker.allow()
    assert second.allowed is False  # E is still unresolved: no second probe
    assert breaker.state == "probing"

    # Only the caller that actually holds the probe (E itself) can release
    # it, and doing so returns the breaker to its exact pre-probe state.
    breaker.abort_probe(first.is_probe)
    assert breaker.state == "open"

    third = breaker.allow()
    assert third == BreakerDecision(allowed=True, is_probe=True)


def test_the_last_route_is_never_skipped(model, fake_openai):
    """An open breaker on the LAST route must not zero out the row.

    The ring is 3-deep [gmi, deepseek, deepinfra], so it's `deepinfra` --
    not the merely-middle `deepseek` -- whose breaker must be open here:
    opening deepseek's exercises nothing, since deepseek is skipped over by
    ordinary `allow()` regardless of whether the `not is_last` exemption
    exists at all. Both earlier routes are made to fail divertibly so the
    ring genuinely reaches deepinfra with its breaker open.
    """
    lm = model()
    breaker = breaker_for("deepinfra")
    for _ in range(breaker.failure_threshold):
        breaker.record_failure()
    assert breaker.state == "open"
    by_provider(fake_openai, "gmi").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        api_error(500)
    )
    by_provider(fake_openai, "deepseek").behaviour = lambda _n, _kw: (_ for _ in ()).throw(
        api_error(500)
    )
    assert lm.complete([{"role": "user", "content": "hi"}]).text == "ok"
    assert len(by_provider(fake_openai, "deepinfra").calls) == 1


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
    with pytest.raises(dataclasses.FrozenInstanceError):
        route.model = "other"  # type: ignore[misc]
