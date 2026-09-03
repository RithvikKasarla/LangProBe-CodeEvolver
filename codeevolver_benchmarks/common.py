from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI

from ._tracing import traceable

try:  # Keep the seed runnable before runtime dependencies are installed.
    from opentelemetry import trace as _otel_trace
except ImportError:
    _otel_trace = None


_CONFIG_PATH = Path(__file__).with_name("runtime_config.json")
_LOGGER = logging.getLogger(__name__)
_DEFAULT_RESPONSE_JSON_DECODE_RETRIES = 2

# Retry budget, matched to whether a cover exists. With an equivalent provider
# idle, the third attempt's backoff is dead time; with no cover, retrying is the
# only thing that can save the row.
_NUM_RETRIES_WITH_COVER = 1
_NUM_RETRIES_NO_COVER = 5

_DEFAULT_BREAKER_FAILURES = 3
_DEFAULT_BREAKER_COOLDOWN = 60.0

_FALSEY = {"0", "false", "no", "off", "none"}
_TRUTHY = {"1", "true", "yes", "on"}

# A 4xx that says the REQUEST is the problem: it fails identically on every
# provider, so diverting only doubles the cost and hides a fault in the program
# being evolved. Everything else -- including GMI's 402 "Insufficient balance"
# and any 5xx -- is a provider fault and diverts.
_NON_DIVERTIBLE_PATTERNS = (
    "context_length_exceeded",
    "context length",
    "maximum context",
    "reduce the length",
    "unsupported_parameter",
    "unsupported parameter",
    "unrecognized request argument",
    "unknown parameter",
)


def load_runtime_config(name: str) -> dict[str, Any]:
    with _CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)[name]


def _span_attr(key: str, value: Any) -> None:
    """Stamp the active span, if tracing is installed and a span is running."""
    if _otel_trace is None:
        return
    span = _otel_trace.get_current_span()
    if span is not None and span.is_recording():
        span.set_attribute(key, value)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _FALSEY:
        return False
    if value in _TRUTHY:
        return True
    raise RuntimeError(
        f"{name}={raw!r} is not a recognised boolean "
        f"(use one of {sorted(_TRUTHY | _FALSEY)})"
    )


def _env_number(name: str, default: float, cast: Any) -> Any:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return cast(default)
    try:
        return cast(raw.strip())
    except ValueError:
        raise RuntimeError(f"{name}={raw!r} is not a number") from None


class ProviderResponseError(Exception):
    """The provider's HTTP call "succeeded" but the response is unusable
    (right now: an empty ``choices`` list). Raised only from `_send`, while
    validating the PROVIDER's own response -- see the invariant noted there
    and on `should_fallback` below.
    """


class ProviderAPIError(RuntimeError):
    """Every route in the fallback ring failed with a divertible (provider)
    fault -- the row has no chances left, and the failure is infrastructure's
    fault, not the evolving program's.

    The class name is deliberate: the engine's row-failure classifier builds
    `f"{type(exc).__name__}: {exc}"`, lowercases it, and requires a GATE 1
    term -- lowercased "ProviderAPIError" is "providerapierror", which
    contains "apierror" -- AND a GATE 2 term, satisfied by this message's
    literal phrase "temporarily unavailable". Both must stay literally true
    of what happened: an exhausted provider genuinely is temporarily
    unavailable. This string must never be adjusted just to satisfy the
    classifier -- see also why it deliberately avoids "insufficient balance"
    as a signature (AppWorld's simulated payment app can legitimately emit
    that phrase as program output, and colliding with it would misclassify a
    real program bug as infrastructure).
    """

    def __init__(self, provider: str, cause: BaseException) -> None:
        detail = str(cause)[:200]
        super().__init__(
            f"provider {provider!r} temporarily unavailable: every route in "
            f"the fallback ring was exhausted; last error was "
            f"{type(cause).__name__}: {detail}"
        )


def should_fallback(exc: BaseException) -> bool:
    """Is `exc` a provider fault worth re-issuing on the cover?

    Never diverts a non-API exception: a ValueError raised by the program must
    stay visible to whoever is evolving it, not be retried somewhere else.
    The one deliberate exception is `json.JSONDecodeError` /
    `ProviderResponseError`: both are raised only inside `_send`, validating
    the PROVIDER's own response, never from parsing what the program
    produced. INVARIANT this whole classification depends on: no program-side
    JSON parsing (or other response validation) may ever happen inside
    `complete()`'s try block -- only inside `_send` -- or a program's own
    parsing failure would be misclassified as a provider fault and diverted.

    Ordering matters below: `json.JSONDecodeError` is a `ValueError`
    subclass, so it must be special-cased before the `not isinstance(exc,
    openai.APIError)` check below, or it would fall through to `return False`.
    """
    if isinstance(exc, (json.JSONDecodeError, ProviderResponseError)):
        return True
    if not isinstance(exc, openai.APIError):
        return False
    if isinstance(exc, openai.BadRequestError):
        text = str(exc).lower()
        return not any(pattern in text for pattern in _NON_DIVERTIBLE_PATTERNS)
    return True


@dataclass(frozen=True)
class BreakerDecision:
    """`ProviderBreaker.allow()`'s answer.

    `allowed` says whether the caller may issue the request. `is_probe` says
    whether THIS caller is the one `allow()` promoted to the sanctioned,
    cooldown-gated probe (see `ProviderBreaker` docstring) -- at most one
    caller can hold `is_probe=True` at a time, enforced under `allow()`'s own
    lock. A caller must hand its own `is_probe` back to `abort_probe()`:
    that is what lets `abort_probe()` release only the promotion it actually
    holds, never a different, still-unresolved caller's.
    """

    allowed: bool
    is_probe: bool


class ProviderBreaker:
    """Process-wide health flag for one provider.

    A sustained outage must cost O(1), not O(calls): after `failure_threshold`
    consecutive diverted failures the provider is skipped outright -- no
    request, no retries, no backoff -- until `cooldown_seconds` elapses and
    exactly one probe call re-tests it. Shared across threads on purpose;
    per-thread state would make each of N worker threads learn about the same
    outage independently, N waves of failures instead of one.
    """

    def __init__(
        self,
        provider: str,
        failure_threshold: int = _DEFAULT_BREAKER_FAILURES,
        cooldown_seconds: float = _DEFAULT_BREAKER_COOLDOWN,
    ) -> None:
        self.provider = provider
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "probing" if self._probing else "open"

    def allow(self) -> BreakerDecision:
        """May a request be issued to this provider right now?

        Returns a `BreakerDecision`. At most one caller is ever handed
        `is_probe=True` while the promotion is unresolved: the check and the
        `_probing = True` write happen under the same lock, so two
        concurrent callers can never both be promoted for the same outage.
        """
        promoted = False
        with self._lock:
            if self._opened_at is None:
                return BreakerDecision(allowed=True, is_probe=False)
            if self._probing:
                return BreakerDecision(allowed=False, is_probe=False)
            if (time.monotonic() - self._opened_at) < self.cooldown_seconds:
                return BreakerDecision(allowed=False, is_probe=False)
            self._probing = True
            promoted = True
        if promoted:
            _LOGGER.warning(
                "provider %r cooldown elapsed; letting one probe call through",
                self.provider,
            )
            _span_attr("lm.breaker.probe", self.provider)
        return BreakerDecision(allowed=True, is_probe=True)

    def abort_probe(self, is_probe: bool) -> None:
        """Undo `allow()`'s probe promotion without treating it as evidence.

        A non-divertible failure during a probe (a request-shaped fault: it
        would fail identically against any provider) says nothing about this
        provider's health. Closing the breaker on it would be wrong (the
        provider was never actually re-tested); leaving `_probing` stuck True
        would be worse (`allow()` then returns False forever, with no
        cooldown to escape it -- retiring the provider for the rest of the
        process). Set `_probing` back to False and leave `_opened_at` /
        `_consecutive_failures` untouched: the breaker returns to its exact
        pre-probe state, so the next call re-probes.

        `is_probe` MUST be the `BreakerDecision.is_probe` that THIS caller's
        own `allow()` call returned. A non-divertible failure on some other,
        unrelated in-flight attempt against the same shared breaker --
        including one that bypassed `allow()` entirely, e.g. the fallback
        ring's last route, which is always attempted regardless of breaker
        state -- must never release a different, still-unresolved caller's
        probe: that would let a second request through while the first
        probe hasn't reported back yet, defeating "exactly one probe call"
        above. A caller that never held the probe passes `is_probe=False`
        and this is a no-op.
        """
        if not is_probe:
            return
        with self._lock:
            self._probing = False

    def record_success(self) -> None:
        with self._lock:
            recovered = self._opened_at is not None
            self._consecutive_failures = 0
            self._opened_at = None
            self._probing = False
        if recovered:
            _LOGGER.warning("provider %r healthy again; breaker closed", self.provider)
            _span_attr("lm.breaker.closed", self.provider)

    def record_failure(self) -> None:
        opened = False
        with self._lock:
            if self._probing:
                # The probe failed: straight back to open, cooldown restarts.
                self._probing = False
                self._opened_at = time.monotonic()
            else:
                self._consecutive_failures += 1
                if (
                    self._opened_at is None
                    and self._consecutive_failures >= self.failure_threshold
                ):
                    self._opened_at = time.monotonic()
                    opened = True
        if opened:
            _LOGGER.warning(
                "provider %r failed %d consecutive calls; skipping it for %.0fs",
                self.provider,
                self.failure_threshold,
                self.cooldown_seconds,
            )
            _span_attr("lm.breaker.opened", self.provider)


_BREAKERS: dict[str, ProviderBreaker] = {}
_BREAKERS_LOCK = threading.Lock()


def breaker_for(provider: str) -> ProviderBreaker:
    with _BREAKERS_LOCK:
        breaker = _BREAKERS.get(provider)
        if breaker is None:
            breaker = ProviderBreaker(
                provider,
                failure_threshold=_env_number(
                    "LM_BREAKER_FAILURES", _DEFAULT_BREAKER_FAILURES, int
                ),
                cooldown_seconds=_env_number(
                    "LM_BREAKER_COOLDOWN", _DEFAULT_BREAKER_COOLDOWN, float
                ),
            )
            _BREAKERS[provider] = breaker
        return breaker


def reset_breakers() -> None:
    """Drop all remembered provider health. For tests and long-lived REPLs."""
    with _BREAKERS_LOCK:
        _BREAKERS.clear()


@dataclass(frozen=True)
class ModelResponse:
    text: str
    reasoning: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class Route:
    """One provider's way of serving the pinned model.

    Every route in a config block serves the SAME weights, which is what makes
    diverting between them sound. Never point one at a different model: that
    changes what is being measured.
    """

    provider: str
    model: str
    base_url: str
    api_key_env: str
    provider_preferences: dict[str, Any] | None = None


def _routes_for(config: dict[str, Any]) -> dict[str, Route]:
    """The config block's routing table, one entry per provider.

    A block with no `routes` key is single-provider (alfworld_vision, whose
    resilience comes from OpenRouter's own provider pool) and keeps the flat
    model/base_url/api_key_env/provider_preferences fields. Each entry in a
    multi-route block may carry its own `provider_preferences`; a route with
    none simply omits `extra_body` from its request (see `_build_request`).
    """
    raw = config.get("routes")
    if not raw:
        return {
            config["provider"]: Route(
                provider=config["provider"],
                model=config["model"],
                base_url=config["base_url"],
                api_key_env=config["api_key_env"],
                provider_preferences=config.get("provider_preferences"),
            )
        }
    routes: dict[str, Route] = {}
    for provider, entry in raw.items():
        try:
            routes[provider] = Route(
                provider=provider,
                model=entry["model"],
                base_url=entry["base_url"],
                api_key_env=entry["api_key_env"],
                provider_preferences=entry.get("provider_preferences"),
            )
        except KeyError as exc:
            raise RuntimeError(
                f"route {provider!r} is missing {exc.args[0]!r} in runtime_config.json"
            ) from None
    return routes


def resolve_provider(config: dict[str, Any], routes: dict[str, Route]) -> Route:
    """Which provider serves the request first ($LM_PROVIDER, else the config)."""
    name = os.environ.get("LM_PROVIDER", "").strip() or config["provider"]
    route = routes.get(name)
    if route is None:
        raise RuntimeError(
            f"provider {name!r} has no route; known providers: {sorted(routes)}"
        )
    return route


def resolve_fallback(
    config: dict[str, Any], routes: dict[str, Route], primary: Route
) -> Route | None:
    """Which provider covers for `primary`.

    Default is the next entry after it in the single preference order, wrapping
    -- so restoring the head of that order is the only edit needed when a
    provider comes back. `$LM_FALLBACK` disarms it (falsey) or names a cover.
    """
    raw = os.environ.get("LM_FALLBACK", "").strip()
    if raw and raw.lower() not in (_TRUTHY | _FALSEY):
        route = routes.get(raw)
        if route is None:
            raise RuntimeError(
                f"LM_FALLBACK={raw!r} has no route; known providers: {sorted(routes)}"
            )
        if route.provider == primary.provider:
            raise RuntimeError(
                f"LM_FALLBACK={raw!r} is also the primary provider; "
                f"set LM_FALLBACK=0 to run without a cover"
            )
        return route
    if not _env_flag("LM_FALLBACK", True):
        return None

    preference = [p for p in config.get("provider_preference", []) if p in routes]
    if primary.provider not in preference or len(preference) < 2:
        return None
    index = preference.index(primary.provider)
    return routes[preference[(index + 1) % len(preference)]]


class OpenAICompatibleModel:
    def __init__(self, config_name: str) -> None:
        self.config = load_runtime_config(config_name)
        routes = _routes_for(self.config)
        self.route = resolve_provider(self.config, routes)
        self.fallback_route = resolve_fallback(self.config, routes, self.route)

        if not os.environ.get(self.route.api_key_env, ""):
            raise RuntimeError(f"{self.route.api_key_env} is not configured")
        if self.fallback_route is not None and not os.environ.get(
            self.fallback_route.api_key_env, ""
        ):
            # Degrade to no cover rather than divert into a guaranteed failure.
            _LOGGER.warning(
                "%s is not configured; running %r with no fallback provider",
                self.fallback_route.api_key_env,
                self.route.provider,
            )
            self.fallback_route = None

        self.breaker_enabled = _env_flag("LM_BREAKER", True)
        self._ring = self._provider_ring(routes)
        num_retries = (
            _NUM_RETRIES_WITH_COVER
            if len(self._ring) > 1
            else _NUM_RETRIES_NO_COVER
        )
        self._clients: dict[str, OpenAI] = {}
        for route in self._ring:
            self._client_for(route, num_retries)
        self.client = self._clients[self.route.provider]
        if self.fallback_route is not None:
            _LOGGER.info(
                "task model %s: %s -> %s (cover)",
                self.route.model,
                self.route.provider,
                self.fallback_route.provider,
            )

    def _provider_ring(self, routes: dict[str, Route]) -> list[Route]:
        """Every route `complete()` may attempt, in try-order, from the primary.

        `self.fallback_route` (from `resolve_fallback`) names only the
        immediate next hop -- enough to size the retry budget and log a
        cover. This walks the REST of `provider_preference` too, wrapping
        from the primary and deduping, so a three-(or more-)deep preference
        list is fully reachable instead of stopping at the second entry.
        `$LM_FALLBACK`'s three meanings mirror `resolve_fallback` exactly: an
        explicit name pins a two-deep ring `[primary, name]`; a falsey value
        disarms every cover (`[primary]` alone); anything else walks the
        full ring, dropping any entry whose API key env var is unset.
        """
        raw = os.environ.get("LM_FALLBACK", "").strip()
        if raw and raw.lower() not in (_TRUTHY | _FALSEY):
            # Explicit pin: two-deep only. `self.fallback_route` already
            # reflects resolve_fallback's validation and the missing-key
            # degrade above, so it is the single source of truth here.
            if self.fallback_route is None:
                return [self.route]
            return [self.route, self.fallback_route]
        if not _env_flag("LM_FALLBACK", True):
            return [self.route]

        preference = [
            p for p in self.config.get("provider_preference", []) if p in routes
        ]
        if self.route.provider not in preference:
            return [self.route]
        start = preference.index(self.route.provider)
        ordered = preference[start:] + preference[:start]

        ring: list[Route] = []
        seen: set[str] = set()
        for name in ordered:
            if name in seen:
                continue
            seen.add(name)
            route = routes[name]
            if route.provider != self.route.provider and not os.environ.get(
                route.api_key_env, ""
            ):
                continue
            ring.append(route)
        return ring

    def _client_for(self, route: Route, num_retries: int) -> OpenAI:
        client = self._clients.get(route.provider)
        if client is None:
            client = OpenAI(
                api_key=os.environ[route.api_key_env],
                base_url=route.base_url,
                timeout=900.0,
                max_retries=num_retries,
            )
            self._clients[route.provider] = client
        return client

    def _build_request(
        self, route: Route, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """The request body for one attempt against `route`.

        Built fresh per route, not once per `complete()` call: OpenRouter's
        `provider_preferences` is per-route data (see `Route`/`_routes_for`),
        so forwarding one route's `extra_body` to a different, non-OpenRouter
        route would hand it an argument it doesn't recognise -- which 400s,
        and that phrasing is itself in `_NON_DIVERTIBLE_PATTERNS`, so it would
        fail closed with no divert.
        """
        request: dict[str, Any] = {
            "messages": messages,
            "reasoning_effort": self.config["reasoning_effort"],
            "max_tokens": self.config["max_tokens"],
        }
        if "temperature" in self.config:
            request["temperature"] = self.config["temperature"]
        if "seed" in self.config:
            request["seed"] = self.config["seed"]
        if route.provider_preferences:
            request["extra_body"] = {
                "provider": route.provider_preferences,
            }
        return request

    def _send(self, route: Route, request: dict[str, Any]) -> Any:
        """One provider's attempt, including the malformed-response retry.

        INVARIANT: everything in this try block validates the PROVIDER's own
        response (malformed JSON, an empty `choices` list). No program-side
        parsing may ever be added here or inside `complete()`'s try block --
        `should_fallback` treats `json.JSONDecodeError` and
        `ProviderResponseError` as provider faults precisely because they can
        only originate from this validation, never from the program's output.
        """
        json_decode_retries = max(
            0,
            int(
                self.config.get(
                    "response_json_decode_retries",
                    _DEFAULT_RESPONSE_JSON_DECODE_RETRIES,
                )
            ),
        )
        client = self._clients[route.provider]
        for attempt in range(json_decode_retries + 1):
            try:
                response = client.chat.completions.create(model=route.model, **request)
                if not response.choices:
                    raise ProviderResponseError(
                        f"{route.provider} returned a response with no choices"
                    )
                return response
            except (json.JSONDecodeError, ProviderResponseError) as exc:
                if attempt >= json_decode_retries:
                    raise
                delay_seconds = float(2**attempt)
                _LOGGER.warning(
                    "Provider returned a malformed response (%s); retrying "
                    "chat completion in %.1f seconds (retry %d/%d)",
                    type(exc).__name__,
                    delay_seconds,
                    attempt + 1,
                    json_decode_retries,
                )
                time.sleep(delay_seconds)
        raise AssertionError("unreachable")

    @traceable("llm", name="model.complete", max_attr_chars=12_000)
    def complete(self, messages: list[dict[str, Any]]) -> ModelResponse:
        attempts = self._ring

        for index, route in enumerate(attempts):
            is_last = index == len(attempts) - 1
            breaker = breaker_for(route.provider) if self.breaker_enabled else None
            # Never skip the last remaining route in the ring: an open
            # breaker there is still the only remaining chance at a scored row.
            # `is_probe` defaults to False -- and stays False -- for the last
            # route, which bypasses `allow()` entirely and therefore never
            # holds a probe promotion it could release.
            is_probe = False
            if breaker is not None and not is_last:
                decision = breaker.allow()
                if not decision.allowed:
                    _LOGGER.debug("skipping unhealthy provider %r", route.provider)
                    _span_attr("lm.primary_skipped", route.provider)
                    continue
                is_probe = decision.is_probe
            request = self._build_request(route, messages)
            try:
                # See the INVARIANT on `_send`: no program-side parsing may
                # ever happen in this try block, or should_fallback's
                # classification of the exceptions it can raise breaks.
                response = self._send(route, request)
            except Exception as exc:
                divertible = should_fallback(exc)
                if breaker is not None:
                    if divertible:
                        breaker.record_failure()
                    else:
                        # Zero evidence about provider health: a
                        # non-divertible failure is the request's fault and
                        # would fail identically anywhere. Don't let it
                        # close the breaker or strand a probe -- but only
                        # release the probe THIS attempt actually holds
                        # (`is_probe`); never a different, still-unresolved
                        # attempt's promotion on the same shared breaker.
                        breaker.abort_probe(is_probe)
                if not divertible:
                    raise
                if is_last:
                    # The ring is exhausted and the last fault was the
                    # provider's, not the program's. Re-raise it as the type
                    # that states that plainly, so the engine's row-failure
                    # classifier books the row as infrastructure rather than
                    # as a bug in the program being scored. This is protocol
                    # conformance, not a workaround: every claim the new
                    # exception makes is true of what just happened, and the
                    # original is chained, so nothing is hidden. It is the
                    # ONLY place in this file that substitutes an exception,
                    # and it is reachable only when `divertible` is True.
                    raise ProviderAPIError(route.provider, exc) from exc
                _LOGGER.warning(
                    "provider %r failed (%s: %s); re-issuing on %r",
                    route.provider,
                    type(exc).__name__,
                    str(exc)[:200],
                    attempts[index + 1].provider,
                )
                _span_attr("lm.fallback.from", route.provider)
                _span_attr("lm.fallback.to", attempts[index + 1].provider)
                _span_attr("lm.fallback.reason", type(exc).__name__)
                continue
            if breaker is not None:
                breaker.record_success()
            _span_attr("lm.provider", route.provider)
            return self._parse(response)

        raise AssertionError("unreachable")

    @staticmethod
    def _parse(response: Any) -> ModelResponse:
        message = response.choices[0].message
        content = message.content or ""
        if not isinstance(content, str):
            content = "".join(str(part) for part in content)
        reasoning = getattr(message, "reasoning_content", None) or ""
        usage = response.usage
        return ModelResponse(
            text=content.strip(),
            reasoning=str(reasoning).strip(),
            prompt_tokens=int(usage.prompt_tokens) if usage else 0,
            completion_tokens=int(usage.completion_tokens) if usage else 0,
            total_tokens=int(usage.total_tokens) if usage else 0,
        )


def strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines)
    return value.strip()
