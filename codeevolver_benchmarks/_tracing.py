"""Small OpenTelemetry decorator using CodeEvolver's trace conventions."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode
except ImportError:  # Keep the seed runnable before runtime dependencies are installed.
    trace = None
    StatusCode = None


F = TypeVar("F", bound=Callable[..., Any])


def _text(value: Any, limit: int) -> str:
    return str(value)[:limit]


def _inputs(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        return {
            key: value
            for key, value in bound.arguments.items()
            if key not in {"self", "cls"}
        }
    except TypeError:
        return kwargs


def traceable(
    span_kind: str,
    *,
    name: str,
    max_attr_chars: int = 12_000,
) -> Callable[[F], F]:
    """Capture a bounded sync call as a CodeEvolver-readable OTel span."""

    def decorate(fn: F) -> F:
        if trace is None:
            return fn
        tracer = trace.get_tracer(fn.__module__)

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("ce.span_kind", span_kind)
                for key, value in _inputs(fn, args, kwargs).items():
                    span.set_attribute(
                        f"ce.inputs.{key}", _text(value, max_attr_chars)
                    )
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    detail = _text(f"{type(exc).__name__}: {exc}", max_attr_chars)
                    span.set_attribute("ce.error", detail)
                    span.set_status(StatusCode.ERROR, detail)
                    span.record_exception(exc)
                    raise
                span.set_attribute("ce.output", _text(result, max_attr_chars))
                return result

        return wrapped  # type: ignore[return-value]

    return decorate
