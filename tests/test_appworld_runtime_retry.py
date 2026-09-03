"""Bounded retry for AppWorldClient.request against transient simulator faults.

No network: every test drives `AppWorldClient` with `requests.request` replaced
by a scripted fake, and `time.sleep` replaced by a no-op so nothing here spends
meaningful wall-clock.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from codeevolver_benchmarks import appworld_runtime
from codeevolver_benchmarks.appworld_runtime import (
    AppWorldClient,
    AppWorldServer,
    AppWorldServerPool,
    AppWorldServiceUnavailable,
)


class FakeResponse:
    """Minimal stand-in for `requests.Response`."""

    def __init__(self, status_code: int, json_data: object = None) -> None:
        self.status_code = status_code
        self._json_data = {} if json_data is None else json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Error for url: fake", response=self
            )

    def json(self) -> object:
        return self._json_data


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every test gets a patched sleep so retries never cost real wall-clock."""
    slept: list[float] = []
    monkeypatch.setattr(appworld_runtime.time, "sleep", lambda seconds: slept.append(seconds))
    return slept


def _client() -> AppWorldClient:
    return AppWorldClient("http://127.0.0.1:18124")


def test_500_then_success_returns_payload_and_logs_one_retry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A SAFE-TO-REPEAT (idempotent=True) endpoint keeps the original
    retry-then-succeed policy for a 5xx -- see the /execute-specific tests
    below for why a SIDE-EFFECTING endpoint behaves differently."""
    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        if len(calls) == 1:
            return FakeResponse(500)
        return FakeResponse(200, {"output": "ok"})

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    with caplog.at_level(logging.WARNING, logger=appworld_runtime.__name__):
        result = _client().request(
            "POST", "/task_completed", {"task_id": "t1"}, idempotent=True
        )

    assert result == "ok"
    assert len(calls) == 2
    retry_warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(retry_warnings) == 1
    message = retry_warnings[0].getMessage()
    assert "/task_completed" in message
    assert "500" in message or "attempt 1/3" in message


def test_three_consecutive_500s_on_a_safe_endpoint_raises_simulator_error_after_exactly_3_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        return FakeResponse(500)

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    with pytest.raises(AppWorldServiceUnavailable) as excinfo:
        _client().request("POST", "/task_completed", {"task_id": "t1"}, idempotent=True)

    assert len(calls) == 3
    # Type name and message both make the fault attributable to the
    # simulator/infrastructure, not the evolved program.
    assert "ServiceUnavailable" in type(excinfo.value).__name__
    assert "temporarily unavailable" in str(excinfo.value).lower()
    assert "not a fault in the evolved program" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, requests.exceptions.HTTPError)


def test_404_raises_immediately_with_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        return FakeResponse(404)

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    with pytest.raises(requests.exceptions.HTTPError):
        _client().request("GET", "/tasks/missing")

    assert len(calls) == 1


def test_connection_error_then_success(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """"connection refused" is a never-established-connection message, so it
    is retried even on /execute (SIDE-EFFECTING, idempotent=False by
    default): the handshake itself never completed, so the request
    provably never reached the simulator. Contrast with
    test_execute_mid_request_connection_error_is_not_retried below, where
    the message indicates a live connection dropped partway through."""
    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        if len(calls) == 1:
            raise requests.exceptions.ConnectionError("connection refused")
        return FakeResponse(200, {"output": {"done": True}})

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    result = _client().request("POST", "/execute", {"task_id": "t1", "code": "1"})

    assert result == {"done": True}
    assert len(calls) == 2
    assert len(no_sleep) == 1


def test_timeout_then_success_on_a_safe_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A SAFE-TO-REPEAT endpoint (health probe, close, ...) timing out says
    nothing about the evolved program's own code, so it keeps the original
    retry-then-succeed policy. Contrast with
    test_execute_read_timeout_propagates_raw_as_a_program_failure below."""
    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        if len(calls) == 1:
            raise requests.exceptions.Timeout("timed out")
        return FakeResponse(200, {"output": "resumed"})

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    result = _client().request("POST", "/task_completed", {"task_id": "t1"}, idempotent=True)

    assert result == "resumed"
    assert len(calls) == 2


def test_payload_without_output_key_returns_whole_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        return FakeResponse(200, {"status": "ok", "value": 42})

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    result = _client().request("GET", "/tasks/t1")

    assert result == {"status": "ok", "value": 42}


def test_does_not_retry_a_response_that_was_received_and_understood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4xx means the request was received; retrying it would waste time and
    could mask a real program bug, so it must fail on the very first call."""
    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        return FakeResponse(400)

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    with pytest.raises(requests.exceptions.HTTPError):
        _client().request("POST", "/execute", {"task_id": "t1", "code": "bad"})

    assert calls == ["http://127.0.0.1:18124/execute"]


# These two tuples are copied verbatim from the (out-of-repo) "engine" repo's
# dead-row classifier. It builds `summary = f"{type(exc).__name__}: {exc}"`
# for whatever exception killed a row and requires a match against BOTH
# gates before booking the row as an infrastructure fault rather than a
# fault in the evolved program being scored. If you rename
# AppWorldServiceUnavailable or reword its message, and don't update the
# engine's classifier in lockstep, these tests will still pass locally but
# the real fix silently regresses: rows will go back to being misclassified
# as program failures. Treat any failure here as a signal to go check the
# engine's classifier too, not just this repo.
_ENGINE_GATE_1 = (
    "litellm",
    "apierror",
    "apistatuserror",
    "apiconnectionerror",
    "serviceunavailable",
    "internalservererror",
    "ratelimiterror",
    "timeout",
)
_ENGINE_GATE_2 = (
    "error code: 402",
    "insufficient balance",
    "model_access_denied",
    "connection reset",
    "connection aborted",
    "remote end closed connection",
    "temporarily unavailable",
)


def _assert_classified_as_infrastructure_fault(exc: BaseException) -> None:
    summary = f"{type(exc).__name__}: {exc}"
    lowered = summary.lower()
    assert any(token in lowered for token in _ENGINE_GATE_1), (
        f"summary {summary!r} matches none of the engine's GATE 1 tokens"
    )
    assert any(token in lowered for token in _ENGINE_GATE_2), (
        f"summary {summary!r} matches none of the engine's GATE 2 tokens"
    )


def test_5xx_exhaustion_on_a_safe_endpoint_passes_engine_dead_row_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        return FakeResponse(500)

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    with pytest.raises(AppWorldServiceUnavailable) as excinfo:
        _client().request("POST", "/task_completed", {"task_id": "t1"}, idempotent=True)

    _assert_classified_as_infrastructure_fault(excinfo.value)


def test_connection_error_exhaustion_on_execute_passes_engine_dead_row_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"connection refused" is retried even on /execute (see
    test_connection_error_then_success), so it can exhaust all 3 attempts
    there too; the resulting AppWorldServiceUnavailable must still classify
    as infrastructure."""

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    with pytest.raises(AppWorldServiceUnavailable) as excinfo:
        _client().request("POST", "/execute", {"task_id": "t1", "code": "1"})

    _assert_classified_as_infrastructure_fault(excinfo.value)


# --- Fix A/B: request() distinguishes SAFE-TO-REPEAT endpoints (retry
# ConnectionError/Timeout/5xx, as above) from the SIDE-EFFECTING /execute
# endpoint, which is not idempotent. On /execute:
#   - ConnectTimeout (never connected)                    -> retried
#   - ConnectionError, never-established message           -> retried
#   - ConnectionError, ambiguous (e.g. mid-request reset)   -> NOT retried,
#     wrapped as AppWorldServiceUnavailable (infra's fault, just not safe
#     to resend)
#   - a read Timeout                                        -> NOT retried,
#     propagated RAW (not wrapped) -- the program's own code hanging is the
#     likely cause, and that must reach the optimizer as a program failure
#   - a 5xx                                                  -> NOT retried,
#     wrapped as AppWorldServiceUnavailable
# ---


def test_execute_5xx_raises_immediately_without_retry_and_passes_engine_gates(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """This is the actual fix for the 4 production rows that died on a
    local /execute 500 and were booked as program bugs: classification (an
    immediate, correctly-named/-worded AppWorldServiceUnavailable), not
    retry, is what repairs them -- retrying a 5xx here would risk applying
    a side effect twice for no benefit, since a 5xx already means the
    simulator itself errored out."""
    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        return FakeResponse(500)

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    with pytest.raises(AppWorldServiceUnavailable) as excinfo:
        _client().execute("t1", "print(1)")

    assert len(calls) == 1
    assert no_sleep == []
    _assert_classified_as_infrastructure_fault(excinfo.value)


def test_execute_read_timeout_propagates_raw_as_a_program_failure(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """Ordinary exceptions raised by the evolved program's generated code
    are caught INSIDE the simulator and come back as a 200 -- so the
    overwhelming likely cause of /execute going silent for the full 300s
    is the program's own code stuck in an infinite loop, not the simulator.
    Wrapping this as infrastructure would erase a genuinely bad candidate's
    failure from the optimizer's fitness signal, so it must propagate the
    raw requests.Timeout, unretried, and its summary must NOT satisfy both
    engine gates -- i.e. it is booked as a program failure, exactly as
    before retry existed."""
    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        raise requests.exceptions.ReadTimeout("Read timed out.")

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    with pytest.raises(requests.exceptions.Timeout) as excinfo:
        _client().execute("t1", "while True: pass")

    assert len(calls) == 1
    assert no_sleep == []
    assert not isinstance(excinfo.value, AppWorldServiceUnavailable)

    summary = f"{type(excinfo.value).__name__}: {excinfo.value}".lower()
    gate_1_hit = any(token in summary for token in _ENGINE_GATE_1)
    gate_2_hit = any(token in summary for token in _ENGINE_GATE_2)
    assert not (gate_1_hit and gate_2_hit), (
        f"summary {summary!r} would satisfy the engine's classifier and be "
        "misbooked as infrastructure instead of a program failure"
    )


def test_execute_connect_timeout_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordering pin: requests.exceptions.ConnectTimeout subclasses BOTH
    ConnectionError and Timeout. If request() caught either broader clause
    first, this would be mis-classified: as the ambiguous-ConnectionError
    case it would raise immediately, wrapped (1 call); as the
    never-retried-on-/execute read-Timeout case it would raise immediately,
    raw (1 call). It only retries -- succeeding on the second call -- if
    ConnectTimeout is dispatched to its own always-safe-to-retry branch,
    which requires that except clause to be ordered before both broader
    ones in AppWorldClient.request()."""
    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        if len(calls) == 1:
            raise requests.exceptions.ConnectTimeout("Connection to simulator timed out")
        return FakeResponse(200, {"output": "ok"})

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    result = _client().execute("t1", "print(1)")

    assert result == "ok"
    assert len(calls) == 2


def test_execute_mid_request_connection_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """A ConnectionError whose message doesn't indicate the TCP handshake
    never completed (here: a peer reset partway through) is ambiguous --
    the simulator may already have executed the side-effecting request
    before the connection died. Not retried on /execute, but still wrapped
    as an infrastructure fault: the simulator dropped a live connection,
    which is not the evolved program's own doing."""
    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        raise requests.exceptions.ConnectionError(
            "('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))"
        )

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    with pytest.raises(AppWorldServiceUnavailable) as excinfo:
        _client().execute("t1", "print(1)")

    assert len(calls) == 1
    assert no_sleep == []
    _assert_classified_as_infrastructure_fault(excinfo.value)


# --- Base-class regression: two pre-existing handlers in this module catch
# `requests.RequestException` around calls into AppWorldClient. Both were
# written when request() only ever raised bare `requests` exceptions, so
# AppWorldRequestError/AppWorldServiceUnavailable MUST also be a
# requests.exceptions.RequestException or both handlers silently break. ---


def test_service_unavailable_is_caught_by_the_pre_existing_request_exception_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both AppWorldServer.ensure_started() and AppWorldServerPool.acquire()
    guard calls into AppWorldClient with `except requests.RequestException:`.
    AppWorldServiceUnavailable must match that clause, exactly like the bare
    `requests` errors request() used to raise before retry was added."""

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        return FakeResponse(500)

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    with pytest.raises(AppWorldServiceUnavailable) as excinfo:
        _client().request("POST", "/execute", {"task_id": "t1", "code": "1"})

    assert isinstance(excinfo.value, requests.exceptions.RequestException)

    caught_by_pre_existing_pattern = False
    try:
        raise excinfo.value
    except requests.RequestException:
        caught_by_pre_existing_pattern = True
    assert caught_by_pre_existing_pattern


class _StubClient:
    """Stands in for AppWorldClient inside AppWorldServer for pool tests."""

    def __init__(self) -> None:
        self.close_calls: list[str] = []

    def initialize(self, experiment_name: str, task_id: str) -> dict[str, object]:
        return {"ok": True}

    def close(self, task_id: str) -> None:
        self.close_calls.append(task_id)
        raise AppWorldServiceUnavailable(
            "AppWorld simulator at http://127.0.0.1:18124/close is "
            "temporarily unavailable after 3 attempts "
            "(ConnectionError: connection refused); this is a simulator/"
            "infrastructure fault, not a fault in the evolved program."
        )


class _StubServer:
    """Stands in for AppWorldServer: exercises AppWorldServerPool.acquire()'s
    locking/cleanup without touching a real process or client."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.client = _StubClient()
        self.ensure_started_calls = 0
        self.stop_calls = 0

    def ensure_started(self) -> None:
        self.ensure_started_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def test_acquire_releases_lock_when_close_raises_service_unavailable() -> None:
    """Regression test for the leaked-lock bug: acquire()'s `finally` calls
    close() then release()s the pool lock; if close() raises a type that
    `except requests.RequestException:` does not match, that exception
    escapes the `finally` and the lock is never released. Bypass
    AppWorldServerPool.__init__ (which would spin up real AppWorldServer
    objects) and wire in a stub server instead."""
    pool = AppWorldServerPool.__new__(AppWorldServerPool)
    server = _StubServer()
    pool.servers = [server]

    with pool.acquire("experiment", "task-1"):
        pass

    assert server.ensure_started_calls == 1
    assert server.client.close_calls == ["task-1"]
    # close() raised -> acquire() must have fallen back to stop()...
    assert server.stop_calls == 1
    # ...and, above all, the lock must not have leaked.
    assert not server.lock.locked()


def test_acquire_reacquires_a_lock_left_by_a_prior_close_failure() -> None:
    """If the lock had leaked (the bug), a second acquire() on a
    single-server pool would spin forever waiting for a lock nothing ever
    releases. Proves the pool is usable again immediately afterwards."""
    pool = AppWorldServerPool.__new__(AppWorldServerPool)
    server = _StubServer()
    pool.servers = [server]

    with pool.acquire("experiment", "task-1"):
        pass
    with pool.acquire("experiment", "task-2"):
        pass

    assert server.client.close_calls == ["task-1", "task-2"]
    assert not server.lock.locked()


def test_ensure_started_keeps_polling_past_one_client_side_exhaustion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test for the other half of the base-class bug:
    ensure_started()'s startup loop wraps each health probe in
    `except requests.RequestException: time.sleep(0.25)` and is meant to
    keep polling for up to ~90s while the simulator process boots. Before
    AppWorldRequestError was a RequestException, request()'s own retry
    exhaustion (~2s) produced a type that handler did not catch, so startup
    aborted after one exhaustion cycle instead of continuing to poll. Here
    the fake transport fails through more than one full 3-attempt cycle
    before succeeding, so this only passes if the outer loop survives it."""
    monkeypatch.setenv("APPWORLD_PYTHON", sys.executable)

    server = AppWorldServer(port=65432, repo_root=tmp_path)

    fake_process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(appworld_runtime.subprocess, "Popen", lambda *a, **k: fake_process)

    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        # Fails through attempt 1's full 3-try exhaustion (calls 1-3) plus
        # two more tries into attempt 2 (calls 4-5), then succeeds on
        # attempt 2's third try (call 6) -- i.e. spans two outer polls.
        if len(calls) <= 5:
            raise requests.exceptions.ConnectionError("still booting")
        return FakeResponse(200, {"output": {}})

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    server.ensure_started()

    assert len(calls) == 6
    if server.log_handle is not None:
        server.log_handle.close()


# --- Malformed 200 body: the request already took effect (2xx means the
# simulator applied it), /execute is not idempotent, so this must NOT be
# retried -- but it must still classify as an infrastructure fault. ---


class _MalformedBodyResponse:
    """A 2xx response whose body cannot be parsed."""

    def __init__(self, exc: Exception) -> None:
        self.status_code = 200
        self._exc = exc

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        raise self._exc


def test_malformed_json_body_raises_service_unavailable_with_exactly_one_call(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        return _MalformedBodyResponse(ValueError("Expecting value: line 1 column 1 (char 0)"))

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    with pytest.raises(AppWorldServiceUnavailable) as excinfo:
        _client().request("POST", "/execute", {"task_id": "t1", "code": "1"})

    # The 2xx means the request already took effect -- retrying it could
    # execute the same agent action a second time, so exactly one call is
    # made and no backoff is slept.
    assert len(calls) == 1
    assert no_sleep == []
    assert isinstance(excinfo.value, requests.exceptions.RequestException)
    assert excinfo.value.__cause__ is not None
    _assert_classified_as_infrastructure_fault(excinfo.value)


def test_chunked_encoding_error_reading_body_raises_service_unavailable_with_no_retry(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    calls: list[str] = []

    def fake_request(method: str, url: str, json: object = None, timeout: float | None = None):
        calls.append(url)
        return _MalformedBodyResponse(
            requests.exceptions.ChunkedEncodingError("Connection broken: stream ended early")
        )

    monkeypatch.setattr(appworld_runtime.requests, "request", fake_request)

    with pytest.raises(AppWorldServiceUnavailable) as excinfo:
        _client().request("POST", "/execute", {"task_id": "t1", "code": "1"})

    assert len(calls) == 1
    assert no_sleep == []
    _assert_classified_as_infrastructure_fault(excinfo.value)
