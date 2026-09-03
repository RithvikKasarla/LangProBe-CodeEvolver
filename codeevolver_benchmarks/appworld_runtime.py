from __future__ import annotations

import atexit
import logging
import os
import random
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import requests

from ._tracing import traceable

_LOGGER = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_BASE_SECONDS = 0.5
_RETRY_BACKOFF_JITTER_SECONDS = 0.25

# `AppWorldClient.request()` takes an explicit `idempotent` flag instead of
# sniffing the endpoint string, so a caller adding a new endpoint later has
# to make a deliberate choice instead of silently inheriting whatever policy
# some other endpoint happens to use. The default is the conservative one --
# NOT safe to blindly repeat -- because resending a side-effecting call
# (paying, emailing, anything /execute's generated code does inside the
# simulated apps) risks applying the same action twice; a caller has to
# explicitly opt into the more aggressive retry policy.
#
# Endpoints this client calls, and why each is classified the way it is:
#   GET  /                health probe (ensure_started)     -- SAFE:
#                          read-only, no state change.
#   POST /initialize       starts/resets a task's session    -- SAFE: this
#                          (re)establishes the starting state for task_id;
#                          it is not an action taken inside the simulated
#                          world, so repeating it does not compound.
#   POST /execute           runs the evolved program's code   -- SIDE-
#                          EFFECTING: the generated code can call simulated
#                          apps that send money, send email, etc. Not
#                          idempotent -- see `AppWorldClient.execute()`.
#   POST /task_completed    query of task completion status   -- SAFE:
#                          read-only.
#   POST /evaluate          scores final state vs. ground truth -- SAFE:
#                          read-only comparison, does not act in the world.
#   POST /close             tears down a task's session        -- SAFE:
#                          closing an already-closed (or never-opened)
#                          session is a no-op, not a repeated action.
#   GET  /tasks/{id}        task metadata lookup (show_task)    -- SAFE:
#                          read-only.
_DEFAULT_IDEMPOTENT = False

# A requests.exceptions.ConnectionError covers two very different failures:
#   - the TCP handshake itself never completed (connection refused, DNS
#     failure, host unreachable, ...) -- the simulator never saw the
#     request, so resending is provably safe.
#   - the connection existed and then died partway (peer reset, aborted,
#     closed while reading the response, broken pipe while writing the
#     request body) -- the simulator may already have received, started, or
#     even finished a side-effecting request before the connection dropped.
# We only know which one happened from the exception's message, since
# `requests`/urllib3 don't expose a typed distinction. These substrings are
# the standard OS/urllib3 wording for "never connected"; anything else
# (including the "reset"/"aborted"/"closed" wording the engine's own Gate 2
# also keys on -- see tests/test_appworld_runtime_retry.py) is treated as
# the ambiguous mid-request case, per the fail-safe default above.
_NEVER_ESTABLISHED_CONNECTION_PATTERNS = (
    "connection refused",
    "failed to establish a new connection",
    "name or service not known",
    "nodename nor servname provided",
    "no route to host",
    "network is unreachable",
)


def _connection_never_established(exc: requests.exceptions.ConnectionError) -> bool:
    """True only for connection failures that provably never reached the
    simulator (handshake-phase failures) -- see the constant above."""
    message = str(exc).lower()
    return any(pattern in message for pattern in _NEVER_ESTABLISHED_CONNECTION_PATTERNS)


class AppWorldRequestError(requests.exceptions.RequestException):
    """Base for AppWorld HTTP client failures.

    Deliberately a `requests.exceptions.RequestException` (not a bare
    RuntimeError): two pre-existing call sites in this module --
    `AppWorldServer.ensure_started()`'s startup poll and
    `AppWorldServerPool.acquire()`'s cleanup -- catch
    `requests.RequestException` around calls into `AppWorldClient`. Before
    this class existed, `request()` only ever raised bare `requests`
    exceptions, which those handlers were written to catch. Raising a type
    unrelated to `requests.RequestException` here would silently break both
    handlers (see tests/test_appworld_runtime_retry.py). Note
    `RequestException` derives from `OSError`, so this class must NOT also
    inherit from `RuntimeError` -- combining the two layouts raises
    "multiple bases have instance lay-out conflict" at class-creation time.
    """

    def __init__(self, message: str, *, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class AppWorldServiceUnavailable(AppWorldRequestError):
    """The local AppWorld simulator service failed on every retry attempt.

    This is an infrastructure fault -- the simulator process returned
    repeated 5xx responses, or the connection dropped/timed out on every
    attempt -- not a fault in the evolved program being scored. The
    service is genuinely, if transiently, unavailable; the request almost
    certainly never took effect.

    Name and message matter beyond readability: the engine's row
    classifier pattern-matches `f"{type(exc).__name__}: {exc}"` against a
    fixed vocabulary (see tests/test_appworld_runtime_retry.py) to decide
    whether a dead row is an infrastructure fault rather than a program
    fault. "ServiceUnavailable" and "temporarily unavailable" below are
    load-bearing for that classification.
    """


def _retry_delay_seconds(attempt: int) -> float:
    """Short exponential backoff plus jitter, indexed from attempt 1."""
    return _RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(
        0, _RETRY_BACKOFF_JITTER_SECONDS
    )


class AppWorldClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    @traceable("tool", name="appworld.request", max_attr_chars=12_000)
    def request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        *,
        idempotent: bool = _DEFAULT_IDEMPOTENT,
    ) -> Any:
        """Call the simulator, retrying transient faults with backoff.

        `idempotent` must be True only for calls that are genuinely safe to
        repeat (see the endpoint table above `_DEFAULT_IDEMPOTENT`). It
        defaults to False -- the conservative choice -- because resending a
        side-effecting call risks applying it twice.
        """
        url = f"{self.base_url}{endpoint}"
        last_exc: BaseException | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = requests.request(method, url, json=data, timeout=300)
                response.raise_for_status()
            except requests.exceptions.ConnectTimeout as exc:
                # ConnectTimeout subclasses BOTH ConnectionError and Timeout
                # (see tests/test_appworld_runtime_retry.py for a test that
                # pins this), so it MUST be caught here, before either of
                # the broader clauses below -- otherwise it would fall into
                # the ambiguous-ConnectionError or the never-retried-Timeout
                # branch and be mis-classified. A ConnectTimeout means the
                # TCP handshake itself never completed: nothing was sent, so
                # this is safe to retry on every endpoint, idempotent or not.
                last_exc = exc
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                if not (idempotent or _connection_never_established(exc)):
                    # Ambiguous: a connection existed and then died, so the
                    # simulator may already have received, started, or even
                    # finished a side-effecting request before it dropped.
                    # Not retried -- resending could re-run the evolved
                    # program's code a second time -- but this is still the
                    # simulator dropping a live connection, not the
                    # program's own doing, so it is wrapped as an
                    # infrastructure fault, same as a 5xx below.
                    raise AppWorldServiceUnavailable(
                        f"AppWorld simulator at {url} dropped the connection "
                        f"mid-request ({type(exc).__name__}: {exc}); the "
                        "request may already have taken effect so this is "
                        "not retried, but the simulator is not responding "
                        "correctly right now -- temporarily unavailable. "
                        "This is a simulator/infrastructure fault, not a "
                        "fault in the evolved program.",
                        status_code=0,
                    ) from exc
                # Else: never established (or a SAFE endpoint, where any
                # ConnectionError is fine to repeat) -- fall through to retry.
            except requests.exceptions.Timeout as exc:
                last_exc = exc
                if not idempotent:
                    # A read timeout on a side-effecting endpoint means the
                    # simulator accepted /execute and then never answered
                    # within 300s. Ordinary exceptions raised by the evolved
                    # program's generated code are caught INSIDE the
                    # simulator and come back as a 200 -- so the
                    # overwhelmingly likely cause of a silent 300s hang is
                    # the program's own code stuck in an infinite loop, not
                    # simulator infrastructure. We must not excuse that: a
                    # hanging candidate is exactly what the optimizer needs
                    # to see in its fitness signal. Propagate the raw
                    # requests.Timeout, unretried and unwrapped, so it is
                    # booked as a program failure -- exactly as it was
                    # before retry existed. A SAFE endpoint (health probe,
                    # close, ...) timing out says nothing about the
                    # program's own code, so those keep the old policy of
                    # retrying and, on exhaustion, being treated as infra.
                    raise
                # Else: SAFE endpoint -- fall through to retry.
            except requests.exceptions.HTTPError as exc:
                last_exc = exc
                is_server_error = exc.response is not None and exc.response.status_code >= 500
                if not is_server_error:
                    # A 4xx is the caller's own malformed request -- the
                    # simulator received and understood it, and it will fail
                    # identically on every attempt, on every endpoint.
                    # Raise immediately instead of wasting wall-clock (and
                    # potentially masking a real program bug) on retries.
                    raise
                if not idempotent:
                    # A 5xx on a side-effecting endpoint is the simulator's
                    # own fault, but it is not retried: we cannot tell
                    # whether it failed before or after applying side
                    # effects, and /execute is not idempotent.
                    status_code = exc.response.status_code if exc.response is not None else 0
                    raise AppWorldServiceUnavailable(
                        f"AppWorld simulator at {url} returned HTTP "
                        f"{status_code} on a side-effecting request "
                        f"({type(exc).__name__}: {exc}); the request may "
                        "already have taken effect so this is not retried, "
                        "but the simulator is not responding correctly "
                        "right now -- temporarily unavailable. This is a "
                        "simulator/infrastructure fault, not a fault in the "
                        "evolved program.",
                        status_code=status_code,
                    ) from exc
                # Else: SAFE endpoint -- fall through to retry.
            else:
                # A 2xx response: the simulator received, understood, and
                # applied the request. A malformed or truncated body at this
                # point (bad JSON, or the stream breaking mid-read) is still
                # a simulator fault, not a program fault -- but critically it
                # is NOT retried like the failures above: the 2xx means the
                # request already took effect, /execute is not idempotent,
                # and re-issuing it here could execute the same agent action
                # a second time and silently corrupt the row's measurement.
                # Raise immediately, without consuming another attempt.
                try:
                    payload = response.json()
                except (ValueError, requests.exceptions.ChunkedEncodingError) as exc:
                    raise AppWorldServiceUnavailable(
                        f"AppWorld simulator at {url} returned a malformed "
                        f"response body ({type(exc).__name__}: {exc}); the "
                        "request already took effect so this is not retried, "
                        "but the simulator is not responding correctly right "
                        "now -- temporarily unavailable. This is a simulator/"
                        "infrastructure fault, not a fault in the evolved "
                        "program.",
                        status_code=response.status_code,
                    ) from exc
                return payload.get("output", payload)

            if attempt >= _MAX_ATTEMPTS:
                break
            _LOGGER.warning(
                "AppWorld simulator request failed (%s: %s) on %s; retrying (attempt %d/%d)",
                type(last_exc).__name__,
                last_exc,
                url,
                attempt,
                _MAX_ATTEMPTS,
            )
            time.sleep(_retry_delay_seconds(attempt))

        assert last_exc is not None, "loop always sets last_exc before exhausting attempts"
        status_code = getattr(getattr(last_exc, "response", None), "status_code", 0) or 0
        raise AppWorldServiceUnavailable(
            f"AppWorld simulator at {url} is temporarily unavailable after "
            f"{_MAX_ATTEMPTS} attempts ({type(last_exc).__name__}: {last_exc}); "
            "this is a simulator/infrastructure fault, not a fault in the "
            "evolved program.",
            status_code=status_code,
        ) from last_exc

    def initialize(self, experiment_name: str, task_id: str) -> Any:
        # SAFE: (re)establishes task_id's starting state; see the endpoint
        # table above _DEFAULT_IDEMPOTENT.
        return self.request(
            "POST",
            "/initialize",
            {"experiment_name": experiment_name, "task_id": task_id},
            idempotent=True,
        )

    def execute(self, task_id: str, code: str) -> str:
        # SIDE-EFFECTING: runs the evolved program's code inside the
        # simulator, which can call simulated apps with real side effects
        # (send a payment, send an email, ...). Explicit even though it
        # matches the default, so this call site can never silently drift
        # from that classification if the default ever changes.
        return str(
            self.request(
                "POST",
                "/execute",
                {"task_id": task_id, "code": code},
                idempotent=False,
            )
        )

    def task_completed(self, task_id: str) -> bool:
        # SAFE: read-only query.
        return bool(
            self.request("POST", "/task_completed", {"task_id": task_id}, idempotent=True)
        )

    def evaluate(self, task_id: str) -> dict[str, Any]:
        # SAFE: read-only comparison against ground truth, does not act in
        # the world.
        result = self.request(
            "POST",
            "/evaluate",
            {"task_id": task_id, "suppress_errors": True, "report": False},
            idempotent=True,
        )
        return dict(result)

    def close(self, task_id: str) -> None:
        # SAFE: tearing down an already-closed (or never-opened) session is
        # a no-op, not a repeated action.
        self.request("POST", "/close", {"task_id": task_id}, idempotent=True)

    def show_task(self, task_id: str) -> dict[str, Any]:
        # SAFE: read-only metadata lookup.
        return dict(self.request("GET", f"/tasks/{task_id}", idempotent=True))


class AppWorldServer:
    def __init__(self, port: int, repo_root: Path) -> None:
        self.port = port
        self.repo_root = repo_root
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.client = AppWorldClient(f"http://127.0.0.1:{port}")

    def ensure_started(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        python = Path(
            os.environ.get(
                "APPWORLD_PYTHON",
                str(self.repo_root / ".runtime" / "appworld-venv" / "bin" / "python"),
            )
        )
        root = Path(
            os.environ.get(
                "APPWORLD_ROOT", str(self.repo_root / ".runtime" / "appworld-root")
            )
        )
        if not python.exists():
            raise RuntimeError("AppWorld runtime is missing; run scripts/prepare_wsl.sh")
        log_dir = self.repo_root / ".runtime" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = (log_dir / f"appworld-{self.port}.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(python),
                "-m",
                "appworld.cli",
                "serve",
                "environment",
                "--root",
                str(root),
                "--port",
                str(self.port),
            ],
            cwd=self.repo_root,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"AppWorld server on port {self.port} exited during startup")
            try:
                self.client.request("GET", "/", idempotent=True)
                return
            except requests.RequestException:
                time.sleep(0.25)
        raise TimeoutError(f"AppWorld server on port {self.port} did not become healthy")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


class AppWorldServerPool:
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        count = max(1, int(os.environ.get("APPWORLD_SERVER_COUNT", "4")))
        base_port = int(os.environ.get("APPWORLD_BASE_PORT", "18123"))
        self.servers = [AppWorldServer(base_port + index, repo_root) for index in range(count)]
        atexit.register(self.stop)

    @contextmanager
    def acquire(self, experiment_name: str, task_id: str) -> Iterator[AppWorldClient]:
        server: AppWorldServer | None = None
        while server is None:
            for candidate in self.servers:
                if candidate.lock.acquire(blocking=False):
                    server = candidate
                    break
            if server is None:
                time.sleep(0.05)
        initialized = False
        try:
            server.ensure_started()
            server.client.initialize(experiment_name, task_id)
            initialized = True
            yield server.client
        finally:
            if initialized:
                try:
                    server.client.close(task_id)
                except requests.RequestException:
                    server.stop()
            server.lock.release()

    def stop(self) -> None:
        for server in self.servers:
            server.stop()


appworld_pool = AppWorldServerPool()
