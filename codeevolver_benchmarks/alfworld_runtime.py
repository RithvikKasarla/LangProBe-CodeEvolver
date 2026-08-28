from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import requests

from ._tracing import traceable


class AlfWorldRequestError(RuntimeError):
    def __init__(self, message: str, *, status: str = "error", status_code: int = 0):
        super().__init__(message)
        self.status = status
        self.status_code = status_code


class AlfWorldWorkerBusy(AlfWorldRequestError):
    """The worker is healthy but leased by another evaluator process."""


class AlfWorldInvalidSession(AlfWorldRequestError):
    """The caller no longer owns the worker's active episode."""


class AlfWorldFatalWorker(AlfWorldRequestError):
    """The worker must exit and be replaced by its external supervisor."""


class AlfWorldClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None

    @traceable("tool", name="alfworld.request", max_attr_chars=6_000)
    def request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        *,
        timeout: float | tuple[float, float] = 900,
    ) -> Any:
        response = requests.request(
            method,
            f"{self.base_url}{endpoint}",
            json=data,
            timeout=timeout,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise AlfWorldRequestError(
                f"ALFWorld worker returned invalid JSON (HTTP {response.status_code})",
                status_code=response.status_code,
            ) from exc
        if payload.get("status") != "success":
            status = str(payload.get("status", "error"))
            error = str(payload.get("error", "ALFWorld worker request failed"))
            exception_type: type[AlfWorldRequestError]
            if status == "busy":
                exception_type = AlfWorldWorkerBusy
            elif status == "invalid_session":
                exception_type = AlfWorldInvalidSession
            elif status == "fatal":
                exception_type = AlfWorldFatalWorker
            else:
                exception_type = AlfWorldRequestError
            raise exception_type(
                error,
                status=status,
                status_code=response.status_code,
            )
        response.raise_for_status()
        return payload.get("output")

    def initialize(self, game_file: str) -> dict[str, Any]:
        if self.session_id is not None:
            raise RuntimeError("ALFWorld client already owns an episode")
        output = dict(self.request("POST", "/initialize", {"game_file": game_file}))
        session_id = str(output.get("session_id", ""))
        if not session_id:
            raise AlfWorldRequestError("ALFWorld worker returned no episode session")
        self.session_id = session_id
        return output

    def step(self, action: str) -> dict[str, Any]:
        if self.session_id is None:
            raise AlfWorldInvalidSession("ALFWorld client has no active episode")
        return dict(
            self.request(
                "POST",
                "/step",
                {"action": action, "session_id": self.session_id},
            )
        )

    def close(self) -> None:
        session_id = self.session_id
        if session_id is None:
            return
        try:
            self.request("POST", "/close", {"session_id": session_id}, timeout=30)
        finally:
            self.session_id = None

    def live(self, *, timeout: float = 2) -> dict[str, Any]:
        return dict(self.request("GET", "/live", timeout=timeout))

    def ready(self, *, timeout: float = 2) -> dict[str, Any]:
        return dict(self.request("GET", "/ready", timeout=timeout))

    def status(self, *, timeout: float = 5) -> dict[str, Any]:
        return dict(self.request("GET", "/status", timeout=timeout))

    def shutdown(self, *, timeout: float = 10) -> None:
        self.request("POST", "/shutdown", timeout=timeout)


class AlfWorldServer:
    def __init__(self, port: int, repo_root: Path) -> None:
        self.port = port
        self.repo_root = repo_root
        self.lock = threading.Lock()
        self.external = os.environ.get("ALFWORLD_EXTERNAL_SERVERS", "").casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.client = AlfWorldClient(f"http://127.0.0.1:{port}")

    def ensure_started(self) -> None:
        if self.external:
            self._wait_until_live()
            return
        if self.process is not None and self.process.poll() is None:
            return
        python = Path(
            os.environ.get(
                "ALFWORLD_PYTHON",
                str(self.repo_root / ".runtime" / "alfworld-venv" / "bin" / "python"),
            )
        )
        if not python.exists():
            raise RuntimeError("ALFWorld runtime is missing; run scripts/prepare_wsl.sh")
        log_dir = self.repo_root / ".runtime" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = (log_dir / f"alfworld-{self.port}.log").open("a", encoding="utf-8")
        env = os.environ.copy()
        env.setdefault(
            "ALFWORLD_DATA", str(self.repo_root / ".runtime" / "alfworld-data")
        )
        popen_options: dict[str, Any] = {}
        if os.name == "posix":
            popen_options["start_new_session"] = True
        elif os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = subprocess.Popen(
            [
                str(python),
                str(self.repo_root / "scripts" / "alfworld_server.py"),
                "--port",
                str(self.port),
                "--repo-root",
                str(self.repo_root),
            ],
            cwd=self.repo_root,
            env=env,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            **popen_options,
        )
        try:
            self._wait_until_live()
        except Exception:
            self.stop()
            raise

    def _wait_until_live(self) -> None:
        deadline = time.monotonic() + float(
            os.environ.get("ALFWORLD_STARTUP_TIMEOUT_SECONDS", "120")
        )
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            if (
                not self.external
                and self.process is not None
                and self.process.poll() is not None
            ):
                raise RuntimeError(f"ALFWorld worker on port {self.port} exited during startup")
            try:
                self.client.live(timeout=2)
                return
            except (requests.RequestException, AlfWorldRequestError) as exc:
                last_error = exc
                time.sleep(0.25)
        suffix = f": {last_error}" if last_error else ""
        raise TimeoutError(
            f"ALFWorld worker on port {self.port} did not become live{suffix}"
        )

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str], force: bool) -> None:
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL if force else signal.SIGTERM)
                return
            except (ProcessLookupError, PermissionError):
                pass
        if force:
            process.kill()
        else:
            process.terminate()

    def stop(self) -> None:
        if self.external:
            return
        process = self.process
        if process is not None and process.poll() is None:
            try:
                self.client.shutdown(timeout=10)
            except (requests.RequestException, AlfWorldRequestError):
                self._terminate_process_group(process, force=False)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._terminate_process_group(process, force=True)
                process.wait(timeout=5)
        self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


class AlfWorldServerPool:
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        count = max(1, int(os.environ.get("ALFWORLD_SERVER_COUNT", "2")))
        base_port = int(os.environ.get("ALFWORLD_BASE_PORT", "19123"))
        self.servers = [AlfWorldServer(base_port + index, repo_root) for index in range(count)]
        atexit.register(self.stop)

    @contextmanager
    def acquire(self, game_file: str) -> Iterator[tuple[AlfWorldClient, dict[str, Any]]]:
        server: AlfWorldServer | None = None
        state: dict[str, Any] | None = None
        deadline = time.monotonic() + float(
            os.environ.get("ALFWORLD_ACQUIRE_TIMEOUT_SECONDS", "300")
        )
        last_error: BaseException | None = None
        while server is None:
            for candidate in self.servers:
                if not candidate.lock.acquire(blocking=False):
                    continue
                try:
                    candidate.ensure_started()
                    state = candidate.client.initialize(game_file)
                    server = candidate
                    break
                except AlfWorldWorkerBusy:
                    pass
                except (
                    requests.RequestException,
                    AlfWorldFatalWorker,
                    TimeoutError,
                ) as exc:
                    last_error = exc
                    candidate.stop()
                except AlfWorldRequestError as exc:
                    if exc.status_code < 500:
                        raise
                    last_error = exc
                    candidate.stop()
                finally:
                    if server is not candidate:
                        candidate.lock.release()
            if server is not None:
                break
            if time.monotonic() >= deadline:
                suffix = f": {last_error}" if last_error else ""
                raise TimeoutError(f"No ALFWorld worker became available{suffix}")
            time.sleep(0.05)

        assert state is not None
        try:
            yield server.client, state
        finally:
            try:
                server.client.close()
            except (requests.RequestException, AlfWorldRequestError):
                server.stop()
            server.lock.release()

    def stop(self) -> None:
        for server in self.servers:
            server.stop()


alfworld_pool = AlfWorldServerPool()
