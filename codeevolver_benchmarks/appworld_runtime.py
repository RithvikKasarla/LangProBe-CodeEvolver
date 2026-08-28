from __future__ import annotations

import atexit
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import requests

from ._tracing import traceable


class AppWorldClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    @traceable("tool", name="appworld.request", max_attr_chars=12_000)
    def request(self, method: str, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        response = requests.request(
            method,
            f"{self.base_url}{endpoint}",
            json=data,
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("output", payload)

    def initialize(self, experiment_name: str, task_id: str) -> Any:
        return self.request(
            "POST",
            "/initialize",
            {"experiment_name": experiment_name, "task_id": task_id},
        )

    def execute(self, task_id: str, code: str) -> str:
        return str(self.request("POST", "/execute", {"task_id": task_id, "code": code}))

    def task_completed(self, task_id: str) -> bool:
        return bool(self.request("POST", "/task_completed", {"task_id": task_id}))

    def evaluate(self, task_id: str) -> dict[str, Any]:
        result = self.request(
            "POST",
            "/evaluate",
            {"task_id": task_id, "suppress_errors": True, "report": False},
        )
        return dict(result)

    def close(self, task_id: str) -> None:
        self.request("POST", "/close", {"task_id": task_id})

    def show_task(self, task_id: str) -> dict[str, Any]:
        return dict(self.request("GET", f"/tasks/{task_id}"))


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
                self.client.request("GET", "/")
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
