#!/usr/bin/env python3
"""Keep long-lived ALFWorld HTTP workers alive across evaluator subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Deque


@dataclass
class WorkerSlot:
    port: int
    process: subprocess.Popen[str] | None = None
    log_handle: IO[str] | None = None
    started_at: float = 0.0
    restart_times: Deque[float] = field(default_factory=deque)
    total_starts: int = 0


class WorkerSupervisor:
    def __init__(
        self,
        *,
        repo_root: Path,
        python: Path,
        base_port: int,
        count: int,
        log_dir: Path,
        max_restarts: int,
        restart_window_seconds: float,
    ) -> None:
        self.repo_root = repo_root.resolve()
        # A virtualenv's bin/python is commonly a symlink to its base
        # interpreter. Resolving that symlink discards the virtualenv prefix
        # and therefore all packages installed in the ALFWorld runtime.
        self.python = Path(os.path.abspath(python))
        self.log_dir = log_dir.resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_restarts = max_restarts
        self.restart_window_seconds = restart_window_seconds
        self.slots = [WorkerSlot(base_port + index) for index in range(count)]
        self.stop_requested = threading.Event()
        self.status_path = self.log_dir / "alfworld-supervisor-status.json"

    def worker_command(self, slot: WorkerSlot) -> list[str]:
        return [
            str(self.python),
            str(self.repo_root / "scripts" / "alfworld_server.py"),
            "--port",
            str(slot.port),
            "--repo-root",
            str(self.repo_root),
        ]

    def _write_status(self, state: str, error: str | None = None) -> None:
        payload = {
            "state": state,
            "supervisor_pid": os.getpid(),
            "error": error,
            "updated_at_unix": time.time(),
            "workers": [
                {
                    "port": slot.port,
                    "pid": slot.process.pid if slot.process is not None else None,
                    "returncode": slot.process.poll() if slot.process is not None else None,
                    "total_starts": slot.total_starts,
                    "started_at_monotonic": slot.started_at,
                }
                for slot in self.slots
            ],
        }
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.status_path)

    def _spawn(self, slot: WorkerSlot) -> None:
        now = time.monotonic()
        while slot.restart_times and now - slot.restart_times[0] > self.restart_window_seconds:
            slot.restart_times.popleft()
        if slot.total_starts and len(slot.restart_times) >= self.max_restarts:
            raise RuntimeError(
                f"worker {slot.port} exceeded {self.max_restarts} restarts "
                f"in {self.restart_window_seconds:g} seconds"
            )
        if slot.log_handle is not None:
            slot.log_handle.close()
        slot.log_handle = (self.log_dir / f"alfworld-{slot.port}.log").open(
            "a", encoding="utf-8"
        )
        slot.log_handle.write(
            f"\n--- supervisor start {slot.total_starts + 1} at {time.time():.3f} ---\n"
        )
        slot.log_handle.flush()
        environment = os.environ.copy()
        environment.setdefault(
            "ALFWORLD_DATA", str(self.repo_root / ".runtime" / "alfworld-data")
        )
        slot.process = subprocess.Popen(
            self.worker_command(slot),
            cwd=self.repo_root,
            env=environment,
            stdout=slot.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name == "posix",
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
        slot.started_at = now
        slot.restart_times.append(now)
        slot.total_starts += 1
        print(
            f"started ALFWorld worker port={slot.port} pid={slot.process.pid} "
            f"start={slot.total_starts}",
            flush=True,
        )

    @staticmethod
    def _request_shutdown(port: int) -> None:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/shutdown",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10):
                pass
        except Exception:
            pass

    @staticmethod
    def _signal_group(process: subprocess.Popen[str], force: bool) -> None:
        if os.name == "posix":
            try:
                # Every worker is started with start_new_session=True, so its
                # PID is also its process-group ID. Use that known PGID rather
                # than os.getpgid(): the Python leader may already have exited
                # while its Unity child remains alive.
                os.killpg(
                    process.pid,
                    signal.SIGKILL if force else signal.SIGTERM,
                )
                return
            except (ProcessLookupError, PermissionError):
                pass
        if process.poll() is not None:
            return
        if force:
            process.kill()
        else:
            process.terminate()

    def _stop_slot(self, slot: WorkerSlot) -> None:
        process = slot.process
        if process is None:
            return
        if process.poll() is None:
            self._request_shutdown(slot.port)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._signal_group(process, force=False)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._signal_group(process, force=True)
                    process.wait(timeout=5)
        slot.process = None
        if slot.log_handle is not None:
            slot.log_handle.close()
            slot.log_handle = None

    def run(self) -> None:
        failure: str | None = None
        try:
            for slot in self.slots:
                self._spawn(slot)
            self._write_status("running")
            while not self.stop_requested.wait(0.5):
                changed = False
                for slot in self.slots:
                    process = slot.process
                    if process is None or process.poll() is None:
                        continue
                    returncode = process.returncode
                    print(
                        f"ALFWorld worker port={slot.port} pid={process.pid} "
                        f"exited returncode={returncode}",
                        flush=True,
                    )
                    # A watchdog uses os._exit so the Python worker cannot
                    # clean up Unity itself. Reap the whole old session before
                    # replacement to avoid recreating the original GLX leak.
                    self._signal_group(process, force=True)
                    if slot.log_handle is not None:
                        slot.log_handle.close()
                        slot.log_handle = None
                    slot.process = None
                    if self.stop_requested.wait(min(5.0, 0.25 * slot.total_starts)):
                        break
                    self._spawn(slot)
                    changed = True
                if changed:
                    self._write_status("running")
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            self._write_status("failed", failure)
            raise
        finally:
            self.stop_requested.set()
            for slot in self.slots:
                self._stop_slot(slot)
            self._write_status("failed" if failure else "stopped", failure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.environ.get("ALFWORLD_PYTHON", sys.executable)),
    )
    parser.add_argument(
        "--base-port", type=int, default=int(os.environ.get("ALFWORLD_BASE_PORT", "19123"))
    )
    parser.add_argument(
        "--count", type=int, default=int(os.environ.get("ALFWORLD_SERVER_COUNT", "2"))
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=int(os.environ.get("ALFWORLD_MAX_RESTARTS", "20")),
    )
    parser.add_argument(
        "--restart-window-seconds",
        type=float,
        default=float(os.environ.get("ALFWORLD_RESTART_WINDOW_SECONDS", "600")),
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.max_restarts < 1:
        parser.error("--max-restarts must be at least 1")

    supervisor = WorkerSupervisor(
        repo_root=args.repo_root,
        python=args.python,
        base_port=args.base_port,
        count=args.count,
        log_dir=args.log_dir,
        max_restarts=args.max_restarts,
        restart_window_seconds=args.restart_window_seconds,
    )

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        supervisor.stop_requested.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    supervisor.run()


if __name__ == "__main__":
    main()
