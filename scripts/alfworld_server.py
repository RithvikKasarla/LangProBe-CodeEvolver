from __future__ import annotations

import argparse
import base64
import io
import json
import os
import signal
import sys
import threading
import time
import traceback
import types
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from PIL import Image


def install_oracle_only_import_stubs() -> None:
    textworld = types.ModuleType("textworld")
    textworld.__path__ = []
    logic = types.ModuleType("textworld.logic")

    class Variable:
        def __init__(self, name: str, type: str) -> None:
            self.name = name
            self.type = type

    class Proposition:
        def __init__(self, name: str, arguments: list[Any]) -> None:
            self.name = name
            self.arguments = arguments

    class Agent:
        pass

    textworld.Agent = Agent
    logic.Variable = Variable
    logic.Proposition = Proposition
    textworld.logic = logic
    sys.modules["textworld"] = textworld
    sys.modules["textworld.logic"] = logic

    detector_name = "alfworld.agents.detector.mrcnn"
    detector = types.ModuleType(detector_name)
    detector.load_pretrained_model = lambda *args, **kwargs: None
    sys.modules[detector_name] = detector
    for module_name, class_name in (
        ("alfworld.agents.controller.mrcnn", "MaskRCNNAgent"),
        ("alfworld.agents.controller.mrcnn_astar", "MaskRCNNAStarAgent"),
    ):
        module = types.ModuleType(module_name)
        setattr(module, class_name, type(class_name, (), {}))
        sys.modules[module_name] = module


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class WorkerBusyError(RuntimeError):
    """Raised when another evaluator owns this worker's episode lease."""


class InvalidSessionError(RuntimeError):
    """Raised when a step or release does not own the active episode."""


class FatalWorkerError(RuntimeError):
    """Raised after a controller/rendering fault requires process replacement."""


class EnvironmentState:
    """One long-lived ALFWorld/Unity controller with serialized episode leases."""

    def __init__(
        self,
        repo_root: Path,
        environment_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.repo_root = repo_root
        self.env: Any = None
        self._environment_factory = environment_factory
        self._lock = threading.RLock()
        self._service_id = uuid.uuid4().hex
        self._session_id: Optional[str] = None
        self._session_last_activity = 0.0
        self._session_ttl = max(
            60.0, float(os.environ.get("ALFWORLD_SESSION_TTL_SECONDS", "1800"))
        )
        self._fatal_error: Optional[str] = None
        self._controller_starts = 0
        self._episode_resets = 0
        self._episode_steps = 0

    @property
    def service_id(self) -> str:
        # Immutable and deliberately lock-free so /live remains responsive
        # while the renderer is inside a long reset or action.
        return self._service_id

    def _operation_watchdog(self, operation: str) -> threading.Timer:
        timeout = max(
            30.0,
            float(os.environ.get("ALFWORLD_OPERATION_TIMEOUT_SECONDS", "300")),
        )

        def abort_hung_worker() -> None:
            print(
                f"fatal: ALFWorld {operation} exceeded {timeout:g} seconds; "
                "exiting for supervisor replacement",
                flush=True,
            )
            os._exit(70)

        timer = threading.Timer(timeout, abort_hung_worker)
        timer.daemon = True
        timer.start()
        return timer

    def _worker_locked(self) -> Any:
        if self.env is None or not getattr(self.env, "envs", None):
            raise FatalWorkerError("ALFWorld controller is not initialized")
        return self.env.envs[0]

    @staticmethod
    def _process_alive(pid: Any) -> bool:
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except (OSError, TypeError, ValueError):
            return False
        return True

    def _controller_readiness_locked(self) -> tuple[bool, str]:
        if self._fatal_error:
            return False, self._fatal_error
        if self.env is None:
            return False, "controller has not started"
        try:
            worker = self._worker_locked()
            if hasattr(worker, "is_alive") and not worker.is_alive():
                return False, "ALFWorld worker thread exited"
            controller = worker.env
            server_thread = getattr(controller, "server_thread", None)
            if server_thread is None or not server_thread.is_alive():
                return False, "AI2-THOR server thread exited"
            unity_pid = getattr(controller, "unity_pid", None)
            container_id = getattr(controller, "container_id", None)
            if not container_id and not self._process_alive(unity_pid):
                return False, "Unity process exited"
            event = getattr(controller, "last_event", None)
            frame = getattr(event, "frame", None)
            if frame is None or tuple(getattr(frame, "shape", ())) != (300, 300, 3):
                return False, "AI2-THOR has no valid 300x300 RGB frame"
            return True, "ready"
        except Exception as exc:  # noqa: BLE001 - readiness must never escape
            return False, f"controller readiness failed: {type(exc).__name__}: {exc}"

    def _expire_session_locked(self) -> None:
        if (
            self._session_id is not None
            and time.monotonic() - self._session_last_activity > self._session_ttl
        ):
            print(
                f"expiring stale ALFWorld session {self._session_id}",
                flush=True,
            )
            self._session_id = None
            self._session_last_activity = 0.0

    def _require_session_locked(self, session_id: str) -> None:
        self._expire_session_locked()
        if not session_id or session_id != self._session_id:
            raise InvalidSessionError("request does not own the active ALFWorld session")
        self._session_last_activity = time.monotonic()

    def _close_controller_locked(self) -> None:
        environment = self.env
        self.env = None
        self._session_id = None
        self._session_last_activity = 0.0
        if environment is not None:
            try:
                environment.close()
            except Exception:  # noqa: BLE001 - preserve the original fatal error
                traceback.print_exc()

    def _latch_fatal_locked(self, exc: BaseException) -> None:
        if self._fatal_error is None:
            self._fatal_error = f"{type(exc).__name__}: {exc}"
        self._close_controller_locked()

    def _load_config(self, task_file: Path) -> dict[str, Any]:
        with (self.repo_root / "configs" / "alfworld_vision.yaml").open() as file:
            config = yaml.safe_load(file) or {}
        dataset = config.setdefault("dataset", {})
        dataset["eval_ood_data_path"] = str(task_file.parent)
        dataset["num_eval_games"] = 1
        return config

    def _new_environment(self, config: dict[str, Any]) -> Any:
        if self._environment_factory is not None:
            return self._environment_factory(
                config, train_eval="eval_out_of_distribution"
            )

        display = os.environ.get("DISPLAY", "")
        import alfworld.gen.constants as constants

        if display:
            constants.X_DISPLAY = display.lstrip(":").split(".", 1)[0]
        install_oracle_only_import_stubs()
        from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv

        return AlfredThorEnv(config, train_eval="eval_out_of_distribution")

    def _ensure_controller_locked(self, task_file: Path) -> None:
        if self._fatal_error:
            raise FatalWorkerError(self._fatal_error)
        if self.env is not None:
            return

        environment = self._new_environment(self._load_config(task_file))
        # Assign before init_env so a partially started controller is still
        # available for best-effort cleanup if Unity initialization raises.
        self.env = environment
        environment.init_env(1)
        self._controller_starts += 1

    @staticmethod
    def _result(worker: Any) -> tuple[str, bool, dict[str, Any]]:
        feedback, done, actions, won, goal_condition, expert_actions = (
            worker.get_results()
        )
        info = {
            "admissible_commands": [actions],
            "won": [won],
            "goal_condition_success_rate": [goal_condition],
            "extra.gamefile": [worker.traj_root],
            "extra.expert_plan": [expert_actions],
        }
        return feedback, bool(done), info

    def shutdown(self) -> None:
        with self._lock:
            self._close_controller_locked()

    def frame_data_url(self) -> str:
        frame = self._worker_locked().env.last_event.frame
        if tuple(getattr(frame, "shape", ())) != (300, 300, 3):
            raise RuntimeError(
                f"AI2-THOR returned invalid frame shape {getattr(frame, 'shape', None)}"
            )
        image = Image.fromarray(frame).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def initialize(self, game_file: str) -> dict[str, Any]:
        # Do not let concurrent evaluator processes queue behind a slow Unity
        # reset. Return busy immediately so their pools can try another port.
        if not self._lock.acquire(blocking=False):
            raise WorkerBusyError("ALFWorld worker is processing another request")
        try:
            data_root = Path(os.environ["ALFWORLD_DATA"]).resolve()
            task_file = (data_root / game_file).resolve()
            if data_root not in task_file.parents or not task_file.is_file():
                raise ValueError(f"Invalid ALFWorld task path: {game_file}")
            if self._fatal_error:
                raise FatalWorkerError(self._fatal_error)
            self._expire_session_locked()
            if self._session_id is not None:
                raise WorkerBusyError("ALFWorld worker already has an active episode")

            try:
                self._ensure_controller_locked(task_file)
                # Call the persistent inner worker directly. The outer
                # AlfredThorEnv queue thread is intentionally kept constant and
                # is not recreated per episode; direct calls also propagate
                # reset exceptions to this request instead of losing them in a
                # daemon thread.
                worker = self._worker_locked()
                watchdog = self._operation_watchdog("reset")
                try:
                    worker.reset(str(task_file))
                finally:
                    watchdog.cancel()
                observation, _, info = self._result(worker)
                ready, reason = self._controller_readiness_locked()
                if not ready:
                    raise RuntimeError(reason)
                session_id = uuid.uuid4().hex
                self._session_id = session_id
                self._session_last_activity = time.monotonic()
                self._episode_resets += 1
                return {
                    "observation": observation,
                    "info": jsonable(info),
                    "frame": self.frame_data_url(),
                    "session_id": session_id,
                    "service_id": self._service_id,
                    "unity_pid": getattr(worker.env, "unity_pid", None),
                    "episode_reset": self._episode_resets,
                }
            except (WorkerBusyError, InvalidSessionError, FatalWorkerError, ValueError):
                raise
            except Exception as exc:
                self._latch_fatal_locked(exc)
                raise
        finally:
            self._lock.release()

    def step(self, action: str, session_id: str) -> dict[str, Any]:
        with self._lock:
            self._require_session_locked(session_id)
            try:
                worker = self._worker_locked()
                watchdog = self._operation_watchdog("step")
                try:
                    worker.step(action)
                finally:
                    watchdog.cancel()
                observation, done, info = self._result(worker)
                ready, reason = self._controller_readiness_locked()
                if not ready:
                    raise RuntimeError(reason)
                self._episode_steps += 1
                return {
                    "observation": observation,
                    "done": done,
                    "info": jsonable(info),
                    "frame": self.frame_data_url(),
                    "session_id": session_id,
                }
            except (InvalidSessionError, FatalWorkerError):
                raise
            except Exception as exc:
                self._latch_fatal_locked(exc)
                raise

    def release(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            self._require_session_locked(session_id)
            self._session_id = None
            self._session_last_activity = 0.0
            return {"released": True, "controller_persistent": self.env is not None}

    def readiness(self) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            # Readiness is a traffic gate and must remain fast even while a
            # software-rendered reset takes tens of seconds. Liveness remains
            # independently available at /live.
            return {
                "service_id": self._service_id,
                "ready": False,
                "reason": "worker is processing an episode request",
                "fatal_error": self._fatal_error,
                "session_active": self._session_id is not None,
                "controller_starts": self._controller_starts,
                "episode_resets": self._episode_resets,
                "episode_steps": self._episode_steps,
                "unity_pid": None,
                "thor_thread_alive": None,
                "python_thread_count": threading.active_count(),
                "rss_kib": None,
                "fd_count": None,
            }
        try:
            ready, reason = self._controller_readiness_locked()
            return {**self._status_locked(), "ready": ready, "reason": reason}
        finally:
            self._lock.release()

    def _status_locked(self) -> dict[str, Any]:
        worker = None
        controller = None
        if self.env is not None and getattr(self.env, "envs", None):
            worker = self.env.envs[0]
            controller = getattr(worker, "env", None)
        ready, _ = self._controller_readiness_locked()
        try:
            rss_kib = int(
                next(
                    line.split()[1]
                    for line in Path("/proc/self/status").read_text().splitlines()
                    if line.startswith("VmRSS:")
                )
            )
        except (OSError, StopIteration, ValueError):
            rss_kib = None
        try:
            fd_count = len(list(Path("/proc/self/fd").iterdir()))
        except OSError:
            fd_count = None
        return {
            "service_id": self._service_id,
            "ready": ready,
            "fatal_error": self._fatal_error,
            "session_active": self._session_id is not None,
            "controller_starts": self._controller_starts,
            "episode_resets": self._episode_resets,
            "episode_steps": self._episode_steps,
            "unity_pid": getattr(controller, "unity_pid", None),
            "thor_thread_alive": bool(worker and worker.is_alive()),
            "python_thread_count": threading.active_count(),
            "rss_kib": rss_kib,
            "fd_count": fd_count,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()


class Handler(BaseHTTPRequestHandler):
    state: EnvironmentState

    def log_message(self, format: str, *args: Any) -> None:
        print(format % args, flush=True)

    def respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/live", "/health"):
            self.respond(
                200,
                {
                    "status": "success",
                    "output": {
                        "alive": True,
                        "service_id": self.state.service_id,
                    },
                },
            )
            return
        if self.path == "/status":
            self.respond(200, {"status": "success", "output": self.state.status()})
            return
        if self.path == "/ready":
            output = self.state.readiness()
            self.respond(
                200 if output["ready"] else 503,
                {
                    "status": "success" if output["ready"] else "error",
                    "output": output,
                    "error": None if output["ready"] else output["reason"],
                },
            )
            return
        else:
            self.respond(404, {"status": "error", "error": "not found"})

    def _schedule_server_stop(self) -> None:
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/initialize":
                output = self.state.initialize(str(data["game_file"]))
            elif self.path == "/step":
                output = self.state.step(
                    str(data["action"]), str(data.get("session_id", ""))
                )
            elif self.path == "/close":
                output = self.state.release(str(data.get("session_id", "")))
            elif self.path == "/shutdown":
                self.state.shutdown()
                output = {"shutdown": True}
            else:
                self.respond(404, {"status": "error", "error": "not found"})
                return
            self.respond(200, {"status": "success", "output": output})
            if self.path == "/shutdown":
                self._schedule_server_stop()
        except WorkerBusyError as exc:
            self.respond(409, {"status": "busy", "error": str(exc)})
        except InvalidSessionError as exc:
            self.respond(409, {"status": "invalid_session", "error": str(exc)})
        except FatalWorkerError as exc:
            self.respond(503, {"status": "fatal", "error": str(exc)})
            self._schedule_server_stop()
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self.respond(400, {"status": "error", "error": str(exc)})
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            self.respond(500, {"status": "error", "error": error})
            if self.state.status()["fatal_error"]:
                self._schedule_server_stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    Handler.state = EnvironmentState(args.repo_root.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True

    def request_shutdown(signum: int, frame: Any) -> None:
        del signum, frame
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        server.serve_forever()
    finally:
        Handler.state.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
