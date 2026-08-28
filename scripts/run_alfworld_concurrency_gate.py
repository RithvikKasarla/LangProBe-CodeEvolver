#!/usr/bin/env python3
"""Validate N persistent ALFWorld/Unity workers on one shared display."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


class WorkerRequestError(RuntimeError):
    pass


def request(
    base_url: str,
    method: str,
    endpoint: str,
    data: dict[str, Any] | None = None,
    timeout: float = 300,
) -> dict[str, Any]:
    encoded = None if data is None else json.dumps(data).encode("utf-8")
    http_request = urllib.request.Request(
        f"{base_url.rstrip('/')}{endpoint}",
        data=encoded,
        method=method,
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("error") or body
        except json.JSONDecodeError:
            detail = body
        raise WorkerRequestError(
            f"{method} {endpoint}: HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise WorkerRequestError(f"{method} {endpoint}: {exc}") from exc
    if payload.get("status") != "success":
        raise WorkerRequestError(
            f"{method} {endpoint}: {payload.get('status')}: {payload.get('error')}"
        )
    output = payload.get("output")
    if not isinstance(output, dict):
        raise WorkerRequestError(f"{method} {endpoint}: output is not an object")
    return output


def validate_frame(data_url: str) -> tuple[int, str]:
    prefix = "data:image/jpeg;base64,"
    if not data_url.startswith(prefix):
        raise AssertionError("frame is not a JPEG data URL")
    frame = base64.b64decode(data_url[len(prefix) :], validate=True)
    if len(frame) < 1_000 or not frame.startswith(b"\xff\xd8") or not frame.endswith(b"\xff\xd9"):
        raise AssertionError(f"invalid JPEG frame ({len(frame)} bytes)")
    return len(frame), hashlib.sha256(frame).hexdigest()


def choose_visual_action(commands: list[str]) -> tuple[str, bool]:
    movement_prefixes = (
        "go to ",
        "rotate right",
        "rotate left",
        "look up",
        "look down",
    )
    for prefix in movement_prefixes:
        for command in commands:
            if command.casefold().startswith(prefix):
                return command, True
    if "look" in commands:
        return "look", False
    if not commands:
        raise AssertionError("worker returned no admissible commands")
    return commands[0], False


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-port", type=int, default=19123)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--episodes-per-worker", type=int, default=20)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=32)
    parser.add_argument("--stagger-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or args.episodes_per_worker < 1 or args.task_count < 1:
        parser.error("workers, episodes-per-worker, and task-count must be positive")

    task_rows = json.loads(args.tasks.read_text(encoding="utf-8"))
    game_files = [str(row["game_file"]) for row in task_rows[: args.task_count]]
    if not game_files:
        raise SystemExit("task list is empty")
    base_urls = [
        f"http://127.0.0.1:{args.base_port + index}"
        for index in range(args.workers)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(".summary.json")
    output_lock = threading.Lock()
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    identities: dict[int, dict[str, Any]] = {}

    # Start controllers one at a time. This avoids conflating shared-display
    # concurrency with first-download/cache extraction races.
    for worker_index, base_url in enumerate(base_urls):
        session_id: str | None = None
        try:
            initial = request(
                base_url,
                "POST",
                "/initialize",
                {"game_file": game_files[worker_index % len(game_files)]},
                timeout=600,
            )
            session_id = str(initial["session_id"])
            validate_frame(str(initial["frame"]))
            request(
                base_url,
                "POST",
                "/close",
                {"session_id": session_id},
                timeout=30,
            )
            session_id = None
            status = request(base_url, "GET", "/ready", timeout=10)
            identity = {
                "service_id": str(initial["service_id"]),
                "unity_pid": int(initial["unity_pid"]),
            }
            if int(status["controller_starts"]) != 1:
                raise AssertionError(f"worker started multiple controllers: {status}")
            if not status["ready"] or not status["thor_thread_alive"]:
                raise AssertionError(f"worker did not become renderer-ready: {status}")
            identities[worker_index] = identity
            print(
                f"bootstrap worker={worker_index} port={args.base_port + worker_index} "
                f"pid={identity['unity_pid']}",
                flush=True,
            )
            if args.stagger_seconds:
                time.sleep(args.stagger_seconds)
        except Exception as exc:
            if session_id is not None:
                try:
                    request(
                        base_url,
                        "POST",
                        "/close",
                        {"session_id": session_id},
                        timeout=30,
                    )
                except Exception:
                    pass
            errors.append(
                f"bootstrap worker {worker_index}: {type(exc).__name__}: {exc}"
            )
            break

    barrier = threading.Barrier(args.workers)
    stop_requested = threading.Event()

    def run_worker(worker_index: int) -> dict[str, Any]:
        base_url = base_urls[worker_index]
        identity = identities[worker_index]
        worker_records: list[dict[str, Any]] = []
        frame_changes = 0
        visual_actions = 0
        try:
            barrier.wait(timeout=120)
            for episode in range(args.episodes_per_worker):
                if stop_requested.is_set():
                    raise RuntimeError("another worker failed")
                game_file = game_files[
                    (worker_index * args.episodes_per_worker + episode)
                    % len(game_files)
                ]
                session_id: str | None = None
                episode_started = time.monotonic()
                try:
                    initial = request(
                        base_url,
                        "POST",
                        "/initialize",
                        {"game_file": game_file},
                        timeout=600,
                    )
                    session_id = str(initial["session_id"])
                    initial_bytes, initial_hash = validate_frame(str(initial["frame"]))
                    if str(initial["service_id"]) != identity["service_id"]:
                        raise AssertionError("worker service process changed")
                    if int(initial["unity_pid"]) != identity["unity_pid"]:
                        raise AssertionError("Unity PID changed")
                    commands = list(initial["info"]["admissible_commands"][0])
                    action, visual_action = choose_visual_action(commands)
                    stepped = request(
                        base_url,
                        "POST",
                        "/step",
                        {"action": action, "session_id": session_id},
                        timeout=300,
                    )
                    final_bytes, final_hash = validate_frame(str(stepped["frame"]))
                    changed = initial_hash != final_hash
                    if visual_action:
                        visual_actions += 1
                        frame_changes += int(changed)
                    request(
                        base_url,
                        "POST",
                        "/close",
                        {"session_id": session_id},
                        timeout=30,
                    )
                    session_id = None
                    status = request(base_url, "GET", "/ready", timeout=10)
                    if not status["ready"] or status["fatal_error"]:
                        raise AssertionError(f"worker became unready: {status}")
                    if status["session_active"]:
                        raise AssertionError("episode lease remained active")
                    if int(status["controller_starts"]) != 1:
                        raise AssertionError("controller was recreated")
                    if int(status["unity_pid"]) != identity["unity_pid"]:
                        raise AssertionError("status Unity PID changed")
                    record = {
                        "worker": worker_index,
                        "port": args.base_port + worker_index,
                        "episode": episode + 1,
                        "game_file": game_file,
                        "duration_seconds": round(
                            time.monotonic() - episode_started, 3
                        ),
                        "action": action,
                        "visual_action": visual_action,
                        "frame_changed": changed,
                        "initial_frame_bytes": initial_bytes,
                        "final_frame_bytes": final_bytes,
                        "initial_frame_sha256": initial_hash,
                        "final_frame_sha256": final_hash,
                        "service_id": identity["service_id"],
                        "unity_pid": identity["unity_pid"],
                        "controller_starts": status["controller_starts"],
                        "episode_resets": status["episode_resets"],
                        "episode_steps": status["episode_steps"],
                        "thor_thread_alive": status["thor_thread_alive"],
                        "python_thread_count": status["python_thread_count"],
                        "rss_kib": status["rss_kib"],
                        "fd_count": status["fd_count"],
                    }
                    worker_records.append(record)
                    with output_lock:
                        records.append(record)
                        output_file.write(json.dumps(record, sort_keys=True) + "\n")
                        output_file.flush()
                finally:
                    if session_id is not None:
                        try:
                            request(
                                base_url,
                                "POST",
                                "/close",
                                {"session_id": session_id},
                                timeout=30,
                            )
                        except Exception:
                            pass
            return {
                "worker": worker_index,
                "records": worker_records,
                "visual_actions": visual_actions,
                "frame_changes": frame_changes,
            }
        except Exception:
            stop_requested.set()
            try:
                barrier.abort()
            except threading.BrokenBarrierError:
                pass
            raise

    worker_results: list[dict[str, Any]] = []
    started = time.monotonic()
    if len(identities) == args.workers and not errors:
        with args.output.open("w", encoding="utf-8") as output_file:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(run_worker, worker_index): worker_index
                    for worker_index in range(args.workers)
                }
                for future in as_completed(futures):
                    worker_index = futures[future]
                    try:
                        worker_results.append(future.result())
                    except Exception as exc:
                        errors.append(
                            f"worker {worker_index}: {type(exc).__name__}: {exc}"
                        )

    final_statuses: list[dict[str, Any]] = []
    for worker_index, base_url in enumerate(base_urls):
        try:
            final_statuses.append(
                {
                    "worker": worker_index,
                    "port": args.base_port + worker_index,
                    **request(base_url, "GET", "/status", timeout=10),
                }
            )
        except Exception as exc:
            errors.append(
                f"final status worker {worker_index}: {type(exc).__name__}: {exc}"
            )

    worker_summaries: list[dict[str, Any]] = []
    for worker_index in range(args.workers):
        worker_records = sorted(
            (record for record in records if record["worker"] == worker_index),
            key=lambda record: record["episode"],
        )
        result = next(
            (item for item in worker_results if item["worker"] == worker_index),
            {"visual_actions": 0, "frame_changes": 0},
        )
        durations = [float(record["duration_seconds"]) for record in worker_records]
        rss_values = [
            int(record["rss_kib"])
            for record in worker_records
            if record["rss_kib"] is not None
        ]
        fd_values = [
            int(record["fd_count"])
            for record in worker_records
            if record["fd_count"] is not None
        ]
        thread_values = [int(record["python_thread_count"]) for record in worker_records]
        worker_summaries.append(
            {
                "worker": worker_index,
                **identities.get(worker_index, {}),
                "completed_episodes": len(worker_records),
                "visual_actions": int(result["visual_actions"]),
                "changed_visual_frames": int(result["frame_changes"]),
                "duration_seconds": {
                    "mean": round(statistics.mean(durations), 3) if durations else None,
                    "p95": percentile(durations, 0.95),
                    "max": max(durations) if durations else None,
                },
                "rss_kib": {
                    "min": min(rss_values) if rss_values else None,
                    "max": max(rss_values) if rss_values else None,
                },
                "fd_count": {
                    "min": min(fd_values) if fd_values else None,
                    "max": max(fd_values) if fd_values else None,
                },
                "python_thread_count": {
                    "min": min(thread_values) if thread_values else None,
                    "max": max(thread_values) if thread_values else None,
                },
            }
        )

    service_ids = [identity["service_id"] for identity in identities.values()]
    unity_pids = [identity["unity_pid"] for identity in identities.values()]
    expected_records = args.workers * args.episodes_per_worker
    invariants = {
        "all_workers_bootstrapped": len(identities) == args.workers,
        "all_episodes_completed": len(records) == expected_records,
        "unique_service_processes": len(service_ids) == args.workers
        and len(set(service_ids)) == args.workers,
        "unique_unity_processes": len(unity_pids) == args.workers
        and len(set(unity_pids)) == args.workers,
        "one_controller_per_worker": len(final_statuses) == args.workers
        and all(int(status["controller_starts"]) == 1 for status in final_statuses),
        "exact_reset_counts": len(final_statuses) == args.workers
        and all(
            int(status["episode_resets"]) == args.episodes_per_worker + 1
            for status in final_statuses
        ),
        "exact_step_counts": len(final_statuses) == args.workers
        and all(
            int(status["episode_steps"]) == args.episodes_per_worker
            for status in final_statuses
        ),
        "all_workers_ready": len(final_statuses) == args.workers
        and all(
            status["ready"]
            and not status["fatal_error"]
            and not status["session_active"]
            and status["thor_thread_alive"]
            for status in final_statuses
        ),
        "changing_frames_per_worker": len(worker_summaries) == args.workers
        and all(
            summary["visual_actions"] > 0
            and summary["changed_visual_frames"] > 0
            for summary in worker_summaries
        ),
        "thread_counts_stable": len(worker_summaries) == args.workers
        and all(
            summary["python_thread_count"]["min"] is not None
            and summary["python_thread_count"]["max"]
            - summary["python_thread_count"]["min"]
            <= 2
            for summary in worker_summaries
        ),
        "fd_counts_stable": len(worker_summaries) == args.workers
        and all(
            summary["fd_count"]["min"] is not None
            and summary["fd_count"]["max"] - summary["fd_count"]["min"] <= 4
            for summary in worker_summaries
        ),
        "no_errors": not errors,
    }
    summary = {
        "passed": all(invariants.values()),
        "workers": args.workers,
        "episodes_per_worker": args.episodes_per_worker,
        "completed_episodes": len(records),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "task_count": len(game_files),
        "service_ids": service_ids,
        "unity_pids": unity_pids,
        "invariants": invariants,
        "errors": errors,
        "worker_summaries": worker_summaries,
        "final_statuses": final_statuses,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
