#!/usr/bin/env python3
"""Exercise repeated ALFWorld resets without model calls or evaluator code."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class WorkerRequestError(RuntimeError):
    pass


def request(
    base_url: str,
    method: str,
    endpoint: str,
    data: dict[str, Any] | None = None,
    timeout: float = 900,
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
            payload = json.loads(body)
            detail = payload.get("error") or body
        except json.JSONDecodeError:
            detail = body
        raise WorkerRequestError(f"{method} {endpoint}: HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise WorkerRequestError(f"{method} {endpoint}: {exc}") from exc
    if payload.get("status") != "success":
        raise WorkerRequestError(
            f"{method} {endpoint}: {payload.get('status')}: {payload.get('error')}"
        )
    output = payload.get("output")
    if not isinstance(output, dict):
        raise WorkerRequestError(f"{method} {endpoint}: response output is not an object")
    return output


def validate_frame(data_url: str) -> tuple[int, str]:
    prefix = "data:image/jpeg;base64,"
    if not data_url.startswith(prefix):
        raise AssertionError("worker frame is not a JPEG data URL")
    frame = base64.b64decode(data_url[len(prefix) :], validate=True)
    if len(frame) < 1_000 or not frame.startswith(b"\xff\xd8") or not frame.endswith(b"\xff\xd9"):
        raise AssertionError(f"worker returned an invalid JPEG ({len(frame)} bytes)")
    return len(frame), hashlib.sha256(frame).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:19123")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--task-count", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rss-growth-mib", type=float, default=1024.0)
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 0 or args.task_count < 1:
        parser.error("episodes/task-count must be positive and steps cannot be negative")

    rows = json.loads(args.tasks.read_text(encoding="utf-8"))
    game_files = [str(row["game_file"]) for row in rows[: args.task_count]]
    if not game_files:
        raise SystemExit("task list is empty")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(".summary.json")
    start_status = request(args.base_url, "GET", "/status", timeout=5)
    start_resets = int(start_status.get("episode_resets") or 0)
    service_ids: set[str] = set()
    unity_pids: set[int] = set()
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    started = time.monotonic()

    with args.output.open("w", encoding="utf-8") as output:
        for episode in range(args.episodes):
            game_file = game_files[episode % len(game_files)]
            session_id: str | None = None
            episode_started = time.monotonic()
            try:
                initial = request(
                    args.base_url,
                    "POST",
                    "/initialize",
                    {"game_file": game_file},
                )
                session_id = str(initial["session_id"])
                frame_bytes, frame_sha256 = validate_frame(str(initial["frame"]))
                service_ids.add(str(initial["service_id"]))
                unity_pid = int(initial["unity_pid"])
                unity_pids.add(unity_pid)

                info = initial["info"]
                steps_completed = 0
                for _ in range(args.steps):
                    admissible = list(info["admissible_commands"][0])
                    if not admissible:
                        raise AssertionError("worker returned no admissible actions")
                    action = "look" if "look" in admissible else admissible[0]
                    stepped = request(
                        args.base_url,
                        "POST",
                        "/step",
                        {"action": action, "session_id": session_id},
                    )
                    validate_frame(str(stepped["frame"]))
                    info = stepped["info"]
                    steps_completed += 1
                    if bool(stepped["done"]):
                        break

                request(
                    args.base_url,
                    "POST",
                    "/close",
                    {"session_id": session_id},
                    timeout=30,
                )
                session_id = None
                status = request(args.base_url, "GET", "/ready", timeout=5)
                if not status.get("ready"):
                    raise AssertionError(f"worker became unready: {status.get('reason')}")
                if status.get("session_active"):
                    raise AssertionError("episode lease remained active after release")
                record = {
                    "episode": episode + 1,
                    "game_file": game_file,
                    "duration_seconds": round(time.monotonic() - episode_started, 3),
                    "steps": steps_completed,
                    "service_id": initial["service_id"],
                    "unity_pid": unity_pid,
                    "controller_starts": status["controller_starts"],
                    "episode_resets": status["episode_resets"],
                    "episode_steps_total": status["episode_steps"],
                    "thor_thread_alive": status["thor_thread_alive"],
                    "python_thread_count": status["python_thread_count"],
                    "rss_kib": status["rss_kib"],
                    "fd_count": status["fd_count"],
                    "frame_bytes": frame_bytes,
                    "frame_sha256": frame_sha256,
                }
                records.append(record)
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                if (episode + 1) % 10 == 0 or episode == 0:
                    print(
                        f"episode={episode + 1}/{args.episodes} pid={unity_pid} "
                        f"rss_kib={status['rss_kib']} duration={record['duration_seconds']}s",
                        flush=True,
                    )
            except Exception as exc:
                error = f"episode {episode + 1}: {type(exc).__name__}: {exc}"
                errors.append(error)
                print(error, flush=True)
                if session_id is not None:
                    try:
                        request(
                            args.base_url,
                            "POST",
                            "/close",
                            {"session_id": session_id},
                            timeout=30,
                        )
                    except Exception:
                        pass
                break

    final_status: dict[str, Any] | None
    try:
        final_status = request(args.base_url, "GET", "/status", timeout=5)
    except Exception as exc:
        final_status = None
        errors.append(f"final status: {type(exc).__name__}: {exc}")

    durations = [float(record["duration_seconds"]) for record in records]
    rss_values = [int(record["rss_kib"]) for record in records if record["rss_kib"] is not None]
    fd_values = [int(record["fd_count"]) for record in records if record["fd_count"] is not None]
    thread_values = [int(record["python_thread_count"]) for record in records]
    rss_growth_kib = max(rss_values) - min(rss_values[: min(5, len(rss_values))]) if rss_values else 0

    invariants = {
        "all_episodes_completed": len(records) == args.episodes,
        "one_service_process": len(service_ids) == 1,
        "one_unity_process": len(unity_pids) == 1,
        "one_controller_start": bool(records)
        and all(int(record["controller_starts"]) == 1 for record in records),
        "one_thor_thread_alive": bool(records)
        and all(bool(record["thor_thread_alive"]) for record in records),
        "reset_count_exact": final_status is not None
        and int(final_status["episode_resets"]) - start_resets == len(records),
        "thread_count_stable": bool(thread_values)
        and max(thread_values) - min(thread_values) <= 2,
        "rss_growth_bounded": rss_growth_kib <= args.max_rss_growth_mib * 1024,
        "no_errors": not errors,
    }
    passed = all(invariants.values())
    summary = {
        "passed": passed,
        "base_url": args.base_url,
        "requested_episodes": args.episodes,
        "completed_episodes": len(records),
        "requested_steps_per_episode": args.steps,
        "unique_tasks": len(set(game_files)),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "service_ids": sorted(service_ids),
        "unity_pids": sorted(unity_pids),
        "duration_seconds": {
            "mean": round(statistics.mean(durations), 3) if durations else None,
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "max": max(durations) if durations else None,
        },
        "rss_kib": {
            "min": min(rss_values) if rss_values else None,
            "max": max(rss_values) if rss_values else None,
            "growth": rss_growth_kib,
        },
        "fd_count": {
            "min": min(fd_values) if fd_values else None,
            "max": max(fd_values) if fd_values else None,
        },
        "python_thread_count": {
            "min": min(thread_values) if thread_values else None,
            "max": max(thread_values) if thread_values else None,
        },
        "invariants": invariants,
        "errors": errors,
        "start_status": start_status,
        "final_status": final_status,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
