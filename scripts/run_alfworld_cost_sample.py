from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def token_totals(result: dict[str, Any]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for step in result.get("trace", []):
        usage = step.get("usage", {})
        for key in totals:
            totals[key] += int(usage.get(key, 0))
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded ALFWorld cost sample")
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--model")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)
    os.environ["ALFWORLD_SERVER_COUNT"] = str(args.workers)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from codeevolver_benchmarks.alfworld_runtime import alfworld_pool
    from codeevolver_benchmarks.alfworld_vision_program import AlfWorldVisionProgram
    from codeevolver_benchmarks.common import load_runtime_config

    resolved_model = args.model or load_runtime_config("alfworld_vision")["model"]

    examples = json.loads(
        (repo_root / "data" / "alfworld_vision_dev.json").read_text(encoding="utf-8")
    )[: args.rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def run_one(index: int, example: dict[str, Any]) -> dict[str, Any]:
        row_started = time.monotonic()
        program = AlfWorldVisionProgram()
        if args.model:
            program.model.config = {**program.model.config, "model": args.model}
        program.config = {**program.config, "max_steps": args.max_steps}
        try:
            result = program(example["game_file"])
            return {
                "index": index,
                "game_file": example["game_file"],
                "elapsed_seconds": round(time.monotonic() - row_started, 3),
                "steps": len(result.get("trace", [])),
                **token_totals(result),
                "success": bool(result.get("success")),
                "goal_condition_success_rate": result.get(
                    "goal_condition_success_rate", 0.0
                ),
                "error": None,
            }
        except Exception as exc:
            return {
                "index": index,
                "game_file": example["game_file"],
                "elapsed_seconds": round(time.monotonic() - row_started, 3),
                "steps": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "success": False,
                "goal_condition_success_rate": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    try:
        with args.output.open("w", encoding="utf-8") as output:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(run_one, index, example)
                    for index, example in enumerate(examples, start=1)
                ]
                for completed, future in enumerate(as_completed(futures), start=1):
                    row = future.result()
                    row["wall_seconds"] = round(time.monotonic() - started, 3)
                    output.write(json.dumps(row, sort_keys=True) + "\n")
                    output.flush()
                    print(
                        f"[{completed}/{len(examples)}] row={row['index']} "
                        f"steps={row['steps']} tokens={row['total_tokens']} "
                        f"elapsed={row['elapsed_seconds']:.1f}s error={row['error']}",
                        flush=True,
                    )
    finally:
        alfworld_pool.stop()

    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    summary = {
        "rows": len(rows),
        "workers": args.workers,
        "max_steps": args.max_steps,
        "model": resolved_model,
        "wall_seconds": round(time.monotonic() - started, 3),
        "steps": sum(row["steps"] for row in rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "errors": sum(bool(row["error"]) for row in rows),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
