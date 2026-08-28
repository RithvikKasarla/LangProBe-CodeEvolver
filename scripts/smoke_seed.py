from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["appworld", "alfworld_vision"])
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.benchmark == "appworld":
        from codeevolver_benchmarks.appworld_program import AppWorldProgram

        row = json.loads((repo_root / "data" / "appworld_train.json").read_text())[0]
        program = AppWorldProgram()
    else:
        from codeevolver_benchmarks.alfworld_vision_program import AlfWorldVisionProgram

        row = json.loads(
            (repo_root / "data" / "alfworld_vision_train.json").read_text()
        )[0]
        program = AlfWorldVisionProgram()

    program.config["max_steps"] = args.max_steps
    result = program(**row)
    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "success": bool(result.get("success")),
                "trace_steps": len(result.get("trace", [])),
                "goal_condition_success_rate": result.get(
                    "goal_condition_success_rate"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
