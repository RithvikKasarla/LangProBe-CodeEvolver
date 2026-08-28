from __future__ import annotations

import json
import random
from pathlib import Path

from appworld.task import load_task_ids


def write(path: Path, task_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"task_id": task_id} for task_id in task_ids], indent=2) + "\n")


def main() -> None:
    train = load_task_ids("train")
    dev = load_task_ids("dev")
    test_normal = load_task_ids("test_normal")
    test_challenge = load_task_ids("test_challenge")

    optimization_pool = train + dev + test_normal
    scenarios = sorted({task_id.split("_")[0] for task_id in optimization_pool})
    random.Random(42).shuffle(scenarios)
    train_scenarios = set(scenarios[:67])
    custom_train = [
        task_id
        for task_id in optimization_pool
        if task_id.split("_")[0] in train_scenarios
    ]
    custom_val = [
        task_id
        for task_id in optimization_pool
        if task_id.split("_")[0] not in train_scenarios
    ]
    assert len(custom_train) == 201
    assert len(custom_val) == 114
    assert len(test_challenge) == 417

    data_dir = Path(__file__).resolve().parents[1] / "data"
    write(data_dir / "appworld_train.json", custom_train)
    write(data_dir / "appworld_val.json", custom_val)
    write(data_dir / "appworld_test.json", test_challenge)
    write(data_dir / "appworld_test_normal.json", test_normal)
    write(data_dir / "appworld_dev.json", dev)


if __name__ == "__main__":
    main()
