from __future__ import annotations

import json
import os
from pathlib import Path


def rows(root: Path, split: str) -> list[dict[str, str]]:
    split_root = root / "json_2.1.1" / split
    return [
        {"game_file": path.relative_to(root).as_posix()}
        for path in sorted(split_root.rglob("traj_data.json"))
        if "movable" not in path.as_posix() and "Sliced" not in path.as_posix()
    ]


def main() -> None:
    root = Path(os.environ["ALFWORLD_DATA"]).resolve()
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for output_name, split in {
        "train": "train",
        "val": "valid_train",
        "dev": "valid_seen",
        "test": "valid_unseen",
    }.items():
        output = data_dir / f"alfworld_vision_{output_name}.json"
        output.write_text(json.dumps(rows(root, split), indent=2) + "\n")


if __name__ == "__main__":
    main()
