from __future__ import annotations

from typing import Any

from ._tracing import traceable


@traceable("function", name="alfworld_vision.metric", max_attr_chars=4_000)
def alfworld_vision_metric(output: Any, example: Any) -> dict[str, Any]:
    result = dict(output or {})
    success = bool(result.get("success", False))
    partial = float(result.get("goal_condition_success_rate", 0.0))
    feedback = (
        "Task completed successfully."
        if success
        else f"Task failed; goal-condition completion was {partial:.3f}."
    )
    return {"score": float(success), "feedback": feedback}


def alfworld_vision_smoke_metric(output: Any, example: Any) -> dict[str, Any]:
    result = dict(output or {})
    valid = isinstance(result.get("trace"), list) and "success" in result
    return {
        "score": 0.1 if valid else 0.0,
        "feedback": "Smoke output shape is valid." if valid else "Smoke output is incomplete.",
    }
