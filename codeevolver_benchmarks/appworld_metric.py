from __future__ import annotations

from typing import Any

from ._tracing import traceable


@traceable("function", name="appworld.metric", max_attr_chars=4_000)
def appworld_metric(output: Any, example: Any) -> dict[str, Any]:
    result = dict(output or {})
    report = result.get("eval_report") or {}
    success = bool(result.get("success", False))
    feedback = "Task completed successfully." if success else f"Task failed. Evaluation: {report}"
    return {"score": float(success), "feedback": feedback}


def appworld_smoke_metric(output: Any, example: Any) -> dict[str, Any]:
    result = dict(output or {})
    valid = isinstance(result.get("trace"), list) and "eval_report" in result
    return {
        "score": 0.1 if valid else 0.0,
        "feedback": "Smoke output shape is valid." if valid else "Smoke output is incomplete.",
    }
