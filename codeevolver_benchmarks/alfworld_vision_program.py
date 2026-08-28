from __future__ import annotations

import json
import re
from typing import Any

from .alfworld_runtime import alfworld_pool
from ._tracing import traceable
from .common import OpenAICompatibleModel, load_runtime_config, strip_code_fence


SYSTEM_PROMPT = """You control an embodied household agent. Use the current camera image, task, environment feedback, action history, and admissible actions to select exactly one next action. Return JSON only: {\"action\": \"...\"}. The action must exactly match one admissible action. Plan carefully, track inventory and object state, and avoid repeating failed actions."""


class AlfWorldVisionProgram:
    """Visual ALFWorld seed using LangProBe's iterative action loop."""

    def __init__(self) -> None:
        self.model = OpenAICompatibleModel("alfworld_vision")
        self.config = load_runtime_config("alfworld_vision")

    @staticmethod
    def _select_action(text: str, admissible: list[str]) -> str:
        if not admissible:
            raise RuntimeError("ALFWorld returned no admissible actions")
        value = strip_code_fence(text)
        try:
            parsed = json.loads(value)
            proposed = str(parsed.get("action", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            match = re.search(r"(?:^|\n)\s*action\s*:\s*(.+)", value, re.IGNORECASE)
            lines = value.splitlines()
            proposed = (
                match.group(1).strip().strip('"`')
                if match
                else lines[0].strip() if lines else ""
            )
        normalized = proposed.casefold()
        for action in admissible:
            if action.casefold() == normalized:
                return action
        for action in sorted(admissible, key=len, reverse=True):
            if action.casefold() in normalized:
                return action
        return "look" if "look" in admissible else admissible[0]

    @traceable("chain", name="alfworld_vision.episode", max_attr_chars=12_000)
    def __call__(self, game_file: str) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        won = False
        goal_condition = 0.0
        with alfworld_pool.acquire(game_file) as (world, state):
            info = state["info"]
            initial_observation = state["observation"]
            current_observation = initial_observation
            current_frame = state["frame"]
            for _ in range(int(self.config["max_steps"])):
                admissible = list(info["admissible_commands"][0])
                history = "\n".join(
                    f"{index}. {step['action']} -> {step['observation']}"
                    for index, step in enumerate(trace[-12:], start=max(1, len(trace) - 11))
                ) or "No actions yet."
                prompt = (
                    f"Initial task and scene:\n{initial_observation}\n\n"
                    f"Current feedback:\n{current_observation}\n\n"
                    f"Recent history:\n{history}\n\n"
                    "Admissible actions:\n- " + "\n- ".join(admissible)
                )
                response = self.model.complete(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": current_frame},
                                },
                            ],
                        },
                    ]
                )
                action = self._select_action(
                    response.text or response.reasoning, admissible
                )
                state = world.step(action)
                info = state["info"]
                current_observation = state["observation"]
                current_frame = state["frame"]
                won = bool(info["won"][0])
                goal_condition = float(info.get("goal_condition_success_rate", [0.0])[0])
                trace.append(
                    {
                        "action": action,
                        "observation": current_observation,
                        "reasoning": response.reasoning[:8000],
                        "usage": {
                            "prompt_tokens": response.prompt_tokens,
                            "completion_tokens": response.completion_tokens,
                            "total_tokens": response.total_tokens,
                        },
                    }
                )
                if won or bool(state["done"]):
                    break
        return {
            "success": won,
            "goal_condition_success_rate": goal_condition,
            "trace": trace,
        }


class AlfWorldVisionSmokeProgram(AlfWorldVisionProgram):
    """Environment-reset wiring check; full runs use the regular program."""

    def __init__(self) -> None:
        super().__init__()
        self.config = {**self.config, "max_steps": 0}
