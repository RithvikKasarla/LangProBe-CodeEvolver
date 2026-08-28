from __future__ import annotations

import re
import uuid
from typing import Any

from .appworld_runtime import appworld_pool
from ._tracing import traceable
from .common import OpenAICompatibleModel, load_runtime_config, strip_code_fence


SYSTEM_PROMPT = """You autonomously complete the user's AppWorld task by writing one small Python code block at a time. The code runs in a persistent Python REPL with an `apis` object.

Useful discovery calls:
- print(apis.api_docs.show_app_descriptions())
- print(apis.api_docs.show_api_descriptions(app_name='spotify'))
- print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))

Inspect an API specification before calling it. Handle every page of paginated APIs. Make irreversible changes only after checking inputs. The supervisor app provides account information and the phone app provides contacts. When finished, call apis.supervisor.complete_task(); pass answer=<answer> only when the task requests information.

Return only executable Python for the next step, with no commentary and no Markdown fence."""


class AppWorldProgram:
    """LangProBe AppWorld ReAct seed adapted to CodeEvolver's callable contract."""

    def __init__(self) -> None:
        self.model = OpenAICompatibleModel("appworld")
        self.config = load_runtime_config("appworld")

    @staticmethod
    def _format_trace(trace: list[dict[str, str]]) -> str:
        if not trace:
            return "No steps have been executed yet."
        return "\n\n".join(
            f"Step {index}\nCode:\n{step['code']}\nOutput:\n{step['output']}"
            for index, step in enumerate(trace, start=1)
        )

    @staticmethod
    def _safe_code(text: str) -> str:
        code = strip_code_fence(text)
        if "os.path.expanduser" in code:
            return "print('Use the file_system app rather than direct home-directory access.')"
        return code

    @traceable("chain", name="appworld.episode", max_attr_chars=12_000)
    def __call__(self, task_id: str) -> dict[str, Any]:
        experiment_name = f"codeevolver-{uuid.uuid4().hex}"
        trace: list[dict[str, str]] = []
        with appworld_pool.acquire(experiment_name, task_id) as world:
            task = world.show_task(task_id)
            supervisor = task["supervisor"]
            supervisor_name = f"{supervisor['first_name']} {supervisor['last_name']}"
            for _ in range(int(self.config["max_steps"])):
                user_prompt = (
                    f"Task: {task['instruction']}\n"
                    f"Supervisor: {supervisor_name}; {supervisor['email']}; "
                    f"{supervisor['phone_number']}\n\n"
                    f"Past steps:\n{self._format_trace(trace)}\n\n"
                    "Generate the next code step."
                )
                response = self.model.complete(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                code = self._safe_code(response.text)
                output = world.execute(task_id, code)
                trace.append(
                    {
                        "code": code,
                        "output": output,
                        "reasoning": response.reasoning[:8000],
                    }
                )
                if world.task_completed(task_id):
                    break

            report = world.evaluate(task_id)
            return {
                "success": bool(report.get("success", False)),
                "eval_report": report,
                "trace": trace,
            }


class AppWorldSmokeProgram(AppWorldProgram):
    """One-step wiring check; full runs use ``AppWorldProgram`` unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.config = {**self.config, "max_steps": 1}
