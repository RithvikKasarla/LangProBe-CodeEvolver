from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from ._tracing import traceable


_CONFIG_PATH = Path(__file__).with_name("runtime_config.json")
_LOGGER = logging.getLogger(__name__)
_DEFAULT_RESPONSE_JSON_DECODE_RETRIES = 2


def load_runtime_config(name: str) -> dict[str, Any]:
    with _CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)[name]


@dataclass(frozen=True)
class ModelResponse:
    text: str
    reasoning: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAICompatibleModel:
    def __init__(self, config_name: str) -> None:
        self.config = load_runtime_config(config_name)
        api_key = os.environ.get(self.config["api_key_env"], "")
        if not api_key:
            raise RuntimeError(f"{self.config['api_key_env']} is not configured")
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config["base_url"],
            timeout=900.0,
            max_retries=5,
        )

    @traceable("llm", name="model.complete", max_attr_chars=12_000)
    def complete(self, messages: list[dict[str, Any]]) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self.config["model"],
            "messages": messages,
            "reasoning_effort": self.config["reasoning_effort"],
            "max_tokens": self.config["max_tokens"],
        }
        if "temperature" in self.config:
            request["temperature"] = self.config["temperature"]
        if "seed" in self.config:
            request["seed"] = self.config["seed"]
        if "provider_preferences" in self.config:
            request["extra_body"] = {
                "provider": self.config["provider_preferences"],
            }
        json_decode_retries = max(
            0,
            int(
                self.config.get(
                    "response_json_decode_retries",
                    _DEFAULT_RESPONSE_JSON_DECODE_RETRIES,
                )
            ),
        )
        for attempt in range(json_decode_retries + 1):
            try:
                response = self.client.chat.completions.create(**request)
                break
            except json.JSONDecodeError:
                if attempt >= json_decode_retries:
                    raise
                delay_seconds = float(2**attempt)
                _LOGGER.warning(
                    "Provider returned malformed JSON; retrying chat completion "
                    "in %.1f seconds (retry %d/%d)",
                    delay_seconds,
                    attempt + 1,
                    json_decode_retries,
                )
                time.sleep(delay_seconds)
        message = response.choices[0].message
        content = message.content or ""
        if not isinstance(content, str):
            content = "".join(str(part) for part in content)
        reasoning = getattr(message, "reasoning_content", None) or ""
        usage = response.usage
        return ModelResponse(
            text=content.strip(),
            reasoning=str(reasoning).strip(),
            prompt_tokens=int(usage.prompt_tokens) if usage else 0,
            completion_tokens=int(usage.completion_tokens) if usage else 0,
            total_tokens=int(usage.total_tokens) if usage else 0,
        )


def strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines)
    return value.strip()
