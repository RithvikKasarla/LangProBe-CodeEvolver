from __future__ import annotations

import json
import re
from typing import Any

from .alfworld_runtime import alfworld_pool
from ._tracing import traceable
from .common import OpenAICompatibleModel, _span_attr, load_runtime_config, strip_code_fence


SYSTEM_PROMPT = """You control an embodied household agent. Use the current camera image, task, environment feedback, action history, and admissible actions to select exactly one next action. Return JSON only: {\"action\": \"...\"}. The action must exactly match one admissible action. Plan carefully, track inventory and object state, and avoid repeating failed actions."""


# (i) An embedded JSON object carrying an "action" key, found anywhere in the
# text. Deliberately does not require the object to be the whole string, and
# does not require whitespace before "action" -- that is what lets it catch
# `{"action": "..."}` wrapped in prose, which the line-oriented regex below
# cannot (the leading `{"` breaks its `^`/`\n` anchor).
_JSON_OBJECT_RE = re.compile(r'\{[^{}]*"action"[^{}]*\}')

# (ii) Plain "action: ..." style lines (no surrounding braces/quotes).
_ACTION_LINE_RE = re.compile(r"(?:^|\n)[ \t]*action[ \t]*:[ \t]*(.+)", re.IGNORECASE)

_LEADING_MARKER_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT = " \t.,;:!"


class AlfWorldVisionProgram:
    """Visual ALFWorld seed using LangProBe's iterative action loop."""

    def __init__(self) -> None:
        self.model = OpenAICompatibleModel("alfworld_vision")
        self.config = load_runtime_config("alfworld_vision")

    @staticmethod
    def _extract_candidates(value: str) -> list[tuple[str, str]]:
        """Return (source, raw_candidate) pairs in extraction priority order.

        Priority: (i) embedded JSON action objects, with the LAST one in the
        text tried first, (ii) action:-style lines with the LAST one tried
        first, (iii) the LAST non-empty line, then the first. Model reasoning
        states its final decision at the end, so "last" is tried before
        "first" throughout.
        """
        candidates: list[tuple[str, str]] = []

        json_matches = list(_JSON_OBJECT_RE.finditer(value))
        for json_match in reversed(json_matches):
            try:
                parsed = json.loads(json_match.group(0))
                action_value = parsed.get("action")
            except (json.JSONDecodeError, AttributeError, TypeError):
                action_value = None
            if action_value is not None:
                candidates.append(("json", str(action_value).strip()))

        action_lines = _ACTION_LINE_RE.findall(value)
        for raw in reversed(action_lines):
            candidates.append(("action_line", raw.strip()))

        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if lines:
            candidates.append(("line", lines[-1]))
            if len(lines) > 1:
                candidates.append(("line", lines[0]))

        return candidates

    @staticmethod
    def _clean_candidate(raw: str) -> str:
        """Strip quotes/backticks, list markers, and trailing punctuation."""
        text = raw.strip()
        if len(text) >= 2 and text[0] in "\"'`" and text[-1] == text[0]:
            text = text[1:-1].strip()
        text = text.strip("`\"'").strip()
        text = _LEADING_MARKER_RE.sub("", text)
        text = text.rstrip(_TRAILING_PUNCT)
        return _WHITESPACE_RE.sub(" ", text).strip()

    @staticmethod
    def _normalize(text: str) -> str:
        return _WHITESPACE_RE.sub(" ", text).strip().casefold()

    @staticmethod
    def _match_candidate(cleaned: str, admissible: list[str]) -> tuple[str, str] | None:
        """Try the cleaned candidate against admissible actions, in order:

        (1) exact match, (2) longest prefix-anchored match, (3) last-resort
        substring match ranked by rightmost occurrence (not length), so an
        action named earlier -- e.g. one explicitly rejected -- loses to one
        named later in the same line.
        """
        normalized_candidate = AlfWorldVisionProgram._normalize(cleaned)
        if not normalized_candidate:
            return None
        normalized_actions = [
            (action, AlfWorldVisionProgram._normalize(action)) for action in admissible
        ]

        for action, normalized_action in normalized_actions:
            if normalized_action and normalized_action == normalized_candidate:
                return action, "exact"

        prefix_matches = [
            (action, normalized_action)
            for action, normalized_action in normalized_actions
            if normalized_action and normalized_candidate.startswith(normalized_action)
        ]
        if prefix_matches:
            best_action, _ = max(prefix_matches, key=lambda pair: len(pair[1]))
            return best_action, "prefix"

        substring_matches: list[tuple[int, int, str]] = []
        for action, normalized_action in normalized_actions:
            if not normalized_action:
                continue
            index = normalized_candidate.rfind(normalized_action)
            if index != -1:
                substring_matches.append((index, len(normalized_action), action))
        if substring_matches:
            substring_matches.sort(key=lambda item: (item[0], item[1]))
            _, _, best_action = substring_matches[-1]
            return best_action, "substring"

        return None

    @staticmethod
    def _select_action(text: str, admissible: list[str]) -> tuple[str, str]:
        """Convert model output into one admissible action.

        Returns (action, match_method). match_method records how the action
        was chosen (e.g. "json_exact", "action_line_prefix", "line_substring",
        "fallback_noop", "empty_response") so parse quality is measurable
        instead of silently absorbed into a "look" no-op.
        """
        if not admissible:
            raise RuntimeError("ALFWorld returned no admissible actions")
        value = strip_code_fence(text)
        candidates = AlfWorldVisionProgram._extract_candidates(value)
        if not candidates:
            fallback = "look" if "look" in admissible else admissible[0]
            return fallback, "empty_response"

        for source, raw_candidate in candidates:
            cleaned = AlfWorldVisionProgram._clean_candidate(raw_candidate)
            if not cleaned:
                continue
            match = AlfWorldVisionProgram._match_candidate(cleaned, admissible)
            if match is not None:
                action, stage = match
                return action, f"{source}_{stage}"

        fallback = "look" if "look" in admissible else admissible[0]
        return fallback, "fallback_noop"

    @traceable("chain", name="alfworld_vision.episode", max_attr_chars=12_000)
    def __call__(self, game_file: str) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        won = False
        goal_condition = 0.0
        parse_failures = 0
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
                action, match_method = self._select_action(
                    response.text or response.reasoning, admissible
                )
                # Defensive: another route may or may not attach finish_reason
                # to the response object. Never branch control flow on it --
                # it is audit-only.
                finish_reason = getattr(response, "finish_reason", None)
                _span_attr("alfworld.match_method", match_method)
                if match_method in ("fallback_noop", "empty_response"):
                    parse_failures += 1
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
                        "match_method": match_method,
                        "finish_reason": finish_reason,
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
            "parse_failures": parse_failures,
        }


class AlfWorldVisionSmokeProgram(AlfWorldVisionProgram):
    """Environment-reset wiring check; full runs use the regular program."""

    def __init__(self) -> None:
        super().__init__()
        self.config = {**self.config, "max_steps": 0}
