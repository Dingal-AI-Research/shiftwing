#!/usr/bin/env python3
"""Ornith model registry, chat rendering, and Qwen3-XML response parsing."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ORNITH_STOP_IDS = (248046, 248044)


@dataclass(frozen=True)
class ModelProfile:
    family: str
    source_repo: str | None
    source_revision: str | None
    stop_token_ids: tuple[int, ...]
    pad_token_id: int | None
    chat_template_sha256: str | None


def _ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return tuple(dict.fromkeys(value))
    return ()


def load_model_profile(snapshot: Path) -> ModelProfile:
    config = json.loads((snapshot / "config.json").read_text())
    generation_path = snapshot / "generation_config.json"
    generation = (
        json.loads(generation_path.read_text()) if generation_path.exists() else {}
    )
    text = config.get("text_config", config)
    stop_ids = _ids(generation.get("eos_token_id"))
    if not stop_ids:
        stop_ids = _ids(config.get("eos_token_id")) or _ids(
            text.get("eos_token_id")
        )
    pad = generation.get(
        "pad_token_id",
        config.get("pad_token_id", text.get("pad_token_id")),
    )
    family = config.get("colib_model_family", "qwen3.5")
    template_path = snapshot / "chat_template.jinja"
    digest = (
        hashlib.sha256(template_path.read_bytes()).hexdigest()
        if template_path.exists()
        else None
    )
    if family == "ornith-1.0" and not set(ORNITH_STOP_IDS).issubset(stop_ids):
        raise ValueError(
            "Ornith snapshot must stop on both <|im_end|> and <|endoftext|>"
        )
    return ModelProfile(
        family=family,
        source_repo=config.get("colib_source_repo"),
        source_revision=config.get("colib_source_revision"),
        stop_token_ids=stop_ids,
        pad_token_id=pad if isinstance(pad, int) else None,
        chat_template_sha256=digest,
    )


def render_chat(
    snapshot: Path,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """Render the snapshot's own pinned Jinja template.

    Keeping the template beside the converted model avoids silently applying a
    Qwen or 35B template to an Ornith-397B snapshot; the two current official
    Ornith templates are similar but not byte-identical.
    """
    template_path = snapshot / "chat_template.jinja"
    if not template_path.exists():
        raise FileNotFoundError(f"missing chat template: {template_path}")
    from transformers.utils.chat_template_utils import render_jinja_template

    normalized_messages = _normalize_messages_for_template(messages)
    rendered, _ = render_jinja_template(
        [normalized_messages],
        tools=list(tools) if tools else None,
        chat_template=template_path.read_text(),
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        add_vision_id=False,
    )
    return rendered[0]


def _normalize_messages_for_template(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI wire-format arguments to the mapping Ornith Jinja expects.

    OpenAI chat responses encode ``function.arguments`` as a JSON string. The
    released Ornith template applies Jinja's ``items`` filter to that field, so
    replaying an assistant tool call without normalization raises a template
    error. Keep the public/wire representation unchanged and adapt only the
    private copy passed to the snapshot template.
    """
    normalized = deepcopy([dict(message) for message in messages])
    for message in normalized:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function", call)
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "assistant tool-call arguments are not valid JSON"
                    ) from exc
                if not isinstance(arguments, dict):
                    raise ValueError(
                        "assistant tool-call arguments must decode to an object"
                    )
                function["arguments"] = arguments
    return normalized


_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_.:-]*)>\s*"
    r"(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_.:-]*)>\s*"
    r"(.*?)\s*</parameter>",
    re.DOTALL,
)


def _parameter_value(text: str) -> Any:
    value = text.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_assistant_response(text: str) -> dict[str, Any]:
    """Split reasoning and convert Qwen3-XML calls to OpenAI-style tool calls."""
    reasoning = ""
    body = text.strip()
    if body.startswith("<think>") and "</think>" in body:
        raw_reasoning, body = body[len("<think>") :].split("</think>", 1)
        reasoning = raw_reasoning.strip()
        body = body.strip()

    calls: list[dict[str, Any]] = []
    matches = list(_TOOL_CALL.finditer(body))
    for index, match in enumerate(matches):
        arguments: dict[str, Any] = {}
        for parameter in _PARAMETER.finditer(match.group(2)):
            arguments[parameter.group(1)] = _parameter_value(parameter.group(2))
        calls.append(
            {
                "id": f"call_{index + 1}",
                "type": "function",
                "function": {
                    "name": match.group(1),
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
        )

    if matches:
        content = body[: matches[0].start()].strip() or None
        suffix = body[matches[-1].end() :].strip()
        if suffix:
            # The official prompt forbids a suffix. Preserve it instead of
            # silently discarding malformed model output.
            content = ((content + "\n\n") if content else "") + suffix
    else:
        content = body or None
    return {
        "reasoning_content": reasoning or None,
        "content": content,
        "tool_calls": calls or None,
    }
