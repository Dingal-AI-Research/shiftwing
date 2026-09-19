#!/usr/bin/env python3
"""Strict colib wrapper around the pinned official DeepSeek-V4 encoding."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .deepseek_v4_spec import SOURCE_REVISION, validate_reasoning_effort
    from .vendor.deepseek_v4 import encoding_dsv4 as official
except ImportError:  # Direct script/test import with c/tools on sys.path.
    from deepseek_v4_spec import SOURCE_REVISION, validate_reasoning_effort
    from vendor.deepseek_v4 import encoding_dsv4 as official


ENCODING_SHA256 = "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6"


def verify_vendor() -> str:
    path = Path(official.__file__).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != ENCODING_SHA256:
        raise RuntimeError(
            "vendored DeepSeek-V4 encoder hash mismatch: "
            f"expected {ENCODING_SHA256}, got {actual}"
        )
    return actual


def render_deepseek_chat(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Render OpenAI messages using only the pinned official implementation."""
    verify_vendor()
    validate_reasoning_effort(reasoning_effort)
    normalized = copy.deepcopy(list(messages))
    if not normalized:
        raise ValueError("at least one chat message is required")
    if tools:
        target = next(
            (
                message
                for message in normalized
                if message.get("role") in ("system", "developer")
            ),
            None,
        )
        if target is None:
            target = {"role": "system", "content": ""}
            normalized.insert(0, target)
        target["tools"] = copy.deepcopy(list(tools))
    thinking_mode = "thinking" if reasoning_effort is not None else "chat"
    return official.encode_messages(
        normalized,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
    )


def _plain_recovery(text: str, thinking_mode: str, error: BaseException) -> dict[str, Any]:
    value = text.replace(official.bos_token, "")
    if value.startswith(official.ASSISTANT_SP_TOKEN):
        value = value[len(official.ASSISTANT_SP_TOKEN) :]
    if value.startswith(official.thinking_start_token):
        value = value[len(official.thinking_start_token) :]
    value = value.removesuffix(official.eos_token)
    tool_marker = f"\n\n<{official.dsml_token}{official.tool_calls_block_name}"
    if tool_marker in value:
        value = value.split(tool_marker, 1)[0]
    reasoning = ""
    content = value
    if thinking_mode == "thinking":
        if official.thinking_end_token in value:
            reasoning, content = value.split(official.thinking_end_token, 1)
        else:
            reasoning, content = value, ""
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": [],
        "colib_recovery": {
            "recovered": True,
            "error": f"{type(error).__name__}: {error}",
            "source_revision": SOURCE_REVISION,
        },
    }


def parse_deepseek_completion(
    text: str, *, reasoning_effort: str | None = None
) -> dict[str, Any]:
    """Parse official DSML, conservatively recovering malformed plain content."""
    verify_vendor()
    validate_reasoning_effort(reasoning_effort)
    thinking_mode = "thinking" if reasoning_effort is not None else "chat"
    candidate = text
    if not candidate.endswith(official.eos_token):
        candidate += official.eos_token
    try:
        parsed = official.parse_message_from_completion_text(
            candidate, thinking_mode=thinking_mode
        )
        parsed["colib_recovery"] = {"recovered": False}
        return parsed
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return _plain_recovery(text, thinking_mode, exc)


verify_vendor()
