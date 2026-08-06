"""Shared model-family prompt preparation for production qualification tools."""

from __future__ import annotations

from pathlib import Path

from openai_server import render_model_chat, snapshot_model_family


def prepare_user_prompt(
    model: Path,
    prompt: str,
    *,
    family: str | None = None,
    raw: bool = False,
) -> tuple[str, str]:
    """Return the selected family and either raw or family-template chat input."""
    selected_family = family or snapshot_model_family(model)
    if raw:
        return selected_family, prompt
    rendered = render_model_chat(
        model,
        selected_family,
        [{"role": "user", "content": prompt}],
        enable_thinking=False,
        cache_prefix_compatible=True,
    )
    return selected_family, rendered
