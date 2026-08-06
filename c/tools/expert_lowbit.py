"""Shared fail-closed loading for optional routed-expert sidecar manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_complete_expert_sidecar(model: Path, *, bits: int) -> dict[str, Any]:
    if bits not in (2, 3):
        raise ValueError("expert sidecar bits must be 2 or 3")
    path = model / f"expert-q{bits}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"int{bits} expert manifest is unavailable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root is not an object")
    expected_format = f"colib-routed-expert-int{bits}-sidecar-v1"
    if value.get("format") != expected_format:
        raise ValueError(
            f"{path}: format {value.get('format')!r} != {expected_format!r}"
        )
    if value.get("complete") is not True:
        raise ValueError(f"{path}: expert sidecar is incomplete")
    for name in ("config_sha256", "signature"):
        if not value.get(name):
            raise ValueError(f"{path}: {name} is missing")
    for name in ("layers", "file_count", "tensor_count", "data_bytes"):
        observed = value.get(name)
        if not isinstance(observed, int) or observed <= 0:
            raise ValueError(f"{path}: {name} must be positive")
    return {
        "format": value["format"],
        "complete": True,
        "config_sha256": value["config_sha256"],
        "signature": value["signature"],
        "layers": value["layers"],
        "file_count": value["file_count"],
        "tensor_count": value["tensor_count"],
        "data_bytes": value["data_bytes"],
        "manifest_sha256": sha256_file(path),
    }
