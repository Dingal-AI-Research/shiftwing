#!/usr/bin/env python3
"""Bounded one-shard lookahead for an active Hub streaming conversion."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root is not an object")
    return value


def process_matches(pid: int, needle: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return needle.encode() in command


def source_shards(index: dict[str, Any]) -> tuple[str, ...]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("source index has no weight_map object")
    shards = tuple(sorted({value for value in weight_map.values() if isinstance(value, str)}))
    if not shards:
        raise ValueError("source index has no shard names")
    return shards


def lookahead_target(
    shards: tuple[str, ...],
    completed: set[str],
) -> str | None:
    """Return the shard after the converter's first uncommitted shard."""
    remaining = [name for name in shards if name not in completed]
    return remaining[1] if len(remaining) >= 2 else None


def validate_signature(
    state: dict[str, Any],
    *,
    repo: str,
    revision: str,
) -> None:
    signature = state.get("signature")
    expected = f"hf://{repo}@{revision}"
    if not isinstance(signature, dict) or signature.get("source") != expected:
        observed = signature.get("source") if isinstance(signature, dict) else None
        raise ValueError(
            f"conversion source mismatch: expected {expected!r}, observed {observed!r}"
        )


def run(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - project dependency
        raise RuntimeError("huggingface_hub is required") from error

    staging = args.staging_dir.resolve()
    state_path = args.state.resolve()
    index_path = staging / "model.safetensors.index.json"
    shards = source_shards(read_object(index_path))
    last_message: tuple[int, str | None, bool] | None = None

    while process_matches(args.converter_pid, "convert_qwen.py"):
        try:
            state = read_object(state_path)
            validate_signature(state, repo=args.repo, revision=args.revision)
            completed_value = state.get("completed")
            if not isinstance(completed_value, dict):
                raise ValueError(f"{state_path}: completed is not an object")
            completed = set(completed_value)
            target = lookahead_target(shards, completed)
            if target is None:
                print("[prefetch] no second uncommitted shard remains", flush=True)
                return 0
            destination = staging / target
            present = destination.is_file()
            message = (len(completed), target, present)
            if message != last_message:
                print(
                    f"[prefetch] committed={len(completed)}/{len(shards)} "
                    f"target={target} present={present}",
                    flush=True,
                )
                last_message = message
            if present:
                time.sleep(args.poll_seconds)
                continue
            free_gb = shutil.disk_usage(staging).free / (1024**3)
            if free_gb < args.min_free_gb:
                raise RuntimeError(
                    f"disk guard failed: {free_gb:.2f} GiB free "
                    f"< {args.min_free_gb:.2f} GiB"
                )
            downloaded = Path(
                hf_hub_download(
                    args.repo,
                    target,
                    revision=args.revision,
                    local_dir=staging,
                )
            )
            if downloaded.resolve() != destination.resolve() or not destination.is_file():
                raise RuntimeError(
                    f"download returned unexpected path {downloaded}; "
                    f"expected {destination}"
                )
            print(f"[prefetch] ready {target}", flush=True)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"[prefetch] retry after {type(error).__name__}: {error}", flush=True)
            time.sleep(args.poll_seconds)

    print("[prefetch] converter exited", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--converter-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--min-free-gb", type=float, default=100.0)
    args = parser.parse_args()
    if args.converter_pid <= 0 or args.poll_seconds <= 0 or args.min_free_gb <= 0:
        parser.error("PID, poll interval, and disk guard must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
