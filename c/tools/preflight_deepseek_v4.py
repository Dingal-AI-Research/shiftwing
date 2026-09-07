#!/usr/bin/env python3
"""Read-only DeepSeek-V4 conversion preflight with reproducible JSON evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from runtime_env import deepseek_engine_source_sha256

from deepseek_v4_spec import (
    MIN_FINAL_FREE_BYTES,
    PLANNED_CHECKPOINT_BYTES,
    PLANNED_CONTAINER_BYTES,
    PLANNED_STAGING_BYTES,
    source_identity,
)


SCHEMA = "colib.deepseek-v4.preflight.v1"
RELEVANT_ENV_PREFIXES = ("COLI_", "CUDA_", "NVIDIA_", "OMP_", "HF_HUB_")
RELEVANT_ENV_EXACT = {"CUDA_VISIBLE_DEVICES", "PATH"}
SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "KEY")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    try:
        source_fingerprint = deepseek_engine_source_sha256(root)
        fingerprint_status = "complete"
    except FileNotFoundError as error:
        source_fingerprint = None
        missing = Path(error.filename) if error.filename else None
        try:
            missing_text = missing.relative_to(root.resolve()).as_posix() if missing else "unknown"
        except ValueError:
            missing_text = missing.as_posix() if missing else "unknown"
        fingerprint_status = f"missing:{missing_text}"
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
        "deepseek_engine_source_sha256": source_fingerprint,
        "deepseek_engine_source_status": fingerprint_status,
    }


def memory_metadata() -> dict[str, int | None]:
    values: dict[str, int | None] = {"total_bytes": None, "available_bytes": None}
    path = Path("/proc/meminfo")
    if not path.is_file():
        return values
    parsed: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, _, rest = line.partition(":")
        fields = rest.split()
        if fields and fields[0].isdigit():
            parsed[name] = int(fields[0]) * 1024
    values["total_bytes"] = parsed.get("MemTotal")
    values["available_bytes"] = parsed.get("MemAvailable")
    return values


def gpu_metadata() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            gpus.append(
                {
                    "name": fields[0],
                    "uuid": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "driver_version": fields[3],
                    "compute_capability": fields[4],
                }
            )
    return gpus


def safe_environment() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in os.environ.items():
        selected = name in RELEVANT_ENV_EXACT or name.startswith(RELEVANT_ENV_PREFIXES)
        secret = any(marker in name.upper() for marker in SECRET_MARKERS)
        if selected and not secret:
            result[name] = value
    return dict(sorted(result.items()))


def make_report(
    root: Path,
    target: Path,
    source_bytes_remaining: int,
    container_bytes: int,
    staging_bytes: int,
    min_final_free_bytes: int,
    argv: list[str],
) -> dict[str, Any]:
    root = root.resolve()
    target_parent = target.resolve() if target.exists() else target.parent.resolve()
    usage = shutil.disk_usage(target_parent)
    peak_new_bytes = source_bytes_remaining + container_bytes + staging_bytes
    projected_final_free = usage.free - peak_new_bytes
    failures: list[str] = []
    if projected_final_free < min_final_free_bytes:
        failures.append(
            "projected free space after conversion is below the required floor"
        )
    return {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "accepted": not failures,
        "failures": failures,
        "source": source_identity(),
        "command": shlex.join(argv),
        "repository": git_metadata(root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "variables": safe_environment(),
        },
        "hardware": {"memory": memory_metadata(), "gpus": gpu_metadata()},
        "storage": {
            "filesystem_path": str(target_parent),
            "capacity_bytes": usage.total,
            "free_before_bytes": usage.free,
            "planned_container_bytes": container_bytes,
            "planned_staging_bytes": staging_bytes,
            "source_bytes_remaining": source_bytes_remaining,
            "projected_peak_new_bytes": peak_new_bytes,
            "projected_final_free_bytes": projected_final_free,
            "required_final_free_bytes": min_final_free_bytes,
        },
        "cleanup_policy": [
            "stale download/conversion staging",
            "unqualified large-model scratch",
            "c/ornith35 (only if necessary)",
            "c/ornith397 only after its fresh control evidence is independently verified",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--container-bytes", type=int, default=PLANNED_CONTAINER_BYTES)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-bytes", type=int, default=PLANNED_CHECKPOINT_BYTES)
    parser.add_argument("--staging-bytes", type=int, default=PLANNED_STAGING_BYTES)
    parser.add_argument("--min-final-free-gib", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    minimum = int(args.min_final_free_gib * 1024**3)
    present_source_bytes = 0
    if args.source and args.source.is_dir():
        present_source_bytes = sum(
            path.stat().st_size for path in args.source.glob("*.safetensors")
        )
    source_remaining = max(0, args.source_bytes - present_source_bytes)

    report = make_report(
        args.root,
        args.target,
        source_remaining,
        args.container_bytes,
        args.staging_bytes,
        minimum,
        sys.argv,
    )
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
