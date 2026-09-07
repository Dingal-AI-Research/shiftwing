#!/usr/bin/env python3
"""Audit pinned DeepSeek metadata before any weight-shard transfer."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from deepseek_v4_spec import SOURCE_REPO, SOURCE_REVISION, validate_config
from fetch_deepseek_v4 import API_FILE, SOURCE_MARKER, STATE_FILE, sha256_file, shard_files


SCHEMA = "colib.deepseek-v4.metadata-inventory.v1"
BASE_EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
MTP_EXPERT_RE = re.compile(
    r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
EXPECTED_COMPONENTS = {
    (projection, kind)
    for projection in ("w1", "w2", "w3")
    for kind in ("weight", "scale")
}


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

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def analyze_weight_map(weight_map: Mapping[str, str]) -> dict[str, Any]:
    base: dict[int, dict[int, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    mtp: dict[int, dict[int, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    categories = Counter()
    failures: list[str] = []
    for name in weight_map:
        match = BASE_EXPERT_RE.fullmatch(name)
        if match:
            layer, expert = int(match.group(1)), int(match.group(2))
            base[layer][expert].add((match.group(3), match.group(4)))
            categories["routed_expert"] += 1
            continue
        match = MTP_EXPERT_RE.fullmatch(name)
        if match:
            layer, expert = int(match.group(1)), int(match.group(2))
            mtp[layer][expert].add((match.group(3), match.group(4)))
            categories["dspark_expert"] += 1
            continue
        if name.startswith("mtp."):
            categories["dspark_dense"] += 1
        else:
            categories["dense"] += 1

    expected_layers = set(range(43))
    if set(base) != expected_layers:
        failures.append(f"base expert layers: expected 0..42, got {sorted(base)}")
    for layer in sorted(base):
        experts = base[layer]
        if set(experts) != set(range(256)):
            failures.append(f"layer {layer}: expected experts 0..255, got {len(experts)}")
            continue
        incomplete = [
            expert
            for expert, components in experts.items()
            if components != EXPECTED_COMPONENTS
        ]
        if incomplete:
            failures.append(
                f"layer {layer}: {len(incomplete)} experts lack native w1/w2/w3 weight/scale pairs"
            )
    for layer in sorted(mtp):
        experts = mtp[layer]
        if set(experts) != set(range(256)):
            failures.append(
                f"MTP layer {layer}: expected experts 0..255, got {len(experts)}"
            )
        incomplete = [
            expert
            for expert, components in experts.items()
            if components != EXPECTED_COMPONENTS
        ]
        if incomplete:
            failures.append(
                f"MTP layer {layer}: {len(incomplete)} experts lack native pairs"
            )
    if not mtp:
        failures.append("no bundled MTP/DSpark expert layers found")
    return {
        "passed": not failures,
        "failures": failures,
        "categories": dict(sorted(categories.items())),
        "base_expert_layers": len(base),
        "base_experts_per_layer": {
            str(layer): len(experts) for layer, experts in sorted(base.items())
        },
        "dspark_expert_layers": sorted(mtp),
    }


def audit(source: Path, root: Path, argv: list[str]) -> dict[str, Any]:
    source = source.resolve()
    failures: list[str] = []
    api_path = source / API_FILE
    state_path = source / STATE_FILE
    marker_path = source / SOURCE_MARKER
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    required = (api_path, state_path, marker_path, config_path, index_path)
    for path in required:
        if not path.is_file():
            failures.append(f"missing {path.name}")
    if failures:
        return {
            "schema": SCHEMA,
            "created_at": utc_now(),
            "accepted": False,
            "failures": failures,
        }

    api = json.loads(api_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for name, value in (
        ("API", api.get("sha")),
        ("fetch state", state.get("revision")),
        ("source marker", marker.get("revision")),
    ):
        if value != SOURCE_REVISION:
            failures.append(f"{name} revision {value!r} does not match pin")
    if marker.get("repository") != SOURCE_REPO:
        failures.append("source marker repository does not match pin")
    failures.extend(validate_config(config))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        failures.append("index has no weight_map")
        weight_map = {}
    try:
        shards = shard_files(index)
    except ValueError as exc:
        failures.append(str(exc))
        shards = []
    analysis = analyze_weight_map(weight_map)
    failures.extend(analysis["failures"])
    total_size = index.get("metadata", {}).get("total_size")
    if not isinstance(total_size, int) or total_size <= 0:
        failures.append("index metadata.total_size is missing or invalid")
        total_size = 0
    for name, evidence in state.get("completed", {}).items():
        path = source.joinpath(*Path(name).parts)
        if not path.is_file():
            failures.append(f"completed metadata file is absent: {name}")
            continue
        if path.stat().st_size != evidence.get("size"):
            failures.append(f"completed metadata size changed: {name}")
        elif sha256_file(path) != evidence.get("sha256"):
            failures.append(f"completed metadata hash changed: {name}")

    tensor_count = len(weight_map)
    padding_upper_bound = tensor_count * 4095
    return {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "accepted": not failures,
        "failures": failures,
        "command": shlex.join(argv),
        "repository": git_metadata(root.resolve()),
        "source": {
            "repository": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "api_sha256": sha256_file(api_path),
            "fetch_state_sha256": sha256_file(state_path),
            "source_marker_sha256": sha256_file(marker_path),
            "config_sha256": sha256_file(config_path),
            "index_sha256": sha256_file(index_path),
            "fetch_status": state.get("status"),
        },
        "checkpoint": {
            "tensor_count": tensor_count,
            "weight_shards": len(shards),
            "indexed_payload_bytes": total_size,
            "alignment_padding_upper_bound": padding_upper_bound,
            "container_payload_upper_bound": total_size + padding_upper_bound,
        },
        "layout": analysis,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.source, args.root, sys.argv)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
