#!/usr/bin/env python3
"""Lossless, resumable DeepSeek-V4 native-layout converter.

The converter never dequantizes or requantizes a tensor.  It copies the pinned
checkpoint's native FP4/FP8 byte ranges into aligned colib segment files,
packing each layer's routed-expert gate/up/down records in expert-major order.
Dense weights and the optional DSpark/next-token head are stored separately.

Work is committed one segment at a time.  ``conversion-state.json`` is fsynced
and atomically replaced after each segment, so an interruption repeats at most
the segment that was in flight.  A final manifest binds source byte ranges,
per-record hashes, output offsets, segment hashes, config identity, command,
repository commit, and timestamps.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

from runtime_env import deepseek_engine_source_sha256

from deepseek_v4_layout import validate_records
from deepseek_v4_spec import (
    DEFAULT_CONTEXT,
    MAX_CONTEXT,
    MIN_FINAL_FREE_BYTES,
    MODEL_ID,
    SOURCE_REPO,
    SOURCE_REVISION,
    WEIGHT_SHARDS,
    source_identity,
    validate_config,
)


STATE_SCHEMA = "colib.deepseek-v4.conversion-state.v1"
MANIFEST_SCHEMA = "colib.deepseek-v4.model-manifest.v1"
STATE_FILE = "conversion-state.json"
MANIFEST_FILE = "model-manifest.json"
INDEX_FILE = "model.safetensors.index.json"
SOURCE_MARKER = "colib-source.json"
DEFAULT_ALIGNMENT = 4096
COPY_CHUNK = 8 * 1024 * 1024

METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
EXPLICIT_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
PROJECTION_RE = re.compile(
    r"\.(gate_proj|up_proj|down_proj|w1|w2|w3)(?:\.|$)"
)
EXPERT_MARKERS = (".experts.", ".switch_mlp.")
SHARED_MARKERS = (".shared_expert", ".shared_experts")
DSPARK_MARKERS = (
    "mtp.",
    ".nextn.",
    "nextn_predict",
    ".draft.",
)
PROJECTION_ORDER = {
    "gate_proj": 0,
    "w1": 0,
    "up_proj": 1,
    "w3": 1,
    "down_proj": 2,
    "w2": 2,
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(COPY_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
        "deepseek_engine_source_sha256": deepseek_engine_source_sha256(root),
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_relative_file(name: str) -> str:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or path.name != name:
        raise ValueError(f"unsafe shard name in index: {name!r}")
    return name


@dataclass(frozen=True)
class TensorRecord:
    name: str
    source_name: str
    source_shard: str
    source_offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]
    group: str
    layer: int | None
    expert: int | None
    projection: str | None
    slice_index: int | None = None

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.expert if self.expert is not None else -1,
            PROJECTION_ORDER.get(self.projection or "", 9),
            self.source_name,
            self.slice_index if self.slice_index is not None else -1,
        )


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors header length: {path}")
        header_length = int.from_bytes(raw, "little", signed=False)
        if header_length <= 0 or header_length > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length in {path}")
        header_raw = handle.read(header_length)
    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON in {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not a mapping: {path}")
    return 8 + header_length, header


def _projection(name: str) -> str | None:
    match = PROJECTION_RE.search(name)
    return match.group(1) if match else None


def _base_classification(name: str, source_shard: str) -> tuple[str, int | None]:
    if any(marker in name for marker in DSPARK_MARKERS):
        return f"dspark/{Path(source_shard).stem}.bin", None
    layer_match = LAYER_RE.search(name)
    layer = int(layer_match.group(1)) if layer_match else None
    is_shared = any(marker in name for marker in SHARED_MARKERS)
    is_expert = any(marker in name for marker in EXPERT_MARKERS) and not is_shared
    if layer is not None and is_expert:
        return f"experts/layer-{layer:02d}.bin", layer
    return f"dense/{Path(source_shard).stem}.bin", layer


def _expand_record(
    name: str,
    source_shard: str,
    data_start: int,
    metadata: Mapping[str, Any],
) -> list[TensorRecord]:
    offsets = metadata.get("data_offsets")
    shape = metadata.get("shape")
    dtype = metadata.get("dtype")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(item, int) for item in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
        or not isinstance(shape, list)
        or not all(isinstance(item, int) and item >= 0 for item in shape)
        or not isinstance(dtype, str)
    ):
        raise ValueError(f"invalid tensor metadata for {name}")
    nbytes = offsets[1] - offsets[0]
    group, layer = _base_classification(name, source_shard)
    projection = _projection(name)
    explicit = EXPLICIT_EXPERT_RE.search(name)
    expert_group = group.startswith("experts/")

    if explicit:
        expert = int(explicit.group(1))
        return [
            TensorRecord(
                name=name,
                source_name=name,
                source_shard=source_shard,
                source_offset=data_start + offsets[0],
                nbytes=nbytes,
                dtype=dtype,
                shape=tuple(shape),
                group=group,
                layer=layer,
                expert=expert,
                projection=projection,
            )
        ]

    # Official checkpoints may fuse all experts into the leading dimension.
    # Split only when the byte layout is evenly divisible, preserving the
    # native payload without decoding it.
    if expert_group:
        if not shape or shape[0] != 256 or nbytes % 256:
            raise ValueError(
                f"routed-expert tensor {name} has no explicit expert id and "
                "cannot be losslessly split into 256 contiguous records"
            )
        slice_bytes = nbytes // 256
        return [
            TensorRecord(
                name=f"{name}#expert={expert}",
                source_name=name,
                source_shard=source_shard,
                source_offset=data_start + offsets[0] + expert * slice_bytes,
                nbytes=slice_bytes,
                dtype=dtype,
                shape=tuple(shape[1:]),
                group=group,
                layer=layer,
                expert=expert,
                projection=projection,
                slice_index=expert,
            )
            for expert in range(256)
        ]

    return [
        TensorRecord(
            name=name,
            source_name=name,
            source_shard=source_shard,
            source_offset=data_start + offsets[0],
            nbytes=nbytes,
            dtype=dtype,
            shape=tuple(shape),
            group=group,
            layer=layer,
            expert=None,
            projection=projection,
        )
    ]


def verify_source_revision(source: Path, allow_fixture: bool) -> dict[str, str]:
    if allow_fixture:
        return {"revision": "fixture", "verification": "fixture override"}
    if SOURCE_REVISION in source.resolve().parts:
        return {"revision": SOURCE_REVISION, "verification": "snapshot path"}
    marker = source / SOURCE_MARKER
    if marker.is_file():
        value = _read_json(marker)
        revision = value.get("revision") if isinstance(value, dict) else None
        repository = value.get("repository") if isinstance(value, dict) else None
        if revision == SOURCE_REVISION and repository == SOURCE_REPO:
            return {"revision": SOURCE_REVISION, "verification": SOURCE_MARKER}
    raise ValueError(
        "source revision is not independently bound to the pinned release; "
        f"use a Hugging Face snapshot path containing {SOURCE_REVISION} or "
        f"a {SOURCE_MARKER} written by the pinned fetcher"
    )


def build_plan(
    source: Path, allow_fixture: bool = False, header_provider=None
) -> tuple[dict[str, list[TensorRecord]], dict[str, Any]]:
    source = source.resolve()
    revision = verify_source_revision(source, allow_fixture)
    config = _read_json(source / "config.json")
    if not isinstance(config, dict):
        raise ValueError("config.json is not a mapping")
    if not allow_fixture:
        failures = validate_config(config)
        if failures:
            raise ValueError("pinned config validation failed: " + "; ".join(failures))
    index_path = source / INDEX_FILE
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index has no non-empty weight_map")
    shards = sorted({_validate_relative_file(value) for value in weight_map.values()})
    if not allow_fixture and len(shards) != WEIGHT_SHARDS:
        raise ValueError(f"expected {WEIGHT_SHARDS} shards, found {len(shards)}")

    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    for shard in shards:
        path = source / shard
        if header_provider is not None:
            headers[shard] = header_provider(shard)
        else:
            if not path.is_file():
                raise FileNotFoundError(f"missing source shard: {path}")
            headers[shard] = read_safetensors_header(path)

    groups: dict[str, list[TensorRecord]] = {}
    for name, shard in sorted(weight_map.items()):
        if not isinstance(name, str) or not isinstance(shard, str):
            raise ValueError("weight_map keys and values must be strings")
        data_start, header = headers[shard]
        metadata = header.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"index tensor {name} is absent from {shard}")
        for record in _expand_record(name, shard, data_start, metadata):
            groups.setdefault(record.group, []).append(record)
    for records in groups.values():
        records.sort(key=TensorRecord.sort_key)
    if not allow_fixture:
        failures = validate_records(groups)
        if failures:
            raise ValueError(
                "pinned tensor layout validation failed: " + "; ".join(failures)
            )

    plan_identity = {
        "source": {**source_identity(), **revision},
        "source_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "source_shards": shards,
        "source_tensor_count": len(weight_map),
        "output_record_count": sum(len(records) for records in groups.values()),
        "groups": {
            name: {
                "records": len(records),
                "payload_bytes": sum(record.nbytes for record in records),
            }
            for name, records in sorted(groups.items())
        },
    }
    canonical = json.dumps(plan_identity, sort_keys=True, separators=(",", ":"))
    plan_identity["plan_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return groups, plan_identity


def projected_output_bytes(
    groups: Mapping[str, Iterable[TensorRecord]], alignment: int
) -> int:
    total = 0
    for records in groups.values():
        offset = 0
        for record in records:
            offset = _align(offset, alignment) + record.nbytes
        total += offset
    return total


def preflight_storage(target: Path, required_bytes: int, min_final_free: int) -> None:
    parent = target if target.exists() else target.parent
    usage = shutil.disk_usage(parent.resolve())
    projected = usage.free - required_bytes
    if projected < min_final_free:
        raise OSError(
            f"conversion requires {required_bytes} new bytes and would leave "
            f"{projected} bytes free, below the {min_final_free}-byte floor"
        )


def _copy_range(
    source: BinaryIO, target: BinaryIO, offset: int, nbytes: int
) -> str:
    source.seek(offset)
    remaining = nbytes
    digest = hashlib.sha256()
    while remaining:
        block = source.read(min(COPY_CHUNK, remaining))
        if not block:
            raise EOFError(f"source tensor ended with {remaining} bytes remaining")
        target.write(block)
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def write_group(
    source: Path,
    target: Path,
    group: str,
    records: list[TensorRecord],
    alignment: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = target / group
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    partial.unlink(missing_ok=True)
    handles: dict[str, BinaryIO] = {}
    inventory: list[dict[str, Any]] = []
    try:
        with partial.open("wb") as destination:
            offset = 0
            for record in records:
                aligned = _align(offset, alignment)
                if aligned != offset:
                    destination.write(b"\0" * (aligned - offset))
                handle = handles.get(record.source_shard)
                if handle is None:
                    handle = (source / record.source_shard).open("rb")
                    handles[record.source_shard] = handle
                digest = _copy_range(
                    handle, destination, record.source_offset, record.nbytes
                )
                inventory.append(
                    {
                        "name": record.name,
                        "source_name": record.source_name,
                        "source_shard": record.source_shard,
                        "source_offset": record.source_offset,
                        "slice_index": record.slice_index,
                        "dtype": record.dtype,
                        "shape": list(record.shape),
                        "file": group,
                        "offset": aligned,
                        "nbytes": record.nbytes,
                        "sha256": digest,
                        "layer": record.layer,
                        "expert": record.expert,
                        "projection": record.projection,
                    }
                )
                offset = aligned + record.nbytes
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        for handle in handles.values():
            handle.close()
    os.replace(partial, output)
    _fsync_directory(output.parent)
    segment = {
        "file": group,
        "size": output.stat().st_size,
        "sha256": sha256_file(output),
        "records": len(inventory),
        "completed_at": utc_now(),
    }
    return segment, inventory


def _load_or_create_state(
    path: Path, plan: Mapping[str, Any], command: str, repository: Mapping[str, Any]
) -> dict[str, Any]:
    if path.is_file():
        state = _read_json(path)
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("unrecognized conversion state schema")
        if state.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("conversion state belongs to a different source/plan")
        return state
    state = {
        "schema": STATE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "source": plan["source"],
        "source_index_sha256": plan["source_index_sha256"],
        "repository": repository,
        "command": command,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "completed": {},
        "inventory": {},
    }
    atomic_json(path, state)
    return state


def convert(
    source: Path,
    target: Path,
    alignment: int = DEFAULT_ALIGNMENT,
    min_final_free: int = MIN_FINAL_FREE_BYTES,
    allow_fixture: bool = False,
    command: str | None = None,
    stop_after_groups: int | None = None,
    header_provider=None,
    prepare_group=None,
    release_group=None,
) -> dict[str, Any]:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    source = source.resolve()
    target = target.resolve()
    groups, plan = build_plan(source, allow_fixture=allow_fixture, header_provider=header_provider)
    required = projected_output_bytes(groups, alignment)
    # Completed segments are verified below; account only for new allocation on resume.
    existing = sum((target / group).stat().st_size for group in groups if (target / group).is_file())
    preflight_storage(target, max(0, required - existing), min_final_free)
    target.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[2]
    repository = _git_metadata(repository_root)
    state_path = target / STATE_FILE
    state = _load_or_create_state(
        state_path,
        plan,
        command or shlex.join(sys.argv),
        repository,
    )

    newly_completed = 0
    for group, records in sorted(groups.items()):
        completed = state["completed"].get(group)
        output = target / group
        if completed:
            if output.stat().st_size != completed["size"]:
                raise ValueError(f"completed segment size changed: {group}")
            if sha256_file(output) != completed["sha256"]:
                raise ValueError(f"completed segment hash changed: {group}")
            if release_group is not None:
                # Resume a crash after state commit but before staging release.
                release_group(records, completed, state["inventory"][group])
            continue
        if prepare_group is not None:
            prepare_group(records)
        segment, inventory = write_group(source, target, group, records, alignment)
        state["completed"][group] = segment
        state["inventory"][group] = inventory
        state["updated_at"] = utc_now()
        atomic_json(state_path, state)
        if release_group is not None:
            release_group(records, segment, inventory)
        newly_completed += 1
        print(f"[converted {len(state['completed'])}/{len(groups)}] {group}", flush=True)
        if stop_after_groups is not None and newly_completed >= stop_after_groups:
            return state

    for name in METADATA_FILES:
        source_path = source / name
        if source_path.is_file():
            shutil.copy2(source_path, target / name)
    config_path = target / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "colib_model_family": "deepseek-v4",
            "colib_model_id": MODEL_ID,
            "colib_source_repository": SOURCE_REPO,
            "colib_source_revision": SOURCE_REVISION,
            "colib_default_context": DEFAULT_CONTEXT,
            "colib_validated_max_context": MAX_CONTEXT,
        }
    )
    atomic_json(config_path, config)
    marker = {**source_identity(), "verified_by": plan["source"]["verification"]}
    atomic_json(target / SOURCE_MARKER, marker)
    metadata = {
        name: {
            "size": (target / name).stat().st_size,
            "sha256": sha256_file(target / name),
        }
        for name in (*METADATA_FILES, SOURCE_MARKER)
        if (target / name).is_file()
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "source": plan["source"],
        "source_index_sha256": plan["source_index_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "repository": repository,
        "command": state["command"],
        "alignment": alignment,
        "packing": {
            "routed_experts": "layer/expert/projection/metadata",
            "dense": "source-shard segments",
            "dspark": "separate source-shard segments",
            "weight_transform": "none (native bytes preserved)",
        },
        "source_tensor_count": plan["source_tensor_count"],
        "output_record_count": plan["output_record_count"],
        "metadata": metadata,
        "segments": state["completed"],
        "inventory": state["inventory"],
    }
    atomic_json(target / MANIFEST_FILE, manifest)
    manifest_sha = sha256_file(target / MANIFEST_FILE)
    state["status"] = "complete"
    state["completed_at"] = utc_now()
    state["manifest"] = {"file": MANIFEST_FILE, "sha256": manifest_sha}
    state["updated_at"] = utc_now()
    atomic_json(state_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment", type=int, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--min-final-free-gib", type=float, default=100.0)
    parser.add_argument("--allow-fixture", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    groups, plan = build_plan(args.source, allow_fixture=args.allow_fixture)
    plan["projected_output_bytes"] = projected_output_bytes(groups, args.alignment)
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    state = convert(
        args.source,
        args.output,
        alignment=args.alignment,
        min_final_free=int(args.min_final_free_gib * 1024**3),
        allow_fixture=args.allow_fixture,
    )
    print(json.dumps({"status": state["status"], "manifest": state.get("manifest")}, indent=2))
    return 0 if state["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
