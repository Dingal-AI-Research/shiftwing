#!/usr/bin/env python3
"""Independently and resumably validate a converted DeepSeek-V4 container."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import convert_deepseek_v4 as converter
from deepseek_v4_spec import SOURCE_REPO, SOURCE_REVISION


SCHEMA = "colib.deepseek-v4.conversion-validation.v1"
COPY_CHUNK = 8 * 1024 * 1024


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
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
    }


def git_metadata(root: Path) -> dict[str, Any]:
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
    }


def _binding(
    source: Path,
    model: Path,
    manifest_path: Path,
    state_path: Path,
    allow_fixture: bool,
) -> dict[str, Any]:
    marker = source / converter.SOURCE_MARKER
    tools = Path(__file__).resolve().parent
    dependency_sha256 = {
        name: sha256_file(tools / name)
        for name in (
            "convert_deepseek_v4.py",
            "deepseek_v4_layout.py",
            "deepseek_v4_spec.py",
            "runtime_env.py",
            "validate_deepseek_v4_conversion.py",
        )
    }
    value = {
        "source": str(source),
        "model": str(model),
        "manifest_sha256": sha256_file(manifest_path),
        "conversion_state_sha256": sha256_file(state_path),
        "source_index_sha256": sha256_file(source / converter.INDEX_FILE),
        "source_marker_sha256": sha256_file(marker) if marker.is_file() else None,
        "dependency_sha256": dependency_sha256,
        "allow_fixture": allow_fixture,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    value["signature_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return value


def _metadata_for_record(record: converter.TensorRecord) -> dict[str, Any]:
    return {
        "name": record.name,
        "source_name": record.source_name,
        "source_shard": record.source_shard,
        "source_offset": record.source_offset,
        "slice_index": record.slice_index,
        "dtype": record.dtype,
        "shape": list(record.shape),
        "file": record.group,
        "layer": record.layer,
        "expert": record.expert,
        "projection": record.projection,
        "nbytes": record.nbytes,
    }


def _read_exact(handle: BinaryIO, nbytes: int, label: str) -> bytes:
    value = handle.read(nbytes)
    if len(value) != nbytes:
        raise EOFError(f"{label} ended {nbytes - len(value)} bytes early")
    return value


def _validate_group(
    source: Path,
    model: Path,
    group: str,
    expected_records: list[converter.TensorRecord],
    inventory: list[Mapping[str, Any]],
    segment: Mapping[str, Any],
) -> dict[str, Any]:
    if len(expected_records) != len(inventory):
        raise ValueError(f"record count mismatch for {group}")
    if segment.get("file") != group or segment.get("records") != len(inventory):
        raise ValueError(f"segment descriptor mismatch for {group}")
    output = model / group
    before = stat_identity(output)
    if before["size"] != segment.get("size"):
        raise ValueError(f"segment size mismatch for {group}")
    source_names = sorted({str(item["source_shard"]) for item in inventory})
    source_before = {name: stat_identity(source / name) for name in source_names}
    segment_digest = hashlib.sha256()
    converted_hashes: dict[str, str] = {}
    payload_bytes = 0
    position = 0
    with output.open("rb") as converted:
        for expected, item in zip(expected_records, inventory, strict=True):
            expected_metadata = _metadata_for_record(expected)
            for name, value in expected_metadata.items():
                if item.get(name) != value:
                    raise ValueError(
                        f"record metadata mismatch for {expected.name}: {name}"
                    )
            offset = item.get("offset")
            nbytes = item.get("nbytes")
            if not isinstance(offset, int) or offset < position:
                raise ValueError(f"invalid output offset for {expected.name}")
            if not isinstance(nbytes, int) or nbytes < 0:
                raise ValueError(f"invalid byte count for {expected.name}")
            padding = offset - position
            while padding:
                size = min(COPY_CHUNK, padding)
                block = _read_exact(converted, size, f"padding before {expected.name}")
                if any(block):
                    raise ValueError(f"nonzero alignment padding before {expected.name}")
                segment_digest.update(block)
                padding -= size
            converted_digest = hashlib.sha256()
            remaining = nbytes
            while remaining:
                size = min(COPY_CHUNK, remaining)
                block = _read_exact(
                    converted, size, f"converted record {expected.name}"
                )
                segment_digest.update(block)
                converted_digest.update(block)
                remaining -= size
            digest = converted_digest.hexdigest()
            if digest != item.get("sha256"):
                raise ValueError(f"record hash mismatch for {expected.name}")
            converted_hashes[expected.name] = digest
            payload_bytes += nbytes
            position = offset + nbytes
        if position != before["size"]:
            raise ValueError(
                f"unmanifested trailing bytes in {group}: {before['size'] - position}"
            )
        if converted.read(1):
            raise ValueError(f"segment grew during validation: {group}")
    observed_segment_hash = segment_digest.hexdigest()
    if observed_segment_hash != segment.get("sha256"):
        raise ValueError(f"segment hash mismatch for {group}")

    source_records = sorted(
        zip(expected_records, inventory, strict=True),
        key=lambda pair: (pair[0].source_shard, pair[0].source_offset),
    )
    source_handle: BinaryIO | None = None
    current_shard: str | None = None
    try:
        for expected, item in source_records:
            if current_shard != expected.source_shard:
                if source_handle is not None:
                    source_handle.close()
                current_shard = expected.source_shard
                source_handle = (source / current_shard).open("rb")
            assert source_handle is not None
            if source_handle.tell() != expected.source_offset:
                source_handle.seek(expected.source_offset)
            source_digest = hashlib.sha256()
            remaining = expected.nbytes
            while remaining:
                size = min(COPY_CHUNK, remaining)
                block = _read_exact(
                    source_handle, size, f"source record {expected.name}"
                )
                source_digest.update(block)
                remaining -= size
            if (
                source_digest.hexdigest() != converted_hashes[expected.name]
                or source_digest.hexdigest() != item.get("sha256")
            ):
                raise ValueError(f"native-byte hash mismatch for {expected.name}")
    finally:
        if source_handle is not None:
            source_handle.close()

    after = stat_identity(output)
    source_after = {name: stat_identity(source / name) for name in source_names}
    if before != after or source_before != source_after:
        raise ValueError(f"input changed during validation: {group}")
    return {
        "file": group,
        "sha256": observed_segment_hash,
        "size": before["size"],
        "records": len(inventory),
        "payload_bytes": payload_bytes,
        "source_bytes_compared": payload_bytes,
        "output_identity": after,
        "source_identities": source_after,
        "completed_at": utc_now(),
    }


def _completed_is_current(
    source: Path, model: Path, value: Mapping[str, Any]
) -> bool:
    try:
        if stat_identity(model / str(value["file"])) != value["output_identity"]:
            return False
        return all(
            stat_identity(source / name) == identity
            for name, identity in value["source_identities"].items()
        )
    except (FileNotFoundError, KeyError, TypeError):
        return False


def validate(
    source: Path,
    model: Path,
    output: Path,
    *,
    allow_fixture: bool = False,
    command: str | None = None,
    stop_after_groups: int | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    model = model.resolve()
    output = output.resolve()
    manifest_path = model / converter.MANIFEST_FILE
    conversion_state_path = model / converter.STATE_FILE
    manifest = read_json(manifest_path)
    conversion_state = read_json(conversion_state_path)
    if manifest.get("schema") != converter.MANIFEST_SCHEMA:
        raise ValueError("unrecognized model manifest schema")
    if conversion_state.get("schema") != converter.STATE_SCHEMA:
        raise ValueError("unrecognized conversion state schema")
    if conversion_state.get("status") != "complete":
        raise ValueError("conversion state is not complete")
    manifest_hash = sha256_file(manifest_path)
    if conversion_state.get("manifest") != {
        "file": converter.MANIFEST_FILE,
        "sha256": manifest_hash,
    }:
        raise ValueError("conversion state does not bind the current manifest")
    if manifest.get("segments") != conversion_state.get("completed"):
        raise ValueError("manifest/conversion segment state mismatch")
    if manifest.get("inventory") != conversion_state.get("inventory"):
        raise ValueError("manifest/conversion inventory mismatch")
    if manifest.get("plan_sha256") != conversion_state.get("plan_sha256"):
        raise ValueError("manifest/conversion plan mismatch")
    if not allow_fixture and (
        manifest.get("source", {}).get("repository") != SOURCE_REPO
        or manifest.get("source", {}).get("revision") != SOURCE_REVISION
    ):
        raise ValueError("manifest is not bound to the pinned DeepSeek release")
    groups, plan = converter.build_plan(source, allow_fixture=allow_fixture)
    for name in (
        "plan_sha256",
        "source_index_sha256",
        "source_tensor_count",
        "output_record_count",
    ):
        if manifest.get(name) != plan.get(name):
            raise ValueError(f"manifest plan field mismatch: {name}")
    segments = manifest.get("segments")
    inventory = manifest.get("inventory")
    if not isinstance(segments, dict) or set(segments) != set(groups):
        raise ValueError("manifest segment set does not match source plan")
    if not isinstance(inventory, dict) or set(inventory) != set(groups):
        raise ValueError("manifest inventory set does not match source plan")
    for name, metadata in manifest.get("metadata", {}).items():
        path = model / name
        if path.stat().st_size != metadata.get("size"):
            raise ValueError(f"metadata size mismatch: {name}")
        if sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"metadata hash mismatch: {name}")
    binding = _binding(
        source, model, manifest_path, conversion_state_path, allow_fixture
    )
    if output.is_file():
        state = read_json(output)
        if state.get("schema") != SCHEMA:
            raise ValueError("unrecognized validation state schema")
        if state.get("binding", {}).get("signature_sha256") != binding["signature_sha256"]:
            raise ValueError("validation state belongs to different inputs or tool code")
    else:
        root = Path(__file__).resolve().parents[2]
        state = {
            "schema": SCHEMA,
            "status": "running",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "command": command or shlex.join(sys.argv),
            "repository": git_metadata(root),
            "binding": binding,
            "completed": {},
        }
        atomic_json(output, state)
    newly_completed = 0
    for group in sorted(groups):
        prior = state["completed"].get(group)
        if prior and _completed_is_current(source, model, prior):
            print(f"validated {len(state['completed'])}/{len(groups)} {group} (resume)", flush=True)
            continue
        result = _validate_group(
            source, model, group, groups[group], inventory[group], segments[group]
        )
        state["completed"][group] = result
        state["updated_at"] = utc_now()
        atomic_json(output, state)
        newly_completed += 1
        print(f"validated {len(state['completed'])}/{len(groups)} {group}", flush=True)
        if stop_after_groups is not None and newly_completed >= stop_after_groups:
            return state
    completed = state["completed"]
    state["status"] = "complete"
    state["completed_at"] = utc_now()
    state["updated_at"] = state["completed_at"]
    state["totals"] = {
        "segments": len(completed),
        "records": sum(item["records"] for item in completed.values()),
        "segment_bytes": sum(item["size"] for item in completed.values()),
        "payload_bytes": sum(item["payload_bytes"] for item in completed.values()),
        "source_bytes_compared": sum(
            item["source_bytes_compared"] for item in completed.values()
        ),
    }
    atomic_json(output, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    state = validate(
        args.source, args.model, args.output, allow_fixture=args.allow_fixture
    )
    print(
        json.dumps(
            {
                "status": state["status"],
                "binding": state["binding"]["signature_sha256"],
                "totals": state.get("totals"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if state["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
