#!/usr/bin/env python3
"""Read-only installation and resource diagnostics for Shiftwing."""

from __future__ import annotations

import hashlib
import json
import re
import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

from tools.resource_plan import GIB, plan_resources


def _check(identifier: str, status: str, summary: str, **details: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"id": identifier, "status": status, "summary": summary}
    if details:
        item["details"] = details
    return item


def cuda_linkage(engine: Path) -> dict[str, bool]:
    if not engine.is_file() or os.name != "posix":
        return {"linked": False, "missing": False}
    try:
        result = subprocess.run(
            ["ldd", str(engine)],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"linked": False, "missing": False}
    lines = [line for line in result.stdout.splitlines() if "libcudart" in line]
    return {
        "linked": any("not found" not in line for line in lines),
        "missing": any("not found" in line for line in lines),
    }


def nvidia_gpu_info() -> dict[str, Any] | None:
    """Return the first NVIDIA device's name and MiB totals without CUDA init."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        line = result.stdout.splitlines()[0]
        name, total_mib, free_mib = (part.strip() for part in line.split(",", 2))
        return {
            "name": name,
            "total_gib": float(total_mib) / 1024.0,
            "free_gib": float(free_mib) / 1024.0,
        }
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


# A routed expert, in either naming convention the converters emit.
_ROUTED_EXPERT_RE = re.compile(r"\.(?:mlp\.)?experts\.\d+\.")

# Stored width divided by resident width. Weights that arrive unquantised are
# quantised to int8 when they are loaded; anything already packed is kept.
_RAM_DIVISOR = {"BF16": 2, "F16": 2, "F32": 4}


def _safetensors_header(path: Path) -> tuple[dict[str, Any], int, int]:
    """Read and bounds-check one safetensors header without touching payloads."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("truncated length prefix")
        (header_size,) = struct.unpack("<Q", raw)
        if header_size < 2 or header_size > 64 * 1024 * 1024:
            raise ValueError(f"invalid header size {header_size}")
        if 8 + header_size > size:
            raise ValueError("header extends beyond file")
        header = json.loads(handle.read(header_size))
    if not isinstance(header, dict):
        raise ValueError("header root is not an object")
    data_size = size - 8 - header_size
    tensors: dict[str, Any] = {}
    ranges = []
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError("invalid tensor record")
        offsets = record.get("data_offsets")
        shape = record.get("shape")
        dtype = record.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_size
            or not isinstance(shape, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
            or not isinstance(dtype, str)
        ):
            raise ValueError(f"invalid metadata for tensor {name!r}")
        tensors[name] = record
        ranges.append((offsets[0], offsets[1], name))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise ValueError(
                f"overlapping tensor payloads {previous[2]!r} and {current[2]!r}"
            )
    tensor_bytes = sum(end - start for start, end, _ in ranges)
    return tensors, tensor_bytes, size


def inspect_container(model_path: Path) -> dict[str, Any]:
    """Validate index-to-header ownership and payload byte bounds."""
    index_path = model_path / "model.safetensors.index.json"
    errors: list[str] = []
    expected: dict[str, str] = {}
    metadata_total = None
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("weight_map is missing or empty")
            for name, shard in weight_map.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(shard, str)
                    or Path(shard).name != shard
                ):
                    raise ValueError("weight_map contains an unsafe or invalid entry")
            expected = dict(weight_map)
            metadata = index.get("metadata") or {}
            if isinstance(metadata.get("total_size"), int):
                metadata_total = metadata["total_size"]
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return {
                "ok": False,
                "summary": f"tensor index is invalid: {error}",
                "details": {"index": str(index_path)},
            }
        shard_names = sorted(set(expected.values()))
    else:
        shard_names = sorted(path.name for path in model_path.glob("model*.safetensors"))
        if not shard_names:
            return {
                "ok": False,
                "summary": "no safetensors container files found",
                "details": {},
            }

    actual: dict[str, str] = {}
    tensor_bytes = 0
    file_bytes = 0
    parsed_shards = 0
    routed_bytes = 0
    resident_bytes = 0
    resident_ram_bytes = 0
    for shard_name in shard_names:
        shard_path = model_path / shard_name
        if not shard_path.is_file():
            errors.append(f"missing shard {shard_name}")
            continue
        try:
            tensors, payload_bytes, shard_bytes = _safetensors_header(shard_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{shard_name}: {error}")
            continue
        parsed_shards += 1
        tensor_bytes += payload_bytes
        file_bytes += shard_bytes
        for name, record in tensors.items():
            if name in actual:
                errors.append(f"duplicate tensor {name}")
            actual[name] = shard_name
            # Routed experts stream; everything else stays resident. Deriving
            # this as "total minus a computed expert size" was wrong by ~10 GiB
            # on GLM-5.3-Flash, because the MTP layer carries a full set of
            # routed experts that also stream. Reading it off the names is
            # exact and the headers are already open.
            start, end = record["data_offsets"]
            if _ROUTED_EXPERT_RE.search(name):
                routed_bytes += end - start
            else:
                span = end - start
                resident_bytes += span
                # What the engine will actually hold, which is not what the
                # file holds: it quantises on load, so a bf16 tensor costs half
                # its stored size in RAM. Measured on GLM-5.3-Flash: 18.1 GiB
                # of resident tensors on disk, 9 GiB resident in the process.
                # Sizing RAM off the file blocked a configuration that fits.
                resident_ram_bytes += span // _RAM_DIVISOR.get(record["dtype"], 1)

    if expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        wrong = sorted(
            name
            for name in set(expected) & set(actual)
            if expected[name] != actual[name]
        )
        if missing:
            errors.append(f"{len(missing)} indexed tensors missing from headers")
        if extra:
            errors.append(f"{len(extra)} header tensors absent from index")
        if wrong:
            errors.append(f"{len(wrong)} tensors mapped to the wrong shard")
        if metadata_total is not None and metadata_total != tensor_bytes:
            errors.append(
                f"metadata total_size {metadata_total} != header payload {tensor_bytes}"
            )
    details = {
        "shards": len(shard_names),
        "parsed_shards": parsed_shards,
        "indexed_tensors": len(expected),
        "header_tensors": len(actual),
        "tensor_bytes": tensor_bytes,
        "file_bytes": file_bytes,
        "routed_expert_bytes": routed_bytes,
        "resident_bytes": resident_bytes,
        "resident_ram_bytes": resident_ram_bytes,
        "metadata_total_size": metadata_total,
    }
    if errors:
        details["errors"] = errors[:8]
        return {
            "ok": False,
            "summary": f"container inventory failed ({len(errors)} issue(s))",
            "details": details,
        }
    return {
        "ok": True,
        "summary": (
            f"{len(actual):,} tensors across {parsed_shards} shard(s), "
            f"{tensor_bytes / GIB:.2f} GiB payload"
        ),
        "details": details,
    }


def inspect_quantization_manifest(
    model_path: Path, container: dict[str, Any]
) -> dict[str, Any]:
    """Cross-check an optional converter completion manifest against headers."""
    path = model_path / "quantization.json"
    if not path.is_file():
        return {
            "status": "skip",
            "summary": "quantization manifest not present (valid for oracle snapshots)",
            "details": {},
        }
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("root is not an object")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "fail",
            "summary": f"quantization manifest is invalid: {error}",
            "details": {"path": str(path)},
        }

    details = container.get("details", {})
    expected = {
        "data_bytes": details.get("tensor_bytes"),
        "tensor_count": details.get("header_tensors"),
        "output_shards": details.get("shards"),
    }
    errors: list[str] = []
    if manifest.get("complete") is not True:
        errors.append("complete is not true")
    for name, actual in expected.items():
        value = manifest.get(name)
        if name == "output_shards" and value is None:
            # Backward compatibility for manifests written before source-only
            # shards were observed in official multimodal checkpoints.
            value = manifest.get("source_shards")
        if isinstance(actual, int) and value != actual:
            errors.append(f"{name} {value!r} != container {actual}")
    source_shards = manifest.get("source_shards")
    output_shards = manifest.get("output_shards", source_shards)
    if (
        isinstance(source_shards, int)
        and isinstance(output_shards, int)
        and source_shards < output_shards
    ):
        errors.append(
            f"source_shards {source_shards} < output_shards {output_shards}"
        )
    loader = manifest.get("loader_inventory")
    if isinstance(loader, dict) and loader.get("present") != loader.get("expected"):
        errors.append(
            "loader inventory "
            f"{loader.get('present')!r} != {loader.get('expected')!r}"
        )
    result_details = {
        "path": str(path),
        "source": manifest.get("source"),
        "xbits": manifest.get("xbits"),
        "include_mtp": manifest.get("include_mtp"),
        "source_shards": source_shards,
        "output_shards": output_shards,
        "data_bytes": manifest.get("data_bytes"),
        "tensor_count": manifest.get("tensor_count"),
    }
    if errors:
        result_details["errors"] = errors
        return {
            "status": "fail",
            "summary": f"quantization manifest disagrees with container ({len(errors)} issue(s))",
            "details": result_details,
        }
    return {
        "status": "pass",
        "summary": "quantization completion manifest matches container",
        "details": result_details,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_conversion_ledger(
    model_path: Path,
    *,
    verify_hashes: bool = False,
    expected_source: str | None = None,
    expected_source_fingerprint: str | None = None,
    expected_source_shards: int | None = None,
    expected_output_shards: int | None = None,
    expected_logical_tensors: int | None = None,
    expected_physical_tensors: int | None = None,
    expected_data_bytes: int | None = None,
) -> dict[str, Any]:
    """Validate the converter's durable source-to-output commit ledger."""
    path = model_path / ".conversion-state.json"
    if not path.is_file():
        return {
            "status": "skip",
            "summary": "conversion ledger not present",
            "details": {},
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("root is not an object")
        signature = state.get("signature")
        completed = state.get("completed")
        inventory = state.get("inventory")
        if not isinstance(signature, dict):
            raise ValueError("signature is missing or invalid")
        if not isinstance(completed, dict):
            raise ValueError("completed map is missing or invalid")
        if not isinstance(inventory, dict):
            raise ValueError("inventory is missing or invalid")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "fail",
            "summary": f"conversion ledger is invalid: {error}",
            "details": {"path": str(path)},
        }

    errors: list[str] = []
    output_names: set[str] = set()
    output_count = 0
    data_bytes = 0
    file_bytes = 0
    tensor_count = 0
    hashes_checked = 0
    for source_name, record in completed.items():
        if (
            not isinstance(source_name, str)
            or Path(source_name).name != source_name
            or not isinstance(record, dict)
        ):
            errors.append(f"unsafe or invalid source record {source_name!r}")
            continue
        output = record.get("output")
        if output is None:
            if any(
                record.get(name) not in (None, 0)
                for name in ("file_size", "data_bytes", "sha256", "tensor_count")
            ):
                errors.append(f"{source_name}: source-only record has output metadata")
            continue
        if not isinstance(output, str) or Path(output).name != output:
            errors.append(f"{source_name}: unsafe or invalid output name")
            continue
        if output in output_names:
            errors.append(f"{source_name}: duplicate output {output}")
            continue
        output_names.add(output)
        output_count += 1
        target = model_path / output
        try:
            actual_size = target.stat().st_size
        except OSError as error:
            errors.append(f"{source_name}: output unavailable: {error}")
            continue
        expected_size = record.get("file_size")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            errors.append(f"{source_name}: invalid file_size")
        elif actual_size != expected_size:
            errors.append(
                f"{source_name}: file_size {expected_size} != output {actual_size}"
            )
        else:
            file_bytes += expected_size
        for name in ("data_bytes", "tensor_count"):
            value = record.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"{source_name}: invalid {name}")
            elif name == "data_bytes":
                data_bytes += value
            else:
                tensor_count += value
        expected_hash = record.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(char not in "0123456789abcdef" for char in expected_hash)
        ):
            errors.append(f"{source_name}: invalid sha256")
        elif verify_hashes:
            try:
                actual_hash = _sha256_file(target)
            except OSError as error:
                errors.append(f"{source_name}: hash read failed: {error}")
            else:
                hashes_checked += 1
                if actual_hash != expected_hash:
                    errors.append(
                        f"{source_name}: sha256 {expected_hash} != output {actual_hash}"
                    )

    manifest_path = model_path / "quantization.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("root is not an object")
            manifest = loaded
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"completion manifest is invalid: {error}")
    if manifest is not None and manifest.get("complete") is True:
        expected = {
            "source_shards": len(completed),
            "output_shards": output_count,
            "data_bytes": data_bytes,
            "tensor_count": tensor_count,
            "logical_tensor_count": len(inventory),
        }
        for name, actual in expected.items():
            if manifest.get(name) != actual:
                errors.append(
                    f"manifest {name} {manifest.get(name)!r} != ledger {actual}"
                )
        for name in (
            "source",
            "source_fingerprint",
            "xbits",
            "io_bits",
            "shared_bits",
            "group_size",
            "include_mtp",
        ):
            if manifest.get(name) != signature.get(name):
                errors.append(
                    f"manifest {name} {manifest.get(name)!r} "
                    f"!= ledger {signature.get(name)!r}"
                )

    observed = {
        "source": signature.get("source"),
        "source_fingerprint": signature.get("source_fingerprint"),
        "source_shards": len(completed),
        "output_shards": output_count,
        "logical_tensors": len(inventory),
        "physical_tensors": tensor_count,
        "data_bytes": data_bytes,
    }
    preregistered = {
        "source": expected_source,
        "source_fingerprint": expected_source_fingerprint,
        "source_shards": expected_source_shards,
        "output_shards": expected_output_shards,
        "logical_tensors": expected_logical_tensors,
        "physical_tensors": expected_physical_tensors,
        "data_bytes": expected_data_bytes,
    }
    for name, expected in preregistered.items():
        if expected is not None and observed[name] != expected:
            errors.append(
                f"ledger {name} {observed[name]!r} != expected {expected!r}"
            )

    details = {
        "path": str(path),
        "source": signature.get("source"),
        "source_fingerprint": signature.get("source_fingerprint"),
        "committed_sources": len(completed),
        "output_shards": output_count,
        "logical_tensors": len(inventory),
        "physical_tensors": tensor_count,
        "data_bytes": data_bytes,
        "file_bytes": file_bytes,
        "hashes_requested": verify_hashes,
        "hashes_checked": hashes_checked,
        "preregistered_expectations": {
            name: value for name, value in preregistered.items() if value is not None
        },
    }
    if errors:
        details["errors"] = errors[:8]
        return {
            "status": "fail",
            "summary": f"conversion ledger failed ({len(errors)} issue(s))",
            "details": details,
        }
    hash_summary = (
        f"; all {hashes_checked} output SHA-256 digests match"
        if verify_hashes
        else "; content hashes not requested"
    )
    return {
        "status": "pass",
        "summary": (
            f"{len(completed)} durable source commits match output metadata"
            f"{hash_summary}"
        ),
        "details": details,
    }


def run_doctor(
    model: str | Path,
    engine: str | Path,
    *,
    slots: int = 2,
    context: int = 4096,
    cuda_expert_gib: float = 7.0,
    ram_cache_gib: float = 18.0,
    runtime_headroom_gib: float = 1.5,
    verify_hashes: bool = False,
    expected_source: str | None = None,
    expected_source_fingerprint: str | None = None,
    expected_source_shards: int | None = None,
    expected_output_shards: int | None = None,
    expected_logical_tensors: int | None = None,
    expected_physical_tensors: int | None = None,
    expected_data_bytes: int | None = None,
) -> dict[str, Any]:
    model_path = Path(model).expanduser().resolve()
    engine_path = Path(engine).expanduser().resolve()
    checks: list[dict[str, Any]] = []

    readable = model_path.is_dir() and os.access(model_path, os.R_OK)
    checks.append(
        _check(
            "model.path",
            "pass" if readable else "fail",
            "model directory is readable" if readable else "model directory is missing or unreadable",
            path=str(model_path),
        )
    )

    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_ok = isinstance(config, dict)
    except (OSError, ValueError):
        config, config_ok = {}, False
    checks.append(
        _check(
            "model.config",
            "pass" if config_ok else "fail",
            "config.json is valid" if config_ok else "config.json is missing or invalid",
        )
    )
    tokenizer_ok = (model_path / "tokenizer.json").is_file()
    checks.append(
        _check(
            "model.tokenizer",
            "pass" if tokenizer_ok else "fail",
            "tokenizer.json found" if tokenizer_ok else "tokenizer.json is missing",
        )
    )

    state_path = model_path / ".conversion-state.json"
    conversion_complete = False
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            conversion_complete = bool(state.get("complete"))
        except (OSError, ValueError):
            pass
    index_path = model_path / "model.safetensors.index.json"
    inventory = inspect_container(model_path) if readable else {
        "ok": False,
        "summary": "model directory is unavailable",
        "details": {},
    }
    resumable_incomplete = state_path.is_file() and not index_path.is_file()
    conversion_complete = bool(inventory["ok"] and not resumable_incomplete)
    checks.append(
        _check(
            "model.container",
            "warn"
            if resumable_incomplete
            else "pass"
            if inventory["ok"]
            else "fail",
            "conversion is still resumable/incomplete"
            if resumable_incomplete
            else inventory["summary"],
            conversion_complete=conversion_complete,
            **inventory["details"],
        )
    )
    quantization = inspect_quantization_manifest(model_path, inventory)
    checks.append(
        _check(
            "model.quantization",
            quantization["status"],
            quantization["summary"],
            **quantization["details"],
        )
    )
    ledger = inspect_conversion_ledger(
        model_path,
        verify_hashes=verify_hashes,
        expected_source=expected_source,
        expected_source_fingerprint=expected_source_fingerprint,
        expected_source_shards=expected_source_shards,
        expected_output_shards=expected_output_shards,
        expected_logical_tensors=expected_logical_tensors,
        expected_physical_tensors=expected_physical_tensors,
        expected_data_bytes=expected_data_bytes,
    )
    checks.append(
        _check(
            "model.ledger",
            ledger["status"],
            ledger["summary"],
            **ledger["details"],
        )
    )

    executable = engine_path.is_file() and (
        os.name == "nt" or os.access(engine_path, os.X_OK)
    )
    checks.append(
        _check(
            "engine.binary",
            "pass" if executable else "fail",
            "engine executable is ready" if executable else "engine is missing or not executable",
            path=str(engine_path),
        )
    )
    linkage = cuda_linkage(engine_path)
    gpu = nvidia_gpu_info()
    accelerator_status = (
        "fail"
        if linkage["missing"]
        else "warn"
        if linkage["linked"] and gpu is None
        else "pass"
        if linkage["linked"]
        else "skip"
    )
    checks.append(
        _check(
            "accelerator.cuda",
            accelerator_status,
            (
                f"CUDA runtime linked; {gpu['name']} has "
                f"{gpu['total_gib']:.2f} GiB total / {gpu['free_gib']:.2f} GiB free"
            )
            if linkage["linked"] and gpu
            else "CUDA runtime is linked but no NVIDIA device was queried"
            if linkage["linked"]
            else "CUDA runtime library is missing"
            if linkage["missing"]
            else "CPU build; CUDA is optional",
            **(gpu or {}),
        )
    )

    plan = None
    if config_ok:
        memory_total = 0
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            memory_total = int(pages) * int(page_size)
        except (AttributeError, OSError, ValueError):
            pass
        disk_free = shutil.disk_usage(model_path if model_path.exists() else ".").free
        # GLM's container size is measured rather than predicted; summing the
        # snapshot is cheap and is the ground truth the disk check is about.
        # Both numbers come from the headers inspect_container already read,
        # so the plan sizes the real container rather than a prediction of it.
        container = inventory.get("details") or {}
        snapshot_bytes = int(container.get("tensor_bytes") or 0)
        routed_bytes = int(container.get("routed_expert_bytes") or 0)
        resident_ram_bytes = int(container.get("resident_ram_bytes") or 0)
        if not snapshot_bytes and model_path.exists():
            try:
                snapshot_bytes = sum(
                    entry.stat().st_size
                    for entry in model_path.glob("*.safetensors")
                )
            except OSError:
                snapshot_bytes = 0
        plan = plan_resources(
            config,
            snapshot_bytes=snapshot_bytes,
            routed_bytes=routed_bytes,
            resident_ram_bytes=resident_ram_bytes,
            slots=slots,
            context=context,
            cuda_expert_gib=cuda_expert_gib,
            ram_cache_gib=ram_cache_gib,
            system_ram_gib=memory_total / GIB if memory_total else 0.0,
            gpu_gib=gpu["free_gib"] if gpu else 0.0,
            disk_free_gib=disk_free / GIB,
            runtime_headroom_gib=runtime_headroom_gib,
            include_mtp=False,
        )
        checks.append(
            _check(
                "runtime.state",
                "pass",
                f"{plan['state_bytes']['total'] / GIB:.2f} GiB for {slots} slots × {context} context",
            )
        )
        failed_resources = [
            name for name, passed in plan["checks"].items() if not passed
        ]
        checks.append(
            _check(
                "runtime.resources",
                "fail" if failed_resources else "pass",
                (
                    "resource plan fits measured host, currently free device, and disk capacity"
                    if not failed_resources
                    else "resource plan exceeds " + ", ".join(failed_resources)
                ),
                runtime_headroom_gib=runtime_headroom_gib,
                checks=plan["checks"],
            )
        )

    statuses = {item["status"] for item in checks}
    status = "error" if "fail" in statuses else "warning" if "warn" in statuses else "ok"
    return {
        "schema_version": 1,
        "status": status,
        "model": str(model_path),
        "engine": str(engine_path),
        "checks": checks,
        "plan": plan,
    }


def format_doctor(report: dict[str, Any]) -> str:
    lines = [f"shiftwing doctor · {report['model']}"]
    for item in report["checks"]:
        lines.append(f"[{item['status']:>4}] {item['id']:<18} {item['summary']}")
    lines.append(f"result {report['status']}")
    return "\n".join(lines)


def exit_code(report: dict[str, Any]) -> int:
    return 1 if report["status"] == "error" else 0
