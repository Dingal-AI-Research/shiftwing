#!/usr/bin/env python3
"""Independently audit a complete routed-expert low-bit sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from safetensors import safe_open


INDEX_FILE = "model.safetensors.index.json"
EXPERT_RE = re.compile(
    r"^(?P<prefix>(?:model(?:\.language_model)?\.)?layers\.(?P<layer>\d+)"
    r"\.mlp\.experts\.(?P<expert>\d+)\.)"
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root is not an object")
    return value


def text_config(snapshot: Path) -> dict[str, Any]:
    config = read_object(snapshot / "config.json")
    value = config.get("text_config", config)
    if not isinstance(value, dict):
        raise ValueError("config text_config is not an object")
    return value


def source_experts(snapshot: Path) -> dict[int, dict[int, dict[str, str]]]:
    index_path = snapshot / INDEX_FILE
    index = read_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path}: weight_map is not an object")
    grouped: dict[int, dict[int, dict[str, str]]] = {}
    for raw_name in weight_map:
        name = str(raw_name)
        match = EXPERT_RE.match(name)
        if not match:
            continue
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        grouped.setdefault(layer, {}).setdefault(expert, {})[
            match.group("projection")
        ] = name
    return grouped


def expected_tensor_specs(
    projections: dict[int, dict[str, str]],
    experts: range,
    *,
    bits: int,
    hidden: int,
    intermediate: int,
    group_size: int,
) -> dict[str, tuple[str, tuple[int, ...], int | None]]:
    suffix = f"q{bits}"
    expected: dict[str, tuple[str, tuple[int, ...], int | None]] = {}
    for expert in experts:
        names = projections.get(expert)
        if names is None or set(names) != {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"expert {expert} does not have exactly three projections")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            rows = intermediate if projection != "down_proj" else hidden
            columns = hidden if projection != "down_proj" else intermediate
            groups = math.ceil(columns / group_size)
            padded = groups * group_size
            packed_columns = padded // 4 if bits == 2 else padded // 8 * 3
            target = f"{names[projection]}.{suffix}"
            expected[target] = ("U8", (rows, packed_columns), None)
            expected[f"{target}.qs"] = ("F32", (rows, groups), None)
            expected[f"{target}.qtype"] = ("U8", (1,), bits)
    return expected


def audit(
    snapshot: Path,
    *,
    bits: int,
    verify_hashes: bool,
    hash_workers: int = 1,
) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    if hash_workers < 1:
        raise ValueError("hash_workers must be positive")
    manifest_path = snapshot / f"expert-q{bits}.json"
    manifest = read_object(manifest_path)
    config_path = snapshot / "config.json"
    index_path = snapshot / INDEX_FILE
    config = text_config(snapshot)
    try:
        layers = int(config["num_hidden_layers"])
        experts_per_layer = int(config["num_experts"])
        hidden = int(config["hidden_size"])
        intermediate = int(config["moe_intermediate_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("config lacks routed-expert geometry") from error

    expected_format = f"colib-routed-expert-int{bits}-sidecar-v1"
    if manifest.get("format") != expected_format:
        raise ValueError(
            f"format {manifest.get('format')!r} does not equal {expected_format!r}"
        )
    if manifest.get("complete") is not True:
        raise ValueError("sidecar manifest is incomplete")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("sidecar config hash does not match config.json")
    if manifest.get("layers") != layers:
        raise ValueError(
            f"manifest layers={manifest.get('layers')} does not equal config layers={layers}"
        )

    signature = manifest.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("sidecar signature is not an object")
    source_identity = sha256_file(index_path)
    if signature.get("source_identity") != source_identity:
        raise ValueError("sidecar source identity does not match the base tensor index")
    group_size = signature.get("group_size")
    iterations = signature.get("iterations")
    experts_per_file = signature.get("experts_per_file")
    alignment = 8 if bits == 3 else 4
    if not isinstance(group_size, int) or group_size <= 0 or group_size % alignment:
        raise ValueError(
            f"signature group_size must be a positive multiple of {alignment}"
        )
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("signature iterations must be positive")
    if not isinstance(experts_per_file, int) or experts_per_file <= 0:
        raise ValueError("signature experts_per_file must be positive")

    grouped = source_experts(snapshot)
    expected_layers = set(range(layers))
    if set(grouped) != expected_layers:
        raise ValueError("source expert layers do not exactly cover the config")
    expected_experts = set(range(experts_per_layer))
    for layer in range(layers):
        if set(grouped[layer]) != expected_experts:
            raise ValueError(f"source layer {layer} does not exactly cover all experts")

    records = manifest.get("files")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("manifest files is not an object list")
    expected_files = layers * math.ceil(experts_per_layer / experts_per_file)
    expected_tensors = layers * experts_per_layer * 3 * 3
    if manifest.get("file_count") != expected_files or len(records) != expected_files:
        raise ValueError(
            f"file count does not equal expected inventory {expected_files}"
        )
    if manifest.get("tensor_count") != expected_tensors:
        raise ValueError(
            f"tensor count does not equal expected inventory {expected_tensors}"
        )

    expected_records: dict[str, tuple[int, int, int]] = {}
    for layer in range(layers):
        for first in range(0, experts_per_layer, experts_per_file):
            last = min(first + experts_per_file, experts_per_layer) - 1
            filename = (
                f"expert-q{bits}-l{layer:03d}-e{first:04d}-{last:04d}.safetensors"
            )
            expected_records[filename] = (layer, first, last)
    ledger = {str(row.get("file")): row for row in records}
    if len(ledger) != len(records) or set(ledger) != set(expected_records):
        raise ValueError("manifest filenames do not exactly match the expected chunks")
    physical = {
        path.name
        for path in snapshot.glob(f"expert-q{bits}-l*-e*.safetensors")
    }
    if physical != set(expected_records):
        raise ValueError("physical sidecar files do not exactly match the manifest")

    observed_hashes: dict[str, str] = {}
    if verify_hashes:
        filenames = sorted(expected_records)
        with ThreadPoolExecutor(max_workers=hash_workers) as executor:
            digests = executor.map(
                sha256_file,
                (snapshot / filename for filename in filenames),
            )
            observed_hashes = dict(zip(filenames, digests, strict=True))

    total_bytes = 0
    verified_hashes = 0
    verified_headers = 0
    for filename, (layer, first, last) in sorted(expected_records.items()):
        row = ledger[filename]
        path = snapshot / filename
        size = path.stat().st_size
        expert_count = last - first + 1
        tensor_count = expert_count * 9
        expected_metadata = {
            "layer": layer,
            "first_expert": first,
            "last_expert": last,
            "experts": expert_count,
            "tensor_count": tensor_count,
            "bytes": size,
        }
        mismatches = {
            key: {"expected": value, "observed": row.get(key)}
            for key, value in expected_metadata.items()
            if row.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{filename}: ledger mismatch {mismatches}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{filename}: invalid ledger SHA-256")
        if verify_hashes:
            observed = observed_hashes[filename]
            if observed != digest:
                raise ValueError(f"{filename}: SHA-256 mismatch")
            verified_hashes += 1

        expected = expected_tensor_specs(
            grouped[layer],
            range(first, last + 1),
            bits=bits,
            hidden=hidden,
            intermediate=intermediate,
            group_size=group_size,
        )
        with safe_open(path, framework="pt", device="cpu") as handle:
            names = set(handle.keys())
            if names != set(expected):
                raise ValueError(f"{filename}: tensor names do not match the chunk")
            for name, (dtype, shape, scalar) in expected.items():
                value = handle.get_slice(name)
                observed_shape = tuple(value.get_shape())
                observed_dtype = value.get_dtype()
                if observed_dtype != dtype or observed_shape != shape:
                    raise ValueError(
                        f"{filename}:{name}: {observed_dtype}/{observed_shape} "
                        f"does not equal {dtype}/{shape}"
                    )
                if scalar is not None:
                    observed_scalar = int(handle.get_tensor(name).reshape(-1)[0])
                    if observed_scalar != scalar:
                        raise ValueError(
                            f"{filename}:{name}: qtype={observed_scalar}, expected {scalar}"
                        )
        verified_headers += 1
        total_bytes += size

    if manifest.get("data_bytes") != total_bytes:
        raise ValueError(
            f"manifest data_bytes={manifest.get('data_bytes')} does not equal {total_bytes}"
        )
    return {
        "schema_version": 1,
        "status": "ok",
        "snapshot": str(snapshot),
        "bits": bits,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_identity": source_identity,
        "layers": layers,
        "experts_per_layer": experts_per_layer,
        "group_size": group_size,
        "iterations": iterations,
        "experts_per_file": experts_per_file,
        "file_count": expected_files,
        "tensor_count": expected_tensors,
        "data_bytes": total_bytes,
        "headers_verified": verified_headers,
        "hashes_verified": verified_hashes,
        "hash_workers": hash_workers if verify_hashes else 0,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(2, 3), required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--hash-workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.snapshot,
        bits=args.bits,
        verify_hashes=args.verify_hashes,
        hash_workers=args.hash_workers,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
