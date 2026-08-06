#!/usr/bin/env python3
"""Add resumable grouped-int2/int3 routed-expert sidecars to a colib snapshot.

The source snapshot remains immutable.  Each output tensor appends ``.q2`` or
``.q3`` to the canonical q4 payload name, followed by matching scale and type
tensors.  The C runtime selects these alternatives only when the corresponding
``EXPERT_Q2=1`` or ``EXPERT_Q3=1`` switch is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from convert_qwen import (
    dequantize_int4_grouped,
    quantize_int2_grouped,
    quantize_int3_grouped,
)


INDEX_FILE = "model.safetensors.index.json"
EXPERT_RE = re.compile(
    r"^(?P<prefix>(?:model(?:\.language_model)?\.)?layers\.(?P<layer>\d+)"
    r"\.mlp\.experts\.(?P<expert>\d+)\.)"
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_map(snapshot: Path) -> dict[str, str]:
    index_path = snapshot / INDEX_FILE
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        mapping = payload.get("weight_map")
        if not isinstance(mapping, dict):
            raise SystemExit(f"{index_path} has no weight_map object")
        return {str(name): str(file) for name, file in mapping.items()}
    mapping: dict[str, str] = {}
    for path in sorted(snapshot.glob("*.safetensors")):
        if path.name.startswith(("expert-q2-", "expert-q3-")):
            continue
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in mapping:
                    raise SystemExit(f"duplicate tensor {name}")
                mapping[name] = path.name
    if not mapping:
        raise SystemExit(f"no safetensors found in {snapshot}")
    return mapping


def read_config(snapshot: Path) -> tuple[int, int]:
    payload = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    config = payload.get("text_config", payload)
    try:
        return int(config["hidden_size"]), int(config["moe_intermediate_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("config lacks hidden_size/moe_intermediate_size") from error


def load_tensor(snapshot: Path, mapping: dict[str, str], name: str) -> torch.Tensor:
    filename = mapping.get(name)
    if filename is None:
        for suffix in (".qs", ".qtype"):
            if name.endswith(suffix):
                filename = mapping.get(name[: -len(suffix)])
                if filename is not None:
                    break
    if filename is None:
        raise SystemExit(f"missing source tensor {name}")
    with safe_open(snapshot / filename, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def load_optional_aux(
    snapshot: Path, mapping: dict[str, str], weight_name: str, suffix: str
) -> torch.Tensor | None:
    name = f"{weight_name}{suffix}"
    filename = mapping.get(name, mapping.get(weight_name))
    if filename is None:
        return None
    with safe_open(snapshot / filename, framework="pt", device="cpu") as handle:
        if name not in handle.keys():
            return None
        return handle.get_tensor(name)


def source_group_size(
    packed: torch.Tensor, scales: torch.Tensor, rows: int
) -> int:
    if packed.ndim != 2 or scales.ndim != 2 or packed.shape[0] != rows:
        raise SystemExit(
            f"bad q4 geometry payload={tuple(packed.shape)} "
            f"scales={tuple(scales.shape)} rows={rows}"
        )
    groups = scales.shape[-1]
    if groups <= 0 or packed.shape[-1] * 2 % groups:
        raise SystemExit("q4 payload and scale grid do not define whole groups")
    return packed.shape[-1] * 2 // groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="sidecar directory (defaults to the snapshot directory)",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--experts-per-file", type=int, default=16)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="quantize independent expert projections concurrently",
    )
    parser.add_argument(
        "--resume-hash-workers",
        type=int,
        help="validate ledgered chunk hashes concurrently (defaults to --workers)",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        help="intra-operation CPU threads (defaults to PyTorch's setting)",
    )
    parser.add_argument("--bits", type=int, choices=(2, 3), default=2)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--max-experts", type=int)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace completed chunks instead of validating and resuming them",
    )
    parser.add_argument(
        "--adopt-existing",
        action="store_true",
        help=(
            "validate and ledger exact unmanifested sidecar chunks left by an "
            "older interrupted converter"
        ),
    )
    return parser.parse_args()


def inspect_sidecar_chunk(
    path: Path,
    *,
    layer: int,
    chunk: list[int],
    projections: dict[int, dict[str, str]],
    bits: int,
    group_size: int,
    hidden: int,
    intermediate: int,
) -> dict[str, Any]:
    """Validate one unledgered atomic chunk before explicitly adopting it."""
    suffix = f"q{bits}"
    expected: dict[str, tuple[torch.dtype, tuple[int, ...], int | None]] = {}
    for expert in chunk:
        names = projections[expert]
        if set(names) != {"gate_proj", "up_proj", "down_proj"}:
            raise SystemExit(
                f"layer {layer} expert {expert} lacks three projections"
            )
        for projection in ("gate_proj", "up_proj", "down_proj"):
            source_name = names[projection]
            rows = intermediate if projection != "down_proj" else hidden
            columns = hidden if projection != "down_proj" else intermediate
            groups = (columns + group_size - 1) // group_size
            padded = groups * group_size
            packed_columns = padded // 4 if bits == 2 else padded // 8 * 3
            target = f"{source_name}.{suffix}"
            expected[target] = (torch.uint8, (rows, packed_columns), None)
            expected[f"{target}.qs"] = (torch.float32, (rows, groups), None)
            expected[f"{target}.qtype"] = (torch.uint8, (1,), bits)
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            observed = set(handle.keys())
            if observed != set(expected):
                missing = sorted(set(expected) - observed)
                extra = sorted(observed - set(expected))
                raise SystemExit(
                    f"{path}: sidecar tensor inventory mismatch "
                    f"missing={missing[:3]} extra={extra[:3]}"
                )
            for name, (dtype, shape, scalar) in expected.items():
                value = handle.get_tensor(name)
                if value.dtype != dtype or tuple(value.shape) != shape:
                    raise SystemExit(
                        f"{path}:{name} has {value.dtype}/{tuple(value.shape)}, "
                        f"expected {dtype}/{shape}"
                    )
                if scalar is not None and int(value.reshape(-1)[0]) != scalar:
                    raise SystemExit(
                        f"{path}:{name} does not declare qtype={scalar}"
                    )
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot validate existing sidecar {path}: {error}") from error
    return {
        "file": path.name,
        "layer": layer,
        "first_expert": chunk[0],
        "last_expert": chunk[-1],
        "experts": len(chunk),
        "tensor_count": len(expected),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def publish_manifest(
    path: Path,
    *,
    bits: int,
    snapshot: Path,
    config_hash: str,
    signature: dict[str, Any],
    layers: int,
    records: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: item["file"])
    manifest = {
        "format": f"colib-routed-expert-int{bits}-sidecar-v1",
        "complete": complete,
        "snapshot": str(snapshot),
        "config_sha256": config_hash,
        "signature": signature,
        "layers": layers,
        "files": ordered,
        "file_count": len(ordered),
        "tensor_count": sum(int(item["tensor_count"]) for item in ordered),
        "data_bytes": sum(int(item["bytes"]) for item in ordered),
    }
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return manifest


def main() -> None:
    args = parse_args()
    snapshot = args.snapshot.resolve()
    output = (args.output or snapshot).resolve()
    alignment = 8 if args.bits == 3 else 4
    if args.group_size <= 0 or args.group_size % alignment:
        raise SystemExit(
            f"--group-size must be a positive multiple of {alignment} for "
            f"int{args.bits}"
        )
    resume_hash_workers = (
        args.workers
        if args.resume_hash_workers is None
        else args.resume_hash_workers
    )
    if (
        args.iterations < 1
        or args.experts_per_file < 1
        or args.workers < 1
        or resume_hash_workers < 1
    ):
        raise SystemExit(
            "iterations, file size, quantization workers, and resume hash workers "
            "must be positive"
        )
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise SystemExit("--torch-threads must be positive")
        torch.set_num_threads(args.torch_threads)
    if args.max_layers is not None and args.max_layers < 1:
        raise SystemExit("--max-layers must be positive")
    if args.max_experts is not None and args.max_experts < 1:
        raise SystemExit("--max-experts must be positive")
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"q{args.bits}"
    manifest_name = f"expert-{suffix}.json"

    mapping = tensor_map(snapshot)
    hidden, intermediate = read_config(snapshot)
    grouped: dict[int, dict[int, dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for name in mapping:
        match = EXPERT_RE.match(name)
        if not match:
            continue
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        grouped[layer][expert][match.group("projection")] = name
    layers = sorted(grouped)
    if args.max_layers is not None:
        layers = layers[: args.max_layers]
    if not layers:
        raise SystemExit("no unfused routed-expert tensors found")

    config_hash = sha256_file(snapshot / "config.json")
    index_path = snapshot / INDEX_FILE
    source_identity = (
        sha256_file(index_path) if index_path.is_file() else config_hash
    )
    prior_path = output / manifest_name
    prior: dict[str, Any] = {}
    if prior_path.is_file() and not args.force:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        signature = prior.get("signature")
        expected = {
            "source_identity": source_identity,
            "group_size": args.group_size,
            "iterations": args.iterations,
            "experts_per_file": args.experts_per_file,
        }
        if signature != expected:
            raise SystemExit(
                f"existing {manifest_name} has a different conversion signature; "
                "use --force or a different output directory"
            )
    completed: dict[str, dict[str, Any]] = {
        item["file"]: item
        for item in prior.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("file"), str)
    }
    signature = {
        "source_identity": source_identity,
        "group_size": args.group_size,
        "iterations": args.iterations,
        "experts_per_file": args.experts_per_file,
    }
    existing = sorted(output.glob(f"expert-{suffix}-l*-e*.safetensors"))
    unledgered = [path for path in existing if path.name not in completed]
    if unledgered and not args.adopt_existing and not args.force:
        raise SystemExit(
            f"found {len(unledgered)} unledgered {suffix} sidecar files; "
            "use --adopt-existing to validate them or --force to replace them"
        )
    verified_completed: set[str] = set()
    if not args.force:
        candidates = [path for path in existing if path.name in completed]
        if candidates:
            with ThreadPoolExecutor(max_workers=resume_hash_workers) as executor:
                observed = executor.map(sha256_file, candidates)
                for path, digest in zip(candidates, observed, strict=True):
                    if completed[path.name].get("sha256") == digest:
                        verified_completed.add(path.name)
            print(
                f"resume-validated={len(verified_completed)}/{len(candidates)} "
                f"hash-workers={resume_hash_workers}",
                flush=True,
            )
    records: list[dict[str, Any]] = []
    converted = skipped = adopted = total_tensors = total_bytes = 0

    for layer in layers:
        experts = sorted(grouped[layer])
        if args.max_experts is not None:
            experts = experts[: args.max_experts]
        for offset in range(0, len(experts), args.experts_per_file):
            chunk = experts[offset : offset + args.experts_per_file]
            filename = (
                f"expert-{suffix}-l{layer:03d}-e{chunk[0]:04d}-{chunk[-1]:04d}"
                ".safetensors"
            )
            path = output / filename
            old = completed.get(filename)
            adopted_now = False
            if (
                not args.force
                and old is None
                and args.adopt_existing
                and path.is_file()
            ):
                old = inspect_sidecar_chunk(
                    path,
                    layer=layer,
                    chunk=chunk,
                    projections=grouped[layer],
                    bits=args.bits,
                    group_size=args.group_size,
                    hidden=hidden,
                    intermediate=intermediate,
                )
                completed[filename] = old
                adopted += 1
                adopted_now = True
            if (
                not args.force
                and old
                and path.is_file()
                and (adopted_now or filename in verified_completed)
            ):
                records.append(old)
                skipped += 1
                total_tensors += int(old["tensor_count"])
                total_bytes += int(old["bytes"])
                continue

            tasks: list[tuple[int, str, str]] = []
            for expert in chunk:
                projections = grouped[layer][expert]
                if set(projections) != {"gate_proj", "up_proj", "down_proj"}:
                    raise SystemExit(
                        f"layer {layer} expert {expert} lacks three projections"
                    )
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    tasks.append((expert, projection, projections[projection]))

            def convert_projection(
                task: tuple[int, str, str],
            ) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
                _expert, projection, name = task
                packed = load_tensor(snapshot, mapping, name)
                scales = load_tensor(snapshot, mapping, f"{name}.qs")
                qtype = load_optional_aux(snapshot, mapping, name, ".qtype")
                if packed.dtype != torch.uint8 or (
                    qtype is not None and int(qtype.reshape(-1)[0]) != 4
                ):
                    raise SystemExit(f"{name} is not a qtype=4 uint8 matrix")
                columns = hidden if projection != "down_proj" else intermediate
                group_size = source_group_size(packed, scales, packed.shape[0])
                value = dequantize_int4_grouped(
                    packed,
                    scales,
                    columns=columns,
                    group_size=group_size,
                )
                quantizer = (
                    quantize_int2_grouped
                    if args.bits == 2
                    else quantize_int3_grouped
                )
                packed_low, low_scale = quantizer(
                    value,
                    group_size=args.group_size,
                    iterations=args.iterations,
                )
                return (
                    f"{name}.{suffix}",
                    packed_low,
                    low_scale,
                    torch.tensor([args.bits], dtype=torch.uint8),
                )

            payload: dict[str, torch.Tensor] = {}
            if args.workers == 1:
                converted_projections = map(convert_projection, tasks)
                executor = None
            else:
                executor = ThreadPoolExecutor(max_workers=args.workers)
                converted_projections = executor.map(convert_projection, tasks)
            try:
                for target, packed_low, low_scale, qtype in converted_projections:
                    payload[target] = packed_low
                    payload[f"{target}.qs"] = low_scale
                    payload[f"{target}.qtype"] = qtype
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            save_file(payload, temporary)
            os.replace(temporary, path)
            record = {
                "file": filename,
                "layer": layer,
                "first_expert": chunk[0],
                "last_expert": chunk[-1],
                "experts": len(chunk),
                "tensor_count": len(payload),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            records.append(record)
            completed[filename] = record
            converted += 1
            total_tensors += len(payload)
            total_bytes += path.stat().st_size
            publish_manifest(
                prior_path,
                bits=args.bits,
                snapshot=snapshot,
                config_hash=config_hash,
                signature=signature,
                layers=len(layers),
                records=list(completed.values()),
                complete=False,
            )
            print(
                f"wrote {filename}: {len(chunk)} experts, "
                f"{path.stat().st_size / (1 << 20):.2f} MiB",
                flush=True,
            )

    manifest = publish_manifest(
        prior_path,
        bits=args.bits,
        snapshot=snapshot,
        config_hash=config_hash,
        signature=signature,
        layers=len(layers),
        records=records,
        complete=args.max_layers is None and args.max_experts is None,
    )
    print(
        f"complete={manifest['complete']} converted={converted} "
        f"resumed={skipped} adopted={adopted} files={len(records)} "
        f"bytes={total_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
