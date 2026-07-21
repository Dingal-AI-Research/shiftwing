#!/usr/bin/env python3
"""Qwen3.5 quantization primitives and resumable shard-streaming converter.

The converter accepts either a local Hugging Face snapshot or ``--repo``.  It
loads one source shard at a time, drops vision tensors, unfuses routed experts,
applies the Qwen3.5 mixed-precision map, atomically writes one output shard,
and can delete a downloaded source shard before advancing.  A conversion state
file makes interrupted runs resumable without trusting partial output files.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


DEFAULT_GROUP_SIZE = 128
STATE_FILE = ".conversion-state.json"
INDEX_FILE = "model.safetensors.index.json"
METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
)


def _require_finite(weight: torch.Tensor) -> None:
    if not torch.isfinite(weight).all():
        raise ValueError("cannot quantize a tensor containing NaN or infinity")


def _matrix_view(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Return rows×columns view used by all row-wise quantizers."""
    if weight.ndim < 2:
        raise ValueError(f"quantization requires ndim >= 2, got {weight.shape}")
    shape = weight.shape
    return weight.detach().float().reshape(-1, shape[-1]), shape


def quantize_int8_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8, stored as two's-complement U8 plus F32 scales."""
    _require_finite(weight)
    rows, shape = _matrix_view(weight)
    max_abs = rows.abs().amax(dim=1)
    scales = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
    signed = torch.round(rows / scales[:, None]).clamp_(-127, 127).to(torch.int8)
    packed = signed.view(torch.uint8).reshape(*shape[:-1], shape[-1]).contiguous()
    return packed, scales.reshape(*shape[:-1]).float().contiguous()


def dequantize_int8_rows(
    packed: torch.Tensor, scales: torch.Tensor, *, columns: int | None = None
) -> torch.Tensor:
    """Decode :func:`quantize_int8_rows` to float32."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"int8 payload must be uint8, got {packed.dtype}")
    cols = packed.shape[-1] if columns is None else columns
    if packed.shape[-1] != cols:
        raise ValueError("int8 payload column count does not match")
    signed = packed.contiguous().view(torch.int8).float()
    return signed * scales.float().unsqueeze(-1)


def quantize_int4_grouped(
    weight: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric grouped int4, two signed nibbles per byte, low nibble first.

    Groups run along the final tensor dimension.  The final group and final
    nibble are zero padded when necessary; callers recover the logical shape
    from the model configuration/tensor name.
    """
    _require_finite(weight)
    if group_size <= 0 or group_size % 2:
        raise ValueError("group_size must be a positive even integer")
    rows, shape = _matrix_view(weight)
    nrows, cols = rows.shape
    groups = math.ceil(cols / group_size)
    padded_cols = groups * group_size
    if padded_cols != cols:
        rows = torch.nn.functional.pad(rows, (0, padded_cols - cols))
    grouped = rows.reshape(nrows, groups, group_size)
    max_abs = grouped.abs().amax(dim=2)
    scales = torch.where(max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs))
    signed = torch.round(grouped / scales[:, :, None]).clamp_(-8, 7).to(torch.int8)
    flat = signed.reshape(nrows, padded_cols)
    if padded_cols & 1:
        flat = torch.nn.functional.pad(flat, (0, 1))
    # colibri wire format is offset-coded: nibble 0..15 decodes to -8..7.
    nibble = (flat.to(torch.int16) + 8).to(torch.uint8)
    packed = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
    packed_shape = (*shape[:-1], packed.shape[-1])
    scale_shape = (*shape[:-1], groups)
    return packed.reshape(packed_shape).contiguous(), scales.reshape(scale_shape).float().contiguous()


def dequantize_int4_grouped(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    columns: int,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> torch.Tensor:
    """Decode grouped int4 to float32, trimming padding to ``columns``."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"int4 payload must be uint8, got {packed.dtype}")
    if columns <= 0 or group_size <= 0:
        raise ValueError("columns and group_size must be positive")
    rows = packed.reshape(-1, packed.shape[-1])
    lo = rows & 0xF
    hi = rows >> 4
    vals = torch.stack((lo, hi), dim=-1).reshape(rows.shape[0], -1).to(torch.int8)
    vals = vals.float() - 8.0
    groups = math.ceil(columns / group_size)
    vals = vals[:, : groups * group_size].reshape(rows.shape[0], groups, group_size)
    dequant = vals * scales.reshape(rows.shape[0], groups).float()[:, :, None]
    logical = dequant.reshape(rows.shape[0], -1)[:, :columns]
    return logical.reshape(*packed.shape[:-1], columns).contiguous()


def quantize_dequantize(weight: torch.Tensor, mode: str, group_size: int = 128) -> torch.Tensor:
    """Round-trip a matrix through the exact container quantizer."""
    if mode == "int8":
        q, s = quantize_int8_rows(weight)
        return dequantize_int8_rows(q, s)
    if mode == "int4g128":
        q, s = quantize_int4_grouped(weight, group_size)
        return dequantize_int4_grouped(q, s, columns=weight.shape[-1], group_size=group_size)
    if mode == "none":
        return weight.detach().float().clone()
    raise ValueError(f"unknown quantization mode: {mode}")


def is_quantizable(name: str, tensor: torch.Tensor) -> bool:
    """Matrices are quantized; norms, biases, recurrent constants and conv stay BF16."""
    return tensor.ndim >= 2 and ".conv1d." not in name


def unfuse_experts(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert HF fused expert tensors to streamable per-expert 2D tensors."""
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name.endswith(".mlp.experts.gate_up_proj"):
            if tensor.ndim != 3 or tensor.shape[1] % 2:
                raise ValueError(f"unexpected fused expert shape for {name}: {tensor.shape}")
            mid = tensor.shape[1] // 2
            stem = name[: -len("gate_up_proj")]
            for expert in range(tensor.shape[0]):
                out[f"{stem}{expert}.gate_proj.weight"] = tensor[expert, :mid]
                out[f"{stem}{expert}.up_proj.weight"] = tensor[expert, mid:]
        elif name.endswith(".mlp.experts.down_proj"):
            if tensor.ndim != 3:
                raise ValueError(f"unexpected fused expert shape for {name}: {tensor.shape}")
            stem = name[: -len("down_proj")]
            for expert in range(tensor.shape[0]):
                out[f"{stem}{expert}.down_proj.weight"] = tensor[expert]
        else:
            out[name] = tensor
    return out


def containerize_state(
    state: Mapping[str, torch.Tensor], mode: str = "none", group_size: int = 128
) -> dict[str, torch.Tensor]:
    """Build safetensors payload using the colibri-compatible ``NAME``/``NAME.qs`` format."""
    if mode not in {"none", "int8", "int4g128"}:
        raise ValueError(f"unsupported mode: {mode}")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in sorted(unfuse_experts(state).items()):
        value = tensor.detach().cpu().contiguous()
        if mode == "none" or not is_quantizable(name, value):
            result[name] = value.to(torch.bfloat16)
        elif mode == "int8":
            result[name], result[f"{name}.qs"] = quantize_int8_rows(value)
        else:
            result[name], result[f"{name}.qs"] = quantize_int4_grouped(value, group_size)
    return result


def is_text_tensor(name: str, include_mtp: bool = False) -> bool:
    """Return whether ``name`` belongs in the text-only runtime container."""
    if name == "lm_head.weight":
        return True
    if name.startswith("mtp."):
        return include_mtp
    return name.startswith("model.language_model.") or name.startswith("model.layers.") or name.startswith(
        ("model.embed_tokens.", "model.norm.")
    )


def _bits_mode(bits: int) -> str:
    return "int8" if bits == 8 else "int4g128"


def precision_for_name(
    name: str,
    ndim: int,
    *,
    xbits: str = "int4g128",
    io_bits: int = 8,
    shared_bits: int = 8,
) -> str:
    """Return the Phase-4 storage mode for one text tensor.

    ``f32`` is deliberately used for every norm, router, convolution and
    recurrent constant. MTP matrices stay int8 even when routed main experts
    use int4-g128.
    """
    if ndim < 2:
        return "f32"
    if name.startswith("mtp."):
        return "int8"
    if ".mlp.experts." in name:
        return xbits
    if ".mlp.shared_expert." in name:
        return _bits_mode(shared_bits)
    if name.endswith(("embed_tokens.weight", "lm_head.weight")):
        return _bits_mode(io_bits)
    if ".self_attn." in name:
        if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
            return xbits
        if name.endswith("o_proj.weight"):
            return "int8"
        return "f32"
    if ".linear_attn." in name:
        if name.endswith(("in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight")):
            return xbits
        return "f32"
    # Router and shared-expert scalar gate are intentionally exact.
    return "f32"


def precision_for_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    xbits: str = "int4g128",
    io_bits: int = 8,
    shared_bits: int = 8,
) -> str:
    return precision_for_name(
        name, tensor.ndim, xbits=xbits, io_bits=io_bits, shared_bits=shared_bits
    )


def estimate_tensor_bytes(
    name: str,
    shape: list[int] | tuple[int, ...],
    *,
    xbits: str = "int4g128",
    io_bits: int = 8,
    shared_bits: int = 8,
    group_size: int = DEFAULT_GROUP_SIZE,
    include_mtp: bool = False,
) -> int:
    """Predict payload bytes from a safetensors header without loading data."""
    if not is_text_tensor(name, include_mtp):
        return 0
    mode = precision_for_name(name, len(shape), xbits=xbits, io_bits=io_bits, shared_bits=shared_bits)
    count = math.prod(shape)
    if mode == "f32":
        return count * 4
    qtype_count = 1
    if name.endswith(".mlp.experts.gate_up_proj"):
        qtype_count = shape[0] * 2
    elif name.endswith(".mlp.experts.down_proj") and len(shape) == 3:
        qtype_count = shape[0]
    rows, columns = math.prod(shape[:-1]), shape[-1]
    if mode == "int8":
        return count + rows * 4 + qtype_count
    groups = math.ceil(columns / group_size)
    return rows * groups * (group_size // 2 + 4) + qtype_count


def containerize_state_mixed(
    state: Mapping[str, torch.Tensor],
    *,
    xbits: str = "int4g128",
    io_bits: int = 8,
    shared_bits: int = 8,
    group_size: int = DEFAULT_GROUP_SIZE,
    include_mtp: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    """Convert a source shard and return payload plus logical inventory."""
    if xbits not in {"int8", "int4g128"}:
        raise ValueError(f"unsupported xbits: {xbits}")
    selected = {name: value for name, value in state.items() if is_text_tensor(name, include_mtp)}
    result: dict[str, torch.Tensor] = {}
    inventory: dict[str, dict[str, Any]] = {}
    for name, tensor in sorted(unfuse_experts(selected).items()):
        value = tensor.detach().cpu().contiguous()
        mode = precision_for_tensor(name, value, xbits=xbits, io_bits=io_bits, shared_bits=shared_bits)
        if mode == "f32":
            result[name] = value.float().contiguous()
        elif mode == "int8":
            result[name], result[f"{name}.qs"] = quantize_int8_rows(value)
            result[f"{name}.qtype"] = torch.tensor([8], dtype=torch.uint8)
        else:
            result[name], result[f"{name}.qs"] = quantize_int4_grouped(value, group_size)
            result[f"{name}.qtype"] = torch.tensor([4], dtype=torch.uint8)
        inventory[name] = {"shape": list(value.shape), "mode": mode}
    return result, inventory


def save_container(
    state: Mapping[str, torch.Tensor], outdir: Path, mode: str = "none", group_size: int = 128
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / "model.safetensors"
    save_file(containerize_state(state, mode, group_size), target)
    return target


def load_safetensors_dir(indir: Path) -> dict[str, torch.Tensor]:
    paths = sorted(indir.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no safetensors files in {indir}")
    state: dict[str, torch.Tensor] = {}
    for path in paths:
        shard = load_file(path, device="cpu")
        overlap = state.keys() & shard.keys()
        if overlap:
            raise ValueError(f"duplicate tensors across shards: {sorted(overlap)[:3]}")
        state.update(shard)
    return state


def copy_metadata(indir: Path, outdir: Path) -> None:
    for name in METADATA_FILES:
        src = indir / name
        if src.exists():
            shutil.copy2(src, outdir / name)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_save_file(tensors: Mapping[str, torch.Tensor], target: Path) -> None:
    """Write one safetensors shard without ever exposing a partial target."""
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        save_file(dict(tensors), temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _source_index(indir: Path) -> tuple[list[str], dict[str, Any], str]:
    index_path = indir / INDEX_FILE
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise ValueError(f"empty weight_map in {index_path}")
        return shards, index, sha256_file(index_path)
    shards = sorted(path.name for path in indir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors files in {indir}")
    synthetic = {"metadata": {}, "weight_map": {}}
    return shards, synthetic, hashlib.sha256("\n".join(shards).encode()).hexdigest()


def _output_bytes(payload: Mapping[str, torch.Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in payload.values())


def _check_free_space(path: Path, minimum_gb: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    required = int(minimum_gb * (1024**3))
    if free < required:
        raise OSError(f"only {free / 1024**3:.1f} GiB free at {path}; require {minimum_gb:.1f} GiB")


def expected_loader_names(config: Mapping[str, Any]) -> set[str]:
    """Build the complete text tensor inventory expected by ``qwen.c``."""
    text = config.get("text_config", config)
    layers = int(text.get("num_hidden_layers", 0))
    experts = int(text.get("num_experts", 0))
    if layers <= 0 or experts <= 0:
        return set()
    layer_types = text.get("layer_types")
    interval = int(text.get("full_attention_interval", 4))
    prefix = "model.language_model."
    names = {"lm_head.weight", f"{prefix}embed_tokens.weight", f"{prefix}norm.weight"}
    common_mlp = (
        "mlp.gate.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    )
    linear = (
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.conv1d.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
    )
    full = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    )
    for layer in range(layers):
        stem = f"{prefix}layers.{layer}."
        names.update({f"{stem}input_layernorm.weight", f"{stem}post_attention_layernorm.weight"})
        is_full = (
            layer_types[layer] == "full_attention"
            if isinstance(layer_types, list) and len(layer_types) == layers
            else (layer + 1) % interval == 0
        )
        names.update(stem + suffix for suffix in (full if is_full else linear))
        names.update(stem + suffix for suffix in common_mlp)
        for expert in range(experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                names.add(f"{stem}mlp.experts.{expert}.{projection}.weight")
    return names


def validate_loader_inventory(outdir: Path, inventory: Mapping[str, Any]) -> dict[str, int]:
    config_path = outdir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("converted container is missing config.json")
    expected = expected_loader_names(json.loads(config_path.read_text()))
    if not expected:  # synthetic/unit-test config
        return {"expected": 0, "present": len(inventory)}
    def canonical(name: str) -> str:
        if name.startswith("model.language_model."):
            return name[len("model.language_model.") :]
        if name.startswith("model."):
            return name[len("model.") :]
        return name

    expected_canonical = {canonical(name) for name in expected}
    present_canonical = {canonical(name) for name in inventory}
    missing = sorted(expected_canonical - present_canonical)
    if missing:
        raise ValueError(f"converted container is missing {len(missing)} loader tensors: {missing[:5]}")
    return {"expected": len(expected_canonical), "present": len(present_canonical)}


def convert_streaming(
    *,
    shards: list[str],
    get_shard: Callable[[str], Path],
    outdir: Path,
    source_id: str,
    source_fingerprint: str,
    xbits: str = "int4g128",
    io_bits: int = 8,
    shared_bits: int = 8,
    group_size: int = DEFAULT_GROUP_SIZE,
    include_mtp: bool = False,
    delete_source: bool = False,
    resume: bool = True,
    max_shards: int | None = None,
) -> dict[str, Any]:
    """Convert ``shards`` one at a time and atomically maintain resume state."""
    outdir.mkdir(parents=True, exist_ok=True)
    state_path = outdir / STATE_FILE
    signature = {
        "source": source_id,
        "source_fingerprint": source_fingerprint,
        "xbits": xbits,
        "io_bits": io_bits,
        "shared_bits": shared_bits,
        "group_size": group_size,
        "include_mtp": include_mtp,
    }
    if state_path.exists() and resume:
        state = json.loads(state_path.read_text())
        if state.get("signature") != signature:
            raise ValueError("existing conversion state uses different source or precision options")
    else:
        if any(outdir.glob("*.safetensors")):
            raise FileExistsError(f"{outdir} contains safetensors but has no reusable conversion state")
        state = {"version": 1, "signature": signature, "completed": {}, "inventory": {}}
        _atomic_json(state_path, state)

    converted_now = 0
    for ordinal, shard_name in enumerate(shards, 1):
        completed = state["completed"].get(shard_name)
        if completed:
            target = outdir / completed["output"]
            if target.exists() and target.stat().st_size == completed["file_size"] and sha256_file(target) == completed["sha256"]:
                continue
            raise ValueError(f"completed output shard failed integrity check: {target}")
        if max_shards is not None and converted_now >= max_shards:
            break
        source = get_shard(shard_name)
        if source.resolve().parent == outdir.resolve():
            raise ValueError("source and output shard directories must differ")
        print(f"[{ordinal}/{len(shards)}] loading {source.name}", flush=True)
        source_state = load_file(source, device="cpu")
        payload, inventory = containerize_state_mixed(
            source_state,
            xbits=xbits,
            io_bits=io_bits,
            shared_bits=shared_bits,
            group_size=group_size,
            include_mtp=include_mtp,
        )
        del source_state
        if payload:
            target = outdir / shard_name
            atomic_save_file(payload, target)
            digest = sha256_file(target)
            record = {
                "output": target.name,
                "file_size": target.stat().st_size,
                "data_bytes": _output_bytes(payload),
                "sha256": digest,
                "tensor_count": len(payload),
            }
        else:
            record = {"output": None, "file_size": 0, "data_bytes": 0, "sha256": None, "tensor_count": 0}
        state["inventory"].update(inventory)
        state["completed"][shard_name] = record
        _atomic_json(state_path, state)
        del payload, inventory
        gc.collect()
        if delete_source:
            source.unlink(missing_ok=True)
        converted_now += 1

    done = len(state["completed"]) == len(shards)
    if done:
        validation = validate_loader_inventory(outdir, state["inventory"])
        weight_map: dict[str, str] = {}
        for source_name in shards:
            record = state["completed"][source_name]
            output = record["output"]
            if output is None:
                continue
            with safe_open(outdir / output, framework="pt", device="cpu") as handle:
                weight_map.update({name: output for name in handle.keys()})
        total_size = sum(record["data_bytes"] for record in state["completed"].values())
        _atomic_json(outdir / INDEX_FILE, {"metadata": {"total_size": total_size}, "weight_map": weight_map})
        manifest = {
            **signature,
            "complete": True,
            "source_shards": len(shards),
            "tensor_count": len(weight_map),
            "logical_tensor_count": len(state["inventory"]),
            "data_bytes": total_size,
            "loader_inventory": validation,
        }
        _atomic_json(outdir / "quantization.json", manifest)
        _atomic_json(outdir / "tensor_inventory.json", state["inventory"])
    return {"complete": done, "converted_now": converted_now, **state}


def convert_local_directory(
    indir: Path,
    outdir: Path,
    *,
    min_free_gb: float = 35.0,
    delete_source: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    _check_free_space(outdir, min_free_gb)
    shards, _index, fingerprint = _source_index(indir)
    copy_metadata(indir, outdir)
    return convert_streaming(
        shards=shards,
        get_shard=lambda name: indir / name,
        outdir=outdir,
        source_id=str(indir.resolve()),
        source_fingerprint=fingerprint,
        delete_source=delete_source,
        **kwargs,
    )


def convert_hub_repo(
    repo: str,
    outdir: Path,
    *,
    revision: str = "main",
    staging_dir: Path | None = None,
    min_free_gb: float = 35.0,
    keep_source: bool = False,
    dry_run: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Resolve one immutable Hub revision and stream its shards locally."""
    try:
        from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is in the project venv
        raise RuntimeError("--repo requires huggingface_hub") from exc

    _check_free_space(outdir, min_free_gb)
    api = HfApi()
    resolved = api.model_info(repo, revision=revision).sha
    if not resolved:
        raise RuntimeError(f"could not resolve immutable revision for {repo}@{revision}")
    staging = (staging_dir or outdir.parent / f".{outdir.name}.source").resolve()
    staging.mkdir(parents=True, exist_ok=True)

    def download(filename: str) -> Path:
        return Path(hf_hub_download(repo, filename, revision=resolved, local_dir=staging))

    index_path = download(INDEX_FILE)
    index = json.loads(index_path.read_text())
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise ValueError(f"Hub index for {repo} has no shards")
    for name in METADATA_FILES:
        try:
            download(name)
        except Exception as exc:
            # Only config and tokenizer.json are hard requirements for this runtime.
            if name in {"config.json", "tokenizer.json"}:
                raise
            print(f"metadata {name}: {type(exc).__name__} (optional, skipped)", flush=True)
    copy_metadata(staging, outdir)
    if dry_run:
        total = int(index.get("metadata", {}).get("total_size", 0))
        repo_metadata = get_safetensors_metadata(repo, revision=resolved)
        estimate = 0
        kept = 0
        for file_metadata in repo_metadata.files_metadata.values():
            for tensor_name, tensor_info in file_metadata.tensors.items():
                size = estimate_tensor_bytes(
                    tensor_name,
                    tensor_info.shape,
                    xbits=kwargs.get("xbits", "int4g128"),
                    io_bits=kwargs.get("io_bits", 8),
                    shared_bits=kwargs.get("shared_bits", 8),
                    group_size=kwargs.get("group_size", DEFAULT_GROUP_SIZE),
                    include_mtp=kwargs.get("include_mtp", False),
                )
                estimate += size
                kept += size > 0
        return {
            "complete": False,
            "dry_run": True,
            "repo": repo,
            "revision": resolved,
            "source_shards": len(shards),
            "source_bytes": total,
            "kept_source_tensors": kept,
            "estimated_output_data_bytes": estimate,
            "staging_dir": str(staging),
        }
    return convert_streaming(
        shards=shards,
        get_shard=download,
        outdir=outdir,
        source_id=f"hf://{repo}@{resolved}",
        source_fingerprint=sha256_file(index_path),
        delete_source=not keep_source,
        **kwargs,
    )


def selftest() -> None:
    samples = [
        torch.zeros(2, 129),
        torch.linspace(-3.0, 3.0, 3 * 257).reshape(3, 257),
        torch.tensor([[float("-inf"), 0.0, float("inf")]]),
    ]
    for sample in samples[:2]:
        for mode in ("int8", "int4g128"):
            got = quantize_dequantize(sample, mode)
            if got.shape != sample.shape or not torch.isfinite(got).all():
                raise AssertionError(f"{mode} round-trip failed for {sample.shape}")
    for fn in (quantize_int8_rows, quantize_int4_grouped):
        try:
            fn(samples[2])
        except Exception:
            pass
        else:
            raise AssertionError(f"{fn.__name__} must reject non-finite scale input")
    print("quantization self-test: OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="Hugging Face repo, e.g. Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--revision", default="main", help="Hub branch/tag/commit (resolved to a commit before download)")
    parser.add_argument("--indir", type=Path, help="local safetensors snapshot")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--staging-dir", type=Path, help="download staging directory for --repo")
    parser.add_argument("--xbits", choices=("int8", "int4g128"), default="int4g128")
    parser.add_argument("--io-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--shared-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=35.0)
    parser.add_argument("--max-shards", type=int, help="convert at most this many new shards (resume testing)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-source", action="store_true", help="retain Hub shards after conversion")
    parser.add_argument("--delete-source", action="store_true", help="delete local --indir shards after conversion")
    parser.add_argument("--dry-run", action="store_true", help="resolve metadata and preflight without weight downloads")
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        if args.indir is None and args.repo is None:
            return
    if bool(args.repo) == bool(args.indir):
        raise SystemExit("select exactly one of --repo or --indir")
    if args.outdir is None:
        raise SystemExit("--outdir is required")
    if args.keep_source and args.delete_source:
        raise SystemExit("--keep-source and --delete-source are mutually exclusive")
    common = {
        "xbits": args.xbits,
        "io_bits": args.io_bits,
        "shared_bits": args.shared_bits,
        "group_size": args.group_size,
        "include_mtp": args.mtp,
        "resume": not args.no_resume,
        "max_shards": args.max_shards,
    }
    if args.repo:
        result = convert_hub_repo(
            args.repo,
            args.outdir,
            revision=args.revision,
            staging_dir=args.staging_dir,
            min_free_gb=args.min_free_gb,
            keep_source=args.keep_source,
            dry_run=args.dry_run,
            **common,
        )
    else:
        if args.dry_run:
            shards, index, fingerprint = _source_index(args.indir)
            result = {
                "complete": False,
                "dry_run": True,
                "source_shards": len(shards),
                "source_bytes": int(index.get("metadata", {}).get("total_size", 0)),
                "source_fingerprint": fingerprint,
            }
        else:
            result = convert_local_directory(
                args.indir,
                args.outdir,
                min_free_gb=args.min_free_gb,
                delete_source=args.delete_source,
                **common,
            )
    print(json.dumps({key: result[key] for key in result if key not in {"inventory", "completed"}}, indent=2))


if __name__ == "__main__":
    main()
