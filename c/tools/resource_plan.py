#!/usr/bin/env python3
"""Plan Qwen3.5 tiered weight/state resources before conversion or launch."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


GIB = 1024**3


def _qbytes(rows: int, columns: int, mode: str, group_size: int = 128) -> int:
    if mode == "f32":
        return rows * columns * 4
    if mode == "int8":
        return rows * columns + rows * 4 + 1
    if mode == "int4g128":
        groups = math.ceil(columns / group_size)
        return rows * groups * (group_size // 2 + 4) + 1
    raise ValueError(f"unknown storage mode {mode}")


def _full_layer_count(config: Mapping[str, Any]) -> int:
    layers = int(config["num_hidden_layers"])
    types = config.get("layer_types")
    if isinstance(types, list) and len(types) == layers:
        return sum(value == "full_attention" for value in types)
    interval = int(config.get("full_attention_interval", 4))
    return layers // interval


def _is_glm53(config: Mapping[str, Any]) -> bool:
    """True for a GLM-5.3-Flash checkpoint.

    The marker the converter stamps is checked first; the architecture string
    and GLM's own expert field are accepted too, so a snapshot converted before
    the marker existed still plans correctly.
    """
    if str(config.get("shiftwing_model_family", "")).startswith("glm"):
        return True
    architectures = config.get("architectures")
    if isinstance(architectures, list) and any(
        isinstance(name, str) and name.startswith("Glm5Next") for name in architectures
    ):
        return True
    text = config.get("text_config", config)
    return "n_routed_experts" in text and "kv_lora_rank" in text


def plan_resources_glm53(
    config: Mapping[str, Any],
    *,
    slots: int,
    context: int,
    kv_bytes: int,
    cuda_expert_gib: float,
    ram_cache_gib: float,
    system_ram_gib: float,
    gpu_gib: float,
    disk_free_gib: float,
    largest_source_shard_gib: float,
    disk_reserve_gib: float,
    runtime_headroom_gib: float,
    snapshot_bytes: int = 0,
    routed_bytes: int = 0,
    resident_ram_bytes: int = 0,
    group_size: int = 64,
) -> dict[str, Any]:
    """Resource plan for GLM-5.3-Flash.

    The Qwen planner cannot be reused by renaming fields: GLM stores KV as an
    MLA latent of kv_lora_rank per token per full layer, where Qwen stores
    2 * kv_heads * head_dim. Using Qwen's formula here overstates the KV state
    by more than an order of magnitude and would fail a machine that fits.

    Container size is measured from the converted snapshot when one is given.
    Predicting it would mean re-deriving every tensor shape the converter
    writes, and the file sizes are already the ground truth the disk check is
    about. The routed-expert share is computed analytically -- it is exact and
    cheap -- so the resident part (which must fit in RAM) is total minus routed.
    """
    c = config.get("text_config", config)
    h = int(c["hidden_size"])
    layers = int(c["num_hidden_layers"])
    experts = int(c["n_routed_experts"])
    topk = int(c["num_experts_per_tok"])
    moe_inter = int(c["moe_intermediate_size"])
    dense_layers = int(c.get("first_k_dense_replace", 0))
    # The MTP layer carries its own full set of routed experts and streams them
    # like any other. Counting it as resident overstated the resident footprint
    # by ~14 GiB on GLM-5.3-Flash, which under-budgeted the expert cache below
    # topk and made every layer of every token split into two blocks.
    mtp_layers = 1 if int(c.get("num_nextn_predict_layers", 0)) > 0 else 0
    sparse_layers = layers - dense_layers + mtp_layers
    kv_lora = int(c["kv_lora_rank"])
    qk_rope = int(c.get("qk_rope_head_dim", 0))

    types = c.get("layer_types")
    if isinstance(types, list) and len(types) == layers:
        full_layers = sum(value != "linear_attention" for value in types)
    else:
        full_layers = layers // 4
    linear_layers = layers - full_layers

    linear = c.get("linear_attn_config") or {}
    kda_heads = int(linear.get("num_heads", c.get("num_attention_heads", 1)))
    kda_dim = int(linear.get("head_dim", 128))
    conv_k = int(linear.get("short_conv_kernel_size", 4))

    if min(h, layers, experts, topk, moe_inter, kv_lora, kda_heads, kda_dim,
           slots, context) <= 0:
        raise ValueError("configuration and runtime dimensions must be positive")
    if kv_bytes not in (2, 4):
        raise ValueError("kv_bytes must be 2 or 4")

    # One routed expert: gate and up are [moe_inter, h], down is [h, moe_inter].
    routed_one = (
        _qbytes(moe_inter, h, "int4g128", group_size) * 2
        + _qbytes(h, moe_inter, "int4g128", group_size)
    )
    # Measured beats computed: the caller reads the routed total off the
    # shard headers, which needs no assumption about which layers carry
    # experts or how the converter packed them.
    routed = (int(routed_bytes) if routed_bytes > 0
              else sparse_layers * experts * routed_one)

    output_bytes = int(snapshot_bytes) if snapshot_bytes > 0 else routed * 2
    # Whatever is not a routed expert stays resident: dense core, shared
    # experts, embeddings, norms, routers, the indexer.
    dense_shared = max(output_bytes - routed, 0)
    # What the container holds and what the process holds are different
    # numbers: unquantised weights shrink when they are loaded. The caller
    # measures the resident width when it can; the container size is the
    # conservative fallback.
    resident_ram = int(resident_ram_bytes) if resident_ram_bytes > 0 else dense_shared

    # KDA carries a per-head recurrent state and a short-convolution window.
    recurrent_per_slot = linear_layers * kda_heads * kda_dim * kda_dim * 4
    conv_per_slot = linear_layers * 3 * kda_heads * kda_dim * conv_k * 4
    last_hidden_per_slot = h * 4
    # MLA: one latent per token per full layer, plus any rope carried alongside.
    kv_per_token = full_layers * (kv_lora + qk_rope) * kv_bytes

    fixed_per_slot = recurrent_per_slot + conv_per_slot + last_hidden_per_slot
    fixed_state = slots * fixed_per_slot
    kv_state = slots * context * kv_per_token
    snapshot_per_slot_at_context = fixed_per_slot + context * kv_per_token
    sequential_switch_bytes_per_token = 2 * snapshot_per_slot_at_context
    runtime_state = fixed_state + kv_state

    ram_cache_bytes = int(ram_cache_gib * GIB)
    ram_required = (
        resident_ram + ram_cache_bytes + runtime_state + int(runtime_headroom_gib * GIB)
    )
    # The CPU engine keeps no dense weights on the GPU; a CUDA build would add
    # them, so the VRAM line stays the conservative one.
    vram_required = int(cuda_expert_gib * GIB)
    disk_required = (
        output_bytes + int(largest_source_shard_gib * GIB) + int(disk_reserve_gib * GIB)
    )
    expert_cache_slots = int(cuda_expert_gib * GIB) // routed_one if routed_one else 0
    ram_expert_slots_per_layer = (
        ram_cache_bytes // (sparse_layers * routed_one) if routed_one else 0
    )

    # A per-layer cache smaller than topk cannot hold one token's experts, so
    # the engine splits every layer into blocks and evicts entries it is about
    # to need again. Measured at cap 6 against topk 8: the parallel read never
    # went wider than six and the tail two evicted part of the block just read.
    experts_per_layer = (ram_cache_bytes // (sparse_layers * routed_one)
                         if routed_one and sparse_layers else 0)
    checks = {
        "disk": disk_free_gib <= 0 or disk_required <= disk_free_gib * GIB,
        "ram": ram_required <= system_ram_gib * GIB,
        "vram": gpu_gib <= 0 or vram_required <= gpu_gib * GIB,
        "expert_cache_holds_topk": experts_per_layer >= topk,
    }
    return {
        "family": "glm-5.3",
        "architecture": {
            "hidden": h,
            "layers": layers,
            "full_layers": full_layers,
            "linear_layers": linear_layers,
            "experts": experts,
            "topk": topk,
            "expert_layers": sparse_layers,
        },
        "container_bytes": {
            "routed_experts": routed,
            "shared_experts": 0,
            "dense_core": dense_shared,
            "resident_ram": resident_ram,
            "mtp": 0,
            "total": output_bytes,
            "measured": bool(snapshot_bytes > 0),
        },
        "state_bytes": {
            "recurrent_per_slot": recurrent_per_slot,
            "conv_per_slot": conv_per_slot,
            "last_hidden_per_slot": last_hidden_per_slot,
            "kv_per_token_per_slot": kv_per_token,
            "snapshot_per_slot_at_context": snapshot_per_slot_at_context,
            "sequential_switch_bytes_per_token_per_active_slot": (
                sequential_switch_bytes_per_token
            ),
            "fixed_all_slots": fixed_state,
            "kv_all_slots": kv_state,
            "total": runtime_state,
        },
        "tier_plan": {
            "dense_shared_bytes": dense_shared,
            "routed_expert_bytes": routed_one,
            "expert_cache_slots": expert_cache_slots,
            "ram_expert_slots_per_layer": ram_expert_slots_per_layer,
            "experts_per_layer": experts_per_layer,
            "topk": topk,
        },
        "requirements_bytes": {
            "ram": ram_required,
            "vram": vram_required,
            "disk": disk_required,
        },
        "checks": checks,
    }


def plan_resources(
    config: Mapping[str, Any],
    *,
    slots: int = 4,
    context: int = 4096,
    kv_bytes: int = 4,
    cuda_expert_gib: float = 7.0,
    ram_cache_gib: float = 18.0,
    system_ram_gib: float = 32.0,
    gpu_gib: float = 16.0,
    disk_free_gib: float = 0.0,
    largest_source_shard_gib: float = 4.4,
    disk_reserve_gib: float = 20.0,
    runtime_headroom_gib: float = 2.0,
    include_mtp: bool = True,
    snapshot_bytes: int = 0,
    routed_bytes: int = 0,
    resident_ram_bytes: int = 0,
) -> dict[str, Any]:
    """Return conservative container and runtime resource estimates."""
    if _is_glm53(config):
        return plan_resources_glm53(
            config,
            slots=slots,
            context=context,
            kv_bytes=kv_bytes,
            cuda_expert_gib=cuda_expert_gib,
            ram_cache_gib=ram_cache_gib,
            system_ram_gib=system_ram_gib,
            gpu_gib=gpu_gib,
            disk_free_gib=disk_free_gib,
            largest_source_shard_gib=largest_source_shard_gib,
            disk_reserve_gib=disk_reserve_gib,
            runtime_headroom_gib=runtime_headroom_gib,
            snapshot_bytes=snapshot_bytes,
            routed_bytes=routed_bytes,
            resident_ram_bytes=resident_ram_bytes,
        )
    c = config.get("text_config", config)
    h = int(c["hidden_size"])
    layers = int(c["num_hidden_layers"])
    experts = int(c["num_experts"])
    expert_i = int(c["moe_intermediate_size"])
    shared_i = int(c["shared_expert_intermediate_size"])
    vocab = int(c["vocab_size"])
    topk = int(c["num_experts_per_tok"])
    heads = int(c["num_attention_heads"])
    kv_heads = int(c["num_key_value_heads"])
    head_dim = int(c["head_dim"])
    full_layers = _full_layer_count(c)
    linear_layers = layers - full_layers
    key_heads = int(c["linear_num_key_heads"])
    value_heads = int(c["linear_num_value_heads"])
    key_dim = int(c["linear_key_head_dim"])
    value_dim = int(c["linear_value_head_dim"])
    conv_kernel = int(
        c.get("linear_conv_kernel_dim", c.get("linear_conv_kernel_size", 4))
    )
    if min(
        h,
        layers,
        experts,
        expert_i,
        shared_i,
        vocab,
        topk,
        heads,
        kv_heads,
        head_dim,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        slots,
        context,
    ) <= 0:
        raise ValueError("configuration and runtime dimensions must be positive")
    if kv_bytes not in (2, 4):
        raise ValueError("kv_bytes must be 2 or 4")

    routed_one = (
        _qbytes(expert_i, h, "int4g128") * 2
        + _qbytes(h, expert_i, "int4g128")
    )
    routed = layers * experts * routed_one
    shared_one = (
        _qbytes(shared_i, h, "int8") * 2
        + _qbytes(h, shared_i, "int8")
        + _qbytes(1, h, "f32")
    )
    shared = layers * shared_one
    routers = layers * _qbytes(experts, h, "f32")
    norms = (layers * 2 + 1) * h * 4

    key_total = key_heads * key_dim
    value_total = value_heads * value_dim
    channels = 2 * key_total + value_total
    gdn_one = (
        _qbytes(channels, h, "int4g128")
        + _qbytes(value_total, h, "int4g128")
        + _qbytes(value_heads, h, "f32") * 2
        + channels * conv_kernel * 4
        + value_heads * 2 * 4
        + value_dim * 4
        + _qbytes(h, value_total, "int4g128")
    )
    qrows = heads * head_dim * 2
    kvrows = kv_heads * head_dim
    attention_one = (
        _qbytes(qrows, h, "int4g128")
        + _qbytes(kvrows, h, "int4g128") * 2
        + _qbytes(h, heads * head_dim, "int8")
        + (head_dim * 2 * 4)
    )
    dense_core = (
        linear_layers * gdn_one
        + full_layers * attention_one
        + _qbytes(vocab, h, "int8") * 2
        + norms
        + routers
    )
    mtp = 0
    if include_mtp:
        mtp = (
            _qbytes(h, 2 * h, "int8")
            + attention_one
            + shared_one
            + experts * (
                _qbytes(expert_i, h, "int8") * 2
                + _qbytes(h, expert_i, "int8")
            )
            + _qbytes(experts, h, "f32")
            + (5 * h * 4)
        )
    output_bytes = routed + shared + dense_core + mtp

    recurrent_per_slot = (
        linear_layers * value_heads * key_dim * value_dim * 4
    )
    conv_per_slot = linear_layers * channels * conv_kernel * 4
    kv_per_token = full_layers * 2 * kv_heads * head_dim * kv_bytes
    if include_mtp:
        kv_per_token += 2 * kv_heads * head_dim * kv_bytes
    last_hidden_per_slot = h * 4
    fixed_per_slot = recurrent_per_slot + conv_per_slot + last_hidden_per_slot
    fixed_state = slots * fixed_per_slot
    kv_state = slots * context * kv_per_token
    snapshot_per_slot_at_context = fixed_per_slot + context * kv_per_token
    sequential_switch_bytes_per_token = 2 * snapshot_per_slot_at_context

    dense_shared = dense_core + shared + mtp
    runtime_state = fixed_state + kv_state
    vram_required = (
        dense_shared
        + int(cuda_expert_gib * GIB)
        + runtime_state
        + int(runtime_headroom_gib * GIB)
    )
    ram_cache_bytes = int(ram_cache_gib * GIB)
    ram_required = (
        dense_shared
        + ram_cache_bytes
        + runtime_state
        + int(runtime_headroom_gib * GIB)
    )
    disk_required = (
        output_bytes
        + int(largest_source_shard_gib * GIB)
        + int(disk_reserve_gib * GIB)
    )
    source_reads_per_token = layers * topk * routed_one
    expert_cache_slots = int(cuda_expert_gib * GIB) // routed_one
    ram_expert_slots_per_layer = ram_cache_bytes // (layers * routed_one)
    nominal_combined_slots_per_layer = (
        ram_expert_slots_per_layer + expert_cache_slots / layers
    )

    checks = {
        "disk": disk_free_gib <= 0 or disk_required <= disk_free_gib * GIB,
        "ram": ram_required <= system_ram_gib * GIB,
        "vram": vram_required <= gpu_gib * GIB,
    }
    return {
        "architecture": {
            "hidden": h,
            "layers": layers,
            "full_layers": full_layers,
            "linear_layers": linear_layers,
            "experts": experts,
            "topk": topk,
        },
        "container_bytes": {
            "routed_experts": routed,
            "shared_experts": shared,
            "dense_core": dense_core,
            "mtp": mtp,
            "total": output_bytes,
        },
        "state_bytes": {
            "recurrent_per_slot": recurrent_per_slot,
            "conv_per_slot": conv_per_slot,
            "last_hidden_per_slot": last_hidden_per_slot,
            "kv_per_token_per_slot": kv_per_token,
            "snapshot_per_slot_at_context": snapshot_per_slot_at_context,
            "sequential_switch_bytes_per_token_per_active_slot": (
                sequential_switch_bytes_per_token
            ),
            "fixed_all_slots": fixed_state,
            "kv_all_slots": kv_state,
            "total": runtime_state,
        },
        "tier_plan": {
            "dense_shared_bytes": dense_shared,
            "routed_expert_bytes_each": routed_one,
            "cuda_expert_cache_slots": expert_cache_slots,
            "cuda_expert_cache_slots_per_layer_average": expert_cache_slots / layers,
            "ram_expert_cache_slots_per_layer": ram_expert_slots_per_layer,
            "nominal_combined_expert_slots_per_layer": nominal_combined_slots_per_layer,
            "nominal_combined_expert_coverage": min(
                1.0, nominal_combined_slots_per_layer / experts
            ),
            "cold_source_bytes_per_token": source_reads_per_token,
            "vram_required": vram_required,
            "ram_required": ram_required,
            "disk_required": disk_required,
        },
        "available_bytes": {
            "gpu": int(gpu_gib * GIB),
            "ram": int(system_ram_gib * GIB),
            "disk": int(disk_free_gib * GIB),
        },
        "checks": checks,
        "safe": all(checks.values()),
    }


def _gib(value: int) -> str:
    return f"{value / GIB:.2f} GiB"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--kv-bytes", type=int, choices=(2, 4), default=4)
    parser.add_argument("--cuda-expert-gib", type=float, default=7.0)
    parser.add_argument("--ram-cache-gib", type=float, default=18.0)
    parser.add_argument("--system-ram-gib", type=float, default=32.0)
    parser.add_argument("--gpu-gib", type=float, default=16.0)
    parser.add_argument("--disk-path", type=Path, default=Path("."))
    parser.add_argument("--largest-source-shard-gib", type=float, default=4.4)
    parser.add_argument("--disk-reserve-gib", type=float, default=20.0)
    parser.add_argument("--runtime-headroom-gib", type=float, default=2.0)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = json.loads(args.config.read_text())
    disk_free = shutil.disk_usage(args.disk_path).free / GIB
    plan = plan_resources(
        config,
        slots=args.slots,
        context=args.context,
        kv_bytes=args.kv_bytes,
        cuda_expert_gib=args.cuda_expert_gib,
        ram_cache_gib=args.ram_cache_gib,
        system_ram_gib=args.system_ram_gib,
        gpu_gib=args.gpu_gib,
        disk_free_gib=disk_free,
        largest_source_shard_gib=args.largest_source_shard_gib,
        disk_reserve_gib=args.disk_reserve_gib,
        runtime_headroom_gib=args.runtime_headroom_gib,
        include_mtp=not args.no_mtp,
    )
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(
            f"container={_gib(plan['container_bytes']['total'])} "
            f"dense+shared={_gib(plan['tier_plan']['dense_shared_bytes'])} "
            f"experts={_gib(plan['container_bytes']['routed_experts'])}"
        )
        print(
            f"state={_gib(plan['state_bytes']['total'])} "
            f"kv/token/slot={plan['state_bytes']['kv_per_token_per_slot']} B "
            f"sequential-switch/token/slot="
            f"{_gib(plan['state_bytes']['sequential_switch_bytes_per_token_per_active_slot'])} "
            f"cuda-cache-experts={plan['tier_plan']['cuda_expert_cache_slots']}"
        )
        print(
            f"required: VRAM={_gib(plan['tier_plan']['vram_required'])} "
            f"RAM={_gib(plan['tier_plan']['ram_required'])} "
            f"disk={_gib(plan['tier_plan']['disk_required'])}"
        )
        print(
            "checks: "
            + " ".join(
                f"{name}={'PASS' if passed else 'FAIL'}"
                for name, passed in plan["checks"].items()
            )
        )
    return 0 if plan["safe"] or args.no_fail else 2


if __name__ == "__main__":
    sys.exit(main())
