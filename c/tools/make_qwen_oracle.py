#!/usr/bin/env python3
"""Generate deterministic tiny Qwen3.5-MoE snapshots and reference outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import hf_hub_download
from packaging.version import Version
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

from convert_qwen import is_quantizable, quantize_dequantize, save_container


MIN_TRANSFORMERS = Version("5.14")
DEFAULT_SEED = 20260720
ROOT = Path(__file__).resolve().parents[1]


def require_runtime() -> None:
    current = Version(transformers.__version__.split("+")[0])
    if current < MIN_TRANSFORMERS:
        raise SystemExit(
            f"transformers >= {MIN_TRANSFORMERS} is required; found {transformers.__version__}"
        )


def tiny_config() -> Qwen3_5MoeTextConfig:
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=8,
        num_experts_per_tok=2,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
        rope_parameters={
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=False,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    # The text-only HF implementation ignores MTP, but the real config and C
    # loader need this architecture fact in the generated config.
    cfg.mtp_num_hidden_layers = 1
    cfg.mtp_use_dedicated_embeddings = False
    cfg.architectures = ["Qwen3_5MoeForCausalLM"]
    return cfg


def rounded_model(mode: str, seed: int) -> Qwen3_5MoeForCausalLM:
    torch.manual_seed(seed)
    model = Qwen3_5MoeForCausalLM(tiny_config()).eval()
    state = model.state_dict()
    rounded: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        base = value.detach().to(torch.bfloat16).float()
        rounded[name] = quantize_dequantize(base, mode) if mode != "none" and is_quantizable(name, base) else base
    model.load_state_dict(rounded, strict=True)
    return model


def synthetic_mtp(
    model: Qwen3_5MoeForCausalLM, mode: str, seed: int
) -> dict[str, torch.Tensor]:
    """Create a deterministic executable Qwen3.5 MTP block.

    Transformers deliberately ignores the checkpoint's top-level ``mtp.*``
    tensors.  The decoder block is copied from the tiny model's full-attention
    layer, while the concat projection is independently initialized.  Every
    MTP matrix is rounded through int8 for quantized snapshots, including the
    otherwise int4-g128 tiny mode, matching the real converter precision map.
    """
    cfg = model.config
    hidden = cfg.hidden_size
    state = model.state_dict()
    # Keep the synthetic tiny fixture away from accidental near-ties: the C
    # int8 kernels and PyTorch's dequantized float path may differ by a few
    # ulps after a full decoder block, while real MTP correctness is guarded
    # by target verification.  This fixed offset gives every fixture row a
    # useful argmax margin without changing determinism.
    gen = torch.Generator().manual_seed(seed + 27)

    def rounded(name: str, value: torch.Tensor) -> torch.Tensor:
        base = value.detach().to(torch.bfloat16).float()
        if mode != "none" and is_quantizable(name, base):
            return quantize_dequantize(base, "int8")
        return base

    result: dict[str, torch.Tensor] = {
        "mtp.fc.weight": torch.randn(hidden, hidden * 2, generator=gen) * cfg.initializer_range,
        "mtp.pre_fc_norm_embedding.weight": torch.zeros(hidden),
        "mtp.pre_fc_norm_hidden.weight": torch.zeros(hidden),
        "mtp.norm.weight": torch.zeros(hidden),
    }
    source_prefix = "model.layers.3."
    for name, value in state.items():
        if not name.startswith(source_prefix):
            continue
        suffix = name[len(source_prefix) :]
        if suffix.startswith("self_attn.") or suffix in {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert.down_proj.weight",
            "mlp.shared_expert_gate.weight",
        }:
            target = f"mtp.layers.0.{suffix}"
            result[target] = rounded(target, value)
        elif suffix == "mlp.experts.gate_up_proj":
            mid = value.shape[1] // 2
            for expert in range(value.shape[0]):
                gate = f"mtp.layers.0.mlp.experts.{expert}.gate_proj.weight"
                up = f"mtp.layers.0.mlp.experts.{expert}.up_proj.weight"
                result[gate] = rounded(gate, value[expert, :mid])
                result[up] = rounded(up, value[expert, mid:])
        elif suffix == "mlp.experts.down_proj":
            for expert in range(value.shape[0]):
                target = f"mtp.layers.0.mlp.experts.{expert}.down_proj.weight"
                result[target] = rounded(target, value[expert])
    result["mtp.fc.weight"] = rounded("mtp.fc.weight", result["mtp.fc.weight"])
    return result


def mtp_reference(
    model: Qwen3_5MoeForCausalLM,
    mtp_state: Mapping[str, torch.Tensor],
    token_ids: list[int],
) -> dict[str, Any]:
    """Evaluate the vLLM/Qwen MTP equation independently of Transformers CausalLM.

    For pair ``i``, the MTP block receives the main model's post-final-norm
    hidden state at token ``i`` and the embedding of token ``i+1``.  Its logit
    row predicts token ``i+2``.  A standalone one-layer HF text model supplies
    the full-attention/MoE decoder math after the concat projection.
    """
    if len(token_ids) < 3:
        raise ValueError("MTP reference needs at least three tokens")
    cfg_dict = model.config.to_dict()
    cfg_dict["num_hidden_layers"] = 1
    cfg_dict["layer_types"] = ["full_attention"]
    cfg = Qwen3_5MoeTextConfig(**cfg_dict)
    block = Qwen3_5MoeTextModel(cfg).eval()
    block_state = block.state_dict()
    block_state["embed_tokens.weight"] = model.model.embed_tokens.weight.detach().clone()
    block_state["norm.weight"] = mtp_state["mtp.norm.weight"].detach().clone()
    direct_suffixes = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    )
    for suffix in direct_suffixes:
        block_state[f"layers.0.{suffix}"] = mtp_state[f"mtp.layers.0.{suffix}"].detach().clone()
    gate_up = []
    down = []
    for expert in range(cfg.num_experts):
        prefix = f"mtp.layers.0.mlp.experts.{expert}."
        gate_up.append(
            torch.cat(
                (mtp_state[prefix + "gate_proj.weight"], mtp_state[prefix + "up_proj.weight"]),
                dim=0,
            )
        )
        down.append(mtp_state[prefix + "down_proj.weight"])
    block_state["layers.0.mlp.experts.gate_up_proj"] = torch.stack(gate_up)
    block_state["layers.0.mlp.experts.down_proj"] = torch.stack(down)
    block.load_state_dict(block_state, strict=True)

    ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        main_hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
        embedding = model.model.embed_tokens(ids[:, 1:])
        hidden = main_hidden[:, :-1]

        def rms_zero(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + cfg.rms_norm_eps).to(x.dtype) * (
                1.0 + weight
            )

        embedding = rms_zero(embedding, mtp_state["mtp.pre_fc_norm_embedding.weight"])
        hidden = rms_zero(hidden, mtp_state["mtp.pre_fc_norm_hidden.weight"])
        fused = F.linear(torch.cat((embedding, hidden), dim=-1), mtp_state["mtp.fc.weight"])
        mtp_hidden = block(inputs_embeds=fused, use_cache=False).last_hidden_state
        logits = F.linear(mtp_hidden, model.lm_head.weight)
    return {
        "mtp_input_ids": token_ids,
        "mtp_pred": logits[0].argmax(dim=-1).tolist(),
        "mtp_logits": _tolist(logits[0]),
    }


def reference_tokens(model: Qwen3_5MoeForCausalLM, seed: int) -> dict[str, Any]:
    prompt_ids = [1, 17, 42, 99, 7, 255]
    full_ids = list(prompt_ids)
    with torch.inference_mode():
        for _ in range(32):
            ids = torch.tensor([full_ids], dtype=torch.long)
            logits = model(input_ids=ids, use_cache=False).logits
            full_ids.append(int(logits[0, -1].argmax()))
        teacher = model(input_ids=torch.tensor([full_ids[:-1]]), use_cache=False).logits
    start = len(prompt_ids) - 1
    tf_pred = teacher[0, start:].argmax(dim=-1).tolist()
    if tf_pred != full_ids[len(prompt_ids) :]:
        raise AssertionError("teacher-forced predictions differ from greedy generation")
    return {
        "seed": seed,
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
        "tf_pred": tf_pred,
    }


def _tolist(value: torch.Tensor) -> Any:
    return value.detach().cpu().float().tolist()


def deltanet_fixture(seed: int) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed + 1)
    seq, heads, dim, kernel = 5, 2, 4, 4
    raw = torch.randn(seq, heads, dim * 3, generator=gen) * 0.25
    q_raw, k_raw, v_raw = raw.chunk(3, dim=-1)
    z = torch.randn(seq, heads, dim, generator=gen) * 0.25
    a = torch.randn(seq, heads, generator=gen) * 0.25
    b = torch.randn(seq, heads, generator=gen) * 0.25
    conv_weight = torch.randn(heads * dim * 3, 1, kernel, generator=gen) * 0.2
    channels = raw.reshape(seq, -1).transpose(0, 1).unsqueeze(0)
    conv_pre = F.conv1d(F.pad(channels, (kernel - 1, 0)), conv_weight, groups=channels.shape[1])
    conv = F.silu(conv_pre).squeeze(0).transpose(0, 1).reshape(seq, heads, dim * 3)
    q, k, v = conv.chunk(3, dim=-1)
    a_log = torch.randn(heads, generator=gen) * 0.1
    dt_bias = torch.randn(heads, generator=gen) * 0.1
    norm_weight = torch.rand(dim, generator=gen) + 0.5
    state = torch.zeros(heads, dim, dim)
    trajectory, outputs, beta_all, decay_all = [], [], [], []
    for token in range(seq):
        beta = torch.sigmoid(b[token])
        log_decay = -torch.exp(a_log) * F.softplus(a[token] + dt_bias)
        decay = torch.exp(log_decay)
        qn = F.normalize(q[token], dim=-1, eps=1e-6) / math.sqrt(dim)
        kn = F.normalize(k[token], dim=-1, eps=1e-6)
        state = state * decay[:, None, None]
        memory = torch.einsum("hij,hi->hj", state, kn)
        state = state + torch.einsum("hi,hj->hij", kn, (v[token] - memory) * beta[:, None])
        out = torch.einsum("hij,hi->hj", state, qn)
        rms = out * torch.rsqrt(out.square().mean(dim=-1, keepdim=True) + 1e-6)
        outputs.append(rms * norm_weight * F.silu(z[token]))
        trajectory.append(state.clone())
        beta_all.append(beta)
        decay_all.append(decay)
    return {
        "raw_qkv": _tolist(raw),
        "z": _tolist(z),
        "a": _tolist(a),
        "b": _tolist(b),
        "conv_weight": _tolist(conv_weight),
        "conv_pre": _tolist(conv_pre.squeeze(0).transpose(0, 1)),
        "conv_silu_qkv": _tolist(conv),
        "A_log": _tolist(a_log),
        "dt_bias": _tolist(dt_bias),
        "norm_weight": _tolist(norm_weight),
        "beta": _tolist(torch.stack(beta_all)),
        "decay": _tolist(torch.stack(decay_all)),
        "state_trajectory": _tolist(torch.stack(trajectory)),
        "output": _tolist(torch.stack(outputs)),
        "eps": 1e-6,
    }


def rope_fixture(seed: int) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed + 2)
    seq, heads, head_dim, rotary_dim = 6, 2, 64, 16
    q = torch.randn(seq, heads, head_dim, generator=gen) * 0.2
    k = torch.randn(seq, heads, head_dim, generator=gen) * 0.2
    pos = torch.arange(seq, dtype=torch.float32)
    inv = 10_000_000.0 ** (-torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    angles = pos[:, None] * inv[None, :]
    cos = torch.cat((angles, angles), dim=-1).cos()[:, None, :]
    sin = torch.cat((angles, angles), dim=-1).sin()[:, None, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
        half = rotary_dim // 2
        rotated_half = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
        return torch.cat((rot * cos + rotated_half * sin, passthrough), dim=-1)

    return {
        "q": _tolist(q),
        "k": _tolist(k),
        "positions": pos.int().tolist(),
        "theta": 10_000_000.0,
        "rotary_dim": rotary_dim,
        "q_out": _tolist(rotate(q)),
        "k_out": _tolist(rotate(k)),
    }


def router_fixture(seed: int) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed + 3)
    logits = torch.randn(7, 8, generator=gen)
    probs = torch.softmax(logits.float(), dim=-1)
    weights, indices = torch.topk(probs, 2, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return {
        "logits": _tolist(logits),
        "top_k": 2,
        "indices": indices.tolist(),
        "weights": _tolist(weights),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_tokenizer(outdir: Path, repo: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True)
    tokenizer.backend_tokenizer.save(str(outdir / "tokenizer.json"))
    source = Path(hf_hub_download(repo_id=repo, filename="tokenizer_config.json"))
    shutil.copy2(source, outdir / "tokenizer_config.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_paths(mode: str, root: Path) -> tuple[Path, Path]:
    if mode == "none":
        return root / "qwen_tiny", root / "ref_qwen.json"
    if mode == "int8":
        return root / "qwen_tiny_int8", root / "ref_qwen_int8.json"
    return root / "qwen_tiny_i4", root / "ref_qwen_i4.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=("none", "int8", "int4g128"), default="none")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--tokenizer-repo", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--skip-tokenizer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_runtime()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = rounded_model(args.quant, args.seed)
    mtp_source = rounded_model("none", args.seed)
    mtp_state = synthetic_mtp(mtp_source, args.quant, args.seed)
    outdir, refpath = output_paths(args.quant, args.output_root)
    outdir.mkdir(parents=True, exist_ok=True)
    model.config.to_json_file(outdir / "config.json", use_diff=False)
    if not args.skip_tokenizer:
        copy_tokenizer(outdir, args.tokenizer_repo)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state.update(mtp_state)
    snapshot = save_container(state, outdir, args.quant)
    reference = reference_tokens(model, args.seed)
    reference.update(mtp_reference(model, mtp_state, reference["full_ids"][:12]))
    reference.update(
        {
            "quant": args.quant,
            "transformers_version": transformers.__version__,
            "snapshot_sha256": sha256(snapshot),
        }
    )
    write_json(refpath, reference)
    if args.quant == "none":
        fixtures = args.output_root / "fixtures"
        fixtures.mkdir(parents=True, exist_ok=True)
        write_json(fixtures / "deltanet.json", deltanet_fixture(args.seed))
        write_json(fixtures / "rope_partial.json", rope_fixture(args.seed))
        write_json(fixtures / "router.json", router_fixture(args.seed))
    print(f"transformers={transformers.__version__} seed={args.seed} quant={args.quant}")
    print("state_dict (HF + synthetic MTP):")
    for name, value in sorted(state.items()):
        print(f"  {name}: {list(value.shape)} {value.dtype}")
    print(f"wrote {snapshot} ({reference['snapshot_sha256']})")
    print(f"wrote {refpath}; greedy/teacher-forced 32/32")


if __name__ == "__main__":
    main()
