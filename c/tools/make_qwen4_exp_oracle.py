#!/usr/bin/env python3
"""Build a tiny Qwen4-Exp model and its reference activations.

The engine loads the real 91 GB container and produces fluent-looking but
meaningless text. Every novel component passes parity in isolation, so the fault
is in how they compose -- and composition bugs are found by diffing against a
reference layer by layer, which needs an oracle small enough to run both ways.

This builds a random-weight Qwen4-Exp exercising every path that matters: linear
attention, a QSA attention layer, the PLE trigram layer, MoE routing, and the
four-stream gated residual including the terminal mixer. It then records the
per-layer residual stream so `compare_qwen4_exp_oracle.py` can say which layer
first disagrees.

Needs the overlay interpreter (transformers main has qwen4_exp, the pinned
project venv does not):

    PYTHONPATH=.tf-main/site:.tf-main/transformers-src/src .venv/bin/python \\
        c/tools/make_qwen4_exp_oracle.py --outdir c/qwen4exp_tiny_src
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig


def tiny_config() -> Qwen4ExpTextConfig:
    """Smallest configuration that still exercises every Qwen4-Exp path.

    ple_layer_ids is one-indexed, so [2] puts the trigram layer at index 1 --
    which must be a linear_attention layer, as the upstream config validates.
    """
    return Qwen4ExpTextConfig(
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=4,
        hc_lowrank=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        indexer_budget=8,
        indexer_compress_ratio=4,
        indexer_head_dim=8,
        # partial rotary mirrors the real model (0.25), so rotary_dim = 32*0.25 = 8
        # and fits indexer_head_dim; mrope_section must sum to rotary_dim/2.
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
        indexer_n_heads=2,
        indexer_kv_heads=1,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=1000,
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ple_layer_ids=[2],
        split_ngram_parts=4,
        make_ngram_vocab_size_divisible_by=8,
        mtp_num_hidden_layers=0,
        rms_norm_eps=1e-6,
        eos_token_id=2,
        bos_token_id=1,
        tie_word_embeddings=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--real-dims", action="store_true",
                        help="use Qwen3.8-Flash-Next per-layer dimensions (few layers, small vocab)")
    parser.add_argument("--dims", default="",
                        help="comma-separated groups to take from the real model: hidden,gdn,attn,moe,hc,ple")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--indexer-budget", type=int, default=0,
                        help="force a small QSA budget so the indexer actually restricts")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = tiny_config()
    if args.layers != config.num_hidden_layers:
        config.num_hidden_layers = args.layers
        config.layer_types = [
            'qwen_sparse_attention' if (i + 1) % 4 == 0 else 'linear_attention' for i in range(args.layers)
        ]
    config.num_experts = args.experts
    groups = set(filter(None, args.dims.split(","))) or (
        {"hidden", "gdn", "attn", "moe", "hc", "ple"} if args.real_dims else set()
    )
    if groups:
        # Per-layer dimensions from Qwen/Qwen3.8-Flash-Next, selectable in
        # groups. Depth, expert count and both vocabularies stay small, so a
        # bug that depends on a real dimension reproduces in seconds and can be
        # bisected one group at a time.
        if "hidden" in groups:
            config.hidden_size = 2560
            config.ple_embed_dim = 2560
        if "attn" in groups:
            config.num_attention_heads = 24
            config.num_key_value_heads = 2
            config.head_dim = 256
            config.indexer_head_dim = 128
            config.indexer_n_heads = 4
            config.indexer_kv_heads = 1
            config.indexer_budget = 2048
            config.indexer_compress_ratio = 4
            config.rope_parameters = {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            }
        if "moe" in groups:
            config.moe_intermediate_size = 640
            config.shared_expert_intermediate_size = 640
            config.num_experts_per_tok = 10
        if "hc" in groups:
            config.hc_lowrank = 320
        if "gdn" in groups:
            config.linear_num_key_heads = 16
            config.linear_num_value_heads = 48
            config.linear_key_head_dim = 128
            config.linear_value_head_dim = 128
        if "ple" in groups:
            config.heads_per_ngram = 8
            config.ngram_vocab_size_base = 10000
            config.split_ngram_parts = 8

    if args.indexer_budget:
        config.indexer_budget = args.indexer_budget
    model = Qwen4ExpForCausalLM(config).eval()

    # Default init leaves many tensors near zero, which hides sign and ordering
    # mistakes. Give every parameter real spread so a wrong composition shows.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                param.normal_(0.0, 0.05)
            else:
                param.normal_(0.0, 0.05)

    args.outdir.mkdir(parents=True, exist_ok=True)
    # Force sharding so a model.safetensors.index.json is emitted --
    # convert_local_directory reads it to enumerate shards.
    model.save_pretrained(args.outdir, max_shard_size="200KB", safe_serialization=True)

    ids = torch.randint(0, config.vocab_size, (1, args.tokens))
    ids[0, args.tokens // 2] = config.eos_token_id  # exercise the PLE segment reset

    states: list[torch.Tensor] = []
    hooks = []

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        states.append(hidden.detach().float().clone())

    for layer in model.model.layers[: config.num_hidden_layers]:
        hooks.append(layer.register_forward_hook(capture))
    with torch.no_grad():
        out = model(ids, use_cache=False)
    for handle in hooks:
        handle.remove()

    wide = config.hc_count * config.hidden_size
    acts = args.outdir / "reference_acts"
    acts.mkdir(exist_ok=True)
    for index, hidden in enumerate(states):
        assert hidden.shape[-1] == wide, f"layer {index} width {hidden.shape[-1]} != {wide}"
        hidden[0].contiguous().numpy().astype("float32").tofile(acts / f"layer-{index:03d}.f32")

    reference = {
        "tokens": ids[0].tolist(),
        "hidden_width": wide,
        "hidden_size": config.hidden_size,
        "hc_count": config.hc_count,
        "layers": config.num_hidden_layers,
        "logits_last": out.logits[0, -1].float().tolist(),
        "argmax_per_position": out.logits[0].argmax(-1).tolist(),
    }
    (args.outdir / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")
    print(f"wrote {args.outdir}: {args.tokens} tokens, {config.num_hidden_layers} layers, width {wide}")
    print(f"  argmax per position: {reference['argmax_per_position']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
