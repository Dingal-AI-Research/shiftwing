#!/usr/bin/env python3
"""Generate the Qwen4-Exp gated-residual (hyper-connection) parity fixture.

Qwen3.8-Flash-Next replaces the plain residual add with a gated multi-stream
residual: the inter-layer hidden state is `hc_count * hidden_size` wide, each
block reads a single collapsed `hidden_size` vector, and the block output is
injected back into every stream with its own learned weight.

The reference math is transcribed from `Qwen4ExpTextGatedResidual` and
`Qwen4ExpTextDecoderLayer.forward` in transformers' `modular_qwen4_exp.py`.
Transformers 5.14.1 does not ship `qwen4_exp`, so the module is rebuilt here
from primitives rather than imported; keeping it self-contained also means the
fixture does not move when transformers is upgraded.

Writes `fixtures/hyper_connection.json` consumed by `tests/test_hyper_connection.c`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "hyper_connection.json"


def grouped_rmsnorm(x: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float) -> torch.Tensor:
    """RMSNorm over independent groups, then a full-width elementwise weight.

    `Qwen4ExpTextRMSNorm(dim=hc*H, group_size=H)` reshapes to (..., hc, H) and
    normalizes each stream on its own before flattening and scaling. A single
    norm over all hc*H features would couple the streams and is not equivalent.

    The scale is `(1 + weight)`, not `weight`: `Qwen4ExpTextRMSNorm` overrides
    only `_norm` and inherits `Qwen3_5RMSNorm.forward`, whose weight is
    zero-centered and initialized to zeros. This matches the engine's existing
    `rmsnorm_zero` (`x*r*(1.f+w[i])`), so converted checkpoints need no
    rebasing.
    """
    grouped = x.reshape(*x.shape[:-1], -1, group_size)
    normed = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + eps)
    return normed.flatten(-2) * (1.0 + weight)


def gated_residual(
    hyper_input: torch.Tensor,
    hc_norm_weight: torch.Tensor,
    mix_down: torch.Tensor,
    mix_up: torch.Tensor,
    block_inject: torch.Tensor | None,
    hc_count: int,
    hidden_size: int,
    eps: float,
):
    """One `Qwen4ExpTextGatedResidual` forward.

    Returns `(mixed_input, injection_weights)`; `injection_weights` is None for
    the final `hyper_connection_mixer`, which is built with use_combine=False
    and only collapses the streams before the output norm.
    """
    normed = grouped_rmsnorm(hyper_input, hc_norm_weight, hidden_size, eps)
    low = F.silu(F.linear(normed, mix_down) / hc_count)
    gate = torch.sigmoid(F.linear(low, mix_up))
    gate = gate.unflatten(-1, (hc_count, hidden_size))
    mixed = (gate * normed.unflatten(-1, (hc_count, hidden_size))).mean(dim=-2)
    if block_inject is None:
        return mixed, None
    injection = 2 * torch.sigmoid(F.linear(normed, block_inject) / hc_count)
    return mixed, injection


def inject(hyper_input: torch.Tensor, block_out: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
    """Scatter one block output back across the streams.

    `hidden_states = hyper_input + (block_out ⊗ injection).flatten(-2)`. Note the
    base is the *raw* hyper input, not the normalized copy the gates were
    computed from.
    """
    return hyper_input + (block_out.unsqueeze(-2) * injection.unsqueeze(-1)).flatten(-2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--hc-count", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--lowrank", type=int, default=6)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    T, hc, H, R = args.tokens, args.hc_count, args.hidden, args.lowrank
    wide = hc * H

    # float64 throughout: the C engine accumulates in float, and a float32
    # reference would fold its own rounding into the tolerance we are trying
    # to measure.
    dtype = torch.float64
    hyper = torch.randn(T, wide, dtype=dtype)
    # zero-centered, matching the checkpoint convention (weight init is zeros)
    hc_norm_w = torch.randn(wide, dtype=dtype) * 0.2
    mix_down = torch.randn(R, wide, dtype=dtype) * 0.1
    mix_up = torch.randn(wide, R, dtype=dtype) * 0.1
    block_inject = torch.randn(hc, wide, dtype=dtype) * 0.1
    block_out = torch.randn(T, H, dtype=dtype)

    mixed, injection = gated_residual(hyper, hc_norm_w, mix_down, mix_up, block_inject, hc, H, args.eps)
    injected = inject(hyper, block_out, injection)

    # use_combine=False path: the model-level mixer collapses streams and
    # returns no injection weights.
    mixer_only, none_injection = gated_residual(hyper, hc_norm_w, mix_down, mix_up, None, hc, H, args.eps)
    assert none_injection is None

    fixture = {
        "config": {
            "tokens": T,
            "hc_count": hc,
            "hidden": H,
            "lowrank": R,
            "eps": args.eps,
            "seed": args.seed,
        },
        "hyper_input": hyper.tolist(),
        "hc_norm_weight": hc_norm_w.tolist(),
        "mix_down": mix_down.tolist(),
        "mix_up": mix_up.tolist(),
        "block_inject": block_inject.tolist(),
        "block_out": block_out.tolist(),
        "expected_mixed": mixed.tolist(),
        "expected_injection": injection.tolist(),
        "expected_injected": injected.tolist(),
        "expected_mixer_only": mixer_only.tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"wrote {args.output} T={T} hc={hc} H={H} lowrank={R}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
