#!/usr/bin/env python3
"""Generate the Qwen4-Exp PLE layer parity fixture.

The PLE layer turns a token's hashed n-gram embedding into an additive
correction on every hyper-connection stream. Per stream it scores the projected
n-gram key against the normalized stream, squashes that score through a
sign-preserving square root, gates a shared value with it, and finally adds a
dilated depthwise convolution over the normalized result.

Two details are easy to get wrong and are the reason this is a fixture rather
than an inspection:

* the gate uses ``sign(g) * sqrt(max(|g|, 1e-6))`` -- a plain ``sqrt`` would
  drop every negative score;
* the convolution is depthwise with ``dilation = ngram_size`` and a left pad of
  ``(kernel - 1) * dilation``, so tap ``j`` reads ``t - (kernel-1-j)*dilation``.

The n-gram row gather itself is covered by `make_ple_ngram_fixture.py`; this
fixture starts from the gathered embedding.

Reference: Qwen4ExpTextPLELayer.forward in transformers' modular_qwen4_exp.py.
Writes `fixtures/ple_layer.json` consumed by `tests/test_ple_layer.c`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "ple_layer.json"


def grouped_rmsnorm(x: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float) -> torch.Tensor:
    grouped = x.reshape(*x.shape[:-1], -1, group_size)
    normed = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + eps)
    return normed.flatten(-2) * (1.0 + weight)


def ple_forward(
    hidden: torch.Tensor,       # [T, hc*H] current stream
    embeddings: torch.Tensor,   # [T, D] gathered n-gram embedding
    key_w: torch.Tensor,        # [hc*H, D]
    value_w: torch.Tensor,      # [H, D]
    norm_key: torch.Tensor,     # [hc*H]
    norm_query: torch.Tensor,   # [hc*H]
    norm_conv: torch.Tensor,    # [hc*H]
    conv_w: torch.Tensor,       # [hc*H, kernel]
    hc: int,
    H: int,
    dilation: int,
    eps: float,
) -> torch.Tensor:
    T = hidden.shape[0]
    key_normed = grouped_rmsnorm(F.linear(embeddings, key_w), norm_key, H, eps).unflatten(-1, (hc, H))
    value = F.linear(embeddings, value_w)
    query_normed = grouped_rmsnorm(hidden, norm_query, H, eps).unflatten(-1, (hc, H))
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(H)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_normed = grouped_rmsnorm(gated.flatten(-2), norm_conv, H, eps)
    gated = gated.flatten(-2)

    # Depthwise causal conv with dilation, left-padded by (kernel-1)*dilation.
    kernel = conv_w.shape[-1]
    pad = (kernel - 1) * dilation
    padded = F.pad(gated_normed.transpose(0, 1).unsqueeze(0), (pad, 0))
    conv = F.conv1d(padded, conv_w.unsqueeze(1), groups=hc * H, dilation=dilation)
    conv = F.silu(conv).squeeze(0).transpose(0, 1)[:T]
    return gated + conv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=5)
    parser.add_argument("--hc-count", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=6)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--kernel", type=int, default=4)   # ple_conv_kernel_size
    parser.add_argument("--dilation", type=int, default=3)  # ngram_size
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    T, hc, H, D = args.tokens, args.hc_count, args.hidden, args.embed_dim
    wide = hc * H
    dtype = torch.float64

    hidden = torch.randn(T, wide, dtype=dtype)
    embeddings = torch.randn(T, D, dtype=dtype)
    key_w = torch.randn(wide, D, dtype=dtype) * 0.3
    value_w = torch.randn(H, D, dtype=dtype) * 0.3
    norm_key = torch.randn(wide, dtype=dtype) * 0.2
    norm_query = torch.randn(wide, dtype=dtype) * 0.2
    norm_conv = torch.randn(wide, dtype=dtype) * 0.2
    conv_w = torch.randn(wide, args.kernel, dtype=dtype) * 0.4

    expected = ple_forward(
        hidden, embeddings, key_w, value_w, norm_key, norm_query, norm_conv,
        conv_w, hc, H, args.dilation, args.eps,
    )

    fixture = {
        "config": {
            "tokens": T, "hc_count": hc, "hidden": H, "embed_dim": D,
            "kernel": args.kernel, "dilation": args.dilation, "eps": args.eps,
        },
        "hidden": hidden.tolist(),
        "embeddings": embeddings.tolist(),
        "key_w": key_w.tolist(),
        "value_w": value_w.tolist(),
        "norm_key": norm_key.tolist(),
        "norm_query": norm_query.tolist(),
        "norm_conv": norm_conv.tolist(),
        "conv_w": conv_w.tolist(),
        "expected_output": expected.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"wrote {args.output} T={T} hc={hc} H={H} D={D} kernel={args.kernel} dilation={args.dilation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
