#!/usr/bin/env python3
"""Generate the Qwen4-Exp QSA indexer selection parity fixture.

The `full_attention` layers of Qwen3.8-Flash-Next are not plain GQA: a
`Qwen4ExpTextQSAIndexer` first restricts what each query may attend to. Visible
keys are grouped into blocks of `indexer_compress_ratio`, each block is scored
against the query heads, and the top `indexer_budget / indexer_compress_ratio`
blocks are admitted along with the trailing incomplete block.

This fixture covers the novel part -- block scoring, top-k selection, and the
always-admitted tail. Mean pooling, RMSNorm and RoPE are composed from
primitives the engine already tests (`test_gqa_rope`, `rope_partial.json`), so
block keys are supplied post-RoPE rather than rebuilt here.

Selection feeds an additive attention mask, so only the admitted *set* is
semantically meaningful, not the top-k ordering. The fixture therefore records
a boolean mask.

Reference: Qwen4ExpTextQSAIndexer.forward in transformers' modular_qwen4_exp.py.
Writes `fixtures/qsa_indexer.json` consumed by `tests/test_qsa_indexer.c`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "qsa_indexer.json"


def select(
    query: torch.Tensor,        # [n_heads, head_dim], post-norm and post-RoPE
    block_keys: torch.Tensor,   # [n_blocks, head_dim], pooled/normed/RoPE'd
    block_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score blocks and return (scores, selected block indices).

    `scores = relu(q @ K^T).sum(over heads) / sqrt(head_dim)`. The ReLU before
    the sum is what makes this a selection rule rather than a soft attention:
    a head that disagrees contributes nothing instead of cancelling a head that
    agrees.
    """
    head_dim = query.shape[-1]
    raw = torch.matmul(query.float(), block_keys.float().transpose(-1, -2))
    scores = torch.relu(raw).sum(dim=0) / math.sqrt(head_dim)
    chosen = scores.topk(min(block_topk, block_keys.shape[0]), dim=0).indices
    return scores, chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible", type=int, default=19, help="visible key positions for this query")
    parser.add_argument("--n-heads", type=int, default=4)      # indexer_n_heads
    parser.add_argument("--head-dim", type=int, default=8)     # scaled down from 128
    parser.add_argument("--compress-ratio", type=int, default=4)
    parser.add_argument("--block-topk", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    visible, ratio = args.visible, args.compress_ratio
    complete = visible // ratio
    dtype = torch.float64

    query = torch.randn(args.n_heads, args.head_dim, dtype=dtype)
    block_keys = torch.randn(complete, args.head_dim, dtype=dtype)
    scores, chosen = select(query, block_keys, args.block_topk)

    # Admitted set: every token of each chosen block, plus the trailing
    # incomplete block, which bypasses scoring entirely.
    mask = [False] * visible
    for block in chosen.tolist():
        for offset in range(ratio):
            mask[block * ratio + offset] = True
    for position in range(complete * ratio, visible):
        mask[position] = True

    fixture = {
        "config": {
            "visible": visible,
            "n_heads": args.n_heads,
            "head_dim": args.head_dim,
            "compress_ratio": ratio,
            "block_topk": args.block_topk,
            "complete_blocks": complete,
        },
        "query": query.tolist(),
        "block_keys": block_keys.tolist(),
        "expected_scores": scores.tolist(),
        "expected_mask": [int(value) for value in mask],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        f"wrote {args.output} visible={visible} complete_blocks={complete} "
        f"topk={args.block_topk} admitted={sum(mask)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
