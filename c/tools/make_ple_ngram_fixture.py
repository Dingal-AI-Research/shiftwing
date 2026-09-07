#!/usr/bin/env python3
"""Generate the Qwen4-Exp PLE n-gram index parity fixture.

Layer `ple_layer_ids[0] - 1` injects hashed n-gram features. Each token maps to
`ngram_heads = (ngram_size - 1) * heads_per_ngram` rows of a single embedding
table (stored as `split_ngram_parts` row shards), and the row index is a
multiply-xor hash of the token and its predecessors, taken modulo a per-head
prime and shifted by a per-head offset.

Getting this wrong is silent: every index still lands inside the table, so the
model keeps running and simply reads the wrong 51.2B-parameter rows. Hence a
parity fixture rather than an inspection.

The multipliers, per-head vocabulary sizes and offsets are *stored buffers* in
the checkpoint, so the runtime reads them rather than recomputing splitmix64
and the prime search. This generator reproduces them only to build a
self-contained fixture.

Reference: Qwen4ExpTextNGramEmbedding in transformers' modular_qwen4_exp.py.
Writes `fixtures/ple_ngram.json` consumed by `tests/test_ple_ngram.c`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "ple_ngram.json"

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> list[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    out = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        out.append(2 * (_splitmix64(value) % half_bound) + 1)
    return out


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def shift_right_ignore_eos(tokens: list[int], shift: int, eos: int) -> list[int]:
    """Shift right within the current EOS-delimited segment.

    Positions closer to the segment start than `shift` -- and positions that
    would read before the sequence -- yield EOS rather than crossing into the
    previous document.
    """
    if shift == 0:
        return list(tokens)
    n = len(tokens)
    previous_eos_inclusive, running = [], -1
    for position, token in enumerate(tokens):
        if token == eos:
            running = position
        previous_eos_inclusive.append(running)
    previous_eos = [-1] + previous_eos_inclusive[:-1]
    out = []
    for position in range(n):
        segment_start = previous_eos[position] + 1
        position_in_segment = position - segment_start
        source = position - shift
        out.append(tokens[source] if position_in_segment >= shift and source >= 0 else eos)
    return out


def ngram_ids(
    tokens: list[int],
    multipliers: list[int],
    vocab_sizes: list[int],
    offsets: list[int],
    ngram_size: int,
    heads_per_ngram: int,
    eos: int,
) -> list[list[int]]:
    shifted = [shift_right_ignore_eos(tokens, shift, eos) for shift in range(ngram_size)]
    blocks: list[list[list[int]]] = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = [token * multipliers[0] for token in shifted[0]]
        for position in range(1, ngram):
            mixed = [value ^ (shifted[position][i] * multipliers[position]) for i, value in enumerate(mixed)]
        block = [
            [(value % vocab_sizes[start + h]) + offsets[start + h] for h in range(end - start)]
            for value in mixed
        ]
        blocks.append(block)
    return [sum((block[position] for block in blocks), []) for position in range(len(tokens))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults mirror Qwen/Qwen3.8-Flash-Next's text_config.
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--heads-per-ngram", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--ngram-vocab-size-base", type=int, default=20000000)
    parser.add_argument("--eos", type=int, default=248044)
    parser.add_argument("--seed", type=int, default=1234)  # Qwen4ExpTextConfig default
    parser.add_argument("--ple-layer-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    heads = (args.ngram_size - 1) * args.heads_per_ngram
    multipliers = build_layer_multipliers(args.vocab_size, args.ngram_size, args.ple_layer_index, args.seed)
    vocab_sizes, offsets, total = [], [], 0
    for head in range(heads):
        global_head = args.ple_layer_index * heads + head
        size = _find_nth_prime_after(args.ngram_vocab_size_base - 1, global_head + 1)
        vocab_sizes.append(size)
        offsets.append(total)
        total += size

    # A deliberately awkward sequence: an EOS mid-stream so the segment-aware
    # shift is exercised, a repeated token, and the vocabulary extremes.
    tokens = [7, 1234, args.eos, 5, 5, 248319, 0, 99, args.eos, 42]
    expected = ngram_ids(
        tokens, multipliers, vocab_sizes, offsets, args.ngram_size, args.heads_per_ngram, args.eos
    )

    fixture = {
        "config": {
            "tokens": len(tokens),
            "ngram_size": args.ngram_size,
            "heads_per_ngram": args.heads_per_ngram,
            "ngram_heads": heads,
            "eos": args.eos,
        },
        "input_ids": tokens,
        "layer_multipliers": multipliers,
        "ngram_heads_vocab_sizes": vocab_sizes,
        "ngram_heads_offsets": offsets,
        "expected_ngram_ids": expected,
        "expected_shifted": [
            shift_right_ignore_eos(tokens, shift, args.eos) for shift in range(args.ngram_size)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"wrote {args.output} tokens={len(tokens)} heads={heads} total_vocab={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
