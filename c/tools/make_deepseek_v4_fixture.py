#!/usr/bin/env python3
"""Generate the tiny DeepSeek-V4 scalar oracle from pinned reference math."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from deepseek_v4_spec import SOURCE_REVISION


REFERENCE_HASHES = {
    "config.json": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
    "inference/model.py": "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f",
    "inference/kernel.py": "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def hc_split(mixes: list[float], scale: list[float], base: list[float]):
    eps = 1e-6
    pre = [sigmoid(mixes[j] * scale[0] + base[j]) + eps for j in range(4)]
    post = [2 * sigmoid(mixes[4 + j] * scale[1] + base[4 + j]) for j in range(4)]
    comb = [
        [mixes[8 + j * 4 + k] * scale[2] + base[8 + j * 4 + k] for k in range(4)]
        for j in range(4)
    ]
    for row in comb:
        maximum = max(row)
        values = [math.exp(item - maximum) for item in row]
        total = sum(values)
        row[:] = [item / total + eps for item in values]
    for iteration in range(20):
        columns = [sum(comb[j][k] for j in range(4)) for k in range(4)]
        comb = [[comb[j][k] / (columns[k] + eps) for k in range(4)] for j in range(4)]
        if iteration == 19:
            break
        rows = [sum(row) for row in comb]
        comb = [[comb[j][k] / (rows[j] + eps) for k in range(4)] for j in range(4)]
    return pre, post, [value for row in comb for value in row]


def hc_forward_fixture():
    dim = 4
    residual = [(index - 7) / 5 for index in range(4 * dim)]
    function_weight = [((index * 7) % 23 - 11) / 37 for index in range(24 * 4 * dim)]
    square_mean = sum(value * value for value in residual) / len(residual)
    inverse_rms = 1 / math.sqrt(square_mean + 1e-6)
    mixes = [
        sum(function_weight[mix * len(residual) + index] * residual[index]
            for index in range(len(residual))) * inverse_rms
        for mix in range(24)
    ]
    scale = [0.5, 0.75, 1.25]
    base = [(index % 5 - 2) / 11 for index in range(24)]
    pre, post, comb = hc_split(mixes, scale, base)
    reduced = [
        sum(pre[copy] * residual[copy * dim + axis] for copy in range(4))
        for axis in range(dim)
    ]
    module = [(axis - 2) / 3 for axis in range(dim)]
    expanded = [
        post[to] * module[axis] +
        sum(comb[source * 4 + to] * residual[source * dim + axis]
            for source in range(4))
        for to in range(4) for axis in range(dim)
    ]
    return {
        "dim": dim, "residual": residual, "function_weight": function_weight,
        "scale": scale, "base": base, "mixes": mixes, "pre": pre,
        "post": post, "comb": comb, "reduced": reduced, "module": module,
        "expanded": expanded,
    }


def compressor_decode_fixture(overlap: bool):
    ratio, dim = 4, 2
    coefficient = 2 if overlap else 1
    width = coefficient * dim
    rows = coefficient * ratio
    kv_state = [[0.0] * width for _ in range(rows)]
    score_state = [[float("-inf")] * width for _ in range(rows)]
    ape = [
        [((slot * 3 + axis) % 7 - 3) / 9 for axis in range(width)]
        for slot in range(ratio)
    ]
    kv_inputs = [
        [((position + 1) * (axis + 1)) / 5 for axis in range(width)]
        for position in range(8)
    ]
    score_inputs = [
        [((position * 5 + axis) % 11 - 5) / 7 for axis in range(width)]
        for position in range(8)
    ]
    outputs = []
    for position, (kv, score) in enumerate(zip(kv_inputs, score_inputs)):
        slot = position % ratio
        target = ratio + slot if overlap else slot
        kv_state[target] = kv[:]
        score_state[target] = [score[axis] + ape[slot][axis]
                                     for axis in range(width)]
        if slot != ratio - 1:
            continue
        pooled = []
        for axis in range(dim):
            if overlap:
                values = [kv_state[row][axis] for row in range(ratio)] + [
                    kv_state[ratio + row][dim + axis] for row in range(ratio)
                ]
                scores = [score_state[row][axis] for row in range(ratio)] + [
                    score_state[ratio + row][dim + axis] for row in range(ratio)
                ]
            else:
                values = [kv_state[row][axis] for row in range(ratio)]
                scores = [score_state[row][axis] for row in range(ratio)]
            maximum = max(scores)
            weights = [math.exp(value - maximum) for value in scores]
            pooled.append(sum(value * weight for value, weight in zip(values, weights)) /
                          sum(weights))
        outputs.append({"position": position, "value": pooled})
        if overlap:
            for row in range(ratio):
                kv_state[row] = kv_state[ratio + row][:]
                score_state[row] = score_state[ratio + row][:]
    return {
        "ratio": ratio,
        "dim": dim,
        "overlap": overlap,
        "ape": [value for row in ape for value in row],
        "kv": [value for row in kv_inputs for value in row],
        "score": [value for row in score_inputs for value in row],
        "outputs": outputs,
        "final_kv_state": [value for row in kv_state for value in row],
        "final_score_state": [value for row in score_state for value in row],
    }


def indexer_fixture():
    heads, dim, tokens, topk = 3, 4, 6, 3
    query = [((head * 7 + axis * 3) % 13 - 6) / 5
             for head in range(heads) for axis in range(dim)]
    kv = [((token * 5 + axis * 2) % 17 - 8) / 6
          for token in range(tokens) for axis in range(dim)]
    weights = [0.75, -0.25, 1.125]
    scores = []
    for token in range(tokens):
        score = 0.0
        for head in range(heads):
            dot = sum(query[head * dim + axis] * kv[token * dim + axis]
                      for axis in range(dim))
            score += max(dot, 0.0) * weights[head]
        scores.append(score)
    indices = sorted(range(tokens), key=lambda token: scores[token],
                     reverse=True)[:topk]
    return {"heads": heads, "dim": dim, "tokens": tokens, "topk": topk,
            "query": query, "kv": kv, "weights": weights,
            "scores": scores, "indices": indices}


def router(logits: list[float], bias: list[float]):
    original = [math.sqrt(math.log1p(math.exp(value))) for value in logits]
    selected = [score + delta for score, delta in zip(original, bias)]
    indices = sorted(range(len(logits)), key=lambda index: selected[index], reverse=True)[:6]
    total = sum(original[index] for index in indices)
    return indices, [1.5 * original[index] / total for index in indices]


def rope(values: list[float], position: int):
    result = values[:]
    dim, original, base, factor = len(values), 65536, 160000.0, 16.0
    low = max(math.floor(dim * math.log(original / (32 * 2 * math.pi)) / (2 * math.log(base))), 0)
    high = min(math.ceil(dim * math.log(original / (1 * 2 * math.pi)) / (2 * math.log(base))), dim - 1)
    for pair in range(dim // 2):
        ramp = min(1.0, max(0.0, (pair - low) / (high - low if high != low else 0.001)))
        smooth = 1 - ramp
        frequency = base ** (-(2 * pair) / dim)
        frequency = frequency / factor * (1 - smooth) + frequency * smooth
        angle = position * frequency
        a, b = result[2 * pair : 2 * pair + 2]
        result[2 * pair] = a * math.cos(angle) - b * math.sin(angle)
        result[2 * pair + 1] = a * math.sin(angle) + b * math.cos(angle)
    return result


def build(reference_root: Path) -> dict:
    observed = {name: sha256(reference_root / name) for name in REFERENCE_HASHES}
    if observed != REFERENCE_HASHES:
        raise ValueError(f"pinned reference hash mismatch: {observed!r}")
    mixes = [(index - 12) / 7 for index in range(24)]
    base = [(index % 5 - 2) / 11 for index in range(24)]
    scale = [0.5, 0.75, 1.25]
    pre, post, comb = hc_split(mixes, scale, base)
    indices, weights = router(list(range(-4, 4)), [100.0] + [0.0] * 7)
    return {
        "schema": "colib.deepseek-v4.reference-fixture.v1",
        "source_revision": SOURCE_REVISION,
        "reference_sha256": observed,
        "fp4_e2m1": [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        "hc": {"mixes": mixes, "scale": scale, "base": base, "pre": pre, "post": post, "comb": comb},
        "hc_forward": hc_forward_fixture(),
        "router": {"logits": list(range(-4, 4)), "bias": [100.0] + [0.0] * 7,
                   "indices": indices, "weights": weights},
        "indexer": indexer_fixture(),
        "rope": {"input": [1, 2, 3, 4], "position": 1234,
                 "output": rope([1, 2, 3, 4], 1234)},
        "compress_pool": {"kv": [1, 10, 3, 30], "score": [0, 0, 0, 0],
                          "ape": [0, 0, 0, 0], "tokens": 2, "dim": 2,
                          "output": [2, 20]},
        "compressor_decode": {
            "overlap": compressor_decode_fixture(True),
            "plain": compressor_decode_fixture(False),
        },
        "fp4_expert": {"hidden": 128, "intermediate": 128, "route": 0.5,
                       "input_nonzero": {"0": 1.0}, "output_value": 48.0},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.reference_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
