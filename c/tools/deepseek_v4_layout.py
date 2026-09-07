#!/usr/bin/env python3
"""Exact tensor-name/dtype/physical-shape contract for the pinned 0731 model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

Descriptor = tuple[str, tuple[int, ...]]


def _add(target: dict[str, Descriptor], name: str, dtype: str, *shape: int) -> None:
    if name in target:
        raise AssertionError(f"duplicate expected tensor: {name}")
    target[name] = (dtype, tuple(shape))


def _fp8(target: dict[str, Descriptor], name: str, rows: int, cols: int) -> None:
    _add(target, name + ".weight", "F8_E4M3", rows, cols)
    _add(target, name + ".scale", "F8_E8M0", (rows + 127) // 128,
         (cols + 127) // 128)


def _fp4_expert(target: dict[str, Descriptor], prefix: str) -> None:
    _add(target, prefix + ".w1.weight", "I8", 2048, 2048)
    _add(target, prefix + ".w1.scale", "F8_E8M0", 2048, 128)
    _add(target, prefix + ".w2.weight", "I8", 4096, 1024)
    _add(target, prefix + ".w2.scale", "F8_E8M0", 4096, 64)
    _add(target, prefix + ".w3.weight", "I8", 2048, 2048)
    _add(target, prefix + ".w3.scale", "F8_E8M0", 2048, 128)


def _block(target: dict[str, Descriptor], prefix: str, *, hashed: bool) -> None:
    _add(target, prefix + ".attn.attn_sink", "F32", 64)
    _add(target, prefix + ".attn.kv_norm.weight", "BF16", 512)
    _add(target, prefix + ".attn.q_norm.weight", "BF16", 1024)
    _fp8(target, prefix + ".attn.wkv", 512, 4096)
    _fp8(target, prefix + ".attn.wo_a", 8192, 4096)
    _fp8(target, prefix + ".attn.wo_b", 4096, 8192)
    _fp8(target, prefix + ".attn.wq_a", 1024, 4096)
    _fp8(target, prefix + ".attn.wq_b", 32768, 1024)
    _add(target, prefix + ".attn_norm.weight", "BF16", 4096)
    for expert in range(256):
        _fp4_expert(target, f"{prefix}.ffn.experts.{expert}")
    _add(target, prefix + ".ffn.gate.weight", "BF16", 256, 4096)
    if hashed:
        _add(target, prefix + ".ffn.gate.tid2eid", "I64", 129280, 6)
    else:
        _add(target, prefix + ".ffn.gate.bias", "F32", 256)
    _fp8(target, prefix + ".ffn.shared_experts.w1", 2048, 4096)
    _fp8(target, prefix + ".ffn.shared_experts.w2", 4096, 2048)
    _fp8(target, prefix + ".ffn.shared_experts.w3", 2048, 4096)
    _add(target, prefix + ".ffn_norm.weight", "BF16", 4096)
    for operation in ("attn", "ffn"):
        _add(target, f"{prefix}.hc_{operation}_base", "F32", 24)
        _add(target, f"{prefix}.hc_{operation}_fn", "F32", 24, 16384)
        _add(target, f"{prefix}.hc_{operation}_scale", "F32", 3)


def _compressor(target: dict[str, Descriptor], prefix: str, ratio: int,
                dim: int) -> None:
    coefficient = 2 if ratio == 4 else 1
    _add(target, prefix + ".ape", "F32", ratio, coefficient * dim)
    _add(target, prefix + ".norm.weight", "BF16", dim)
    _add(target, prefix + ".wgate.weight", "BF16", coefficient * dim, 4096)
    _add(target, prefix + ".wkv.weight", "BF16", coefficient * dim, 4096)


def expected_layout() -> dict[str, Descriptor]:
    expected: dict[str, Descriptor] = {}
    _add(expected, "embed.weight", "BF16", 129280, 4096)
    _add(expected, "head.weight", "BF16", 129280, 4096)
    _add(expected, "norm.weight", "BF16", 4096)
    _add(expected, "hc_head_base", "F32", 4)
    _add(expected, "hc_head_fn", "F32", 4, 16384)
    _add(expected, "hc_head_scale", "F32", 1)
    ratios = [0, 0, *([4, 128] * 20), 4]
    for layer, ratio in enumerate(ratios):
        prefix = f"layers.{layer}"
        _block(expected, prefix, hashed=layer < 3)
        if ratio:
            _compressor(expected, prefix + ".attn.compressor", ratio, 512)
        if ratio == 4:
            _compressor(expected, prefix + ".attn.indexer.compressor", 4, 128)
            _add(expected, prefix + ".attn.indexer.weights_proj.weight",
                 "BF16", 64, 4096)
            _fp8(expected, prefix + ".attn.indexer.wq_b", 8192, 1024)
    for stage in range(3):
        prefix = f"mtp.{stage}"
        _block(expected, prefix, hashed=False)
    _add(expected, "mtp.0.main_norm.weight", "BF16", 4096)
    _fp8(expected, "mtp.0.main_proj", 4096, 12288)
    _add(expected, "mtp.2.confidence_head.proj.weight", "BF16", 1, 4352)
    _add(expected, "mtp.2.hc_head_base", "F32", 4)
    _add(expected, "mtp.2.hc_head_fn", "F32", 4, 16384)
    _add(expected, "mtp.2.hc_head_scale", "F32", 1)
    _add(expected, "mtp.2.markov_head.markov_w1.weight", "BF16", 129280, 256)
    _add(expected, "mtp.2.markov_head.markov_w2.weight", "BF16", 129280, 256)
    _add(expected, "mtp.2.norm.weight", "BF16", 4096)
    return expected


def category_counts(layout: Mapping[str, Descriptor]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for name in layout:
        if name.startswith("mtp.") and ".experts." in name:
            counts["dspark_expert"] += 1
        elif name.startswith("mtp."):
            counts["dspark_dense"] += 1
        elif ".ffn.experts." in name:
            counts["routed_expert"] += 1
        else:
            counts["dense"] += 1
    return dict(sorted(counts.items()))


def validate_descriptors(actual: Mapping[str, Descriptor]) -> list[str]:
    expected = expected_layout()
    failures: list[str] = []
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        failures.append(f"missing {len(missing)} tensors; first={missing[:5]}")
    if extra:
        failures.append(f"unexpected {len(extra)} tensors; first={extra[:5]}")
    mismatches = [
        (name, expected[name], actual[name])
        for name in sorted(set(expected) & set(actual))
        if expected[name] != actual[name]
    ]
    if mismatches:
        failures.append(f"descriptor mismatches {len(mismatches)}; first={mismatches[:5]}")
    expected_categories = {
        "dense": 1564,
        "dspark_dense": 97,
        "dspark_expert": 4608,
        "routed_expert": 66048,
    }
    if category_counts(actual) != expected_categories:
        failures.append(
            f"category counts: expected {expected_categories}, got {category_counts(actual)}"
        )
    return failures


def validate_records(groups: Mapping[str, Iterable[Any]]) -> list[str]:
    actual: dict[str, Descriptor] = {}
    duplicates: list[str] = []
    for records in groups.values():
        for record in records:
            if record.name in actual:
                duplicates.append(record.name)
            actual[record.name] = (record.dtype, tuple(record.shape))
    failures = validate_descriptors(actual)
    if duplicates:
        failures.insert(0, f"duplicate output records {len(duplicates)}; first={duplicates[:5]}")
    return failures
