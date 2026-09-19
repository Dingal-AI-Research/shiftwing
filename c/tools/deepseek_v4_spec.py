#!/usr/bin/env python3
"""Pinned DeepSeek-V4-Flash-0731 identity and runtime contract.

This module is intentionally dependency-free so the converter, preflight,
runtime launcher, tests, and evidence tooling all consume one fail-closed
source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SOURCE_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
SOURCE_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
TOKENIZER_SHA256 = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
MODEL_DIRECTORY = "deepseek-v4-flash-0731"
MODEL_ID = "deepseek-v4-flash-0731-colib"
MODEL_FAMILY = "deepseek-v4"

DEFAULT_CONTEXT = 16_384
REVIEW_INPUT_TOKENS = 92_160
REVIEW_OUTPUT_TOKENS = 8_192
REVIEW_CONTEXT = REVIEW_INPUT_TOKENS + REVIEW_OUTPUT_TOKENS
MAX_CONTEXT = REVIEW_CONTEXT
ORIGINAL_CONTEXT = 65_536
UPSTREAM_MAX_CONTEXT = 1_048_576
WEIGHT_SHARDS = 48

# Planning values only.  The converter replaces them with exact index/header
# totals before writing a byte and records both estimates and exact totals.
PLANNED_CHECKPOINT_BYTES = 167_000_000_000
PLANNED_CONTAINER_BYTES = 168_000_000_000
PLANNED_STAGING_BYTES = 5_000_000_000
MIN_FINAL_FREE_BYTES = 100 * 1024**3

DS_PARK_CHOICES = ("auto", "on", "off")
DS_PARK_DRAFT_LENGTHS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class SamplingProfile:
    name: str
    temperature: float
    top_p: float


DETERMINISTIC_SAMPLING = SamplingProfile("deterministic", 0.0, 1.0)
AGENT_SAMPLING = SamplingProfile("agent", 1.0, 0.95)


# Inference-critical fields from the pinned official config.json.  Extra
# upstream fields are accepted; a mismatch in any field below is fatal.
EXPECTED_CONFIG: dict[str, Any] = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "bos_token_id": 0,
    "eos_token_id": 1,
    "expert_dtype": "fp4",
    "hc_eps": 1e-6,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "head_dim": 512,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "index_head_dim": 128,
    "index_n_heads": 64,
    "index_topk": 512,
    "max_position_embeddings": UPSTREAM_MAX_CONTEXT,
    "model_type": "deepseek_v4",
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 6,
    "num_hash_layers": 3,
    "num_hidden_layers": 43,
    "num_key_value_heads": 1,
    "num_nextn_predict_layers": 1,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10_000,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "sliding_window": 128,
    "swiglu_limit": 10.0,
    "tie_word_embeddings": False,
    "topk_method": "noaux_tc",
    "vocab_size": 129_280,
    "compress_rope_theta": 160_000,
    "dspark_block_size": 5,
    "dspark_markov_rank": 256,
    "dspark_noise_token_id": 128_799,
    "dspark_target_layer_ids": [40, 41, 42],
}

EXPECTED_QUANTIZATION = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
}

EXPECTED_ROPE_SCALING = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": ORIGINAL_CONTEXT,
    "type": "yarn",
}

EXPECTED_COMPRESS_RATIOS = [
    0,
    0,
    *([4, 128] * 20),
    4,
    0,
    0,
    0,
]


def source_identity() -> dict[str, Any]:
    return {
        "repository": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "weight_shards": WEIGHT_SHARDS,
        "model_family": MODEL_FAMILY,
        "model_id": MODEL_ID,
    }


def validate_context(context: int) -> int:
    if context <= 0:
        raise ValueError("context must be positive")
    if context > MAX_CONTEXT:
        raise ValueError(
            f"DeepSeek context {context} exceeds configured maximum {MAX_CONTEXT}"
        )
    return context


def validate_reasoning_effort(value: str | None) -> str | None:
    if value in (None, "low", "high", "max"):
        return value
    raise ValueError(f"unknown reasoning_effort={value!r}")


def sampling_profile(temperature: float, top_p: float) -> SamplingProfile:
    for profile in (DETERMINISTIC_SAMPLING, AGENT_SAMPLING):
        if temperature == profile.temperature and top_p == profile.top_p:
            return profile
    raise ValueError(
        "DeepSeek-V4 currently supports only temperature=0, top_p=1 "
        "or temperature=1, top_p=0.95"
    )


def _check_mapping(
    actual: Mapping[str, Any], expected: Mapping[str, Any], prefix: str
) -> list[str]:
    failures: list[str] = []
    for key, value in expected.items():
        if key not in actual:
            failures.append(f"missing {prefix}{key}")
        elif actual[key] != value:
            failures.append(
                f"{prefix}{key}: expected {value!r}, got {actual[key]!r}"
            )
    return failures


def validate_config(config: Mapping[str, Any]) -> list[str]:
    """Return all pinned-config contract failures without short-circuiting."""
    failures = _check_mapping(config, EXPECTED_CONFIG, "")
    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping):
        failures.append("missing quantization_config mapping")
    else:
        failures.extend(
            _check_mapping(quant, EXPECTED_QUANTIZATION, "quantization_config.")
        )
    rope = config.get("rope_scaling")
    if not isinstance(rope, Mapping):
        failures.append("missing rope_scaling mapping")
    else:
        failures.extend(_check_mapping(rope, EXPECTED_ROPE_SCALING, "rope_scaling."))
    if config.get("compress_ratios") != EXPECTED_COMPRESS_RATIOS:
        failures.append("compress_ratios do not match the pinned 43-layer schedule")
    return failures

