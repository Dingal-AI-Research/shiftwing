# Phase 7 Preflight 3: Real FP8 Tensor Probe

**Project:** colib
**Target:** Qwen3.5-397B-A17B-FP8
**Revision:** `ea5b4f81096f3901c91dea97f81324302495781d`
**Date:** 25 July 2026
**Status:** Complete; FP8 arithmetic prerequisite passed

## Abstract

One official 2.67 GB shard was downloaded after the metadata and resource
preflights passed. A real shared-expert gate matrix was decoded with colib's
new block-FP8 primitive and independently with Transformers 5.14.1. All
4,194,304 decoded fp32 elements are bit-identical. Re-quantization to colib's
int8 shared-expert format has root-mean-square error \(5.98\times10^{-5}\) and
maximum absolute error \(6.42\times10^{-4}\).

This validates actual safetensors dtype handling, BF16 inverse scales,
128×128 scale orientation, and the selected runtime precision. A one-shard
production conversion pilot may proceed.

## 1. Method

The probe used:

```text
model.safetensors-00094-of-00094.safetensors
model.language_model.layers.0.mlp.shared_expert.gate_proj.weight
```

The source tensors are:

| Tensor | Dtype | Shape |
|---|---|---:|
| weight | E4M3FN | 1,024 × 4,096 |
| `weight_scale_inv` | BF16 | 8 × 32 |

Colib expands each scale over its corresponding 128×128 tile and multiplies
after promoting both operands to fp32. The independent reference is
Transformers 5.14.1 `Fp8Dequantize._dequantize_one`, requested with fp32
output. `c/tools/probe_fp8_shard.py` reports elementwise equality and target
quantization error.

## 2. Results

| Metric | Result |
|---|---:|
| decoded elements | 4,194,304 |
| Transformers maximum difference | 0 |
| exact tensor equality | yes |
| decoded SHA-256 | `dd7879bd7349cf5cc990e1f1bd16e7838c6f8e598182c78c531f0e3ec831d847` |
| target format | int8 per row |
| target payload | 4,198,401 bytes |
| int8 maximum absolute error | 0.0006420100 |
| int8 RMSE | 0.0000597738 |

## 3. Supporting evidence

The metadata-only converter dry-run additionally validates:

- 94,078 weight/inverse-scale pairs;
- zero orphan scales;
- zero cross-shard pairs;
- zero block-grid shape mismatches;
- 212,634,789,241 predicted output bytes without MTP.

Raw-bit unit tests cover zero, subnormal, negative, maximum finite, NaN
rejection, scale-grid orientation, and partial edge blocks. The real official
matrix dimensions are exactly divisible by 128.

## 4. Limitations

- This probe covers one shared-expert matrix, not all 94 shards.
- Routed experts use the same E4M3FN/block-scale decoding but target grouped
  int4 rather than int8; the production pilot must exercise that path.
- Equality is against the pinned Transformers implementation and does not by
  itself establish final-model quality after int4 quantization.

## 5. Decision

Retain the FP8 decoder and authorize a one-shard resumable production
conversion pilot. Do not authorize the remaining 93 shards until the first
output shard is atomically written, hashed, recorded in conversion state, and
can be re-opened by safetensors.
