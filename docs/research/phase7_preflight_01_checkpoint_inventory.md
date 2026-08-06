# Phase 7 Preflight 1: Qwen3.5-397B FP8 Checkpoint Inventory

**Project:** colib
**Target:** Qwen3.5-397B-A17B
**Date:** 25 July 2026
**Status:** Read-only inventory complete; conversion implementation not started

## Abstract

This preflight inspects the official 397B FP8 repository before any weight
shard is downloaded. Revision
`ea5b4f81096f3901c91dea97f81324302495781d` contains 94 safetensors shards and
406,125,181,280 indexed weight bytes. Its quantization configuration specifies
fine-grained FP8 weights with 128×128 blocks and dynamic activation scaling.
The index contains 189,042 logical tensors, including 94,078
`weight_scale_inv` tensors.

The existing 35B converter has no FP8 or `weight_scale_inv` implementation.
All 94 shards contain text tensors, and two also contain vision tensors, so a
text-only conversion cannot avoid downloading any shard. The safe next step is
therefore a small deterministic FP8-block dequantization fixture and converter
dry-run extension, not a 406 GB download.

## 1. Source and immutable revision

The selected source is the official
[Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)
repository. The model card describes fine-grained FP8 quantization with block
size 128. The repository file view reports approximately 406 GB across 94
safetensors shards.

| Property | Observed value |
|---|---:|
| immutable revision | `ea5b4f81096f3901c91dea97f81324302495781d` |
| repository files | 107 |
| safetensors shards | 94 |
| index size | 23,926,385 bytes |
| indexed model size | 406,125,181,280 bytes |
| logical tensors | 189,042 |
| inverse-scale tensors | 94,078 |

Only `config.json`, repository metadata, and
`model.safetensors.index.json` were fetched. No weight shard was downloaded.

## 2. Architecture confirmation

The text configuration confirms:

| Field | Value |
|---|---:|
| hidden width | 4,096 |
| decoder layers | 60 |
| routed experts per layer | 512 |
| selected experts per token | 10 |
| routed expert intermediate width | 1,024 |
| shared expert intermediate width | 1,024 |
| MTP layers | 1 |

These values match the architecture recorded in `PLAN.md`.

## 3. FP8 storage contract

The repository quantization configuration is:

```json
{
  "quant_method": "fp8",
  "activation_scheme": "dynamic",
  "weight_per_tensor": false,
  "act_per_tensor": false,
  "weight_block_size": [128, 128]
}
```

Each converted matrix must pair its FP8 weight tensor with the corresponding
`weight_scale_inv` tensor. The conversion operation is conceptually:

\[
W_{ij}^{fp32} =
\operatorname{decode}_{E4M3}(Q_{ij})\,
S_{\lfloor i/128\rfloor,\lfloor j/128\rfloor}.
\]

The exact scale orientation, edge-block behavior, special-value policy, and
rounding path must be established with a fixture from the official
Transformers/quantizer implementation before full conversion.

## 4. Shard and vision analysis

The weight map assigns:

- 92 shards exclusively to text/MTP tensors;
- 2 shards to a mixture of text and vision tensors;
- 0 shards exclusively to vision.

Consequently, dropping the vision tower reduces the converted output but not
the number of source shards that must be fetched. The Phase-4
download→convert→durably-write→delete loop remains necessary for all 94
shards.

## 5. Converter gap

`c/tools/convert_qwen.py` currently supports bf16/fp32 source tensors and has
no references to FP8, float8 decoding, or `weight_scale_inv`. Starting the full
download now would therefore create approximately 406 GB of unusable staging
data.

Required implementation order:

1. implement a reusable E4M3 + 128×128 inverse-scale dequantizer;
2. test full blocks, partial edge blocks, zeros, maximum finite values, NaN,
   and infinity policy against an independent reference;
3. teach the dry-run inventory to pair weights and inverse scales and predict
   text-only output bytes;
4. verify one small real tensor pair from a single shard;
5. only then start the resumable 94-shard conversion.

## 6. Preliminary resource bounds

Routed expert parameters alone are:

\[
60 \times 512 \times 3 \times 4096 \times 1024
= 386{,}547{,}056{,}640.
\]

Grouped int4-g128 uses one half-byte weight plus one fp32 scale per 128
weights, or approximately 0.53125 byte per parameter. Routed experts therefore
require about 205.35 GB decimal before container metadata. This supports the
roadmap's approximately 205 GB expert-store estimate.

Per inference slot, the 45 GDN layers require at least:

\[
45 \times 64 \times 128 \times 128 \times 4
= 188{,}743{,}680\ \text{bytes}
\]

for recurrent matrices, plus approximately 8.85 MB for convolution tails.
Full-attention KV storage grows by 61,440 bytes per token per slot at fp32.
Four 4,096-token slots therefore need approximately 1.01 GB of KV storage in
addition to about 790 MB of fixed recurrent/conv state.

These are lower bounds; the Phase-7 resource planner must also account for
activations, allocator headroom, CUDA context memory, dense/shared weights,
RAM expert cache, and converter scratch space.

## 7. Decision

Retain the official FP8 repository and pinned revision as the Phase-7 source.
Do not download weight shards yet. The next experiment must implement and
validate the FP8 block-dequantization primitive and add a resource planner with
hard disk/RAM/VRAM guards.
