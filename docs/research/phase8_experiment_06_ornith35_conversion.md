# Phase 8 Experiment 6: Ornith-35B Streaming Conversion

**Model:** `deepreinforce-ai/Ornith-1.0-35B-FP8`
**Immutable revision:** `1ab57ce0b44950e498a88756f40ad1ed4d0f30ca`
**Date:** 2026-07-28
**Status:** conversion and structural audit complete; numerical and generated
tool-use gates remain open

## Abstract

This experiment converted the official Ornith-1.0-35B FP8 checkpoint into
colib's text-only mixed-precision container. Sixteen source shards were
downloaded and processed sequentially. Each source shard was decoded using
its per-output-channel FP8 scale, converted to grouped 4-bit routed experts
and 8-bit protected paths, written atomically, hashed, and released before
the next shard was acquired.

The completed container has 93,277 physical tensors, 31,333 logical model
tensors, and exactly 19,081,810,684 tensor payload bytes. This equals the
metadata-only prediction made before downloading weights. An independent
container doctor validates all 16 output shards, the converter manifest,
tokenizer, CUDA-linked engine, and a one-slot 4,096-token resource plan.
This proves structural completeness; it does not yet prove numerical
agreement or generated tool use.

## 1. Research question

Can the official Ornith compressed-tensors FP8 checkpoint be converted
completely and reproducibly by the Qwen-compatible runtime pipeline, without
holding the entire 37.7 GB source checkpoint in staging at once?

## 2. Method

The source identity was fixed before conversion:

```text
hf://deepreinforce-ai/Ornith-1.0-35B-FP8@1ab57ce0b44950e498a88756f40ad1ed4d0f30ca
```

Its checkpoint-index fingerprint is:

```text
15281dc0352464f68ec93e805283a7c070aeb06f22622d975648381009cfe1d6
```

The converter used:

- grouped int4, group size 128, for routed experts;
- int8 for dense input/output and shared-expert matrices;
- no MTP tensors, consistent with the current target-only qualification;
- a 100 GiB minimum-free-space guard;
- atomic output files and a resumable hash ledger.

The source is a compressed-tensors checkpoint. For a matrix row \(W_q\) and
its channel scale \(S\), the converter first reconstructs:

\[
W = W_q S
\]

and then applies the same colib quantizers already tested on Qwen. Vision
tensors are omitted because this project is text-only.

## 3. Streaming behavior

Only one source shard was acquired at a time. After conversion, the source
shard was released while its committed output remained. At the midpoint,
eight committed outputs occupied 9.5 GB while the visible staging directory
was only 33 MB. This demonstrates the intended bounded staging behavior.

The unauthenticated Hugging Face transfer varied between shards and was the
dominant elapsed-time cost. Local conversion typically followed each
download in tens of seconds. No source shard, tensor-scale pair, or atomic
write failed.

## 4. Results

| Measurement | Result |
|---|---:|
| source shards converted | 16 / 16 |
| output shards | 16 |
| physical tensors | 93,277 |
| logical model tensors | 31,333 |
| tensor payload | 19,081,810,684 bytes |
| preflight prediction | 19,081,810,684 bytes |
| prediction error | 0 bytes |
| converted precision | routed int4-g128; protected int8 |
| MTP included | no |

The converter manifest records `complete=true`, the immutable source
revision, fingerprint, precision map, and exact counts. The generated config
also records:

```text
colib_model_family = ornith-1.0
colib_source_repo = deepreinforce-ai/Ornith-1.0-35B-FP8
colib_source_revision = 1ab57ce0b44950e498a88756f40ad1ed4d0f30ca
```

## 5. Independent structural audit

`colib doctor` reports:

- valid model directory, configuration, and tokenizer;
- 93,277 tensors across 16 shards;
- 17.77 GiB independently parsed tensor payload;
- a completion manifest matching the container;
- a CUDA-linked engine and visible RTX 5070 Ti;
- 0.22 GiB state for one slot at 4,096-token context;
- safe host RAM, free VRAM, and disk resources under an 18 GiB host expert
  cache, 8 GiB GPU expert cache, and 1 GiB runtime headroom.

## 6. Limitations

Structural agreement cannot detect every numerical error. A wrong FP8 scale
orientation could still produce the expected number of tensors and bytes.
The official Q4_K_M reference comparison, fixed-corpus perplexity, coherent
generation, termination behavior, and live generated tool round trip remain
required.

MTP was intentionally excluded. The experiment also does not measure
throughput because network transfer and conversion are not inference
workloads.

## 7. Decision

Accept the Ornith-35B container as structurally complete and retain it for
Gate-8 numerical qualification. Do not claim Gate 8 passed until the pinned
external GGUF, perplexity, and generated HTTP tool-use checks pass.

Artifacts:

- `c/ornith35/quantization.json`
- `c/ornith35/model.safetensors.index.json`
- `c/ornith35/config.json`
