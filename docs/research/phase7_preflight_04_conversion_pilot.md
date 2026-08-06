# Phase 7 Preflight 4: Production Conversion Pilot

**Project:** colib
**Target:** Qwen3.5-397B-A17B-FP8
**Revision:** `ea5b4f81096f3901c91dea97f81324302495781d`
**Date:** 25 July 2026
**Status:** Complete; full resumable conversion authorized

## Abstract

The production converter was stopped after one official 4.3 GB source shard.
It applied real FP8 inverse scales, requantized 1,024 routed-expert matrices to
grouped int4-g128, atomically wrote a 2.282 GB output shard, computed its
SHA-256, durably updated resume state, and deleted the source only after those
steps succeeded.

The output reopens through safetensors and contains 3,072 physical tensors:
payload, fp32 scale, and qtype for every logical matrix. All 1,024 qtype tags
are grouped int4. The pilot satisfies the preregistered authorization
conditions for resuming shards 2–94.

## 1. Controls

```text
source revision: ea5b4f81096f3901c91dea97f81324302495781d
source shard: model.safetensors-00001-of-00094.safetensors
target: text only, no MTP
routed precision: symmetric int4-g128
shared precision: symmetric int8 per row
I/O precision: symmetric int8 per row
minimum free disk: 230 GiB
maximum new shards: 1
```

The existing Phase-4 durability order was retained:

```mermaid
flowchart LR
    D["Download one source shard"] --> F["Decode FP8 blocks"]
    F --> Q["Quantize target tensors"]
    Q --> T["Write temporary safetensors"]
    T --> A["Atomic rename"]
    A --> H["Hash and persist resume state"]
    H --> X["Delete source shard"]
```

## 2. Results

| Property | Result |
|---|---:|
| logical converted matrices | 1,024 |
| physical output tensors | 3,072 |
| output data bytes | 2,281,702,400 |
| output file bytes | 2,282,123,368 |
| output SHA-256 | `d98c8f57a9f3c5c9c50ad5f59de53a1a39c1c45be9a14cbd337efe9963010630` |
| qtype values | `{4}` |
| source retained after commit | no |
| resumable state | yes |

The shard contains routed experts from layer 43. A representative saved gate
matrix has:

```text
payload dtype U8, shape [1024, 2048]
scale dtype F32, shape [1024, 32]
qtype dtype U8, shape [1], value 4
```

The 2,048-byte payload row represents 4,096 logical columns at two int4
weights per byte. Thirty-two scale values correspond to 32 groups of 128
columns.

## 3. Resume evidence

`.conversion-state.json` records:

- immutable Hub source ID and index fingerprint;
- int4-g128, int8 I/O/shared, no-MTP signature;
- output filename, size, data bytes, tensor count, SHA-256;
- 1,024-entry logical inventory.

On resume, the converter verifies both output size and SHA-256 before skipping
the completed source shard. It emits one progress line per verification so a
100+ GiB recovery audit after an unclean host restart is distinguishable from
a stalled process, and another line before each potentially multi-gigabyte
source acquisition so network time is also observable. A changed revision or
precision signature is rejected.

## 4. Decision

Authorize the remaining 93 shards under the same 230 GiB disk guard. The
conversion remains incomplete until final loader-inventory validation,
aggregate index generation, and `quantization.json` are written.
