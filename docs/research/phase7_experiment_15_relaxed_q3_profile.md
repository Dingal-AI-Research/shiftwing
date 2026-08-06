# Phase 7 Experiment 15: User-Authorized Relaxed 3-Bit Profile

**Target:** Qwen3.5-397B-A17B on the 32 GiB reference workstation
**Date:** 2026-07-28
**Status:** permanently retired as an intermediate-model experiment after the
conversion pilot; not part of any active or future gate

## Abstract

Experiment 14 rejected grouped 3-bit routed experts under the original
strict numerical rule. The result was nevertheless close: 35B agreement
fell from 96.17% to 95.23% overall, and the weakest prompt fell from 87.50%
to 82.81%. After reviewing these measurements, the project owner explicitly
accepted this limited weakening because the expected cache-capacity and
storage-bandwidth gains may be more valuable for local 397B use.

This report does not rewrite the original gate. It defines a second,
clearly-labeled relaxed profile before any 397B 3-bit conversion or
throughput measurement. The profile must still pass structural, perplexity,
coherence, memory-safety, and sustained-speed checks.

## 1. Rationale

The strict profile protects against localized regressions by requiring at
least 85% teacher-forced agreement on every 35B prompt. Three-bit failed only
that condition:

| Profile | Aggregate | Weakest frozen prompt |
|---|---:|---:|
| accepted 4-bit | 96.17% | 87.50% |
| experimental 3-bit | 95.23% | 82.81% |
| difference | -0.94 points | -4.69 points |

These numbers measure next-token fidelity, not a direct percentage of
reasoning or coding ability. The user accepts the observed numerical risk;
the runtime must expose the format as experimental rather than silently
replacing 4-bit.

## 2. Frozen relaxed acceptance conditions

The 397B 3-bit profile will be accepted for local experimental use only if:

1. the complete sidecar manifest matches the accepted 397B source identity;
2. every routed projection is present with valid 3-bit/group-128 geometry;
3. fixed-corpus perplexity is finite and no more than 5% worse than the
   accepted 397B 4-bit baseline;
4. all four frozen chat prompts produce coherent, non-degenerate output and
   terminate within their token budgets;
5. the frozen RAM/VRAM guards pass without OOM;
6. the 64-token sustained qualifier reaches at least 2.0 tok/s after the
   specified warm-up;
7. telemetry proves that the 3-bit host/storage tensors were selected and
   that CUDA expert execution remained active.

The original 35B strict failure remains visible in every release note. A
speed result does not erase it.

## 3. Expected resource effect

For the 397B expert geometry:

| Format | Bytes per routed expert |
|---|---:|
| grouped 4-bit | 6,684,672 |
| grouped 3-bit | 5,111,808 |

Three-bit therefore uses 76.47% of the host/storage bytes. Under an 18 GiB
host tier, ideal per-layer capacity rises from approximately 48 to 63
experts. The GPU expands selected values into the qualified 4-bit compute
layout, so its capacity and arithmetic kernel are not reduced by the same
ratio. The combined ideal host/device working set is expected to approach
81 experts per layer. Whether fewer read bytes and higher host coverage are
enough for 2 tok/s is an empirical question.

## 4. Method

The existing accepted 397B container remains immutable. A resumable converter
will read one routed 4-bit matrix at a time and write `.q3`, `.qs`, and
`.qtype` sidecars with atomic shard replacement and SHA-256 records. A pilot
conversion will first measure per-expert time and projected total duration.
The complete job will start only after disk headroom and the estimate are
recorded.

## 5. Decision boundary

This profile is an opt-in engineering trade-off authorized by the project
owner. It may become the recommended profile for this 32 GiB machine if all
conditions above pass. It is not equivalent to the strict 4-bit reference,
and failure of perplexity, coherence, safety, or sustained throughput will
reject it even though the 35B weakening was accepted.

## 6. Conversion pilot and pause checkpoint

A one-expert serial pilot processed all three 397B routed projections in
2.20 seconds with 483,116 KiB maximum resident memory. Direct extrapolation
was 18.8 hours, so the full job was not started with that configuration.

The converter was extended to quantize independent projections concurrently.
An eight-expert representative chunk produced:

| Configuration | Eight experts | CPU utilization | Peak resident memory | Full projection |
|---|---:|---:|---:|---:|
| 4 workers × 2 Torch threads | 4.24 s | 405% | 1,013,388 KiB | about 4.5 h |
| 8 workers × 1 Torch thread | **4.06 s** | 419% | 1,514,096 KiB | about **4.3 h** |

All nine tensors for the first expert are byte-identical between serial and
concurrent conversion. The two concurrent pilot shards also have identical
SHA-256 hashes.

The full job was started with 8 workers, one Torch thread, and 64 experts per
atomic file, then paused at the user's request. It left two complete files:

```text
expert-q3-l000-e0000-0063.safetensors  327,234,632 bytes
expert-q3-l000-e0064-0127.safetensors  327,234,976 bytes
```

No converter process remains and no `.partial` file remains. The current
script writes its manifest only at final completion, so these two valid files
are not yet indexed. Before resuming, add per-file manifest checkpoints plus
validation/import of unindexed complete chunks; do not recompute or trust
them merely by filename.

## 7. Product-target disposition

After the pause, the project owner revised the roadmap: Ornith-1.0 is the
intended production model, so spending approximately 4.3 hours completing a
second Qwen397 representation does not provide enough additional product
evidence. The Qwen397 3-bit conversion is therefore retired rather than
resumed.

The later roadmap decision makes this retirement permanent: Gate 8 closes on
the direct Ornith397 int4-g128/int8 container, with no Qwen397 3-bit artifact,
benchmark, or retry. Any future Ornith lower-bit work would require a new,
separately authorized experiment and would not reopen this Qwen study.

This changes sequencing, not the experimental findings:

1. Qwen397's accepted 4-bit profile remains the architecture, correctness,
   and capacity reference.
2. The two complete 3-bit files remain unindexed research artifacts. They do
   not form a complete runtime and are not selected by the loader.
3. The deterministic quantizer, exact CUDA expansion, concurrency pilot, and
   measured quality trade-off remain reusable.
4. Ornith35 will first pass its official 4-bit numerical and tool-use gates.
5. If Ornith397 repeats the same 32 GiB capacity boundary, grouped 3-bit will
   be quality-gated on Ornith35 and generated only for Ornith397.

This direct-to-end-model sequence avoids approximately 157 GB of redundant
Qwen sidecars and their conversion time while preserving the evidence needed
to evaluate the same treatment on Ornith.
