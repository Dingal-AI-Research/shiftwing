# Phase 8 Experiment 10: Lossless Warm Route Atlas

**Model:** `deepreinforce-ai/Ornith-1.0-397B-FP8`
**Converted revision:** `8b61f97a8512d9d01bff1a9625c9a16730e115bb`
**Date:** 2026-07-31
**Status:** rejected on measured capacity; experimental code removed

## Abstract

The best lossless Ornith397 pilot sustains 0.867369 token/s. Its measured
16-token decode performs 2,394 disk misses and reads 16.003 GB, even though a
single preceding warm turn follows the same deterministic prompt and token
path. This experiment tests whether the prior turn's exact decode working set
can be partitioned across the existing 18 GiB RAM and 6 GiB VRAM expert tiers
without changing weight precision, model arithmetic, or the accepted
container.

The pre-experiment `HITS` bitmap reported 2,934 unique layer/expert pairs.
The tier snapshot classified 2,051 experts as RAM-only and 963 as VRAM,
suggesting that the working set might fit if duplicates were removed. This
became the testable capacity hypothesis. Instrumentation inside the actual
decode route path disproved it: the bitmap omitted resident-batch routes, and
the complete working set contains 4,779 pairs.

## 1. Research question

Can an explicit cross-tier atlas eliminate disk reads during a deterministic
warm decode and thereby close the frozen 2-token/s Ornith397 requirement?

## 2. Terms

**Route** means the router's selection of one expert for one model layer and
token. Ornith selects multiple experts per layer.

**Working set** means the unique layer/expert pairs used during one decode.
Repeated use of the same expert counts once for capacity and multiple times
for computation.

**Atlas** means a temporary map learned from the immediately preceding
decode. It records frequency per layer, assigns enough entries to VRAM that
the remainder fits in that layer's RAM cache, and preloads both tiers after
the next prompt prefill but before timed decode.

**Lossless** means every packed int4 value and scale remains byte-identical to
the accepted container. The atlas changes only residency and timing.

## 3. Prior evidence

The bounded non-lossy control used persistent `io_uring`, pinned upload,
decode protection, one warm pass, and one measured 16-token pass:

| Quantity | Result |
|---|---:|
| sustained decode | 0.867369 token/s |
| decode disk time | 11.386176 s |
| decode expert compute | 6.876265 s |
| decode expert misses | 2,394 |
| decode expert reads | 16,003,111,950 bytes |
| telemetry-bitmap unique experts | 2,934 |
| physical host slots | 2,880 (48 per layer) |
| device slots | 963 |

A read-only microbenchmark of the real tensor loader reached 2.079 GiB/s
with two to four `io_uring` workers. Eight workers fell to 1.233 GiB/s. Worker
count is therefore not the next variable; the experiment targets avoidable
misses.

## 4. Method

The implementation is opt-in through `DECODE_ATLAS=1` and a matching
qualifier argument. During decode it records a saturating per-layer frequency
for every routed expert. On the next request, after prefill:

1. rank the prior turn's experts within each layer;
2. reserve the layer's RAM capacity for the most frequently reused entries;
3. assign the remaining prior-turn entries to the global VRAM tier, provided
   the exact packed byte budget permits them;
4. preload VRAM entries first, detach their temporary host owners, then fill
   the RAM cache;
5. clear the turn-local counts and begin the ordinary timed decode.

The accepted replacement paths remain available when the new option is off.
The atlas may increase time to first token because preloading occurs between
prefill and decode; that cost must be reported and is not hidden from wall or
TTFT telemetry.

## 5. Preregistered controls and decision rule

Before the real pilot:

1. CPU and CUDA tiny int4 greedy and teacher-forced references must remain
   32/32 with the option both off and on;
2. the existing C and Python suites covering cache eviction, quantized expert
   loading, and qualifier configuration must remain green;
3. the real pilot must use the same prompt, one warm pass, one measured pass,
   16 tokens, 4,096 context, 18 GiB RAM experts, 6 GiB VRAM experts, 1 GiB
   headroom in each tier, persistent rings, pinned upload, and decode
   protection as the 0.867369 control;
4. generated token IDs and text must be byte-identical to that control;
5. resident telemetry must report CUDA active, device MoE on every layer, and
   zero host-MoE fallback;
6. atlas telemetry must report a complete preload within both byte budgets;
7. the measured decode must record zero expert disk misses and zero expert
   read bytes; and
8. sustained measured throughput must be at least 2.0 token/s.

Any correctness, capacity, residency, zero-miss, or throughput failure rejects
the implementation and prohibits using its result to close Gate 8. Passing
the one-prompt pilot authorizes the already frozen four-prompt qualification,
not Gate 8 by itself. The full run must still be coherent, finite, CUDA
resident, tool-correct, and at least 2 token/s.

## 6. Scope limitation

This is a warm-route optimization. A new prompt can select a different expert
set, so the atlas does not promise zero-miss cold TTFT or first-turn decode.
Its production value is repeated workloads, session extensions, and the
preregistered warm qualification. Cold behavior remains the accepted tiered
fallback and must continue to work unchanged.

## 7. Results

### 7.1 Correctness controls

The opt-in implementation built cleanly. CPU and CUDA tiny mux controls
generated byte-identical repeated outputs, and the focused server, qualifier,
and gateway tests passed. Both real pilots generated exactly the same
16-token text as the non-atlas control:

```text
A **cache miss** occurs when a processor requests data from the cache, but
```

The experiment therefore found no numerical or protocol regression.

### 7.2 Capacity discovery

The first real run rejected activation at its combined low-bit/capacity
guard. A reduced-cache load-only control independently proved that the direct
Ornith397 container contains 0 int2 and 0 int3 matrices. A second run split
the guard and reported the exact reason before attempting any preload:

```text
entries=4779 needed-vram=1901 slots=963
expert-bytes=6684672 budget=6442450944
```

The 60 host caches provide 2,880 physical slots, while the 6 GiB device tier
provides 963. Per-layer imbalance leaves two host slots unused for this route
set, so 1,901 entries would have to reside on the device. Only 963 fit. The
shortfall is 938 expert slots, approximately 5.84 GiB. The complete
4,779-entry working set occupies approximately 29.75 GiB.

The earlier 2,934 value came from `HITS`, which is updated by the ordinary
cache-touch helper but not by every expert selected in the resident batched
route. It is valid as a partial activity visualization, not as a capacity
inventory. The atlas-local route counter was independent and therefore
exposed the omission before any result could be accepted.

### 7.3 Performance

Neither real run activated the atlas, so neither is an atlas performance
result. They are retained as failed controls:

| Run | Decode rate | Decode misses | Decode reads | Decision |
|---|---:|---:|---:|---|
| guard v1 | 0.938357 tok/s | 2,392 | 15,989,742,600 bytes | guard fail |
| capacity v2 | 0.610446 tok/s | 2,393 | 15,996,427,275 bytes | exact capacity fail |

Both are below 2 token/s and retain roughly the same miss volume as the
non-atlas control. The lower v2 rate coincided with slower startup, prefill,
and expert compute, so it does not establish an additional algorithmic
regression; the branch never entered its preload path.

## 8. Decision

The zero-miss and throughput criteria fail because the accepted cache budgets
cannot contain the true deterministic working set. No threshold is changed,
and no full four-prompt qualification is authorized. All atlas runtime,
protocol, qualifier, and test code was removed after the negative result; the
accepted replacement path remains unchanged.

The measured 5.84 GiB shortfall motivates a different lossless question:
whether host copies of CUDA-resident dense matrices can be released after a
successful device preload and their memory reassigned to experts. That is a
new experiment with separate fallback and resource-safety controls, not a
retry of this rejected atlas.
