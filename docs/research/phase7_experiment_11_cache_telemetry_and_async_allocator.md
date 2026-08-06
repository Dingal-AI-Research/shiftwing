# Phase 7 Experiment 11: Cache Telemetry and CUDA Allocation

**Target:** Qwen3.5-397B-A17B tiered decode
**Hardware:** Ryzen 7 7700X, RTX 5070 Ti, NVMe, 29.4 GiB RAM
**Date:** 2026-07-28
**Status:** allocator treatment accepted; Gate 7 still open

## Abstract

Experiment 10 proved that the complete model worked but did not explain its
low throughput accurately. This experiment added disjoint tier accounting and
per-request expert-cache counters, made learned expert maps survive graceful
server shutdown, rejected an ineffective cache-layout treatment, and tested a
CUDA stream-ordered allocator with a controlled same-atlas A/B comparison.

The corrected bounded workload had 1,157 host-cache hits, 517 GPU-cache hits,
726 storage misses, and 4,853,074,050 bytes read during four measured decode
tokens. Thus 69.75% of routed expert requests avoided storage, while each
token still required approximately 1.213 GB of reads. Replacing synchronous
`cudaMalloc`/`cudaFree` calls with CUDA's stream-ordered memory pool preserved
the exact cache requests and generated text, improved decode from 0.3363 to
0.7468 tokens per second, and reduced expert matmul/upload time from 40.05 to
10.72 seconds. The 2.22-fold gain is substantial but still below the 2.0
tokens-per-second gate.

## 1. Definitions

- **RAM** is the computer's main memory. It is larger than GPU memory here but
  is accessed by the CPU.
- **VRAM** is memory attached to the GPU. GPU kernels can use it directly.
- **NVMe** is the solid-state storage device. It holds the complete model but
  is much slower than RAM for repeated expert access.
- A **tier** is one of those storage levels. The fastest available copy of an
  expert determines whether it is classified as VRAM, RAM, or disk.
- **Direct I/O** transfers aligned data without filling the operating
  system's page cache. It makes the engine's explicit cache easier to reason
  about.
- A CUDA **allocator** obtains and releases GPU-memory regions. Synchronous
  allocation can stop the GPU stream and wait for outstanding work.
  Stream-ordered allocation reuses a memory pool while respecting the order
  of operations in that CUDA stream.
- An **expert atlas** is the persisted table of routes observed in earlier
  requests. It allows later engine processes to begin with learned route
  information.

## 2. Research questions

1. How many decode-time expert requests are served from RAM, VRAM, and NVMe?
2. Is GPU allocation synchronization a significant cost when experts are
   repeatedly inserted into and evicted from the small VRAM cache?
3. Can a treatment improve throughput without changing the route sequence,
   expert-cache outcomes, quantization, or generated text?

## 3. Measurement corrections

The engine now emits two request records:

- `CACHE` counts prompt prefill plus decode;
- `DCACHE` begins immediately after prefill and counts decode only.

Each record contains CPU hits, GPU hits, misses, total read bytes, and direct
read bytes. `DONE.cache_hit_percent` is computed from the decode interval so
it describes the same work as decode tokens per second.

Tier-map construction now queries the device cache independently of the host
cache slot. This includes experts that still exist in VRAM after their host
owner has been detached. Tier counts are disjoint: every one of the 60 × 512
= 30,720 experts belongs to its fastest present tier. Physical host-cache
bytes are still reported separately because a GPU expert may also occupy
RAM.

Finally, the Python engine closes the mux input first and waits for normal
exit. The C process can therefore run its `atexit` handler and atomically save
`expert_map.bin`. Forced termination remains a bounded fallback.

## 4. Controlled allocator method

A single short prompt, “Briefly explain a cache miss.”, was used with one warm
pass and one four-token measured pass. Context was 512; host and device expert
caches remained 18 and 6 GiB. Before each allocator arm, the same saved expert
atlas was restored byte for byte. No GPU compute process or competing CPU job
was present, and approximately 28 GiB of system RAM was available.

The control used synchronous `cudaMalloc` and `cudaFree`. The treatment used
`cudaMallocAsync` and `cudaFreeAsync` on the inference stream, backed by the
device's default memory pool with a retained release threshold. The fallback
is selected with:

```sh
CUDA_ASYNC_ALLOC=0
```

The treatment is accepted only if output and cache/read counters remain
identical. This constraint isolates allocation behavior from model routing
and storage locality.

## 5. Results

### 5.1 Corrected tier state

| Tier | Experts | Fraction of 30,720 |
|---|---:|---:|
| VRAM | 963 | 3.13% |
| RAM, not in VRAM | 1,991 | 6.48% |
| disk only | 27,766 | 90.38% |

The disjoint warm union is 2,954 experts, or 9.62% of the full expert table.
The physical host cache is larger than 1,991 entries because many VRAM
experts also retain a host copy.

### 5.2 Decode-only cache behavior

Both allocator arms produced exactly:

| Counter | Value |
|---|---:|
| host hits | 1,157 |
| GPU hits | 517 |
| storage misses | 726 |
| total routed requests | 2,400 |
| non-disk hit rate | 69.75% |
| bytes read | 4,853,074,050 |
| direct-I/O bytes | 4,853,071,872 |
| bytes read per generated token | 1.213 GB |

The equality of these values is the most important control in the experiment:
the allocator did not receive an easier route sequence or a warmer cache.

### 5.3 Performance

| Measurement | Synchronous | Stream-ordered | Change |
|---|---:|---:|---:|
| measured decode | 0.3363 tok/s | 0.7468 tok/s | 2.22× |
| measured wall time | 89.26 s | 54.09 s | −39.4% |
| expert matmul/upload | 40.05 s | 10.72 s | −73.2% |
| expert storage phase | 27.88 s | 22.29 s | variable |
| attention phase | 20.89 s | 20.91 s | unchanged |

The generated prefix was identical: `A **cache miss`. CUDA upload, eviction,
and cache-hit totals over the complete bounded run were also equal.

Artifacts:

- `c/qwen397_bounded_sync_atlas.json`
- `c/qwen397_bounded_async_atlas.json`
- `c/qwen397_bounded_exclusive.json`
- `c/qwen397_bounded_control.json`

### 5.4 Rejected host-exclusive policy

An intermediate treatment preferentially evicted host copies of experts
already present in VRAM. It increased the distinct RAM-plus-VRAM union by only
39 experts and slowed measured decode to 0.2655 tok/s. It was reverted.

This negative result matters because “remove duplicate copies” sounds
obviously beneficial but changes replacement pressure. A VRAM entry may be
evicted shortly afterward; retaining its hot host copy can then prevent a
much more expensive disk read.

## 6. Interpretation

The allocator A/B shows that repeated globally synchronizing GPU allocations
were a major implementation cost. Stream-ordered reuse removes much of that
cost and is now the default when the device reports memory-pool support. Both
forced modes and the default pass the independent CUDA kernel suite.

The remaining gap is primarily a data-movement problem. At the observed
69.75% hit rate, four tokens require 4.853 GB of storage reads. If bandwidth
and other compute costs remain similar, reaching 2 tok/s would require a
non-disk hit rate near 86%, or a combination of higher locality, higher
effective storage bandwidth, and lower per-token compute cost.

The attention phase is almost identical between arms, which supports the
causal claim that the speedup came from expert allocation/upload rather than
an unrelated attention change. Storage time varies because the operating
system and drive are not a perfectly deterministic laboratory instrument, so
the exact 2.22× factor should not be generalized to every machine.

## 7. Conclusion and next experiment

Stream-ordered CUDA allocation is an accepted optimization with a tested
synchronous fallback. It more than doubles the bounded decode rate without
changing model output or cache behavior. Gate 7 remains open because 0.7468
tok/s is still below 2.0 tok/s.

The next experiment should expand the effective expert-cache union or reduce
bytes read per token while preserving the frozen precision map. Every
treatment must report decode-only hits, misses, and bytes, so an apparent
speedup cannot be mistaken for a different cache state.
