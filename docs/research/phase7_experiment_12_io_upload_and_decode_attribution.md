# Phase 7 Experiment 12: I/O, Upload Staging, and Decode Attribution

**Target:** Qwen3.5-397B-A17B tiered decode
**Hardware:** Ryzen 7 7700X, RTX 5070 Ti, NVMe, 29.4 GiB RAM
**Date:** 2026-07-28
**Status:** two treatments rejected as defaults; bottleneck attributed

## Abstract

Experiment 11 showed that stream-ordered CUDA allocation more than doubled
bounded decode speed, but its inclusive request profile mixed prompt prefill
with generation. This experiment tested two further data-movement treatments
and added a decode-only phase record.

Reusing one `io_uring` instance eliminated 6,228 repeated ring constructions
but reduced the controlled measured rate from 0.8104 to 0.4995 tokens per
second. A bounded pinned-host upload arena reduced the inclusive expert
upload/matmul phase from 14.48 to 8.68 seconds, but its measured decode rate
was 0.7691 tokens per second rather than the control's 0.8104. Neither
treatment is enabled by default.

The new decode-only profile resolves the important causal question. Four
tokens took 5.2509 seconds inside decode: 3.7580 seconds (71.6%) reading
experts from storage, 1.4506 seconds (27.6%) in expert upload and computation,
and only 0.0407 seconds (0.8%) in hybrid attention plus the language-model
head. At the same compute cost and effective storage rate, the 2 tok/s gate
requires reducing 4.853 GB of reads to about 0.66 GB, corresponding to
approximately a 96% non-disk expert-hit rate. The next useful experiment is
therefore cache-locality policy, not another attention kernel.

## 1. Definitions

- **Prompt prefill** processes all input tokens and constructs the model
  state needed to begin generation.
- **Decode** produces tokens one at a time after prefill. The Gate-7
  throughput requirement applies to this interval.
- **`io_uring`** is a Linux interface for submitting several storage
  operations as one asynchronous batch.
- A **pinned host buffer** is RAM that cannot be paged out. CUDA can transfer
  from it without first making an internal pinned copy.
- **Upload staging** copies pageable expert weights into a bounded pinned
  arena before sending them to VRAM.
- **Phase attribution** assigns elapsed time to storage, expert
  upload/computation, hybrid attention, and the final vocabulary projection.
  It does not by itself prove that phases can be removed independently.

## 2. Research questions

1. Does retaining `io_uring` state across expert batches improve real
   complete-model throughput?
2. Does one reusable pinned upload arena reduce the cost of repeatedly
   transferring experts to the GPU?
3. Which phase actually limits decode after the accepted asynchronous CUDA
   allocator change?

## 3. Controlled method

All complete-model arms used the same short prompt, one warm pass, one
four-token measured pass, context 512, 18 GiB of host expert cache, and 6 GiB
of device expert cache. Before each arm, the same expert atlas was restored:

```text
SHA-256 3ceb220c640bcfd1efcc58c7bc99fe44316e8ebf61b754b47d89464e9501b216
```

Each treatment had to preserve the generated prefix and the exact decode
cache counters: 1,157 host hits, 517 device hits, 726 misses, and
4,853,074,050 bytes read. This controls routing and locality even when the
wall clock varies.

The persistent-ring arm retained a thread-local ring and aligned direct-I/O
buffer. The pinned-upload arm deduplicated the layer's selected matrices,
reserved at most one layer's unique routed-expert payload, copied the
pageable quantized values and scales into that arena, and then enqueued host
to device transfers.

Finally, `DPERF` baselines were captured immediately after prefill. They use
the same phase fields as `PERF`, but every counter and timer covers decode
only. `PERF` remains the inclusive request record for backward compatibility.

## 4. Results

### 4.1 Persistent `io_uring`

| Measurement | Per-batch control | Persistent ring |
|---|---:|---:|
| ring constructions | 6,231 | 3 |
| ring reuses | 0 | 6,228 |
| measured decode | 0.8104 tok/s | 0.4995 tok/s |
| measured request wall | 55.45 s | 58.67 s |
| inclusive expert storage phase | 19.98 s | 24.89 s |

The model output and cache/read counters were identical. Removing ring setup
did not improve the complete workload, and the persistent treatment was
materially slower in the observed run. `URING_PERSIST=1` remains available
for hardware-specific study but is disabled by default.

An alternating direct-I/O microbenchmark was also inconclusive: the two
control observations were 1.401 and 1.496 GiB/s, while the persistent
observations were 1.544 and 1.349 GiB/s. This variability is another reason
not to promote the treatment.

### 4.2 Pinned upload arena

| Measurement | Control | Pinned staging |
|---|---:|---:|
| measured decode | 0.8104 tok/s | 0.7691 tok/s |
| measured request wall | 55.45 s | 52.04 s |
| time to first token | 50.56 s | 46.85 s |
| inclusive expert upload/matmul | 14.48 s | 8.68 s |
| inclusive expert storage | 19.98 s | 22.53 s |

The treatment staged 87.669 GiB over the full warm-plus-measured run through
a 0.062 GiB arena. It preserved exact output, cache counters, storage bytes,
and CUDA transaction totals. The phase movement and lower request wall time
are promising, but the decode-rate metric did not beat the control and
storage variability is large relative to this four-token sample.
`CUDA_PINNED_UPLOAD=1` therefore remains opt-in rather than becoming a
production default.

### 4.3 Decode-only phase attribution

The accepted defaults were rerun from the same atlas with both experimental
switches disabled:

| Decode phase, four tokens | Seconds | Fraction |
|---|---:|---:|
| expert storage | 3.758021 | 71.57% |
| expert upload and compute | 1.450550 | 27.62% |
| GDN/GQA attention core | 0.026783 | 0.51% |
| language-model head | 0.013910 | 0.26% |
| unassigned/rounding | 0.001681 | 0.03% |
| **decode total** | **5.250945** | **100.00%** |

This run measured 0.7594 tok/s. The exact route/cache totals again were
1,157 host hits, 517 device hits, 726 misses, and 4.853 GB read.

The inclusive `PERF.attention_s` value of approximately 20.68 seconds was
almost entirely prompt work. Comparing it directly with decode throughput
had falsely suggested an attention ceiling. `DPERF` shows that attention and
the language-model head together consume only 0.0407 seconds for the four
measured tokens.

## 5. Gate budget

At 2 tok/s, four tokens have a total budget of 2.0 seconds. Holding the
measured non-storage decode cost fixed leaves approximately:

```text
2.000 - (1.450550 + 0.026783 + 0.013910) = 0.508757 seconds
```

The run read 4.853 GB in 3.758 seconds, an effective rate of about 1.29 GB/s.
At that rate, 0.509 seconds permits about 0.66 GB of reads. With equal-size
routed expert payloads, misses must fall from 726 to roughly 99. That is
about a 95.9% non-disk hit rate across 2,400 route requests.

This is a local linear budget, not a universal law. Faster storage, overlap
between reading and compute, faster expert uploads, or a different GPU can
relax the required hit rate. It nevertheless replaces the earlier 86%
storage-only estimate for this reference machine because it reserves the
measured compute time inside the same 2-second gate.

## 6. Decision and next experiment

The persistent ring and pinned staging mechanisms are retained as explicit
experimental switches, but neither is accepted as a default from these
results. The `DPERF` protocol and gateway parsing are retained because they
correctly align phase timings with the decode throughput metric.

The next experiment will analyze per-layer route concentration in the saved
atlas and redistribute or protect the existing cache budget. A useful policy
must substantially reduce measured misses and bytes without increasing RAM
or changing model output. Attention-kernel work is deferred because its
maximum possible contribution here is less than one percent.

Artifacts:

- `c/qwen397_bounded_ring_control.json`
- `c/qwen397_bounded_ring_persistent.json`
- `c/qwen397_bounded_pinned_upload.json`
- `c/qwen397_bounded_decode_profile.json`
