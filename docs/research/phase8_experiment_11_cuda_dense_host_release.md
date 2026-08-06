# Phase 8 Experiment 11: CUDA Dense-Host Release

**Model:** `deepreinforce-ai/Ornith-1.0-397B-FP8`
**Converted revision:** `8b61f97a8512d9d01bff1a9625c9a16730e115bb`
**Date:** 2026-07-31
**Status:** completed; rejected by the preregistered throughput rule

## Abstract

The rejected warm-route atlas measured a complete 16-token decode working set
of 4,779 layer/expert pairs, approximately 29.75 GiB. The current 18 GiB host
expert tier plus 6 GiB device tier is short by approximately 5.84 GiB. At the
same time, the runtime retains about 6.78 GiB of non-expert container weights
in host memory after their successful CUDA upload.

This experiment tests an explicit GPU-only production mode. After verifying
that every eligible dense matrix has a persistent device copy, the engine
releases its no-longer-used host payload and scale buffers, retains the token
embedding and CPU-required vectors, and grows the per-layer routed-expert
cache from an 18 GiB startup budget to a 24 GiB final budget. Weight values,
quantization, routing, and CUDA kernels are unchanged.

## 1. Research question

Can host shadow removal transfer enough physical RAM from dense matrices to
routed experts to close the 2-token/s Ornith397 decode target without changing
model output or exceeding peak memory?

## 2. Terms

**Host shadow** means the CPU-side packed data and scale arrays retained after
the same matrix has been copied to persistent CUDA allocations.

**GPU-only fail-closed mode** means CPU matrix fallback is no longer valid
after host shadows are released. Any later CUDA failure terminates the request
or process instead of dereferencing freed host data or silently changing the
execution path.

**Two-stage cache budget** means startup uses the already safe 18 GiB expert
budget while dense host shadows still exist. Only after releasing them may the
cache grow toward 24 GiB. This prevents a transient 32+ GiB allocation peak.

## 3. Method

Two opt-in settings are proposed:

```text
CUDA_DROP_DENSE=1
RAM_GROW_GB=24
```

The implementation must:

1. preload all CUDA-eligible non-expert matrices and synchronize the device;
2. verify every matrix selected for release has a valid persistent CUDA cache
   entry;
3. preserve token embeddings and any vector or matrix still read directly by
   CPU code;
4. release only host payload/scale storage while retaining matrix geometry and
   device handles;
5. enter a recorded fail-closed state that prohibits CPU fallback;
6. recompute the final RAM plan from retained host bytes, recurrent/KV state,
   final expert budget, and the unchanged 1 GiB headroom;
7. grow each layer atomically, preserving existing experts and initializing
   only new slots; and
8. report released bytes, final cap per layer, and actual RAM/VRAM tier bytes.

## 4. Preregistered controls

Before the real pilot:

1. ordinary CPU and CUDA paths with both options absent must remain unchanged;
2. the tiny int4 oracle must remain greedy and teacher-forced 32/32 in the new
   mode;
3. mux resident output must be byte-identical with the mode off and on;
4. a synthetic CUDA failure after release must fail closed rather than enter a
   host fallback;
5. startup must use 18 GiB, and growth must occur only after the release;
6. both peak and final resource plans must preserve 1 GiB RAM and VRAM
   headroom;
7. the real pilot must retain the fixed prompt, one warm pass, one measured
   pass, 16 tokens, 4,096 context, 6 GiB device experts, persistent rings,
   pinned upload, and decode protection used by the 0.867369 control; and
8. generated token IDs and text must be byte-identical to that control.

## 5. Preregistered decision rule

The one-prompt pilot passes only if all structural and correctness controls
hold, resident telemetry reports CUDA on every layer and zero host-MoE
fallback, the final host expert tier is at least 23.9 GiB, and sustained decode
is at least 2.0 token/s. TTFT, misses, read bytes, dense release bytes, and
compute time are reported even on pass.

Any resource-plan, release-verification, correctness, CUDA-residency, or
throughput failure rejects the branch. A passing pilot authorizes the frozen
four-prompt Gate-8 qualification; it does not close Gate 8 alone.

## 6. Expected limitation

This mode intentionally sacrifices automatic CPU fallback after initialization.
It is suitable only when CUDA is mandatory and healthy. The default engine
must retain host shadows and its existing fallback behavior.

## 7. Implementation controls

The experimental branch enumerated the same non-expert matrices selected by
the dense CUDA preloader. It excluded the token embedding, floating-point
matrices, vectors, routed experts, and tied output storage. Before freeing any
host allocation, it synchronized CUDA and checked that every selected matrix
had a live device payload, scale allocation, and cache owner. It then enabled
a process-wide fail-closed flag, recomputed the final physical-RAM plan, and
expanded each layer's host cache before accepting requests.

On the tiny int4 oracle, the treatment retained 32/32 teacher-forced and 32/32
greedy agreement. The resident mux emitted byte-identical output with the mode
off and on. An injected CUDA error after release exited with status 2 and the
diagnostic `GPU-only mode cannot fall back to CPU`; it did not enter the CPU
matrix path. The focused seven-test CUDA mux suite and the 21-test qualifier
and gateway suite passed before the real pilot.

## 8. Full-container load-only control

The direct container completed the two-stage startup without swap growth or
direct-I/O fallback:

| Measurement | Result |
|---|---:|
| original expert cache | 48 experts/layer, 18.00 GiB budget |
| original peak plan | 26.20 GiB required / 29.375 GiB physical |
| verified dense matrices released | 376 |
| host storage released | 5,653,882,880 bytes (5.266 GiB) |
| retained dense host storage | 1.516 GiB |
| final expert cache | 64 experts/layer, 23.906 GiB |
| recurrent/KV state | 0.418 GiB |
| final plan including 1 GiB headroom | 26.841 GiB |
| load-only wall time | 93.8 s |

The measured release is smaller than the 6.78 GiB container-wide estimate
because CPU-required embeddings, floating-point matrices, and vectors must
remain resident. The final plan nevertheless fit with 2.534 GiB between the
planned requirement and physical RAM.

## 9. Fixed one-prompt pilot

The pilot is recorded in `c/bench/ornith397_dense_release_pilot.json`. It used
the preregistered 19-token prompt, one warm pass, one measured pass, 16 output
tokens, 4,096-token context, 18-to-24 GiB host budgets, 6 GiB device experts,
persistent `io_uring`, pinned upload, and decode protection. The emitted text
was byte-identical to the accepted control:

```text
A **cache miss** occurs when a processor requests data from the cache, but
```

The production mux artifact does not expose token IDs separately, so this
pilot proves byte identity but not an independent real-model ID stream. That
does not affect the decision because throughput independently fails.

| Measured decode quantity | 18 GiB control | Dense-release treatment | Change |
|---|---:|---:|---:|
| throughput | 0.867369 tok/s | 0.133596 tok/s | -84.60% |
| decode wall time | 18.431010 s | 119.684617 s | 6.49x |
| expert disk time | 11.386176 s | 10.314163 s | -9.41% |
| expert execution excluding measured reads | 6.876265 s | 108.913129 s | 15.84x |
| decode misses | 2,394 | 1,500 | -37.34% |
| decode expert reads | 16,003,111,950 B | 10,027,012,500 B | -37.34% |
| reported cache hit rate | 75.0625% | 84.3750% | +9.3125 points |
| TTFT | 142.637283 s | 140.278710 s | -1.65% |

All structural telemetry passed: 1,920 layer executions used the resident CUDA
MoE path, host-MoE fallback was zero, the device tier remained 5.99939 GiB,
and the host tier reached 23.90625 GiB. Startup took 151.975 s.

## 10. Interpretation

The treatment demonstrates that nominal capacity and useful capacity are not
the same. The final physical allocations held 3,840 host slots plus 963 device
slots, but telemetry classified only 2,960 host-unique and 963 device-unique
experts. Approximately 880 device entries duplicated host entries, leaving
3,923 unique resident layer/expert pairs. This remains 856 below the measured
4,779-pair warm working set and explains why 1,500 decode misses survived.

More importantly, reducing read volume did not reduce total latency. The saved
1.072 s of measured disk time was overwhelmed by an additional 102.037 s in
the residual MoE interval. The experiment did not enable CUDA stage events, so
that residual cannot be divided rigorously among allocator, upload, kernel,
and synchronization time. The concurrent evidence—roughly 26.4 GiB process
RSS on a 29.375 GiB host and unchanged device budget—supports memory pressure
and host/device cache churn as the primary hypothesis, not a model arithmetic
change. This is an inference and not a separately isolated causal result.

## 11. Decision and disposition

The treatment reaches only 6.68% of the 2.0 tok/s target and is substantially
slower than the immutable control. It is rejected, and the four-prompt Gate-8
qualification is not authorized. The experimental runtime, protocol,
qualifier, and test hooks were removed; the default fallback-capable engine is
restored. The benchmark artifact and this report retain the negative evidence.

The result also corrects the capacity model used to motivate this experiment:
future proposals must count unique host/device layer-expert pairs after
duplication, not add physical tier slots. Further remediation must eliminate
decode-time replacement or reserve more host headroom, and must measure CUDA
upload/kernel/synchronization stages directly before it can claim a
compute-side improvement.
