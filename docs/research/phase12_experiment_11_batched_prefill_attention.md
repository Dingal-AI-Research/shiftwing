# Phase 12 Experiment 11: Batched Full-Attention Prefill

**Project:** Shiftwing
**Model:** Ornith397 q3 — 397B total / 17B active parameters, Mixture-of-Experts
**GPU:** NVIDIA GeForce RTX 5070 Ti, sm_120, 70 multiprocessors, 15.92 GiB VRAM
**Host:** AMD Ryzen 7 7700X, 29.375 GiB RAM recorded
**CUDA compiler:** nvcc 12.9.86
**Report date:** 23 August 2026
**Status:** Measured, retained, default on

## Abstract

Full-attention prefill in the Shiftwing engine was executing the single-token
decode path once per prompt position. On an 8,451-token prompt that meant
126,825 kernel chains, each ending in a host stream synchronization, and each
launching only 32 thread blocks on a 70-multiprocessor GPU. The attention stage
measured 0.041 TFLOP/s, roughly a thousandth of the device's fp32 capability.

Batching the rows into one call per layer, while leaving every arithmetic
operation and its order untouched, cut the attention stage **9.709x** on the
real model — from 421.850426 to 43.450325 seconds — and lowered time to first
token **33.75%**, from 19m19.300s to **12m48.086s**. Generated output was
byte-identical, matching the bound baseline's SHA-256.

Attention was the only stage that moved materially; every other stage stayed
within 2% except the expert-I/O terms, which rose as predicted once there was
less compute to hide storage behind. A separate defect was found in the same
code: the attention score buffer overflowed the 48 KB shared-memory cap at
position 12,256, which disabled CUDA for the remainder of a run and made the
10,000-16,000-token target workload unservable.

The first real-model attempt at this measurement was a **negative result** and
is preserved. It is reported in full in section 8, because it is the reason the
experiment has an engagement gate at all.

## 1. Research question

Phase 12 optimized every prefill stage except attention. After that work,
attention accounted for 421.850426 of the 1,159.299940 seconds of time to first
token — 36.4%, the largest remaining stage and the only one that grows
quadratically with prompt length.

An earlier report, `phase12_preflight_09_prefill_compute_candidates.md`, had
closed this candidate:

> **Candidate C is closed with no work required.** Disabling CUDA attention made
> attention 3x *slower*, which proves the GPU path is already active and already
> helping. [...] attention is simply genuinely expensive at quadratic cost, and
> there is nothing to fix.

The first sentence is correct and is confirmed here. The conclusion does not
follow from it. An experiment showing that the GPU path beats the CPU fallback
by 3x establishes that the path is active; it does not establish that the path
is anywhere near what the hardware can do. This experiment asks what the actual
bound is.

## 2. Technical background

**Prefill** is the phase that reads a prompt before any output token is
produced. It dominates time to first token for long prompts.

**Attention** compares every token against every earlier token. For a prompt of
T tokens the work grows with T², so a prompt four times longer costs roughly
sixteen times as much attention. Ornith397 uses 32 query heads sharing 2
key/value heads, a head dimension of 256, and applies full attention on 15 of
its 60 layers; the other 45 use a linear-attention variant.

A **kernel** is a function run by many GPU threads at once. A **kernel launch**
is the CPU scheduling that function. A **thread block** is a group of threads
that can cooperate; a GPU runs many blocks at once across its
**multiprocessors** — this device has 70. A kernel launched with only 32 blocks
therefore cannot use even half the machine, no matter how efficient its inner
loop is. **Occupancy** is how much of the device is kept busy.

**Shared memory** is a small fast scratch space private to a thread block,
capped by default at 48 KB. Requesting more makes the launch fail.

**Bit-identical** here means the new code produces the same output bits, not
merely close numbers. This matters because floating-point addition is not
associative: changing the order of a sum changes the result slightly, which
changes which token is chosen, which changes the text. Shiftwing's
qualification harness binds generated text by SHA-256, so a numerical change
invalidates the bound baseline and every quality gate behind it.

## 3. The defect

Phase 12 gave the linear-attention layers a batched prefill routine and their
cost fell 2.9x. The full-attention layers never received one. The prefill loop
read:

```c
if (l->type == LT_LINEAR) gdn_prefill_layer(m, l, n, T, mix);
else for (int t = 0; t < T; t++) { m->pos = t; attn_forward(m, l, ...); }
```

Every one of those `attn_forward` calls performed an activation upload, three
matrix-vector launches, normalization and rotary embedding, a key/value cache
append, the attention kernel, an output projection, a download, and a full host
stream synchronization. The engine's own counter confirms the shape:
`GQA-calls=126825`, which is 15 layers × 8,455 rows.

The attention kernel itself launched with one block per query head — **32
blocks on 70 multiprocessors** — with the loop over key positions serial inside
each block and the softmax section single-threaded across three passes.

Three defects follow: insufficient occupancy, per-token serialization behind
126,765 host synchronizations, and a hard ceiling described in section 7.

## 4. Method: batch the rows, keep the arithmetic

The available gain is in scheduling, not arithmetic, so nothing about the
computation needed to change. Batching a block of consecutive rows is safe
because row *t* writes its key and value at position *t* and reads only
positions up to *t*; rows above it write strictly above, where it never reads.
Projecting and appending the whole block before the causal pass therefore
produces exactly the values the sequential loop produced.

Each element preserves operation order deliberately:

- the projections reuse an existing batched kernel that is the single-row
  kernel plus a sample index, with an identical lane-strided accumulation and
  reduction tree;
- the normalization and rotary kernels are the existing kernels with a row
  index, evaluated at the same position;
- the attention kernel keeps the original body verbatim per (row, head) pair —
  the same block-wide reduction for the dot product, the same sequential
  maximum, the same sequential exponential and denominator, the same sequential
  weighted accumulation. Only the number of blocks in flight changes.

Three implementation details are load-bearing:

**Grid-stride pairs.** Work is walked so that scratch memory is sized by
resident blocks rather than by row count, which would reach gigabytes at 16,384
tokens. Consecutive blocks are assigned to the same row, so the 32 query heads
sharing 2 key/value heads read the same cached data and hit the GPU's L2 cache
instead of re-reading it from memory sixteen times.

**Chunked softmax staging.** The softmax scan must stay sequential to stay
exact, but it need not read global memory one element at a time. Staging scores
through shared memory in 1,024-element chunks preserves the order while making
the loads block-wide and coalesced. This alone lifted the 8,451-token result
from 7.57x to 8.63x.

**Scores in global memory.** This removes the ceiling described in section 7.

## 5. Instrumentation and its validation

A full-model arm costs roughly 20 minutes, which is too slow to iterate on.
`c/tests/bench_gqa_prefill.c` builds one full-attention layer of synthetic
weights at the model's real dimensions and runs the same rows through both
paths, comparing every output float with `memcmp`.

Before drawing any conclusion from it, the harness was checked against
production telemetry:

| prompt tokens | harness, 15-layer projection | measured on the real model | delta |
|---:|---:|---|---:|
| 2,335 | 40.23 s | 42.6 s (preflight 09 baseline arm) | -5.6% |
| 8,451 | 417.75 s | 421.83 s (experiment 10 best stack) | -1.0% |

Synthetic weights reproducing the production stage to 1% at 8,451 tokens is the
basis for trusting the microbenchmark results below.

## 6. Microbenchmark results

Device: 70 multiprocessors, 49,152 B shared memory per block. Per full-attention
layer:

| T | per-token | batched | speedup | differing floats |
|---:|---:|---:|---:|---|
| 512 | 0.268 s | 0.102 s | 2.64x | 0 / 2,097,152 |
| 2,335 | 2.738 s | 0.534 s | 5.12x | 0 / 9,564,160 |
| 8,451 | 28.214 s | 2.962 s | **9.53x** | 0 / 34,615,296 |
| 16,384 | 100.958 s | 9.967 s | **10.13x** | 0 / 67,108,864 |

Key and value caches also matched exactly at every size. The per-token path
measured 0.016, 0.033, 0.041 and 0.044 TFLOP/s respectively, against roughly 44
TFLOP/s of fp32 capability — about **0.1% of roofline**. The stage was
latency-bound, not arithmetic-bound.

Speedup rises with prompt length because longer prompts have more rows with
which to fill the device. Attention cost is independent of prompt *content* —
every token attends to every earlier token regardless of what the tokens are,
and the kernel contains no data-dependent branching — so prompt length is the
only variable that governs this curve.

## 7. A latent ceiling at 12,256 tokens

The attention score buffer was allocated in shared memory sized
`(position + 1) × 4` bytes, and nothing raised the 48 KB default cap. The launch
therefore fails once a prompt passes roughly 12.25 thousand positions. Measured:
it fails at **position 12,256**, requesting 49,028 bytes.

The consequence is worse than a failed kernel. The failure path calls
`cuda_backend_fail`, which clears the runtime's active flag and drops the entire
model to the CPU for the remainder of the run. The 10,000-16,000-token code
review that motivates this work would have hit it.

Moving the score buffer to global memory removes the ceiling. The change is a
storage location only, so it is bit-identical, and it was applied to the
single-token path as well.

## 8. The first real-model attempt was negative

The first attempt at the full-model measurement **failed to execute the new code
at all**. It is preserved as
`c/bench/ornith397_attn_prefill_longctx/attn_prefill.notengaged.20260823T194147.json`.

It reported `GQA-calls=126825` — precisely the per-token count — with
`GQA-prefill-calls=0`, and an attention stage of 423.321491 s, unchanged. Its
time to first token of 1,127.913853 s was 2.71% below the best-stack baseline.
That difference is run-to-run variation, not an effect.

The cause was integration, not the kernel. The batched routine had been wired
into the command-line prefill function and the speculative verifier, but the
qualifier drives the engine through its OpenAI-compatible server into the
serving multiplexer, whose prefill runs in a third per-token attention loop in
`serve_prefill_step`. Every unit test passed, because the tests exercised the
two call sites that had been wired.

This is why the harness gates on the engagement counter covering the whole
prompt rather than on timing alone. Without that gate the run would have been
recorded as a 2.7% improvement and attributed to a code path that never ran. The
counter was added specifically for this failure mode, and it is the reason the
negative result was caught rather than published.

## 9. Full-model result

One hash-bound, zero-warmup arm on the preserved 8,451-token fixture, compared
with the best-stack configuration from experiment 10 under the same model
manifests, prompt, context, four-token limit, and exact-output gate. Stage
timers are the engine's raw values and include the four-token decode component
in both arms.

| | best stack | + batched attention | change |
|---|---:|---:|---:|
| Time to first token | 1,159.299940 s | **768.086294 s** | **-391.213645 s (-33.75%, 1.509x)** |
| TTFT, clock form | 19m19.300s | **12m48.086s** | -6m31.214s |
| **Full attention** | **421.850426 s** | **43.450325 s** | **-378.400101 s (-89.70%, 9.709x)** |
| Linear attention (GDN) | 193.733646 s | 182.032615 s | -11.701031 s (-6.04%) |
| Expert matmul | 540.350517 s | 529.813365 s | -10.537152 s (-1.95%) |
| Expert disk | 157.737780 s | 176.837196 s | +19.099416 s (+12.11%) |
| Pipeline consumer wait | 5.592311 s | 14.315415 s | +8.723104 s |
| Output SHA-256 | `181415b0…12235f` | `181415b0…12235f` | identical |

Engagement was verified rather than assumed: `GQA-prefill-calls=15`, one per
full-attention layer, and `GQA-prefill-rows=126765`, which is 15 × 8,451 — the
entire prompt.

**Attention is the only stage that moved materially.** This makes the experiment
a far cleaner attribution than the stacked best-stack arm, which changed several
things at once and could not apportion its result. The two expert-I/O terms
moved in the direction that removing compute predicts: with less arithmetic to
hide storage reads behind, previously concealed wait becomes visible. That
effect consumed about 28 seconds of the 378 saved.

The microbenchmark predicted the outcome closely. It projected a 44.43 s
attention stage; the model measured 43.450325 s, an error of 2.2%. A TTFT band
of 750-900 s was stated before the run; the measured value was 768.086294 s.

The result clears the owner's roughly 18-minute operating point by 5m11.914s.
Experiment 10 missed the strict 18-minute threshold by 79.299940 seconds; this
configuration passes it.

## 10. Exactness evidence

Bit-identical was the acceptance bar for every element, not a tolerance.

- `c/tests/test_attn_prefill.c` compares the batched routine against the
  sequential loop and requires exactly zero differing floats, tighter than the
  neighbouring slot-batch test's 1e-6 tolerance. It covers a prefill block, a
  mid-sequence verifier block, and two blocks below the batching threshold. It
  also asserts *engagement*, because a test that silently falls back compares a
  path against itself and always passes — the first draft of this test did
  exactly that, having never started the CUDA backend.
- End-to-end on a tiny model, two prompts of 200 and 64 tokens produced
  identical token streams with the batched path engaged and disabled.
- Speculative decoding accept rate was 24/24 in three arms: batching off,
  batching at its shipping threshold, and batching forced into the four-row
  verifier blocks. A verifier divergence does not fail loudly — it quietly
  collapses accept rate — so the forced arm is the control for that mode.
- On the full model, generated text matched the bound baseline's SHA-256.

## 11. Limitations

- **One run, against a preserved baseline.** This is not an interleaved,
  repeated, paired experiment. It establishes what this configuration did once;
  it does not provide a confidence interval.
- **Startup is excluded**, following experiment 10's convention. The runs were
  not contemporaneous.
- **Four completion tokens.** Decode is untouched by this change, and a real
  400-token review adds roughly 8 minutes at the measured decode rate. The
  headline is a prefill result, not a review time.
- **Cache state.** A short smoke run preceded this arm, so the page cache was
  not cold. Expert disk time rose 12.11% rather than falling, which argues
  against warm-cache flattery, but the machine was not reset.
- **Length coverage.** Only 8,451 tokens was measured on the real model; the
  512-16,384 curve is microbenchmark.
- **16,384 tokens has never been run end-to-end**, so the ceiling fix is
  verified by unit test and microbenchmark only.
- **Concurrent serving is unchanged.** The multi-slot path receives the ceiling
  fix but not the batching.

## 12. Decision

Retain, default on, controlled by a recorded flag so an arm is reproducible
rather than ambient. Open next: repeated paired arms for a confidence interval,
a 16,384-token arm, and a run long enough to evaluate decode separately.

The ceiling fix should be treated as a correctness fix independent of the
performance work. Without it, a 16,000-token prompt silently disables GPU
execution for an entire run.

## 13. Reproduction

```sh
make -C c CUDA=1 CUDA_ARCH=native bench-gqa-prefill
make -C c CUDA=1 CUDA_ARCH=native test-cuda
```

Full-model harness:
`c/bench/ornith397_attn_prefill_longctx/run_attn_prefill_longctx.sh`.
Artifact: `attn_prefill.json`. Preserved negative arm:
`attn_prefill.notengaged.20260823T194147.json`.

Source: `c/backend_cuda.cu` (`gqa_prefill_kernel`,
`coli_cuda_gqa_prefill_q4_f16`, `gqa_decode_launch`), `c/qwen.c`
(`attn_prefill_layer`, `cuda_attn_prefill_try`, and the three call sites),
`c/tests/bench_gqa_prefill.c`, `c/tests/test_attn_prefill.c`.
