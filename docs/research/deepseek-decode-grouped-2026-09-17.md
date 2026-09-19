# DeepSeek-V4 decode: grouped per-layer expert stage and in-place reads

Date: 2026-09-17. Scope: decode only (the token-by-token phase after prefill).
Prefill is unchanged and was not re-measured beyond the paired runs below.

## Question

Decode ran at 2.98 s/token in serving (2048-token review, 528 output tokens,
`decode_s=1574`, engine log of 2026-09-11). Hypothesis: the cost is not the
drive but the serial shape of the per-layer expert stage -- six single-expert
fetches per layer, each a queue-depth-one read, a single-threaded record hash,
three host copies and three synchronised GEMM round trips -- and restructuring
it to one grouped submission per layer removes most of it without touching
arithmetic.

## What was measured before changing anything

Read-only measurements on this host (WSL2, `/dev/sdd` ext4 VHDX,
`dd iflag=direct` on `experts/layer-2x.bin`):

| access pattern | rate |
| --- | ---: |
| random 13 MB expert reads, one stream | 1.26 GB/s (~10 ms each; dd includes process spawn) |
| same, six parallel streams | 2.6 GB/s |
| sequential direct, 13 MB blocks | 2.4 GB/s |
| buffered (page cache) | 0.55 GB/s |

SHA-256 through libcrypto with SHA-NI: 2.5 GB/s per core.

The serving decode loop (`dsv4_execute_request`) runs each token through
`dsv4_runtime_prefill_chunk(count=1)`. After prefill `prefill_entries` is
NULL, so `dsv4_prefill_layer_chunk` fetched each of the six experts with its
own `dsv4_expert_cache_acquire_many(..., 1, ...)` and ran each through
`dsv4_prefill_expert` (three `dsv4_prefill_linear` calls, each an upload,
launch, download and `cudaSync`). On disk every expert's six records are
contiguous, 4 KiB-aligned and exactly `dsv4_expert_payload_bytes()` long, in
the order s1,w1,s3,w3,s2,w2; the reader nevertheless bounced them through an
io_uring buffer, then staging, then copied record by record into the packed
entry layout, then again into pinned upload staging.

## Change

1. **Decode profile** (`DSV4_DECODE_PROFILE` on stderr per request): per-step
   seconds for attention, route, routed experts, shared expert, head, and
   inside the routed stage read / hash / copy / upload / kernel, plus experts
   and bytes read, uring batches, host and device hit/miss counts. Cumulative
   counters live in `dsv4_expert_cache` and `dsv4_runtime`.
2. **Grouped decode path** (`dsv4_prefill_layer_chunk`, `count == 1`, no
   prefill window): sort the six routed ids ascending and call
   `dsv4_routed_experts_forward` -- one `acquire_many` of six (one io_uring
   batch, so the drive sees queue depth up to six and the 36 record hashes
   run on six threads), one device acquire of six (six async pinned uploads),
   one grouped FP4 kernel, one download, one sync -- then the shared expert
   and the existing `round_bf16(routed + shared)` and hyper-connection post.
   This is the path the smoke binary already took. `DSV4_DECODE_GROUPED=0`
   restores the serial route for paired comparison.

   **The fused grouped kernel is not byte-identical to the serial path**, and
   the first paired run showed it: same binary, same prompt, greedy, and the
   two arms' outputs diverged after roughly 45 tokens. The per-layer FFN
   agreed bit for bit on 86 real-weight trials (six experts, all 43 layers,
   serial vs fused vs CPU oracle), so the cause is rare. It is `expf`:
   `dsv4_group_middle_quant_kernel` evaluates the clamped SwiGLU on the
   device, and CUDA `expf` differs from glibc by one ulp in 31.7 % of
   arguments (measured on 16.7 M BF16-valued gate/up pairs on this GPU);
   after BF16 rounding the middle value differs 4 times in 16.7 M, i.e.
   about 0.13 flipped FP8 codes per decoded token, enough to move a near-tie
   over a long decode. `ceilf(log2f(·))` for the MXFP scale never differed.
   The serving path therefore uses a split form (new backend entry points
   `coli_cuda_dsv4_grouped_fp4_hidden` and `coli_cuda_dsv4_grouped_fp4_down`):
   the six hidden GEMMs run on the device and come down to the host, the
   middle stage (route weight, clamped SwiGLU, BF16, MXFP quantization) runs
   on the host with the per-expert path's own code, and the six down GEMMs
   plus the id-ordered reduction run on the device. One extra 98 KB download
   and 12 KB upload per layer. The fused kernel remains for its fixture test
   and is no longer used by the runtime.
3. **Parallel record hashing** for any batch of more than one record
   (`dsv4_store_verify_records`: `if(count>1)` instead of `if(count>6)`), so a
   single expert's three 4 MB records hash concurrently.
4. **In-place extent reads**: entry storage is page-aligned
   (`posix_memalign`); when an expert's extent is packed and aligned it is
   read straight into the entry and hashed there, and the entry's record
   pointers follow the on-disk order. `st.h` gained an aligned fast path
   (`st_direct_in_place`) so O_DIRECT lands in the destination with no bounce
   buffer; unaligned callers are untouched. Device slots record the six
   record offsets they were uploaded with, so device pointers never depend on
   how the host entry is laid out later. Two host copies per miss disappear.

5. **Direct upload from registered entries.** The profile of the split arm
   still charged 0.25–0.28 s per step to `upload`, which is the 13 MB memcpy
   of every device miss into pinned staging. Measured on this host with six
   13 MB experts (`cudaMemcpyAsync` + sync, caches flushed between runs):

   | upload route | effective |
   | --- | ---: |
   | memcpy into 80 MB pinned staging, then H2D (engine default) | 10.1 GB/s |
   | pageable `cudaMemcpy` straight from the page-aligned entry | 15.5 GB/s |
   | H2D from a `cudaHostRegister`ed entry, no copy | 28.1 GB/s |
   | H2D from the 80 MB pinned block alone | 28.8 GB/s |

   `cudaHostRegister` on an already-filled 13 MB entry costs 0.67 ms
   (`cudaHostAlloc` 18.5 ms, and WSL2 pinned 9 GiB of 13 MB entries without
   complaint). Entries are therefore registered once after their first fill
   (`dsv4_expert_pin`), uploads DMA directly from them, and `prefill_begin` /
   `prefill_end` / close unregister before freeing. `DSV4_PINNED_CACHE=0`
   keeps the staging route; a registration failure falls back to it and is
   counted (`pin_failures` on the profile line).

## Correctness controls

- `make -C c CUDA=1 test-deepseek` passes (all DSV4 suites, CUDA build).
- `test_deepseek_v4_cuda.c`: the split hidden/host-middle/down form matches
  the CPU reference with `absolute == 0` on the grouped fixture.
- `test_deepseek_v4_prefill.c`: CPU `prefill_chunk(chunk=1)` grouped and
  serial both `memcmp`-equal to `decode_token`; on CUDA, grouped chunk=1 is
  byte-identical to serial chunk=1 and within `max_error=0` of the CPU
  scalar; a second fixture with the converter's page-aligned disk-order
  records decodes to byte-identical logits on CPU and CUDA while never
  allocating a staging buffer (`read_staging == NULL`) and with entry
  pointers verified to follow the on-disk order.
- `test_st`, `test_st_pread`, `test_st_batch` pass; `qwen` and `glm53` build.
- Paired real-model arms below share one engine binary
  (`binary_sha256=2c7ae9c5...`), one prompt (`negative-transfer`, 2048
  tokens, `payload_sha256=0771e5d1...`), greedy sampling, 128-token budget,
  fresh process each (cold decode cache, as in production), the production
  environment (`RAM_GB=8 CUDA_EXPERT_GB=2 DIRECT=1 URING=1 URING_PERSIST=1`).

## Results

All runs: fresh engine process per arm, greedy (`temperature=0`), production
environment, prompts from `docs/research/deepseek-review-inputs-2026-09-10`.
`DSV4_DECODE_TRACE=1` records the chosen token, the top two logits and a
logit checksum at every step, so arms are compared step by step, not only by
final text.

### Determinism of the baseline and the fused-kernel divergence

`upper-bound` (512 tokens), 96-token budget, engine `2c7ae9c5…` (grouped path
still using the fused device middle):

| arm | output sha256 | decode | s/step | first differing step |
| --- | --- | ---: | ---: | --- |
| serial, run a | `3d3e8394…` | 195.4 s | 2.06 | – |
| serial, run b | `3d3e8394…` | 187.7 s | 1.98 | none: 96/96 trace lines identical |
| grouped, fused device middle | `f9e13ff8…` | 139.8 s | 1.47 | logits at step 16 (top logit 39.6626 → 39.5133, margin 4.618 → 4.429); token at step 74 (margin 0.052) |

The serial path is deterministic run to run. The fused grouped path agreed
with it for 16 complete steps, then carried a 0.03–0.15 logit shift until a
0.05-margin near-tie flipped at step 74. That is the `expf` mechanism above,
and it is why the runtime now uses the split form.

### Split form (host middle stage): exactness restored

Same case and budget, engine rebuilt with the split path:

| arm | output sha256 | trace | decode | s/step | tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| serial (above) | `3d3e8394…` | – | 195.4 s | 2.06 | 0.49 |
| grouped, split | `3d3e8394…` | 96/96 lines identical to serial | 137.6 s | 1.45 | 0.70 |

Per-step profile of the grouped split arm: attention 0.140, route 0.038,
routed 1.242 (read 0.665, hash 0.154, copy 0.000, upload 0.249, kernel
0.172), shared 0.025, head 0.003; 133 experts (1.78 GB) read per step in
41.6 io_uring batches (one per layer with a miss), host hit 48 %, device
misses 194 per step.

### Matched pair on the 2048-token case (split form, engine `f0ca804a…`)

`negative-transfer`, 128-token budget, serial arm first:

| arm | output | decode | s/step | tok/s |
| --- | --- | ---: | ---: | ---: |
| serial | identical (128/128 trace lines) | 240.5 s | 1.893 | 0.532 |
| grouped, split | identical | 205.0 s | 1.613 | 0.624 |

Per-step components, serial → grouped: read 0.823 → 0.721, hash 0.281 →
0.196, upload 0.190 → 0.279, kernel (launches through sync, which also
absorbs the queued expert DMA) 0.376 → 0.180, attention 0.154 → 0.163;
io_uring batches 138 → 41.5 per step. This pair shows the smallest gain of
the four measured: the serial arm's queue-depth-one reads ran at 2.25 GB/s
here against 1.8 GB/s in the first pair and 1.26 GB/s from `dd`. The WSL2
VHDX sits behind the Windows host's file cache, so repeated runs over the
same prompt warm it and the "drive" is not a stable instrument between arms;
the io_uring batch count and the device-side numbers are.

### Direct upload from registered entries

`upper-bound`, 96-token budget, same binary, `DSV4_PINNED_CACHE=1` then `=0`:

| arm | output | host cache | misses/step | s/step | read | hash | upload | kernel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| registered entries | identical to serial | 5.9 GiB (11 slots/layer) | 146.4 | 1.574 | 0.898 | 0.182 | 0.006 | 0.254 |
| staging memcpy | identical to serial | 7.6 GiB (14 slots/layer) | 133.2 | 1.679 | 0.806 | 0.180 | 0.280 | 0.184 |

The registered arm was handicapped: it started with 21.8 GiB of host memory
available (the pinning probes had just run) and `dsv4_plan_memory` shrank
its host cache, costing 13 extra misses and ~0.17 GB per step, and it still
finished faster. Upload time fell from 0.28 s to 0.006 s per step (26,387
direct uploads, zero registration failures); the sync now absorbs the DMA
it used to overlap with the next memcpy, so `kernel` rose by 0.07 s. At equal
cache capacity the registered route is worth about 0.2 s per step. This is
also a reminder for every paired run: check `host_cache` on the
`DSV4_MEMORY` line of both arms before comparing them.

## Summary and decision

| stage | s/step (2048 case) | tok/s | output |
| --- | ---: | ---: | --- |
| 2026-09-11 serving run (528 tokens) | 2.98 | 0.34 | reference |
| serial route on this branch (in-place reads, parallel hash) | 1.89–2.09 | 0.48–0.53 | identical, deterministic |
| grouped split route, staging uploads | 1.45–1.61 | 0.62–0.69 | identical (2048: 128/128 steps; 512: 96/96) |
| grouped split route, registered entries | ≈1.50 (512 case, handicapped cache) | ≈0.67 | identical |

Retained, all default-on: grouped decode (`DSV4_DECODE_GROUPED=0` to
compare), in-place extent reads, parallel record hashing, split grouped
kernels with the host middle stage, registered-entry uploads
(`DSV4_PINNED_CACHE=0` to compare), and the `DSV4_DECODE_PROFILE` /
`DSV4_DECODE_TRACE` telemetry. Rejected: the fused device middle stage for
serving (not byte-identical). Not claimed: any change to prefill or to
review quality; the planted-defect ladder was not re-run because the token
stream is byte-identical to the serial route's on every measured case.

What is left per step is now read 0.7–0.9 s at the drive's measured ceiling,
hash 0.18, attention 0.15, DMA-plus-launch 0.18–0.25, and shared/route/head
0.07: the disk is on the critical path and cannot be batched further. The
next levers are the ones the plan lists as Ways 2–3, overlap and fewer
bytes.
