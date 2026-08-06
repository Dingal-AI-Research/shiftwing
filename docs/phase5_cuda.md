# Phase 5 CUDA backend — Gate passed 2026-07-21

The workspace-local CUDA 12.9.86 toolkit at `.toolchains/cuda-12.9` compiles
both `CUDA_ARCH=sm_120` and the Makefile default `CUDA_ARCH=native`. The CUDA
runtime identifies the RTX 5070 Ti as compute capability 12.0. CUDA kernels are
compiled by nvcc; the C engine and final link use GCC 13.3.

## Backend and kernels

`c/backend_cuda.h` is the C-compatible boundary between `qwen.c` and CUDA.
CUDA runtime types do not cross it. The backend provides device discovery, a
nonblocking stream, device and pinned-host allocation, asynchronous copies,
and deterministic reference-tested kernels for:

- zero-centered Qwen RMSNorm, SiLU multiplication, AXPY, and conversion;
- container-layout int8 and grouped-int4 GEMV, including fp16 activations;
- direct-half grouped routed experts and a fused shared-expert transaction;
- causal convolution plus fp32 Gated DeltaNet recurrent state;
- zero-centered q/k norm, 64-dimension partial split-half RoPE, gated GQA,
  persistent KV state, and int8/int4 output projection.

Run the independent kernel gate with:

```sh
cd c
make CUDA=1 CUDA_ARCH=sm_120 test-cuda
```

The deterministic test covers non-power-of-two dimensions, padded row
strides, grouped-int4 tail groups, and fp16 conversion. Maximum absolute
differences against scalar references are 4.77e-7 for RMSNorm, 9.31e-10 for
SiLU, 7.45e-9 for AXPY, 8.11e-6 for int8 GEMV, 1.31e-6 for int4/fp32 GEMV,
and 1.00e-3 for int4/fp16 GEMV.

## Inference architecture

Build with `make CUDA=1`; runtime opt-in is `COLI_CUDA=1 CUDA_DENSE=1`.
`CUDA_EXPERTS=1` includes routed experts and `CUDA_F16=1` selects fp16
activations for grouped int4. Unsupported matrices automatically retain the
CPU implementation.

On CUDA devices with memory-pool support, transient device allocations use
stream-ordered `cudaMallocAsync`/`cudaFreeAsync` by default. This avoids a
global synchronization during high-churn expert-cache replacement. Set
`CUDA_ASYNC_ALLOC=0` to restore synchronous allocation; both modes are covered
by `test_backend_cuda`. The Qwen397 same-atlas study and its limitations are
reported in
`docs/research/phase7_experiment_11_cache_telemetry_and_async_allocator.md`.

The completed decode path contains:

- all eight routed experts in two grouped kernels. The hidden kernel computes
  gate/up/SwiGLU directly into fp16; the down kernel combines route weights;
- the int4 shared expert in three specialized kernels, while the official
  int8 shared expert uses the generic fused CUDA MLP; both stay inside the
  same one-upload/one-download MoE transaction;
- persistent GDN convolution rings and recurrent state, with projection,
  causal convolution, recurrence, gated norm, and output projection in one
  host transaction;
- persistent GQA KV state, q/k normalization, partial RoPE, attention gate,
  and output projection in one host transaction;
- a tensor-name-keyed device-weight LRU independent of CPU expert slots. It is
  bounded by `CUDA_EXPERT_GB` (8 GiB default) and serves experts directly even
  after their CPU copies are evicted. Dense weights preload at startup and
  prompt routes populate expert VRAM while host pages are hot.

CPU prompt prefill remains intentional: it uses the validated 64-token chunked
GDN implementation, then uploads recurrent/KV state once before CUDA decode.
`WARMUP=N` runs a deterministic route/kernel warmup and resets model state
before measured generation. This is a cache-warm decode measurement, not a
claim about cold model load or prompt ingestion.

## Gate evidence

CPU and CUDA both replay the fp32, int8, and int4 tiny snapshots at
teacher-forced and greedy 32/32. The int4 run exercises 370 grouped MoE calls,
248 fused GDN calls, and 74 fused GQA calls. The int8 path uses its generic CUDA
fallbacks and is also 32/32; fp32 correctly stays on CPU.

The fixed real-model result is in the ignored artifact
`c/bench/qwen35_prefix_cuda.json`. Against the same saved llama.cpp Q4_K
continuations used by Gate 4, CUDA scores **1,231/1,280 = 96.171875%** over
20 prompts × 64 positions. The aggregate equals CPU. Floating-point ordering
shifts one match from prompt 9 to prompt 7 (CPU 60/55, CUDA 61/54), so real
scale is not claimed bit-identical. Free-running agreement is 756/1,280
(59.06%); as in Gate 4, it is diagnostic because an early near-tie changes all
later context.

The repeatable performance command is:

```sh
cd c
OMP_NUM_THREADS=8 COLI_CUDA=1 CUDA_DENSE=1 CUDA_EXPERTS=1 CUDA_F16=1 \
CUDA_EXPERT_GB=8 EXPERT_RAM=64 PREFETCH_THREADS=4 SNAP=./qwen35 \
CHAT=1 TEXT=0 PROF=1 PROF_DETAIL=1 WARMUP=64 NGEN=64 PROMPT=Hello ./qwen
```

Three final-binary 64-token decode samples are **31.75, 31.87, and 30.46
tok/s**, median **31.75 tok/s**. The representative 31.75 run took 2.016 s:
GDN 0.692 s, attention 0.156 s, MoE 1.094 s, LM head 0.053 s, and other
0.021 s. The expert
cache occupied 7.05 GiB with zero decode-time misses or evictions. Cold model
load varied from 9.3 to 64.2 s with OS page state and is reported separately;
CPU prefill was 5.4–5.9 s for this prompt.

Final validation is green: 13 C tests, 8 Python tests, the standalone CUDA
kernel test, tokenizer encode/decode 10,000/10,000, and fp32/int8/int4 tiny
oracles on both CPU and CUDA.

## Controls and fallback

`CUDA_GDN=0`, `CUDA_ATTN=0`, `CUDA_MLP=0`, `CUDA_GROUPED=0`,
`CUDA_GROUPED_KERNEL=0`, and `CUDA_PROJECTIONS=0` independently restore earlier
or CPU paths. `CUDA_PRELOAD=0` disables startup preload. `CUDA_GDN_CHECK=1`
runs CPU and GPU recurrence side by side for debugging. Omitting
`COLI_CUDA=1` keeps the CPU engine unchanged.
