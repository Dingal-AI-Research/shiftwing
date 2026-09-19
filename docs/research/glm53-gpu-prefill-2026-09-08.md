# GLM batched GPU prefill — 2026-09-08

Implements proposal 2 from `glm53-prefill-bottleneck-2026-09-08.md`.
The native GLM engine can now batch its large matrix operations on CUDA.
This targets prompt reading. It does not change expert quantization, review
approval rules, timeouts or LocalForge orchestration.

## Measured result

One matched 128-token input, separate freshly loaded model processes, same
int4 weights, 23 expert slots per sparse layer (14 GB configured RAM budget),
16 CPU threads and 128-token chunk. LocalForge and the orchestrator were not
running during the measurement. Each process had a 300-second wall limit.

| Measurement | Previous scalar CPU path | Batched GPU path |
| --- | ---: | ---: |
| Prompt reading | 102.6 s | 89.2 s |
| Prompt tokens/sec | 1.25 | 1.43 |
| Model loading | 67.7 s | 126.7 s |
| Complete cold CLI process, including printed logits | 176.7 s | 222.9 s |
| Peak sampled total GPU use | 14 MiB | 4975 MiB |

Prompt reading was **1.15× faster**, or **13.1% less time**, in this single sample.
This is below the original 1.5–3× planning estimate. The complete cold CLI run
was slower because loading took longer in the second run. The reason for that
load-time difference has not been isolated; this result is not an overall
review-speed improvement claim.
Teacher-forced token choices identical: **True**.
Maximum final-position logit difference: **1.907e-05**.
The input, output logits, stderr and resource samples are under
`../smoke/glm53-gpu-batch-2026-09-08/`.

This is a short prefill comparison, not a completed review, long-context test
or quality evaluation. It does not establish the speed of a 5K review or its
thinking/answer generation. CPU ran first; model processes were fresh but OS
file caches were not explicitly flushed. Most expert reads use the existing
direct-I/O path. Loading and prompt computation are reported separately.
GPU availability alongside another resident model was not benchmarked.

## Where the work happens

| Resource | Work |
| --- | --- |
| GPU | Batched attention/indexer projections, dense/shared feed-forward networks, expert feed-forward networks with multiple routed tokens, final output projection. |
| CPU | Expert selection and weighted combination, normalization and residual connections, ordered KDA memory updates/convolutions, sparse token selection and attention pooling, single-token generation. |
| RAM | Existing resident weights, expert cache, attention/recurrent state, gathered token batches. GPU copies do not remove their CPU originals because fallback and generation need them. |
| Disk | The same int4 routed experts, read through the existing cache and direct-I/O path. The full model remains too large for RAM or VRAM. |

GPU weights retain their compact FP32/int8/int4 source formats. Quantized rows
are expanded into a reusable FP32 tile of at most 32 MiB for cuBLAS. Immutable
shared matrices stay uploaded between operations. Routed expert weights use
three reusable buffers, uploaded again for each expert; CPU pointer reuse is
never interpreted as a cache hit. All tokens assigned to an expert are gathered
and computed together before moving on, preserving expert accumulation order.
The gate, up and down operations of an expert remain on the GPU together.

FP32 cuBLAS pedantic math is used, without TF32 or half-precision activation
conversion. GLM's asymmetric gate clamp and the ordering of recurrent state
updates are preserved. GPU failures synchronize pending work before the caller
recomputes the complete operation on CPU. CPU fallback also supports batched
matrix operations.

## Build and controls

The local binary is built and GPU prefill is enabled by default in CUDA builds.
LocalForge was not restarted. Its next normal GLM launch uses this binary.
No change to the Qwen backend or its GPU configuration was made.

```sh
# Local CUDA 12.9 toolkit was already installed. Install matching cuBLAS:
uv pip install --target .toolchains/cublas nvidia-cublas-cu12==12.9.2.10
make -C c CUDA=1 glm53

# Independent kernel checks and model oracle / cache checks:
make -C c CUDA=1 test-glm53-cuda test-glm53
cd c
python3 tools/test_glm53_gpu_parity.py
```

The Makefile also accepts `CUDA_HOME`, `NVCC`, `CUDA_ARCH`, `CUBLAS_HOME` and
`CUBLAS_LIBDIR`. The cuBLAS dependency is isolated under `.toolchains/cublas`.
A CPU-only build remains available with `make -C c CUDA=0 glm53`.
CUDA support here is for the standalone GLM binary and its native serve mode;
embedded segment/edge adapters are explicitly excluded from this backend.

| Setting | Default / effect |
| --- | --- |
| `GLM53_CUDA` | Enabled in a CUDA build. `0` disables CUDA. |
| `GLM53_CUDA_MB` | 12288 MiB maximum for backend-managed allocations, additionally limited by free GPU memory at initialization. |
| `GLM53_CUDA_DEVICE` | Device 0. |
| `GLM53_BATCH_MATH` | Enabled when CUDA initialization succeeds. `1` explicitly enables batching, including CPU fallback; `0` restores scalar math. |
| `GLM53_PREFILL_CHUNK` | Now 1024 tokens; the measurements above used 128. See the proposal 1 follow-up. |

The GPU budget reserves up to 1 GiB (or one quarter of free memory when less)
for CUDA/cuBLAS and other allocations; library/driver memory is not included
in the backend's allocation counter. Resident uploads leave up to 512 MiB
of additional workspace headroom within the backend budget. Operations fall
back to CPU when they cannot fit. This avoids deliberately oversubscribing
VRAM; it cannot prevent another application from allocating memory later.
To restore the previous math without rebuilding, set both `GLM53_CUDA=0` and
`GLM53_BATCH_MATH=0` for the GLM process.

## Validation

- Independent Hugging Face tiny-model oracle: expected teacher-forced and
  greedy tokens at FP32, int8 and int4; converted expert-streaming fixture too.
- GPU kernel tests: all three formats; odd widths and partial quantization
  groups; matrix row slices; output strides across 32 MiB tiles; immutable
  resident reuse; different experts at the same host addresses; asymmetric
  activation clamps; bounded-allocation fallback.
- Model parity: forced cache sizes 1/2/3 and chunks 1/2/7/128, 32 prompt tokens
  and six greedy continuation tokens. Every token matched CPU; maximum logit
  error under 4e-7. Tests require positive GPU call counters when applicable.
- Batched CPU fallback, a 1 MiB GPU allocation budget with FP32 fixture weights, and an unavailable device checked against scalar CPU. CPU-only build and review-stream tests passed.

CUDA statistics from the real GPU probe:

```text
calls=7188 batched=7188 mlp=5385 max_batch=128 uploaded_mb=82049.032 resident_mb=4611.170 peak_mb=4759.151 budget_mb=12884.902 fallbacks=0
```

## Remaining limits

Both paths read exactly **102,997,426,176 bytes (103.0 GB)** of routed experts:
7,276 expert-cache misses and zero hits in this first chunk. The GPU path
uploaded **82.0 GB** in total, while keeping **4.61 GB** of shared weights on
the device. Thus substantial disk and CPU-to-GPU traffic remains despite
the faster math. Both probes recorded zero swap use. Peak sampled process RAM
was approximately 22.68 GiB on CPU and 22.95 GiB with GPU batching.

The final installed binary differs from the probe binary only in permanent
device-error fallback-counter accounting; successful-operation math is unchanged.
Both binary hashes and the final validation commands/results are saved with
the probe artifacts.

Disk reads still dominate part of prefill. This implementation does not overlap
the next disk-read group with the current GPU group. Attention pooling and
the KDA recurrence remain on CPU. Single-token generation uses the existing
CPU path, so the long thinking/answer phase is not accelerated by this change.
Additional speed work should be measured separately with an actual review
packet rather than extrapolating this short prefill comparison.

## Proposal 1 implemented

The default chunk has since increased from 128 to 1024 tokens.
See [larger chunk measurements and controls](glm53-larger-chunks-2026-09-08.md).
The earlier measurements in this document remain historical results.
