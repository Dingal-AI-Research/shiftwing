# Reproducible Environment

This document records the environment used for the current colib research
results. It distinguishes required version constraints from the specific
reference machine; a benchmark is not assumed portable merely because the
code compiles elsewhere.

## Reference machine

Recorded 2026-07-25:

| Component | Reference value |
|---|---|
| host | Windows with WSL2 |
| guest OS | Ubuntu 24.04.4 LTS |
| kernel | Linux 6.6.87.2-microsoft-standard-WSL2 |
| CPU | AMD Ryzen 7 7700X, 8 cores / 16 threads |
| CPU ISA used | AVX2, AVX-512, AVX-512 VNNI (`ARCH=native`) |
| RAM visible to WSL | approximately 29.4 GiB |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| compute capability | 12.0 (`sm_120`) |
| VRAM | 16,303 MiB reported by `nvidia-smi` |
| NVIDIA driver | 591.86 |
| compiler | GCC 13.3.0 |
| CUDA toolkit | nvcc 12.9.86 |
| Python | 3.12.3 |
| Node.js / npm | 22.22.1 / 10.9.4 |

The current workspace-local CUDA toolkit is
`.toolchains/cuda-12.9`. It is ignored by Git and is not part of the source
artifact.

## Python oracle and converter

The checked workspace virtual environment contains:

| Package | Version |
|---|---:|
| PyTorch | 2.13.0+cpu |
| Transformers | 5.14.1 |
| safetensors | 0.8.0 |

Transformers 5.14.1 is a correctness dependency for oracle regeneration: the
architecture report and deterministic fixtures were derived from that
implementation. The runtime C engine does not import Python or PyTorch.

Create the environment with the project's existing lock/input files when
available, then verify:

```sh
.venv/bin/python - <<'PY'
import torch, transformers, safetensors
print(torch.__version__)
print(transformers.__version__)
print(safetensors.__version__)
PY
```

The full 397B converter uses CPU PyTorch/safetensors and Hugging Face Hub
downloads. It deliberately processes one source shard at a time.

## Active model progression

Recorded 2026-07-29:

- Qwen35 is the accepted correctness/CUDA/MTP reference.
- Qwen397 conversion and qualification are complete; its 0.772 tok/s
  sustained result closes Gate 7 negatively on this 32 GiB machine.
- The incomplete Qwen397 grouped-3-bit conversion is retired and must not be
  resumed or auto-selected.
- Ornith35 is the accepted family-specific numerical and tool-use control.
- Ornith397 is the direct production target and is converting from pinned
  revision `8b61f97a8512d9d01bff1a9625c9a16730e115bb`.

Lower-bit work is conditional: first qualify the complete direct
Ornith397 int4-g128/int8 profile. Only a measured tier-capacity or storage
bottleneck can trigger a fresh Ornith35 grouped-3-bit quality gate and then
an Ornith397-only sidecar.

## Build configurations

CPU reference:

```sh
make -C c clean
make -C c qwen ARCH=native
```

Portable x86-64-v3 release check:

```sh
make -C c clean
make -C c qwen ARCH=x86-64-v3
make -C c test-c ARCH=x86-64-v3
```

CUDA reference:

```sh
make -C c CUDA=1 CUDA_ARCH=native qwen test-cuda
```

At runtime the CUDA backend is explicit:

```sh
COLI_CUDA=1 CUDA_DENSE=1 CUDA_EXPERTS=1 CUDA_F16=1 \
CUDA_EXPERT_GB=6 SNAP=c/ornith35 ./c/qwen
```

The `colib ... --cuda` commands set the first four switches. They do not enable
`CUDA_PROFILE_STAGES`; device-event profiling is deliberately opt-in because
it perturbs short multi-process tests substantially.

## Complete test matrix

From the repository root:

```sh
make test-c
make -C c test-python
make -C c test-tokenizer
make test-docs
make test-source-package
make -C c CUDA=1 test-cuda

cd web
npm ci
npm test
npm run build
npm audit
```

The earlier complete uncontended run on 2026-07-31 recorded:

- 21/21 C test executables;
- 114/114 Python tests;
- 10,000/10,000 tokenizer cases;
- 18/18 web tests;
- 82 Markdown documents, 101 local links, and 66 indexed research reports;
- 235 tracked prospective-release paths with zero package violations;
- the CUDA backend and CUDA session suites on the RTX 5070 Ti; and
- zero npm audit vulnerabilities.

An earlier expanded Python discovery before release hardening passed 98/98
tests. A later 102-test run under active multi-gigabyte shard writes
completed 101 tests and hit the strict engine-close timeout once; the
identical expert-map persistence test passed unchanged in 7.648 seconds when
rerun alone. That discovery set contained 114 tests after adding
manifest/engine-bound controllers, structural-doctor evidence, documentation,
source-package fixtures, the strict 0.849 production-threshold boundary, and
the bounded conversion-prefetch controller. Its
new 29-test focused release/controller/prefetch subset passes. That
uncontended composite run confirmed all 114 tests together.

The final post-Gate-9 rerun on 2026-08-06 supersedes those repository counts:

- 21/21 C test executables;
- 137/137 Python tests;
- 10,000/10,000 tokenizer cases;
- 18/18 web tests;
- 91 Markdown documents, 111 local links, and 75 indexed research reports;
- 273 tracked prospective-release paths after final evidence staging, with
  zero package violations;
- the CUDA backend and CUDA session suites on the RTX 5070 Ti, including exact
  native-q3 versus expanded-q4 Ornith-shape grouped-MoE output; and
- a successful production web build with zero npm audit vulnerabilities.

The combined release regression is:

```sh
make check
```

It starts from a clean tree, builds the portable x86-64-v3 executable, runs
the native, tokenizer, Python, and browser checks, builds and audits the web
application, resolves all local documentation links, checks every research
report is indexed, rejects generated/model/native artifacts in Git's index,
and automatically rebuilds and tests CUDA when `nvcc` is available. It does
not replace Ornith397 production evidence. The historical 2.0 tok/s and later
≥0.85 tok/s int4 criteria failed; the separate owner-approved q3 profile later
passed Gate 8 at 0.836866773 tok/s, passed Gate 9, and passed its 20/20 release
audit before the final composite rerun above.

## Production-model controls

Qwen35 container audit:

```sh
./c/colib doctor --model c/qwen35 --kv-slots 2 --context 4096 --json
```

Expected current inventory:

- 14/14 parsed shards;
- 64,646 indexed and header tensors;
- 19,929,665,806 tensor payload bytes;
- index `metadata.total_size` equal to the header sum.

Ornith397's final audit additionally rehashes every committed output and
turns its preregistered identity into executable checks:

```sh
./c/colib doctor --model c/ornith397 \
  --kv-slots 1 --context 4096 \
  --cuda-expert-gb 6 --ram-cache-gb 18 \
  --runtime-headroom-gb 1 --verify-hashes \
  --expect-source \
    hf://deepreinforce-ai/Ornith-1.0-397B-FP8@8b61f97a8512d9d01bff1a9625c9a16730e115bb \
  --expect-source-fingerprint \
    4b1c5ab0824e25b3768e3c6df8392394194fce86c4e9f4f3da24e3134ccf4c94 \
  --expect-source-shards 122 \
  --expect-output-shards 122 \
  --expect-logical-tensors 93078 \
  --expect-physical-tensors 278152 \
  --expect-data-bytes 212634789241 --json
```

The remaining numerical, tier, coherence, and HTTP tool commands are fixed in
[`phase8_preflight_06_ornith397_qualification_protocol.md`](research/phase8_preflight_06_ornith397_qualification_protocol.md).

Web qualification:

```sh
.venv/bin/python c/tools/smoke_web_qwen.py \
  --model c/ornith35 --cuda --requests 2 --kv-slots 2 \
  --max-tokens 16 --context 4096 \
  --ram-gb 8 --ram-headroom-gb 1 \
  --cuda-expert-gb 6 --cuda-headroom-gb 1 \
  --expect-exact 'colib ready' --output c/ornith35_web_gate.json
```

Sequential/concurrent and cancellation studies:

```sh
.venv/bin/python c/tools/bench_serve_batch.py \
  --model c/ornith35 --cuda --requests 2 --max-tokens 8 --context 4096 \
  --prompt 'Reply with exactly: colib batch ready' \
  --expect-exact 'colib batch ready' --warmup-passes 2 \
  --ram-gb 8 --ram-headroom-gb 1 \
  --cuda-expert-gb 6 --cuda-headroom-gb 1 \
  --order sequential,concurrent --output c/ornith35_batch_ab.json

.venv/bin/python c/tools/bench_cancel_mux.py \
  --model c/ornith35 --cuda --max-tokens 8 --context 4096 \
  --warmup-passes 2 --trials 20 \
  --ram-gb 8 --ram-headroom-gb 1 \
  --cuda-expert-gb 6 --cuda-headroom-gb 1 \
  --max-cancel-ack-s 1 --output c/ornith35_cancel_gate.json
```

The reversed batch order, comparator, and corresponding Ornith397 commands
are fixed in
[`phase9_preflight_07_ornith_production_qualification.md`](research/phase9_preflight_07_ornith_production_qualification.md).

Do not compare performance runs while conversion, antivirus scanning, model
copying, or another inference process is contending for storage or page cache.
Every research report must state whether those controls held.

## Important defaults and experimental switches

Production defaults:

- `SERVE_RESIDENT=1`: slot-major recurrent/KV state;
- greedy decoding in the Phase-9 mux;
- fp32 KV unless `KV16=1` is selected;
- chunked GDN prompt prefill unless `GDN_CHUNK=0`;
- CUDA stream-ordered allocation on devices with memory-pool support;
- predictive expert prefetch disabled.

Experimental/diagnostic:

- `CUDA_PROFILE_STAGES=1`: CUDA-event fused-MoE suffix in `PERF`;
- `SERVE_SUFFIX_BLOCK=1`: causal block continuation, currently slower for the
  measured short 35B chat suffix;
- `PREFETCH_LOAD=1`: learned route-transition prefetch, rejected by the first
  35B A/B and disabled by default;
- `CUDA_ASYNC_ALLOC=0`: tested synchronous CUDA-allocation fallback;
- `URING_PERSIST=1`: reuse a thread-local ring and aligned direct-I/O arena;
  rejected by the first same-atlas 397B A/B and disabled by default;
- `CUDA_PINNED_UPLOAD=1`: stage routed experts through one bounded
  pinned-host arena; promising phase timings but no decode-rate win in the
  first same-atlas 397B A/B, so disabled by default;
- `DECODE_PROTECT=1`: protect up to `host-cap - top-k` experts learned during
  decode from the following prefill; reaches 1.296 tok/s on the repeated
  four-token 397B probe but only 0.772 tok/s over 64 tokens;
- `DECODE_PROTECT_PREWARM=1`: materialize protected device-only experts in
  RAM between requests; paired with `DECODE_PROTECT=1`, reaches 2.736 tok/s
  and zero decode reads for the four-token probe, but remains opt-in because
  the sustained working set does not fit;
- `EXPERT_Q3=1`: select the complete grouped-int3 routed-expert sidecar;
- `Q3_ROUTE_ATLAS=1`: unvalidated native-q3-only adaptive hot-route
  partitioning. It requires `EXPERT_Q3=1`, `COLI_CUDA=1`, `CUDA_EXPERTS=1`,
  and `DECODE_PROTECT=1`. The deferred reference profile uses `RAM_GB=20`
  and `CUDA_EXPERT_GB=6`; it must not be treated as release-qualified until
  the tests in `research/phase8_preflight_10_native_q3_hot_route_atlas.md`
  pass;
- `COLI_ENGINE_TRACE=1`: prompt-free mux request/response headers on stderr.

## Reproducibility limits

Token correctness gates are deterministic. Wall-clock throughput is sensitive
to driver version, WSL memory allocation, CPU frequency, expert-cache warmth,
filesystem/page cache, storage contention, and CUDA event instrumentation.
The recorded RTX 5070 Ti results are not performance promises for a 3090 or
another architecture; each GPU needs its own warm controlled qualification.
