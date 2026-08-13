# colib

A from-scratch, pure-C inference engine for the **Qwen3.5 MoE** family and
**Ornith-1.0** (Qwen3.5-based) on consumer hardware — in the spirit of
[colibri](https://github.com/JustVugg/colibri): VRAM, RAM and NVMe treated as
one managed memory hierarchy, int4 weights, expert streaming with a learning
cache, MTP speculative decoding, CPU + CUDA backends, OpenAI-compatible server.

**Status:** Ornith397 grouped-int3 is the owner-selected production precision,
with the accepted expanded-q4 CUDA representation as its release default.
Phases 0–6 pass. Gate 7 remains a measured 32 GiB hardware no-go: Qwen397 is
coherent and safe but reaches **0.772 tok/s** rather than the ≥2 tok/s target.
Ornith35 passes structural, official-GGUF numerical, HTTP tool-use, and
clean-stop controls. Ornith397 converts completely and passes its independent
122-shard audit, coherence, generated-tool, CUDA-residency, and telemetry
controls. Its historical int4 results remain negative: the original full run
sustains **0.527355 tok/s**, a short lossless pilot reaches **0.867369 tok/s**,
and the mandatory full optimized int4 run reaches only **0.631964 tok/s** under
the owner-revised **0.85 tok/s** minimum. The owner accepts the Ornith35 int3
calibration (95.55% token agreement, +10.66% PPL), waived a separate
Ornith397 q3 PPL run, and selected q3 under a **0.70 tok/s** sustained
regression floor.

Gate 8 now passes: the complete controlled Ornith397 q3 retry sustains
**0.836866773 tok/s**, with a **0.8581515 tok/s median** and
**0.773247 tok/s minimum**. Gate 9 also passes. Ornith35 records a
**1.022585286** continuous-batch geometric mean and **0.162146312 s**
cancellation p95; Ornith397 q3 records **1.109807093** and
**1.935733776 s**, respectively. Both models pass exact two-turn web history,
and all accepted CUDA serving evidence records zero host-MoE fallback. The
q3-bound release audit is **20/20 green**, and the final full `make check`
passes. Review of the prospective release tree, an intentional clean release
commit, and the `v0.1` tag remain pending.
The retired full Qwen35/Qwen397 containers and Qwen-only bench scratch were
removed, releasing about 220 GB of reusable ext4 space; the tiny Qwen oracle
and all source/tests/reports remain.

The official 35B Qwen model runs at a median
31.75 tok/s CUDA decode on an RTX 5070 Ti, with 96.17% teacher-forced
next-token agreement over 20×64 real-model positions. The lossless depth-1 MTP
path reaches a median **41.08 tok/s** versus a matched **29.47 tok/s** baseline
on the `Hello` gate prompt (**1.394×**), using confidence admission and a
bit-exact batched int8 shared-expert kernel. All measured speculative streams
match D0 exactly. The mux server now has resident CPU batching, CUDA-resident
GDN/GQA session state, exact warm reload, telemetry, a Qwen-aware OpenAI
gateway, and a rebranded web/CLI surface. The real 35B production web path has
passed two deterministic streamed turns, live two-slot admission, cooperative
cancellation, and immediate slot reuse. The final post-Gate-9 composite
regression passes **21 C executables, 137 Python tests, 10,000 tokenizer cases,
18 web tests, 91-document/111-link/75-report documentation validation, a
273-path package audit after final evidence staging, and both CUDA suites**.
The historical
int4 production-evidence audit remains negative at 7/15, while the selected q3
profile passes its separate 20/20 audit. This remains a tested research build
rather than a production release until the final clean release sequence. See
[PLAN.md](PLAN.md),
[docs/phase5_cuda.md](docs/phase5_cuda.md), and
[docs/phase6_mtp.md](docs/phase6_mtp.md).

Supported model ladder:

| model | total / active | converted payload | role / status |
|---|---|---:|---|
| Qwen/Qwen3.5-35B-A3B | 35B / 3B | 19.93 GB with MTP | correctness and CUDA/MTP reference; passed |
| Qwen/Qwen3.5-397B-A17B | 397B / 17B | 212.63 GB | tier/reference study; 0.772 tok/s negative result |
| deepreinforce-ai/Ornith-1.0-35B | 35B / 3B | 19.08 GB | production-quality control; passed |
| deepreinforce-ai/Ornith-1.0-397B | 397B / 17B | 212.63 GB + 157.07 GB q3 sidecar | q3 selected; Gate 8 passes at 0.836867 sustained / 0.858152 median tok/s |

Text-only; MoE variants only. Apache-2.0. Vendored colibri components: see
[NOTICE](NOTICE).

## Build

```sh
make qwen                         # CPU engine (GCC, ARCH=native)
make test-c                       # 21 C executables
make -C c test-python             # Python integration/unit suite
make -C c test-tokenizer          # 10,000 HF/C parity cases
make test-docs                     # links + research index
make test-source-package           # reject tracked weights/build binaries
make -C c test-web                # built browser -> gateway -> tiny engine
make -C c CUDA=1 CUDA_ARCH=native test-cuda
cd c && SNAP=./qwen_tiny TF=1 ./qwen  # 32-token oracle replay
```

The reference machine uses GCC 13.3 and nvcc 12.9.86. Clang is selected on
macOS; MinGW GCC is supported on Windows for CPU builds. Exact recorded
dependencies, hardware, environment variables, and reproduction commands are
in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

For a system installation:

```sh
sudo make -C c CUDA=1 CUDA_ARCH=native install PREFIX=/usr/local
COLI_MODEL=/models/ornith35 colib doctor
COLI_MODEL=/models/ornith35 colib web --cuda --kv-slots 2
```

The install target copies the launcher, engine, Python support tools, and
built web client. It deliberately does not copy model containers. In the
source tree the default is `c/ornith35`; after installation, either place an
Ornith35 container at the corresponding libexec path or set `COLI_MODEL` to
its actual location. `COLI_ENGINE` and `COLI_PYTHON` provide equivalent
explicit overrides for nonstandard deployments.

The Makefile auto-detects the ignored workspace-local toolkit at
`.toolchains/cuda-12.9`; set `CUDA_HOME` to override it. Build with `CUDA=1`,
then enable inference with `COLI_CUDA=1 CUDA_DENSE=1 CUDA_EXPERTS=1
CUDA_F16=1`. GDN and GQA decode state, grouped routed experts, the shared
expert, and dense projections run on CUDA. `CUDA_EXPERT_GB=N` bounds the
independent expert VRAM LRU (8 GiB by default), while CPU prefill remains the
correct fallback. The measured Gate-5 command and all kill-switches are in
[docs/phase5_cuda.md](docs/phase5_cuda.md).

The launcher explicitly selects the int4 container by default, even if
low-bit research variables are present in its parent environment. The opt-in
q3 path (`--expert-q3`) sets `EXPERT_Q3=1` and expands the packed host payload
to the already benchmarked q4 CUDA representation. The optional native path adds
`Q3_NATIVE=1`; its atlas additionally sets `Q3_ROUTE_ATLAS=1` and
`DECODE_PROTECT=1`, with `RAM_GB=20` and
`CUDA_EXPERT_GB=6` on the reference machine. It keeps q3 packed in VRAM and
repartitions learned hot routes disjointly across RAM and GPU caches after
each request. Do not treat this path as release-qualified until the deferred
native/CUDA, numerical-parity, and real-model performance tests pass. See
[the native-q3 atlas preflight](docs/research/phase8_preflight_10_native_q3_hot_route_atlas.md).

The corresponding deferred-test launch surface is:

```sh
./c/colib web --model c/ornith397 --cuda --expert-q3 --q3-native --q3-route-atlas
```

For an immediately visible, non-release directional benchmark:

```sh
.venv/bin/python -u c/tools/stream_qwen_benchmark.py \
  --model c/ornith397 --expert-q3 \
  --prompt "Reply with exactly: colib q3 ready" \
  --warmup-turns 1 --measured-turns 1 --max-tokens 8
```

The wrapper flushes each decoded piece to the terminal and can atomically save
the JSON record with `--output` and the exact response with `--save-response`.

CPU controls: `GDN_CHUNK=0` selects recurrent prompt prefill, `IDOT=0` disables
activation-int8 expert dots, `KV16=1` halves KV-cache storage, `EXPERT_RAM=N`
bounds resident experts per layer, and `PREFETCH_THREADS=N` controls background
file-page prefetch (0–4). `PREFETCH_LOAD=1` enables the experimental
route-transition loader; it is off by default because the first 35B A/B
increased reads and slowed decode. fp32 KV and the validated compute paths are
the defaults.

## Chat, API, and web UI

The source-tree launcher keeps Phase 9 on the mux protocol and greedy decoding:

```sh
./c/colib doctor --model c/ornith35
./c/colib chat --model c/ornith35 --cuda
./c/colib serve --model c/ornith35 --cuda --kv-slots 2
./c/colib web --model c/ornith35 --cuda --kv-slots 2
```

`web` builds the pinned React/Vite client when needed and serves it from the
same process at `http://127.0.0.1:8000`. `--session-dir /path` enables atomic
warm-session checkpoints. Non-loopback binds require an API key unless
explicitly overridden by the gateway's insecure-bind switch.

`doctor` validates more than file presence. It matches every safetensors index
entry to a bounded shard-header tensor, rejects missing/duplicate/wrong-shard
entries and overlapping offsets, and checks the exact payload-byte total
without reading model data. `--verify-hashes` additionally recomputes every
converter-ledger SHA-256; `--expect-source`, `--expect-source-fingerprint`,
`--expect-source-shards`, `--expect-output-shards`,
`--expect-logical-tensors`, `--expect-physical-tensors`, and
`--expect-data-bytes` turn a preregistered model identity and container
cardinality into executable acceptance checks.

## Convert a pinned Qwen3.5 or Ornith checkpoint

The Phase-4 converter pins one immutable Hub revision, processes one source
shard at a time, drops vision, unfuses routed experts, and deletes each
downloaded source shard only after the output shard and resume state are
durable. The default text-only mixed-precision container is about 19.1 GB.

```sh
cd c
../.venv/bin/python tools/convert_qwen.py \
  --repo Qwen/Qwen3.5-35B-A3B --outdir qwen35 --dry-run
../.venv/bin/python tools/convert_qwen.py \
  --repo Qwen/Qwen3.5-35B-A3B --outdir qwen35

# Coherent ChatML smoke test and performance counters
SNAP=./qwen35 CHAT=1 TEXT=1 PROF=1 PROMPT='Explain delta attention.' NGEN=64 ./qwen

# Loader-only inventory/shape smoke (no forward pass)
SNAP=./qwen35 LOAD_ONLY=1 ./qwen
```

Use `--mtp` to retain the optional MTP block at int8. It can also add MTP to a
completed non-MTP container by downloading and atomically merging only the
source shards that contain `mtp.*`. Use `--max-shards N` to exercise
interruption/resume, and `--keep-source` to retain downloaded bf16 shards.
`tools/eval_qwen.py` and `tools/compare_qwen_prefix.py` implement the
fixed-corpus perplexity and GGUF prefix-agreement gates. Fetch the pinned,
hash-verified public-domain corpus with `tools/fetch_eval_corpus.py`.

Ornith uses the same C architecture but a distinct per-channel FP8 source
format and snapshot-specific chat template. The production-size direct path
is:

```sh
cd c
../.venv/bin/python tools/convert_qwen.py \
  --repo deepreinforce-ai/Ornith-1.0-397B-FP8 \
  --revision 8b61f97a8512d9d01bff1a9625c9a16730e115bb \
  --outdir ornith397 --staging-dir bench/ornith397_stage \
  --xbits int4g128 --io-bits 8 --shared-bits 8 \
  --group-size 128 --min-free-gb 100
```

The exact post-conversion audit, numerical, tier, throughput, and HTTP
tool-use protocol is preregistered in
[Phase 8 Preflight 6](docs/research/phase8_preflight_06_ornith397_qualification_protocol.md).
