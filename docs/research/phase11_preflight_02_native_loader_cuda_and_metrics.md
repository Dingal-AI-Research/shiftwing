# DeepSeek-V4 native loader, CUDA correctness, and serving metrics

Date: 2026-08-14
Phase: 11.1-11.3 fixture boundary
Disposition: native loader and grouped-expert fixture parity pass; full forward and promotion remain open

## Bound source and execution identity

All work remains pinned to `deepseek-ai/DeepSeek-V4-Flash-0731` commit
`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`. The repository HEAD at this
boundary is `35d9ab86fa2b8f8acbf56c8438413dfb08f2a2a7`; the tree is intentionally
dirty while Phase 11 is implemented. The execution-time DeepSeek engine source
fingerprint is `bd95db91489c52638698b00127c93dfc85f1a1aa12a57709f896912b351856be`.

The pinned runtime contract now also checks DSpark block size 5, Markov rank
256, noise token 128799, target layers 40/41/42, and all 46 compression-ratio
entries. The C loader accepts the original pinned config and the converter's
hash-bound annotations, but rejects architecture or context drift.

## Native manifest loader

`c/deepseek_v4_store.h` adapts the converter manifest to colib's existing raw
segment reader. Each record now retains a validated native storage dtype,
physical shape, layer, expert, and projection. It checks shape multiplication
for overflow, verifies dtype ? physical element count equals the declared byte
range, rejects duplicate names and unsafe paths, and checks every range against
the opened segment before building its hash index.

Header inspection of the pinned checkpoint confirms the relevant physical
representations: BF16, F32, F8_E4M3, F8_E8M0, I8, and I64. Routed FP4 weights
are physically stored as I8 packed-byte tensors?for example
`layers.0.ffn.experts.0.w1.weight` is `[2048, 2048]`, representing a logical
`[2048, 4096]` E2M1 matrix. The loader test exercises real I8/UE8M0 descriptor
semantics, direct reads, batched reads, byte counters, and persistent io_uring.

Command and result:

```text
make -C c tests/test_deepseek_v4_store
c/tests/test_deepseek_v4_store
DeepSeek-V4 manifest store tests: ok
```

The layout audit also caught a pre-conversion grouping defect: global
`hc_head_fn`, `hc_head_base`, and `hc_head_scale` are mandatory main-model
weights, not optional draft weights. They now remain in base dense segments;
only `mtp.*` records enter DSpark segments. The corrected index categories are
1,564 base dense, 66,048 base routed-expert, 97 DSpark dense, and 4,608 DSpark
expert tensors. A synthetic regression test proves `hc_head_fn` cannot migrate
back into the DSpark group.

An exhaustive converter-side contract now enumerates all 72,317 expected
tensor names, native storage dtypes, and physical shapes: 43 base blocks, their
alternating 4x/128x compressor/indexer modules, and all three DSpark stages and
heads. Real conversion calls this contract after reading all headers and before
writing the first segment. Its drift fixture passes, and a partial real audit
of the 17,287 tensor headers currently downloaded reports zero mismatches.

The reproducibility fingerprint was correspondingly widened to cover the
launcher, gateway, converter, exact layout contract, pinned spec, official
protocol/encoder, native loader/session, and CUDA code. Its value at this
boundary is `8006a58ae538d115d7c188052b42122488b7a7866bb93bc92082efb15b209359`.

## Scalar and SM120 expert correctness

The dependency-free scalar oracle covers native E2M1, E4M3FN, and UE8M0,
MXFP activation quantization, FP8/FP4 projections, BF16 normalization, the
complete routed expert, sqrt-softplus/hash routing, four-way mHC/Sinkhorn,
YaRN/RoPE, window wrap, 4?/128? compression indices, learned pooling, DSpark
indices, and attention sinks.

A comparison with the pinned `Expert.forward` caught and fixed a material
ordering requirement: the route weight multiplies the clamped SwiGLU hidden
state before the dynamically quantized `w2` projection. The CPU oracle and CUDA
path preserve this order so routing participates in the FP8 scale selection.

The SM120 path now includes native FP8/FP4 GEMMs plus a grouped top-k expert
operation spanning `w1`, `w3`, clamped SwiGLU, route-aware E4M3/UE8M0
requantization, and `w2`. It reuses persistent context allocations and accepts
host arrays of device-resident, converter-native weight/scale pointers.

Commands and measured result on NVIDIA GeForce RTX 5070 Ti, compute capability
12.0, CUDA 12.9:

```text
make -C c CUDA=1 CUDA_ARCH=sm_120 tests/test_deepseek_v4_cuda
c/tests/test_deepseek_v4_cuda
DeepSeek CUDA sm_120: fp8=0 fp4=0 swiglu=1.19e-07
DeepSeek grouped FP4 experts: absolute=0 relative=0
DeepSeek actual 4096x2048 top-6 FP4 experts: absolute=0 relative=0
```

The actual-shape control uses six native packed expert triplets at the pinned
4096 hidden / 2048 intermediate dimensions and validates every output row.
This establishes kernel correctness only. The kernels are not yet the final
tensor-core implementation, and no tok/s or TTFT improvement is claimed.

## Serving interfaces and metrics

The DeepSeek family uses the hash-pinned official encoding/DSML implementation,
rejects unsupported reasoning/sampling profiles, and exposes 16,384 default /
65,536 maximum context plus `--dspark auto|on|off`. The native engine remains
fail-closed for serving until its 43-layer quality gate passes.

The gateway now emits a final schema-1 `colib.metrics` event containing queue,
TTFT, prefill, decode and total time; true engine decode tok/s; prompt and
completion counts; cache hits/misses; and disk/direct bytes. The non-streaming
response carries the same object. The web client consumes this final engine
telemetry and no longer estimates TTFT or throughput from text chunks.

At this boundary, 23 C executables, the focused 35-test Python protocol/tooling
set, 19 web tests, and the production web build have passed. A complete release
suite will be rerun only after the remaining native-forward changes settle.

## Storage and active long operation

The tracked atomic preflight at `2026-08-14T15:37:26.304642+00:00` passes with:

| Quantity | Bytes |
|---|---:|
| Free before the recorded step | 496,527,085,568 |
| Source bytes then remaining | 137,214,445,429 |
| Planned native container | 167,174,674,555 |
| Planned staging | 5,000,000,000 |
| Projected peak new storage | 309,389,119,984 |
| Projected final free | 187,137,965,584 |
| Required floor | 107,374,182,400 |

At `2026-08-14T15:57:24.729702+00:00`, the resumable fetch process had run for
1:19:21 and committed 23/61 files atomically. Available filesystem space was
491,534,630,912 bytes. No Qwen or Ornith file has been deleted; Ornith397 stays
protected for the fresh paired control.

## Versioned session snapshot boundary

A subsequent same-day step added `c/deepseek_v4_session.h`. Its atomic schema
binds independent 32-byte model-manifest, tokenizer/protocol, and native-engine
fingerprints and persists position/context, token history, four-way mHC state,
window KV, compressed KV, compressor buffers, sampling profile, and sampler RNG
state. Reads reject the wrong context or any identity drift, verify exact file
length before allocation, enforce a 2 GiB safety ceiling, and checksum header
metadata plus every state section. Writes use a same-directory temporary file,
`fsync`, atomic rename, and directory `fsync`.

```text
make -C c tests/test_deepseek_v4_session
c/tests/test_deepseek_v4_session
DeepSeek-V4 session snapshot tests: ok
```

The DeepSeek engine source fingerprint after adding this schema is
`9ed2f356dd9632b6b81f79b8b218efe3665170a37839e4151b7e60a3c25b3415`.
Live mux slot wiring remains open until the complete native runtime state exists.

## Open gate

The 48-shard fetch and interrupted real conversion/resume proof remain open,
as do the full attention/indexer/shared-expert/block forward path, versioned
DeepSeek sessions, real-model oracle/quality, 64K soak, DSpark A/B, paired
Ornith comparison, tensor-core performance work, and all promotion actions.
The CLI and web defaults therefore remain on the qualified rollback model.


## Interrupted fetch resume and compiled full-layout boundary

At `2026-08-14T16:21:19.385230+00:00`, the single-worker downloader (Linux PID
17960) was stopped with `SIGTERM` at an atomic file boundary and restarted as
Linux PID 60971 with the exact command:

```text
./.venv/bin/python c/tools/fetch_deepseek_v4.py \
  --output c/.deepseek-v4-flash-0731.source --workers 4
```

The restart reused the same `colib.deepseek-v4.fetch-state.v1` ledger and all
completed hashes/partial ranges. It also migrated the metadata plan from 61 to
62 files to include pinned `inference/kernel.py`. At
`2026-08-14T16:34:13+00:00`, the process remained healthy with 28/62 committed
files (14/48 complete weight shards), 53,961,005,337 source-directory bytes,
and 441 GiB filesystem space available. No Qwen or Ornith artifact was removed.
The full fetch and conversion gate remain open.

The native loader now exposes typed bounded operations over those future
converted records: one BF16 embedding row expands to all four hC copies, one I64
hash-router row selects the token's six experts without reading the 6.2 MiB
route table, and a CPU-correct output-head fallback scans contiguous BF16 row
blocks. The fixture checks exact values, bounds failures, and disk/direct-byte
accounting.

A startup complexity audit found that duplicate detection was linear for every
inserted manifest record. It now builds the open-addressed hash table before
record ingestion and inserts each name incrementally. An exhaustive sparse
fixture generated from `deepseek_v4_layout.py` then binds all 72,317 records to
the compiled C contract?43 base blocks, 3 DSpark blocks, compressors/indexers,
heads, and all experts?in 1.415 seconds. Changing `embed.weight` to an unknown
name is rejected with `tensor contract mismatch: embed.weight`.

```text
./.venv/bin/python -m unittest \
  c.tests.test_deepseek_v4_tooling.DeepSeekLayoutTests.test_native_contract_accepts_all_pinned_descriptors
.
Ran 1 test in 1.415s
OK
```

The DeepSeek source fingerprint at this boundary is
`03710780ed22150ff4795141a080931e7c510eb9c5191c82e94eb66d4953c3db`.
It now covers the semantic model reader plus fetch, preflight, inventory,
fixture/oracle, converter, protocol, session, C/CUDA, gateway, and launcher
sources. The native engine invokes the full descriptor contract whenever a
converted manifest is present, but serving remains fail-closed until the real
forward/quality gates pass.


## Stateful compression and tiered MoE boundary

The scalar reference now replays the official decode-time `Compressor.forward`
transaction over two complete windows. Ratio-4 overlap pools the preceding
window's first learned projection half with the current window's second half;
non-overlap pools the current window only. Learned APE scores, emitted values,
and final KV/score buffers match the hash-pinned reference, and a repeated or
non-contiguous position is rejected. The ratio-4 learned indexer also matches
per-head ReLU dot scores, signed head-weight aggregation, top-k order, and
compressed-cache offsets.

The exact physical layout establishes this RAM/storage split:

| Category | Bytes | GiB |
|---|---:|---:|
| Base dense | 8,845,959,388 | 8.238 |
| Base routed experts | 147,169,738,752 | 137.063 |
| DSpark dense | 595,190,812 | 0.554 |
| DSpark routed experts | 10,267,656,192 | 9.563 |

`c/deepseek_v4_dense.h` therefore loads the base dense set once into a
64-byte-aligned, byte-preserving RAM arena with strict production record/byte
counts. `c/deepseek_v4_tier.h` retains routed experts in a bounded per-layer
LFRU. A cold expert reads its contiguous `w1/s1/w2/s2/w3/s3` records through
one direct-I/O/io_uring request group; grouped prompt acquisition submits all
unique cold experts together. Heat survives eviction, active references are
protected, kernel prefetch is available, and a warm hit reads zero bytes.

A complete tiny MoE fixture now executes directly from the manifest:
BF16 gate projection, token-specific I64 hash routing, normalized route
weights, grouped native-FP4 routed experts, the aligned-RAM native-FP8 shared
expert, and the final sum. The warm second execution performs no model read and
matches the exact expected result. The same routed entries are accepted by the
already-tested SM120 grouped kernel; live device-LRU wiring remains open.

At this boundary the complete regression results are:

```text
make -C c test-c ARCH=x86-64-v3
# 25 C executables passed
make -C c test-python ARCH=x86-64-v3
# Ran 160 tests in 27.460s ? OK
./.venv/bin/python c/tools/check_source_package.py --root .
# tracked=278 failures=0
./.venv/bin/python c/tools/check_docs.py --root .
# documents=94 local_links=114 research_reports=78
```

The source fingerprint is
`38555104f5081946fb0ed6b5f323a6df15a4879865649d99e40f7798273e1e32`.
At `2026-08-14T16:48:35.567495+00:00` the four-worker fetch had committed
32/62 files, including 18/48 weight shards, with four durable partial ranges
and 426 GiB free. No cleanup or model deletion occurred. The full forward,
real conversion/oracle, 64K soak, performance comparison, and promotion gates
remain open.


## Three-mode attention and block-assembly boundary

The native scalar reference now assembles every attention mode in the pinned
43-layer schedule rather than testing compression and selection only as
isolated primitives:

- layers 0-1 use the pure 128-token sliding MLA path;
- even layers 2-42 use the ratio-4 overlapping main compressor plus the
  independent learned FP4-simulated indexer and at most 512 compressed keys;
- odd layers 3-41 use the ratio-128 non-overlapping compressor and all causally
  visible compressed keys.

Each path performs the full FP8 low-rank Q projection and normalization, main
KV projection/QAT, cache transaction, sink-aware sparse attention, inverse
RoPE, grouped low-rank output projection, and final dense projection. The
ratio-4 implementation advances its main compressor and indexer atomically and
rejects diverged positions. `deepseek_v4_block.h` now binds all three modes to
the same four-way hC attention stage and complete tier-backed MoE stage with
explicit checks against the official layer schedule.

Inspection of the pinned `inference/config.json` also caught a fixture-only
constant error before real-model execution: compressed attention and the
indexer now use the official `compress_rope_theta=160000`,
`original_seq_len=65536`, and YaRN factor 16, not the earlier 40000
placeholder. All attention binaries were rebuilt after the correction.

```text
make -C c test-c ARCH=x86-64-v3
# 28 C executables passed
c/tests/test_deepseek_v4_attention
c/tests/test_deepseek_v4_indexer
c/tests/test_deepseek_v4_indexed_attention
# all passed
```

The registered DeepSeek source fingerprint at this boundary is
`6c7af31e9f2a489e0f57d90b17e0dea208600ec5bd140d9cb104ab4b97a9631a`.
At `2026-08-14T17:33:46.980493521Z`, downloader PID 60971 remained the only
active fetcher. Its atomic ledger was `running` with 41/62 files committed,
including 27/48 complete weight shards and four durable partial ranges.
Filesystem space was 400 GiB available, so no cleanup or Qwen/Ornith deletion
was performed.

This closes scalar attention composition, not Gate 11.1: the live engine still
needs its bounded 43-layer state/scratch allocator, embedding-to-logits decode
loop, mux scheduling, snapshots, and pinned real-model oracle agreement.
