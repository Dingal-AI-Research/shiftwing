# Phase 12 experiment 04: ordered grouped-prefill expert I/O

Date: 2026-08-18

## Outcome

A second Ornith397 optimization candidate is implemented behind
`PREFILL_EXPERT_BATCH` and the qualifier option
`--prefill-expert-batch`. The default remains `1`, which preserves the
accepted one-cold-expert-per-submission behavior. The first candidate arm is
`4`.

The tiny mux integration control emits exactly the same tokens and preserves
44 host-cache hits, 30 misses, and 300 physical tensor reads. Combining four
consecutive cold experts reduces persistent `io_uring` submissions from 50
to 41, an 18% reduction. This is implementation evidence only. It is not an
Ornith397 TTFT or tok/s result.

## Rationale

The fresh Ornith397 control attributes roughly 21 seconds of a representative
38-token prompt to expert I/O. Grouped prefill already gathers all prompt
tokens routed to one expert and computes that expert once, but it still calls
the six-record expert loader one expert at a time. The production qualifier
uses `PIPE=1`, `URING=1`, direct I/O, and a persistent ring while disabling
prefetch threads. Submitting several ordered experts together increases ring
depth without speculative reads or additional bytes.

The treatment preserves the original expert-id order. It flushes a pending
host batch before an atlas-resident CUDA expert, protects every member of the
current batch from cache eviction, retains the same CPU q3 matrix path and
route-order output reduction, and clamps the batch to 1--32 and to the actual
per-layer host-cache capacity.

## Change boundary

- `c/qwen.c` batches only cold grouped-prompt expert loads. Decode, router
  selection, shared experts, cached CUDA prefill, attention, sampling, and
  output reduction are unchanged.
- `c/tools/qualify_tiered_model.py` records the selected batch size and
  rejects values outside 1--32.
- `c/tests/test_serve_mux_qwen.py` runs an exact batch-1/batch-4 mux control
  and requires fewer ring submissions with equal semantic output.
- No LocalForge source or configuration is changed.

## Local validation

The focused command is:

```sh
./.venv/bin/python -m unittest \
  c.tests.test_serve_mux_qwen \
  c.tests.test_cli_qwen \
  c.tests.test_openai_server_qwen \
  c.tests.test_qualify_tiered_model \
  c.tests.test_runtime_env
```

All 43 tests pass. The SM120 CUDA backend suite also passes, including
packed-q3 versus expanded-q4 parity at maximum difference 0. The cached atlas
prefill seam still matches its CPU q3 reference at
`1.1641532e-09` maximum difference. Python compilation and
`git diff --check` pass.

The tiny I/O control uses `PIPE=1 URING=1 URING_PERSIST=1`, the standard
`c/qwen_tiny_i4` fixture, and vocabulary-safe prompt bytes
`21 00 01`. Both arms report the same hits, misses, and reads:

| Arm | Batch | Host hits | Misses | Ring batches | Ring reads |
|---|---:|---:|---:|---:|---:|
| control | 1 | 44 | 30 | 50 | 300 |
| candidate | 4 | 44 | 30 | 41 | 300 |

The small cached fixture cannot establish real NVMe latency or TTFT.

## Preregistered real-model protocol

The isolated comparison runs before any promotion and uses the reconstructed,
hash-verified Ornith397 q3 model. Both arms disable the route atlas so this
experiment measures I/O batching rather than its interaction with a separate
candidate. All other controls match: four legacy prompts, 64 decode tokens,
context 4096, eight CPU threads, 18 GB expert RAM, 6 GB expert VRAM, two
warmup passes, deterministic sampling, direct persistent `io_uring`, pinned
uploads, decode protection, and expanded-q4 CUDA execution.

Control:

```sh
./.venv/bin/python c/tools/qualify_tiered_model.py \
  --model c/ornith397 --engine c/qwen \
  --prompt-file c/fixtures/ornith397_legacy_prompts.json \
  --warmup-passes 2 --measured-passes 1 --max-tokens 64 \
  --context 4096 --threads 8 --expert-ram-gb 18 \
  --cuda-expert-gb 6 --minimum-tps 0.70 \
  --uring-persist 1 --pinned-upload 1 \
  --decode-protect 1 --decode-protect-prewarm 1 \
  --expert-q3 1 --q3-route-atlas 0 --q3-native 0 \
  --prefill-expert-batch 1 --cuda-events --output {output}
```

Candidate is identical except:

```sh
--prefill-expert-batch 4
```

Five complete pairs alternate AB/BA order. The atomic trial harness binds the
model, q3 sidecar, prompt fixture, engine, qualifier, and command environment
and resumes only at a verified trial boundary.

## Acceptance

Promotion requires all of the following:

1. Exact emitted outputs and identical per-prompt expert hit, miss, and read
   counts between arms.
2. The 95% upper confidence bound for candidate/control TTFT is below 1.0.
3. The 95% lower confidence bound for candidate/control decode tok/s is at
   least 0.98, with no per-prompt decode regression above 5%.
4. Candidate ring submissions are lower on every prompt with more than one
   cold expert; direct-I/O fallback and extra read bytes remain zero.
5. No OOM, silent CUDA fallback, empty output, session, cancellation, batch,
   tool, web, or quality regression.

The route atlas has its own independent protocol. Neither candidate can
inherit the other's result. If both pass independently, the final combined
configuration still requires an exact-output paired check before either
becomes a default.

## Identity

- repository HEAD:
  `35d9ab86fa2b8f8acbf56c8438413dfb08f2a2a7`
- engine-source fingerprint:
  `4a7e2593485aaf8eead81f2d9f06fcfaea9c1bc4307e07d09ab0aa7075111ebf`
- CUDA engine SHA-256:
  `b47aadfc3ead42b1b31695d52898e36dc4668bef85d2f4f59a4aed0f82f24271`
- qualifier SHA-256:
  `df58026e4c270434e9812ecd7d1ad33c2311b266aee3d31443ce7d1a7aa1cc45`
- mux test SHA-256:
  `fc079594f228c11b063cba9bfd1cf90e41387deabd5b3858016b808c80ee964d`
- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB, driver 591.86.

The working tree is intentionally dirty. These identities must be recaptured
immediately before the first real trial; any drift invalidates them.
