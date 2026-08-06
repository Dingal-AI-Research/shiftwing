# Phase 9 Preflight 7: Ornith Production Serving Qualification

**Models:** Ornith-1.0-35B control and Ornith-1.0-397B production target
**Date:** 2026-07-29
**Status:** preregistered before uncontended production results

## Abstract

Earlier Phase 9 work proved mux framing, resident multi-slot state, CUDA
execution, session reload, web streaming, cooperative cancellation, and slot
reuse. The only real-model batch and cancellation timings were collected
while a large checkpoint converter was using CPU, RAM, storage, and network
resources. They are useful diagnostics but cannot close a performance gate.

This preflight fixes the final uncontended protocol for the actual product
family. It uses Ornith's snapshot-specific chat template, two reversed
sequential/concurrent orders, deterministic output comparison, repeated
cancellation trials in one warm engine, and the real web/gateway/C engine
boundary. Qwen remains historical architecture evidence; new Gate 9
production measurements use Ornith35 and Ornith397.

## 1. Terms

**Continuous batching** means advancing multiple independent requests through
one model step together. It should improve total throughput without mixing
their recurrent or KV state.

**AB/BA control** means running sequential then concurrent mode (AB), then
concurrent then sequential mode (BA). Reversing order reduces the chance that
page-cache warmth makes whichever mode ran second appear unfairly faster.

**Geometric-mean speedup** is
\(\sqrt{s_{AB}s_{BA}}\), where each \(s\) is concurrent aggregate throughput
divided by sequential aggregate throughput. It treats reciprocal slowdowns
and speedups symmetrically.

**Cancel-to-ack latency** is time from the first streamed content piece, which
triggers cancellation, until the engine acknowledges `CANCELLED`.
Cancellation is cooperative at decode-batch boundaries rather than a
mid-kernel interrupt.

**Nearest-rank p95** sorts 20 samples and selects sample
\(\lceil 0.95 \times 20\rceil=19\). One extreme observation may therefore
exceed p95, but two cannot.

## 2. Isolation and identity controls

No acceptance timing begins until:

- all conversion and model downloads have stopped;
- no other inference, build, antivirus scan, or storage benchmark is active;
- the model has a complete converter manifest;
- the relevant Gate 8 structural/numerical qualification passes; and
- CUDA event profiling is disabled because earlier work measured substantial
  observer overhead.

Every artifact retains model source, revision fingerprint, tensor/byte counts,
precision map, family, prompt framing, cache budgets, hardware, and resident
telemetry.

The harnesses set `CTX=4096` themselves and force predictive prefetch,
persistent io_uring, pinned-upload treatment, decode protection, grouped
2/3-bit sidecars, and CUDA event profiling off. They also force all CUDA
switches to zero for a declared CPU run. Consequently, inherited shell
variables cannot silently change the frozen profile.

### 2.1 Live resource-plan verification

The read-only doctor was run against the exact preregistered budgets before
performance testing. The observed device was an NVIDIA GeForce RTX 5070 Ti
with 15.92 GiB total and 15.62 GiB currently free; measured available host
memory was 29.38 GiB. Both production plans pass:

| Profile | Slots × context | Predicted VRAM requirement | Predicted RAM requirement | Result |
|---|---:|---:|---:|---|
| Ornith35, 6 GiB expert tier | 2 × 4096 | 9.27 GiB | 11.27 GiB | pass |
| Ornith397, 5 GiB expert tier | 2 × 4096 | 13.62 GiB | 26.62 GiB | pass |

The one-slot Ornith397 Gate 8 plan with a 6 GiB expert tier also passes at
14.20 GiB VRAM and 26.20 GiB RAM. The Ornith397 container was still
incomplete during this calculation, so the doctor correctly returned an
overall warning while independently passing the architecture-derived state
and resource checks. These results establish feasibility only; timing remains
deferred until conversion and download activity have stopped.

## 3. Family-correct prompt boundary

Production harnesses now pass user text through
`colib_model_family`. Ornith reads its pinned `chat_template.jinja`; Qwen
continues to use Qwen ChatML. Raw `"!"` input remains available only through
an explicit `--raw-prompt` diagnostic switch and is rejected by the AB/BA
production comparator.

This matters because engine architecture and model input protocol are
different layers. A correct tensor kernel can still be evaluated unfairly if
the model receives another model's conversation format.

## 4. Continuous-batch AB/BA protocol

For each model, run the two orders with the identical user prompt:

```sh
PROMPT='Reply with exactly: colib batch ready'

.venv/bin/python c/tools/bench_serve_batch.py \
  --model MODEL --prompt "$PROMPT" --expect-exact 'colib batch ready' \
  --requests 2 --max-tokens 8 --context 4096 --warmup-passes 2 \
  --order sequential,concurrent \
  --ram-gb RAM --ram-headroom-gb 1 \
  --cuda --cuda-expert-gb VRAM --cuda-headroom-gb 1 \
  --request-timeout 1800 --output AB.json

.venv/bin/python c/tools/bench_serve_batch.py \
  --model MODEL --prompt "$PROMPT" --expect-exact 'colib batch ready' \
  --requests 2 --max-tokens 8 --context 4096 --warmup-passes 2 \
  --order concurrent,sequential \
  --ram-gb RAM --ram-headroom-gb 1 \
  --cuda --cuda-expert-gb VRAM --cuda-headroom-gb 1 \
  --request-timeout 1800 --output BA.json

.venv/bin/python c/tools/compare_batch_ab.py \
  --ab AB.json --ba BA.json \
  --expect-family ornith-1.0 --expect-source EXPECT_SOURCE \
  --minimum-each 0.95 --minimum-geomean 1.0 \
  --output ABBA.json
```

Use:

| Model | `MODEL` | `RAM` | `VRAM` | `EXPECT_SOURCE` |
|---|---|---:|---:|---|
| Ornith35 | `c/ornith35` | 8 GiB | 6 GiB | `hf://deepreinforce-ai/Ornith-1.0-35B-FP8@1ab57ce0b44950e498a88756f40ad1ed4d0f30ca` |
| Ornith397 | `c/ornith397` | 18 GiB | 5 GiB | `hf://deepreinforce-ai/Ornith-1.0-397B-FP8@8b61f97a8512d9d01bff1a9625c9a16730e115bb` |

Write the first order to `c/MODEL_batch_ab.json`, the reversed order to
`c/MODEL_batch_ba.json`, and the comparator result to
`c/MODEL_batch_abba.json`.

Acceptance requires:

1. both source artifacts pass their lifecycle/resident-CUDA checks;
2. modes occur in the required reversed orders;
3. every response is exactly `colib batch ready`, and all sequential and
   concurrent outputs match within and across artifacts;
4. the prompt is family-rendered rather than raw;
5. each order's aggregate throughput ratio is at least 0.95; and
6. the geometric-mean concurrent speedup is at least 1.00.

The 0.95 per-order floor tolerates modest measurement noise but prevents one
large regression from being hidden by the other order. The geometric mean
requires batching to be non-regressive overall. Absolute Ornith397 decode
speed is separately controlled by Gate 8's ≥2 tokens/s sustained rule.

## 5. Repeated cancellation protocol

The harness keeps one engine warm and runs 20 cancellation/peer/reuse trials:

```sh
.venv/bin/python c/tools/bench_cancel_mux.py \
  --model MODEL --prompt 'Reply briefly: cancellation probe' \
  --max-tokens 8 --context 4096 --warmup-passes 2 --trials 20 \
  --ram-gb RAM --ram-headroom-gb 1 \
  --cuda --cuda-expert-gb VRAM --cuda-headroom-gb 1 \
  --request-timeout 1800 --max-cancel-ack-s LIMIT \
  --output CANCEL.json
```

Use `c/MODEL_cancel_gate.json` for `CANCEL.json`.

Use:

| Model | RAM | VRAM | p95 limit |
|---|---:|---:|---:|
| Ornith35 | 8 GiB | 6 GiB | 1.0 s |
| Ornith397 | 18 GiB | 5 GiB | 3.0 s |

The 35B limit is deliberately much tighter than the 9.88-second contended
observation. The 397B limit allows several decode-boundary intervals at the
separate 2 tokens/s product floor while still rejecting an interactive
multi-second stall distribution.

Every trial must:

- emit exactly one piece before cancellation;
- acknowledge cancellation;
- let the peer produce positive, nonempty output;
- immediately serve a new request in the released slot; and
- remain on device MoE with zero host-MoE fallback.

The artifact records all 20 samples and computes nearest-rank p95. A single
sample cannot be reported as p95.

## 6. Web/gateway/engine streaming protocol

The real bundled web and OpenAI streaming route runs twice per model:

```sh
.venv/bin/python c/tools/smoke_web_qwen.py \
  --model MODEL --cuda \
  --requests 2 --kv-slots 2 --max-tokens 16 --context 4096 \
  --ram-gb RAM --ram-headroom-gb 1 \
  --cuda-expert-gb VRAM --cuda-headroom-gb 1 \
  --timeout 1800 --expect-exact 'colib ready' \
  --output WEB.json
```

Use `c/MODEL_web_gate.json` for `WEB.json`.

Acceptance requires the built page, exact `colib ready` content on both
streamed turns, positive completion-token usage, `[DONE]`, an empty scheduler
afterward, at least two completed admissions, a real CUDA GPU, and a nonempty
VRAM expert tier. `/profile` must also report positive resident-layer and
device-MoE execution, zero host-MoE fallback, and positive router/logit
transfers. The complete converter manifest and model family are stored with
the result.

## 7. Session correctness control

The existing deterministic tests remain the acceptance evidence for state
persistence:

- recurrent matrix, convolution tail, GQA KV, position, hidden state, and
  token history survive atomic disk serialization;
- deliberate checksum corruption is rejected;
- an exact prompt extension regenerates the same eight tokens as fresh
  prefill on CPU and CUDA; and
- a new engine process restores a cancelled slot and emits the same
  continuation.

These are exact correctness tests rather than throughput measurements and may
run in the final release suite after the production benchmarks.

## 8. Manifest-bound controlled execution

`c/tools/run_ornith_gate9.py` packages the ten commands above into a
fail-fast, resumable sequence. It does not weaken or replace any harness
acceptance rule. Before starting, it calls the release evidence auditor and
requires every failure, if any, to be one of the six still-missing Gate-9
artifacts. Therefore an incomplete Ornith397 manifest or a failed Gate-8 PPL,
tier, or tool result prevents all production timing work. The CLI also
requires `--acknowledge-gate8`, making review of the completed 397B evidence
an explicit boundary rather than an inferred side effect of file presence.

For each model, the controller executes AB, BA, the independent comparator,
20-trial cancellation, and two-turn web streaming in that order. A step is
resumed only if:

- both model `quantization.json` SHA-256 values still match the state file;
- the controller's explicit CUDA rebuild still has the recorded engine
  SHA-256;
- its complete command line is unchanged;
- the output artifact still has the recorded SHA-256; and
- the harness returned zero with `acceptance.passed=true`.

After the tenth step, the controller runs the complete release evidence
auditor again and reports Gate 9 passed only when that independent audit has
no failed evidence checks. It does not run `make check`, commit, tag, or
publish a release; those remain Phase 10.

Four controller tests cover the Gate-8 prerequisite, required CUDA build,
exact two-model command matrix, verified resume, artifact tampering, and
manifest/engine rebinding. Together with the AB/BA, cancellation, web, and
release-auditor tests, 18 focused tests pass. The controller is intentionally
not launched while the conversion or Gate-8 measurements are active.

## 9. Gate rule

Gate 9 closes only after both Ornith models pass family-correct AB/BA,
20-trial cancellation, and web streaming acceptance, and the complete session
and server suites remain green. Ornith397 runs only after Gate 8 establishes a
numerically usable production profile. Contended Qwen measurements remain
historical diagnostics and cannot substitute for these results.
