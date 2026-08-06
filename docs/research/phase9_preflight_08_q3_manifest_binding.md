# Phase 9 Preflight 8: Q3 Manifest-Bound Production Harnesses

**Models:** Ornith-1.0-35B int4 control and Ornith-1.0-397B q3 candidate

**Date:** 2026-08-01

**Status:** implementation and focused tests complete; execution prohibited until Gate 8 passes

## Abstract

The reopened Gate 8 may accept an Ornith397 model whose routed experts are
stored in grouped 3-bit sidecar files. The existing Gate-9 production
harnesses deliberately forced every model to use its base int4 experts. If
left unchanged, they could accidentally test the int4 rollback model while
claiming a q3 production result.

This preflight adds explicit, fail-closed q3 selection to the continuous-batch,
cancellation, and web-serving harnesses. Each artifact records a summary and
SHA-256 identity of the complete sidecar manifest. The Gate-9 controller keeps
Ornith35 on int4, selects q3 only for Ornith397, and binds resume state to all
three model identities: Ornith35 int4, Ornith397 int4 base, and Ornith397 q3
sidecar. Fifteen focused manifest, controller, comparator, cancellation, and
web tests pass. No Gate-9 model experiment was run because q3 Gate 8 remains
incomplete.

## 1. Research question

If q3 passes Gate 8, can the unchanged ten-step production-serving experiment
prove that every Ornith397 request actually uses the accepted q3 sidecar?

The required evidence is stronger than setting an environment variable. A
result must identify the exact completed sidecar, both reversed batch orders
must agree on that identity, and a resumed controller must reject any change
to it.

## 2. Terms

**Continuous batch.** Multiple requests occupy separate model-state slots and
advance together through the neural network. The test compares this with
serving the same requests sequentially.

**Cancellation acknowledgement.** The time between requesting cancellation
after the first streamed piece and the engine reporting that the request has
stopped. A peer request must continue, and the released slot must be reusable.

**AB/BA control.** Cache warming can favor whichever benchmark mode runs
second. One experiment runs sequential then concurrent (AB); another reverses
the order (BA). The geometric mean reduces ordering bias.

**Manifest binding.** A controller records a SHA-256 for each model manifest.
It refuses to resume if any identity changes, even if filenames and command
arguments remain the same.

## 3. Fail-closed sidecar loading

`expert_lowbit.py` is a shared metadata boundary used by all three production
harnesses. When `--expert-q3` is requested, it requires:

- `expert-q3.json` to exist and contain a JSON object;
- format `colib-routed-expert-int3-sidecar-v1`;
- `complete=true`;
- configuration and conversion-signature fields; and
- positive layer, file, tensor, and byte counts.

The returned artifact summary includes the manifest's own SHA-256. Full file
and tensor auditing remains Gate 8's responsibility; Gate 9 verifies that the
already-audited identity is the one actually selected for serving.

## 4. Harness changes

The following tools now accept `--expert-q3`:

1. `bench_serve_batch.py` for sequential/concurrent AB and BA measurements;
2. `bench_cancel_mux.py` for the 20-trial peer-aware cancellation test; and
3. `smoke_web_qwen.py` for the exact two-turn web/gateway test.

Each tool sets `EXPERT_Q3=1` only after loading a complete manifest. Its result
contains `expert_lowbit_manifest` and records `expert_lowbit="int3g128"` in
the tier configuration. Schema versions were incremented because this new
identity is part of the evidence contract.

Each model-running harness also starts from the shared isolated engine
environment. Ambient MTP, teacher-force, session, partial-q3, mmap, debug,
cache-policy, and experimental CUDA controls are removed before the recorded
profile is installed. Batch, cancellation, and web artifacts record an
explicit eight-thread OpenMP setting. This prevents shell state from becoming
an unreported AB/BA, latency, or exact-output confound.

`compare_batch_ab.py` now treats the sidecar manifest as an AB/BA identity
field and copies it into the comparator result. A run cannot pass if the two
orders used different sidecars.

## 5. Controller binding

`run_ornith_gate9.py` now declares precision per model:

| Model | Routed-expert profile |
|---|---|
| Ornith35 | accepted int4 base |
| Ornith397 | Gate-8 q3 candidate |

The controller adds `--expert-q3` to exactly four Ornith397 model commands:
AB batch, BA batch, cancellation, and web. The AB/BA comparator consumes the
two recorded artifacts and therefore does not load a model itself. No
Ornith35 command receives the q3 flag.

Resume state includes SHA-256 identities for:

- `c/ornith35/quantization.json`;
- `c/ornith397/quantization.json`; and
- `c/ornith397/expert-q3.json`.

The existing CUDA-engine hash, exact-command binding, artifact hashes, Gate-8
release-audit precondition, and explicit `--acknowledge-gate8` requirement
remain unchanged.

## 6. Verification

Twenty-three focused tests pass. They cover:

- complete sidecar summary and SHA-256 binding;
- rejection of incomplete or incorrectly formatted sidecars;
- exact ten-step Gate-9 command ordering;
- q3 selection on four Ornith397 commands and zero Ornith35 commands;
- controller refusal to resume against different model or engine hashes;
- AB/BA order, family, source, output, and geometric-mean checks;
- deterministic nearest-rank cancellation p95; and
- complete CUDA web lifecycle plus negative content/slot-leak cases.
- a complete synthetic 20-check q3 release evidence graph; and
- rejection when a Gate-9 artifact reports a different sidecar identity.
- ambient engine-control removal with preservation of ordinary process state;
  and
- an explicit eight-thread setting on all eight model-running Gate-9 commands.

These are implementation tests. They do not supply the production throughput,
cancellation, or web evidence that only real model runs can provide.

## 7. Decision boundary

Gate 9 is still prohibited. The q3 conversion, independent audit, numerical
quality tests, coherence review, tool round trip, CUDA-residency checks, and
0.85 token/s Gate-8 threshold must all pass first.

If Gate 8 fails, these q3-capable harnesses remain unused and the int4 rollback
result stays negative. If Gate 8 passes and is explicitly reviewed, run the
same ten production steps with the newly bound q3 identity. No release claim
is permitted until the final evidence auditor also accepts those artifacts.
