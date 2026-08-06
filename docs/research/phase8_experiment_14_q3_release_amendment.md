# Phase 8 Experiment 14: Q3 Release-Policy Amendment and Reproducible CUDA Path

**Model:** `deepreinforce-ai/Ornith-1.0-397B-FP8` with grouped-int3 routed experts
**Date:** 2026-08-02
**Status:** implementation and focused validation complete; ordered Gate-8 experiment running

## Abstract

The complete Ornith397 grouped-int3 benchmark sustains 0.718432662 token/s
when its three-bit host payload is expanded into the established grouped-four-
bit CUDA kernels. The project owner selected this profile, waived a separate
Ornith397 perplexity experiment, and accepted 0.70 token/s as the sustained
production regression floor. Earlier automation still encoded a 12% perplexity
limit and a 0.85 token/s floor, so it could not represent the accepted product
decision without either rejecting valid evidence or rewriting historical data.

This experiment implements a schema-2 release amendment. It replaces the q3
perplexity measurement with an explicit, machine-checked waiver; changes only
the q3 throughput floor to 0.70 token/s; and preserves manifest integrity,
teacher-forced agreement, semantic coherence, generated tool use, CUDA
residency, Gate 9, and clean-release requirements. It also makes the measured
q3-to-q4 CUDA representation the default and moves newer packed-q3 kernels and
their adaptive route atlas behind explicit experimental switches. Twenty-eight
focused Python tests, the CUDA build, native-q3 kernel comparisons, and CUDA
session tests pass. The full manifest-bound Gate-8 sequence was safely paused
during its initial independent hash audit and has now resumed from that stage.

## 1. Research question

Can the owner-approved q3 product decision be encoded reproducibly without
weakening unrelated safety and correctness gates or changing the meaning of
previous negative experiments?

The amendment must satisfy four constraints:

1. retain the raw 0.718432662 token/s artifact unchanged;
2. record the PPL waiver as policy, not as a fabricated numerical pass;
3. make the released runtime reproduce the CUDA representation used by the
   accepted benchmark; and
4. keep every remaining numerical, semantic, serving, and release check
   fail-closed.

## 2. Methods

### 2.1 Policy representation

`run_ornith397_q3_gate8.py` now writes a schema-2 pipeline ledger containing:

- `q3_ppl_waived=true`;
- `q3_minimum_tps=0.70`; and
- a fixed textual policy basis naming the accepted 0.718432662 token/s
  expanded-q4 result.

The controller no longer runs `ornith397_q3_ppl_gate.json`. It still binds the
historical int4 PPL artifact by SHA-256 so the model/corpus context remains
identifiable, but no q3 PPL value is interpreted as an acceptance condition.
The independent release auditor rejects a missing, false, stale, or ambiguous
waiver and rejects any schema-2 ledger that still claims a `q3_ppl` execution
step.

### 2.2 Preserved controls

The following conditions are unchanged:

- 480 sidecar files, 276,480 tensors, exact hashes and headers;
- at least 90% aggregate and 85% per-prompt teacher-forced agreement;
- four nonempty, CUDA-resident coding/diagnostic outputs;
- explicit human review after those outputs;
- the generated Paris weather-tool call/result/final-answer sequence;
- zero host-MoE fallback in accepted CUDA evidence;
- the complete two-warm-pass, four-prompt, 64-token tier profile;
- all Gate-9 batch, cancellation, and web conditions; and
- a 20/20 q3 release audit, green `make check`, and intentional release commit.

This isolates the product decision to the two quantities actually changed by
the owner: q3 perplexity and sustained throughput.

### 2.3 Reproducible CUDA execution

The accepted benchmark loaded packed q3 tensors from storage but expanded them
to four-bit CUDA bytes before grouped matrix multiplication. Later development
changed the default to newly written packed-q3 kernels, creating an unmeasured
implementation change between evidence and release source.

The runtime now restores the measured behavior as the default. `EXPERT_Q3=1`
selects the q3 sidecar and expands its weights to q4 in the bounded GPU cache.
`Q3_NATIVE=1` explicitly selects packed-q3 CUDA uploads and kernels.
`Q3_ROUTE_ATLAS=1` additionally requires native q3 and decode protection. The
CLI and qualification tools expose and record the same distinction. This
design permits continued optimization while keeping the release candidate
traceable to its accepted benchmark.

## 3. Focused verification

The following command group passed on the Ryzen 7 7700X and RTX 5070 Ti:

```sh
.venv/bin/python -m unittest \
  c.tests.test_run_ornith397_q3_gate8 \
  c.tests.test_audit_release \
  c.tests.test_runtime_env \
  c.tests.test_qualify_tiered_model \
  c.tests.test_cli_qwen
make -C c CUDA=1 CUDA_ARCH=native qwen test-cuda
```

The Python group passed 28 tests. The CUDA backend reported maximum q3 GEMV
differences of `3.87e-07` with fp32 inputs and `6.04e-04` with fp16 inputs;
grouped and batched q3 MoE comparisons were exact. CUDA session restore,
two-slot alternation, disk checkpointing, and exact-prefix extension also
passed.

The amended release auditor currently passes 8 of 20 checks. Its 12 failures
are expected current-state evidence gaps: the waiver-bearing completed Gate-8
ledger, q3 teacher-forced/coherence/tier/tool artifacts, and six dependent
Gate-9 artifacts. This is a useful fail-closed intermediate state, not a
release result.

## 4. Full-model experiment checkpoint

The prior schema-1 pipeline ledger contained only the completed q3 doctor step
and was preserved as `ornith397_q3_gate8_pipeline.pre-policy.json`. A fresh
schema-2 controller was launched against the exact base and q3 manifests. It
rebuilds and hashes the CUDA executable, independently verifies the sidecar,
captures the int4 reference, performs q3 teacher forcing, and generates the
four fixed coherence outputs. It must stop at the semantic-review boundary
before tool and tier qualification unless those exact outputs are reviewed.
The run was deliberately stopped during the read-only 480-file sidecar audit.
No schema-2 state had been published, so the resumed controller repeated that
audit from the beginning; no model artifact or conversion output was modified.
The repeat completed in 1,282.37 seconds and passed all 480 file hashes, all
480 headers, and the complete 276,480-tensor inventory. The controller then
atomically published the schema-2 doctor record and captured the 20-prompt,
64-position int4 reference in 3,879.536 seconds. All 1,280 required positions
were present, the reference acceptance record passed with no failures, and the
artifact SHA-256 is
`a79219e2ce29057238f7a3b01619d53c61412ac2ef25b2d7b7190bd43b459bf0`.
The controller checkpointed this result and advanced to the exact q3
teacher-forced comparison. That comparison completed in 2,935.231 seconds and
matched 1,243 of 1,280 baseline token choices, or 97.109375%. All 20 prompts
cleared the 85% per-prompt floor, the weakest prompt scored 92.1875%, and the
90% aggregate gate passed without failures. Its artifact SHA-256 is
`462dce23cdf046d08cc7f263f97fd8c3a086bcc925add361acddb632c136c5ae`.
The controller then advanced to the four-prompt frozen coherence generation.
Its machine checks passed, but the required semantic review rejected all four
outputs as garbled or incomplete. The controller remains unacknowledged and
Gate 8 remains open. The preserved negative result and its controlled follow-up
are reported in `phase8_experiment_15_q3_cold_start_coherence.md`.

The durable runtime files are:

- `c/bench/ornith397_q3_gate8_pipeline.json`;
- `c/bench/ornith397_q3_gate8_pipeline.log`; and
- `c/bench/ornith397_q3_gate8_pipeline.pre-policy.json`.

## 5. Limitations

The Snake trials provide useful directional coding evidence, and the owner
accepts them as such, but both reached explicit token ceilings. They do not
replace the frozen four-prompt semantic review. Likewise, packed-q3 unit
correctness does not establish whole-model numerical parity or better
throughput. Packed q3 and the adaptive atlas therefore remain experimental and
are not required to close the release gate.

## 6. Conclusion

The release policy now represents the actual owner decision directly and
reproducibly. Historical 0.85/PPL experiments remain valid historical records,
the raw 0.718 benchmark remains unchanged, and the candidate runtime defaults
to the same expanded-q4 CUDA method used to obtain it. Gate 8 is not closed:
the zero-warmup semantic control failed even though its automatic checks
passed, demonstrating why explicit human review is a required conjunct.
