# Phase 10 Preflight 6: Q3 Release Evidence Audit

**Target:** Ornith-1.0-397B grouped-int3 production profile

**Date:** 2026-08-01

**Status:** implementation and boundary fixtures complete; real evidence pending

## Abstract

The historical release auditor evaluates 15 checks for the Ornith35 and
Ornith397 int4 study. The owner-authorized q3 branch adds evidence that cannot
be represented safely by replacing filenames in those checks: a new sidecar
manifest, an independent sidecar doctor, teacher-forced comparison, a separate
coherence run, and an explicit human coherence review.

This preflight adds a separate 20-check q3 audit profile while preserving the
historical 15-check behavior unchanged. The q3 profile binds every artifact to
the accepted Ornith397 base container and exact q3 sidecar. It recomputes the
12% perplexity ceiling and 0.85 token/s threshold, validates all seven
hash-bound Gate-8 pipeline steps, and requires an explicit coherence-review
acknowledgement before tool, serving, or release evidence can pass. A complete
synthetic boundary fixture passes 20/20. Changing only the sidecar byte count
in the Ornith397 web artifact causes the corresponding release check to fail.

## 1. Why the historical audit is insufficient

The old auditor correctly rejects the current int4 production result, but it
also requires `expert_lowbit_manifest=null`. Reusing it for q3 would create one
of two errors:

1. a valid q3 result would always be rejected; or
2. weakening the old predicate could permit an int4 artifact to masquerade as
   q3 evidence.

The solution is an explicit audit mode, not automatic detection. Automatic
detection is unsafe during conversion because an incomplete progress manifest
already exists and must never change the selected release profile.

## 2. Q3 audit inventory

The q3 profile contains 20 checks:

| Evidence group | Checks |
|---|---:|
| Ornith35 and Ornith397 base manifests | 2 |
| Ornith397 q3 sidecar manifest | 1 |
| Ornith35 prefix, PPL, and tool controls | 3 |
| Ornith397 base-container doctor | 1 |
| Ornith397 q3 doctor and coherence-review ledger | 2 |
| Ornith397 q3 prefix, PPL, coherence, tier, and tool gates | 5 |
| Ornith35/397 batch, cancellation, and web gates | 6 |
| **Total** | **20** |

The original profile remains 15 checks. Historical reports and negative
results therefore retain their exact meaning.

## 3. Sidecar identity

The auditor requires the preregistered conversion signature:

- format `colib-routed-expert-int3-sidecar-v1`;
- configuration SHA-256
  `c31964d2d920c10228a40d77afe02b19122b66f7bf3f6eb4b0809ba42527b0d6`;
- source-index SHA-256
  `93f6769bc6d8e2f7d669a8571be1471905a8e6eb487b596dd6836ee08fd2a813`;
- group size 128, three refinement iterations, and 64 experts per file;
- 60 layers, 480 files, and 276,480 tensors; and
- a positive final byte count plus exactly 480 file records.

Every q3 model artifact must repeat the matching sidecar summary. The Gate-8
pipeline and Gate-9 controller additionally bind the full manifest SHA-256.

## 4. Independent structural evidence

The q3 doctor must report all 480 headers and all 480 file hashes verified,
the exact tensor count, byte count, source identity, grouping parameters, and
the same sidecar-manifest SHA-256. The accepted int4 base doctor remains a
separate check because the sidecar depends on that container and does not
replace it.

## 5. Numerical and runtime evidence

The q3 profile requires:

- 20 teacher-forced prompts with 64 positions each, at least 90% aggregate
  agreement, and at least 85% for every prompt;
- finite PPL no greater than
  `1.050067545 × 1.12 = 1.1760756504`;
- four nonempty coherence outputs with q3 selected and complete CUDA telemetry;
- a full two-warm-pass production result at or above 0.85 token/s;
- the exact Paris weather-tool call/result/final-answer behavior; and
- zero host-MoE fallback in all CUDA-resident evidence.

The release audit cannot infer semantic coherence from token counts. It
therefore checks the Gate-8 pipeline ledger for
`coherence_review_acknowledged=true` and validates the artifact SHA-256 for
all seven pipeline steps against the files currently present.

## 6. Gate-9 evidence

For Ornith397, batch comparison, cancellation, and web artifacts must carry
the same q3 sidecar identity and `expert_lowbit="int3g128"`. Ornith35 must
continue to report no low-bit sidecar. Existing AB/BA speedup, 20-trial p95,
exact output, scheduler lifecycle, resource, and resident-CUDA requirements
are unchanged.

`run_ornith_gate9.py` now invokes the q3 audit profile both before execution
and after all ten production steps. Before Gate 8 passes, the 14 expected
missing or incomplete q3/Gate-9 checks keep it fail-closed.

## 7. Verification and limitations

Ten release-auditor tests pass, including the historical 15/15 fixture, the
new q3 20/20 fixture, strict 0.849 token/s rejection, forged base-doctor hash
evidence, forged numerical/tool evidence, cancellation-p95 recomputation, and
mixed q3 web-manifest rejection. The conditional Gate-9 controller tests also
remain green.

The synthetic fixture proves predicate boundaries and identity joins; it is
not model evidence. The real audit must remain negative until conversion,
Gate 8, and Gate 9 produce all required artifacts. `make check`, a clean
release commit, and the `v0.1` tag remain separate final requirements.
