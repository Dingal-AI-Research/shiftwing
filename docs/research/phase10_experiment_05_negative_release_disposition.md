# Phase 10 Experiment 5: Negative Release Disposition

**Models:** Ornith-1.0-35B and Ornith-1.0-397B

**Reference hardware:** Ryzen 7 7700X, 29.375 GiB RAM, RTX 5070 Ti with
15.92 GiB VRAM

**Date:** 2026-07-31

**Status:** roadmap closed; production release rejected

## Abstract

This study asks whether the completed `colib` implementation satisfies its
frozen production-release criteria. The answer is no. Both Ornith containers
have the expected checkpoint identities and tensor inventories. The 397B
container passes an independent 122-shard header, ledger, hash, and resource
audit; its fixed numerical smoke reaches finite perplexity 1.050067545; and
its direct qualification produces coherent output with the intended CUDA
resident graph. However, its preregistered four-prompt decode rate is
0.527355 token/s against a minimum of 2 token/s. The best bounded lossless
pilot reaches 0.867369 token/s, and the final exact capacity experiment
remains short by 135 routed experts.

The strict release auditor consequently passes 7 of 15 checks and rejects
eight. The production Gate-9 controller independently refuses to begin its
ten-step serving sequence, even when supplied the required review
acknowledgement. This is the intended fail-closed behavior. Phases 8–10 are
therefore closed with negative or not-authorized outcomes, the 2 token/s rule
is unchanged, and no `v0.1` tag is created.

## 1. Research question

Do the final structural, numerical, performance, tool-use, continuous-batch,
cancellation, web, and repository artifacts jointly satisfy the previously
defined production release criteria?

The null hypothesis is that at least one mandatory criterion remains false.
The release hypothesis requires all 15 independent evidence checks, the
composite repository gate, a clean release commit, and the `v0.1` tag. These
requirements are conjunctive: a strong numerical result cannot compensate
for a failed throughput result, and an unexecuted dependent experiment cannot
be counted as a pass.

## 2. Method

### 2.1 Frozen model evidence

The audit reads the live conversion manifests rather than trusting names in a
report. The Ornith35 identity requires 16 output shards, 31,333 logical and
93,277 physical tensors, and 19,081,810,684 payload bytes. Ornith397 requires
122 output shards, 93,078 logical and 278,152 physical tensors, and
212,634,789,241 payload bytes. Both use routed int4-g128, int8 input/output and
shared-expert weights, and exclude MTP.

The direct Ornith397 criterion remains the preregistered no-low-bit profile:
two warm-up passes, one measured 64-token pass for each of four prompts,
4,096-token context, 18 GiB host experts, 6 GiB device experts, 1 GiB
headroom, resident CUDA telemetry, coherent nonempty outputs, and at least
2 token/s sustained.

### 2.2 Independent release audit

The following command regenerated the final audit atomically:

```sh
.venv/bin/python c/tools/audit_release.py \
  --root . --output c/release_audit.json
```

The auditor recomputes thresholds and model bindings for 15 checks rather
than accepting a top-level Boolean. Its result has SHA-256
`196897d40700eefc445775d0dc06e12bde399691e8b03150aef9be628d192d79`.

### 2.3 Dependent Gate-9 preflight

The production controller was invoked with its explicit human-review flag:

```sh
.venv/bin/python c/tools/run_ornith_gate9.py --acknowledge-gate8
```

This call is a preflight, not an authorization to ignore a failure. The
controller first removes the six expected Gate-9 artifact failures from the
audit result and rejects any remaining pre-Gate-9 failure. It performs that
check before rebuilding CUDA or launching a benchmark.

### 2.4 Decision rule

A production release is accepted only if all 15 evidence checks pass, the
full repository gate passes, the release commit is clean, and `v0.1` is
tagged intentionally. If the direct tier fails, the ordered Ornith397 tool
step and Gate 9 remain unexecuted. Missing artifacts from those prohibited
steps are evidence of dependency enforcement, not evidence that their
underlying behavior failed.

## 3. Results

### 3.1 Release-audit result

Seven checks pass:

| Check group | Result |
|---|---|
| Ornith35 and Ornith397 manifests | pass; exact pinned identities and inventories |
| Ornith397 independent doctor | pass; 122-shard header/ledger/hash/resource audit |
| Ornith35 teacher-forced, PPL, and generated tools | pass |
| Ornith397 PPL corruption smoke | pass; PPL 1.050067545 |

Eight checks fail or are unavailable by design:

| Check | Result and interpretation |
|---|---|
| `ornith397.tier` | fail; direct acceptance is false at 0.527355 token/s |
| `ornith397.tools` | absent; ordered supervisor stopped after tier failure |
| `ornith35.batch`, `.cancel`, `.web` | absent; Gate 9 was not authorized |
| `ornith397.batch`, `.cancel`, `.web` | absent; Gate 9 was not authorized |

The audit summary is therefore 7/15 passed and 8/15 failed, with overall
`passed: false`. The associated Ornith397 qualification artifact has SHA-256
`8f50265f4bcb531fcb4b314381822712996e9fafe21e849006b0d6ffed476509`.

### 3.2 Throughput evidence

| Treatment | Sustained decode | Fraction of 2 token/s target | Decision |
|---|---:|---:|---|
| frozen direct four-prompt profile | 0.527355 token/s | 26.37% | reject |
| best bounded non-lossy pilot | 0.867369 token/s | 43.37% | reject |
| Qwen397 best long control | 0.772 token/s | 38.60% | Gate 7 negative |

The best Ornith397 pilot is 64.5% faster than the frozen baseline, but it is
still 56.6% below the absolute threshold. The final disjoint-atlas guard
shows why another placement retry is not justified on this machine: the
4,779-pair working set needs 1,088 pinned device experts after a 66-per-layer
host allocation, while only 953 safely fit. The residual shortage is 135
experts, or 0.840454 GiB of packed data.

The separate Ornith35 grouped-int3 control also cannot supply an accepted
capacity path. It passes the token-agreement limits but raises perplexity by
10.660342% relative to int4, more than twice the fixed 5% maximum. No
Ornith397 int3 sidecar is produced.

### 3.3 Gate-9 dependency behavior

The controller exits during validation with:

```text
Gate 9 cannot start; pre-Gate-9 release checks failed:
ornith397.tier, ornith397.tools
```

It does not build a new engine and does not launch the first AB measurement.
This confirms that the implementation cannot silently convert a Gate-8
failure into partial Gate-9 evidence.

### 3.4 Composite repository result

After the negative disposition and its artifacts were staged, an uncontended
`make check` completed successfully in 69.8 seconds:

| Component | Result |
|---|---:|
| native C | 21 test executables passed |
| tokenizer | 10,000/10,000 encode and decode cases exact |
| Python | 113 tests passed |
| web | 18 tests, production build, zero audit vulnerabilities |
| documentation | 79 documents, 98 local links, 63 indexed reports |
| source package | 232 tracked paths, zero violations |
| CUDA | backend and session suites passed on RTX 5070 Ti (`sm_120`) |

This separates software-integrity closure from production-model acceptance:
the repository is mechanically healthy even though the strict model evidence
audit rejects release.

## 4. Interpretation

The project distinguishes three outcomes:

1. **implemented and passing**, such as the model containers, numerical
   controls, 35B generated tool use, session mechanisms, and test harnesses;
2. **implemented and rejected**, such as the direct 397B throughput profile,
   int3 quality control, and lossless residency treatments; and
3. **implemented but not authorized to run**, namely the final dependent
   Gate-9 production sequence.

This separation matters scientifically. Running Gate 9 after Gate 8 failed
would consume substantial time without making the build releasable, and it
would weaken the preregistered dependency. Marking it complete as
“not authorized” records the experimental decision without inventing a
measurement.

The negative result is hardware-profile specific. It does not prove that
Ornith397 cannot exceed 2 token/s on a larger-memory GPU or system. It does
show that repeated cache-placement changes cannot safely cover the observed
route set within 29.375 GiB RAM, 15.92 GiB VRAM, and the retained headroom.

## 5. Limitations

- The throughput result applies to this CPU, GPU, WSL storage stack, precision
  map, context, prompts, and memory headroom.
- The lossless pilots use deterministic repeated prompts and do not establish
  arbitrary-prompt cache coverage.
- The final Gate-9 AB/BA, cancellation p95, and two-model web measurements do
  not exist because their prerequisite failed. No inference about their
  numerical values is made.
- The strict auditor reports missing dependent artifacts as failures because
  it is a release auditor, not a roadmap-disposition auditor.
- A different algorithm, such as a new compressed expert representation that
  passes a fresh quality gate, would require a new preregistered study.

## 6. Reproducibility and final disposition

The final machine-readable artifacts retained with the source are:

| Artifact | SHA-256 |
|---|---|
| `c/ornith397_doctor.json` | `57538e0a75a12a591028aedcd86d4db4b2cc331181d1f17e37a36ea33182c08a` |
| `c/ornith397_ppl_smoke.json` | `76612b67614dc69e014ee45683ff033c0789b3a5da8b7d8c27ef401b0644bc49` |
| `c/ornith397_qualification.json` | `8f50265f4bcb531fcb4b314381822712996e9fafe21e849006b0d6ffed476509` |
| `c/release_audit.json` | `196897d40700eefc445775d0dc06e12bde399691e8b03150aef9be628d192d79` |

Gate 8 is closed negative on the reference hardware. Gate 9 is closed without
production acceptance because its prerequisite failed. Phase 10 is closed
without release because its strict evidence audit rejects the candidate. The
performance threshold is not revised, and no `v0.1` release tag is created.

Reopening requires one material change: a larger-memory target, an explicitly
revised product threshold, or a new algorithm/kernel hypothesis with fresh
controls. The same residency experiments should not be repeated.
