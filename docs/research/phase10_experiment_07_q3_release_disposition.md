# Phase 10 Experiment 7: Q3 Positive Release Disposition

**Models:** Ornith-1.0-35B and Ornith-1.0-397B grouped-int3

**Reference hardware:** Ryzen 7 7700X, 29.375 GiB RAM, RTX 5070 Ti with
15.92 GiB VRAM

**Date:** 2026-08-06

**Status:** model and repository evidence accepted; intentional release
commit/tag remain separate final actions

## Abstract

This study applies the amended, owner-approved grouped-three-bit release
criteria to the completed Ornith397 evidence chain. It does not revise the
historical rejection of the lossless int4 profile. The selected q3 sidecar is
complete and independently verified, teacher-forced agreement is 97.109375%,
semantic coherence and generated tool use pass, and the explicit owner-bound
perplexity waiver is present. After repairing an expanded-q4 prefill dispatch
defect, the complete production tier sustains 0.836866773 token/s against the
amended 0.70 token/s floor. All four measured turns exceed the floor.

The dependent Gate-9 sequence also passes. Ornith397 concurrent serving is
faster than sequential serving in both orderings, with a 1.109807 geometric-
mean speedup; cooperative cancellation p95 is 1.935734 seconds against a
3-second ceiling; and two HTTP history turns return exact output. Every
accepted CUDA artifact records zero host-MoE fallback. The independent q3
release audit consequently passes 20 of 20 checks.

The first final `make check` run passed its executable test phases and then
stopped because the research index already linked to this report but the file
did not yet exist. Publishing this report closed that documentation gap. The
complete post-publication rerun then exited successfully: 21 C executables,
10,000 tokenizer cases, 18 web tests, 137 Python tests, the production web
build and dependency audit, 91-document documentation audit, 273-path source-
package audit after final evidence staging, and both CUDA suites passed. An intentional clean release
commit and `v0.1` tag remain separate actions and are not inferred from
evidence acceptance.

## 1. Research question

Do the final structural, numerical, semantic, tool-use, throughput, multi-slot,
cancellation, web, and repository artifacts jointly satisfy the amended q3
production criteria on the reference machine?

The release-evidence hypothesis requires every one of the 20 machine-readable
q3 audit checks to pass. The broader release hypothesis additionally requires
the complete repository check, an intentional clean release commit, and a
deliberate `v0.1` tag. These requirements are conjunctive. The model can pass
its scientific gates while the repository or release operation remains open.

## 2. Protocol amendments and preserved history

The original int4 study required at least 2 token/s and later 0.85 token/s.
Its complete optimized result reached only 0.631963761 token/s, so that profile
remains rejected. The project owner subsequently accepted the measured
Ornith35 q3 quality class, waived a separate Ornith397 q3 perplexity run, and
selected a prospective minimum of 0.70 token/s sustained for the Ornith397 q3
production profile.

The amendment is explicit rather than retroactive:

- the historical 15-check int4 audit and negative report are unchanged;
- the q3 profile has its own 20-check auditor and exact sidecar identity;
- the perplexity waiver is stored in the Gate-8 ledger and rechecked by the
  release auditor;
- teacher-forced, semantic-coherence, generated-tool, CUDA-residency, Gate-9,
  and repository requirements remain mandatory; and
- expanded-q4 execution of q3 expert values remains the validated default,
  while native packed q3 and the adaptive atlas remain experimental.

## 3. Model identity and structural evidence

The Ornith397 base container is pinned to publisher revision
`8b61f97a8512d9d01bff1a9625c9a16730e115bb`. It contains 122 output shards,
93,078 logical tensors, 278,152 physical tensors, and 212,634,789,241 payload
bytes. The q3 routed-expert sidecar contains 60 layers, 480 files, 276,480
tensors, and 157,073,113,440 payload bytes. Its manifest SHA-256 is
`5231fbe79bc9edf27b86222e4df503066a0b8f04f02fc612ab993696096b0180`.

The independent sidecar doctor verifies every file hash and tensor header.
All accepted Ornith397 quality, throughput, batch, cancellation, and web
artifacts repeat the same sidecar identity. This prevents an int4 result, an
incomplete sidecar, or a different conversion from satisfying the q3 profile.

## 4. Gate-8 results

### 4.1 Numerical and semantic controls

The repaired q3 teacher-forced comparison matches 1,243 of 1,280 token
positions, or 97.109375%. All 20 prompts exceed the 85% per-prompt floor; the
weakest prompt reaches 92.1875%. The regenerated four-prompt free-running
control answers all prompts coherently and is explicitly acknowledged in the
hash-bound Gate-8 ledger.

The generated HTTP tool control selects
`get_weather({"city":"Paris"})`, consumes the deterministic 18-degree clear
weather result, returns a correct final answer, and stops without a second tool
call. Its 2,580 MoE layer-forwards all execute on CUDA.

### 4.2 Throughput control

The first repaired full tier run was retained as a genuine failure: one
0.243102 token/s transient reduced sustained throughput to 0.520672 token/s.
Component timing showed simultaneous storage, expert-matmul, and language-
model-head slowdown, followed by immediate recovery. One unchanged full retry
was preregistered rather than selectively replaying the slow prompt.

After confirming an idle machine, the complete retry produced:

| Prompt | Decode rate (token/s) | TTFT (s) |
|---|---:|---:|
| hash-table explanation | 0.858887 | 82.880 |
| TypeScript grouping | 0.857416 | 76.494 |
| service diagnosis | 0.865132 | 91.217 |
| RAM versus NVMe | 0.773247 | 81.065 |

The aggregate sustained rate is 0.836866773 token/s, the median turn is
0.8581515 token/s, and the minimum is 0.773247 token/s. This exceeds the
0.70-token/s floor by 19.55%. All 46,080 MoE layer-forwards use CUDA and none
fall back to host MoE. The tier artifact SHA-256 is
`7d4d1fe8add387fbdd274a6bd287629023b76c7562a67ac328ffc731507c4106`.

## 5. Gate-9 results

### 5.1 Reversed-order batch comparison

Both models run sequential/concurrent A/B, concurrent/sequential B/A, and an
independent ABBA comparator. Ornith397 uses the exact q3 sidecar and produces
the required `colib batch ready` output in both modes.

| Metric | Ornith35 | Ornith397 q3 | Acceptance |
|---|---:|---:|---:|
| A/B aggregate speedup | 1.008885 | 1.123472 | at least 0.95 |
| B/A aggregate speedup | 1.036472 | 1.096309 | at least 0.95 |
| geometric mean | 1.022585 | 1.109807 | at least 1.00 |
| minimum ordering | 1.008885 | 1.096309 | at least 0.95 |

The two Ornith397 batch artifacts jointly record 6,600 device-MoE layer-
forwards and zero host-MoE fallback.

### 5.2 Cooperative cancellation

Each model runs 20 trials. A trial observes exactly one streamed piece before
cancelling one live request, allows its peer to finish, and immediately reuses
the released slot. Ornith35 nearest-rank p95 is 0.162146 seconds against a
1-second ceiling. Ornith397 p95 is 1.935734 seconds against a 3-second ceiling.
The Ornith397 artifact records 12,720 device-MoE layer-forwards and zero host
fallback.

### 5.3 Exact web history

Both Ornith397 HTTP turns return exactly `colib ready`; the scheduler ends with
zero active and queued requests. TTFT falls from 160.179 seconds on the first
turn to 74.148 seconds on the history-extension turn. All 480 MoE layer-
forwards execute on CUDA and none use host MoE.

The Gate-9 pipeline finishes with `status: passed`. Its SHA-256 is
`98f359c3d966f26b22235cbf23098019c3c12b7046122c366e684218eb3a853e`.
The Ornith397 Gate-9 artifact hashes are:

| Artifact | SHA-256 |
|---|---|
| batch A/B | `e703bc9343d0f78794049fc96c52ba4386c77a84ae94a4a5f2e4e3709b2cc1ee` |
| batch B/A | `625554c513a6b5875ba6456a6d3e9d09c60e2aa7cc7184fd34e0735e36c82eb6` |
| ABBA comparator | `b9f95d9462f33d530370b5929a9bb11044ccbf71692782b525f9b5870ea41032` |
| cancellation | `606c4383e341e09db746f117b95f2bc1c83286837df19a6f87b667cb858d69ce` |
| web | `e5aa4d5ee2211686bfeb4af00b2bacfa50a3c1ef12b4256d085f1fb91a0dbb36` |

## 6. Independent release audit

The q3 release audit reads live manifests and artifacts rather than accepting
the controller's top-level status. It checks 20 independent joins spanning
both model manifests, the q3 sidecar and doctor, the explicit PPL waiver,
teacher-forced/coherence/tool/tier evidence, semantic acknowledgement, and all
six model-level Gate-9 batch/cancellation/web results.

The audit passes 20/20 with zero failures. Its SHA-256 is
`83393026d49402116de337cc7f014d094b1c4ffab603c0ddb5822adb10c6edad`.
The auditor deliberately reports two requirements outside its own scope:
`make check` must pass, and release still requires a clean intentional commit
plus the `v0.1` tag.

## 7. Repository check

The first final `make check` execution rebuilt the portable CPU target, passed
all 134 Python tests, and reached the documentation validator. The validator
reported one failure: the research index linked to
`phase10_experiment_07_q3_release_disposition.md`, but that required final
paper did not exist. No implementation or model test failure was reported.

This document supplied the missing indexed paper. The complete post-
publication `make check` rerun exited with status zero on 2026-08-06. It
passed 21 native C test executables, 10,000/10,000 tokenizer cases, 18/18 web
tests, and 137/137 Python tests. The web production build succeeded, `npm
audit` found zero vulnerabilities, the documentation audit covered 91
documents, 111 local links, and 75 indexed research reports, and the source-
package audit accepted 273 tracked paths with zero failures after the final
Gate-8/Gate-9 evidence artifacts and closing papers were staged.

The CUDA rebuild and tests also passed on the RTX 5070 Ti. In particular, the
real Ornith-shape native-q3 versus expanded-q4 grouped-MoE comparison was
exact (`maxdiff=0`), and the CUDA session suite preserved recurrent/KV state,
disk-checkpoint rejection, two-slot restore, and exact-prefix reuse. This
closes the repository-check requirement independently of the model benchmarks.

### 7.1 Rebuild-stable engine binding

A final live regeneration initially contradicted the stored result: the audit
fell from 20/20 to 19/20 after `make check` rebuilt the CUDA engine. Every
model/artifact hash and the manual review flag still matched. Only the raw
engine executable hash differed. Two forced builds from unchanged sources
produced distinct SHA-256 values,
`b5484f0d478c4b7003ad732aac36df8672248d2f2caece1261422ca4123a1848`
and
`7c2c2ed17ccd7ea8158c19de73f024f0950c53e4e1f050569265c0b4684e0fb7`.
The CUDA fat binary is therefore not a reproducible content identifier on this
toolchain, even when its source and build command are unchanged.

The controller and auditor now retain the original raw executable hash as
historical provenance but bind resumption and live release acceptance to a
deterministic length-delimited digest of the Makefile, C/CUDA translation
units, and every local header that determines the engine. The accepted source
fingerprint is
`c7751e414009bff6059c7f79dbbf28269c68c489254cbde2b4810cc3a3b07527`.
A changed source file fails the audit; byte-different rebuilds from identical
sources do not. Both real Gate-8 and Gate-9 controllers subsequently resumed
all hash-bound artifacts without inference reruns, and the regenerated release
audit returned to 20/20. This repair makes the mandatory clean rebuild and the
release-evidence check logically compatible.

The release surfaces now expose one consistent application version: `colib
--version` reports `colib 0.1.0`, matching `web/package.json`. The planned
short tag `v0.1` denotes this semantic version without changing the package
version required by npm.

## 8. Limitations

- Results apply to the specified CPU, RTX 5070 Ti, WSL storage stack, memory
  budgets, prompts, and context length.
- The accepted q3 profile is explicitly lossy. The owner waived a separate
  Ornith397 perplexity run based on the measured Ornith35 q3 quality class;
  this report does not claim q3 is lossless.
- One full tier run experienced a severe transient. The successful controlled
  retry establishes the acceptance result, while production telemetry should
  retain per-turn storage and compute timing.
- The batch benchmark uses short exact responses and establishes scheduler/
  resident-graph behavior, not arbitrary-workload scaling.
- Packed native-q3 kernels and the adaptive route atlas are not the release
  path because their complete real-model protocols remain open.
- Evidence acceptance does not create a clean Git commit or release tag.

## 9. Disposition

The Ornith397 q3 model evidence satisfies Gates 8 and 9, and the independent
release auditor accepts all 20 checks. This is a positive model-evidence
disposition and supersedes only the later q3-open status, not the historical
int4 rejection.

Repository acceptance is positive: the post-publication full `make check`
rerun passes. No `v0.1` tag is yet authorized because the current worktree has
not been reviewed and intentionally committed as a clean release state. The
remaining commit and tag are source-control publication actions, kept separate
so scientific evidence cannot silently publish a release.
