# Phase 10 Preflight 4: Release Evidence Audit

**Date:** 2026-07-29
**Status:** implemented and tested; Ornith397 threshold amended prospectively

## Abstract

The project has many individually useful JSON artifacts. A release decision
would still be fragile if a person had to remember which files belong to
which model, whether they used the same checkpoint revision, and which nested
boolean indicates acceptance. This preflight adds one read-only release
auditor that joins model manifests with every required Gate 8 and Gate 9
result.

The auditor does not merely trust `passed: true`. It independently checks
model identity and recomputes the numerical thresholds visible in the
artifacts. It deliberately does not replace `make check`, source review, a
release commit, or the `v0.1` tag.

## 1. Research question

Can one deterministic command demonstrate that all production-model evidence
belongs to the pinned Ornith snapshots and satisfies the frozen release
thresholds?

## 2. Required evidence

`c/tools/audit_release.py` requires 15 checks:

| Group | Required evidence |
|---|---|
| model identity | complete pinned manifests for Ornith35 and Ornith397 |
| Ornith35 Gate 8 | teacher-forced gate, PPL gate, generated tool gate |
| Ornith397 Gate 8 | independent 122-shard header/ledger/hash/resource doctor, PPL smoke, direct tier/coherence/throughput gate, generated tool gate |
| Ornith35 Gate 9 | AB/BA batch, 20-trial cancellation, exact web streaming |
| Ornith397 Gate 9 | AB/BA batch, 20-trial cancellation, exact web streaming |

Every artifact's `model_manifest` must equal the corresponding live
container's source, fingerprint, shard counts, tensor counts, byte total,
precision map, and MTP status. A valid result from a different checkpoint
therefore fails.

The live manifests are also checked against preregistered container
cardinality. Ornith35 must contain 16 output shards and 93,277 physical
tensors. Ornith397 must contain 122 output shards and 278,152 physical
tensors. These values cannot be replaced by whatever a completed conversion
happens to report.

The precision identity is equally explicit: routed and dense Q/K/V weights
use grouped int4 with group size 128, input/output and shared experts use
int8, and MTP is excluded. A manifest with the right family but a different
precision map fails even if its own artifacts agree with it.

## 3. Artifact names

Existing accepted Ornith35 numerical evidence:

```text
c/ornith35_prefix_gate.json
c/ornith35_ppl_gate.json
c/ornith35_tool_gate.json
```

Pending Ornith397 Gate 8 evidence:

```text
c/ornith397_doctor.json
c/ornith397_ppl_smoke.json
c/ornith397_qualification.json
c/ornith397_tool_qualification.json
```

Pending Gate 9 evidence for `MODEL=ornith35` and `MODEL=ornith397`:

```text
c/MODEL_batch_ab.json
c/MODEL_batch_ba.json
c/MODEL_batch_abba.json
c/MODEL_cancel_gate.json
c/MODEL_web_gate.json
```

The first two batch files are experimental inputs. The release auditor
consumes the comparator's `*_batch_abba.json`, which already binds both
orders.

## 4. Recomputed rules

In addition to nested acceptance fields, the auditor requires:

- Ornith35 teacher-forced totals to recompute from exactly 20 rows × 64
  positions, every row at least 85%, and aggregate agreement at least 90%;
- finite positive perplexity, Ornith397 perplexity below 50, and Ornith35's
  reported relative GGUF delta to recompute exactly within the 5% bound;
- generated tool evidence to parse as exactly
  `get_weather({"city":"Paris"})`, finish first with `tool_calls`, use 18 in
  the final answer, finish with `stop`, and retain device-MoE telemetry;
- Ornith397 direct tier qualification to have no low-bit sidecar, the exact
  preregistered cache/context/warm-up profile, four nonempty measured outputs,
  and at least 2.0 sustained tokens/s recomputed from their token counts and
  rates;
- its EMAP and tier dimensions/counts to cover exactly 60 × 512 experts, a
  nonempty VRAM tier and real GPU to be present, and every resident layer to
  use device MoE with zero host fallback and consistent hidden/router/logit
  transfers;
- the exact family-rendered batch prompt and frozen 0.95/1.0 thresholds, with
  minimum order speedup and geometric mean recomputed from the recorded AB
  and BA ratios rather than trusted as summary fields;
- the exact family-rendered cancellation prompt, model-specific cache
  budgets, CUDA enabled, experimental switches off, and exactly 20 lifecycle
  trials whose individual acknowledgement latencies equal the 20 summary
  samples;
- nearest-rank p95 no more than 1.0 seconds for Ornith35 and 3.0 seconds for
  Ornith397, plus real GPU/VRAM and device-MoE resident telemetry;
- the model-specific frozen RAM/VRAM budgets and all experimental switches
  off for web qualification, a built bundle, CUDA enabled, a real GPU and
  nonempty VRAM tier;
- exactly two web results, each with content `colib ready` and positive token
  usage; and
- resident layer/device-MoE equality, positive router/logit transfers, zero
  host-MoE fallback, and an empty final scheduler.

This is stronger than searching files for a true boolean. A self-consistent
artifact from Qwen, a stale Ornith revision, a one-sample “p95,” or a web run
that only detected a GPU all fail.

## 5. Command and final sequence

After Gate 8 and Gate 9 experiments:

```sh
.venv/bin/python c/tools/audit_release.py \
  --root . --output c/release_audit.json

git add -A
make test-source-package
make check
git diff --cached --check
```

Acceptance requires `release_audit.json` to report 15/15 checks and zero
failures. The intended release set is staged before the source-package and
composite checks so newly added files are included in the index audit rather
than silently omitted as untracked work. After the complete uncontended
`make check`, inspect the staged diff, create the intentional release commit,
verify the clean tree, and tag exactly `v0.1`.
The report is written to a sibling temporary file and atomically renamed, so
an interrupted audit cannot expose a truncated final evidence artifact.

The audit output retains two explicit reminders:

```text
requires_make_check: true
requires_clean_release_commit_and_v0_1_tag: true
```

This prevents the JSON join from being misrepresented as the entire release
process.

## 6. Tests

One fixture constructs both complete model manifests and all required
manifest-bound artifacts, including threshold values, and requires 15/15
acceptance. Five negative fixtures independently:

1. change only the Ornith397 web artifact's source fingerprint;
2. change the Ornith397 output-shard and physical-tensor counts; and
3. reduce the doctor's independently checked output-hash count from 122 to
   121;
4. forge teacher-forced counts, a relative PPL delta, and tool arguments; and
5. forge a reported cancellation p95 that disagrees with its 20 samples.

The corresponding web, manifest, structural/hash, numerical, tool, and
cancellation checks must fail. These tests cover the complete path,
cross-checkpoint rejection, cardinality/hash rejection, and independent
threshold recomputation.

The complete CPU-side C unit target was also rerun while the converter was
active. All 21 test binaries passed, covering safetensors I/O, quantized
matrices, DeltaNet, partial RoPE, routing, slot isolation, session
serialization, cancellation-supporting scheduler primitives, and the
resident whole-model batch path. This is useful correctness evidence but is
not the final release suite.

The tokenizer and deterministic web components were then checked separately:
all 10,000 frozen mixed-language tokenizer cases matched exact encode and
decode IDs, all 18 web unit tests passed, the TypeScript/Vite production
bundle built successfully, and `npm audit` reported zero vulnerabilities.
The isolated CUDA target also passed on the RTX 5070 Ti (`sm_120`): dense and
quantized kernels stayed within their fixture tolerances, grouped routed and
shared MoE matched exactly, and the CUDA session test preserved recurrent,
KV, slot, checkpoint, and exact-prefix state.

The required clean `make check`, real tiny web round trip, and full Python
discovery run remain deferred until conversion activity has stopped. The
final suite must rerun the successful components from a clean build rather
than treating these incremental checks as a substitute.

Documentation consistency was checked independently: 74 Markdown files
contain 93 local links, all of which resolve, and the research-folder index
names all 58 phase reports. The fully staged prospective release diff also
passes `git diff --cached --check`.
This does not assess the truth of experimental claims; it establishes that
the release documentation is navigable and mechanically well formed.

A source-control packaging audit excludes the local Ornith containers, the
20 GiB GGUF reference, Qwen diagnostic JSON, and source-inspection scratch
output while retaining the small accepted gate artifacts. It also found one
historically tracked ELF, `c/tests/test_st_pread`, even though the target is
generated. The binary has now been removed from the Git index while the local
file and `test_st_pread.c` remain intact; explicit native and `.exe` ignore
rules prevent it from returning. The release commit will therefore record
only the deletion of the generated executable, and the test target will
continue to rebuild it from source. A new index-driven package gate detects
ELF/PE magic plus forbidden model, object, engine, and web-build paths; its
four fixtures and the complete 219-path prospective release index pass.

## 7. Prospective threshold amendment

On 2026-07-31, after the original negative disposition, the project owner
revised only the Ornith397 production threshold to 0.85 token/s. The historical
2.0 token/s protocol and results above remain unchanged. The live auditor now
requires the full lossless optimized profile, `minimum_tps: 0.85`, and
recomputes a strict sustained rate of at least 0.85. Persistent I/O, pinned
upload, decode protection, and prewarming must all be recorded as enabled.
A passing fixture uses 0.86 token/s and a dedicated 0.849 boundary fixture
fails. See
`docs/research/phase8_preflight_07_revised_production_threshold.md`.

The independent auditor was also executed in the expected pre-Gate-8 state.
It reports exactly 4/15 checks passing and the 11 preregistered missing
results: the Ornith397 manifest, doctor, PPL, tier, and tool checks, followed
by batch, cancellation, and web checks for each Ornith model. No completed
Ornith35 result regressed, and no unexpected check failed. This is a
fail-closed baseline, not release acceptance.

The install layout was exercised under an isolated `DESTDIR` with the CUDA
build enabled. It installed the launcher, engine, gateway, doctor, support
tools, and web bundle with the intended executable/data permissions. Running
the installed launcher with `COLI_MODEL` pointed at the accepted Ornith35
container selected the installed engine and passed the two-slot production
doctor/resource plan. The temporary installation was then removed. This
confirms that model weights are an explicit deployment input rather than an
accidental part of the software install.

## 7. Limitation and decision boundary

The frozen audit accepts only the direct Ornith397 int4-g128/int8 profile.
Qwen397 3-bit is permanently retired, and grouped 3-bit is not part of Gate
8. Any future Ornith-only lower-bit experiment would require a separately
authorized protocol and audit amendment; it cannot weaken or bypass this
release check.

Accept the auditor as the final evidence-joining step, not as a substitute
for model execution, the complete regression suite, source review, or release
tagging.
