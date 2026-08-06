# Phase 10 Experiment 3: Composite Release Gate

## Abstract

The repository release check now exercises the complete portable build and
test surface from a clean tree. A first execution exposed a Makefile
conditional-scoping defect that silently omitted CUDA validation from the
default configuration. Moving CUDA-check discovery outside the `CUDA=1`
branch corrected the defect. The repeated end-to-end gate passed every native,
tokenizer, Python, web, dependency-audit, and CUDA check.

## Method

The root `make check` target delegates to `c/Makefile`, where it performs the
following ordered operations:

1. remove prior binaries and build metadata;
2. compile the inference executable for portable x86-64-v3;
3. compile and run the native C test executables;
4. build the tokenizer harness and compare 10,000 deterministic cases with the
   official Qwen tokenizer;
5. install the pinned web dependencies, run the JavaScript tests, build the
   production bundle, and audit the dependency graph;
6. discover and run all Python unit and integration tests;
7. validate every local Markdown link and require every research report to
   appear in `docs/research/README.md`; and
8. inspect Git's index for native executables, model weights, compiled
   objects, engine binaries, and generated web/model directories; and
9. when `nvcc` is present, rebuild the executable with CUDA support and run the
   CUDA backend and session suites.

The initial composite run completed the first six operations but printed no
CUDA command. A dry run showed that `CHECK_CUDA` and `CHECK_CUDA_RUN` were
defined only inside the `ifeq ($(CUDA),1)` branch. Since the release command
starts in its portable CPU configuration, the variables were undefined.
Their definitions were moved outside that branch. A second dry run verified
the expanded `make CUDA=1 CUDA_ARCH=native qwen test-cuda` command before the
complete gate was repeated.

## Results

The corrected clean run produced:

| Surface | Result |
|---|---:|
| Native C executables | 21/21 passed |
| Qwen tokenizer parity | 10,000/10,000 exact |
| Python tests | 59/59 passed |
| Web tests | 18/18 passed |
| Production web build | passed |
| npm dependency audit | 0 vulnerabilities |
| CUDA primitive/backend suite | passed on RTX 5070 Ti (`sm_120`) |
| CUDA recurrent/KV session suite | passed with exact restore and prefix extension |

The final `c/qwen` artifact was relinked with the CUDA backend, rather than
leaving the intermediate portable CPU executable as the repository's active
binary.

## Interpretation

A release gate must verify its own feature detection as well as the tests it
dispatches. The first run was superficially green but scientifically
incomplete: absence of an expected command is different from a passing
result. Including the CUDA rebuild in the same clean target prevents stale
objects and manually selected build modes from masking backend regressions.

The composite gate establishes build and regression integrity. It does not
establish the remaining model-scale acceptance criteria: warm 35B continuous
batching, completed 397B conversion and sustained throughput, or Ornith
end-to-end tool calling. Therefore it is evidence for one Phase 10 checklist
item, not evidence that the release is ready to tag.

## Decision

- Mark the composite `make check` requirement complete.
- Keep the benchmark table and `v0.1` tag explicitly blocked on Gates 7–9.
- Preserve automatic CUDA execution when `nvcc` is available and a clear skip
  message when it is not.
- Require future changes to the test matrix to pass through this clean target.

## Documentation-gate hardening addendum (2026-07-29)

The original release run checked documentation manually. That left link and
research-index integrity outside the command later used to justify the tag.
`c/tools/check_docs.py` now performs the same check deterministically and is a
required `make check` step.

The checker enumerates `PLAN.md`, `README.md`, and all Markdown files below
`docs/`; resolves every local link without permitting paths to escape the
repository; and requires every research Markdown file except the index itself
to be named by `docs/research/README.md`. Three fixtures prove acceptance of a
complete archive and rejection of both a missing target and an unindexed
report.

The live audit passes:

| Documentation surface | Result |
|---|---:|
| Markdown documents | 74 |
| local links resolved | 93/93 |
| research reports indexed | 58/58 |
| focused checker tests | 3/3 |

This addendum does not revise the historical 59-test count in the completed
composite run. The expanded Python suite and documentation step remain
subject to the final uncontended `make check` after model qualification.

## Contended expanded-suite observation

An expanded 102-test Python discovery was attempted while the Ornith397
converter committed multi-gigabyte shards. It completed 101 tests and raised
one error in `test_graceful_close_persists_learned_expert_map`: the tiny C
engine did not leave the kernel within the fixed 10-second graceful,
5-second terminate, and 5-second kill windows. The process disappeared after
the exception. A delayed `SIGKILL` cannot be explained by application signal
handling and is consistent with an uninterruptible WSL filesystem operation
during the concurrent shard commit.

The identical test was rerun alone without changing code or timeouts. It
passed in 7.648 seconds, printed the expected tier/prefetch telemetry, and
atomically wrote a valid learned expert map. The strict close contract is
therefore retained rather than weakening it to accommodate a deliberately
confounded run.

This observation reinforces the preregistered release rule: incremental and
contended tests are diagnostic, while the final expanded `make check` must be
run after conversion and all production benchmarks have stopped.

## Source-package gate addendum

The earlier packaging audit found one generated ELF retained from the initial
repository state. Removing it once was necessary but not sufficient:
extensionless native test binaries could be accidentally staged again.
`c/tools/check_source_package.py` is therefore a required composite step.

It obtains the authoritative tracked-path set from `git ls-files`, rejects
known model and build suffixes, the production model directories, `web/dist`,
and engine executable names, and additionally reads the first four bytes of
every present tracked file to detect extensionless ELF and PE executables.
Four fixtures require ordinary source and harmless `MZ` text to pass and ELF,
valid PE, model-weight, GGUF, and web-build examples to fail.

The first live pre-release audit covered only 62 previously tracked paths.
After staging the complete prospective release set, the audit covers 219
paths and passes with zero violations. The staged deletion of
`c/tests/test_st_pread` leaves its ignored local binary and tracked C source
available for testing. The first cached-diff audit also found trailing
whitespace and extra EOF blank lines in the research archive; these were
normalized, after which `git diff --cached --check` passed. The package gate
will run again during the final clean `make check`.
