# Phase 8 Preflight 9: Ornith397 Q3 Qualification Automation

**Model:** `deepreinforce-ai/Ornith-1.0-397B-FP8`

**Date:** 2026-08-01

**Status:** implementation and focused controls complete; conversion and results pending

## Abstract

This preflight converts the owner-approved grouped-3-bit decision into a
fail-closed experimental procedure. The original Ornith397 container remains
unchanged. A sidecar converter is writing only the routed-expert matrices, and
a new independent auditor will verify the completed files without trusting the
converter's own resume ledger. A separate Gate-8 controller binds every result
to the exact base manifest, q3 manifest, int4 perplexity baseline, CUDA engine,
commands, and output hashes.

The sequence tests structural integrity, perplexity, teacher-forced token
agreement, four coding and diagnostic responses, generated tool use, CUDA
residency, and sustained speed. It stops immediately when a required result
fails. It also pauses after the four generated responses so their meaning can
be reviewed before later tests proceed. Sixteen focused reference-capture, q3,
and historical Gate-8 controller tests pass. At the recorded live checkpoint, 41 of 480
atomic files (8.54%) had been committed, representing 13,416,641,552 bytes,
with no temporary file or resource fault. This is progress evidence, not a
completed conversion or a Gate-8 result.

## 1. Research question

Can the Ornith397 routed experts be reduced from grouped 4-bit to grouped
3-bit storage while meeting all of the following prospective conditions?

1. the new files are complete and exactly bound to the accepted int4 model;
2. perplexity increases by no more than 12%;
3. teacher-forced agreement remains at least 90% overall and 85% for every
   prompt;
4. coding and diagnostic answers remain coherent;
5. the model completes the specified weather-tool interaction correctly;
6. every neural-network layer stays on the intended CUDA path with no CPU
   fallback for expert matrix multiplication; and
7. the complete optimized profile sustains at least 0.85 token/s.

These requirements are conjunctive: all must pass. A faster model that fails
quality is rejected, and a high-quality model below the speed threshold is
also rejected.

## 2. Terms

**Routed expert.** Ornith is a mixture-of-experts model. For each token, a
router selects a small subset of many specialist feed-forward networks. Only
those selected specialists execute, but every specialist must still be stored
somewhere.

**Grouped 3-bit quantization.** Each weight is approximated by one of eight
integer levels. One floating-point scale is shared by each group of 128
weights. This reduces storage and disk traffic but introduces numerical error.

**Sidecar.** The 3-bit tensors live in additional files beside the accepted
4-bit container. The runtime uses them only when `EXPERT_Q3=1`. Turning that
switch off returns to the unmodified 4-bit model.

**Manifest.** A JSON inventory describing the conversion method and every
output file. It contains file sizes and cryptographic hashes. A manifest with
`complete=false` is a progress checkpoint and is not runnable.

**SHA-256.** A cryptographic digest used as a strong file identity. Recomputing
the digest detects accidental changes or corruption.

**Perplexity.** A probability-based error measure on known text. Lower is
better. Here, relative change from the matching 4-bit model measures the
calibration cost of 3-bit experts.

**Teacher forcing.** Both models receive the same correct token history at
every position. Their next-token choices can therefore be compared without an
early generation difference changing all later inputs.

**CUDA residency.** Dense layers and expert matrix operations remain on the
GPU. Only the explicitly designed routing control and final outputs cross the
CPU/GPU boundary.

## 3. Conversion method and live checkpoint

The deterministic conversion command is:

```sh
.venv/bin/python c/tools/requantize_expert_q2.py \
  --snapshot c/ornith397 \
  --bits 3 \
  --group-size 128 \
  --iterations 3 \
  --experts-per-file 64 \
  --workers 8 \
  --torch-threads 1
```

Eight workers process independent expert projections. Each worker uses one
Torch thread, limiting oversubscription on the eight-core Ryzen 7 7700X. A
file covers 64 experts in one layer and is renamed into place only after
Safetensors writing succeeds. The progress manifest is also replaced
atomically after each file.

The model contains 60 layers and 512 experts per layer. Eight files per layer
therefore produce 480 files. Each expert has three projections; every
projection has a packed payload, a scale tensor, and a type tensor. The exact
expected physical inventory is consequently:

```text
60 layers × 512 experts × 3 projections × 3 tensors = 276,480 tensors
```

At 12 minutes 18 seconds, the process used approximately 6.13 CPU cores and
1.23 GB resident memory. It had committed 41 files and 13,416,641,552 bytes.
The last ledger entry was
`expert-q3-l005-e0000-0063.safetensors`, SHA-256
`290cb9514fb43a05f52b2ea6684cc0dca88beb6f1d3105e15890bfa80a56222a`.
Linux free space was 646 GB. These observations show healthy progress and
large safety margins, but do not predict the final performance result.

### Safely paused checkpoint

At the owner's pause request, the Gate-8 supervisor was terminated first and
the converter received `SIGINT`. Python waited for its active threaded chunk
work to return before exiting; no force kill was used. The resulting ledger is
still deliberately `complete=false` and contains 218 of 480 files, 125,568
tensors, and exactly 71,337,346,464 bytes. An independent filesystem join found
218 actual files, zero missing ledger entries, zero unledgered files, and zero
`.partial` files. The latest committed file is
`expert-q3-l027-e0064-0127.safetensors` (327,235,552 bytes), whose recomputed
SHA-256 matches the ledger value
`a7e2d25071745aef25305129278126779c5c4f3648be43e2aa91b47f31fedc2d`.
Both converter and supervisor processes are absent. Linux retains 592 GB free.

Resume with the identical converter command in Section 3. After obtaining its
new PID, start the Gate-8 controller with `--converter-pid <PID>
--poll-seconds 60`. The converter will validate and skip the 218 ledgered files
before continuing. Gate 9 remains prohibited.

The owner subsequently resumed work. Converter PID 95109 uses the identical
recorded command, and supervisor PID 95154 is attached to it. Before writing a
new chunk, the converter recomputes SHA-256 for each of the 218 committed
files; observed reads advance at roughly 98 MB/s during this deliberately
single-threaded integrity pass. The unchanged file count during validation is
therefore expected. Eight-worker quantization begins only after every prior
hash matches.

The resume integrity pass completed successfully. The converter then atomically
published `expert-q3-l027-e0128-0191.safetensors` and
`expert-q3-l027-e0192-0255.safetensors`, advancing the ledger from 218 to 220
files and from 71,337,346,464 to 71,991,818,208 bytes. This proves both halves
of the recovery protocol on the real model: prior artifacts are revalidated,
then conversion continues at the first missing chunk without rewriting the
accepted prefix.

The same bounded hashing strategy is now available to future converter
restarts through `--resume-hash-workers 8`. It validates only ledgered files
within the sidecar directory and leaves the quantization signature unchanged.
A deterministic recovery fixture proves that two exact files are both skipped;
after one is tampered, the next resume validates and skips its intact peer and
regenerates only the mismatched chunk. The already-running converter does not
reload this code and was intentionally left uninterrupted.

## 4. Independent structural audit

`audit_expert_sidecar.py` does not import or call the converter. It derives
the expected expert inventory from `config.json` and the accepted base tensor
index, then checks:

- the q3 format, completion flag, configuration hash, source-index hash, group
  size, refinement count, and 64-expert chunk size;
- exactly 60 layers, 512 experts per layer, 480 files, and 276,480 tensors;
- exact filenames and layer/expert ranges with neither missing nor extra
  files;
- every ledger byte count and total byte count;
- every Safetensors name, dtype, shape, and qtype value; and
- an independently recomputed SHA-256 for every file.

This separation matters because a converter can consistently repeat its own
bookkeeping bug. Re-deriving the inventory in another program reduces that
common-mode risk.

Hash calculation is now bounded to eight independent worker threads for the
real 480-file audit. Each immutable file still receives a complete SHA-256 and
must exactly match its own ledger digest; only scheduling changed. Tensor-name,
dtype, shape, qtype, cardinality, and byte checks remain sequential and
independent of the converter. This avoids repeating the resume pass's
single-stream disk bottleneck without weakening any acceptance predicate. An
18-test focused auditor/controller/release suite passes, including threaded
success, digest tampering, and non-positive worker-count rejection. Supervisor
PID 99190 holds this exact command while converter PID 95109 continues
uninterrupted.

## 5. Ordered qualification

`run_ornith397_q3_gate8.py` executes the following stages in order:

1. independent full sidecar audit;
2. fixed-corpus q3 perplexity with an exact ceiling of
   `1.050067545 × 1.12 = 1.1760756504`;
3. capture 20 deterministic, 64-token int4 reference continuations;
4. replay those references through q3 teacher forcing with 90% aggregate and
   85% per-prompt floors;
5. produce four fixed, 64-token coding/diagnostic responses with complete q3
   and CUDA telemetry;
6. pause for semantic review of those four responses;
7. perform the live generated weather-tool round trip; and
8. run the complete two-warm-pass, four-prompt, 64-token optimized profile at
   the unchanged 0.85 token/s minimum.

The reference-capture mode now explicitly writes `reference_tokens` and sets
both low-bit environment switches, preventing an unrelated shell variable
from contaminating the int4 control. A feasibility review also found that the
older comparison harness defaulted to its CPU-oriented cache profile. Both the
int4 capture and q3 teacher-forced stages now explicitly select the accepted
18 GB host cache, 6 GB CUDA expert cache, persistent `io_uring`, pinned upload,
decode protection, and decode prewarming. These settings are written into the
result JSON instead of being implicit shell state.

Every completed step is recorded with its exact command and artifact SHA-256.
Resume is refused if the base manifest, q3 manifest, int4 PPL baseline, or
CUDA executable changes. A failure stops the sequence; later stages cannot
make an earlier failure pass.

## 6. Manual coherence boundary

Nonempty output is necessary but not sufficient for coherence. A program can
produce many tokens that are repetitive, contradictory, or irrelevant. The
controller therefore exits with `awaiting_coherence_review` after the four
machine-valid responses. It will not run the tool or final performance stages
until restarted with `--acknowledge-q3-coherence` after those outputs have
been inspected.

This review does not change numerical thresholds. It only addresses a property
that is not represented reliably by token count or throughput.

## 7. Focused verification

Sixteen focused tests covering reference capture, the new q3 controls, and the
historical Gate-8 controller pass. The q3 tests prove that:

- an incomplete manifest cannot be mistaken for a finished conversion;
- changing the conversion signature is rejected;
- tampering with a sidecar file is detected;
- exact tensor headers and hashes pass;
- every production command selects q3 except the deliberately int4 reference;
- the final tier command retains all four accepted I/O optimizations and the
  exact 0.85 threshold; and
- the pipeline cannot cross the semantic-review boundary without explicit
  acknowledgement.

A one-prompt real-35B smoke was attempted while the 397B converter occupied
most CPU and storage bandwidth. The first attempt correctly exposed that the
installed executable was an older CPU build. After rebuilding with CUDA, a
second attempt remained dominated by concurrent model loading and was stopped
at the three-minute bound. It produced no result artifact and is not counted
as performance or correctness evidence. The post-conversion controller always
rebuilds and hashes the CUDA executable before any accepted model run.

After the Gate-9, release-audit, shared-manifest, and environment-isolation
bindings were added, the complete Python regression suite passed 130 of 130
tests in 99.682 seconds under converter contention after Gate-9 environment
isolation was included.
The native suite also passed all 21 C executables, including deterministic
grouped-int3 packing/matmul cases for row widths 1, 31, 64, 128, 129, 256,
and 257. These controls establish that the new orchestration did not regress
the converter, runtime, serving, or historical audit paths. They do not
substitute for the real 397B numerical and performance measurements.

A final handoff review found that the command harnesses rejected incomplete
sidecar manifests, but a direct engine invocation could request `EXPERT_Q3=1`
and silently fall back to the corresponding int4 tensor when a selected q3
tensor was absent. The C loader now terminates on that condition. A direct
runtime regression proves the incomplete selection is rejected, while an
explicit `EXPERT_Q3=0` control still loads the unchanged int4 snapshot. This
closes the last mixed-precision escape path outside the guarded controllers.

The final directory contains 122 base shards plus 480 sidecar shards. A new
native regression creates and indexes this exact 602-file layout, proving the
loader's 4,096-shard table and dynamic tensor inventory do not impose a hidden
small-fixture limit. The control passes with both the reference machine's
10,240-file soft descriptor limit and a constrained 1,024-file limit; direct
I/O twins that cannot be opened under the latter use the existing buffered
fallback. The accepted reference profile has sufficient descriptors for all
buffered and direct handles simultaneously.

The perplexity, prefix-comparison, tier, tool, batch, cancellation, and web
harnesses now obtain their low-bit identity from one shared fail-closed
loader. It requires the expected format, `complete=true`, configuration and
conversion signatures, positive layer/file/tensor/byte counts, and records
the manifest SHA-256. This removes weaker duplicated parsers that checked only
the completion flag. Thirty-three focused identity, controller, protocol, and
release-auditor tests pass after the consolidation.

Each numerical and runtime harness now starts from a shared isolated engine
environment. Ambient variables that could silently select MTP, teacher-force
mode, a partial q3 layer range, mmap, debugging, a different cache policy, or
an experimental CUDA kernel are removed before the command's recorded profile
is installed. All six model-executing Gate-8 steps explicitly use eight
OpenMP threads, and the tier/tool artifacts record that value. The waiting
supervisor was restarted after this preregistration change without touching
the converter, so the live process holds the exact tested command set.

## 8. Conversion result and active qualification

The resumed conversion completed on 1 August 2026. It reused 218 previously
verified files and atomically generated the remaining 262. The complete
sidecar contains 480 files, 276,480 tensors, and 157,073,113,440 data bytes
across 60 layers and 512 routed experts per layer. No temporary output remains.
The manifest SHA-256 is
`5231fbe79bc9edf27b86222e4df503066a0b8f04f02fc612ab993696096b0180`.

The controller then rebuilt the CUDA executable and ran the independent
sidecar audit with eight bounded hash workers. All 480 file hashes and all 480
Safetensors headers passed, and the auditor independently counted the expected
276,480 tensors. This closes the structural conversion item but is not model
quality or performance evidence.

The frozen 1,024-token q3 perplexity evaluation ran for approximately 25
minutes before the owner elected to skip perplexity and requested a direct
tok/s result. It was terminated cleanly before producing an artifact and is
not evidence. The exact production CUDA tier profile was then launched with
the frozen 18 GiB RAM, 6 GiB VRAM, persistent `io_uring`, pinned upload,
decode protection, two warm-up passes, four 64-token prompts, and one measured
pass. At the subsequent owner-requested pause it was still executing warm-up
prompt 1/4 and had not produced a completed turn or result artifact. Therefore
no q3 tok/s is claimed from the interrupted run.

All associated process groups were terminated and GPU memory was released.
The complete sidecar and independent audit remain valid and unchanged. The
next continuation may rerun the exact tier command to answer throughput, but
token agreement, coherence, tool behavior, and the skipped perplexity result
will remain unknown unless separately authorized. Gate 9 stays prohibited.

## 9. Subsequent throughput result and superseding decision

A later continuation completed the same two-warm-pass, four-prompt,
64-token profile after adding an exact CPU SIMD expansion path for q3-to-q4
CUDA staging. The measured turns were 0.945809, 0.454950, 0.893894, and
0.837777 token/s. Their token-weighted sustained rate is 0.718432662 token/s,
the median is 0.8658355, and the minimum is 0.45495. All outputs were nonempty,
telemetry was complete, CUDA residency covered 46,080 layer-forwards, and
host-MoE fallback was zero. The result artifact was preserved as
`c/ornith397_q3_qualification_expand_q4.json`, SHA-256
`047fdf9d6d8e5fcbd494dcdbb1d8d4ac752f799dae9bf3f10f456c0223a8d9f7`.

The owner accepted this sustained result, selected q3 for production, and
deferred the remaining tests and release finalization. A subsequent native
packed-q3 run was deliberately stopped after two warm-up prompts and produced
no artifact. The original automation and 0.85 threshold remain documented
above as historical protocol; later finalization must encode the explicit PPL
waiver and prospective 0.70 regression floor without rewriting or fabricating
the skipped evidence. Native q3 and adaptive-atlas work is described in
`phase8_preflight_10_native_q3_hot_route_atlas.md`.
