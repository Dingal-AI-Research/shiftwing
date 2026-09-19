# DeepSeek reviewer: current model handoff

> **Superseded on 2026-09-11 for everything about performance and acceptance.**
> Read [deepseek-reviewer-acceptance-findings-2026-09-11.md](deepseek-reviewer-acceptance-findings-2026-09-11.md)
> first. The checkpoint is now complete and byte-validated (91/91 groups,
> 166.9 GB compared), and the complete model has been measured end to end.
>
> Two corrections that matter before trusting anything below:
>
> 1. Every measurement in this document was taken with **buffered reads**.
>    `_GNU_SOURCE` was defined after system headers in `deepseek_v4.c` and in
>    the `deepseek_v4_execute.h` include chain, so `O_DIRECT` was invisible and
>    the direct-I/O twin descriptors silently became -1. Fixed; worth 2.95x on
>    decode and 1.25x on prefill, output bit-identical. The "isolated direct vs
>    buffered" comparison in this document was real, but the engine and probes
>    were never actually taking the direct path.
> 2. The acceptance ladder **does not pass**: one of five cases completes
>    within 20 minutes. The gap at 32,768 tokens is about 2.6x and the cost is
>    broadly distributed across CPU state transitions, FP8 dense projections
>    and the sparse-attention kernel, so no single optimisation closes it.
>
> The plan below is otherwise still accurate, and steps 1 and 2 are done.

This is the authoritative continuation snapshot for 2026-09-09. Read this before
older appendices in `deepseek-native-wsl-handoff.md`. The user requested a handoff
to another model and a written record of the work and next steps.

## Successor-model continuation status

The handoff has occurred and the successor model is actively continuing the same
workspace task. At the latest 2026-09-09 snapshot, checkpoint recovery is running
under PID 84615 at **80/91 verified groups**, with about **185 GiB free**. It has
not been restarted or duplicated.

Work completed after the handoff:

- Added an independent libm decoder oracle for all 256 FP8 codes and all 256
  UE8M0 scale codes, including signed zero and NaN, and removed decoder
  co-validation from the million-pattern FP8 encoder oracle.
- Rebuilt the current CUDA backend and passed the complete native DeepSeek fixture
  suite. The new independent encoding test passes.
- Rebuilt the four-layer populated prefill probe from current headers and ran the
  exact prepared 512-token review prompt with direct io_uring. It completed in
  50.955 seconds. The first layer spent 32.654 seconds on reads while recovery
  contended for the disk; later layers read in 2.892–3.853 seconds. This is a
  contention-affected partial-model baseline, not complete-model TTFT or decode
  throughput.
- Hardened the complete-model prefill runner: native log records reject malformed
  or duplicate fields, numeric/memory evidence is strictly parsed, allocator and
  device high-water readings are combined, and the full token/chunk schedule is
  explicit. Four focused tests pass.
- Hardened aggregate qualification: exact integer schedules are required; released
  reference evidence must compare 43-layer logits; prefill must contain all 11
  required full-43-layer token/chunk measurements bound to the exact engine and
  manifest; aggregate reviews and memory must exactly match their checksummed
  evidence. Ten tamper/failure tests pass.
- Passed all **79 DeepSeek Python unit and fault tests** and the current LocalForge
  TypeScript check. Refreshed the still-unapplied startup proposal so it includes
  the stronger evidence gates; it remains unapplied pending explicit approval.

The active plan remains:

1. Let recovery finish, then run the independent 91-group byte-integrity checker
   and preserve the 100 GiB free-space floor.
2. Rebuild the complete 43-layer engine from current sources and run full released
   reference/logit parity before treating generation as meaningful.
3. Run populated prefill at 128/512/2048/8192/32768 tokens with chunk 1024, plus
   2048/32768 comparisons at chunks 256/512/2048. Measure actual TTFT, memory and
   I/O. Make only bounded, measured optimizations needed to meet the warm limit.
4. Run all five exact planted-defect reviews, including the 32,768-token case, and
   measure actual decode tokens/second. Every review must be complete, correct and
   at or under 20 minutes; truncation never passes.
5. Assemble checksummed qualification evidence for the exact engine, checkpoint
   and 40,960-token reviewer profile. After the separately pending startup change
   is authorized, validate its compiled integration, then make DeepSeek selectable
   through LocalForge without altering live saved settings during qualification.

## User objective and boundaries

The user said **“dont stop until deepseek is useable as reviewer model.”** Continue
through actual complete-model acceptance. A downloaded checkpoint, a build,
fixtures, partial-model results or estimated performance do not meet that goal.
The latest user question asked for TTFT and generation tok/s: **neither has been
measured for the complete model yet**. Do not present input processing throughput
as generation speed.

The candidate is pinned DeepSeek-V4-Flash-0731, native FP4 experts and FP8 dense
weights, without additional quantization or DSpark. The required review profile:

- Exactly counted, fully rendered input up to32,768 tokens;8,192 tokens shared by
  thinking and final output; total context40,960. Low reasoning effort opens the
  thinking block without the high/max instruction prefix. Omitted effort is
  nonthinking.
- Five complete planted-defect reviews, including an actual32,768-token input;
  every complete warm review must finish within20 minutes. Keep the existing
  one-hour production timeout. Partial/truncated output never approves.
- CPU exact counting before taking the GPU lease. Preserve mandatory context,
  selected source and every changed line; split oversized evidence into required
  coverage passes rather than silently truncating.
- Exclusive GPU sequence: prepare → unload managed orchestrator → reviewer →
  confirm reviewer release → restore orchestrator, including failure/cancellation.
- Host expert budget8 GiB, GPU expert budget2 GiB, scratch at most2 GiB,
  GPU headroom1.5 GiB. Account for actual allocations and shrink/reject if needed.
- Layer-major prefill with bounded256/512/1024/2048 matrix batches; start with1024.
  Report committed input separately from current chunk/layer/activity, thinking
  and output budget used.
- Fixtures → independent released reference → I/O/kernel measurements → full
  43-layer prefill128/512/2048/8192/32768 and chunk comparisons → five reviews.

Do not change/restart live services or activate the reviewer in saved settings.
No GLM probes or full LocalForge E2E. No staging, commits, pushes or broad cleanup.
Both repositories contain extensive pre-existing edits. Do not revert unrelated
work or remove the preserved old partial checkpoint shard.

## Workspace, tools and approval

- Main repo: `/home/dinga/Projects/shiftwing`
- Integration repo: `/home/dinga/Projects/localforge`
- Python with NumPy/Torch/tokenizers: `/home/dinga/Projects/shiftwing/.venv/bin/python`
- CUDA12.9: `/home/dinga/Projects/shiftwing/.toolchains/cuda-12.9`
- RTX5070Ti,16,303 MiB; Ryzen7700X; WSL about29.38 GiB RAM.
- Filesystem sandbox launcher is broken because bubblewrap is unavailable. Scoped
  `exec_command` calls used `sandbox_permissions=require_escalated` with a concise
  justification. Auto-review permits ordinary scoped edits/tests. `apply_patch`
  also failed; use quoted heredocs/Python. No subagents were used.
- Give useful commentary at least once/minute. The user is frustrated by stopping
  at intermediate milestones. Long processes should continue while other useful
  work proceeds; do not restart recovery casually.

**A specific production-startup approval remains pending.** Automatic approval
review rejected replacing the permanent C startup lock and routing native doctor,
saying the broad task authorization was insufficient for that startup change.
The rejected command did not execute. A concrete two-file startup patch was then
prepared and an asynchronous approval question sent. No answer has arrived.

- `/home/dinga/Projects/shiftwing/docs/research/deepseek-qualification-startup-proposal.md`
- `/home/dinga/Projects/shiftwing/docs/research/deepseek-qualification-startup.proposed.patch`
- Generator: `/tmp/prepare_deepseek_startup_patch.py`

Do not apply that patch until approval arrives. It replaces the permanent lock
with validation of passing evidence bound to the exact engine, checkpoint and
profile; it does not activate services. `c/deepseek_v4.c` still returns78 without
`DSV4_EXPERIMENTAL=1`. The actual Python qualification validator still has no CLI.
Native doctor routing is also not implemented. No passing qualification report
exists. The proposal was regenerated after the latest validator hardening and
passes `git apply --check`; recheck hashes/applicability before any approved
application.

## Active checkpoint recovery

At the latest successor-model snapshot: **80/91 groups** (45 dense,3 DSpark,32
expert layers0–31). The latest log had staged source shard34. Always read the
journal for a newer count. All base dense weights are already available.

PID **84615** (parent84607) is still running:

```sh
python3 c/tools/recover_deepseek_v4.py \
  --source c/.deepseek-v4-flash-0731.source \
  --output c/deepseek-v4-flash-0731 --download-workers 12
```

Log:
`/home/dinga/Projects/shiftwing/c/.deepseek-v4-flash-0731.source/recovery-twelve-workers-2026-09-09.log`
Journal:
`/home/dinga/Projects/shiftwing/c/deepseek-v4-flash-0731/conversion-state.json`

Recovery uses exact8 MiB HTTP ranges, bounded12 workers, retries, ownership
inode/device journaling, fsync-before-journal, hashes and independent source vs
output comparisons before releasing owned staging. It has24 passing fault tests.
The old6-worker PID34537 was safely interrupted at65 groups. Restarting incurred
about15 minutes rehashing77 GB; avoid another restart. A later diagnostic paused
PID84615 for11.47 seconds and resumed it in a finally block; resume was confirmed.
**It is not currently paused.** Do not create a duplicate downloader.

Pinned source revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.
Final bytes:166,881,088,004. Preserve100 GiB free floor; the latest disk reading
was185 GiB free. Preserve the pre-existing shard4 partial of
897,581,056 bytes.

Historical dense groups1–3 have independent proof in `historical-validation.json`;
all other groups use `recovery-validation.json`. Historical owned staging was
released after proof. The old `deepseek-recovery-status-2026-09-09.json` is stale.
Final config/tokenizer/manifest are published only when conversion completes.

New final checker: `c/tools/validate_deepseek_v4_recovery.py`. It rebuilds the pinned
source plan from cached headers/index/config, binds both proof ledgers to all
91 groups and inventories, verifies current full segment hashes and metadata,
and checks files did not change during validation. Five tests pass. Run its
`--help` and execute against the complete checkpoint when recovery finishes;
no real complete-conversion validation report has yet been produced.

## Native implementation already present

Core files: `c/deepseek_v4.c`, `deepseek_v4_execute.h`, `deepseek_v4_server.h`,
`deepseek_v4_memory.h`, `deepseek_v4_prefill.h`, `deepseek_v4_attention_batch.h`,
`deepseek_v4_attention_cuda.h`, `deepseek_v4_runtime.h`, `deepseek_v4_dense.h`,
`deepseek_v4_tier.h`, `deepseek_v4_cuda.inc` and `backend_cuda.*`.

The experimental native server has strict length-framed one-slot SUBMIT/DATA/
CANCEL/DONE handling, a separate cancellation reader, strict request bounds,
independent BUSY IDs, cleanup before terminal response and clean next-request
state. Tokenization/counting is CPU-only and pinned; dense weights load once,
expert caches persist, each request gets fresh causal state. Decode is greedy at
zero temperature or deterministic seeded top-p otherwise. Nonfinite logits fail.
EOS1 is distinguished from output exhaustion. Logs bind exact payload, engine,
manifest, tokenizer, timing, reads and termination; actual allocator and scratch
measurements are emitted separately.

The host prefill cache is a whole-layer window, separate from bounded decode
caches. Expert records are read as six-record contiguous extents. Prefill finishes
attention chunks for a layer, then evaluates experts in increasing ID order,
keeping each expert on GPU across its assigned tokens. **Assignments now pack
across the prompt into batches no larger than the configured chunk**, instead of
underfilled batches for every original token interval. This preserves independent
FFN mathematics and expert accumulation order. Progress reports assigned-token
batch size and its first position. The prompt HC/FFN banks are fully accounted.

Attention batches precompute projections. Existing scalar state transitions still
construct each token's query, compression and causal selected indices in order.
A collector preserves the old ring, appends new raw/compressed keys and remaps each
captured selection to a linear chunk KV bank. A single CUDA batch computes sparse
attention, then inverse rotation/output projections follow. `DSV4_BATCH_ATTENTION=0`
allows a diagnostic scalar-attention comparison. Extra host/device buffers are in
the memory plan. CPU HC preparation and activation quantization batch across six
workers. Cancellation clears collector pointers and poisons incomplete state.

**Latest format optimization, just applied:** CPU FP8 encoding/decoding and UE8M0
scales now use exact IEEE bit construction instead of repeated libm calls. CUDA
FP8 decoding/encoding and scale decoding do likewise. No checkpoint values or
quantization precision change. Native fixtures including a million encoding bit
patterns and every rounding boundary pass. CPU quantization of1024×8192 values
improved from about68 ms to27 ms before counting the new parallel batch helper.
CUDA FP4 decoder still uses its old switch; optimizing it is optional, untested.

The complete `c/deepseek_v4` executable is **stale relative to current headers**.
Rebuild it before actual serving/qualification. Current temporary prefix binaries
are also stale relative to the latest format optimization. `c/backend_cuda.o` was
rebuilt for the latest optimization and the full native fixture suite passed.

## Correctness findings and independent evidence

- RoPE must use `1.0f / powf(base, positive_exponent)`, matching pinned FP32 operation
  order. Negative exponent rounding caused long-position errors. New fixture
  generator/test cover16 positions through65,535, plain/YaRN and forward/inverse;
  all four cases are exact against unchanged pinned model.py.
- FP4 ties-even/negative zero, FP8 scale floor1e-4, FP4 floor6×2^-126 are corrected.
- q-vector normalization has BF16 intermediate operations, unlike float RMSNorm.
  HC/RMS reductions accumulate accurately; HC FP32 weights are not quantized.
- wo_a is stored FP8 but evaluated with BF16 activations/dequantized BF16 weights.
  Routed SwiGLU order, route weighting, MoE BF16 output before HC, Hadamard/index
  scaling and index score BF16 dot/product/sum boundaries are corrected.
- Sparse attention uses64-key online softmax, FP32 denominator and BF16
  unnormalized probabilities before value GEMM. Scalar and batched CUDA fixtures
  match an independent pinned-kernel adapter, including variable row lengths and
  masked tails.
- Released expert parity: three experts, CPU/CUDA batch17, exact in the latest
  pre-format-change report `docs/research/deepseek-current-expert-parity-2026-09-09.json`.
- Released dense parity: six projections including router and wo_a pass in
  `docs/research/deepseek-current-dense-parity-2026-09-09.json`.
  Re-run after final kernel changes; these are not whole-model qualification.
- `c/tools/reference_deepseek_v4_prefix.py` loads unchanged pinned model.py Block
  with independent Torch CPU adapters/lazy SHA-checked weights. It now supports
  all43 layers and optional final logits. The full head path has not yet been
  exercised with all43 layers. `--stage-input` supports controlled module inputs.
- Earlier4-layer128-token final hidden error was about3% relative L2;17-layer
  one-token error1.12%, cosine0.999945. Diagnostics are in
  `docs/research/deepseek-prefix-diagnostics-2026-09-09.json`. Complete-model logits
  and meaningful generation remain untested; do not declare numerical readiness.
- New2048-token/four-layer batched attention was **exact at every layer** against
  scalar attention. Packed expert assignments were also exact at every layer
  against the preceding implementation.

Pinned model.py SHA:
`c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f`
Pinned kernel.py SHA:
`59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2`
Pinned tokenizer SHA:
`8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`
**Use the authoritative TOKENIZER_SHA256 constant in `c/tools/deepseek_v4_spec.py`;
verify it rather than copying hashes from prose.**

## Honest performance status and next optimization

**No complete-model TTFT or generation tokens/s measurement exists.**
The last user-facing status said the8,192-token diagnostic took181.5 seconds
through4 of43 layers, about45 input tokens/s. This is not generation throughput.

Representative partial-model results before the latest format optimization:

| Run | Prefill | Notes |
| --- | ---: | --- |
| 512 tokens, old/new expert schedule |44.85 /47.68 s|GPU weight uploads22.83 /11.70 GB; no speedup claimed|
| 2048 tokens, packed expert schedule |86.46 s|4 layers,11.89 GB read/upload,440 MB GPU scratch|
| 8192 rendered review tokens |181.46 s|4 layers,13.06 GB reads,450.7 MB scratch,6.433 GB peak RSS|

8k phase times per layer ranged6.56–34.65 s attention,0.45–0.68 s routing,
6.47–12.09 s reads,11.43–19.43 s routed experts and1.66–1.77 s shared/post.
Recovery causes large I/O timing swings. Controlled warm measurements must follow
completion/settling, not overlap conversion/source hashing.

`DIRECT`, `URING`, `URING_PERSIST` default OFF in `st_env_enabled`. Earlier diagnostic
commands accidentally omitted them. They are now explicit in acceptance runners
and the LocalForge DeepSeek candidate profile. An isolated0.896 GiB comparison
had equal checksums,72 io_uring reads, one setup/11 reuses, no fallback; total
buffered7.31 s versus direct3.99 s, with later direct batches80 MB/78 ms. Log:
`/tmp/deepseek-isolated-io.json`. Recovery was resumed after11.47 s.

Isolated CUDA sparse kernel,1024 queries×128 selected keys: scalar+sync100 ms,
batch24.7 ms. `/tmp/deepseek-sparse-bench.log`.

**Very latest GEMM/copy microbenchmark** (after bit conversions), no checkpoint I/O:
`c/tests/bench_deepseek_v4_gemm.c`, `/tmp/deepseek-gemm-bench.log`.

- FP8 batch1024, rows32768, K1024: warm kernel16.14 ms,4.26 TFLOP/s;
  pageable134 MB download13.4–13.9 ms versus pinned4.75 ms.
  First pageable transfer118.6 ms versus pinned4.79 ms.
- FP4 batch192, rows2048, K4096: warm kernel0.87–0.90 ms,3.6–3.7 TFLOP/s;
  pageable output0.19–0.25 ms versus pinned0.10 ms.

A promising next experiment is reusing a bounded pinned host attention bank and
possibly expert working buffers, instead of malloc/free pageable chunk buffers.
This is **not implemented**. Avoid a fresh cudaHostAlloc/free for each chunk;
measure actual whole-layer benefit and account for retained pinned bytes. Do not
expand whole weights or add quantization. First rebenchmark the latest exact
format/parallel quantization changes, which have not yet had a populated run.

Useful temporary paths:

- `/tmp/dsv4-prefix-subset-path` points to a symlink-only4-layer dense/expert subset
  (1.666 GB dense). `/tmp/dsv4-prefix17-subset-path` points to17 layers (3.713 GB).
- `/tmp/dsv4-prefix-128.tokens`, `/tmp/dsv4-prefix-2048.tokens` (latter repeated128
  token sequence for performance); actual8k review IDs are in prepared cases below.
- Last packed binary: `/tmp/deepseek-prefix-probe-packed` (stale after bit changes).
- Logs: `/tmp/deepseek-prefix-2048-packed.log`, `/tmp/deepseek-prefix-8192-packed.log`.
- Native probe source: `c/tests/probe_deepseek_v4_prefill.c`; build partial layers
  with `-DDSV4_RUNTIME_LAYERS=4`, or omit override for43. Supports `PROBE_TRACE=layers`,
  `PROBE_LEGACY_CHUNKS=1`, and `PROBE_LOGITS=1` only for43 layers. Hard600-second cap.
- All GPU diagnostic jobs were complete at this handoff; only recovery is ongoing.
  Recheck before loading GPU work. Do not manage unrelated processes/services.

## Acceptance tooling prepared, not yet run on the complete model

`c/tools/qualify_deepseek_v4_reviews.py` prepares and measures five standalone
native reviews, without binding an HTTP server. Prepared inputs:
`/home/dinga/Projects/shiftwing/docs/research/deepseek-review-inputs-2026-09-09`.

Cases: upper-bound512 tokens; negative-transfer2048; cache-key8192;
path-containment16384; partial-transaction32768. Every prompt is complete official
low-thinking rendering. Native full token-ID equality and byte roundtrip were
verified for all five. Request/prompt/count/ID files and independent defect oracles
are retained. Four oracle tests pass: malformed/truncated/wrong-location/wrong-
semantics/approval answers fail. The runner verifies prepared case identities,
uses the actual engine with experimental validation, max8192, explicit8/2 GiB
caches,1.5 GiB headroom, direct I/O, and a1200-second case stop gate. It records
answers, thinking, progress, statistics, hashes and native logs. It does not write
aggregate qualification. Actual complete-review measurements have not run.

Newest tool, **only syntax-checked so far**:
`c/tools/qualify_deepseek_v4_prefill.py`. It runs the complete serving executable
with one output token to measure actual prefill. Schedule:1024 chunk at all five
input lengths128/512/2048/8192/32768; other chunks256/512/2048 compare2048 and32768.
It checks requested chunk was admitted, binds native logs to prompt/engine/model,
checks allocator high-water and scratch bytes including device baseline, and
confirms child exit. These diagnostic length-limited requests are never reviews.
Add meaningful fault/measurement-parser tests before relying on it. Its pending
CLI assumes execution via `PYTHONPATH=c .venv/bin/python -m tools.<module>`.

`c/tools/deepseek_v4_qualification.py` has six passing validator tests, strict
profile/current engine+manifest+tokenizer binding, required8 evidence stages,
5 complete reviews under20 minutes including32768 input, and actual memory limits.
**No passing real report, CLI or startup integration.** Possible next hardening:
strict integer elements in prefill/chunk lists; require per-stage full43-layer
measurement contents and bind summary review/memory fields to their evidence
instead of only trusting stage/status/hashes. Regenerate the pending startup patch
if this validator changes; do not accidentally apply its unapproved integration.

## LocalForge work and verification

Implemented in the working tree, no saved settings or live services changed:

- Exact CPU count and bound payload SHA before GPU swap;8k shared output reserve,
 40,960 context, low thinking, native FP4/FP8, DSpark off.
- Diff, work-spec and scaffold selected-evidence splitting. Every coverage pass
  retains mandatory context and must complete/pass. Conflicting complete scaffold
  amendments block and preserve the original. Oversized mandatory context fails
  explicitly. Changed lines/source are not silently lost.
- Cancellation during orchestrator unload restores with a fresh signal. Reviewer
  exit must be confirmed after force stop before restoration. Process groups are
  used on POSIX; unresolved stop remains an error instead of being overwritten.
- UI shows current prefill activity separately from committed tokens and displays
  output budget used, not completion percentage.
- Latest candidate `production-service.ts` engineEnv explicitly enables
  DIRECT/URING/URING_PERSIST and forces DSV4_EXPERIMENTAL=0. This prevents validation
  flags in inherited/configured environments from bypassing the production lock.
  It does not activate DeepSeek or change saved settings.27 focused tests passed.

Earlier89 focused integration tests passed; isolated React production build and
backend TypeScript check passed. Re-run relevant checks after further changes.
Latest profile change has not yet had a new TypeScript check, only those27 tests.

Key files under LocalForge: `src/inference/deepseek-review.ts`, `audit-client.ts`,
`production-service.ts`, `colib-runtime.ts`, `hybrid-coordinator.ts`,
`src/mediated-agent-harness/review-exact.ts`, `scaffold-amendment.ts`,
`src/orchestrator/tools.ts`, `src/server/chat.ts`, stream/parser/UI types.

## Verification and immediate continuation order

1. Recheck recovery PID/journal/free space; keep it running. Do not duplicate or
   casually restart it. It must finish15 remaining groups from the76-group snapshot.
2. Read the latest source changes. Native format-bit suite passed:
   `/tmp/deepseek-format-bits-suite.log`. The million-pattern encoding oracle uses
   the decoder; strengthen it with an independent libm comparison of all256
   FP8/UE8M0 decode codes so the new encoder and decoder do not validate each other.
3. Rebuild a populated prefix probe using current headers/CUDA object. Measure the
   latest format/parallel quantization changes. Experiment with bounded reusable
   pinned buffers only if evidence supports it; retain parity/memory/cancel checks.
4. Larger partial diagnostics can identify bottlenecks while downloading, but
   must not substitute for complete43-layer numerical/prefill/review gates.
5. When all weights finish, validate all91 groups against both independent source
   proof ledgers. Build the complete engine. Run full43-layer independent reference
   and logits, escalating actual prefill/input and chunk gates, with actual memory.
6. Run all five complete planted-defect reviews; inspect actual answers and native
   logs, not just automated semantic anchors. Measure real TTFT, decode tok/s and
   total warm time. Optimize/fix until every required gate passes.
7. Bind actual evidence to the final exact engine/checkpoint/profile. If startup
   approval arrives, apply/test the reviewed gate patch and finish native doctor
   integration within authorized scope. Do not bypass a pending/rejected approval.
8. Leave a selectable, qualified reviewer candidate and report true measured
   performance. Do not switch live settings/services without separate authorization.

Other useful logs: `/tmp/deepseek-python-current-suite.log` (80 passed),
`/tmp/deepseek-packed-expert-suite.log` (native passed),
`/tmp/deepseek-review-oracle-tests.log` (4 passed),
`/tmp/deepseek-localforge-io-profile-tests.log` (27 passed),
`/tmp/deepseek-localforge-final-typecheck.log` (older successful typecheck).
