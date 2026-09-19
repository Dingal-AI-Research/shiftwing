# Native DeepSeek reviewer — WSL Codex handoff

**Current handoff:** read [the model handoff snapshot](deepseek-reviewer-model-handoff-2026-09-09.md) first. It supersedes historical status sections below.

Updated 2026-09-09 after the native WSL continuation. The user asked to continue using Codex installed directly in WSL.
This file records authorized work and implementation state; it is not evidence of qualification.

## Active update — rotary correctness, batched attention and acceptance cases

Work is still active; production is not qualified or usable yet. Do not stop at
this update. Startup approval described below is still pending; do not apply the
proposed startup patch until the user answers. No services/settings were changed.

Recovery PID84615 continues with12 workers; at least73/91 groups (45 dense,
3 DSpark,25 base expert layers). Read its journal for the current count.

New fixes and checks:
- RoPE now computes `1.0f / powf(base, positive_exponent)`, exactly matching the
  pinned operation order. The prior negative exponent introduced frequency
  rounding differences magnified at long positions. New pinned model.py fixture
  generator and C test cover16 positions through65535, both plain/YaRN and
  forward/inverse; all four cases are exact. Native fixture suite passes.
- Native prefill now optionally batches sparse attention for a whole bounded
  chunk. Existing scalar transitions still construct queries, compression and
  causal selections in order. A collector saves the prior ring and every new raw
  key, remaps captured selections to a linear KV bank, then runs one CUDA batch.
  No future key becomes selectable. The same kernel preserves64-key online
  softmax. CPU/CUDA causal fixtures remain exact for all chunk sizes. A batched
  independent sparse fixture also verifies varying row lengths ignore tail keys.
- Attention HC preparation is parallel across six CPU workers. All extra host
  and GPU collection buffers are included in the memory plan. GPU scratch for a
  four-layer2048-token diagnostic was433,873,472 bytes; GPU pool peak4.239GB.
- Released-weight2048-token/four-layer outputs match the preceding scalar
  attention run **exactly at every layer**. Timings are badly affected by storage
  stalls during recovery:97.52s prior vs212.45s batch, with a134s first-layer read
  stall in the latter. Do not claim an end-to-end speedup.
- Isolated sparse kernel benchmark (`c/tests/bench_deepseek_v4_sparse.c`) measures
 1024 queries: scalar+sync0.100s versus batch0.0247s. Log:
  `/tmp/deepseek-sparse-bench.log`. This isolates kernels, not whole-model timing.
- Diagnostic probes previously omitted `DIRECT=1 URING=1 URING_PERSIST=1`.
  `st_env_enabled` defaults these OFF. The actual LocalForge DeepSeek profile also
  lacks these settings. A current explicit-direct run is in session87447,
  `/tmp/deepseek-prefix-2048-direct.log`; binary
  `/tmp/deepseek-prefix-probe-batch-attention`. Read/compare before changing the
  candidate launch profile. No I/O profile patch has been applied yet.

New standalone acceptance tool: `c/tools/qualify_deepseek_v4_reviews.py`. It
prepares/validates exact API-rendered requests and measures native experimental
serving without binding a server or changing live services. Five prepared cases
are in `docs/research/deepseek-review-inputs-2026-09-09`, with native-verified
counts512,2048,8192,16384,32768. Cases cover upper bounds, negative transfers,
incomplete cache keys, path containment and partial transactions. A concise
structured verdict must identify the planted file/line and semantic anchors;
malformed/truncated/incorrect answers never pass. Four oracle tests pass.
**No actual review measurements have run.** The tool writes only review-stage
results, not aggregate production qualification. Its cache env now correctly
usesRAM_GB/CUDA_EXPERT_GB/CUDA_HEADROOM_GB. It still needs explicit I/O settings
aligned with the final measured profile, strict prepared-case identity checking,
and actual native run validation after the checkpoint completes.

Latest test logs: `/tmp/deepseek-python-current-suite.log` (80 passed),
`/tmp/deepseek-batch-attention-suite.log` (native suite passed),
`/tmp/deepseek-review-oracle-tests.log` (4 passed). LocalForge final TypeScript
check passed. Latest native probe supports PROBE_LOGITS only for43 layers and
writes per-layer phase timing (attention, routing, reads, routed experts,
shared/post). Pinned CPU reference head code was cleaned up; full43-layer
reference/logit comparison is still required.

Next: isolate storage contention and compare direct I/O; measure larger partial
prefixes only as diagnostics while recovery continues; finish complete-source
validation, full43-layer numerical/prefill/chunk/memory measurements and five
complete reviews. Production startup patch and native doctor remain pending
approval/integration. All previous notes below are historical where superseded.

## Latest continuation addendum — 2026-09-09, still active

The user still requires **do not stop until DeepSeek is usable as reviewer**.
This addendum supersedes earlier counts and numerical notes below.

Recovery was safely interrupted at65 groups and resumed with
`--download-workers 12` (new bounded CLI accepts1–24, default6). Current PID84615,
log `c/.deepseek-v4-flash-0731.source/recovery-twelve-workers-2026-09-09.log`.
All completed outputs were rehashed on resume; downloading is active again.
Check the journal for current count (at least67). Prior PID34537 exited.
All24 recovery fault tests pass. Preserve the old partial shard4.

New numerical fixes and evidence:
- Pinned sparse attention uses **64-key online softmax, FP32 denominator, BF16
  probabilities before each value GEMM**. Both native CPU/CUDA and the independent
  CPU adapter were missing this. New exact CPU/CUDA fixtures cover selected-key
  counts1,63,64,65,127,128,137 against independent pinned-kernel equations.
- Index dot/product/final-sum BF16 boundaries now match upstream. FP4 ties/negative
  zero and heap selection have regressions. Routed SwiGLU ordering and FP32 SiLU
  extreme underflow match pinned Torch. Released experts remain exact CPU/CUDA
  at batch17: `deepseek-current-expert-parity-2026-09-09.json`.
- HC dot products accumulate in FP64, cast before FP32 scaling; controlled HC mix
  error improved about10x. Earlier accurate RMS square sums remain. Fixtures pass.
- Controlled first-layer attention relative L2 is0.1384%; controlled FFN0.00356%.
  Full4-layer128-token final hidden differs3.09%;17-layer1-token finishes at1.12%,
  cosine0.999945. These are diagnostics, not qualification. Report:
  `deepseek-prefix-diagnostics-2026-09-09.json`. The independent adapter now has
  the full43-layer compression schedule and optional final logits at43 layers.

**Newest prefill implementation:** `dsv4_prefill_layer_prompt` finishes causal
attention chunks for one layer, then keeps each expert on GPU across all prompt
chunks. It adds full-prompt FFN input/sum, post/comb and routing arrays; all are
included in `dsv4_prefill_bank_bytes`. Matrix work remains chunk-bounded. Router
BF16-weight/FP32-output projections batch on CUDA. HC FFN preparation parallelizes
across six CPU workers. Expert accumulation follows increasing expert id in decode
and both old/new prefill paths, matching upstream.

Current full native fixture suite passes. The cache fixture holds only top-k
experts (fewer than the working set) and proves uploads stay<=layers*experts for
all chunk sizes. Cancellation while an expert is held releases references and
poisons state. Complete-checkpoint contract checking now has a separate target:
`make test-deepseek-contract DSV4_SNAPSHOT=...`; it requires a real snapshot.
The prefix probe defaults to the new schedule; `PROBE_LEGACY_CHUNKS=1` permits A/B
checks. Probe stage traces now write at the correct chunk offset. Rebuild probes.
Six released dense projection cases, including the new router, pass CPU/CUDA:
`deepseek-current-dense-parity-2026-09-09.json`.

LocalForge now splits selected source for work-spec/scaffold reviews too. Every
part retains mandatory context, and all must pass. Oversized mandatory context
fails explicitly. Conflicting complete scaffold amendments preserve the original
and block; no response silently wins. Exact counting bypasses the old estimated
work-spec gate. All89 focused LocalForge tests and backend TypeScript check pass.
No services, full E2E or live configuration changes.

New final checker `c/tools/validate_deepseek_v4_recovery.py` rebuilds the source
plan from the pinned index/cached headers and binds current segment hashes and
inventory to both independent source-comparison ledgers after staging is released.
Five tampering/released-staging tests pass. Run against all91 groups when complete.
Qualification validator is hardened with six passing tests; no passing report
exists and no CLI/startup integration is applied.

**Startup approval is pending** from an asynchronous user question. Unapplied
patch: `deepseek-qualification-startup.proposed.patch`; explanation:
`deepseek-qualification-startup-proposal.md`. Approval covers replacing the
permanent lock with exact engine/checkpoint/profile qualification checks in two
files. It does not activate the reviewer or edit live services/settings. Automatic
approval review had rejected this startup change earlier. Do not apply without
the answer. Native doctor routing remains separate and pending. Production is
still hardlocked. No startup/doctor changes from the rejected command were applied.

Next: benchmark the new schedule at128 then512/2048 bounded prefix sizes; finish
full-model reference, prefill/chunk/I/O/memory and five complete planted-defect
review checks when recovery finishes. Keep the one-hour production timeout.
Do not label partial models, fixtures, downloads or estimated timings ready.

## Current continuation — 2026-09-09, still active

The user explicitly said: **do not stop until DeepSeek is usable as reviewer model**.
Do not end at a download, build or fixture milestone. The older status sections
below are historical; this section and the on-disk journals supersede them.

Recovery remains unbounded in PID34537, log
`c/.deepseek-v4-flash-0731.source/recovery-continue-2026-09-09.log`.
At this update, **62/91 groups** are complete. Check the journal for the current
count. Historical dense groups1–3 now have independent source-byte comparisons in
`c/deepseek-v4-flash-0731/historical-validation.json`; owned historical staging was
released. The main and historical ledgers together cover all completed groups.
The historical resume helper now checks inventory binding and releases only its
own group instead of restaging already-proven groups. The old partial shard4 is
still preserved. No live services/configuration were changed, no commits made.

New native work:
- Length-framed one-slot experimental serving, cancellation/reset/error handling,
  pinned tokenizer and per-record SHA256 verification remain implemented.
- Exact FP8 direct encoder replaces the exhaustive 127-code scan; one million
  float patterns and every rounding boundary match the original encoder.
- Native FP8/FP4 tensor-core tiles decode only bounded shared tiles. Independent
  released-expert parity at batch17 is exact on CPU and CUDA for three experts.
- `deepseek_v4_attention_batch.h` batches projections while reusing the ordered
  scalar ring/compressor/indexer transitions. CUDA sparse attention and index
  scoring retain reusable bounded state buffers in `deepseek_v4_attention_cuda.h`.
- Index top-k now uses a heap with the original tie order, removing the former
  quadratic scan of already selected entries.
- Upstream `inference/model.py` and `convert.py` exposed semantic gaps: wo_a is
  stored FP8 but evaluated using BF16 activations and dequantized BF16 weights;
  q normalization has BF16 intermediate rounding; MoE output rounds to BF16
  before HC; Hadamard and index-head scaling also require BF16 boundaries.
  These are corrected. FP4 ties now use nearest-even and simulation scale bounds
  match the pinned kernels. Native checkpoint bytes are unchanged.
- RMS/HC square sums now accumulate accurately before the FP32 mean/RMS step,
  reducing long serial-FP32 reduction drift. This latest change is under released
  prefix comparison; do not assert full numerical qualification.
- Optional runtime trace callbacks are used by the standalone prefix probe only.
  Actual CUDA allocator used/reserved high-water marks and allocated scratch
  capacities are exposed in DSV4_ALLOCATIONS. Run a fresh probe for current metrics.

Current numerical evidence:
- Expanded 137-token/four-layer fixtures include sliding, overlap and the 128-token
  compressor boundary. Scalar vs batch and CUDA whole-prompt chunks1,35,69,103,137
  all match exactly. Cancellation and memory-plan fixtures also pass.
- `deepseek-released-dense-parity-2026-09-09.json` compares five released FP8/BF16
  projections to independent Torch equations. All pass; wo_a now tests its correct
  BF16 arithmetic. This is not whole-model qualification.
- `reference_deepseek_v4_prefix.py` loads the pinned upstream Block code unchanged
  and adapts TileLang primitives/lazy weight reads to CPU Torch. It writes per-layer
  and component outputs. `--stage-input PREFIX` compares modules on identical
  native inputs. Reference runs are bounded to four layers and128 tokens.
- Initial populated four-layer128-token baseline:203.687s prefill,120.590s load.
  Corrected optimized trace:23.305s prefill,3.686s load of a1.666GB dense subset,
  10.963GB expert reads. Load figures are not comparable across full/subset arenas.
- The corrected native trace differs from the independent full prefix by1.49%
  relative L2 at layer0 and3.26% at layer3. Controlled-input tracing places initial
  drift in RMS normalization; quantization amplifies it. Do not scale until this
  is understood and accepted component/full-model criteria are met. A newer
  normalization run is in `/tmp/deepseek-prefix-128-normalized.log`.
- Original full-prefix reference: `/tmp/dsv4-reference-128.hc.layerN`.
  Corrected native trace: `/tmp/dsv4-prefix-128-trace.hc.layerN[.stage]`.
  Controlled reference: `/tmp/dsv4-reference-controlled-128.hc.layerN[.stage]`.
  Current normalized native: `/tmp/dsv4-prefix-128-normalized.hc.layerN[.stage]`.
  Prefix snapshots are symlink-only temporary subsets; path files are
  `/tmp/dsv4-prefix-probe-path` and `/tmp/dsv4-prefix-subset-path`.

LocalForge work and verification:
- Exact CPU prompt counting before GPU swap, bound prompt SHA, profile budgets,
  selected-evidence/diff splitting, low thinking, progress and output-budget UI
  are implemented in the working tree. Live settings remain unchanged.
- The oversized request byte limit is checked before spawning the counter.
- Fitting selected source can get its own required coverage pass when it leaves
  no room for a changed line. A new regression fixture is being finalized.
- Cancellation during orchestrator unload restores it with a fresh cleanup
  signal. An unconfirmed reviewer exit blocks restoration. A fault test found
  the outer lease overwrote that error status with stopped; that is fixed.
- Isolated React production build succeeded in `/tmp/deepseek-localforge-react-build`;
  backend TypeScript check succeeded. No LocalForge E2E/services were started.
- Re-run focused suites after the current changes; logs:
  `/tmp/deepseek-localforge-cleanup-tests.log`, `/tmp/deepseek-norm-fixture.log`.

Production startup remains locked. A Python qualification validator exists at
`c/tools/deepseek_v4_qualification.py`, but no real passing report exists.
**Automatic approval review rejected** the proposed change replacing the permanent
startup lock with identity-bound qualification checks and adding native doctor
handling, saying that broad production startup change lacked authorization.
The rejected command did not execute: no C startup/doctor/launcher gate changes
from that command were applied. Do not bypass this rejection. Prepare a concrete
reviewable patch and obtain the required approval when the unaffected work is
complete, or establish additional authorization/low-risk evidence before retrying.
The user has already been told about the rejection. No approval question is pending.

Outstanding: finish numerical reference investigation; final conversion integrity
and manifest checks; current full native build; full43-layer populated prefill
128→512→2048→8192→32768 and chunk256/512/1024/2048 comparisons with stop gates;
five complete planted-defect reviews including actual32768 input, each warm≤20min;
actual memory/headroom and progress/cancellation evidence; complete LocalForge
oversized scaffold/work-spec paths; qualification report and approved startup/doctor
integration. Keep the production one-hour timeout; incomplete reviews never approve.

## User-authorized objective and restrictions

Implement the native Shiftwing DeepSeek-V4-Flash-0731 reviewer and LocalForge integration.
Use the pinned checkpoint from c/tools/deepseek_v4_spec.py:
deepseek-ai/DeepSeek-V4-Flash-0731 at
9e165c30e2704aec5d9d593cce3eebd58bbef1cb.

Requirements:
- 32,768 ACTUAL fully rendered input tokens, 8,192 combined thinking/answer tokens,
  40,960 total context. Existing engine maximum stays 65,536.
- Low thinking opens the thinking block WITHOUT high/max instruction prefix.
  Wrapper supports low/high/max; omitted effort remains non-thinking for compatibility.
- Complete warm review within 20 minutes is a qualification criterion, NOT a prediction.
  Keep the production one-hour timeout. Truncated/incomplete answers never approve.
- Native FP4 experts and FP8 dense weights; no additional quantization or DSpark.
- Blocking reviews with exclusive GPU use. Prepare with orchestrator, unload its
  managed model process, run reviewer, release reviewer, restore orchestrator even
  on cancellation/failure. Preserve conversation/original request, stale checks and
  authoritative scaffold replacement.
- Do NOT restart existing services, activate the candidate or alter live reviewer
  configuration without separately requested rollout. No GLM probes/full LocalForge
  E2E run during implementation.
- Recover/convert incrementally with bounded staging. Preserve existing models and
  a 100 GiB free-space floor. Delete newly downloaded staging only after conversion
  output and integrity evidence have been verified.
- Contiguous expert weight+scale reads, persistent asynchronous reads and bounded
  pinned staging. Compare buffered/direct modes on identical extents.
- Layer-major chunk prefill, token routing grouped by expert, one expert load per
  chunk/layer. Preserve causal/sliding/compression/routing state and original
  quantization semantics. GPU dense/expert matrix multiplication; bounded expansions.
  Evaluate chunks 256,512,1024,2048; start1024.
- Initial budgets host expert cache8GiB, GPU expert cache2GiB, GPU scratch<=2GiB,
  GPU headroom1.5GiB. Account dense8.24GiB, all state/staging and actual allocations;
  shrink cache/chunk or fail visibly if minimum cannot fit.
- Native engine protocol: request/stream, generation/stops, cancellation, clean reset,
  errors and ONE review slot. Production remains unavailable until qualification.
- LocalForge profile-specific input ceiling (DeepSeek32768, existing8K defaults).
  Exact DeepSeek tokenizer count over fully rendered prompt. Mandatory instructions
  and selected source/diff evidence preserved exactly. Split oversized reviews;
  never silently remove changed code.
- Expanded tool panel shows loading, committed prompt tokens/total, current chunk
  activity, throughput, thinking, answer generation, output budget and restoration.
  Output budget is NOT completion percentage.
- Log exact payload, checkpoint/build identity, thinking effort, timings, memory
  peaks, expert-read bytes and termination reason.
- Validation: fixtures/mocks -> independent released-weight/reference checks ->
  I/O/kernel probes -> populated prefill128,512,2048,8192,32768 with stop gates ->
  five planted-defect standalone Shiftwing reviews with long-context evidence.
  Stop scaling on correctness failure or measured projections beyond target.
  Report checkpoint load/restore separately and total visible wait.
  Do not mark ready unless correctness/quality/memory/progress/time all pass.

## Workspace

Shiftwing: /home/dinga/Projects/shiftwing
LocalForge: /home/dinga/Projects/localforge
Both working trees contain extensive PRE-EXISTING modifications. Do not reset,
clean, or blindly stage them. No commits/pushes were made in this implementation.
No AGENTS.md was found in the inspected repo/root/c tree. Recheck applicable rules.
No subagents were requested or used.

Hardware previously recorded: Ryzen7700X8C16T, RTX5070Ti16303MiB,
WSL RAM29.38GiB. Orchestrator occupies most VRAM while loaded. Do not load a full
candidate alongside it. Native CPU/disk path was vastly too slow before this work.
Keep candidate disabled if measured gates cannot be met.

## Implemented so far (unfinished, unqualified)

1. c/tools/deepseek_v4_spec.py: review budget constants; low/high/max effort support.
   c/tools/deepseek_v4_protocol.py: official pinned encoder wrapper, thinking mode
   for non-None effort. Official encoding SHA256 remains
   abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6.
2. c/openai_server.py: explicit DeepSeek family detection/render/response parsing,
   DeepSeek sampling profiles and effort validation. Reject recovered malformed
   responses; validate tool names. Other pre-existing edits preserved.
3. c/shiftwing: restores DeepSeek binary selection, --dspark off and
   --experimental-deepseek flags. DeepSeek env sets exclusive defaults8/2/1.5GiB
   and chunk1024; rejects Q3/DSpark. Doctor/default context still need review.
4. c/tools/runtime_env.py: restored deepseek_engine_source_sha256().
   Include all new source files in identity and isolate DSV4 env knobs.
5. c/Makefile, backend_cuda.cu/.h, deepseek_v4_cuda.inc: restores native DeepSeek
   build targets and shared CUDA integration, scratch fields and declarations.
   Added BF16 include. Existing kernels are basic, not yet optimized tensor-core.
6. c/st.h: explicit contiguous extent support through persistent io_uring plus
   buffered/direct fallback and metrics. Existing tensor batch API preserved.
   c/deepseek_v4_tier.h: six expert records -> one validated extent per expert,
   up to TOPK extents/read. Bounds/overlap/same-fd checks. malloc staging currently
   NOT reusable pinned storage. Record checksum validation before trust still needs
   strengthening. Remove unused requests variable warning.
7. c/deepseek_v4_dense.h: shared-expert batch API with single-token wrapper.
8. NEW c/deepseek_v4_prefill.h: bounded layer-major chunk path, causal per-token
   attention within each layer, group tokens by expert, batched expert GPU/CPU
   projection, route-weight application BEFORE intermediate quantization, retain
   original route reduction order. Progress callback can cancel; partial attention
   is poisoned, committed position unchanged. Head computed only at final chunk
   token when logits requested. Attention projections are still per-token and may
   remain a major bottleneck. Native server does NOT yet call this header.
9. NEW c/tests/test_deepseek_v4_prefill.c: reuses tiny runtime fixture, nonuniform
   weights and token-dependent routing. Chunk sizes1,4,7,10,13 match scalar logits,
   HC state and sliding history exactly; cancellation/context checks pass.
   This is NOT independent released-weight evidence.
10. c/tools/recover_deepseek_v4.py and converter hooks now have tested sparse range
    resume. Verified ranges are rehashed and reused; only missing spans are fetched.
    Up to six 8 MiB requests run concurrently, with three attempts for transient
    network failures. Hash records follow data fsync. Exclusive creation, inode/device
    ownership, symlink/hardlink rejection, a process lock, and sparse allocation/floor
    checks protect existing files. Successful in-flight transfers survive another
    request's failure. Independent per-record source/output comparison and a durable
    recovery-validation.json precede owned staging deletion. Converter resume now
    finishes cleanup after a crash between output commit and staging release.
11. NEW c/deepseek_v4_tokenizer.h, deepseek_v4_unicode.h, deepseek_v4_tokenize.c:
    separate native DeepSeek split sequence, preserving existing Qwen/GLM behavior.
    Generic tok_encode failed 382/3287 released-tokenizer cases; the new wrapper
    matches all 7630 expanded reference cases, including actual 32768/32769-token
    fully rendered prompts. Use dsv4_tok_load/dsv4_tok_encode in the future server.
    Native serving is still not implemented and therefore does not call this yet.
12. NEW c/tools/deepseek_v4_count.py; c/openai_server.py now provides
    POST /v1/chat/count_tokens and --tokenizer-only. Counting and generation share
    the same render path; exact prompt, token count, fits flag, budgets and identity
    hashes are returned. No model weights/CUDA are needed. Tokenizer-only startup
    constructs no Engine and generation endpoints return 503. Actual HTTP tests
    confirmed 32768 fits / 32769 does not, in roughly 0.4 seconds per count.
    Omitted DeepSeek effort remains non-thinking even if COLI_THINK=1.
13. c/tools/runtime_env.py now removes all DSV4_* controls and DSPARK from isolated
    environments unless explicitly preserved. New native files are included by the
    existing DeepSeek source-fingerprint glob. See deepseek-exact-count.md for usage.

## Latest continuation status

Historical checkpoint below: the earlier run paused at **46/91 groups**. This
pause was a mistake; the user explicitly required continuing until the reviewer is usable.
Recovery resumed without a stop-after limit and is RUNNING in
`recovery-continue-2026-09-09.log`. Read conversion-state.json for current progress.
The historical boundary included
including all 45 dense groups and dspark/model-00046-of-00048.bin. The bounded
resume succeeded after the upstream timeout, reused 224 verified ranges
(1,879,048,192 bytes), and fetched only the remaining payload spans. All newly owned
staging was released after independent source/output comparison. No recovery
process remains. The old 897,581,056-byte partial is preserved.

43 groups have new independent byte-comparison evidence; the first three historical
groups still need that evidence. The next group is dspark/model-00047-of-00048.bin,
then shard 48 and the 43 base expert layers. Production remains unqualified.
Machine-readable status: deepseek-recovery-status-2026-09-09.json.

This continuation's final focused suite passed **82 tests** in the repo virtual
environment: recovery (24), exact-count/gateway (10), converter/tooling, independent
conversion validator, DeepSeek protocol, Qwen/GLM rendering and mocked HTTP, and
runtime environment isolation. The existing C digit-tokenizer regression also
passed. System Python lacks transformers; use .venv/bin/python for the reference
renderer tests. No full-model GLM probe or LocalForge application E2E was run.

Reference tokenizer parity: **7630/7630 exact ID matches**, all byte round trips,
including actual 32768/32769-token fully rendered inputs. A real ephemeral HTTP
counter returned the correct fits flags in approximately 0.4 s without an Engine.
Evidence and reproduction commands are in deepseek-exact-count.md and the adjacent
JSON reports. That temporary counter was closed after the check.

No LocalForge files were edited during this continuation. Native experimental
serving, memory preflight, exact-count integration into LocalForge, expert staging
and checksums, GPU prefill optimization, and inference/quality/time qualification
remain to implement or validate. Neither native tokenizer parity nor checkpoint
conversion qualifies the reviewer.

WSL environment note: sandbox commands and apply_patch could not start because
bubblewrap is missing. This continuation used explicitly escalated shell commands
for reads, edits and tests; no system package or service was changed.

## Validation already run

- python3 -m unittest c.tests.test_deepseek_v4_protocol
  c.tests.test_openai_server_deepseek c.tests.test_deepseek_v4_tooling:
  22 tests PASS after launcher integration.
- make -C c CUDA=1 deepseek_v4 -j2: PASS after restoring CUDA integration,
  BEFORE the new prefill header (which is not wired into native main).
- make -C c CUDA=0 tests/test_deepseek_v4_prefill
  env OMP_NUM_THREADS=2 c/tests/test_deepseek_v4_prefill: PASS.
- No real-model inference, production rollout or service restart in this implementation.
- New chunk GPU path has not been exercised. Compressed/indexed chunk parity and
  independent reference comparisons still required. Never weaken tolerances.

## Recovery state — inspect before resuming

Source metadata and pinned reference:
  c/.deepseek-v4-flash-0731.source/
Converted output/state:
  c/deepseek-v4-flash-0731/conversion-state.json
Independent per-group comparison evidence:
  c/deepseek-v4-flash-0731/recovery-validation.json

48 shard headers are cached; the plan has 91 groups and 166,881,088,004 output bytes.
The current count/process state is recorded in the continuation status below;
conversion-state.json remains authoritative. Retain all verified outputs.

The exact-range recovery implementation is now tested: 24 fault-injection/local
HTTP tests pass. Actual pinned HTTP206/Content-Range was verified on a 512-byte
range, then on converted dense groups. Maximum sparse staging for one planned
conversion group is 3,692,777,472 bytes. The 100 GiB floor is still enforced.

A full recovery invocation advanced from 4 to 45 converted groups, then stopped
on an upstream read timeout during dspark/model-00046-of-00048.bin. It preserved
224 verified ranges (1,879,048,192 bytes). A bounded resume invocation reuses those
ranges and stops after that group; inspect the continuation status below.

Logs:
- c/.deepseek-v4-flash-0731.source/recovery-run-2026-09-09.log
- c/.deepseek-v4-flash-0731.source/recovery-resume-2026-09-09.log

The original .model-00004-of-00048.safetensors.partial (897,581,056 bytes) is preserved.
No broad cleanup was performed. Files predating range recovery are never claimed
for deletion. An ambiguous crash between exclusive file creation and the ownership
identity journal deliberately stops and preserves the file instead of adopting it.

The first three groups were converted by the previous implementation and have no
new independent recovery-validation.json entries. Their segment hashes are checked
on resume; obtain independent source comparisons for those groups before treating
the full conversion as independently verified. Recovery comparison records validate
native bytes, not numerical inference or model quality. The separate full converter
validator still expects source shard files; it cannot directly validate deleted
sparse staging without an appropriate evidence-aware follow-up.

Resume under the recovery lock (already enforced by the CLI):

```sh
python3 c/tools/recover_deepseek_v4.py \
  --source c/.deepseek-v4-flash-0731.source \
  --output c/deepseek-v4-flash-0731
```

Use --stop-after-groups N for an explicit durable boundary. Corrupt verified ranges,
changed file identities, missing owned staging, wrong HTTP ranges, or insufficient
free space fail visibly. Do not delete integrity evidence or unverified files to
force a restart.

## Active continuation after the user required no further milestone stops

The recovery command is still running without --stop-after-groups. Do not stop at
another download/build milestone. No services or live settings have been changed.

New native implementation (still experimental and not model-qualified):
- deepseek_v4_memory.h accounts dense host/device arenas, host attention state,
  persistent and chunk scratch, staging, request memory and a full-prompt HC bank.
  It checks actual available host/device memory before dense loading, reduces
  cache/chunk or rejects. Smoke has the same preflight. GPU headroom minimum1.5GiB.
- deepseek_v4_server.h: one-slot SUBMIT/DATA/DONE/ERROR with an independent reader
  accepting CANCEL while the worker computes. Invalid framing closes the process;
  supported bounded frames with bad parameters are consumed/rejected. Runtime is
  reset before terminal output. Ten protocol/hash tests pass.
- deepseek_v4_execute.h connects exact pinned tokenization, prefill, sampling,
  EOS/length termination, cancellation, fresh request state, and identity/timing/
  read-byte logs. Native serving now exists behind DSV4_EXPERIMENTAL=1. Production
  remains locked until a real qualification gate is implemented and passed.
- deepseek_v4_sha256.h verifies dense/expert record bytes before trust; SHA256
  dynamically uses installed libcrypto.so.3 acceleration, with a tested portable
  fallback. Native cache read staging is reusable and pinned when CUDA is enabled.
- The first real expert I/O probe (two completed layers, identical extents, recovery
  running concurrently) read481296384 bytes in0.683s buffered and0.664s direct:
  0.656/0.675GiB/s. Direct:36 reads, one io_uring setup/five reuses/no fallbacks.
  Both modes had matching checksums. This is NOT a full-model performance gate.
- Based on that bottleneck, prefill now processes the whole prompt layer by layer
  in bounded chunks, retaining/recycling one layer's experts. It owns a host HC
  bank (2GiB at32768 tokens) and frees the decode host cache before that expert
  window. Each layer's expert is read at most once across all prompt chunks.
  Two-layer nonuniform scalar comparison and cancellation tests pass for chunks
  1,4,7,10,13. The entire prompt commits only after every layer succeeds; current
  chunk/layer activity is separate from committed tokens. GPU/dense batching and
  released-weight numerical qualification still need work.
- API forwards PREFILL_ACTIVITY (optional chunk_start), and validates
  expected_prompt_sha256 at generation against its shared renderer.
- CPU-only count CLI: from c, python3 -m tools.deepseek_v4_count --snapshot PATH,
  accepting one chat request JSON on stdin; no port or weights needed.

LocalForge source changes now exist (no live settings changed):
- inference/deepseek-review.ts pins32768/8192/40960 and source/tokenizer/encoder
  hashes, invokes the CPU count CLI, validates its response, binds the request hash.
- production-service/colib-runtime/types propagate an explicit inactive DeepSeek
  profile,8/2GiB caches, DSparkoff/nativeweights and low thinking.
- server/chat.ts checks all packet counts before a GPU swap; audit-client compares
  the actual request to the counted request and sends expected_prompt_sha256.
- review-exact.ts preserves all selected source and diff changed lines while
  partitioning with actual rendered counts; existing model defaults stay8K.
  Oversized indivisible work-spec/scaffold packets currently reject without dropping
  evidence; further scaffold partition handling may be needed.
- Cleanup catches cancellation during orchestrator unload. Colib now fails if an
  owned process does not exit after SIGKILL, retaining its PID; restoration is
  blocked until GPU release is confirmed, instead of starting two model processes.
- Expanded review UI shows committed/total input, chunk position/layer, throughput,
  thinking, output budget (no misleading completion percentage), restoration.
-77 focused mocked LocalForge tests pass and backend TypeScript check is clean.
  New tests: deepseek-review.test.ts. Add explicit stuck-exit/unload-cancel tests.
- CUDA build passed. Existing bounded CUDA FP8/FP4/grouped fixtures passed (FP8/FP4
  exact; BF16 max error2.2888e-5). Whole-prompt CUDA validation remains to run.

Outstanding: finish recovery + historical3-group independent validation; native
  integrity fault tests and memory allocation accounting; doctor/qualification gate;
  stronger multi-mode/real-weight/reference and populated prefill gates; optimize
  measured bottlenecks; five complete planted-defect reviews. Do not claim ready.

## Native server outstanding

c/deepseek_v4.c currently only LOAD_ONLY/smoke; serving ends locked with code78.
Implement experimental serving with robust memory preflight BEFORE huge allocations.
Production unlock must be bound to manifest/tokenizer/build and actual passing gates.

Existing Python Engine expects:
- READY bytes: \x01\x01READY\x01\x01\n
- stdin: SUBMIT id slot payload_bytes max_tokens temperature top_p [grammar_bytes]\n
  then raw prompt and grammar, then newline; CANCEL id\n.
- stdout:
  DATA id bytes\n<raw bytes>\n
  PREFILL_BEGIN id total cached\n
  PREFILL_PROGRESS id committed total elapsed_ms [milli_percent]\n
  PREFILL_END id total elapsed_ms\n
  DECODE_PROGRESS id generated maximum elapsed_ms\n
  DONE id STAT completion_tokens tok_s cache_hit_percent rss_gb prompt_tokens length_limited\n
  ERROR id code details\n
- EOF/incomplete/length limits must never approve.
- Cancellation must be read while worker computes; clean request reset closes and
  reinitializes dsv4_runtime while keeping shared dense/expert cache as appropriate.
- Use the newly verified dsv4_tok_load/dsv4_tok_encode path, not generic tok_encode.
  Allocate an input-byte-count upper bound and enforce the actual input/output
  context budgets. The helper pins tokenizer SHA256 in deepseek_v4_spec.py.
- Exact count/render endpoint is implemented and tested without candidate CUDA
  loading. LocalForge still needs to call it before entering the GPU swap lease;
  preserve its exact rendered payload/hash through eventual submission.

Runtime APIs:
dsv4_runtime_init/close/decode_token, dsv4_runtime_prefill_chunk.
Attention state is host-backed currently; account its real location.
dsv4_runtime_state_bytes(context), dsv4_prefill_host_bytes(chunk).
Dense arena8.24GiB duplicated host/device when CUDA enabled.
Cache capacity per layer includes base+DSpark layers currently; budget accurately.
coli_cuda_memory_info available.
Quantization functions in deepseek_v4.h; FP4 activation scales are per128,
expert weight scales per32. Route multiplication occurs before middle quantization.
Existing CUDA grouped-expert path is one-token/many-experts; prefill helper implements
one-expert/many-token calls via three native FP4 GEMMs and CPU middle quantization.

## LocalForge outstanding (no edits during THIS implementation yet)

Existing integration from prior work:
- src/inference/hybrid-coordinator.ts auditWithinLease already serializes GPU swap:
  unload previous llama target, colib.withModel(auditor397b), finally restore using
  cleanup signal. Verify reviewer release happens BEFORE orchestrator restore.
  auditor397b is logical role; do not assume actual model architecture.
- src/inference/production-service.ts resolves reviewer settings; special GLM
  output8192/non-thinking behavior currently. Add candidate profile without activation.
- src/mediated-agent-harness/review-preparation.ts has fixed input8192,
  preparation input16384/output2048. Add profile-driven limit and exact final count.
- Existing orchestrator preparation JSON parsing was repaired earlier; preserve.
- src/server/chat.ts preparation happens before audit lease.
- src/inference/review-preparation-client.ts and review-request-log.ts exist.
- Preserve exact selected evidence/original request across compaction and swaps.
- Add profile/count/progress/cleanup mocks; do not run full application E2E first.

## Next work

Finish checkpoint recovery and independent validation of the three historical
segments; finish server/memory gates and LocalForge wiring to the implemented exact
counter; improve batched GPU/dense path; independent reference
checks and bounded storage/kernel/preload qualification. Stop real-model scaling on
measured infeasibility and record blocker. The feature is NOT complete or ready.
