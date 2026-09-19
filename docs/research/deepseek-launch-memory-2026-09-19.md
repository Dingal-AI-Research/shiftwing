# DeepSeek reviewer launch failures under LocalForge: host-memory floor and swap race

Date: 2026-09-19. Scope: engine start-up only; no arithmetic, prefill or
decode path changed.

## Symptom

LocalForge's review gate (`auditAgentModel = deepseek-flash`) launches
`c/shiftwing serve … --context 100352 --experimental-deepseek --ram-gb 8
--cuda-expert-gb 2` after unloading the Qwen `llama-server`. On 2026-09-18
11:45 the engine exited during the handshake:

    deepseek_v4: minimum native memory budget cannot fit (host_available=22464114688 device_available=15738077184)
    RuntimeError: invalid engine handshake: b''

and LocalForge restored the orchestrator. `.localforge/logs/inference-colib.log`
and `inference-llama-server.log` give the sequence: `llama-server` "cleaning
up before exit" at 11:45:09.7, the DeepSeek engine started at 11:45:10.9 and
sampled `MemAvailable` at 11:45:12.5 while the previous process's memory was
still being returned (driver teardown, pinned buffers, page cache), so it saw
20.9 GiB. The two successful launches on 2026-09-17 had 27.9 and 29.3 GB
available and only fit by shrinking the expert cache to 5.5–6.8 GB.

## Cause

Two things compound:

1. **The floor was 26 GB on a 29.5 GB VM.** At the review context the plan
   needs dense arena 8.85 GB + attention state 1.40 GB + whole-prompt prefill
   banks 9.81 GB + staging/request/headroom 2.1 GB + a minimum expert cache
   of 3.7 GB. Anything else resident, or a single low sample, and it cannot
   start; when it does start it steals from the expert cache, which is
   decode throughput.
2. **The plan is evaluated once, ~1 s after another model is told to exit.**
   Memory comes back over several seconds.

## Change

1. **Release the FP8 dense pages after the device upload**
   (`dsv4_dense_arena_release_fp8`, `c/deepseek_v4_dense.h`). Every FP8
   consumer (attention `wq_a/wq_b/wkv/wo_a/wo_b`, indexer `wq_b`, shared
   experts) dispatches to the device once CUDA is enabled and uses the host
   address only to locate the device copy (`dsv4_dense_cuda_pointer`). The
   whole pages inside each FP8 record are `madvise(MADV_DONTNEED)`ed and
   `mprotect(PROT_NONE)`ed, so the memory is returned to the OS and any host
   path that wrongly touches FP8 bytes faults instead of computing on zeros.
   Edge pages shared with neighbouring records stay. Access is restored
   before the arena is freed. `DSV4_DENSE_RELEASE=0` disables it. Measured
   on the real model: 5.45 GiB released, engine RSS 3.1 GiB after load
   (was ~9), peak RSS over a request 12.7 GB (was 16–23).
2. **Plan accounting** (`dsv4_plan_memory`, new `host_dense_released`
   argument, `dense_released` on the `DSV4_MEMORY` line): the load-time peak
   (whole arena resident) is checked separately, and the steady-state fixed
   footprint uses `dense_bytes - dense_released`. At the review context the
   floor drops from ~26 GB to ~20.7 GB, and 24.4 GB grants the full 8 GiB
   expert budget. `test_deepseek_v4_memory.c` reproduces the 22,464,114,688
   byte case: rejected without the release, accepted with it.
3. **Wait for memory to settle** (`dsv4_execution_init`,
   `c/deepseek_v4_execute.h`): the plan is re-evaluated every second for up
   to `DSV4_MEMORY_WAIT_S` (default 90; LocalForge's transition timeout is
   12 minutes). The first fitting sample is *not* accepted: the engine
   continues until the plan grants the full host and device cache budgets or
   availability has stopped rising for three samples, so a launch during a
   swap does not pin a small expert cache for the life of the process.
   Logged as `DSV4_MEMORY_WAIT … fits=… limit_s=…` and
   `DSV4_MEMORY_WAIT done waited_s=…`. Invalid configuration still fails at
   once.

## Verification

- Exact LocalForge launch (same launcher, arguments and environment as
  `buildColibLaunchSpec` + `resolveAuditLaneSettings`): `DSV4_MEMORY
  … host_required=24409755412 … dense_released=5854355456 … host_cache=7994867712`,
  `DSV4_LOADED seconds=19.2`, `/v1/models` lists `deepseek-flash`.
- Chat completion through the HTTP API with `reasoning_effort: low`,
  temperature 0, on a planted `a + b → a - b` diff: reasoning stream, then
  `{"verdict": "REJECT", "issues": ["The function 'add' now returns the
  difference …"]}`; decode 1.35 s/step.
- Streaming request carrying `expected_prompt_sha256` from the CPU counter
  (`python3 -m tools.deepseek_v4_count`, run as LocalForge runs it): 351 SSE
  events, progress and metrics frames, `[DONE]`.
- Swap race reproduced with a 12 GiB hog released after 30 s and the engine
  launched 10 s in: `DSV4_MEMORY_WAIT host_available=16114520064 … fits=0`,
  `DSV4_MEMORY_WAIT done waited_s=31`, then a plan at 27.8 GB available with
  the full 8 GiB cache and a successful load.
- `make -C c CUDA=1 test-deepseek` (13 suites), the CPU-only compile of
  `deepseek_v4.c`, 85 DeepSeek Python tests and `test_openai_server_deepseek`
  pass; `git diff --check` clean.

## Not changed

The whole-prompt prefill banks (9.8 GB at 92,160 input tokens) remain the
largest fixed host cost; storing the HC bank as BF16 (its values are already
BF16-rounded) would halve it and is the next lever if the floor needs to
drop further. LocalForge's own swap sequencing was not modified.

## Follow-up the same day: reviews still dying before the reviewer ran

With the engine starting reliably, the scaffold audit in LocalForge session 8
(`shop-diary-apps`, defect E7) still failed three times with

    Invalid source range returned by review preparation. No review was submitted.

which LocalForge classifies as an infrastructure failure. The DeepSeek engine
was never reached: the error comes from LocalForge's own orchestrator
preparation pass (`src/mediated-agent-harness/review-preparation.ts`), where
Qwen3.8-27B selects source excerpts as `block_id` + original line numbers
and the host rejects any selection it cannot ground.

Replaying the exact preparation prompt against the live orchestrator
reproduced it deterministically (temperature 0): `inventory.service.ts` is
460 lines, so the host had split it into `source-2` (lines 1–377) and
`source-3` (378–460); the model returned the right file and the right line
numbers (338–392 and 412–422) under `source-2`, the file's first block id,
and the validator treated a range past the block's end as fabrication.

Fix in LocalForge (uncommitted WIP tree, not in this repository): a
selection is resolved by the named block's file and validated against every
block of that file in the same batch, lines are copied from whichever block
holds them, and the error now names the offending selection and the shown
range. Unknown ids, diff blocks, inverted ranges, lines outside the shown
evidence and empty reasons are still rejected. The prompt also tells the
preparer that long files arrive as several consecutive blocks. Two
regression tests were added (`review-preparation.test.ts`); the three tests
that were already failing in that file (batch-size budgets) fail identically
without the change. The real scaffold then prepared 15 excerpts in 43.5 s
with the two cross-block selections logged as resolved.

The running LocalForge dev server is a plain `node` process and must be
restarted to pick the change up.
