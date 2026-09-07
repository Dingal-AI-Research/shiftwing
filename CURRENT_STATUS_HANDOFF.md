# Shiftwing current-status handoff

Snapshot: `2026-08-21T09:40:00+01:00` (Europe/London). Verify live state
before acting. This is the authoritative continuation summary.

## Decision

- DeepSeek-V4-Flash-0731 is rejected on the RTX 5070 Ti. Do not optimize,
  promote, or make it the default.
- Reconstruct pinned Ornith397 base and q3 byte-for-byte, then improve tok/s
  and TTFT through exact-output paired A/B tests.
- LocalForge is no longer documentation-only. On 2026-08-23 the owner directed
  changes to `/home/dinga/Projects/localforge`: the code-review gate now runs on
  the colib 397B lane for tasks the orchestrator classifies as complex at
  triage, simple tasks are not reviewed at all, and the `explore` SWE phase was
  removed in favour of `localize` (its `explore_agent` and `swarm` delegation
  tools remain, and the 9B lane is made resident on demand). Earlier statements
  in this file and in the Phase-12 papers describing LocalForge as unmodified
  are contemporaneous and remain as written.
- The two DeepSeek weight directories were deleted on `2026-08-20` after
  explicit owner confirmation. `c/deepseek-v4-flash-0731` (98 files,
  `166,973,631,781` bytes) and `c/.deepseek-v4-flash-0731.source` (69 files,
  `166,898,734,250` bytes) are gone, freeing about `310.6 GiB`; free space rose
  to `683,464,658,944` bytes. All DeepSeek evidence survives in tracked files:
  the six `docs/research/artifacts/deepseek_v4_*.json` artifacts,
  `c/deepseek_v4_preflight.json`, and the tooling. Revisiting DeepSeek now
  requires a full re-download.

## Performance ground truth

Fresh five-trial Ornith397 control: median sustained decode
`0.824033698 tok/s`, token-weighted decode `0.818532306 tok/s`, and median
trial TTFT `43.508444 s`.

Those five control trials also rise from `0.773577249` to `0.838411793 tok/s`
as the session warms. Any sequential control-then-candidate comparison would
report that 8.4% warm-up as candidate speedup, which is why the paired
controller alternates AB/BA and supports unmeasured priming trials.

DeepSeek never reached qualified paired serving. Its best two-token diagnostic
was `0.011083 tok/s`, with `200.970412 s` initialization and token times
`83.758211 s` and `96.703443 s`. That is about 74x slower in decode, but its
first-token smoke timing is not a qualified chat TTFT. The second token missed
230/258 experts and read 3,074,949,120 bytes; expert I/O, host work, and upload
dominated. DeepSeek used FP4 experts plus FP8 dense weights, never 3-bit.

## Ornith reconstruction checkpoint

Base reconstruction is **complete**. Attempt 3 exited 0 at
`2026-08-19T10:30:09+01:00` with `122/122` committed, `complete: true`, and 64
shards converted in that attempt. Log:
`docs/research/logs/ornith397_base_resume_20260819T005504.log`. The snapshot is
`212,757,753,436` bytes over 122 shards plus `quantization.json` and
`tensor_inventory.json`.

Base reconstruction is **verified byte-exact**. The pinned verification with
`--rehash` reports `passed: true`: 122/122 shards rehashed, size-verified, and
matched record for record, with ledger SHA-256
`cf00084d2e534bed7f1fa8eb5de5fef5fd4dcc857737cfd6a78fa81f397ed67a` and
quantization SHA-256 `b0623a5f83d3890cbb095bbec25f0aeaec62edbc804fae918a2051189d1933a2`
equal to the values recorded at retirement. Artifact:
`c/ornith397_pinned_base_verification.json`.

An int4 smoke run confirmed the reconstruction serves coherent text. It used
zero warmup passes, so its rates are a cold-start floor and are **not**
comparable to any baseline: load `481.9 s`, prompt 1 TTFT `455.5 s` at
`0.072024 tok/s`, prompt 2 TTFT `91.2 s` at `0.150925 tok/s`, `host_moe 0`,
tiers vram `963` / ram `2204`.

**The pinned Ornith397 reconstruction is COMPLETE and verified byte-exact,
base and q3 both.** `c/ornith397_pinned_verification.json` reports
`passed: true` with `rehash: true`:

- base: 122 expected, 122 live, 122 rehashed, 122 size-verified, 122 records
  matched, zero failures;
- expert-q3: 480 expected, 480 live, 480 rehashed, 480 size-verified, 480
  records matched, zero failures;
- live sidecar manifest SHA-256
  `5231fbe79bc9edf27b86222e4df503066a0b8f04f02fc612ab993696096b0180` equals the
  pinned retirement value.

The q3 rebuild reproducing all 480 hashes is a strong determinism result: that
path runs three refinement iterations across 30,720 experts, far more floating
point than the base conversion, and still landed bit-identical to the original.

Timeline of the final run: chain resumed from 241/480 at `23:32`, reached
480/480 near `00:04`, verification passed at `01:22:34`, and the machine was
powered off at `01:23:31` by an owner-requested shutdown watcher.

**Why long jobs kept dying (root cause, 2026-08-20):** the WSL VM restarts.
`uptime -s` moved to `16:17:36`, `20:58:27`, and `21:40:00` while jobs were
running, and each restart killed them. `C:\Users\dinga\.wslconfig` sets
`memory=30GB` on a 32 GB host, so WSL may take nearly all host RAM; streaming
212 GB of shards through the page cache balloons it to the cap, Windows starves
(the owner reported Chrome crashing), and WSL goes down. The owner chose to
leave `.wslconfig` unchanged, so **expect further restarts** during any long
job. This is not agent teardown and `setsid` cannot prevent it.

Both conversions are resumable and hash-validated, so a restart costs only
wall-clock. Use `setsid --fork`, not bare `setsid`: plain `setsid` silently
no-ops when the caller is already a process group leader, which was observed
leaving a job in the agent's own session. Detect a running builder by scanning
`/proc` cmdlines and excluding shells; a bare `pgrep -f` also matches the
agent's own command line and once produced a false duplicate-builder alarm.

Verified atomically before launch: ledger SHA-256, all 58 committed shard
hashes against the preserved manifest (58/58 match, zero mismatch, zero
missing), base directory size, staged shard 59 size, converter/engine/
qualifier hashes, and free space. Every value matched the previous snapshot.

- source: `deepreinforce-ai/Ornith-1.0-397B-FP8`;
- revision: `8b61f97a8512d9d01bff1a9625c9a16730e115bb`;
- source fingerprint:
  `4b1c5ab0824e25b3768e3c6df8392394194fce86c4e9f4f3da24e3134ccf4c94`;
- base: `int4g128` experts, int8 I/O/shared weights;
- ledger: `c/ornith397/.conversion-state.json`;
- committed at launch: `58/122`; next shard: 59;
- ledger SHA-256:
  `b2f4bbdb73dc86096202c1e758e746d9b3685dec411b292cf506d978fb41f5d9`;
- base directory at launch: `101,716,700,399` bytes;
- preserved partial staged shard 59: `2,404,667,544` bytes;
- free filesystem at launch: `511,674,691,584` bytes; conservative final
  projection is about `243,645,474,510` free bytes, above the 100 GiB floor;
- q3 complete: `c/ornith397/expert-q3.json` reports `complete: true`,
  `file_count: 480`. Never pass `--force` (it discards completed chunks); add
  `--adopt-existing` only if the tool reports unledgered sidecar files.

Resume command if the attempt stops again (it revalidates committed outputs
before converting new ones):

```sh
./.venv/bin/python c/tools/convert_qwen.py \
  --repo deepreinforce-ai/Ornith-1.0-397B-FP8 \
  --revision 8b61f97a8512d9d01bff1a9625c9a16730e115bb \
  --outdir c/ornith397 --staging-dir c/.ornith397.source \
  --xbits int4g128 --io-bits 8 --shared-bits 8 \
  --group-size 128 --min-free-gb 100
```

After 122/122 base hashes match the preserved manifest, build q3:

```sh
./.venv/bin/python c/tools/requantize_expert_q2.py \
  --snapshot c/ornith397 --bits 3 --group-size 128 \
  --iterations 3 --experts-per-file 64 --workers 8 --torch-threads 1
```

Match base hashes against
`docs/research/artifacts/ornith397_retirement/base-shards.json` and all 480
q3 hashes against
`docs/research/artifacts/ornith397_retirement/expert-q3.json` before serving
or benchmarking. Runtime target: q3 on NVMe/RAM, expanded q4 in CUDA.
`Q3_NATIVE=1` is diagnostic.

## Implemented but unqualified

1. Expanded-q4 adaptive route atlas; cached tiny seam max difference
   `1.1641532e-09` versus CPU q3.
2. Ordered grouped-prefill I/O; batch 4 kept exact tiny output, 44 hits,
   30 misses, and 300 reads while ring submissions fell 50 to 41. Default is 1.

Extended request-scoped ring telemetry is implemented. Last post-telemetry
validation: Python compile, SM120 build, 33 relevant tests, and
`git diff --check` passed. Earlier candidate gates included 43 focused tests,
CUDA parity, and the cached seam. The final broad suite remains open.

Two notes from reading the candidate diff before the A/B, so the results are
not misread:

- Candidate 1 is not guaranteed token-exact. Its atlas prefill executes the
  expanded-q4 device copy, which matched the CPU q3 reference at
  `1.1641532e-09`, not bit-exactly. Over 60 layers and a 64-token turn that
  can in principle flip a token. If the paired controller reports
  `outputs_identical` false for candidate 1, treat that as the preregistered
  exactness gate doing its job, not as a controller defect, and do not promote
  it.
- Candidate 2 widens no concurrency hazard. `moe_prefill_cpu_expert_batch`
  holds pointers to up to four cache-resident experts across the lock
  boundary, which would race with a concurrent evictor, but decode and prefill
  run in one slot-major forward on a single scheduler thread, so no second
  thread loads experts for the same layer. `expert_victim_slot` additionally
  protects every eid in the active batch.

## Pinned-manifest verifier (new, 2026-08-19)

`c/tools/verify_pinned_ornith397.py` and
`c/tests/test_verify_pinned_ornith397.py` close a real evidence gap.
`shiftwing doctor --verify-hashes` proves the shards match the *live* ledger and
`audit_expert_sidecar.py --verify-hashes` does the same for the sidecar, but
neither reads the retirement evidence, so neither can prove the
reconstruction reproduced the *pinned* model rather than a merely
self-consistent one.

The verifier compares the live ledger, quantization manifest, and sidecar
manifest field by field with
`docs/research/artifacts/ornith397_retirement/`, checks the aggregate ledger
and quantization hashes, checks every physical file size, and with `--rehash`
independently recomputes every digest so the result depends on neither audit.
It writes a `passed` artifact and exits nonzero on any mismatch.

```sh
./.venv/bin/python c/tools/verify_pinned_ornith397.py   --snapshot c/ornith397   --base-manifest docs/research/artifacts/ornith397_retirement/base-shards.json   --sidecar-manifest docs/research/artifacts/ornith397_retirement/expert-q3.json   --rehash --workers 4 --output c/ornith397_pinned_verification.json
```

Eight focused tests pass, covering a matching snapshot, a flipped body byte
that only `--rehash` can catch, a truncated shard, ledger drift with intact
files, a missing shard, sidecar count drift, an incomplete quantization
manifest, and the missing-manifest guard.

## Paired controller (new, 2026-08-19)

`c/tools/run_paired_perf_trials.py` and
`c/tests/test_run_paired_perf_trials.py` now exist and close the gap the
previous handoff flagged. The tool alternates AB/BA per pair, supports
unmeasured `--prime-trials`, hash-verifies and resumes every artifact, refuses
resume across binding or pair-count drift, and promotes only when outputs are
identical, no prompt slows, host-MoE fallback is zero, the 95% decode lower
bound exceeds 1.0, and the 95% TTFT upper bound is below 1.0. Student's t is
computed from the regularized incomplete beta; no SciPy is added.

- controller SHA-256:
  `4310ff622b5a9a142e607f3e5d176f2ee0a6defc4ea767ba30f097e2adb2f718`;
- test SHA-256:
  `07ff23cc27bb4f027a46d21d619a9b15264720e8826c45a3c2845cfca64d2087`;
- 20 focused tests pass; `test_run_perf_trials.py` still passes unchanged;
  `check_docs.py` and `git diff --check` pass.
- Validated against the five preserved control artifacts: it reproduces
  median `0.824033698`, minimum `0.773577249`, maximum `0.838411793` tok/s,
  and median trial TTFT `43.508444 s`, and confirms all five outputs hash
  identically.
- Replaying those artifacts through the whole controller reports `hold` on a
  null effect and `promote` on a uniform 1.12 decode / 0.93 TTFT scaling. That
  replay found and fixed a real defect: hardware equality now covers stable
  identity only, because available RAM drifts 2.289-2.312 GiB between real
  trials and would otherwise fail every comparison.
- Report: `docs/research/phase12_preflight_05_paired_ab_controller.md`.

`c/tools/run_perf_trials.py` is unchanged and remains the single-arm
controller for historical artifacts.

## Python gate and the rejected DeepSeek toolchain

A full `unittest discover` over `c/tests` runs 183 tests: **180 pass, 3 error**.
All three are import-time collection errors, not assertion failures, and all
three are DeepSeek-only modules:

- `tests/test_deepseek_v4_tooling.py`;
- `tests/test_openai_server_deepseek.py`;
- `tests/test_validate_deepseek_v4_conversion.py`.

Each imports `deepseek_engine_source_sha256` from `c/tools/runtime_env.py`.
That symbol no longer exists: last session refactored `engine_source_sha256`
into `_source_sha256` plus a wrapper for the Ornith candidates and dropped the
DeepSeek variant. It is in no commit — the entire DeepSeek toolchain, including
these three test modules, is untracked working-tree work — so git cannot
restore the original definition.

The function defines a *fingerprint*. Any reconstructed file list that is not
exactly the original produces digests that silently disagree with the recorded
DeepSeek evidence, which is worse than a visible import error. It was therefore
deliberately left unrepaired rather than guessed at.

Recommended disposition, pending owner confirmation: judge the Ornith gate on
the 180 non-DeepSeek tests, record the three exclusions and this reason, and
leave the rejected toolchain broken. Repairing it would need the original
pinned DeepSeek source list, which only the owner can supply. Nothing in the
Ornith path depends on it.

## Expert-map persistence (new, 2026-08-20)

The engine can seed its expert cache from a persisted heat map (`AUTOPIN=1`,
`EMAP_PATH`, defaulting to `{snapshot}/expert_map.bin`), which is what lets a
fresh process start on the experts traffic actually routes to instead of
experts `0..cap-1`. It was not working: the map was saved only from `atexit`,
and a graceful shutdown that overruns `openai_server.py`'s 10 s wait escalates
to `SIGTERM`, which runs no `atexit` handler. No `expert_map.bin` existed after
a normal run.

`c/qwen.c` now checkpoints the map on turn boundaries via
`expert_map_checkpoint`, so persistence no longer depends on a clean exit. The
write is ~123 KiB and was already atomic. `EMAP_SAVE_EVERY=0` restores
exit-only behavior. `EMAP_FREEZE=1` makes the map a read-only input so a
benchmark cannot mutate its own seed map, which a hash-bound A/B requires. The
`[EMAP]` line now reports `frozen` and `save-every`.

`c/tests/test_expert_map_persistence.py` has four passing tests: the map
survives `kill -9`, the next process reports `loaded=1`, freeze leaves bytes
identical and reports `saved=0`, and `EMAP_SAVE_EVERY=0` writes nothing on turn
boundaries. Validated against the tiny fixture with a scratch CPU build; note
the tiny tokenizer holds 512 entries, so prompts must be in-vocab (`"!"`), and
`"hello"` fails with `prefill token outside vocab`. That fixture limit, not an
engine fault, is why an earlier tiny-model qualifier attempt returned
`INTERNAL`.

**`c/qwen` has deliberately not been rebuilt**, so the pinned binary still
hashes to `3bb2e6d1...5d663d751` and the q3 smoke run stays comparable to the
int4 one. The engine source now differs from that binary: rebuild and rebind
every hash before any A/B.

Measured ceiling for this work, from the five preserved control trials: once
warm, cache contents are already within ~1.5 percentage points of a perfect
same-size selection (trial 1 is 5.22 pp off, trials 3-5 about 1.5 pp). So
seeding cannot raise steady-state throughput; it can only reach that state
immediately instead of after a warmup, which is worth roughly the trial-1 to
trial-3 gap (0.774 to 0.838 tok/s) plus a large early-TTFT improvement.
Covering 80% of routing mass would need about 30% of experts, roughly twice the
current cache, so cache capacity - not contents - is the real throughput lever.

## Source/worktree boundary

- branch `prefill-throughput-and-serve-fixes`, HEAD
  `35d9ab86fa2b8f8acbf56c8438413dfb08f2a2a7`;
- dirty tree: do not reset, clean, or broadly stage;
- engine fingerprint
  `8d8992c20907c7b376751d61119289d4d05127fd318e2bcf974ad412136da6fa`;
- `c/qwen` SHA-256
  `3bb2e6d10e24e3b40ad72c1f676a4c743f054fe865f3ea55c5be6475d663d751`;
- qualifier SHA-256
  `92de98e7a85ae583be8154674a47442e30f682c90d16c2ec866c7f03efcc2b1c`;
- converter hashes: base
  `034d8103459cc6ab13189a04f53f6f7245112fb221ce75807f4ab9072dbf6fc3`,
  q3 `960a4c00703645809f8d26babc415318029f734a2f561c0badbfbe3eb1f3e950`.

The engine has deliberately not been rebuilt during reconstruction, so the
pinned `c/qwen` hash still holds. The full gate suite rebuilds it; run that
once after reconstruction, then rebind every hash before the first pair.

Candidate-report hashes predate telemetry edits; rebind everything before A/B.
Preserve unrelated changes and the old root untracked backslash-named directory.
DeepSeek targets still present:

- `c/deepseek-v4-flash-0731`: `166,973,631,781` bytes;
- `c/.deepseek-v4-flash-0731.source`: `166,898,734,250` bytes.

## Next actions

1. Let attempt 3 and the armed chain finish; if either stops, record the
   attempt and resume by hand.
2. Verify 122 base hashes with `--rehash`, then verify the 480 q3 files the
   same way plus `audit_expert_sidecar.py --verify-hashes`.
3. Run full CPU/CUDA/Python/server/web/documentation gates, then rebind the
   rebuilt engine and qualifier hashes.
4. Re-baseline the control, then run five pairs per candidate independently
   with `run_paired_perf_trials.py`, one prime trial, and every binding. Both
   arms share this preregistered qualifier template; only the differing flag
   changes, and `c/qwen` must be rebuilt and rebound first:

   ```sh
   Q="./.venv/bin/python c/tools/qualify_tiered_model.py \
     --model c/ornith397 --engine c/qwen \
     --prompt-file c/fixtures/ornith397_legacy_prompts.json \
     --warmup-passes 2 --measured-passes 1 \
     --max-tokens 64 --context 4096 --threads 8 \
     --expert-ram-gb 18 --cuda-expert-gb 6 \
     --minimum-tps 0.70 --uring-persist 1 --pinned-upload 1 \
     --decode-protect 1 --decode-protect-prewarm 1 \
     --expert-q3 1 --q3-native 0 --cuda-events"
   ```

   Candidate 1, expanded-q4 route atlas, adds `--q3-route-atlas 0` to the
   control arm and `--q3-route-atlas 1` to the candidate arm, both with
   `--prefill-expert-batch 1`. Candidate 2, ordered grouped-prefill I/O, adds
   `--prefill-expert-batch 1` versus `4`, both with `--q3-route-atlas 0`.
   Each arm ends with `--output {output}`, the two templates are separated by
   `:::`, and the run binds `c/qwen`,
   `c/fixtures/ornith397_legacy_prompts.json`,
   `c/tools/qualify_tiered_model.py`, `c/ornith397/expert-q3.json`,
   `c/ornith397/.conversion-state.json`, and `--bind-engine-source`. Run the
   two candidates as separate states so neither can inherit the other's
   result.
5. Promote only after confidence, exact-output, quality, CUDA, and all serving
   regression gates pass. Keep LocalForge unchanged.

Next-model prompt: "Read `CURRENT_STATUS_HANDOFF.md` and Phase 12 of
`PLAN.md`; verify the atomic state, then continue from the first incomplete
action without deleting or resetting any artifact."
