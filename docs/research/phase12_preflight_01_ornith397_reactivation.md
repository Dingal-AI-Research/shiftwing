# Phase 12 preflight 01: Ornith397 reactivation

Date: 2026-08-18

## Decision

The owner rejected DeepSeek after its real native path measured roughly
0.011 tok/s and redirected work to Ornith397 on the RTX 5070 Ti. DeepSeek
promotion is frozen. LocalForge is out of implementation scope; only a later
carry-forward note is authorized.

## Revalidated reconstruction boundary

The compact retirement bundle revalidates the pinned
`deepreinforce-ai/Ornith-1.0-397B-FP8@8b61f97a8512d9d01bff1a9625c9a16730e115bb`
source, 122 base output hashes, 480 q3 hashes, tokenizer/template, and five
fresh control trials. The base and q3 converter hashes exactly match the
retirement record.

The metadata-only dry-run resolves 122 shards and 405,108,696,032 source
bytes, validates 92,400 compressed-FP8 pairs, keeps 93,078 tensors, and
projects 212,634,789,241 base output bytes. Source shards are downloaded,
converted, committed, and removed incrementally. Completed base plus the
157,073,113,440-byte q3 layout projects about 246 GB free, above the required
100 GiB floor even while rejected DeepSeek weights await deletion
confirmation.

## Active exact command

```sh
./.venv/bin/python c/tools/convert_qwen.py \
  --repo deepreinforce-ai/Ornith-1.0-397B-FP8 \
  --revision 8b61f97a8512d9d01bff1a9625c9a16730e115bb \
  --outdir c/ornith397 \
  --staging-dir c/.ornith397.source \
  --xbits int4g128 --io-bits 8 --shared-bits 8 \
  --group-size 128 --min-free-gb 100
```

The operation is resumable and records atomic state after each output shard.
Completion is accepted only when every base and q3 file hash reproduces the
compact retirement manifests. Structured evidence is in
`docs/research/artifacts/ornith397_reactivation_preflight.json`.

## Resume record

Attempt 1 ended with exit code 1 while downloading source shard 16. It had
already atomically committed 15/122 output shards totaling 25,708,187,394
bytes. The retained terminal output contains no converter traceback, so no
more specific cause is asserted. The checkpoint SHA-256 is
`128d9049ea12af06ec22e3a5eff9eb3097a8d853b1df70d8dfd13099e4de5574`.
Attempt 2 started at 2026-08-18T17:13:15+01:00 with the identical command and
rehashed every committed shard before resuming shard 16. At the read-only
checkpoint taken at 2026-08-18T17:47:56+01:00 it had committed 21/122 outputs,
occupied 36,235,656,580 directory bytes, and was acquiring source shard 22.
The live ledger hashes to
`60f0eb9026a4695e5ff376e308aaf74e76500d232b5a7aa75867d78a11f84a2e`;
its signature binds the pinned source fingerprint
`4b1c5ab0824e25b3768e3c6df8392394194fce86c4e9f4f3da24e3134ccf4c94`.
No completed output was deleted or rewritten, and 579,514,429,440 bytes
remained free.

## Finalized continuation checkpoint

At the read-only handoff checkpoint `2026-08-19T00:38:31+01:00`, no converter
or quantizer process is running. Attempt 2 committed 58/122 base outputs; its
atomic ledger hashes to
`b2f4bbdb73dc86096202c1e758e746d9b3685dec411b292cf506d978fb41f5d9`.
The next source is shard 59. Its 2,404,667,544-byte staged file is retained and
must not be deleted before resume. The completed model directory occupies
101,716,700,399 bytes, 511,674,691,584 filesystem bytes remain available, and
q3 reconstruction has not started. The prior terminal session is unavailable,
so the reason the process stopped is intentionally recorded as unknown.

The exact resume command is unchanged. It must rehash all 58 committed
outputs, finish 122/122, and independently reproduce the preserved base
manifest before the q3 command begins. The root-level
`CURRENT_STATUS_HANDOFF.md` binds the continuation order, current source/code
hashes, dirty-worktree boundary, candidates, and remaining gates.

## LocalForge boundary

No LocalForge repository, configuration, deployment, or runtime is modified.
After device-specific A/B results exist, a carry-forward note will capture
the model fingerprint, required artifacts, runtime flags, metrics, and proven
optimization settings for later LocalForge work.
