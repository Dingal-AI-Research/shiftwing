# Phase 11 Experiment 5: Fresh Ornith397 control and retirement record

**Model:** `deepreinforce-ai/Ornith-1.0-397B-FP8`
**Immutable revision:** `8b61f97a8512d9d01bff1a9625c9a16730e115bb`
**Date:** 2026-08-16
**Status:** five-trial control accepted; compact retirement evidence accepted;
DeepSeek comparison and promotion remain open

## Result

Five independent process-fresh Ornith397 trials completed on the current
branch between `2026-08-16T16:47:55.053814+00:00` and
`2026-08-16T19:32:08.894201+00:00`. Each trial used the exact four prompts
from the historical accepted artifact, two warmup passes, one measured pass,
64 deterministic tokens per prompt, an 18 GiB RAM expert tier, a 6 GiB VRAM
expert tier, and the expanded-q4 execution of the complete q3 sidecar.

All five automatic gates passed with nonempty output, complete expert
telemetry, active CUDA/VRAM tiers, an active resident graph, and no host-MoE
fallback. The hash-bound controller state is complete and every published
trial was independently rehashed against it.

| Trial | sustained tok/s | median TTFT (s) | Artifact SHA-256 |
|---:|---:|---:|---|
| 1 | 0.773577249 | 45.976887 | `85a3a6ca12760d1d3fbf8c5061b9bb1092a2d31137aea0729dfe5ccb257df31f` |
| 2 | 0.823691479 | 42.989518 | `9f87dbc2c20e4fd67ea4497ff9ed9e13ef4baeca28a3dadff5879839f26244a8` |
| 3 | 0.824033698 | 44.113513 | `5a278be6f043ea9229402b642144af9b2e2ecb74d6e6f148c61fbcbf4c504fa5` |
| 4 | 0.838411793 | 43.508444 | `302af5f62f76502b1baa346cdc776a20e5051e01facb15c5e8f0e6ddfe38d59e` |
| 5 | 0.836485522 | 43.421626 | `ec18e0103fa735df7617927fd567b496afeab1046eef26536ca2490974dd53f2` |

The median trial throughput is **0.824033698 tok/s**, aggregate
token-weighted decode is **0.818532306 tok/s**, and median trial TTFT is
**43.508444 s**. Compared with the historical 0.836866773 tok/s and
81.9725 s control, current median throughput is 0.984665x and median TTFT is
0.530769x. DeepSeek must beat both this fresh control and the historical
legacy-subset thresholds; this Ornith-only run does not pass the paired
DeepSeek promotion gate.

## Reproducibility boundary

The valid controller signature is
`512a4c74f0634b316f73e5d6f4f57b403de3b4c732a99dd9621e10d9765c404c`;
the complete state hashes to
`1efee0a3c8929a61fbd18f79ab2c4a938fe3f6bad081f1eff1f4cc78bbf0156b`.
It binds the exact command, model manifests, legacy fixture, tokenizer and
template, qualification/controller code, native engine sources, and CUDA
binary. The binary SHA-256 is
`89b870b88bb41a45785378e4e0e3ba0fc9ac456b711fe5627380783723938cd7`;
the engine source-contract fingerprint is
`6de45ac315fba7adfbcab41a2e6a1e03cc05d1146b4b09ae523691d4a70bde6b`.
The repository was at
`35d9ab86fa2b8f8acbf56c8438413dfb08f2a2a7` on
`prefill-throughput-and-serve-fixes` with the dirty state explicitly
recorded and all behavior-determining files bound individually.

## Invalid workload attempt

The first launch pointed at the current 20-prompt Qwen fixture instead of the
historical four-prompt subset. It was interrupted during the first warmup,
published no result, and was atomically closed as `interrupted`. Its state
hash is
`7a061ebb4066162dc7b81ef7e68f6d6746b7ef0b2d3fbe2e6e6df3976079d803`.
A dedicated tracked fixture was then proven exactly equal to the four
historical measured prompt strings and used for all valid trials. This invalid
attempt is not included in any statistic.

## Compact retirement evidence

`record_ornith397_retirement.py` rehashes every current performance binding
and trial artifact, proves configuration/model-manifest identity, joins the
base conversion signature to the published quantization manifest, checks all
base totals, and checks the complete q3 file inventory before publishing
atomic compact evidence.

| Artifact | SHA-256 |
|---|---|
| `control.json` | `980eaa16d7cb47768503ab709a9ac4bfb845c784ebddfde67115493ae820757d` |
| `base-shards.json` | `d4fc004ec2f450bb4fbae352af4651d79b56641c24d23bc53d9cad1058aa1e23` |
| `quantization.json` | `b0623a5f83d3890cbb095bbec25f0aeaec62edbc804fae918a2051189d1933a2` |
| `expert-q3.json` | `5231fbe79bc9edf27b86222e4df503066a0b8f04f02fc612ab993696096b0180` |
| `config.json` | `c31964d2d920c10228a40d77afe02b19122b66f7bf3f6eb4b0809ba42527b0d6` |
| `chat_template.jinja` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `tokenizer_config.json` | `792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b` |
| `generation_config.json` | `303aba891d66ab63908a7b3cc9163bcb835fdf8b9f6301c73216f3f1eb3992dd` |

The base-shard record preserves 122 output hashes covering 212,634,789,241
tensor-data bytes, 212,672,804,033 file bytes, 278,152 physical tensors, and
93,078 logical tensors. The q3 manifest preserves 480 file hashes covering
157,073,113,440 bytes and 276,480 tensors. The original 14.6 MB conversion
state hashes to
`cf00084d2e534bed7f1fa8eb5de5fef5fd4dcc857737cfd6a78fa81f397ed67a`;
its nonredundant signature and per-output hashes are retained in the compact
base-shard record.

## Exact reconstruction

~~~sh
./.venv/bin/python c/tools/convert_qwen.py \
  --repo deepreinforce-ai/Ornith-1.0-397B-FP8 \
  --revision 8b61f97a8512d9d01bff1a9625c9a16730e115bb \
  --outdir c/ornith397 \
  --staging-dir c/.ornith397.source \
  --xbits int4g128 --io-bits 8 --shared-bits 8 \
  --group-size 128 --min-free-gb 100
~~~

~~~sh
./.venv/bin/python c/tools/requantize_expert_q2.py \
  --snapshot c/ornith397 --bits 3 --group-size 128 \
  --iterations 3 --experts-per-file 64 \
  --workers 8 --torch-threads 1
~~~

The base converter hashes to
`034d8103459cc6ab13189a04f53f6f7245112fb221ce75807f4ab9072dbf6fc3`;
the q3 converter hashes to
`960a4c00703645809f8d26babc415318029f734a2f561c0badbfbe3eb1f3e950`.
Reconstruction must reproduce the preserved base and q3 file hashes before
the model can serve as a rollback comparator.

## Statistical and storage disposition

These five trials complete the fresh Ornith side of the performance study.
They are a sequential control block, not a completed DeepSeek/Ornith AB/BA
pair, because DeepSeek does not yet have a real converted runnable engine.
Consequently no confidence-bound speedup or promotion is claimed. A true
temporally interleaved comparison would require reconstructing Ornith from
the evidence above after the DeepSeek runtime exists.

The user explicitly authorized exact-target Ornith deletion if required for
space after progress became reproducible. After re-verifying the evidence
hashes and confirming that no model process was running, only the resolved,
non-symlink directory `/home/dinga/Projects/colib/c/ornith397` was
permanently removed. It occupied 369,831,132,527 bytes. Ext4 free space rose
from 412,941,623,296 to 782,772,477,952 bytes immediately after deletion.
The original weights are recoverable only through the exact pinned
reconstruction commands and hashes above.

`fstrim` then reported 538,290,511,872 trimmed bytes. With WSL shut down and
no running distribution, Microsoft `OpenVirtualDisk`/`CompactVirtualDisk`
was applied to the exact ext4 VHDX. Its physical length fell from
618,354,704,384 to 248,025,972,736 bytes, reclaiming 370,328,731,648 bytes.
Windows C: free space rose from 46,854,950,912 to 417,219,334,144 bytes, a
gain of 370,364,383,232 bytes.

After restart, `/dev/sdd` mounted ext4 read/write with discard and reported
782,775,443,456 available bytes. The deleted target remained absent, the
pinned DeepSeek source remained present, and the compact control, base-shard,
and q3 evidence hashes revalidated as
`980eaa16d7cb47768503ab709a9ac4bfb845c784ebddfde67115493ae820757d`,
`d4fc004ec2f450bb4fbae352af4651d79b56641c24d23bc53d9cad1058aa1e23`,
and
`5231fbe79bc9edf27b86222e4df503066a0b8f04f02fc612ab993696096b0180`.
No ext4 or block-I/O error was present in the restart log; systemd-journald
only rotated its two unclean journal files after the deliberate shutdown.
