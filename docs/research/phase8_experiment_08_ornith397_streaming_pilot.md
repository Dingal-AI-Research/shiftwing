# Phase 8 Experiment 8: Ornith-397B Streaming Conversion Pilot

**Model:** `deepreinforce-ai/Ornith-1.0-397B-FP8`
**Immutable revision:** `8b61f97a8512d9d01bff1a9625c9a16730e115bb`
**Date:** 2026-07-28
**Status:** pilot accepted; 10-, 20-, and 30-shard projection checks passed;
full resumable conversion proceeds

## Abstract

Before transferring the complete 405 GB Ornith-397B source, this experiment
ran the production converter on one real source shard. The purpose was to
test the checkpoint's actual FP8 tensors, quantify peak memory, verify atomic
output and source cleanup, and prove that the job can resume from a durable
ledger.

The pilot completed in 294.25 seconds. It produced a 1,103,829,815-byte
output containing 30 physical tensors and 1,103,825,927 tensor-data bytes.
Peak resident memory was 14,758,896 KiB. The output was hashed and recorded
before its downloaded source shard was removed. Only 45 MB of tokenizer,
configuration, index, and transfer metadata remained in staging. The result
supports resuming the remaining 121 shards without revisiting the retired
Qwen397 3-bit conversion.

## 1. Research question

Can the direct Ornith-397B conversion begin safely on the reference machine,
using bounded staging and a restartable commit record, before the project
spends the full network and storage cost?

## 2. Method

The exact command used the same precision map accepted for Ornith-35B:

```text
source: deepreinforce-ai/Ornith-1.0-397B-FP8
revision: 8b61f97a8512d9d01bff1a9625c9a16730e115bb
routed experts: grouped int4, group size 128
dense input/output paths: int8
shared experts: int8
MTP: excluded
minimum free disk: 100 GiB
maximum new shards: 1
```

The `--max-shards 1` control stopped the converter after its first successful
atomic commit. `/usr/bin/time -v` measured elapsed time and peak resident
memory. The output directory and conversion ledger were then inspected
independently.

## 3. Terms

**Shard** means one file-sized partition of a large checkpoint. It is a
transport and storage unit, not a separate model.

**Atomic commit** means writing a temporary file completely and renaming it
to its final name only after it is durable. A crash therefore leaves either
the previous valid state or a complete new shard, not a trusted half-file.

**Resume ledger** means the small JSON record that binds each completed
source shard to its output filename, byte counts, tensor count, and SHA-256
hash. On restart, the converter validates this record instead of assuming
that a file is correct because its name exists.

## 4. Results

| Measurement | Result |
|---|---:|
| source inventory | 122 shards |
| shards committed in pilot | 1 |
| elapsed wall time | 294.25 s |
| user CPU time | 28.73 s |
| system time | 188.52 s |
| peak resident memory | 14,758,896 KiB |
| output file bytes | 1,103,829,815 |
| tensor-data bytes | 1,103,825,927 |
| physical tensors | 30 |
| remaining staging footprint | 45 MB |
| free disk after pilot | 607 GiB |

The committed file is:

```text
model-00001-of-00122.safetensors
SHA-256:
beb6a1f75e3fd02cbeccac0490f739f2b13f60edd49c0c13a14570c1ff85707c
```

The resume signature fixes the source index fingerprint:

```text
4b1c5ab0824e25b3768e3c6df8392394194fce86c4e9f4f3da24e3134ccf4c94
```

as well as the immutable repository revision and every quantization option.
The first shard includes the 248,320 by 4,096 token embedding and the
beginning of layer 0, so it is not necessarily representative of later
shard output sizes.

## 5. Interpretation

The pilot exercised the real source representation rather than only
metadata. Its successful commit confirms that the converter can reconstruct
Ornith's per-channel FP8 values and map the 397B tensor shapes into the same
container consumed by the C runtime.

The high system time reflects network/file transfer, memory mapping, durable
writeback, and source cleanup. It does not describe inference speed. The
14.1 GiB peak is below the machine's 29.4 GiB physical RAM, leaving useful
headroom during this isolated conversion.

### 5.1 Ten-shard projection check

The full run was resumed from the pilot ledger. After 10 of 122 shards, the
converter had committed 17,425,431,584 tensor-data bytes and
17,428,372,160 total file bytes. This is 8.1950% of the predicted final
tensor payload; 10/122 is 8.1967% of the shard inventory.

Multiplying the measured mean by 122 gives 212,590,265,325 bytes, compared
with the metadata-only prediction of 212,634,789,241 bytes. The early linear
projection is 0.0209% lower. Shards are not guaranteed to have equal content,
so this is not a replacement for the final exact-byte gate. It is an
independent early indication that no systematic precision expansion or
missing large tensor class has appeared.

At 20 committed shards, measured tensor data was 34,973,618,751 bytes. A
second linear projection gives 213,339,074,381 bytes, 0.3312% above the same
metadata prediction. The direction change from -0.0209% at 10 shards to
+0.3312% at 20 demonstrates the expected content imbalance between transport
shards. Both measurements remain close enough to reject a large systematic
format error, while only the final exact-byte comparison can close
completeness.

At 30 committed shards, the durable ledger contained 52,521,805,918
tensor-data bytes and 52,531,065,502 total file bytes. This is 24.7005% of
the predicted payload after 24.5902% of the shard inventory. The linear
projection is 213,588,677,400 bytes, 0.4486% above the metadata prediction.
This remains consistent with uneven shard contents rather than systematic
precision inflation: the projection has moved by less than half a percent
while the observed fraction has tripled. Shard 30 itself contains 3,072
physical tensors, 2,281,702,400 tensor-data bytes, and has SHA-256
`b386f60d6af2fccc228cf16bc217bf38ca0a07275e35031c889ef3ae9912a875`.

Commit intervals alternate with shard contents. Across commits 2 through 6,
the observed intervals were 500.8, 186.5, 382.2, 174.7, and 379.6 seconds.
The median was 379.6 seconds and the mean 324.7 seconds. Network transfer
remains the dominant and variable cost.

The active downloader includes `hf_xet` and leaves
`HF_XET_HIGH_PERFORMANCE` unset. This was not changed mid-run. Hugging Face
documents adaptive concurrency as the default and recommends the
high-performance override primarily for machines with at least 64 GB RAM
because it raises concurrency and memory-buffer bounds
([Hugging Face, 2026](https://huggingface.co/docs/hub/models-downloading#faster-downloads)).
The present host exposes about 29.38 GiB of available RAM and the converter
has already demonstrated safe resumability, so restarting into a
higher-memory transfer mode would add rehash cost and resource risk without
evidence that the default has failed to adapt.

### 5.2 Interruption and supervision recovery

Two foreground status turns were interrupted during the long conversion.
The first interruption occurred while shard 17 was in flight and left the
ledger at 16 committed shards. Restarting the identical command re-hashed
those 16 outputs and subsequently committed shards 17 and 18. A second
interruption stopped that foreground process without changing either
committed file.

The command was then moved into a detached WSL session, with its parent
adopted by WSL init, a unique session identifier, no controlling terminal,
and a persistent log. On startup it re-hashed all 18 committed shards before
printing:

```text
[source 19/122] acquiring model-00019-of-00122.safetensors
```

At this checkpoint, committed tensor data was 31,465,334,841 bytes. The last
committed output, shard 18, had SHA-256
`37648bc2f4ed67f29b1889468ceef4fdfd3aa50196b8e12b6103669a5b38496c`.
The abandoned download markers were only small incomplete destination files;
none was indexed or counted as output.

This is a direct recovery test of the ledger rather than an inferred safety
claim. It shows that interruption loses at most the uncommitted source
partition, while committed outputs are verified again before conversion
continues. Detached supervision prevents a later status-turn interruption
from terminating the multi-hour transfer.

### 5.3 Preregistered completion audit

The final conversion will not be accepted merely because the process exits
with status zero. The following checks were fixed at the 30-shard checkpoint:

1. the ledger must contain all 122 pinned source shard names and retain the
   exact repository revision, source-index fingerprint, and quantization
   signature recorded above;
2. every ledger output must exist, and an independent SHA-256 pass must match
   its recorded digest and file size;
3. the converter must atomically publish `model.safetensors.index.json`,
   `quantization.json`, and `tensor_inventory.json`, with
   `complete: true`;
4. every source shard must produce an output shard, fixing the output count
   at 122, and the completed container must contain exactly 278,152 physical
   tensors;
5. the logical inventory must contain exactly 93,078 tensors, and the sum of
   tensor payload ranges must equal exactly 212,634,789,241 bytes, the
   preflight prediction;
6. the index weight map must agree with every safetensors header, with no
   missing, duplicate, extra, overlapping, unsafe, or wrongly owned tensor;
7. the independent container doctor and the one-slot reference-hardware
   resource profile must pass before the C loader or numerical qualifier is
   trusted; and
8. any abandoned download or temporary marker remains outside the index and
   completion manifest.

After these structural checks, Gate 8 still requires a real loader smoke,
finite perplexity/coherent generation, required tier/CUDA telemetry, the
Ornith tool-call round trip, and the bounded sustained-throughput
qualification. Conversion completeness is necessary but does not establish
model quality or speed.

The completion audit is implemented as the opt-in
`colib doctor --verify-hashes` mode. Its focused regression constructs a
valid ledger/manifest pair, verifies the output digest, then corrupts the
output and requires a hard failure. On the live partial container, ordinary
non-hashing doctor mode correctly reports the container as incomplete while
separately passing metadata checks for 31 durable commits; it does not
misrepresent resumability as model completion.

### 5.4 Family-specific qualifier preflight

A pre-run inspection found that the tiered throughput harness still called
the Qwen renderer directly. The engine architecture is shared, but chat
templates are model-level input protocols; using Qwen framing would measure
an input Ornith was not actually released to consume. The qualifier now
dispatches through the snapshot's explicit `colib_model_family` marker and
renders Ornith prompts from that snapshot's pinned `chat_template.jinja`.
The result JSON records the selected family. Six focused qualifier tests
pass, including a synthetic Ornith template that would fail if the Qwen
fallback were silently used.

### 5.5 Cardinality hardening and 39-shard checkpoint

The source-index partition was inspected independently of the conversion
ledger. Applying the text-only selection rule leaves at least one retained
weight in every one of the 122 source shards, so the expected output count is
122 rather than an observed post hoc value. Ornith and Qwen share the same
93,078-name loader inventory and per-name precision map. Exact tensors
contribute one container entry; quantized tensors contribute data, scale, and
qtype entries. The predicted physical count is therefore 278,152.

Both quantities are now executable `doctor` expectations and are also bound
into the final release auditor. A focused negative test changes the expected
output and physical counts while leaving all other fields valid; both
mismatches are rejected. The combined doctor, release-auditor, family prompt,
tier qualifier, batch comparator, cancellation, and web harness suite passes
30 tests.

At the durable 39-shard checkpoint, the ledger contained 67,795,058,302
tensor-data bytes, 67,807,068,470 total output-file bytes, and 88,109 physical
tensors. Shard 39 has SHA-256
`6e26fe3363a50836587baa28ab9a9e883992727c090807444252b915ff312e04`.
This is 31.8833% of the fixed payload after 31.9672% of the source shards.
The simple linear projection is 212,076,849,047 bytes, 0.2624% below the
metadata prediction and still consistent with nonuniform shard contents.
The next commit advanced the ledger to 40 shards, 70,076,760,702 payload
bytes, and 91,181 physical tensors; shard 40 has SHA-256
`aaebe1c2320ff3aef152a4dcfed33300ea02ad6560da9b0c0caf3f4360bba3c6`.

At the durable 46-shard checkpoint, the ledger contained 80,601,612,432
tensor-data bytes, 80,615,946,720 total output-file bytes, and 105,089
physical tensors. This is 37.9061% of the predicted payload after 37.7049%
of the source shards. Its linear projection is 213,769,493,841 bytes,
0.5336% above the metadata prediction; the small drift remains consistent
with the alternating large and small transport-shard contents. Shard 46 has
SHA-256
`8e191acec00e39916c7b904ba9ebeaabce71c00f4b92ca60b33cebe7b98fffd6`.

At the round 50-shard checkpoint, the ledger contained 87,624,947,869
tensor-data bytes, 87,640,550,741 total output-file bytes, and 114,360
physical tensors. This is 41.2091% of the predicted payload after 40.9836%
of the source shards. The linear projection is 213,804,872,800 bytes,
0.5503% above the metadata prediction. Shard 50 has SHA-256
`d775251b729ef1a536e324e2d27080b1201b32491d34e382b6d4941626f5fe3e`.
The projection remains close to the prior checkpoints and shows no
precision-expansion or omitted-weight discontinuity.

At 53/122 durable outputs, the ledger contains 92,359,813,289 tensor-data
bytes, 92,376,263,961 total output-file bytes, and 120,560 physical tensors.
The linear projection is 212,601,834,363 bytes, just 0.0155% below the exact
metadata prediction. Shard 53 has SHA-256
`d1b7964d806a9e7ea559e0f1fb1ea30128f44b86ffea3b43413fdfe4242b1762`.
The converter is acquiring shard 54 with 522 GiB free, so the 100 GiB
fail-safe remains comfortably satisfied.

An operational concurrency check found no safe reason to restart the proven
stream. Although conversion commits one shard at a time, PyTorch is configured
for eight compute threads and 16 inter-op threads; three one-second live
samples of the active process used 150% to 714% CPU while resident memory
remained about 3.7--3.8 GiB. The absence of a shard-worker CLI therefore does
not imply single-core quantization. Starting a second writer would compete
with those kernels and introduce shared-ledger publication races, so the
atomic single-writer design is retained.

Transfer, rather than quantization, later became the observable idle boundary:
while shard 56 was being acquired, its partial download advanced by only about
21 MB during one minute. `c/tools/prefetch_conversion_shards.py` overlaps one
future transfer without adding a second writer. It derives the sorted shard
set from the pinned source index, reads the converter's atomic state, and
selects exactly the second uncommitted shard. The converter therefore owns the
first uncommitted shard while the companion downloads one lookahead file
through Hugging Face's per-file lock.

The companion validates the exact `hf://repo@revision` signature, retains the
100 GiB free-space guard, never writes the ledger or output directory, and
exits when the converter PID no longer matches `convert_qwen.py`. Four
deterministic tests cover unique sorted source shards, the exact one-shard
window, final-shard termination, and source-identity rejection. The live
process started at 55/122 committed shards and selected shard 57 while the
converter acquired shard 56. This records the mechanism before claiming a
speed benefit; effectiveness will be evaluated from later commit intervals.

The first completed overlap behaved correctly. Shard 57 became locally ready
before the converter requested it, was loaded through the normal conversion
path, committed atomically, and was then removed from staging. Its output
timestamp is 187 seconds after shard 56. The preceding three comparable small
output intervals were 322, 477, and 394 seconds. This is evidence that transfer
latency was hidden for one shard, but one observation is insufficient for a
sustained-rate conclusion; later intervals remain the primary evaluation.

The first large-shard observation also remained inside the established range.
Shard 58 committed 602 seconds after shard 57, compared with 565, 573, 639,
and 674 seconds for the preceding four large outputs measured from their
immediately preceding small output. During that interval, shard 59 completed
its transfer and was ready when the converter requested it. Thus the helper
hid useful transfer work without a measured large-shard regression, although
the conversion remains dominated by the large-shard quantization/write path.

The next small output strengthened the transfer-hiding result. Shard 59 was
already local when requested and committed 70 seconds after shard 58, versus
the 322--477-second recent small-shard range before lookahead. At 59/122, the
ledger contains 102,891,432,636 payload bytes and 134,467 physical tensors;
the linear final-size projection is 212,758,555,620 bytes, 0.0582% above the
metadata prediction. Shard 59 hashes to
`32366ac80458e83854c0bd09420d85acdc9142a2598259e119c9eb898c0f885b`.

At the round 60-shard checkpoint, the ledger contains 105,173,135,036
payload bytes, 105,191,909,644 total output-file bytes, and 137,539 physical
tensors. The large-shard linear projection is 213,852,041,240 bytes, 0.5725%
above the metadata prediction and close to the 0.5503% round-50 projection.
Shard 60 contains 2,281,702,400 payload bytes and 3,072 tensors and hashes to
`8a397b6b919dc66418cef802b15e2e2626abdaec0c7639064287e2a4627143ee`.
It committed 664 seconds after shard 59, within the prior 565--674-second
large-shard range. Shard 61 was already local and began loading immediately.

Shard 61 then committed 91 seconds after shard 60, and its staged source was
removed before shard 62 was requested. At this exact 61/122 halfway point, the
ledger contains 106,399,716,546 payload bytes, 50.0387% of the predicted final
payload. Shard 61 hashes to
`f54d4fdb45d3b035aaf547db5bb35eb713e7472976fb0babaea054edc3c8fbfc`.

The next pair supplied the necessary regression check. Shard 62's large arm
took 796 seconds, exceeding the earlier 565--674-second range, while its
prefetched shard 63 then committed in 81 seconds. The total pair took 877
seconds. The three complete pairs immediately before lookahead took 996,
1,042, and 967 seconds; the first four lookahead pairs took 826, 672, 755,
and 877 seconds. Median pair time therefore fell from 996 to 790.5 seconds,
or 20.6%, and the observed ranges do not overlap. The individual large arm
can regress under concurrent reconstruction, but the small-arm transfer
savings are larger. The bounded helper is retained on the controlling
end-to-end pair result rather than an isolated component time.

### 5.6 Unplanned-restart recovery

An unplanned WSL restart occurred after shard 66 had committed and while
shard 67 was loading. The restart removed all three Linux processes but did
not damage the atomic conversion state. At that boundary, the ledger still
contained exactly 66 completed sources, 115,704,754,383 tensor-payload bytes,
115,725,431,959 total output-file bytes, and 151,446 physical tensors. Shard
66 retained SHA-256
`8f0011e56294039cb44b2d58a2162e15349c936323d24ca561ed77ca02c16508`.
No completion manifest existed, so neither the supervisor nor any inference
gate could mistake the partial model for a release artifact.

The converter was relaunched with the identical pinned command and durable
state. Before accepting the resume point, it recomputed and checked the
stored SHA-256 digest of all 66 output shards, covering 115.7 GB of output
payload. Every digest passed. It then selected source shard 67, the first
source absent from the ledger, while the independent bounded helper retained
already staged shards 67 and 68. Thus the recovery reused every committed
output, repeated only the interrupted uncommitted shard, and preserved the
one-writer publication rule. Fresh Gate-8 and prefetch processes were bound
to the replacement converter PID. This is direct system-level evidence for
the planned crash-consistency and resumability properties rather than only a
synthetic unit-test result.

At the subsequent round-70 checkpoint, the recovered ledger contains
122,721,322,203 tensor-payload bytes, 122,743,268,547 total output-file bytes,
and 160,718 physical tensors. This is 57.7146% of the predicted payload after
57.3770% of the source shards. Its linear projection is 213,885,732,982
bytes, 0.5883% above the metadata prediction and consistent with the earlier
round-50 and round-60 projections. Shard 70 hashes to
`fe74abeb2658151e60935fc12a0c46b97d2d965376cde93768e368dd059a06af`.
The converter then selected shard 71 with 493 GiB free, leaving the 100 GiB
guard unchallenged.

Shard 71 committed 182.822 seconds after shard 70. Measured from the preceding
shard-69 commit, the complete shard-70/shard-71 pair took 566.085 seconds:
383.263 seconds for the large output and 182.822 seconds for the small output.
This is the first clean post-recovery pair and is below every 967--1,042
second pre-lookahead baseline pair. The shard-66/shard-67 and
shard-68/shard-69 pairs are excluded from the performance sample because the
restart hash replay and the retained staged downloads changed their timing.
Across the original five valid lookahead pairs and this sixth observation,
the median remains 790.5 seconds and the range expands downward to 566--877
seconds. Shard 71 hashes to
`608cb79c8a41880f212cdf41565b79a514b54d9542eb555ac7284e1a0531c1bf`.

Four additional uncontaminated pairs completed before the round-80
checkpoint. Pairs 72/73, 74/75, 76/77, and 78/79 took 524.743, 547.514,
522.810, and 544.172 seconds respectively. Together with pair 70/71, this
post-recovery series has a 544.172-second median and a 522.810--566.085-second
range. Across all ten valid lookahead pairs, the median is 619.043 seconds
and the range is 522.810--877 seconds. Relative to the 996-second baseline
median, this is a 37.847% improvement. The sustained result strengthens the
decision to retain the one-shard helper; it is no longer dependent on the
first four observations.

At round 80, the ledger contains 140,276,276,987 tensor-payload bytes,
140,301,394,883 total output-file bytes, and 183,896 physical tensors. This
is 65.9705% of the predicted payload after 65.5738% of the source shards. Its
linear projection is 213,921,322,405 bytes, 0.6050% above the exact metadata
prediction and still consistent with rounds 50, 60, and 70. Shard 80 hashes
to `605bc553b540a659ab212269ed3f590a5b11cc61c89efa9682ef377357a2a55e`.
The converter then selected shard 81 with 477 GiB free.

At round 90, the ledger contains 157,824,464,154 tensor-payload bytes,
157,852,753,786 total output-file bytes, and 207,075 physical tensors. This
is 74.2233% of the predicted payload after 73.7705% of the source shards. Its
linear projection is 213,939,829,187 bytes, 0.6137% above the metadata
prediction. Shard 90 hashes to
`de563a6606b2d6d6d17b758aa99a3fcedf9933928af82fff03046b5c763538f7`.

The user then requested a safe pause while source shard 91 was being acquired.
By the time the signals were delivered, shard 91 had completed and committed
atomically. The final durable pause boundary is therefore 91/122, with
159,051,045,664 tensor-payload bytes, 159,079,548,712 total output-file
bytes, and 208,639 physical tensors. Shard 91 hashes to
`0275cfd636d5c25fb99ec2d09bcff34957b47ba2483bc03e39a433c8373f8e03`.
The Gate-8 waiter and prefetcher were stopped before the converter; all three
Linux processes and their Windows wrappers exited. No output temporary file
or completion manifest remains, so resume will begin by verifying the 91
committed hashes and then selecting shard 92. The pause leaves 460 GiB free
and preserves the same 100 GiB guard.

Work resumed on 2026-07-31 using the same pinned revision, conversion
signature, output directory, and staging directory. Before accepting any new
source data, the converter read and independently rehashed all 91 durable
outputs, covering 159,079,548,712 file bytes. Every hash matched the ledger,
so it selected shard 92 without rewriting a committed output. This is a
second real resume exercise, distinct from the earlier unplanned restart at
round 66, and demonstrates the same fail-closed recovery behavior after a
controlled shutdown.

At round 100, the ledger contains 175,372,651,321 tensor-payload bytes,
175,404,112,689 total output-file bytes, and 230,254 physical tensors. This is
82.4760% of the predicted payload after 81.9672% of the source shards. Its
linear projection is 213,954,634,612 bytes, 0.6207% above the exact metadata
prediction and remains consistent with the round-80 and round-90 estimates.
Shard 100 hashes to
`6153d7ab7b6b5875188211c7b6eac7013266107a2fa4b2e0b27768b1410afd4b`.
The converter then selected shard 101 with 444 GiB free, leaving the 100 GiB
guard unchallenged.

### 5.7 Host-capacity incident and recovery

The converter reached 109/122 durable outputs before a storage-accounting
failure exposed a boundary not represented by the in-guest disk guard. The
ledger at that point contains 190,639,136,088 tensor-payload bytes,
190,673,348,224 output-file bytes, and 250,361 physical tensors. Shard 109
hashes to
`1135ccdd0d2946c4d1a4ff18b650d18ed8b49dd55c9ea517b0ac4b4927bc029a`.

WSL reported hundreds of GiB available in the ext4 filesystem, but its
dynamically expanding `ext4.vhdx` exhausted the underlying Windows C:
volume. The host reached approximately 30 MB free and then zero usable
capacity while shard 110 was being written. Linux filesystem reads began
returning I/O errors, new WSL processes could not read `/etc/passwd`, and the
distribution could not restart. This is a host-capacity failure rather than
an Ornith tensor, quantizer, or GPU failure. It also demonstrates that an
in-guest `statvfs` guard alone cannot detect exhaustion of the physical
volume backing a dynamic VHD.

The failure occurred before shard 110's final rename and ledger update.
After a controlled WSL shutdown, the state remained at 109/122, the
completion manifest was absent, and the only output residue was one
2,282,123,368-byte unindexed temporary file. That orphan was removed; no
committed output was changed. The interrupted source fragments were retained
for resumable acquisition.

Recovery first removed 8.94 GB of regenerable Windows npm download cache,
restoring 7.08 GiB of host capacity and allowing WSL to start. The already
closed Phase-4 Qwen numerical comparison had retained
`Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf`, a 22,285,080,384-byte downloaded reference
whose source revision, byte count, and digest are recorded in
`docs/phase4_handoff.md`. Repository search confirmed that no active gate,
test, or release audit consumes the file, so it was removed. This does not
immediately shrink the non-sparse VHD; instead, it gives ext4 22.3 GB of
already host-allocated blocks to reuse for the remaining Ornith outputs.
After also removing the failed atomic temporary, ext4 reported 447 GiB free
while the host retained 7.15 GiB.

Conversion resumed with a third complete hash replay, this time over all 109
durable outputs. A new prefetch worker staged shard 111 while replay ran, and
a new Gate-8 supervisor remained bound to the replacement converter PID.
Host and guest free space are monitored separately for the remaining writes.
Future long conversions should add an explicit backing-volume guard when
running under WSL rather than treating guest free space as sufficient.

All 109 replayed SHA-256 values matched. Shard 110 was then regenerated from
the retained source and committed atomically. Conversion continued through
the final source without another capacity event; Windows host free space
remained above 7.1 GiB while ext4 reused blocks from the retired Qwen
reference.

The final manifest was published only after all 122 outputs committed and
loader inventory validation completed. It reports exactly 122 source shards,
122 output shards, 93,078 logical tensors, 278,152 physical tensors, and
212,634,789,241 tensor-payload bytes, identical to the metadata-derived
preregistration. Total output-file bytes are 212,672,804,033. The final
source shard contains four physical tensors and commits as a
1,018,128,753-byte output with SHA-256
`40691b0de74e22332afbaecc42268fb836fee3d3997c71c6b5f031d09d79c87b`.
The Windows host retained 7.236 GiB when the manifest became visible.

### 5.8 Manifest-bound post-conversion supervisor

The conversion and Gate-8 inference experiments have very different resource
profiles. Running inference while the converter downloads, dequantizes, and
writes shards would confound throughput and cancellation measurements.
Conversely, requiring an interactive handoff after a multi-hour conversion
would leave a completed 212.6 GB artifact unaudited until the next session.

`c/tools/run_ornith397_gate8.py` resolves this boundary without overlapping
the workloads. A detached, low-resource process polls the known converter
PID. It starts qualification only after that process has exited and the
published `quantization.json` exactly matches the pinned source revision,
source-index fingerprint, 122 input and output shards, 93,078 logical
tensors, 278,152 physical tensors, 212,634,789,241 payload bytes, and the
fixed mixed-precision signature.

After conversion, the supervisor first rebuilds `c/qwen` with
`CUDA=1 CUDA_ARCH=native`. This is required because ordinary Python test
targets intentionally build a portable CPU engine and update the build-config
stamp. Without an explicit production rebuild, an otherwise correct long
conversion could fail its tier gate merely because the most recent regression
suite replaced the executable.

The supervisor then executes four preregistered operations in order:

1. a complete independent container/resource audit with every ledger output
   rehashed;
2. a 1,024-token, 512-context finite-perplexity corruption smoke with
   \(PPL < 50\);
3. the four-prompt, two-warm-pass, one-measured-pass tiered qualification
   with the fixed 18 GiB RAM and 6 GiB VRAM expert caches and
   \(\geq 2\) sustained tokens/s;
4. the generated HTTP tool-call/result/final-answer round trip.

The state file records the completed manifest SHA-256, rebuilt CUDA-engine
SHA-256, each exact command line, and each result artifact SHA-256. Resume
skips a step only when all four still agree. A state file from a different
manifest or engine fails closed, and a modified result is regenerated. Six
focused supervisor tests plus one CUDA-build fixture cover exact manifest
validation, missing completion,
absence of Qwen/3-bit work, verified resume, artifact tampering,
cross-manifest rejection, and cross-engine rejection. Together with the
release-auditor and tier-qualifier tests, 20 focused tests pass.

The first complete Python regression after adding the supervisor exposed six
failures in the mux integration test parser. The engine emitted `DPERF`,
`CACHE`, and `DCACHE`, which are documented telemetry frames and already
parsed by the HTTP gateway, but the older test allowlist recognized only
`PERF` and the original advisory frames. Extending the test contract to the
published protocol made all six live mux/session cases pass. After adding the
Gate-9 controller tests, the repeated complete suite passes 98/98 tests. This
was a verification-harness defect;
the wire producer and production consumer already agreed.

Automation deliberately stops after Gate 8. It does not begin Gate 9 because
the warm two-model AB/BA and cancellation experiments require an
uncontended-state review after the 397B coherence and throughput evidence is
known.

### 5.9 Independent validation and first tier result

The post-conversion supervisor completed the two corruption controls before
attempting performance qualification. The independent doctor read and hashed
all 122 output shards and confirmed 278,152 physical tensors containing
212,634,789,241 payload bytes. This cold audit took approximately 48 minutes
because it deliberately reread the complete 212.6 GB container rather than
trusting the converter ledger.

The fixed 1,024-token corpus smoke scored 510 teacher-forced tokens in two
512-token chunks. It returned finite NLL 24.915790427 and perplexity
1.050067545, passing the preregistered upper bound of 50. Its CPU
teacher-forced rate of 0.254 token/s is a corruption-test measurement, not the
production decode target.

The first complete four-prompt CUDA tier qualification remained coherent and
used the resident device graph for all 46,080 measured layer-forwards, with
zero host-MoE fallback. RSS and VRAM were stable at approximately 27.0 GB and
13.2 GB. It nevertheless failed the throughput criterion: the token-weighted
measured rate was 0.527355 token/s, compared with the fixed 2.0 token/s
minimum. The four measured turn rates were 0.618, 0.445, 0.559, and 0.518
token/s.

The detailed counters localize the failure. Across the full warm and measured
run, routed experts caused 1.839 TiB of direct reads and 291,540 misses; the
combined RAM/VRAM hit rate was 60.40%. A representative 64-token measured
decode spent 96.34 seconds reading experts, 26.44 seconds in expert matrix
operations, 0.41 seconds in attention, and 0.37 seconds in the language-model
head. The runtime also constructed 233,024 one-shot `io_uring` instances and
reused none. Therefore this result is classified as an expert-I/O locality
failure, not a converter, numerical, CUDA-residency, or model-coherence
failure.

Before changing numerical precision, a bounded one-prompt pilot was
preregistered with 16 warm and 16 measured tokens. It enables only mechanisms
already implemented in the runtime: persistent `io_uring` and aligned
buffers, reusable pinned upload staging, protection of decode-hot experts
from prompt eviction, and inter-request prewarming of that protected set.
Weights, cache budgets, CUDA kernels, prompts, and sampling remain unchanged.
This pilot asks whether avoidable setup and eviction overhead can move the
runtime into the 2-token/s performance class. If it cannot, the direct result
supplies the capacity evidence required to begin an Ornith-only grouped-3-bit
experiment, starting with the Ornith35 quality control.

The pilot reached 0.867369 token/s. This is a 64.48% improvement over the
0.527355-token/s frozen baseline, but remains 56.63% below the production
criterion. The I/O mechanism behaved as designed: 13,172 expert batches used
only three ring setups and 13,169 reuses, while pinned staging moved
149.364 GiB. Decode protection refreshed twice, prewarmed 1,190 experts
covering 7.408 GiB, and reduced the measured 16-token decode to 2,394 disk
misses. That measured decode spent 11.386 seconds reading 16.003 GB of expert
payloads and 6.876 seconds in expert matrix work, completing in 18.431
seconds. The measured host/device hit fraction was 75.06%.

This result rejects ring construction and pageable upload as the sole cause.
It also exposes avoidable duplication between the two bounded caches:
telemetry classified only about 51 unique RAM-or-VRAM experts per layer even
though the nominal capacities sum to approximately 64. An opt-in
`TIER_EXCLUSIVE` replacement mode was therefore added. When choosing a host
victim, it first selects an expert whose three projections are already
resident in VRAM; decode-protected experts remain protected unless VRAM
already holds the same expert. The accepted tensor files and device LRU are
unchanged. The int4 tiny oracle remains 32/32 teacher-forced and 32/32 greedy,
and all 12 focused qualifier/tiering tests pass.

A second bounded pilot repeats the same prompt, token counts, cache budgets,
and four first-pilot controls with only `TIER_EXCLUSIVE=1` added. Its purpose
is to measure whether recovered unique capacity can close the remaining
hit-rate gap before the numerical precision is changed.

The second pilot rejected the policy. It sustained 0.677430 token/s, 21.90%
below the first pilot. Although it performed 16,830 duplicate evictions, the
final telemetry exposed only 2,196 unique RAM experts plus 963 VRAM experts:
about 52.65 unique entries per layer, versus about 50.63 without the rule.
The measured decode incurred 3,811 misses and read 25.475 GB, compared with
2,394 misses and 16.003 GB in the first pilot. Device eviction removed the
only remaining copy of too many formerly duplicated hot experts, converting
later accesses into disk reads. The experimental code and qualifier option
were removed after the negative result; the 32/32 oracle and test evidence
remain the safety control for the attempted mechanism.

The direct baseline, first locality pilot, and duplication-aware negative now
satisfy the conditional lower-bit protocol: generation is coherent and
numerically finite, throughput remains below 2 token/s, and expert misses/read
volume dominate dense compute and attention. The next experiment is therefore
the owner-approved grouped-3-bit treatment on Ornith35. Ornith397 sidecars
remain prohibited until that smaller quality control completes.

## 6. Limitations

The conversion now proves complete container cardinality, exact
metadata-predicted payload size, and finite teacher-forced inference, in
addition to atomic restartability. The first production tier profile also
proves coherent CUDA-resident generation, but it does not meet the sustained
throughput target. The generated-tool round trip remains intentionally
unexecuted until the tier path passes.

The recovery also demonstrates that the 100 GiB in-guest guard is
insufficient under WSL when a dynamically expanding VHD is backed by a nearly
full Windows volume. This experiment monitored the host manually after
recovery; a future converter revision should make backing-volume telemetry an
explicit platform-aware control.

## 7. Decision

Accept the direct conversion and its independent structural/numerical checks
as complete, and preserve the published manifest unchanged. Keep Gate 8 open
because the first no-MTP production profile reached only 0.527355 token/s.
Complete the bounded non-lossy locality pilot before changing precision. Do
not produce Qwen397 3-bit sidecars. If the locality pilot remains materially
below 2 token/s, begin lower-bit work only as an Ornith35 quality control
followed by an Ornith397 routed-expert sidecar.

Primary artifact:

- `c/ornith397/.conversion-state.json`
