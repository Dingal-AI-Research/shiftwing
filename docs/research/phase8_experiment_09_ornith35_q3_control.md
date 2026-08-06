# Phase 8 Experiment 9: Ornith-35B Grouped-3-Bit Quality Control

**Model:** `deepreinforce-ai/Ornith-1.0-35B-FP8`
**Converted revision:** `1ab57ce0b44950e498a88756f40ad1ed4d0f30ca`
**Date:** 2026-07-31
**Status:** complete negative result; Ornith397 int3 prohibited

## Abstract

The direct Ornith397 int4-g128 profile produced coherent, finite output but
failed its 2-token/s target because routed-expert disk traffic dominated.
Non-lossy I/O and cache controls improved a bounded decode from 0.527355 to
0.867369 token/s but did not close the target. This experiment therefore
applies the owner-approved grouped-3-bit routed-expert treatment to the
accepted Ornith35 control before any Ornith397 low-bit artifact is created.

The base container remains immutable. A resumable sidecar stores only routed
expert projections in signed grouped int3 with group size 128; dense weights,
attention, shared experts, embeddings, and the language-model head retain
their accepted precision. Quality was measured against the already pinned
Ornith35 int4 and publisher-GGUF evidence. The int3 treatment retained 95.55%
teacher-forced argmax agreement, but fixed-corpus perplexity rose from
1.108495 to 1.226664, a 10.66% regression and more than twice the
preregistered 5% allowance. The experiment therefore stopped before
coherence and HTTP tool controls. No Ornith397 int3 artifact is authorized.

## 1. Research question

Does grouped-3-bit quantization reduce Ornith routed-expert storage while
retaining the model behavior needed for the owner-approved local performance
profile?

## 2. Why this experiment is now allowed

The conditional protocol required three observations before lower precision:

1. the direct int4 model must be numerically finite and generate coherent
   output;
2. its sustained qualification must fail 2 token/s; and
3. expert misses, read volume, or tier capacity must dominate the failure.

All three now hold. The independent 397B corruption smoke returned perplexity
1.050067545. The four-prompt CUDA profile was coherent but sustained only
0.527355 token/s. Its complete run read 1.839 TiB of experts, and a
representative measured decode spent 96.34 seconds in expert reads versus
26.44 seconds in expert matrix work. The best non-lossy pilot reached
0.867369 token/s but still read 16.003 GB for 16 measured tokens.

## 3. Terms

**Three-bit quantization** represents each weight with one of eight signed
levels. The scale for each group of 128 weights maps those small integers
back to approximate real values.

**Sidecar** means a separate set of tensor files selected only when
`EXPERT_Q3=1`. The accepted int4 container is not overwritten.

**Teacher forcing** gives both treatments the same known token history before
each prediction. It measures local numerical differences without allowing
one early free-running choice to change every later input.

**Relaxed profile** means the result may be recommended for this
resource-constrained local machine under explicitly weaker numerical
thresholds. It does not replace or erase the strict int4 reference.

## 4. Conversion method

The shared converter reads one accepted int4 expert projection at a time,
dequantizes it, and performs three deterministic scale-refinement iterations
for grouped int3. It writes `.q3`, `.qs`, and `.qtype` tensors into atomic
64-expert files. Eight independent workers run with one Torch thread each:

```sh
.venv/bin/python c/tools/requantize_expert_q2.py \
  --snapshot c/ornith35 \
  --bits 3 \
  --group-size 128 \
  --iterations 3 \
  --experts-per-file 64 \
  --workers 8 \
  --torch-threads 1
```

Completion required all 40 layers, 256 experts per layer, three routed
projections per expert, a complete manifest, and valid hashes for every
atomic output. The resumed command added `--adopt-existing` after the resume
defect described in Section 7 was corrected.

## 5. Storage control

The Windows volume had only about 6.4 GiB free because the WSL VHD was
non-sparse. Before conversion, 640 rejected Qwen35 int2 files, 160 historical
Qwen35 int3 files, and their two manifests were retired. The exact deletion
was 802 ignored artifacts totaling 22,171,170,971 bytes. Their format,
cardinality, numerical results, and regeneration method remain recorded in
the Phase 7 reports. No accepted model container, source file, Qwen397 pilot,
or Git-indexed source was removed. Ext4 can reuse those already allocated
blocks without expanding the host VHD.

## 6. Preregistered acceptance

The relaxed Ornith35 control passes only if all of the following hold:

1. the sidecar manifest is complete and bound to the accepted Ornith35
   container identity;
2. all 20 frozen prompts contain exactly 64 teacher-forced positions;
3. aggregate agreement with the pinned official-GGUF reference is at least
   90%;
4. every prompt has at least 80% agreement;
5. fixed-corpus perplexity is finite and no more than 5% above the accepted
   Ornith35 int4 value of 1.108495;
6. the four qualification prompts produce nonempty, coherent,
   non-degenerate text;
7. the live HTTP interaction emits exactly
   `get_weather({"city":"Paris"})`, consumes the deterministic 18 °C result,
   answers with 18, leaks no raw/second call, and stops cleanly; and
8. result artifacts identify the int3 sidecar and CUDA expert execution.

The 80% per-prompt floor was fixed before observing Ornith results. It is
below the original strict 85% gate but still below neither the
owner-accepted Qwen int3 precedent (82.81% worst prompt) nor the 90%
aggregate safeguard. Both strict int4 and relaxed int3 statistics will be
reported together.

## 7. Conversion pause, recovery, and resume correction

The converter reached 62 of 160 atomic files, covering 5,075,765,184 bytes.
The last durable output is
`expert-q3-l015-e0064-0127.safetensors`, 81,867,480 bytes, with SHA-256:

```text
664d2e1126959a87af4bf08105a42444dfa660fc346452b140c86ebb101c55ba
```

Windows free space nevertheless fell from about 6.4 GB to about 1.7 GB.
Although ext4 reported 483.7 GB free after the Qwen sidecars were retired,
the VHDX remained non-sparse and appended physical blocks rather than
reusing the released host allocation. The process was sent `SIGTERM` before
the volume filled. It exited, left no temporary file, and correctly withheld
`expert-q3.json`; the 62 completed files are independently hash-validatable
resume inputs.

The unrelated gateway was stopped cleanly for a brief maintenance window,
ext4 reported 502,070,693,888 trimmed bytes, and Ubuntu was terminated. The
documented sparse command was then attempted:

```powershell
wsl --manage Ubuntu --set-sparse true
```

The installed WSL release refused this operation unless the explicitly
unsafe `--allow-unsafe` option was supplied because sparse VHD mode can risk
filesystem corruption. That unsafe override was not used. `Optimize-VHD`
was unavailable and the session was not elevated, so recovery instead
removed only regenerable Gradle, NVIDIA shader, and npm caches. Downloads,
repositories, Codex runtimes, accepted containers, and user data were not
touched. Host free space recovered first to about 13.6 GB and later settled
near 24.7 GB after trim and cache reclamation completed. Ubuntu restarted,
the gateway returned to service, and the workspace head and checkpoint
identity were reverified.

The first converter restart exposed a resume defect: because an interrupted
run had correctly withheld the completion manifest, the tool treated the 62
valid files as unledgered and began regenerating layer zero. It was stopped
before publishing another chunk. The converter now fails closed on
unledgered outputs by default and accepts them only with explicit
`--adopt-existing`. Adoption validates every expected tensor name, dtype,
shape, quantization type, and hash. It also publishes an atomic
`complete=false` progress manifest after each file, so future interruptions
resume without repeating this ambiguity. A focused regression test proves
that implicit adoption is rejected and explicit adoption preserves valid
chunks while converting only missing chunks.

The corrected run adopted 62 files and converted 98. It published 160 files,
92,160 tensors, and 13,098,786,560 tensor bytes with `complete=true`. A
separate read-only pass recomputed all 160 SHA-256 values; none differed.

## 8. Results

### 8.1 Teacher-forced agreement

The single preregistered run compared 20 prompts with exactly 64 positions
each against the pinned official-Q4 reference. It matched 1,223 of 1,280
positions, or 95.546875%, exceeding the 90% aggregate minimum. Every prompt
also exceeded the 80% floor; the worst prompt matched 55/64 positions
(85.9375%). The accepted int4 treatment matched 1,232/1,280 (96.25%), so
int3 changed only nine additional argmax decisions in this test.

### 8.2 Perplexity

The frozen 1,024-source-token, 510-scored-token corpus produced:

| Treatment | PPL | Relative to accepted int4 | Decision |
|---|---:|---:|---|
| Ornith35 int4 | 1.108495 | baseline | accepted control |
| Ornith35 routed-expert int3 | 1.226664355 | +10.660342% | fail |
| Preregistered int3 ceiling | 1.16391975 | +5.000000% | maximum |

The int3 result is finite, but exceeds the ceiling by 0.062744605. This is
an acceptance failure, not a corruption failure: the model remains locally
similar in argmax space while assigning materially less probability to the
correct corpus tokens.

### 8.3 Stopped controls

Coherence and HTTP tool tests were not run. The decision rule is conjunctive,
so the perplexity failure already rejects the treatment; executing later
tests could not reverse that result and would create misleading
post-selection evidence.

## 9. Decision and implications

If any structural, perplexity, coherence, tool, or relaxed token-agreement
criterion fails, Ornith397 int3 conversion is prohibited. If every criterion
passes, the result authorizes a separately capacity-guarded Ornith397
sidecar. The larger model must then repeat finite PPL, four-prompt coherence,
CUDA-residency, generated-tool, and sustained ≥2-token/s controls; the 35B
result alone cannot close Gate 8.

The structural and token-agreement criteria passed, but perplexity failed.
Therefore Ornith397 int3 conversion is prohibited and no projected
approximately 157 GB sidecar will be created. The accepted direct Ornith397
int4 container remains unchanged. Gate 8 returns to non-lossy performance
work: the current measured ceiling is 0.867369 token/s against the frozen
2-token/s requirement.
