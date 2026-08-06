# Phase 8 Preflight 8: Owner-Accepted Ornith397 Grouped 3-Bit Profile

**Models:** Ornith-1.0-35B quality control and Ornith-1.0-397B production target

**Date:** 2026-07-31

**Status:** authorized; Qwen storage retired; Ornith397 conversion pending

## Abstract

The lossless Ornith397 int4-g128/int8 profile passes structural, numerical,
coherence, and CUDA-residency controls but reaches 0.631964 token/s against
the owner-revised 0.85 production minimum. The earlier Ornith35 grouped-int3
control retains 95.546875% aggregate next-token agreement with a worst prompt
of 85.9375%, while increasing perplexity by 10.660342% relative to int4. That
experiment failed the original 5% PPL limit.

The project owner now explicitly accepts this measured class of weakening in
exchange for the expected capacity and speed benefit, based on the expectation
that a 397B/17B-active model at 3-bit remains more capable than the
35B/3B-active int4 control. This preflight authorizes an Ornith397 routed-expert
grouped-int3 sidecar. It does not assert capability equivalence and does not
guarantee the performance target. A prospective 12% relative-PPL ceiling,
unchanged token-agreement floors, coherent coding responses, generated tool
use, CUDA residency, and the 0.85 token/s production threshold remain required.

## 1. Research question

Can grouped 3-bit routed experts reduce the Ornith397 storage/cache working set
enough to sustain at least 0.85 token/s while retaining the owner-accepted
quality class demonstrated by Ornith35?

## 2. Prior evidence

### 2.1 Quality control

| Ornith35 treatment | Aggregate TF | Worst prompt | PPL | Relative PPL |
|---|---:|---:|---:|---:|
| int4-g128 | 96.25% | 90.625% | 1.108495 | baseline |
| routed int3 | 95.546875% | 85.9375% | 1.226664355 | +10.660342% |

The int3 model is not corrupted: it preserves most argmax decisions and has
finite low perplexity. Its probability calibration is measurably worse, so
the result must be described as a quality trade-off rather than lossless
compression.

### 2.2 Capacity motivation

The full optimized Ornith397 int4 run reads 414.88 GiB of expert payload
across four measured 64-token decodes and sustains 0.631964 token/s. A 3-bit
payload uses approximately 75% of the grouped 4-bit data before small scale
and header overheads. This should increase effective RAM/NVMe bandwidth and
cache coverage, but the exact improvement must be measured.

## 3. Revised quality rule

The original 5% relative-PPL ceiling remains historical for the lossless
fallback decision. It is prospectively replaced only for the explicitly
lossy Ornith397 int3 production profile:

- Ornith35 calibration must retain at least 90% aggregate and 85% per-prompt
  teacher-forced agreement; the completed 95.55%/85.94% result passes;
- the Ornith397 int3 fixed-corpus PPL must be finite and no more than 12%
  above its matching int4 result;
- four fixed coding/diagnostic outputs must remain nonempty and coherent;
- the generated HTTP tool round trip must still select the correct function,
  consume its result, and stop cleanly; and
- no numerical, manifest, resource, CUDA-residency, or server criterion is
  relaxed.

The 12% PPL ceiling gives 1.339658 percentage points of prospective margin
above the observed 35B control while preventing an open-ended quality waiver.
It is a product trade-off selected after observing the 35B result and is
labelled as such.

## 4. Conversion protocol

The accepted int4 container remains immutable. The existing resumable
converter appends only `.q3`, `.q3.qs`, and `.q3.qtype` tensors in atomic
sidecar files:

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

Before committing each resume record, the converter validates tensor names,
dtypes, shapes, qtype, byte length, and SHA-256. It publishes an incomplete
progress manifest atomically after every file and a complete
`expert-q3.json` only after every expected layer/expert projection exists.
The runtime must never auto-select an incomplete sidecar.

The 397B model has 60 layers × 512 experts. At 64 experts per file, the
expected inventory is 480 atomic files and 276,480 physical tensors. The
payload is expected to be roughly 147–157 GB; exact bytes are a completion
result, not a preregistered assertion.

## 5. Qualification sequence

After independent hash verification:

1. run a fixed-corpus int3 perplexity comparison against the existing int4
   result and apply the 12% ceiling;
2. run teacher-forced agreement against the retained int4 target on the
   frozen prompt set;
3. run the four-prompt coherence profile with `EXPERT_Q3=1` and require
   complete low-bit/CUDA telemetry;
4. run the generated Ornith397 HTTP tool round trip;
5. run the complete two-warm-pass, four-prompt, 64-token production profile
   at the unchanged ≥0.85 token/s threshold; and
6. only on Gate-8 pass, update the independent auditor to require the exact
   q3 manifest and run the unchanged Gate-9 AB/BA, cancellation, and web
   sequence.

The int4 container remains the rollback path. Any missing sidecar tensor,
hash mismatch, nonfinite PPL, quality-limit failure, host-MoE fallback, or
throughput below 0.85 closes the int3 branch negatively.

## 6. Storage disposition

The Qwen35 and Qwen397 full containers are retired research targets and are
not inputs to any active Ornith gate. Their architecture facts, benchmark
JSON, hashes, and scientific reports remain in Git. To prevent the WSL VHD
from growing while the large Ornith397 sidecar is written, the owner authorizes
removal of:

- `c/qwen35/` and `c/qwen397/`;
- their `.qwen35.source/` and `.qwen397.source/` caches; and
- regenerable Qwen-only bench directories.

The tiny fp32/int8/int4 Qwen oracle, reference JSON fixtures, tokenizer data,
C implementation, tests, and research documentation remain because they are
small and continue to protect numerical correctness. The 20 GB GGUF in
`c/reference/` is Ornith35 quality evidence and is retained.

Deleting data inside ext4 makes blocks reusable by the Ornith conversion. It
does not guarantee an immediate reduction in the Windows-visible VHDX size;
that would require a separate safe trim/compaction operation.

## 7. Limitations

- The claim that Ornith397 int3 remains stronger than Ornith35 int4 is an
  informed capability expectation, not yet a benchmark result.
- A smaller disk representation does not guarantee proportional decode speed;
  routing locality, dequantization, upload, and cache replacement still matter.
- The 12% PPL limit is an owner-approved product boundary rather than a
  universal scientific quality threshold.
- Conversion is long-running and must remain resumable and hash-bound.
