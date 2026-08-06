# Phase 7 Experiment 14: Low-Bit Routed-Expert Sidecars

**Target:** Qwen3.5-397B-A17B capacity treatment, quality-gated on
Qwen3.5-35B-A3B
**Hardware:** Ryzen 7 7700X, RTX 5070 Ti, 29.4 GiB WSL RAM
**Date:** 2026-07-28
**Status:** 2-bit and 3-bit treatments rejected; Gate 7 closed as a measured
32 GiB hardware no-go, not passed

## Abstract

The preceding experiment found that sustained 397B decoding needed an ideal
working set of about 88 different experts per layer, while the reference
workstation could retain approximately 48 experts in host RAM and 16 mostly
overlapping experts in GPU memory. This experiment tested whether reducing
only the routed-expert weights from 4 bits to 2 or 3 bits could increase
cache capacity without materially changing model predictions.

Reusable grouped 2-bit and 3-bit quantizers, resumable sidecar conversion,
CPU dot products, and CUDA expansion into the existing 4-bit kernels were
implemented. The original model container was not modified. The complete
35B 2-bit sidecar occupied 9.07 GB but matched only 12 of 128
teacher-forced predictions, so it was rejected immediately. The 3-bit
sidecar occupied 13.10 GB and matched 1,219 of 1,280 predictions (95.23%).
However, one frozen prompt matched only 53 of 64 predictions (82.81%),
below the preregistered 85% per-prompt minimum. Two contiguous
mixed-precision experiments also failed to repair this prompt.

The 3-bit result is scientifically close but does not pass the fixed quality
rule. It is therefore not used to convert or qualify the 397B model. Since
unchanged precision needs more memory, and both authorized lower-bit
treatments failed quality, the 32 GiB reference profile is closed as a
measured Gate-7 no-go. The original throughput requirement remains unmet.

## 1. Definitions

- An **expert** is one of many small feed-forward networks in a
  mixture-of-experts model. The router selects only a few experts for each
  token.
- **Quantization** stores a weight using a small integer plus a scale instead
  of a 32-bit floating-point number. Fewer bits reduce storage, but introduce
  rounding error.
- A **group** is a short block of weights sharing one scale. This study uses
  groups of 128 weights.
- A **sidecar** is an additional set of files beside the accepted model. It
  can be selected experimentally without overwriting the source tensors.
- **Teacher forcing** feeds the trusted reference token at every step and
  asks whether the tested model predicts the same next token. This prevents
  one early disagreement from making every later position incomparable.
- **Aggregate agreement** combines all predictions across prompts.
  **Per-prompt agreement** prevents a high average from hiding a severe
  failure on one input.
- A **negative gate closure** means the experiment has reached a supported
  no-go decision. It does not mean the performance gate passed.

## 2. Research questions and hypotheses

The primary question was whether lower-bit routed experts could provide
enough additional RAM capacity to make the 397B sustained decode target
plausible on the installed machine.

The preregistered numerical acceptance rule was unchanged:

1. at least 85% teacher-forced agreement on every prompt;
2. at least 90% teacher-forced agreement over all prompts;
3. 64 compared positions per prompt.

We expected 2-bit weights to be risky because four representable levels are
coarse. We expected 3-bit weights to be more accurate because eight levels
provide twice as many choices, while still saving approximately one quarter
of routed-weight bytes.

## 3. Implementation

### 3.1 Shared quantization primitives

`c/tools/convert_qwen.py` now provides deterministic grouped 2-bit and 3-bit
quantize/dequantize functions. The 2-bit representation uses four odd
levels, while the 3-bit representation tests both orientations of an
eight-level codebook and retains the lower squared-error result. Tests cover
exact packing, zero groups, partial final groups, leading dimensions,
non-finite inputs, invalid contracts, and deterministic output.

The same functions are used by `c/tools/requantize_expert_q2.py`, which is a
resumable converter despite its historical filename. `--bits 2` writes
`.q2`, `.qs`, and `.qtype` tensors; `--bits 3` writes the corresponding
`.q3` tensors. Every output shard is written atomically and recorded by hash
in a completion manifest. Existing 4-bit expert matrices are processed one
at a time, so peak conversion memory does not scale with model size.

### 3.2 C and CUDA execution

The C loader recognizes quantization tags 2 and 3 and validates their row,
group, and scale geometry. CPU scalar and row kernels decode the packed
values directly. On CUDA, compact 2-bit or 3-bit values are expanded into
equivalent 4-bit nibbles during upload; the already-qualified 4-bit matrix
kernels then perform the multiplication. Signed 3-bit scales preserve the
selected mirrored codebook exactly.

The sidecars are opt-in through `EXPERT_Q2=1` or `EXPERT_Q3=1`; the accepted
4-bit model remains the default. The safetensors reader's shard limit was
raised from 512 to 4,096 after the deliberately small 2-bit shard layout
produced 640 files.

### 3.3 Diagnostic layer boundaries

After the complete 3-bit model narrowly failed one prompt,
`EXPERT_Q3_MIN_LAYER` and `EXPERT_Q3_MAX_LAYER` were added as diagnostic
switches. The comparison harness exposes and records these bounds. They
allow a contiguous part of the model to retain the accepted 4-bit experts
without changing either tensor set.

## 4. Controlled method

The official converted Qwen3.5-35B-A3B snapshot and its frozen 20-prompt
Q4_K_M reference were used as a lower-cost quality proxy before any 397B
conversion. All runs used:

- the same tokenizer, rendered prompts, and 64 reference tokens;
- greedy next-token comparison;
- eight CPU threads and the qualified CUDA expert path;
- complete sidecar manifests tied to the source-container identity;
- the original 4-bit dense, shared-expert, attention, embedding, and output
  tensors.

The 2-bit format was first tested on two prompts. Its large error made a
20-prompt run scientifically unnecessary. The 3-bit format passed the
two-prompt screen and therefore advanced to the complete frozen suite. The
single failing prompt then received three bounded layer-sensitivity tests.
Acceptance thresholds were not changed after observing results.

## 5. Results

### 5.1 Artifacts and storage

| Expert format | Sidecar files | Physical tensors | 35B sidecar bytes | Approximate bytes per 397B expert |
|---|---:|---:|---:|---:|
| grouped 2-bit | 640 | 92,160 | 9,072,147,520 | 3,538,944 |
| grouped 3-bit | 160 | 92,160 | 13,098,786,560 | 5,111,808 |
| accepted grouped 4-bit | original container | — | — | 6,684,672 |

For the 397B geometry, grouped 3-bit is 76.47% of the accepted routed-expert
size. An 18 GiB host tier can ideally retain about 63 3-bit experts per
layer instead of 48 4-bit experts. A 7 GiB GPU tier stores expanded 4-bit
entries and contributes about 18 additional experts per layer. The ideal
union is therefore about 81, still below the 88-entry static-coverage target
from Experiment 13, although reduced storage traffic could narrow that gap.

### 5.2 Numerical quality

| Treatment | Prompts | Teacher-forced matches | Aggregate | Worst prompt | Decision |
|---|---:|---:|---:|---:|---|
| all experts 2-bit | 2 | 12 / 128 | 9.38% | 4 / 64 | reject |
| all experts 3-bit, screen | 2 | 123 / 128 | 96.09% | 61 / 64 | advance |
| all experts 3-bit, frozen suite | 20 | 1,219 / 1,280 | 95.23% | **53 / 64** | reject |

Nineteen of the twenty 3-bit prompts met the 85% rule. Prompt 14 needed at
least 55 of 64 matches but obtained 53. For comparison, the accepted 4-bit
model obtained 56 of 64 on that prompt. The failure is small in absolute
count but real under the rule established before the test.

### 5.3 Layer-sensitivity tests

| 3-bit layers | 4-bit layers | Prompt-14 matches | Agreement | Pass? |
|---|---|---:|---:|---|
| 0–39 | none | 53 / 64 | 82.81% | no |
| 0–19 | 20–39 | 53 / 64 | 82.81% | no |
| 0–9 | 10–39 | 54 / 64 | 84.38% | no |
| 10–39 | 0–9 | 53 / 64 | 82.81% | no |

Both halves can reproduce the failure when quantized separately. This shows
that the difference is distributed or nonlinear; it cannot be repaired by a
simple early/late precision boundary. An exhaustive search over layer
subsets would tune the format to one observed prompt and would require many
multi-minute production-model runs, so it was not pursued.

## 6. Interpretation

The 2-bit result demonstrates that capacity alone is not useful if the model
being cached is no longer numerically faithful. The 3-bit result is much
more informative: average behavior remains close to 4-bit, but the
per-prompt guard correctly detects a localized degradation that the 95.23%
aggregate score hides.

Even a quality-passing all-3-bit treatment would provide an ideal
approximately 81-entry RAM/VRAM union, not the 88 entries associated with
95.33% route coverage. It was plausible enough to test because each storage
miss would also read 23.53% fewer bytes, but it was not certain to cross
2 tok/s. Converting 397B after the 35B quality failure would therefore spend
substantial disk, network, and conversion time without an accepted format.

## 7. Limitations

The quality proxy is the 35B checkpoint, not the 397B checkpoint. A larger
model could be either more or less tolerant of the same quantizer. However,
the approved sequencing requires a smaller-model quality gate before a
large conversion, so this uncertainty is a reason not to proceed rather
than permission to skip the gate.

Teacher-forced argmax agreement does not measure every aspect of language
quality. The experiment also did not test a trained quantizer, activation-
aware calibration, per-layer bit allocation, or a different 3-bit
codebook. Those are new research programs, not minor extensions of the
authorized treatment.

The mixed-format timings include cache behavior from two tensor sets and are
diagnostic only. No throughput claim is made from them.

## 8. Decision

Grouped 2-bit and grouped 3-bit routed-expert sidecars remain implemented for
reproducible research, but neither is an accepted runtime default. No 397B
low-bit sidecar is generated.

The three Gate-7 choices stated in Experiment 13 are now resolved:

1. unchanged precision requires a machine with at least 48 GB RAM and is not
   testable on the installed hardware;
2. the separately gated 2-bit and 3-bit treatments fail numerical quality;
3. therefore the 32 GiB reference profile is closed as a measured no-go.

Gate 7 did **not** pass: sustained throughput remains 0.772 tok/s against a
2 tok/s requirement. Later 35B and software-only work may proceed, but any
397B release claim remains conditional on requalification with greater host
memory or a future lower-bit method that independently passes quality.

## 9. Reproducibility artifacts

- `c/qwen35/expert-q2.json`
- `c/qwen35/expert-q3.json`
- `c/bench/qwen35_q2_prefix_probe2.json`
- `c/bench/qwen35_q3_prefix_probe2.json`
- `c/bench/qwen35_q3_tf_full.json`
- `c/bench/qwen35_q3_cut09_p14.json`
- `c/bench/qwen35_q3_min10_p14.json`
- `c/tools/requantize_expert_q2.py`

## 10. Post-study project decision

After this strict-gate result was recorded, the project owner explicitly
accepted the measured 3-bit weakening as a local performance trade-off. This
does not change the data or retroactively pass the original rule.
`phase7_experiment_15_relaxed_q3_profile.md` defines the separate experimental
397B profile and its preregistered safety, perplexity, coherence, telemetry,
and sustained-throughput conditions.
