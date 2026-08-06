# Phase 7 Experiment 10: Complete-Model Qualification

**Target:** Qwen3.5-397B-A17B-FP8 converted container
**Source revision:** `ea5b4f81096f3901c91dea97f81324302495781d`
**Date:** 2026-07-28
**Status:** correctness and resource criteria pass; throughput criterion fails

## Abstract

This experiment ran the preregistered Gate 7 qualification on the complete
397-billion-parameter model. The test first verified every container tensor
and resource guard, then measured perplexity, generated four kinds of
technical answer, and ran two warm passes followed by one measured pass
without restarting the inference engine. The container and numerical checks
passed. The model produced coherent answers, stayed within the specified
memory profile, and executed the resident CUDA graph. However, its
token-weighted measured decode rate was 0.3701 tokens per second, below the
fixed requirement of 2.0 tokens per second. Gate 7 therefore remains open.

The initial telemetry also exposed a measurement defect: the reported cache
percentage was always zero and the tier map omitted experts whose host cache
entry had been detached after upload to the GPU. This did not affect model
output, but it made the cache explanation incomplete. Experiment 11 corrects
those counters before testing performance treatments.

## 1. Terms used in this report

- A **parameter** is a learned number in the model. “397B” means approximately
  397 billion learned numbers in the original model.
- An **expert** is one small feed-forward subnetwork in a
  mixture-of-experts layer. The router selects 10 of 512 experts per token in
  each layer, so only a fraction of all expert weights are needed at once.
- A **cache hit** occurs when a requested expert is already in RAM or GPU
  memory. A **miss** requires reading it from the NVMe drive.
- **Perplexity (PPL)** is a language-model error measure; lower is better. A
  finite value below the preregistered limit provides a numerical smoke test,
  not a complete intelligence score.
- **Time to first token (TTFT)** includes prompt processing and any expert
  loading before the first generated piece.
- **Decode throughput** is the number of generated tokens per second after
  prompt processing.

## 2. Research question

Does the complete converted Qwen397 model meet every Gate 7 condition under
the frozen one-slot 18 GiB host expert cache and 6 GiB GPU expert cache
profile?

The conditions were frozen before the conversion completed:

1. exact container and manifest integrity;
2. finite fixed-corpus PPL below 50;
3. coherent nonempty output on four technical prompts;
4. complete hardware, tier, expert-map, hit, profile, and resident-CUDA
   telemetry;
5. no out-of-memory failure under the guarded resource profile;
6. token-weighted warm decode throughput of at least 2.0 tokens per second.

## 3. Method

The converted model contains 93 output shards, 278,152 physical tensors,
93,078 logical tensors, and 212,634,789,241 tensor-payload bytes. Routed
experts use grouped int4 with groups of 128; dense and shared matrices use
int8; MTP was excluded from the Gate 7 resource profile.

The PPL smoke scored 510 tokens from a hash-pinned corpus at context 512. The
chat qualifier used context 4,096, one runtime slot, 18 GiB of host expert
cache, 6 GiB of GPU expert cache, two complete warm passes, and one measured
pass. Each of four prompts generated 64 greedy tokens. The prompts covered an
undergraduate data-structure explanation, TypeScript programming, systems
debugging, and RAM-versus-NVMe trade-offs.

For measured turns \(i\), the aggregate rate was computed from token counts
\(n_i\) and individual decode rates \(r_i\):

\[
R = \frac{\sum_i n_i}{\sum_i n_i/r_i}.
\]

This is a token-weighted harmonic aggregation. It reconstructs total decode
time and prevents one unusually fast short trial from dominating the result.

## 4. Results

| Criterion | Observation | Outcome |
|---|---:|---|
| source/output shards | 94 / 93, as specified | pass |
| physical/logical tensors | 278,152 / 93,078 | pass |
| exact payload bytes | 212,634,789,241 | pass |
| PPL smoke | 1.01912 over 510 tokens | pass |
| PPL processing rate | 0.267 tok/s | diagnostic |
| generated outputs | four nonempty coherent answers | pass |
| CUDA graph | resident graph active, no host-MoE fallback | pass |
| memory safety | no OOM under frozen guards | pass |
| measured turn rates | 0.3159, 0.4342, 0.4160, 0.3406 tok/s | fail |
| token-weighted rate | **0.3701 tok/s** | **fail** |

The full run required approximately 85 minutes. Engine startup was 546.4
seconds. Measured TTFT values ranged from 96.0 to 144.7 seconds. Outputs were
stable across warm and measured passes, which is useful evidence that the
model was numerically functional even though it was slow.

The complete machine-readable artifact is
`c/qwen397_qualification.json`; the numerical smoke artifact is
`c/qwen397_ppl_smoke.json`.

## 5. Interpretation

Five of the six preregistered conditions passed. The result separates two
questions that are often incorrectly combined:

- **Can the model run correctly on this machine?** Yes.
- **Can it run at the promised interactive rate with this cache profile?**
  Not yet.

The measured rate is only 18.5% of the required rate. This gap is too large to
describe as timing noise. The coherent output and low PPL also make a gross
tensor-layout or quantization failure unlikely.

The run’s original `cache_hit_percent` field was zero because the mux did not
capture the correct counter interval. The reported VRAM tier also omitted
device-only entries after their host owner was detached. Therefore this
experiment establishes the throughput failure but cannot, by itself,
quantitatively assign that failure to cache misses. Experiment 11 introduces
decode-only counters and repeats a bounded workload.

## 6. Conclusion

The complete Qwen397 model is structurally valid, numerically plausible,
coherent, CUDA-resident, and memory-safe on the reference workstation. Gate 7
does not close because the unchanged 2.0 tok/s criterion is not met. Future
work must improve the measured implementation rather than relax the gate.
