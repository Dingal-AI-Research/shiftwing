# Phase 7 Preflight 5: Preregistered 397B Qualification Protocol

**Target:** Qwen3.5-397B-A17B, official FP8 revision
`ea5b4f81096f3901c91dea97f81324302495781d`
**Converted format:** routed int4-g128, dense/shared int8, no MTP
**Reference hardware:** Ryzen 7 7700X, RTX 5070 Ti 16 GB, 29.4 GiB RAM,
NVMe under WSL2
**Status:** protocol frozen before complete-model results

## Abstract

Gate 7 is performance-sensitive and the equivalent-capacity 35B proxy predicts
less cache locality than the two-token-per-second target probably requires.
This preflight therefore fixes the complete-model acceptance procedure before
the conversion result is available. It defines container integrity, resource,
language-quality, telemetry, warm-up, and throughput criteria and supplies a
one-process harness that preserves the learned RAM/VRAM expert tiers across
trials.

## 1. Research question

Can the converted 397B target produce coherent text without numerical failure
and sustain at least 2.0 generated tokens per second after a realistic
expert-cache warm-up, while staying inside the one-slot 16 GB GPU and 29.4 GiB
host profile?

The null result is operationally important: if warmed throughput is below
2.0 tokens/s, Gate 7 remains open even when the output is coherent.

## 2. Frozen runtime profile

The initial run uses:

| Control | Value |
|---|---:|
| KV slots | 1 |
| context | 4,096 tokens |
| CUDA expert cache | 6 GiB |
| host expert cache | 18 GiB |
| CUDA / host runtime headroom | 1 GiB / 1 GiB |
| MTP | disabled |
| predictive prefetch | disabled |
| layer-batched I/O | enabled |
| `io_uring` + direct I/O | enabled |
| persistent route heat | enabled |

This is the resource plan accepted in Preflight 2 and reconstructs its
14.20 GiB post-context VRAM and 26.20 GiB host requirement. Predictive prefetch remains
off because Experiment 8 increased bytes read and reduced throughput on the
35B control. The harness sets `RAM_GB=18`; `EXPERT_RAM` is deliberately not
used because that older control denotes a count of expert slots per layer,
not a byte budget.

## 3. Integrity and resource checks

After conversion and before model execution:

```sh
./c/colib doctor --model c/qwen397 --kv-slots 1 --context 4096 \
  --cuda-expert-gb 6 --ram-cache-gb 18 --runtime-headroom-gb 1 --json
```

The inventory must account for all 94 source shards and parse every retained
output shard, find no missing, extra,
duplicate, misowned, overlapping, or out-of-bounds tensor, and match the
index's exact payload byte count. The resource plan must retain runtime
headroom and the CUDA-linked engine must be detected.

The converter, downloads, web builds, and other storage benchmarks are stopped
before timing. No result collected under conversion contention can close the
gate.

### Interim resource audit

At 50/94 committed shards, the stricter doctor reports the expected
conversion warning but no structural or resource failure:

| Observation | Value |
|---|---:|
| parsed partial shards | 50 / 50 |
| physical tensors | 153,600 |
| committed payload | 114,085,120,000 bytes |
| planned one-slot VRAM | 15,247,031,673 bytes (14.20 GiB) |
| planned host RAM | 28,131,933,561 bytes (26.20 GiB) |
| planned output/shard/reserve disk | 238,834,089,746 bytes (222.43 GiB) |
| measured GPU total / currently free | 15.92 / 15.62 GiB |
| resource checks | disk, RAM, VRAM pass |

The final audit must replace the warning with a complete 94-shard
index/manifest pass; this interim result proves only the durable prefix and
resource assumptions.

## 4. Language-quality smoke

The existing `eval_qwen.py` teacher-forced path evaluates at least 1,024 fixed
corpus tokens in complete contexts. Acceptance requires:

- finite mean negative log likelihood and perplexity;
- no loader, NaN, or context error;
- perplexity below 50 on the frozen mixed prose/code corpus.

The bound is executable rather than manually interpreted:

```sh
.venv/bin/python c/tools/eval_qwen.py \
  --snapshot c/qwen397 --corpus c/bench/qwen35_eval.txt \
  --max-tokens 1024 --ctx-size 512 \
  --ram-gb 18 --ram-headroom-gb 1 --max-ppl 50 \
  --require-complete-manifest \
  --output c/qwen397_ppl_smoke.json
```

This is a corruption smoke test, not a claim of benchmark quality or parity
with an unquantized 397B implementation. The required completion manifest
binds the PPL artifact to the source revision/fingerprint, shard and tensor
counts, payload bytes, and precision map. Four deterministic chat prompts
then cover conceptual explanation, TypeScript generation, systems diagnosis,
and memory/storage reasoning. Their complete texts are retained for human
coherence review.

## 5. Warm throughput method

`c/tools/qualify_tiered_model.py` starts one persistent engine. It renders
every prompt through the Qwen chat template, performs two complete unmeasured
passes, then one measured 64-token pass. Reusing one process is essential:
restarting between prompts would discard the host and device expert caches
whose steady state Gate 7 is intended to measure.
Before startup, the harness requires a complete converter manifest and copies
its immutable source/fingerprint, shard/tensor/byte totals, precision map, and
MTP status into the result artifact.

```sh
.venv/bin/python c/tools/qualify_tiered_model.py \
  --model c/qwen397 --max-tokens 64 \
  --warmup-passes 2 --measured-passes 1 \
  --expert-ram-gb 18 --cuda-expert-gb 6 \
  --minimum-tps 2 --output c/qwen397_qualification.json
```

For measured request \(i\), let \(n_i\) be generated tokens and \(r_i\) the
engine-reported decode rate. The preregistered sustained rate is

\[
R = \frac{\sum_i n_i}{\sum_i n_i/r_i}.
\]

This token-weighted harmonic aggregation represents total generated tokens
divided by reconstructed decode time; it cannot be inflated by averaging a
few short fast turns equally with longer slow turns.

## 6. Acceptance criteria

The automatic chat result passes only if:

1. every measured output is non-empty;
2. hardware, tier, expert-map, and hit telemetry are all present; EMAP must be
   exactly 60×512, contain no reserved tier, agree exactly with TIERS counts,
   and HITS must have the exact bit length with zero padding bits;
3. CUDA must report a real GPU and at least one VRAM-resident expert;
4. `RESIDENT` telemetry must report positive resident-layer/device-MoE
   transactions, zero host-MoE fallback, and positive hidden/logit/router
   transfer counters;
5. the sustained rate \(R\) is at least 2.0 tokens/s.

Gate 7 closes only when the automatic result is joined by:

- the exact container and memory checks;
- finite perplexity below 50;
- human review finding all four retained outputs coherent;
- no OOM, process restart, or fallback that violates the frozen profile;
- telemetry showing that configured RAM, VRAM, and disk tiers actually
  participated as reported.

## 7. Interpretation plan

If throughput fails, the result will be reported without changing the
threshold or warm-up count. Route heat and stage telemetry will determine the
next bounded experiment:

- insufficient non-disk hit rate motivates an expert-atlas/cache-allocation
  study;
- high hit rate with large expert compute time motivates kernel work;
- high disk wait with acceptable predictor precision may justify a new,
  controlled prefetch policy;
- memory failure requires a revised resource profile and a complete rerun.

The first failure is diagnostic evidence, not permission to close the gate
with a weaker definition.
