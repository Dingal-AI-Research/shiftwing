# Phase 8 Preflight 6: Ornith-397B Qualification Protocol

**Model:** `deepreinforce-ai/Ornith-1.0-397B-FP8`
**Immutable revision:** `8b61f97a8512d9d01bff1a9625c9a16730e115bb`
**Date:** 2026-07-29
**Status:** preregistered before full conversion and inference results

## Abstract

This report fixes the tests that Ornith-397B must pass after streaming
conversion. It separates four questions that are easy to confuse:

1. **Completeness:** did every required tensor arrive intact?
2. **Numerical usability:** can the converted model assign finite,
   non-degenerate probabilities to real text?
3. **Runtime correctness:** did the intended GPU, RAM, and NVMe paths execute?
4. **Product usefulness:** is output coherent, does tool use work, and is
   sustained generation at least 2 tokens per second?

The tests and thresholds are recorded before seeing Ornith-397B outputs.
Qwen397 3-bit conversion is not part of this protocol. Grouped 3-bit is a
conditional Ornith-only experiment and begins only if the direct Ornith397
profile proves that memory capacity or storage traffic is the limiting
factor.

## 1. Terms

**Container** means the converted collection of safetensors files plus its
index and manifests. It is the representation read by the C engine.

**Ledger** means `.conversion-state.json`, the converter's durable record of
which source shard produced which output, including size and SHA-256.

**SHA-256** is a content fingerprint. Recomputing it detects accidental file
changes that a filename or file-size check can miss.

**Perplexity** measures how surprised a language model is by known text.
Here it is a broad corruption smoke test: finite, low perplexity makes a
major scale/layout error unlikely, but does not by itself prove intelligence.

**Tiering** means keeping some expert weights in GPU memory, some in ordinary
RAM, and the rest on NVMe storage. Telemetry is required to prove which path
actually ran.

**Sustained throughput** is total measured output tokens divided by their
reconstructed decode time after fixed warm-up passes. It is not a short
four-token cache diagnostic.

## 2. Frozen model and precision identity

The accepted direct profile is:

| Property | Required value |
|---|---|
| source | `hf://deepreinforce-ai/Ornith-1.0-397B-FP8@8b61f97a8512d9d01bff1a9625c9a16730e115bb` |
| source-index fingerprint | `4b1c5ab0824e25b3768e3c6df8392394194fce86c4e9f4f3da24e3134ccf4c94` |
| source shards | 122 |
| expected output shards | 122 |
| logical tensors | 93,078 |
| physical container tensors | 278,152 |
| predicted tensor payload | 212,634,789,241 bytes |
| routed experts | grouped int4, group size 128 |
| dense input/output paths | int8 |
| shared experts | int8 |
| MTP | excluded |

The architecture matches Qwen397, but the input checkpoint uses Ornith's
per-channel compressed-tensors FP8 representation. Therefore Qwen's completed
container is useful as a structural control, not as substitute evidence for
this conversion.

The two container-cardinality predictions are metadata-derived. Applying the
text-only selection rule to the pinned Ornith index leaves at least one
retained weight in every source shard, so all 122 shards must produce output.
Ornith and Qwen expose the same 93,078-name loader inventory and the same
per-name precision map. Each exact tensor contributes one physical entry,
whereas each quantized tensor contributes data, scale, and qtype entries;
this fixes the expected physical count at 278,152 independently of the
observed conversion ledger.

## 3. Isolation control

No performance result may close Gate 8 while model conversion, model
download, a web build, or another storage benchmark is active. Before timing:

- the detached converter must have exited;
- no source download or temporary output may remain in flight;
- the completion manifests must exist;
- the GPU must have enough free memory for the fixed plan; and
- the same one-slot process must remain alive across warm-up and measured
  prompts.

Results collected under known contention may diagnose behavior but are not
acceptance measurements.

## 4. Structural audit

The first complete-model command is:

```sh
./c/colib doctor --model c/ornith397 \
  --kv-slots 1 --context 4096 \
  --cuda-expert-gb 6 --ram-cache-gb 18 \
  --runtime-headroom-gb 1 --verify-hashes \
  --expect-source \
    hf://deepreinforce-ai/Ornith-1.0-397B-FP8@8b61f97a8512d9d01bff1a9625c9a16730e115bb \
  --expect-source-fingerprint \
    4b1c5ab0824e25b3768e3c6df8392394194fce86c4e9f4f3da24e3134ccf4c94 \
  --expect-source-shards 122 \
  --expect-output-shards 122 \
  --expect-logical-tensors 93078 \
  --expect-physical-tensors 278152 \
  --expect-data-bytes 212634789241 --json
```

Acceptance requires:

1. 122 source records and 122 output shards in the ledger;
2. exact source, revision, fingerprint, and quantization signature;
3. exactly 278,152 physical container tensors;
4. every recorded output present at the recorded file size;
5. every independently recomputed SHA-256 equal to the ledger;
6. `quantization.json` with `complete: true`;
7. exactly 93,078 logical tensors;
8. exactly 212,634,789,241 tensor-payload bytes;
9. agreement among ledger totals, manifest totals, safetensors headers, and
   the index weight map;
10. no missing, duplicate, extra, overlapping, unsafe, or wrongly owned
   tensor; and
11. passing one-slot host RAM, free VRAM, and disk guards.

The hash mode is opt-in because reading roughly 198 GiB is intentionally
expensive. Its regression first accepts a valid fixture, then corrupts its
output and requires a hard failure.

## 5. Numerical corruption smoke

After the structural audit:

```sh
.venv/bin/python c/tools/eval_qwen.py \
  --snapshot c/ornith397 \
  --corpus c/bench/qwen35_eval.txt \
  --max-tokens 1024 --ctx-size 512 \
  --ram-gb 18 --ram-headroom-gb 1 \
  --max-ppl 50 --require-complete-manifest \
  --output c/ornith397_ppl_smoke.json
```

The frozen corpus has SHA-256
`01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`.
Acceptance requires a finite negative log likelihood, finite perplexity below
50, no NaN, no loader/context error, and an artifact bound to the complete
model manifest. The wide upper bound detects gross corruption; it is not a
claim that perplexity values from different tokenizers are directly
comparable.

## 6. Family-correct coherence and throughput

The fixed qualifier is:

```sh
.venv/bin/python c/tools/qualify_tiered_model.py \
  --model c/ornith397 --max-tokens 64 \
  --warmup-passes 2 --measured-passes 1 \
  --expert-ram-gb 18 --cuda-expert-gb 6 \
  --ram-headroom-gb 1 --cuda-headroom-gb 1 \
  --minimum-tps 2 \
  --output c/ornith397_qualification.json
```

The four prompts test an undergraduate data-structure explanation,
TypeScript code generation, systems debugging, and RAM-versus-NVMe reasoning.
The qualifier now dispatches through `colib_model_family`; Ornith uses its own
pinned `chat_template.jinja`, not Qwen's chat framing.

For measured request \(i\), let \(n_i\) be generated tokens and \(r_i\) its
reported decode rate. Sustained throughput is

\[
R = \frac{\sum_i n_i}{\sum_i n_i/r_i}.
\]

Automatic acceptance requires:

- every measured output is nonempty;
- \(R \ge 2.0\) tokens per second;
- hardware, TIERS, EMAP, HITS, and resident-graph telemetry are present;
- EMAP dimensions are exactly 60 layers by 512 routed experts and its tier
  counts equal TIERS;
- CUDA and the VRAM expert tier are active;
- all measured model layers use device MoE, with zero host-MoE fallback; and
- resident hidden/logit/router transfer counters are positive and
  structurally consistent.

Automatic acceptance is followed by human coherence review. A fluent but
incorrect answer is recorded rather than hidden by the speed statistic.

## 7. Generated tool-use round trip

The real HTTP boundary is exercised with:

```sh
.venv/bin/python c/tools/qualify_ornith_tools.py \
  --model c/ornith397 \
  --ram-gb 18 --ram-headroom-gb 1 \
  --cuda-expert-gb 6 --cuda-headroom-gb 1 \
  --startup-timeout 1800 --request-timeout 1800 \
  --output c/ornith397_tool_qualification.json
```

The first answer must call exactly `get_weather(city="Paris")` and finish
with `tool_calls`. The harness returns a deterministic 18 °C result. The
second answer must mention 18, emit no further/raw XML call, and terminate
with `stop`. Both HTTP envelopes and runtime telemetry are retained.

## 8. Conditional lower-bit decision

Direct int4 qualification runs first. Grouped 3-bit is considered only if:

1. output is numerically usable and coherent;
2. the ≥2 tokens/s criterion fails; and
3. decode attribution shows that misses, read bytes, or expert-tier capacity
   dominate rather than dense compute, attention, or a correctness defect.

If all three conditions hold, the next experiment quantizes Ornith35 routed
experts and repeats its frozen numerical/tool controls under a separately
recorded relaxed profile. Only after that control passes may Ornith397
sidecars be generated. The incomplete Qwen397 3-bit pilot is never resumed.

## 9. Gate rule

Gate 8 closes only when structural, numerical, coherence, runtime-path,
tool-use, and sustained-throughput evidence all pass. A successful conversion
alone cannot close the gate. If a conditional Ornith397 3-bit profile is
needed, Gate 8 remains open until that profile completes the same applicable
checks and its quality trade-off is explicitly reported.
