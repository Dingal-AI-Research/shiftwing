# Phase 7 Experiment 9: Complete 397B Streaming Conversion

**Target:** Qwen3.5-397B-A17B-FP8
**Source revision:** `ea5b4f81096f3901c91dea97f81324302495781d`
**Converted precision:** routed int4-g128, dense/shared int8, no MTP
**Status:** conversion and structural qualification complete; numerical and
throughput qualification remain open

## Abstract

This experiment converted the official 397B FP8 checkpoint into the
text-runtime format used by Colibri. The converter downloaded one source shard
at a time, decoded block-scaled FP8 tensors, quantized the retained tensors,
wrote each output atomically, recorded a SHA-256 digest, and removed the
temporary source data. The process was interrupted and resumed without losing
completed work. A fresh integrity sweep verified all 90 previously committed
outputs before the final four source shards were processed.

The completed container has 93 retained output shards, 278,152 physical
tensors, 93,078 logical tensors, and 212,634,789,241 tensor payload bytes. This
is exactly the byte count predicted before conversion. One source shard
contained only the deferred multi-token-prediction component and correctly
produced no retained output. The initial diagnostic tool confused source-shard
count with output-shard count; separating those fields fixed the contract and
is covered by a regression test. The final structural and resource diagnostic
passes. These findings establish container integrity but do not yet establish
language quality or the two-token-per-second Gate 7 performance target.

## 1. Research question

Can the 94-file official FP8 checkpoint be transformed into a complete,
resumable, loader-compatible text container whose exact tensor inventory and
resource plan can be verified before inference?

## 2. Method

The converter used the immutable source identifier:

```text
hf://Qwen/Qwen3.5-397B-A17B-FP8@ea5b4f81096f3901c91dea97f81324302495781d
```

Its source-index fingerprint was:

```text
5a0489df2921265612613cd9b212842d2ba3af693eac1cf33d35f04fb9c15a9e
```

For each source shard, the procedure was:

1. download into a staging directory;
2. decode the source FP8 weights using their inverse-scale tensors;
3. omit vision and the Phase 6-deferred MTP component;
4. translate tensor names and expert layouts to the loader contract;
5. quantize routed experts to symmetric int4 in groups of 128 and retain
   dense/shared paths at int8 where specified;
6. atomically write the output safetensors file;
7. compute and record its SHA-256 digest, payload bytes, and tensor count;
8. atomically update the conversion ledger and delete the source file.

After a user-requested pause at 90/94 source shards, the same command resumed.
It re-read and re-hashed every committed output. Only after all 90 digests
matched did it download shards 91 through 94.

## 3. Results

| Measurement | Result |
|---|---:|
| source shards accounted for | 94 / 94 |
| retained output shards | 93 |
| physical tensors | 278,152 |
| logical loader tensors | 93,078 |
| loader inventory present / expected | 93,078 / 93,078 |
| tensor payload | 212,634,789,241 bytes |
| preflight prediction | 212,634,789,241 bytes |
| prediction error | 0 bytes |
| MTP included | no |
| completed outputs re-hashed on resume | 90 / 90 |

Source shard 93 contained 2,304 indexed tensors, all under `mtp.*`. Because the
frozen Gate 7 profile excludes MTP, its conversion record correctly has no
output file and zero retained bytes. Source shard 94 supplied the remaining
retained tensors and completed the global inventory.

The generated index reports 278,152 tensor-to-file mappings across 93 files.
Its `metadata.total_size` equals both the converter ledger sum and the
quantization manifest's `data_bytes`.

## 4. Diagnostic-contract correction

The first complete `colib doctor` run passed container parsing but reported:

```text
source_shards 94 != container 93
```

This was a diagnostic-schema error. `source_shards` counts files in the
publisher checkpoint, while the parsed container count is the number of
retained output files. The converter now emits both:

```json
{
  "source_shards": 94,
  "output_shards": 93
}
```

The diagnostic compares `output_shards` with the container and preserves
backward compatibility for older manifests where every source shard produced
an output. A regression test constructs a two-source/one-output manifest and
requires it to pass.

## 5. Structural and resource qualification

The frozen one-slot command was:

```sh
./c/colib doctor --model c/qwen397 --kv-slots 1 --context 4096 \
  --cuda-expert-gb 6 --ram-cache-gb 18 --runtime-headroom-gb 1
```

Every check passed:

- model path, configuration, and tokenizer;
- 278,152 tensors across 93 output shards;
- complete quantization manifest;
- CUDA-linked engine and RTX 5070 Ti detection;
- 0.42 GiB one-slot recurrent/KV state;
- host RAM, VRAM, and disk resource guards.

## 6. Interpretation and limitations

The exact agreement with the preregistered byte prediction, complete logical
inventory, atomic resume behavior, and passing diagnostic provide strong
evidence that the conversion is structurally correct. They do not prove that
int4 quantization preserved useful model behavior. A model can be complete yet
numerically damaged by a scale, layout, or kernel error.

Gate 7 therefore remains open pending:

1. finite fixed-corpus perplexity below 50;
2. coherent retained outputs on four frozen prompts;
3. exact tier, expert-map, hit, and CUDA-residency telemetry;
4. no OOM under the frozen 18 GiB RAM / 6 GiB VRAM-cache profile;
5. token-weighted warm decode throughput of at least 2.0 tokens per second.
