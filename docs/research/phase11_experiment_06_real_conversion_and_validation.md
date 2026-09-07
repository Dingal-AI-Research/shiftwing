# Phase 11 Experiment 6: real conversion, resume, and native-byte audit

## Disposition

Gate 11.0 passes for the pinned
`deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
checkpoint. The source was converted losslessly into the native colib layout,
an external SIGINT proved segment-level resume, every converted record was
independently compared with its pinned source range, and both Python and C
accept all 72,317 real tensor descriptors.

This result qualifies the model container only. It does not establish
end-to-end model quality, live mux completeness, tok/s, TTFT, DSpark benefit,
or promotion over Ornith397.

## Storage preflight

The post-retirement preflight ran at
`2026-08-16T20:49:44.646607+00:00` and accepted the exact conversion plan.

| Field | Bytes |
|---|---:|
| Filesystem capacity | 1,081,101,176,832 |
| Free before conversion | 782,779,924,480 |
| Exact planned segment bytes | 166,881,088,004 |
| Projected peak new bytes | 166,893,286,918 |
| Projected final free | 615,886,637,562 |
| Required final free | 107,374,182,400 |

The source was already complete. The conservative preflight counted
12,198,914 metadata bytes as remaining source work and still left more than
508 GB above the required floor. The preflight artifact hashes to
`39f50f3faabf04412fbd1c64461f97198f648174d16d4cf7ed507a37f7184279`.

## Conversion and interruption proof

The exact converter command retained in atomic state is:

~~~sh
./.venv/bin/python c/tools/convert_deepseek_v4.py \
  --source c/.deepseek-v4-flash-0731.source \
  --output c/deepseek-v4-flash-0731 \
  --alignment 4096 \
  --min-final-free-gib 100
~~~

Conversion began at `2026-08-16T20:50:14.930571+00:00`. An external
15-second SIGINT landed during segment-ten `fsync` after nine prior segments
had been committed. The ninth segment's completion timestamp is
`2026-08-16T20:50:27.752555+00:00`; the uncommitted segment remained only as
a hidden partial. Resume rehashed and skipped all nine committed files,
replaced the partial, and completed at
`2026-08-16T21:50:43.867162+00:00`.

The unchanged real plan is
`ec527bb2d8dad257876e0dd1255e6df50666004f82190a186753a9b26e27ae93`.
It contains 45 dense segments, three separately isolated DSpark segments, and
43 expert-layer segments. Expert records are packed layer -> expert ->
projection with 4 KiB alignment; no tensor is decoded, dequantized, or
requantized.

## Final container identity

| Artifact or total | Value |
|---|---|
| Segment files | 91 |
| Output records | 72,317 |
| Native payload bytes | 166,878,536,440 |
| Aligned segment bytes | 166,881,088,004 |
| Converted directory bytes including metadata/state | 166,973,631,781 |
| Manifest SHA-256 | `468d29fd3262af88ec4c31ef29a94e62631387475917ec7bdb4da9d0441e4d86` |
| Conversion-state SHA-256 | `385b9b3785fa36c66c0e57a6331d02163aeae2fc255518b31e99eef91b96a02c` |
| Final filesystem free bytes | 615,806,476,288 |

No hidden `*.partial` file remains. The final free-space result differs from
preflight by less than the small metadata, state, manifest, and unrelated
filesystem activity allowance.

## Independent native-byte validation

The validator reconstructs the source plan independently, matches every
manifest name/dtype/physical shape/source range/output range, rejects overlap,
trailing bytes, nonzero alignment padding, record hash drift, segment hash
drift, source/output identity changes, or metadata/state disagreement. It
hashes converted records in output order and source records in
shard/offset order, comparing the cryptographic record identities while
preserving sequential I/O. Each completed segment is atomically recorded with
source and output inode/size/mtime identities for safe resume.

The canonical run began at `2026-08-16T23:04:36.884905+00:00`, was
interrupted by a task continuation after 82 segments, resumed by verifying and
skipping those 82 identities, and completed at
`2026-08-17T13:22:32.159246+00:00`.

| Validation result | Value |
|---|---|
| Status | `complete` |
| Segments | 91/91 |
| Records | 72,317 |
| Source bytes compared | 166,878,536,440 |
| Payload bytes accepted | 166,878,536,440 |
| Segment bytes rehashed | 166,881,088,004 |
| Binding signature | `765b7c2dc2abf7d5941ecda6b769879d12c7eaf1d2ba50781242d6f6c3d8ee1c` |
| Evidence SHA-256 | `845ec7924c2c6f16468b5ee042adb06630328b73c66a9d75844e24b70a42b428` |

The signature binds the model/source paths, source marker/index, manifest,
conversion state, fixture mode, and these direct implementation dependencies:

| Dependency | SHA-256 |
|---|---|
| `convert_deepseek_v4.py` | `1f07779675959b3a29e80816953cb12839d501453a49205191fac18b2324e660` |
| `deepseek_v4_layout.py` | `a54a7698cddf19586efa7dfc6ac64931ac221424186ea45523ccb8689b98f56d` |
| `deepseek_v4_spec.py` | `ec214304940e536b73996bb04186d3055d1a2b79b3e9b0f46bc6d45f1dd79e21` |
| `runtime_env.py` | `e3d4f8204788879f6d889c73461ff9526a7e2e1615ee1887087ca504816c87df` |
| `validate_deepseek_v4_conversion.py` | `689e54265bbd3063b53478b1787873e82c0da7fb2a91fa315e48373ca80f5dd9` |

Two audit-history artifacts are intentionally retained. The random-source-order
attempt stopped at 51 segments to remove avoidable seeks and, after being
marked `interrupted`, hashes to
`2ff55e5a39124bb0da56806d28bf3058adcd65f313087fa89533cbae5f1d4360`.
The optimized complete run before dependency hashes were added is valid byte
evidence but not the canonical reproducibility record; it hashes to
`058d2c1bb42a8dcb77c2ad121dcd91adb43e8e092460a429dc51d61c62502fa3`.

## Descriptor-contract correction

The first compiled real-manifest check rejected
`mtp.2.confidence_head.proj.weight`: the official source and native manifest
are BF16 `[1,4352]`, while the generated Python and C expectations said F32.
The contract was corrected to BF16. The converter also now calls its previously
imported but unused `validate_records` function before any real output write,
so a future name, dtype, shape, category, duplicate, missing, or unexpected
record fails before conversion.

After the correction, the exact real plan retains the same plan hash and byte
projection. Python accepts all 72,317 descriptors, and the compiled check
reports:

~~~text
DeepSeek-V4 contract: tensors=72317 layers=43 dspark=3
~~~

Focused regression tests cover the BF16 descriptor, mandatory real-plan drift
rejection, conversion interruption/resume, completed-segment corruption,
validator resume, same-size corruption, dependency binding, and CLI
invocation.

## Remaining gates

Gate 11.0 is complete. The following remain prerequisites for replacement or
any performance claim:

- assemble and run the full 43-layer live mux engine over this real container;
- complete tokenizer, DSML tools, streaming, batching, cancellation, sessions,
  prefix reuse, and 16K/64K serving controls;
- pass the layer-streamed Python oracle, teacher-forced agreement, perplexity,
  coherence, tool-call, and long-context requirements;
- measure DeepSeek tok/s and TTFT, run DSpark on/off qualification, and compare
  against the preserved fresh Ornith397 control under the preregistered gate;
- promote defaults only if every correctness, quality, performance, and
  operational criterion passes.
