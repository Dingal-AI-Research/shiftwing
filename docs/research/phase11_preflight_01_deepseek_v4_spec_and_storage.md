# DeepSeek-V4 pinned specification and storage preflight

Date: 2026-08-14  
Phase: 11.0  
Disposition: pinned metadata inventory and storage gate pass; shard fetch in progress

## Decision boundary

DeepSeek-V4-Flash-0731 is a new architecture and must not be launched through
the Qwen/Ornith forward path. This preflight starts an independent engine track
and freezes promotion until correctness, quality, tok/s, and TTFT all pass.
The source repository is `deepseek-ai/DeepSeek-V4-Flash-0731`; model,
tokenizer, encoding, configuration, and reference implementation are pinned to
commit `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.

## Implemented controls

`c/tools/deepseek_v4_spec.py` is the common fail-closed contract for:

- 43 layers, hidden size 4096, 64 query heads, one KV head, head size 512;
- four-way mHC with 20 Sinkhorn iterations and epsilon `1e-6`;
- 256 routed experts, one shared expert, top-6 sqrt-softplus routing, and the
  first three hash-routed layers;
- top-512 indexing, a 128-token sliding window, and the exact 4×/128×
  compression schedule;
- native FP4 routed experts plus FP8 E4M3 dynamic dense execution with UE8M0
  scales and 128×128 weight blocks;
- 16,384 default and 65,536 validated maximum context;
- non-thinking/high-reasoning behavior and the deterministic `(0, 1)` and
  agent/tool `(1, 0.95)` sampler profiles.

`c/tools/preflight_deepseek_v4.py` is read-only unless an explicit JSON output
is requested. It records UTC time, command, source pin, repository commit and
dirty state, secret-redacted relevant environment, platform/RAM/GPU identity,
and the full storage arithmetic. It never performs cleanup.

`c/tools/convert_deepseek_v4.py` parses safetensors headers directly and copies
native tensor ranges without a floating-point round trip. It supports explicit
or fused routed-expert tensors, packs each layer in expert-major gate/up/down
order with aligned offsets, stores dense and DSpark records separately, and
commits one segment plus atomic JSON state at a time. The final manifest binds
every source range and output offset to per-record and per-segment SHA-256.
Resume rehashes every completed segment before trusting it.

## Executed storage preflight

Command:

```text
./.venv/bin/python c/tools/preflight_deepseek_v4.py \
  --source c/.deepseek-v4-flash-0731.source \
  --source-bytes 166888735421 \
  --container-bytes 167174674555 \
  --staging-bytes 5000000000 \
  --target c/deepseek-v4-flash-0731
```

Repository commit at execution:
`35d9ab86fa2b8f8acbf56c8438413dfb08f2a2a7`; the tree was dirty because the
Phase-11 implementation was under construction.

| Quantity | Bytes |
|---|---:|
| Filesystem capacity | 1,081,101,176,832 |
| Free before conversion | 526,791,335,936 |
| Source bytes still to download | 166,888,735,421 |
| Native-container upper bound | 167,174,674,555 |
| Planned staging high-water | 5,000,000,000 |
| Projected peak new storage | 339,063,409,976 |
| Projected free after conversion | 187,727,925,960 |
| Required free-space floor | 107,374,182,400 |

Result: **accepted**. The host reports 31,541,358,592 bytes RAM and an NVIDIA
GeForce RTX 5070 Ti with 16,303 MiB VRAM, compute capability 12.0, and driver
591.86. No Qwen or Ornith artifact was removed.

The first preflight omitted the not-yet-downloaded source copy and therefore
undercounted peak use. The command and table above are the corrected rerun;
the gate still passes with more than 74 GiB of margin over the required floor.

## Pinned metadata inventory

The metadata-first fetch resolved the requested revision to the exact commit
and downloaded the config, tokenizer/encoding/reference files, and safetensors
index. The independent inventory passes with 48 shards, 72,317 tensors, and
166,878,536,440 indexed payload bytes. Its alignment-aware container upper
bound is 167,174,674,555 bytes. The corrected records comprise 1,564 base
dense tensors, 66,048 base routed-expert tensors, 97 DSpark dense tensors, and
4,608 DSpark expert tensors across the three bundled draft layers. An initial
category-only audit counted the mandatory global `hc_head_*` triplet as DSpark;
the converter and inventory were corrected before real conversion so only the
`mtp.*` namespace is optional draft data.

The official checkpoint has 46 compression-ratio entries: 43 base layers plus
three zero-compression draft layers. This was verified from the pinned config
and is now part of the fail-closed specification. The 48-shard download is
running through atomic, hash-bound resume state; it does not authorize
conversion or promotion by itself.

## Tests

```text
./.venv/bin/python -m unittest c.tests.test_deepseek_v4_tooling -v
```

All nine tooling tests pass. They cover config drift,
context/reasoning/sampling rejection, environment secret redaction, fused and
explicit expert packing, interruption/resume with manifest binding,
completed-segment corruption rejection, metadata fetch resume, and the real
pinned layout analyzer. Four additional protocol tests cover the hash-pinned
official encoder, reasoning, tool rendering/parsing, and conservative
malformed-output recovery.

## Open gate

Gate 11.0 remains open until the pinned fetcher finishes all 48 shards and an
interrupted real download/conversion proves durable resume. The release
metadata and exact index inventory have passed. No performance or TTFT claim
is made at this boundary, and DeepSeek is not the runtime default.
