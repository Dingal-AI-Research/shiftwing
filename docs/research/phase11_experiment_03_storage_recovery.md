# Phase 11 Experiment 3: Host-storage recovery

**Date:** 2026-08-16
**Workspace:** `/home/dinga/Projects/colib`

## Cause and safe disposition

Repeated WSL ext4 write failures during the pinned DeepSeek fetch were caused by
the Windows host volume reaching capacity, not by a model-format failure. At the
preflight boundary, `C:\` had 46,899,200 bytes free and the Ubuntu dynamic VHDX
was 665,518,604,288 bytes. The VHDX is not sparse. WSL refused safe sparse
conversion and offered only the explicit `--allow-unsafe` corruption-risk
override; that override was not used.

The recovery order follows the Phase 11 storage policy: preserve compact
evidence, remove Ornith35 before Ornith397, trim ext4, shut WSL down, and compact
the exact detached VHDX. Ornith397 remains protected until a fresh paired
control result has been independently recorded.

## Ornith35 reconstruction evidence

The removed container is reproducible from:

- source: `hf://deepreinforce-ai/Ornith-1.0-35B-FP8@1ab57ce0b44950e498a88756f40ad1ed4d0f30ca`;
- source fingerprint: `15281dc0352464f68ec93e805283a7c070aeb06f22622d975648381009cfe1d6`;
- conversion signature: `int4g128`, group size 128, int8 I/O and shared
  experts, no MTP;
- base completion: 16 output shards, 31,333 logical tensors, 93,277 physical
  tensors, and 19,081,810,684 payload bytes;
- q3 sidecar: `colib-routed-expert-int3-sidecar-v1`, complete, 160 files,
  40 layers, 92,160 tensors, and 13,098,786,560 data bytes.

The exact conversion commands, validation protocol, and results remain in
`phase8_experiment_06_ornith35_conversion.md` and
`phase8_experiment_09_ornith35_q3_control.md`. The actual compact local
metadata was moved, without rewriting, to
`c/bench/ornith35_reconstruction/` before removing weights.

| preserved file | bytes | SHA-256 |
|---|---:|---|
| `.conversion-state.json` | 4,874,201 | `86308551c2231f842f7184b62abe30b29a12df4df799ff173ace0423ad6cea52` |
| `chat_template.jinja` | 7,536 | `182e77dd83bd8e9ca818b240b82e28f243762cd5dda32e6eef327df7b1cd107e` |
| `config.json` | 4,927 | `3d37a307f3e7d62f57e0b4da97b99f580a3a042b5cf5f5d3147a2a05745e1a28` |
| `expert-q3.json` | 47,668 | `b097358eb8a499f1525cdcf43cb052599fd0a1df724ce715009be7eb1ab0b869` |
| `expert_map.bin` | 40,980 | `1d8783d88e744aeb2099644fb69d2b2981ee64a796ece91bda37b725419d7e6f` |
| `generation_config.json` | 213 | `896eeba60a1195e034dc6b9a7bbddd64b6174992348f5dc7d75daaa93d0bdd45` |
| `model.safetensors.index.json` | 10,130,288 | `a5d79ae6440e3cb45b5eabca02134c6081a38e2a56086bc30484856cbc85d313` |
| `quantization.json` | 522 | `b9b54701d38c0ea2836c9863f320f0f61ce4e70eac8cf7444babe146c0b6115d` |
| `tensor_inventory.json` | 4,431,077 | `d77885b96a3c37eeac425c64663ddccfe9c59d52c56f59d146ceb23cd802e57c` |
| `tokenizer.json` | 19,989,325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | 1,165 | `792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b` |
| `vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

The separate official numerical reference is also reproducible:
`deepreinforce-ai/Ornith-1.0-35B-GGUF`,
`ornith-1.0-35b-Q4_K_M.gguf`, 21,166,757,760 bytes, SHA-256
`ff25291b2599fb927a835e624d2b3540106af61761c3fa57ac4264046dbec002`.
Its exact download command is retained in
`phase8_preflight_04_official_gguf_reference.md`.

## Acceptance boundary

This operation removed only regenerable Ornith35 model weights. Tiny Qwen
fixtures, source, tests, JSON qualification evidence, research reports,
DeepSeek source shards, and Ornith397 remain.

## Recovery result

- Converted Ornith35 target removed:
  `/home/dinga/Projects/colib/c/ornith35`, 32,193,227,292 bytes after compact
  metadata was moved out.
- Official GGUF target removed:
  `/home/dinga/Projects/colib/c/reference/ornith-1.0-35b-Q4_K_M.gguf`,
  21,166,757,760 bytes, after its SHA-256 was independently reverified.
- Root ext4 trim: 424,089,743,360 bytes (395 GiB).
- Detached VHDX before/after:
  665,566,838,784 / 611,744,481,280 bytes.
- Host bytes reclaimed: 53,822,357,504.
- Host `C:\` free after compaction: 53,990,948,864 bytes.
- Ext4 free after restart: 419,620,769,792 bytes; the filesystem mounted
  read/write with no new ext4 or block-I/O error.

`diskpart` was not used because its administrator manifest blocked on an
interactive UAC prompt. The same documented operation was performed directly
with Microsoft's `OpenVirtualDisk`/`CompactVirtualDisk` API in the VHDX
owner's security context. The VHD path was resolved and checked exactly before
the call. Unsafe WSL sparse mode was never enabled.
