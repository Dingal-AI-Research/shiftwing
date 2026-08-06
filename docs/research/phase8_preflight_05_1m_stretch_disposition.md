# Phase 8 Preflight 5: One-Million-Token Stretch Disposition

## Abstract

The roadmap listed a `-1M` YaRN model as an optional Phase 8 stretch. A
current official-registry inspection found no such Ornith release. The pinned
35B and 397B configurations both declare 262,144 maximum positions and no
`rope_scaling` block. Moreover, the full-attention KV allocation for one
1,048,576-token 397B slot is approximately 60 GiB in fp32 or 30 GiB in bf16,
before dense weights, expert caches, recurrent state, and runtime headroom.

The stretch is therefore closed as not applicable to the v0.1 artifact set:
there is neither an official checkpoint/configuration to implement against nor
a safe resource profile on the reference 29.4 GiB RAM / 16 GB GPU system.

## 1. Registry evidence

The publisher's current
[official model registry](https://huggingface.co/deepreinforce-ai/models)
lists Ornith 9B, 35B, and 397B releases plus FP8/GGUF packaging variants. It
does not list a `-1M`, YaRN, or other million-token checkpoint.

The already pinned immutable metadata reports:

| Model | `max_position_embeddings` | `rope_scaling` | full-attention layers |
|---|---:|---|---:|
| Ornith-1.0-35B-FP8 | 262,144 | absent | 10 |
| Ornith-1.0-397B-FP8 | 262,144 | absent | 15 |

Inventing YaRN parameters without a released configuration would be
architecture speculation rather than checkpoint support and would have no
upstream numerical oracle.

## 2. Resource bound

Only full-attention layers grow a token-indexed KV cache; DeltaNet recurrence
is fixed-size. The existing resource planner computes:

\[
\mathrm{KV/token}_{397B}
 = 15 \times 2 \times 2 \times 256 \times 4
 = 61{,}440\ \mathrm{bytes}
\]

where the factors are full-attention layers, key/value, KV heads, head
dimension, and fp32 bytes. At 1,048,576 tokens:

| Model | fp32 KV / slot | bf16 KV / slot |
|---|---:|---:|
| Ornith-35B | 40 GiB | 20 GiB |
| Ornith-397B | 60 GiB | 30 GiB |

The 397B bf16 minimum already exceeds host RAM once any model weights and
runtime allocations are included. The current loader clamps requested context
to the checkpoint's declared maximum, and `resource_plan.py`/`doctor.py`
account only full-attention KV layers and reject memory-unsafe slot profiles.

## 3. Decision

- Do not synthesize an unsupported YaRN configuration.
- Keep the existing full-attention-only KV sizing and runtime memory guards.
- Exclude the optional 1M stretch from the v0.1 model set because no official
  target exists and the reference system cannot hold a 397B 1M slot.
- Reopen this item only if the publisher releases an immutable checkpoint with
  explicit RoPE/YaRN parameters and a resource profile exists on target
  hardware.

This disposition does not weaken Gates 8 or 9: the released 262K models still
must pass their numerical, tool-use, server, and session acceptance tests.
