# Phase 8 Preflight 1: Ornith Registry, FP8 Layout, and Agent Protocol

**Project:** colib
**Models:** Ornith-1.0-35B-FP8 and Ornith-1.0-397B-FP8
**Publisher:** deepreinforce-ai
**Report date:** 25 July 2026
**Status:** metadata and protocol mechanisms validated; numerical model gates open

## Abstract

Phase 8 initially assumed that Ornith would be an architecture-identical Qwen
checkpoint with a Hermes-style JSON tool-call envelope. Read-only inspection
showed that the architecture statement is correct but two operational
assumptions are not. First, the official FP8 checkpoints use the
`compressed-tensors` per-output-channel scale format rather than Qwen397's
128-by-128 inverse-scale grid. Second, the official chat templates request
Qwen3 XML function calls rather than JSON bodies, and the 35B and 397B
templates are not byte-identical.

The converter was extended with per-channel FP8 pairing, metadata validation,
and dequantization. Metadata-only dry runs validate 30,880 pairs for 35B and
92,400 pairs for 397B with no orphan, cross-shard, or shape errors. The
estimated no-MTP outputs are 19,081,810,684 bytes and 212,634,789,241 bytes,
respectively. An explicit `colib_model_family` marker is now written during
official Hub conversion because the upstream `model_type` intentionally
remains `qwen3_5_moe`.

Snapshot-specific Jinja rendering and a Qwen3-XML-to-OpenAI parser were added.
The C runtime now reads both end tokens from `generation_config.json`, stopping
on IDs 248046 and 248044. Protocol and converter unit tests pass. Full
conversion, numerical comparison, and tool-call generation remain required
before Gate 8 can close.

## 1. Research questions

1. Can the existing Qwen converter decode the official Ornith weight format
   without loading an entire model?
2. Is there an unambiguous local signal that distinguishes Ornith from its
   Qwen base architecture?
3. What exact text and tool protocol did the released models train against?
4. Which token IDs terminate generation?

## 2. Sources and controls

The inspected immutable revisions were:

| Model | Revision |
|---|---|
| `deepreinforce-ai/Ornith-1.0-35B-FP8` | `1ab57ce0b44950e498a88756f40ad1ed4d0f30ca` |
| `deepreinforce-ai/Ornith-1.0-397B-FP8` | `8b61f97a8512d9d01bff1a9625c9a16730e115bb` |

The study used the official
[35B model card](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-FP8),
[397B config](https://huggingface.co/deepreinforce-ai/Ornith-1.0-397B-FP8/blob/main/config.json),
checkpoint indices, `generation_config.json`, tokenizer configuration, and
`chat_template.jinja`. No weight shard was required for this metadata study.

Transformers 5.14.1 remained the project reference. The Ornith-35B model card
requires Transformers 5.8.1 or newer, so the pinned reference satisfies the
stated minimum.

## 3. FP8 format study

Both checkpoints declare:

```text
quant_method = compressed-tensors
format = float-quantized
weight strategy = channel
weight bits = 8, type = float, symmetric = true
```

Each compressed matrix has a sibling called `weight_scale`. This differs from
Qwen397, where a sibling `weight_scale_inv` describes a two-dimensional
128-by-128 block grid. The compressed-tensors 0.17.1 source defines
dequantization as:

\[
W_{\mathrm{decoded}} =
\left(W_{\mathrm{FP8}} - Z\right) S .
\]

The official configuration is symmetric, so no zero point \(Z\) is stored.
For a matrix with shape `[output, input]`, the observed scale shape is
`[output, 1]`. The converter therefore multiplies each FP8 output row by its
own floating-point scale before applying colib's normal int8 or grouped-int4
quantizer.

Metadata validation produced:

| Property | Ornith-35B | Ornith-397B |
|---|---:|---:|
| Source shards | 16 | 122 |
| Indexed source bytes | 37,667,035,872 | 405,108,696,032 |
| FP8 weight/scale pairs | 30,880 | 92,400 |
| Orphan scales | 0 | 0 |
| Cross-shard pairs | 0 | 0 |
| Valid scale shapes | 30,880 | 92,400 |
| Invalid scale shapes | 0 | 0 |
| Kept no-MTP text tensors | 31,333 | 93,078 |
| Estimated output bytes | 19,081,810,684 | 212,634,789,241 |

The 397B output estimate and logical tensor count exactly match the Qwen397
plan. Thus, after FP8 dequantization, both families enter the same container
and runtime architecture.

```mermaid
flowchart LR
    A["Official Ornith FP8 weight"] --> B["Per-output-channel scale"]
    B --> C["Decode to fp32 working tile"]
    C --> D["Shared colib quantizer"]
    D --> E["Int8 or grouped-int4 container"]
    E --> F["Existing Qwen3.5 C and CUDA kernels"]
```

## 4. Registry design

Architecture keys cannot identify the fine-tune:

```text
architectures = Qwen3_5MoeForConditionalGeneration
model_type = qwen3_5_moe
```

Guessing from end tokens is also unsafe because the Qwen generation
configuration uses the same pair. The converter already knows the immutable
Hub repository at the point of conversion, so it records:

```json
{
  "colib_model_family": "ornith-1.0",
  "colib_source_repo": "deepreinforce-ai/Ornith-1.0-…",
  "colib_source_revision": "…"
}
```

Unknown configuration fields are ignored by Transformers, while colib gains a
deterministic registry key. An ordinary Qwen conversion is left byte-for-byte
unchanged by this annotation function.

## 5. Stop-token study

Both official generation configurations list:

```json
"eos_token_id": [248046, 248044],
"pad_token_id": 248044
```

ID 248046 represents the normal end of a ChatML message and ID 248044 is the
end-of-text/padding sentinel. The previous C configuration retained only one
integer and usually selected 248044 from `text_config`. This could continue
generation past a valid message terminator. The loader now treats the first
two generation-config IDs as valid stops in normal, MTP, warm-up, and
token-prefix modes.

## 6. Chat and tool protocol

The official templates have SHA-256 values:

| Template | SHA-256 |
|---|---|
| Ornith-35B | `182e77dd83bd8e9ca818b240b82e28f243762cd5dda32e6eef327df7b1cd107e` |
| Ornith-397B | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |

The 397B template changes how earlier assistant reasoning is replayed and how
mapping/sequence argument values are serialized. Applying one hard-coded
template to both models would therefore be unjustified. `render_chat` loads the
template stored beside each converted snapshot and renders it through the
pinned Transformers Jinja environment.

Tool declarations are placed in a system message. The response format is:

```text
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

This is Qwen3 XML, not Hermes JSON. The parser:

1. separates an initial `<think>…</think>` block;
2. extracts one or more function blocks;
3. converts parameter values that are valid JSON into structured values;
4. retains plain text values as strings;
5. emits OpenAI-style `tool_calls`;
6. preserves a forbidden trailing suffix rather than silently losing output.

The model card independently recommends vLLM's `qwen3_xml` tool parser and
`qwen3` reasoning parser, which agrees with the template inspection.

## 7. Results and limitations

Five new focused tests cover family annotation, registry stops, snapshot Jinja
rendering, structured XML arguments, reasoning separation, and malformed
suffix preservation. The official 35B and 397B templates both rendered a
tool-enabled user request and opened the expected assistant `<think>` block.

This preflight does not show that a converted Ornith weight produces correct
logits. A real FP8 tensor must still be compared with the upstream
compressed-tensors dequantizer, followed by a complete 35B conversion and the
Phase-4 statistical checks. Tool parsing is structurally correct, but an
end-to-end generated function call is still required.

## 8. Engineering decision

Retain the explicit registry marker, dual stop IDs, per-snapshot renderer,
Qwen3 XML parser, and compressed-tensors FP8 implementation. Revise the Phase
8 roadmap to remove the Hermes-JSON assumption. Do not close either numerical
or agentic portions of Gate 8 until the official 35B model passes them.
