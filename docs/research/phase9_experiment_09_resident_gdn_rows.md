# Phase 9 Experiment 9: Resident Batched GDN Rows

## Abstract

Experiment 8 showed that save/restore switching moves 0.44–0.84 GiB per
full-context active token for the target models. This experiment introduced the
first layer primitive that avoids those copies. `gdn_forward_slot_batch`
accepts multiple independent decode rows, their positions, and row-major
convolution/recurrent states. It batches the layer's matrix projections and
updates each row's state in place. For three slots at different positions,
outputs, recurrent matrices, and convolution tails matched independent
sequential evaluation exactly.

## Hypothesis

If slot identity is represented as a leading tensor dimension, Gated DeltaNet
rows can share weight reads and matrix operations without sharing or copying
their sequence state.

## Method

The new CPU primitive receives:

- an input matrix `[active_rows, hidden]`;
- one position per row;
- convolution state
  `[active_rows, kernel, qkv_channels]`;
- DeltaNet state
  `[active_rows, value_heads, key_dim, value_dim]`.

The A, B, QKV, Z, and output matrices use the existing `qmat_mul_batch`
implementation. Convolution and the recurrent update operate independently on
each row and may execute in parallel. No session serializer or model-global
state is used.

The regression loaded `qwen_tiny_i4`, selected a real linear-attention layer,
and constructed three deterministic inputs with positions 3, 5, and 7 plus
non-zero initial recurrent and ring-buffer states. It then:

1. evaluated all three rows with the resident batch primitive;
2. restored each row's same initial state into the original sequential layer;
3. evaluated the rows individually;
4. compared every output, recurrent-state, and convolution-state element.

## Results

```
resident GDN slot batch: rows=3 output=0 state=0 conv=0
```

All differences were exactly zero for the int4 tiny layer. The new regression
raises the CPU suite to 18 C tests; the 47 Python tests remain green.

## Interpretation

This result establishes the correct ownership boundary for continuous
batching: weights belong to the model, while position and recurrent/KV state
belong to rows. It also reuses the already validated batch matrix kernel, so
the arithmetic order of each row is unchanged.

The experiment does not yet demonstrate an end-to-end throughput gain. GDN is
only one component of a hybrid layer, and the reference server still snapshots
state around whole-token forwards. A performance claim would be premature until
full-attention KV, MoE routing/expert execution, residuals, and the LM head all
accept the same active-row representation.

## Limitations and next step

The primitive is CPU-only. CUDA currently stores one recurrent state per layer,
so its GDN kernel needs a slot stride and row-index array. Full-attention KV must
likewise change from `[position,...]` to `[slot,position,...]`. The next
experiment should implement resident batched GQA/partial-RoPE on CPU, compare it
against independent sequential rows, and then connect both attention types to a
single hybrid-layer batch driver.
