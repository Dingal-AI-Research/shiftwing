# Phase 9 Experiment 10: Resident Batched GQA Rows

## Abstract

This experiment added the full-attention counterpart to the resident Gated
DeltaNet batch seam. `attn_forward_slot_batch` accepts active rows at different
positions and a separate GQA key/value cache slice for every slot. It batches
Q, K, V, and output projections, then performs each row's normalization,
partial RoPE, causal attention, gating, and KV update without serializing
session state. Three deterministic rows matched sequential evaluation within
`6.98e-10`; every key and value cache element matched exactly.

## Method

The primitive uses the layout

```
input       [active_rows, hidden]
position    [active_rows]
key/value   [active_rows, max_seq, kv_heads, head_dim]
output      [active_rows, hidden]
```

Q/K/V and output matrices use `qmat_mul_batch`, retaining the same per-row dot
reduction as the validated quantized path. Q and K receive the model's
zero-centered per-head RMS normalization and partial split-half RoPE at each
row's own position. Attention addresses only that row's cache slice.

The regression loaded the tiny int4 model, initialized three distinct cache
slices and inputs, and evaluated positions 3, 5, and 7 together. Each row was
then rerun through the original sequential `attn_forward` after restoring its
initial cache.

## Results

```
resident GQA slot batch: rows=3 output=6.98e-10 key=0 value=0
```

The output deviation is far below the Phase 2 activation tolerance, and both KV
caches are byte-equivalent as fp32 arrays. The CPU suite now contains 19 C
tests.

## Interpretation

Both attention families now have a tested resident-row representation. This
removes the principal mathematical uncertainty from the CPU continuous-batch
driver: rows may have different positions while sharing immutable weights.
The very small non-zero output difference comes from the batched projection
execution context and does not alter state or token-level correctness in this
fixture.

## Limitations

The seam is not yet a complete transformer layer. Residual updates,
normalization, routing, routed/shared MoE, final normalization, and the LM head
still use single-row ownership. The CUDA backend also retains one state/cache
per model layer. No end-to-end server speedup is claimed.

## Next experiment

Experiment 11 subsequently composed the resident attention seams with
residual/norm and grouped MoE operations. Whole-model state ownership and a
token-for-token driver comparison are still required before `run_serve_mux`
can stop using `SessionState` between tokens.
