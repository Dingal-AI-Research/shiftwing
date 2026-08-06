# Phase 9 Experiment 11: Resident Hybrid Layer

## Abstract

The two resident attention seams were composed with normalization, residual
connections, and the existing grouped routed/shared MoE implementation. The new
`layer_forward_slot_batch` therefore evaluates one complete Qwen transformer
layer for multiple independent rows without session serialization. On two
deterministic GDN rows, the complete layer output differed from independent
sequential evaluation by at most `7.45e-09`; recurrent and convolution state
matched exactly.

## Method

For every active row, the driver performs:

1. zero-centered input RMS normalization;
2. resident GDN or resident GQA attention;
3. the attention residual;
4. zero-centered post-attention RMS normalization;
5. grouped routed and shared MoE evaluation across all active rows;
6. the MoE residual.

The test used a real int4 tiny-model GDN layer, two positions, different
non-zero recurrent/ring states, and deterministic hidden vectors. The batched
result was compared with the original `layer_forward_one`, which evaluates
each row separately through the same layer and expert weights.

## Results

```
resident hybrid layer batch: rows=2 output=7.45e-09 state=0 conv=0
```

The C suite now contains 20 tests. The difference is numerically negligible and
the state transition is exact.

## Interpretation

This is the first complete layer boundary suitable for a continuous-batch model
driver. It reuses grouped MoE work that was previously exercised only for
multi-token prefill/speculative verification. In a server batch, those rows now
represent different conversations instead.

## Remaining work

The complete model still stores layer state in the old single-sequence
structures. A resident model driver must allocate per-slot state once, iterate
all layers with `layer_forward_slot_batch`, apply final normalization and the LM
head as a batch, and advance each row's position. It must then be compared
token-for-token and state-for-state with the sequential mux before that mux can
switch implementations. CUDA needs equivalent slot-strided state and KV
kernels; the CPU result does not imply GPU throughput.
