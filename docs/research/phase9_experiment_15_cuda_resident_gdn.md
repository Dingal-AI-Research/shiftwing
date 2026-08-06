# Phase 9 Experiment 15: CUDA-Resident Multi-Slot GDN

## Abstract

This experiment moved the first authoritative mux session state from CPU RAM
to CUDA memory. Every linear-attention layer now owns slot-major convolution
tails and DeltaNet recurrent matrices on the device. A new CUDA operation
projects all active rows together and updates each row's independently selected
slot and position. CPU and CUDA mux output remains identical, warm checkpoints
still reload exactly, and the complete 21 C and 49 Python tests pass.

## Method

The resident layer state was extended with device allocations:

```
conv[slot, kernel, channels]
gdn[slot, value_head, key_dim, value_dim]
```

The new `coli_cuda_gdn_slots_q4_f16` operation receives an active-row list:

```
row -> (slot[row], position[row], normalized_hidden[row])
```

It performs QKV and Z grouped-int4 matrix multiplication for the complete
batch, then indexes the appropriate circular convolution tail and recurrent
matrix for each row. Output projection is again performed as a batch. Unlike
the earlier block verifier, rows do not need consecutive positions and do not
share recurrence.

Model-to-resident import uses device-to-device copies when chunked prefill left
the single-model CUDA state authoritative. Session export downloads only the
selected slot at a turn boundary. Session import uploads only that slot. Thus
normal decode no longer copies GDN recurrence across PCIe, while persistence
keeps its established exact state format.

The CUDA mux test was run with:

```
COLI_CUDA=1 CUDA_DENSE=1 CUDA_F16=1 CUDA_EXPERTS=1
SERVE_BATCH=1 SERVE_RESIDENT=1
```

It additionally requires the process summary to report a non-zero GDN kernel
count, preventing an accidental CPU fallback from satisfying the semantic
tests.

## Results

All five mux integration tests passed with the resident CUDA GDN operation
active. They cover byte-counted streaming, cancellation/reuse, two active
slots, equality with the sequential snapshot reference, and exact warm reload.

The wider validation remained:

| Check | Result |
|---|---:|
| complete C suite | 21/21 |
| complete Python suite | 49/49 |
| CUDA backend primitives | pass |
| CUDA session continuation | exact |
| CUDA resident mux integration | 5/5 |

The standalone CUDA numerical bounds and the tiny session's 93.88% tier hit
rate were unchanged.

## Interpretation

The principal architectural result is ownership, not merely faster matrix
multiplication. Before this experiment, the CUDA build could accelerate
projections while the recurrent state still lived in host arrays. It now
persists across turns in device memory and crosses PCIe only for explicit
checkpoint import/export.

This removes the larger and more frequently updated half of hybrid session
state from the decode-time host path. The remaining full-attention layers still
use CPU-owned slot-major KV in the resident mux path.

## Limitations

Kernel launches for the per-row recurrence are currently issued from a host
loop on one CUDA stream. The expensive projections are batched, but a later
kernel can fuse row dispatch. Inputs and outputs still cross the host boundary
between transformer sublayers. GQA KV and end-to-end activations are not yet
device-resident, so this experiment is an incremental Gate 9 result rather than
a final throughput qualification.

The test model is intentionally tiny. Production-size memory use and latency
must be measured on the converted 35B and 397B snapshots.

## Conclusion

Independent GDN decode slots now retain their authoritative recurrence on
CUDA, survive exact checkpoint round trips, and preserve mux semantics. The
next state-residency step is slot-major CUDA GQA KV.
