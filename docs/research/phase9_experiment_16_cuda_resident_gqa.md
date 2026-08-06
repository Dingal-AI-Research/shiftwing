# Phase 9 Experiment 16: CUDA-Resident Multi-Slot GQA

## Abstract

This experiment completed device ownership of the target mux's hybrid session
state. Full-attention layers now retain slot-major key/value caches on CUDA and
evaluate active rows with batched Q/K/V/O projections while permitting
different slots and positions. Together with Experiment 15, both DeltaNet
recurrence and GQA KV are device-resident. Exact mux output, cancellation,
two-slot isolation, and warm reload remain green.

## Method

The resident full-attention state uses:

```
key[slot, context, kv_head, head_dim]
value[slot, context, kv_head, head_dim]
```

`coli_cuda_gqa_slots_q4_f16` receives normalized hidden rows plus a row-to-slot
and row-to-position mapping. It:

1. batches grouped-int4 Q, K, and V projections;
2. applies zero-centered per-head normalization and partial RoPE at each row's
   position;
3. appends K/V to that row's slot-major cache;
4. attends over only that slot's prefix;
5. batches the int4 or int8 output projection.

The model-prefill handoff copies current CUDA KV directly into the selected
resident slot. A turn-boundary checkpoint downloads only the used prefix of one
slot; restore uploads that prefix. The normal decode loop performs neither
operation.

CUDA KV allocation is guarded before allocation. The runtime calculates the
complete slot-major requirement, retains 512 MiB of free-memory margin, and
falls back to host-resident KV if the plan does not fit. `CUDA_RESIDENT_KV=0`
also disables it explicitly.

Tests run the actual CUDA mux with both dense and fp16 activation paths enabled.
The engine summary must contain non-zero GDN and GQA call counts.

## Results

The complete validation matrix passed:

| Check | Result |
|---|---:|
| complete C suite | 21/21 |
| complete Python suite | 49/49 |
| CUDA backend and session tests | pass |
| CUDA resident mux integration | 5/5 |
| CUDA gateway integration | 2/2 |
| CPU resident whole-model continuation | token exact |
| warm process reload on CUDA mux | byte exact |

For two slots at 4,096 context and fp32 state, the resource planner reports:

| Model | target session state | total 16-GiB plan with 7-GiB expert cache |
|---|---:|---:|
| Qwen3.5-35B-A3B | 0.44 GiB | 11.27 GiB |
| Qwen3.5-397B-A17B | 0.84 GiB | 16.62 GiB |

The 35B plan fits. The 397B example exceeds 16 GiB by 0.62 GiB, but the expert
cache is a tunable seven-GiB term; reducing it permits state residency at the
cost of more expert traffic. The runtime guard makes this tradeoff explicit
instead of failing an allocation during service.

## Interpretation

The target model's persistent conversation state is now genuinely on the
device in the CUDA resident path. Host memory remains the serialization
boundary, not the per-token owner. This removes the state-copy architecture
that Experiment 8 showed would cost 0.44 GiB per active token for 35B and
0.84 GiB for 397B at the same context geometry.

The row mapping also preserves the mux scheduler's central property: requests
can have different history lengths without padding their cache ownership into
one shared sequence.

## Limitations

Activations still return to host memory between major sublayers, and per-row
attention/recurrence kernels are launched from host loops after their batched
projections. Therefore “state resident” is not yet “end-to-end graph
resident.” CUDA event-based transfer/kernel timing and production-model
throughput qualification remain necessary.

The current attention kernel uses dynamic shared memory proportional to the
attended prefix. Long-context production qualification must establish the
device-specific shared-memory limit or replace this score buffer with a tiled
online-softmax kernel.

The 397B conversion was still streaming during this experiment, so no 397B
latency or web-session result is claimed.

## Conclusion

Both forms of Qwen3.5 target session state—DeltaNet recurrence and GQA KV—now
remain on CUDA across mux decode steps, cross PCIe only at explicit
import/export boundaries, and preserve exact warm-session behavior. Gate 9's
remaining compute work is end-to-end activation residency and production-size
performance/cancellation qualification.
