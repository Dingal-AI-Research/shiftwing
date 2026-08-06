# Phase 9 Experiment 12: Whole-Model Resident Batch

## Abstract

This experiment extended the resident layer operation through the complete
target model and connected it to the multiplexed server. `ResidentBatchState`
allocates recurrent, convolution, GQA KV, hidden, and logit storage once for
each slot. `resident_forward_tokens` evaluates arbitrary active slot IDs through
embedding, every hybrid layer, final normalization, and a batched language-model
head without saving or restoring session snapshots. Two slots remained
token-exact for eight decode steps, and the opt-in mux emitted the same token
bytes and request lifecycle as the snapshot reference.

## Method

Two prompts were prefetched independently with the established chunked prefill
path. Their resulting target states were imported once into slots 0 and 1.
Seven subsequent token advances were evaluated as a two-row batch. At every
step, the next token was selected independently from the slot's own logits.

The experiment then recomputed both conversations with the original sequential
model and compared:

- all eight greedy token IDs;
- final normalized hidden vectors and vocabulary logits;
- every GDN recurrent and convolution element;
- every used full-attention key/value element;
- final positions.

Server integration was enabled only by `SERVE_RESIDENT=1`. The same two-request
wire stream was run through resident and snapshot modes. Tests compared all
`DATA` bytes, errors, request IDs, and `DONE` ordering while ignoring timing
fields.

## Results

```
resident whole-model batch: rows=2 tokens=exact
hidden=7.15e-07 logits=2.09e-07 recurrent=2.46e-07 kv=9.54e-07
```

The resident and snapshot mux modes emitted identical semantic wire events.
Queued cancellation, slot reuse, and two simultaneously active request IDs also
passed with resident mode enabled.

The complete regression state is:

| Suite | Result |
|---|---:|
| C tests | 21/21 pass |
| Python tests | 48/48 pass |
| Standalone CUDA kernels | pass |
| CUDA session save/restore, disk reload, alternating slots, prefix extension | exact |

## Interpretation

This closes the CPU algorithmic gap identified in Experiment 8. Decode state is
now resident in the structure used by computation, so state traffic no longer
scales as one complete save plus restore per emitted token. Batched projections
also reuse immutable weights across active conversations.

The observed state differences are bounded floating-point ordering effects from
batched execution and are below the existing activation tolerances. They did
not change any greedy token in the tested trajectories.

## Limitations

The path is opt-in and target-only. Prefill remains serial, sampling is still
greedy-only, and MTP session state is not represented. Most importantly, CUDA
still owns only one recurrent/KV state per layer. A CUDA build may accelerate
some batched matrices, but this experiment does not provide device-resident
multi-slot state or a throughput claim.

The tiny model verifies ownership and arithmetic but cannot predict large-model
expert-cache contention. Real 35B benchmarking must measure batch throughput,
per-request latency, cache misses, and output parity before resident mode
becomes the default.

## Next experiment

Add a slot stride and active-row index to CUDA GDN and GQA state kernels, retain
all slot state on device, and compare the device-resident two-slot trajectory
against this CPU resident oracle. In parallel, complete mux telemetry so the
server can report batch utilization and tier behavior during that benchmark.
