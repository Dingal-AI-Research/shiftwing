# Phase 9 Experiment 21: Resident CUDA Activation Graph

## Abstract

This experiment extended CUDA residency from persistent recurrent/KV state to
the token activation graph used by the multi-request mux. The residual stream,
both RMS normalizations, Gated DeltaNet or grouped-query attention, routed
experts, shared expert, residual additions, final normalization, and language
model head now remain on the device for eligible int4/int8 snapshots. The CPU
receives only router logits, which are small control metadata rather than
hidden-state vectors. The embedding row is uploaded once at the start of a
resident step; the final normalized hidden row and vocabulary logits are
downloaded once as model outputs. Tiny-model CUDA mux output remains
byte-identical to the snapshot reference. Production-model warm throughput remains the final Gate-9
measurement and is not inferred from this functional experiment.

## Research Question

Can the resident mux execute successive hybrid layers without transferring the
full hidden activation to and from host memory at each GDN/GQA and MoE
boundary, while preserving the established token and session semantics?

## Method

The implementation added three classes of CUDA primitive:

1. batched zero-centered RMS normalization, fp32 matrix multiplication, and
   elementwise sigmoid for the exact router and DeltaNet gate matrices;
2. device-I/O forms of the resident slot-major GDN and GQA transactions;
3. device-scale shared-int8 expert execution, complementing the existing
   grouped-int4 routed and shared expert kernels.

`ResidentBatchState` now owns device buffers for the residual stream,
normalized rows, attention output, MoE output, fp16 expert input, DeltaNet
gates, router logits, and shared-expert scales. Each layer performs:

```text
device residual
  -> device input RMSNorm
  -> device GDN or GQA using slot-major device state
  -> device residual add
  -> device post-attention RMSNorm
  -> device router logits
  -> CPU top-k selection and expert-cache management
  -> device routed + shared experts
  -> device residual add
  -> next layer
  -> device final RMSNorm and LM head
  -> final hidden row and vocabulary logits
```

Only `batch × experts × sizeof(float)` router logits cross device-to-host per
layer. Expert identifiers, weights, and pointer tables return as small kernel
setup metadata. If the active routes cannot fit the configured expert cache,
the code explicitly downloads the normalized rows and uses the established CPU
fallback; it does not silently consume an uninitialized host buffer.

The backend unit test compares the new batched normalization, fp32 GEMM, and
sigmoid kernels with scalar CPU references. The mux tests run the quantized
tiny snapshot with CUDA dense/expert execution and compare resident output with
the independent snapshot-mode path. A two-slot manual run additionally
requires grouped device transactions and prints transfer counters.

## Results

The backend arithmetic test passed with:

| Primitive | Maximum absolute difference |
|---|---:|
| batched RMSNorm | `1.07e-6` |
| fp32 GEMM | `3.43e-5` |
| sigmoid | `5.96e-8` |
| shared-int8 device-scale path | `0` |

The focused two-slot parity tests passed for five-token resident-versus-snapshot
generation and independent request IDs. After final device-MoE and output-head
integration, the complete six-test CUDA mux suite passed in 5.92 seconds.

A two-slot, two-token manual run executed ten resident device-MoE transactions
covering 40 routed expert uses and 34 unique expert uses. This demonstrates
that the route batch reached the grouped CUDA kernel rather than falling back
to the prior host MoE path. Its transfer counters were:

```text
[CUDA_RESIDENT] layers=10 device-moe=10 host-moe=0
activation-h2d=2048 activation-d2h=2048 logits-d2h=8192
router-d2h=640 bytes
```

The 2,048-byte activation upload and download are exactly one two-row,
64-component fp32 boundary at model input and output. They do not scale with
the five layer count. Vocabulary logits are reported separately because they
are model output, not a hidden activation. The 640 router bytes are
`5 layers × 2 rows × 8 experts × 4 bytes`.

The production Qwen3.5-35B-A3B snapshot then exercised the converter's actual
fp32 router, int4 routed experts, int8 shared expert, fp32 shared gate, int8 LM
head, and 2,048-component hidden state. With a deliberately small 2 GiB host
expert cache, both one-token requests completed without protocol errors:

```text
[CUDA_BATCH] transactions=40 routes=640 unique=462 overlap=27.81%
[CUDA_RESIDENT] layers=40 device-moe=40 host-moe=0
activation-h2d=16384 activation-d2h=16384 logits-d2h=1986560
router-d2h=81920 bytes
```

These dimensions are exact: two hidden rows require 16,384 bytes, the router
metadata is `40 × 2 × 256 × 4 = 81,920` bytes, and two 248,320-entry fp32
vocabulary rows require 1,986,560 bytes. The host tier incurred 596 misses and
4.619 GiB of direct reads, yet the activation graph did not fall back to host
MoE. The two generated token byte strings were exactly equal before and after
moving final RMSNorm/LM-head execution onto CUDA.

The runtime now emits a `CUDA_RESIDENT` line containing layer count,
device-versus-host MoE transactions, full-activation H2D/D2H bytes, and router
metadata bytes. The mux publishes the same values as a `RESIDENT` advisory
line; the gateway parses it and Gate 7 requires positive device activity with
zero host-MoE fallback. These counters distinguish graph residency from merely
retaining recurrent state.

## Interpretation

Persistent state residency and activation residency solve different transfer
problems. Experiments 15 and 16 removed repeated copies of GDN recurrence and
GQA KV, but every major sublayer still treated host memory as the owner of the
current token vector. This experiment makes the device residual stream the
owner across layers. CPU routing remains appropriate because top-k selection
and tiered expert-cache decisions are control operations; downloading a few
hundred logits is materially different from downloading and re-uploading the
full hidden representation.

The final normalized hidden row is retained on the host for session
checkpoint/state semantics, and logits must reach the scheduler for sampling.
Those are output transfers rather than intermediate compute ownership. The
final RMSNorm and LM head themselves execute on CUDA.

## Decision

- Accept the resident activation graph as the implementation closure of the
  Gate-9 “end-to-end CUDA between layers” item.
- Retain CPU top-k and expert-tier control with router-logit-only downloads.
- Retain the explicit host fallback when a route set exceeds device-eligible
  cache capacity.
- Do not close Gate 9 until warm 35B and 397B continuous-batch throughput and
  cancellation measurements pass their production criteria.
- Use `CUDA_RESIDENT` counters in those runs to reject accidental host-MoE
  fallback.

## Limitations

The functional oracle is a small quantized model and cannot predict
production-size throughput. The CUDA path changes floating-point reduction
order within documented tolerances, so token parity is the primary
end-to-end invariant. Converter I/O was active during development; no timing
from this experiment is presented as an uncontended benchmark. Embedding
lookup remains host-owned at the input boundary, output hidden/logit rows are
downloaded for session/sampling semantics, and expert weight movement is
intentionally tiered rather than permanently device-resident.
