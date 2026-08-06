# Phase 9 Experiment 8: Session-Switch Cost and Reusable Buffers

## Abstract

The functional mux stores each request as a `SessionState`, restores it before a
token, and saves it afterward. This experiment quantified the amount of state
handled by that reference algorithm and removed avoidable allocation work.
At a 4,096-token context, one save-plus-restore transfers 0.44 GiB for
Qwen3.5-35B and 0.84 GiB for Qwen3.5-397B per active emitted token. Therefore,
sequential snapshot switching cannot be the production continuous-batching
architecture. Reusable geometrically grown buffers remove allocator churn while
preserving exact CPU and CUDA continuation results.

## Research question

Is the functional server's state-switch method a plausible implementation for
real multi-slot decoding, or only a semantic reference?

## State model

A target-model session contains:

- the DeltaNet recurrent matrices for every linear-attention layer;
- every causal-convolution tail;
- the full-attention key/value cache up to the current position;
- the final normalized hidden vector used to reconstruct next-token logits.

For a slot at position \(T\), the snapshot size is

\[
B(T)=B_{\mathrm{GDN}}+B_{\mathrm{conv}}+4H+T B_{\mathrm{KV/token}}.
\]

The sequential reference performs one restore and one save per emitted token,
so its state traffic is approximately \(2B(T)\) for each active slot-token. On
CUDA, authoritative recurrent/KV buffers also cross the device boundary during
these operations.

## Method

`resource_plan.py` was extended to calculate the last-hidden vector, the
snapshot size at the selected context, and the save-plus-restore byte count. The
calculation used the converted snapshots' real `config.json`, `CTX=4096`,
target-only operation, and fp32 KV, matching the current server reference.

`SessionState` was then changed from exact-size allocation on every save to
capacity-tracked buffers that grow geometrically. Per-slot vocabulary logits
were also retained. A C regression saved an advancing session twice and checked
that the recurrent, KV, and hidden buffer addresses did not change while exact
continuation tests still passed.

## Results

| Quantity at 4K context | Qwen3.5-35B | Qwen3.5-397B |
|---|---:|---:|
| GDN recurrent state | 60.00 MiB | 180.00 MiB |
| Convolution tails | 3.75 MiB | 8.44 MiB |
| KV bytes per token | 40,960 B | 61,440 B |
| Complete slot snapshot | 223.76 MiB | 428.45 MiB |
| Restore + save per emitted token | 447.52 MiB | 856.91 MiB |
| Reported binary-unit traffic | 0.44 GiB | 0.84 GiB |

The tiny regression reported stable buffer addresses
(`recurrent-cap=65,536`, `kv-cap=4,096` floats for the tested state). All 17 C
tests and all 47 Python tests passed. CUDA kernel and session checks also
remained exact.

## Interpretation

Removing allocation is necessary but insufficient. At the measured 35B decode
rate of roughly 30 tokens/s, a full-context single-slot reference would request
about 13 GiB/s of snapshot copying before accounting for model computation. The
397B path would request about 1.7 GiB/s even at only 2 tokens/s. Additional
active slots multiply this traffic. These are calculated upper-context costs,
not direct end-to-end bandwidth measurements, but they are large enough to
reject the architecture without a benchmark.

The production design must keep each slot's recurrent and KV state resident in
its final layout and pass a slot/row index to batched kernels. Host snapshots
should be created only for persistence, eviction, or migration—not at every
token.

## Threats to validity

The calculation uses fp32 KV because that is the conservative default.
`KV16=1` reduces the KV term but not the fp32 GDN recurrent matrices. Short
conversations transfer less KV than the 4K row, while longer contexts transfer
more. The estimate also does not include allocator metadata, synchronization,
or the extra host/device copy paths, so it should be read as payload geometry
rather than a timing prediction.

## Conclusion and next experiment

The sequential mux remains a correct oracle for request lifecycle and exact
state ownership, but it is not a viable performance implementation. The next
kernel experiment should introduce a slot dimension into GDN recurrent state
and full-attention KV, execute at least two active rows without snapshot copies,
and compare every output token and state element with this reference.
