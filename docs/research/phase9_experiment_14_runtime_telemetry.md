# Phase 9 Experiment 14: Runtime Telemetry

## Abstract

This experiment completed the implemented Phase 9 mux telemetry contract.
The engine now publishes host hardware, expert-tier residency, route heat,
per-turn expert hits, and inclusive phase timings without changing the
request-control frames. The vendored HTTP gateway consumes the new records and
retains the older upstream `PROF` spelling for compatibility. Focused CPU and
CUDA mux tests, the complete 21-test C suite, and all 49 Python tests passed.

## Method

Five line-shaped records were added to the mux output:

- `HWINFO` reports online CPU cores, total/available RAM, the active CUDA
  device name, and total VRAM;
- `TIERS` counts experts represented in the CUDA, CPU, and storage tiers and
  reports their resident bytes;
- `EMAP` encodes one byte per layer/expert, combining a two-bit tier with a
  six-bit logarithmic route-heat bucket;
- `HITS` encodes one bit per layer/expert routed since the previous turn
  boundary;
- `PERF` reports the request ID, wall interval, expert-load time, expert
  matrix time, hybrid sequence-core time, and LM-head time.

`HWINFO`, `TIERS`, and `EMAP` are emitted after the `READY` sentinel.
`PERF`, `TIERS`, `EMAP`, and `HITS` are emitted immediately before each
`DONE`. The control contract remains only `DATA`, `DONE`, and `ERROR`;
telemetry remains advisory and unknown record kinds are ignored.

The profiler records a cumulative counter snapshot when decode starts and
subtracts it at request completion. This makes a single request's values
proper deltas. In a continuous batch, requests intentionally receive inclusive
interval values: shared batched work can appear in more than one request, so
the fields diagnose where latency occurred but must not be summed for billing.

The HTTP gateway's dispatcher was extended to parse `PERF` into its rolling
`/profile` state. It continues accepting the old `PROF` schema. Integration
tests validate row/column dimensions, encoded bitmap lengths, request IDs, and
gateway state after a real engine request.

## Results

The validation matrix was:

| Check | Result |
|---|---:|
| focused mux and HTTP gateway tests | 16/16 |
| complete C suite | 21/21 |
| complete Python suite | 49/49 |
| CUDA backend primitives on RTX 5070 Ti | pass |
| CUDA session continuation | token/state exact |
| CUDA mux and gateway telemetry tests | 7/7 |

The CUDA backend test retained its numerical bounds, including grouped MoE
exactness and maximum differences of `8.11e-06` for int8 and `1.31e-06` for
grouped int4. The CUDA session test remained exact and reported 93.88% combined
CPU/GPU expert-cache hits in the tiny stress configuration.

## Interpretation

The server can now distinguish three different runtime questions:

1. what hardware is available;
2. where expert weights currently reside and which experts were used;
3. where the elapsed interval was spent.

This is enough for the existing web/API observability endpoints without
coupling request parsing to a fixed telemetry version. It also supplies the
instrumentation needed to compare CPU-resident and future device-resident
continuous batching.

## Limitations

`ENTROPY`, per-device `GPUS`, live `REPIN`, and `.coli_usage` persistence are
not implemented. The current `TIERS` record describes the active cache view
rather than reconstructing every allocation owned by the CUDA driver. The
phase counters are CPU wall timers around operations; a later CUDA-resident
server should add device-event timings for asynchronous kernels and transfers.

No 35B or 397B web-session performance claim is made here. The 397B snapshot
conversion was still streaming while this experiment ran.

## Conclusion

The documented core mux telemetry (`HWINFO`, `TIERS`, `EMAP`, `HITS`, and
`PERF`) now works end-to-end on CPU and CUDA builds, is shape-validated by the
test suite, and is exposed through the HTTP gateway. The remaining Gate 9
work is device-resident continuous state, production-size web validation, and
the combined gateway/UI/CLI surface.
