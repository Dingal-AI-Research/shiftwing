# Phase 9 Experiment 19: CUDA Events and Live Continuous-Batch Admission

## Abstract

This experiment added request-scoped CUDA event telemetry and a repeatable
sequential-versus-concurrent mux benchmark. The first live simultaneous run
found a persistent-pipe deadlock that static byte-stream tests had hidden:
stdio read-ahead consumed a second `SUBMIT`, then `poll()` waited on an empty
file descriptor while the frame remained in the private `FILE` buffer.
Unbuffered mux stdin fixes the boundary, and two gateway threads now complete
on separate slots in one persistent engine.

Event profiling is numerically and structurally valid but intentionally
opt-in because enabling it increased the six-process CUDA mux qualification
from roughly 14 seconds to 111 seconds. A real 35B two-slot run remained
token-exact, but concurrent execution was 0.953-fold the sequential aggregate
throughput while the 397B converter caused 42–62 seconds of expert-storage time
per inclusive interval. This is a controlled negative result for the contended
cold-tier condition, not the warm Gate-9 throughput result.

## Research questions

1. Can the mux accept two `SUBMIT` frames arriving through a live persistent
   pipe and execute both slots without EOF?
2. Can CUDA work be measured with device events rather than inferred only from
   CPU wall clocks?
3. Does the current two-slot CUDA path improve aggregate throughput on 35B?

## Method

### Live-pipe admission

Two Python threads share the production `Engine`, synchronize at a barrier,
then call `generate` on slots zero and one. Unlike the existing integration
test, stdin remains open after both writes. Optional `COLI_ENGINE_TRACE=1`
records the submitted and received request IDs without dumping prompt data.

The failed trace was:

```
[engine->] SUBMIT 1 slot=1
[engine->] SUBMIT 2 slot=0
[engine<-] DATA 1 ...
[engine<-] DONE 1 ...
```

The C process then slept in `poll`, Python request 2 remained pending, and
there were no active scheduler rows. `mux_read_frame` uses stdio, so its first
read had buffered the second frame above the file-descriptor layer. The fix is
`setvbuf(stdin, NULL, _IONBF, 0)` on entry to `run_serve_mux`.

### CUDA events

The existing nine-event fused-MoE instrument was promoted to a public
snapshot API. `CUDA_PROFILE_STAGES=1` enables it. Each `PERF` record may append:

1. completed CUDA transaction count;
2. setup;
3. routed hidden projection;
4. routed down projection;
5. route reduction;
6. shared hidden projection;
7. shared scaling;
8. shared down projection;
9. device-to-host download.

Times are seconds on the wire. The gateway retains them as named profile
fields. The suffix is optional, so CPU builds and older parsers keep the
original nine-field record.

`PERF.wall_s` now starts before prompt prefill. A separate decode clock still
feeds the `DONE` token-rate statistic. Fresh prefill, resident hybrid core,
MoE, and LM-head paths now all contribute to the phase counters; previously
the resident path emitted zeros for compute phases.

The CUDA backend unit test enables events around one fused two-row routed plus
shared MoE transaction and requires exactly one completed transaction. Full
mux profiling remains opt-in rather than a CLI default.

### Batch harness

`c/tools/bench_serve_batch.py` starts a fresh engine for each requested mode,
performs two complete in-process warm-up passes by default, excludes startup
and warm-up from measured throughput, and records per-slot TTFT, elapsed time,
engine statistics, measured-only profiles, tiers, hardware, resident-graph
telemetry, output equality, and aggregate speedup. It can reverse mode order
in a separate run to control page cache bias and has an explicit worker
timeout that reports pending IDs.
For production 397B qualification it also accepts `--ram-gb` and explicit
host/device headroom values. This selects the runtime's byte-budgeted
`RAM_GB` path instead of treating the number as `EXPERT_RAM` experts per
layer, and records every tier control plus the immutable conversion-manifest
identity in the JSON artifact. CUDA acceptance rejects any host-MoE fallback.

The real-model trial used:

- Qwen3.5-35B-A3B int4/int8 container;
- RTX 5070 Ti, CUDA 12.9, fp16 activations;
- two slots, identical one-token prompts, two generated tokens per slot;
- 7 GiB CUDA expert budget;
- live 397B conversion left running as an explicit contention condition.

## Results

### Correctness and lifecycle

- The new live-pipe regression completes both two-token requests in 3.55 s.
- Tiny sequential and concurrent outputs are identical.
- Real 35B sequential and concurrent outputs are both `\n\n#` for both slots.
- CUDA kernel/session tests and all six mux tests remain exact.
- The complete post-change suite passes 21 C executables and 59 Python tests.

### Device-event reference transaction

| CUDA fused-MoE stage | Time (ms) |
|---|---:|
| setup | 0.042 |
| routed hidden | 0.157 |
| routed down | 0.091 |
| route reduction | 0.074 |
| shared hidden | 0.201 |
| shared scale | 0.132 |
| shared down | 0.103 |
| download | 0.148 |

The event totals are not interpreted as a production layer profile; the unit
shapes are deliberately small. They prove that event order, accumulation, and
snapshot transport execute.

### Tiny CPU mechanism run

| Mode | Elapsed (s) | Aggregate completion tok/s |
|---|---:|---:|
| sequential | 0.184 | 43.56 |
| concurrent | 0.835 | 9.58 |

Concurrent speedup was 0.220-fold. The tiny five-layer model is dominated by
threading and batch setup, so this is expected to be a mechanism stress rather
than a model-scale prediction.

### Real 35B contended run

| Mode | Startup (s) | Measured elapsed (s) | Aggregate completion tok/s |
|---|---:|---:|---:|
| sequential | 292.35 | 80.39 | 0.0498 |
| concurrent | 192.12 | 84.32 | 0.0474 |

Concurrent speedup was 0.953-fold. The first sequential request took 74.01 s,
of which 61.99 s was attributed to expert loading; the warmed second request
took 6.30 s with 0.16 s expert loading. Concurrent inclusive intervals
contained 53.95 s and 42.86 s of expert loading. Both modes ended with 320
experts in approximately 1.17 GiB VRAM.

## Interpretation

The live-pipe defect was a transport implementation bug, not a scheduler or
model failure. EOF-based tests are insufficient for a server protocol because
stdio buffering changes the visibility relationship between `FILE*` reads and
descriptor polling. The new test reproduces the real gateway ownership model.

The CUDA event mechanism is useful for controlled studies, but pervasive event
recording and synchronization perturb this backend strongly. It is therefore
an experimental diagnostic, not always-on production telemetry.

The 35B trial does not show a continuous-batch gain under cold storage
contention. It also does not falsify the warm-cache hypothesis: most of the
elapsed interval was expert I/O, and the two modes started independent engines
with different startup/page-cache conditions. The warmed single-slot request
at 2.75 tok/s demonstrates the scale of the confound.

## Decision

- Retain unbuffered mux stdin and the simultaneous persistent-pipe regression.
- Retain `COLI_ENGINE_TRACE=1` as prompt-free protocol diagnostics.
- Retain the CUDA snapshot suffix behind `CUDA_PROFILE_STAGES=1`.
- Do not enable stage events by default in `colib --cuda`.
- Retain the benchmark harness.
- Record the contended 0.953-fold result as negative and do not use it to close
  the warm continuous-batch gate.

## Next experiment

After 397B conversion releases storage and page-cache pressure:

1. preload a fixed 35B expert working set;
2. run sequential/concurrent in both AB and BA order;
3. sweep two, four, and eight output tokens;
4. record corrected wall phase counters without CUDA events;
5. repeat one short run with events for device-stage attribution;
6. measure cancel-to-ack latency while another slot remains in decode.

The same harness then becomes the 397B Gate-9 qualification after Gate 7 has a
coherent, warmed model.

The frozen two-slot 397B invocation is:

```sh
.venv/bin/python c/tools/bench_serve_batch.py \
  --model c/qwen397 --requests 2 --max-tokens 8 \
  --warmup-passes 2 \
  --order sequential,concurrent --ram-gb 18 --ram-headroom-gb 1 \
  --cuda --cuda-expert-gb 5 --cuda-headroom-gb 1 \
  --request-timeout 1800 --output c/qwen397_batch_qualification.json
```

Supplying `--ram-gb` also enables the validated PIPE/`io_uring`/direct-I/O
path, persistent expert heat, and disables the rejected predictive prefetch
policy. The two-slot run uses 5 rather than 6 GiB of device expert cache:
6 GiB leaves only about 40 MiB beyond the specified headroom at the measured
post-context free VRAM, whereas 5 GiB preserves roughly 1 GiB of additional
operational margin.
