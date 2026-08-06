# Phase 8 Preflight 10: Native Q3 and an Adaptive Hot-Route Atlas

**Model:** `deepreinforce-ai/Ornith-1.0-397B-FP8` with the complete grouped-int3 sidecar
**Date:** 2026-08-01
**Status:** CUDA unit validation passed; full-model optimization validation remains open

## Abstract

The accepted expanded-q4 execution of the Ornith397 grouped-int3 sidecar
sustains 0.718432662 token/s, with a 0.8658355 token/s median and a 0.45495
token/s minimum turn. This is 13.68% faster in sustained throughput and 28.46%
faster at the median than the completed int4 profile, although its minimum turn
is 14.27% slower. The owner selected q3 for production based on the sustained
and median results.

Two implementation changes target the remaining storage boundary. First, q3
weights now remain packed at three bits on CUDA rather than expanding to four
bits in the device cache. Second, an opt-in adaptive atlas partitions learned
hot routes disjointly across host and device caches. The implementation is
present. Its dedicated CUDA kernels now pass controlled fp32/fp16, single,
grouped, and batched comparisons, but its complete real-model numerical and
performance protocol remains unfinished. The release path therefore retains
the accepted q3-to-q4 CUDA expansion by default; `Q3_NATIVE=1` opts into the
packed representation.

## 1. Motivation

The expanded-q4 q3 profile still used approximately the same device bytes per
expert as int4. It therefore improved disk and host storage size without
increasing the number of routed experts retained in 6 GiB of VRAM. Telemetry
also showed a slow outlier associated with storage and upload variability.

Native packed q3 changes this capacity boundary. One routed Ornith397 q3
expert occupies 5,111,808 bytes including scales. Under the planned 20 GiB
host and 6 GiB device expert budgets:

| Quantity | Exact capacity |
|---|---:|
| Host slots per layer | 70 |
| Host slots reserved for one unseen top-k route | 10 per layer |
| Host hot entries across 60 layers | 3,600 |
| Device hot entries after a ten-expert transaction reserve | 1,250 |
| Combined adaptive hot-set capacity | 4,850 |
| Previously observed deterministic route set | 4,779 |
| Nominal spare entries | 71 |

This arithmetic is a feasibility hypothesis, not a speed claim. New prompts
may have a larger or differently distributed working set.

## 2. Native packed-q3 method

The CUDA backend now stores the original 24-bit q3 triplets and their grouped
scales directly. Dedicated fp32-input, fp16-input, single-expert, and grouped
expert kernels unpack signed three-bit values during multiplication. Device
cache accounting uses packed q3 bytes, and machine-readable `Q3NATIVE`
telemetry records native uploads and kernel calls.

A full real-model qualification was started, then stopped when the owner chose
to defer testing. It completed only two warm-up turns:

| Warm-up observation | Decode rate | TTFT | Wall time |
|---|---:|---:|---:|
| Prompt 1 | 0.488 token/s | 215.8 s | 347.0 s |
| Prompt 2 | 0.885 token/s | 97.1 s | 169.5 s |

These incomplete warm-up observations are not a benchmark. No measured pass
or JSON result artifact was produced.

## 3. Adaptive disjoint atlas

`Q3_ROUTE_ATLAS=1` extends decode protection only when grouped-int3 experts,
CUDA experts, and decode protection are explicitly enabled. At each completed
request it:

1. ranks every observed decode route by frequency and recency;
2. retains the hottest `host_capacity - top_k` entries per layer in RAM;
3. removes matching CUDA copies so host and device hot sets are disjoint;
4. ranks the remaining routes globally;
5. pins as many as fit in the packed-q3 device budget after reserving one
   top-k transaction;
6. detaches host copies after successful device pinning, freeing their RAM
   slots for ordinary misses; and
7. unpins and recomputes the partition after the next completed request.

Unlike the rejected fixed atlas, this policy does not assume that future
prompts reproduce one exact route set. Routes outside the retained hot set use
the reserved ordinary cache slots. The policy exposes `Q3ATLAS` telemetry with
activation, refresh, route, host/device-entry, capacity, load, byte, and
uncovered-route counts.

The qualification harness adds `--q3-route-atlas {0,1}` and records both the
requested configuration and observed telemetry. The intended later pilot is:

```bash
.venv/bin/python c/tools/qualify_tiered_model.py \
  --model c/ornith397 \
  --expert-q3 1 --q3-native 1 --q3-route-atlas 1 \
  --expert-ram-gb 20 --cuda-expert-gb 6 \
  --ram-headroom-gb 1 --cuda-headroom-gb 1 \
  --threads 8 --uring-persist 1 --pinned-upload 1 \
  --decode-protect 1 --decode-protect-prewarm 1 \
  --warmup-passes 2 --measured-passes 1 --max-tokens 64 \
  --minimum-tps 0.70 \
  --output c/ornith397_q3_atlas_qualification.json
```

## 4. Validation status

The CUDA target and native-q3 kernel regressions now pass. The focused backend
test measures maximum differences of `3.87e-07` for fp32-input q3 GEMV,
`6.04e-04` for fp16-input q3 GEMV, and zero for grouped and batched q3 MoE
against their CPU references. Before this path can become the release default,
full-model work must still establish:

1. q3 results remain numerically equivalent with packed execution and with
   the atlas off and on;
2. every pinned device entry is q3, host/device hot-set intersection is zero,
   cache bytes remain within both budgets, and ten transaction slots remain;
3. unseen routes continue through ordinary bounded caching without deadlock or
   silent host fallback;
4. the complete four-prompt profile produces nonempty coherent output and
   complete native-q3, atlas, CUDA, storage, and residency telemetry; and
5. sustained, median, minimum, TTFT, disk-read, and upload results are compared
   with the accepted 0.718432662 token/s expanded-q4 q3 baseline.

## 5. Decision

Q3 is the selected production precision. The reproducible release candidate
uses the accepted expanded-q4 CUDA representation. Native packed-q3 execution
and the adaptive atlas remain explicit experiments until the deferred
full-model validation is completed; unit correctness alone does not establish
production parity or speed.

## 6. Live-stream directional probes

After the owner authorized small tests, `stream_qwen_benchmark.py` was added as
a thin wrapper over the existing byte-framed mux engine. It prints each
decoded piece with immediate flushing, retains the ordinary engine telemetry,
atomically writes a compact JSON artifact, and can save the exact final
response. A visible unbuffered terminal was used for both probes.

The first probe requested exactly `colib q3 ready`, with one six-token warm-up
and one six-token measured turn. Both streamed the exact requested text. The
measured turn reached 1.084803 token/s with 175.760 seconds TTFT and a 100%
decode cache-hit rate. Native q3 reported 3,240 grouped calls, and the VRAM tier
held 1,260 experts. The atlas refreshed once, but all 2,597 learned routes fit
the host hot set, so the device-pinned overflow branch was not exercised. The
4,985-byte artifact is `c/ornith397_q3_atlas_stream_small.json`, SHA-256
`dbf0bc6fdf69eb7fa90794bda35877e8ff34025c2c037a403ff4c4fca45d5a8f`.

The second probe asked for a self-contained HTML/JavaScript Snake game and
streamed one cold response without the atlas. It reached 0.574879 token/s,
446.963 seconds TTFT, and a 55.2296% decode cache-hit rate. All 23,040 layer
forwards used device MoE with zero host fallback, and native q3 reported 27,720
grouped calls. The response hit the 384-token limit after 1,114.942 seconds and
ended inside `start()`, so it is an incomplete game rather than a functional
quality pass. The exact partial HTML is
`c/bench/ornith397_q3_snake.html`, SHA-256
`fff527e49d0c5998fbee4d05ed36d31ecd382d7d9f7c3dcb4a1b79cb348d9be7`;
the JSON record hashes to
`106a6987f9e213e6f696a5e73e99b29683860326c6484bcd0f228d843bda1175`.

A second Snake attempt raised the safety ceiling to 1,024 tokens. It sustained
0.579593 token/s with 440.306 seconds TTFT and emitted 2,790 bytes of
self-contained HTML through most of the keyboard handler. It again consumed
the complete token budget before closing the document. The owner accepts the
two attempts as a directional coding-quality pass, but neither artifact is a
syntactically complete game and neither substitutes for the frozen four-prompt
coherence gate. The second JSON artifact is
`c/ornith397_q3_snake_complete_stream.json`.

This probe also exposed a wrapper-only latency issue: one-shot runs inherited
decode protection and performed an unnecessary post-response host prewarm.
The wrapper now enables decode protection and prewarming only when the atlas is
explicitly requested. Neither directional probe replaces the deferred full
correctness or performance protocol.
