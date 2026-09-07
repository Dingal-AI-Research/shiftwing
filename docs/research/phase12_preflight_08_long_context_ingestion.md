# Phase 12 preflight 08: long-context ingestion

Date: 2026-08-21

## Why long context is a different regime

Every performance measurement in this project uses the four legacy prompts,
which are about 38 tokens each. The owner intends to load **10,000-16,000
tokens on the first request**, and the dominant cost changes shape at that
length.

At 38 prompt tokens a turn touches **2,684 distinct experts**, 8.7% of the
30,720 total (`telemetry_summary.turn_hit_experts` in the preserved control).
At 10,000 tokens each layer sees 100,000 routes across 512 experts, roughly 195
per expert, so the distinct set **saturates at essentially every expert**.

That is an ~11x increase in expert I/O, not a proportional one with token
count. First-request prefill must read close to the whole 157 GB q3 sidecar.
Subtracting the 3,948 cached experts leaves about **137 GB of misses**, and at
the measured 1.40 GB/s decode rate that is **~98 seconds of expert I/O alone**,
before any compute. Treat that as a floor rather than an estimate.

A consequence worth stating plainly: **the frozen expert-map preload
(candidate 3) helps far less at long context.** It fills a 12.85% cache with
the hottest experts, but a 10k prefill needs every expert regardless of heat.
It remains valuable for decode and short follow-up turns; it will not
meaningfully change first-request TTFT on a large prompt.

## What is already implemented

Expert-major grouped prefill is **on by default and always has been**.
`forward_prefill` passes `grouped_moe=1` unconditionally, and
`moe_prefill_grouped` loops over experts, gathers every token routed to each,
runs one batched SwiGLU, and scatters the results back. Each expert is
therefore streamed **once per layer**, not once per routed token.

The source notes this is bit-identical to the per-token path on CPU, because
`mlp_batch` and `mlp_one` both bottom out in `qmat_dot_row` with
`allow_idot=1` and the grouped scatter preserves top-k accumulation order. So
it is a pure ordering win with no output risk.

An earlier draft of this analysis proposed expert-major batching as an
unmeasured opportunity. That was wrong: it already exists, and its remaining
gain is zero. Reading the prefill path before ranking would have caught it.

## Candidate 6: prefix KV caching for a stable preamble

The engine already checkpoints and restores session state
(`SESSION_DIR`, `mux_checkpoint_slot`, `mux_session_write`) and performs
prefix matching, which the logs report as
`[SESSION] prompt-prefix-miss slot=0 checkpoint=42 prompt=38 common=3`.

It is **off by default**: `SESSION_DIR` is unset unless `colib --session-dir`
is passed, and the client must reuse the same slot.

If the 10-16k context is a stable preamble reused across requests -- a
codebase, a document set, a long system prompt -- the prefill is paid once and
subsequent requests reuse the cached KV. That is the largest available win for
this workload by a wide margin, and it is wiring rather than new engine work.

The constraint is exactness of the prefix. The log line above shows what
happens otherwise: a differing prompt collapses the common prefix to 3 tokens
and the whole context is recomputed. Anything that varies -- a timestamp, a
request id, reordered context -- must sit *after* the stable block, not before
it.

## Candidate 2 rescoped: `PREFILL_EXPERT_BATCH` at long context

The ordered grouped-prefill I/O candidate is implemented and defaults to
`PREFILL_EXPERT_BATCH=1`, the legacy one-expert-per-submission path. It clamps
to `[1, 32]` and additionally to the per-layer cache capacity.

It was validated only on the tiny fixture, where batch 4 preserved exact output
while reducing ring submissions from 50 to 41. Long-context prefill is the
workload it should actually matter for, because that is when nearly every
expert is cold and must be submitted. It is a single environment variable, so
testing it is minutes rather than hours.

## Candidate 4 extended: coalescing helps prefill too

The read-coalescing candidate was scoped to decode, but the same six-reads-per
-expert pattern applies during prefill, where nearly every expert is read.
Raising 1.40 GB/s toward the measured 2.9 GB/s device ceiling would cut the
~98 second prefill floor to roughly 55 seconds. This strengthens the case for
candidate 4, since it now pays on both phases.

## Open question

Whether the serve path hands the whole prompt to the expert-major loop in one
pass or splits it. `moe_prefill_grouped` is expert-major over whatever `T` it
receives, and the only chunking found in the prefill path is
`gdn_prefill_chunked`, which is the linear-attention primitive rather than the
MoE loop. If the serve path splits, experts are re-read per split and
`PREFILL_EXPERT_BATCH` matters considerably more.

One 10k-token prefill with telemetry settles it: about 137 GB of prefill expert
reads means a single pass, and a multiple of that means splitting. That
measurement also replaces every estimate in this report with a fact, and costs
about five minutes.

## Status

No long-context measurement has been taken. Every number here is derived from
38-token telemetry and the expert-count arithmetic, not observed at 10k. No
promotion is proposed.
