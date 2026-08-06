# Phase 9 Experiment 22: Production Cancellation Qualifier

## Abstract

This experiment converted the earlier cooperative-cancellation probe into a
repeatable production qualification harness. The revised harness verifies the
model snapshot identity, separates warm-up work from measured work, measures
cancel-to-ack latency while a peer slot continues decoding, proves that the
cancelled slot can be reused immediately, and rejects CUDA runs that fall back
from device MoE to host MoE. A real Qwen3.5-35B-A3B validation passed all
functional and residency checks. Because the 397B converter was active and the
test deliberately used zero warm-up passes, its timing is diagnostic rather
than the final Gate-9 latency result.

## Research Question

Can one command produce auditable evidence that cancellation is cooperative,
does not interrupt another request, releases its slot, and stays on the
resident CUDA graph?

## Method

`c/tools/bench_cancel_mux.py` now performs the following protocol:

1. require a complete `quantization.json` and retain its immutable source,
   fingerprint, shard count, byte count, tensor counts, and quantization
   parameters in the result;
2. optionally run a declared number of two-slot warm-up passes, then exclude
   their request profiles from measurement;
3. start two requests simultaneously through the real mux and gateway engine;
4. cancel slot 0 immediately after its first streamed data piece while slot 1
   continues to its normal length limit;
5. submit a fresh one-token request to slot 0 after cancellation is
   acknowledged;
6. collect request profiles, tier placement, hardware identity, and
   `CUDA_RESIDENT` counters; and
7. write a schema-versioned JSON record and fail the process if any declared
   acceptance condition is false.

The production validation command was:

```bash
.venv/bin/python c/tools/bench_cancel_mux.py \
  --model c/qwen35 \
  --cuda \
  --warmup-passes 0 \
  --max-tokens 2 \
  --ram-gb 2 \
  --cuda-expert-gb 7 \
  --request-timeout 180 \
  --max-cancel-ack-s 60 \
  --output /tmp/bench-cancel-q35-resident.json
```

The 60-second bound is intentionally loose for this harness-validation run.
It detects deadlock or missing acknowledgement but is not the final warm p95
criterion.

## Results

The run completed with `acceptance.passed=true` and no failures.

| Observation | Result |
|---|---:|
| Engine startup | 7.29 s |
| First cancelled-slot content | 10.28 s |
| Cancel-to-ack | 1.17 s |
| Pieces emitted before cancel | 1 |
| Peer request completion | 11.90 s |
| Released-slot reuse | 4.93 s |
| Measured layer-forwards | 120 |
| Device-MoE layer-forwards | 120 |
| Host-MoE layer-forwards | 0 |

The peer produced two completion tokens and the reused slot produced one
completion token. Resident transfer telemetry reported 32,768 bytes for the
input activation boundary, 32,768 bytes for the output activation boundary,
3,973,120 bytes of vocabulary logits, and 163,840 bytes of router logits.
These counts cover the peer's two forwards plus the reuse request's one
forward over 40 layers.

The two measured request profiles also show why the timing cannot be treated
as an uncontended benchmark. The peer request spent 1.54 seconds in expert
disk work, while the slot-reuse request had no expert disk time. The Qwen397
converter was concurrently downloading and converting multi-gigabyte shards.

## Interpretation

Cancellation is acknowledged at a token boundary, so cancel-to-ack is bounded
by the currently executing forward rather than by a host-thread signal alone.
The 1.17-second observation demonstrates that the resident graph checks for
cancellation promptly after the first emitted piece under this configuration.
It does not establish the eventual p95 latency for warm production workloads.

The peer and reuse checks are essential. A low cancellation time by itself
could hide engine termination, cross-request interference, or a slot that was
never actually released. Likewise, a correct lifecycle result could silently
exercise the CPU fallback. The resident counters make these failure modes
observable and machine-checkable.

## Decision

- Accept schema version 2 of `bench_cancel_mux.py` as the Gate-9 cancellation
  qualification harness.
- Accept this 35B run as functional and CUDA-residency validation.
- Do not use its latency as the final Gate-9 number because it had no warm-up
  passes and shared storage with the active 397B conversion.
- Run the frozen warm 35B and 397B cancellation trials after conversion and
  other large downloads stop; report repeated-trial p95 rather than selecting
  a single favorable sample.

## Limitations

The validation used one RTX 5070 Ti, a two-token peer request, and a permissive
60-second ceiling. It did not estimate a latency distribution, test 397B, or
measure behavior at long context. The final experiment must use controlled
warm-up, repeated trials, fixed prompts and cache budgets, and no converter,
download, build, or unrelated storage contention.
