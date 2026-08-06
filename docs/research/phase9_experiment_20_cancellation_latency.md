# Phase 9 Experiment 20: Cancellation Latency with a Live Peer Slot

## Abstract

This experiment measured cancellation as a concurrent scheduling operation
rather than an isolated protocol response. One slot requested cancellation
after its first streamed piece, a second slot continued decoding, and the
cancelled slot was reused immediately afterward. The tiny CPU model
acknowledged cancellation in 194 ms. The real 35B CUDA model acknowledged it
in 9.88 s while the peer completed normally in 30.31 s; converter contention
and expert loading make this an operational upper-bound observation rather
than a warm latency target.

## Method

`c/tools/bench_cancel_mux.py` starts the production `Engine` with two resident
KV slots. Two threads synchronize at a barrier:

- slot 0 requests up to N tokens and returns true from its cancellation
  callback after the first content piece;
- slot 1 requests N tokens and must finish normally.

Cancel latency is measured from receipt of the first content piece—immediately
before the gateway writes `CANCEL`—until `Engine.generate` receives
`ERROR <id> CANCELLED` and raises `ClientCancelled`. After both threads end,
the harness submits a one-token request to slot 0 to prove release and reuse.
An explicit worker deadline reports pending request IDs instead of hanging.
The harness accepts and records a byte-budgeted `--ram-gb` cache and explicit
RAM/CUDA headroom values, allowing the 397B run to reproduce its approved
resource profile rather than only a per-layer expert count.

The regression test performs the same lifecycle on the tiny model with a
32-token cancelled request and four-token peer. It requires exactly one piece
before cancellation, four peer tokens, and a successful one-token reuse.

The real run used Qwen3.5-35B-A3B, CUDA 12.9 on the RTX 5070 Ti, fp16
activations, a 7 GiB expert budget, two generated tokens per request, and a
one-token raw prompt. The 397B converter remained active.

## Results

| Measurement | Tiny CPU | 35B CUDA |
|---|---:|---:|
| startup | 12.99 s | 140.07 s |
| first content | 38 ms | 14.50 s |
| cancel-to-ack | 194 ms | 9.88 s |
| peer request | 245 ms | 30.31 s |
| cancelled pieces | 1 | 1 |
| post-cancel slot reuse | 24 ms | 5.97 s |

The peer produced all requested tokens in both runs. The 35B peer's inclusive
profile attributed 18.48 s to expert loading, 9.64 s to GDN/GQA sequence work,
1.74 s to expert compute, and 0.10 s to the LM head. The reuse request reduced
expert loading to 0.63 s and completed in 5.96 s.

The new peer-aware regression passed on both CPU and CUDA builds. It extends
the prior isolated cancellation test by proving that another resident row
continues and that the released slot is usable.
The complete suite later passes 21 C executables and 59 Python tests.

## Interpretation

Cancellation is cooperative at mux scheduling boundaries. The gateway can
only observe a content piece after the current batched forward has completed,
and the engine checks the newly written `CANCEL` before beginning the next
decode batch. On 35B, the 9.88 s acknowledgement therefore approximates one
contended iteration plus pipe/dispatch overhead; it is not an interruptible
mid-kernel cancellation.

The result demonstrates safety and bounded progress: cancelling one row does
not cancel, corrupt, or indefinitely block its peer. It does not yet meet a
low-latency interactive target. Reducing the bound requires faster per-step
execution or finer-grained kernel interruption, the latter of which would add
substantial state-rollback complexity.

## Decision

- Retain cooperative cancellation between decode batches.
- Retain the peer-completion and immediate-slot-reuse regression.
- Retain the benchmark as the Gate-9 latency instrument.
- Record 9.88 s as the 35B contended observation, not the warm acceptance
  threshold.
- Rerun after conversion/cache warmup and define the production latency target
  from the warm p95 distribution.
- Repeat on 397B only after Gate 7 produces coherent warmed decode.

The corresponding 397B command is:

```sh
.venv/bin/python c/tools/bench_cancel_mux.py \
  --model c/qwen397 --max-tokens 8 \
  --ram-gb 18 --ram-headroom-gb 1 \
  --cuda --cuda-expert-gb 5 --cuda-headroom-gb 1 \
  --request-timeout 1800
```

## Limitations

The sample count is one per model and the 35B run shared CPU, RAM, and storage
with conversion. No `SESSION_DIR` was configured, so the measurement excludes
durable checkpoint write latency. Cancellation arrived after a content piece,
not during prefill; prefill cancellation remains unsupported.
