# Phase 11 experiment 08: cold/warm real-model profile

Date: 2026-08-17

## Decision

The first two-token resident profile passes its diagnostic gate and fails the
performance gate. It rejects the hypothesis that the first smoke was slow
only because its expert cache was cold.

Token 0 takes 134.734045 seconds and misses all 258 routed experts. Token 1
takes 84.532588 seconds, but only 28 routes hit while 230 miss. It reads
3,074,949,120 additional bytes and uploads the changed expert pages. Aggregate
throughput is 0.009121 tok/s.

The 667 CUDA dense projections are not the dominant cost in this run. FP8 plus
BF16 dense wall time is 2.023834 seconds for token 0 and 1.912695 seconds for
token 1. The output head is about 0.05 seconds. Nearly all remaining time is
inside layers whose expert routes miss.

## Frozen control

```sh
make -C c CUDA=1 CUDA_ARCH=native deepseek_v4 test-cuda
SNAP=/home/dinga/Projects/colib/c/deepseek-v4-flash-0731 \
  CTX=16384 DSV4_SMOKE_CONTEXT=128 DSV4_SMOKE_TOKEN=0 \
  DSV4_SMOKE_TOKENS=2 DSV4_SMOKE_PROFILE=1 COLI_CUDA=1 \
  RAM_GB=12 CUDA_EXPERT_GB=5 c/deepseek_v4
```

The binary SHA-256 is
`8199bb1de51d2117382d9dd355aa02cda45599b40a59c5770dd7d1730819b863`.
The run intentionally leaves `DIRECT`, `URING`, and pinned staging off to
measure the pre-optimization path. The current source retains
`DSV4_PINNED_UPLOAD=0` as the reproducible transfer control.

Full per-layer values, exact environment, hardware, command, model identity,
and counters are in
`docs/research/artifacts/deepseek_v4_cold_warm_profile.json`.

## Interpretation

The 12 GiB host budget provides 20 slots per layer and the 5 GiB device budget
holds more than the first token's 258 experts. Nevertheless, consecutive
routes overlap by only 10.85%. A cache sized for this machine therefore cannot
turn the current path into a mostly resident workload.

The native FP4 payload is 13,369,344 bytes per expert. Six experts across 43
layers require about 3.45 GB on a fully cold token. A resident token in this
sample still needs 3.075 GB. Repeated pageable host-to-device copies and
buffered synchronous reads dominate the measured layer times.

## Next controlled intervention

The implementation now adds two reversible controls:

1. A bounded reusable pinned host staging area for each top-6 expert group and
   chunked pinned staging for the one-time dense upload.
2. The existing persistent `io_uring` and direct-I/O path enabled together
   for expert misses.

The next run must use the same token sequence and cache budgets, record pinned
bytes plus direct/uring counters, and compare every emitted token/logit before
claiming a speed improvement. Dense tensor-core work remains valuable, but it
is not the first-order bottleneck exposed by this profile.
