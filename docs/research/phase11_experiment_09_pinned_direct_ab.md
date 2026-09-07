# Phase 11 experiment 09: pinned/direct matched A/B

Date: 2026-08-17

## Decision

The combined pinned-upload and persistent-direct-I/O intervention preserves
the exact two-token output and improves aggregate time, but fails the
resident-token no-regression condition. It is not accepted for decode
qualification.

Initialization falls from 408.372656 to 200.970412 seconds. Token 0 falls from
134.734045 to 83.758211 seconds. Token 1 rises from 84.532588 to 96.703443
seconds, a 14.40% regression. The same 28 routes hit and 230 miss on token 1,
so the performance change is an I/O-path effect rather than routing drift.

## Intervention

```sh
SNAP=/home/dinga/Projects/colib/c/deepseek-v4-flash-0731 \
  CTX=16384 DSV4_SMOKE_CONTEXT=128 DSV4_SMOKE_TOKEN=0 \
  DSV4_SMOKE_TOKENS=2 DSV4_SMOKE_PROFILE=1 COLI_CUDA=1 \
  RAM_GB=12 CUDA_EXPERT_GB=5 DIRECT=1 URING=1 URING_PERSIST=1 \
  URING_WORKERS=4 DSV4_DENSE_PINNED_MB=64 c/deepseek_v4
```

The intervention binary SHA-256 is
`1bfc55c990294201f9c4157ec406ec12c251b960f1487e2e6846db544b149c9b`.
The matched control is
`docs/research/artifacts/deepseek_v4_cold_warm_profile.json`, SHA-256
`f0d0358c4ce3839ca0b8f46cbcac3efaa6023440ef49d68be4e8773c34dc00c1`.

The tokens and top logits are exact in both arms: token 0 selects 5 at
16.258379 and token 1 selects 223 at 18.0251122. The intervention attributes
15,370,206,592 bytes to pinned model uploads, covering the dense arena and
expert pages. CUDA activation uploads account for the small remainder of the
15,375,519,072 total upload bytes.

## Interpretation

Pinned staging clearly improves the pageable-transfer problem, and the
one-time initialization result is useful. The combined experiment cannot
separate that win from direct-I/O behavior, however. The resident-token
regression and wide per-layer variance show that 36 separate tensor reads per
layer are a poor qualification unit on ext4-in-VHDX.

The converter already stores each expert's six records contiguously and
aligned. The next implementation step is therefore to read one contiguous
expert extent rather than six separate records, reducing a top-6 layer miss
from 36 I/O requests to six. Before another expensive end-to-end run, a short
real-container benchmark must compare buffered synchronous, buffered
`io_uring`, direct synchronous, and direct persistent-`io_uring` modes.

The complete matched values and acceptance result are in
`docs/research/artifacts/deepseek_v4_pinned_direct_ab.json`.
