# Ornith35 explicit-int4 default and throughput record

**Model:** `deepreinforce-ai/Ornith-1.0-35B-FP8`  
**Converted revision:** `1ab57ce0b44950e498a88756f40ad1ed4d0f30ca`  
**Date:** 2026-08-06  
**Status:** complete directional measurement; not release-gate evidence

## Decision

The source-tree launcher defaults to the Ornith35 base container, whose
quantization manifest records grouped int4 routed weights with group size 128.
The complete routed-expert int3 sidecar remains on disk for reproducibility,
but it is selected only by the explicit `--expert-q3` option.

The launcher now writes `EXPERT_Q2=0`, `EXPERT_Q3=0`, `Q3_NATIVE=0`, and
`Q3_ROUTE_ATLAS=0` into ordinary chat, server, and web environments. This
prevents ambient research variables from silently changing the default away
from int4. Explicit q3 options still override those defaults.

## Measurement

The isolated benchmark used the release engine and base Ornith35 manifest,
with no expert sidecar selected:

```sh
.venv/bin/python -u c/tools/stream_qwen_benchmark.py \
  --model c/ornith35 \
  --prompt "Count from one to one hundred, one number per line, and do not stop early." \
  --warmup-turns 1 --measured-turns 3 --max-tokens 64 \
  --threads 8 --expert-ram-gb 20 --cuda-expert-gb 6 \
  --output c/bench/ornith35_int4_tps_20260806.json
```

The warm-up produced 64 tokens at 24.912394 token/s. The three measured
64-token turns produced 39.832177, 41.082213, and 40.355589 token/s.
Token-weighted sustained throughput was **40.416843 token/s**, the median was
40.355589, and the minimum was 39.832177.

All 10,240 recorded layer-forwards used device MoE, host fallback was zero,
and the final tier state placed 3,855 experts in VRAM, 6,385 in RAM, and none
on disk. The result is a directional single-user decode measurement, not a
replacement for the frozen release qualification protocol.

## Evidence identity

- Base manifest SHA-256:
  `b9b54701d38c0ea2836c9863f320f0f61ce4e70eac8cf7444babe146c0b6115d`
- Engine SHA-256:
  `8659db94bb6904b8710389848532af6e2d2b169bd790a6b3c9bd3ed2bdb9c467`
- Raw ignored artifact SHA-256:
  `d8ebdea459032d014e20284f45b77b054773b777a4d10173d5dc14599d8ed8f4`
- Tracked compact record: `c/ornith35_int4_tps.json`
