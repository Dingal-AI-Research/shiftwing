# Phase 8 Preflight 7: Revised Ornith397 Production Threshold

**Model:** `deepreinforce-ai/Ornith-1.0-397B-FP8`

**Converted revision:** `8b61f97a8512d9d01bff1a9625c9a16730e115bb`

**Date:** 2026-07-31

**Status:** preregistered amendment; production rerun pending

## Abstract

The original Ornith397 release protocol required at least 2.0 generated
tokens per second. The direct four-prompt profile reached 0.527355 token/s,
while a bounded lossless optimization pilot reached 0.867369 token/s. After
reviewing these results, the project owner revised the product requirement to
approximately 0.85 token/s.

This amendment converts “approximately” into the machine-checkable rule
**sustained throughput ≥0.85 token/s**. It applies prospectively to Ornith397
production qualification only. It does not alter the historical Qwen397
Gate-7 result, rewrite the original experiments, or automatically promote the
one-prompt 16-token pilot. Release still requires the full four-prompt,
64-token, two-warm-pass qualification under the lossless optimized profile,
followed by generated-tool and Gate-9 production evidence.

## 1. Research question

Does the lossless optimized Ornith397 profile sustain at least 0.85 token/s
over the complete frozen four-prompt workload while preserving coherence,
model identity, resource limits, CUDA residency, and telemetry?

## 2. Reason for the protocol amendment

The 2.0 token/s target was a product requirement rather than a numerical
correctness limit. All structural and numerical evidence passed, and the
best lossless pilot demonstrated that 0.85 token/s is plausible on the
reference workstation. The owner therefore accepts that lower service rate
for the 397B production role.

This is a post-result requirement change and is labelled accordingly. To
avoid selecting a favorable short observation after seeing it, the pilot is
treated only as the motivation for a new frozen run. Its 16 measured tokens,
single prompt, and one warm pass are insufficient release evidence.

## 3. Exact revised profile

The qualification retains the pinned int4-g128/int8 container and all original
identity and resource controls:

| Parameter | Value |
|---|---:|
| prompts | four frozen family-rendered prompts |
| warm-up | two complete passes |
| measurement | one complete pass |
| generated tokens | 64 per prompt |
| context | 4,096 tokens |
| host expert cache | 18 GiB |
| device expert cache | 6 GiB |
| RAM/VRAM headroom | 1 GiB each |
| precision | routed int4-g128; I/O/shared int8; no low-bit sidecar |
| minimum sustained rate | **0.85 token/s** |

Four already implemented, lossless switches become part of the production
profile because the bounded pilot verified that they executed and preserved
the output:

- persistent `io_uring` and aligned-buffer reuse;
- reusable pinned upload staging;
- protection of decode-hot experts from prompt eviction; and
- inter-request prewarming of the protected decode set.

Predictive prefetch and all 2-bit/3-bit expert paths remain disabled.

## 4. Command

```sh
.venv/bin/python c/tools/qualify_tiered_model.py \
  --model c/ornith397 \
  --max-tokens 64 \
  --warmup-passes 2 \
  --measured-passes 1 \
  --expert-ram-gb 18 \
  --cuda-expert-gb 6 \
  --ram-headroom-gb 1 \
  --cuda-headroom-gb 1 \
  --minimum-tps 0.85 \
  --uring-persist 1 \
  --pinned-upload 1 \
  --decode-protect 1 \
  --decode-protect-prewarm 1 \
  --output c/ornith397_qualification.json
```

The Gate-8 supervisor now freezes this same argument vector. The independent
release auditor requires every value exactly and recomputes sustained rate
from completion-token counts and per-turn rates.

## 5. Acceptance rule

The revised tier check passes only when all of the following are true:

1. the artifact is bound to the exact 122-shard Ornith397 manifest;
2. exactly four nonempty measured outputs are present;
3. the token-weighted sustained rate is at least 0.85 token/s;
4. hardware, tier-map, expert-map, and request telemetry are complete;
5. a real CUDA device and nonempty VRAM tier are reported;
6. every resident layer uses device MoE with zero host-MoE fallback;
7. no grouped-2-bit or grouped-3-bit sidecar is selected; and
8. all four optimized lossless controls have the frozen values above.

The boundary is strict: 0.849 token/s fails. A tier pass authorizes the
existing generated HTTP tool qualification. Gate 8 passes only after both
tier and tool artifacts pass. Gate 9 remains prohibited until then.

## 6. Downstream sequence

If Gate 8 passes, the manifest-bound Gate-9 controller runs its frozen ten
steps for Ornith35 and Ornith397: AB and BA batch orders, the comparator,
20-trial cooperative cancellation, and exact two-turn web streaming. The
strict release audit then recomputes all 15 checks. `make check`, source
packaging, an intentional release commit, and `v0.1` remain separate final
requirements.

If the full optimized profile remains below 0.85 token/s, Gate 8 closes
negative again. No further residency retry or threshold change is implied.

## 7. Limitations

- The revised value is an owner-approved product trade-off, not a new estimate
  of model quality or a general hardware performance law.
- The measured rate remains specific to the reference WSL/NVMe/CUDA system.
- The optimized profile may perform differently across four long turns than
  in the one-prompt pilot; this is why a complete rerun is mandatory.
- Changing the Gate-8 threshold does not relax Gate-9 batch ratios,
  cancellation latency, exact web output, numerical, or packaging criteria.
