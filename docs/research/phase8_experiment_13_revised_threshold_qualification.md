# Phase 8 Experiment 13: Revised-Threshold Production Qualification

**Model:** `deepreinforce-ai/Ornith-1.0-397B-FP8`

**Converted revision:** `8b61f97a8512d9d01bff1a9625c9a16730e115bb`

**Date:** 2026-07-31

**Status:** completed; rejected at 0.631964 token/s

## Abstract

The project owner revised the Ornith397 production minimum from 2.0 to
0.85 token/s after a one-prompt, 16-token lossless pilot reached 0.867369
token/s. To prevent a short favorable observation from becoming release
evidence, a prospective amendment required the complete four-prompt workload:
two warm passes and one measured 64-token pass, with persistent I/O, pinned
upload, decode protection, and inter-request prewarming enabled.

The full profile completed in 2,664.8 seconds. All four measured outputs are
nonempty and coherent; the exact manifest, CUDA device, resident graph,
expert-map, tier, and request telemetry controls pass; and all 46,080
resident layer-forwards use device MoE with zero host fallback. The measured
turn rates are 0.676676, 0.673275, 0.674737, and 0.530693 token/s. Their
token-weighted sustained rate is **0.631963761 token/s**, 25.65% below the
revised 0.85 minimum. Gate 8 therefore closes negative again, and the ordered
Ornith397 tool and Gate-9 production sequences remain prohibited.

## 1. Research question

Does the optimized, numerically unchanged Ornith397 profile sustain at least
0.85 token/s over the complete production workload while retaining coherent
output, safe memory use, exact model identity, and CUDA-resident execution?

## 2. Method

The executed command exactly matches the prospective amendment:

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

The RTX 5070 Ti CUDA engine was freshly rebuilt by the passing composite
repository gate immediately before qualification. No converter, download,
build, or other GPU process ran concurrently. At launch the system reported
about 28 GiB available RAM; after tier allocation, the runtime retained about
2.3–2.7 GiB available, above the fixed 1 GiB headroom.

The model uses the exact 122-shard, 93,078-logical/278,152-physical-tensor,
212,634,789,241-byte container. Routed experts remain int4-g128, input/output
and shared experts remain int8, MTP is excluded, and no lower-bit sidecar is
selected.

## 3. Acceptance criteria

The run passes only if:

1. four measured outputs are nonempty and coherent;
2. sustained rate recomputed from 256 completion tokens is at least 0.85
   token/s;
3. hardware, tier, route, and request telemetry are complete;
4. the CUDA expert tier is nonempty;
5. every resident layer-forward uses device MoE with zero host fallback;
6. the exact 18 GiB RAM, 6 GiB VRAM, and 1 GiB headroom profile executes;
7. all four optimized lossless controls are enabled; and
8. predictive prefetch and grouped-2-bit/grouped-3-bit experts remain off.

Failure of throughput stops the ordered Gate-8 sequence before generated-tool
qualification. Gate 9 remains dependent on both checks.

## 4. Results

### 4.1 Warm-up behavior

| Pass | Prompt rates (token/s) |
|---|---|
| warm 1 | 0.467592, 0.602798, 0.593834, 0.555495 |
| warm 2 | 0.644223, 0.665839, 0.663889, 0.583850 |

The second warm pass improves every prompt, confirming that the learned cache
and prewarm mechanisms execute. It does not reproduce the short pilot's
0.867369 rate on a 64-token turn.

### 4.2 Measured behavior

| Prompt | TTFT (s) | Decode rate (token/s) | Decode wall (s) | Cache hit |
|---|---:|---:|---:|---:|
| hash-table complexity | 91.372 | 0.676676 | 94.565 | 58.24% |
| TypeScript grouping | 85.867 | 0.673275 | 95.043 | 57.99% |
| long-running service diagnosis | 95.071 | 0.674737 | 94.836 | 58.23% |
| RAM versus NVMe | 96.193 | 0.530693 | 120.582 | 51.99% |

The token-weighted sustained rate is 0.631963761 token/s; the median turn rate
is 0.674006 and the minimum is 0.530693. The shortfall from 0.85 is
0.218036 token/s, or 25.65%.

Across the four measured decodes, 66,641 expert misses read
445,473,426,675 bytes (414.88 GiB). Decode spends 298.475 seconds in expert
I/O and 103.861 seconds in expert upload/matrix work out of 405.026 seconds
total. The fourth prompt alone reads 123.225 GB and spends 90.965 seconds in
expert I/O, explaining its lower rate.

### 4.3 Execution controls

| Control | Result |
|---|---:|
| startup | 80.602 s |
| RAM expert tier | 2,068 experts / 17.930 GiB |
| VRAM expert tier | 963 experts / 5.999 GiB |
| persistent rings | 3 setups, 238,289 reuses |
| pinned staging | 71,530 batches / 2,612.312 GiB |
| decode protection | 12 refreshes, zero pin fallbacks |
| inter-request prewarm | 3,336 loads / 20.769 GiB |
| resident graph | 46,080 device-MoE, zero host-MoE |
| output/coherence | four nonempty, deterministic coherent responses |

The complete artifact has SHA-256
`83eee0dc45d4fae581f03dafb731b47927d09a3eec63680cda5f839403900fce`.

## 5. Interpretation

The short pilot measured a favorable local working set: one prompt, 16 warm
tokens, and 16 measured tokens. The full run covers four prompts and 768 total
generated tokens. Its measured route sets retain only about 52–58% cache hits
and repeatedly read more than 100 GB per 64-token turn. Persistent ring reuse
removes setup overhead, and pinned staging removes pageable-upload overhead,
but neither changes the fundamental expert working-set capacity.

This result validates the decision not to promote 0.867369 directly. The
optimized path improves the original four-prompt baseline from 0.527355 to
0.631964 token/s, a 19.84% gain, but does not meet the revised product
requirement.

## 6. Decision and downstream disposition

The revised Gate-8 tier criterion fails. Per the frozen ordering:

- the new qualification artifact is retained as negative evidence;
- Ornith397 generated-tool qualification is not run;
- the Gate-9 ten-step Ornith35/397 production sequence is not authorized;
- the strict release audit remains 7/15 with eight failures; and
- `v0.1` remains untagged.

Gate 8 is closed negative at both the original and owner-revised thresholds.
No additional residency retry or further threshold change is implied.

## 7. Limitations

- The result is specific to the reference WSL/NVMe/RTX 5070 Ti system and its
  retained memory headroom.
- Four prompts cannot characterize every workload, but they are the frozen
  production sample and include distinct route patterns.
- The experiment evaluates one-slot decode; Gate-9 continuous batching was
  intentionally not run after the prerequisite failed.
- A materially larger memory target or new expert representation could alter
  the capacity boundary and would require a new protocol.
