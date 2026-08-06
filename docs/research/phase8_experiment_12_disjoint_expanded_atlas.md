# Phase 8 Experiment 12: Disjoint Expanded Warm Atlas

**Model:** `deepreinforce-ai/Ornith-1.0-397B-FP8`
**Converted revision:** `8b61f97a8512d9d01bff1a9625c9a16730e115bb`
**Date:** 2026-07-31
**Status:** completed; rejected by the preregistered capacity guard

## Abstract

Two rejected lossless experiments established complementary facts. The exact
16-token decode working set contains 4,779 layer/expert pairs, which does not
fit the ordinary 48-per-layer host cache plus 963 device slots. Dense-host
release safely raises the host cache to 64 per layer, but ordinary independent
replacement leaves 880 device entries duplicated in RAM. It therefore exposes
only 3,923 unique residents, retains 1,500 decode misses, and collapses to
0.133596 token/s under replacement and memory pressure.

This experiment tests one final capacity hypothesis: release verified dense
host shadows, allocate 66 host slots per layer, learn the exact route set from
one deterministic warm turn, and partition that set disjointly across RAM and
VRAM. Device atlas entries are pinned for the next identical turn and are
excluded from RAM. No weight, scale, route, logit, or kernel arithmetic changes.

## 1. Research question

Can an explicitly disjoint, zero-replacement warm atlas eliminate all measured
decode reads and recover the control's approximately 6.88-second CUDA expert
execution interval, thereby reaching at least 2 token/s?

## 2. Capacity and resource hypothesis

One expert occupies 6,684,672 packed bytes. Sixty layers with 66 host slots
consume 24.65332 GiB. With the measured 1.516 GiB retained dense host storage,
0.418 GiB recurrent/KV state, and 1 GiB headroom, the preregistered final plan
is approximately 27.59 GiB on the 29.375 GiB host. The 6 GiB device budget
holds 963 experts.

The physical capacity is 4,923 layer/expert pairs. Ten device slots must remain
available for one routed CUDA transaction, leaving a persistent-atlas capacity
of 4,913, 134 more than the 4,779-pair working set. Raw capacity is not
accepted as proof. After the warm turn, the
engine must compute

```text
needed_vram = sum(max(0, unique_routes_in_layer - 66))
```

and stop before preload if `needed_vram > 953` or if the pinned atlas plus the
ten-slot transaction reserve exceeds the exact device-byte budget. This closes
the per-layer-imbalance error exposed by the
first atlas experiment.

## 3. Method

The implementation is opt-in and fail-closed. It combines the previously
validated dense-host release safety checks with a new disjoint atlas:

1. start with the accepted 18 GiB/48-per-layer host cache;
2. preload and synchronize every eligible dense CUDA matrix;
3. verify device ownership, release only verified host shadows, recompute the
   physical plan, and grow to exactly 66 host slots per layer;
4. record every routed expert in the warm decode, including resident-batch
   routes omitted by the legacy `HITS` visualization;
5. assign each layer's mandatory overflow to VRAM, reserve ten device slots
   for one routed transaction, then move the globally hottest remaining
   entries from RAM to otherwise unused persistent VRAM capacity;
6. materialize the RAM partition, preload and pin the disjoint VRAM partition,
   then verify the union and intersection counts before measured decode;
7. prohibit host eviction, device eviction, and disk loading while the atlas is
   active; an unseen route or CUDA error fails closed; and
8. emit atlas capacity, partition, preload, miss, and CUDA-stage telemetry.

The default engine remains unchanged when the option is absent. MTP and
background prefetch remain disabled in this target-only experiment.

## 4. Controls

Before the real pilot:

1. ordinary CPU/CUDA behavior with the option absent must have no source diff
   relative to the accepted baseline;
2. tiny int4 teacher-forced and greedy checks must remain 32/32;
3. tiny resident mux output must be byte-identical off and on;
4. injected post-release CUDA failure, atlas overflow, unseen route, and device
   eviction must each fail closed;
5. host and device atlas sets must have an empty intersection, and their union
   must equal the recorded route set exactly;
6. startup must retain the safe 18 GiB peak and preserve 1 GiB RAM/VRAM
   headroom after growth;
7. the real run retains the fixed prompt, one warm pass, one measured pass,
   16 tokens, 4,096 context, persistent rings, pinned upload, and resident CUDA
   path used by the 0.867369 control; and
8. CUDA stage events must be enabled so residual expert time is attributable.

## 5. Decision rule

The capacity guard is evaluated before atlas preload. A guard failure rejects
the experiment without a performance claim. If capacity fits, the one-prompt
pilot passes only if:

- emitted text is byte-identical to the accepted control;
- every layer uses device MoE and host-MoE fallback is zero;
- the atlas reports 4,779 union entries, zero intersection, and a complete
  preload inside both exact byte budgets;
- measured decode records zero expert misses, zero expert read bytes, and zero
  atlas violations; and
- sustained decode is at least 2.0 token/s.

Any failure removes the experimental runtime/protocol branch and prohibits the
four-prompt run. A pass authorizes, but does not replace, the frozen four-prompt
Gate-8 qualification and generated-tool gate.

## 6. Scientific limitation

This is deliberately a deterministic warm-workload experiment. It does not
claim cold-prompt acceleration or general coverage for arbitrary prompts. Its
purpose is to test whether the reference hardware can meet the established
decode target when disk and replacement are rigorously removed. If it still
fails, Gate 8 has direct evidence that the current 397B target cannot reach the
threshold through residency policy alone on this hardware.

## 7. Implementation controls

The experimental implementation retained the safe 18 GiB startup, verified
376 persistent CUDA dense matrices, released 5.266 GiB of host shadows, and
grew to exactly 66 host slots per layer. It recorded resident-batch routes in
the decode phase, reserved ten device slots for a live routed transaction, and
computed the exact per-layer overflow before attempting any atlas preload.

On the tiny int4 model, teacher-forced and greedy references remained 32/32.
Two sequential resident-mux turns were byte-identical with the option off and
on. The tiny atlas produced an exact union, zero host/device intersection,
zero misses, and zero violations. An injected post-release CUDA failure exited
with status 2 rather than falling back to released host matrices. The focused
28-test mux, qualifier, and gateway suite passed before the real run.

## 8. Exact capacity result

The real warm turn reproduced the complete 4,779-pair working set. The guard
then rejected activation:

```text
entries=4779 needed-vram=1088 available-vram=953 cap/layer=66 result=reject
```

The result explains why aggregate slot arithmetic was optimistic. Sixty layers
provide 3,960 physical host slots, but only 3,691 can hold members of this
route set at a per-layer cap of 66. The remaining 269 host slots cannot be
transferred to layers with larger working sets. The usable 3,691 host entries
plus 953 safely pinned device entries cover 4,644 pairs, leaving 135 uncovered
experts (902,430,720 bytes, 0.840454 GiB). Even spending the ten transaction
reserve would leave a 125-entry deficit.

The full memory control itself remained valid:

| Quantity | Result |
|---|---:|
| released host shadows | 5.266 GiB |
| retained dense host data | 1.516 GiB |
| host expert allocation | 24.653 GiB |
| recurrent/KV state | 0.418 GiB |
| preregistered headroom | 1.000 GiB |
| planned total | 27.588 GiB / 29.375 GiB |
| observed available RAM at result capture | 1.214 GiB |
| startup | 82.159 s |

## 9. Inactive fallback observations

Because the capacity guard failed, the atlas did not preload or activate. The
following measured turn is therefore an expanded-cache fallback control, not
an atlas performance result. It retained the exact reference text but reached
only 0.292602 token/s, with 3,038 decode misses and 20,308,042,650 read bytes.
The warm fallback reached 0.179281 token/s.

CUDA stage events attribute about 4.55 seconds of the measured 54.65-second
decode to routed/shared kernels and downloads, while the residual expert
interval was 33.96 seconds and measured disk time was 20.29 seconds. This
supports the previous finding that a nearly full host cache adds substantial
replacement/allocation overhead, but it is not evidence about the unexecuted
zero-miss atlas.

## 10. Decision and disposition

The exact capacity guard fails by 135 expert slots, so no preload, zero-miss
claim, or four-prompt qualification is authorized. Per the preregistration,
the experimental runtime, protocol, qualifier, and test branch was removed and
the accepted fallback-capable CUDA source was rebuilt.

This closes residency-only performance remediation negatively for the stated
29.375 GiB RAM / 15.92 GiB VRAM reference machine and the immutable 1 GiB
headroom requirement. It does not claim that the model can never meet 2 tok/s
on larger hardware. It establishes that changing tier placement alone cannot
meet the current Gate-8 threshold safely on this machine. Any continuation now
requires a changed hardware target, a changed performance threshold, or a new
algorithmic/kernel objective rather than another cache-capacity retry.
