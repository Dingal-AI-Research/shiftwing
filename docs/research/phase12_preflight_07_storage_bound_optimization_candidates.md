# Phase 12 preflight 07: storage-bound optimization candidates

Date: 2026-08-21

## Measured basis

Decode cost, summed over the four measured turns of the median preserved
control trial (`ornith397-legacy4-control-trial-03.json`), 256 tokens in
310.6 s = `0.8242 tok/s`:

| decode cost | time | share |
|---|---|---|
| expert disk I/O | 218.1 s | **70.2%** |
| expert matmul | 89.4 s | 28.8% |
| attention | 2.2 s | 0.7% |
| LM head | 0.9 s | 0.3% |

- expert bytes read: `305.8 GB`, i.e. **1,195 MB per decoded token**;
- effective read rate during decode: **1.40 GB/s**;
- decode hit rate `61.05%`, `59,823` misses, **5.11 MB per miss**.

This is a storage-bound workload, and it sets a hard ceiling: removing *all*
disk time leaves 92.5 s, or **2.77 tok/s**. No storage-side change can exceed
about 3.4x, and candidates that attack the same 218.1 s do not multiply.

## Candidate 4: NVMe read-path throughput

1.40 GB/s is low for NVMe, but the filesystem is a VHDX on Windows NTFS, so
the device ceiling may genuinely be near that. This candidate is therefore
**gated on measurement before any implementation work**.

`c/iobench` performs random O_DIRECT reads with configurable block size and
thread count. The queued sweep covers 1, 2, 4, and 8 MB at 1, 2, 4, and 8
threads against a base shard, and answers three questions:

1. **Is there headroom?** The 4-8 MB rows at high thread count give the device
   ceiling for the shape decode actually uses. If it lands near 1.5 GB/s the
   candidate is dead and should be dropped rather than optimized.
2. **Is the engine splitting reads?** A routed expert is three tensors
   (`gate`, `up`, `down`). If the engine issues three ~1.7 MB reads where one
   contiguous 5.11 MB read would serve, the gap between the 1-2 MB rows and
   the 4-8 MB rows is what coalescing would recover.
3. **Is it queue-depth limited?** If throughput scales strongly with threads,
   the fix is io_uring submission depth rather than read size.

Estimated gain **1.3-1.5x (about 1.08-1.27 tok/s)** at 1.5-2x throughput.
Confidence is **low until the sweep lands**; it may be zero.

## Candidate 5: heat-tiered q2 cold tail

Every miss reads 5.11 MB at q3. At 2-bit the same expert is about 3.4 MB, so
demoting the rarely-routed tail cuts the dominant decode cost by roughly a
third while leaving the hot core at q3.

Disk time `218.1 s` to about `145 s`, decode wall `310.6 s` to about `238 s`:
estimated **1.31x, roughly 1.08 tok/s**. Confidence on the speed arithmetic is
**medium-high** because it is a direct proportion; the risk is entirely in
quality.

The machinery already exists: `requantize_expert_q2.py --bits 2` builds the
sidecar and the runtime selects it with `EXPERT_Q2=1`
(`qualify_tiered_model.py --expert-q2 1`). What does not exist is the *mixed*
representation — hot experts at q3, tail at q2 — which is the point of the
candidate, since a uniform q2 model was already considered and rejected in
favour of q3.

Quality risk is real and must not be waved through. Misses are by definition
tail accesses, so q2 would cover about 39% of expert activations. Ornith35
routed int3 already costs 95.546875% teacher-forced agreement and +10.66% PPL;
q2 is more aggressive. Promotion requires the existing teacher-forcing,
perplexity, and coherence gates, not only the paired throughput bounds.

Selecting the tier boundary needs the same frozen heat map as candidate 3, so
the two share a profiling corpus:
`c/fixtures/ornith397_profile_prompts.json`.

## Hardware note (not a code candidate)

The largest single lever is cache capacity, and it is a purchase rather than a
change. The board reports **two DIMM slots, both occupied** by 16 GB DDR5-4800
modules, with a 128 GB maximum, so any upgrade replaces both sticks.

| configuration | expert cache | experts cached | routing mass | est. tok/s |
|---|---|---|---|---|
| today, 2x16 = 32 GB | 18 GiB | 3,948 (12.9%) | 51.8% | 0.824 |
| 2x32 = 64 GB | ~40 GiB | ~9,400 (30%) | ~78% | ~1.50 |
| 2x48 = 96 GB | ~70 GiB | ~15,000 (49%) | ~93% | ~2.13 |
| 2x64 = 128 GB | ~100 GiB | ~21,000 (68%) | ~97% | ~2.40 |

96 GB is where the model stops being storage-bound and the expert matmul
becomes dominant; beyond that the 2.77 tok/s ceiling dominates and returns
diminish sharply. These assume hit rate tracks routing-mass coverage, which
held at the current operating point (61.05% hit against 51.8% mass) but will
not scale perfectly: treat them as plus or minus 15%.

## Status

Both candidates are proposals with no implementation and no real-model
measurement. Candidate 4 is blocked on its own sweep, queued to run
uncontended after the five-trial q3 re-baseline. Candidate 5 is unstarted.
Neither may be promoted without the preregistered paired AB/BA protocol, and
candidate 5 additionally requires the quality gates.
