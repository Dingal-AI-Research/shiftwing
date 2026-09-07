# Phase 12 preflight 06: frozen expert-map cold-start preload

Date: 2026-08-21

## Owner's design

Preload the expert cache on cold start so a fresh process does not have to
discover the hot experts through warmup:

1. Run a variety of prompts and record which experts are hot.
2. Save that as a static artifact rather than rewriting it at every exit, so a
   measured hot set simply sits there ready.
3. Preload from it on every cold start.

The intent is sound and the mechanism now exists. One quantity in it has to be
restated against measurement, and the expected benefit is narrower than it
first appears.

## What "80% of the experts" can and cannot mean

The cache cannot hold 80% of the experts. Ornith397 has 30,720 routed experts;
80% of them is 24,576, roughly 125 GB in q3 against a 24 GiB cache of 18 GiB
RAM plus 6 GiB VRAM. The q3 run holds **3,948 experts, 12.85%**.

Nor can it cover 80% of routing *mass*. The concentration curve, measured from
the preserved control trials, is:

| top N% of experts | routing mass covered |
|---|---|
| 5% | 30.8% |
| 12.85% (current cache) | 51.8% |
| 20% | 65.8% |
| 30% | 78.1% |
| 50% | 93.5% |

Covering 80% needs about 30% of all experts, roughly twice the cache. The
achievable target is therefore: **fill 100% of the cache slots with the hottest
experts**, which covers about 52% of routing mass and is the ceiling for this
cache size. That is the form the candidate takes.

## Expected benefit: cold start only

Preloading cannot raise steady-state throughput, because the existing LRU and
decode-protect machinery already converges to near-optimal cache *contents*:

| trial | cache covers | perfect same-size cache | headroom |
|---|---|---|---|
| 1 (coldest) | 47.31% | 52.53% | **5.22 pp** |
| 2 | 50.49% | 52.27% | 1.78 pp |
| 3 | 51.61% | 53.11% | 1.50 pp |
| 4 | 51.86% | 53.38% | 1.52 pp |
| 5 | 51.87% | 53.39% | 1.52 pp |

Once warm there is ~1.5 pp left to win, so no cache-content strategy can move
sustained tok/s meaningfully. What preloading buys is arriving at the trial-3
state on trial 1: the 5.22 pp gap, worth roughly the 0.773577249 to
0.838411793 tok/s spread plus a large improvement on early-turn TTFT. That is a
cold-start and first-impression win, not a throughput win, and it should be
claimed as such.

The real throughput lever remains cache capacity measured in experts, which is
why q3 (3,948 experts) beats int4 (3,167) in the same memory.

## Mechanism (implemented, unqualified)

The engine already seeded from a persisted heat map under `AUTOPIN`, but the
map never survived: it was written only from `atexit`, and a graceful shutdown
that overruns its wait is escalated to `SIGTERM`, which runs no `atexit`
handler. No `expert_map.bin` existed after a normal run.

- `expert_map_checkpoint` persists the map on turn boundaries, so persistence
  no longer depends on a clean exit. It is **opt-in** via `EMAP_SAVE_EVERY=N`
  so that rebuilding the engine cannot change measured behavior.
- `EMAP_FREEZE=1` pins the map as a read-only input.
- `qualify_tiered_model.py --expert-map PATH` seeds read-only from a frozen map
  and records its path, size, and SHA-256 in the artifact. Without the flag the
  qualifier sets `AUTOPIN=0` and no map I/O happens at all, which reproduces
  the historical baseline exactly and makes trials mutually independent.

That last point is the reason the freeze exists. A map that rewrites itself is
hidden mutable state: trial 1 would create it, trials 2-5 would seed from it,
each trial would look faster than the last, and the baseline would be silently
invalid. This was nearly introduced by defaulting checkpointing to on.

## Preregistered protocol

1. Build the map from a **diverse corpus that excludes the four benchmark
   prompts**. Seeding from the same prompts that are then measured is
   test-set leakage: it would report a large gain that real traffic never sees.
2. Freeze the map, record its SHA-256, and treat it as a bound input.
3. Compare with `run_paired_perf_trials.py`, five pairs, alternating AB/BA:
   control is the current qualifier line; candidate adds `--expert-map`.
4. Because the effect is concentrated in the first turns, the preregistered
   metric is **cold-start TTFT and first-turn decode**, with warm sustained
   tok/s reported as a non-regression check rather than the headline.
5. Promotion still requires identical outputs, no per-prompt slowdown, zero
   host-MoE fallback, and the paired bounds.

Point 4 differs from candidates 1 and 2 deliberately. Running the standard two
warmup passes would erase the very effect being measured, since warmup is what
preloading replaces.

## Status

Mechanism implemented and unit-tested on the tiny fixture; four tests in
`c/tests/test_expert_map_persistence.py` cover checkpoint-survives-kill,
load-on-next-process, freeze-is-read-only, and opt-in cadence. No profiling
corpus has been built and no real-model measurement exists, so no cold-start
claim is made here.
