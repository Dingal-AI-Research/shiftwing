# Phase 12 experiment 02: expanded-q4 route-atlas candidate

Date: 2026-08-18

## Outcome

The first Ornith397 optimization candidate is implemented and locally
validated. It is not promoted and has no new real-model tok/s or TTFT result
yet because the pinned model reconstruction is still active.

The selected representation is deliberately mixed:

- NVMe and RAM retain the accepted grouped-int3 expert sidecar.
- Atlas-selected VRAM experts expand losslessly to the existing q4 CUDA
  layout.
- Grouped prefill executes already-resident atlas experts with batched CUDA
  GEMM and never uploads a cold expert; misses retain the CPU panel path.
- `Q3_NATIVE=1` remains a separate diagnostic arm and is not enabled by the
  atlas.

This preserves the q3 storage and host-cache capacity benefit while using the
faster, already-qualified q4 device kernels. The loader now accounts the
actual device bytes for native-q3 and expanded-q4 copies separately, so atlas
capacity cannot be overstated.

## Change boundary

`c/qwen.c` no longer makes the route atlas imply native-q3 execution,
uses the selected device representation when computing expert-cache capacity,
and adds a cached-only batched CUDA prefill path. The path reports its batch
and token counts in `Q3ATLAS` telemetry. `c/colib`,
`c/tools/qualify_tiered_model.py`, and
`c/tools/stream_qwen_benchmark.py` expose the two switches independently.
CLI tests cover the expanded-q4 atlas default and the explicit native-q3
override.

DeepSeek runtime/build paths are not part of this candidate. Authoritative
`colib.metrics` response telemetry is retained because the A/B must use
engine TTFT and true decode rate rather than web text-chunk estimates.

## Local validation

- 43 focused mux, CLI, HTTP, qualifier, and runtime-environment tests pass.
- The SM120 CUDA backend suite passes on the RTX 5070 Ti.
- Actual-shape packed-q3 versus expanded-q4 grouped-MoE comparison reports
  maximum difference 0.
- A composed tiny-model seam test loads a grouped-int3 expert, computes a
  three-token CPU reference, reuses the cached expanded-q4 device copy with no
  cold weight upload, and reports `maxdiff=1.1641532e-09` against a
  `0.0020128214` tolerance. Its `prefill-batches` and `prefill-tokens`
  counters both advance exactly once/by three.
- Python compilation and `git diff --check` pass.
- The web metrics parser has 19 passing tests and its production build
  succeeds.

These are implementation and numerical controls, not performance evidence.


The focused CUDA command is reproducible from the standard tiny int4 fixture.
First copy `c/qwen_tiny_i4` to a temporary directory and generate its q3
sidecar with:

```sh
./.venv/bin/python c/tools/requantize_expert_q2.py \
  --snapshot {temporary-fixture} --bits 3 --group-size 128 \
  --iterations 1 --experts-per-file 8 --workers 2 --torch-threads 1
```

Then run:

```sh
SNAP={temporary-fixture} MTP=0 EXPERT_Q3=1 Q3_NATIVE=0 \
  COLI_CUDA=1 CUDA_DENSE=1 CUDA_F16=1 CUDA_EXPERTS=1 \
  CUDA_EXPERT_GB=0.1 CUDA_HEADROOM_GB=1 \
  RAM_GB=0.01 RAM_HEADROOM_GB=1 PREFETCH_THREADS=0 \
  Q3_PREFILL_CACHE_TEST=1 ./c/qwen
```

`MTP=0` is required only because the standard tiny fixture contains MTP base
weights while the deliberately minimal generated q3 sidecar covers base-model
experts. The real Ornith397 q3 sidecar and production protocol retain their
ordinary MTP selection; this fixture limitation does not change that policy.
## Preregistered real-model protocol

Both arms use the pinned Ornith397 source and reconstructed q3 sidecar,
four legacy prompts, 64 decoded tokens, context 4096, eight CPU threads,
18 GB RAM expert cache, 6 GB VRAM expert cache, persistent io_uring,
direct I/O, pinned uploads, decode-route protection and prewarming, CUDA
events, deterministic sampling, and identical warmups.

Control qualifier template:

```sh
./.venv/bin/python c/tools/qualify_tiered_model.py \
  --model c/ornith397 --engine c/qwen \
  --prompt-file c/fixtures/ornith397_legacy_prompts.json \
  --warmup-passes 2 --measured-passes 1 \
  --max-tokens 64 --context 4096 --threads 8 \
  --expert-ram-gb 18 --cuda-expert-gb 6 \
  --minimum-tps 0.70 --uring-persist 1 --pinned-upload 1 \
  --decode-protect 1 --decode-protect-prewarm 1 \
  --expert-q3 1 --q3-route-atlas 0 --q3-native 0 \
  --cuda-events --output {output}
```

Candidate qualifier template is identical except:

```sh
--q3-route-atlas 1 --q3-native 0
```

`c/tools/run_perf_trials.py` binds the engine, prompt fixture, qualifier,
q3 manifest, and model manifest hashes and atomically records five trials per
arm. Launch order alternates AB/BA. A stale running attempt becomes
`interrupted` and resumes at the next unverified boundary.

## Acceptance

Promotion requires all of the following:

1. Five complete paired AB/BA trials with identical outputs and hardware
   state.
2. The 95% lower confidence bound for candidate/control decode tok/s exceeds
   1.0.
3. The 95% upper confidence bound for candidate/control TTFT is below 1.0.
4. No per-prompt slowdown, empty output, OOM, host-MoE fallback, incomplete
   telemetry, session, cancellation, batch, tool, or web regression.
5. Atlas telemetry is active, device capacity matches expanded-q4 bytes, and
   disk reads or cache misses improve consistently.

Native packed-q3 may be measured as a diagnostic third arm, but it cannot
replace either preregistered arm or inherit a passing result.

## Identity

At preregistration:

- repository HEAD:
  `35d9ab86fa2b8f8acbf56c8438413dfb08f2a2a7`
- engine-source fingerprint:
  `4a7e2593485aaf8eead81f2d9f06fcfaea9c1bc4307e07d09ab0aa7075111ebf`
- CUDA engine SHA-256:
  `b47aadfc3ead42b1b31695d52898e36dc4668bef85d2f4f59a4aed0f82f24271`
- legacy fixture SHA-256:
  `c977a1530534b9e7b10cbe91dcdeda2bfca468bcb36b20446cdad70ed05abb38`
- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB, driver 591.86.

The working tree is intentionally dirty. The atomic trial state must capture
the final source fingerprint and binary hash again immediately before launch;
any drift invalidates these preregistration identities.
