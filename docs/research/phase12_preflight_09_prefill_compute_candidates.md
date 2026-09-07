# Phase 12 preflight 09: prefill compute candidates for 10-16k context

Date: 2026-08-21

## Goal and gap

Target use is a LocalForge code reviewer taking a git diff plus the original
prompt, 10,000-16,000 tokens on the first request, with about 10 minutes per
review accepted because the inference is free.

Measured time to first token, `--cuda-expert-gb 4`, `--warmup-passes 0`:

| prompt tokens | TTFT | artifact |
|---|---|---|
| 38 | 0.7 min | `flagcheck.json` |
| 2,335 | **11.2 min** | `attn_cuda1.json` |
| 8,451 | **33.0 min** | `split_telemetry.json` |

Add decode: a 400-token review at ~0.8 tok/s is another ~8 minutes. So a 10k
review costs about **41 minutes** today, and 16k projects to **over an hour**.
The gap to target is 4x at 10k and about 7x at 16k.

**No optimization has been applied yet.** Every number here is the unmodified
engine. What exists so far is diagnosis and the ability to test.

## Where the 33 minutes go (8,451 tokens)

| stage | minutes | share |
|---|---|---|
| expert matmul, cold experts on CPU | 14.0 | 42.6% |
| GDN, 45 linear-attention layers | 9.0 | 27.3% |
| full attention, 15 layers | 6.8 | 20.7% |
| expert disk I/O | 3.0 | 9.0% |

Prefill is **91% compute, 9% storage**, which retired the storage-side
candidates 4 and 5 for this workload. It also reads 135.4 GB against 157.1 GB
for one full pass, so the expert-major loop already reads each expert once.

Disk time is **178.5 s at 2,335 tokens and 178.0 s at 8,451** - flat. The
expert set saturates by roughly 2k tokens, so past that point extra context
costs pure compute. This is also why the frozen expert-map preload does
nothing for a long first request: prefill needs every expert regardless of
which are hot.

## Infrastructure already in place

- **Split timers.** `attention_s` was `prof_gdn + prof_attn` summed, leaving
  953.6 s unattributable. `PERF`/`DPERF` now emit both parts as a trailing
  pair, `openai_server.py` parses them by row length, and
  `docs/serve_protocol.md` documents the layout. Values reconcile exactly.
- **Recorded flags.** `qualify_tiered_model.py` gained `--cuda-attn` and
  `--cuda-spec-gdn`. These *must* be flags: `isolated_engine_env` strips 102
  engine control keys, so an ambient `CUDA_ATTN=0` or `CUDA_SPEC_GDN=1` never
  reaches the engine. Two experiments were run and wasted before this was
  understood; both arms were byte-identical, which is the signature of a
  variable that never applied. Recorded flags also land in the artifact's
  `configuration`, so a result is reproducible rather than ambient.

## Candidate A: cold experts on CUDA during prefill (14.0 min)

The largest slice. `moe_prefill_grouped` deliberately keeps cold experts on the
host: *"Cold experts retain the once-per-layer CPU path: uploading prompt-wide
routes churned the bounded device LRU and was measured 2.2x slower."* Only
already-resident atlas experts execute on CUDA.

That 2.2x was measured at short prompts, where an uploaded expert serves a
couple of tokens and the transfer never amortizes. At 8,451 tokens each expert
serves about 165 tokens per layer, so the upload is paid once against a
165-row GEMM. The economics plausibly invert, and 26,496 of 30,216 prefill
expert activations are cold.

Needs **code**, not a flag: a prefill-only path that uploads a cold expert,
executes the batched GEMM on device, and releases it without disturbing the
decode LRU. Gate it on batch size so short prompts keep the current behavior,
and put it behind a recorded flag so the A/B can measure it.

Risk: device cache churn is exactly what the original measurement warned
about; the mitigation is that prefill uploads should not enter the decode LRU
at all.

## Candidate B: GDN prefill on the CUDA block kernel (9.0 min)

`coli_cuda_gdn_block_q4_f16` exists, this model's GDN weights are all
`qtype 4` so `cuda_gdn_eligible` passes, and `--cuda-spec-gdn 1` now reaches
the engine. Verified working: `GDN-calls` rose from 180 to 1,800 on a 38-token
prompt, and the artifact records `cuda_spec_gdn: true`.

Ready to measure with no further code. Three cautions:

- The path is reached through `cuda_spec_gdn_enabled()`, named for speculative
  decoding, and `cuda_gdn_block_try` failure calls `die()` rather than falling
  back to the host path.
- It is not established as bit-identical. The CPU chunked GDN path is already
  documented as *not* bit-identical to the sequential form, with a 1e-4 gate.
  Per the owner's decision, a non-bit-exact path is acceptable only with its
  divergence measured against that gate and the generated text compared.
- `GDN-calls=1800` on 40 positions is 45 layers x 40, i.e. one call per
  position rather than one per block. If that holds at long context the gain
  may be much smaller than the 9.0 minutes suggests. Check the call count
  before trusting the result.

## Candidate C: full-attention prefill (6.8 min)

Diagnosis is **incomplete** and the first proposed fix was withdrawn.

Established: attention scales quadratically (42.4 s at 2,335 tokens, 410.2 s
at 8,451; 3.6x tokens gave 9.7x time against 13.1x for pure quadratic), and
410 s is close to a CPU estimate of 351 s for its 35.1 TFLOP, while the same
work on this GPU should take about 0.9 s.

Not established: whether the CUDA attention path is running at all. The
`GQA-calls` counter matched `15 layers x tokens` exactly, suggesting per-token
CUDA invocation, but the counter was identical with `CUDA_ATTN=0` - which is
now known to be because the variable was stripped, so that test proved
nothing.

An earlier proposal to batch the q/k/v/o projections with `qmat_mul_batch`,
mirroring the GDN fix, was **withdrawn**: if attention is already on CUDA per
token, batching projections on the host moves work off the GPU and would
likely be slower. Three further mechanisms were considered and rejected on
arithmetic: KV cache re-upload (`qwen.c:755`, should not trigger in sequential
prefill), attention weights being evicted (only `.experts.` keys are subject
to the LRU), and `calloc` churn before the CUDA early return (a real waste of
~1.25 MB per call including a 1.08 MB `scores` buffer the CUDA path never
touches, but worth seconds, not minutes).

Next step is one measurement, not another hypothesis: `--cuda-attn 0` versus
`1` at 2,335 tokens, now that the flag applies.

## Measurement protocol

Use the 2,335-token prompt for iteration - about 17 minutes per arm against 35
at 8,451 - and confirm the winner at 8,451. Every arm records its flags in the
artifact.

```sh
./.venv/bin/python c/tools/qualify_tiered_model.py \
  --model c/ornith397 --engine c/qwen \
  --prompt-file {2335-token prompt} \
  --warmup-passes 0 --measured-passes 1 --max-tokens 4 \
  --context 4096 --threads 8 --expert-ram-gb 18 --cuda-expert-gb 6 \
  --minimum-tps 0.0001 --expert-q3 1 --cuda-events \
  --cuda-attn {0,1} --cuda-spec-gdn {0,1} \
  --output {artifact}
```

Order: C first (flag only, and it settles whether attention is even on the
GPU), then B (flag only), then A (needs code). Read `gdn_s` and `attn_s` from
the artifact, and check `GDN-calls`/`GQA-calls` in the log to confirm the path
actually engaged before believing any timing.

At 16k, `--cuda-expert-gb 4` is mandatory: at 6 GiB the VRAM plan needs about
15.5 GiB against 14.66 GiB free and the engine refuses to start. Even 8k is
marginal at 6 GiB. This costs about a third of GPU-resident experts, so
long-context runs are not directly comparable to the 4k baseline.

## Measured results (2,335 tokens, 2026-08-22)

Four arms, all flags recorded in their artifacts, in
`c/bench/ornith397_prefill_ab/`:

| arm | TTFT | vs baseline | stage that moved |
|---|---|---|---|
| baseline | 646.8 s | - | - |
| `attn_off` | 716.8 s | +10.8% | attention 42.6 -> 127.5 s |
| **`gdn_cuda`** | **614.7 s** | **-5.0%** | **GDN 147.5 -> 56.6 s** |
| `route_atlas` | 668.3 s | +3.3% | expert matmul 280.9 -> 305.0 s |

**Candidate B works: GDN falls 2.6x.** First confirmed improvement.

**Candidate C is closed with no work required.** Disabling CUDA attention made
attention 3x *slower*, which proves the GPU path is already active and already
helping. The earlier reasoning that 410 s "looks like CPU speed" was wrong;
attention is simply genuinely expensive at quadratic cost, and there is nothing
to fix.

**The route-atlas proxy for candidate A made things worse.** Expert matmul rose
and TTFT went up 3.3%, which is exactly the device-cache churn the original
`moe_prefill_grouped` comment warned about. The scratch-buffer design that keeps
prefill uploads out of the decode LRU is therefore mandatory, not an
optimisation.

## Owner decision (2026-08-22): 18 minutes accepted

Extrapolated to 8,451 tokens:

| | now | with B | with A+B |
|---|---|---|---|
| expert matmul | 14.0 | 14.0 | ~4.7 |
| GDN | 9.0 | ~3.5 | ~3.5 |
| attention | 6.8 | 6.8 | 6.8 |
| disk | 3.0 | 3.0 | 3.0 |
| **total (min)** | **33.0** | **27.5** | **~18** |

The owner accepts ~18 minutes per cold review, so candidates A and B are the
committed path and prefix caching (candidate 6) becomes optional rather than
required. Reaching 10 minutes is not possible by making prefill faster:
attention's 6.8 minutes is now proven already optimal, and disk is 3.0. Only
prefix reuse, which avoids paying prefill at all, would get below ~18.

## Ceiling

The three candidates together target 30 of the 33 minutes. If all three
succeed completely, prefill lands around 5-10 minutes rather than seconds:
ingesting 10k tokens through a 397B model on one consumer GPU is a large
amount of arithmetic, and placement work cannot remove it, only stop wasting
it. Whether that clears the 10-minute bar depends on whether that budget was
prefill or prefill plus decode.
