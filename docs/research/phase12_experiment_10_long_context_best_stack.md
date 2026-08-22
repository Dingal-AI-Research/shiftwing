# Phase 12 experiment 10: 8,451-token best-stack long-context qualification

Date: 2026-08-22

## Outcome

On the preserved 8,451-token LocalForge review fixture, the Shiftwing
best-stack configuration reached the first generated token in
**1,159.299940 seconds (19 minutes 19.300 seconds)**. The hash-bound baseline
needed **1,977.830198 seconds (32 minutes 57.830 seconds)**. This is an
**818.530259-second reduction**, or **41.3853% lower TTFT (1.7061x)**.

The candidate emitted the same four-token text as the baseline,
`The tool mitigates`, with SHA-256
`181415b040160dc8010f7089238d7c1729d361da99046b0bc280ab104312235f`.
The qualifier acceptance gate and its automatic telemetry gates passed.

This is a measured engineering result, not a completed code review and not a
general latency claim. The run had zero warmups, one measured pass, and a
four-token output limit. It tested a stack of changes together rather than a
factorial A/B. It also missed the strict 18-minute TTFT threshold by
**79.299940 seconds**.

## Question

Can the implemented resource-shifting stack bring an 8,451-token cold
prefill on the 397B-parameter Ornith397 mixture-of-experts model close to the
owner's accepted roughly 18-minute boundary on one consumer PC, while
preserving the frozen prompt, model manifests, output, and execution gates?

## Evidence classification

**Measured engineering result, single stacked arm.** The candidate is compared
with the preserved split-telemetry baseline that the harness binds by hash.
The comparison is not an interleaved, repeated, paired experiment. The result
establishes what this complete configuration did in one qualified run; it does
not independently estimate the effect of each switch.

## Frozen workload and machine

| Item | Recorded value |
|---|---|
| Model | Ornith397 q3, 397B total / 17B active MoE |
| Prompt fixture | `c/fixtures/ornith397_longctx_prompt.json` |
| Prompt tokens | 8,451 |
| Context capacity | 16,384 |
| Completion limit | 4 tokens |
| Warmups / measured passes | 0 / 1 |
| CPU | AMD Ryzen 7 7700X, 16 logical cores recorded |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| System RAM | 29.375 GiB recorded |
| VRAM | 15.92 GiB recorded |
| Expert RAM / VRAM budgets | 18.0 / 4.0 GiB |
| Engine threads | 8 |

TTFT is measured after model initialization. Startup took 103.185926 seconds
for the historical baseline artifact and 72.141250 seconds for the candidate,
but those values are **not compared**: the runs were not contemporaneous and
the valid candidate was relaunched immediately after an invalid timed-out
attempt.

## Treatment stack

The candidate retained persistent `io_uring`, pinned uploads, decode
protection and prewarming, q3 expert storage, CUDA attention, and 4 GiB of
expert VRAM. It added or selected the following prefill behavior:

- `prefill_expert_batch = 4`
- `prefill_cache_bypass = true`
- `prefill_load_pipeline = true`
- `prefill_cold_device = 64`
- `cuda_spec_gdn = true`
- `cuda_attn = true`
- `prefetch_threads = 0`
- `q3_route_atlas = false`
- `q3_native = false`

`prefetch_threads = 0` is deliberate. The measured overlap came from the
actual-load prefill pipeline, not the older predictive-prefetch mechanism.
Cache bypass let prompt-wide expert traffic stream through without using the
bounded decode expert cache, while the cold-device threshold sent sufficiently
large expert batches to CUDA scratch storage outside that cache.

## End-to-end result

| Metric | Baseline | Best stack | Change |
|---|---:|---:|---:|
| TTFT | 1,977.830198 s | **1,159.299940 s** | **-818.530259 s (-41.3853%)** |
| TTFT, clock form | 32m 57.830s | **19m 19.300s** | **-13m 38.530s** |
| Full four-token wall time | 2,028.934302 s | 1,211.211960 s | -817.722342 s (-40.3030%) |
| Decode wall | 51.332895 s | 58.578526 s | +7.245631 s (+14.1150%) |
| Reported decode rate | 0.077854 tok/s | 0.068242 tok/s | -12.3462% |
| Output | `The tool mitigates` | `The tool mitigates` | exact match |

The TTFT result is the headline. The decode regression is also real in these
two artifacts and must not be hidden. With only four completion tokens and one
run per configuration, it is a signal for a future paired decode check rather
than a stable throughput estimate.

## Prefill-only stage telemetry

The table subtracts each artifact's recorded decode component from its total
stage timer. These are observed timer deltas, not independent causal effects.
In particular, the candidate intentionally overlaps expert loading and
compute, so stage times cannot be added as if they were disjoint wall-clock
contributions.

| Stage | Baseline | Best stack | Observed change |
|---|---:|---:|---:|
| Expert loading | 177.951768 s | 115.317923 s | -62.633845 s (-35.1971%, 1.5431x) |
| Broad expert MoE / matmul timer | 841.752888 s | 531.698300 s | -310.054588 s (-36.8344%, 1.5831x) |
| GDN | 539.580103 s | 186.329116 s | -353.250987 s (-65.4678%, 2.8958x) |
| Full attention | 410.222391 s | 421.829358 s | +11.606967 s (+2.8294%) |

The broad expert timer includes more than a pure matrix multiply: it spans the
MoE block around routing, shared and routed experts, pipeline consumer work,
and reduction. The table therefore uses the engine's recorded label only as a
stage boundary, not as a hardware-GEMM claim.

## Direct evidence of shifting and overlap

The candidate's actual-load pipeline reported:

| Pipeline counter | Value |
|---|---:|
| Batches | 7,562 |
| Materialized experts | 30,200 |
| Expert bytes | 135,376,090,713 |
| Producer load time | 115.317923 s |
| Consumer compute time | 357.503075 s |
| Consumer wait time | 5.592311 s |
| Pipeline wall time | 363.372839 s |

The directly recorded overlap is
`115.317923 + 357.503075 - 363.372839 = 109.448159 seconds`.
That hid **94.91% of producer load time** behind consumer work. Only
5.592311 seconds was recorded as consumer wait.

The improvement did not come from reading materially fewer expert weights:

| Prefill I/O counter | Baseline | Best stack | Change |
|---|---:|---:|---:|
| Expert misses | 26,496 | 26,483 | -13 |
| Expert bytes read | 135,442,544,256 | 135,376,090,713 | -0.0491% |
| `io_uring` batches | 26,496 | 6,667 | -74.8377% |

Instead, the pipeline grouped submissions and overlapped almost the same byte
volume with compute. The candidate log also records zero direct-I/O fallbacks
and zero `io_uring` fallbacks.

Cold-device engagement was not inferred from the flag. The engine reported
`22,330` cold experts, `4,844,229` routed tokens, and `139.02 GiB` staged.
CUDA GDN engagement reported `380,475` calls, while full-attention engagement
reported `126,825` GQA calls. These counters establish that the intended paths
ran; they do not by themselves apportion the end-to-end saving.

## Exactness and qualification gates

The harness rejects the artifact unless all of the following hold:

1. Candidate configuration and run configuration match their canonical
   hashes.
2. Prompt text, prompt-token count, model manifest, and q3 low-bit manifest
   match the bound baseline.
3. Generated text is nonempty and its SHA-256 matches the baseline output.
4. Qualifier acceptance, complete telemetry, CUDA activity, resident CUDA
   graph activity, and automatic gates are true.
5. The load pipeline reports active with positive batch, expert, byte, load,
   compute, and wall counters.
6. The stderr log proves batch-4 pipeline, cold-device, and CUDA GDN
   engagement.

The valid artifact passed. It records 240 device MoE layers and zero resident
host-MoE fallbacks.

## The invalid first attempt

The first launch used the server's 900-second first-output timeout. It ended
with `FirstModelOutputTimeoutError: no model output within 900000 ms` before a
token was returned. The engine did not report a numerical or CUDA failure.
The harness preserved that attempt as invalid, increased the bound timeout to
2,700,000 ms, and immediately relaunched.

This recovery is operationally correct but experimentally important. It means
the valid arm may have inherited host, NVMe, firmware, or thermal state from
the preceding attempt. Expert reads used direct I/O with zero recorded
fallbacks, but the experiment did not independently reset every relevant
cache or thermal condition. Startup time is therefore excluded from the
performance claim, and the TTFT result requires repeated paired confirmation
before it can be called typical.

## Claim boundary and limitations

- **Supported:** this exact best-stack configuration produced its first token
  after 19m19.300s on the recorded consumer PC and passed its exact-output and
  engagement gates.
- **Supported:** compared with the bound 32m57.830s split-telemetry baseline,
  TTFT was 41.3853% lower and four-token wall time was 40.3030% lower.
- **Supported:** the actual-load pipeline overlapped 109.448159 seconds of its
  own load and compute timers while processing nearly unchanged expert bytes.
- **Not supported:** that every cold run, every 8k prompt, or every 397B model
  will reach 19 minutes.
- **Not supported:** that the stack completes a code review in 19 minutes.
  Only four completion tokens were requested.
- **Not supported:** a separate causal speedup for cache bypass, load
  pipelining, batch four, cold-device CUDA, or CUDA GDN. The experiment did not
  run the factorial controls needed for that attribution.
- **Not supported:** an 18-minute pass. The measured TTFT was 79.299940 seconds
  over the strict threshold.
- **Not supported:** a startup improvement or a stable decode-throughput
  result.

The baseline is a preserved historical artifact rather than an adjacent
control run. There were no warmups, repeats, median, confidence interval, or
AB/BA order control. The next statistical qualification should use repeated
paired arms from a defined machine reset state and should generate enough
tokens to evaluate decode separately.

## Reproduction and identity

Primary artifacts:

- candidate: `c/bench/ornith397_best_stack_longctx/best_stack.json`
- candidate hash record:
  `c/bench/ornith397_best_stack_longctx/best_stack.sha256`
- baseline: `c/bench/ornith397_longctx/split_telemetry.json`
- harness:
  `c/bench/ornith397_best_stack_longctx/run_best_stack_longctx.sh`
- valid logs:
  `c/bench/ornith397_best_stack_longctx/best_stack.stderr.log` and
  `best_stack.stdout.log`
- invalid timeout logs:
  `best_stack.invalid.timeout.20260822T182816.stderr.log` and
  `best_stack.invalid.timeout.20260822T182816.stdout.log`
- controller evidence:
  `c/bench/ornith397_best_stack_longctx/controller.20260822T182816.log`

Key SHA-256 bindings:

| Object | SHA-256 |
|---|---|
| Candidate artifact | `ec394a03e16956854e366032b0b67a8ed2da9a019b599855ed2fd4a6b039cd6a` |
| Split-telemetry baseline | `dba4b65c712e7aa0d0d57fb6206e452ae932683b437af273aeb3f725ee454030` |
| Harness | `0945d12ba264f701d0bd136a5fb0e97844296307f6caa8558cc51c98a37cc011` |
| Engine binary | `9b46505474d41717411593c92b7b648dcdac0e6d8ee5d982a2db3ce1c4ef4062` |
| Engine source | `abb86bf831314a6be3b4fd396c6f3e67d525bebcaf459df2da030b54549bb8d6` |
| Qualifier | `ca979201b143e7c8195d3184611cf4d914dafbc8fb2a92ea44940fe93d691889` |
| Prompt fixture file | `b1ec285c337a8627c44a6dd8c988a9bf761d7fda079e176a5784ba52f0cebba1` |
| Prompt value | `29a279b2b7385972ed65d8dfd1490493df9f6fc6c40095daf8c70139b8b80ff1` |
| Qualifier configuration | `851ace9f58bd39f607f847f92936bee827d9b3f2ac4cb12244265ef5479bd633` |
| Run configuration | `3354d18e9eab5c8f81e5064c7e1c2cbd9ac94f7ec2892646187fd1ba885efa1d` |
| Output text | `181415b040160dc8010f7089238d7c1729d361da99046b0bc280ab104312235f` |

The run recorded branch `prefill-throughput-and-serve-fixes` at repository
HEAD `35d9ab86fa2b8f8acbf56c8438413dfb08f2a2a7`. The working tree was
intentionally dirty, so the content hashes above, not the branch name alone,
are the reproduction boundary.

## Decision

Retain the best stack as the strongest measured long-context configuration.
The result is close enough to the accepted roughly 18-minute operating point
to be useful, but it does not satisfy a strict 18-minute gate and should not be
described as a 19-minute review. Publish it as **19m19s time to first token on
one exact 8,451-token qualification**, with the 33-minute baseline, the
four-token limit, the single-run design, and the stacked-attribution boundary
kept adjacent to the headline.
