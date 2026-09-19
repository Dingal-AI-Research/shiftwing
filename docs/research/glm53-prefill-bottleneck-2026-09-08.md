# GLM prompt-reading bottleneck — 2026-09-08

The isolated test was stopped at the user's request. No performance changes or further model runs were started during this investigation. LocalForge was not started or restarted.

## What the run actually measured

- Same synthetic request as the previous context-limit test: **5,258 prompt tokens**, 64K allocated context, thinking enabled, 8,192-token generation budget.
- This is a context-boundary stress prompt containing **230 repetitive reference cases**, not a replay of the failing production scaffold. This distinction should have been stated before the long rerun. It exposes a real long-prompt bottleneck, but does not establish normal production-review latency.
- Stopped after **2,924.767 seconds / 48m 44.8s**, at **51.968%** reported reading progress. **2,688** tokens were fully processed; the percentage also includes processing within the next 128-token chunk.
- **No thinking or answer tokens**. The first-output hour did not expire. The terminal `engine_error` in the raw stream is a consequence of terminating the owned test engine at the user's request, not a newly discovered spontaneous crash.
- Native process counters recorded **2.301 TB of storage reads**, including loading. That is roughly **856 MB per fully processed prompt token**, with a small loading contribution. These are Linux process I/O counters, not a physical SSD telemetry measurement.
- Native RAM use was about **22.7 GiB**, with **zero process swap**. At the middle of the run it averaged roughly nine logical CPU cores of CPU time. CPU usage alone cannot separate useful work from thread synchronization.

## Where the resources went

| Resource | Observed allocation/work |
| --- | --- |
| Storage | Local int4 model directory is about 194.7 GB. Most expert weights remain here and are repeatedly read. |
| RAM | WSL is capped at 30 GB on a host with approximately 32 GB physical RAM. Expert budget is 14 GB; the engine allocates 23 expert slots per layer, about 13.7 GB total. |
| CPU | Ryzen 7 7700X, 8 cores / 16 hardware threads. This build performs all GLM matrix, attention, and expert computation on CPU. |
| GPU | RTX 5070 Ti, 16 GB. The review uses no GPU backend. After stopping, telemetry showed 14 MiB used and 0% utilization. |

Each of 42 sparse layers chooses 8 experts per token from 288. RAM can retain only 23 per layer. The prompt is processed in chunks of 128 tokens, walking through all layers and then starting the next chunk. Experts are reused within a chunk, but many have to be read again for the next chunk. The source already groups selected experts and reads misses in parallel; neither feature is a new optimization to propose.

This is also a compute problem. The expert loop visits matching tokens one at a time, and the matrix function is called with a batch size of one. Much of the resident attention projection work uses the same pattern. Larger chunks alone reduce storage traffic but do not turn these operations into efficient matrix batches.

The source has a Vulkan path, but the active build lacks `COLI_VULKAN`. Its current matrix-level path is not a qualified implementation of a persistent GPU expert cache or a batched GPU prefill pipeline. A flag alone is not a complete remedy.

## What we have already tried

The prior roadmap records a **436.7-second** small planted-defect review: 100.0 seconds prefill and 336.7 seconds decode. Its prompt is not the current padded 5K request. The 16K context capacity from earlier discussions must not be treated as a measured count of occupied prompt tokens.

- Parallel direct expert reads are already enabled. Earlier measurements found direct reads substantially better than buffered reads on this host.
- `io_uring` variants previously took 593.0, 494.5, and 438.2 seconds versus a 436.7-second baseline; they did not improve that review.
- The custom MTP experiment reported 39% draft acceptance, below its estimated break-even. It addresses generation, which this run never reached.
- Frequency-based expert replacement exists in current source. The old roadmap's blanket claim of uniform expert use is superseded by newer source observations; a new cache-hit measurement would be needed for this long prompt.
- Increasing the first-output limit removed the premature 15-minute stop, without changing execution speed.

## Ranked proposals and predicted gains

These are **engineering estimates**, not benchmark results. Speedups below describe prompt-reading latency unless stated otherwise. They overlap and must not be multiplied together.

| Priority | Method | Estimated gain | Why / limits |
| --- | --- | --- | --- |
| 1 | Increase `GLM53_PREFILL_CHUNK` from 128 to 512, then 1024 if memory permits. | **1.2–2.0× faster reading**; target roughly 2–5× fewer expert bytes read per prompt. | Amortizes each expert read across more tokens. More temporary memory is required, and the present one-token math limits the gain. Removing unused output-score rows can reduce memory pressure for larger chunks. |
| 2 | Implement true batched CPU/GPU math, with resident/shared matrices on GPU and a bounded working area for streamed experts. Evaluate a separate GLM-capable CUDA backend build. | **1.5–3× faster reading** as a planning range. | Reduces repeated weight unpacking and memory scans, and uses the idle GPU. Needs correct batched kernels and transfer scheduling; storage remains a lower bound. The whole model cannot fit in 16 GB VRAM. |
| 3 | Quantize expert weights more aggressively while protecting attention/router precision. | **3-bit: 1.1–1.3×; 2-bit: 1.2–1.6× faster reading**, with potentially larger savings in storage-bound generation. | Fewer bytes to read and more experts per cache budget. Current custom expert kernels accept the int4 format, so this needs conversion/kernel support or a different backend. `GLM53_BITS` changes resident matrices and does not convert expert weights. Review quality must be evaluated. |
| 4 | Build a minimal review packet from the complete task, current scaffold/diff, and relevant evidence; remove duplicated logs/history. Preserve the original task and correctness-critical evidence. | **2–4× less reading time when a 5K packet can validly become 1–2K**. | Conditional on actual redundancy; this padded test has extensive artificial repetition. For repeated reviews, retaining a validated prompt cache can help further, but there is no first-run benefit and model unloads lose in-memory reuse. |
| 5 | Use a GPU-sized reviewer for routine blocking reviews, and reserve GLM for narrower high-risk questions. | **Order-of-magnitude target, roughly 10–30×**, low confidence until a matched quality/latency test. | The existing 27B GPU model previously generated about 50 tok/s on a short prompt, but its matched review prefill and quality are unmeasured. A smaller reviewer changes review capability; reusing the orchestrator also reduces independence. |

For retaining GLM, begin with **1**, then **2**. For a strict few-minute review requirement on the current machine, **5** is the most plausible route. Aggressive quantization alone is unlikely to turn this run into a few-minute review.

The current Unsloth guide offers GLM GGUFs around 93 GB (smallest 1-bit), 109 GB (2-bit), and 120–148 GB (3-bit). Even the smallest exceeds this machine's combined RAM and VRAM before overhead, so a backend change or a lower-bit file will not make the entire model memory-resident here. Their advertised 3.3× improvement mainly concerns long-context generation on a B200; their listed 512-token prefill throughput is essentially unchanged. It is not evidence for a 3.3× gain on this CPU/storage workload. See the [official guide](https://unsloth.ai/docs/models/glm-5.3-flash) and [GLM support PR](https://github.com/ggml-org/llama.cpp/pull/27754). The installed local llama.cpp master lacks the GLM5-Next model implementation, so a separate supported build would be required.

## Additional measurements from the saved stream

Layer-progress timestamps account for about 2,922.7 seconds. The output collapse/normalization/scoring stage accounts for only **23.224 seconds / 0.795%**. Computing only the needed final output row is sensible, especially for temporary memory, but its direct time saving is small.

The eleven sparse-attention-containing layers account for about **47.3%** of measured stage time. This includes their expert computation and I/O, so it is not a measurement of attention alone. New phase timers would be needed to separate disk wait, expert math, resident projections, and attention precisely.

The first 128-token chunk took 119.4 seconds; the final completed chunk took 152.6 seconds. Performance is not perfectly linear with prompt length. Estimates obtained simply by doubling the half-run time should not be presented as a measured full-review duration.

The next useful experiment would be a short, fixed production-shaped prompt with explicit disk-byte and phase counters, comparing chunk sizes under a firm wall-time bound. No further experiment has been started.

## Local evidence

- `c/glm53.c:1271`: expert cache initialization and direct-read defaults.
- `c/glm53.c:1316`: per-layer slot calculation.
- `c/glm53.c:1530`: expert union and cache-sized blocks.
- `c/glm53.c:1664`: expert math loops over matching tokens individually.
- `c/glm53.c:955`: matrix operation and optional Vulkan dispatch.
- `c/glm53.c:2310`: unnecessary per-token output scoring during served prefill.
- `c/glm53.c:2353`: 128-token prefill chunk default.
- `docs/glm53_performance_roadmap.md`: prior seven-minute review, I/O and MTP experiments; some high-level status/cache commentary is stale relative to current code.
- `docs/smoke/glm53-review-one-hour-2026-09-08/`: request, raw SSE, timeline, runtime settings, resource counters, cancellation record, and derived phase analysis.

## Proposal 2 implemented

The standalone GLM CUDA batching backend is built and enabled for the next GLM
launch. A matched 128-token probe measured 102.6s scalar CPU prefill versus
89.2s GPU prefill (1.15× throughput, 13.1% less reading time), below the planning
range above. Cold loading was slower in the GPU run, so the full cold CLI run
did not improve. See [implementation, resource map and measurements](glm53-gpu-prefill-2026-09-08.md).
LocalForge was not restarted; no full review was run.

## Proposal 1 implemented

The default chunk has since increased from 128 to 1024 tokens.
See [larger chunk measurements and controls](glm53-larger-chunks-2026-09-08.md).
The earlier measurements in this document remain historical results.
