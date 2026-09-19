# GLM larger prompt chunks — 2026-09-08

Proposal 1 is implemented alongside the CUDA batching from proposal 2.
The native default is now **1024 prompt tokens per chunk**, up from 128.
The explicit `GLM53_PREFILL_CHUNK` override remains available. LocalForge was
not restarted, and no full review or LocalForge end-to-end run was performed.

## Matched real-model comparison

Both configurations read the **same 1024 tokens** from an actual source-code
excerpt (`c/glm53_gpu.inc`), without repeated padding. One model load was shared:
71.7s load plus 1.1s to populate common GPU weights.
The two cases used fresh sequence state and reset expert identities and usage
counts; the host expert buffer allocations and shared GPU weights were reused.
Most expert reads use the engine's existing direct-I/O path. This design avoids
confounding chunk comparison with the load-time variation seen in the earlier
CPU/GPU experiment. The 512 case ran first; no repeated-trial statistics are claimed.

| Measurement | Chunk 512 | Chunk 1024 |
| --- | ---: | ---: |
| Time to read the 1024-token input | 388.9s | 294.7s |
| Prompt tokens/second | 2.63 | 3.47 |
| Expert bytes read | 313.6 GB | 164.9 GB |
| Expert cache misses | 22,155 | 11,650 |
| Peak sampled process RAM | 22.89 GiB | 23.18 GiB |
| Peak sampled GPU use | 4.61 GiB | 4.72 GiB |
| Maximum process swap | 0 KiB | 0 KiB |

1024-token chunks delivered **1.32× throughput** compared
with 512, using **47.4% fewer expert bytes**.
The next-token choice and the ranking of the top ten candidates matched.
Scores were not bit-identical: maximum final-logit difference was
0.00431266, while the largest change in next-token
probability was 2.26925e-07.
Only this final-position distribution was checked on the real model; this is
not an assessment of complete review quality.

The original 128-token chunk size was not benchmarked on this same input.
The earlier 128-token CPU/GPU probe used a different, shorter input, so its
throughput cannot establish a matched speedup against the new default.
Long-context throughput, concurrent orchestrator memory use and full review
latency are not established by this comparison.

## Why it helps

Within a chunk, the engine reads an expert once and applies it to every token
that selected it. Larger chunks give more tokens a chance to share that read.
The expert cache budget is unchanged at 14 GB configured (23 slots per sparse
layer on this model). Expert quantization is still int4 and the GPU math still
uses the FP32 backend from proposal 2.

Served prefill now produces only the final position's output scores. Previously
it allocated and calculated scores for all positions before throwing away the
unused rows. On this vocabulary (154880 entries), the removed score storage is
about **302 MiB per 512-token chunk** or **604 MiB per 1024-token chunk** on the
host, plus the unused GPU output buffer when the head was batched there.
Final collapse/normalization also operates only on the needed row. The final
unnormalized hidden state used for MTP is preserved. Teacher-forcing and other
callers that need all rows still receive them.

The extra capacity is for each work chunk, not a new context limit. Context
capacity, 90% orchestrator compaction, review timeouts and approval rules are
unchanged. Review reading progress continues to update after each layer,
including partial progress inside the larger chunk; completion is reported
only after that chunk has finished. Single-token answer generation remains on
the existing CPU path.

## Validation and reproduction

- The 1031-token converted-model fixture exercises full chunks and partial
  tails at 128/512/1024, including forced one-slot expert-cache eviction.
- Compact final scores match full teacher-forced scores. Token choices,
  continuation tokens, recurrent state behavior and final MTP hidden state match.
- Progress is monotonic, remains within the prompt size and ends at 100%.
- Default chunk behavior, CPU-only execution, the independent tiny-model
  oracle and native review-stream checks are covered by the saved validation.

```sh
make -C c CUDA=1 glm53 test-glm53-prefill test-glm53
cd c
python3 -m unittest discover -s tests -p 'test_glm53_review_stream.py'
```

`GLM53_PREFILL_CHUNK=512` selects the smaller tested chunk.
`GLM53_PREFILL_CHUNK=128` restores the original chunk size while retaining GPU
batching and compact served outputs. No environment override was present in
LocalForge's `.env`; the next normal GLM launch will therefore use the new
native default. No LocalForge restart was performed.

The isolated benchmark used 16 CPU threads, int4 resident weights, the existing
14 GB expert-cache setting and a 12 GiB CUDA allocation ceiling. Input IDs,
raw progress, resource samples, final logits, commands and results are stored
in `../smoke/glm53-larger-chunks-2026-09-08/`. Each case had an eight-minute
limit and the process had a swap guard; these are benchmark limits and do not
change the production one-hour first-output timeout.
