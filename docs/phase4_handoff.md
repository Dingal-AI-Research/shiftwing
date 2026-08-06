# Phase 4 handoff — Gate passed 2026-07-21

Phase 4 is complete through official-model conversion, real CPU inference,
benchmarking, and statistical comparison. Phase 5 subsequently passed; its
CUDA implementation and measurements are in `docs/phase5_cuda.md`.

## Durable model artifacts

- Official source: `Qwen/Qwen3.5-35B-A3B`, pinned commit
  `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- Converted snapshot: `c/qwen35/` (ignored by git), 14 shards,
  19,081,779,712 indexed bytes / 19,090,239,288 filesystem bytes.
- Inventory: 31,333 logical tensors and 62,305 physical payload/scale tensors.
  The loader reports 40 layers (10 full attention, 30 GDN) and 140 f32,
  132 int8, and 30,840 int4 matrices.
- Reference GGUF:
  `c/bench/gguf/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf`, 22,285,080,384 bytes,
  from `bartowski/Qwen_Qwen3.5-35B-A3B-GGUF` commit
  `3d2a22c535a105ce3ea4cc690b4e6a66ddbbf273`.
- llama.cpp reference tools were built under `/tmp/llama.cpp` at commit
  `91d2fc3` (`llama-cli`, `llama-server`, and `llama-perplexity`).

The converter streams one immutable official shard at a time, writes and
fsyncs the corresponding output plus resume state, then removes the source
shard. The completed source cache is empty. The conversion output passed the
full loader-only inventory and shape smoke test.

## Measured results

Compiler: GCC 13.3 with `-O3 -march=native -fopenmp -pthread` on a Ryzen 7
7700X under WSL2.

- Coherent ChatML generation: pass.
- Best 64-token CPU decode: **9.34 tok/s** with
  `OMP_NUM_THREADS=8 EXPERT_RAM=64 PREFETCH_THREADS=4` and copied expert cache.
  Load 14.631 s, one-token prefill 4.369 s, decode 6.852 s. Decode detail:
  GDN 3.073 s, attention 0.454 s, MoE 2.615 s (1.080 s expert loading,
  4,724 misses), LM head 0.687 s, other 0.022 s.
- Resident-all-experts decode: 1.64 tok/s because random expert access over the
  19 GB allocation caused severe WSL page pressure. `EXPERT_RAM=64` is the
  measured default; `COLI_MMAP=1` is optional and was slower (3.55 tok/s).
- Fixed 20-prompt, 64-token teacher-forced next-token comparison:
  **1,231/1,280 = 96.171875%**, with all 20 prompts at or above 85%.
  Free-running agreement is **737/1,280 = 57.578125%** because early
  quantizer-dependent rank flips cascade into different continuations. Result
  JSON (including both complete token streams):
  `c/bench/qwen35_prefix_tf.json` (ignored by git).
- Fixed corpus-window comparison: source 1,024 tokens in two 512-token chunks,
  510 scored tokens. With grouped int4 GEMM and mmap expert views, colib PPL is
  **1.088495505** versus llama.cpp **1.0712**, a **1.6145%** relative delta.
  C scoring speed is 1.281 tok/s. Results:
  `c/bench/qwen35_ppl_gemm_mmap_smoke.json` and the original reference-bearing
  `c/bench/qwen35_ppl_smoke.json` (ignored by git).
- Corpus: Project Gutenberg Alice in Wonderland #11,
  `c/bench/qwen35_eval.txt`, 174,311 bytes, SHA-256
  `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`.
  The first 1,024 token IDs hash to
  `90d23bdd12555dfb041ee5166026b47e1f726d64f3e3a6b3dec0c8603c1f2c5e`.

## Implementation state and known issues

- `c/tools/convert_qwen.py` is the shared quantization implementation and now
  supports resumable full-shard streaming, text-only filtering, expert
  unfusing, mixed precision, revision pinning, manifests, and inventory. It
  materializes the effective fast tokenizer into the output `tokenizer.json`,
  including special tokens injected by `tokenizer_config.json`, so the
  standalone C loader has exact tokenizer semantics.
- `c/qwen.c` performs full Qwen3.5-MoE inference, prefix-batch comparison,
  evaluation, ChatML generation, bounded expert caching, and detailed timing.
- The LM head uses the int8 integer-dot path. A fixed OpenMP bug previously
  allowed worker threads to see a null thread-local activation buffer; the
  implementation now captures the allocated buffer before entering the team,
  with a 512x2048 regression case.
- Evaluation groups tokens by expert, transposes the activation batch, and
  executes packed int4 small GEMMs so each nibble is unpacked once per batch.
  Shared-expert and LM-head operations are also batched. `eval_qwen.py` defaults
  to mmap expert views because evaluation visits experts sequentially;
  `--copy-experts`, `EVAL_GROUPED=0`, and `EVAL_LM_BATCH=N` are fallbacks and
  tuning controls. int8 batching is bit-exact, int4 batch-vs-GEMV differs by at
  most 8.34e-7 in its deterministic unit case, and tiny grouped NLL differs by
  less than 5e-7 at fp32 and int4.
- Cross-quantizer free-running equality is retained as a diagnostic, not a
  gate. On an initially divergent prompt both engines ranked the same five
  tokens at the top; llama.cpp preferred token 760 over 8160 by only 0.35
  log-probability, while colib reversed them. Replaying the reference path gave
  15/16 agreement, motivating the complete teacher-forced measurement without
  changing the approved 85% threshold.
- Final regressions are green: 13 C binaries, 8 Python tests, converter
  self-test, all fp32/int8/int4 tiny oracles at teacher-forced and greedy 32/32,
  and 10,000/10,000 tokenizer encode plus decode-round-trip parity using the
  materialized snapshot tokenizer.

## Current resume order

Phase 5 is now complete; its implementation and measurements are recorded in
`docs/phase5_cuda.md`. Resume at Phase 6 in `PLAN.md`: generate numerical MTP
oracle references, implement the one-layer checkpoint-compatible MTP block,
then connect it to the speculative driver while preserving `MTP=0`.
