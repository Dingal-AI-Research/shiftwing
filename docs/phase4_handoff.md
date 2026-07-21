# Phase 4 handoff — paused 2026-07-20

Phase 4 is implemented through official-model conversion, real CPU inference,
benchmarking, and statistical comparison, but **Gate 4 is not passed**. Phase 5
must not begin until the real-model prefix disagreement is diagnosed and the
full fixed-corpus perplexity run completes.

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
- Fixed 20-prompt, 64-token prefix comparison: **737/1,280 = 57.578125%**;
  8/20 prompts reached at least 85%. Result JSON:
  `c/bench/qwen35_prefix.json` (ignored by git).
- Bounded corpus comparison: source 1,024 tokens in two 512-token chunks,
  510 scored tokens. colib PPL **1.097995016**, llama.cpp PPL **1.0712**,
  relative delta **2.5014%** (pass). C scoring speed was 0.799 tok/s. Result:
  `c/bench/qwen35_ppl_smoke.json` (ignored by git).
- Corpus: Project Gutenberg Alice in Wonderland #11,
  `c/bench/qwen35_eval.txt`, 174,311 bytes, SHA-256
  `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`.
  The first 1,024 token IDs hash to
  `90d23bdd12555dfb041ee5166026b47e1f726d64f3e3a6b3dec0c8603c1f2c5e`.

## Implementation state and known issues

- `c/tools/convert_qwen.py` is the shared quantization implementation and now
  supports resumable full-shard streaming, text-only filtering, expert
  unfusing, mixed precision, revision pinning, manifests, and inventory.
- `c/qwen.c` performs full Qwen3.5-MoE inference, prefix-batch comparison,
  evaluation, ChatML generation, bounded expert caching, and detailed timing.
- The LM head uses the int8 integer-dot path. A fixed OpenMP bug previously
  allowed worker threads to see a null thread-local activation buffer; the
  implementation now captures the allocated buffer before entering the team,
  with a 512x2048 regression case.
- Evaluation has an opt-in grouped-MoE prefill path that loads each used expert
  once per layer and is numerically consistent with sequential prefill on the
  tiny model (NLL difference about 3.7e-7). It still executes each routed token
  as a separate GEMV, so the full corpus would take hours. Implement batched
  per-expert matmul before running the full gate.
- The latest grouped-evaluation, mmap-option, and profiling changes compiled
  and the tiny grouped NLL smoke passed. The complete C/Python/tokenizer suite
  was green before those last edits and needs one final rerun on resume.

## Resume order

1. Reproduce one of the early-diverging fixed prompts and use
   `c/tools/compare_acts.py` to locate the first real-model divergence. Check
   model math and the mixed precision map before changing the 85% criterion.
2. Batch the evaluation-only expert matmuls across all tokens assigned to each
   expert; retain the sequential and current grouped paths as correctness
   fallbacks.
3. Re-run `c/tools/compare_qwen_prefix.py` for all 20 prompts and
   `c/tools/eval_qwen.py` on the full pinned corpus.
4. Run `make -C c test-c test-python`, all fp32/int8/int4 32-token tiny
   oracles, and `git diff --check`.
5. Mark Gate 4 passed only if prefix agreement reaches 85%, full-corpus PPL is
   within 5–10%, coherent chat remains good, and decode stays above 8 tok/s.
