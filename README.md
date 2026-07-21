# colib

A from-scratch, pure-C inference engine for the **Qwen3.5 MoE** family and
**Ornith-1.0** (Qwen3.5-based) on consumer hardware — in the spirit of
[colibri](https://github.com/JustVugg/colibri): VRAM, RAM and NVMe treated as
one managed memory hierarchy, int4 weights, expert streaming with a learning
cache, MTP speculative decoding, CPU + CUDA backends, OpenAI-compatible server.

**Status:** paused in Phase 4; Gate 3 is complete and the official 35B model is
converted and running coherently at a measured 9.34 tok/s on CPU. The bounded
perplexity comparison is within 2.50% of llama.cpp, but the fixed greedy-prefix
agreement is only 57.58% versus the required 85%, so Gate 4 is not passed and
CUDA work has not started. See [PLAN.md](PLAN.md) and
[docs/phase4_handoff.md](docs/phase4_handoff.md).

Supported models (planned ladder):

| model | total / active | est. int4 size | target |
|---|---|---|---|
| Qwen/Qwen3.5-35B-A3B | 35B / 3B | ~19 GB | correctness first |
| Qwen/Qwen3.5-397B-A17B | 397B / 17B | ~210 GB | ≥2 tok/s tiered |
| deepreinforce-ai/Ornith-1.0-35B | 35B / 3B | ~19 GB | agentic coding |
| deepreinforce-ai/Ornith-1.0-397B | 397B / 17B | ~210 GB | agentic coding |

Text-only; MoE variants only. Apache-2.0. Vendored colibri components: see
[NOTICE](NOTICE).

## Build

```sh
make qwen        # CPU engine (ARCH=native)
make test-c      # infrastructure + DeltaNet/RoPE/router tests
SNAP=./qwen_tiny TF=1 ./qwen  # 32-token oracle replay (run from c/)
make bench-qmat  # int4-g128 GEMV bandwidth
make bench-stack # synthetic 35B-shaped active expert stack
```

CPU controls: `GDN_CHUNK=0` selects recurrent prompt prefill, `IDOT=0` disables
activation-int8 expert dots, `KV16=1` halves KV-cache storage, `EXPERT_RAM=N`
bounds resident experts per layer, and `PREFETCH_THREADS=N` controls background
file-page prefetch (0–4). fp32 KV and all fast paths enabled are the defaults.

## Convert Qwen3.5-35B-A3B

The Phase-4 converter pins one immutable Hub revision, processes one source
shard at a time, drops vision, unfuses routed experts, and deletes each
downloaded source shard only after the output shard and resume state are
durable. The default text-only mixed-precision container is about 19.1 GB.

```sh
cd c
../.venv/bin/python tools/convert_qwen.py \
  --repo Qwen/Qwen3.5-35B-A3B --outdir qwen35 --dry-run
../.venv/bin/python tools/convert_qwen.py \
  --repo Qwen/Qwen3.5-35B-A3B --outdir qwen35

# Coherent ChatML smoke test and performance counters
SNAP=./qwen35 CHAT=1 TEXT=1 PROF=1 PROMPT='Explain delta attention.' NGEN=64 ./qwen

# Loader-only inventory/shape smoke (no forward pass)
SNAP=./qwen35 LOAD_ONLY=1 ./qwen
```

Use `--mtp` to retain the optional MTP block at int8, `--max-shards N` to
exercise interruption/resume, and `--keep-source` to retain downloaded bf16
shards. `tools/eval_qwen.py` and `tools/compare_qwen_prefix.py` implement the
fixed-corpus perplexity and GGUF prefix-agreement gates. Fetch the pinned,
hash-verified public-domain corpus with `tools/fetch_eval_corpus.py`.
