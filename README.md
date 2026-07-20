# colib

A from-scratch, pure-C inference engine for the **Qwen3.5 MoE** family and
**Ornith-1.0** (Qwen3.5-based) on consumer hardware — in the spirit of
[colibri](https://github.com/JustVugg/colibri): VRAM, RAM and NVMe treated as
one managed memory hierarchy, int4 weights, expert streaming with a learning
cache, MTP speculative decoding, CPU + CUDA backends, OpenAI-compatible server.

**Status: early development.** See [PLAN.md](PLAN.md) for the phased roadmap,
validation gates, and current status.

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
make test-c      # header/infra tests
```
