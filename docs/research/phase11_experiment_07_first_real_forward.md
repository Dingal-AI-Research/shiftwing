# Phase 11 experiment 07: first real 43-layer CUDA forward

Date: 2026-08-17

## Decision

The first manifest-bound CUDA forward through all 43 base layers of the
converted DeepSeek-V4-Flash-0731 checkpoint passes the narrow correctness
smoke. It does not pass performance qualification and does not authorize
serving or promotion.

With BOS token 0 as input, the runtime returns finite logits with top token 5
and logit 16.258379. It loads exactly 258 routed experts, the expected six
experts for each of 43 layers. It attributes 667 CUDA dense calls,
12,295,250,140 bytes read from storage, and 12,297,913,712 bytes uploaded to
CUDA. The process completes without OOM.

Initialization takes 334.191103 seconds. The first cold decode takes
118.610136 seconds, or 0.008431 tok/s. Those values are diagnostic failures
against the objective to improve both throughput and TTFT over the fresh
Ornith397 control.

## Identity and command

- Source: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Repository commit: `35d9ab86fa2b8f8acbf56c8438413dfb08f2a2a7`
- Branch: `prefill-throughput-and-serve-fixes`
- Manifest SHA-256:
  `468d29fd3262af88ec4c31ef29a94e62631387475917ec7bdb4da9d0441e4d86`
- Binary SHA-256:
  `19666f1188aeb498b0619c8dcdddcefd5db9d216ed670d007902818749c00b46`
- Entry-point SHA-256:
  `3786a9aa4a225800a4ec16513d7f5d15136b3048e34d8896f49562cd811ee599`

```sh
make -C c CUDA=1 CUDA_ARCH=native deepseek_v4 test-cuda
SNAP=/home/dinga/Projects/colib/c/deepseek-v4-flash-0731 \
  CTX=16384 DSV4_SMOKE_CONTEXT=128 DSV4_SMOKE_TOKEN=0 \
  COLI_CUDA=1 RAM_GB=6 CUDA_EXPERT_GB=4 c/deepseek_v4
```

The JSON evidence records the compiler, toolkit, driver, WSL kernel, hardware,
source hashes, exact environment, counters, and acceptance result:
`docs/research/artifacts/deepseek_v4_first_real_forward.json`.

## Controls

Before the real forward:

- the compiled load-only check bound all 72,317 records in 91 segments;
- the complete native CPU suite passed;
- the SM120 CUDA suite passed, including exact generic FP8/FP4 fixtures,
  the actual 4096x2048 top-6 expert shape, and the one-layer CUDA runtime;
- the machine had 28 GiB host memory available, 15,995 MiB VRAM free, and
  574 GiB filesystem space free.

The run used a 6 GiB bounded host expert-cache budget and a 4 GiB bounded
device expert-cache budget. The 128-token smoke context avoids confusing
context-state allocation with the model-residency measurement; normal serving
remains configured for 16,384 and capped at 65,536.

## Interpretation and next gate

The output proves that the native converted records, strict dense arena,
real expert descriptors, CUDA dense path, CUDA expert cache, 43-layer state
schedule, and output head can complete together on the target 16 GiB GPU. It
does not prove model quality.

The counters split the immediate problem into two parts:

1. The 8.85 GB dense arena takes several minutes to assemble before upload,
   which would dominate a process-fresh TTFT unless startup is redesigned.
2. Cold decode reads and uploads 258 expert payloads, but the observed
   one-core saturation and low GPU duty cycle show substantial host work
   between CUDA calls.

The next controlled run must retain the process, decode additional tokens with
the same resident dense arena and warmed expert caches, and report each token
separately. Profiling should then target the host stages that dominate the
warm path. No mux, paired AB/BA, DSpark auto-enable, or default change is
authorized from this smoke.
