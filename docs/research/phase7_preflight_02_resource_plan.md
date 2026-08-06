# Phase 7 Preflight 2: 397B Tier and State Resource Plan

**Project:** colib
**Target:** Qwen3.5-397B-A17B
**Hardware budget:** 16 GiB VRAM, 32 GiB RAM, 868 GiB free disk
**Date:** 25 July 2026
**Status:** Planner implemented; Gate-7 launch profile selected

## Abstract

This study replaces the roadmap's informal 397B memory estimates with an
executable resource planner. The initial proposal—four slots, MTP retained,
and a 7 GiB CUDA expert cache—requires an estimated 23.65 GiB VRAM and fails
before launch. The main cause is the approximately 6.13 GiB int8 MTP block,
which contains a complete 512-expert layer.

The selected Gate-7 profile omits MTP, uses one 4,096-token slot, a 6.0 GiB
CUDA expert cache, 18 GiB RAM cache, and 1.0 GiB post-context headroom. It
requires 14.20 GiB after CUDA context creation, 26.20 GiB RAM, and 222.43 GiB
conversion disk space. A later four-slot server profile fits by reducing the
CUDA expert cache to 5.0 GiB.

The RAM value corrects the first version of this report. That version counted
the expert cache and runtime state but omitted the 6.78 GiB of dense and shared
weights retained by the present C loader. The executable planner, C launch
guard, tests, and this report now use the same resident-memory definition.

## 1. Method

`c/tools/resource_plan.py` derives storage from the model configuration and
the converter's exact precision policy:

- routed experts: grouped int4 with one fp32 scale per 128 columns;
- shared experts, embeddings, LM head, and selected outputs: int8;
- routers, norms, recurrent constants, and scalar gates: fp32;
- optional MTP matrices and experts: int8.

It separately calculates:

- converted container bytes;
- fixed Gated DeltaNet recurrent and convolution state;
- full-attention and MTP KV growth;
- CUDA and RAM expert-cache capacity;
- cold routed-expert bytes per token;
- conversion disk requirement including one source shard and a reserve;
- explicit disk, RAM, and VRAM pass/fail checks.

The RAM requirement is:

\[
\text{dense and shared weights}+\text{expert cache}+\text{state}+
\text{runtime headroom}.
\]

This is intentionally a resident-set upper bound for the current loader; CUDA
preloading does not free the corresponding host matrices.

Unit tests verify the 45/15 linear/full layer split, recurrent-state geometry,
KV formula, MTP increment, and honest failure under deliberately undersized
resources.

The C runtime now enforces the plan as well:

- `RAM_GB` derives the per-layer expert-slot count from the exact packed size;
- `[RAM_PLAN]` reports dense bytes, cache budget, state, headroom, physical
  memory, expert size, and resulting slots;
- startup aborts before dense loading when the RAM plan exceeds physical
  memory;
- `[VRAM_PLAN]` queries free/total device memory after CUDA context creation
  and aborts before preload when dense + expert cache + state + headroom does
  not fit.

A tiny-container integration test proves both successful cache derivation and
the deliberate 100 GiB RAM rejection.

## 2. Weight storage

| Component | Bytes | GiB |
|---|---:|---:|
| routed experts | 205,353,216,000 | 191.25 |
| shared experts | 757,432,500 | 0.71 |
| dense core without MTP | 6,524,140,741 | 6.08 |
| **text container without MTP** | **212,634,789,241** | **198.03** |
| optional int8 MTP increment | ~6.58 billion | ~6.13 |

Each routed expert occupies 6,684,675 bytes. A 6.0 GiB CUDA cache can hold
approximately 963 experts globally, while a 5.0 GiB cache can hold about 803.
The 18 GiB host cache holds 48 experts per layer. Treating the CUDA slots as
uniformly distributed gives 64.05 nominal slots per layer, or 12.51% of the
512 experts. This is a capacity bound, not a predicted hit rate, because the
two cache tiers can overlap.

The 198.03 GiB total exactly matches the independent metadata-only converter
dry-run, strengthening confidence that both calculations implement the same
precision map.

## 3. Runtime state

For one slot:

\[
45 \times 64 \times 128 \times 128 \times 4
=188{,}743{,}680\ \text{bytes}
\]

of recurrent GDN matrices are required. Convolution state adds 8,847,360
bytes. The 15 full-attention layers add 61,440 bytes per token at fp32 KV.

At 4,096 tokens:

| Slots | Fixed recurrent + conv | KV | Total state |
|---:|---:|---:|---:|
| 1 | 0.184 GiB | 0.234 GiB | 0.42 GiB |
| 4 | 0.736 GiB | 0.938 GiB | 1.67 GiB |

Retaining MTP adds 4,096 bytes per token per slot and, more importantly, about
6.13 GiB of int8 weights. It is excluded from the first 397B gate.

## 4. Evaluated profiles

| Profile | MTP | Slots | CUDA experts | Headroom | VRAM required | Result |
|---|---:|---:|---:|---:|---:|---|
| original roadmap | yes | 4 | 7.0 GiB | 2.0 GiB | 23.65 GiB | fail |
| no-MTP original cache | no | 1 | 7.0 GiB | 2.0 GiB | 16.20 GiB | fail |
| **Gate 7** | **no** | **1** | **6.0 GiB** | **1.0 GiB** | **14.20 GiB** | **pass** |
| later server | no | 4 | 5.0 GiB | 1.0 GiB | 14.46 GiB | pass |

The pass margin is intentionally modest because the calculation already
reserves explicit runtime headroom. Actual allocator telemetry and a hard
launch guard remain mandatory.

## 5. Disk and RAM

With a 198.03 GiB output, 4.4 GiB largest source shard, and 20 GiB reserve,
streaming conversion requires 222.43 GiB free disk. The observed 868 GiB
passes with wide margin.

The Gate-7 RAM plan uses:

- 6.78 GiB retained dense and shared weights;
- 18 GiB expert cache;
- 0.42 GiB model state;
- 1.0 GiB post-context runtime headroom.

The corrected 26.20 GiB estimate passes the observed 29.38 GiB WSL
physical-memory guard with 3.18 GiB of calculated margin. This is substantially
less margin than the first report claimed, so increasing `RAM_GB` before
full-model telemetry is not approved.

## 6. Decision

Use a text-only, no-MTP 397B container for Gate 7. Configure:

```text
KV_SLOTS=1
CTX=4096
CUDA_EXPERT_GB=6
CUDA_HEADROOM_GB=1
RAM_GB=18
```

Do not restore MTP until its expert layer can remain tiered instead of pinned
or a larger GPU is available. Phase 9 may use four slots only with an
approximately 5.0 GiB CUDA expert cache and runtime telemetry confirming the
calculated state.
