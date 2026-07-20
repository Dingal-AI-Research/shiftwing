# PLAN.md — `colib`: a from-scratch C inference engine for Qwen3.5 MoE + Ornith-1.0

> **How to use this file (for any model/human picking up mid-work):** work top to bottom. Each phase has checkboxes and a **GATE** — a measurable validation that MUST pass before the next phase starts. Update the *Current status* block and check boxes as you go. All architecture facts and design decisions are recorded here; you do not need the original conversation. [colibri](https://github.com/JustVugg/colibri) (Apache-2.0) is the design reference — consult it freely (clone it if needed), but **this repo is NOT a fork**: it supports only Qwen3.5 MoE + Ornith-1.0, text-only.

## Current status

- **Active phase:** 1 (oracle + tokenizer parity)
- **Last gate passed:** GATE 0 (2026-07-20: `make test-c` 8/8 green, `make qwen` compiles clean `-Wall -Wextra`)
- **Blockers:** none
- **Machine:** AMD Ryzen 7 7700X (Zen4; AVX-512F/DQ/BW/VL/VNNI/BF16 confirmed), 32 GB RAM, RTX 5070 Ti 16 GB (Blackwell **sm_120**; WSL2 driver 591.86 OK; **nvcc NOT installed** — CUDA toolkit ≥12.8 needed before Phase 5), NVMe: **910 GB free** (recorded 2026-07-20; 397B needs ≥250 GB — OK). Python 3.12.3; **no torch/transformers yet** — create `.venv` in Phase 1 (CPU torch is enough until Phase 5).
- **Colibri reference clone:** `/tmp/claude-1000/-home-dinga-Projects-colib/de80692d-17fa-4193-8ee0-1c2cc8b7db9a/scratchpad/colibri` (scratchpad; if gone: `git clone --depth 1 https://github.com/JustVugg/colibri`).

## Scope (fixed, user-approved)

- Models in order: **Qwen/Qwen3.5-35B-A3B** (correctness target) → **Qwen/Qwen3.5-397B-A17B** (perf target ≥2 tok/s) → **deepreinforce-ai/Ornith-1.0-35B** → **Ornith-1.0-397B**. Text-only (vision tower dropped at conversion). MoE only (no dense 9B/31B).
- Full colibri-parity capability set: int4 container, VRAM/RAM/NVMe expert tiering + learning cache, io_uring/O_DIRECT async I/O, MTP speculative decoding, CPU + CUDA backends (Metal deferred), mux serve protocol + OpenAI-compatible server + web dashboard, token-exact oracle validation.
- Non-goals: GLM/OLMoE support, vision, dense variants, Windows/macOS (keep `compat.h` so ports stay possible).

## Architecture facts (verified from HF configs 2026-07-20; ⚠ items re-verified in Phase 1 against `modeling_qwen3_5_moe.py`)

Both Qwen models are `qwen3_5_moe` (`Qwen3_5MoeForConditionalGeneration` = vision_config + text_config wrapper; we use text_config only). Hybrid: repeating **3× Gated DeltaNet (linear attention) + 1× full attention** (`full_attention_interval: 4`); **every** layer has MoE (`mlp_only_layers: []`).

| | 35B-A3B | 397B-A17B |
|---|---|---|
| layers | 40 = 10×[3 lin + 1 full] | 60 = 15×[3+1] |
| hidden | 2048 | 4096 |
| experts / top-k / expert-inter | 256 / 8 / 512 | 512 / 10 / 1024 |
| shared expert inter | 512 (⚠ `shared_expert_gate` sigmoid — Qwen2/3-MoE lineage) | 1024 |
| full attn | 16 Q / 2 KV heads (GQA), head_dim 256, `attn_output_gate: true` (⚠ gate fusion layout) | 32 Q / 2 KV |
| RoPE (full-attn only) | partial_rotary_factor 0.25 (64 of 256 dims), θ=1e7, **interleaved MRoPE sections [11,11,10]** — text-only ⇒ all position streams equal ⇒ effectively 1D RoPE but dim pairing must match HF exactly (⚠) | same |
| DeltaNet | 16 K-heads×128, 32 V-heads×128, conv k=4, `mamba_ssm_dtype: float32` | 16 K, **64 V**×128 |
| MTP | `mtp_num_hidden_layers: 1`, shared embeddings (⚠ block structure) | same |
| vocab / eps / ctx | 248,320 untied / 1e-6 / 262,144 | same |
| router | softmax-top-k Qwen lineage (⚠ `norm_topk_prob`) — NOT GLM's sigmoid+bias | same |

**Gated DeltaNet per-layer weights:** `in_proj_qkvz`, `in_proj_ba`, depthwise `conv1d` (k=4, SiLU, over concat(q,k,v)), `A_log`, `dt_bias`, gated RMSNorm (`norm`), `out_proj`. Decode recurrence per V-head (state S = 128×128 **fp32**; K-heads repeat-interleaved to V-heads):

```
g = exp(-exp(A_log)·softplus(a + dt_bias));  β = σ(b);  q̃,k̃ = L2-normalized
S ← g·S + k̃ ⊗ (β·(v − (g·S)ᵀ k̃));   o = q̃ᵀ S / √d_k
out = RMSNorm(o) · SiLU(z) → out_proj
```

Prefill fast path = chunked WY decomposition (chunk 64); sequential recurrence is the always-correct fallback (`GDN_CHUNK=0`). Consequence: 30/40 (45/60) layers have **no KV cache** — fixed recurrent state (S + 4-token conv tail) instead. KV cache exists only for the 10/15 full-attn layers (2 KV heads × 256 → ~61 KB/token fp32 across all layers at 397B scale, ~41 KB at 35B).

**Ornith-1.0:** config verified **byte-identical architecture** to Qwen3.5 same-size (diffs: eos 248046, pad 248044, cosmetic flags). Ornith = model registry entry + tokenizer/chat-template/eos handling only (Phase 8). A `-1M` YaRN long-context variant exists (stretch; YaRN affects full-attn layers only).

**Validation references:** HF transformers `qwen3_5_moe` (PIN the version and hard-gate it in the oracle script — colibri's `make_glm_oracle.py:21-33` shows why: a too-old transformers silently produces a wrong oracle); llama.cpp already supports the arch (GGUFs exist for all four models) — statistical cross-check at real scale; tiny-random oracle model for token-exact gates.

## Key colibri findings (what to reuse vs rewrite)

- Colibri precedent: a new arch = a new **self-contained `.c`** reusing header-only infra (`olmoe.c` is the proof). No plugin registry exists; don't invent one.
- Residual seam to replicate (glm.c:3926-3936): `rmsnorm → mixer → +residual → rmsnorm → MoE → +residual`; per-layer type array precedent = `idx_type`.
- GQA attention structural reference: `olmoe.c:271-317` (adapt: 2 KV heads, head_dim 256, partial interleaved RoPE, output gate).
- Expert streaming (`expert_load_impl`, glm.c:1762): coalesced {gate,up,down} slab reads keyed (layer,eid), O_DIRECT, prefetch threads, pinned hot store, per-layer LRU + heat counters — generic over exactly Qwen's SwiGLU expert shape. Port it.
- Container convention (KEEP IDENTICAL): dir of `out-NNNNN.safetensors`; quantized tensor = U8-packed `NAME` + F32 scales `NAME.qs`; fmt auto-detected from byte counts (int8/row, int4/row, int2, int4-grouped w/ group size from `.qs` length); norms/router/bias f32; config/tokenizer JSONs copied alongside.
- Spec-decode driver (glm.c:4439-4518) is logit-only and architecture-agnostic; drafting optional (MTP → n-gram → grammar). Port it.
- Oracle/TF mode (glm.c:6575-6625): `SNAP=<tiny> TF=1` replays `ref.json` teacher-forced, prints `[ORACLE] n/32`; `DEBUG_LOGITS=1` dumps top-5. Replicate exactly.
- Serve wire protocol (`docs/serve_protocol.md`): keep **byte-identical** (READY sentinel, SUBMIT/CANCEL, DATA/DONE/ERROR, TIERS/HWINFO/EMAP/HITS/PERF/ENTROPY/GPUS telemetry) so vendored `openai_server.py` + `web/` drop in.
- **MTP head must be int8** — int4 MTP gives ~0% acceptance (colibri issue #8; `repair_mtp_int8.py` exists for damaged containers).
- Env-var surface: keep colibri names verbatim (`SNAP TF REF NGEN TEMP NUCLEUS TOPK SEED CTX KV_SLOTS KVSAVE MTP DRAFT SERVE SERVE_BATCH SERVE_TOPK PIPE URING DIRECT PREFETCH PIN PIN_GB AUTOPIN RAM_GB PROF IDOT COLI_CUDA COLI_GPU CUDA_DENSE CUDA_EXPERT_GB COLI_MMAP MLOCK`). Drop GLM-only ones (`ABSORB`, DSA/`CACHE_ROUTE`).

## Repo layout

```
colib/
├── PLAN.md  README.md  LICENSE (Apache-2.0)  NOTICE (credits colibri/JustVugg)  Makefile
├── c/
│   ├── Makefile                # targets: qwen, test-c, CUDA=1 CUDA_ARCH=native
│   ├── qwen.c                  # THE engine (new, self-contained; ports generic glm.c subsystems w/ attribution)
│   ├── st.h json.h tok.h tok_unicode.h tier.h uring.h compat.h grammar.h schema_gbnf.h decode_batch.h  # vendored
│   ├── backend_cuda.cu backend_cuda.h    # Phase 5 (new; generic kernels ported)
│   ├── openai_server.py resource_plan.py doctor.py colib   # vendored+adapted (Phase 9 / Phase 4 CLI)
│   ├── iobench.c
│   ├── tools/
│   │   ├── make_qwen_oracle.py convert_qwen.py compare_acts.py preflight.py eval_qwen.py gen_unicode.py
│   └── tests/                  # vendored generic C tests + new arch tests
├── web/                        # vendored wholesale later (Phase 9), rebrand strings only
└── docs/  serve_protocol.md (vendored verbatim)  qwen35_arch.md (Phase 1 findings)  ENVIRONMENT.md
```

**Vendor decisions:** verbatim w/ attribution header: `st.h json.h tier.h uring.h compat.h tok_unicode.h grammar.h schema_gbnf.h decode_batch.h iobench.c gen_unicode.py docs/serve_protocol.md` + generic C tests. Vendored-then-gated: `tok.h` (cl100k split regex hardcoded — Phase 1 must verify vs Qwen's pretokenizer). Not vendored: `glm.c`, `olmoe.c` (port subsystems function-by-function into fresh `qwen.c`). Rewritten: converter (`convert_qwen.py` keeps quant math + shard-streaming loop + CLI surface `--repo --indir --outdir --xbits --io-bits --shared-bits --group-size --mtp --selftest --min-free-gb`; new `classify()` for Qwen tensor names; **skip vision tensors**; strip `model.language_model.` prefix). Adapted: `openai_server.py` (ChatML + Hermes `<tool_call>` JSON), `resource_plan.py` (MLA-KV formula → hybrid GDN-state+GQA-KV formula, expert regex), `doctor.py`, `coli`→`colib` CLI.

---

## Phase 0 — Repo bootstrap and vendored infra green

- [x] `git init` (branch main); record machine + disk in Current status.
- [x] LICENSE (Apache-2.0), NOTICE crediting colibri, `.gitignore`, README stub.
- [x] Vendor header-only infra + `iobench.c` + 9 generic C tests (glm.c-dependent tests — idot/i4/topp/stops/sample_nan/logit_nan/kv_alloc/uring — get recreated against qwen.c in Phases 2–3) + adapted `c/Makefile`.
- [x] `qwen.c` skeleton: `SNAP` load, text_config-aware config parse (flat or nested `rope_parameters`; `layer_types[]` or `full_attention_interval` fallback), tokenizer load, prefix-tolerant tensor lookup (bare / `model.` / `model.language_model.`), embed→final-norm→lm_head noop forward.
- [x] `tools/preflight.py` (disk table 35b/397b, RAM, `--cuda` nvcc≥12.8 gate, `--iobench`).

**GATE 0:** ✅ 2026-07-20 — `make test-c` 8/8 green; `make qwen` compiles clean `-Wall -Wextra` (note: skeleton not yet run against a model dir — first exercised by the Phase-1 tiny oracle).

## Phase 1 — Oracle, tokenizer parity, ⚠-fact verification

- [ ] `python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu transformers safetensors` — record exact transformers version HERE: ______ ; hard-gate it in the oracle script.
- [ ] `tools/make_qwen_oracle.py` (pattern: colibri `make_glm_oracle.py`): tiny-random **text-only** qwen3_5_moe — ≥5 layers (≥1 full-attn), hidden 128, 8 experts top-2 + shared expert, DeltaNet 4 K/8 V heads ×32, conv 4, head_dim 64 partial-rotary 0.25 interleaved MRoPE, vocab 512, MTP block. Emit `c/qwen_tiny/` (bf16 safetensors + configs) + `c/ref_qwen.json` = `{prompt_ids, full_ids (greedy 32 new), tf_pred}`. Print full state_dict names (defines loader name-map). `--quant {int8,int4g128}` mode: round-trip weights through `convert_qwen.py` quant functions BEFORE computing refs → `ref_qwen_int8.json`, `ref_qwen_i4.json`.
- [ ] Also emit unit fixtures (JSON): DeltaNet single-layer in/out + intermediate (conv out, g/β, S trajectory), partial-RoPE q/k in/out, router logits→weights.
- [ ] Write `docs/qwen35_arch.md` answering every ⚠: exact MRoPE interleaved dim pairing + rotate convention; q/k per-head norm presence/type; `attn_output_gate` fusion (q_proj out layout, sigmoid placement); router softmax + `norm_topk_prob`; `shared_expert_gate` formula; gated-RMSNorm exact form (per-head? weight placement); MTP block structure (mixer type, norms, embedding sharing); which HF code path is reference (force eager/recurrent; note chunked-vs-recurrent numerics).
- [ ] Tokenizer parity `tests/test_tok_qwen.py`: download real Qwen3.5 tokenizer.json (tokenizer only); 10k mixed strings (code, CJK, emoji, whitespace runs, specials) `tok.h` (via a small C test binary) vs HF `AutoTokenizer`; fix `tok.h` split regex if Qwen's pretokenizer differs.

**GATE 1:** oracle regenerates deterministically (fixed seed); tokenizer 10k/10k exact; every ⚠ has a written answer in `docs/qwen35_arch.md`.

## Phase 2 — CPU token-exact correctness  ← THE CORE

- [ ] fp32 reference kernels in `qwen.c`: `causal_conv1d` (prefill + 4-slot decode ring), `gdn_decode_step` (recurrence above), `gdn_prefill_seq`, `gdn_gated_rmsnorm`, `rope_partial_interleaved`, `qk_head_rmsnorm` (if confirmed), gated GQA attention prefill+decode (conventional K/V cache, 2 KV heads repeated to Q heads), `moe_router_softmax_topk` (+renorm iff confirmed), `shared_expert_gated`, per-layer `layer_types[]` driver on the colibri residual seam.
- [ ] Loader name-map from oracle's printed state_dict (`model.language_model.layers.N.{linear_attn.*, self_attn.*, mlp.gate, mlp.experts.E.*, mlp.shared_expert.*, mlp.shared_expert_gate}` — confirm exact names in Phase 1).
- [ ] TF/greedy self-test identical UX to colibri: `SNAP=./qwen_tiny TF=1 ./qwen` → `[ORACLE] n/32`; `REF=` override; `DEBUG_LOGITS=1` top-5 dump.
- [ ] Debug rig: `DUMP_ACTS=1` per-layer hidden-state `.f32` dumps + `tools/compare_acts.py` (HF forward hooks; prints first divergent layer/element).
- [ ] C unit tests from Phase-1 fixtures: `test_deltanet`, `test_gqa_rope`, `test_router` → add to TEST_BINS.
- [ ] Quantized: run converter on qwen_tiny (int8, int4-g128), validate vs `--quant` oracles.

**GATE 2:** TF **32/32** on tiny oracle at fp32; 32/32 int8; 32/32 int4-g128; greedy `full_ids` match end-to-end; unit tests green.

## Phase 3 — CPU performance (still exact)

- [ ] Port colibri AVX-512/VNNI int8/int4/g128 matmul kernels into all dense + expert matmuls (+ their accuracy tests into TEST_BINS).
- [ ] `gdn_prefill_chunked` (WY, chunk 64, mirror HF chunked math): must match `gdn_prefill_seq` ≤1e-4 max-abs on fixtures AND keep TF 32/32. `GDN_CHUNK=0` kill-switch stays forever.
- [ ] OpenMP across heads/experts/rows; KV bf16 option (`KV16=1`, fp32 default).
- [ ] Integrate ported expert tiering/streaming (tier.h + expert_load_impl port + prefetch threads + heat/pin) — exercised even when RAM-resident.

**GATE 3:** tiny oracle still 32/32 with all fast paths on; chunked==sequential on fixtures; microbenchmarks recorded here (int4 GEMM GB/s; synthetic 35B-shaped layer-stack tok/s).

## Phase 4 — Real Qwen3.5-35B-A3B on CPU + conversion pipeline

- [ ] `convert_qwen.py` full pipeline from **bf16 35B** (~70 GB), shard-streaming download→convert→delete (peak disk ≈ 1 shard + ~20 GB output; preflight ≥35 GB — trivially OK on this box, could also keep source with 910 GB free).
- [ ] Precision map (defaults): routed experts **int4-g128**; shared expert **int8**; attention q/k/v **int4-g128**, o_proj **int8**; DeltaNet in_proj_qkvz/out_proj **int4-g128**; in_proj_ba/conv1d/A_log/dt_bias/norms/router **f32**; embed+lm_head **int8**; **MTP int8** (colibri issue #8).
- [ ] Expected container ≈19–20 GB → RAM-resident on 32 GB. Sizing: experts 40×256×3×(2048×512)≈32 B params.
- [ ] Statistical gates (HF bf16 can't fit in 32 GB): (a) llama.cpp Q4_K GGUF greedy 64-tok prefix agreement ≥85% over 20 fixed prompts; (b) perplexity (`eval_qwen.py`) within ~5–10% of `llama-perplexity` on a fixed ~100 KB corpus; (c) coherent ChatML chat via CLI.

**GATE 4:** (a)+(b)+(c) pass; CPU decode ≥ **8 tok/s** (A3B ⇒ ~1.5 GB active-weight reads/token @ ~60 GB/s DDR5 ⇒ 10–20 realistic); `PROF=1` numbers recorded here.

## Phase 5 — CUDA backend (sm_120)

- [ ] Install CUDA toolkit ≥12.8 (WSL2); verify `nvcc` emits sm_120 with `CUDA_ARCH=native`.
- [ ] Port generic colibri kernels: w4a16 GEMM, grouped expert dispatch (`grouped_hidden_w4`/`_dual`/`grouped_down_w4`), `silu_mul`, rmsnorm, pinned-staging/async. **No int4-WMMA reliance** (unsupported on Blackwell consumer): w4a16 dequant path.
- [ ] New kernels: `causal_conv1d` (decode+prefill), batched `gdn_decode` (per-slot states VRAM-resident), `rope_partial_interleaved`, gated RMSNorm + qknorm/gate elementwise, GQA decode attention (2 KV heads×256). GPU chunked-GDN prefill = stretch; CPU prefill + GPU decode acceptable.
- [ ] Env parity `COLI_CUDA=1 CUDA_DENSE=1 CUDA_EXPERT_GB=N`; per-kernel CPU-fallback kill-switches.

**GATE 5:** CUDA vs CPU token-identical on tiny oracle (32/32); ≥95% 64-tok prefix agreement on 20 real-35B prompts (document FP-order tolerance if not bit-stable); 35B ≥ **30 tok/s** with `CUDA_DENSE=1 CUDA_EXPERT_GB≈10`.

## Phase 6 — MTP speculative decoding

- [ ] MTP block per Phase-1 findings (1 hidden layer, shared embeddings/lm_head), loaded from converter `--mtp` pass at int8; port `repair_mtp_int8.py`.
- [ ] Spec-decode driver (already ported): draft MTP → verify batched → accept-on-match; adaptive pause on acceptance collapse (colibri pattern).
- [ ] Extend tiny oracle with MTP-head refs; TF mode checks MTP logits.

**GATE 6:** `MTP=1` output **identical** to `MTP=0` greedy; acceptance ≥50% on natural text; ≥1.3× end-to-end tok/s on 35B; `MTP=0` kill-switch works.

## Phase 7 — Qwen3.5-397B-A17B (tiered)

- [ ] Source: official **FP8** (~400 GB) → int4-g128 via streaming loop (vendored FP8-block dequant). Peak disk ≈ output ~205 GB + 1 shard (910 GB free — can even keep source). Fallback: GPTQ-Int4 repack ONLY if symmetric-g128-compatible (container has no zero-points — do not add asymmetric int4).
- [ ] Tier plan (this box): dense+shared+embeds ≈6 GB → VRAM; experts ~205 GB NVMe (`URING=1 DIRECT=1 PIPE=1`); VRAM expert hot cache `CUDA_EXPERT_GB≈7–8` (~1100 experts @ ~6.7 MB); RAM cache `RAM_GB≈18–20`; learning cache + pilot prefetch on; `PIN=auto`.
- [ ] Per-slot state: 45 lin-layers ×64 heads×128×128 f32 = **188 MB** GDN state + KV 15×2×256×2×4 B ≈ 61 KB/token → default `KV_SLOTS=4`, capped CTX; `resource_plan.py` must compute this (replace MLA formula).
- [ ] Cold worst case: 60 layers ×10 experts ×6.7 MB ≈ 4 GB reads/token ⇒ ≥2 tok/s needs ~40–50% cache hit — learning cache + prefetch are the lever; run expert-atlas-style sweep to seed pins.

**GATE 7:** coherent output; ppl smoke sane; sustained ≥ **2 tok/s** decode after cache warm-up on realistic chat; TIERS/EMAP telemetry correct; no OOM with RAM guard.

## Phase 8 — Ornith-1.0 (35B, then 397B)

- [ ] Registry: detect Ornith via config (eos 248046 / pad 248044 / name); set eos/pad; converter passthrough (arch identical).
- [ ] Chat template from Ornith `tokenizer_config.json` → `render_chat` in openai_server.py; Hermes-style `<tool_call>` JSON parse for agentic coding; verify actual template before assuming.
- [ ] Stretch: `-1M` YaRN variant (full-attn layers only; KV sizing guard).

**GATE 8:** Ornith-35B passes Phase-4-style statistical gates vs its own GGUF; tool-call round-trip e2e green; eos stops cleanly (no runaway). Then 397B via Phase-7 pipeline.

## Phase 9 — Server, web UI, sessions

- [ ] `SERVE_BATCH=1` mux protocol byte-per `docs/serve_protocol.md` (READY/SUBMIT/CANCEL/DATA/DONE/ERROR + TIERS/HWINFO/EMAP/HITS/PERF telemetry); continuous batching with batched GDN decode across slots.
- [ ] Vendored `openai_server.py` adapted (ChatML render, `<think>` handling, Hermes tools); `web/` vendored + rebranded; `colib` CLI (chat/serve/web/convert/doctor).
- [ ] Session persistence with recurrent state: per-slot save = token history + KV + per-layer (S, conv tail) at position T. Prefix reuse ONLY on exact extension of stored history; else full re-prefill (fast via chunked GDN). Optional: NVMe checkpoint ring every 1–2k tokens.

**GATE 9:** server test suite green; streaming chat via web UI on 35B and 397B; CANCEL/slot-reuse correct; byte-exact regeneration after warm reload of an extended session.

## Phase 10 — Hardening, docs, release

- [ ] `make check` (clean build + all tests); `doctor.py` retargeted; README/ENVIRONMENT.md accurate; benchmark table recorded here; tag v0.1.

---

## Risk register

| # | Risk | Mitigation |
|---|---|---|
| 1 | DeltaNet numerical exactness vs HF (fp32 state, gated-norm details) | fp32 recurrence; pinned transformers + version hard-gate; `compare_acts.py` first-divergence rig; oracle-emitted unit fixtures |
| 2 | Chunked WY prefill complexity | sequential fallback ships first and is never removed; dual validation (fixtures ≤1e-4 AND TF 32/32); `GDN_CHUNK=0` |
| 3 | MRoPE interleaved layout mismatch | resolved definitionally in Phase 1 from modeling code; tiny oracle exercises real partial-rotary before any real-model spend |
| 4 | Router/shared-gate assumption wrong | Phase-1 ⚠ checklist; caught at 32/32 gate |
| 5 | 32 GB RAM conversion | shard-streaming converter (never materializes model); `--min-free-gb`; proven on 35B first |
| 6 | Disk for 397B | preflight hard-gate; 910 GB free recorded — OK |
| 7 | Blackwell sm_120 | CUDA ≥12.8 preflight; w4a16 (no int4 WMMA); CPU fallback per kernel |
| 8 | MTP acceptance collapse | int8 MTP enforced in converter; adaptive draft pause |
| 9 | Tokenizer pretokenizer regex mismatch | Phase-1 10k-string parity gate |
| 10 | Recurrent state × slots memory | resource_plan computes; `KV_SLOTS=4` default on 32 GB; exact-extension-only session reuse |

## Definition of done

All four models convert and run; tiny-oracle 32/32 on CPU and CUDA at fp32/int8/int4; 35B ≥30 tok/s (GPU) with MTP; 397B ≥2 tok/s sustained; Ornith tool-calling e2e green over OpenAI server + web UI; `make check` green; this file fully checked with recorded benchmarks.
