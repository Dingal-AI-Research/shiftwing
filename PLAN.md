# PLAN.md — `colib`: a from-scratch C inference engine for Qwen3.5 MoE + Ornith-1.0

> **How to use this file (for any model/human picking up mid-work):** work top to bottom. Each phase has checkboxes and a **GATE** — a measurable validation that MUST pass before the next phase starts. Update the *Current status* block and check boxes as you go. All architecture facts and design decisions are recorded here; you do not need the original conversation. [colibri](https://github.com/JustVugg/colibri) (Apache-2.0) is the design reference — consult it freely (clone it if needed), but **this repo is NOT a fork**: it supports only Qwen3.5 MoE + Ornith-1.0, text-only.

## Current status

- **Active phase:** **PAUSED in Phase 4** (real Qwen3.5-35B-A3B CPU validation). Do not start Phase 5 until Gate 4 is resolved.
- **Last gate passed:** GATE 3 (2026-07-20: packed AVX/VNNI kernels, chunked WY prefill, OpenMP/KV16, and bounded expert streaming; all fast-path tiny oracles 32/32).
- **Blockers:** Gate 4 prefix agreement is **57.58%** (737/1,280 tokens; 8/20 prompts at or above 85%), below the required 85%. The bounded 1,024-token perplexity comparison passes at **2.50% relative delta**, and coherent chat plus **9.34 tok/s** decode pass. Full fixed-corpus perplexity is still pending because the evaluator needs batched expert GEMM rather than repeated GEMV to run in reasonable time.
- **Python env:** `.venv` created with **uv** (`export PATH="$HOME/.local/bin:$PATH"`; system python3-venv broken, sudo needs password). CPU torch + **transformers 5.14.1** (has `Qwen3_5MoeForCausalLM` — text-only class, use it for the oracle) + safetensors installed. Pin/gate transformers ≥ 5.14 in the oracle script.
- **Next steps for whoever resumes (in order):**
  1. Reproduce a failing real-model prompt from `c/bench/qwen35_prefix.json`, then use `c/tools/compare_acts.py` to find the first divergent layer/tensor against the Q4_K reference. Investigate model math or the mixed-precision map before relaxing the approved 85% threshold.
  2. Add batched grouped-expert matmul to the evaluation-only prefill path; the current grouped implementation is correct but processes routed assignments as repeated GEMVs (the 1,024-token C run took about 10 minutes at 0.799 scored tok/s).
  3. Re-run the complete 20×64 prefix comparison and full hash-pinned corpus comparison. Preserve the current bounded results as a regression baseline.
  4. Re-run `make -C c test-c test-python`, all three 32-token tiny oracles, and `git diff --check`; these were green before the final grouped-evaluation/mmap/profile edits but have not been rerun since them.
- **Phase-4 handoff:** exact commands, revisions, hashes, timings, artifacts, and known limitations are in `docs/phase4_handoff.md`.
- **Machine:** AMD Ryzen 7 7700X (Zen4; AVX-512F/DQ/BW/VL/VNNI/BF16 confirmed), 32 GB RAM, RTX 5070 Ti 16 GB (Blackwell **sm_120**; WSL2 driver 591.86 OK; **nvcc NOT installed** — CUDA toolkit ≥12.8 needed before Phase 5), NVMe: **910 GB free** (recorded 2026-07-20; 397B needs ≥250 GB — OK). Python 3.12.3; `.venv` has CPU torch 2.13.0 and transformers 5.14.1.
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

**Gated DeltaNet per-layer weights:** split `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, depthwise `conv1d` (k=4, SiLU, over concat(q,k,v)), `A_log`, `dt_bias`, gated RMSNorm (`norm`), `out_proj`. Decode recurrence per V-head (state S = 128×128 **fp32**; K-heads repeat-interleaved to V-heads):

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
│   ├── st.h json.h tok.h tok_unicode.h tok_nfc.h tier.h uring.h compat.h grammar.h schema_gbnf.h decode_batch.h
│   ├── backend_cuda.cu backend_cuda.h    # Phase 5 (new; generic kernels ported)
│   ├── openai_server.py resource_plan.py doctor.py colib   # vendored+adapted (Phase 9 / Phase 4 CLI)
│   ├── iobench.c
│   ├── tools/
│   │   ├── make_qwen_oracle.py convert_qwen.py compare_acts.py preflight.py eval_qwen.py gen_unicode.py
│   └── tests/                  # vendored generic C tests + new arch tests
├── web/                        # vendored wholesale later (Phase 9), rebrand strings only
└── docs/  serve_protocol.md (vendored verbatim)  qwen35_arch.md (Phase 1 findings)  ENVIRONMENT.md
```

**Vendor decisions:** verbatim w/ attribution header: `st.h json.h tier.h uring.h compat.h grammar.h schema_gbnf.h decode_batch.h iobench.c docs/serve_protocol.md` + generic C tests. Adapted after parity gating: `tok.h`, `tok_unicode.h`, and generated `tok_nfc.h` now implement Qwen's BPE/NFC behavior exactly. Not vendored: `glm.c`, `olmoe.c` (port subsystems function-by-function into fresh `qwen.c`). Rewritten: converter (`convert_qwen.py` keeps quant math + shard-streaming loop + CLI surface `--repo --indir --outdir --xbits --io-bits --shared-bits --group-size --mtp --selftest --min-free-gb`; new `classify()` for Qwen tensor names; **skip vision tensors**; strip `model.language_model.` prefix). Adapted: `openai_server.py` (ChatML + Hermes `<tool_call>` JSON), `resource_plan.py` (MLA-KV formula → hybrid GDN-state+GQA-KV formula, expert regex), `doctor.py`, `coli`→`colib` CLI.

---

## Phase 0 — Repo bootstrap and vendored infra green

- [x] `git init` (branch main); record machine + disk in Current status.
- [x] LICENSE (Apache-2.0), NOTICE crediting colibri, `.gitignore`, README stub.
- [x] Vendor header-only infra + `iobench.c` + 9 generic C tests (glm.c-dependent tests — idot/i4/topp/stops/sample_nan/logit_nan/kv_alloc/uring — get recreated against qwen.c in Phases 2–3) + adapted `c/Makefile`.
- [x] `qwen.c` skeleton: `SNAP` load, text_config-aware config parse (flat or nested `rope_parameters`; `layer_types[]` or `full_attention_interval` fallback), tokenizer load, prefix-tolerant tensor lookup (bare / `model.` / `model.language_model.`), embed→final-norm→lm_head noop forward.
- [x] `tools/preflight.py` (disk table 35b/397b, RAM, `--cuda` nvcc≥12.8 gate, `--iobench`).

**GATE 0:** ✅ 2026-07-20 — `make test-c` 8/8 green; `make qwen` compiles clean `-Wall -Wextra` (note: skeleton not yet run against a model dir — first exercised by the Phase-1 tiny oracle).

## Phase 1 — Oracle, tokenizer parity, ⚠-fact verification

- [x] Python env: `.venv` via uv; CPU torch; transformers **5.14.1** (recorded; hard-gate ≥5.14 in the oracle script).
- [x] ⚠-fact verification: all answered in `docs/qwen35_arch.md` from installed 5.14.1 source (zero-centered RMSNorm w/ `1+weight`; split GDN projections in 5.14.1 vs possibly-fused checkpoint — OPEN ITEM there; conv over qkv only, bias-free, SiLU; exact recurrence w/ q-scaling and l2norm eps 1e-6; per-head gated RMSNorm plain-weight; q_proj fuses per-head [q|gate], sigmoid gate after attention; per-head zero-centered q/k norms pre-RoPE; partial 64-dim split-half RoPE, MRoPE = no-op for text; router fp32-softmax→topk→always-renorm; fused 3D expert tensors in HF format; shared_expert_gate sigmoid scalar; MTP ignored by HF ⇒ native implementation from checkpoint names + vLLM reference).
- [x] `tools/make_qwen_oracle.py`: deterministic five-layer text-only tiny model; bf16/int8/int4-g128 snapshots (`qwen_tiny`, `qwen_tiny_int8`, `qwen_tiny_i4`); 32-token greedy/TF references; checkpoint-compatible synthetic MTP block; full state inventory.
- [x] Unit fixtures (JSON): DeltaNet input/output plus conv, decay/β and state trajectory; partial RoPE q/k; router logits/top-k weights.
- [x] `docs/qwen35_arch.md` answers every inference-critical ⚠ and records the real checkpoint's split projections and MTP tensor inventory. MTP executable math is explicitly deferred to Phase 6 because HF ignores it.
- [x] Tokenizer parity `tests/test_tok_qwen.py`: real effective Qwen3.5 tokenizer, 10k mixed strings, exact C/HF encode parity; `tok.h` now handles Qwen merge format, NFC, marks, one-digit numeric splitting, and effective special tokens.

**GATE 1:** ✅ 2026-07-20 — all three oracle modes regenerate byte-for-byte deterministically; each reference records greedy/teacher-forced 32/32; tokenizer 10,000/10,000 encode and decode parity; every inference-critical ⚠ has a written answer in `docs/qwen35_arch.md`.

## Phase 2 — CPU token-exact correctness  ← THE CORE

- [x] fp32 reference kernels in `qwen.c`: causal conv ring, sequential GDN recurrence and gated norm, partial split-half RoPE, zero-centered per-head q/k norm, gated GQA with KV cache, softmax-top-k router, routed/shared SwiGLU experts, and the hybrid residual driver.
- [x] Full loader name-map for split DeltaNet, attention, per-expert container matrices, router/shared experts, norms, embeddings, and LM head; Phase 2 first validated int8/int4 through eager dequantization (replaced by retained packed weights in Phase 3).
- [x] TF/greedy self-test: `SNAP=./qwen_tiny TF=1 ./qwen` plus `REF=` and `DEBUG_LOGITS=1`; both modes compare all 32 generated tokens.
- [x] Debug rig: `DUMP_ACTS=1` emits per-layer/token `.f32`; `tools/compare_acts.py` runs HF hooks and reports the first divergent layer/token/element.
- [x] Fixture-driven `test_deltanet`, `test_gqa_rope`, and `test_router` are part of `TEST_BINS`.
- [x] Quantized tiny snapshots: int8 and int4-g128 both pass their quantized oracle references.

**GATE 2:** ✅ 2026-07-20 — TF and greedy **32/32** at fp32, int8, and int4-g128; 11 C tests and 5 Python quantization tests green; HF/C activation comparison checks 185 layer-token states per mode within 1e-4 (worst observed 8.94e-8).

## Phase 3 — CPU performance (still exact)

- [x] Packed matrices remain quantized after load; AVX-512/AVX2 exact int8 and grouped-int4 kernels cover every dense/expert matmul, with AVX-512 VNNI activation-int8 acceleration on expert int8 paths. Deterministic integer-dot, dequantized-matmul, zero-scale, tail, and g128 tests are in `TEST_BINS`; `IDOT=0` is the accuracy fallback.
- [x] `gdn_prefill_chunked` mirrors Transformers 5.14.1's 64-token WY math and is wired into prompt prefill. It matches `gdn_prefill_seq` across 1/5/63/64/65/127-token tests within 1.2e-7 and the HF fixture within 4.2e-5; `GDN_CHUNK=0` remains the permanent fallback.
- [x] OpenMP covers matrix rows, attention heads, routed experts, and prefill projections. `KV16=1` stores the KV cache in bf16 (fp32 remains default) and passes the oracle.
- [x] Bounded per-layer expert residency uses `tier.h` LFRU heat/recency, `expert_load_impl` eviction/reload, and asynchronous file-page prefetch workers. `EXPERT_RAM=2` forces repeated streaming through the tiny oracle; full RAM residence still exercises heat and prefetch accounting.

**GATE 3:** ✅ 2026-07-20 — fp32, int8, and int4-g128 tiny oracles remain TF/greedy **32/32** with `OMP_NUM_THREADS=8 KV16=1 EXPERT_RAM=2 PREFETCH_THREADS=2` (chunked prefill and IDOT defaults on). Chunked/recurrent fixture deltas are below 1e-4; 13 C tests, 5 Python tests, the x86-64-v3 clean build, and 10,000-case tokenizer parity are green. Ryzen 7 7700X, GCC 13.3, `-O3 -march=native`: 4096² int4-g128 GEMV **40.34 GB/s**; synthetic 40-layer × 9-active-expert, H=2048/I=512 stack **52.99 tok/s, 31.88 GB/s** over 0.56 GiB unique packed weights (`OMP_NUM_THREADS=8`).

## Phase 4 — Real Qwen3.5-35B-A3B on CPU + conversion pipeline

- [x] `convert_qwen.py` full pipeline from **bf16 35B** (~70 GB), shard-streaming download→convert→delete. Converted official commit `59d61f3ce65a6d9863b86d2e96597125219dc754` into 14 output shards; each source shard was deleted only after durable conversion.
- [x] Precision map (defaults): routed experts **int4-g128**; shared expert **int8**; attention q/k/v **int4-g128**, o_proj **int8**; DeltaNet in_proj_qkvz/out_proj **int4-g128**; in_proj_ba/conv1d/A_log/dt_bias/norms/router **f32**; embed+lm_head **int8**; **MTP int8** (colibri issue #8). Header-only dry-run predicts 31,333 loader tensors and ~19.1 GB text-only output without materializing a source shard.
- [x] Container is **19,081,779,712 indexed bytes** (19,090,239,288 bytes on disk), with 31,333 logical / 62,305 physical tensors. Full loader smoke passes: 40 layers (10 attention + 30 GDN), 140 f32 / 132 int8 / 30,840 int4 matrices.
- [ ] Statistical gates (HF bf16 can't fit in 32 GB): (a) **FAIL** — llama.cpp Q4_K GGUF greedy agreement is 57.58% over 20×64 tokens; (b) **bounded PASS** — 1,024 source tokens / 510 scored tokens give colib PPL 1.097995 vs llama.cpp 1.0712 (2.50% delta), while the full fixed corpus remains pending; (c) **PASS** — coherent ChatML output via CLI.

**GATE 4:** ❌ **NOT PASSED / project paused 2026-07-20.** Prefix agreement misses the approved threshold and full-corpus PPL has not run. CPU performance passes: **9.34 tok/s** for 64-token decode with `OMP_NUM_THREADS=8 EXPERT_RAM=64 PREFETCH_THREADS=4` (load 14.631 s, one-token prefill 4.369 s, decode 6.852 s; detail: GDN 3.073 s, attention 0.454 s, MoE 2.615 s including 1.080 s expert load / 4,724 misses, LM head 0.687 s). Resident-all-experts was only 1.64 tok/s under WSL page pressure, so bounded copied expert caching remains the measured default. See `docs/phase4_handoff.md`.

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
