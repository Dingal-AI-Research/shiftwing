# PLAN.md — shiftwing: a phase-aware C inference engine for Qwen3.5 MoE + Ornith-1.0

> **How to use this file (for any model/human picking up mid-work):** work top to bottom. Each phase has checkboxes and a **GATE** — a measurable validation that MUST pass before the next phase starts. Update the *Current status* block and check boxes as you go. All architecture facts and design decisions are recorded here; you do not need the original conversation. [colibri](https://github.com/JustVugg/colibri) (Apache-2.0) is the design reference — consult it freely (clone it if needed), but **this repo is NOT a fork**: it supports only Qwen3.5 MoE + Ornith-1.0, text-only.

## Current status

- **Active phase:** **Phase 12 Ornith397 reactivation and RTX 5070 Ti optimization in progress**. DeepSeek is rejected and frozen. The resumable Ornith397 base reconstruction is stopped, not running: 58/122 outputs are atomically committed, shard 59 is partially staged, and the stop cause was not retained. Q3 reconstruction has not started. Two reversible candidates are locally validated but have no real-model A/B result. `CURRENT_STATUS_HANDOFF.md` is the authoritative continuation snapshot. LocalForge is no longer documentation-only; see `CURRENT_STATUS_HANDOFF.md` for the 2026-08-23 owner-directed changes.
- **Last full gate passed:** **GATE 9** (2026-08-05: both Ornith models pass the manifest-bound AB/BA, cancellation, and exact-web production sequence; the independent q3 release audit is 20/20).
- **Current evidence and decision:** the complete optimized int4 run sustains **0.631963761 tok/s** with a 0.674006 median and 0.530693 minimum. Expanded-q4 q3 is **13.68% faster sustained** and **28.46% faster at the median**, but its minimum turn is **14.27% slower**; no twofold speed claim is supported. Ornith35 routed int3 retains 95.546875% teacher-forced agreement with an 85.9375% worst prompt and increases PPL 10.660342%. The owner accepts this quality class, waived the Ornith397 PPL run, selected q3, and accepted a **≥0.70 tok/s sustained regression floor**. The release auditor now encodes that waiver and floor explicitly without rewriting historical evidence; 28 focused policy/runtime tests and the CUDA/native-q3 suites pass.
- **Python env:** `.venv` created with **uv** (`export PATH="$HOME/.local/bin:$PATH"`; system python3-venv broken, sudo needs password). CPU torch + **transformers 5.14.1** (has `Qwen3_5MoeForCausalLM` — text-only class, use it for the oracle) + safetensors installed. Pin/gate transformers ≥ 5.14 in the oracle script.
- **Reopening decisions (2026-07-31):** the first amendment set Ornith397 production throughput to ≥0.85 tok/s; its full int4 rerun failed. The second owner decision authorizes the measured Ornith35 int3 quality trade-off and a full Ornith397 q3 sidecar. Historical reports remain unchanged. See `docs/research/phase8_preflight_08_owner_accepted_q3.md`.
- **Q3 implementation verification (2026-08-01):** during conversion preflight, the complete current Python suite passed **130/130** and all **21/21 C test executables** passed, including the grouped-int3 matmul cases. The C loader now also fails closed if `EXPERT_Q3=1` selects a missing int3 expert tensor instead of silently mixing in its int4 base tensor, a production-layout regression opens all 122 base shards plus 480 q3 sidecars successfully, and every Gate-8/9 harness uses one fail-closed sidecar identity loader that records the manifest SHA-256. Gate-8 and Gate-9 engine environments discard ambient model/mode/kernel settings and explicitly freeze eight OpenMP threads. These are implementation controls, not model-quality or throughput evidence; the Gate-8 supervisor is now stopped at the conversion boundary.
- **Q3 completion evidence (2026-08-01):** the final sidecar manifest is complete at 60 layers × 512 experts, 480 files, 276,480 tensors, and 157,073,113,440 data bytes. Its SHA-256 is `5231fbe79bc9edf27b86222e4df503066a0b8f04f02fc612ab993696096b0180`; the independent audit verified 480/480 hashes and 480/480 headers. The accepted expanded-q4 qualification artifact is `c/ornith397_q3_qualification_expand_q4.json`, SHA-256 `047fdf9d6d8e5fcbd494dcdbb1d8d4ac752f799dae9bf3f10f456c0223a8d9f7`.
- **Optional q3 optimization:** native q3 stores packed 24-bit triplets in VRAM and adds dedicated single/grouped CUDA kernels. Its CUDA unit tests pass, but its historical real-model path was slower. The release default therefore expands q3 to the faster q4 CUDA representation. Phase 12 decouples `Q3_ROUTE_ATLAS=1` from `Q3_NATIVE=1`, so the atlas can retain compact q3 on NVMe/RAM while caching expanded-q4 experts in VRAM. Full-model A/B qualification remains open. See `docs/research/phase12_experiment_02_expanded_q4_route_atlas.md`.
- **Directional streaming probes (2026-08-01):** the CUDA target builds. A visible six-token warm/measured q3-atlas probe streamed the exact requested text and measured 1.084803 tok/s with 100% decode cache hits; all 2,597 learned routes fit in the host hot set, so device-atlas overflow remains untested. Two cold Snake-game generations streamed at 0.574879 and 0.579593 tok/s with complete CUDA residency; both reached their 384/1,024-token safety ceilings, while the second produced a substantially complete game through most of its keyboard handler. The owner accepts this as a directional coding-quality pass. It does not replace the frozen four-output Gate-8 control.
- **Q3 coherence repair (2026-08-02):** the q3 sidecar and native packed kernel were coherent, and disabling pinned staging did not change the expanded-route failure. A new actual-dimension CUDA equivalence test reports zero maximum difference between packed-q3 and expanded-q4 grouped MoE. The defect was a representation mismatch limited to resident batched prefill: expanded q4 buffers were sent to the q3 kernel based only on the source `fmt`. Dispatch now additionally requires `Q3_NATIVE=1`. The fixed cold control is coherent with 3,840/3,840 device-MoE forwards and zero host fallback; the old malformed Gate-8 artifact remains rejected and will not be acknowledged.
- **Repaired Gate-8 controls and tier attempt (2026-08-02):** regenerated q3 teacher forcing passes at 1,243/1,280 positions, all four frozen coherence prompts pass semantic review, and the HTTP tool gate emits the exact Paris weather call and final answer with complete device-MoE telemetry. The first repaired full tier attempt is a real 0.520672 tok/s sustained failure despite a 0.796441 median because one turn falls to 0.243102 tok/s and then immediately recovers. One unchanged complete retry is preregistered after an idle health check. Its first launch was owner-interrupted after three warm-up turns, produced no measured evidence, and left the failed artifact unchanged. See `docs/research/phase8_experiment_16_repaired_q3_gate8_retry.md`.
- **Gate-8 closure (2026-08-03):** after Windows GPU contention was removed, the unchanged complete retry sustains 0.836866773 tok/s, with a 0.8581515 median and 0.773247 minimum. Every measured turn exceeds 0.70; the earlier 0.243102 TypeScript outlier repeats at 0.857416 instead. The artifact hashes to `7d4d1fe8…c4106`, all 46,080 MoE layer-forwards use CUDA, host fallback is zero, and the schema-2 pipeline closes passed. Gate 9 is now authorized. See `docs/research/phase8_experiment_16_repaired_q3_gate8_retry.md`.
- **Gate-9 closure (2026-08-05):** the resumed controller verifies all five Ornith35 artifacts, restarts Ornith397 A/B from its atomic boundary, and completes all five remaining controls. Ornith397 batching passes at 1.109807× geometric mean with a 1.096309× minimum ordering; cancellation passes at 1.935734 s p95 over 20 trials with 12,720/12,720 device-MoE forwards; and the web gate returns exact output twice with 480/480 device-MoE forwards. The final controller state and regenerated q3 release audit both report 20/20 checks passed. See `docs/research/phase9_experiment_23_gate9_production_execution.md`.
- **Phase-10 evidence closure (2026-08-06):** after the missing indexed disposition paper was added, the complete clean-build `make check` rerun exits zero: 21 C executables, 10,000 tokenizer cases, 18 web tests, 137 Python tests, production web build, zero-vulnerability audit, 91-document/111-link/75-report documentation validation, 273-path source-package validation after final evidence staging, and both CUDA suites pass. The actual-shape native-q3 versus expanded-q4 CUDA comparison is exact. A final live audit exposed nondeterministic CUDA executable hashes across identical forced rebuilds; Gate-8/9 resumption now retains the raw hash as provenance and binds to deterministic engine-source fingerprint `c7751e41…b07527`. Source drift fails, identical-source rebuilds resume, both real controllers replay successfully, and the regenerated q3 audit returns 20/20. The CLI now reports release version `0.1.0`, matching the web package. See `docs/research/phase10_experiment_07_q3_release_disposition.md`.
- **Phase-11 preflight (2026-08-14):** source/model/tokenizer/reference identity is pinned to `deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb`. The corrected read-only preflight includes both the remaining 166,888,735,421-byte source copy and the 167,174,674,555-byte native-container upper bound: from 526,791,335,936 bytes free it projects 339,063,409,976 new bytes at peak and 187,727,925,960 bytes free afterward, above the mandatory 100 GiB floor. No Qwen or Ornith artifact is deleted. The pinned metadata inventory passes at 48 shards, 72,317 tensors, and 166,878,536,440 indexed payload bytes; the resumable shard fetch is active. Nine tooling and four protocol tests pass. See `docs/research/phase11_preflight_01_deepseek_v4_spec_and_storage.md`.
- **Phase-11 host recovery (2026-08-16):** repeated ext4 write failures were traced to Windows `C:\` having only 46,899,200 bytes free. Compact Ornith35 reconstruction metadata and exact source/hash commands were preserved; only the verified 32,193,227,292-byte Ornith35 container and independently rehashed 21,166,757,760-byte GGUF were removed. A root `fstrim` discarded 424,089,743,360 bytes, and direct use of Microsoft's `CompactVirtualDisk` API shrank the detached VHDX by 53,822,357,504 bytes, leaving 53,990,948,864 host bytes free. Ornith397 and the partial DeepSeek source remain intact. See `docs/research/phase11_experiment_03_storage_recovery.md`.
- **Phase-11 CUDA residency (2026-08-16):** the native runtime now has a bounded reference-protected device expert cache, a persistent FP8 dense arena mirror, and a resident BF16 output head. Focused CPU/CUDA parity controls pass; the full post-change gate, embedding residency, live 43-layer mux, and end-to-end tok/s/TTFT evidence remain open.
- **Phase-11 source completion (2026-08-16):** attempt 3 resumes at 58/62 and completes all 48 pinned shards; no partial remains. A separate full-file pass accepts 72,317 tensors and 166,878,536,440 indexed payload bytes with all 43 base and three DSpark expert layers complete. Host free space is now 47,417,536,512 bytes, so real conversion remains storage-locked until the fresh Ornith397 control is captured. See `docs/research/phase11_experiment_04_pinned_fetch_completion.md`.
- **Phase-11 first real forward (2026-08-17):** the manifest-bound SM120 binary loads the 8,845,959,388-byte dense arena, streams exactly 258 cold routed experts, traverses all 43 base layers, and returns finite logits (BOS input 0, top token 5, logit 16.258379). Correctness smoke passes without OOM, but performance does not: initialization is 334.191103 seconds and cold decode is 118.610136 seconds (0.008431 tok/s), with 12,295,250,140 storage bytes and 12,297,913,712 CUDA upload bytes attributed. This is a diagnostic failure, not DeepSeek qualification or promotion evidence. See `docs/research/phase11_experiment_07_first_real_forward.md`.
- **Phase-11 cold/warm profile (2026-08-17):** a two-token resident run isolates the bottleneck. Token 0 takes 134.734045 seconds with 258/258 expert misses; token 1 takes 84.532588 seconds with 230 misses and only 28 hits. The 667 dense projections consume just 2.023834 and 1.912695 seconds respectively, while the second token reads another 3,074,949,120 expert bytes. Cache-only tuning cannot qualify this path; direct persistent I/O and pinned transfers are the next controlled intervention. See `docs/research/phase11_experiment_08_cold_warm_profile.md`.
- **Phase-11 pinned/direct A/B (2026-08-17):** matched two-token output is exact and all 15,370,206,592 model-upload bytes use pinned staging. Initialization improves 408.372656 to 200.970412 seconds and cold decode improves 134.734045 to 83.758211 seconds, but the resident token regresses 84.532588 to 96.703443 seconds. The combined optimization is rejected for decode qualification; the next step is a short storage-mode benchmark and expert-extent coalescing rather than another full blind run. See `docs/research/phase11_experiment_09_pinned_direct_ab.md`.
- **Phase-4 handoff:** exact commands, revisions, hashes, timings, artifacts, revised statistical-gate rationale, and evaluation controls are in `docs/phase4_handoff.md`.
- **Phase-5 handoff:** CUDA architecture, controls, correctness evidence, and repeatable benchmark command are in `docs/phase5_cuda.md`.
- **Phase-6 handoff:** implemented MTP math, additive conversion, rollback design, controls, correctness evidence, and the open performance results are in `docs/phase6_mtp.md`.
- **Research archive:** retrospective Phase 0–5 papers and the reporting policy are indexed at `docs/research/README.md`.
- **Machine:** AMD Ryzen 7 7700X (Zen4; AVX-512F/DQ/BW/VL/VNNI/BF16 confirmed), 29.4 GiB RAM, RTX 5070 Ti 15.92 GiB (Blackwell **sm_120**; WSL2 driver 591.86 OK; workspace-local CUDA 12.9.86 toolkit). After owner-authorized Qwen retirement, ext4 usage fell from 518 to 298 GB and free space rose from 439 to **659 GB**. Windows-visible free space remains **26 GB** because the VHDX was not compacted; the released ext4 blocks are reusable for q3 conversion. The Ornith397 int4 manifest remains 122 shards, 93,078 logical/278,152 physical tensors, and 212,634,789,241 payload bytes. Python 3.12.3; `.venv` has CPU torch 2.13.0 and transformers 5.14.1.
- **Colibri reference clone:** `/tmp/claude-1000/-home-dinga-Projects-colib/de80692d-17fa-4193-8ee0-1c2cc8b7db9a/scratchpad/colibri` (scratchpad; if gone: `git clone --depth 1 https://github.com/JustVugg/colibri`).

## Scope (fixed, user-approved)

- Models in order: **Qwen/Qwen3.5-35B-A3B** (correctness target) → **Qwen/Qwen3.5-397B-A17B** (completed tier/reference study under its historical ≥2 tok/s rule) → **deepreinforce-ai/Ornith-1.0-35B** (quality control) → **Ornith-1.0-397B q3** (selected production model; final Gate-8 qualification 0.836866773 tok/s sustained against the ≥0.70 regression floor). Text-only (vision tower dropped at conversion). MoE only (no dense 9B/31B).
- **Production-target decision (confirmed 2026-07-29):** Qwen397 3-bit conversion is permanently skipped and is not a prerequisite for any later gate. Qwen remains the measured architecture/correctness reference; all new 397B conversion and optimization effort goes directly to Ornith397 because it is the end model. The two complete but unindexed Qwen 3-bit pilot files are research evidence only and must never be auto-discovered as a runnable sidecar.
- **Active-roadmap cut:** Phase 7 is historical and closed. No Qwen397 3-bit conversion, artifact, benchmark, or retry remains in the work queue; Gate 8 proceeds directly through the pinned Ornith397 FP8 → int4-g128/int8 conversion.
- **Selected precision:** Ornith35 grouped int3 passes token agreement and regresses PPL 10.660342%. The owner accepts that quality class, selected Ornith397 q3 based on its complete 0.718432662 tok/s sustained result, and waived a separate Ornith397 perplexity run. The release amendment preserves the waiver explicitly rather than manufacturing a passing PPL artifact.
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
shiftwing/
├── PLAN.md  README.md  LICENSE (Apache-2.0)  NOTICE (credits colibri/JustVugg)  Makefile
├── c/
│   ├── Makefile                # targets: qwen, test-c, CUDA=1 CUDA_ARCH=native
│   ├── qwen.c                  # THE engine (new, self-contained; ports generic glm.c subsystems w/ attribution)
│   ├── st.h json.h tok.h tok_unicode.h tok_nfc.h tier.h uring.h compat.h grammar.h schema_gbnf.h decode_batch.h
│   ├── backend_cuda.cu backend_cuda.h    # Phase 5 (new; generic kernels ported)
│   ├── openai_server.py resource_plan.py doctor.py shiftwing   # vendored+adapted (Phase 9 / Phase 4 CLI)
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
- [x] Statistical gates (HF bf16 can't fit in 32 GB): (a) **PASS** — teacher-forced next-token argmax agreement against llama.cpp Q4_K is 96.17% (1,231/1,280) over 20×64 tokens, and every prompt is ≥85%; (b) **PASS** — the fixed hash-pinned 1,024-token corpus window / 510 scored tokens gives colib PPL 1.088496 vs llama.cpp 1.0712 (1.61% delta); (c) **PASS** — coherent ChatML output via CLI. Free-running agreement remains recorded as 57.58% because one quantizer-dependent near-tie cascades into a different valid continuation.

**GATE 4:** ✅ **2026-07-21.** The original 85% threshold is retained, but the cross-quantizer comparison is teacher-forced rather than free-running: the focused failure showed the same top-five candidates with a first-token rank flip, followed by 15/16 agreement when the reference path was replayed. The complete replay scores **96.17%** and all 20 prompts pass. PPL delta is **1.61%**, chat is coherent, and CPU performance is **9.34 tok/s** for 64-token decode with `OMP_NUM_THREADS=8 EXPERT_RAM=64 PREFETCH_THREADS=4` (load 14.631 s, one-token prefill 4.369 s, decode 6.852 s; detail: GDN 3.073 s, attention 0.454 s, MoE 2.615 s including 1.080 s expert load / 4,724 misses, LM head 0.687 s). Evaluation-only grouped int4 GEMM plus mmap expert views score at 1.281 tok/s; `EVAL_GROUPED=0` and `--copy-experts` retain the slow fallbacks. The exhaustive 32K-token PPL soak is optional because it is estimated at several CPU hours and no longer gates correctness. See `docs/phase4_handoff.md`.

## Phase 5 — CUDA backend (sm_120)

- [x] CUDA toolkit **12.9.86** installed without sudo/driver replacement from NVIDIA redistributable components under `.toolchains/cuda-12.9`; `nvcc -arch=sm_120` compile and runtime kernel smoke pass on the RTX 5070 Ti.
- [x] C-compatible backend/link seam plus independent `test-cuda`: device/stream and pinned/device memory primitives, zero-centered RMSNorm, SiLU-multiply, int8 GEMV, grouped-int4/fp32 GEMV, and grouped-int4/fp16 activation GEMV agree with deterministic scalar references. See `docs/phase5_cuda.md`.
- [x] Opt-in `COLI_CUDA=1 CUDA_DENSE=1` inference correctness path with lazy persistent matrix uploads, reusable activation buffers, and automatic per-matrix CPU fallback. Dense-only and `CUDA_EXPERTS=1` tiny int8/int4 runs are TF/greedy 32/32; fp16 int4 activations and `EXPERT_RAM=2` eviction/reload are also 32/32.
- [x] First device-resident fusion increment: three-stage MLP transactions, true grouped int4/fp16 hidden+down expert kernels, and shared-input Q/K/V plus compatible GDN projection groups. The stressed tiny oracle remains TF/greedy 32/32. A real-35B eight-token smoke improved the intermediate CUDA path from 5.19 to **7.88 tok/s**, still below the 9.34 tok/s CPU baseline and therefore not a Gate-5 result.
- [x] Generic decode kernels: warp-tiled container-layout w4a16 GEMV, direct-half grouped routed/shared expert dispatch, fused SiLU/gating/weighted reduction, RMSNorm, pinned host allocation, and async copies on one nonblocking stream. No int4-WMMA dependency is used on Blackwell consumer GPUs.
- [x] Qwen decode kernels: causal conv + fp32 recurrent GDN state, gated DeltaNet norm and output projection, zero-centered q/k norm, partial split-half RoPE, gated GQA with persistent KV, and int8/int4 output projections. CPU chunked-GDN/attention prefill remains the accepted fallback; decode state is synchronized to CUDA once at the boundary.
- [x] Independent bounded VRAM LRU (`CUDA_EXPERT_GB`, 8 GiB default) survives CPU expert-slot eviction; dense/prompt-route preload, direct cache descriptors, and fused routed+shared MoE avoid decode-time host expert loads on a warmed route.
- [x] Env parity `COLI_CUDA=1 CUDA_DENSE=1 CUDA_EXPERT_GB=N`; `CUDA_GDN`, `CUDA_ATTN`, `CUDA_MLP`, `CUDA_GROUPED`, `CUDA_GROUPED_KERNEL`, and `CUDA_PROJECTIONS` provide per-path CPU fallbacks. `CUDA_PRELOAD=0` and `WARMUP=N` control benchmark preparation.

**GATE 5:** ✅ **2026-07-21.** CPU and CUDA replay all fp32/int8/int4 tiny references at teacher-forced and greedy **32/32**. On the fixed 20×64 real-model corpus, CUDA teacher-forced agreement against the saved llama.cpp Q4_K reference is **1,231/1,280 = 96.171875%**; its aggregate equals CPU, with one-token FP-order shifts between prompts 7 and 9 documented in `docs/phase5_cuda.md`. Three warmed 64-token 35B decode samples from the final binary are 31.75, 31.87, and 30.46 tok/s (**median 31.75 tok/s**) with `CUDA_DENSE=1 CUDA_EXPERT_GB=8`; the cache occupies 7.05 GiB and records zero decode-time misses. CPU prefill remains intentionally outside this decode gate. Clean regressions: 13 C, 8 Python, standalone CUDA kernels, and tokenizer 10,000/10,000 green.

## Phase 6 — MTP speculative decoding

- [x] MTP block per Phase-1 findings: zero-centered pre-embedding/pre-hidden norms → concat projection → one full-attention+MoE layer → MTP norm → shared lm head. `convert_qwen.py --mtp` now performs a resumable additive int8 upgrade of an existing container, replacing the planned standalone repair script.
- [x] Lossless speculative driver: MTP draft → selectable exact CPU or CUDA target-block verification → accept-on-match; GDN recurrent snapshot/replay on rejection; prompt preparation for the MTP attention cache; telemetry, `MTP=0` kill switch, and adaptive pause below `MTP_MIN_ACCEPT`.
- [x] Tiny oracle has executable MTP-head/logit references at fp32, int8, and mixed int4+MTP-int8. Integration tests cover all three references, successful batching, forced-rejection rollback, and output identity.
- [x] CUDA verification increment: device-resident GDN/KV state, device-to-device recurrent snapshots, rejection restore/KV rewind, fused multi-token GDN, and the Phase-5 attention/MoE/lm-head path. The mixed-precision transaction batches int4 routed experts while retaining the int8 shared-expert path. The initial 1.221→1.116 s MoE result motivated the path; Experiment 2 subsequently made its arithmetic order exact and superseded the earlier end-to-end timing.
- [x] Multi-domain correctness control: MTP now retains hidden states from the normal chunked target prefill instead of preparing a numerically different sequential target state. The routed-MoE batch kernel matches the single-token multiply and top-k reduction order; standalone and 480 real-model comparisons have maximum difference zero. All 15 D1/D2/D3 streams match five D0 baselines.
- [x] Stage-specific cache attribution corrected the MTP expert tensor key and placed encountered MTP experts in a separately accounted non-evicting CUDA pool. The current five-domain D1 run has zero draft misses, 1.031× paired geometric speed, and exact output; the measured pool costs 0.42–0.48 GiB.
- [x] Confidence-aware admission uses the MTP top-two logit margin as an opt-in gate. `MTP_MIN_MARGIN=2` skips low-confidence transactions, removes measured replay, preserves exact output, reaches 1.104× across five domains, and raises the official median to 35.25 tok/s.
- [x] CUDA event attribution resets after warmup and separates routed hidden/down/reduction, shared hidden/scale/down, setup, and download. It exposed that the real checkpoint's int8 shared expert bypassed the all-int4 shared batch path.
- [x] Format-aware int8 shared-expert batching uses one warp per row/sample and fuses the scaled shared down projection into the routed output. The standalone kernel and 320 real-model layer comparisons have maximum difference zero; MoE time fell from 1.021 to 0.816 seconds.

**GATE 6:** ✅ **2026-07-25.** CPU and CUDA verification are lossless against their matching `MTP=0` paths, including forced rejection, kill switch, confidence fallback, multi-domain continuations, and the final shared-expert kernel. Three alternating exact `Hello` pairs with `MTP_MIN_MARGIN=2` are D0 29.47/31.07/29.17 and D1 41.08/40.86/41.28 tok/s; medians are **29.47** and **41.08 tok/s** (**1.394×**). D1 exceeds the fixed **40.29 tok/s** target, accepts 30/30 admitted proposals in every pair, and skips four low-margin proposals without replay. Regressions: 13 C, 13 Python, standalone CUDA, and tokenizer 10,000/10,000 pass. See `docs/research/phase6_experiment_06_shared_expert_gate_closure.md`.

## Phase 7 — Qwen3.5-397B-A17B (tiered)

- [x] Read-only source preflight pins official FP8 revision `ea5b4f81096f3901c91dea97f81324302495781d`: 94 shards, 406,125,181,280 indexed bytes, 128×128 weight blocks, 94,078 inverse-scale tensors. Every shard contains text weights, so none can be skipped. No weight shard was downloaded during the inventory. See `docs/research/phase7_preflight_01_checkpoint_inventory.md`.
- [x] FP8 conversion preflight: raw-bit E4M3FN tests, strict inverse-scale consumption, and metadata-only pairing validate all 94,078 weight/scale pairs with zero orphan, cross-shard, or shape errors. Dry-run predicts 212,634,789,241 bytes (198.03 GiB) for the no-MTP text container.
- [x] Real FP8 probe: one official 1024×4096 shared-expert matrix with an 8×32 BF16 scale grid decodes bit-identically to Transformers 5.14.1 (`maxdiff=0`, SHA-256 `dd7879bd…d847`). Target int8 requantization RMSE is 5.98e-5. See `docs/research/phase7_preflight_03_real_fp8_probe.md`.
- [x] One-shard production pilot: shard 1 converts 1,024 routed matrices into a 2,282,123,368-byte atomic output (SHA-256 `d98c8f57…010630`), all qtype 4; resume state is durable and the committed source is deleted. Remaining shards are authorized. See `docs/research/phase7_preflight_04_conversion_pilot.md`.
- [x] Source conversion: the pinned official **FP8** revision streamed all 94 source shards into a complete no-MTP int4-g128/int8 container. The result is 93 retained output shards, 278,152 physical / 93,078 logical tensors, and exactly 212,634,789,241 payload bytes (the preflight prediction matched exactly); source shard 93 was MTP-only and therefore produced no text-runtime output. A fresh 90-shard resume sweep matched every SHA-256 digest, the last four shards committed atomically, and the exact one-slot `colib doctor` profile passes container, manifest, CUDA, RAM, VRAM, and disk checks. See `docs/research/phase7_experiment_09_conversion_completion.md`.
- [x] Tier/resource plan: the original MTP + 7 GiB expert-cache proposal honestly fails at 23.65 GiB VRAM. CUDA context telemetry leaves 14.66 GiB free, so Gate 7 uses no MTP, `KV_SLOTS=1`, `CTX=4096`, `CUDA_EXPERT_GB=6`, `CUDA_HEADROOM_GB=1`, and `RAM_GB=18` (14.20 GiB post-context VRAM, **26.20 GiB corrected host RAM**, 222.43 GiB disk). The host estimate includes 6.78 GiB of dense/shared weights retained after CUDA preload; the earlier 19.42 GiB figure was incomplete. A later four-slot profile requires a 5.0 GiB CUDA cache. See `docs/research/phase7_preflight_02_resource_plan.md`.
- [x] Per-slot state: `resource_plan.py` computes 188,743,680 bytes GDN matrix state + 8,847,360 bytes conv state and 61,440 bytes/token fp32 KV. One 4,096-token slot is 0.42 GiB; four slots are 1.67 GiB.
- [x] Tier mechanism: `PIPE=1` reserves all layer misses together; `URING=1 DIRECT=1` submits their packed matrices/scales through aligned batched storage with exact `pread` fallback. `AUTOPIN=1` atomically persists per-layer route heat, and `[TIERS]`/`[EMAP]` report the executed path. The stressed tiny int4 oracle remains TF/greedy 32/32 on CPU and CUDA; the current **21 C and 63 Python tests** pass. See `docs/research/phase7_experiment_05_tiered_storage_pipeline.md`.
- [x] Cold/warm storage model: 60 layers ×10 experts ×6.684675 MB = 4.011 GB (3.735 GiB) reads/token in the cold worst case. The complete-model decode counter measures 4.853 GB for four tokens at a 69.75% non-disk hit rate, or 1.213 GB/token. Decode-only attribution measures 3.758 seconds of storage and 1.491 seconds of remaining work. After reserving that compute inside the four-token 2-second gate, the observed storage rate permits about 0.66 GB or 99 misses, requiring approximately 96% non-disk hits rather than the earlier 86% storage-only estimate. See `docs/research/phase7_experiment_06_real_container_io.md`, `docs/research/phase7_experiment_11_cache_telemetry_and_async_allocator.md`, and `docs/research/phase7_experiment_12_io_upload_and_decode_attribution.md`.
- [x] Cache-locality proxy: on Qwen35 with equivalent per-layer cache fractions, 9.4%, 11.7%, and 14.1% cache coverage produced 64.97%, 70.17%, and 74.31% warmed hits and 3.42, 8.01, and 9.00 tok/s, respectively, with identical tokens. Gate 7's nominal two-tier capacity is 12.51%, suggesting roughly 71–73% locality before model-scale effects and cache overlap. This is below the operational target at measured storage bandwidth, so the full 397B gate is explicitly at risk and must measure rather than assume success. See `docs/research/phase7_experiment_07_cache_locality_proxy.md`.
- [x] Predictive-I/O trial: an opt-in per-layer transition learner safely overlaps batched expert reads with attention/GDN and preserves exact tokens, with a completion barrier preventing cache eviction during use. The unfiltered 35B A/B is negative: precision 12.85%, reads 10.31→25.82 GiB, decode 1.86→1.50 tok/s. It is therefore disabled for the first 397B run; lazy allocation and a 0.5 empirical-confidence default retain it only for later controlled sweeps. See `docs/research/phase7_experiment_08_predictive_prefetch.md`.
- [x] Complete-model qualification is preregistered before results: exact 94-shard doctor audit, one-slot resource guard, fixed-corpus finite PPL <50, four retained chat outputs for coherence review, two warm atlas passes, and a token-weighted sustained decode threshold of ≥2 tok/s. `qualify_tiered_model.py` keeps one engine and both expert caches alive across all trials, requires HW/TIERS/EMAP/HITS telemetry, and disables the rejected predictor. See `docs/research/phase7_preflight_05_qualification_protocol.md`.
- [x] First complete-model qualifier: structural integrity, PPL 1.0191, nonempty coherent output, no OOM, required telemetry, and the CUDA-resident graph all pass. Four measured 64-token turns score 0.3159, 0.4342, 0.4160, and 0.3406 tok/s, for a token-weighted **0.3701 tok/s**. The original request-level hit field was found to be non-diagnostic and is superseded by the counters below. See `docs/research/phase7_experiment_10_complete_model_qualification.md`.
- [x] Tier/counter correction and allocator treatment: EMAP/TIERS now include detached device-only experts, engine shutdown saves the learned atlas through the mux, and `CACHE`/`DCACHE` report inclusive and decode-only CPU/GPU hits, misses, direct bytes, and total read bytes. A host-exclusive cache policy was rejected after slowing the bounded control. With an identical restored atlas and identical 1,157 CPU hits, 517 GPU hits, 726 misses, and 4,853,074,050 read bytes, the CUDA stream-ordered allocator improves decode from **0.3363 to 0.7468 tok/s** and reduces expert matmul/upload time from 40.05 to 10.72 seconds. It is now default-on where CUDA memory pools are supported; `CUDA_ASYNC_ALLOC=0` is the tested fallback. See `docs/research/phase7_experiment_11_cache_telemetry_and_async_allocator.md`.
- [x] Storage/upload treatments and decode attribution: persistent `io_uring` removes 6,228 setup operations but slows the same-atlas measured arm from 0.8104 to 0.4995 tok/s, so it remains opt-in. A 0.062 GiB pinned-host arena reduces inclusive expert upload/matmul from 14.48 to 8.68 seconds but measures 0.7691 tok/s, so it is also not default. New `DPERF` baselines after prefill show the accepted path's four-token decode spends 3.758 seconds in expert reads, 1.451 in expert upload/compute, 0.027 in GDN/GQA, and 0.014 in the output head. This rejects attention as the current bottleneck and sets the next experiment at cache locality under the unchanged memory budgets. See `docs/research/phase7_experiment_12_io_upload_and_decode_attribution.md`.
- [x] Decode-set protection and capacity boundary: an opt-in phase-aware policy protects 38 of 48 host entries per layer and can materialize selected device-only experts between requests without contaminating request telemetry. It improves the same-atlas four-token arm from 726 misses/0.759–0.810 tok/s to zero misses/**2.736 tok/s**, proving the CUDA compute path exceeds the gate when warm. Over 64 tokens it falls to **0.772 tok/s**, 67.57% hits, and 83.25 GB read. The measured long-arm budget requires approximately 94.4% hits; atlas coverage reaches that range only near an ideal 88-entry disjoint union, beyond the safe capacity of the installed 32 GB system. A lossless-compression probe averages 75.0% but has an 86.2% median and is insufficient by itself. See `docs/research/phase7_experiment_13_decode_cache_protection.md`.
- [x] Lower-bit treatment and negative closure: reusable grouped 2-bit/3-bit quantizers, resumable sidecars, C decoding, and exact CUDA expansion were implemented and tested without modifying the accepted container. The complete 35B 2-bit sidecar matches only 12/128 teacher-forced positions. Three-bit reaches 1,219/1,280 (95.23%) overall but one frozen prompt is 53/64 (82.81%), below the unchanged 85% rule; early/late mixed splits also fail. Neither format is accepted and no 397B sidecar is generated. With unchanged precision requiring more memory, the installed 32 GiB profile is closed as a measured no-go. See `docs/research/phase7_experiment_14_low_bit_expert_sidecars.md`.
- [x] Qwen397 relaxed 3-bit disposition: the project owner accepts the measured weakening in principle but retires the ~4.3-hour full Qwen conversion because Ornith is the production target. Two complete 64-expert files (654,469,608 bytes) are retained as unindexed research artifacts; they are not a usable sidecar and must not be auto-selected. Reuse the tested quantizer/concurrency mechanism only on Ornith after an Ornith35 quality gate. See `docs/research/phase7_experiment_15_relaxed_q3_profile.md`.

**GATE 7 — CLOSED NEGATIVE FOR QWEN ON REFERENCE HARDWARE:** coherent output, PPL, telemetry, and memory safety pass; sustained decode is **0.772 tok/s**, so the ≥ **2 tok/s** performance criterion does not pass. Qwen remains the architecture/correctness reference. Production 397B capacity work moves directly to Ornith and is not blocked on another full Qwen conversion.

## Phase 8 — Ornith-1.0 (35B control, then direct 397B production target)

- [x] Roadmap disposition: permanently skip the incomplete Qwen397 3-bit sidecar and proceed directly to the pinned Ornith397 FP8 source, the actual end model. No Qwen397 3-bit artifact, benchmark, or gate is required. Preserve its two unindexed pilot files only for reproducibility; reuse the quantization mechanism on Ornith only if the direct Ornith397 tier profile demonstrates a capacity need.
- [x] Read-only official preflight pins 35B-FP8 revision `1ab57ce0…f30ca` and 397B-FP8 revision `8b61f97a…115bb`. Their architecture is Qwen3.5-compatible, but their `compressed-tensors` per-channel FP8 format is distinct from Qwen397 block FP8. Metadata validates 30,880/92,400 pairs with zero orphan, cross-shard, or shape errors; predicted no-MTP outputs are 19,081,810,684/212,634,789,241 bytes. See `docs/research/phase8_preflight_01_ornith_registry_and_protocol.md`.
- [x] Registry and termination plumbing: Hub conversion writes an explicit `colib_model_family=ornith-1.0` plus source repo/revision because upstream retains `model_type=qwen3_5_moe`. The C loader reads `generation_config.json` and stops on both 248046 and 248044; pad is 248044.
- [x] Protocol mechanism: render the `chat_template.jinja` stored with each snapshot (35B and 397B templates differ). The live OpenAI/Anthropic gateway and interactive CLI dispatch through the explicit family marker, while ordinary Qwen keeps its byte-exact renderer. The actual released tool format is Qwen3 XML, not the planned Hermes JSON; `ornith_protocol.py` separates reasoning and converts `<function=…><parameter=…>` blocks to OpenAI `tool_calls`. OpenAI JSON-string arguments are normalized back to objects only for history rendering, and the official 35B template passes gateway render → parse → tool result → next-assistant round trip. Invalid argument JSON fails explicitly. See `docs/research/phase8_experiment_02_tool_history_round_trip.md`.
- [x] Generated tool-use acceptance is preregistered at the real HTTP boundary: `qualify_ornith_tools.py` requires one `get_weather(city="Paris")` call with `finish_reason=tool_calls`, replays a deterministic 18 °C result, then requires a nonempty final answer that uses 18, leaks no second/raw XML call, and terminates with `finish_reason=stop`; it retains both OpenAI envelopes plus PERF/expert telemetry. This executes after Ornith-35B conversion and numerical qualification. See `docs/research/phase8_preflight_03_e2e_tool_gate.md`.
- [x] Official numerical reference pinned without downloading weights: publisher revision `383064f7…89aac`, `ornith-1.0-35b-Q4_K_M.gguf`, 21,166,757,760 bytes, SHA-256 `ff25291b…ec002`. Gate 8 retains Phase 4's ≥85% per-prompt teacher-forced threshold, adds ≥90% aggregate and ≤5% relative PPL-delta bounds, and treats free-running equality as diagnostic. See `docs/research/phase8_preflight_04_official_gguf_reference.md`.
- [x] Ornith-35B numerical/tool sub-gate: the pinned 21,166,757,760-byte official GGUF has the expected SHA-256; teacher-forced agreement is 1,232/1,280 = 96.25%, all 20 prompts exceed 85% (weakest 90.625%), and perplexity is 1.108495 versus 1.1086 (-0.00943%). The live HTTP path emits exactly `get_weather({"city":"Paris"})`, consumes the deterministic 18 °C result, gives a coherent answer, and terminates with `finish_reason=stop`; all 1,720 measured layer-forwards use device MoE. See `docs/research/phase8_experiment_07_ornith35_numerical_and_tool_gate.md`.
- [x] Ornith-397B production-shard pilot: the first of 122 pinned source shards commits atomically as a 1,103,829,815-byte output with SHA-256 `beb6a1f7…5707c`; the source is released, staging returns to 45 MB, the hash-bound ledger is resumable, elapsed time is 294.25 s, and peak RSS is 14,758,896 KiB. At 10/20/30 committed shards, measured data projects to 212,590,265,325/213,339,074,381/213,588,677,400 bytes, respectively -0.0209%/+0.3312%/+0.4486% from the 212,634,789,241-byte metadata prediction; only the final exact-byte comparison will close completeness. Two foreground interruptions were recovered by re-hashing 16 and then 18 committed outputs. Detached WSL supervision reached 60/122 durable outputs with 105,173,135,036 payload bytes and 137,539 physical tensors; shard 60 hashes to `8a397b6b…7143ee`. Its large-shard linear projection was 213,852,041,240 bytes, +0.5725% from the exact metadata prediction and consistent with the +0.5503% round-50 projection. After a deliberate pause at 91/122 and successful replay of all 91 hashes, conversion reached 100/122 with 175,372,651,321 payload bytes, 175,404,112,689 output-file bytes, and 230,254 physical tensors. The round-100 projection is 213,954,634,612 bytes, +0.6207% from the metadata prediction; shard 100 hashes to `6153d7ab…afd4b`. See `docs/research/phase8_experiment_08_ornith397_streaming_pilot.md`.
- [x] Bounded transfer lookahead: `prefetch_conversion_shards.py` reads but never writes the durable ledger, validates the exact pinned Hub source, and fetches only the second uncommitted source shard into Hugging Face's per-file-locked staging cache while the converter owns the first. It retains the 100 GiB disk guard and exits with the converter. Four deterministic tests cover source-index normalization, the one-shard window, final-shard termination, and source mismatch. The live companion began at 55/122 with shard 57 selected while the converter acquired shard 56. The first four prefetched small shards committed in 187, 70, 91, and 81 seconds versus 322–477 seconds before lookahead. One large arm increased to 796 seconds, so individual-arm non-regression is not claimed. Excluding two restart-contaminated pairs, ten valid lookahead pairs now have a 619.0-second median and 522.8–877-second range versus the 996-second median and 967–1,042-second baseline range, a 37.8% median improvement. The five clean post-recovery pairs have a 544.2-second median. Retain the bounded helper based on total pipeline throughput.
- [x] Original Ornith-397B qualification preregistration: the direct profile fixed the full-ledger SHA-256 audit, exact 93,078-logical-tensor/212,634,789,241-byte checks, finite PPL <50 corruption smoke, four family-template coherence prompts, one-slot 18 GiB RAM/6 GiB VRAM tier telemetry, generated HTTP tool round trip, and the then-current ≥2 tok/s sustained threshold. This historical protocol produced the negative direct result and is superseded prospectively, not rewritten. See `docs/research/phase8_preflight_06_ornith397_qualification_protocol.md`.
- [x] Resumable Gate-8 supervisor: `run_ornith397_gate8.py` validates the exact pinned completion manifest, rebuilds the CUDA engine, then runs the hash-verifying doctor, PPL smoke, tier qualification, and HTTP tool gate in order. Its revised tier vector now requires ≥0.85 tok/s plus persistent `io_uring`, pinned upload, decode protection, and inter-request prewarming; exact manifest, engine, command, and artifact hashes still guard resume. Focused controller/auditor tests include the strict 0.849 failure boundary.
- [x] Direct Ornith397 structural and numerical validation: conversion published the exact preregistered 122-output, 93,078-logical/278,152-physical-tensor, 212,634,789,241-byte manifest. The independent doctor rehashed all 122 shards successfully. The fixed 510-token corruption smoke is finite at PPL 1.050067545 (NLL 24.915790427), below the preregistered limit of 50.
- [x] Ornith397 performance remediation (negative closure on reference hardware): the first frozen four-prompt qualification is coherent, CUDA-resident, and telemetry-complete but fails throughput at 0.527355 tok/s. It records a 60.40% aggregate host/device expert hit rate, 1.839 TiB of direct expert reads, 233,024 `io_uring` setups with zero reuse, stable roughly 27 GB RSS/13.2 GB VRAM, and no host-MoE fallback. The best bounded non-lossy pilot improves to 0.867369 tok/s and proves persistent rings (3 setups/13,169 reuses), pinned upload (149.364 GiB staged), and decode protection work, but remains 56.6% below target. Duplication-aware replacement, dense-host growth, and the guarded disjoint atlas are all rejected; the final exact guard shows the 4,779-pair route set still exceeds safe per-layer RAM plus pinned VRAM capacity by 135 experts. The Ornith35 grouped-3-bit control is rejected on its perplexity limit, so Ornith397 int3 is prohibited. Further progress requires changed hardware, threshold, or an explicitly new algorithmic/kernel objective rather than another residency retry.
- [x] Ornith35 grouped-3-bit control (negative closure): a safe cache-only host recovery replaced the WSL sparse transition because WSL refused sparse mode without the corruption-risk `--allow-unsafe` override. The converter now fails closed on unledgered chunks, supports explicit validated adoption, and atomically publishes incomplete progress manifests. It adopted 62 files, converted 98, and completed 160 files/92,160 tensors/13,098,786,560 bytes; an independent pass verified every SHA-256. The fixed 20×64 comparison passed at 1,223/1,280 (95.546875%) with a worst prompt of 55/64 (85.9375%). The fixed-corpus PPL was 1.226664355, however, 10.660342% above the accepted int4 value of 1.108495 and beyond the preregistered 5%/1.16391975 ceiling. The treatment is rejected before coherence/tool controls; no Ornith397 int3 artifact may be created. See `docs/research/phase8_experiment_09_ornith35_q3_control.md`.
- [x] Lossless warm-route atlas (negative closure): independent decode-route instrumentation corrected the prior `HITS` proxy from 2,934 to 4,779 unique layer/expert pairs. The exact atlas needs 1,901 device slots after filling 48 host slots per layer, but 6 GiB provides only 963; the shortfall is about 5.84 GiB. Both guarded pilots retained the exact reference text but never activated, kept roughly 2,393 decode misses/15.99 GB reads, and remained below target. The experimental runtime/protocol branch was removed. See `docs/research/phase8_experiment_10_warm_route_atlas.md`.
- [x] CUDA dense-host release (negative closure): the fail-closed implementation passed tiny 32/32 teacher-forced and greedy parity, byte-identical mux output, injected-failure termination, and the safe two-stage memory plan. On Ornith397 it verified 376 CUDA matrices, released 5.266 GiB, and grew the host tier from 48 to 64 experts/layer (23.906 GiB). The fixed output stayed byte-identical and decode misses fell 37.34%, but only 3,923 unique host/device pairs were resident because 880 device entries duplicated host entries. Residual expert execution rose from 6.876 s to 108.913 s and throughput collapsed from 0.867369 to 0.133596 tok/s. The branch is rejected, the four-prompt run is prohibited, and all experimental runtime/protocol hooks were removed. See `docs/research/phase8_experiment_11_cuda_dense_host_release.md`.
- [x] Disjoint expanded warm atlas (negative closure): tiny 32/32 parity, sequential mux identity, disjoint-union validation, and injected fail-closed behavior passed. The safe 66-per-layer real profile planned 27.588/29.375 GiB, but the exact 4,779-pair warm set required 1,088 pinned device experts while only 953 fit after the live-transaction reserve. Per-layer imbalance stranded 269 of 3,960 host slots; the usable union covered 4,644 pairs and remained short by 135 experts/0.840454 GiB. The guard rejected before preload, so the 0.292602 tok/s fallback is not an atlas result. The branch was removed, and residency-only remediation is closed negatively on the reference hardware. See `docs/research/phase8_experiment_12_disjoint_expanded_atlas.md`.
- [x] Direct Ornith-397B conversion and validation (negative acceptance closure): the published int4-g128/int8 manifest exactly matches 122 source/output shards, 93,078 logical tensors, 278,152 physical tensors, and 212,634,789,241 payload bytes. Independent doctor and numerical smoke pass. The frozen direct tier run is coherent and CUDA-resident but fails the then-current ≥2 tok/s requirement at 0.527355 tok/s, so the ordered Gate-8 supervisor correctly stops before generated-tool qualification. This historical int4 decision was later superseded by the owner-selected q3 branch.
- [x] Ornith-35B structural conversion: all 16 pinned FP8 source shards produced a complete 93,277-physical/31,333-logical-tensor container with exactly 19,081,810,684 payload bytes, matching the preflight prediction; independent doctor/resource audit passes. See `docs/research/phase8_experiment_06_ornith35_conversion.md`.
- [x] Optional `-1M` stretch disposition: the official publisher registry has no `-1M`/YaRN checkpoint; both pinned Ornith configs declare 262,144 positions with no `rope_scaling`. Full-attention-only sizing gives one 397B 1M slot 60 GiB fp32 / 30 GiB bf16 KV before weights and caches, so no safe reference-hardware profile exists. Retain the existing context clamp and resource guards; reopen only for an immutable official configuration. See `docs/research/phase8_preflight_05_1m_stretch_disposition.md`.
- [x] Revised Ornith397 production qualification (negative closure): the complete two-warm-pass, four-prompt, 64-token optimized lossless profile produces coherent nonempty output, complete tier/route telemetry, real CUDA residency, and 46,080/46,080 device-MoE layer-forwards with zero host fallback. Persistent rings, pinned upload, decode protection, and prewarming execute, but the four measured turns sustain 0.631963761 tok/s versus the revised ≥0.85 requirement. The ordered generated-tool step is not run. See `docs/research/phase8_preflight_07_revised_production_threshold.md` and `docs/research/phase8_experiment_13_revised_threshold_qualification.md`.
- [x] Owner-approved q3 quality amendment: the original 5% PPL rejection remains historical, and the measured Ornith35 q3 result is accepted as calibration for the explicitly lossy Ornith397 branch. The owner later waived the uncompleted Ornith397 PPL run; this changes the prospective protocol but does not retroactively rewrite the Ornith35 result. See `docs/research/phase8_preflight_08_owner_accepted_q3.md`.
- [x] Qwen storage retirement: removed the ignored full Qwen35/Qwen397 containers, source caches, and regenerable Qwen-only bench directories, releasing about 220 GB inside ext4 (439→659 GB free). The Windows VHDX was not compacted. The tiny Qwen oracle, fixtures, implementation/tests, small benchmark JSON, research reports, and 20 GB Ornith35 GGUF reference remain. Removed model data is not recoverable locally but is reproducible from the documented Hub revisions.
- [x] Historical Q3 qualification guard: added an independent 480-file/276,480-header/hash auditor and a resumable controller bound to the base/q3 manifests, int4 PPL baseline, CUDA engine, exact commands, and artifact hashes. The original ≥0.85/12%-PPL protocol remains preserved in its preregistration report. The active controller and separate q3 audit profile now encode the later owner-approved PPL waiver and ≥0.70 sustained floor as an explicit schema-2 amendment. See `docs/research/phase8_preflight_09_q3_qualification_automation.md`.
- [x] Ornith397 grouped-int3 conversion: the atomic, resumable, hash-ledgered converter completed at 60 layers, 512 experts/layer, 480 files, 276,480 tensors, and 157,073,113,440 data bytes with no temporary residue. The independent audit rechecked every hash and tensor header; `expert-q3.json` hashes to `5231fbe7…18f0`.
- [x] Ornith397 expanded-q4 grouped-int3 throughput profile: four measured 64-token turns sustain 0.718432662 tok/s with a 0.8658355 median and a 0.45495 minimum; output is nonempty, CUDA residency is complete, and host-MoE fallback is zero. The owner accepts this result and selects q3. Artifact SHA-256: `047fdf9d6d8e5fcbd494dcdbb1d8d4ac752f799dae9bf3f10f456c0223a8d9f7`.
- [x] Native packed-q3 and adaptive hot-route atlas implementation: dedicated CUDA kernels pass their fp32/fp16, single, grouped, and batched unit comparisons. Because full-model parity/performance is still unproven, expanded-q4 is the default q3 CUDA representation; `Q3_NATIVE=1` and `Q3_ROUTE_ATLAS=1` are explicit experimental switches. See `docs/research/phase8_preflight_10_native_q3_hot_route_atlas.md`.
- [x] Q3 release-policy amendment: the manifest-bound controller and independent 20-check auditor encode the owner-approved Ornith397 PPL waiver and ≥0.70 sustained floor. Teacher-forced, coherence, tool, manifest, CUDA-residency, Gate-9, and clean-release requirements remain unchanged. Focused policy/runtime tests and CUDA suites pass.
- [x] Q3 Gate-8 execution: the schema-2 doctor, frozen int4 reference (`a79219e2…9bf0`), repaired q3 teacher forcing (`462dce23…c5ae`), repaired four-prompt coherence (`21bc37fd…acd83`), semantic acknowledgement, and generated HTTP tool control (`051917aa…e75`) pass. The unchanged complete production-tier retry sustains 0.836866773 tok/s with a 0.8581515 median and 0.773247 minimum. All measured turns exceed 0.70, all 46,080 MoE layer-forwards use CUDA, host fallback is zero, and artifact `7d4d1fe8…c4106` closes the controller passed.

**GATE 8 — PASS:** the selected Ornith397 q3 profile passes the complete manifest, teacher-forced, semantic-coherence, generated-tool, CUDA-residency, and owner-approved ≥0.70 sustained-throughput protocol. The binary-bound full retry reaches 0.836866773 tok/s sustained with zero host-MoE fallback. Gate 9 is authorized.

## Phase 9 — Server, web UI, sessions

- [x] Mux framing primitive: bounded line parser plus exact byte-counted `SUBMIT`, `CANCEL`, `READY`, `DATA`, `ERROR`, and `DONE` writers. Newline-containing payloads pass; malformed headers recover; truncated/oversized payloads fail connection-fatally rather than desynchronizing the stream. This is framing only, not continuous batching. See `docs/research/phase9_preflight_01_mux_framing.md`.
- [x] Scheduler state primitive: 1–16 slots, globally unique in-flight IDs, stable decode-row collection, explicit prefill/decode/cancel-pending/done-pending transitions, length accounting, and release-before-reuse. Error-code and reuse tests pass without model state. See `docs/research/phase9_preflight_02_scheduler_state.md`.
- [x] CPU/CUDA target session snapshot: save/restore position, last normalized hidden state, every GDN recurrent matrix and convolution tail, and used full-attention KV. The tiny hybrid model regenerates an eight-token continuation exactly after restore at position 6 (`recurrent=40,960`, `KV=1,536` fp32 scalars) on both CPU and CUDA; two different slots also alternate every token with exact IDs and byte-identical state. Device buffers are authoritative only when their recorded position matches the model, fixing the chunked-prefill stale-device case. MTP sessions still reject pending complete drafter state. See `docs/research/phase9_preflight_03_session_state.md`.
- [x] Disk checkpoint primitive: versioned `COLISESS` header, architecture/count validation, FNV-1a payload checksum, file `fsync`, atomic rename, and directory `fsync`. CPU and CUDA regenerate the same eight tokens after a file reload, and a one-bit corruption is rejected. The current v1 format is local/native and target-only; token history, MTP, and cross-architecture portability remain pending. See `docs/research/phase9_preflight_04_disk_checkpoint.md`.
- [x] Exact-extension mechanism: save after a prompt plus four generated tokens, restore, consume a two-token extension, and compare with fresh full prefill of the identical concatenated history. Eight subsequent greedy tokens match exactly on CPU and CUDA. Prefix detection/token-history ownership is still a server task; only exact extensions may use this path. See `docs/research/phase9_preflight_05_exact_extension.md`.
- [x] Functional mux reference: `SERVE_BATCH=1 MTP=0` now connects framing, scheduler, tokenizer, per-slot histories, exact-prefix restore/full-prefill fallback, target state persistence, byte-counted `DATA`, cancellation, slot release/reuse, and `DONE`. Tiny integration tests stream three tokens, prove a queued cancel emits no data before slot reuse, and isolate two simultaneous active request IDs. Sampling is greedy-only and decode rows are sequential state switches, so this does **not** close the continuous-batching item. See `docs/research/phase9_preflight_06_functional_mux.md`.
- [x] Initial OpenAI HTTP gateway: vendored upstream's dependency-free gateway at pinned commit `81f08a09…c0be`, changed engine startup to the mux `READY` handshake, defaulted to `qwen`, and replaced GLM rendering/parsing with Qwen3.5 ChatML plus Qwen3-XML tools. A representative tool-history prompt byte-matches the snapshot's official Jinja template; real tiny-engine dispatch and non-streaming/streaming HTTP tests pass. Qwen reasoning is separated from public content even when `</think>` spans stream chunks. The gateway explicitly rejects sampling and structured output until the mux implements them. See `docs/research/phase9_experiment_07_openai_gateway.md`.
- [x] Session-switch cost model and allocator fix: `SessionState` now retains geometrically grown recurrent/KV/hidden buffers, and mux slots retain their logits buffers. The tiny CPU/CUDA test proves snapshot buffer addresses remain stable across an advancing token. Resource planning now reports the unavoidable copy cost of the sequential reference: at 4K context it is **0.44 GiB per active token for 35B** and **0.84 GiB for 397B**, confirming that resident batched state is required rather than an optional optimization. See `docs/research/phase9_experiment_08_session_switch_cost.md`.
- [x] First resident-state kernel seam: CPU `gdn_forward_slot_batch` performs QKVZ/A/B/output projections across active rows and updates row-major per-slot convolution and DeltaNet recurrent state in place, without `SessionState` copies. Three slots at different positions match three sequential layer evaluations exactly for output, recurrence, and convolution state. See `docs/research/phase9_experiment_09_resident_gdn_rows.md`.
- [x] Resident full-attention seam: CPU `attn_forward_slot_batch` batches Q/K/V/O projections for rows at different positions, applies per-head zero-centered norms and partial RoPE, and reads/writes row-major per-slot GQA KV. Three slots match sequential output within `6.98e-10` and their key/value caches exactly. See `docs/research/phase9_experiment_10_resident_gqa_rows.md`.
- [x] Resident hybrid-layer composition: `layer_forward_slot_batch` now combines per-row normalization, resident GDN/GQA, residuals, the existing grouped routed/shared MoE, and the second residual. A two-row GDN+MoE layer matches independent sequential layers within `7.45e-09`, with exact recurrence and convolution state. See `docs/research/phase9_experiment_11_resident_hybrid_layer.md`.
- [x] Whole-model resident CPU batch: `ResidentBatchState` owns per-slot recurrent/conv/KV/hidden/logit storage; `resident_forward_tokens` runs embeddings, all hybrid layers, final norm, and batched LM head for arbitrary active slot IDs. Two slots over eight decode steps are token-exact against independent sequential models (`hidden 7.15e-07`, `logits 2.09e-07`, recurrent `2.46e-07`, KV `9.54e-07`). `SERVE_RESIDENT=1` connects it to the mux and emits identical token bytes/lifecycle events to the snapshot reference. CUDA-resident state and performance measurement remain open. See `docs/research/phase9_experiment_12_whole_model_resident_batch.md`.
- [x] `SERVE_BATCH=1` core mux protocol byte-per `docs/serve_protocol.md`: READY/SUBMIT/CANCEL/DATA/DONE/ERROR plus advisory TIERS/HWINFO/EMAP/HITS/PERF. The resident CPU path continuously batches active hybrid rows, and the CUDA build preserves identical wire semantics. Startup and turn telemetry is shape-validated through the real gateway; the complete **21 C and 63 Python tests** and focused CUDA mux tests pass. See `docs/research/phase9_experiment_14_runtime_telemetry.md`.
- [x] Manifest-bound Gate-9 controller: `run_ornith_gate9.py` refuses to start unless the release auditor reports every pre-Gate-9 check green and an explicit Gate-8 review acknowledgement is supplied. It rebuilds CUDA, then runs the exact ten frozen Ornith35/397 AB, BA, comparator, 20-trial cancellation, and two-turn web commands; resume is bound to both model-manifest SHA-256 values, the CUDA-engine SHA-256, exact command lines, and artifact hashes. It remains unlaunched until Ornith397 Gate 8 is reviewed. Eighteen focused controller/harness/auditor tests pass.
- [x] End-to-end CUDA production-qualification disposition (not authorized): slot-major state, residual stream, normalizations, hybrid attention, routed/shared experts, residuals, final norm, and LM head remain device-resident; only router control and final outputs cross the host boundary. Tiny exactness, real 35B device-MoE execution, mux concurrency, web streaming, telemetry, manifest guards, reversed AB/BA comparison, and cancellation harnesses are implemented and tested. The frozen Ornith35/397 ten-step production sequence was intentionally not run because its controller's strict preflight reports `ornith397.tier` and `ornith397.tools`; even with `--acknowledge-gate8`, it exits before the CUDA build or first benchmark. Consequently no uncontended Ornith35/397 AB/BA, 20-trial p95, or final Ornith397 web artifact is claimed. See `docs/research/phase9_preflight_07_ornith_production_qualification.md` and `docs/research/phase10_experiment_05_negative_release_disposition.md`.
- [x] Revised Gate-9 production-sequence disposition: the manifest/engine-bound ten-step Ornith35/397 AB, BA, comparator, 20-trial cancellation, and exact-web sequence remains unexecuted because the revised Gate-8 tier check fails. Its ≥0.95 per-order, ≥1.0 geometric-mean, ≤1.0/3.0-second p95, and exact-output rules are unchanged.
- [x] Q3 Gate-9 binding: batch, cancellation, web, AB/BA comparison, and the controller now select q3 only for Ornith397, reject incomplete sidecars, record the exact sidecar manifest SHA-256, and bind resume to Ornith35 int4 plus Ornith397 base/q3 identities. Fifteen focused tests pass. Execution remains prohibited until q3 Gate 8 passes. See `docs/research/phase9_preflight_08_q3_manifest_binding.md`.
- [x] Q3 Gate-9 production sequence: the exact manifest/engine-bound controller completes all ten frozen controls. Ornith35 retains its passing A/B (`81158b57…f75f`), B/A (`52079bb6…9d1a`), ABBA (`ec98cf98…8381`), 20-trial cancellation (`0091fe41…4ea`), and exact-web (`b174733a…ec81`) evidence. Ornith397 restarts from the required A/B boundary and passes A/B (`e703bc93…c1ee`), B/A (`625554c5…2eb6`), ABBA (`b9f95d94…1032`), cancellation (`606c4383…69ce`), and web (`e5aa4d5e…bb36`). Its AB/BA speedups are 1.123471507× and 1.096308874×, geometric mean 1.109807093×; 20-trial cancellation p95 is 1.935733776 s; both web turns equal `colib ready`; every recorded MoE forward is on CUDA with zero host fallback. The final q3 audit passes 20/20. See `docs/research/phase9_experiment_23_gate9_production_execution.md`.
- [x] Vendored `openai_server.py` adapted for Qwen ChatML, `<think>`, and Qwen3 XML tools; pinned/rebranded `web/` uses the implemented greedy and `PERF` contracts; installable `colib` CLI provides chat/serve/web/convert/doctor. The staged install, real static-bundle/gateway/C-engine request, **18 web tests**, production build, and zero-vulnerability production audit pass. See `docs/research/phase9_experiment_17_web_cli_surface.md`.
- [x] Real 35B web/gateway/CUDA qualification: two streamed turns both return exactly `colib ready`; scheduler admission/release and the 5.76 GiB VRAM expert tier remain healthy. A cache-only rendering mode now retains Qwen's generation-time empty thinking envelope, while the public renderer still byte-matches the official template. Exact reuse exposes a 20-token continuation suffix; resident token replay and an opt-in causal block remain slower than batched fresh prefill for this short 43-token transcript, so no TTFT improvement is claimed. The block path is parity-tested behind `SERVE_SUFFIX_BLOCK=1`. See `docs/research/phase9_experiment_18_35b_web_and_prefix_reuse.md`.
- [x] 35B cooperative cancellation qualification: after the first streamed piece, one live slot receives `CANCELLED` in 9.88 s under converter contention while its peer completes normally in 30.31 s; the released slot immediately serves a new request in 5.97 s. Tiny CPU cancel-to-ack is 194 ms. The peer-aware lifecycle is a permanent CPU/CUDA regression. Warm p95 and 397B measurements were assigned to the final production item and then correctly not authorized after Gate 8 failed. See `docs/research/phase9_experiment_20_cancellation_latency.md`.
- [x] Cancellation qualifier hardening: `bench_cancel_mux.py` now rejects incomplete model manifests, isolates measured profiles from explicit warm-up passes, records snapshot identity and resident-graph counters, verifies exactly one pre-cancel piece, peer completion, and slot reuse, and can enforce a cancel-to-ack ceiling. A real 35B CUDA validation with zero warm-up under converter contention passed at 1.17 s cancel-to-ack; all 120 measured layer-forwards used device MoE and none used host MoE. This validates the harness; the uncontended production p95 gate was not authorized because Gate 8 failed. See `docs/research/phase9_experiment_22_cancellation_qualifier.md`.
- [x] Session persistence with recurrent state: `SESSION_DIR` atomically checkpoints each slot on `DONE` and acknowledged cancellation as token history + last hidden + GQA KV + every GDN `(S, conv tail)` at position T. Startup validates architecture/counts/checksum, restores snapshot or resident mode, and reuses only token-exact extensions; divergence still full-prefills. A new engine process restores a cancelled one-token prompt, consumes a known one-token extension, and emits the same continuation as fresh prefill. The optional periodic NVMe ring is not required. See `docs/research/phase9_experiment_13_warm_session_reload.md`.

**GATE 9 — PASS:** both manifest-bound Ornith profiles pass reversed-order continuous batching, 20-trial cooperative cancellation, exact two-turn web history, scheduler cleanup, and complete CUDA-resident MoE execution. The controller finishes `passed`, and the independent q3 release audit reports 20/20 checks green.

## Phase 10 — Hardening, docs, release

- [x] `doctor.py` container/resource audit: read-only safetensors headers validate offset bounds/non-overlap, duplicate ownership, every index↔shard mapping, safe names, exact `metadata.total_size`, and the optional converter manifest against independent byte/tensor/shard totals. Resource checks query actual NVIDIA total/free VRAM, accept explicit runtime headroom, and fail unsafe disk/RAM/VRAM plans instead of assuming 16 GiB. Qwen35 passes with 64,646 physical tensors across 14/14 shards and 19,929,665,806 payload bytes; the tiny single shard passes and a missing-shard/incomplete-manifest fixture fails hard. See `docs/research/phase10_experiment_01_container_doctor.md`.
- [x] README/environment accuracy: `docs/ENVIRONMENT.md` records the reference GCC 13.3/CUDA 12.9.86/Python 3.12/Transformers 5.14.1/Node 22 stack, CPU/GPU hardware, build modes, the final 21 C + 114 Python + 10k tokenizer + 18 web matrix, production-model qualification commands, experimental switches, and benchmark confounds. README distinguishes passing implementation tests from the failed 397B production criterion. See `docs/research/phase10_experiment_02_reproducible_environment.md`.
- [x] Composite `make check`: the final uncontended clean x86-64-v3 run passes 21 C executables, 10,000/10,000 tokenizer cases, 114 Python tests, 18 web tests, the production web build, zero-vulnerability audit, documentation/source-package gates, and CUDA backend/session validation on the RTX 5070 Ti. The current documentation audit covers 82 documents, 101 links, and 66 reports; the staged source package contains 235 paths with zero violations. See `docs/research/phase10_experiment_03_release_gate.md`, `docs/research/phase10_experiment_05_negative_release_disposition.md`, and `docs/research/phase8_experiment_13_revised_threshold_qualification.md`.
- [x] Source-package cleanup: the historically tracked `c/tests/test_st_pread` ELF is removed from the Git index while its source and local rebuild remain intact; both native and `.exe` outputs are ignored. `check_source_package.py` makes this invariant part of `make check` by inspecting Git's index for ELF/PE signatures, model/build suffixes, model directories, the web bundle, and engine binaries. Four positive/negative fixtures pass, including a valid PE header versus harmless `MZ` text. The complete staged prospective set contains 232 paths with zero violations, and `git diff --cached --check` passes.
- [x] Final evidence-audit mechanism: `audit_release.py` binds both pinned Ornith manifests to 13 Gate-8/9 artifacts and independently rechecks exact container cardinality/precision, the Ornith397 122-shard header/ledger/SHA-256 doctor result, teacher-forced totals, PPL, generated tools, the revised Ornith397 ≥0.85 tok/s optimized tier path, raw AB/BA ratios, 20-sample cancellation p95, exact web content, frozen resource profiles, and resident CUDA telemetry. The boundary fixture passes at 0.86 and an explicit 0.849 fixture fails. The prior strict audit remains negative until new evidence replaces the old direct artifact. See `docs/research/phase10_preflight_04_release_evidence_audit.md` and `docs/research/phase8_preflight_07_revised_production_threshold.md`.
- [x] Q3 release-audit profile: the historical 15-check audit remains unchanged; an explicit 20-check q3 mode additionally binds the sidecar manifest, 480-file independent doctor, q3 prefix/PPL/coherence/tool/tier evidence, manual coherence acknowledgement, and every Ornith397 Gate-9 artifact. The 20/20 boundary fixture passes and a mixed sidecar identity fails. See `docs/research/phase10_preflight_06_q3_release_audit.md`.

### Controlled benchmark table

All rows use the Ryzen 7 7700X / RTX 5070 Ti reference workstation. Production
rows were collected without converter, download, or build contention.

| Model/path | Workload and controls | Result | Gate status |
|---|---|---:|---|
| Qwen35 CPU | 64-token greedy decode, 8 threads, `EXPERT_RAM=64` | 9.34 tok/s | Gate 4 pass |
| Qwen35 CUDA target | three warmed 64-token samples, 8 GiB device experts | median 31.75 tok/s | Gate 5 pass |
| Qwen35 CUDA target, MTP off | three alternating exact `Hello` controls | median 29.47 tok/s | Gate 6 control |
| Qwen35 CUDA MTP | margin 2, 30/30 admitted accepted, four skips | median 41.08 tok/s, 1.394× | Gate 6 pass |
| Qwen35 two-slot resident batch | converter-contended, two tokens/slot | 0.953× aggregate | diagnostic only; superseded by negative release disposition |
| Qwen397 one-slot tiered | fixed four-prompt, 2× warm + 64-token measured pass | 0.3701 tok/s | Gate 7 fails throughput only |
| Qwen397 bounded allocator A/B | same saved atlas, one prompt, 1× warm + 4-token measured pass | sync 0.3363; stream-ordered 0.7468 tok/s (2.22×) | optimization evidence; still below gate |
| Ornith35 CUDA | 20×64 teacher-forced suite + 510-token PPL + generated HTTP tool round trip | 96.25% TF; PPL -0.00943% vs GGUF; tool/stop pass | Gate 8 35B sub-gate pass |
| Ornith397 tiered | original baseline, short optimized pilot, and full 2× warm/four-prompt/64-token optimized rerun; 18 GiB RAM/6 GiB VRAM | 0.5274 baseline; 0.8674 short pilot; **0.6320 full revised run**; coherent/CUDA-resident; PPL 1.0501 | Gate 8 closed negative vs revised ≥0.85 |
| Ornith35 routed-expert int3 | frozen 20×64 teacher forcing + 510-token PPL | 95.55% TF, worst 85.94%; PPL 1.2267 (+10.66% vs int4) | original 5% rejection; accepted lossy calibration by owner |
| Ornith397 routed-expert int3 | complete q3 conversion; expanded-q4 CUDA execution; frozen quality/tool controls; repaired full ≥0.70 profile | **0.8369 sustained / 0.8582 median / 0.7732 minimum tok/s; all controls pass** | Gate 8 pass |
| Ornith35/Ornith397 resident batch + cancel | frozen family-correct warm AB/BA, exact web, and peer-aware 20-trial cancel sequence | 35B geomean 1.0226× / cancel p95 0.1621 s; 397B geomean 1.1098× / cancel p95 1.9357 s; both exact-web and zero host-MoE | Gate 9 pass |

- [x] Final controlled benchmark and release disposition: the table records every accepted, rejected, and prohibited production result. The strict release audit passes 7/15 checks and fails eight (`ornith397.tier`, the ordered-but-not-run Ornith397 tool gate, and six dependent Gate-9 artifacts). Because Gates 7–9 do not satisfy their acceptance criteria, `v0.1` is deliberately **not** tagged. See `docs/research/phase10_experiment_05_negative_release_disposition.md`.
- [x] Revised release disposition (negative closure): the strict auditor is regenerated against the new 0.85-bound artifact and remains 7/15 with eight failures. The updated full `make check` passes 21 C executables, 10,000 tokenizer cases, 114 Python tests, 18 web tests, documentation/source-package checks, and both CUDA suites. Because revised Gate 8 fails and Gate 9 is prohibited, `v0.1` is not tagged.
- [x] Q3 evidence disposition: the exact sidecar manifest, explicit Ornith397 PPL waiver, unchanged teacher-forced/coherence/tool controls, 0.836866773 tok/s sustained tier result, all Gate-9 artifacts, 20/20 audit, and green full `make check` are recorded in the positive disposition. See `docs/research/phase10_experiment_07_q3_release_disposition.md`.
- [ ] Release publication: review the large prospective release tree, create an intentional clean release commit, and deliberately tag `v0.1`.

**GATE 10 — EVIDENCE PASS; PUBLICATION PENDING:** q3 model evidence, the 20/20 release audit, and the full post-publication `make check` are green. The code and evidence are release-candidate ready. Gate closure still requires an intentional clean release commit and `v0.1` tag; neither is performed implicitly.

---

## Phase 11 — DeepSeek-V4-Flash-0731 replacement

DeepSeek is a new engine, not a Qwen/Ornith alias. Source weights, tokenizer,
encoding, configuration, and reference code are pinned to
`deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.
The Qwen/Ornith engine and compact evidence remain available for tests and
rollback. DeepSeek becomes the documented/runtime default only at Gate 11.6.

### 11.0 Specification, storage, and reproducibility

- [x] One dependency-free contract defines the pinned source, 48 shards,
  inference-critical official config, FP4 experts/FP8 dense quantization,
  43-layer compression schedule, 16K default/64K validated context, supported
  reasoning modes, sampling profiles, and DSpark draft sweep.
- [x] Read-only preflight records command, UTC timestamp, git commit/dirty
  state, sanitized relevant environment, CPU/RAM/GPU identity, storage
  projection, cleanup order, and a machine-readable acceptance result.
- [x] Peak-storage gate requires at least 100 GiB free after the planned native
  container and staging high-water mark. The 2026-08-14 run passes without any
  deletion; Ornith397 remains protected until the paired control is complete.
- [x] Recover from host-volume exhaustion without unsafe sparse-VHD mode:
  preserve exact Ornith35 reconstruction evidence, remove only verified
  Ornith35 weights, trim ext4, compact the detached VHDX, verify the filesystem
  read/write, and retain Ornith397 until its fresh performance control.
- [x] Resumable converter preserves native tensor bytes, validates source
  identity/config/index/header ranges, packs routed experts by
  layer→expert→gate/up/down with 4 KiB alignment, splits fused expert tensors
  without decoding, isolates dense/DSpark segments, and atomically records
  per-record and per-segment SHA-256 evidence. Before writing, the real plan
  must also match all 72,317 pinned tensor names, dtypes, and physical shapes.
- [x] Pinned fetcher downloads metadata and 48 shards with hash-bound atomic
  state. Two stale attempts are closed as interrupted, and attempt 3 resumes
  from 58/62 to a complete 62/62 state without restarting partial shards.
- [x] Deliberately interrupt the real conversion after nine fsynced segments;
  atomic state remains `running`, binds plan
  `ec527bb2d8dad257876e0dd1255e6df50666004f82190a186753a9b26e27ae93`,
  and leaves only segment ten as an uncommitted partial.
- [x] Resume from that boundary and complete all 91 segments. The final
  166,881,088,004-byte segment set and 72,317 records bind manifest
  `468d29fd3262af88ec4c31ef29a94e62631387475917ec7bdb4da9d0441e4d86`.
- [x] Independently compare all 166,878,536,440 native payload bytes with the
  pinned source, rehash every record/segment, reject nonzero padding and
  descriptor drift, and publish dependency-bound atomic evidence with
  signature
  `765b7c2dc2abf7d5941ecda6b769879d12c7eaf1d2ba50781242d6f6c3d8ee1c`.
  The Python and compiled C contracts both accept all 72,317 real descriptors.

**GATE 11.0:** PASS. Pinned metadata/source, independent 72,317-tensor
inventory, deliberate nine-segment interruption, hash-verified resume,
complete conversion, native-byte audit, and Python/C descriptor contracts all
pass. Fixture conversion/resume/corruption controls also pass.

### 11.1 CPU reference correctness

- [x] Vendor the pinned MIT DeepSeek encoder/parser and malformed-output
  recovery; never substitute a generic Jinja template.
- [x] Pin a generated reference fixture for native E2M1/E4M3/UE8M0 formats,
  MXFP activation quantization, mHC/Sinkhorn, YaRN/RoPE, window/compression
  indices, learned pooling, attention sink, routing, and a complete scalar
  routed-expert projection. The official route weight is applied before the
  quantized `w2` boundary.
- [x] Add typed, bounds-checked model-semantic row reads over the native
  container: BF16 token embeddings expand into four hC copies, I64 hash routes
  read only one token row, and the CPU output-head fallback scans bounded
  contiguous BF16 row blocks with exact logits and attributed direct-I/O bytes.
- [x] Replay official decode-time compressor state for both ratio-4 overlap
  and ratio-128-style non-overlap across successive windows, and implement the
  learned ratio-4 indexer's per-head ReLU scoring, aggregation, top-k ordering,
  cache offsets, and invalid-state rejection from the pinned fixture.
- [x] Assemble pure sliding, ratio-4 overlapping/indexed, and ratio-128
  non-overlapping attention through the complete low-rank Q/KV/O projections,
  cache mutation, learned compressed selection, attention sink, inverse RoPE,
  and output projection. Compressed modes use the pinned 160,000 theta and
  factor-16 YaRN configuration.
- [x] Complete the CPU MoE subgraph over converted records: BF16 gate logits,
  token-row hash or biased sqrt-softplus routing, grouped native-FP4 routed
  experts, native-FP8 shared expert, and exact summed output.
- [ ] Complete the separate `deepseek_v4` live mux engine around the tested
  scalar components: allocate/version all 43 layer states and scratch arenas,
  run embedding -> scheduled hC/attention/MoE blocks -> hC head/logits, and wire
  tokenization, sampling, streaming, batching, cancellation, and snapshots.
- [ ] Generate a tiny pinned fixture and compare every mHC, attention, window
  wrap, route, prefill/decode, FP4/FP8, BOS/EOS, and DSpark intermediate.

**GATE 11.1:** all CPU fixture intermediates and greedy/teacher-forced tokens
agree with the pinned layer-streamed oracle within preregistered tolerances.

### 11.2 CUDA correctness and tiered execution

- [x] Load converted native records through the established direct-I/O and
  persistent-`io_uring` reader while retaining and validating exact dtype,
  physical shape, layer, expert, and projection descriptors.
- [x] Add SM120 scalar-correct FP8/FP4 GEMMs and a grouped top-k expert path
  through `w1`/`w3`, clamped SwiGLU, route-aware MXFP requantization, and
  `w2`; the current CPU/CUDA grouped fixture has zero maximum error after the
  official route-order correction. Tensor-core tuning and actual-shape
  performance qualification remain open.
- [x] Add SM120 FP8 dense and fused grouped native-FP4 expert kernels with
  scalar fallbacks and actual-shape equivalence tests. The 4096-hidden x
  2048-intermediate top-6 control is exact on the RTX 5070 Ti.
- [x] Cross-bind the complete 72,317-record Python layout to a compiled C
  startup contract. It validates every base/DSpark tensor name, dtype, and
  physical shape against a sparse 167 GB logical fixture in 1.4 seconds and
  rejects deliberate drift. Manifest ingestion now inserts names into its hash
  table incrementally instead of performing a quadratic duplicate scan.
- [x] Reuse the bounded adaptive host expert tier, LFRU heat, kernel prefetch,
  persistent `io_uring`, direct I/O, and grouped unique-prompt submissions.
  Cold native-FP4 expert triplets batch their six records; warm hits read zero
  bytes and active scheduler references cannot be evicted.
- [x] Add a bounded reference-protected SM120 device expert cache with pinned
  upload staging, hit/miss/eviction/upload telemetry, a persistent FP8 dense
  arena mirror, and a resident BF16 output head. Focused generic, grouped,
  actual-shape, and one-layer runtime parity controls pass.
- [x] Run the first manifest-bound real-weight CUDA forward over all 43 base
  layers. BOS token 0 produces finite logits with top token 5/logit
  16.258379; the run attributes 667 CUDA dense calls, 258 cold routed-expert
  loads, 12,295,250,140 storage bytes, and 12,297,913,712 CUDA upload bytes,
  with no OOM. The correctness smoke passes, but 334.191103-second
  initialization and 118.610136-second cold decode (0.008431 tok/s) fail the
  performance objective. Warm-path isolation and host-stage profiling are
  required before any performance qualification.
- [x] Isolate cold versus resident-token cost with per-layer and per-kernel
  telemetry. The second token still misses 230/258 experts (89.15%) and reads
  3,074,949,120 bytes; all 667 dense projections take only 1.912695 seconds
  of the 84.532588-second decode. This rejects cache-capacity tuning as the
  sole remedy and selects pinned transfer plus persistent direct I/O next.
- [x] A/B pinned staging plus persistent direct I/O against the frozen
  two-token control. Token IDs and logits remain exact and initialization
  improves 50.79%, but the second token regresses 14.40%; reject the combined
  path for decode and microbenchmark buffered/direct/uring combinations before
  retrying. Pinned staging remains separately reversible.
- [ ] Complete decode-route protection, incremental prefix prefill,
  cancellation, embedding residency, and the byte-identical live mux runtime
  around the complete 43-layer state.
- [ ] Validate 16 GiB VRAM bounds, zero silent fallback, exact prefill/decode
  equivalence, batching, cancellation, session restore, and prefix reuse.

**GATE 11.2:** CPU/CUDA fixture parity passes and the real 43-layer path runs
with all fallbacks and storage traffic explicitly attributed.

### 11.3 Serving and interfaces

- [x] Add `--context` to chat/serve/web (16,384 default; 65,536 maximum; reject
  overflow), `--dspark auto|on|off`, DeepSeek model-family dispatch, and model
  ID `deepseek-v4-flash-0731-colib` without changing the default yet.
- [x] Support non-thinking chat and `reasoning_effort=high`; return explicit
  errors for `low`/`max`; accept only deterministic `(0,1)` and agent/tool
  `(1,0.95)` sampling profiles.
- [x] Define and fixture-test an atomic versioned session format binding
  model, tokenizer/protocol, and engine fingerprints plus position, mHC,
  sliding/compressed attention, compressor buffers, and sampler state; reject
  corruption, context drift, and fingerprint drift.
- [ ] Wire the versioned DeepSeek snapshot format into live mux slot save,
  restore, cancellation, and exact-prefix continuation after the full native
  state object is available.
- [x] Emit final `colib.metrics` telemetry with TTFT, prefill/decode time, true
  decode tok/s, token counts, cache hits, and disk bytes; the web client must
  consume it instead of counting streamed text chunks.

**GATE 11.3:** protocol, tool, malformed-output, sampling, mux, batch,
cancellation, session, prefix, 16K boundary, web, and telemetry tests pass.

### 11.4 Real-model correctness and quality

- [ ] Produce a hash-bound layer-streamed Python oracle from the pinned native
  checkpoint; require ≥95% teacher-forced agreement overall, ≥85% per prompt,
  ≤2% perplexity delta, coherent retained outputs, and exact DSML tool calls.
- [ ] Validate 64K continuation/soak without OOM, truncation, state drift, or
  an unreported CPU/CUDA fallback.

**GATE 11.4:** every numerical, quality, tools, long-context, memory, and
manifest criterion passes on the real pinned model.

### 11.5 DSpark and paired performance qualification

- [ ] Keep DSpark off unless an on/off A/B over draft lengths 1–5 proves a
  decode gain without TTFT, stop-rate, memory, or correctness regression;
  `auto` records its decision and evidence.
- [x] Capture the fresh Ornith397 side on the current commit: five independent
  process-fresh legacy-four-prompt trials all pass, with 0.824033698 tok/s
  median sustained decode and 43.508444 s median trial TTFT. The exact command,
  environment, binary/sources, manifests, trial hashes, and reconstruction
  evidence are preserved.
- [ ] Run the DeepSeek side, then complete the paired AB/BA analysis with
  identical hardware state, cache budgets, sampling, workload, warmup, and
  concurrency. The sequential Ornith control block alone is not a paired pass.
- [ ] Require the bootstrap 95% lower confidence bound for DeepSeek tok/s ÷
  Ornith tok/s >1.0 and the 95% upper bound for DeepSeek TTFT ÷ Ornith TTFT
  <1.0; also beat historical 0.836866773 tok/s and 81.9725 s median TTFT on
  the legacy subset with no short-prompt/agent regression.

**GATE 11.5:** pending. Neither weight format nor speculative decoding is
treated as evidence of a speedup.

### 11.6 Promotion and cleanup

- [ ] Only after Gates 11.0–11.5 pass, change CLI/web defaults to
  `c/deepseek-v4-flash-0731`, install the DeepSeek-bound audit, update active
  docs/NOTICE, and pass complete CPU, CUDA, Python, web, docs, and source gates.
- [x] Independently verify Ornith397's five-trial fresh control, source
  identity, exact base/q3 conversion commands, 122 base and 480 q3 shard
  hashes, environment, and reconstruction instructions; preserve the compact
  tracked evidence and tiny legacy fixture.
- [x] Remove only the exact verified `c/ornith397` weight target after the
  control/evidence gate: 369,831,132,527 bytes removed, ext4 free space rose
  to 782,775,443,456 bytes, and detached VHDX compaction reclaimed
  370,328,731,648 host bytes. The pinned DeepSeek source and retirement
  evidence reverify after a clean read/write restart.

**GATE 11.6 / PROMOTION:** pending. Current runtime/documentation defaults stay
on the qualified rollback model until this gate passes.

## Phase 12 - Ornith397 reactivation and device optimization

- [x] Reject DeepSeek promotion and freeze its runtime work.
- [x] Revalidate the pinned Ornith397 revision, 122 base hashes, 480 q3
  hashes, tokenizer/template, five-trial control, and exact converter hashes.
- [x] Run the pinned metadata-only dry-run: 122 source shards,
  405,108,696,032 source bytes, 93,078 kept tensors, and
  212,634,789,241 estimated base output bytes.
- [ ] Complete the shard-streamed int4-g128 base reconstruction and reproduce
  every preserved base hash. Attempt 1 committed 15/122 outputs before ending
  during source shard 16. Attempt 2 revalidated them and committed through
  58/122 before stopping during staged source shard 59; no process is now
  running and no stop cause was retained. Resume from the hash-bound ledger
  `b2f4bbdb...f41f5d9`, preserving the partial source shard.
- [ ] Rebuild the q3 sidecar and reproduce every preserved q3 hash.
- [ ] Re-baseline 0.824033698 tok/s median decode and 43.508444 s median TTFT,
  then optimize both with matched exact-output A/B trials.
- [x] Implement the resumable hash-bound paired controller the preregistered
  protocol needs. `c/tools/run_paired_perf_trials.py` alternates AB/BA per
  pair, supports unmeasured priming trials, refuses resume across binding
  drift, and promotes only on identical outputs with a 95% decode lower bound
  above 1.0 and a TTFT upper bound below 1.0. Twenty focused tests pass, the
  extractor reproduces the preserved five-trial control aggregates exactly,
  and replaying those artifacts holds on a null effect while promoting a
  uniform 1.12 decode / 0.93 TTFT scaling. See `docs/research/phase12_preflight_05_paired_ab_controller.md`.
- [x] Implement and locally validate the first device candidate: q3 on
  NVMe/RAM, expanded-q4 experts in VRAM, and an independent adaptive route
  atlas. Its cached-only grouped-prefill seam matches the CPU q3 reference on
  the RTX 5070 Ti at `1.1641532e-09` maximum difference and advances exact
  batch/token telemetry. Real-model performance qualification still awaits
  reconstruction.
- [x] Implement and locally validate a second reversible TTFT candidate:
  ordered grouped-prefill expert I/O batches. Batch 4 preserves exact tiny
  mux output, 44 hits, 30 misses, and 300 reads while reducing persistent-ring
  submissions from 50 to 41; the production default remains batch 1 until the
  reconstructed real-model protocol passes.
- [x] Record LocalForge carry-forward requirements without modifying
  LocalForge.
- [x] Removed both rejected DeepSeek weight directories after explicit owner
  confirmation on 2026-08-21, freeing 310.6 GiB. All DeepSeek evidence remains
  in tracked artifacts; the weights are re-downloadable.
- [ ] Candidate 3, frozen expert-map cold-start preload: build a diverse-corpus
  heat map excluding the benchmark prompts, freeze it, and measure cold-start
  TTFT and first-turn decode as the preregistered metric. Mechanism is
  implemented and unit-tested; the achievable target is filling every cache
  slot with the hottest experts, about 52% of routing mass, since 80% of mass
  would need roughly twice the cache. See
  `docs/research/phase12_preflight_06_frozen_expert_map_preload.md`.

- [ ] Candidate 4, NVMe read-path throughput: decode achieves 1.40 GB/s against
  5.11 MB misses while expert I/O is 70.2% of decode. Measure the O_DIRECT
  ceiling by block size and thread count first; pursue coalescing the three
  projection reads or raising ring depth only if headroom exists. Estimated
  1.3-1.5x, confidence low until measured.
- [ ] Candidate 5, heat-tiered q2 cold tail: keep hot experts at q3 and demote
  the rarely-routed tail to 2-bit, cutting bytes per miss from 5.11 MB to about
  3.4 MB. Estimated 1.31x. Requires the teacher-forcing, perplexity, and
  coherence gates because roughly 39% of expert activations would be q2.
  Both are recorded in
  `docs/research/phase12_preflight_07_storage_bound_optimization_candidates.md`.

- [ ] Candidate 6, prefix KV caching for a stable 10-16k preamble: the engine
  already checkpoints and prefix-matches session state, but `SESSION_DIR` is
  unset by default. Reusing a stable preamble pays prefill once instead of per
  request, which is the largest available win for long-context first requests.
  Any varying text must sit after the stable block or the common prefix
  collapses.
- [ ] Long-context measurement (blocks the above): one 10k-token prefill with
  telemetry. About 137 GB of prefill expert reads means a single expert-major
  pass; a multiple means the serve path splits the prompt, which would make
  `PREFILL_EXPERT_BATCH` matter considerably more. Also test that flag at 1
  versus 4, since it defaults to the legacy value and was only ever validated
  on the tiny fixture. See
  `docs/research/phase12_preflight_08_long_context_ingestion.md`.

- [ ] Prefill compute for the 10-16k reviewer. Measured TTFT is 33.0 min at
  8,451 tokens and ~70 min projected at 16k, against a 10-minute target;
  prefill is 91% compute and 9% storage, which retires candidates 4 and 5 for
  this workload. Three candidates, largest first: (A) cold experts on CUDA
  during prefill, 14.0 min, needs code because the host path is deliberate and
  its "2.2x slower" basis was measured at short prompts; (B) GDN prefill via
  the existing CUDA block kernel, 9.0 min, flag ready and verified to engage;
  (C) full attention, 6.8 min, diagnosis incomplete and the first proposed fix
  withdrawn. Split GDN/attention timers and recorded `--cuda-attn` /
  `--cuda-spec-gdn` flags are in place; environment variables cannot be used
  because `isolated_engine_env` strips 102 engine keys. See
  `docs/research/phase12_preflight_09_prefill_compute_candidates.md`.

**GATE 12:** Ornith397 becomes the retained working model after byte-exact
reconstruction and current-device correctness. Each optimization must improve
tok/s or TTFT repeatably without quality, CUDA-residency, batch, session, web,
or cancellation regression.

---
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

## Revised q3 definition of done

Production acceptance requires a complete independently verified Ornith397 q3 sidecar, ≥90% aggregate/≥85% per-prompt teacher-forced agreement, an explicit owner-bound Ornith397 PPL waiver, coherent outputs and generated tools, complete CUDA-resident telemetry, sustained decode ≥0.70 tok/s, the unchanged Gate-9 sequence, a 20/20 q3-bound audit, and green `make check`. The validated expanded-q4 CUDA representation is the release default; packed q3 and the adaptive atlas remain optional experiments until their full-model protocols pass. The int4 container remains the rollback path. No `v0.1` tag is authorized before every condition passes.
