# qwen3_5_moe — verified architecture reference for the C engine

Source of truth: **installed transformers 5.14.1** (`.venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`), read line-by-line 2026-07-20. Line refs below are to that file (GitHub main was identical at read time). This answers every ⚠ item in PLAN.md Phase 1.

## Norms — TWO different conventions in one model (critical)

- `Qwen3_5MoeRMSNorm` (L820): **zero-centered weight** — `out = rms_norm(x.float()) * (1.0 + weight.float())`, then cast back. Used for: `input_layernorm`, `post_attention_layernorm`, `model.norm`, and per-head `q_norm`/`k_norm`. Weight init is zeros. **The C engine must add 1.0** (or the converter must bake `w+1` — decide once, document in converter).
- `Qwen3_5MoeRMSNormGated` (L186, DeltaNet output norm): **plain weight** (ones-init, NOT zero-centered): `out = (rms_norm(x.f32) * weight).to(dtype) * silu(z.f32)`, result cast to dtype. Applied **per-head** on `head_v_dim` (weight length = head_v_dim = 128, shared across heads); input reshaped to (-1, head_v_dim) (L554-556).

## Gated DeltaNet layer (`linear_attn`, L371-560)

Projections (transformers 5.14.1 uses **split** projections; no fused qkvz here):
- `in_proj_qkv`: hidden → key_dim*2 + value_dim (key_dim = n_k_heads*128, value_dim = n_v_heads*128); no bias
- `in_proj_z`: hidden → value_dim (output gate, NOT convolved)
- `in_proj_b`: hidden → num_v_heads (β); `in_proj_a`: hidden → num_v_heads (decay a)
- `conv1d`: depthwise Conv1d(conv_dim = key_dim*2+value_dim, kernel 4, groups=conv_dim, **bias=False**), over concat(q,k,v) channels only, then **SiLU** (L498)
- `A_log` [num_v_heads], `dt_bias` [num_v_heads], `norm` (gated RMSNorm above), `out_proj`: value_dim → hidden

⚠ **OPEN ITEM for the converter**: the real 35B/397B checkpoints were saved with transformers 4.57.0.dev0 and may store the **fused** `linear_attn.in_proj_qkvz` (12288 for 35B) + `in_proj_ba` layout (Qwen3-Next style). No `_checkpoint_conversion_mapping` was found in 5.14.1's qwen3_5_moe. **First task of the converter: fetch `model.safetensors.index.json` of Qwen/Qwen3.5-35B-A3B and list actual names.** If fused, determine the split order empirically: load the tiny oracle... no — build a tiny fused checkpoint is impossible; instead load the real index names AND check how transformers 5.14.1 loads that checkpoint (it must work — Qwen3.5 runs on 5.14; find the mechanism, possibly in `conversion` utils or `from_pretrained` renaming). If HF 5.14.1 can genuinely load the 4.57 checkpoint, replicate its mapping; the container stores the SPLIT layout (what the forward math uses).

Conv state (decode): rolling buffer of the **last 4 raw (pre-conv) qkv columns**; decode step = cat(state, new) → depthwise conv valid → last column → SiLU; state ← last 4 columns (L221-236, L470-500). Prefill saves state as `pad(mixed_qkv, (4 - L, 0))` i.e. last 4 raw columns, left-zero-padded if L<4 (L487).

Per-token math (recurrent reference, L326-367; all in **fp32**, state fp32):
```
β = σ(b)                              # [n_v_heads]
g = -exp(A_log) * softplus(a + dt_bias)   # log-decay, fp32 (L519)
q,k: repeat_interleave k-heads → v-heads (ratio = n_v/n_k = 2 for 35B, 4 for 397B) (L520-522)
q̃ = l2norm(q, eps=1e-6);  k̃ = l2norm(k, eps=1e-6)   # per 128-dim head vector (L239)
q̃ *= 1/sqrt(head_k_dim)               # scale applied to q BEFORE recurrence (L339-340)
S ← S * exp(g)                        # per-head [d_k=128, d_v=128]
kv_mem = Sᵀ k̃                        # (L359: sum over d_k)
S ← S + k̃ ⊗ ((v − kv_mem) * β)
o = Sᵀ q̃                             # [d_v]
out_token = RMSNormGated_perhead(o, z) → out_proj
```
Chunked prefill reference (chunk 64, WY decomposition): `torch_chunk_gated_delta_rule` L245-323 — mirror exactly in Phase 3; the recurrent form above is the always-correct fallback and is what HF uses for cached single-token decode.

## Full attention layer (`self_attn`, L646-720)

- `q_proj`: hidden → n_heads * head_dim * **2** — view(..., n_heads, head_dim*2) then `chunk(2, -1)`: **per-head [q(256) | gate(256)]** (L686-689). Gate flattened per token.
- `k_proj`/`v_proj`: hidden → n_kv_heads*head_dim (2×256). `o_proj`: n_heads*head_dim → hidden. All `attention_bias=false` ⇒ no biases.
- Per-head `q_norm`/`k_norm` (zero-centered RMSNorm on head_dim=256) applied **before RoPE**; no v norm.
- RoPE then standard GQA softmax attention, `scaling = head_dim^-0.5`, softmax in fp32 (L638).
- Output: reshape → `attn_out * sigmoid(gate)` (L717) → o_proj.
- KV cache: conventional per-layer K/V, 2 KV heads × 256, K stored **after** k_norm+RoPE.

## RoPE (partial, MRoPE-interleaved — text ⇒ standard 1D)

- rotary_dim = head_dim * partial_rotary_factor = **64**; inv_freq[j] = theta^(-2j/64), j=0..31, theta=1e7 (L133-144).
- Applied to the **first 64 dims** of each head; dims 64..255 pass through (L595-605).
- Within the 64 dims: **rotate_half (split-half) convention** — pair (i, i+32): `q'[i] = q[i]·cos_i − q[i+32]·sin_i; q'[i+32] = q[i+32]·cos_i + q[i]·sin_i` with cos = cat(f,f).cos() (L162, L563-606).
- MRoPE interleaving (L168-183) only swaps which of the 3 position streams (T/H/W) feeds each frequency index; **for text all streams have identical position ids** (L1286-1287: same arange expanded), so the result equals plain 1D RoPE above. attention_scaling = 1.0 for rope_type "default".
- Position ids: model expands to 4 rows (row 0 = text ids for mask, rows 1-3 = T/H/W for rope) — irrelevant for the C engine beyond using absolute position per token.

## MoE block (every layer; L739-817)

- Router `mlp.gate.weight` [n_experts, hidden], no bias: logits → **softmax over ALL experts in fp32** → top-k (8/10) of the probs → **unconditional renormalize** (sum of top-k = 1) (L787-795). (config `norm_topk_prob` is not even consulted in 5.14.1.)
- Experts in HF format are **fused 3D**: `mlp.experts.gate_up_proj` [E, 2·moe_inter, hidden] (rows: gate then up per expert), `mlp.experts.down_proj` [E, hidden, moe_inter] (L748-749; per-expert compute L770-772: `linear(x, gate_up[e]).chunk(2)` → silu(gate)*up → `linear(·, down[e])`). **Converter unfuses to per-expert 2D `gate_proj/up_proj/down_proj`** (colibri container convention, enables (layer,eid) slab streaming).
- Shared expert: standard SwiGLU MLP at shared_expert_intermediate_size, output scaled by `sigmoid(shared_expert_gate(x))` where `mlp.shared_expert_gate.weight` is [1, hidden] (L804, L813). Added to routed output.

## Decoder layer / model skeleton (L840-896, L1246+)

`x += mixer(input_layernorm(x)); x += moe(post_attention_layernorm(x))` — exactly colibri's residual seam. `layer_types[i]` selects `linear_attn` vs `self_attn` (3:1). Text-only stack: `model.embed_tokens.weight`, `model.layers.N.*`, `model.norm.weight`, untied `lm_head.weight`. **`Qwen3_5MoeForCausalLM` exists** (L1788) — use it for the tiny oracle (text-only, no vision). Full multimodal checkpoint prefixes text weights `model.language_model.*` (ForConditionalGeneration).

## MTP

transformers **ignores** MTP weights: `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` (L907, L1793) ⇒ checkpoint stores top-level `mtp.*` tensors, no HF reference implementation. Same situation colibri faced with GLM: implement natively (Phase 6); recover the block structure from the checkpoint tensor names (index.json) + vLLM's qwen3.5 MTP implementation as cross-reference. Config: `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` (shares embed/lm_head).

## Loader name map (tiny-oracle / ForCausalLM layout = container layout)

```
model.embed_tokens.weight                          [vocab, hidden]
model.norm.weight                                  [hidden]        zero-centered
lm_head.weight                                     [vocab, hidden]
model.layers.N.input_layernorm.weight              [hidden]        zero-centered
model.layers.N.post_attention_layernorm.weight     [hidden]        zero-centered
# linear_attention layers:
model.layers.N.linear_attn.in_proj_qkv.weight      [2*key_dim+value_dim, hidden]
model.layers.N.linear_attn.in_proj_z.weight        [value_dim, hidden]
model.layers.N.linear_attn.in_proj_b.weight        [n_v_heads, hidden]
model.layers.N.linear_attn.in_proj_a.weight        [n_v_heads, hidden]
model.layers.N.linear_attn.conv1d.weight           [conv_dim, 1, 4]  (no bias)
model.layers.N.linear_attn.A_log                   [n_v_heads]
model.layers.N.linear_attn.dt_bias                 [n_v_heads]
model.layers.N.linear_attn.norm.weight             [head_v_dim]    plain (ones-init)
model.layers.N.linear_attn.out_proj.weight         [hidden, value_dim]
# full_attention layers:
model.layers.N.self_attn.q_proj.weight             [n_heads*head_dim*2, hidden]
model.layers.N.self_attn.k_proj.weight             [n_kv*head_dim, hidden]
model.layers.N.self_attn.v_proj.weight             [n_kv*head_dim, hidden]
model.layers.N.self_attn.o_proj.weight             [hidden, n_heads*head_dim]
model.layers.N.self_attn.q_norm.weight             [head_dim]      zero-centered
model.layers.N.self_attn.k_norm.weight             [head_dim]      zero-centered
# every layer:
model.layers.N.mlp.gate.weight                     [n_experts, hidden]
model.layers.N.mlp.experts.gate_up_proj            [E, 2*moe_inter, hidden]  (HF fused; container unfuses)
model.layers.N.mlp.experts.down_proj               [E, hidden, moe_inter]    (HF fused; container unfuses)
model.layers.N.mlp.shared_expert.gate_proj.weight  [shared_inter, hidden]
model.layers.N.mlp.shared_expert.up_proj.weight    [shared_inter, hidden]
model.layers.N.mlp.shared_expert.down_proj.weight  [hidden, shared_inter]
model.layers.N.mlp.shared_expert_gate.weight       [1, hidden]
```

## Environment (this machine)

- venv via **uv** (`~/.local/bin/uv`, add `export PATH="$HOME/.local/bin:$PATH"`; system python3-venv is broken/missing and sudo needs a password). `.venv` has CPU torch + **transformers 5.14.1** + safetensors. Oracle version gate: require `>= 5.14`.
