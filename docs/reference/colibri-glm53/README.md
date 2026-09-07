# Reference: colibri's GLM-5.3-Flash engine

Read-only reference for implementing GLM-5.3-Flash support in shiftwing. Taken
from github.com/JustVugg/colibri (Apache-2.0; see the repository NOTICE, which
already records vendored colibri components).

These files are **not built**. They are here because colibri has a working
engine for this checkpoint and its structure is worth following rather than
rediscovering — this project spent a full cycle rediscovering Qwen4-Exp's
internals from the config alone.

Verified on this machine before the clone was removed:

  glm53 built clean, and colibri's own tiny oracle passed at every precision:
    f32   8 teacher-forced positions exact, 4 greedy tokens exact, logits 9.54e-07
    int8  same tokens, logits 0.0086
    int4  same tokens, logits 0.144

That last line is the one that matters: GLM-5.3-Flash keeps its answers under
int4, which Qwen3.8-Flash-Next did not.

Architecture, read off the released checkpoint by colibri and independently
consistent with this project's own reading of config.json:

  - 45 text layers + 1 MTP, hidden 4096, vocab 154880
  - 34 KDA linear-attention layers, 11 DeepSeek-sparse-attention layers
  - MLA kv_lora 512, q_lora 1536, qk_nope 256, **qk_rope 0** (NoPE)
  - DSA lightning indexer with k-pooling (index_kpool 4, always-select tail)
  - mHC hyper-connections, hc_mult 4, 20 Sinkhorn iterations -- the DeepSeek-V4
    variant, NOT the low-rank sigmoid gate Qwen4-Exp uses
  - 288 routed experts top-8 + 1 shared, from layer 3; first 3 layers dense
  - clamped SwiGLU in the text MLP, not only the vision tower
