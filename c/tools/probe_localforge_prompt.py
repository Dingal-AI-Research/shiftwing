"""Feed localforge's exact orchestrator prompt straight into the engine.

Modelled on stream_qwen_benchmark.py, which drives Engine.generate directly and
is known to produce output (it wrote a full Snake HTML file). The only thing
changed here is the prompt: localforge's real system prompt plus its 36-tool
catalog, rendered through the same render_model_chat the HTTP server uses.

That isolates prompt content from transport: if this emits nothing, the model
genuinely stops on that prompt and the HTTP/SSE layer is exonerated.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from model_prompt import snapshot_model_family  # noqa: E402
from openai_server import Engine, render_model_chat  # noqa: E402
from runtime_env import isolated_engine_env  # noqa: E402

MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "ornith35"
SPEC = json.load(open("/tmp/lf_prompt.json"))
USER = ("Read c/tools/ornith_protocol.py and tell me, with exact line numbers, which "
        "function decides whether a chat request is rendered with the snapshot's own "
        "Jinja template versus the generic Qwen renderer, and what happens if "
        "chat_template.jinja is missing. Do not modify any files.")

CASES = [
    ("user only            ", [{"role": "user", "content": USER}], None),
    ("system + user        ", [{"role": "system", "content": SPEC["system"]},
                               {"role": "user", "content": USER}], None),
    ("user + 36 tools      ", [{"role": "user", "content": USER}], SPEC["tools"]),
    ("system + user + tools", [{"role": "system", "content": SPEC["system"]},
                               {"role": "user", "content": USER}], SPEC["tools"]),
]

env = isolated_engine_env()
env.update({"CTX": "32768", "RAM_GB": "16", "CUDA_EXPERT_GB": "8", "AUTOPIN": "1",
            "CUDA_EXPERT_PREWARM": "1", "COLI_CUDA": "1", "CUDA_DENSE": "1",
            "CUDA_F16": "1", "CUDA_EXPERTS": "1", "SERVE_RESIDENT": "1"})
family = snapshot_model_family(MODEL)
print(f"model family={family}", flush=True)

engine = Engine(HERE.parent / "qwen", MODEL, max_tokens=256, env=env, kv_slots=1)
try:
    for label, messages, tools in CASES:
        rendered = render_model_chat(MODEL, family, messages, enable_thinking=False,
                                     tools=tools, cache_prefix_compatible=True)
        pieces: list[str] = []
        t = time.time()
        stats = engine.generate(rendered, 256, 0.0, 1.0, pieces.append, cache_slot=0)
        out = "".join(pieces)
        print(f"\n=== {label} | prompt_chars={len(rendered)} | {time.time()-t:.1f}s "
              f"| completion_tokens={stats.get('completion_tokens')} "
              f"| length_limited={stats.get('length_limited')}")
        print(f"    RAW OUTPUT ({len(out)} chars): {out[:400]!r}")
        sys.stdout.flush()
finally:
    engine.close()
