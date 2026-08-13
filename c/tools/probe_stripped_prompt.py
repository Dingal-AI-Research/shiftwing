"""Measure the stripped agent prompts end to end: TTFT and tool-call correctness.

Sizing alone is not the answer -- a smaller prompt is only useful if the model
still emits a call the harness can dispatch. This runs each variant through
Engine.generate (the path the working snake wrapper uses) and prints the raw
bytes so the parameter names can be checked against the real schema.
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

MODEL = Path("/home/dinga/Projects/colib/c/ornith35")
USER = "Read package.json in the workspace and tell me the value of the \"name\" field."

CASES = [
    ("core-15  ", "/tmp/lf_prompt_core.json"),
    ("minimal-36", "/tmp/lf_prompt_min.json"),
]

env = isolated_engine_env()
env.update({"CTX": "32768", "RAM_GB": "16", "CUDA_EXPERT_GB": "8", "AUTOPIN": "1",
            "CUDA_EXPERT_PREWARM": "1", "COLI_CUDA": "1", "CUDA_DENSE": "1",
            "CUDA_F16": "1", "CUDA_EXPERTS": "1", "SERVE_RESIDENT": "1",
            # the probe must not inherit the 120s default: a single attention
            # layer on a multi-thousand-token prompt legitimately exceeds it
            "PREFILL_PROGRESS_STALL_TIMEOUT_MS": "900000",
            "PREFILL_FIRST_PROGRESS_TIMEOUT_MS": "900000",
            "FIRST_MODEL_OUTPUT_TIMEOUT_MS": "2700000"})

family = snapshot_model_family(MODEL)
engine = Engine(HERE.parent / "qwen", MODEL, max_tokens=200, env=env, kv_slots=1)
try:
    for label, path in CASES:
        spec = json.load(open(path))
        messages = [{"role": "system", "content": spec["system"]},
                    {"role": "user", "content": USER}]
        rendered = render_model_chat(MODEL, family, messages, enable_thinking=False,
                                     tools=spec["tools"], cache_prefix_compatible=True)
        first = {"t": None}
        started = time.time()

        def emit(piece: str, _f=first, _s=started) -> None:
            if _f["t"] is None:
                _f["t"] = time.time() - _s
            pieces.append(piece)

        pieces: list[str] = []
        stats = engine.generate(rendered, 200, 0.0, 1.0, emit, cache_slot=0)
        out = "".join(pieces)
        wall = time.time() - started
        print(f"\n=== {label} | prompt_chars={len(rendered)} | TTFT={first['t']:.1f}s "
              f"| wall={wall:.1f}s | completion_tokens={stats.get('completion_tokens')}")
        print(f"    RAW: {out[:500]!r}")
        sys.stdout.flush()
finally:
    engine.close()
