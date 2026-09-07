#!/usr/bin/env python3
"""Build a frozen routed-expert heat map from a profiling corpus.

The engine can seed its expert cache from a persisted heat map, which is what
lets a fresh process start on the experts real traffic routes to instead of
experts 0..cap-1. Producing that map is deliberately not a qualifier feature:
`qualify_tiered_model.py` forces `EMAP_FREEZE=1` so a benchmark can never write
the map it is seeded from, because a self-updating map is hidden mutable state
that makes trials depend on each other. Building one is therefore a separate,
explicit act, recorded here with its provenance.

The corpus must differ from the prompts the map is later measured against.
Seeding from the same prompts that are then benchmarked is test-set leakage: it
reports a gain that real traffic never sees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from openai_server import Engine, snapshot_model_family  # noqa: E402
from tools.model_prompt import prepare_user_prompt  # noqa: E402

MAGIC = b"COLIEMAP"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_map(path: Path) -> dict[str, Any]:
    """Decode the map header and summarize the heat it records."""

    raw = path.read_bytes()
    if len(raw) < 20 or raw[:8] != MAGIC:
        raise SystemExit(f"not a COLIEMAP file: {path}")
    version, layers, experts = struct.unpack("<III", raw[8:20])
    count = layers * experts
    expected = 20 + 4 * count
    if len(raw) != expected:
        raise SystemExit(f"map is {len(raw)} bytes, expected {expected}")
    heat = struct.unpack(f"<{count}I", raw[20:])
    nonzero = sum(1 for value in heat if value)
    ordered = sorted(heat, reverse=True)
    total = sum(heat) or 1
    def covered(fraction: float) -> float:
        take = max(1, int(count * fraction))
        return 100.0 * sum(ordered[:take]) / total
    return {
        "version": version,
        "layers": layers,
        "experts_per_layer": experts,
        "experts": count,
        "bytes": len(raw),
        "sha256": sha256_file(path),
        "nonzero_experts": nonzero,
        "nonzero_percent": 100.0 * nonzero / count,
        "total_routes": total,
        "mass_in_top_5_percent": covered(0.05),
        "mass_in_top_13_percent": covered(0.1285),
        "mass_in_top_30_percent": covered(0.30),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=ROOT / "qwen")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--expert-ram-gb", type=float, default=18.0)
    parser.add_argument("--cuda-expert-gb", type=float, default=6.0)
    parser.add_argument("--ram-headroom-gb", type=float, default=1.0)
    parser.add_argument("--cuda-headroom-gb", type=float, default=1.0)
    parser.add_argument("--expert-q3", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue accumulating heat into an existing map",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model = args.model.resolve()
    engine_path = args.engine.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(args.prompt_file.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts:
        raise SystemExit("prompt file must be a non-empty JSON list")
    if output.exists() and not args.resume:
        raise SystemExit(f"{output} exists; pass --resume to accumulate into it")

    env = dict(
        os.environ,
        SNAP=str(model),
        THREADS=str(args.threads),
        CTX=str(args.context),
        EXPERT_RAM=str(args.expert_ram_gb),
        RAM_HEADROOM_GB=str(args.ram_headroom_gb),
        CUDA_EXPERT_GB=str(args.cuda_expert_gb),
        CUDA_HEADROOM_GB=str(args.cuda_headroom_gb),
        COLI_CUDA="1",
        CUDA_DENSE="1",
        CUDA_F16="1",
        CUDA_EXPERTS="1",
        PIPE="1",
        URING="1",
        DIRECT="1",
        URING_PERSIST="1",
        CUDA_PINNED_UPLOAD="1",
        DECODE_PROTECT="1",
        DECODE_PROTECT_PREWARM="1",
        EXPERT_Q2="0",
        EXPERT_Q3=str(args.expert_q3),
        PREFETCH_LOAD="0",
        PREFETCH_THREADS="0",
        # The point of this tool: learn and persist. Both are off in the
        # qualifier so that measurement cannot mutate its own inputs.
        AUTOPIN="1",
        EMAP_PATH=str(output),
        EMAP_FREEZE="0",
        EMAP_SAVE_EVERY=str(args.save_every),
    )

    # prepare_user_prompt returns (family, rendered); the profiling corpus must
    # go through the same chat template the benchmark uses, or the routed
    # experts it learns would not be the ones serving real turns.
    family = snapshot_model_family(model)
    rendered = [prepare_user_prompt(model, text, family=family)[1] for text in prompts]

    started = time.monotonic()
    engine = Engine(engine_path, model, max_tokens=args.max_tokens, env=env)
    turns: list[dict[str, Any]] = []
    try:
        for index, prompt in enumerate(rendered):
            text_parts: list[str] = []

            def emit(chunk: str, sink: list[str] = text_parts) -> None:
                # Stream the response as it decodes. At well under one token per
                # second a turn takes minutes, so buffering it until the end
                # would leave the log silent for the whole turn.
                sink.append(chunk)
                sys.stdout.write(chunk)
                sys.stdout.flush()

            print(
                f"\n=== prompt {index + 1}/{len(prompts)} ===\n"
                f"{prompts[index]}\n--- response ---",
                flush=True,
            )
            turn_started = time.monotonic()
            stats = engine.generate(
                prompt,
                args.max_tokens,
                0.0,
                1.0,
                emit,
            )
            elapsed = time.monotonic() - turn_started
            turns.append(
                {
                    "prompt_index": index,
                    "wall_s": elapsed,
                    "completion_tokens": (stats or {}).get("completion_tokens"),
                    "tokens_per_second": (stats or {}).get("tokens_per_second"),
                    "text_chars": len("".join(text_parts)),
                    "prompt": prompts[index],
                    "text": "".join(text_parts),
                }
            )
            print(
                f"\n[profile] {index + 1}/{len(prompts)} "
                f"{elapsed:.1f}s {(stats or {}).get('tokens_per_second')} tok/s",
                flush=True,
            )
    finally:
        engine.close()

    if not output.is_file():
        raise SystemExit(
            "no expert map was written; the engine may not have reached a "
            "turn boundary, or EMAP_SAVE_EVERY was 0"
        )
    summary = read_map(output)
    manifest = {
        "schema": "colib.expert-map-profile.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "map": str(output),
        "model": str(model),
        "engine": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "corpus": str(args.prompt_file.resolve()),
        "corpus_sha256": sha256_file(args.prompt_file.resolve()),
        "corpus_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "wall_s": time.monotonic() - started,
        "turns": turns,
        "expert_map": summary,
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"map": str(output), **summary}, indent=2, sort_keys=True))
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
