#!/usr/bin/env python3
"""Run a small model benchmark while printing every decoded piece immediately."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
CDIR = ROOT / "c"
if str(CDIR) not in sys.path:
    sys.path.insert(0, str(CDIR))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from expert_lowbit import load_complete_expert_sidecar  # noqa: E402
from model_prompt import prepare_user_prompt  # noqa: E402
from openai_server import Engine  # noqa: E402
from runtime_env import isolated_engine_env  # noqa: E402


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _turn(
    engine: Engine,
    rendered_prompt: str,
    *,
    label: str,
    max_tokens: int,
) -> dict[str, Any]:
    pieces: list[str] = []
    first_piece_at: float | None = None
    started = time.monotonic()
    print(f"\n[{label}] assistant> ", end="", flush=True)

    def emit(piece: str) -> None:
        nonlocal first_piece_at
        if first_piece_at is None:
            first_piece_at = time.monotonic()
        pieces.append(piece)
        print(piece, end="", flush=True)

    stats = engine.generate(
        rendered_prompt,
        max_tokens,
        0.0,
        1.0,
        emit,
        cache_slot=0,
    )
    finished = time.monotonic()
    print(
        f"\n[{label}] done: {stats['completion_tokens']} tokens, "
        f"{stats['tokens_per_second']:.6f} tok/s, "
        f"wall {finished - started:.1f}s"
        + (" [TOKEN LIMIT REACHED]" if stats.get("length_limited") else ""),
        flush=True,
    )
    return {
        "label": label,
        "text": "".join(pieces),
        "wall_s": finished - started,
        "ttft_s": (
            first_piece_at - started if first_piece_at is not None else None
        ),
        "stats": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=CDIR / "qwen")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        default="Reply with exactly: shiftwing q3 ready",
    )
    prompt_group.add_argument("--prompt-file", type=Path)
    prompt_group.add_argument(
        "--continue-file",
        type=Path,
        help=(
            "ask the model for only the missing suffix of a truncated response; "
            "--save-response writes the original prefix plus that suffix"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--warmup-turns", type=int, default=1)
    parser.add_argument("--measured-turns", type=int, default=1)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--expert-ram-gb", type=float, default=20.0)
    parser.add_argument("--cuda-expert-gb", type=float, default=6.0)
    parser.add_argument("--expert-q3", action="store_true")
    parser.add_argument(
        "--q3-native",
        action="store_true",
        help="keep q3 expert payloads packed on CUDA (experimental)",
    )
    parser.add_argument("--q3-route-atlas", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--save-response",
        type=Path,
        help="atomically save the final measured response exactly as emitted",
    )
    parser.add_argument(
        "--wait-on-exit",
        action="store_true",
        help="keep an interactive terminal open after the result is printed",
    )
    args = parser.parse_args()

    if args.max_tokens < 1 or args.context < args.max_tokens:
        parser.error("context must cover a positive max-tokens value")
    if args.warmup_turns < 0 or args.measured_turns < 1:
        parser.error("warmup must be nonnegative and measured-turns positive")
    if args.threads < 1 or args.expert_ram_gb <= 0 or args.cuda_expert_gb <= 0:
        parser.error("threads and memory budgets must be positive")
    if args.q3_route_atlas and not args.expert_q3:
        parser.error("--q3-route-atlas requires --expert-q3")
    if args.q3_native and not args.expert_q3:
        parser.error("--q3-native requires --expert-q3")

    continuation_prefix = ""
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        if not args.prompt:
            parser.error("--prompt-file is empty")
    elif args.continue_file:
        continuation_prefix = args.continue_file.read_text(encoding="utf-8")
        if not continuation_prefix:
            parser.error("--continue-file is empty")
        args.prompt = (
            "Continue the truncated HTML document below. Return only the exact "
            "characters that must be appended to it. Do not repeat existing text, "
            "add Markdown fences, or explain. Start immediately after its final "
            "character, complete the remaining JavaScript, and end exactly with "
            "</html>.\n\nTRUNCATED DOCUMENT:\n" + continuation_prefix
        )

    model = args.model.resolve()
    engine_path = args.engine.resolve()
    if not engine_path.is_file():
        raise SystemExit(f"engine is unavailable: {engine_path}")
    family, rendered_prompt = prepare_user_prompt(model, args.prompt)
    lowbit_manifest = None
    if args.expert_q3:
        lowbit_manifest = load_complete_expert_sidecar(model, bits=3)

    env = isolated_engine_env()
    env.update(
        {
            "SERVE_RESIDENT": "1",
            "CTX": str(args.context),
            "OMP_NUM_THREADS": str(args.threads),
            "RAM_GB": str(args.expert_ram_gb),
            "RAM_HEADROOM_GB": "1",
            "COLI_CUDA": "1",
            "CUDA_DENSE": "1",
            "CUDA_F16": "1",
            "CUDA_EXPERTS": "1",
            "CUDA_EXPERT_GB": str(args.cuda_expert_gb),
            "CUDA_HEADROOM_GB": "1",
            "PIPE": "1",
            "URING": "1",
            "DIRECT": "1",
            "URING_PERSIST": "1",
            "CUDA_PINNED_UPLOAD": "1",
            "DECODE_PROTECT": "1" if args.q3_route_atlas else "0",
            "DECODE_PROTECT_PREWARM": "1" if args.q3_route_atlas else "0",
            "EXPERT_Q3": "1" if args.expert_q3 else "0",
            "Q3_ROUTE_ATLAS": "1" if args.q3_route_atlas else "0",
            "Q3_NATIVE": "1" if args.q3_native else "0",
            "AUTOPIN": "1",
            "PREFETCH_LOAD": "0",
            "PREFETCH_THREADS": "0",
        }
    )

    print(
        f"[stream-bench] model={model} q3={args.expert_q3} "
        f"native={args.q3_native} atlas={args.q3_route_atlas} "
        f"warmup={args.warmup_turns} "
        f"measured={args.measured_turns} max_tokens={args.max_tokens}",
        flush=True,
    )
    runtime = Engine(
        engine_path,
        model,
        max_tokens=args.max_tokens,
        env=env,
        kv_slots=1,
    )
    warmup: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    try:
        for index in range(args.warmup_turns):
            warmup.append(
                _turn(
                    runtime,
                    rendered_prompt,
                    label=f"warmup {index + 1}/{args.warmup_turns}",
                    max_tokens=args.max_tokens,
                )
            )
        for index in range(args.measured_turns):
            measured.append(
                _turn(
                    runtime,
                    rendered_prompt,
                    label=f"measured {index + 1}/{args.measured_turns}",
                    max_tokens=args.max_tokens,
                )
            )
    finally:
        runtime.close()

    rates = [float(turn["stats"]["tokens_per_second"]) for turn in measured]
    total_tokens = sum(int(turn["stats"]["completion_tokens"]) for turn in measured)
    decode_seconds = sum(
        int(turn["stats"]["completion_tokens"])
        / float(turn["stats"]["tokens_per_second"])
        for turn in measured
    )
    result = {
        "schema_version": 1,
        "purpose": "interactive directional benchmark; not release evidence",
        "model": str(model),
        "model_family": family,
        "engine": str(engine_path),
        "engine_sha256": _sha256(engine_path),
        "expert_lowbit_manifest": lowbit_manifest,
        "configuration": {
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "warmup_turns": args.warmup_turns,
            "measured_turns": args.measured_turns,
            "context": args.context,
            "threads": args.threads,
            "expert_ram_gb": args.expert_ram_gb,
            "cuda_expert_gb": args.cuda_expert_gb,
            "expert_q3": args.expert_q3,
            "q3_route_atlas": args.q3_route_atlas,
            "q3_native": args.q3_native,
            "continuation_file": (
                str(args.continue_file.resolve()) if args.continue_file else None
            ),
            "continuation_prefix_sha256": (
                hashlib.sha256(continuation_prefix.encode("utf-8")).hexdigest()
                if continuation_prefix
                else None
            ),
        },
        "warmup": warmup,
        "measured": measured,
        "summary": {
            "sustained_tps": total_tokens / decode_seconds,
            "median_turn_tps": statistics.median(rates),
            "minimum_turn_tps": min(rates),
            "outputs_nonempty": all(turn["text"].strip() for turn in measured),
            "rates_finite": all(math.isfinite(rate) and rate > 0 for rate in rates),
        },
        "hardware": runtime.hwinfo,
        "tiers": runtime.tiers,
        "q3_native": runtime.q3_native,
        "q3_route_atlas": runtime.q3_atlas,
        "resident": runtime.resident,
        "profiles": list(runtime.profile),
    }
    if args.output:
        _atomic_json(args.output.resolve(), result)
        print(f"[stream-bench] artifact={args.output.resolve()}", flush=True)
    if args.save_response:
        _atomic_text(
            args.save_response.resolve(),
            continuation_prefix + measured[-1]["text"],
        )
        print(
            f"[stream-bench] response={args.save_response.resolve()}",
            flush=True,
        )
    print(
        f"[stream-bench] measured sustained={result['summary']['sustained_tps']:.6f} "
        f"tok/s (directional only)",
        flush=True,
    )
    if args.wait_on_exit:
        try:
            input("[stream-bench] complete; press Enter to close this terminal ")
        except EOFError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
