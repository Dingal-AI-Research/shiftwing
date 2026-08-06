#!/usr/bin/env python3
"""Compare sequential and concurrent mux requests on one or more KV slots."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from openai_server import Engine  # noqa: E402
from tools.expert_lowbit import load_complete_expert_sidecar  # noqa: E402
from tools.model_prompt import prepare_user_prompt  # noqa: E402
from tools.runtime_env import isolated_engine_env  # noqa: E402


def run_mode(args: argparse.Namespace, mode: str) -> dict:
    env = isolated_engine_env()
    env.update(
        {
            "SERVE_RESIDENT": "1",
            "CTX": str(args.context),
            "OMP_NUM_THREADS": str(args.threads),
            "PREFETCH_THREADS": "0",
            "PREFETCH_LOAD": "0",
            "URING_PERSIST": "0",
            "CUDA_PINNED_UPLOAD": "0",
            "DECODE_PROTECT": "0",
            "DECODE_PROTECT_PREWARM": "0",
            "EXPERT_Q2": "0",
            "EXPERT_Q3": "1" if args.expert_q3 else "0",
        }
    )
    if args.ram_gb is not None:
        env.pop("EXPERT_RAM", None)
        env.update(
            {
                "RAM_GB": str(args.ram_gb),
                "RAM_HEADROOM_GB": str(args.ram_headroom_gb),
                "PIPE": "1",
                "URING": "1",
                "DIRECT": "1",
                "AUTOPIN": "1",
            }
        )
    else:
        env.pop("RAM_GB", None)
        env["EXPERT_RAM"] = str(args.expert_ram)
    if args.cuda:
        env.update(
            {
                "COLI_CUDA": "1",
                "CUDA_DENSE": "1",
                "CUDA_F16": "1",
                "CUDA_EXPERTS": "1",
                "CUDA_EXPERT_GB": str(args.cuda_expert_gb),
                "CUDA_HEADROOM_GB": str(args.cuda_headroom_gb),
            }
        )
    else:
        env.update(
            {
                "COLI_CUDA": "0",
                "CUDA_DENSE": "0",
                "CUDA_F16": "0",
                "CUDA_EXPERTS": "0",
            }
        )
    if args.cuda_events:
        env["CUDA_PROFILE_STAGES"] = "1"
    else:
        env.pop("CUDA_PROFILE_STAGES", None)
    if args.trace:
        env["COLI_ENGINE_TRACE"] = "1"

    startup_started = time.monotonic()
    engine = Engine(
        args.engine.resolve(),
        args.model.resolve(),
        max_tokens=args.max_tokens,
        env=env,
        kv_slots=args.requests,
    )
    startup_s = time.monotonic() - startup_started
    warmup_started = time.monotonic()
    for _ in range(args.warmup_passes):
        for index in range(args.requests):
            engine.generate(
                args.runtime_prompt,
                args.max_tokens,
                0.0,
                1.0,
                lambda _piece: None,
                cache_slot=index,
            )
    warmup_s = time.monotonic() - warmup_started
    profile_base = len(engine.profile)
    barrier = threading.Barrier(args.requests) if mode == "concurrent" else None
    results: list[dict | None] = [None] * args.requests
    errors: list[str] = []

    def request(index: int) -> None:
        pieces: list[str] = []
        first_content = [None]
        if barrier is not None:
            barrier.wait()
        started = time.monotonic()

        def emit(piece: str) -> None:
            if first_content[0] is None:
                first_content[0] = time.monotonic()
            pieces.append(piece)

        try:
            stats = engine.generate(
                args.runtime_prompt,
                args.max_tokens,
                0.0,
                1.0,
                emit,
                cache_slot=index,
            )
            finished = time.monotonic()
            results[index] = {
                "slot": index,
                "ttft_s": (
                    first_content[0] - started
                    if first_content[0] is not None
                    else None
                ),
                "request_s": finished - started,
                "text": "".join(pieces),
                "stats": stats,
            }
        except Exception as error:  # reported after every worker is joined
            errors.append(f"slot {index}: {error}")

    measured_started = time.monotonic()
    try:
        if mode == "concurrent":
            workers = [
                threading.Thread(target=request, args=(index,), daemon=True)
                for index in range(args.requests)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=args.request_timeout)
            alive = [worker for worker in workers if worker.is_alive()]
            if alive:
                with engine.pending_lock:
                    pending = sorted(engine.pending)
                engine.close()
                for worker in alive:
                    worker.join(timeout=5)
                raise TimeoutError(
                    f"{len(alive)} worker(s) exceeded {args.request_timeout}s; "
                    f"pending request IDs were {pending}"
                )
        else:
            for index in range(args.requests):
                request(index)
        elapsed_s = time.monotonic() - measured_started
        if errors:
            raise RuntimeError("; ".join(errors))
        completions = sum(
            int(result["stats"]["completion_tokens"])
            for result in results
            if result is not None
        )
        return {
            "mode": mode,
            "startup_s": startup_s,
            "warmup_s": warmup_s,
            "elapsed_s": elapsed_s,
            "aggregate_completion_tps": completions / elapsed_s,
            "requests": results,
            "profiles": list(engine.profile)[profile_base:],
            "tiers": engine.tiers,
            "hardware": engine.hwinfo,
            "resident": engine.resident,
        }
    finally:
        engine.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=ROOT / "qwen")
    parser.add_argument("--prompt", default="!")
    parser.add_argument(
        "--expect-exact",
        help="require every measured response to equal this text after stripping",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="bypass family chat rendering for kernel-only diagnostics",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument(
        "--order",
        default="sequential,concurrent",
        help="comma-separated modes; use both orders in separate runs to control cache bias",
    )
    parser.add_argument("--expert-ram", type=int, default=4)
    parser.add_argument(
        "--ram-gb",
        type=float,
        help="byte-budgeted host expert cache; overrides --expert-ram",
    )
    parser.add_argument("--ram-headroom-gb", type=float, default=1.0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda-events", action="store_true")
    parser.add_argument(
        "--expert-q3",
        action="store_true",
        help="select a complete grouped-int3 routed-expert sidecar",
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--cuda-expert-gb", type=float, default=7.0)
    parser.add_argument("--cuda-headroom-gb", type=float, default=1.0)
    parser.add_argument("--minimum-aggregate-tps", type=float)
    parser.add_argument("--minimum-speedup", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.requests <= 16:
        parser.error("--requests must be between 1 and 16")
    modes = [part.strip() for part in args.order.split(",") if part.strip()]
    if not modes or any(mode not in {"sequential", "concurrent"} for mode in modes):
        parser.error("--order accepts only sequential and concurrent")
    if args.expert_ram < 1:
        parser.error("--expert-ram must be positive")
    if args.ram_gb is not None and args.ram_gb <= 0:
        parser.error("--ram-gb must be positive")
    if args.ram_headroom_gb <= 0 or args.cuda_headroom_gb <= 0:
        parser.error("headroom values must be positive")
    if args.cuda_expert_gb < 0:
        parser.error("--cuda-expert-gb cannot be negative")
    if args.warmup_passes < 0:
        parser.error("--warmup-passes cannot be negative")
    if args.max_tokens < 1 or args.context < args.max_tokens or args.threads < 1:
        parser.error("context must cover max-tokens and threads must be positive")
    for name, value in (
        ("--minimum-aggregate-tps", args.minimum_aggregate_tps),
        ("--minimum-speedup", args.minimum_speedup),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name} must be positive and finite")

    model = args.model.resolve()
    model_family, args.runtime_prompt = prepare_user_prompt(
        model,
        args.prompt,
        raw=args.raw_prompt,
    )
    try:
        manifest = json.loads((model / "quantization.json").read_text())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"model quantization manifest is unavailable: {error}")
    if not isinstance(manifest, dict):
        raise SystemExit("model quantization manifest root is not an object")
    if manifest.get("complete") is not True:
        raise SystemExit("model quantization manifest is absent or incomplete")
    model_manifest = {
        name: manifest.get(name)
        for name in (
            "source",
            "source_fingerprint",
            "source_shards",
            "output_shards",
            "data_bytes",
            "tensor_count",
            "logical_tensor_count",
            "xbits",
            "io_bits",
            "shared_bits",
            "group_size",
            "include_mtp",
        )
    }
    try:
        expert_lowbit_manifest = (
            load_complete_expert_sidecar(model, bits=3)
            if args.expert_q3
            else None
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    runs = [run_mode(args, mode) for mode in modes]
    comparison = None
    sequential = next((run for run in runs if run["mode"] == "sequential"), None)
    concurrent = next((run for run in runs if run["mode"] == "concurrent"), None)
    if sequential and concurrent:
        comparison = {
            "elapsed_speedup": sequential["elapsed_s"] / concurrent["elapsed_s"],
            "aggregate_tps_speedup": (
                concurrent["aggregate_completion_tps"]
                / sequential["aggregate_completion_tps"]
            ),
            "outputs_match": [
                result["text"] for result in sequential["requests"]
            ]
            == [result["text"] for result in concurrent["requests"]],
        }
    failures = []
    if any(
        not result["text"]
        for run in runs
        for result in run["requests"]
        if result is not None
    ):
        failures.append("one or more measured outputs are empty")
    if args.expect_exact is not None:
        for run in runs:
            for result in run["requests"]:
                if (
                    result is not None
                    and result["text"].strip() != args.expect_exact
                ):
                    failures.append(
                        f"{run['mode']} response did not exactly match "
                        f"{args.expect_exact!r}"
                    )
    if comparison and not comparison["outputs_match"]:
        failures.append("sequential and concurrent outputs differ")
    if args.cuda:
        for run in runs:
            resident = run.get("resident") or {}
            if resident.get("device_moe", 0) <= 0:
                failures.append(f"{run['mode']} did not execute device MoE")
            if resident.get("host_moe", -1) != 0:
                failures.append(f"{run['mode']} used host MoE fallback")
    target_run = concurrent or runs[-1]
    if (
        args.minimum_aggregate_tps is not None
        and target_run["aggregate_completion_tps"] < args.minimum_aggregate_tps
    ):
        failures.append(
            f"aggregate TPS {target_run['aggregate_completion_tps']} below "
            f"{args.minimum_aggregate_tps}"
        )
    if (
        args.minimum_speedup is not None
        and (
            comparison is None
            or comparison["aggregate_tps_speedup"] < args.minimum_speedup
        )
    ):
        failures.append(
            "aggregate TPS speedup is unavailable or below "
            f"{args.minimum_speedup}"
        )
    payload = {
                "schema_version": 4,
                "model": str(model),
                "model_family": model_family,
                "model_manifest": model_manifest,
                "expert_lowbit_manifest": expert_lowbit_manifest,
                "prompt": {
                    "user_text": args.prompt,
                    "expected_exact_content": args.expect_exact,
                    "raw": args.raw_prompt,
                    "rendered_bytes": len(args.runtime_prompt.encode("utf-8")),
                },
                "cuda": args.cuda,
                "cuda_events": args.cuda_events,
                "tier_configuration": {
                    "warmup_passes": args.warmup_passes,
                    "context": args.context,
                    "threads": args.threads,
                    "expert_ram_per_layer": (
                        None if args.ram_gb is not None else args.expert_ram
                    ),
                    "ram_gb": args.ram_gb,
                    "ram_headroom_gb": args.ram_headroom_gb,
                    "cuda_expert_gb": args.cuda_expert_gb if args.cuda else None,
                    "cuda_headroom_gb": (
                        args.cuda_headroom_gb if args.cuda else None
                    ),
                    "tiered_io": args.ram_gb is not None,
                    "predictive_prefetch": False,
                    "uring_persist": False,
                    "pinned_upload": False,
                    "decode_protect": False,
                    "decode_protect_prewarm": False,
                    "expert_lowbit": "int3g128" if args.expert_q3 else None,
                },
                "runs": runs,
                "comparison": comparison,
                "acceptance": {
                    "minimum_aggregate_tps": args.minimum_aggregate_tps,
                    "minimum_speedup": args.minimum_speedup,
                    "passed": not failures,
                    "failures": failures,
                },
            }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
