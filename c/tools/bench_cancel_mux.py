#!/usr/bin/env python3
"""Measure cancel-to-ack latency while a peer mux slot keeps decoding."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openai_server import ClientCancelled, Engine  # noqa: E402
from tools.expert_lowbit import load_complete_expert_sidecar  # noqa: E402
from tools.model_prompt import prepare_user_prompt  # noqa: E402
from tools.runtime_env import isolated_engine_env  # noqa: E402


def nearest_rank_percentile(values: list[float], percentile: float) -> float:
    """Return a deterministic nearest-rank percentile for non-negative samples."""
    if not values or not 0.0 < percentile <= 1.0:
        raise ValueError("values must be non-empty and percentile in (0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("percentile samples must be finite and non-negative")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def run_trial(
    engine: Engine,
    prompt: str,
    max_tokens: int,
    request_timeout: float,
    trial_index: int,
) -> dict:
    """Cancel one slot, let its peer finish, then prove immediate slot reuse."""
    barrier = threading.Barrier(2)
    result: dict = {"trial": trial_index}
    errors: list[str] = []
    profile_base = len(engine.profile)

    def cancelled_request() -> None:
        pieces: list[str] = []
        first_content = [None]

        def emit(piece: str) -> None:
            if first_content[0] is None:
                first_content[0] = time.monotonic()
            pieces.append(piece)

        try:
            barrier.wait()
            started = time.monotonic()
            engine.generate(
                prompt,
                max_tokens,
                0.0,
                1.0,
                emit,
                cache_slot=0,
                cancelled=lambda: bool(pieces),
            )
            errors.append("slot 0 completed instead of acknowledging cancellation")
        except ClientCancelled:
            acknowledged = time.monotonic()
            result["cancelled"] = {
                "first_content_s": (
                    first_content[0] - started if first_content[0] else None
                ),
                "cancel_ack_s": (
                    acknowledged - first_content[0] if first_content[0] else None
                ),
                "pieces_before_cancel": len(pieces),
                "text_before_cancel": "".join(pieces),
            }
        except Exception as error:
            errors.append(f"slot 0: {error}")

    def peer_request() -> None:
        pieces: list[str] = []
        first_content = [None]

        def emit(piece: str) -> None:
            if first_content[0] is None:
                first_content[0] = time.monotonic()
            pieces.append(piece)

        try:
            barrier.wait()
            started = time.monotonic()
            stats = engine.generate(
                prompt,
                max_tokens,
                0.0,
                1.0,
                emit,
                cache_slot=1,
            )
            result["peer"] = {
                "ttft_s": first_content[0] - started if first_content[0] else None,
                "request_s": time.monotonic() - started,
                "text": "".join(pieces),
                "stats": stats,
            }
        except Exception as error:
            errors.append(f"slot 1: {error}")

    workers = [
        threading.Thread(target=cancelled_request, daemon=True),
        threading.Thread(target=peer_request, daemon=True),
    ]
    measured_started = time.monotonic()
    for worker in workers:
        worker.start()
    deadline = measured_started + request_timeout
    for worker in workers:
        worker.join(max(0.0, deadline - time.monotonic()))
    alive = [worker for worker in workers if worker.is_alive()]
    if alive:
        with engine.pending_lock:
            pending = sorted(engine.pending)
        engine.close()
        for worker in alive:
            worker.join(timeout=5)
        raise TimeoutError(
            f"{len(alive)} worker(s) timed out; pending IDs were {pending}"
        )
    if errors:
        raise RuntimeError("; ".join(errors))

    reuse_pieces: list[str] = []
    reuse_started = time.monotonic()
    reuse_stats = engine.generate(
        prompt,
        1,
        0.0,
        1.0,
        reuse_pieces.append,
        cache_slot=0,
    )
    result["slot_reuse"] = {
        "request_s": time.monotonic() - reuse_started,
        "text": "".join(reuse_pieces),
        "stats": reuse_stats,
    }
    result["measured_s"] = time.monotonic() - measured_started
    result["profiles"] = list(engine.profile)[profile_base:]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=ROOT / "qwen")
    parser.add_argument("--prompt", default="!")
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="bypass family chat rendering for kernel-only diagnostics",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="repeated cancellation trials in the same warm engine",
    )
    parser.add_argument("--expert-ram", type=int, default=4)
    parser.add_argument(
        "--ram-gb",
        type=float,
        help="byte-budgeted host expert cache; overrides --expert-ram",
    )
    parser.add_argument("--ram-headroom-gb", type=float, default=1.0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--expert-q3",
        action="store_true",
        help="select a complete grouped-int3 routed-expert sidecar",
    )
    parser.add_argument("--cuda-expert-gb", type=float, default=7.0)
    parser.add_argument("--cuda-headroom-gb", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--max-cancel-ack-s", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
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
    if not 1 <= args.trials <= 100:
        parser.error("--trials must be between 1 and 100")
    if args.max_cancel_ack_s is not None and (
        not math.isfinite(args.max_cancel_ack_s) or args.max_cancel_ack_s <= 0
    ):
        parser.error("--max-cancel-ack-s must be positive and finite")

    model = args.model.resolve()
    model_family, runtime_prompt = prepare_user_prompt(
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
    env.pop("CUDA_PROFILE_STAGES", None)
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
    if args.trace:
        env["COLI_ENGINE_TRACE"] = "1"

    startup_started = time.monotonic()
    engine = Engine(
        args.engine.resolve(),
        args.model.resolve(),
        max_tokens=args.max_tokens,
        env=env,
        kv_slots=2,
    )
    startup_s = time.monotonic() - startup_started
    warmup_started = time.monotonic()
    for _ in range(args.warmup_passes):
        for slot in range(2):
            engine.generate(
                runtime_prompt,
                args.max_tokens,
                0.0,
                1.0,
                lambda _piece: None,
                cache_slot=slot,
            )
    warmup_s = time.monotonic() - warmup_started
    try:
        trials = []
        for trial_index in range(args.trials):
            print(
                f"[CANCEL] trial {trial_index + 1}/{args.trials}",
                file=sys.stderr,
                flush=True,
            )
            trials.append(
                run_trial(
                    engine,
                    runtime_prompt,
                    args.max_tokens,
                    args.request_timeout,
                    trial_index,
                )
            )
        failures: list[str] = []
        cancel_samples: list[float] = []
        for trial in trials:
            prefix = f"trial {trial['trial'] + 1}"
            if trial["cancelled"]["pieces_before_cancel"] != 1:
                failures.append(
                    f"{prefix}: cancelled request emitted other than one piece"
                )
            if not trial["peer"]["text"]:
                failures.append(f"{prefix}: peer request produced empty output")
            if trial["peer"]["stats"]["completion_tokens"] <= 0:
                failures.append(f"{prefix}: peer request produced no tokens")
            if not trial["slot_reuse"]["text"]:
                failures.append(f"{prefix}: released slot did not produce output")
            sample = trial["cancelled"]["cancel_ack_s"]
            if sample is None:
                failures.append(f"{prefix}: cancel acknowledgement was not timed")
            else:
                cancel_samples.append(float(sample))
        p95_cancel_ack_s = (
            nearest_rank_percentile(cancel_samples, 0.95)
            if len(cancel_samples) == len(trials)
            else None
        )
        if args.max_cancel_ack_s is not None and (
            p95_cancel_ack_s is None
            or p95_cancel_ack_s > args.max_cancel_ack_s
        ):
            failures.append(
                f"cancel acknowledgement p95 exceeds {args.max_cancel_ack_s}s"
            )
        if args.cuda:
            resident = engine.resident or {}
            if resident.get("device_moe", 0) <= 0:
                failures.append("device MoE did not execute")
            if resident.get("host_moe", -1) != 0:
                failures.append("host MoE fallback occurred")
        payload = {
            "schema_version": 5,
            "model": str(model),
            "model_family": model_family,
            "model_manifest": model_manifest,
            "expert_lowbit_manifest": expert_lowbit_manifest,
            "prompt": {
                "user_text": args.prompt,
                "raw": args.raw_prompt,
                "rendered_bytes": len(runtime_prompt.encode("utf-8")),
            },
            "cuda": args.cuda,
            "tier_configuration": {
                "warmup_passes": args.warmup_passes,
                "trials": args.trials,
                "context": args.context,
                "threads": args.threads,
                "expert_ram_per_layer": (
                    None if args.ram_gb is not None else args.expert_ram
                ),
                "ram_gb": args.ram_gb,
                "ram_headroom_gb": args.ram_headroom_gb,
                "cuda_expert_gb": (
                    args.cuda_expert_gb if args.cuda else None
                ),
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
            "startup_s": startup_s,
            "warmup_s": warmup_s,
            "trials": trials,
            "summary": {
                "cancel_ack_samples_s": cancel_samples,
                "p95_method": "nearest-rank",
                "p95_cancel_ack_s": p95_cancel_ack_s,
                "peer_outputs_nonempty": all(
                    bool(trial["peer"]["text"]) for trial in trials
                ),
                "slot_reuse_outputs_nonempty": all(
                    bool(trial["slot_reuse"]["text"]) for trial in trials
                ),
            },
            "tiers": engine.tiers,
            "hardware": engine.hwinfo,
            "resident": engine.resident,
            "acceptance": {
                "maximum_cancel_ack_s": args.max_cancel_ack_s,
                "measured_statistic": "p95_cancel_ack_s",
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
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
