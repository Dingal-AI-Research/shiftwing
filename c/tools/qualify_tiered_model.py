#!/usr/bin/env python3
"""Run a preregistered warm tiered-model chat throughput qualification."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openai_server import Engine, snapshot_model_family  # noqa: E402
from tools.expert_lowbit import load_complete_expert_sidecar  # noqa: E402
from tools.model_prompt import prepare_user_prompt  # noqa: E402
from tools.runtime_env import isolated_engine_env  # noqa: E402


DEFAULT_PROMPTS = [
    "Explain, for an undergraduate programmer, why a hash table can have "
    "constant average lookup time but linear worst-case lookup time.",
    "Write a TypeScript function that groups an array of records by a string "
    "key. Explain its time and space complexity.",
    "A service becomes slow only after several hours. Give a concise, ordered "
    "debugging plan that distinguishes memory growth, lock contention, and I/O.",
    "Summarize the trade-off between caching more data in RAM and reading it "
    "from NVMe on demand. Include one failure mode of each approach.",
]


def _load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError("prompt file must be a non-empty JSON array of strings")
    return value


def sustained_tps(runs: list[dict]) -> float:
    """Token-weighted decode rate reconstructed from per-turn engine rates."""
    tokens = 0
    seconds = 0.0
    for run in runs:
        count = int(run["stats"]["completion_tokens"])
        rate = float(run["stats"]["tokens_per_second"])
        if count <= 0 or not math.isfinite(rate) or rate <= 0:
            raise ValueError("every measured run must have positive tokens and rate")
        tokens += count
        seconds += count / rate
    if not tokens or seconds <= 0:
        raise ValueError("at least one measured run is required")
    return tokens / seconds


def validate_expert_telemetry(
    emap: dict,
    hits_hex: str,
    tiers: dict,
    *,
    expected_rows: int,
    expected_cols: int,
) -> dict:
    """Decode EMAP/HITS and prove their dimensions and tier totals agree."""
    if not isinstance(emap, dict) or not isinstance(tiers, dict):
        raise ValueError("EMAP and TIERS objects are required")
    rows = emap.get("rows")
    cols = emap.get("cols")
    if rows != expected_rows or cols != expected_cols:
        raise ValueError(
            f"EMAP dimensions {rows}x{cols} != expected "
            f"{expected_rows}x{expected_cols}"
        )
    try:
        map_bytes = bytes.fromhex(emap.get("map", ""))
        hit_bytes = bytes.fromhex(hits_hex)
    except (TypeError, ValueError) as error:
        raise ValueError("EMAP/HITS are not valid hexadecimal") from error
    experts = rows * cols
    if len(map_bytes) != experts:
        raise ValueError(f"EMAP has {len(map_bytes)} bytes, expected {experts}")
    expected_hit_bytes = (experts + 7) // 8
    if len(hit_bytes) != expected_hit_bytes:
        raise ValueError(
            f"HITS has {len(hit_bytes)} bytes, expected {expected_hit_bytes}"
        )

    observed = {"disk": 0, "ram": 0, "vram": 0}
    heat = []
    for value in map_bytes:
        tier = value >> 6
        if tier > 2:
            raise ValueError("EMAP contains reserved tier value 3")
        observed[("disk", "ram", "vram")[tier]] += 1
        heat.append(value & 0x3F)
    reported = {
        name: int(tiers.get(name, -1)) for name in ("disk", "ram", "vram")
    }
    if observed != reported:
        raise ValueError(f"EMAP tier totals {observed} != TIERS {reported}")

    hit_count = sum(byte.bit_count() for byte in hit_bytes)
    padding_bits = expected_hit_bytes * 8 - experts
    if padding_bits and hit_bytes[-1] >> (8 - padding_bits):
        raise ValueError("HITS sets padding bits outside the expert map")
    return {
        "rows": rows,
        "cols": cols,
        "experts": experts,
        "tier_counts": observed,
        "nonzero_heat": sum(value > 0 for value in heat),
        "maximum_heat_bucket": max(heat, default=0),
        "turn_hit_experts": hit_count,
    }


def render_qualification_prompt(
    model: Path | None,
    family: str,
    prompt: str,
) -> str:
    """Render one frozen prompt through the selected model family's template."""
    return prepare_user_prompt(
        model,
        prompt,
        family=family,
    )[1]


def _turn(
    engine: Engine,
    prompt: str,
    max_tokens: int,
    index: int,
    *,
    model: Path | None = None,
    family: str = "qwen3.5",
) -> dict:
    rendered = render_qualification_prompt(model, family, prompt)
    pieces: list[str] = []
    first_content: list[float | None] = [None]
    started = time.monotonic()

    def emit(piece: str) -> None:
        if first_content[0] is None:
            first_content[0] = time.monotonic()
        pieces.append(piece)

    stats = engine.generate(
        rendered,
        max_tokens,
        0.0,
        1.0,
        emit,
        cache_slot=0,
    )
    finished = time.monotonic()
    return {
        "prompt_index": index,
        "prompt": prompt,
        "text": "".join(pieces),
        "ttft_s": (
            first_content[0] - started if first_content[0] is not None else None
        ),
        "wall_s": finished - started,
        "stats": stats,
    }


def _run_pass(
    engine: Engine,
    prompts: list[str],
    max_tokens: int,
    *,
    phase: str,
    pass_index: int,
    pass_count: int,
    model: Path | None = None,
    family: str = "qwen3.5",
) -> list[dict]:
    """Run one pass with durable human-readable progress on stderr."""
    turns = []
    for prompt_index, prompt in enumerate(prompts):
        print(
            f"[QUALIFY] {phase} pass {pass_index + 1}/{pass_count}, "
            f"prompt {prompt_index + 1}/{len(prompts)}: starting",
            file=sys.stderr,
            flush=True,
        )
        turn = _turn(
            engine,
            prompt,
            max_tokens,
            prompt_index,
            model=model,
            family=family,
        )
        turn["pass"] = pass_index
        turns.append(turn)
        stats = turn["stats"]
        ttft = turn["ttft_s"]
        print(
            f"[QUALIFY] {phase} pass {pass_index + 1}/{pass_count}, "
            f"prompt {prompt_index + 1}/{len(prompts)}: "
            f"done in {turn['wall_s']:.1f}s, "
            f"TTFT={ttft:.1f}s, "
            f"decode={float(stats['tokens_per_second']):.3f} tok/s",
            file=sys.stderr,
            flush=True,
        )
    return turns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=ROOT / "qwen")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument("--measured-passes", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--expert-ram-gb", type=float, default=18.0)
    parser.add_argument("--cuda-expert-gb", type=float, default=6.0)
    parser.add_argument("--ram-headroom-gb", type=float, default=1.0)
    parser.add_argument("--cuda-headroom-gb", type=float, default=1.0)
    parser.add_argument("--minimum-tps", type=float, default=2.0)
    parser.add_argument(
        "--uring-persist",
        type=int,
        choices=(0, 1),
        default=0,
        help="reuse one io_uring and aligned direct-I/O buffers per thread",
    )
    parser.add_argument(
        "--pinned-upload",
        type=int,
        choices=(0, 1),
        default=0,
        help="stage routed-expert H2D transfers through a reusable pinned arena",
    )
    parser.add_argument(
        "--decode-protect",
        type=int,
        choices=(0, 1),
        default=0,
        help="protect learned decode experts from subsequent prefill eviction",
    )
    parser.add_argument(
        "--decode-protect-prewarm",
        type=int,
        choices=(0, 1),
        default=0,
        help="materialize selected decode experts in RAM between requests",
    )
    parser.add_argument(
        "--expert-q2",
        type=int,
        choices=(0, 1),
        default=0,
        help="select complete grouped-int2 routed-expert sidecars",
    )
    parser.add_argument(
        "--expert-q3",
        type=int,
        choices=(0, 1),
        default=0,
        help="select complete grouped-int3 routed-expert sidecars",
    )
    parser.add_argument(
        "--q3-route-atlas",
        type=int,
        choices=(0, 1),
        default=0,
        help="adaptively partition learned hot q3 routes across RAM and VRAM",
    )
    parser.add_argument(
        "--q3-native",
        type=int,
        choices=(0, 1),
        default=0,
        help="keep q3 routed experts packed on CUDA instead of expanding to q4",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cuda-events", action="store_true")
    args = parser.parse_args()

    if args.warmup_passes < 0 or args.measured_passes < 1:
        parser.error("warmup passes must be non-negative and measured passes positive")
    if args.max_tokens < 1 or args.context < args.max_tokens or args.threads < 1:
        parser.error("context must cover max-tokens and threads must be positive")
    if (
        args.expert_ram_gb <= 0
        or args.cuda_expert_gb < 0
        or args.ram_headroom_gb <= 0
        or args.cuda_headroom_gb <= 0
    ):
        parser.error("RAM/headroom must be positive and CUDA cache non-negative")
    if not math.isfinite(args.minimum_tps) or args.minimum_tps <= 0:
        parser.error("minimum-tps must be positive and finite")

    model = args.model.resolve()
    engine_path = args.engine.resolve()
    model_family = snapshot_model_family(model)
    prompts = _load_prompts(args.prompt_file)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    expected_rows = int(text_config["num_hidden_layers"])
    expected_cols = int(text_config["num_experts"])
    try:
        manifest = json.loads(
            (model / "quantization.json").read_text(encoding="utf-8")
        )
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
    if args.expert_q2 and args.expert_q3:
        parser.error("--expert-q2 and --expert-q3 are mutually exclusive")
    if args.q3_route_atlas and not args.expert_q3:
        parser.error("--q3-route-atlas requires --expert-q3 1")
    if args.q3_route_atlas and not args.decode_protect:
        parser.error("--q3-route-atlas requires --decode-protect 1")
    if args.q3_route_atlas and not args.q3_native:
        parser.error("--q3-route-atlas requires --q3-native 1")
    if args.q3_native and not args.expert_q3:
        parser.error("--q3-native requires --expert-q3 1")
    selected_bits = 2 if args.expert_q2 else 3 if args.expert_q3 else None
    expert_lowbit_manifest = None
    if selected_bits is not None:
        try:
            expert_lowbit_manifest = load_complete_expert_sidecar(
                model, bits=selected_bits
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
    env = isolated_engine_env()
    env.pop("EXPERT_RAM", None)
    env.update(
        {
            "SERVE_RESIDENT": "1",
            "CTX": str(args.context),
            "OMP_NUM_THREADS": str(args.threads),
            "RAM_GB": str(args.expert_ram_gb),
            "RAM_HEADROOM_GB": str(args.ram_headroom_gb),
            "COLI_CUDA": "1",
            "CUDA_DENSE": "1",
            "CUDA_F16": "1",
            "CUDA_EXPERTS": "1",
            "CUDA_EXPERT_GB": str(args.cuda_expert_gb),
            "CUDA_HEADROOM_GB": str(args.cuda_headroom_gb),
            "PIPE": "1",
            "URING": "1",
            "DIRECT": "1",
            "URING_PERSIST": str(args.uring_persist),
            "CUDA_PINNED_UPLOAD": str(args.pinned_upload),
            "DECODE_PROTECT": str(args.decode_protect),
            "DECODE_PROTECT_PREWARM": str(args.decode_protect_prewarm),
            "EXPERT_Q2": str(args.expert_q2),
            "EXPERT_Q3": str(args.expert_q3),
            "Q3_ROUTE_ATLAS": str(args.q3_route_atlas),
            "Q3_NATIVE": str(args.q3_native),
            "AUTOPIN": "1",
            "PREFETCH_LOAD": "0",
            "PREFETCH_THREADS": "0",
        }
    )
    if args.cuda_events:
        env["CUDA_PROFILE_STAGES"] = "1"
    else:
        env.pop("CUDA_PROFILE_STAGES", None)

    started = time.monotonic()
    runtime = Engine(
        engine_path,
        model,
        max_tokens=args.max_tokens,
        env=env,
        kv_slots=1,
    )
    startup_s = time.monotonic() - started
    warmups: list[dict] = []
    measured: list[dict] = []
    try:
        for pass_index in range(args.warmup_passes):
            warmups.extend(
                _run_pass(
                    runtime,
                    prompts,
                    args.max_tokens,
                    phase="warmup",
                    pass_index=pass_index,
                    pass_count=args.warmup_passes,
                    model=model,
                    family=model_family,
                )
            )
        for pass_index in range(args.measured_passes):
            measured.extend(
                _run_pass(
                    runtime,
                    prompts,
                    args.max_tokens,
                    phase="measured",
                    pass_index=pass_index,
                    pass_count=args.measured_passes,
                    model=model,
                    family=model_family,
                )
            )

        rate = sustained_tps(measured)
        per_turn_rates = [
            float(turn["stats"]["tokens_per_second"]) for turn in measured
        ]
        outputs_nonempty = all(turn["text"].strip() for turn in measured)
        telemetry_summary = None
        telemetry_error = None
        try:
            if runtime.hwinfo is None:
                raise ValueError("HWINFO is missing")
            telemetry_summary = validate_expert_telemetry(
                runtime.emap,
                runtime.hits,
                runtime.tiers,
                expected_rows=expected_rows,
                expected_cols=expected_cols,
            )
        except (TypeError, ValueError) as error:
            telemetry_error = str(error)
        telemetry_complete = telemetry_summary is not None
        cuda_active = bool(
            runtime.hwinfo
            and runtime.hwinfo.get("gpus", 0) >= 1
            and runtime.hwinfo.get("vram_total_gb", 0) > 0
            and runtime.tiers
            and runtime.tiers.get("vram", 0) > 0
        )
        resident_graph = bool(
            runtime.resident
            and runtime.resident.get("layers", 0) > 0
            and runtime.resident.get("device_moe", 0) > 0
            and runtime.resident.get("host_moe", -1) == 0
            and runtime.resident.get("activation_h2d_bytes", 0) > 0
            and runtime.resident.get("activation_d2h_bytes", 0) > 0
            and runtime.resident.get("logits_d2h_bytes", 0) > 0
            and runtime.resident.get("router_d2h_bytes", 0) > 0
        )
        q3_native = getattr(runtime, "q3_native", None)
        q3_atlas = getattr(runtime, "q3_atlas", None)
        native_q3_active = bool(
            not args.expert_q3
            or not args.q3_native
            or (
                q3_native
                and q3_native.get("uploads", 0) > 0
                and q3_native.get("upload_bytes", 0) > 0
                and q3_native.get("grouped_calls", 0) > 0
            )
        )
        q3_atlas_active = bool(
            not args.q3_route_atlas
            or (
                q3_atlas
                and q3_atlas.get("active") is True
                and q3_atlas.get("refreshes", 0) > 0
                and q3_atlas.get("device_entries", 0) > 0
            )
        )
        failures = []
        if not outputs_nonempty:
            failures.append("one or more measured outputs are empty")
        if not telemetry_complete:
            failures.append(
                "expert telemetry is incomplete"
                + (f": {telemetry_error}" if telemetry_error else "")
            )
        if not cuda_active:
            failures.append("CUDA or the VRAM expert tier is inactive")
        if not resident_graph:
            failures.append(
                "resident CUDA graph is inactive or used host-MoE fallback"
            )
        if not native_q3_active:
            failures.append("native packed-q3 CUDA execution is inactive")
        if not q3_atlas_active:
            failures.append("adaptive q3 route atlas is inactive")
        if rate < args.minimum_tps:
            failures.append(
                f"sustained throughput {rate:.6f} tok/s is below "
                f"{args.minimum_tps:.6f} tok/s"
            )
        gate_pass = not failures
        result = {
            "schema_version": 2,
            "model": str(model),
            "model_family": model_family,
            "engine": str(engine_path),
            "model_manifest": model_manifest,
            "expert_lowbit_manifest": expert_lowbit_manifest,
            "configuration": {
                "warmup_passes": args.warmup_passes,
                "measured_passes": args.measured_passes,
                "max_tokens": args.max_tokens,
                "context": args.context,
                "threads": args.threads,
                "expert_ram_gb": args.expert_ram_gb,
                "cuda_expert_gb": args.cuda_expert_gb,
                "ram_headroom_gb": args.ram_headroom_gb,
                "cuda_headroom_gb": args.cuda_headroom_gb,
                "minimum_tps": args.minimum_tps,
                "uring_persist": bool(args.uring_persist),
                "pinned_upload": bool(args.pinned_upload),
                "decode_protect": bool(args.decode_protect),
                "decode_protect_prewarm": bool(args.decode_protect_prewarm),
                "expert_q2": bool(args.expert_q2),
                "expert_q3": bool(args.expert_q3),
                "q3_route_atlas": bool(args.q3_route_atlas),
                "q3_native": bool(args.q3_native),
                "predictive_prefetch": False,
            },
            "startup_s": startup_s,
            "warmups": warmups,
            "measured": measured,
            "summary": {
                "sustained_tps": rate,
                "median_turn_tps": statistics.median(per_turn_rates),
                "minimum_turn_tps": min(per_turn_rates),
                "outputs_nonempty": outputs_nonempty,
                "telemetry_complete": telemetry_complete,
                "cuda_active": cuda_active,
                "resident_cuda_graph": resident_graph,
                "native_q3_active": native_q3_active,
                "q3_route_atlas_active": q3_atlas_active,
                "throughput_pass": rate >= args.minimum_tps,
                "automatic_gate_pass": gate_pass,
                "coherence_review_required": True,
            },
            "hardware": runtime.hwinfo,
            "tiers": runtime.tiers,
            "q3_native": q3_native,
            "q3_route_atlas": q3_atlas,
            "emap": runtime.emap,
            "hits": runtime.hits,
            "telemetry_summary": telemetry_summary,
            "telemetry_error": telemetry_error,
            "profiles": list(runtime.profile),
            "resident": runtime.resident,
            "acceptance": {
                "minimum_sustained_tps": args.minimum_tps,
                "passed": gate_pass,
                "failures": failures,
            },
        }
    finally:
        runtime.close()

    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["summary"]["automatic_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
