#!/usr/bin/env python3
"""Run the fixed Phase-6 multi-domain MTP acceptance experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any


PROMPTS = [
    {
        "id": "code",
        "description": "TypeScript source-code generation",
        "text": (
            "Write a TypeScript function that performs a stable topological sort. "
            "Explain its time complexity and include two compact tests."
        ),
    },
    {
        "id": "technical",
        "description": "Technical systems prose",
        "text": (
            "Explain why memory bandwidth often limits single-token transformer "
            "inference, and distinguish bandwidth cost from kernel-launch latency."
        ),
    },
    {
        "id": "multilingual",
        "description": "Mixed-language explanation",
        "text": (
            "用中文解释缓存命中率，然后用日本語で一文に要約し、最後に give a "
            "short English example using an emoji 🚀."
        ),
    },
    {
        "id": "json",
        "description": "Strict structured output",
        "text": (
            'Return only valid JSON for three tasks. Use this schema: '
            '{"tasks":[{"id":1,"title":"string","risk":"low|medium|high"}]}.'
        ),
    },
    {
        "id": "conversation",
        "description": "Ordinary planning conversation",
        "text": (
            "I have a free Saturday and want a calm day with some exercise, reading, "
            "and cooking. Help me make a realistic plan without over-scheduling it."
        ),
    },
]
GATE_PROMPT = {
    "id": "hello_gate",
    "description": "Official Gate-6 single-word control prompt",
    "text": "Hello",
}

TOKEN_RE = re.compile(r"^tokens:\s*(.*)$", re.MULTILINE)
MTP_RE = re.compile(
    r"\[MTP\] accepted=(\d+)/(\d+) \(([0-9.]+)%\), "
    r"target_forwards=(\d+), emitted=(\d+), draft=(\d+) "
    r"timing=([0-9.]+)/([0-9.]+)/([0-9.]+)s draft/verify/replay"
)
CACHE_RE = re.compile(
    r"\[MTP_CACHE\] draft_misses=(\d+) verify_misses=(\d+) "
    r"replay_misses=(\d+) other_misses=(\d+) total_misses=(\d+)"
)
CONF_RE = re.compile(
    r"\[MTP_CONF\] margin_bins=<0\.25,<0\.5,<1,<2,>=2 "
    r"accepted=([0-9,]+) rejected=([0-9,]+) unverified=([0-9,]+) "
    r"mean=([0-9.]+)/([0-9.]+)/([0-9.]+)"
)
ADMIT_RE = re.compile(
    r"\[MTP_ADMIT\] min_margin=([0-9.]+) skipped=(\d+)"
)
BATCH_RE = re.compile(
    r"\[CUDA_BATCH\] transactions=(\d+) routes=(\d+) unique=(\d+) "
    r"overlap=([0-9.]+)% routes/transaction=([0-9.]+) "
    r"unique/transaction=([0-9.]+)"
)
PERF_RE = re.compile(
    r"\[PERF\] load=([0-9.]+)s prefill=([0-9.]+)s "
    r"\(([0-9.]+) tok/s\) decode=([0-9.]+)s \(([0-9.]+) tok/s\)"
)
DETAIL_RE = re.compile(
    r"\[PERF_DETAIL\] gdn=([0-9.]+)s attn=([0-9.]+)s "
    r"moe=([0-9.]+)s \(load=([0-9.]+)s misses=(\d+)\) "
    r"lm=([0-9.]+)s other=([-0-9.]+)s"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=root / "c" / "qwen")
    parser.add_argument("--snapshot", type=Path, default=root / "c" / "qwen35")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "c" / "bench" / "mtp_domains.json",
    )
    parser.add_argument("--ngen", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--expert-ram", type=int, default=64)
    parser.add_argument("--prefetch-threads", type=int, default=4)
    parser.add_argument("--cuda-expert-gb", type=int, default=8)
    parser.add_argument("--mtp-min-margin", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--depth",
        dest="depths",
        type=int,
        choices=(0, 1, 2, 3),
        action="append",
        help="run only selected depths (0 is the non-speculative baseline)",
    )
    parser.add_argument("--debug-logits", action="store_true")
    parser.add_argument("--cuda-spec-gdn", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--cuda-spec-gdn-batch", type=int, choices=(0, 1), default=1
    )
    parser.add_argument("--cuda-spec-full", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--cuda-spec-moe-batch", type=int, choices=(0, 1), default=1
    )
    parser.add_argument("--spec-batch", type=int, choices=(0, 1), default=1)
    parser.add_argument("--cuda-shared-fused", type=int, choices=(0, 1), default=1)
    parser.add_argument("--cuda-spec-moe-check", type=int, choices=(0, 1), default=0)
    parser.add_argument("--cuda-spec-moe-exact", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--prompt",
        action="append",
        choices=[item["id"] for item in PROMPTS],
        help="run only selected prompt IDs; repeat to select more than one",
    )
    parser.add_argument(
        "--gate-hello",
        action="store_true",
        help="run the official single-word Hello gate prompt instead",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run(
    stdout: str, stderr: str, depth: int, require_cuda_batch: bool
) -> dict[str, Any]:
    token_match = TOKEN_RE.search(stdout)
    perf_match = PERF_RE.search(stderr)
    detail_match = DETAIL_RE.search(stderr)
    if not token_match or not perf_match or not detail_match:
        raise RuntimeError(
            "missing token or performance telemetry\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )
    token_text = token_match.group(1).strip()
    tokens = [int(value) for value in token_text.split()] if token_text else []
    result: dict[str, Any] = {
        "tokens": tokens,
        "load_s": float(perf_match.group(1)),
        "prefill_s": float(perf_match.group(2)),
        "prefill_tok_s": float(perf_match.group(3)),
        "decode_s": float(perf_match.group(4)),
        "decode_tok_s": float(perf_match.group(5)),
        "profile": {
            "gdn_s": float(detail_match.group(1)),
            "attention_s": float(detail_match.group(2)),
            "moe_s": float(detail_match.group(3)),
            "expert_load_s": float(detail_match.group(4)),
            "expert_misses": int(detail_match.group(5)),
            "lm_head_s": float(detail_match.group(6)),
            "other_s": float(detail_match.group(7)),
        },
    }
    if depth:
        mtp_match = MTP_RE.search(stderr)
        batch_match = BATCH_RE.search(stderr)
        if not mtp_match or (require_cuda_batch and not batch_match):
            raise RuntimeError(
                "speculative run did not report MTP and CUDA batch telemetry\n"
                f"stderr:\n{stderr}"
            )
        result["mtp"] = {
            "accepted": int(mtp_match.group(1)),
            "proposed": int(mtp_match.group(2)),
            "acceptance_pct": float(mtp_match.group(3)),
            "target_forwards": int(mtp_match.group(4)),
            "emitted": int(mtp_match.group(5)),
            "depth": int(mtp_match.group(6)),
            "draft_s": float(mtp_match.group(7)),
            "verify_s": float(mtp_match.group(8)),
            "replay_s": float(mtp_match.group(9)),
        }
        cache_match = CACHE_RE.search(stderr)
        if cache_match:
            result["mtp_cache"] = {
                "draft_misses": int(cache_match.group(1)),
                "verify_misses": int(cache_match.group(2)),
                "replay_misses": int(cache_match.group(3)),
                "other_misses": int(cache_match.group(4)),
                "total_misses": int(cache_match.group(5)),
            }
        conf_match = CONF_RE.search(stderr)
        if conf_match:
            result["mtp_confidence"] = {
                "margin_bins": [0.25, 0.5, 1.0, 2.0],
                "accepted": [int(value) for value in conf_match.group(1).split(",")],
                "rejected": [int(value) for value in conf_match.group(2).split(",")],
                "unverified": [
                    int(value) for value in conf_match.group(3).split(",")
                ],
                "accepted_mean": float(conf_match.group(4)),
                "rejected_mean": float(conf_match.group(5)),
                "unverified_mean": float(conf_match.group(6)),
            }
        admit_match = ADMIT_RE.search(stderr)
        if admit_match:
            result["mtp_admission"] = {
                "min_margin": float(admit_match.group(1)),
                "skipped": int(admit_match.group(2)),
            }
        if batch_match:
            result["cuda_batch"] = {
                "transactions": int(batch_match.group(1)),
                "routes": int(batch_match.group(2)),
                "unique_experts": int(batch_match.group(3)),
                "overlap_pct": float(batch_match.group(4)),
                "routes_per_transaction": float(batch_match.group(5)),
                "unique_per_transaction": float(batch_match.group(6)),
            }
            if require_cuda_batch and result["cuda_batch"]["transactions"] <= 0:
                raise RuntimeError("speculative run silently skipped CUDA batch path")
    return result


def run_one(
    engine: Path,
    snapshot: Path,
    prompt: dict[str, str],
    depth: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "SNAP": str(snapshot),
            "PROMPT": prompt["text"],
            "CHAT": "1",
            "TEXT": "0",
            "NGEN": str(args.ngen),
            "WARMUP": str(args.warmup),
            "PROF": "1",
            "PROF_DETAIL": "1",
            "OMP_NUM_THREADS": str(args.threads),
            "EXPERT_RAM": str(args.expert_ram),
            "PREFETCH_THREADS": str(args.prefetch_threads),
            "COLI_CUDA": "1",
            "CUDA_DENSE": "1",
            "CUDA_EXPERTS": "1",
            "CUDA_F16": "1",
            "CUDA_EXPERT_GB": str(args.cuda_expert_gb),
            "CUDA_PRELOAD": "0",
            "CUDA_SHARED_FUSED": str(args.cuda_shared_fused),
            "CUDA_SPEC_MOE_CHECK": str(args.cuda_spec_moe_check),
            "CUDA_SPEC_MOE_EXACT": str(args.cuda_spec_moe_exact),
            "MTP": "1" if depth else "0",
        }
    )
    if depth:
        env.update(
            {
                "DRAFT": str(depth),
                "MTP_MIN_ACCEPT": "0",
                "MTP_MIN_MARGIN": str(args.mtp_min_margin),
                "SPEC_BATCH": str(args.spec_batch),
                "CUDA_SPEC_GDN": str(args.cuda_spec_gdn),
                "CUDA_SPEC_GDN_BATCH": str(args.cuda_spec_gdn_batch),
                "CUDA_SPEC_FULL": str(args.cuda_spec_full),
                "CUDA_SPEC_MOE_BATCH": str(args.cuda_spec_moe_batch),
            }
        )
    if args.debug_logits:
        env["DEBUG_LOGITS"] = "1"
    started = time.time()
    completed = subprocess.run(
        [str(engine)],
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{prompt['id']} D{depth} exited {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    parsed = parse_run(
        completed.stdout,
        completed.stderr,
        depth,
        bool(
            args.spec_batch
            and args.cuda_spec_full
            and args.cuda_spec_moe_batch
        ),
        # SPEC_BATCH=0 uses the single-token verifier and emits no batch
        # transactions even if the CUDA batch controls are enabled.
    )
    parsed["wall_s"] = time.time() - started
    parsed["stdout"] = completed.stdout
    parsed["stderr"] = completed.stderr
    return parsed


def main() -> None:
    args = parse_args()
    engine = args.engine.resolve()
    snapshot = args.snapshot.resolve()
    if not engine.is_file():
        raise SystemExit(f"engine not found: {engine}")
    if not (snapshot / "config.json").is_file():
        raise SystemExit(f"snapshot not found: {snapshot}")
    prompts = (
        [GATE_PROMPT]
        if args.gate_hello
        else [
            item
            for item in PROMPTS
            if not args.prompt or item["id"] in args.prompt
        ]
    )
    # Rotate configuration order by prompt to reduce monotonic thermal/order bias.
    configurations = args.depths or [0, 1, 2, 3]
    runs: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        order = configurations[prompt_index % len(configurations) :] + configurations[
            : prompt_index % len(configurations)
        ]
        for depth in order:
            print(f"[bench] {prompt['id']} D{depth}", flush=True)
            run = run_one(engine, snapshot, prompt, depth, args)
            runs.append(
                {
                    "prompt_id": prompt["id"],
                    "depth": depth,
                    "result": run,
                }
            )
            print(
                f"[bench] {prompt['id']} D{depth}: "
                f"{run['decode_tok_s']:.2f} tok/s",
                flush=True,
            )

    by_prompt: dict[str, dict[int, dict[str, Any]]] = {}
    for record in runs:
        by_prompt.setdefault(record["prompt_id"], {})[record["depth"]] = record[
            "result"
        ]
    identities: dict[str, dict[str, bool]] = {}
    if 0 in configurations:
        for prompt in prompts:
            prompt_runs = by_prompt[prompt["id"]]
            baseline = prompt_runs[0]["tokens"]
            identities[prompt["id"]] = {
                f"D{depth}": prompt_runs[depth]["tokens"] == baseline
                for depth in configurations
                if depth
            }
    failed_identity = [
        f"{prompt_id}/{depth}"
        for prompt_id, checks in identities.items()
        for depth, passed in checks.items()
        if not passed
    ]

    payload = {
        "schema_version": 1,
        "experiment": "phase6_multi_domain_mtp",
        "created_unix": time.time(),
        "host": platform.node(),
        "platform": platform.platform(),
        "engine": str(engine),
        "engine_sha256": sha256_file(engine),
        "snapshot": str(snapshot),
        "controls": {
            "ngen": args.ngen,
            "warmup": args.warmup,
            "threads": args.threads,
            "expert_ram": args.expert_ram,
            "prefetch_threads": args.prefetch_threads,
            "cuda_expert_gb": args.cuda_expert_gb,
            "mtp_min_accept": 0,
            "mtp_min_margin": args.mtp_min_margin,
            "depths": configurations,
            "configuration_order": "rotated by prompt",
            "debug_logits": args.debug_logits,
            "cuda_spec_gdn": args.cuda_spec_gdn,
            "cuda_spec_gdn_batch": args.cuda_spec_gdn_batch,
            "cuda_spec_full": args.cuda_spec_full,
            "cuda_spec_moe_batch": args.cuda_spec_moe_batch,
            "spec_batch": args.spec_batch,
            "cuda_shared_fused": args.cuda_shared_fused,
            "cuda_spec_moe_check": args.cuda_spec_moe_check,
            "cuda_spec_moe_exact": args.cuda_spec_moe_exact,
        },
        "prompts": prompts,
        "runs": runs,
        "token_identity": identities,
        "all_token_identical": not failed_identity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if failed_identity:
        raise SystemExit(
            "speculative token identity failed: " + ", ".join(failed_identity)
        )
    print(f"[bench] wrote {args.output}")
    if 0 in configurations:
        print("[bench] all speculative token streams match their D0 baselines")
    else:
        print("[bench] token identity not evaluated because D0 was not selected")


if __name__ == "__main__":
    main()
