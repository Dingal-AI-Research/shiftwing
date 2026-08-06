#!/usr/bin/env python3
"""Tokenize a fixed corpus and run the C engine's teacher-forced perplexity path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from expert_lowbit import load_complete_expert_sidecar  # noqa: E402
from runtime_env import isolated_engine_env  # noqa: E402


PPL_RE = re.compile(r"\[PPL\] tokens=(\d+) nll=([0-9.eE+-]+) ppl=([0-9.eE+-]+) tok_s=([0-9.eE+-]+)")


def binary_version(command: str) -> str:
    run = subprocess.run(
        [command, "--version"],
        text=True,
        capture_output=True,
        check=True,
    )
    return (run.stdout + "\n" + run.stderr).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path(__file__).resolve().parents[1] / "qwen")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--llama-model", type=Path)
    parser.add_argument("--llama-perplexity", default="llama-perplexity")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--expert-ram", type=int, default=64)
    parser.add_argument(
        "--ram-gb",
        type=float,
        help="byte-budgeted host expert cache; overrides --expert-ram",
    )
    parser.add_argument("--ram-headroom-gb", type=float, default=1.0)
    parser.add_argument("--prefetch-threads", type=int, default=4)
    parser.add_argument("--max-ppl", type=float)
    parser.add_argument("--max-relative-ppl-delta", type=float)
    parser.add_argument(
        "--require-complete-manifest",
        action="store_true",
        help="require quantization.json with complete=true and retain its identity",
    )
    parser.add_argument(
        "--copy-experts",
        action="store_true",
        help="copy bounded experts into cache slots instead of using evaluation-optimized mmap views",
    )
    parser.add_argument(
        "--expert-q2",
        action="store_true",
        help="select complete expert-q2.json routed-expert sidecars",
    )
    parser.add_argument(
        "--expert-q3",
        action="store_true",
        help="select complete expert-q3.json routed-expert sidecars",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expert_ram < 1:
        raise SystemExit("--expert-ram must be positive")
    if (
        args.ram_gb is not None
        and (not math.isfinite(args.ram_gb) or args.ram_gb <= 0)
    ):
        raise SystemExit("--ram-gb must be positive")
    if not math.isfinite(args.ram_headroom_gb) or args.ram_headroom_gb <= 0:
        raise SystemExit("--ram-headroom-gb must be positive")
    if (
        args.max_ppl is not None
        and (not math.isfinite(args.max_ppl) or args.max_ppl <= 0)
    ):
        raise SystemExit("--max-ppl must be positive")
    if (
        args.max_relative_ppl_delta is not None
        and (
            not math.isfinite(args.max_relative_ppl_delta)
            or args.max_relative_ppl_delta < 0
        )
    ):
        raise SystemExit("--max-relative-ppl-delta cannot be negative")
    if args.max_relative_ppl_delta is not None and not args.llama_model:
        raise SystemExit("--max-relative-ppl-delta requires --llama-model")
    snapshot, corpus, engine = args.snapshot.resolve(), args.corpus.resolve(), args.engine.resolve()
    manifest_path = snapshot / "quantization.json"
    model_manifest = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"model quantization manifest is invalid: {error}")
        if not isinstance(manifest, dict):
            raise SystemExit("model quantization manifest root is not an object")
        if manifest.get("complete") is not True:
            raise SystemExit("model quantization manifest is incomplete")
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
    elif args.require_complete_manifest:
        raise SystemExit("complete model quantization manifest is required")
    if args.expert_q2 and args.expert_q3:
        raise SystemExit("--expert-q2 and --expert-q3 are mutually exclusive")
    selected_bits = 2 if args.expert_q2 else 3 if args.expert_q3 else None
    expert_lowbit_manifest = None
    if selected_bits is not None:
        try:
            expert_lowbit_manifest = load_complete_expert_sidecar(
                snapshot, bits=selected_bits
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
    corpus_bytes = corpus.read_bytes()
    # Decode bytes directly so CRLF is preserved exactly as llama.cpp reads it.
    text = corpus_bytes.decode("utf-8", errors="replace")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    ids = tokenizer.encode(text, add_special_tokens=False)[: args.max_tokens]
    if args.ctx_size < 2 or len(ids) < 2 * args.ctx_size:
        raise SystemExit("corpus must produce at least two complete evaluation contexts")
    with tempfile.NamedTemporaryFile("w", prefix="qwen-ppl-", suffix=".ids", delete=False) as handle:
        handle.write(" ".join(map(str, ids)))
        ids_path = Path(handle.name)
    try:
        env = isolated_engine_env()
        if args.ram_gb is not None:
            env.pop("EXPERT_RAM", None)
            env["RAM_GB"] = str(args.ram_gb)
            env["RAM_HEADROOM_GB"] = str(args.ram_headroom_gb)
        else:
            env.pop("RAM_GB", None)
            env["EXPERT_RAM"] = str(args.expert_ram)
        env.update(
            {
                "SNAP": str(snapshot),
                "EVAL_IDS": str(ids_path),
                "EVAL_CHUNK": str(args.ctx_size),
                "CTX": str(args.ctx_size),
                "OMP_NUM_THREADS": str(args.threads),
                "PREFETCH_THREADS": str(args.prefetch_threads),
            }
        )
        env["EXPERT_Q2"] = "1" if args.expert_q2 else "0"
        env["EXPERT_Q3"] = "1" if args.expert_q3 else "0"
        if not args.copy_experts:
            # Evaluation visits nearly every expert in ascending grouped order;
            # mmap avoids copying the whole routed-expert set once per context.
            env["COLI_MMAP"] = "1"
        run = subprocess.run([str(engine)], env=env, text=True, capture_output=True, check=True)
    finally:
        ids_path.unlink(missing_ok=True)
    match = PPL_RE.search(run.stdout)
    if not match:
        raise SystemExit(f"could not parse engine result:\n{run.stdout}\n{run.stderr}")
    result: dict[str, object] = {
        "snapshot": str(snapshot),
        "model_manifest": model_manifest,
        "expert_lowbit_manifest": expert_lowbit_manifest,
        "corpus": str(corpus),
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "token_ids_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
        "tokens": int(match.group(1)),
        "source_tokens": len(ids),
        "ctx_size": args.ctx_size,
        "chunks": len(ids) // args.ctx_size,
        "expert_mode": "copied" if args.copy_experts else "mmap",
        "nll": float(match.group(2)),
        "ppl": float(match.group(3)),
        "tok_s": float(match.group(4)),
    }
    if args.llama_model:
        result["llama_runner"] = {
            "command": args.llama_perplexity,
            "version": binary_version(args.llama_perplexity),
        }
        command = [
            args.llama_perplexity,
            "-m",
            str(args.llama_model.resolve()),
            "-f",
            str(corpus),
            "-t",
            str(args.threads),
            "-tb",
            str(args.threads),
            "--ctx-size",
            str(args.ctx_size),
            "--batch-size",
            str(args.ctx_size),
            "--ubatch-size",
            str(args.ctx_size),
            "--chunks",
            str(len(ids) // args.ctx_size),
            "--no-warmup",
            "--no-repack",
        ]
        reference = subprocess.run(command, text=True, capture_output=True, check=True)
        reference_output = reference.stdout + "\n" + reference.stderr
        values = re.findall(r"Final estimate:\s*PPL\s*=\s*([0-9.eE+-]+)", reference_output, re.I)
        if not values:
            values = re.findall(r"perplexity\s*[:=]\s*([0-9.eE+-]+)", reference_output, re.I)
        if not values:
            raise SystemExit(f"could not parse llama-perplexity output:\n{reference_output}")
        llama_ppl = float(values[-1])
        result["llama_ppl"] = llama_ppl
        result["relative_ppl_delta"] = (float(result["ppl"]) - llama_ppl) / llama_ppl
    failures = []
    if args.max_ppl is not None and float(result["ppl"]) > args.max_ppl:
        failures.append(f"ppl {result['ppl']} exceeds {args.max_ppl}")
    if args.max_relative_ppl_delta is not None:
        delta = abs(float(result["relative_ppl_delta"]))
        if delta > args.max_relative_ppl_delta:
            failures.append(
                f"absolute relative PPL delta {delta} exceeds "
                f"{args.max_relative_ppl_delta}"
            )
    result["acceptance"] = {
        "max_ppl": args.max_ppl,
        "max_relative_ppl_delta": args.max_relative_ppl_delta,
        "passed": not failures,
        "failures": failures,
        "ram_gb": args.ram_gb,
        "expert_ram_per_layer": None if args.ram_gb is not None else args.expert_ram,
        "ram_headroom_gb": args.ram_headroom_gb,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    if failures:
        raise SystemExit("perplexity acceptance failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
