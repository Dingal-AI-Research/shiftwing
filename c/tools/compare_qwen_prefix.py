#!/usr/bin/env python3
"""Compare fixed-prompt greedy prefixes between colib and llama.cpp GGUF."""

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
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from expert_lowbit import load_complete_expert_sidecar  # noqa: E402
from runtime_env import isolated_engine_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--gguf",
        type=Path,
        help="reference GGUF (not needed for --c-only or --teacher-forced-only)",
    )
    parser.add_argument("--prompts", type=Path, default=root / "fixtures" / "qwen35_prompts.json")
    parser.add_argument("--engine", type=Path, default=root / "qwen")
    parser.add_argument("--llama-cli", default="llama-cli")
    parser.add_argument("--llama-server")
    parser.add_argument("--server-port", type=int, default=8191)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--expert-ram", type=int, default=64)
    parser.add_argument(
        "--ram-gb",
        type=float,
        help="byte-budgeted host expert cache; overrides --expert-ram",
    )
    parser.add_argument("--ram-headroom-gb", type=float, default=1.0)
    parser.add_argument(
        "--production-cuda",
        action="store_true",
        help="use the accepted CUDA expert/I/O profile for colib runs",
    )
    parser.add_argument("--cuda-expert-gb", type=float, default=6.0)
    parser.add_argument("--cuda-headroom-gb", type=float, default=1.0)
    parser.add_argument("--prefetch-threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--indices",
        help="comma-separated zero-based prompt indices (takes precedence over --limit)",
    )
    parser.add_argument(
        "--diagnostic-topk",
        action="store_true",
        help="record colib raw-logit and llama.cpp log-probability top-5 at the first generated token",
    )
    parser.add_argument("--c-only", action="store_true", help="run only colib (useful for kernel diagnostics)")
    parser.add_argument("--c-idot", choices=("0", "1"), help="override colib's activation-int8 path")
    parser.add_argument(
        "--reference-json",
        type=Path,
        help="reuse reference_tokens saved by an earlier run instead of loading llama.cpp",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expected-gguf-sha256",
        help="stream and verify the reference GGUF against this SHA-256",
    )
    parser.add_argument(
        "--require-complete-manifest",
        action="store_true",
        help="require quantization.json with complete=true and retain its identity",
    )
    parser.add_argument(
        "--expert-q2",
        action="store_true",
        help="select complete grouped-int2 routed-expert sidecars",
    )
    parser.add_argument(
        "--expert-q3",
        action="store_true",
        help="select complete grouped-int3 routed-expert sidecars",
    )
    parser.add_argument(
        "--expert-q3-max-layer",
        type=int,
        help=(
            "diagnostic: use int3 routed experts only through this zero-based "
            "layer, retaining the snapshot's normal expert format afterward"
        ),
    )
    parser.add_argument(
        "--expert-q3-min-layer",
        type=int,
        help=(
            "diagnostic: retain the snapshot's normal routed experts before "
            "this zero-based layer and use int3 afterward"
        ),
    )
    parser.add_argument(
        "--min-tf-prompt-agreement",
        type=float,
        help="require every prompt's teacher-forced agreement to meet this fraction",
    )
    parser.add_argument(
        "--min-tf-aggregate-agreement",
        type=float,
        help="require aggregate teacher-forced agreement to meet this fraction",
    )
    parser.add_argument(
        "--teacher-forced-only",
        action="store_true",
        help="skip free-running generation and replay --reference-json directly",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_version(command: str) -> str:
    run = subprocess.run(
        [command, "--version"],
        text=True,
        capture_output=True,
        check=True,
    )
    return (run.stdout + "\n" + run.stderr).strip()


def c_prefixes(
    engine: Path,
    snapshot: Path,
    rendered: list[str],
    tokens: int,
    threads: int,
    expert_ram: int,
    prefetch_threads: int,
    tokenizer: object,
    diagnostic_topk: bool = False,
    c_idot: str | None = None,
    ram_gb: float | None = None,
    ram_headroom_gb: float = 1.0,
    production_cuda: bool = False,
    cuda_expert_gb: float = 6.0,
    cuda_headroom_gb: float = 1.0,
) -> tuple[list[list[int]], list[list[dict[str, float | int]] | None]]:
    encoded = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered]
    with tempfile.NamedTemporaryFile("w", prefix="qwen-prefix-", suffix=".ids", delete=False) as handle:
        for ids in encoded:
            handle.write(" ".join(map(str, ids)) + "\n")
        ids_path = Path(handle.name)
    env = isolated_engine_env(
        preserve=(
            "EXPERT_Q2",
            "EXPERT_Q3",
            "EXPERT_Q3_MIN_LAYER",
            "EXPERT_Q3_MAX_LAYER",
        )
    )
    for name in ("SERVE_BATCH", "SERVE_RESIDENT", "EVAL_IDS", "TFPREFIX_IDS", "TF", "LOAD_ONLY"):
        env.pop(name, None)
    env.update(
        {
            "SNAP": str(snapshot),
            "PREFIX_IDS": str(ids_path),
            "NGEN": str(tokens),
            "CTX": str(max(map(len, encoded)) + tokens + 1),
            "OMP_NUM_THREADS": str(threads),
            "PREFETCH_THREADS": str(prefetch_threads),
        }
    )
    configure_colib_runtime(
        env,
        expert_ram=expert_ram,
        ram_gb=ram_gb,
        ram_headroom_gb=ram_headroom_gb,
        production_cuda=production_cuda,
        cuda_expert_gb=cuda_expert_gb,
        cuda_headroom_gb=cuda_headroom_gb,
    )
    if diagnostic_topk:
        env["DEBUG_LOGITS"] = "1"
    if c_idot is not None:
        env["IDOT"] = c_idot
    try:
        run = subprocess.run(
            [str(engine)], env=env, text=True, capture_output=True, check=False
        )
    finally:
        ids_path.unlink(missing_ok=True)
    if run.returncode != 0:
        raise RuntimeError(
            f"colib prefix run exited with status {run.returncode}:\n"
            f"{run.stdout}\n{run.stderr}"
        )
    rows = re.findall(r"^PREFIX\s+\d+:((?:\s+\d+)*)$", run.stdout, re.M)
    if len(rows) != len(rendered):
        raise RuntimeError(f"could not parse colib prefix output:\n{run.stdout}\n{run.stderr}")
    topk: list[list[dict[str, float | int]] | None] = [None] * len(rendered)
    if diagnostic_topk:
        for row, values in re.findall(r"^\[ROW (\d+)\]\[LOGITS 0\]((?:\s+\d+:[^\s]+)+)$", run.stderr, re.M):
            topk[int(row)] = [
                {"id": int(token), "logit": float(logit)}
                for token, logit in re.findall(r"(\d+):([^\s]+)", values)
            ]
    return [[int(value) for value in row.split()] for row in rows], topk


def c_teacher_forced(
    engine: Path,
    snapshot: Path,
    prompt_ids: list[list[int]],
    references: list[list[int]],
    threads: int,
    expert_ram: int,
    prefetch_threads: int,
    c_idot: str | None = None,
    ram_gb: float | None = None,
    ram_headroom_gb: float = 1.0,
    production_cuda: bool = False,
    cuda_expert_gb: float = 6.0,
    cuda_headroom_gb: float = 1.0,
) -> list[tuple[int, int]]:
    """Teacher-forced next-token agreement against the reference continuations.

    Free-running greedy comparison measures trajectory divergence: one flipped
    near-tie separates the two token streams permanently, so it reports how long
    two models stay on one path rather than whether they model the same
    distribution. Here the reference continuation is fed to colib as context and
    only the argmax at each position is compared -- the same criterion the
    tiny-model oracle uses (`tf_pred`).
    """
    with tempfile.NamedTemporaryFile("w", prefix="qwen-tf-", suffix=".ids", delete=False) as handle:
        for ids, reference in zip(prompt_ids, references):
            handle.write(" ".join(map(str, [len(ids)] + ids + reference)) + "\n")
        ids_path = Path(handle.name)
    env = isolated_engine_env(
        preserve=(
            "EXPERT_Q2",
            "EXPERT_Q3",
            "EXPERT_Q3_MIN_LAYER",
            "EXPERT_Q3_MAX_LAYER",
        )
    )
    for name in ("SERVE_BATCH", "SERVE_RESIDENT", "EVAL_IDS", "PREFIX_IDS", "TF", "LOAD_ONLY"):
        env.pop(name, None)
    env.update(
        {
            "SNAP": str(snapshot),
            "TFPREFIX_IDS": str(ids_path),
            "CTX": str(max(len(a) + len(b) for a, b in zip(prompt_ids, references)) + 1),
            "OMP_NUM_THREADS": str(threads),
            "PREFETCH_THREADS": str(prefetch_threads),
        }
    )
    configure_colib_runtime(
        env,
        expert_ram=expert_ram,
        ram_gb=ram_gb,
        ram_headroom_gb=ram_headroom_gb,
        production_cuda=production_cuda,
        cuda_expert_gb=cuda_expert_gb,
        cuda_headroom_gb=cuda_headroom_gb,
    )
    if c_idot is not None:
        env["IDOT"] = c_idot
    try:
        run = subprocess.run(
            [str(engine)], env=env, text=True, capture_output=True, check=False
        )
    finally:
        ids_path.unlink(missing_ok=True)
    if run.returncode != 0:
        raise RuntimeError(
            f"colib teacher-forced run exited with status {run.returncode}:\n"
            f"{run.stdout}\n{run.stderr}"
        )
    rows = re.findall(r"^TFPREFIX\s+\d+:\s+(\d+)/(\d+)$", run.stdout, re.M)
    if len(rows) != len(prompt_ids):
        raise RuntimeError(f"could not parse colib teacher-forced output:\n{run.stdout}\n{run.stderr}")
    return [(int(a), int(b)) for a, b in rows]


def configure_colib_runtime(
    env: dict[str, str],
    *,
    expert_ram: int,
    ram_gb: float | None,
    ram_headroom_gb: float,
    production_cuda: bool,
    cuda_expert_gb: float,
    cuda_headroom_gb: float,
) -> None:
    """Apply an explicit cache/backend profile without ambient-shell leakage."""
    if ram_gb is None:
        env.pop("RAM_GB", None)
        env.pop("RAM_HEADROOM_GB", None)
        env["EXPERT_RAM"] = str(expert_ram)
    else:
        env.pop("EXPERT_RAM", None)
        env["RAM_GB"] = str(ram_gb)
        env["RAM_HEADROOM_GB"] = str(ram_headroom_gb)
    if production_cuda:
        env.update(
            {
                "COLI_CUDA": "1",
                "CUDA_DENSE": "1",
                "CUDA_F16": "1",
                "CUDA_EXPERTS": "1",
                "CUDA_EXPERT_GB": str(cuda_expert_gb),
                "CUDA_HEADROOM_GB": str(cuda_headroom_gb),
                "PIPE": "1",
                "URING": "1",
                "DIRECT": "1",
                "URING_PERSIST": "1",
                "CUDA_PINNED_UPLOAD": "1",
                "DECODE_PROTECT": "1",
                "DECODE_PROTECT_PREWARM": "1",
                "AUTOPIN": "1",
                "PREFETCH_LOAD": "0",
                "PREFETCH_THREADS": "0",
            }
        )
    else:
        for name in (
            "COLI_CUDA",
            "CUDA_DENSE",
            "CUDA_F16",
            "CUDA_EXPERTS",
            "CUDA_EXPERT_GB",
            "CUDA_HEADROOM_GB",
            "PIPE",
            "URING",
            "DIRECT",
            "URING_PERSIST",
            "CUDA_PINNED_UPLOAD",
            "DECODE_PROTECT",
            "DECODE_PROTECT_PREWARM",
            "AUTOPIN",
            "PREFETCH_LOAD",
        ):
            env.pop(name, None)


def runtime_record(args: argparse.Namespace) -> dict[str, object]:
    return {
        "threads": args.threads,
        "expert_ram_per_layer": None if args.ram_gb is not None else args.expert_ram,
        "ram_gb": args.ram_gb,
        "ram_headroom_gb": args.ram_headroom_gb,
        "prefetch_threads": 0 if args.production_cuda else args.prefetch_threads,
        "production_cuda": bool(args.production_cuda),
        "cuda_expert_gb": args.cuda_expert_gb if args.production_cuda else None,
        "cuda_headroom_gb": args.cuda_headroom_gb if args.production_cuda else None,
        "persistent_uring": bool(args.production_cuda),
        "pinned_upload": bool(args.production_cuda),
        "decode_protect": bool(args.production_cuda),
        "decode_protect_prewarm": bool(args.production_cuda),
    }


def llama_prefix(command: str, gguf: Path, rendered: str, tokens: int, threads: int, tokenizer: object) -> list[int]:
    run = subprocess.run(
        [
            command,
            "-m", str(gguf),
            "-p", rendered,
            "-n", str(tokens),
            "-t", str(threads),
            "-tb", str(threads),
            "--temp", "0",
            "--repeat-penalty", "1",
            "--no-conversation",
            "--no-display-prompt",
            "--no-warmup",
            "--no-repack",
            "--no-show-timings",
            "--simple-io",
            "--log-disable",
            "--special",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return tokenizer.encode(run.stdout, add_special_tokens=False)[:tokens]


def wait_for_server(process: subprocess.Popen[bytes], base_url: str, timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("timed out waiting for llama-server")


def server_prefix(
    base_url: str, rendered: str, tokens: int, diagnostic_topk: bool = False
) -> tuple[list[int], list[dict[str, float | int]] | None]:
    payload = json.dumps(
        {
            "prompt": rendered,
            "n_predict": tokens,
            "temperature": 0,
            "samplers": ["temperature"],
            "return_tokens": True,
            "cache_prompt": False,
            **({"n_probs": 5} if diagnostic_topk else {}),
        }
    ).encode()
    request = urllib.request.Request(
        base_url + "/completion", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    values = result.get("tokens")
    if not isinstance(values, list):
        raise RuntimeError(f"llama-server response did not include token IDs: {result}")
    topk = None
    if diagnostic_topk:
        probabilities = result.get("completion_probabilities")
        if not isinstance(probabilities, list) or not probabilities:
            raise RuntimeError(f"llama-server response did not include probabilities: {result}")
        topk = [
            {"id": int(item["id"]), "logprob": float(item["logprob"])}
            for item in probabilities[0].get("top_logprobs", [])
        ]
    return [int(value) for value in values[:tokens]], topk


def main() -> None:
    args = parse_args()
    if args.expert_ram < 1 or args.prefetch_threads < 0:
        raise SystemExit("--expert-ram must be positive and --prefetch-threads non-negative")
    for name, value in (
        ("--ram-headroom-gb", args.ram_headroom_gb),
        ("--cuda-expert-gb", args.cuda_expert_gb),
        ("--cuda-headroom-gb", args.cuda_headroom_gb),
    ):
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"{name} must be positive and finite")
    if args.ram_gb is not None and (
        not math.isfinite(args.ram_gb) or args.ram_gb <= 0
    ):
        raise SystemExit("--ram-gb must be positive and finite")
    for name, value in (
        ("--min-tf-prompt-agreement", args.min_tf_prompt_agreement),
        ("--min-tf-aggregate-agreement", args.min_tf_aggregate_agreement),
    ):
        if value is not None and (
            not math.isfinite(value) or value < 0.0 or value > 1.0
        ):
            raise SystemExit(f"{name} must be between 0 and 1")
    if args.gguf is None and not (args.c_only or args.teacher_forced_only):
        raise SystemExit("--gguf is required unless using a reference-only mode")
    snapshot = args.snapshot.resolve()
    gguf = args.gguf.resolve() if args.gguf is not None else None
    engine = args.engine.resolve()
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
    if args.expert_q3_max_layer is not None or args.expert_q3_min_layer is not None:
        if not args.expert_q3:
            raise SystemExit("int3 layer bounds require --expert-q3")
    if args.expert_q3_max_layer is not None:
        if args.expert_q3_max_layer < -1:
            raise SystemExit("--expert-q3-max-layer must be -1 or greater")
    if args.expert_q3_min_layer is not None and args.expert_q3_min_layer < 0:
        raise SystemExit("--expert-q3-min-layer must be zero or greater")
    if (
        args.expert_q3_min_layer is not None
        and args.expert_q3_max_layer is not None
        and args.expert_q3_min_layer > args.expert_q3_max_layer
    ):
        raise SystemExit("--expert-q3-min-layer must not exceed --expert-q3-max-layer")
    expert_lowbit_manifest = None
    selected_bits = 2 if args.expert_q2 else 3 if args.expert_q3 else None
    # Do not allow ambient shell settings to silently change a reference run.
    for name in (
        "EXPERT_Q2",
        "EXPERT_Q3",
        "EXPERT_Q3_MIN_LAYER",
        "EXPERT_Q3_MAX_LAYER",
    ):
        os.environ.pop(name, None)
    os.environ["EXPERT_Q2"] = "1" if args.expert_q2 else "0"
    os.environ["EXPERT_Q3"] = "1" if args.expert_q3 else "0"
    if selected_bits is not None:
        try:
            lowbit_manifest = load_complete_expert_sidecar(
                snapshot, bits=selected_bits
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        expert_lowbit_manifest = lowbit_manifest
        os.environ[f"EXPERT_Q{selected_bits}"] = "1"
        if args.expert_q3_max_layer is not None:
            layers = lowbit_manifest.get("layers")
            if not isinstance(layers, int) or args.expert_q3_max_layer >= layers:
                raise SystemExit(
                    f"--expert-q3-max-layer must be below manifest layer count {layers}"
                )
            os.environ["EXPERT_Q3_MAX_LAYER"] = str(args.expert_q3_max_layer)
        if args.expert_q3_min_layer is not None:
            layers = lowbit_manifest.get("layers")
            if not isinstance(layers, int) or args.expert_q3_min_layer >= layers:
                raise SystemExit(
                    f"--expert-q3-min-layer must be below manifest layer count {layers}"
                )
            os.environ["EXPERT_Q3_MIN_LAYER"] = str(args.expert_q3_min_layer)
    expected_gguf_sha256 = (
        args.expected_gguf_sha256.lower()
        if args.expected_gguf_sha256 is not None
        else None
    )
    if expected_gguf_sha256 is not None and (
        len(expected_gguf_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_gguf_sha256)
    ):
        raise SystemExit("--expected-gguf-sha256 must contain 64 hexadecimal digits")
    gguf_sha256 = None
    if expected_gguf_sha256 is not None:
        assert gguf is not None
        gguf_sha256 = sha256_file(gguf)
        if gguf_sha256 != expected_gguf_sha256:
            raise SystemExit(
                f"GGUF SHA-256 {gguf_sha256} != expected {expected_gguf_sha256}"
            )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    prompt_bytes = args.prompts.read_bytes()
    all_prompts = json.loads(prompt_bytes)
    if args.indices:
        source_indices = [int(value) for value in args.indices.split(",")]
        if not source_indices or any(index < 0 or index >= len(all_prompts) for index in source_indices):
            raise SystemExit(f"--indices must select prompts in [0, {len(all_prompts) - 1}]")
    else:
        source_indices = list(range(len(all_prompts)))
        if args.limit:
            source_indices = source_indices[: args.limit]
    prompts = [all_prompts[index] for index in source_indices]
    rendered_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        for prompt in prompts
    ]
    if args.teacher_forced_only:
        if not args.reference_json:
            raise SystemExit("--teacher-forced-only requires --reference-json")
        reference_bytes = args.reference_json.read_bytes()
        prior = json.loads(reference_bytes)
        saved = {
            int(row["index"]): [int(token) for token in row["reference_tokens"]]
            for row in prior.get("rows", [])
            if "reference_tokens" in row
        }
        missing = [index for index in source_indices if index not in saved]
        if missing:
            raise SystemExit(f"reference JSON is missing prompt indices: {missing}")
        references = [saved[index][: args.tokens] for index in source_indices]
        encoded_prompts = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in rendered_prompts
        ]
        tf_rows = c_teacher_forced(
            engine,
            snapshot,
            encoded_prompts,
            references,
            args.threads,
            args.expert_ram,
            args.prefetch_threads,
            args.c_idot,
            ram_gb=args.ram_gb,
            ram_headroom_gb=args.ram_headroom_gb,
            production_cuda=args.production_cuda,
            cuda_expert_gb=args.cuda_expert_gb,
            cuda_headroom_gb=args.cuda_headroom_gb,
        )
        rows = []
        failures = []
        for index, (matches, compared) in zip(source_indices, tf_rows):
            agreement = matches / compared if compared else 0.0
            rows.append(
                {
                    "index": index,
                    "tf_matches": matches,
                    "tf_compared": compared,
                    "tf_agreement": agreement,
                }
            )
            if compared != args.tokens:
                failures.append(
                    f"prompt {index} has {compared}, not {args.tokens}, positions"
                )
            if (
                args.min_tf_prompt_agreement is not None
                and agreement < args.min_tf_prompt_agreement
            ):
                failures.append(
                    f"prompt {index} teacher-forced agreement {agreement} "
                    f"is below {args.min_tf_prompt_agreement}"
                )
        tf_matches = sum(matches for matches, _ in tf_rows)
        tf_total = sum(compared for _, compared in tf_rows)
        tf_agreement = tf_matches / tf_total if tf_total else 0.0
        if (
            args.min_tf_aggregate_agreement is not None
            and tf_agreement < args.min_tf_aggregate_agreement
        ):
            failures.append(
                f"aggregate teacher-forced agreement {tf_agreement} is below "
                f"{args.min_tf_aggregate_agreement}"
            )
        result = {
            "snapshot": str(snapshot),
            "model_manifest": model_manifest,
            "expert_lowbit_manifest": expert_lowbit_manifest,
            "expert_q3_max_layer": args.expert_q3_max_layer,
            "expert_q3_min_layer": args.expert_q3_min_layer,
            "mode": "teacher-forced-only",
            "colib_runtime": runtime_record(args),
            "reference_json": str(args.reference_json.resolve()),
            "reference_json_sha256": hashlib.sha256(reference_bytes).hexdigest(),
            "prompts_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "prompts": len(prompts),
            "tf_matching_tokens": tf_matches,
            "tf_compared_tokens": tf_total,
            "tf_agreement": tf_agreement,
            "tf_prompts_at_85_percent": sum(
                row["tf_agreement"] >= 0.85 for row in rows
            ),
            "rows": rows,
            "acceptance": {
                "minimum_prompt_tf_agreement": args.min_tf_prompt_agreement,
                "minimum_aggregate_tf_agreement": args.min_tf_aggregate_agreement,
                "required_positions_per_prompt": args.tokens,
                "passed": not failures,
                "failures": failures,
            },
        }
        rendered = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.write_text(rendered)
        print(rendered, end="")
        if failures:
            raise SystemExit(
                "teacher-forced acceptance failed: " + "; ".join(failures)
            )
        return
    actual_prefixes, c_topk = c_prefixes(
        engine,
        snapshot,
        rendered_prompts,
        args.tokens,
        args.threads,
        args.expert_ram,
        args.prefetch_threads,
        tokenizer,
        args.diagnostic_topk,
        args.c_idot,
        ram_gb=args.ram_gb,
        ram_headroom_gb=args.ram_headroom_gb,
        production_cuda=args.production_cuda,
        cuda_expert_gb=args.cuda_expert_gb,
        cuda_headroom_gb=args.cuda_headroom_gb,
    )
    if args.c_only:
        failures = [
            f"prompt {index} produced {len(tokens)}, not {args.tokens}, tokens"
            for index, tokens in zip(source_indices, actual_prefixes)
            if len(tokens) != args.tokens
        ]
        result = {
            "snapshot": str(snapshot),
            "model_manifest": model_manifest,
            "expert_lowbit_manifest": expert_lowbit_manifest,
            "mode": "colib-reference-capture",
            "colib_runtime": runtime_record(args),
            "reference_profile": (
                f"expert-int{selected_bits}" if selected_bits is not None else "base"
            ),
            "prompts_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "prompts": len(prompts),
            "c_idot": args.c_idot,
            "rows": [
                {
                    "index": index,
                    "tokens": tokens,
                    "reference_tokens": tokens,
                    "colib_top5": topk,
                }
                for index, tokens, topk in zip(source_indices, actual_prefixes, c_topk)
            ],
            "acceptance": {
                "required_positions_per_prompt": args.tokens,
                "passed": not failures,
                "failures": failures,
            },
        }
        output = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.write_text(output)
        print(output, end="")
        if failures:
            raise SystemExit("reference capture failed: " + "; ".join(failures))
        return
    saved_references: dict[int, list[int]] = {}
    reference_runner_version = None
    reference_runner_command = None
    if args.reference_json:
        prior = json.loads(args.reference_json.read_text())
        reference_runner_version = prior.get("reference_runner_version")
        reference_runner_command = prior.get("reference_runner_command")
        saved_references = {
            int(row["index"]): [int(token) for token in row["reference_tokens"]]
            for row in prior.get("rows", [])
            if "reference_tokens" in row
        }
        missing = [index for index in source_indices if index not in saved_references]
        if missing:
            raise SystemExit(f"reference JSON is missing prompt indices: {missing}")
    server = None
    base_url = f"http://127.0.0.1:{args.server_port}"
    if args.llama_server and not saved_references:
        reference_runner_command = args.llama_server
        reference_runner_version = binary_version(args.llama_server)
        server = subprocess.Popen(
            [
                args.llama_server,
                "-m", str(gguf),
                "-t", str(args.threads),
                "-tb", str(args.threads),
                "--host", "127.0.0.1",
                "--port", str(args.server_port),
                "--no-repack",
                "--no-warmup",
                "--log-disable",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wait_for_server(server, base_url)
    elif not saved_references:
        reference_runner_command = args.llama_cli
        reference_runner_version = binary_version(args.llama_cli)
    rows = []
    references: list[list[int]] = []
    equal = total = 0
    try:
        for local_index, prompt in enumerate(prompts):
            source_index = source_indices[local_index]
            rendered = rendered_prompts[local_index]
            actual = actual_prefixes[local_index]
            if saved_references:
                reference, reference_topk = saved_references[source_index][: args.tokens], None
            elif server:
                reference, reference_topk = server_prefix(
                    base_url, rendered, args.tokens, args.diagnostic_topk
                )
            else:
                reference = llama_prefix(args.llama_cli, gguf, rendered, args.tokens, args.threads, tokenizer)
                reference_topk = None
            compared = min(len(actual), len(reference), args.tokens)
            matches = sum(actual[i] == reference[i] for i in range(compared))
            common_prefix = 0
            while common_prefix < compared and actual[common_prefix] == reference[common_prefix]:
                common_prefix += 1
            references.append(reference)
            equal += matches
            total += compared
            row = {
                    "index": source_index,
                    "matches": matches,
                    "compared": compared,
                    "agreement": matches / compared if compared else 0,
                    "common_prefix": common_prefix,
                    "first_actual": actual[0] if actual else None,
                    "first_reference": reference[0] if reference else None,
                    "actual_tokens": actual,
                    "reference_tokens": reference,
                }
            if args.diagnostic_topk:
                row["colib_top5"] = c_topk[local_index]
                row["llama_top5"] = reference_topk
            rows.append(row)
            print(
                f"prompt index {source_index} ({local_index + 1}/{len(prompts)}): "
                f"{matches}/{compared}, prefix={common_prefix}",
                flush=True,
            )
    finally:
        if server:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    encoded_prompts = [tokenizer.encode(text, add_special_tokens=False) for text in rendered_prompts]
    token_ids_sha256 = hashlib.sha256(
        json.dumps(encoded_prompts, separators=(",", ":")).encode()
    ).hexdigest()
    tf_rows = c_teacher_forced(
        engine,
        snapshot,
        encoded_prompts,
        references,
        args.threads,
        args.expert_ram,
        args.prefetch_threads,
        args.c_idot,
        ram_gb=args.ram_gb,
        ram_headroom_gb=args.ram_headroom_gb,
        production_cuda=args.production_cuda,
        cuda_expert_gb=args.cuda_expert_gb,
        cuda_headroom_gb=args.cuda_headroom_gb,
    )
    tf_matches = sum(a for a, _ in tf_rows)
    tf_total = sum(b for _, b in tf_rows)
    for row, (a, b) in zip(rows, tf_rows):
        row["tf_matches"], row["tf_compared"] = a, b
        row["tf_agreement"] = a / b if b else 0
    print(
        f"teacher-forced agreement: {tf_matches}/{tf_total} = "
        f"{tf_matches / tf_total if tf_total else 0:.6f}",
        flush=True,
    )

    tf_agreement = tf_matches / tf_total if tf_total else 0
    failures = []
    if (
        args.min_tf_prompt_agreement is not None
        or args.min_tf_aggregate_agreement is not None
    ):
        short_rows = [
            row["index"] for row in rows if row["tf_compared"] != args.tokens
        ]
        if short_rows:
            failures.append(
                f"teacher-forced rows do not contain exactly {args.tokens} "
                f"positions: {short_rows}"
            )
    if args.min_tf_prompt_agreement is not None:
        failed_rows = [
            row["index"]
            for row in rows
            if row["tf_agreement"] < args.min_tf_prompt_agreement
        ]
        if failed_rows:
            failures.append(
                "teacher-forced prompt agreement below "
                f"{args.min_tf_prompt_agreement}: {failed_rows}"
            )
    if (
        args.min_tf_aggregate_agreement is not None
        and tf_agreement < args.min_tf_aggregate_agreement
    ):
        failures.append(
            f"aggregate teacher-forced agreement {tf_agreement} below "
            f"{args.min_tf_aggregate_agreement}"
        )
    result = {
        "snapshot": str(snapshot),
        "model_manifest": model_manifest,
        "expert_lowbit_manifest": expert_lowbit_manifest,
        "colib_runtime": runtime_record(args),
        "expert_q3_max_layer": args.expert_q3_max_layer,
        "expert_q3_min_layer": args.expert_q3_min_layer,
        "gguf": str(gguf) if gguf is not None else None,
        "gguf_bytes": gguf.stat().st_size if gguf is not None and gguf.exists() else None,
        "gguf_sha256": gguf_sha256,
        "expected_gguf_sha256": expected_gguf_sha256,
        "prompts_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "rendered_token_ids_sha256": token_ids_sha256,
        "reference_runner": (
            "saved-json" if saved_references else "llama-server" if args.llama_server else "llama-cli"
        ),
        "reference_runner_command": reference_runner_command,
        "reference_runner_version": reference_runner_version,
        "prompts": len(prompts),
        "matching_tokens": equal,
        "compared_tokens": total,
        "agreement": equal / total if total else 0,
        "prompts_at_85_percent": sum(row["agreement"] >= 0.85 for row in rows),
        "tf_matching_tokens": tf_matches,
        "tf_compared_tokens": tf_total,
        "tf_agreement": tf_agreement,
        "tf_prompts_at_85_percent": sum(row["tf_agreement"] >= 0.85 for row in rows),
        "rows": rows,
        "acceptance": {
            "minimum_prompt_tf_agreement": args.min_tf_prompt_agreement,
            "minimum_aggregate_tf_agreement": args.min_tf_aggregate_agreement,
            "required_positions_per_prompt": (
                args.tokens
                if args.min_tf_prompt_agreement is not None
                or args.min_tf_aggregate_agreement is not None
                else None
            ),
            "passed": not failures,
            "failures": failures,
        },
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    if failures:
        raise SystemExit("teacher-forced acceptance failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
