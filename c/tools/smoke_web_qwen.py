#!/usr/bin/env python3
"""Run one production-model web/gateway/engine streaming smoke request."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "c" / "colib"
sys.path.insert(0, str(ROOT / "c"))
from tools.expert_lowbit import load_complete_expert_sidecar  # noqa: E402
from tools.runtime_env import isolated_engine_env  # noqa: E402


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def acceptance_failures(
    runs: list[dict],
    final_health: dict,
    profile: dict,
    *,
    page_ok: bool,
    expected_content: str,
    cuda: bool,
    expected_requests: int,
) -> list[str]:
    """Return every production web/lifecycle acceptance failure."""
    failures = []
    if not page_ok:
        failures.append("web bundle did not contain the colib title")
    for run in runs:
        if run["content"].strip() != expected_content:
            failures.append(
                f"request {run['index']} content did not exactly match "
                f"{expected_content!r}"
            )
        usage = run.get("usage")
        if (
            not isinstance(usage, dict)
            or int(usage.get("completion_tokens", 0)) <= 0
        ):
            failures.append(f"request {run['index']} usage is missing")
    scheduler = final_health.get("scheduler")
    if not isinstance(scheduler, dict):
        failures.append("final scheduler health is missing")
    else:
        if scheduler.get("active") != 0 or scheduler.get("queued") != 0:
            failures.append("scheduler did not release all requests")
        if int(scheduler.get("completed", 0)) < expected_requests:
            failures.append("scheduler completion count is too small")
    if cuda:
        hwinfo = final_health.get("hwinfo") or {}
        tiers = final_health.get("tiers") or {}
        if int(hwinfo.get("gpus", 0)) < 1:
            failures.append("CUDA GPU telemetry is missing")
        if int(tiers.get("vram", 0)) < 1:
            failures.append("VRAM expert tier is inactive")
        resident = profile.get("resident") if isinstance(profile, dict) else None
        if (
            not isinstance(resident, dict)
            or int(resident.get("layers", 0)) < 1
            or int(resident.get("device_moe", 0)) < 1
            or int(resident.get("host_moe", -1)) != 0
            or int(resident.get("router_d2h_bytes", 0)) < 1
            or int(resident.get("logits_d2h_bytes", 0)) < 1
        ):
            failures.append("resident CUDA graph is inactive or used host MoE")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="Reply with exactly: colib ready")
    parser.add_argument("--expect-exact", default="colib ready")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--kv-slots", type=int, default=2)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--expert-q3",
        action="store_true",
        help="select a complete grouped-int3 routed-expert sidecar",
    )
    parser.add_argument("--cuda-expert-gb", type=float, default=7.0)
    parser.add_argument("--cuda-headroom-gb", type=float, default=1.0)
    parser.add_argument("--expert-ram", type=int, default=2)
    parser.add_argument("--ram-gb", type=float)
    parser.add_argument("--ram-headroom-gb", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.requests < 1 or args.kv_slots < 1:
        parser.error("requests and kv-slots must be positive")
    if args.max_tokens < 1 or args.context < args.max_tokens or args.threads < 1:
        parser.error("context must cover max-tokens and threads must be positive")
    if args.expert_ram < 1:
        parser.error("--expert-ram must be positive")
    if args.ram_gb is not None and args.ram_gb <= 0:
        parser.error("--ram-gb must be positive")
    if (
        args.cuda_expert_gb < 0
        or args.cuda_headroom_gb <= 0
        or args.ram_headroom_gb <= 0
    ):
        parser.error("cache budgets cannot be negative and headroom must be positive")
    model = args.model.resolve()
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (model / "quantization.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"model metadata is unavailable: {error}")
    if not isinstance(config, dict) or not isinstance(manifest, dict):
        raise SystemExit("model metadata roots must be objects")
    if manifest.get("complete") is not True:
        raise SystemExit("model quantization manifest is absent or incomplete")
    model_family = config.get("colib_model_family", "qwen3.5")
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

    port = free_port()
    command = [
        str(CLI),
        "web",
        "--no-build",
        "--model",
        str(model),
        "--port",
        str(port),
        "--max-tokens",
        str(args.max_tokens),
        "--kv-slots",
        str(args.kv_slots),
    ]
    if args.cuda:
        command.append("--cuda")
    env = {
        **isolated_engine_env(),
        "CUDA_EXPERT_GB": str(args.cuda_expert_gb),
        "CUDA_HEADROOM_GB": str(args.cuda_headroom_gb),
        "CTX": str(args.context),
        "OMP_NUM_THREADS": str(args.threads),
        "PREFETCH_THREADS": "0",
        "PREFETCH_LOAD": "0",
        "SERVE_RESIDENT": "1",
        "URING_PERSIST": "0",
        "CUDA_PINNED_UPLOAD": "0",
        "DECODE_PROTECT": "0",
        "DECODE_PROTECT_PREWARM": "0",
        "EXPERT_Q2": "0",
        "EXPERT_Q3": "1" if args.expert_q3 else "0",
    }
    env.pop("CUDA_PROFILE_STAGES", None)
    if args.ram_gb is None:
        env.pop("RAM_GB", None)
        env["EXPERT_RAM"] = str(args.expert_ram)
    else:
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
    if not args.cuda:
        env.update(
            {
                "COLI_CUDA": "0",
                "CUDA_DENSE": "0",
                "CUDA_F16": "0",
                "CUDA_EXPERTS": "0",
            }
        )
    started = time.monotonic()
    # Keep server diagnostics off the pipe used by the test process. The C
    # engine can emit a large telemetry block while shutting down; an unread
    # PIPE can fill and hide or even deadlock the original request failure.
    stderr_log = tempfile.TemporaryFile()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=stderr_log,
    )
    failed = False
    try:
        deadline = started + args.timeout
        health = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr_log.seek(0)
                stderr = stderr_log.read().decode(errors="replace")
                raise RuntimeError(f"server exited during startup:\n{stderr}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    health = json.load(response)
                break
            except (OSError, URLError, json.JSONDecodeError):
                time.sleep(0.1)
        if health is None:
            raise TimeoutError("server startup timed out")
        ready = time.monotonic()
        with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            page_ok = b"<title>colib</title>" in response.read()

        runs = []
        history = []
        for run in range(args.requests):
            messages = [*history, {"role": "user", "content": args.prompt}]
            body = json.dumps(
                {
                    "model": "qwen3.5-colib",
                    "messages": messages,
                    "temperature": 0,
                    "top_p": 1,
                    "max_completion_tokens": args.max_tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "cache_slot": 0,
                }
            ).encode()
            request = Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            request_started = time.monotonic()
            first_content = None
            content: list[str] = []
            usage = None
            done = False
            with urlopen(request, timeout=args.timeout) as response:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    choice = (event.get("choices") or [{}])[0]
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        if first_content is None:
                            first_content = time.monotonic()
                        content.append(text)
                    if event.get("usage"):
                        usage = event["usage"]
            finished = time.monotonic()
            if not done or first_content is None:
                raise RuntimeError("stream did not produce content and [DONE]")
            runs.append(
                {
                    "index": run,
                    "ttft_s": first_content - request_started,
                    "request_s": finished - request_started,
                    "content": "".join(content),
                    "usage": usage,
                }
            )
            history = [*messages, {"role": "assistant", "content": "".join(content)}]
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            final_health = json.load(response)
        with urlopen(f"http://127.0.0.1:{port}/profile", timeout=5) as response:
            profile = json.load(response)
        failures = acceptance_failures(
            runs,
            final_health,
            profile,
            page_ok=page_ok,
            expected_content=args.expect_exact,
            cuda=args.cuda,
            expected_requests=args.requests,
        )
        result = {
            "schema_version": 3,
            "model": str(model),
            "model_family": model_family,
            "model_manifest": model_manifest,
            "expert_lowbit_manifest": expert_lowbit_manifest,
            "cuda": args.cuda,
            "configuration": {
                "prompt": args.prompt,
                "expected_exact_content": args.expect_exact,
                "requests": args.requests,
                "kv_slots": args.kv_slots,
                "max_tokens": args.max_tokens,
                "context": args.context,
                "threads": args.threads,
                "expert_ram_per_layer": (
                    args.expert_ram if args.ram_gb is None else None
                ),
                "ram_gb": args.ram_gb,
                "ram_headroom_gb": args.ram_headroom_gb,
                "cuda_expert_gb": args.cuda_expert_gb if args.cuda else None,
                "cuda_headroom_gb": (
                    args.cuda_headroom_gb if args.cuda else None
                ),
                "predictive_prefetch": False,
                "uring_persist": False,
                "pinned_upload": False,
                "decode_protect": False,
                "decode_protect_prewarm": False,
                "expert_lowbit": "int3g128" if args.expert_q3 else None,
            },
            "web_bundle": page_ok,
            "startup_s": ready - started,
            "requests": runs,
            "startup_health": health,
            "final_health": final_health,
            "profile": profile,
            "acceptance": {
                "passed": not failures,
                "failures": failures,
            },
        }
        rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0 if not failures else 1
    except BaseException:
        failed = True
        raise
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if failed:
            stderr_log.seek(0)
            stderr = stderr_log.read().decode(errors="replace").strip()
            if stderr:
                print("[web-server stderr]", file=sys.stderr)
                print(stderr, file=sys.stderr)
        stderr_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
