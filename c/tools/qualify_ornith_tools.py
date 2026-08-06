#!/usr/bin/env python3
"""Exercise an actual Ornith tool call/result/final-answer round trip over HTTP."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from expert_lowbit import load_complete_expert_sidecar  # noqa: E402
from runtime_env import isolated_engine_env  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _get_json(url: str, timeout: float) -> dict:
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _post_json(url: str, body: dict, timeout: float) -> dict:
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _wait_ready(url: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness (code {process.returncode})")
        try:
            return _get_json(url + "/health", 2.0)
        except (OSError, URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"server did not become ready within {timeout:.0f}s")
            time.sleep(0.25)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=ROOT / "qwen")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ram-gb", type=float, default=8.0)
    parser.add_argument("--ram-headroom-gb", type=float, default=1.0)
    parser.add_argument("--cuda-expert-gb", type=float, default=6.0)
    parser.add_argument("--cuda-headroom-gb", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--expert-q3",
        action="store_true",
        help="select a complete grouped-int3 routed-expert sidecar",
    )
    args = parser.parse_args()

    if args.max_tokens < 1 or args.context < args.max_tokens or args.threads < 1:
        parser.error("context must cover max-tokens and threads must be positive")
    if (
        not math.isfinite(args.ram_gb)
        or args.ram_gb <= 0
        or not math.isfinite(args.cuda_expert_gb)
        or args.cuda_expert_gb < 0
    ):
        parser.error("RAM must be positive and the CUDA expert budget non-negative")
    if (
        not math.isfinite(args.ram_headroom_gb)
        or args.ram_headroom_gb <= 0
        or not math.isfinite(args.cuda_headroom_gb)
        or args.cuda_headroom_gb <= 0
    ):
        parser.error("host and device headroom must be positive and finite")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")

    model = args.model.resolve()
    engine = args.engine.resolve()
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if config.get("colib_model_family") != "ornith-1.0":
        raise SystemExit(
            "model config does not declare colib_model_family=ornith-1.0"
        )
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
    expert_lowbit_manifest = None
    if args.expert_q3:
        try:
            expert_lowbit_manifest = load_complete_expert_sidecar(model, bits=3)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    model_id = "ornith-colib"
    env = isolated_engine_env()
    env.pop("EXPERT_RAM", None)
    env.update(
        {
            "SERVE_RESIDENT": "1",
            "CTX": str(args.context),
            "OMP_NUM_THREADS": str(args.threads),
            "RAM_GB": str(args.ram_gb),
            "RAM_HEADROOM_GB": str(args.ram_headroom_gb),
            "CUDA_EXPERT_GB": str(args.cuda_expert_gb),
            "PIPE": "1",
            "URING": "1",
            "DIRECT": "1",
            "AUTOPIN": "1",
            "PREFETCH_LOAD": "0",
            "PREFETCH_THREADS": "0",
            "URING_PERSIST": "0",
            "CUDA_PINNED_UPLOAD": "0",
            "DECODE_PROTECT": "0",
            "DECODE_PROTECT_PREWARM": "0",
            "EXPERT_Q2": "0",
            "EXPERT_Q3": "1" if args.expert_q3 else "0",
        }
    )
    env.pop("CUDA_PROFILE_STAGES", None)
    if not args.cpu:
        env.update(
            {
                "COLI_CUDA": "1",
                "CUDA_DENSE": "1",
                "CUDA_F16": "1",
                "CUDA_EXPERTS": "1",
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

    command = [
        os.fspath(Path(os.environ.get("COLI_PYTHON", sys.executable))),
        os.fspath(ROOT / "openai_server.py"),
        "--model",
        os.fspath(model),
        "--engine",
        os.fspath(engine),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-id",
        model_id,
        "--max-tokens",
        str(args.max_tokens),
        "--kv-slots",
        "1",
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    user_message = {
        "role": "user",
        "content": "Use get_weather to find the current weather in Paris.",
    }
    started = time.monotonic()
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            health = _wait_ready(base, process, args.startup_timeout)
            ready_s = time.monotonic() - started
            first_started = time.monotonic()
            first = _post_json(
                base + "/v1/chat/completions",
                {
                    "model": model_id,
                    "messages": [user_message],
                    "tools": tools,
                    "tool_choice": "required",
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": args.max_tokens,
                },
                args.request_timeout,
            )
            first_s = time.monotonic() - first_started
            first_choice = first["choices"][0]
            if first_choice.get("finish_reason") != "tool_calls":
                raise AssertionError(
                    "first response did not terminate with finish_reason=tool_calls"
                )
            message = first_choice["message"]
            calls = message.get("tool_calls") or []
            if len(calls) != 1:
                raise AssertionError(f"expected one tool call, received {len(calls)}")
            call = calls[0]
            function = call.get("function") or {}
            if function.get("name") != "get_weather":
                raise AssertionError(f"unexpected tool name {function.get('name')!r}")
            arguments = json.loads(function.get("arguments", ""))
            city = arguments.get("city")
            if not isinstance(city, str) or city.strip().casefold() != "paris":
                raise AssertionError(f"unexpected city argument {city!r}")

            assistant_history = {
                key: value
                for key, value in message.items()
                if key in {"role", "content", "reasoning_content", "tool_calls"}
            }
            second_started = time.monotonic()
            second = _post_json(
                base + "/v1/chat/completions",
                {
                    "model": model_id,
                    "messages": [
                        user_message,
                        assistant_history,
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(
                                {
                                    "city": "Paris",
                                    "temperature_c": 18,
                                    "condition": "clear",
                                },
                                separators=(",", ":"),
                            ),
                        },
                    ],
                    "tools": tools,
                    "tool_choice": "none",
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": args.max_tokens,
                },
                args.request_timeout,
            )
            second_s = time.monotonic() - second_started
            second_choice = second["choices"][0]
            if second_choice.get("finish_reason") != "stop":
                raise AssertionError(
                    "final response did not terminate cleanly with finish_reason=stop"
                )
            final_message = second_choice["message"]
            final_text = final_message.get("content") or ""
            if not final_text.strip() or "18" not in final_text:
                raise AssertionError(
                    "final answer is empty or does not use the 18 °C tool result"
                )
            if final_message.get("tool_calls"):
                raise AssertionError("tool_choice=none response emitted another tool call")
            if "<tool_call>" in final_text or "<function=" in final_text:
                raise AssertionError(
                    "tool_choice=none response leaked raw tool-call markup"
                )

            profile = _get_json(base + "/profile", 5.0)
            experts = _get_json(base + "/experts", 5.0)
            if not args.cpu:
                resident = profile.get("resident") or {}
                if (
                    resident.get("layers", 0) <= 0
                    or resident.get("device_moe", 0) <= 0
                    or resident.get("host_moe", -1) != 0
                    or resident.get("router_d2h_bytes", 0) <= 0
                    or resident.get("logits_d2h_bytes", 0) <= 0
                ):
                    raise AssertionError(
                        "CUDA tool round trip did not retain the resident graph"
                    )
            result = {
                "schema_version": 2,
                "model": str(model),
                "engine": str(engine),
                "model_manifest": model_manifest,
                "expert_lowbit_manifest": expert_lowbit_manifest,
                "cpu": args.cpu,
                "tier_configuration": {
                    "threads": args.threads,
                    "ram_gb": args.ram_gb,
                    "ram_headroom_gb": args.ram_headroom_gb,
                    "cuda_expert_gb": (
                        None if args.cpu else args.cuda_expert_gb
                    ),
                    "cuda_headroom_gb": (
                        None if args.cpu else args.cuda_headroom_gb
                    ),
                    "tiered_io": True,
                    "predictive_prefetch": False,
                    "uring_persist": False,
                    "pinned_upload": False,
                    "decode_protect": False,
                    "decode_protect_prewarm": False,
                    "expert_lowbit": "int3g128" if args.expert_q3 else None,
                },
                "startup_s": ready_s,
                "tool_call_s": first_s,
                "final_answer_s": second_s,
                "health": health,
                "tool_call_response": first,
                "tool_result": {
                    "city": "Paris",
                    "temperature_c": 18,
                    "condition": "clear",
                },
                "final_response": second,
                "profile": profile,
                "experts": experts,
                "passed": True,
            }
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            log.seek(0)
            log_tail = log.read().decode("utf-8", "replace")[-16000:]

    result["server_log_tail"] = log_tail
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
