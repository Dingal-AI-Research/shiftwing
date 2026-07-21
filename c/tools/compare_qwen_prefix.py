#!/usr/bin/env python3
"""Compare fixed-prompt greedy prefixes between colib and llama.cpp GGUF."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=root / "fixtures" / "qwen35_prompts.json")
    parser.add_argument("--engine", type=Path, default=root / "qwen")
    parser.add_argument("--llama-cli", default="llama-cli")
    parser.add_argument("--llama-server")
    parser.add_argument("--server-port", type=int, default=8191)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--expert-ram", type=int, default=64)
    parser.add_argument("--prefetch-threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def c_prefixes(
    engine: Path,
    snapshot: Path,
    rendered: list[str],
    tokens: int,
    threads: int,
    expert_ram: int,
    prefetch_threads: int,
    tokenizer: object,
) -> list[list[int]]:
    encoded = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered]
    with tempfile.NamedTemporaryFile("w", prefix="qwen-prefix-", suffix=".ids", delete=False) as handle:
        for ids in encoded:
            handle.write(" ".join(map(str, ids)) + "\n")
        ids_path = Path(handle.name)
    env = os.environ.copy()
    env.update(
        {
            "SNAP": str(snapshot),
            "PREFIX_IDS": str(ids_path),
            "NGEN": str(tokens),
            "CTX": str(max(map(len, encoded)) + tokens + 1),
            "OMP_NUM_THREADS": str(threads),
            "EXPERT_RAM": str(expert_ram),
            "PREFETCH_THREADS": str(prefetch_threads),
        }
    )
    try:
        run = subprocess.run([str(engine)], env=env, text=True, capture_output=True, check=True)
    finally:
        ids_path.unlink(missing_ok=True)
    rows = re.findall(r"^PREFIX\s+\d+:((?:\s+\d+)*)$", run.stdout, re.M)
    if len(rows) != len(rendered):
        raise RuntimeError(f"could not parse colib prefix output:\n{run.stdout}\n{run.stderr}")
    return [[int(value) for value in row.split()] for row in rows]


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


def server_prefix(base_url: str, rendered: str, tokens: int) -> list[int]:
    payload = json.dumps(
        {
            "prompt": rendered,
            "n_predict": tokens,
            "temperature": 0,
            "samplers": ["temperature"],
            "return_tokens": True,
            "cache_prompt": False,
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
    return [int(value) for value in values[:tokens]]


def main() -> None:
    args = parse_args()
    snapshot, gguf, engine = args.snapshot.resolve(), args.gguf.resolve(), args.engine.resolve()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    prompts = json.loads(args.prompts.read_text())
    if args.limit:
        prompts = prompts[: args.limit]
    rendered_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        for prompt in prompts
    ]
    actual_prefixes = c_prefixes(
        engine,
        snapshot,
        rendered_prompts,
        args.tokens,
        args.threads,
        args.expert_ram,
        args.prefetch_threads,
        tokenizer,
    )
    server = None
    base_url = f"http://127.0.0.1:{args.server_port}"
    if args.llama_server:
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
    rows = []
    equal = total = 0
    try:
        for index, prompt in enumerate(prompts):
            rendered = rendered_prompts[index]
            actual = actual_prefixes[index]
            reference = (
                server_prefix(base_url, rendered, args.tokens)
                if server
                else llama_prefix(args.llama_cli, gguf, rendered, args.tokens, args.threads, tokenizer)
            )
            compared = min(len(actual), len(reference), args.tokens)
            matches = sum(actual[i] == reference[i] for i in range(compared))
            common_prefix = 0
            while common_prefix < compared and actual[common_prefix] == reference[common_prefix]:
                common_prefix += 1
            equal += matches
            total += compared
            rows.append(
                {
                    "index": index,
                    "matches": matches,
                    "compared": compared,
                    "agreement": matches / compared if compared else 0,
                    "common_prefix": common_prefix,
                }
            )
            print(f"prompt {index + 1}/{len(prompts)}: {matches}/{compared}, prefix={common_prefix}", flush=True)
    finally:
        if server:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    result = {
        "snapshot": str(snapshot),
        "gguf": str(gguf),
        "reference_runner": "llama-server" if args.llama_server else "llama-cli",
        "prompts": len(prompts),
        "matching_tokens": equal,
        "compared_tokens": total,
        "agreement": equal / total if total else 0,
        "prompts_at_85_percent": sum(row["agreement"] >= 0.85 for row in rows),
        "rows": rows,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
