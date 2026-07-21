#!/usr/bin/env python3
"""Tokenize a fixed corpus and run the C engine's teacher-forced perplexity path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from transformers import AutoTokenizer


PPL_RE = re.compile(r"\[PPL\] tokens=(\d+) nll=([0-9.eE+-]+) ppl=([0-9.eE+-]+) tok_s=([0-9.eE+-]+)")


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
    parser.add_argument("--prefetch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot, corpus, engine = args.snapshot.resolve(), args.corpus.resolve(), args.engine.resolve()
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
        env = os.environ.copy()
        env.update(
            {
                "SNAP": str(snapshot),
                "EVAL_IDS": str(ids_path),
                "EVAL_CHUNK": str(args.ctx_size),
                "CTX": str(args.ctx_size),
                "OMP_NUM_THREADS": str(args.threads),
                "EXPERT_RAM": str(args.expert_ram),
                "PREFETCH_THREADS": str(args.prefetch_threads),
            }
        )
        run = subprocess.run([str(engine)], env=env, text=True, capture_output=True, check=True)
    finally:
        ids_path.unlink(missing_ok=True)
    match = PPL_RE.search(run.stdout)
    if not match:
        raise SystemExit(f"could not parse engine result:\n{run.stdout}\n{run.stderr}")
    result: dict[str, object] = {
        "snapshot": str(snapshot),
        "corpus": str(corpus),
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "token_ids_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
        "tokens": int(match.group(1)),
        "source_tokens": len(ids),
        "ctx_size": args.ctx_size,
        "chunks": len(ids) // args.ctx_size,
        "nll": float(match.group(2)),
        "ppl": float(match.group(3)),
        "tok_s": float(match.group(4)),
    }
    if args.llama_model:
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
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
