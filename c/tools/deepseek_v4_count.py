"""Exact pinned DeepSeek counts via a CPU-only helper, without model weights."""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

from .deepseek_v4_protocol import ENCODING_SHA256
from .deepseek_v4_spec import (
    REVIEW_CONTEXT, REVIEW_INPUT_TOKENS, REVIEW_OUTPUT_TOKENS,
    SOURCE_REVISION, TOKENIZER_SHA256,
)

MAX_PROMPT_BYTES = 16 * 1024**2


class DeepSeekTokenCounter:
    def __init__(self, snapshot: Path, binary: Path | None = None):
        self.tokenizer = Path(snapshot).resolve() / 'tokenizer.json'
        self.binary = (binary or Path(__file__).resolve().parents[1] / 'deepseek_v4_tokenize').resolve()
        self.lock = threading.Lock()

    def validate_identity(self):
        actual = hashlib.sha256(self.tokenizer.read_bytes()).hexdigest()
        if actual != TOKENIZER_SHA256:
            raise ValueError('DeepSeek tokenizer does not match the pinned checkpoint')
        if not self.binary.is_file():
            raise RuntimeError('Build the CPU-only helper with make -C c deepseek_v4_tokenize')
        return hashlib.sha256(self.binary.read_bytes()).hexdigest()

    def count(self, prompt: str, *, reasoning_effort: str | None = None):
        payload = prompt.encode('utf-8')
        if len(payload) > MAX_PROMPT_BYTES:
            raise ValueError(f'rendered prompt exceeds {MAX_PROMPT_BYTES} bytes')
        started = time.monotonic()
        with self.lock:
            binary_sha = self.validate_identity()
            completed = subprocess.run(
                [str(self.binary), str(self.tokenizer)],
                input=f'COUNT {len(payload)}\n'.encode() + payload + b'\n',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
            )
        if completed.returncode:
            raise RuntimeError('native DeepSeek token counting failed')
        try:
            result = json.loads(completed.stdout)
            count = result['tokens']
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError('native tokenizer returned an invalid count response') from exc
        if type(count) is not int or count < 0 or count > len(payload) or (payload and count == 0):
            raise RuntimeError('native tokenizer returned an invalid token count')
        return {
            'object': 'shiftwing.prompt_token_count',
            'prompt': prompt,
            'prompt_sha256': hashlib.sha256(payload).hexdigest(),
            'prompt_bytes': len(payload),
            'prompt_tokens': count,
            'input_limit': REVIEW_INPUT_TOKENS,
            'output_budget': REVIEW_OUTPUT_TOKENS,
            'review_context': REVIEW_CONTEXT,
            'fits': count <= REVIEW_INPUT_TOKENS,
            'reasoning_effort': reasoning_effort,
            'source_revision': SOURCE_REVISION,
            'tokenizer_sha256': TOKENIZER_SHA256,
            'encoding_sha256': ENCODING_SHA256,
            'tokenizer_binary_sha256': binary_sha,
            'count_time_ms': (time.monotonic() - started) * 1000,
        }


def main():
    """One CPU-only JSON count. Shares API rendering without binding a port."""
    import argparse
    import sys
    from types import SimpleNamespace
    from openai_server import APIHandler
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    body = json.loads(sys.stdin.buffer.read(MAX_PROMPT_BYTES + 1))
    if not isinstance(body, dict):
        raise ValueError("count input must be a chat request object")
    handler = SimpleNamespace(server=SimpleNamespace(model_family="deepseek-v4", model_path=args.snapshot))
    prompt, _, _, _, effort = APIHandler.chat_prompt(handler, body)
    result = DeepSeekTokenCounter(args.snapshot, args.binary).count(prompt, reasoning_effort=effort)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
