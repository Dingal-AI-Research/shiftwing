#!/usr/bin/env python3
"""Compare a generated first-turn prefix with re-rendered Qwen chat history."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from openai_server import render_chat  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--user", default="Reply with exactly: colib ready")
    parser.add_argument("--assistant", default="colib ready")
    parser.add_argument("--next-user", default="Reply with exactly: colib ready")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    first_messages = [{"role": "user", "content": args.user}]
    continued_messages = [
        *first_messages,
        {"role": "assistant", "content": args.assistant},
        {"role": "user", "content": args.next_user},
    ]
    first = render_chat(first_messages, enable_thinking=False)
    continued = render_chat(continued_messages, enable_thinking=False)
    first_ids = tokenizer.encode(first, add_special_tokens=False)
    generated_ids = tokenizer.encode(args.assistant, add_special_tokens=False)
    eos = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    expected = first_ids + generated_ids + eos
    continued_ids = tokenizer.encode(continued, add_special_tokens=False)
    common = 0
    while (
        common < len(expected)
        and common < len(continued_ids)
        and expected[common] == continued_ids[common]
    ):
        common += 1
    print(
        json.dumps(
            {
                "first_text": first,
                "continued_text": continued,
                "first_tokens": len(first_ids),
                "generated_tokens": len(generated_ids),
                "end_tokens": eos,
                "expected_prefix_tokens": len(expected),
                "continued_tokens": len(continued_ids),
                "common_prefix_tokens": common,
                "expected_at_divergence": expected[common : common + 8],
                "continued_at_divergence": continued_ids[common : common + 8],
                "expected_text_at_divergence": tokenizer.decode(
                    expected[max(0, common - 8) : common + 8]
                ),
                "continued_text_at_divergence": tokenizer.decode(
                    continued_ids[max(0, common - 8) : common + 8]
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
