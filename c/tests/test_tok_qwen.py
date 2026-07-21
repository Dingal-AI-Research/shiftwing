#!/usr/bin/env python3
"""10k-case Qwen3.5 tokenizer parity gate for the header-only C tokenizer."""

from __future__ import annotations

import argparse
import random
import re
import subprocess
import tempfile
from pathlib import Path

from transformers import AutoTokenizer


CASE_COUNT = 10_000
SEED = 20260720
ROOT = Path(__file__).resolve().parents[1]


ASCII = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PUNCT = "!@#$%^&*()[]{}<>?/\\|;:'\"`~+-_=,."
CJK = "天地玄黃宇宙洪荒日月盈昃辰宿列張學習模型推理人工智能你好世界"
EMOJI = "😀😅🥲🚀🧠🦜🔥✨❤️👍🏽👩🏽‍💻🏳️‍🌈"
CYRILLIC = "Приветмирмодельданные"
ARABIC = "مرحباالعالمبياناتنموذج"


def escape_case(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def fixed_cases(specials: list[str]) -> list[str]:
    return [
        "",
        "Hello, world!",
        " hello",
        "hello ",
        "hello\nworld",
        "a\tb\rc",
        "    ",
        "\n\n\n",
        "if (x <= 10) { printf(\"%d\\n\", x); }",
        "def f(x: list[int]) -> int:\n    return sum(x)",
        "你好，世界！",
        "こんにちは世界",
        "안녕하세요 세계",
        "مرحبا بالعالم",
        "Привет, мир!",
        "😀😅🥲🚀🧠",
        "👩🏽‍💻 + 🏳️‍🌈 + ❤️",
        "e\u0301 café naïve Straße",
        "\\n is two characters; \n is one newline",
        "<tool_call>{\"name\":\"read_file\",\"arguments\":{}}</tool_call>",
    ] + specials


def random_case(rng: random.Random, index: int, specials: list[str]) -> str:
    family = index % 10
    length = rng.randint(1, 180)
    if family == 0:
        alphabet = ASCII + PUNCT + " " * 8
    elif family == 1:
        alphabet = ASCII + " \t\n\r" * 5
    elif family == 2:
        alphabet = CJK + "，。！？、；：“”" + ASCII
    elif family == 3:
        alphabet = EMOJI + " " + ASCII
    elif family == 4:
        alphabet = CYRILLIC + " ,.!?" + ASCII
    elif family == 5:
        alphabet = ARABIC + " ،.!؟" + ASCII
    elif family == 6:
        words = ["const", "char", "return", "NULL", "size_t", "torch", "model", "router"]
        return " ".join(rng.choice(words) for _ in range(rng.randint(2, 30))) + rng.choice([";", " {}", "\n"])
    elif family == 7:
        runs = [" " * rng.randint(1, 40), "\t" * rng.randint(1, 8), "\n" * rng.randint(1, 8)]
        return rng.choice(runs) + rng.choice(ASCII) + rng.choice(runs)
    elif family == 8 and specials:
        return rng.choice(ASCII) + rng.choice(specials) + rng.choice(ASCII)
    else:
        alphabet = ASCII + CJK + EMOJI + CYRILLIC + ARABIC + PUNCT + " \n\t"
    return "".join(rng.choice(alphabet) for _ in range(length))


def make_cases(tokenizer) -> list[str]:
    specials = list(dict.fromkeys(tokenizer.all_special_tokens))
    cases = fixed_cases(specials)
    rng = random.Random(SEED)
    while len(cases) < CASE_COUNT:
        cases.append(random_case(rng, len(cases), specials))
    return cases[:CASE_COUNT]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--binary", type=Path, default=ROOT / "tests" / "test_tok")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.repo, use_fast=True)
    if not args.binary.is_file():
        raise SystemExit(f"missing {args.binary}; run `make tests/test_tok` from c/")
    lines: list[str] = []
    for text in make_cases(tokenizer):
        ids = tokenizer.encode(text, add_special_tokens=False)
        lines.append(f"{escape_case(text)}\t{','.join(map(str, ids))}\n")
    with tempfile.TemporaryDirectory(prefix="qwen-tokenizer-") as tempdir:
        tokenizer_json = args.tokenizer_json or Path(tempdir) / "tokenizer.json"
        if args.tokenizer_json is None:
            # AutoTokenizer augments tokenizer.json with multimodal special tokens
            # declared by tokenizer_config.json; test the effective tokenizer.
            tokenizer.backend_tokenizer.save(str(tokenizer_json))
        result = subprocess.run(
            [str(args.binary), str(tokenizer_json)],
            input="".join(lines),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    print(result.stdout, end="")
    if result.returncode or not re.search(rf"ENCODE: {CASE_COUNT}/{CASE_COUNT}\b", result.stdout):
        detail = result.stderr.split("MISMATCH", 6)
        mismatch = "MISMATCH" + "MISMATCH".join(detail[1:]) if len(detail) > 1 else result.stderr
        mismatch = mismatch[:8000]
        raise SystemExit(f"tokenizer parity failed (exit {result.returncode})\n{mismatch}")
    print(f"Qwen3.5 tokenizer parity: {CASE_COUNT}/{CASE_COUNT} exact (seed {SEED})")


if __name__ == "__main__":
    main()
