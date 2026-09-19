#!/usr/bin/env python3
"""Compare native token IDs to the pinned released tokenizer, without weights."""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

from deepseek_v4_protocol import render_deepseek_chat
from deepseek_v4_spec import SOURCE_REVISION, TOKENIZER_SHA256


def cases(metadata):
    rng = random.Random(40960)
    words = ['Hello', 'world', 'return', 'None', 'def', 'if', 'while', '你好', 'ありがとう',
             'カタカナ', 'e\u0301', 'café', 'русский', 'مرحبا', '1234567890', '1.0', '0xFF',
             "don't", "I'm", 'foo_bar', 'HTTPRequest', 'True', 'false']
    separators = [' ', '  ', '\n', '\t', '\r\n', '.', ':', ';', '_', '!', "'", '?', '\u00a0', '', '\u2003', '👩\u200d💻']
    result = [''.join(rng.choice(words) + rng.choice(separators) for _ in range(rng.randrange(1, 20))) for _ in range(2000)]
    result.extend(t['content'] for t in metadata['added_tokens'])
    for effort in (None, 'low', 'high', 'max'):
        result.append(render_deepseek_chat([{'role': 'system', 'content': 'Review the exact diff.'},
                                           {'role': 'user', 'content': 'Check this change:\n- return 1\n+ return 0'}], reasoning_effort=effort))
    # Boundary/control cases cannot be represented by the old line/tab harness.
    result += ['', '\x00', 'x\x00y', '  1234', "'return", '你好Hello', 'a\ufe0fx',
               'a\U000e0100x', '\u200dA', '\u2028', '\u2029', '\u0085A', '\r\n  1']
    for cp in range(0x110000):
        # Cover all Unicode categories and block boundaries with deterministic samples.
        if cp % 257 == 0 and not 0xd800 <= cp <= 0xdfff:
            result.append('a' + chr(cp) + '  123你好' + chr(cp) + "'return")
    return result


def verify(tokenizer_path, binary):
    from tokenizers import Tokenizer
    payload = tokenizer_path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != TOKENIZER_SHA256:
        raise ValueError('tokenizer is not the pinned released file')
    oracle = Tokenizer.from_file(str(tokenizer_path))
    texts = cases(json.loads(payload))
    # Exact fully rendered long input, not a nominal word/character estimate.
    messages = [{'role': 'system', 'content': 'Review every changed line; preserve the exact selected evidence.'},
                {'role': 'user', 'content': 'Change:\n- return 1\n+ return 0\n' + (' evidence' * 32700)}]
    prompt = render_deepseek_chat(messages, reasoning_effort='low')
    # The repeated word is one independent token; adjust against the oracle and verify.
    count = len(oracle.encode(prompt, add_special_tokens=False).ids)
    messages[-1]['content'] += ' evidence' * max(0, 32768 - count)
    prompt = render_deepseek_chat(messages, reasoning_effort='low')
    if len(oracle.encode(prompt, add_special_tokens=False).ids) != 32768:
        raise ValueError('long-prompt construction did not reach 32768 actual tokens')
    texts.extend([prompt, prompt + ' evidence'])
    frames = b''.join(f'ENCODE {len(raw)}\n'.encode() + raw + b'\n' for raw in (t.encode() for t in texts))
    completed = subprocess.run([str(binary.resolve()), str(tokenizer_path.resolve())],
                               input=frames, capture_output=True, timeout=120)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode() + completed.stdout.decode()[:1000])
    rows = [json.loads(line) for line in completed.stdout.splitlines()]
    if len(rows) != len(texts):
        raise ValueError('native helper omitted responses')
    failures = []
    roundtrip_failures = 0
    for index, (text, row) in enumerate(zip(texts, rows, strict=True)):
        expected = oracle.encode(text, add_special_tokens=False).ids
        if row.get('ids') != expected or row.get('tokens') != len(expected):
            if len(failures) < 20:
                failures.append({'case': index, 'text': text[:200], 'expected': expected[:100], 'actual': row.get('ids', [])[:100]})
        if bytes.fromhex(row['decoded_hex']) != text.encode():
            roundtrip_failures += 1
    matches = sum(row.get('ids') == oracle.encode(text, add_special_tokens=False).ids for text, row in zip(texts, rows, strict=True))
    return {'revision': SOURCE_REVISION, 'tokenizer_sha256': actual,
            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'reference': 'tokenizers.Tokenizer.from_file; add_special_tokens=False',
            'cases': len(texts), 'seed': 40960, 'matching_ids': matches,
            'roundtrip_failures': roundtrip_failures, 'long_prompt_tokens': rows[-2]['tokens'],
            'over_limit_prompt_tokens': rows[-1]['tokens'], 'first_failures': failures,
            'passed': matches == len(texts) and roundtrip_failures == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path(__file__).resolve().parents[1] / 'deepseek_v4_tokenize')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.tokenizer, args.binary)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
