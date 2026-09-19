"""Compare GPU prefill, recurrent chunk boundaries and cache eviction to CPU.

Requires a CUDA build and the converted tiny GLM fixture. This explicitly
asserts that CUDA ran; a silent CPU fallback cannot pass as GPU coverage.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--binary', type=Path, default=Path('./glm53'))
    ap.add_argument('--fixture', type=Path, default=Path('glm53_tiny_i4'))
    args = ap.parse_args()
    ids = ','.join(str((i*13+5)%90) for i in range(32))
    results = []

    def run(gpu, batch, cap, chunk, budget=512, device=0, bits=4):
        env = {**os.environ, 'OMP_NUM_THREADS': '4', 'GLM53_BITS': str(bits),
               'GLM53_CUDA': str(gpu), 'GLM53_BATCH_MATH': str(batch),
               'GLM53_CUDA_MB': str(budget), 'GLM53_PREFILL_CHUNK': str(chunk),
               'GLM53_CUDA_DEVICE': str(device)}
        p = subprocess.run([str(args.binary.resolve()), '--model', str(args.fixture.resolve()),
                            '--ids', ids, '--greedy', '6', '--logits', str(cap)],
                           env=env, text=True, capture_output=True, timeout=30, check=True)
        lines = {row.split()[0]: row.split()[1:] for row in p.stdout.splitlines() if row.strip()}
        calls = re.search(r'CUDA stats: calls=(\d+) batched=(\d+)', p.stderr)
        if gpu and chunk > 1 and device == 0:
            assert calls and int(calls[2]) > 0, p.stderr
        results.append({'gpu': gpu, 'batch_math': batch, 'cap': cap, 'chunk': chunk,
                        'budget_mb': budget, 'device': device, 'bits': bits, 'fallbacks': int(re.search(r'fallbacks=(\d+)', p.stderr)[1]) if calls else 0, 'calls': int(calls[1]) if calls else 0})
        return lines

    baseline = run(0, 0, 1, 128)
    for cap in (1, 2, 3):
        for chunk in (1, 2, 7, 128):
            got = run(1, 1, cap, chunk)
            for key in ('teacher_forcing', 'greedy'):
                assert got[key] == baseline[key], (cap, chunk, key, got[key], baseline[key])
            worst = max(abs(float(a)-float(b)) for a,b in zip(got['last_logits'],baseline['last_logits'],strict=True))
            assert worst < 5e-5, (cap, chunk, worst)
            results[-1]['max_logit_error'] = worst
    # The batched CPU fallback is useful when no device is available.
    got = run(0, 1, 1, 7)
    assert got['greedy'] == baseline['greedy'] and got['teacher_forcing'] == baseline['teacher_forcing']
    for budget, device in ((1, 0), (512, 9999)):
        # FP32 fixture weights exceed the 1 MiB budget and force partial fallback.
        bits = 32 if budget == 1 else 4
        ref = run(0, 0, 1, 7, bits=bits) if bits != 4 else baseline
        got = run(1, 1, 1, 7, budget, device, bits)
        assert got['greedy'] == ref['greedy'] and got['teacher_forcing'] == ref['teacher_forcing']
        worst = max(abs(float(a)-float(b)) for a,b in zip(got['last_logits'],ref['last_logits'],strict=True))
        assert worst < 5e-5, (budget, device, worst)
        if device == 0: assert results[-1]['fallbacks'] > 0
        else: assert results[-1]['calls'] == 0
    print(json.dumps({'status': 'passed', 'comparisons': results}, indent=2))


if __name__ == '__main__':
    main()
