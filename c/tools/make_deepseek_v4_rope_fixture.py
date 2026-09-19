#!/usr/bin/env python3
"""Generate long-position RoPE outputs with the unchanged pinned model.py."""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from make_deepseek_v4_fixture import REFERENCE_HASHES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.source / 'inference/model.py'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REFERENCE_HASHES['inference/model.py']:
        raise ValueError('pinned model identity changed')
    stub = types.ModuleType('kernel')
    for name in ('act_quant', 'fp4_act_quant', 'hc_split_sinkhorn',
                 'sparse_attn', 'fp8_gemm', 'fp4_gemm'):
        setattr(stub, name, None)
    sys.modules['kernel'] = stub
    spec = importlib.util.spec_from_file_location('pinned_rope_reference', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    positions = [0, 1, 63, 127, 128, 511, 2047, 8191,
                 16383, 32767, 32768, 40959, 49151, 60001, 65534, 65535]
    inputs = torch.tensor([((i * 37) % 97 - 48) / 16
                           for i in range(len(positions) * 64)],
                          dtype=torch.bfloat16).reshape(1, len(positions), 64)
    cases = []
    for original, base, factor in ((0, 10000., 1.), (65536, 160000., 16.)):
        frequencies = module.precompute_freqs_cis(
            64, 65536, original, base, factor, 32, 1)[positions]
        for inverse in (False, True):
            expected = module.apply_rotary_emb(
                inputs.clone(), frequencies, inverse).float().flatten().tolist()
            cases.append(dict(original=original, base=base, factor=factor,
                              inverse=int(inverse), output=expected))
    args.output.write_text(json.dumps(dict(reference_model_sha256=digest,
                                           positions=positions, cases=cases),
                                     indent=2) + '\n')


if __name__ == '__main__':
    main()
