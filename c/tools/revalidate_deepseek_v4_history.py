#!/usr/bin/env python3
"""Independently compare historical converted groups using separate owned staging.

Never rewrite the checkpoint or the active recovery journal. The separate ledger
uses the same durable byte-comparison/release implementation as main recovery.
"""
import argparse
import json
from pathlib import Path

import convert_deepseek_v4 as converter
from recover_deepseek_v4 import Recovery, recovery_lock, json_sha
from deepseek_v4_spec import SOURCE_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, staging, output = (p.resolve() for p in (args.source, args.staging, args.output))
    if staging in (source, output):
        raise ValueError('historical comparison needs a separate staging directory')
    with recovery_lock(staging):
        for name in ('config.json', converter.INDEX_FILE, converter.SOURCE_MARKER, 'recovery-headers.json'):
            data = (source / name).read_bytes()
            target = staging / name
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError('historical staging metadata differs: ' + name)
            else:
                with target.open('xb') as handle:
                    handle.write(data)
        recovery = Recovery(staging, output)
        recovery.validation_path = output / 'historical-validation.json'
        recovery.validation = json.loads(recovery.validation_path.read_text()) if recovery.validation_path.exists() else {}
        groups, _ = converter.build_plan(staging, header_provider=recovery.header)
        state = json.loads((output / converter.STATE_FILE).read_text())
        historical = [f'dense/model-{i:05d}-of-00048.bin' for i in range(1, 4)]
        for group in historical:
            if group in recovery.validation:
                prior = recovery.validation[group]
                binding = {'revision': SOURCE_REVISION, 'segment': state['completed'][group],
                           'inventory_sha256': json_sha(state['inventory'][group])}
                if (prior.get('binding') != binding
                        or converter.sha256_file(output / group) != state['completed'][group]['sha256']):
                    raise ValueError('historical output changed: ' + group)
                recovery.release(groups[group], state['completed'][group], state['inventory'][group])
                continue
            print('[historical comparison] ' + group, flush=True)
            recovery.prepare(groups[group])
            recovery.release(groups[group], state['completed'][group], state['inventory'][group])
        print('[historical comparison complete] 3 groups independently compared; owned staging released', flush=True)


if __name__ == '__main__':
    main()
