#!/usr/bin/env python3
"""Validate a complete checkpoint against retained independent source-byte proofs.

Released staging is not downloaded again. The source plan is rebuilt from its
pinned index and cached shard headers; current output bytes must match the exact
segment and inventory previously compared independently with the source.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import convert_deepseek_v4 as converter
import validate_deepseek_v4_conversion as original
from deepseek_v4_spec import SOURCE_REPO, SOURCE_REVISION


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_file(root: Path, name: str):
    if not isinstance(name, str) or not name or name.startswith('/') or '\\' in name or any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError('unsafe checkpoint path')
    path = root / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('invalid checkpoint file: ' + name)
    return path


def validate(source: Path, model: Path, *, allow_fixture=False):
    source, model = source.resolve(), model.resolve()
    state_path = model / converter.STATE_FILE
    manifest_path = model / converter.MANIFEST_FILE
    manifest = original.read_json(manifest_path); state = original.read_json(state_path)
    manifest_hash = original.sha256_file(manifest_path); state_hash = original.sha256_file(state_path)
    if (manifest.get('schema') != converter.MANIFEST_SCHEMA or state.get('schema') != converter.STATE_SCHEMA
            or state.get('status') != 'complete' or state.get('manifest') != {'file': converter.MANIFEST_FILE, 'sha256': manifest_hash}):
        raise ValueError('complete manifest-bound conversion is required')
    if manifest.get('segments') != state.get('completed') or manifest.get('inventory') != state.get('inventory'):
        raise ValueError('manifest and conversion ledger differ')
    if not allow_fixture and (manifest.get('source', {}).get('repository') != SOURCE_REPO or manifest.get('source', {}).get('revision') != SOURCE_REVISION):
        raise ValueError('checkpoint is not the pinned source')
    headers_path = source / 'recovery-headers.json'; headers = original.read_json(headers_path)
    if headers.get('revision') != SOURCE_REVISION: raise ValueError('cached headers belong to another source')
    def header(name):
        converter._validate_relative_file(name)
        item = headers['shards'][name]
        return item['start'], item['header']
    groups, plan = converter.build_plan(source, allow_fixture=allow_fixture, header_provider=header)
    for key in ('plan_sha256', 'source_index_sha256', 'source_tensor_count', 'output_record_count'):
        if manifest.get(key) != plan.get(key): raise ValueError('source plan mismatch: ' + key)
    if state.get('plan_sha256') != plan['plan_sha256']: raise ValueError('ledger plan mismatch')
    if set(manifest['segments']) != set(groups) or set(manifest['inventory']) != set(groups):
        raise ValueError('checkpoint group coverage differs from the source plan')
    if not allow_fixture and len(groups) != 91: raise ValueError('all 91 native groups are required')
    proof_files = [model / name for name in ('recovery-validation.json', 'historical-validation.json')]
    proof_hashes = {path.name: original.sha256_file(path) for path in proof_files}
    proofs = {}
    for path in proof_files:
        for group, proof in original.read_json(path).items():
            if group in proofs and proofs[group] != proof: raise ValueError('conflicting source proofs: ' + group)
            proofs[group] = proof
    evidence=[]; total=0
    for group, expected in sorted(groups.items()):
        segment = manifest['segments'][group]; inventory = manifest['inventory'][group]
        proof = proofs.get(group, {}); binding = proof.get('binding', {}); measured = proof.get('validation', {})
        if binding != {'revision': SOURCE_REVISION, 'segment': segment, 'inventory_sha256': digest_json(inventory)}:
            raise ValueError('independent source proof does not bind current inventory: ' + group)
        if len(expected) != len(inventory) or segment.get('file') != group or segment.get('records') != len(inventory):
            raise ValueError('record count or segment descriptor mismatch: ' + group)
        position=0; payload=0
        for record, item in zip(expected, inventory, strict=True):
            if any(item.get(k) != v for k, v in original._metadata_for_record(record).items()):
                raise ValueError('native source metadata mismatch: ' + record.name)
            offset=item.get('offset'); size=item.get('nbytes')
            if type(offset) is not int or offset < position or type(size) is not int or size < 0:
                raise ValueError('invalid native record extent: ' + record.name)
            position=offset+size; payload+=size
        if (position != segment['size'] or measured.get('file') != group or measured.get('size') != segment['size']
                or measured.get('sha256') != segment['sha256'] or measured.get('records') != len(inventory)
                or measured.get('payload_bytes') != payload or measured.get('source_bytes_compared') != payload
                or set(measured.get('source_identities', {})) != {r.source_shard for r in expected}):
            raise ValueError('independent source comparison is incomplete: ' + group)
        path=safe_file(model, group); before=original.stat_identity(path)
        if before['size'] != segment['size'] or original.sha256_file(path) != segment['sha256']:
            raise ValueError('current native segment differs from source-verified bytes: ' + group)
        if before != original.stat_identity(path): raise ValueError('segment changed during validation: ' + group)
        total+=payload; evidence.append({'file': group, 'sha256': segment['sha256'], 'source_bytes_compared': payload, 'current_identity': before})
        print(f'[verified {len(evidence)}/{len(groups)}] {group}', flush=True)
    metadata = manifest.get('metadata', {})
    if not metadata: raise ValueError('checkpoint metadata hashes are missing')
    for name, entry in metadata.items():
        path=safe_file(model, name)
        if path.stat().st_size != entry['size'] or original.sha256_file(path) != entry['sha256']:
            raise ValueError('checkpoint metadata changed: ' + name)
    if (original.sha256_file(manifest_path) != manifest_hash or original.sha256_file(state_path) != state_hash
            or any(original.sha256_file(path) != proof_hashes[path.name] for path in proof_files)):
        raise ValueError('checkpoint or proof ledger changed during validation')
    return {'schema': 'colib.deepseek-v4.recovered-conversion-validation.v1', 'stage': 'conversion', 'status': 'passed',
            'source_revision': SOURCE_REVISION, 'manifest_sha256': manifest_hash, 'conversion_state_sha256': state_hash,
            'header_cache_sha256': original.sha256_file(headers_path), 'proof_sha256': proof_hashes,
            'groups': evidence, 'source_bytes_compared': total, 'completed_at': original.utc_now(),
            'fixture': allow_fixture, 'scope': 'native byte integrity; runtime and review qualification remain separate'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'model', 'output'): p.add_argument('--'+key, type=Path, required=True)
    args=p.parse_args(); report=validate(args.source, args.model)
    original.atomic_json(args.output, report)
    print(json.dumps({'status': report['status'], 'groups': len(report['groups']), 'source_bytes_compared': report['source_bytes_compared']}))
if __name__=='__main__': main()
