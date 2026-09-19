"""Identity-bound acceptance checks for a complete native DeepSeek reviewer run.

This module never infers success from checkpoint conversion or fixture results.
A qualification report is created only by the standalone acceptance runner after
all measurements exist. Production startup/doctor integration is still pending.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from .deepseek_v4_spec import SOURCE_REVISION, TOKENIZER_SHA256

SCHEMA = 'colib.deepseek-v4.review-qualification.v1'
STAGES = {'conversion', 'tokenizer', 'protocol', 'released_reference', 'io', 'prefill', 'reviews', 'memory'}
PREFILL_TOKENS = [128, 512, 2048, 8192, 32768]
CHUNKS = [256, 512, 1024, 2048]
GIB = 1024**3


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def evidence_path(snapshot: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.startswith('qualification/') or '\\' in value:
        raise ValueError('qualification evidence must be beneath qualification/')
    relative = Path(value)
    if relative.is_absolute() or any(part in ('..', '.') for part in relative.parts):
        raise ValueError('unsafe qualification evidence path')
    base = snapshot.resolve() / 'qualification'
    target = snapshot / relative
    if base.is_symlink() or not target.resolve().is_relative_to(base):
        raise ValueError('qualification evidence escapes its directory')
    if not target.is_file() or target.stat().st_size > 8 * 1024**2:
        raise ValueError('qualification evidence is missing or exceeds 8 MiB')
    return target


def read_report(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise ValueError('qualification JSON is missing or exceeds 8 MiB')
    value = json.loads(path.read_text())
    if not isinstance(value, dict): raise ValueError('qualification JSON must be an object')
    return value


def exact_int_sequence(value: object, expected: list[int]) -> bool:
    return (isinstance(value, list) and len(value) == len(expected) and
            all(type(actual) is int and actual == wanted
                for actual, wanted in zip(value, expected, strict=True)))


def finite_positive(value: object, *, maximum: float | None = None) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(value) and value > 0 and
            (maximum is None or value < maximum))


def validate_prefill_evidence(proof: dict, required: dict) -> None:
    if type(proof.get('runtime_layers')) is not int or proof['runtime_layers'] != 43:
        raise ValueError('prefill evidence must execute all 43 runtime layers')
    results = proof.get('results')
    expected = {(1024, tokens) for tokens in PREFILL_TOKENS}
    expected.update((size, tokens) for size in CHUNKS if size != 1024
                    for tokens in (2048, 32768))
    if not isinstance(results, list) or len(results) != len(expected):
        raise ValueError('prefill evidence does not contain the complete measurement schedule')
    measured = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError('prefill measurement must be an object')
        size, tokens = result.get('chunk'), result.get('prompt_tokens')
        pair = (size, tokens)
        stats, request = result.get('stats'), result.get('request')
        if (type(size) is not int or type(tokens) is not int or pair not in expected or
                pair in measured or result.get('status') != 'passed' or
                not finite_positive(result.get('warm_seconds'), maximum=1200) or
                not finite_positive(result.get('prefill_seconds'), maximum=1200) or
                type(result.get('read_bytes')) is not int or result['read_bytes'] < 0 or
                not isinstance(result.get('prompt_sha256'), str) or
                not isinstance(stats, dict) or stats.get('prompt_tokens') != tokens or
                stats.get('completion_tokens') != 1 or not isinstance(request, dict) or
                request.get('payload_sha256') != result['prompt_sha256'] or
                request.get('binary_sha256') != required['engine_sha256'] or
                request.get('manifest_sha256') != required['manifest_sha256'] or
                request.get('termination') not in ('length', 'eos')):
            raise ValueError('prefill measurement is incomplete or is not bound to native evidence')
        measured.add(pair)
    if measured != expected:
        raise ValueError('prefill evidence does not cover every required token/chunk pair')


def validate_report(report: dict, snapshot: Path, engine: Path, *, chunk: int = 1024) -> None:
    if not isinstance(report, dict): raise ValueError('qualification report must be an object')
    required = {'schema': SCHEMA, 'status': 'passed', 'source_revision': SOURCE_REVISION,
                'input_tokens': 32768, 'output_tokens': 8192, 'context_tokens': 40960,
                'reasoning_effort': 'low', 'dspark': 'off', 'slots': 1,
                'ram_cache_bytes': 8*GIB, 'device_cache_bytes': 2*GIB, 'gpu_headroom_bytes': 3*GIB//2,
                'manifest_sha256': sha256(snapshot / 'model-manifest.json'),
                'engine_sha256': sha256(engine), 'tokenizer_sha256': TOKENIZER_SHA256}
    if any(type(report.get(key)) is not type(value) or report.get(key) != value for key, value in required.items()):
        raise ValueError('qualification does not bind the current model, engine and reviewer profile')
    if sha256(snapshot / 'tokenizer.json') != TOKENIZER_SHA256:
        raise ValueError('pinned tokenizer bytes changed')
    if (not exact_int_sequence(report.get('prefill_tokens'), PREFILL_TOKENS) or
            not exact_int_sequence(report.get('chunk_sizes'), CHUNKS) or
            type(chunk) is not int or chunk not in CHUNKS):
        raise ValueError('populated prefill and chunk comparison gates are incomplete')
    evidence = report.get('evidence')
    if not isinstance(evidence, list) or len(evidence) != len(STAGES) or any(not isinstance(e, dict) or not isinstance(e.get('stage'), str) for e in evidence) or {e.get('stage') for e in evidence} != STAGES:
        raise ValueError('required qualification evidence is missing')
    proofs = {}
    for item in evidence:
        path = evidence_path(snapshot, item.get('file'))
        if item.get('sha256') != sha256(path):
            raise ValueError('qualification evidence checksum changed: ' + str(path))
        proof = read_report(path)
        if (proof.get('status') != 'passed' or proof.get('stage') != item['stage'] or
                proof.get('source_revision') != SOURCE_REVISION):
            raise ValueError('qualification evidence did not pass: ' + item['stage'])
        if proof.get('engine_sha256') != required['engine_sha256'] or proof.get('manifest_sha256') != required['manifest_sha256']:
            raise ValueError('qualification evidence came from another engine or checkpoint')
        proofs[item['stage']] = proof
    reference = proofs['released_reference']
    if (type(reference.get('runtime_layers')) is not int or reference['runtime_layers'] != 43 or
            reference.get('logits_compared') is not True):
        raise ValueError('released reference evidence must compare complete 43-layer logits')
    validate_prefill_evidence(proofs['prefill'], required)
    cases = report.get('reviews')
    if not isinstance(cases, list) or len(cases) != 5 or any(not isinstance(c, dict) or not isinstance(c.get('id'), str) or not c['id'].strip() for c in cases) or len({c['id'] for c in cases}) != 5:
        raise ValueError('five independent planted-defect review cases are required')
    for case in cases:
        seconds = case.get('warm_seconds')
        if (case.get('complete') is not True or case.get('defect_found') is not True or case.get('termination') != 'eos'
                or type(case.get('prompt_tokens')) is not int or not 1 <= case['prompt_tokens'] <= 32768
                or type(case.get('output_tokens')) is not int or not 1 <= case['output_tokens'] < 8192
                or isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
                or not 0 < seconds <= 1200):
            raise ValueError('a planted-defect review is incomplete, incorrect or over the warm time limit')
    if not any(c['prompt_tokens'] == 32768 for c in cases):
        raise ValueError('an actual 32768-token review is required')
    if proofs['reviews'].get('reviews') != cases:
        raise ValueError('aggregate reviews differ from the checksummed review evidence')
    memory = report.get('memory', {})
    if not isinstance(memory, dict): raise ValueError('memory measurements must be an object')
    fields = ('device_peak_bytes', 'device_total_bytes', 'scratch_peak_bytes', 'host_peak_bytes', 'host_available_bytes')
    if any(type(memory.get(k)) is not int or memory[k] <= 0 for k in fields):
        raise ValueError('actual peak allocation measurements are missing')
    if (memory['device_peak_bytes'] + 3*GIB//2 > memory['device_total_bytes'] or memory['scratch_peak_bytes'] > 2*GIB
            or memory['host_peak_bytes'] > memory['host_available_bytes']):
        raise ValueError('measured allocations violate reviewer memory limits')
    if proofs['memory'].get('memory') != memory:
        raise ValueError('aggregate memory differs from the checksummed allocation evidence')


def load_qualification(snapshot: Path, engine: Path, *, chunk: int = 1024) -> dict:
    report = read_report(snapshot / 'review-qualification.json')
    validate_report(report, snapshot, engine, chunk=chunk)
    return report
