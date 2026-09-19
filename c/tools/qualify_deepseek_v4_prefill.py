#!/usr/bin/env python3
"""Measure populated prefill using the complete native serving executable.

Runs standalone, with one output token per diagnostic request. These requests
cannot qualify as complete reviews. No live service or startup gate is changed.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

from .deepseek_v4_count import DeepSeekTokenCounter
from .deepseek_v4_protocol import render_deepseek_chat
from .deepseek_v4_qualification import CHUNKS, PREFILL_TOKENS, GIB
from .deepseek_v4_spec import SOURCE_REVISION
from .qualify_deepseek_v4_reviews import digest, write_json


def records(path: Path, prefix: str):
    result = []
    with path.open(encoding='utf-8', errors='strict') as handle:
        for raw_line in handle:
            line = raw_line.rstrip('\n')
            if not line.startswith(prefix + ' '):
                continue
            fields = {}
            for field in line.split()[1:]:
                if '=' not in field:
                    raise ValueError(f'malformed {prefix} record field')
                key, value = field.split('=', 1)
                if not key or not value or key in fields:
                    raise ValueError(f'malformed {prefix} record field')
                fields[key] = value
            if not fields:
                raise ValueError(f'empty {prefix} record')
            result.append(fields)
    return result


def _decimal(record, key, *, positive=False):
    if not isinstance(record, dict):
        raise ValueError('native measurement record must be an object')
    value = record.get(key)
    if (not isinstance(value, str) or not value or
            any(character < '0' or character > '9' for character in value)):
        raise ValueError(f'native measurement field {key} is not an unsigned decimal integer')
    number = int(value)
    if positive and number <= 0:
        raise ValueError(f'native measurement field {key} must be positive')
    return number


def _positive_float(record, key):
    if not isinstance(record, dict) or not isinstance(record.get(key), str):
        raise ValueError(f'native measurement field {key} is missing')
    try:
        value = float(record[key])
    except ValueError as error:
        raise ValueError(f'native measurement field {key} is not numeric') from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'native measurement field {key} must be finite and positive')
    return value


def measured_memory(plan, allocations, requests):
    if (not isinstance(plan, dict) or not isinstance(allocations, (list, tuple)) or
            not isinstance(requests, (list, tuple)) or not allocations or not requests):
        raise ValueError('actual native allocation/request records are missing')
    total = _decimal(allocations[0], 'device_total', positive=True)
    available = _decimal(plan, 'device_available', positive=True)
    host_available = _decimal(plan, 'host_available', positive=True)
    if available > total:
        raise ValueError('planned available device memory exceeds device capacity')
    baseline = total - available
    used_values = [_decimal(item, 'pool_used_peak', positive=True) for item in allocations]
    reserved_values = [_decimal(item, 'pool_reserved_peak', positive=True) for item in allocations]
    free_values = [_decimal(item, 'device_free') for item in allocations]
    scratch_values = [_decimal(item, 'scratch_allocated', positive=True) for item in allocations]
    if any(_decimal(item, 'device_total', positive=True) != total for item in allocations):
        raise ValueError('device identity or capacity changed during the measurement')
    if any(used > reserved for used, reserved in zip(used_values, reserved_values, strict=True)):
        raise ValueError('CUDA allocator peak measurements are inconsistent')
    if any(free > total for free in free_values):
        raise ValueError('reported free device memory exceeds device capacity')
    host_peaks = [math.ceil(_positive_float(item, 'peak_rss_gb') * GIB) for item in requests]
    reserved = max(reserved_values)
    peak = max(baseline + reserved, *(total - free for free in free_values))
    result = {'device_peak_bytes': peak, 'device_total_bytes': total,
              'scratch_peak_bytes': max(scratch_values),
              'host_peak_bytes': max(host_peaks),
              'host_available_bytes': host_available}
    if (peak + 3 * GIB // 2 > total or result['scratch_peak_bytes'] > 2 * GIB
            or result['host_peak_bytes'] > result['host_available_bytes']):
        raise ValueError('measured native allocations exceed reviewer limits')
    return result


def prefill_schedule():
    return [(1024, tuple(PREFILL_TOKENS))] + [
        (chunk, (2048, 32768)) for chunk in CHUNKS if chunk != 1024]


def prepare_prompts(snapshot, directory):
    from tokenizers import Tokenizer
    directory.mkdir(parents=True, exist_ok=True)
    counter = DeepSeekTokenCounter(snapshot)
    counter.validate_identity()
    tokenizer = Tokenizer.from_file(str(snapshot / 'tokenizer.json'))
    results = []
    for target in PREFILL_TOKENS:
        def render(padding):
            return render_deepseek_chat([
                {'role': 'system', 'content': 'Review the selected source for correctness.'},
                {'role': 'user', 'content': 'Repository context:\n#' + ' evidence' * padding +
                 '\nSelected source:\ndef lookup(items, index):\n    if index <= len(items):\n        return items[index]\n'}],
                reasoning_effort='low')
        initial = len(tokenizer.encode(render(0), add_special_tokens=False).ids)
        padding = target - initial
        for _ in range(4):
            prompt = render(padding)
            difference = target - len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            if not difference:
                break
            padding += difference
        receipt = counter.count(prompt, reasoning_effort='low')
        if receipt['prompt_tokens'] != target:
            raise ValueError('exact populated prefill length did not match')
        (directory / f'{target}.txt').write_text(prompt)
        write_json(directory / f'{target}.count.json', receipt)
        results.append(receipt)
    return results


def run(snapshot, engine_path, output):
    from openai_server import Engine
    output.mkdir(parents=True, exist_ok=True)
    prompts = prepare_prompts(snapshot, output / 'inputs')
    identities = {'engine_sha256': digest(engine_path), 'manifest_sha256': digest(snapshot / 'model-manifest.json')}
    report = {'status': 'running', 'stage': 'prefill', 'source_revision': SOURCE_REVISION,
              'runtime_layers': 43, **identities, 'results': [], 'memory_runs': []}
    # First pass escalates input size. Other chunk sizes compare the same 2048
    # and32768 inputs; each requested size must actually be admitted by planning.
    schedule = prefill_schedule()
    write_json(output / 'prefill.json', report)
    try:
        for chunk, sizes in schedule:
            log_path = output / f'engine-chunk-{chunk}.log'
            environment = {**os.environ, 'DSV4_EXPERIMENTAL': '1', 'CTX': '40960', 'COLI_CUDA': '1',
                           'RAM_GB': '8', 'CUDA_EXPERT_GB': '2', 'CUDA_HEADROOM_GB': '1.5',
                           'DSPARK': 'off', 'DSV4_PREFILL_CHUNK': str(chunk),
                           'DIRECT': '1', 'URING': '1', 'URING_PERSIST': '1',
                           'FIRST_MODEL_OUTPUT_TIMEOUT_MS': '3600000'}
            active = None
            try:
                with log_path.open('wb') as log:
                    saved = os.dup(2)
                    try:
                        os.dup2(log.fileno(), 2)
                        active = Engine(engine_path, snapshot, max_tokens=8192, env=environment, kv_slots=1)
                    finally:
                        os.dup2(saved, 2)
                        os.close(saved)
                plans = records(log_path, 'DSV4_MEMORY')
                if len(plans) != 1 or int(plans[0]['context']) != 40960 or int(plans[0]['chunk']) != chunk:
                    raise ValueError('memory planning did not admit the requested context/chunk')
                for target in sizes:
                    prompt = next(item for item in prompts if item['prompt_tokens'] == target)
                    started = time.monotonic()
                    last = started
                    pieces = []

                    def progress(event):
                        nonlocal last
                        if time.monotonic() - last >= 20:
                            print({'chunk': chunk, 'tokens': target, 'seconds': time.monotonic() - started, **event}, flush=True)
                            last = time.monotonic()

                    stats = active.generate(prompt['prompt'], 1, 0, 1, pieces.append,
                                            cancelled=lambda: time.monotonic() - started >= 1200,
                                            on_progress=progress)
                    elapsed = time.monotonic() - started
                    request = records(log_path, 'DSV4_REQUEST')[-1]
                    if (stats['prompt_tokens'] != target or stats['completion_tokens'] != 1 or elapsed >= 1200
                            or request.get('payload_sha256') != prompt['prompt_sha256']
                            or request.get('binary_sha256') != identities['engine_sha256']
                            or request.get('manifest_sha256') != identities['manifest_sha256']
                            or request.get('termination') not in ('length', 'eos')):
                        raise ValueError('populated prefill stop gate failed')
                    result = {'chunk': chunk, 'prompt_tokens': target, 'prompt_sha256': prompt['prompt_sha256'],
                              'warm_seconds': elapsed, 'prefill_seconds': float(request['prefill_s']),
                              'read_bytes': int(request['read_bytes']), 'stats': stats, 'request': request,
                              'first_token_text': ''.join(pieces), 'status': 'passed'}
                    report['results'].append(result)
                    write_json(output / 'prefill.json', report)
                    print(result, flush=True)
                report['memory_runs'].append({'chunk': chunk, **measured_memory(
                    plans[0], records(log_path, 'DSV4_ALLOCATIONS'), records(log_path, 'DSV4_REQUEST'))})
            finally:
                if active is not None:
                    active.close()
                    if active.process.poll() is None:
                        raise ValueError('native process release was not confirmed')
            report.setdefault('logs', []).append({'file': log_path.name, 'sha256': digest(log_path)})
            write_json(output / 'prefill.json', report)
        if identities != {'engine_sha256': digest(engine_path), 'manifest_sha256': digest(snapshot / 'model-manifest.json')}:
            raise ValueError('engine/checkpoint changed during qualification')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'prefill.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.snapshot.resolve(), args.engine.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
