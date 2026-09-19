#!/usr/bin/env python3
"""Checkpointed thinking-effort experiment; never a production qualification.

Run from c/: python -m tools.benchmark_deepseek_v4_thinking --output PATH.
The existing acceptance cases/gates are not modified. Completed cells are
immutable and reused after interruption; interrupted attempts remain on disk.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .deepseek_v4_count import DeepSeekTokenCounter
from .deepseek_v4_protocol import parse_deepseek_completion, render_deepseek_chat
from .deepseek_v4_spec import SOURCE_REVISION
from .qualify_deepseek_v4_reviews import CASES, SYSTEM, digest, write_json

EFFORTS = ('off', 'low', 'high', 'max')
# Get a short paired comparison, then the longest context before intermediate sizes.
LENGTHS = (2048, 32768, 8192, 16384, 512)
PROFILE = {'DSV4_EXPERIMENTAL': '1', 'CTX': '40960', 'COLI_CUDA': '1',
           'RAM_GB': '8', 'CUDA_EXPERT_GB': '2', 'CUDA_HEADROOM_GB': '1.5',
           'DSPARK': 'off', 'DSV4_PREFILL_CHUNK': '1024', 'Q3_NATIVE': '0',
           'FIRST_MODEL_OUTPUT_TIMEOUT_MS': '10800000',
           'DIRECT': '1', 'URING': '1', 'URING_PERSIST': '1'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def effort_value(effort):
    if effort not in EFFORTS:
        raise ValueError(f'unsupported effort {effort}')
    return None if effort == 'off' else effort


def assess(text, case, effort):
    """Same planted-defect oracle as acceptance, with mode-aware framing."""
    parsed = parse_deepseek_completion(text, reasoning_effort=effort_value(effort))
    if parsed.get('colib_recovery', {}).get('recovered'):
        return False, parsed, 'malformed completion framing'
    content = parsed.get('content', '').strip()
    if content.startswith('```json\n') and content.endswith('\n```'):
        content = content[8:-4]
    try:
        answer = json.loads(content)
    except ValueError:
        return False, parsed, 'final answer is not complete JSON'
    if not isinstance(answer, dict) or answer.get('verdict') != 'REJECT' or not isinstance(answer.get('issues'), list):
        return False, parsed, 'final answer does not reject with findings'
    for issue in answer['issues']:
        if (isinstance(issue, dict) and issue.get('path') == case['path']
                and type(issue.get('line')) is int and issue['line'] == case['line']
                and isinstance(issue.get('explanation'), str)
                and all(re.search(term, issue['explanation'], re.I) for term in case['terms'])):
            return True, parsed, 'planted location and semantic anchors found'
    return False, parsed, 'planted finding did not match location and semantic anchors'


def prepare(snapshot, output, engine_path, seconds):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(snapshot / 'tokenizer.json'))
    counter = DeepSeekTokenCounter(snapshot)
    counter.validate_identity()
    # These boundaries must each be one native token for the count split below.
    for marker in ('<think>', '</think>'):
        if len(tokenizer.encode(marker, add_special_tokens=False).ids) != 1:
            raise ValueError('thinking marker is not one pinned tokenizer token')
    output.mkdir(parents=True, exist_ok=True)
    identity = {'source_revision': SOURCE_REVISION, 'engine_sha256': digest(engine_path),
                'manifest_sha256': digest(snapshot / 'model-manifest.json'),
                'tokenizer_sha256': digest(snapshot / 'tokenizer.json'),
                'driver_sha256': digest(Path(__file__)), 'profile': PROFILE,
                'output_budget': 8192, 'case_timeout_s': seconds}
    manifest_path = output / 'experiment.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity:
            raise ValueError('experiment identity changed; use a new output directory')
        for cell in manifest['cells']:
            folder = output / 'inputs' / cell['key']
            for name, sha in cell['files'].items():
                if digest(folder / name) != sha:
                    raise ValueError(f'prepared artifact changed: {folder / name}')
        return manifest
    manifest = {'scope': 'capability and performance only; not acceptance or activation',
                'created_at': utc(), 'identity': identity,
                'limitations': ['one deterministic sample per cell',
                    'small planted defects with synthetic reference helpers',
                    'one distinct defect per length; compare efforts within each length',
                    'effort prefixes consume context; helper padding is adjusted to exact total',
                    'expert and OS caches are not flushed; per-request KV starts empty',
                    'existing full-model numerical parity gate is still failing'], 'cells': []}
    helpers = [f'def helper_{i:05d}(value):\n    return value + {i % 17}\n' for i in range(3000)]
    for target in LENGTHS:
        case = next(c for c in CASES if c['target'] == target)
        selected = '\n'.join(f'{i}: {line}' for i, line in enumerate(case['source'].splitlines(), 1))
        tail = f'\nSelected source: {case["path"]}\n```python\n{selected}\n```\nReview this selected file.'
        for effort in EFFORTS:
            mode = effort_value(effort)

            def messages_for(number, padding=0):
                context = 'Repository reference helpers:\n```python\n' + ''.join(helpers[:number])
                context += '# Context note:' + ' evidence' * padding + '\n```\n'
                return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': context + tail}]

            def render(number, padding=0):
                return render_deepseek_chat(messages_for(number, padding), reasoning_effort=mode)

            def count(number, padding=0):
                return len(tokenizer.encode(render(number, padding), add_special_tokens=False).ids)

            low, high = 0, len(helpers)
            if count(0) > target:
                raise ValueError('mandatory review content exceeds input target')
            while low < high:
                middle = (low + high + 1) // 2
                if count(middle) <= target:
                    low = middle
                else:
                    high = middle - 1
            padding = target - count(low)
            for _ in range(5):
                difference = target - count(low, padding)
                if not difference:
                    break
                padding += difference
            prompt = render(low, padding)
            receipt = counter.count(prompt, reasoning_effort=mode)
            if receipt['prompt_tokens'] != target or count(low, padding) != target:
                raise ValueError('native and reference exact input counts must agree')
            key = f'{target:05d}-{effort}'
            folder = output / 'inputs' / key
            folder.mkdir(parents=True, exist_ok=True)
            request = {'model': 'deepseek-v4-flash-0731-colib', 'reasoning_effort': mode,
                       'messages': messages_for(low, padding), 'max_tokens': 8192,
                       'temperature': 0, 'top_p': 1}
            write_json(folder / 'request.json', request)
            (folder / 'prompt.txt').write_text(prompt)
            receipt.pop('prompt')
            write_json(folder / 'count.json', receipt)
            cell = {'key': key, 'effort': effort, 'prompt_tokens': target, 'case': copy.deepcopy(case),
                    'prompt_sha256': receipt['prompt_sha256'],
                    'files': {f: digest(folder / f) for f in ('request.json', 'prompt.txt', 'count.json')}}
            manifest['cells'].append(cell)
            print(json.dumps({'prepared': key, 'prompt_tokens': target}), flush=True)
    write_json(manifest_path, manifest)
    return manifest


class Timeline:
    """Native DATA precedes DECODE_PROGRESS for each non-EOS sampled token."""
    def __init__(self, effort, now=time.monotonic):
        self.effort, self.now = effort, now
        self.started = now()
        self.text = ''
        self.first_token_s = self.first_final_s = self.thinking_end_s = None
        self.boundary_tokens = None
        self.generated = 0
        self.prefill_ms = None
        self.decode_elapsed_ms = 0

    def on_text(self, value):
        elapsed = self.now() - self.started
        if value and self.first_token_s is None:
            self.first_token_s = elapsed
        self.text += value
        if self.effort == 'off':
            final = self.text
        elif '</think>' in self.text:
            if self.thinking_end_s is None:
                self.thinking_end_s = elapsed
            final = self.text.split('</think>', 1)[1]
        else:
            final = ''
        if final.strip() and self.first_final_s is None:
            self.first_final_s = elapsed

    def on_progress(self, event):
        if event.get('phase') == 'decode':
            count = event['completion_tokens']
            if count != self.generated + 1:
                raise ValueError('native decode token sequence has a gap')
            self.generated = count
            self.decode_elapsed_ms = event['elapsed_ms']
            if self.thinking_end_s is not None and self.boundary_tokens is None:
                self.boundary_tokens = count
        elif event.get('event') == 'end':
            self.prefill_ms = event['elapsed_ms']

    def metrics(self, total=None, eos=False):
        if total is not None and total != self.generated + int(eos):
            raise ValueError('native DONE token count disagrees with progress stream')
        if self.effort == 'off':
            thinking, final, framing = 0, self.generated, 0
        elif self.boundary_tokens is not None:
            thinking = self.boundary_tokens - 1
            final = self.generated - self.boundary_tokens
            framing = 1
        else:
            thinking, final, framing = self.generated, 0, 0
        return {'ttft_s': self.first_token_s, 'first_final_s': self.first_final_s,
                'thinking_end_s': self.thinking_end_s,
                'thinking_tokens': thinking, 'final_tokens': final,
                'framing_tokens': framing, 'eos_tokens': int(eos),
                'output_tokens': total if total is not None else self.generated,
                'prefill_s': None if self.prefill_ms is None else self.prefill_ms / 1000,
                'observed_decode_s': self.decode_elapsed_ms / 1000,
                'thinking_decode_s': None if self.thinking_end_s is None or self.first_token_s is None
                    else self.thinking_end_s - self.first_token_s,
                'final_generation_s': None if self.first_final_s is None else self.now() - self.started - self.first_final_s}


def report(output, manifest, status, active=None):
    rows = []
    for cell in manifest['cells']:
        result = output / 'results' / (cell['key'] + '.json')
        rows.append(json.loads(result.read_text()) if result.exists() else
                    {'key': cell['key'], 'prompt_tokens': cell['prompt_tokens'],
                     'effort': cell['effort'], 'status': 'pending'})
    data = {'scope': manifest['scope'], 'status': status, 'updated_at': utc(),
            'active': active, 'identity': manifest['identity'], 'limitations': manifest['limitations'],
            'results': rows}
    write_json(output / 'results.json', data)
    lines = ['# DeepSeek reviewer thinking experiment', '', manifest['scope'] + '.', '',
             f'Status: **{status}**. Updated {data["updated_at"]}. Active: {active or "none"}.', '',
             'Native token counts; thinking/final counts exclude `</think>` and EOS. TTFT includes thinking. '
             'First final is the first non-whitespace final-answer content. Load time is separate. '
             'Timeouts are censored observations, never completed reviews.', '',
             '| Input | Effort | Status | Prefill min | First final min | Total min | Thinking tok | Final tok | Decode tok/s | Defect found |',
             '|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
    def number(value, divisor=1):
        return '—' if value is None else f'{value / divisor:.2f}'
    for row in rows:
        lines.append('| ' + ' | '.join([str(row['prompt_tokens']), row['effort'], row['status'],
            number(row.get('prefill_s'), 60), number(row.get('first_final_s'), 60),
            number(row.get('seconds'), 60), str(row.get('thinking_tokens', '—')),
            str(row.get('final_tokens', '—')), number(row.get('tokens_per_second')),
            str(row.get('defect_found', '—'))]) + ' |')
    lines += ['', 'Limitations:', ''] + ['- ' + note for note in manifest['limitations']]
    temporary = output / 'README.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(output / 'README.md')


def measure(snapshot, engine_path, output, manifest):
    from openai_server import Engine
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('DSV4_', 'Q3_', 'CUDA_', 'COLI_'))}
    environment.update(PROFILE)
    engine = None
    load_seconds = None
    engine_log = None
    (output / 'results').mkdir(exist_ok=True)
    (output / 'attempts').mkdir(exist_ok=True)
    report(output, manifest, 'running')
    try:
        for cell in manifest['cells']:
            result_path = output / 'results' / (cell['key'] + '.json')
            if result_path.exists():
                continue
            if digest(engine_path) != manifest['identity']['engine_sha256']:
                raise ValueError('engine changed during experiment')
            attempt = output / 'attempts' / (cell['key'] + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
            attempt.mkdir()
            if engine is None:
                load_started = time.monotonic()
                engine_log = str((attempt / 'engine.log').relative_to(output))
                with (output / engine_log).open('wb') as log:
                    saved = os.dup(2)
                    try:
                        os.dup2(log.fileno(), 2)
                        engine = Engine(engine_path, snapshot, max_tokens=8192, env=environment, kv_slots=1)
                    finally:
                        os.dup2(saved, 2)
                        os.close(saved)
                load_seconds = time.monotonic() - load_started
                write_json(attempt / 'engine.json', {'pid': engine.process.pid, 'load_seconds': load_seconds})
            prompt_path = output / 'inputs' / cell['key'] / 'prompt.txt'
            if digest(prompt_path) != cell['prompt_sha256']:
                raise ValueError('input changed during experiment')
            timeline = Timeline(cell['effort'])
            timeout_s = manifest['identity']['case_timeout_s']
            timed_out = threading.Event()
            # Kill only this owned native child if it fails to acknowledge cancellation.
            child = engine.process
            def hard_stop():
                timed_out.set()
                if child.poll() is None:
                    child.kill()
            watchdog = threading.Timer(timeout_s + 30, hard_stop)
            watchdog.daemon = True
            watchdog.start()
            row = {'key': cell['key'], 'effort': cell['effort'], 'prompt_tokens': cell['prompt_tokens'],
                   'prompt_sha256': cell['prompt_sha256'], 'started_at': utc(),
                   'engine_load_s': load_seconds, 'request_id': engine.next_request_id,
                   'engine_log': engine_log,
                   'attempt': str(attempt.relative_to(output)), 'status': 'running',
                   'complete': False, 'defect_found': False}
            report(output, manifest, 'running', cell['key'])
            last_saved = -100.0
            with (attempt / 'events.jsonl').open('w') as events, (attempt / 'completion.txt').open('w') as stream:
                def text(value):
                    timeline.on_text(value)
                    stream.write(value)
                    stream.flush()

                def progress(event):
                    nonlocal last_saved
                    timeline.on_progress(event)
                    elapsed = time.monotonic() - timeline.started
                    # Layer activity is only a liveness signal; retain a periodic sample.
                    if event.get('event') != 'activity' or elapsed - last_saved >= 15:
                        events.write(json.dumps({'seconds': elapsed, **event}) + '\n')
                        events.flush()
                    if elapsed - last_saved >= 15:
                        last_saved = elapsed
                        active = {**row, **timeline.metrics(), 'elapsed_s': elapsed,
                                  'native_pid': child.pid, 'updated_at': utc(), 'progress': event}
                        write_json(output / 'active.json', active)
                        print(json.dumps({k: active[k] for k in ('key', 'elapsed_s', 'thinking_tokens', 'final_tokens', 'progress')}), flush=True)

                try:
                    stats = engine.generate(prompt_path.read_text(), 8192, 0, 1, text,
                        cancelled=lambda: time.monotonic() - timeline.started >= timeout_s,
                        on_progress=progress)
                    eos = not stats['length_limited']
                    row.update(timeline.metrics(stats['completion_tokens'], eos), stats=stats,
                               tokens_per_second=stats['tokens_per_second'],
                               status='eos' if eos else 'length',
                               complete=eos and stats['prompt_tokens'] == cell['prompt_tokens'])
                except Exception as error:
                    row.update(timeline.metrics(), status='timeout' if timed_out.is_set() or
                               time.monotonic() - timeline.started >= timeout_s else 'error',
                               error=f'{type(error).__name__}: {error}')
                finally:
                    watchdog.cancel()
            row['seconds'] = time.monotonic() - timeline.started
            found, parsed, reason = assess(timeline.text, cell['case'], cell['effort'])
            row.update(defect_found=bool(row['complete'] and found), oracle_reason=reason,
                       completion_sha256=digest(attempt / 'completion.txt'), finished_at=utc())
            write_json(attempt / 'parsed.json', parsed)
            write_json(result_path, row)
            write_json(output / 'active.json', row)
            report(output, manifest, 'running')
            print(json.dumps(row), flush=True)
            if row['status'] in ('timeout', 'error'):
                engine.close()
                engine = None
        report(output, manifest, 'complete')
    except BaseException:
        report(output, manifest, 'interrupted')
        raise
    finally:
        if engine is not None:
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('deepseek-v4-flash-0731'))
    parser.add_argument('--engine', type=Path, default=Path('deepseek_v4'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-timeout-seconds', type=int, default=10800)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.case_timeout_seconds <= 0:
        parser.error('timeout must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(args.snapshot.resolve(), output, args.engine.resolve(), args.case_timeout_seconds)
        if args.prepare_only:
            report(output, manifest, 'prepared')
        else:
            measure(args.snapshot.resolve(), args.engine.resolve(), output, manifest)


if __name__ == '__main__':
    main()
