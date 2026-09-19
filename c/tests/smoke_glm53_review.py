#!/usr/bin/env python3
"""Opt-in smoke test of a real Shiftwing GLM SSE response (no LocalForge).

Connects to an already running server. Writes request.json, wire.sse,
timeline.jsonl, thinking.txt, response.txt and summary.json to --output-dir.
It fails unless prefill, native decode counts, thinking, final JSON, normal
finish_reason=stop and [DONE] all arrive. It never accepts a partial approval.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18080')
    parser.add_argument('--model', default='glm53-flash')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=4096)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        'model': args.model, 'stream': True, 'stream_options': {'include_usage': True},
        'temperature': 0, 'top_p': 1, 'enable_thinking': True,
        'max_tokens': args.max_tokens, 'cache_slot': 0,
        'messages': [
            {'role': 'system', 'content': 'You are a read-only software reviewer. Think briefly, then return exactly one JSON object with keys verdict (approved or changes_required) and notes (an array of short strings). No markdown fences. Assess correctness, not style. Keep the final response under 150 words.'},
            {'role': 'user', 'content': '''Review this proposed scaffold for a streamed review panel:
1. Accept both shiftwing.progress and colib.progress events, schema version 1.
2. Display completed prompt tokens divided by total prompt tokens as reading progress.
3. Display reasoning_content as live thinking, separately from the final content.
4. Display generated tokens divided by max_tokens as output budget used. Explain that this is not time remaining or percentage of the final answer.
5. When the network stream closes, accept any partial text containing the word approved as an approval, even if finish_reason=length or [DONE] is missing.
6. Keep the orchestrator blocked until the review finishes and the model is unloaded.
7. Retain the expanded panel, progress and final notes after completion.
Should this scaffold be approved? Identify the concrete correctness issue and the required correction. Do not use tools or edit files.'''},
        ],
    }
    (args.output_dir / 'request.json').write_text(json.dumps(payload, indent=2))
    started = time.monotonic()
    summary = {'passed': False, 'model': args.model, 'thinking_enabled': True,
               'max_tokens': args.max_tokens, 'finish_reason': None, 'done': False}
    thought, answer, prefill, decode = [], [], [], []
    last_print = ''
    try:
        request = urllib.request.Request(args.base_url.rstrip('/') + '/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3600) as response, \
             (args.output_dir / 'wire.sse').open('wb') as wire, \
             (args.output_dir / 'timeline.jsonl').open('w') as timeline:
            summary['http_status'] = response.status
            summary['content_type'] = response.headers.get('Content-Type')
            if 'text/event-stream' not in summary['content_type']:
                raise AssertionError('Expected an SSE response')
            for raw in response:
                wire.write(raw)
                wire.flush()
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                elapsed = round(time.monotonic() - started, 3)
                if data == '[DONE]':
                    summary['done'] = True
                    timeline.write(json.dumps({'elapsed_s': elapsed, 'done': True}) + '\n')
                    break
                event = json.loads(data)
                timeline.write(json.dumps({'elapsed_s': elapsed, 'event': event}) + '\n')
                timeline.flush()
                if event.get('error'):
                    summary['stream_error'] = event['error']
                    continue
                if event.get('object') == 'shiftwing.progress':
                    if event.get('phase') == 'prefill':
                        prefill.append(event)
                        summary.setdefault('first_prefill_s', elapsed)
                        percent = event.get('progress_percent', 100 * event['prompt_tokens_prefilled'] / max(1, event['prompt_tokens_total']))
                        key = f'prefill {percent:.0f}%'
                        if event.get('event') == 'end':
                            summary['prefill_end_s'] = elapsed
                    elif event.get('phase') == 'decode':
                        decode.append(event)
                        key = f'generated {event["completion_tokens"] // 100 * 100} / {event["max_tokens"]} tokens'
                    else:
                        key = ''
                    if key and key != last_print:
                        print(f'{elapsed:.1f}s {key}', flush=True)
                        last_print = key
                if event.get('usage'):
                    summary['usage'] = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    reasoning = delta.get('reasoning_content')
                    # The gateway's legacy leading dot is a keepalive, not a thought.
                    if reasoning and (thought or reasoning != '.'):
                        thought.append(reasoning)
                        summary.setdefault('first_thinking_s', elapsed)
                    if delta.get('content'):
                        answer.append(delta['content'])
                        summary.setdefault('first_response_s', elapsed)
                    if choice.get('finish_reason'):
                        summary['finish_reason'] = choice['finish_reason']
        assert not summary.get('stream_error'), str(summary.get('stream_error'))
        assert summary['done'] and summary['finish_reason'] == 'stop', 'No complete normal response'
        assert prefill and prefill[0]['event'] == 'begin', 'Missing prefill begin'
        assert prefill[-1]['event'] == 'end', 'Missing prefill end'
        counts = [p['prompt_tokens_prefilled'] for p in prefill]
        assert counts == sorted(counts), 'Prefill count moved backwards'
        assert len(set(counts)) > 2, 'No intermediate prefill progress'
        assert counts[-1] == prefill[-1]['prompt_tokens_total'], 'Prefill never reached 100%'
        assert decode and decode[0]['completion_tokens'] == 1, 'Missing initial native decode count'
        assert [d['completion_tokens'] for d in decode] == sorted(d['completion_tokens'] for d in decode), 'Decode count moved backwards'
        assert all(d['max_tokens'] == args.max_tokens for d in decode), 'Server silently clamped the output budget'
        assert decode[-1]['completion_tokens'] == summary['usage']['completion_tokens'], 'Final token counts disagree'
        assert thought and answer, 'Thinking or response was not streamed'
        final = json.loads(''.join(answer).strip())
        assert final.get('verdict') == 'changes_required' and final.get('notes'), 'Reviewer did not reject the unsafe approval rule'
        summary['passed'] = True
    except Exception as error:
        summary['error'] = str(error)
        raise
    finally:
        summary.update(elapsed_s=round(time.monotonic() - started, 3), prefill_events=len(prefill),
                       decode_events=len(decode), thinking_chars=len(''.join(thought)), response_chars=len(''.join(answer)))
        (args.output_dir / 'thinking.txt').write_text(''.join(thought))
        (args.output_dir / 'response.txt').write_text(''.join(answer))
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
