"""Exact-count and CPU-only gateway contracts; no inference/model loading."""
import hashlib
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import openai_server as gateway
from tools import deepseek_v4_count as counter
from tools.deepseek_v4_protocol import render_deepseek_chat, official
from tools.deepseek_v4_spec import TOKENIZER_SHA256, MODEL_ID


class CounterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'tokenizer.json').write_text('{}')
        self.binary = self.root / 'helper'
        self.binary.write_bytes(b'fixture binary')
        self.counter = counter.DeepSeekTokenCounter(self.root, self.binary)
        self.hash_patch = mock.patch.object(counter, 'TOKENIZER_SHA256', hashlib.sha256(b'{}').hexdigest())
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)

    def test_fully_rendered_utf8_payload_and_limits_are_preserved(self):
        prompt = render_deepseek_chat([{'role': 'system', 'content': 'Keep every changed line.'},
                                      {'role': 'user', 'content': '你好\n- old\n+ new\n' + ' evidence' * 32768}],
                                     reasoning_effort='low')
        for tokens, fits in ((32768, True), (32769, True), (92160, True), (92161, False)):
            with mock.patch.object(subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps({'tokens': tokens}).encode())) as run:
                result = self.counter.count(prompt, reasoning_effort='low')
            raw = prompt.encode()
            self.assertEqual(run.call_args.kwargs['input'], f'COUNT {len(raw)}\n'.encode() + raw + b'\n')
            self.assertEqual(result['prompt'], prompt)
            self.assertEqual(result['prompt_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(result['prompt_tokens'], tokens)
            self.assertEqual(result['fits'], fits)
            self.assertEqual(result['input_limit'], 92160)
            self.assertEqual(result['output_budget'], 8192)
            self.assertEqual(result['review_context'], 100352)
            self.assertEqual(result['reasoning_effort'], 'low')

    def test_changed_tokenizer_fails_before_helper_execution(self):
        (self.root / 'tokenizer.json').write_text('{"changed":true}')
        with mock.patch.object(subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'pinned checkpoint'):
                self.counter.count('input')
        run.assert_not_called()

    def test_incomplete_invalid_or_failed_helper_never_returns_a_count(self):
        for response in (b'', b'{}', b'{"tokens":true}', b'{"tokens":-1}', b'{"tokens":6}', b'{"tokens":0}', b'{"tokens":2}\n{\"tokens\":3}'):
            with self.subTest(response=response), mock.patch.object(subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=response)):
                with self.assertRaises(RuntimeError):
                    self.counter.count('hello')
        with mock.patch.object(subprocess, 'run', return_value=SimpleNamespace(returncode=2, stdout=b'{"tokens":1}')):
            with self.assertRaises(RuntimeError):
                self.counter.count('hello')
        with mock.patch.object(subprocess, 'run', side_effect=subprocess.TimeoutExpired('helper', 30)):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.counter.count('hello')


class CountGatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'config.json').write_text('{"model_type":"deepseek_v4"}')
        self.engine = mock.Mock()
        self.server = gateway.APIServer(('127.0.0.1', 0), self.engine, MODEL_ID, api_key='fixture-key', model_path=self.root)
        self.server.tokenizer_only = True
        self.server.token_counter = mock.Mock()
        self.server.token_counter.count.side_effect = lambda prompt, **kwargs: {'prompt': prompt, 'prompt_tokens': 123, **kwargs}
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.server.scheduler.close()
        self.thread.join(5)

    def request(self, path, body, authorized=True):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        headers = {'Content-Type': 'application/json'}
        if authorized:
            headers['Authorization'] = 'Bearer fixture-key'
        connection.request('POST', path, json.dumps({'model': MODEL_ID, **body}), headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def test_count_uses_generation_renderer_preserves_tools_and_low_effort(self):
        messages = [{'role': 'system', 'content': 'Mandatory instructions.'},
                    {'role': 'user', 'content': 'Original request.\n- exact old code\n+ exact new code'}]
        tools = [{'type': 'function', 'function': {'name': 'report', 'description': 'Report a finding', 'parameters': {'type': 'object', 'properties': {}}}}]
        body = {'messages': messages, 'reasoning_effort': 'low', 'tools': tools}
        status, result = self.request('/v1/chat/count_tokens', body)
        self.assertEqual(status, 200)
        self.assertEqual(result['prompt'], render_deepseek_chat(messages, tools=tools, reasoning_effort='low'))
        self.assertTrue(result['prompt'].endswith(official.thinking_start_token))
        self.assertEqual(result['reasoning_effort'], 'low')
        handler = object.__new__(gateway.APIHandler)
        handler.server = self.server
        handler.generation = mock.Mock()
        handler.chat_completion(body, 'fixture')
        self.assertEqual(handler.generation.call_args.args[1], result['prompt'])
        self.engine.generate.assert_not_called()

    def test_omitted_effort_stays_nonthinking_even_with_global_think(self):
        messages = [{'role': 'user', 'content': 'Review'}]
        with mock.patch.dict(os.environ, {'COLI_THINK': '1'}):
            status, result = self.request('/v1/chat/count_tokens', {'messages': messages})
        self.assertEqual(status, 200)
        self.assertIsNone(result['reasoning_effort'])
        self.assertEqual(result['prompt'], render_deepseek_chat(messages))

    def test_authentication_errors_and_generation_fail_closed(self):
        body = {'messages': [{'role': 'user', 'content': 'Review'}]}
        status, _ = self.request('/v1/chat/count_tokens', body, authorized=False)
        self.assertEqual(status, 401)
        self.server.token_counter.count.assert_not_called()
        for path in ('/v1/chat/completions', '/v1/completions', '/v1/messages'):
            status, _ = self.request(path, body)
            self.assertEqual(status, 503)
        self.engine.generate.assert_not_called()
        status, _ = self.request('/v1/chat/count_tokens', {**body, 'reasoning_effort': 'medium'})
        self.assertEqual(status, 400)

    def test_tokenizer_only_startup_never_constructs_engine(self):
        fake = mock.Mock(model_path=self.root)
        with mock.patch.object(gateway, 'APIServer', return_value=fake), mock.patch.object(gateway, 'Engine') as engine, mock.patch.object(counter.DeepSeekTokenCounter, 'validate_identity'):
            gateway.serve(self.root, port=18767, tokenizer_only=True)
        engine.assert_not_called()
        self.assertTrue(fake.tokenizer_only)
        fake.server_close.assert_called_once()
        fake.scheduler.close.assert_called_once()


@unittest.skipUnless((ROOT / 'deepseek_v4_tokenize').is_file() and (ROOT / '.deepseek-v4-flash-0731.source/tokenizer.json').is_file(), 'build tokenizer helper and fetch pinned metadata for native checks')
class NativeCountTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ROOT / '.deepseek-v4-flash-0731.source/tokenizer.json'
        self.assertEqual(hashlib.sha256(self.tokenizer.read_bytes()).hexdigest(), TOKENIZER_SHA256)
        self.binary = ROOT / 'deepseek_v4_tokenize'

    def test_native_boundary_cases_and_empty_or_nul_payload(self):
        payloads = [b'', b'  1234', b"'return", b'x\0y', '你好Hello'.encode()]
        frames = b''.join(f'ENCODE {len(raw)}\n'.encode() + raw + b'\n' for raw in payloads)
        result = subprocess.run([self.binary, self.tokenizer], input=frames, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([bytes.fromhex(row['decoded_hex']) for row in rows], payloads)
        self.assertEqual(rows[1]['ids'], [262, 6895, 22])
        self.assertEqual(rows[2]['ids'], [9, 3916])
        self.assertEqual(rows[0]['tokens'], 0)

    def test_native_rendered_prompt_at_90k_input_boundary(self):
        native = counter.DeepSeekTokenCounter(self.tokenizer.parent, self.binary)
        def render(words):
            return render_deepseek_chat([{'role': 'user', 'content': ' evidence' * words}])
        overhead = native.count(render(1))['prompt_tokens'] - 1
        for target, fits in ((92160, True), (92161, False)):
            counted = native.count(render(target - overhead))
            self.assertEqual(counted['prompt_tokens'], target)
            self.assertEqual(counted['fits'], fits)
            self.assertEqual(counted['input_limit'], 92160)
            self.assertEqual(counted['review_context'], 100352)

    def test_invalid_or_truncated_frames_fail(self):
        for frame in (b'COUNT -1\n', b'COUNT 16777217\n', b'COUNT 4 junk\n', b'COUNT 4\nabc', b'COUNT 1\nx!'):
            result = subprocess.run([self.binary, self.tokenizer], input=frame, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('error', json.loads(result.stdout))

    def test_recipe_drift_is_rejected(self):
        metadata = json.loads(self.tokenizer.read_text())
        metadata['pre_tokenizer']['pretokenizers'][0]['pattern']['Regex'] = r'\p{N}'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tokenizer.json'
            path.write_text(json.dumps(metadata))
            result = subprocess.run([self.binary, path], input=b'COUNT 1\nx\n', capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'unsupported', result.stderr)


if __name__ == '__main__':
    unittest.main()
