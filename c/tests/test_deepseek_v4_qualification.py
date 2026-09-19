import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import deepseek_v4_qualification as q


class QualificationValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.model = root / 'model'
        self.model.mkdir()
        self.engine = root / 'engine'
        self.engine.write_bytes(b'fixture engine')
        (self.model / 'model-manifest.json').write_text('{}')
        (self.model / 'tokenizer.json').write_bytes(b'fixture tokenizer')
        token = q.sha256(self.model / 'tokenizer.json')
        mock = patch.object(q, 'TOKENIZER_SHA256', token)
        mock.start()
        self.addCleanup(mock.stop)
        self.report = {
            'schema': q.SCHEMA, 'status': 'passed', 'source_revision': q.SOURCE_REVISION,
            'input_tokens': 32768, 'output_tokens': 8192, 'context_tokens': 40960,
            'reasoning_effort': 'low', 'dspark': 'off', 'slots': 1,
            'ram_cache_bytes': 8 * q.GIB, 'device_cache_bytes': 2 * q.GIB,
            'gpu_headroom_bytes': 3 * q.GIB // 2,
            'manifest_sha256': q.sha256(self.model / 'model-manifest.json'),
            'engine_sha256': q.sha256(self.engine), 'tokenizer_sha256': token,
            'prefill_tokens': q.PREFILL_TOKENS.copy(), 'chunk_sizes': q.CHUNKS.copy(),
            'evidence': [],
            'reviews': [
                {'id': str(i), 'complete': True, 'defect_found': True,
                 'termination': 'eos', 'prompt_tokens': 32768 if i == 4 else 128,
                 'output_tokens': 300, 'warm_seconds': 900}
                for i in range(5)],
            'memory': {'device_peak_bytes': 12 * q.GIB,
                       'device_total_bytes': 16 * q.GIB,
                       'scratch_peak_bytes': q.GIB,
                       'host_peak_bytes': 20 * q.GIB,
                       'host_available_bytes': 25 * q.GIB}}
        (self.model / 'qualification').mkdir()
        for stage in q.STAGES:
            name = 'qualification/' + stage + '.json'
            proof = {'status': 'passed', 'stage': stage,
                     'source_revision': q.SOURCE_REVISION,
                     'engine_sha256': self.report['engine_sha256'],
                     'manifest_sha256': self.report['manifest_sha256']}
            if stage == 'released_reference':
                proof.update(runtime_layers=43, logits_compared=True)
            elif stage == 'prefill':
                proof.update(runtime_layers=43, results=self.prefill_results())
            elif stage == 'reviews':
                proof['reviews'] = copy.deepcopy(self.report['reviews'])
            elif stage == 'memory':
                proof['memory'] = copy.deepcopy(self.report['memory'])
            path = self.model / name
            path.write_text(json.dumps(proof))
            self.report['evidence'].append(
                {'stage': stage, 'file': name, 'sha256': q.sha256(path)})

    def prefill_results(self):
        pairs = [(1024, tokens) for tokens in q.PREFILL_TOKENS]
        pairs.extend((chunk, tokens) for chunk in q.CHUNKS if chunk != 1024
                     for tokens in (2048, 32768))
        return [
            {'chunk': chunk, 'prompt_tokens': tokens, 'prompt_sha256': f'{chunk:04x}{tokens:060x}',
             'warm_seconds': 10.0, 'prefill_seconds': 9.0, 'read_bytes': 100,
             'status': 'passed', 'stats': {'prompt_tokens': tokens, 'completion_tokens': 1},
             'request': {'payload_sha256': f'{chunk:04x}{tokens:060x}',
                         'binary_sha256': self.report['engine_sha256'],
                         'manifest_sha256': self.report['manifest_sha256'],
                         'termination': 'length'}}
            for chunk, tokens in pairs]

    def evidence(self, stage):
        item = next(item for item in self.report['evidence'] if item['stage'] == stage)
        path = self.model / item['file']
        return item, path, json.loads(path.read_text())

    def rewrite_evidence(self, stage, proof):
        item, path, _ = self.evidence(stage)
        path.write_text(json.dumps(proof))
        item['sha256'] = q.sha256(path)

    def check(self, report=None):
        q.validate_report(self.report if report is None else report,
                          self.model, self.engine)

    def test_accepts_complete_bound_fixture(self):
        self.check()

    def test_rejects_changed_binary_and_evidence(self):
        self.engine.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'does not bind'):
            self.check()
        self.engine.write_bytes(b'fixture engine')
        (self.model / self.report['evidence'][0]['file']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            self.check()

    def test_rejects_malformed_objects_and_boolean_slot_count(self):
        for field, value in [('slots', True), ('evidence', [None] * 8),
                             ('reviews', [None] * 5), ('memory', []),
                             ('reviews', [{'id': []}] * 5)]:
            with self.subTest(field=field):
                report = copy.deepcopy(self.report)
                report[field] = value
                with self.assertRaises(ValueError):
                    self.check(report)

    def test_rejects_noninteger_prefill_or_chunk_elements(self):
        for field in ('prefill_tokens', 'chunk_sizes'):
            report = copy.deepcopy(self.report)
            report[field] = [float(value) for value in report[field]]
            with self.assertRaisesRegex(ValueError, 'prefill and chunk'):
                self.check(report)

    def test_rejects_incomplete_slow_truncated_or_nonfinite_review(self):
        for field, value in [('complete', False), ('defect_found', False),
                             ('termination', 'length'), ('warm_seconds', 1201),
                             ('warm_seconds', float('nan')), ('output_tokens', 8192)]:
            report = copy.deepcopy(self.report)
            report['reviews'][0][field] = value
            with self.assertRaises(ValueError):
                self.check(report)

    def test_rejects_missing_32768_case_and_memory_overrun(self):
        report = copy.deepcopy(self.report)
        report['reviews'][-1]['prompt_tokens'] = 32767
        with self.assertRaisesRegex(ValueError, 'actual 32768'):
            self.check(report)
        report = copy.deepcopy(self.report)
        report['memory']['device_peak_bytes'] = 15 * q.GIB
        with self.assertRaisesRegex(ValueError, 'memory limits'):
            self.check(report)

    def test_rejects_incomplete_or_unbound_full_model_prefill(self):
        _, _, proof = self.evidence('prefill')
        proof['runtime_layers'] = 42
        self.rewrite_evidence('prefill', proof)
        with self.assertRaisesRegex(ValueError, '43 runtime layers'):
            self.check()
        proof['runtime_layers'] = 43
        proof['results'][0]['request']['binary_sha256'] = '0' * 64
        self.rewrite_evidence('prefill', proof)
        with self.assertRaisesRegex(ValueError, 'not bound'):
            self.check()

    def test_rejects_reference_without_complete_logits(self):
        _, _, proof = self.evidence('released_reference')
        proof['logits_compared'] = False
        self.rewrite_evidence('released_reference', proof)
        with self.assertRaisesRegex(ValueError, '43-layer logits'):
            self.check()

    def test_rejects_aggregate_review_or_memory_divergence(self):
        report = copy.deepcopy(self.report)
        report['reviews'][0]['warm_seconds'] = 899
        with self.assertRaisesRegex(ValueError, 'aggregate reviews'):
            self.check(report)
        report = copy.deepcopy(self.report)
        report['memory']['host_peak_bytes'] -= 1
        with self.assertRaisesRegex(ValueError, 'aggregate memory'):
            self.check(report)

    def test_rejects_evidence_directory_symlink_escape(self):
        base = self.model / 'qualification'
        outside = self.model.parent / 'outside'
        base.rename(outside)
        base.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.check()


if __name__ == '__main__':
    unittest.main()
