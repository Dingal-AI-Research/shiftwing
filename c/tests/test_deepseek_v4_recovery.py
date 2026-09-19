"""Recovery fault injection and real local HTTP range tests; no model loads."""
import contextlib
import dataclasses
import hashlib
import http.server
import io
import json
import os
import re
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import convert_deepseek_v4 as converter
import fetch_deepseek_v4 as fetcher
import recover_deepseek_v4 as recovery


class Response(io.BytesIO):
    def __init__(self, data, lo, hi, total, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': f'bytes {lo}-{hi-1}/{total}'}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.target = self.root / 'source', self.root / 'output'
        self.source.mkdir()
        self.target.mkdir()
        self.name = 'model-00001-of-00001.safetensors'
        header, payload = {}, bytearray()
        for name, size, value in (
            ('embed.weight', 48, 1),
            ('model.layers.0.mlp.experts.0.gate_proj.weight', 16, 7),
            ('model.layers.0.mlp.experts.0.up_proj.weight', 16, 11),
            ('model.layers.0.mlp.experts.0.down_proj.weight', 16, 13),
            ('model.layers.0.attn.weight', 32, 17),
        ):
            lo = len(payload)
            payload.extend(bytes((i + value) % 256 for i in range(size)))
            header[name] = {'dtype': 'U8', 'shape': [size], 'data_offsets': [lo, len(payload)]}
        raw = json.dumps(header, separators=(',', ':')).encode()
        self.start = len(raw) + 8
        self.blob = struct.pack('<Q', len(raw)) + raw + payload
        self.headers = {'revision': recovery.SOURCE_REVISION, 'shards': {
            self.name: {'start': self.start, 'header': header, 'size': len(self.blob)}}}
        (self.source / 'config.json').write_text('{}')
        (self.source / 'tokenizer.json').write_text('{"fixture":true}')
        (self.source / converter.INDEX_FILE).write_text(json.dumps({'weight_map': {n: self.name for n in header}}))
        (self.source / 'recovery-headers.json').write_text(json.dumps(self.headers))
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(fetcher, 'WEIGHT_SHARDS', 1))
        self.stack.enter_context(mock.patch.object(recovery, 'WORKERS', 1))
        self.stack.enter_context(mock.patch.object(recovery, 'RANGE_BYTES', 16))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.rec = self.restart()
        self.groups, _ = converter.build_plan(self.source, allow_fixture=True, header_provider=self.rec.header)
        self.group, self.records = next((g, r) for g, r in self.groups.items() if g.startswith('dense/'))
        self.calls = []
        self.request = self.stack.enter_context(mock.patch.object(fetcher, '_request', side_effect=self.serve))

    def restart(self):
        return recovery.Recovery(self.source, self.target, floor=0)

    def serve(self, url, headers):
        match = re.fullmatch(r'bytes=(\d+)-(\d+)', headers['Range'])
        lo, hi = int(match[1]), int(match[2]) + 1
        self.calls.append((lo, hi))
        return Response(self.blob[lo:hi], lo, hi, len(self.blob))

    def stage(self):
        self.rec.stage_ranges(self.name, self.records)

    def commit_output(self):
        self.rec.prepare(self.records)
        segment, inventory = converter.write_group(self.source, self.target, self.group, self.records, 16)
        converter.atomic_json(self.target / converter.STATE_FILE, {
            'completed': {self.group: segment}, 'inventory': {self.group: inventory}})
        return segment, inventory

    def test_absolute_offsets_sparse_coverage_and_no_repeated_fetch(self):
        self.stage()
        self.assertEqual(self.records[0].source_offset, self.start)
        self.assertEqual(sum(hi - lo for lo, hi in self.calls), 80)
        path = self.source / self.name
        actual = path.read_bytes()
        for record in self.records:
            lo, hi = record.source_offset, record.source_offset + record.nbytes
            self.assertEqual(actual[lo:hi], self.blob[lo:hi])
        self.assertEqual(actual[self.start + 48:self.start + 96], bytes(48))
        previous = list(self.calls)
        self.rec = self.restart()
        self.stage()
        self.assertEqual(self.calls, previous)
        self.assertTrue(self.rec.owned[self.name]['complete'])

    def test_resume_incomplete_fetches_only_missing_verified_spans(self):
        def interrupted(url, headers):
            if len(self.calls) == 2:
                raise InterruptedError('injected network interruption')
            return self.serve(url, headers)
        self.request.side_effect = interrupted
        with self.assertRaises(InterruptedError):
            self.stage()
        prior = list(self.calls)
        self.assertEqual(len(self.restart().owned[self.name]['ranges']), 2)
        self.rec = self.restart()
        self.request.side_effect = self.serve
        self.stage()
        self.assertEqual(len(set(self.calls)), len(self.calls))
        self.assertEqual(self.calls[:2], prior)
        self.rec.verify_ranges(self.source / self.name, self.rec.owned[self.name], self.records)

    def test_coverage_check_does_not_trust_complete_flag(self):
        self.stage()
        evidence = self.rec.owned[self.name]
        evidence['ranges'].pop()
        with self.assertRaisesRegex(ValueError, 'cover'):
            self.rec.verify_ranges(self.source / self.name, evidence, self.records)

    def test_changed_group_cannot_reuse_or_release_another_groups_staging(self):
        self.stage()
        records = [dataclasses.replace(r, group='dense/other.bin') for r in self.records]
        with self.assertRaisesRegex(ValueError, 'different group'):
            self.rec.stage_ranges(self.name, records)
        self.rec.release(records, {}, [])
        self.assertTrue((self.source / self.name).exists())

    def test_corrupt_range_is_preserved_and_never_refetched(self):
        self.stage()
        path = self.source / self.name
        with path.open('r+b') as handle:
            handle.seek(self.start)
            handle.write(b'BAD')
        before, calls = path.read_bytes(), list(self.calls)
        self.rec = self.restart()
        with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
            self.stage()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.calls, calls)

    def test_existing_unowned_file_and_partial_are_preserved(self):
        path = self.source / self.name
        path.write_bytes(self.blob)
        partial = path.with_name('.' + path.name + '.partial')
        partial.write_bytes(b'older partial download')
        with self.assertRaisesRegex(ValueError, 'refusing to adopt'):
            self.stage()
        with self.assertRaisesRegex(ValueError, 'no download integrity'):
            self.rec.prepare(self.records)
        self.assertEqual(path.read_bytes(), self.blob)
        self.assertEqual(partial.read_bytes(), b'older partial download')
        self.assertFalse(self.rec.owned)
        self.request.assert_not_called()

    def test_verified_preexisting_full_source_is_never_owned_or_deleted(self):
        path = self.source / self.name
        path.write_bytes(self.blob)
        converter.atomic_json(self.source / fetcher.STATE_FILE, {'completed': {self.name: {
            'size': len(self.blob), 'sha256': hashlib.sha256(self.blob).hexdigest()}}})
        segment, inventory = self.commit_output()
        self.rec.release(self.records, segment, inventory)
        self.assertEqual(path.read_bytes(), self.blob)
        self.assertFalse(self.rec.owned)
        self.request.assert_not_called()

    def test_symlink_hardlink_and_replaced_inode_rejected(self):
        path = self.source / self.name
        other = self.root / 'unrelated-model'
        other.write_bytes(self.blob)
        path.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.stage()
        path.unlink()
        self.stage()
        path.rename(self.source / 'saved-owned-shard')
        path.write_bytes(self.blob)
        with self.assertRaisesRegex(ValueError, 'ownership'):
            self.stage()
        path.unlink()
        (self.source / 'saved-owned-shard').rename(path)
        os.link(path, self.source / 'second-link')
        with self.assertRaisesRegex(ValueError, 'ownership'):
            self.stage()
        self.assertEqual(other.read_bytes(), self.blob)

    def test_intent_before_creation_does_not_claim_a_collision(self):
        real_atomic = converter.atomic_json
        def collision(path, value):
            real_atomic(path, value)
            if path == self.rec.owned_path and not (self.source / self.name).exists():
                (self.source / self.name).write_bytes(self.blob)
        with mock.patch.object(converter, 'atomic_json', side_effect=collision):
            with self.assertRaises(FileExistsError):
                self.stage()
        self.rec = self.restart()
        self.assertNotIn('identity', self.rec.owned[self.name])
        with self.assertRaisesRegex(ValueError, 'ownership'):
            self.stage()
        self.assertEqual((self.source / self.name).read_bytes(), self.blob)

    def test_crash_before_creation_resumes_from_intent(self):
        real_open = os.open
        def fail_open(path, flags, *args):
            if path == self.source / self.name:
                raise InterruptedError('before exclusive creation')
            return real_open(path, flags, *args)
        with mock.patch.object(os, 'open', side_effect=fail_open):
            with self.assertRaises(InterruptedError):
                self.stage()
        self.assertFalse((self.source / self.name).exists())
        self.rec = self.restart()
        self.stage()
        self.assertTrue(self.rec.owned[self.name]['complete'])

    def test_crash_during_header_initialization_resumes(self):
        with mock.patch.object(os, 'ftruncate', side_effect=InterruptedError('after ownership commit')):
            with self.assertRaises(InterruptedError):
                self.stage()
        self.rec = self.restart()
        self.assertFalse(self.rec.owned[self.name]['initialized'])
        self.stage()
        self.assertTrue(self.rec.owned[self.name]['complete'])

    def test_range_data_is_fsynced_before_hash_journal_and_crash_refetches(self):
        real_atomic, real_fsync, real_pwrite = converter.atomic_json, os.fsync, os.pwrite
        synced = set()
        writes = []
        def pwrite(fd, data, offset):
            writes.append((fd, offset))
            synced.discard(fd)
            return real_pwrite(fd, data, offset)
        def fsync(fd):
            result = real_fsync(fd)
            synced.add(fd)
            return result
        def atomic(path, value):
            if path == self.rec.owned_path and value[self.name]['ranges']:
                self.assertIn(writes[-1][0], synced)
                raise InterruptedError('after data fsync before range journal')
            return real_atomic(path, value)
        with mock.patch.object(os, 'pwrite', side_effect=pwrite), mock.patch.object(os, 'fsync', side_effect=fsync), mock.patch.object(converter, 'atomic_json', side_effect=atomic):
            with self.assertRaises(InterruptedError):
                self.stage()
        self.rec = self.restart()
        self.assertEqual(self.rec.owned[self.name]['ranges'], [])
        self.stage()
        self.assertEqual(self.calls[0], self.calls[1])
        self.assertTrue(self.rec.owned[self.name]['complete'])

    def test_short_pwrite_retries_without_truncating_data(self):
        real_pwrite = os.pwrite
        with mock.patch.object(os, 'pwrite', side_effect=lambda fd, data, offset: real_pwrite(fd, data[:3], offset)):
            self.stage()
        self.rec.verify_ranges(self.source / self.name, self.rec.owned[self.name], self.records)

    def test_bad_http_responses_never_become_verified(self):
        for kind in ('status', 'range', 'short', 'long', 'encoding'):
            with self.subTest(kind=kind):
                def bad(url, headers):
                    response = self.serve(url, headers)
                    if kind == 'status':
                        response.status = 200
                    elif kind == 'range':
                        response.headers['Content-Range'] = 'bytes 0-7/8'
                    elif kind == 'encoding':
                        response.headers['Content-Encoding'] = 'gzip'
                    else:
                        data = response.getvalue()
                        response = Response(data[:-1] if kind == 'short' else data + b'x', *self.calls[-1], len(self.blob))
                    return response
                self.request.side_effect = bad
                with self.assertRaisesRegex(ValueError, 'range'):
                    self.stage()
                self.rec = self.restart()
                self.assertEqual(self.rec.owned[self.name]['ranges'], [])
        self.request.side_effect = self.serve
        self.stage()

    def test_transient_network_retries_are_bounded_and_preserve_prior_ranges(self):
        self.stack.enter_context(mock.patch.object(recovery, 'RETRY_WAIT_SECONDS', 0))
        attempts = 0
        def timeout_then_succeed(url, headers):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError('transient upstream stall')
            return self.serve(url, headers)
        self.request.side_effect = timeout_then_succeed
        self.stage()
        self.assertEqual(attempts, len(self.calls) + 1)
        self.assertTrue(self.rec.owned[self.name]['complete'])
        # A different group requires fresh staging after verified conversion.
        segment, inventory = self.commit_output()
        self.rec.release(self.records, segment, inventory)
        self.request.reset_mock()
        self.request.side_effect = TimeoutError('persistent upstream stall')
        with self.assertRaises(TimeoutError):
            self.stage()
        self.assertEqual(self.request.call_count, recovery.NETWORK_ATTEMPTS)
        self.assertEqual(self.restart().owned[self.name]['ranges'], [])

    def test_floor_checked_before_staging_and_again_during_transfer(self):
        with mock.patch.object(converter.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 1})()):
            with self.assertRaisesRegex(OSError, 'floor'):
                self.rec.prepare(self.records)
        self.assertFalse((self.source / self.name).exists())
        self.assertFalse(self.rec.owned)
        self.request.assert_not_called()
        actual = self.rec._preflight
        count = 0
        def falling_space(*args):
            nonlocal count
            count += 1
            if count == 3:
                raise OSError('floor consumed by another process')
            return actual(*args)
        with mock.patch.object(self.rec, '_preflight', side_effect=falling_space):
            with self.assertRaisesRegex(OSError, 'floor'):
                self.stage()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.restart().owned[self.name]['ranges']), 1)

    def test_successful_inflight_range_is_saved_when_another_worker_fails(self):
        written = threading.Event()
        real_pwrite = os.pwrite
        def pwrite(fd, data, offset):
            result = real_pwrite(fd, data, offset)
            if offset >= self.start + 16:
                written.set()
            return result
        def fail_first(url, headers):
            if headers['Range'] == f'bytes={self.start}-{self.start + 15}':
                self.assertTrue(written.wait(5))
                raise OSError('one failed range')
            return self.serve(url, headers)
        self.request.side_effect = fail_first
        with mock.patch.object(recovery, 'WORKERS', 2), mock.patch.object(os, 'pwrite', side_effect=pwrite):
            with self.assertRaises(OSError):
                self.stage()
        self.rec = self.restart()
        self.assertIn(self.start + 16, [r['offset'] for r in self.rec.owned[self.name]['ranges']])
        before = list(self.calls)
        self.request.side_effect = self.serve
        self.stage()
        self.assertEqual(self.calls.count(before[0]), 1)

    def test_release_requires_state_commit_and_independent_native_byte_match(self):
        segment, inventory = self.commit_output()
        path = self.target / self.group
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        # Even a self-consistent output/inventory hash cannot replace source comparison.
        segment['sha256'] = hashlib.sha256(data).hexdigest()
        inventory[0]['sha256'] = hashlib.sha256(data[:inventory[0]['nbytes']]).hexdigest()
        with self.assertRaisesRegex(ValueError, 'durably committed'):
            self.rec.release(self.records, segment, inventory)
        converter.atomic_json(self.target / converter.STATE_FILE, {
            'completed': {self.group: segment}, 'inventory': {self.group: inventory}})
        with self.assertRaisesRegex(ValueError, 'native-byte'):
            self.rec.release(self.records, segment, inventory)
        self.assertTrue((self.source / self.name).exists())
        self.assertFalse(self.rec.validation_path.exists())

    def test_release_validates_before_deletion_and_resumes_after_unlink(self):
        segment, inventory = self.commit_output()
        real_atomic = converter.atomic_json
        def crash_after_unlink(path, value):
            if path == self.rec.owned_path and self.name not in value:
                self.assertTrue(self.rec.validation_path.exists())
                raise InterruptedError('unlink durable; journal not yet updated')
            return real_atomic(path, value)
        with mock.patch.object(converter, 'atomic_json', side_effect=crash_after_unlink):
            with self.assertRaises(InterruptedError):
                self.rec.release(self.records, segment, inventory)
        self.assertFalse((self.source / self.name).exists())
        self.rec = self.restart()
        self.assertIn(self.name, self.rec.owned)
        self.rec.release(self.records, segment, inventory)
        self.assertFalse(self.restart().owned)
        evidence = self.rec.validation[self.group]['validation']
        self.assertEqual(evidence['source_bytes_compared'], sum(r.nbytes for r in self.records))

    def test_converter_resume_runs_pending_release_then_preserves_native_bytes(self):
        def interrupted_release(*args):
            raise InterruptedError('after conversion state commit')
        with self.assertRaises(InterruptedError):
            converter.convert(self.source, self.target, allow_fixture=True, min_final_free=0,
                              header_provider=self.rec.header, prepare_group=self.rec.prepare,
                              release_group=interrupted_release)
        self.rec = self.restart()
        final = converter.convert(self.source, self.target, allow_fixture=True, min_final_free=0,
                                  header_provider=self.rec.header, prepare_group=self.rec.prepare,
                                  release_group=self.rec.release)
        self.assertEqual(final['status'], 'complete')
        self.assertFalse(self.rec.owned)
        self.assertEqual(len(self.rec.validation), len(self.groups))
        for group, records in final['inventory'].items():
            data = (self.target / group).read_bytes()
            for record in records:
                actual = data[record['offset']:record['offset'] + record['nbytes']]
                expected = self.blob[record['source_offset']:record['source_offset'] + record['nbytes']]
                self.assertEqual(actual, expected)

    def test_real_local_http_range_server(self):
        blob, seen = self.blob, []
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                match = re.fullmatch(r'bytes=(\d+)-(\d+)', self.headers['Range'])
                lo, end = map(int, match.groups())
                seen.append((lo, end + 1))
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {lo}-{end}/{len(blob)}')
                self.send_header('Content-Length', str(end - lo + 1))
                self.end_headers()
                self.wfile.write(blob[lo:end + 1])
            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # Use urllib's real HTTP transport without sending environment credentials.
            import urllib.request
            def request(url, headers):
                return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5)
            self.request.side_effect = request
            with mock.patch.object(fetcher, 'resolve_url', return_value=f'http://127.0.0.1:{server.server_port}/weights'):
                self.stage()
            self.assertEqual(sum(hi - lo for lo, hi in seen), 80)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

    def test_lock_rejects_concurrent_writer_and_is_reusable(self):
        with recovery.recovery_lock(self.source):
            with self.assertRaisesRegex(RuntimeError, 'another checkpoint recovery'):
                with recovery.recovery_lock(self.source):
                    self.fail('second writer acquired lock')
        with recovery.recovery_lock(self.source):
            pass

    def test_invalid_bounds_and_overlap_evidence_rejected(self):
        for record in (dataclasses.replace(self.records[0], source_offset=0),
                       dataclasses.replace(self.records[0], nbytes=len(self.blob))):
            with self.assertRaisesRegex(ValueError, 'outside shard'):
                self.rec.stage_ranges(self.name, [record])
        self.stage()
        evidence = self.rec.owned[self.name]
        evidence['ranges'].append(dict(evidence['ranges'][0]))
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.rec.verify_ranges(self.source / self.name, evidence, self.records)


class SpanTests(unittest.TestCase):
    def test_partial_verified_spans_and_block_accounting(self):
        self.assertEqual(recovery.missing_spans([(10, 50), (50, 60)], [(8, 12), (20, 30), (40, 80)]), [(12, 20), (30, 40)])
        self.assertEqual(recovery.allocation_bytes([(0, 100), (200, 300), (4095, 4097)]), 8192)


if __name__ == '__main__':
    unittest.main()
