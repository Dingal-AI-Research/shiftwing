import hashlib
import os
from pathlib import Path
import select
import subprocess
import tempfile
import unittest

C = Path(__file__).resolve().parents[1]


class NativeServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temp.name) / 'server'
        subprocess.run(['cc', '-O2', '-pthread', str(C / 'tests/fixtures/deepseek_v4_server.c'), '-lm', '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.process = subprocess.Popen([str(self.binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.assertEqual(self.line(), b'\x01\x01READY\x01\x01\n')

    def tearDown(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=6)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()

    def line(self):
        output = b''
        while not output.endswith(b'\n'):
            self.assertTrue(select.select([self.process.stdout], [], [], 6)[0], 'engine response timed out')
            byte = self.process.stdout.read(1)
            if not byte:
                break
            output += byte
        return output

    def submit(self, id='a', payload=b'hello', slot=0, maximum=8, temperature='0', top_p='1', grammar=b''):
        header = f'SUBMIT {id} {slot} {len(payload)} {maximum} {temperature} {top_p} {len(grammar)}\n'.encode()
        self.process.stdin.write(header + payload + grammar + b'\n')

    def terminal(self, id='a'):
        lines = []
        while True:
            line = self.line()
            self.assertTrue(line)
            if line.startswith(b'DATA '):
                n = int(line.split()[2])
                data = b''
                while len(data) < n + 1:
                    data += self.process.stdout.read(n + 1 - len(data))
                lines.append(data)
            lines.append(line)
            if line.startswith((f'DONE {id} '.encode(), f'ERROR {id} '.encode())):
                return lines

    def test_binary_data_and_clean_next_request(self):
        for id in ('a', 'b'):
            self.submit(id, b'x\x00y\nz')
            lines = self.terminal(id)
            self.assertIn(b'x\x00y\nz\n', lines)
            self.assertTrue(lines[-1].startswith(f'DONE {id} STAT '.encode()))

    def test_cancel_during_prefill_then_reset(self):
        self.submit(payload=b'hold')
        self.assertEqual(self.line(), b'PREFILL_BEGIN a 3 0\n')
        self.process.stdin.write(b'CANCEL a\n')
        self.assertEqual(self.line(), b'ERROR a CANCELLED\n')
        self.submit('b')
        self.assertTrue(self.terminal('b')[-1].startswith(b'DONE b '))

    def test_busy_does_not_cancel_active_request(self):
        self.submit(payload=b'hold')
        self.assertEqual(self.line(), b'PREFILL_BEGIN a 3 0\n')
        self.submit('b')
        self.assertEqual(self.line(), b'ERROR b BUSY\n')
        self.process.stdin.write(b'CANCEL unknown\nCANCEL a\n')
        self.assertEqual(self.line(), b'ERROR a CANCELLED\n')

    def test_unsupported_slot_grammar_budget_consumed_before_reject(self):
        for params in ({'slot': 1}, {'grammar': b'anything\n'}, {'maximum': 8193}, {'maximum': 0}, {'top_p': '0'}):
            self.submit(**params)
            self.assertEqual(self.line(), b'ERROR a BAD_REQUEST\n')
        self.submit('next')
        self.assertTrue(self.terminal('next')[-1].startswith(b'DONE next '))

    def test_forward_error_is_terminal_and_sanitized(self):
        self.submit(payload=b'error')
        self.assertEqual(self.terminal()[-1], b'ERROR a FORWARD_FAILED bad state\n')
        self.submit('next')
        self.assertTrue(self.terminal('next')[-1].startswith(b'DONE next '))

    def test_length_flag_survives_protocol(self):
        self.submit(maximum=1)
        self.assertTrue(self.terminal()[-1].endswith(b' 3 1\n'))

    def test_invalid_length_is_fatal(self):
        self.process.stdin.write(b'SUBMIT a 0 18446744073709551616 8 0 1\n')
        self.assertEqual(self.process.wait(timeout=5), 2)

    def test_eof_cancels_active_worker(self):
        self.submit(payload=b'hold')
        self.assertEqual(self.line(), b'PREFILL_BEGIN a 3 0\n')
        self.process.stdin.close()
        self.assertEqual(self.line(), b'ERROR a CANCELLED\n')
        self.assertEqual(self.process.wait(timeout=5), 0)

    def test_duplicate_active_id_closes_ambiguous_connection(self):
        self.submit(payload=b'hold')
        self.assertEqual(self.line(), b'PREFILL_BEGIN a 3 0\n')
        self.submit('a')
        self.assertEqual(self.line(), b'ERROR a CANCELLED\n')
        self.assertEqual(self.process.wait(timeout=5), 2)

    def test_sha256_reference_bytes_and_padding_boundaries(self):
        path = Path(self.temp.name) / 'hash-input'
        for size in (0, 1, 55, 56, 63, 64, 65, 127, 128, 129, 65535, 65536, 1000000):
            data = bytes((i * 17 + 31) % 256 for i in range(size))
            path.write_bytes(data)
            actual = subprocess.check_output([str(self.binary), str(path)]).decode().strip()
            self.assertEqual(actual, hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    unittest.main()
