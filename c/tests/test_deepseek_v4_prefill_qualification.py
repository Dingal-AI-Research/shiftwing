import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import qualify_deepseek_v4_prefill as prefill


class PrefillEvidenceTests(unittest.TestCase):
    def test_records_parse_native_fields_and_reject_ambiguous_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'engine.log'
            log.write_text('ignored line\nDSV4_MEMORY context=40960 chunk=1024\n')
            self.assertEqual(prefill.records(log, 'DSV4_MEMORY'),
                             [{'context': '40960', 'chunk': '1024'}])
            for line in ('DSV4_MEMORY context=40960 context=1\n',
                         'DSV4_MEMORY context=40960 malformed\n',
                         'DSV4_MEMORY =40960\n'):
                with self.subTest(line=line):
                    log.write_text(line)
                    with self.assertRaisesRegex(ValueError, 'malformed'):
                        prefill.records(log, 'DSV4_MEMORY')

    def test_measured_memory_combines_allocator_and_device_high_water(self):
        gib = prefill.GIB
        plan = {'device_available': str(15 * gib), 'host_available': str(25 * gib)}
        allocations = [
            {'device_total': str(16 * gib), 'pool_used_peak': str(10 * gib),
             'pool_reserved_peak': str(11 * gib), 'device_free': str(4 * gib),
             'scratch_allocated': str(gib)},
            {'device_total': str(16 * gib), 'pool_used_peak': str(9 * gib),
             'pool_reserved_peak': str(10 * gib), 'device_free': str(5 * gib),
             'scratch_allocated': str(gib // 2)},
        ]
        requests = [{'peak_rss_gb': '12.0'}, {'peak_rss_gb': '11.5'}]
        self.assertEqual(prefill.measured_memory(plan, allocations, requests), {
            'device_peak_bytes': 12 * gib, 'device_total_bytes': 16 * gib,
            'scratch_peak_bytes': gib, 'host_peak_bytes': 12 * gib,
            'host_available_bytes': 25 * gib})

    def test_measured_memory_rejects_missing_malformed_or_unsafe_evidence(self):
        gib = prefill.GIB
        valid_plan = {'device_available': str(15 * gib), 'host_available': str(25 * gib)}
        valid_allocation = {'device_total': str(16 * gib),
                            'pool_used_peak': str(10 * gib),
                            'pool_reserved_peak': str(11 * gib),
                            'device_free': str(4 * gib),
                            'scratch_allocated': str(gib)}
        valid_request = {'peak_rss_gb': '12.0'}
        cases = [
            ({}, [valid_allocation], [valid_request]),
            (valid_plan, [], [valid_request]),
            (valid_plan, [{**valid_allocation, 'device_total': 'NaN'}], [valid_request]),
            (valid_plan, [{**valid_allocation, 'pool_used_peak': str(12 * gib)}], [valid_request]),
            (valid_plan, [{**valid_allocation, 'device_free': str(17 * gib)}], [valid_request]),
            (valid_plan, [valid_allocation], [{'peak_rss_gb': 'nan'}]),
            (valid_plan, [{**valid_allocation, 'scratch_allocated': str(3 * gib)}], [valid_request]),
            ({**valid_plan, 'host_available': str(11 * gib)}, [valid_allocation], [valid_request]),
        ]
        for plan, allocations, requests in cases:
            with self.subTest(plan=plan, allocations=allocations, requests=requests):
                with self.assertRaises(ValueError):
                    prefill.measured_memory(plan, allocations, requests)

    def test_schedule_covers_every_required_size_and_chunk(self):
        expected = [(1024, tuple(prefill.PREFILL_TOKENS))] + [
            (chunk, (2048, 32768)) for chunk in prefill.CHUNKS if chunk != 1024]
        schedule = prefill.prefill_schedule()
        self.assertEqual(schedule, expected)
        self.assertEqual({chunk for chunk, _ in schedule}, set(prefill.CHUNKS))
        self.assertEqual(schedule[0][1], tuple(prefill.PREFILL_TOKENS))


if __name__ == '__main__':
    unittest.main()
