"""Accounting and scoring regressions for the long GPU experiment."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark_deepseek_v4_thinking import Timeline, assess
from tools.qualify_deepseek_v4_reviews import CASES


class ThinkingExperimentTests(unittest.TestCase):
    def setUp(self):
        self.clock = 0
        self.answer = json.dumps({'verdict': 'REJECT', 'issues': [
            {'path': 'lookup.py', 'line': 4,
             'explanation': 'Index equal to len(items) passes the bound and raises IndexError.'}]})

    def timeline(self, mode):
        return Timeline(mode, now=lambda: self.clock)

    def token(self, timeline, text, seconds):
        self.clock = seconds
        timeline.on_text(text)
        timeline.on_progress({'phase': 'decode', 'completion_tokens': timeline.generated + 1,
                              'elapsed_ms': int(seconds * 1000)})

    def test_off_is_valid_framing_but_approval_misses_bug(self):
        self.assertTrue(assess(self.answer, CASES[0], 'off')[0])
        self.assertFalse(assess('{"verdict":"APPROVE","issues":[]}', CASES[0], 'off')[0])
        self.assertFalse(assess(self.answer, CASES[0], 'low')[0])

    def test_all_thinking_efforts_require_complete_final(self):
        for mode in ('low', 'high', 'max'):
            self.assertTrue(assess('Reviewing. </think>' + self.answer, CASES[0], mode)[0])
            self.assertFalse(assess('Reviewing. ' + self.answer, CASES[0], mode)[0])
            self.assertFalse(assess('Reviewing. </think>{"verdict":', CASES[0], mode)[0])

    def test_native_counts_separate_marker_and_eos(self):
        line = self.timeline('high')
        self.token(line, 'Review', 10)
        self.token(line, '.', 12)
        self.token(line, '</think>', 14)
        self.token(line, '\n', 16)
        self.token(line, '{', 18)
        self.clock = 20
        metrics = line.metrics(6, eos=True)
        self.assertEqual(metrics['thinking_tokens'], 2)
        self.assertEqual(metrics['framing_tokens'], 1)
        self.assertEqual(metrics['final_tokens'], 2)
        self.assertEqual(metrics['eos_tokens'], 1)
        self.assertEqual(metrics['ttft_s'], 10)
        self.assertEqual(metrics['first_final_s'], 18)
        self.assertEqual(sum(metrics[k] for k in ('thinking_tokens', 'framing_tokens', 'final_tokens', 'eos_tokens')), 6)

    def test_no_thinking_and_partial_thinking(self):
        off = self.timeline('off')
        self.token(off, '{', 2)
        self.assertEqual(off.metrics(2, True)['thinking_tokens'], 0)
        self.assertEqual(off.metrics(2, True)['final_tokens'], 1)
        partial = self.timeline('max')
        self.token(partial, 'Review', 4)
        self.assertEqual(partial.metrics()['thinking_tokens'], 1)
        self.assertEqual(partial.metrics()['final_tokens'], 0)
        self.assertIsNone(partial.metrics()['first_final_s'])

    def test_broken_native_count_evidence_is_rejected(self):
        line = self.timeline('low')
        with self.assertRaises(ValueError):
            line.on_progress({'phase': 'decode', 'completion_tokens': 2, 'elapsed_ms': 0})
        self.token(line, 'Review', 2)
        with self.assertRaises(ValueError):
            line.metrics(4, True)


if __name__ == '__main__':
    unittest.main()
