"""Acceptance findings must be complete, concrete and bound to the planted file."""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.qualify_deepseek_v4_reviews import CASES, assess


class ReviewOracleTests(unittest.TestCase):
    def setUp(self):
        self.case = CASES[0]
        self.answer = {'verdict': 'REJECT', 'issues': [
            {'path': 'lookup.py', 'line': 4,
             'explanation': 'An index equal to len(items) passes the bound and raises IndexError.'}]}

    def completion(self, answer):
        return 'The upper bound includes the invalid terminal index.\n</think>' + json.dumps(answer)

    def test_complete_concrete_finding_passes(self):
        passed, parsed, _ = assess(self.completion(self.answer), self.case)
        self.assertTrue(passed)
        self.assertIn('upper bound', parsed['reasoning_content'])
        self.assertIn('IndexError', parsed['content'])

    def test_wrong_location_or_semantics_fails(self):
        for change in ({'path': 'another.py'}, {'line': 3}, {'line': True},
                       {'explanation': 'There may be a style problem.'}):
            answer = copy.deepcopy(self.answer)
            answer['issues'][0].update(change)
            self.assertFalse(assess(self.completion(answer), self.case)[0])

    def test_approval_malformed_and_thinking_only_fail(self):
        for value in ('unfinished reasoning', '</think>{"verdict":',
                      '</think>[]', '</think>{"verdict":"REJECT","issues":null}',
                      self.completion({**self.answer, 'verdict': 'APPROVE'})):
            self.assertFalse(assess(value, self.case)[0])

    def test_five_distinct_cases_include_exact_long_input(self):
        self.assertEqual(len({case['id'] for case in CASES}), 5)
        self.assertEqual(max(case['target'] for case in CASES), 32768)
        for case in CASES:
            self.assertLessEqual(case['line'], len(case['source'].splitlines()))


if __name__ == '__main__':
    unittest.main()
