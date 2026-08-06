import importlib.util
import math
import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "bench_cancel_mux",
    TOOLS / "bench_cancel_mux.py",
)
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


class CancellationStatisticsTests(unittest.TestCase):
    def test_nearest_rank_p95_is_deterministic(self) -> None:
        self.assertEqual(
            bench.nearest_rank_percentile(list(range(1, 21)), 0.95),
            19.0,
        )
        self.assertEqual(bench.nearest_rank_percentile([0.4], 0.95), 0.4)

    def test_nearest_rank_rejects_invalid_samples(self) -> None:
        for values in ([], [math.nan], [-0.1]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    bench.nearest_rank_percentile(values, 0.95)


if __name__ == "__main__":
    unittest.main()
