import importlib.util
import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "compare_batch_ab",
    TOOLS / "compare_batch_ab.py",
)
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def artifact(order: tuple[str, str], speedup: float, text: str = "ok") -> dict:
    return {
        "schema_version": 3,
        "model_family": "ornith-1.0",
        "model_manifest": {"source": "ornith-fixed"},
        "prompt": {
            "user_text": "hello",
            "expected_exact_content": "ok",
            "raw": False,
            "rendered_bytes": 12,
        },
        "cuda": True,
        "tier_configuration": {"ram_gb": 18, "cuda_expert_gb": 5},
        "runs": [
            {
                "mode": mode,
                "requests": [{"text": text}, {"text": text}],
            }
            for mode in order
        ],
        "comparison": {
            "aggregate_tps_speedup": speedup,
            "outputs_match": True,
        },
        "acceptance": {"passed": True, "failures": []},
    }


class BatchABComparisonTests(unittest.TestCase):
    def test_reversed_orders_pass_geometric_mean_gate(self) -> None:
        result = comparison.compare(
            artifact(("sequential", "concurrent"), 0.98),
            artifact(("concurrent", "sequential"), 1.04),
            minimum_geomean=1.0,
            minimum_each=0.95,
            expected_source="ornith-fixed",
        )
        self.assertTrue(result["acceptance"]["passed"])
        self.assertGreater(result["speedups"]["geometric_mean"], 1.0)

    def test_wrong_order_or_output_fails(self) -> None:
        ba = artifact(("sequential", "concurrent"), 1.1, text="different")
        result = comparison.compare(
            artifact(("sequential", "concurrent"), 1.1),
            ba,
            minimum_geomean=1.0,
            minimum_each=0.95,
            expected_source="ornith-fixed",
        )
        self.assertFalse(result["acceptance"]["passed"])
        self.assertTrue(
            any("mode order" in failure for failure in result["acceptance"]["failures"])
        )
        self.assertTrue(
            any(
                "deterministic outputs" in failure
                for failure in result["acceptance"]["failures"]
            )
        )

    def test_wrong_family_or_source_fails(self) -> None:
        ab = artifact(("sequential", "concurrent"), 1.1)
        ba = artifact(("concurrent", "sequential"), 1.1)
        ab["model_family"] = ba["model_family"] = "qwen3.5"
        result = comparison.compare(
            ab,
            ba,
            minimum_geomean=1.0,
            minimum_each=0.95,
            expected_source="other-source",
        )
        self.assertFalse(result["acceptance"]["passed"])
        self.assertTrue(
            any("model family" in failure for failure in result["acceptance"]["failures"])
        )
        self.assertTrue(
            any("model source" in failure for failure in result["acceptance"]["failures"])
        )


if __name__ == "__main__":
    unittest.main()
