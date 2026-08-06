import importlib.util
import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "smoke_web_qwen",
    TOOLS / "smoke_web_qwen.py",
)
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


class ProductionWebAcceptanceTests(unittest.TestCase):
    def test_complete_cuda_lifecycle_passes(self) -> None:
        failures = smoke.acceptance_failures(
            [
                {
                    "index": 0,
                    "content": "colib ready",
                    "usage": {"completion_tokens": 3},
                },
                {
                    "index": 1,
                    "content": "colib ready",
                    "usage": {"completion_tokens": 3},
                },
            ],
            {
                "scheduler": {
                    "active": 0,
                    "queued": 0,
                    "completed": 2,
                },
                "hwinfo": {"gpus": 1},
                "tiers": {"vram": 8},
            },
            {
                "resident": {
                    "layers": 80,
                    "device_moe": 80,
                    "host_moe": 0,
                    "router_d2h_bytes": 1024,
                    "logits_d2h_bytes": 1024,
                }
            },
            page_ok=True,
            expected_content="colib ready",
            cuda=True,
            expected_requests=2,
        )
        self.assertEqual(failures, [])

    def test_wrong_content_and_leaked_slot_fail(self) -> None:
        failures = smoke.acceptance_failures(
            [
                {
                    "index": 0,
                    "content": "almost ready",
                    "usage": None,
                }
            ],
            {
                "scheduler": {
                    "active": 1,
                    "queued": 0,
                    "completed": 0,
                }
            },
            {},
            page_ok=False,
            expected_content="colib ready",
            cuda=True,
            expected_requests=1,
        )
        self.assertGreaterEqual(len(failures), 7)


if __name__ == "__main__":
    unittest.main()
