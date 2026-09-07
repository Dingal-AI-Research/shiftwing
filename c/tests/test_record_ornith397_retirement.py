import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from record_ornith397_retirement import verify_model, verify_trials


def encoded(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class Ornith397RetirementRecordTests(unittest.TestCase):
    def test_trials_verify_hashes_and_ignore_available_ram_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = root / "engine"
            binding.write_bytes(b"engine")
            completed = {}
            for trial, available in ((1, 2.3), (2, 2.2)):
                result = {
                    "configuration": {"threads": 8},
                    "hardware": {"gpu": "fixture", "ram_avail_gb": available},
                    "model_manifest": {"source": "pinned"},
                    "expert_lowbit_manifest": {"complete": True},
                    "summary": {"sustained_tps": 1.0 + trial / 10},
                    "measured": [
                        {
                            "prompt_index": 0,
                            "ttft_s": 2.0,
                            "wall_s": 3.0,
                            "stats": {
                                "completion_tokens": 2,
                                "tokens_per_second": 1.0,
                            },
                        }
                    ],
                    "acceptance": {"passed": True},
                }
                path = root / f"trial-{trial}.json"
                payload = encoded(result)
                path.write_bytes(payload)
                completed[str(trial)] = {
                    "output": str(path),
                    "bytes": len(payload),
                    "sha256": digest(payload),
                }
            state = {
                "schema": "colib.performance-trials.v1",
                "status": "complete",
                "signature": {
                    "trials": 2,
                    "bindings": [
                        {
                            "path": str(binding),
                            "bytes": binding.stat().st_size,
                            "sha256": digest(binding.read_bytes()),
                        }
                    ],
                },
                "completed": completed,
            }
            state_path = root / "state.json"
            state_path.write_bytes(encoded(state))
            loaded, rows, state_hash = verify_trials(state_path)
            self.assertEqual(loaded["status"], "complete")
            self.assertEqual(len(rows), 2)
            self.assertEqual(state_hash, digest(state_path.read_bytes()))
            binding.write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "binding drifted"):
                verify_trials(state_path)

    def test_model_manifest_totals_are_cross_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            quantization = {
                "complete": True,
                "source": "hf://fixture@commit",
                "source_fingerprint": "fingerprint",
                "output_shards": 1,
                "data_bytes": 10,
                "tensor_count": 2,
                "logical_tensor_count": 1,
            }
            conversion = {
                "version": 1,
                "signature": {
                    "source": "hf://fixture@commit",
                    "source_fingerprint": "fingerprint",
                },
                "completed": {
                    "model.safetensors": {
                        "output": "model.safetensors",
                        "data_bytes": 10,
                        "file_size": 12,
                        "tensor_count": 2,
                        "sha256": "a" * 64,
                    }
                },
                "inventory": {"tensor": {}},
            }
            q3 = {
                "complete": True,
                "file_count": 1,
                "data_bytes": 5,
                "files": [
                    {
                        "file": "expert.safetensors",
                        "bytes": 5,
                        "sha256": "b" * 64,
                    }
                ],
            }
            (model / "quantization.json").write_bytes(encoded(quantization))
            (model / ".conversion-state.json").write_bytes(encoded(conversion))
            (model / "expert-q3.json").write_bytes(encoded(q3))
            (model / "config.json").write_text("{}\n")
            base, manifests = verify_model(model)
            self.assertEqual(base["totals"]["file_bytes"], 12)
            self.assertEqual(len(base["shards"]), 1)
            self.assertEqual(manifests["expert_q3"]["file_count"], 1)
            quantization["data_bytes"] = 11
            (model / "quantization.json").write_bytes(encoded(quantization))
            with self.assertRaisesRegex(ValueError, "data_bytes"):
                verify_model(model)


if __name__ == "__main__":
    unittest.main()
