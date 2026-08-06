import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doctor import (  # noqa: E402
    inspect_conversion_ledger,
    inspect_quantization_manifest,
    run_doctor,
)


class DoctorInventoryTests(unittest.TestCase):
    def test_conversion_ledger_hash_and_manifest_cross_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            output = model / "model-00001-of-00001.safetensors"
            output.write_bytes(b"durable-output")
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            signature = {
                "source": "hf://owner/model@revision",
                "source_fingerprint": "f" * 64,
                "xbits": "int4g128",
                "io_bits": 8,
                "shared_bits": 8,
                "group_size": 128,
                "include_mtp": False,
            }
            (model / ".conversion-state.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "signature": signature,
                        "inventory": {"logical.weight": {}},
                        "completed": {
                            "source-00001.safetensors": {
                                "output": output.name,
                                "file_size": output.stat().st_size,
                                "data_bytes": 12,
                                "tensor_count": 3,
                                "sha256": digest,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (model / "quantization.json").write_text(
                json.dumps(
                    {
                        **signature,
                        "complete": True,
                        "source_shards": 1,
                        "output_shards": 1,
                        "logical_tensor_count": 1,
                        "tensor_count": 3,
                        "data_bytes": 12,
                    }
                ),
                encoding="utf-8",
            )
            result = inspect_conversion_ledger(
                model,
                verify_hashes=True,
                expected_source=signature["source"],
                expected_source_fingerprint=signature["source_fingerprint"],
                expected_source_shards=1,
                expected_output_shards=1,
                expected_logical_tensors=1,
                expected_physical_tensors=3,
                expected_data_bytes=12,
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["details"]["hashes_checked"], 1)

            result = inspect_conversion_ledger(
                model,
                expected_source_shards=2,
            )
            self.assertEqual(result["status"], "fail")
            self.assertIn(
                "ledger source_shards 1 != expected 2",
                result["details"]["errors"],
            )

            result = inspect_conversion_ledger(
                model,
                expected_output_shards=2,
                expected_physical_tensors=4,
            )
            self.assertEqual(result["status"], "fail")
            self.assertIn(
                "ledger output_shards 1 != expected 2",
                result["details"]["errors"],
            )
            self.assertIn(
                "ledger physical_tensors 3 != expected 4",
                result["details"]["errors"],
            )

            output.write_bytes(b"corrupt-output")
            result = inspect_conversion_ledger(model, verify_hashes=True)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(
                any(
                    "file_size" in error or "sha256" in error
                    for error in result["details"]["errors"]
                )
            )

    def test_source_only_shard_does_not_disagree_with_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "quantization.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "data_bytes": 100,
                        "tensor_count": 4,
                        "source_shards": 2,
                        "output_shards": 1,
                        "loader_inventory": {"present": 4, "expected": 4},
                    }
                ),
                encoding="utf-8",
            )
            result = inspect_quantization_manifest(
                model,
                {
                    "details": {
                        "tensor_bytes": 100,
                        "header_tensors": 4,
                        "shards": 1,
                    }
                },
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["details"]["source_shards"], 2)
            self.assertEqual(result["details"]["output_shards"], 1)

    def test_single_shard_tiny_container_inventory(self) -> None:
        report = run_doctor(
            ROOT / "qwen_tiny_i4",
            ROOT / "qwen",
            slots=1,
            context=64,
        )
        container = next(
            item for item in report["checks"] if item["id"] == "model.container"
        )
        self.assertEqual(container["status"], "pass")
        self.assertEqual(container["details"]["shards"], 1)
        self.assertGreater(container["details"]["header_tensors"], 0)
        self.assertGreater(container["details"]["tensor_bytes"], 0)
        quantization = next(
            item for item in report["checks"] if item["id"] == "model.quantization"
        )
        self.assertEqual(quantization["status"], "skip")

        unsafe = run_doctor(
            ROOT / "qwen_tiny_i4",
            ROOT / "qwen",
            slots=1,
            context=64,
            cuda_expert_gib=100,
            ram_cache_gib=100,
            runtime_headroom_gib=1,
        )
        resources = next(
            item for item in unsafe["checks"] if item["id"] == "runtime.resources"
        )
        self.assertEqual(resources["status"], "fail")
        self.assertFalse(resources["details"]["checks"]["ram"])
        self.assertEqual(unsafe["status"], "error")

    def test_missing_indexed_shard_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            shutil.copy2(ROOT / "qwen_tiny_i4" / "config.json", model / "config.json")
            shutil.copy2(
                ROOT / "qwen_tiny_i4" / "tokenizer.json",
                model / "tokenizer.json",
            )
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 4},
                        "weight_map": {"missing.weight": "missing.safetensors"},
                    }
                ),
                encoding="utf-8",
            )
            (model / "quantization.json").write_text(
                json.dumps(
                    {
                        "complete": False,
                        "data_bytes": 99,
                        "tensor_count": 1,
                        "source_shards": 1,
                    }
                ),
                encoding="utf-8",
            )
            report = run_doctor(model, ROOT / "qwen", slots=1, context=64)
            container = next(
                item for item in report["checks"] if item["id"] == "model.container"
            )
            quantization = next(
                item for item in report["checks"] if item["id"] == "model.quantization"
            )
            self.assertEqual(container["status"], "fail")
            self.assertEqual(quantization["status"], "fail")
            self.assertIn(
                "complete is not true",
                quantization["details"]["errors"],
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("missing shard", container["details"]["errors"][0])


if __name__ == "__main__":
    unittest.main()
