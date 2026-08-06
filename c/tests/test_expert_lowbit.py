import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.expert_lowbit import load_complete_expert_sidecar


class ExpertLowbitManifestTests(unittest.TestCase):
    def write_manifest(self, model: Path, **updates):
        model.mkdir(parents=True, exist_ok=True)
        value = {
            "format": "colib-routed-expert-int3-sidecar-v1",
            "complete": True,
            "config_sha256": "a" * 64,
            "signature": {"group_size": 128},
            "layers": 2,
            "file_count": 4,
            "tensor_count": 18,
            "data_bytes": 1024,
            **updates,
        }
        (model / "expert-q3.json").write_text(json.dumps(value))
        return value

    def test_complete_manifest_is_summarized_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            value = self.write_manifest(model)
            result = load_complete_expert_sidecar(model, bits=3)
            self.assertEqual(result["format"], value["format"])
            self.assertEqual(result["file_count"], 4)
            self.assertEqual(len(result["manifest_sha256"]), 64)

    def test_incomplete_or_wrong_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            self.write_manifest(model, complete=False)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_complete_expert_sidecar(model, bits=3)
            self.write_manifest(model, format="wrong", complete=True)
            with self.assertRaisesRegex(ValueError, "format"):
                load_complete_expert_sidecar(model, bits=3)


if __name__ == "__main__":
    unittest.main()
