import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "audit_expert_sidecar",
    TOOLS / "audit_expert_sidecar.py",
)
auditor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = auditor
SPEC.loader.exec_module(auditor)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExpertSidecarAuditTests(unittest.TestCase):
    def make_snapshot(self, root: Path) -> Path:
        snapshot = root / "model"
        snapshot.mkdir()
        config = {
            "hidden_size": 4,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 1,
            "num_experts": 2,
        }
        (snapshot / "config.json").write_text(json.dumps(config))
        weight_map = {}
        for expert in range(2):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.0.mlp.experts.{expert}.{projection}.weight"
                weight_map[name] = "model.safetensors"
        index = snapshot / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": weight_map}))

        records = []
        for expert in range(2):
            filename = f"expert-q3-l000-e{expert:04d}-{expert:04d}.safetensors"
            path = snapshot / filename
            payload = {}
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = (
                    f"model.layers.0.mlp.experts.{expert}.{projection}.weight.q3"
                )
                payload[name] = torch.zeros((4, 3), dtype=torch.uint8)
                payload[f"{name}.qs"] = torch.ones((4, 1), dtype=torch.float32)
                payload[f"{name}.qtype"] = torch.tensor([3], dtype=torch.uint8)
            save_file(payload, path)
            records.append(
                {
                    "file": filename,
                    "layer": 0,
                    "first_expert": expert,
                    "last_expert": expert,
                    "experts": 1,
                    "tensor_count": 9,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        manifest = {
            "format": "colib-routed-expert-int3-sidecar-v1",
            "complete": True,
            "snapshot": str(snapshot),
            "config_sha256": sha256(snapshot / "config.json"),
            "signature": {
                "source_identity": sha256(index),
                "group_size": 8,
                "iterations": 3,
                "experts_per_file": 1,
            },
            "layers": 1,
            "files": records,
            "file_count": 2,
            "tensor_count": 18,
            "data_bytes": sum(row["bytes"] for row in records),
        }
        (snapshot / "expert-q3.json").write_text(json.dumps(manifest))
        return snapshot

    def test_complete_inventory_headers_and_hashes_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.make_snapshot(Path(directory))
            result = auditor.audit(
                snapshot,
                bits=3,
                verify_hashes=True,
                hash_workers=2,
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["file_count"], 2)
            self.assertEqual(result["tensor_count"], 18)
            self.assertEqual(result["headers_verified"], 2)
            self.assertEqual(result["hashes_verified"], 2)
            self.assertEqual(result["hash_workers"], 2)

    def test_hash_worker_count_must_be_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.make_snapshot(Path(directory))
            with self.assertRaisesRegex(ValueError, "hash_workers"):
                auditor.audit(
                    snapshot,
                    bits=3,
                    verify_hashes=True,
                    hash_workers=0,
                )

    def test_tampered_chunk_fails_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.make_snapshot(Path(directory))
            path = snapshot / "expert-q3-l000-e0000-0000.safetensors"
            with path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "ledger mismatch|SHA-256"):
                auditor.audit(snapshot, bits=3, verify_hashes=True)

    def test_incomplete_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.make_snapshot(Path(directory))
            path = snapshot / "expert-q3.json"
            manifest = json.loads(path.read_text())
            manifest["complete"] = False
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                auditor.audit(snapshot, bits=3, verify_hashes=False)


if __name__ == "__main__":
    unittest.main()
