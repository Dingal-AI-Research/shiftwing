import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
RUNNER = TOOLS / "verify_pinned_ornith397.py"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PinnedVerificationTests(unittest.TestCase):
    def build(self, root: Path) -> tuple[Path, Path, Path]:
        """Create a two-shard snapshot plus its preserved manifests."""

        snapshot = root / "snapshot"
        preserved = root / "preserved"
        snapshot.mkdir()
        preserved.mkdir()

        shards = {}
        completed = {}
        for index in (1, 2):
            name = f"model-{index:05d}-of-00002.safetensors"
            payload = f"shard-{index}".encode() * 16
            (snapshot / name).write_bytes(payload)
            record = {
                "output": name,
                "sha256": digest(payload),
                "file_size": len(payload),
                "data_bytes": len(payload) - 8,
                "tensor_count": 10 * index,
            }
            shards[name] = record
            completed[f"source-{index}"] = record

        signature = {
            "source": "hf://fixture@rev",
            "source_fingerprint": "fixture-fingerprint",
            "xbits": "int4g128",
            "io_bits": 8,
            "shared_bits": 8,
            "group_size": 128,
            "include_mtp": False,
        }
        ledger = {"completed": completed, "signature": signature, "version": 1}
        ledger_text = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
        (snapshot / ".conversion-state.json").write_text(ledger_text)

        quantization = {
            "complete": True,
            "data_bytes": sum(r["data_bytes"] for r in shards.values()),
            "tensor_count": sum(r["tensor_count"] for r in shards.values()),
            "logical_tensor_count": 7,
            "output_shards": 2,
        }
        quantization_text = json.dumps(quantization, indent=2, sort_keys=True) + "\n"
        (snapshot / "quantization.json").write_text(quantization_text)

        base_manifest = preserved / "base-shards.json"
        base_manifest.write_text(
            json.dumps(
                {
                    "schema": "colib.ornith397-base-shards.v1",
                    "conversion_state_sha256": digest(ledger_text.encode()),
                    "quantization_sha256": digest(quantization_text.encode()),
                    "signature": signature,
                    "shards": shards,
                    "totals": {
                        "data_bytes": quantization["data_bytes"],
                        "tensor_count": quantization["tensor_count"],
                        "logical_tensor_count": 7,
                        "output_shards": 2,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )

        config = json.dumps({"num_hidden_layers": 1}, sort_keys=True)
        (snapshot / "config.json").write_text(config)
        files = []
        for first in (0, 64):
            name = f"expert-q3-l000-e{first:04d}-{first + 63:04d}.safetensors"
            payload = f"sidecar-{first}".encode() * 8
            (snapshot / name).write_bytes(payload)
            files.append(
                {
                    "file": name,
                    "sha256": digest(payload),
                    "bytes": len(payload),
                    "tensor_count": 576,
                    "layer": 0,
                    "experts": 64,
                    "first_expert": first,
                    "last_expert": first + 63,
                }
            )
        sidecar = {
            "format": "colib-routed-expert-int3-sidecar-v1",
            "complete": True,
            "config_sha256": digest(config.encode()),
            "layers": 1,
            "file_count": 2,
            "tensor_count": 1152,
            "data_bytes": sum(entry["bytes"] for entry in files),
            "signature": {
                "experts_per_file": 64,
                "group_size": 128,
                "iterations": 3,
                "source_identity": "fixture-identity",
            },
            "files": files,
        }
        (snapshot / "expert-q3.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True)
        )
        sidecar_manifest = preserved / "expert-q3.json"
        sidecar_manifest.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
        return snapshot, base_manifest, sidecar_manifest

    def invoke(
        self,
        snapshot: Path,
        *manifests: Path,
        rehash: bool = True,
        output: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        argv = [sys.executable, str(RUNNER), "--snapshot", str(snapshot)]
        base, sidecar = manifests
        if base is not None:
            argv.extend(["--base-manifest", str(base)])
        if sidecar is not None:
            argv.extend(["--sidecar-manifest", str(sidecar)])
        if rehash:
            argv.append("--rehash")
        if output is not None:
            argv.extend(["--output", str(output)])
        return subprocess.run(argv, text=True, capture_output=True, check=False)

    def test_matching_snapshot_passes_and_publishes_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, base, sidecar = self.build(root)
            artifact = root / "verification.json"
            result = self.invoke(snapshot, base, sidecar, output=artifact)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(artifact.read_text())
            self.assertTrue(report["passed"])
            self.assertTrue(report["base"]["passed"])
            self.assertTrue(report["expert_q3"]["passed"])
            self.assertEqual(report["base"]["records_matched"], 2)
            self.assertEqual(report["base"]["physical"]["hash_verified"], 2)
            self.assertEqual(report["expert_q3"]["physical"]["hash_verified"], 2)
            self.assertEqual(report["base"]["failures"], [])

    def test_corrupted_shard_body_fails_only_under_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, base, sidecar = self.build(root)
            target = snapshot / "model-00001-of-00002.safetensors"
            payload = bytearray(target.read_bytes())
            payload[0] ^= 0xFF
            target.write_bytes(bytes(payload))
            without = self.invoke(snapshot, base, None, rehash=False)
            self.assertEqual(without.returncode, 0, without.stderr)
            with_rehash = self.invoke(snapshot, base, None)
            self.assertEqual(with_rehash.returncode, 1)
            self.assertIn("hashes to", with_rehash.stdout)

    def test_truncated_shard_fails_on_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, base, sidecar = self.build(root)
            target = snapshot / "model-00002-of-00002.safetensors"
            target.write_bytes(target.read_bytes()[:-4])
            result = self.invoke(snapshot, base, None, rehash=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("bytes, expected", result.stdout)

    def test_ledger_drift_fails_even_when_files_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, base, sidecar = self.build(root)
            ledger_path = snapshot / ".conversion-state.json"
            ledger = json.loads(ledger_path.read_text())
            ledger["signature"]["xbits"] = "int8"
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True))
            result = self.invoke(snapshot, base, None)
            self.assertEqual(result.returncode, 1)
            self.assertIn("live ledger hashes to", result.stdout)
            self.assertIn("ledger signature xbits", result.stdout)

    def test_missing_shard_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, base, sidecar = self.build(root)
            (snapshot / "model-00001-of-00002.safetensors").unlink()
            result = self.invoke(snapshot, base, None)
            self.assertEqual(result.returncode, 1)
            self.assertIn("is absent from the snapshot", result.stdout)

    def test_sidecar_file_count_drift_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, base, sidecar = self.build(root)
            live_path = snapshot / "expert-q3.json"
            live = json.loads(live_path.read_text())
            live["file_count"] = 3
            live["files"] = live["files"][:1]
            live_path.write_text(json.dumps(live, indent=2, sort_keys=True))
            result = self.invoke(snapshot, None, sidecar)
            self.assertEqual(result.returncode, 1)
            self.assertIn("sidecar file_count is 3", result.stdout)
            self.assertIn("is missing from the snapshot", result.stdout)

    def test_incomplete_quantization_manifest_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, base, sidecar = self.build(root)
            path = snapshot / "quantization.json"
            value = json.loads(path.read_text())
            value["complete"] = False
            path.write_text(json.dumps(value, indent=2, sort_keys=True))
            result = self.invoke(snapshot, base, None, rehash=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("quantization manifest is incomplete", result.stdout)

    def test_a_manifest_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, _, _ = self.build(root)
            result = subprocess.run(
                [sys.executable, str(RUNNER), "--snapshot", str(snapshot)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("at least one of", result.stderr)


if __name__ == "__main__":
    unittest.main()
