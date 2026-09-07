import importlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
converter = importlib.import_module("convert_deepseek_v4")
validator = importlib.import_module("validate_deepseek_v4_conversion")


def write_safetensors(path: Path, tensors):
    header = {}
    payload = bytearray()
    for name, (dtype, shape, value) in tensors.items():
        start = len(payload)
        payload.extend(value)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def make_fixture(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    shard = source / "model-00001-of-00001.safetensors"
    tensors = {
        "model.embed_tokens.weight": ("U8", [16], bytes(range(16))),
        "model.layers.42.mtp.proj.weight": ("U8", [4], b"MTP!"),
    }
    for expert in (0, 1):
        for projection, byte in (
            ("gate_proj", 10 + expert),
            ("up_proj", 20 + expert),
            ("down_proj", 30 + expert),
        ):
            name = f"model.layers.0.mlp.experts.{expert}.{projection}.weight"
            tensors[name] = ("F4", [4], bytes([byte]) * 4)
    write_safetensors(shard, tensors)
    index = {"weight_map": {name: shard.name for name in tensors}}
    (source / converter.INDEX_FILE).write_text(json.dumps(index))
    (source / "config.json").write_text("{}")
    (source / "tokenizer.json").write_text('{"fixture":true}')
    return source


class DeepSeekConversionValidatorTests(unittest.TestCase):
    def convert_fixture(self, root: Path):
        source = make_fixture(root)
        model = root / "model"
        converter.convert(
            source,
            model,
            alignment=16,
            min_final_free=0,
            allow_fixture=True,
            command="fixture conversion",
        )
        return source, model

    def test_validates_every_record_and_resumes_by_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, model = self.convert_fixture(root)
            evidence = root / "validation.json"
            first = validator.validate(
                source,
                model,
                evidence,
                allow_fixture=True,
                command="fixture validation",
                stop_after_groups=1,
            )
            self.assertEqual(first["status"], "running")
            self.assertEqual(len(first["completed"]), 1)
            final = validator.validate(
                source,
                model,
                evidence,
                allow_fixture=True,
                command="ignored on resume",
            )
            self.assertEqual(final["status"], "complete")
            self.assertEqual(final["command"], "fixture validation")
            self.assertEqual(final["totals"]["records"], 8)
            self.assertEqual(final["totals"]["source_bytes_compared"], 44)
            self.assertEqual(
                set(final["binding"]["dependency_sha256"]),
                {
                    "convert_deepseek_v4.py",
                    "deepseek_v4_layout.py",
                    "deepseek_v4_spec.py",
                    "runtime_env.py",
                    "validate_deepseek_v4_conversion.py",
                },
            )

    def test_completed_segment_identity_drift_forces_revalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, model = self.convert_fixture(root)
            evidence = root / "validation.json"
            first = validator.validate(
                source,
                model,
                evidence,
                allow_fixture=True,
                stop_after_groups=1,
            )
            group = next(iter(first["completed"]))
            path = model / group
            value = bytearray(path.read_bytes())
            value[-1] ^= 1
            path.write_bytes(value)
            with self.assertRaisesRegex(ValueError, "native-byte mismatch|record hash mismatch"):
                validator.validate(source, model, evidence, allow_fixture=True)


class DeepSeekValidatorCliTests(unittest.TestCase):
    def test_help_invokes_main(self):
        result = subprocess.run(
            [sys.executable, str(TOOLS / "validate_deepseek_v4_conversion.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source", result.stdout)


if __name__ == "__main__":
    unittest.main()
