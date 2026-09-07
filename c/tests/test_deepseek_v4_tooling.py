import hashlib
import importlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.import_module("deepseek_v4_spec")
preflight = importlib.import_module("preflight_deepseek_v4")
converter = importlib.import_module("convert_deepseek_v4")
fetcher = importlib.import_module("fetch_deepseek_v4")
inventory_tool = importlib.import_module("inventory_deepseek_v4")
layout = importlib.import_module("deepseek_v4_layout")


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]):
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


class DeepSeekSpecTests(unittest.TestCase):
    def test_context_reasoning_and_sampling_contracts(self):
        self.assertEqual(spec.validate_context(16_384), 16_384)
        self.assertEqual(spec.validate_context(65_536), 65_536)
        with self.assertRaisesRegex(ValueError, "validated maximum"):
            spec.validate_context(65_537)
        self.assertIsNone(spec.validate_reasoning_effort(None))
        self.assertEqual(spec.validate_reasoning_effort("high"), "high")
        for value in ("low", "max"):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                spec.validate_reasoning_effort(value)
        self.assertEqual(spec.sampling_profile(0, 1).name, "deterministic")
        self.assertEqual(spec.sampling_profile(1, 0.95).name, "agent")
        with self.assertRaisesRegex(ValueError, "supports only"):
            spec.sampling_profile(0.7, 0.9)

    def test_pinned_config_contract_reports_all_drift(self):
        config = dict(spec.EXPECTED_CONFIG)
        config["quantization_config"] = dict(spec.EXPECTED_QUANTIZATION)
        config["rope_scaling"] = dict(spec.EXPECTED_ROPE_SCALING)
        config["compress_ratios"] = list(spec.EXPECTED_COMPRESS_RATIOS)
        self.assertEqual(spec.validate_config(config), [])
        config["expert_dtype"] = "int4"
        config["quantization_config"]["scale_fmt"] = "wrong"
        config["compress_ratios"] = []
        failures = spec.validate_config(config)
        self.assertEqual(len(failures), 3)


class DeepSeekLayoutTests(unittest.TestCase):
    def test_exact_pinned_layout_and_drift_rejection(self):
        expected = layout.expected_layout()
        self.assertEqual(len(expected), 72_317)
        self.assertEqual(
            expected["mtp.2.confidence_head.proj.weight"],
            ("BF16", (1, 4352)),
        )
        self.assertEqual(
            layout.category_counts(expected),
            {
                "dense": 1_564,
                "dspark_dense": 97,
                "dspark_expert": 4_608,
                "routed_expert": 66_048,
            },
        )
        self.assertEqual(layout.validate_descriptors(expected), [])
        drifted = dict(expected)
        drifted["layers.0.attn.wq_a.weight"] = ("F8_E4M3", (1024, 2048))
        failures = layout.validate_descriptors(drifted)
        self.assertTrue(any("descriptor mismatches" in item for item in failures))
        del drifted["hc_head_fn"]
        failures = layout.validate_descriptors(drifted)
        self.assertTrue(any("missing 1 tensors" in item for item in failures))

    def test_native_contract_accepts_all_pinned_descriptors(self):
        c_root = Path(__file__).resolve().parents[1]
        binary = c_root / "tests" / "test_deepseek_v4_contract"
        subprocess.run(
            ["make", "tests/test_deepseek_v4_contract", "ARCH=x86-64-v3"],
            cwd=c_root,
            check=True,
            capture_output=True,
            text=True,
        )
        dtype_bytes = {
            "BF16": 2,
            "F32": 4,
            "F8_E4M3": 1,
            "F8_E8M0": 1,
            "I8": 1,
            "I64": 8,
        }
        records = []
        offset = 0
        for name, (dtype, shape) in layout.expected_layout().items():
            elements = 1
            for dimension in shape:
                elements *= dimension
            nbytes = elements * dtype_bytes[dtype]
            records.append(
                {
                    "name": name,
                    "file": "layout.bin",
                    "offset": offset,
                    "nbytes": nbytes,
                    "dtype": dtype,
                    "shape": list(shape),
                }
            )
            offset += nbytes
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "layout.bin").open("wb") as segment:
                segment.truncate(offset)
            manifest = {
                "schema": "colib.deepseek-v4.model-manifest.v1",
                "source": {"revision": spec.SOURCE_REVISION},
                "inventory": {"layout.bin": records},
            }
            manifest_path = root / "model-manifest.json"
            manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
            accepted = subprocess.run(
                [str(binary), str(root)], capture_output=True, text=True
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn("tensors=72317 layers=43 dspark=3", accepted.stdout)

            records[0]["name"] = "broken.embed.weight"
            manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
            rejected = subprocess.run(
                [str(binary), str(root)], capture_output=True, text=True
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("tensor contract mismatch: embed.weight", rejected.stderr)


class DeepSeekPreflightTests(unittest.TestCase):
    def test_storage_gate_and_secret_redaction(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {
                "COLI_CUDA": "1",
                "HF_HUB_TOKEN": "must-not-leak",
                "CUDA_VISIBLE_DEVICES": "0",
            },
            clear=False,
        ):
            root = Path(tmp)
            accepted = preflight.make_report(
                root,
                root / "model",
                1,
                1,
                1,
                0,
                ["preflight", "--target", "model"],
            )
            self.assertTrue(accepted["accepted"])
            self.assertIsNone(accepted["repository"]["deepseek_engine_source_sha256"])
            self.assertTrue(
                accepted["repository"]["deepseek_engine_source_status"].startswith("missing:")
            )
            variables = accepted["environment"]["variables"]
            self.assertEqual(variables["COLI_CUDA"], "1")
            self.assertNotIn("HF_HUB_TOKEN", variables)
            free = accepted["storage"]["free_before_bytes"]
            rejected = preflight.make_report(
                root, root / "model", free, 1, 1, 1, ["preflight"]
            )
            self.assertFalse(rejected["accepted"])


class DeepSeekConverterTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        shard = source / "model-00001-of-00001.safetensors"
        tensors: dict[str, tuple[str, list[int], bytes]] = {
            "model.embed_tokens.weight": ("U8", [16], bytes(range(16))),
            "hc_head_fn": ("F32", [1], b"HEAD"),
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
        for projection, byte in (
            ("gate_proj", 40),
            ("up_proj", 50),
            ("down_proj", 60),
        ):
            name = f"model.layers.1.mlp.experts.{projection}.weight"
            tensors[name] = ("F4", [256, 1], bytes([byte]) * 256)
        write_safetensors(shard, tensors)
        index = {"weight_map": {name: shard.name for name in tensors}}
        (source / "model.safetensors.index.json").write_text(json.dumps(index))
        (source / "config.json").write_text("{}")
        (source / "tokenizer.json").write_text('{"fixture":true}')
        return source

    def test_plan_splits_fused_experts_and_separates_dspark(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_fixture(Path(tmp))
            groups, plan = converter.build_plan(source, allow_fixture=True)
            self.assertIn("dense/model-00001-of-00001.bin", groups)
            self.assertIn("dspark/model-00001-of-00001.bin", groups)
            dense_names = {record.name for record in groups["dense/model-00001-of-00001.bin"]}
            dspark_names = {record.name for record in groups["dspark/model-00001-of-00001.bin"]}
            self.assertIn("hc_head_fn", dense_names)
            self.assertNotIn("hc_head_fn", dspark_names)
            self.assertEqual(len(groups["experts/layer-00.bin"]), 6)
            self.assertEqual(len(groups["experts/layer-01.bin"]), 3 * 256)
            first = groups["experts/layer-00.bin"][:3]
            self.assertEqual([record.expert for record in first], [0, 0, 0])
            self.assertEqual(
                [record.projection for record in first],
                ["gate_proj", "up_proj", "down_proj"],
            )
            self.assertEqual(plan["source"]["revision"], "fixture")

    def test_interruption_resume_and_hash_bound_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_fixture(root)
            output = root / "output"
            first = converter.convert(
                source,
                output,
                alignment=16,
                min_final_free=0,
                allow_fixture=True,
                command="fixture conversion",
                stop_after_groups=1,
            )
            self.assertEqual(first["status"], "running")
            completed_name = next(iter(first["completed"]))
            completed_path = output / completed_name
            completed_hash = hashlib.sha256(completed_path.read_bytes()).hexdigest()

            final = converter.convert(
                source,
                output,
                alignment=16,
                min_final_free=0,
                allow_fixture=True,
                command="ignored on resume",
            )
            self.assertEqual(final["status"], "complete")
            self.assertEqual(
                hashlib.sha256(completed_path.read_bytes()).hexdigest(), completed_hash
            )
            manifest_path = output / converter.MANIFEST_FILE
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["schema"], converter.MANIFEST_SCHEMA)
            self.assertEqual(manifest["command"], "fixture conversion")
            self.assertEqual(
                final["manifest"]["sha256"], converter.sha256_file(manifest_path)
            )
            self.assertEqual(
                manifest["packing"]["weight_transform"],
                "none (native bytes preserved)",
            )
            self.assertTrue((output / "tokenizer.json").is_file())

    def test_resume_rejects_changed_completed_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_fixture(root)
            output = root / "output"
            state = converter.convert(
                source,
                output,
                alignment=16,
                min_final_free=0,
                allow_fixture=True,
                stop_after_groups=1,
            )
            name = next(iter(state["completed"]))
            with (output / name).open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "size changed"):
                converter.convert(
                    source,
                    output,
                    alignment=16,
                    min_final_free=0,
                    allow_fixture=True,
                )


    def test_real_plan_rejects_descriptor_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_fixture(Path(tmp))
            (source / converter.SOURCE_MARKER).write_text(
                json.dumps(
                    {
                        "repository": spec.SOURCE_REPO,
                        "revision": spec.SOURCE_REVISION,
                    }
                )
            )
            with (
                mock.patch.object(converter, "WEIGHT_SHARDS", 1),
                mock.patch.object(converter, "validate_config", return_value=[]),
                mock.patch.object(
                    converter, "validate_records", return_value=["forced drift"]
                ) as validate,
            ):
                with self.assertRaisesRegex(
                    ValueError, "pinned tensor layout validation failed: forced drift"
                ):
                    converter.build_plan(source, allow_fixture=False)
            validate.assert_called_once()

class DeepSeekFetcherTests(unittest.TestCase):
    def inventory(self):
        shards = [
            f"model-{index:05d}-of-00048.safetensors"
            for index in range(1, 49)
        ]
        return {
            "sha": spec.SOURCE_REVISION,
            "colib_files": sorted(
                set(fetcher.REQUIRED_METADATA)
                | {"encoding/encoding_dsv4.py", "inference/model.py"}
                | set(shards)
            ),
        }

    def test_inventory_contract_requires_metadata_and_48_shards(self):
        inventory = self.inventory()
        selected = fetcher.metadata_files(inventory["colib_files"])
        self.assertTrue(set(fetcher.REQUIRED_METADATA).issubset(selected))
        shards = [
            name
            for name in inventory["colib_files"]
            if name.endswith(".safetensors")
        ]
        index = {
            "weight_map": {
                f"tensor.{number}": name
                for number, name in enumerate(shards)
            }
        }
        self.assertEqual(len(fetcher.shard_files(index)), 48)
        with self.assertRaisesRegex(ValueError, "48"):
            fetcher.shard_files({"weight_map": {"x": shards[0]}})
        with self.assertRaises(ValueError):
            fetcher.safe_relative("../escape")

    def test_metadata_to_full_fetch_is_resumable_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "source"
            inventory = self.inventory()
            shards = [
                name
                for name in inventory["colib_files"]
                if name.endswith(".safetensors")
            ]
            index = {
                "weight_map": {
                    f"tensor.{number}": name
                    for number, name in enumerate(shards)
                }
            }

            def fake_download(url, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.name == "model.safetensors.index.json":
                    value = json.dumps(index).encode()
                else:
                    value = path.as_posix().encode()
                path.write_bytes(value)
                return {
                    "size": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                    "etag": "fixture",
                    "url": url,
                    "resolved_url": url,
                    "completed_at": "fixture",
                }

            with mock.patch.object(
                fetcher, "download_one", side_effect=fake_download
            ):
                metadata_state = fetcher.fetch(
                    output, metadata_only=True, api=inventory
                )
                self.assertEqual(metadata_state["status"], "metadata_complete")
                metadata_hashes = dict(metadata_state["completed"])
                stale = json.loads((output / fetcher.STATE_FILE).read_text())
                stale["status"] = "running"
                stale["attempts"][-1]["status"] = "running"
                fetcher.atomic_json(output / fetcher.STATE_FILE, stale)

                final_state = fetcher.fetch(
                    output, metadata_only=False, api=inventory, workers=4
                )
            self.assertEqual(final_state["status"], "complete")
            self.assertEqual(len(final_state["attempts"]), 2)
            self.assertEqual(final_state["attempts"][0]["status"], "interrupted")
            self.assertIn("stale running",
                          final_state["attempts"][0]["reason"])
            self.assertEqual(final_state["attempts"][1]["workers"], 4)
            self.assertEqual(final_state["attempts"][1]["completed_before"],
                             len(metadata_hashes))
            self.assertEqual(final_state["attempts"][1]["completed_after"], 55)
            with self.assertRaisesRegex(ValueError, "workers"):
                fetcher.fetch(output, api=inventory, workers=0)
            self.assertEqual(len(final_state["weight_shards"]), 48)
            for name, evidence in metadata_hashes.items():
                self.assertEqual(final_state["completed"][name], evidence)
            marker = json.loads((output / fetcher.SOURCE_MARKER).read_text())
            self.assertEqual(marker["revision"], spec.SOURCE_REVISION)

    def test_failure_attempt_records_errno_without_error_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            state = {
                "status": "running",
                "completed": {"one": {"sha256": "fixture"}},
                "attempts": [{"attempt": 1, "status": "running"}],
            }
            fetcher.atomic_json(output / fetcher.STATE_FILE, state)
            fetcher._record_attempt_failure(
                output, OSError(5, "must-not-enter-ledger")
            )
            recorded = json.loads(
                (output / fetcher.STATE_FILE).read_text()
            )
            self.assertEqual(recorded["status"], "failed")
            attempt = recorded["attempts"][0]
            self.assertEqual(attempt["status"], "failed")
            self.assertEqual(attempt["completed_after"], 1)
            self.assertEqual(
                attempt["error"], {"type": "OSError", "errno": 5}
            )
            self.assertNotIn(
                "must-not-enter-ledger", json.dumps(recorded)
            )

    def test_real_layout_analyzer_requires_complete_base_and_dspark_experts(self):
        weight_map = {}
        for prefix, layers in (("layers", range(43)), ("mtp", range(1))):
            for layer in layers:
                for expert in range(256):
                    for projection in ("w1", "w2", "w3"):
                        for kind in ("weight", "scale"):
                            name = (
                                f"{prefix}.{layer}.ffn.experts.{expert}."
                                f"{projection}.{kind}"
                            )
                            weight_map[name] = "fixture.safetensors"
        weight_map["embed.weight"] = "fixture.safetensors"
        got = inventory_tool.analyze_weight_map(weight_map)
        self.assertTrue(got["passed"])
        self.assertEqual(got["base_expert_layers"], 43)
        self.assertEqual(got["dspark_expert_layers"], [0])
        del weight_map["layers.0.ffn.experts.0.w1.scale"]
        got = inventory_tool.analyze_weight_map(weight_map)
        self.assertFalse(got["passed"])
        self.assertIn("lack native", got["failures"][0])


if __name__ == "__main__":
    unittest.main()
