import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("convert_qwen", TOOLS / "convert_qwen.py")
convert_qwen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = convert_qwen
SPEC.loader.exec_module(convert_qwen)


class QuantizationTests(unittest.TestCase):
    @staticmethod
    def _manual_e4m3fn(byte: int) -> float:
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0:
            return sign * mantissa * (2.0**-9)
        if exponent == 0xF and mantissa == 0x7:
            return float("nan")
        return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))

    def test_fp8_e4m3_block_decode_and_edge_orientation(self):
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("torch build has no float8_e4m3fn")
        raw = torch.full((130, 129), 0x38, dtype=torch.uint8)
        edge_patterns = {
            (0, 0): 0x01,
            (0, 128): 0x30,
            (128, 0): 0xB8,
            (129, 128): 0x7E,
        }
        for index, byte in edge_patterns.items():
            raw[index] = byte
        fp8 = raw.view(torch.float8_e4m3fn)
        scales = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        got = convert_qwen.dequantize_fp8_blocks(fp8, scales)
        self.assertEqual(got.dtype, torch.float32)
        self.assertEqual(got.shape, raw.shape)
        self.assertEqual(float(got[1, 1]), 2.0)
        self.assertEqual(float(got[1, 128]), 3.0)
        self.assertEqual(float(got[128, 1]), 5.0)
        for index, byte in edge_patterns.items():
            block_scale = scales[index[0] // 128, index[1] // 128]
            expected = self._manual_e4m3fn(byte) * float(block_scale)
            self.assertEqual(float(got[index]), expected)

    def test_fp8_contract_rejects_bad_grid_dtype_and_nan(self):
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("torch build has no float8_e4m3fn")
        fp8 = torch.full((2, 3), 0x38, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        with self.assertRaises(ValueError):
            convert_qwen.dequantize_fp8_blocks(fp8, torch.ones(2, 1))
        with self.assertRaises(TypeError):
            convert_qwen.dequantize_fp8_blocks(
                torch.ones(2, 3), torch.ones(1, 1)
            )
        nan_fp8 = torch.tensor([[0x7F]], dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        with self.assertRaises(ValueError):
            convert_qwen.dequantize_fp8_blocks(nan_fp8, torch.ones(1, 1))

    def test_compressed_fp8_channel_decode_pairing_and_consumption(self):
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("torch build has no float8_e4m3fn")
        raw = torch.tensor(
            [[0x38, 0x40, 0xB8], [0x30, 0x38, 0x40]], dtype=torch.uint8
        )
        weight = raw.view(torch.float8_e4m3fn)
        scale = torch.tensor([[2.0], [3.0]], dtype=torch.float32)
        got = convert_qwen.dequantize_fp8_channels(weight, scale)
        expected = weight.float() * scale
        self.assertTrue(torch.equal(got, expected))

        name = "model.language_model.layers.0.self_attn.q_proj.weight"
        scale_name = convert_qwen.compressed_fp8_scale_name(name)
        pairs = convert_qwen.inspect_compressed_fp8_pairs(
            {name: "a.safetensors", scale_name: "a.safetensors"}
        )
        self.assertEqual(pairs["pairs"], {name: scale_name})
        shapes = convert_qwen.validate_channel_fp8_pair_shapes(
            pairs["pairs"], {name: [2, 3], scale_name: [2, 1]}
        )
        self.assertEqual(shapes["validated"], 1)
        self.assertFalse(shapes["mismatches"])

        payload, inventory = convert_qwen.containerize_state_mixed(
            {name: weight, scale_name: scale},
            xbits="int8",
            io_bits=8,
            shared_bits=8,
        )
        logical = name
        self.assertIn(logical, payload)
        self.assertNotIn(scale_name, payload)
        self.assertEqual(inventory[logical]["shape"], [2, 3])

    def test_ornith_family_annotation_is_explicit_and_qwen_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            original = {"model_type": "qwen3_5_moe"}
            (out / "config.json").write_text(json.dumps(original))
            family = convert_qwen.annotate_model_family(
                out,
                source_repo="deepreinforce-ai/Ornith-1.0-35B-FP8",
                source_revision="abc123",
            )
            got = json.loads((out / "config.json").read_text())
            self.assertEqual(family, "ornith-1.0")
            self.assertEqual(got["colib_model_family"], "ornith-1.0")
            self.assertEqual(got["colib_source_revision"], "abc123")

            (out / "config.json").write_text(json.dumps(original))
            family = convert_qwen.annotate_model_family(
                out,
                source_repo="Qwen/Qwen3.5-35B-A3B",
                source_revision="def456",
            )
            self.assertEqual(family, "qwen3.5")
            self.assertEqual(json.loads((out / "config.json").read_text()), original)

    def test_fp8_index_pairing_and_container_consumes_scale(self):
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("torch build has no float8_e4m3fn")
        weight_name = (
            "model.language_model.layers.0.self_attn.q_proj.weight"
        )
        scale_name = convert_qwen.fp8_scale_name(weight_name)
        weight_map = {
            weight_name: "model-1.safetensors",
            scale_name: "model-1.safetensors",
        }
        pairs = convert_qwen.inspect_fp8_pairs(weight_map)
        self.assertEqual(pairs["pair_count"], 1)
        self.assertFalse(pairs["orphan_scales"])
        self.assertFalse(pairs["cross_shard_weights"])
        fp8 = torch.full((2, 128), 0x38, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        payload, inventory = convert_qwen.containerize_state_mixed(
            {weight_name: fp8, scale_name: torch.tensor([[2.0]])},
            xbits="int4g128",
        )
        self.assertNotIn(scale_name, payload)
        self.assertNotIn(scale_name, inventory)
        self.assertEqual(int(payload[f"{weight_name}.qtype"][0]), 4)
        expected_q, expected_s = convert_qwen.quantize_int4_grouped(
            torch.full((2, 128), 2.0)
        )
        torch.testing.assert_close(payload[weight_name], expected_q)
        torch.testing.assert_close(payload[f"{weight_name}.qs"], expected_s)

    def test_fp8_index_pairing_rejects_orphan_and_cross_shard(self):
        orphan = convert_qwen.inspect_fp8_pairs(
            {"x.weight_scale_inv": "a.safetensors"}
        )
        self.assertEqual(orphan["orphan_scales"], ["x.weight_scale_inv"])
        cross = convert_qwen.inspect_fp8_pairs(
            {
                "x.weight": "a.safetensors",
                "x.weight_scale_inv": "b.safetensors",
            }
        )
        self.assertEqual(cross["cross_shard_weights"], ["x.weight"])

    def test_fp8_metadata_shape_validation(self):
        pairs = {"x.weight": "x.weight_scale_inv"}
        good = convert_qwen.validate_fp8_pair_shapes(
            pairs,
            {"x.weight": [129, 257], "x.weight_scale_inv": [2, 3]},
        )
        self.assertEqual(good, {"validated": 1, "mismatches": []})
        bad = convert_qwen.validate_fp8_pair_shapes(
            pairs,
            {"x.weight": [129, 257], "x.weight_scale_inv": [2, 2]},
        )
        self.assertEqual(bad["validated"], 0)
        self.assertEqual(len(bad["mismatches"]), 1)
        self.assertEqual(bad["mismatches"][0]["expected"], (2, 3))

    def test_int8_zero_row_and_bounds(self):
        value = torch.tensor([[0.0, 0.0], [-2.0, 2.0]])
        packed, scales = convert_qwen.quantize_int8_rows(value)
        got = convert_qwen.dequantize_int8_rows(packed, scales)
        self.assertEqual(packed.dtype, torch.uint8)
        signed = packed.view(torch.int8)
        self.assertGreaterEqual(int(signed.min()), -127)
        self.assertLessEqual(int(signed.max()), 127)
        self.assertTrue(torch.isfinite(scales).all())
        self.assertTrue((scales > 0).all())
        self.assertEqual(float(scales[0]), 1.0)
        self.assertAlmostEqual(float(scales[1]), 2.0 / 127.0, places=8)
        torch.testing.assert_close(got, value)

    def test_int4_nibble_order_and_padding(self):
        value = torch.tensor([[-8.0, -1.0, 0.0, 7.0, 1.0]])
        packed, scales = convert_qwen.quantize_int4_grouped(value, group_size=8)
        got = convert_qwen.dequantize_int4_grouped(packed, scales, columns=5, group_size=8)
        self.assertEqual(packed.shape, (1, 4))
        self.assertEqual(int(packed[0, 0]), 0x71)  # offset nibbles: low=-7, high=-1
        self.assertTrue(torch.isfinite(scales).all())
        self.assertTrue((scales > 0).all())
        self.assertEqual(got.shape, value.shape)
        self.assertLessEqual(float((got - value).abs().max()), 8.0 / 7.0 / 2.0 + 1e-6)

    def test_int2_code_order_zero_group_and_padding(self):
        exact = torch.tensor([[-3.0, -1.0, 1.0, 3.0]])
        packed, scales = convert_qwen.quantize_int2_grouped(
            exact, group_size=4
        )
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(int(packed[0, 0]), 0xE4)
        torch.testing.assert_close(scales, torch.ones_like(scales))
        torch.testing.assert_close(
            convert_qwen.dequantize_int2_grouped(
                packed, scales, columns=4, group_size=4
            ),
            exact,
        )

        value = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, -2.0, -0.5, 0.5]]
        )
        packed, scales = convert_qwen.quantize_int2_grouped(
            value, group_size=4
        )
        got = convert_qwen.dequantize_int2_grouped(
            packed, scales, columns=7, group_size=4
        )
        self.assertEqual(packed.shape, (1, 2))
        self.assertEqual(scales.shape, (1, 2))
        self.assertEqual(got.shape, value.shape)
        self.assertEqual(float(scales[0, 0]), 0.0)
        self.assertTrue(torch.equal(got[0, :4], torch.zeros(4)))
        self.assertTrue(torch.isfinite(scales).all())
        self.assertTrue((scales >= 0).all())

    def test_int2_is_deterministic_and_preserves_leading_dimensions(self):
        value = torch.linspace(-2, 2, 2 * 3 * 129).reshape(2, 3, 129)
        first = convert_qwen.quantize_int2_grouped(value)
        second = convert_qwen.quantize_int2_grouped(value)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(first[0].shape, (2, 3, 64))
        self.assertEqual(first[1].shape, (2, 3, 2))
        got = convert_qwen.dequantize_int2_grouped(
            first[0], first[1], columns=129
        )
        self.assertEqual(got.shape, value.shape)
        self.assertTrue(torch.isfinite(got).all())
        self.assertLess(float(torch.mean((got - value) ** 2).sqrt()), 0.5)

    def test_int2_rejects_bad_contracts(self):
        matrix = torch.ones(1, 8)
        for group_size in (0, 2, 6):
            with self.assertRaises(ValueError):
                convert_qwen.quantize_int2_grouped(
                    matrix, group_size=group_size
                )
        with self.assertRaises(ValueError):
            convert_qwen.quantize_int2_grouped(matrix, group_size=4, iterations=0)
        with self.assertRaises(TypeError):
            convert_qwen.dequantize_int2_grouped(
                torch.ones(1, 2), torch.ones(1, 2),
                columns=8, group_size=4,
            )

    def test_int3_pack_order_and_mirrored_scale(self):
        exact = torch.arange(-4.0, 4.0).reshape(1, 8)
        packed, scales = convert_qwen.quantize_int3_grouped(
            exact, group_size=8
        )
        self.assertEqual(packed.tolist(), [[0x88, 0xC6, 0xFA]])
        torch.testing.assert_close(scales, torch.ones_like(scales))
        torch.testing.assert_close(
            convert_qwen.dequantize_int3_grouped(
                packed, scales, columns=8, group_size=8
            ),
            exact,
        )

        mirrored = torch.arange(-3.0, 5.0).reshape(1, 8)
        packed, scales = convert_qwen.quantize_int3_grouped(
            mirrored, group_size=8
        )
        self.assertLess(float(scales[0, 0]), 0.0)
        torch.testing.assert_close(
            convert_qwen.dequantize_int3_grouped(
                packed, scales, columns=8, group_size=8
            ),
            mirrored,
        )

    def test_int3_padding_zero_and_determinism(self):
        value = torch.zeros(2, 3, 129)
        value[1, 2] = torch.linspace(-2, 2, 129)
        first = convert_qwen.quantize_int3_grouped(value)
        second = convert_qwen.quantize_int3_grouped(value)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(first[0].shape, (2, 3, 96))
        self.assertEqual(first[1].shape, (2, 3, 2))
        got = convert_qwen.dequantize_int3_grouped(
            first[0], first[1], columns=129
        )
        self.assertEqual(got.shape, value.shape)
        self.assertTrue(torch.equal(got[0], torch.zeros_like(got[0])))
        self.assertTrue(torch.isfinite(got).all())
        self.assertLess(float(torch.mean((got - value) ** 2).sqrt()), 0.2)

    def test_int3_rejects_bad_contracts(self):
        matrix = torch.ones(1, 8)
        for group_size in (0, 4, 12):
            with self.assertRaises(ValueError):
                convert_qwen.quantize_int3_grouped(
                    matrix, group_size=group_size
                )
        with self.assertRaises(ValueError):
            convert_qwen.quantize_int3_grouped(matrix, iterations=0)
        with self.assertRaises(ValueError):
            convert_qwen.dequantize_int3_grouped(
                torch.ones(1, 4, dtype=torch.uint8),
                torch.ones(1, 1),
                columns=8,
                group_size=8,
            )

    def test_grouped_quant_preserves_leading_dimensions(self):
        value = torch.linspace(-1, 1, 2 * 3 * 129).reshape(2, 3, 129)
        packed, scales = convert_qwen.quantize_int4_grouped(value)
        got = convert_qwen.dequantize_int4_grouped(packed, scales, columns=129)
        self.assertEqual(packed.shape, (2, 3, 128))
        self.assertEqual(scales.shape, (2, 3, 2))
        self.assertEqual(got.shape, value.shape)

    def test_nonfinite_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                convert_qwen.quantize_int8_rows(torch.tensor([[value]]))
            with self.assertRaises(ValueError):
                convert_qwen.quantize_int4_grouped(torch.tensor([[value]]))
            with self.assertRaises(ValueError):
                convert_qwen.quantize_int2_grouped(torch.tensor([[value]]))
            with self.assertRaises(ValueError):
                convert_qwen.quantize_int3_grouped(torch.tensor([[value]]))

    def test_unfuse_experts(self):
        gate_up = torch.arange(2 * 6 * 4).reshape(2, 6, 4)
        down = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
        got = convert_qwen.unfuse_experts(
            {
                "model.layers.0.mlp.experts.gate_up_proj": gate_up,
                "model.layers.0.mlp.experts.down_proj": down,
            }
        )
        self.assertEqual(len(got), 6)
        torch.testing.assert_close(got["model.layers.0.mlp.experts.0.gate_proj.weight"], gate_up[0, :3])
        torch.testing.assert_close(got["model.layers.0.mlp.experts.1.up_proj.weight"], gate_up[1, 3:])
        torch.testing.assert_close(got["model.layers.0.mlp.experts.1.down_proj.weight"], down[1])

    def test_phase4_precision_map(self):
        matrix = torch.ones(3, 4)
        vector = torch.ones(4)
        mode = convert_qwen.precision_for_tensor
        self.assertEqual(mode("model.language_model.embed_tokens.weight", matrix), "int8")
        self.assertEqual(mode("lm_head.weight", matrix), "int8")
        self.assertEqual(mode("model.language_model.layers.0.mlp.experts.0.gate_proj.weight", matrix), "int4g128")
        self.assertEqual(mode("model.language_model.layers.0.mlp.shared_expert.up_proj.weight", matrix), "int8")
        self.assertEqual(mode("model.language_model.layers.0.self_attn.q_proj.weight", matrix), "int4g128")
        self.assertEqual(mode("model.language_model.layers.0.self_attn.o_proj.weight", matrix), "int8")
        self.assertEqual(mode("model.language_model.layers.0.linear_attn.in_proj_b.weight", matrix), "f32")
        self.assertEqual(mode("model.language_model.layers.0.mlp.gate.weight", matrix), "f32")
        self.assertEqual(mode("model.language_model.layers.0.input_layernorm.weight", vector), "f32")
        self.assertEqual(mode("mtp.layers.0.mlp.experts.0.gate_proj.weight", matrix), "int8")

    def test_tiny_int4_container_keeps_mtp_int8(self):
        payload = convert_qwen.containerize_state(
            {
                "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 8),
                "mtp.fc.weight": torch.randn(4, 8),
            },
            mode="int4g128",
            group_size=8,
        )
        self.assertEqual(int(payload["model.layers.0.self_attn.q_proj.weight.qtype"][0]), 4)
        self.assertEqual(int(payload["mtp.fc.weight.qtype"][0]), 8)
        self.assertEqual(payload["mtp.fc.weight"].numel(), 4 * 8)

    def test_int2_expert_sidecar_is_resumable_and_non_destructive(self):
        with tempfile.TemporaryDirectory() as td:
            snapshot = Path(td)
            state = {}
            for expert in range(2):
                for projection, shape in (
                    ("gate_proj", (4, 8)),
                    ("up_proj", (4, 8)),
                    ("down_proj", (8, 4)),
                ):
                    name = (
                        "model.language_model.layers.0.mlp.experts."
                        f"{expert}.{projection}.weight"
                    )
                    value = torch.linspace(
                        -1.0 + expert * 0.1,
                        1.0 + expert * 0.1,
                        shape[0] * shape[1],
                    ).reshape(shape)
                    state[name], state[f"{name}.qs"] = (
                        convert_qwen.quantize_int4_grouped(
                            value, group_size=8
                        )
                    )
            source = snapshot / "model.safetensors"
            save_file(state, source)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            weight_map = {name: source.name for name in state}
            (snapshot / convert_qwen.INDEX_FILE).write_text(
                json.dumps({"weight_map": weight_map})
            )
            (snapshot / "config.json").write_text(
                json.dumps(
                    {
                        "text_config": {
                            "hidden_size": 8,
                            "moe_intermediate_size": 4,
                        }
                    }
                )
            )
            command = [
                sys.executable,
                str(TOOLS / "requantize_expert_q2.py"),
                "--snapshot",
                str(snapshot),
                "--group-size",
                "8",
                "--experts-per-file",
                "1",
                "--resume-hash-workers",
                "2",
            ]
            first = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            second = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            self.assertIn("converted=2", first.stdout)
            self.assertIn("converted=0 resumed=2", second.stdout)
            self.assertIn("resume-validated=2/2 hash-workers=2", second.stdout)
            damaged = snapshot / "expert-q2-l000-e0000-0000.safetensors"
            with damaged.open("ab") as handle:
                handle.write(b"tamper")
            repaired = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            self.assertIn("resume-validated=1/2 hash-workers=2", repaired.stdout)
            self.assertIn("converted=1 resumed=1", repaired.stdout)
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(), source_hash
            )
            manifest = json.loads(
                (snapshot / "expert-q2.json").read_text()
            )
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["file_count"], 2)
            sidecar = load_file(
                snapshot / "expert-q2-l000-e0000-0000.safetensors"
            )
            target = (
                "model.language_model.layers.0.mlp.experts.0."
                "gate_proj.weight.q2"
            )
            self.assertEqual(sidecar[target].dtype, torch.uint8)
            self.assertEqual(int(sidecar[f"{target}.qtype"][0]), 2)
            self.assertTrue(torch.isfinite(sidecar[f"{target}.qs"]).all())

            q3_command = [*command, "--bits", "3"]
            q3 = subprocess.run(
                q3_command, check=True, capture_output=True, text=True
            )
            self.assertIn("converted=2", q3.stdout)
            q3_manifest = json.loads(
                (snapshot / "expert-q3.json").read_text()
            )
            self.assertTrue(q3_manifest["complete"])
            q3_sidecar = load_file(
                snapshot / "expert-q3-l000-e0000-0000.safetensors"
            )
            q3_target = target.removesuffix(".q2") + ".q3"
            self.assertEqual(int(q3_sidecar[f"{q3_target}.qtype"][0]), 3)

            # Older converter versions published only the final manifest.
            # Recovery must fail closed unless adoption is explicit, then
            # validate and retain the durable chunk instead of rewriting it.
            (snapshot / "expert-q3.json").unlink()
            (snapshot / "expert-q3-l000-e0001-0001.safetensors").unlink()
            refused = subprocess.run(
                q3_command, check=False, capture_output=True, text=True
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("unledgered q3 sidecar files", refused.stderr)
            adopted = subprocess.run(
                [*q3_command, "--adopt-existing"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("converted=1 resumed=1 adopted=1", adopted.stdout)
            adopted_manifest = json.loads(
                (snapshot / "expert-q3.json").read_text()
            )
            self.assertTrue(adopted_manifest["complete"])
            self.assertEqual(adopted_manifest["file_count"], 2)

    def test_loader_inventory_map(self):
        config = {
            "text_config": {
                "num_hidden_layers": 2,
                "num_experts": 3,
                "layer_types": ["linear_attention", "full_attention"],
            }
        }
        names = convert_qwen.expected_loader_names(config)
        self.assertEqual(len(names), 50)
        self.assertIn("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", names)
        self.assertIn("model.language_model.layers.1.self_attn.q_proj.weight", names)
        self.assertIn("model.language_model.layers.1.mlp.experts.2.down_proj.weight", names)
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)
            (outdir / "config.json").write_text(json.dumps(config))
            bare_names = {name.replace("model.language_model.", "model."): {} for name in names}
            got = convert_qwen.validate_loader_inventory(outdir, bare_names)
            self.assertEqual(got, {"expected": 50, "present": 50})

    def test_streaming_resume_unfuse_and_text_filter(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, output = root / "source", root / "output"
            source.mkdir()
            shard1 = {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.arange(2 * 6 * 4).reshape(2, 6, 4).float(),
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": torch.randn(8, 4),
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": torch.randn(2, 4),
                "model.visual.blocks.0.weight": torch.randn(2, 2),
            }
            shard2 = {
                "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(2, 4, 3),
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(3, 4),
                "model.language_model.layers.0.mlp.gate.weight": torch.randn(2, 4),
                "model.language_model.embed_tokens.weight": torch.randn(10, 4),
                "model.language_model.norm.weight": torch.randn(4),
                "mtp.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(3, 4),
            }
            names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
            save_file(shard1, source / names[0])
            save_file(shard2, source / names[1])
            weight_map = {name: names[0] for name in shard1} | {name: names[1] for name in shard2}
            (source / convert_qwen.INDEX_FILE).write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}))
            (source / "config.json").write_text("{}")
            (source / "tokenizer.json").write_text("{}")

            partial = convert_qwen.convert_local_directory(source, output, min_free_gb=0, max_shards=1)
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["converted_now"], 1)
            first_mtime = (output / names[0]).stat().st_mtime_ns
            complete = convert_qwen.convert_local_directory(source, output, min_free_gb=0)
            self.assertTrue(complete["complete"])
            self.assertEqual((output / names[0]).stat().st_mtime_ns, first_mtime)

            out1, out2 = load_file(output / names[0]), load_file(output / names[1])
            self.assertNotIn("model.visual.blocks.0.weight", out1)
            self.assertNotIn("mtp.layers.0.mlp.experts.0.gate_proj.weight", out2)
            gate = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
            self.assertIn(gate, out1)
            self.assertIn(f"{gate}.qs", out1)
            self.assertEqual(int(out1[f"{gate}.qtype"][0]), 4)
            self.assertEqual(out1[gate].dtype, torch.uint8)
            self.assertEqual(out1["model.language_model.layers.0.linear_attn.in_proj_b.weight"].dtype, torch.float32)
            self.assertEqual(out2["model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"].shape, (3, 4))
            self.assertIn("model.language_model.embed_tokens.weight.qs", out2)
            index = json.loads((output / convert_qwen.INDEX_FILE).read_text())
            self.assertIn(gate, index["weight_map"])
            self.assertIn(f"{gate}.qs", index["weight_map"])
            self.assertIn(f"{gate}.qtype", index["weight_map"])

            again = convert_qwen.convert_local_directory(source, output, min_free_gb=0)
            self.assertTrue(again["complete"])
            self.assertEqual(again["converted_now"], 0)
            self.assertEqual((output / names[0]).stat().st_mtime_ns, first_mtime)

            upgraded = convert_qwen.convert_local_directory(
                source, output, min_free_gb=0, include_mtp=True
            )
            self.assertTrue(upgraded["complete"])
            self.assertEqual(upgraded["converted_now"], 1)
            out2 = load_file(output / names[1])
            mtp = "mtp.layers.0.mlp.experts.0.gate_proj.weight"
            self.assertIn(mtp, out2)
            self.assertEqual(int(out2[f"{mtp}.qtype"][0]), 8)
            state = json.loads((output / convert_qwen.STATE_FILE).read_text())
            self.assertTrue(state["signature"]["include_mtp"])

            upgraded_again = convert_qwen.convert_local_directory(
                source, output, min_free_gb=0, include_mtp=True
            )
            self.assertTrue(upgraded_again["complete"])
            self.assertEqual(upgraded_again["converted_now"], 0)


if __name__ == "__main__":
    unittest.main()
