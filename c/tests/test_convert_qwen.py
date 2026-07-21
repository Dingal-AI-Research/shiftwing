import importlib.util
import json
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


if __name__ == "__main__":
    unittest.main()
