import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "compare_qwen_prefix_reference",
    TOOLS / "compare_qwen_prefix.py",
)
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


class DummyTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return f"user:{messages[0]['content']}"


class PrefixReferenceCaptureTests(unittest.TestCase):
    def test_production_profile_replaces_ambient_cache_and_cuda_settings(self):
        env = {"EXPERT_RAM": "999", "RAM_GB": "999", "COLI_CUDA": "0"}
        compare.configure_colib_runtime(
            env,
            expert_ram=48,
            ram_gb=18.0,
            ram_headroom_gb=1.0,
            production_cuda=True,
            cuda_expert_gb=6.0,
            cuda_headroom_gb=1.0,
        )
        self.assertNotIn("EXPERT_RAM", env)
        self.assertEqual(env["RAM_GB"], "18.0")
        self.assertEqual(env["COLI_CUDA"], "1")
        self.assertEqual(env["CUDA_EXPERT_GB"], "6.0")
        self.assertEqual(env["URING_PERSIST"], "1")
        self.assertEqual(env["CUDA_PINNED_UPLOAD"], "1")
        self.assertEqual(env["DECODE_PROTECT_PREWARM"], "1")

    def test_c_only_capture_needs_no_gguf_and_is_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "model"
            snapshot.mkdir()
            prompts = root / "prompts.json"
            prompts.write_text(json.dumps(["hello"]))
            output = root / "reference.json"
            args = SimpleNamespace(
                snapshot=snapshot,
                gguf=None,
                prompts=prompts,
                engine=root / "engine",
                llama_cli="llama-cli",
                llama_server=None,
                server_port=8191,
                tokens=2,
                threads=1,
                expert_ram=1,
                ram_gb=2.0,
                ram_headroom_gb=1.0,
                production_cuda=True,
                cuda_expert_gb=1.0,
                cuda_headroom_gb=1.0,
                prefetch_threads=1,
                limit=None,
                indices=None,
                diagnostic_topk=False,
                c_only=True,
                c_idot=None,
                reference_json=None,
                output=output,
                expected_gguf_sha256=None,
                require_complete_manifest=False,
                expert_q2=False,
                expert_q3=False,
                expert_q3_max_layer=None,
                expert_q3_min_layer=None,
                min_tf_prompt_agreement=None,
                min_tf_aggregate_agreement=None,
                teacher_forced_only=False,
            )
            with (
                patch.object(compare, "parse_args", return_value=args),
                patch.object(
                    compare.AutoTokenizer,
                    "from_pretrained",
                    return_value=DummyTokenizer(),
                ),
                patch.object(
                    compare,
                    "c_prefixes",
                    return_value=([[7, 11]], [None]),
                ),
            ):
                compare.main()
            result = json.loads(output.read_text())
            self.assertEqual(result["mode"], "colib-reference-capture")
            self.assertEqual(result["reference_profile"], "base")
            self.assertTrue(result["acceptance"]["passed"])
            self.assertTrue(result["colib_runtime"]["production_cuda"])
            self.assertEqual(result["colib_runtime"]["ram_gb"], 2.0)
            self.assertEqual(result["rows"][0]["tokens"], [7, 11])
            self.assertEqual(result["rows"][0]["reference_tokens"], [7, 11])


if __name__ == "__main__":
    unittest.main()
