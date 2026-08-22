import importlib.util
import io
import math
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "qualify_tiered_model", TOOLS / "qualify_tiered_model.py"
)
qualifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualifier
SPEC.loader.exec_module(qualifier)


class TieredQualifierTests(unittest.TestCase):
    def test_prefill_cache_controls_default_off(self):
        args = qualifier._argument_parser().parse_args(["--model", "."])
        self.assertEqual(args.prefill_cache_bypass, 0)
        self.assertEqual(args.prefill_load_pipeline, 0)

    def test_prefill_load_pipeline_requires_cache_bypass(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "qualify_tiered_model.py"),
                "--model",
                ".",
                "--prefill-load-pipeline",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "prefill-load-pipeline requires prefill-cache-bypass",
            completed.stderr,
        )

    def test_ornith_qualification_uses_snapshot_template(self):
        template = """\
{% for message in messages -%}[{{ message.role }}]{{ message.content }}
{% endfor -%}
{% if add_generation_prompt -%}[assistant]{% if enable_thinking %}<think>
{% else %}<direct>{% endif %}{% endif -%}
"""
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "chat_template.jinja").write_text(
                template,
                encoding="utf-8",
            )
            rendered = qualifier.render_qualification_prompt(
                model,
                "ornith-1.0",
                "hello",
            )
        self.assertEqual(rendered, "[user]hello\n[assistant]<direct>")

    def test_sustained_tps_is_token_weighted(self):
        runs = [
            {"stats": {"completion_tokens": 10, "tokens_per_second": 2.0}},
            {"stats": {"completion_tokens": 20, "tokens_per_second": 4.0}},
        ]
        self.assertEqual(qualifier.sustained_tps(runs), 3.0)

    def test_sustained_tps_rejects_invalid_measurements(self):
        invalid = (
            [],
            [{"stats": {"completion_tokens": 0, "tokens_per_second": 2.0}}],
            [{"stats": {"completion_tokens": 1, "tokens_per_second": 0.0}}],
            [{"stats": {"completion_tokens": 1, "tokens_per_second": math.nan}}],
        )
        for runs in invalid:
            with self.subTest(runs=runs):
                with self.assertRaises(ValueError):
                    qualifier.sustained_tps(runs)

    def test_expert_telemetry_decodes_tiers_heat_and_hits(self):
        # Two disk, two RAM, and two VRAM entries. Bits 0 and 5 were touched.
        emap = {
            "rows": 2,
            "cols": 3,
            "map": bytes([0, 3, 0x41, 0x42, 0x81, 0x82]).hex(),
        }
        result = qualifier.validate_expert_telemetry(
            emap,
            bytes([0b00100001]).hex(),
            {"disk": 2, "ram": 2, "vram": 2},
            expected_rows=2,
            expected_cols=3,
        )
        self.assertEqual(result["tier_counts"], {"disk": 2, "ram": 2, "vram": 2})
        self.assertEqual(result["turn_hit_experts"], 2)
        self.assertEqual(result["nonzero_heat"], 5)
        self.assertEqual(result["maximum_heat_bucket"], 3)

    def test_expert_telemetry_rejects_reserved_and_padding_bits(self):
        base = {"rows": 1, "cols": 2, "map": bytes([0, 0x40]).hex()}
        with self.assertRaisesRegex(ValueError, "padding bits"):
            qualifier.validate_expert_telemetry(
                base,
                bytes([0x80]).hex(),
                {"disk": 1, "ram": 1, "vram": 0},
                expected_rows=1,
                expected_cols=2,
            )
        reserved = {"rows": 1, "cols": 2, "map": bytes([0, 0xC0]).hex()}
        with self.assertRaisesRegex(ValueError, "reserved tier"):
            qualifier.validate_expert_telemetry(
                reserved,
                bytes([0]).hex(),
                {"disk": 1, "ram": 0, "vram": 1},
                expected_rows=1,
                expected_cols=2,
            )

    def test_run_pass_reports_each_slow_turn_immediately(self):
        original_turn = qualifier._turn

        def fake_turn(engine, prompt, max_tokens, index, **kwargs):
            return {
                "prompt_index": index,
                "prompt": prompt,
                "text": "ok",
                "ttft_s": 1.25,
                "wall_s": 2.5,
                "stats": {
                    "completion_tokens": max_tokens,
                    "tokens_per_second": 3.0,
                },
            }

        qualifier._turn = fake_turn
        output = io.StringIO()
        try:
            with redirect_stderr(output):
                turns = qualifier._run_pass(
                    object(),
                    ["one", "two"],
                    4,
                    phase="measured",
                    pass_index=0,
                    pass_count=1,
                )
        finally:
            qualifier._turn = original_turn
        self.assertEqual([turn["pass"] for turn in turns], [0, 0])
        rendered = output.getvalue()
        self.assertIn("prompt 1/2: starting", rendered)
        self.assertIn("prompt 2/2: done in 2.5s", rendered)
        self.assertIn("decode=3.000 tok/s", rendered)


if __name__ == "__main__":
    unittest.main()
