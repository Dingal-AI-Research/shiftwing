import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai_server import (  # noqa: E402
    APIError,
    generation_options,
    parse_model_reply,
    render_model_chat,
    snapshot_model_family,
)
from tools.deepseek_v4_protocol import official, render_deepseek_chat  # noqa: E402


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


class DeepSeekGatewayTests(unittest.TestCase):
    def test_snapshot_family_detects_official_model_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            (snapshot / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v4"})
            )
            self.assertEqual(snapshot_model_family(snapshot), "deepseek-v4")

    def test_cli_selects_deepseek_binary_id_context_and_dspark(self):
        namespace = runpy.run_path(str(ROOT / "colib"), run_name="deepseek_cli_test")
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            (snapshot / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v4"})
            )
            (snapshot / "tokenizer.json").write_text("{}")
            parser = namespace["build_parser"]()
            args = parser.parse_args(
                [
                    "serve",
                    "--model",
                    str(snapshot),
                    "--context",
                    "65536",
                    "--dspark",
                    "off",
                ]
            )
            argv = namespace["_server_argv"](args)
            self.assertEqual(
                Path(argv[argv.index("--engine") + 1]),
                namespace["DEEPSEEK_ENGINE"],
            )
            self.assertEqual(
                argv[argv.index("--model-id") + 1],
                "deepseek-v4-flash-0731-colib",
            )
            environment = namespace["_runtime_env"](args)
            self.assertEqual(environment["CTX"], "65536")
            self.assertEqual(environment["DSPARK"], "off")
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    ["serve", "--model", str(snapshot), "--context", "65537"]
                )

    def test_render_dispatches_to_pinned_official_encoder(self):
        messages = [{"role": "user", "content": "Hello"}]
        got = render_model_chat(
            Path("unused"),
            "deepseek-v4",
            messages,
            enable_thinking=True,
            reasoning_effort="high",
            tools=TOOLS,
        )
        expected = render_deepseek_chat(
            messages, tools=TOOLS, reasoning_effort="high"
        )
        self.assertEqual(got, expected)

    def test_deepseek_sampling_profiles_and_agent_default(self):
        self.assertEqual(
            generation_options({}, 8, "deepseek-v4"),
            (8, 0.0, 1.0, None),
        )
        self.assertEqual(
            generation_options({"tools": TOOLS}, 8, "deepseek-v4"),
            (8, 1.0, 0.95, None),
        )
        self.assertEqual(
            generation_options(
                {"temperature": 1, "top_p": 0.95}, 8, "deepseek-v4"
            ),
            (8, 1.0, 0.95, None),
        )
        with self.assertRaisesRegex(APIError, "supports only"):
            generation_options(
                {"temperature": 0.7, "top_p": 0.9}, 8, "deepseek-v4"
            )

    def test_dsml_reply_parses_and_filters_unknown_tools(self):
        completion = (
            "Checking.\n\n"
            f"<{official.dsml_token}tool_calls>\n"
            f"<{official.dsml_token}invoke name=\"weather\">\n"
            f"<{official.dsml_token}parameter name=\"city\" string=\"true\">"
            f"London</{official.dsml_token}parameter>\n"
            f"</{official.dsml_token}invoke>\n"
            f"</{official.dsml_token}tool_calls>"
            f"{official.eos_token}"
        )
        reasoning, content, calls = parse_model_reply(
            "deepseek-v4", completion, TOOLS
        )
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "Checking.")
        self.assertEqual(calls[0]["function"]["name"], "weather")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"city": "London"}
        )
        _, _, filtered = parse_model_reply(
            "deepseek-v4",
            completion.replace("weather", "unknown"),
            TOOLS,
        )
        self.assertEqual(filtered, [])


if __name__ == "__main__":
    unittest.main()
