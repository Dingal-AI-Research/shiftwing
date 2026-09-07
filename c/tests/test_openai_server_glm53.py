"""GLM-5.3-Flash is rendered with its own chat template, not Qwen's.

openai_server.py carries two chat renderers. For a long time the GLM one was
shadowed by a later definition of the same name, so every non-Ornith model was
rendered as Qwen. That is invisible at import time and produces a model that
looks broken rather than an error: GLM-5.3's template shares no control token
with Qwen's, so the model is prompted entirely in tokens it never saw in
training.

These tests pin the dispatch by behaviour rather than by definition order.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai_server import (  # noqa: E402
    parse_model_tool_calls,
    render_glm_chat,
    render_model_chat,
    snapshot_model_family,
)

MESSAGES = [
    {"role": "system", "content": "Return one JSON object."},
    {"role": "user", "content": "Review this diff."},
]


class Glm53ChatRendering(unittest.TestCase):
    def test_glm_family_uses_the_glm_template(self):
        prompt = render_model_chat(None, "glm-5.3", MESSAGES)
        self.assertEqual(
            prompt,
            "[gMASK]<sop><|system|>Return one JSON object."
            "<|user|>Review this diff.<|assistant|><think></think>",
        )

    def test_glm_template_carries_no_qwen_control_tokens(self):
        # The specific failure this guards: Qwen's markers reaching GLM.
        prompt = render_model_chat(None, "glm-5.3", MESSAGES)
        self.assertNotIn("<|im_start|>", prompt)
        self.assertNotIn("<|im_end|>", prompt)

    def test_qwen_family_is_unaffected(self):
        prompt = render_model_chat(None, "qwen3.5", MESSAGES)
        self.assertIn("<|im_start|>", prompt)
        self.assertNotIn("[gMASK]", prompt)

    def test_dispatch_reaches_the_named_glm_renderer(self):
        # Definition order must not decide which renderer runs.
        self.assertEqual(
            render_model_chat(None, "glm-5.3", MESSAGES),
            render_glm_chat(MESSAGES),
        )

    def test_thinking_opens_the_block_for_glm(self):
        prompt = render_model_chat(None, "glm-5.3", MESSAGES, enable_thinking=True)
        self.assertTrue(prompt.endswith("<|assistant|><think>"))

    def test_family_is_read_from_the_converted_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            (snapshot / "config.json").write_text(
                json.dumps({"shiftwing_model_family": "glm-5.3"}), encoding="utf-8")
            self.assertEqual(snapshot_model_family(snapshot), "glm-5.3")

    def test_unstamped_snapshot_still_defaults_to_qwen(self):
        # The default is deliberate; the converter stamps GLM explicitly.
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            (snapshot / "config.json").write_text(json.dumps({}), encoding="utf-8")
            self.assertEqual(snapshot_model_family(snapshot), "qwen3.5")


class Glm53ToolCallParsing(unittest.TestCase):
    def test_glm_family_parses_glm_boxes(self):
        reply = ("<tool_call>get_weather\n"
                 "<arg_key>city</arg_key><arg_value>Rome</arg_value>"
                 "</tool_call>")
        tools = [{"type": "function",
                  "function": {"name": "get_weather", "parameters": {}}}]
        _content, calls = parse_model_tool_calls("glm-5.3", reply, tools)
        self.assertEqual([c["function"]["name"] for c in calls], ["get_weather"])

    def test_qwen_family_parses_qwen_xml(self):
        reply = ("<tool_call><function=get_weather>"
                 "<parameter=city>Rome</parameter>"
                 "</function></tool_call>")
        tools = [{"type": "function",
                  "function": {"name": "get_weather", "parameters": {}}}]
        _content, calls = parse_model_tool_calls("qwen3.5", reply, tools)
        self.assertEqual([c["function"]["name"] for c in calls], ["get_weather"])


if __name__ == "__main__":
    unittest.main()
