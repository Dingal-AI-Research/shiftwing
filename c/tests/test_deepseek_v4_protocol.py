import importlib
import json
import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
protocol = importlib.import_module("deepseek_v4_protocol")
official = protocol.official


class DeepSeekV4ProtocolTests(unittest.TestCase):
    def test_vendor_is_bound_to_pinned_hash(self):
        self.assertEqual(protocol.verify_vendor(), protocol.ENCODING_SHA256)

    def test_nonthinking_and_high_reasoning_render(self):
        messages = [{"role": "user", "content": "Hello"}]
        chat = protocol.render_deepseek_chat(messages)
        self.assertTrue(chat.startswith(official.bos_token + official.USER_SP_TOKEN))
        self.assertTrue(
            chat.endswith(official.ASSISTANT_SP_TOKEN + official.thinking_end_token)
        )
        thinking = protocol.render_deepseek_chat(
            messages, reasoning_effort="high"
        )
        self.assertIn("Reasoning Effort: Absolute maximum", thinking)
        self.assertTrue(thinking.endswith(official.thinking_start_token))
        for unsupported in ("low", "max"):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                protocol.render_deepseek_chat(
                    messages, reasoning_effort=unsupported
                )

    def test_tools_use_official_dsml_template(self):
        tools = [
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
        prompt = protocol.render_deepseek_chat(
            [{"role": "user", "content": "Weather?"}], tools=tools
        )
        self.assertIn("## Tools", prompt)
        self.assertIn("weather", prompt)
        self.assertIn(official.dsml_token, prompt)

    def test_parse_and_malformed_recovery(self):
        parsed = protocol.parse_deepseek_completion("hello")
        self.assertEqual(parsed["content"], "hello")
        self.assertFalse(parsed["colib_recovery"]["recovered"])

        malformed = (
            f"answer\n\n<{official.dsml_token}tool_calls>\n"
            f"<{official.dsml_token}invoke broken"
        )
        recovered = protocol.parse_deepseek_completion(malformed)
        self.assertEqual(recovered["content"], "answer")
        self.assertEqual(recovered["tool_calls"], [])
        self.assertTrue(recovered["colib_recovery"]["recovered"])

        thinking = protocol.parse_deepseek_completion(
            "private reasoning", reasoning_effort="high"
        )
        self.assertEqual(thinking["reasoning_content"], "private reasoning")
        self.assertEqual(thinking["content"], "")
        self.assertTrue(thinking["colib_recovery"]["recovered"])


if __name__ == "__main__":
    unittest.main()
