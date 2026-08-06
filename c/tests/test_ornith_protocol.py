import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
OFFICIAL_META = Path(__file__).resolve().parents[1] / "bench" / "ornith35_meta"
sys.path.insert(0, str(TOOLS.parent))
SPEC = importlib.util.spec_from_file_location(
    "ornith_protocol", TOOLS / "ornith_protocol.py"
)
ornith = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ornith
SPEC.loader.exec_module(ornith)


TEMPLATE = """\
{%- if tools -%}<|im_start|>system
<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools><|im_end|>
{% endif -%}
{% for message in messages -%}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor -%}
{% if add_generation_prompt -%}<|im_start|>assistant
{% if enable_thinking %}<think>
{% else %}<think>

</think>

{% endif %}{% endif -%}
"""

ROUND_TRIP_TEMPLATE = """\
{%- for message in messages -%}
{%- if message.role == "assistant" -%}
<|im_start|>assistant
{%- for call in message.tool_calls %}
<tool_call><function={{ call.function.name }}>
{%- for name, value in call.function.arguments|items %}
<parameter={{ name }}>{{ value|tojson }}</parameter>
{%- endfor %}
</function></tool_call>
{%- endfor %}<|im_end|>
{%- elif message.role == "tool" -%}
<|im_start|>user
<tool_response>{{ message.content }}</tool_response><|im_end|>
{%- else -%}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt %}<|im_start|>assistant
<think>
{%- endif -%}
"""


class OrnithProtocolTests(unittest.TestCase):
    def test_profile_requires_explicit_family_and_both_stop_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp)
            (snap / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe",
                        "colib_model_family": "ornith-1.0",
                        "colib_source_repo": "deepreinforce-ai/Ornith-1.0-35B-FP8",
                        "colib_source_revision": "abc",
                    }
                )
            )
            (snap / "generation_config.json").write_text(
                json.dumps(
                    {
                        "eos_token_id": [248046, 248044],
                        "pad_token_id": 248044,
                    }
                )
            )
            (snap / "chat_template.jinja").write_text(TEMPLATE)
            profile = ornith.load_model_profile(snap)
            self.assertEqual(profile.family, "ornith-1.0")
            self.assertEqual(profile.stop_token_ids, (248046, 248044))
            self.assertEqual(profile.pad_token_id, 248044)
            self.assertIsNotNone(profile.chat_template_sha256)

    def test_render_uses_snapshot_template_and_thinking_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp)
            (snap / "chat_template.jinja").write_text(TEMPLATE)
            messages = [{"role": "user", "content": "weather"}]
            tool = {
                "type": "function",
                "function": {"name": "weather", "parameters": {"type": "object"}},
            }
            thinking = ornith.render_chat(snap, messages, tools=[tool])
            direct = ornith.render_chat(
                snap, messages, tools=[tool], enable_thinking=False
            )
            self.assertIn("<tools>", thinking)
            self.assertTrue(thinking.endswith("<think>\n"))
            self.assertTrue(direct.endswith("<think>\n\n</think>\n\n"))

    def test_qwen3_xml_tool_calls_become_openai_arguments(self):
        response = """<think>
Need live weather.
</think>
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
<parameter=units>
{"temperature": "celsius"}
</parameter>
</function>
</tool_call>"""
        parsed = ornith.parse_assistant_response(response)
        self.assertEqual(parsed["reasoning_content"], "Need live weather.")
        self.assertIsNone(parsed["content"])
        call = parsed["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(call["function"]["arguments"]),
            {"city": "Paris", "units": {"temperature": "celsius"}},
        )

    def test_malformed_suffix_is_not_silently_lost(self):
        response = (
            "<tool_call><function=f><parameter=x>1</parameter>"
            "</function></tool_call>unexpected"
        )
        parsed = ornith.parse_assistant_response(response)
        self.assertEqual(parsed["content"], "unexpected")
        self.assertEqual(json.loads(parsed["tool_calls"][0]["function"]["arguments"]), {"x": 1})

    def test_openai_tool_call_round_trip_normalizes_arguments_for_jinja(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp)
            (snap / "chat_template.jinja").write_text(ROUND_TRIP_TEMPLATE)
            parsed = ornith.parse_assistant_response(
                "<think>Use the tool.</think>"
                "<tool_call><function=get_weather>"
                "<parameter=city>Paris</parameter>"
                "<parameter=days>2</parameter>"
                "</function></tool_call>"
            )
            history = [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", **parsed},
                {"role": "tool", "content": '{"temperature":18}'},
            ]
            rendered = ornith.render_chat(snap, history)
            self.assertIn("<function=get_weather>", rendered)
            self.assertIn('<parameter=city>"Paris"</parameter>', rendered)
            self.assertIn("<parameter=days>2</parameter>", rendered)
            self.assertIn(
                '<tool_response>{"temperature":18}</tool_response>', rendered
            )
            self.assertTrue(rendered.endswith("<|im_start|>assistant\n<think>"))
            self.assertIsInstance(
                parsed["tool_calls"][0]["function"]["arguments"], str
            )

    def test_invalid_openai_arguments_fail_before_template_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp)
            (snap / "chat_template.jinja").write_text(ROUND_TRIP_TEMPLATE)
            messages = [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "f", "arguments": "{bad"},
                        }
                    ],
                }
            ]
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                ornith.render_chat(snap, messages)

    @unittest.skipUnless(
        (OFFICIAL_META / "chat_template.jinja").exists(),
        "official Ornith metadata cache is not present",
    )
    def test_official_template_accepts_parsed_call_and_tool_result(self):
        from openai_server import render_model_chat

        parsed = ornith.parse_assistant_response(
            "<think>Use live data.</think>"
            "<tool_call><function=get_weather>"
            "<parameter=city>Paris</parameter>"
            "</function></tool_call>"
        )
        messages = [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", **parsed},
            {"role": "tool", "content": '{"temperature":18}'},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Return current weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        rendered = ornith.render_chat(OFFICIAL_META, messages, tools=tools)
        gateway_rendered = render_model_chat(
            OFFICIAL_META,
            "ornith-1.0",
            messages,
            True,
            "high",
            tools,
        )
        self.assertEqual(gateway_rendered, rendered)
        required = render_model_chat(
            OFFICIAL_META,
            "ornith-1.0",
            [{"role": "user", "content": "Weather in Paris?"}],
            False,
            None,
            tools,
            "required",
        )
        self.assertIn(
            "You must call one of the provided functions. Do not answer directly.",
            required,
        )
        self.assertIn("<function=get_weather>", rendered)
        self.assertIn("<parameter=city>\nParis\n</parameter>", rendered)
        self.assertIn(
            '<tool_response>\n{"temperature":18}\n</tool_response>', rendered
        )
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n<think>\n"))


if __name__ == "__main__":
    unittest.main()
