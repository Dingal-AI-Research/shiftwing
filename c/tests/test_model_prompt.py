import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.model_prompt import prepare_user_prompt  # noqa: E402


TEMPLATE = """\
{% for message in messages -%}[{{ message.role }}]{{ message.content }}
{% endfor -%}
{% if add_generation_prompt -%}[assistant]{% if enable_thinking %}<think>
{% else %}<direct>{% endif %}{% endif -%}
"""


class ModelPromptTests(unittest.TestCase):
    def test_ornith_uses_snapshot_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps({"colib_model_family": "ornith-1.0"}),
                encoding="utf-8",
            )
            (model / "chat_template.jinja").write_text(
                TEMPLATE,
                encoding="utf-8",
            )
            family, rendered = prepare_user_prompt(model, "hello")
        self.assertEqual(family, "ornith-1.0")
        self.assertEqual(rendered, "[user]hello\n[assistant]<direct>")

    def test_raw_diagnostic_preserves_input_but_records_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps({"colib_model_family": "ornith-1.0"}),
                encoding="utf-8",
            )
            family, rendered = prepare_user_prompt(
                model,
                "!",
                raw=True,
            )
        self.assertEqual(family, "ornith-1.0")
        self.assertEqual(rendered, "!")


if __name__ == "__main__":
    unittest.main()
