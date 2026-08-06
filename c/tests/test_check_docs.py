import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "check_docs",
    TOOLS / "check_docs.py",
)
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


class DocumentationCheckerTests(unittest.TestCase):
    def fixture(self, root: Path) -> None:
        research = root / "docs" / "research"
        research.mkdir(parents=True)
        (root / "PLAN.md").write_text(
            "[report](docs/research/report.md)\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text("# Project\n", encoding="utf-8")
        (research / "README.md").write_text(
            "[Report](report.md)\n",
            encoding="utf-8",
        )
        (research / "report.md").write_text("# Report\n", encoding="utf-8")

    def test_complete_links_and_index_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            result = checker.check_docs(root)
            self.assertTrue(result["passed"], result["failures"])
            self.assertEqual(result["research_reports"], 1)
            self.assertEqual(result["local_links"], 2)

    def test_missing_link_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "README.md").write_text(
                "[missing](docs/missing.md)\n",
                encoding="utf-8",
            )
            result = checker.check_docs(root)
            self.assertFalse(result["passed"])
            self.assertTrue(
                any("missing link target" in item for item in result["failures"])
            )

    def test_unindexed_research_report_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "docs" / "research" / "extra.md").write_text(
                "# Extra\n",
                encoding="utf-8",
            )
            result = checker.check_docs(root)
            self.assertFalse(result["passed"])
            self.assertIn(
                "research report is not indexed: extra.md",
                result["failures"],
            )


if __name__ == "__main__":
    unittest.main()
