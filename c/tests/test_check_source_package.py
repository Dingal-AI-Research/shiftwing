import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "check_source_package",
    TOOLS / "check_source_package.py",
)
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


class SourcePackageCheckerTests(unittest.TestCase):
    def test_source_files_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.c").write_text("int main(void) { return 0; }\n")
            self.assertEqual(
                checker.check_source_package(root, ["main.c"]),
                [],
            )

    def test_native_executable_magic_fails_without_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "c" / "tests" / "test_generated"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"\x7fELFfixture")
            failures = checker.check_source_package(
                root,
                ["c/tests/test_generated"],
            )
            self.assertEqual(
                failures,
                ["tracked native executable: c/tests/test_generated"],
            )

    def test_pe_signature_fails_but_plain_mz_text_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pe = root / "tool"
            header = bytearray(68)
            header[:2] = b"MZ"
            header[60:64] = (64).to_bytes(4, "little")
            header[64:68] = b"PE\0\0"
            pe.write_bytes(header)
            text = root / "notes.txt"
            text.write_text("MZ is a model-size abbreviation.\n")
            failures = checker.check_source_package(
                root,
                ["tool", "notes.txt"],
            )
            self.assertEqual(
                failures,
                ["tracked native executable: tool"],
            )

    def test_model_and_web_outputs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failures = checker.check_source_package(
                root,
                [
                    "c/ornith397/model-00001.safetensors",
                    "reference.gguf",
                    "web/dist/index.html",
                ],
            )
            self.assertEqual(len(failures), 4)
            self.assertTrue(
                any("tracked generated/model path" in item for item in failures)
            )
            self.assertTrue(
                any("tracked generated/model suffix" in item for item in failures)
            )


if __name__ == "__main__":
    unittest.main()
