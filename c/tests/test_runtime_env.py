import os
import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from runtime_env import (  # noqa: E402
    ENGINE_ENV_KEYS,
    ENGINE_SOURCE_FILES,
    engine_source_sha256,
    isolated_engine_env,
)


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_engine_controls_are_removed_and_process_values_are_preserved(self) -> None:
        source = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": "/example/home",
            "TEMP": "/example/tmp",
            "MTP": "1",
            "TF": "1",
            "EXPERT_Q3_MAX_LAYER": "4",
            "PREFILL_CACHE_BYPASS": "1",
            "PREFILL_EXPERT_BATCH": "4",
            "PREFILL_LOAD_PIPELINE": "1",
            "Q3_NATIVE": "1",
            "CUDA_SPEC_FULL": "1",
            "COLI_MMAP": "1",
        }
        result = isolated_engine_env(source)
        self.assertEqual(result["PATH"], source["PATH"])
        self.assertEqual(result["HOME"], source["HOME"])
        self.assertEqual(result["TEMP"], source["TEMP"])
        self.assertNotIn("PREFILL_CACHE_BYPASS", result)
        self.assertNotIn("PREFILL_LOAD_PIPELINE", result)
        self.assertFalse(ENGINE_ENV_KEYS.intersection(result))

    def test_explicitly_preserved_controls_survive(self) -> None:
        source = {
            "PATH": "bin",
            "EXPERT_Q3": "1",
            "EXPERT_Q3_MIN_LAYER": "2",
            "MTP": "1",
        }
        result = isolated_engine_env(
            source,
            preserve=("EXPERT_Q3", "EXPERT_Q3_MIN_LAYER"),
        )
        self.assertEqual(result["EXPERT_Q3"], "1")
        self.assertEqual(result["EXPERT_Q3_MIN_LAYER"], "2")
        self.assertNotIn("MTP", result)

    def test_engine_source_fingerprint_is_stable_and_content_bound(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, relative in enumerate(ENGINE_SOURCE_FILES):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"source-{index}\n".encode())
            first = engine_source_sha256(root)
            second = engine_source_sha256(root)
            self.assertEqual(first, second)
            changed = root / ENGINE_SOURCE_FILES[1]
            changed.write_bytes(changed.read_bytes() + b"changed\n")
            self.assertNotEqual(first, engine_source_sha256(root))
            empty = root / ENGINE_SOURCE_FILES[2]
            empty.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "input is empty"):
                engine_source_sha256(root)


if __name__ == "__main__":
    unittest.main()
