from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prefetch_conversion_shards",
    ROOT / "tools" / "prefetch_conversion_shards.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrefetchConversionShardTests(unittest.TestCase):
    def test_source_shards_are_unique_and_sorted(self) -> None:
        index = {
            "weight_map": {
                "b": "model-00002.safetensors",
                "a": "model-00001.safetensors",
                "c": "model-00002.safetensors",
            }
        }
        self.assertEqual(
            MODULE.source_shards(index),
            ("model-00001.safetensors", "model-00002.safetensors"),
        )

    def test_lookahead_skips_exactly_one_uncommitted_shard(self) -> None:
        shards = ("s1", "s2", "s3", "s4")
        self.assertEqual(MODULE.lookahead_target(shards, {"s1"}), "s3")
        self.assertEqual(MODULE.lookahead_target(shards, {"s1", "s3"}), "s4")

    def test_lookahead_stops_at_final_uncommitted_shard(self) -> None:
        self.assertIsNone(MODULE.lookahead_target(("s1", "s2"), {"s1"}))

    def test_signature_is_pinned_to_repo_and_revision(self) -> None:
        state = {"signature": {"source": "hf://owner/model@deadbeef"}}
        MODULE.validate_signature(
            state,
            repo="owner/model",
            revision="deadbeef",
        )
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            MODULE.validate_signature(
                state,
                repo="owner/model",
                revision="cafebabe",
            )


if __name__ == "__main__":
    unittest.main()
