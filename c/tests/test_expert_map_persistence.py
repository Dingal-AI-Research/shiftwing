"""Prove the learned expert map survives a turn, a signal, and a freeze.

The engine seeds its expert cache from a persisted heat map when `AUTOPIN` is
set, which is what lets a fresh process start with the experts real traffic
actually routes to instead of experts 0..cap-1. That is only worth anything if
the map is on disk when the next process starts. Persisting exclusively from
`atexit` did not achieve that: a graceful shutdown that overruns its wait is
escalated to `SIGTERM`, which runs no `atexit` handler, so the map was silently
lost. These tests pin the checkpoint-on-turn-boundary behavior, the freeze
switch that keeps a benchmark from mutating its own seed map, and the header
the format promises.
"""

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("COLI_TEST_ENGINE", ROOT / "qwen"))
MODEL = ROOT / "qwen_tiny_i4"
MAGIC = b"COLIEMAP"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def engine_available() -> bool:
    return ENGINE.is_file() and os.access(ENGINE, os.X_OK) and MODEL.is_dir()


def base_env(emap: Path, extra=None) -> dict[str, str]:
    # Turn-boundary checkpointing is opt-in so that rebuilding the engine
    # cannot change measured benchmark behavior. These tests exercise the
    # checkpoint path, so they request it explicitly; the exit-only test
    # overrides EMAP_SAVE_EVERY back to 0.
    env = dict(
        os.environ,
        AUTOPIN="1",
        EMAP_PATH=str(emap),
        EMAP_SAVE_EVERY="1",
        EXPERT_RAM="4",
        PREFETCH_THREADS="0",
        MTP="0",
    )
    env.update(extra or {})
    return env


def read_header(path: Path) -> tuple[bytes, int, int, int]:
    with path.open("rb") as handle:
        magic = handle.read(8)
        version, layers, experts = struct.unpack("<III", handle.read(12))
    return magic, version, layers, experts


@unittest.skipUnless(engine_available(), "engine or tiny fixture is unavailable")
class ExpertMapPersistenceTests(unittest.TestCase):
    def load_only(self, emap: Path, extra=None) -> str:
        """Load, report, and exit normally so the [EMAP] line is emitted."""

        env = base_env(emap, {**(extra or {}), "LOAD_ONLY": "1", "SNAP": str(MODEL)})
        result = subprocess.run(
            [str(ENGINE)],
            cwd=ROOT.parent,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return result.stderr

    def serve_turns(self, emap: Path, *, turns: int = 2, extra=None, kill=False):
        from openai_server import Engine  # noqa: E402

        engine = Engine(ENGINE, MODEL, max_tokens=2, env=base_env(emap, extra))
        try:
            for _ in range(turns):
                engine.generate("!", 2, 0.0, 1.0, lambda _text: None)
        finally:
            if kill:
                engine.process.kill()
                engine.process.wait(timeout=60)
            else:
                engine.close()

    def test_map_is_written_while_serving_not_only_at_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            emap = Path(directory) / "expert_map.bin"
            self.serve_turns(emap, turns=2, kill=True)
            self.assertTrue(
                emap.is_file(),
                "map must survive a killed process, which runs no atexit handler",
            )
            magic, version, layers, experts = read_header(emap)
            self.assertEqual(magic, MAGIC)
            self.assertEqual(version, 1)
            self.assertGreater(layers, 0)
            self.assertGreater(experts, 0)
            self.assertEqual(emap.stat().st_size, 20 + 4 * layers * experts)

    def test_saved_map_is_loaded_by_the_next_process(self):
        with tempfile.TemporaryDirectory() as directory:
            emap = Path(directory) / "expert_map.bin"
            self.serve_turns(emap, turns=2, kill=True)
            self.assertIn("loaded=1", self.load_only(emap))

    def test_freeze_keeps_the_map_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            emap = Path(directory) / "expert_map.bin"
            self.serve_turns(emap, turns=2, kill=True)
            original = emap.read_bytes()
            self.serve_turns(emap, turns=2, extra={"EMAP_FREEZE": "1"})
            self.assertEqual(
                emap.read_bytes(),
                original,
                "a frozen map is an input; a benchmark must not rewrite its own seed",
            )
            report = self.load_only(emap, {"EMAP_FREEZE": "1"})
            self.assertIn("frozen=1", report)
            self.assertIn("saved=0", report)

    def test_save_every_zero_restores_exit_only_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            emap = Path(directory) / "expert_map.bin"
            self.serve_turns(emap, turns=2, extra={"EMAP_SAVE_EVERY": "0"}, kill=True)
            self.assertFalse(
                emap.is_file(),
                "EMAP_SAVE_EVERY=0 must not checkpoint on turn boundaries",
            )


if __name__ == "__main__":
    unittest.main()
