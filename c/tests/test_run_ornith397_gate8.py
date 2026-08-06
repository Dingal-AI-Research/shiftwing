import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "run_ornith397_gate8",
    TOOLS / "run_ornith397_gate8.py",
)
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class Ornith397Gate8PipelineTests(unittest.TestCase):
    def write_manifest(self, model: Path, **updates):
        value = {**pipeline.EXPECTED_MANIFEST, **updates}
        model.mkdir(parents=True, exist_ok=True)
        (model / "quantization.json").write_text(
            json.dumps(value),
            encoding="utf-8",
        )
        return value

    def test_manifest_requires_exact_pinned_identity_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            expected = self.write_manifest(model)
            self.assertEqual(pipeline.validate_manifest(model), expected)
            self.write_manifest(model, source_shards=121)
            with self.assertRaisesRegex(ValueError, "source_shards"):
                pipeline.validate_manifest(model)

    def test_cuda_build_is_required_and_hashes_the_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "c" / "qwen"
            binary.parent.mkdir()
            binary.write_bytes(b"cuda-engine")
            with patch.object(
                pipeline.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0),
            ) as run:
                digest = pipeline.build_cuda_engine(root)
            self.assertEqual(digest, pipeline._sha256(binary))
            argv = run.call_args.args[0]
            self.assertIn("CUDA=1", argv)
            self.assertIn("CUDA_ARCH=native", argv)
            with patch.object(
                pipeline.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=2),
            ):
                with self.assertRaisesRegex(RuntimeError, "exit code 2"):
                    pipeline.build_cuda_engine(root)

    def test_wait_accepts_only_complete_manifest_after_converter_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            expected = self.write_manifest(model)
            self.assertEqual(
                pipeline.wait_for_conversion(
                    model,
                    converter_pid=2**30,
                    poll_seconds=0.001,
                ),
                expected,
            )
            (model / "quantization.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "exited without"):
                pipeline.wait_for_conversion(
                    model,
                    converter_pid=2**30,
                    poll_seconds=0.001,
                )

    def test_preregistered_sequence_has_no_qwen_low_bit_work(self):
        steps = pipeline.gate8_steps(Path("/repo"))
        self.assertEqual(
            [step.name for step in steps],
            ["doctor", "ppl", "tier", "tools"],
        )
        argv = " ".join(part for step in steps for part in step.argv)
        self.assertNotIn("expert-q3", argv)
        self.assertNotIn("qwen397", argv)
        self.assertIn("--verify-hashes", steps[0].argv)
        self.assertIn("--minimum-tps", steps[2].argv)
        minimum_index = steps[2].argv.index("--minimum-tps")
        self.assertEqual(steps[2].argv[minimum_index + 1], "0.85")
        for option in (
            "--uring-persist",
            "--pinned-upload",
            "--decode-protect",
            "--decode-protect-prewarm",
        ):
            index = steps[2].argv.index(option)
            self.assertEqual(steps[2].argv[index + 1], "1")

    def test_pipeline_resumes_only_hash_verified_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.json"
            state_path = root / "state.json"
            script = (
                "from pathlib import Path; import json, sys; "
                "Path(sys.argv[1]).write_text("
                "json.dumps({'acceptance': {'passed': True}}))"
            )
            step = pipeline.Step(
                "fixture",
                (sys.executable, "-c", script, str(artifact)),
                artifact,
                ("acceptance", "passed"),
            )
            with patch.object(pipeline, "gate8_steps", return_value=(step,)):
                self.assertEqual(
                    pipeline.run_pipeline(
                        root,
                        state_path,
                        manifest_sha256="a" * 64,
                        engine_sha256="b" * 64,
                    ),
                    0,
                )
                first_mtime = artifact.stat().st_mtime_ns
                self.assertEqual(
                    pipeline.run_pipeline(
                        root,
                        state_path,
                        manifest_sha256="a" * 64,
                        engine_sha256="b" * 64,
                    ),
                    0,
                )
                self.assertEqual(artifact.stat().st_mtime_ns, first_mtime)
                artifact.write_text('{"acceptance":{"passed":false}}')
                self.assertEqual(
                    pipeline.run_pipeline(
                        root,
                        state_path,
                        manifest_sha256="a" * 64,
                        engine_sha256="b" * 64,
                    ),
                    0,
                )
                self.assertTrue(
                    json.loads(artifact.read_text())["acceptance"]["passed"]
                )
            state = json.loads(state_path.read_text())
            self.assertEqual(state["status"], "passed")
            self.assertEqual(state["manifest_sha256"], "a" * 64)
            self.assertEqual(state["engine_sha256"], "b" * 64)
            self.assertEqual(state["expected_manifest"], pipeline.EXPECTED_MANIFEST)

    def test_pipeline_rejects_state_from_another_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "running",
                        "manifest_sha256": "a" * 64,
                        "steps": {},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "bound to manifest"):
                pipeline.run_pipeline(
                    root,
                    state_path,
                    manifest_sha256="b" * 64,
                    engine_sha256="c" * 64,
                )

    def test_pipeline_rejects_state_from_another_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "running",
                        "manifest_sha256": "a" * 64,
                        "engine_sha256": "b" * 64,
                        "steps": {},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "bound to engine"):
                pipeline.run_pipeline(
                    root,
                    state_path,
                    manifest_sha256="a" * 64,
                    engine_sha256="c" * 64,
                )


if __name__ == "__main__":
    unittest.main()
