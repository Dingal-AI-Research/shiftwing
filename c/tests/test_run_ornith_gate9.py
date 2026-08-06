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
    "run_ornith_gate9",
    TOOLS / "run_ornith_gate9.py",
)
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class OrnithGate9PipelineTests(unittest.TestCase):
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

    def test_gate8_precondition_allows_only_gate9_failures(self):
        accepted = {
            "summary": {
                "failures": sorted(pipeline.GATE9_CHECK_IDS),
                "passed": False,
            }
        }
        with patch.object(pipeline, "audit_release", return_value=accepted):
            self.assertEqual(pipeline.validate_gate8(Path("/repo")), accepted)
        rejected = {
            "summary": {
                "failures": ["ornith397.tier"],
                "passed": False,
            }
        }
        with patch.object(pipeline, "audit_release", return_value=rejected):
            with self.assertRaisesRegex(ValueError, "ornith397.tier"):
                pipeline.validate_gate8(Path("/repo"))

    def test_sequence_matches_frozen_two_model_protocol(self):
        steps = pipeline.gate9_steps(Path("/repo"))
        self.assertEqual(len(steps), 10)
        self.assertEqual(
            [step.name for step in steps],
            [
                "ornith35.batch_ab",
                "ornith35.batch_ba",
                "ornith35.batch_abba",
                "ornith35.cancel",
                "ornith35.web",
                "ornith397.batch_ab",
                "ornith397.batch_ba",
                "ornith397.batch_abba",
                "ornith397.cancel",
                "ornith397.web",
            ],
        )
        argv = " ".join(part for step in steps for part in step.argv)
        self.assertNotIn("qwen397", argv)
        self.assertIn("--trials 20", argv)
        self.assertIn("--minimum-each 0.95", argv)
        self.assertIn("--minimum-geomean 1.0", argv)
        for step in steps:
            if not step.name.endswith("batch_abba"):
                self.assertEqual(
                    step.argv[step.argv.index("--threads") + 1],
                    "8",
                )
        ornith397 = " ".join(
            part
            for step in steps
            if step.name.startswith("ornith397.")
            for part in step.argv
        )
        self.assertIn("--ram-gb 18", ornith397)
        self.assertIn("--cuda-expert-gb 5", ornith397)
        self.assertIn("--max-cancel-ack-s 3.0", ornith397)
        self.assertEqual(ornith397.count("--expert-q3"), 4)
        ornith35 = " ".join(
            part
            for step in steps
            if step.name.startswith("ornith35.")
            for part in step.argv
        )
        self.assertNotIn("--expert-q3", ornith35)

    def test_pipeline_resumes_verified_artifact_and_rejects_rebinding(self):
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
            )
            passing_audit = {"summary": {"passed": True, "failures": []}}
            manifests = {"ornith35": "a" * 64, "ornith397": "b" * 64}
            with (
                patch.object(pipeline, "gate9_steps", return_value=(step,)),
                patch.object(
                    pipeline,
                    "audit_release",
                    return_value=passing_audit,
                ),
            ):
                self.assertEqual(
                    pipeline.run_pipeline(
                        root,
                        state_path,
                        bound_manifests=manifests,
                        engine_sha256="c" * 64,
                        engine_source_fingerprint="s" * 64,
                    ),
                    0,
                )
                mtime = artifact.stat().st_mtime_ns
                self.assertEqual(
                    pipeline.run_pipeline(
                        root,
                        state_path,
                        bound_manifests=manifests,
                        engine_sha256="d" * 64,
                        engine_source_fingerprint="s" * 64,
                    ),
                    0,
                )
                self.assertEqual(artifact.stat().st_mtime_ns, mtime)
                artifact.write_text('{"acceptance":{"passed":false}}')
                self.assertEqual(
                    pipeline.run_pipeline(
                        root,
                        state_path,
                        bound_manifests=manifests,
                        engine_sha256="c" * 64,
                        engine_source_fingerprint="s" * 64,
                    ),
                    0,
                )
            state = json.loads(state_path.read_text())
            self.assertEqual(state["status"], "passed")
            self.assertEqual(state["engine_sha256"], "c" * 64)
            self.assertEqual(state["engine_source_sha256"], "s" * 64)
            with self.assertRaisesRegex(ValueError, "different manifests"):
                pipeline.run_pipeline(
                    root,
                    state_path,
                    bound_manifests={
                        "ornith35": "c" * 64,
                        "ornith397": "d" * 64,
                    },
                    engine_sha256="c" * 64,
                    engine_source_fingerprint="s" * 64,
                )
            with self.assertRaisesRegex(ValueError, "different engine sources"):
                pipeline.run_pipeline(
                    root,
                    state_path,
                    bound_manifests=manifests,
                    engine_sha256="d" * 64,
                    engine_source_fingerprint="t" * 64,
                )


if __name__ == "__main__":
    unittest.main()
