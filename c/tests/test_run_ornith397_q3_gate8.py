import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "run_ornith397_q3_gate8",
    TOOLS / "run_ornith397_q3_gate8.py",
)
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class Ornith397Q3Gate8Tests(unittest.TestCase):
    def write_manifests(self, model: Path, **q3_updates):
        model.mkdir(parents=True, exist_ok=True)
        (model / "quantization.json").write_text(
            json.dumps(pipeline.EXPECTED_BASE),
            encoding="utf-8",
        )
        q3 = {
            **pipeline.EXPECTED_Q3,
            "data_bytes": 123,
            "files": [{} for _ in range(480)],
            **q3_updates,
        }
        (model / "expert-q3.json").write_text(json.dumps(q3), encoding="utf-8")
        return q3

    def test_complete_q3_manifest_requires_exact_conversion_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            q3 = self.write_manifests(model)
            _, observed = pipeline.validate_manifests(model)
            self.assertEqual(observed, q3)
            self.write_manifests(
                model,
                signature={**pipeline.EXPECTED_Q3["signature"], "iterations": 2},
            )
            with self.assertRaisesRegex(ValueError, "signature"):
                pipeline.validate_manifests(model)

    def test_incomplete_q3_is_not_treated_as_finished_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            self.write_manifests(model, complete=False)
            with self.assertRaisesRegex(RuntimeError, "inactive"):
                pipeline.wait_for_q3(
                    model,
                    converter_pid=2**30,
                    poll_seconds=0.001,
                )

    def test_sequence_is_q3_bound_and_preserves_manual_coherence_review(self):
        steps = pipeline.gate8_steps(Path("/repo"))
        self.assertEqual(
            [step.name for step in steps],
            [
                "q3_doctor",
                "int4_reference",
                "q3_teacher_forced",
                "q3_coherence",
                "q3_tools",
                "q3_tier",
            ],
        )
        argv = {step.name: step.argv for step in steps}
        self.assertIn("--verify-hashes", argv["q3_doctor"])
        self.assertEqual(
            argv["q3_doctor"][argv["q3_doctor"].index("--hash-workers") + 1],
            "8",
        )
        self.assertNotIn("--expert-q3", argv["int4_reference"])
        self.assertIn("--production-cuda", argv["int4_reference"])
        self.assertIn("--ram-gb", argv["int4_reference"])
        self.assertIn("--expert-q3", argv["q3_teacher_forced"])
        self.assertIn("--production-cuda", argv["q3_teacher_forced"])
        self.assertEqual(
            argv["q3_teacher_forced"][
                argv["q3_teacher_forced"].index("--min-tf-prompt-agreement") + 1
            ],
            "0.85",
        )
        self.assertIn("--expert-q3", argv["q3_tools"])
        for name in (
            "int4_reference",
            "q3_teacher_forced",
            "q3_coherence",
            "q3_tools",
            "q3_tier",
        ):
            self.assertEqual(argv[name][argv[name].index("--threads") + 1], "8")
        tier = argv["q3_tier"]
        self.assertEqual(tier[tier.index("--minimum-tps") + 1], "0.70")
        self.assertEqual(tier[tier.index("--q3-native") + 1], "0")
        self.assertTrue(pipeline.Q3_PPL_WAIVED)
        self.assertEqual(pipeline.Q3_MINIMUM_TPS, 0.70)
        for option in (
            "--uring-persist",
            "--pinned-upload",
            "--decode-protect",
            "--decode-protect-prewarm",
            "--expert-q3",
        ):
            self.assertEqual(tier[tier.index(option) + 1], "1")

    def test_pipeline_pauses_after_coherence_until_acknowledged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.json"
            artifacts = [root / f"{name}.json" for name in ("quality", "coherence", "tools")]
            script = (
                "from pathlib import Path; import json,sys; "
                "Path(sys.argv[1]).write_text(json.dumps({'passed': True}))"
            )
            steps = (
                pipeline.Step(
                    "quality",
                    (sys.executable, "-c", script, str(artifacts[0])),
                    artifacts[0],
                    ("passed",),
                ),
                pipeline.Step(
                    "q3_coherence",
                    (sys.executable, "-c", script, str(artifacts[1])),
                    artifacts[1],
                    ("passed",),
                ),
                pipeline.Step(
                    "q3_tools",
                    (sys.executable, "-c", script, str(artifacts[2])),
                    artifacts[2],
                    ("passed",),
                ),
            )
            bindings = {
                "base_manifest_sha256": "a" * 64,
                "q3_manifest_sha256": "b" * 64,
                "baseline_sha256": "c" * 64,
                "engine_sha256": "d" * 64,
                "engine_source_fingerprint": "e" * 64,
            }
            with patch.object(pipeline, "gate8_steps", return_value=steps):
                self.assertEqual(
                    pipeline.run_pipeline(
                        root,
                        state,
                        **bindings,
                        acknowledge_coherence=False,
                    ),
                    2,
                )
                self.assertFalse(artifacts[2].exists())
                self.assertEqual(
                    json.loads(state.read_text())["status"],
                    "awaiting_coherence_review",
                )
                self.assertEqual(
                    pipeline.run_pipeline(
                        root,
                        state,
                        **bindings,
                        acknowledge_coherence=True,
                    ),
                    0,
                )
                self.assertTrue(artifacts[2].exists())
                self.assertTrue(
                    json.loads(state.read_text())["coherence_review_acknowledged"]
                )
                self.assertEqual(
                    pipeline.run_pipeline(
                        root,
                        state,
                        **{**bindings, "engine_sha256": "f" * 64},
                        acknowledge_coherence=True,
                    ),
                    0,
                )
                with self.assertRaisesRegex(ValueError, "binding mismatch"):
                    pipeline.run_pipeline(
                        root,
                        state,
                        **{
                            **bindings,
                            "engine_source_fingerprint": "0" * 64,
                        },
                        acknowledge_coherence=True,
                    )


if __name__ == "__main__":
    unittest.main()
