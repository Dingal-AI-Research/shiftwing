import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
RUNNER = TOOLS / "run_perf_trials.py"


class PerformanceTrialRunnerTests(unittest.TestCase):
    def command(self, root: Path, accepted: bool = True) -> list[str]:
        counter = root / "counter.txt"
        code = (
            "import json,sys; from pathlib import Path; "
            "out=Path(sys.argv[1]); counter=Path(sys.argv[2]); "
            "count=int(counter.read_text())+1 if counter.exists() else 1; "
            "counter.write_text(str(count)); "
            f"out.write_text(json.dumps({{'acceptance':{{'passed':{accepted!r}}},"
            "'summary':{'sustained_tps':1.0},'hardware':{'gpu':'fixture'}}))"
        )
        return [sys.executable, "-c", code, "{output}", str(counter)]

    def invoke(
        self,
        root: Path,
        command: list[str],
        *,
        trials: int = 2,
        binding: Path | None = None,
        recover_only: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        argv = [
            sys.executable,
            str(RUNNER),
            "--root",
            str(root),
            "--state",
            str(root / "state.json"),
            "--output-dir",
            str(root / "out"),
            "--label",
            "fixture",
            "--trials",
            str(trials),
        ]
        if binding is not None:
            argv.extend(["--bind", str(binding)])
        if recover_only:
            argv.append("--recover-only")
        argv.extend(["--", *command])
        return subprocess.run(argv, text=True, capture_output=True, check=False)

    def test_completed_trials_are_hash_verified_and_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self.command(root)
            first = self.invoke(root, command)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual((root / "counter.txt").read_text(), "2")
            second = self.invoke(root, command)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual((root / "counter.txt").read_text(), "2")
            state = json.loads((root / "state.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(len(state["completed"]), 2)
            self.assertIn("[resume 1/2]", second.stdout)

    def test_binding_drift_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = root / "engine"
            binding.write_bytes(b"one")
            command = self.command(root)
            first = self.invoke(root, command, trials=1, binding=binding)
            self.assertEqual(first.returncode, 0, first.stderr)
            binding.write_bytes(b"two")
            second = self.invoke(root, command, trials=1, binding=binding)
            self.assertEqual(second.returncode, 2)
            self.assertIn("signature does not match", second.stderr)

    def test_false_acceptance_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.invoke(root, self.command(root, accepted=False), trials=1)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((root / "out" / "fixture-trial-01.json").exists())
            state = json.loads((root / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["attempts"][-1]["status"], "failed")

    def test_recover_only_marks_stale_attempt_without_rerunning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self.command(root, accepted=False)
            first = self.invoke(root, command, trials=1)
            self.assertEqual(first.returncode, 2)
            state_path = root / "state.json"
            state = json.loads(state_path.read_text())
            state["attempts"][-1]["status"] = "running"
            state["status"] = "running"
            state_path.write_text(json.dumps(state))
            recovered = self.invoke(
                root, command, trials=1, recover_only=True
            )
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
            self.assertEqual((root / "counter.txt").read_text(), "1")
            state = json.loads(state_path.read_text())
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["attempts"][-1]["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
