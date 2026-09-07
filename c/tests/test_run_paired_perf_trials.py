import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
RUNNER = TOOLS / "run_paired_perf_trials.py"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_paired_perf_trials import (  # noqa: E402
    hardware_identity,
    pair_order,
    paired_ratio_interval,
    student_t_critical,
)


FIXTURE = '''
import json, sys
from pathlib import Path

output = Path(sys.argv[1])
arm = sys.argv[2]
root = Path(sys.argv[3])
options = set(sys.argv[4:])

counter = root / "counter.txt"
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
order = root / "order.log"
order.write_text((order.read_text() if order.exists() else "") + arm + "\\n")

jitter = 1.0 + 0.01 * ((count % 3) - 1)
rates = [0.80, 0.90]
ttfts = [40.0, 44.0]
if arm == "candidate":
    rates = [rate * 1.15 * jitter for rate in rates]
    ttfts = [value * 0.90 * jitter for value in ttfts]
    if "slow-prompt" in options:
        rates[1] = 0.85
if "fail" in options:
    accepted = False
else:
    accepted = True

measured = []
for index, (rate, ttft) in enumerate(zip(rates, ttfts)):
    text = "answer-%d" % index
    if arm == "candidate" and "diverge" in options:
        text += "-diverged"
    measured.append(
        {
            "prompt_index": index,
            "prompt": "prompt-%d" % index,
            "text": text,
            "ttft_s": ttft,
            "wall_s": ttft + 64.0 / rate,
            "stats": {
                "tokens_per_second": rate,
                "completion_tokens": 64,
                "prompt_tokens": 38,
                "profile": {
                    "decode_expert_read_bytes": 1000 if arm == "control" else 900,
                    "decode_expert_misses": 100 if arm == "control" else 80,
                    "decode_expert_cpu_hits": 10,
                    "decode_expert_gpu_hits": 20,
                },
            },
        }
    )

result = {
    "measured": measured,
    "summary": {
        "sustained_tps": sum(rates) / len(rates),
        "median_turn_tps": sum(rates) / len(rates),
        "minimum_turn_tps": min(rates),
        "outputs_nonempty": True,
        "telemetry_complete": True,
        "cuda_active": True,
        "resident_cuda_graph": True,
        "q3_route_atlas_active": arm == "candidate",
        "automatic_gate_pass": "gate-fail" not in options,
    },
    "resident": {"host_moe": 4 if "host-moe" in options else 0, "device_moe": 46080},
    "hardware": {
        "gpu": "other-fixture" if "other-gpu" in options else "fixture",
        "cores": 16,
        "ram_total_gb": 29.375,
        "ram_avail_gb": 2.289 + 0.001 * count,
    },
    "acceptance": {"passed": accepted},
}
output.write_text(json.dumps(result))
'''


class PairedStatisticsTests(unittest.TestCase):
    def test_student_t_critical_matches_published_values(self):
        for degrees, expected in ((1, 12.706205), (4, 2.776445), (30, 2.042272)):
            self.assertAlmostEqual(
                student_t_critical(degrees, 0.95), expected, places=5
            )
        self.assertAlmostEqual(student_t_critical(4, 0.99), 4.604095, places=5)

    def test_paired_interval_brackets_the_geometric_mean(self):
        interval = paired_ratio_interval([1.10, 1.20, 1.15, 1.18, 1.12], 0.95)
        self.assertEqual(interval["count"], 5)
        self.assertLess(interval["lower"], interval["geometric_mean"])
        self.assertGreater(interval["upper"], interval["geometric_mean"])
        self.assertGreater(interval["lower"], 1.0)

    def test_single_pair_has_no_interval(self):
        interval = paired_ratio_interval([1.10], 0.95)
        self.assertIsNone(interval["lower"])
        self.assertEqual(interval["reason"], "one pair only")

    def test_launch_order_alternates(self):
        self.assertEqual(pair_order(1), ("control", "candidate"))
        self.assertEqual(pair_order(2), ("candidate", "control"))
        self.assertEqual(pair_order(3), ("control", "candidate"))


class PairedTrialRunnerTests(unittest.TestCase):
    def fixture_script(self, root: Path) -> Path:
        script = root / "fixture.py"
        script.write_text(FIXTURE)
        return script

    def command(self, root: Path, arm: str, *options: str) -> list[str]:
        return [
            sys.executable,
            str(self.fixture_script(root)),
            "{output}",
            arm,
            str(root),
            *options,
        ]

    def invoke(
        self,
        root: Path,
        *,
        pairs: int = 3,
        control_options: tuple[str, ...] = (),
        candidate_options: tuple[str, ...] = (),
        binding: Path | None = None,
        prime_trials: int = 0,
        recover_only: bool = False,
        analyze_only: bool = False,
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
            "--pairs",
            str(pairs),
            "--report",
            str(root / "report.json"),
        ]
        if prime_trials:
            argv.extend(["--prime-trials", str(prime_trials)])
        if binding is not None:
            argv.extend(["--bind", str(binding)])
        if recover_only:
            argv.append("--recover-only")
        if analyze_only:
            argv.append("--analyze-only")
        argv.extend(["--", *self.command(root, "control", *control_options)])
        argv.append(":::")
        argv.extend(self.command(root, "candidate", *candidate_options))
        return subprocess.run(argv, text=True, capture_output=True, check=False)

    def test_pairs_alternate_and_promote_a_faster_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.invoke(root, pairs=3)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(
                (root / "order.log").read_text().split(),
                [
                    "control",
                    "candidate",
                    "candidate",
                    "control",
                    "control",
                    "candidate",
                ],
            )
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["decision"], "promote")
            self.assertEqual(report["failures"], [])
            self.assertTrue(all(report["gates"].values()))
            self.assertGreater(report["decode_tps_ratio"]["lower"], 1.0)
            self.assertLess(report["ttft_ratio"]["upper"], 1.0)
            self.assertEqual(len(report["pairs"]), 3)
            self.assertEqual(
                [record["order"] for record in report["pairs"]],
                ["AB", "BA", "AB"],
            )
            state = json.loads((root / "state.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(len(state["completed"]), 6)

    def test_prime_trials_alternate_and_are_excluded_from_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.invoke(root, pairs=3, prime_trials=2)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(
                (root / "order.log").read_text().split(),
                [
                    "control",
                    "candidate",
                    "control",
                    "candidate",
                    "candidate",
                    "control",
                    "control",
                    "candidate",
                ],
            )
            self.assertTrue((root / "out" / "control-prime-01.json").is_file())
            self.assertTrue((root / "out" / "candidate-prime-02.json").is_file())
            state = json.loads((root / "state.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(len(state["completed"]), 8)
            self.assertIn("prime:1:control", state["completed"])
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["decision"], "promote")
            self.assertEqual(len(report["pairs"]), 3)
            self.assertEqual(report["decode_tps_ratio"]["count"], 3)
            resumed = self.invoke(root, pairs=3, prime_trials=2)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertIn("[resume prime 1/2 control] verified", resumed.stdout)
            self.assertEqual((root / "counter.txt").read_text(), "8")

    def test_drifting_available_ram_does_not_fail_a_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.invoke(root)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["decision"], "promote")
            self.assertEqual(report["failures"], [])
            spread = report["hardware_volatile"]["ram_avail_gb"]
            self.assertLess(spread["minimum"], spread["maximum"])
            self.assertEqual(report["hardware_identity"]["gpu"], "fixture")

    def test_differing_gpu_identity_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.invoke(root, candidate_options=("other-gpu",))
            self.assertEqual(result.returncode, 1)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["decision"], "hold")
            self.assertIn(
                "pair 1 hardware identity differs between arms", report["failures"]
            )

    def test_hardware_identity_drops_volatile_fields(self):
        self.assertEqual(
            hardware_identity({"cores": 16, "gpu": "x", "ram_avail_gb": 2.3}),
            {
                "cores": 16,
                "cpu": None,
                "gpu": "x",
                "gpus": None,
                "ram_total_gb": None,
                "vram_total_gb": None,
            },
        )

    def test_verified_trials_are_not_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.invoke(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual((root / "counter.txt").read_text(), "6")
            second = self.invoke(root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual((root / "counter.txt").read_text(), "6")
            self.assertIn("[resume pair 1/3 AB control] verified", second.stdout)

    def test_one_pair_cannot_establish_a_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.invoke(root, pairs=1)
            self.assertEqual(result.returncode, 1)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["decision"], "hold")
            self.assertTrue(report["gates"]["pairs_complete"])
            self.assertTrue(report["gates"]["outputs_identical"])
            self.assertFalse(report["gates"]["decode_lower_bound_above_one"])
            self.assertFalse(report["gates"]["ttft_upper_bound_below_one"])
            self.assertEqual(report["decode_tps_ratio"]["reason"], "one pair only")

    def test_binding_drift_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = root / "engine"
            binding.write_bytes(b"one")
            first = self.invoke(root, pairs=1, binding=binding)
            self.assertEqual(first.returncode, 1, first.stderr)
            binding.write_bytes(b"two")
            second = self.invoke(root, pairs=1, binding=binding)
            self.assertEqual(second.returncode, 2)
            self.assertIn("signature does not match", second.stderr)

    def test_divergent_candidate_output_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.invoke(root, candidate_options=("diverge",))
            self.assertEqual(result.returncode, 1)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["decision"], "hold")
            self.assertFalse(report["gates"]["outputs_identical"])
            self.assertIn(
                "pair 1 candidate output differs from control", report["failures"]
            )

    def test_per_prompt_slowdown_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.invoke(root, candidate_options=("slow-prompt",))
            self.assertEqual(result.returncode, 1)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["decision"], "hold")
            self.assertFalse(report["gates"]["no_per_prompt_slowdown"])
            self.assertLess(report["minimum_per_prompt_ratio"], 1.0)

    def test_host_moe_fallback_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.invoke(root, candidate_options=("host-moe",))
            self.assertEqual(result.returncode, 1)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["decision"], "hold")
            self.assertFalse(report["gates"]["no_structural_failure"])
            self.assertTrue(
                any("host-MoE fallback" in failure for failure in report["failures"])
            )

    def test_false_acceptance_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.invoke(root, pairs=1, candidate_options=("fail",))
            self.assertEqual(result.returncode, 2)
            self.assertFalse((root / "out" / "candidate-pair-01.json").exists())
            self.assertTrue((root / "out" / "control-pair-01.json").exists())
            state = json.loads((root / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["attempts"][-1]["status"], "failed")

    def test_recover_only_marks_stale_attempt_without_rerunning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.invoke(root, pairs=1, candidate_options=("fail",))
            self.assertEqual(first.returncode, 2)
            state_path = root / "state.json"
            state = json.loads(state_path.read_text())
            state["attempts"][-1]["status"] = "running"
            state["status"] = "running"
            state_path.write_text(json.dumps(state))
            recovered = self.invoke(
                root, pairs=1, candidate_options=("fail",), recover_only=True
            )
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
            self.assertEqual((root / "counter.txt").read_text(), "2")
            state = json.loads(state_path.read_text())
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["attempts"][-1]["status"], "interrupted")

    def test_analyze_only_recomputes_without_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.invoke(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            (root / "report.json").unlink()
            again = self.invoke(root, analyze_only=True)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual((root / "counter.txt").read_text(), "6")
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["decision"], "promote")

    def test_incomplete_state_reports_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.invoke(root, candidate_options=("fail",))
            self.assertEqual(first.returncode, 2)
            result = self.invoke(
                root, candidate_options=("fail",), analyze_only=True
            )
            self.assertEqual(result.returncode, 1)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["decision"], "hold")
            self.assertFalse(report["gates"]["pairs_complete"])
            self.assertIn("pair 1 candidate trial is incomplete", report["failures"])

    def test_pair_count_change_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.invoke(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = self.invoke(root, pairs=4)
            self.assertEqual(second.returncode, 2)
            self.assertIn("signature does not match", second.stderr)


if __name__ == "__main__":
    unittest.main()
