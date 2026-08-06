import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / ("qwen.exe" if os.name == "nt" else "qwen")


def run_qwen(snapshot: str, **options: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"SNAP": str(ROOT / snapshot), **{key: str(value) for key, value in options.items()}})
    return subprocess.run(
        [str(BINARY)], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )


def tokens(output: str) -> list[int]:
    match = re.search(r"^tokens:(.*)$", output, re.MULTILINE)
    if not match:
        raise AssertionError(f"missing token output:\n{output}")
    return [int(value) for value in match.group(1).split()]


class MtpIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not BINARY.exists():
            raise unittest.SkipTest("qwen binary has not been built")

    def test_quantized_mtp_oracles(self) -> None:
        for snapshot in ("qwen_tiny", "qwen_tiny_int8", "qwen_tiny_i4"):
            with self.subTest(snapshot=snapshot):
                got = run_qwen(snapshot, TF="1")
                self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
                self.assertIn("[MTP_ORACLE] 11/11", got.stdout)
                self.assertIn("[ORACLE] 32/32", got.stdout)
                self.assertIn("[GREEDY] 32/32", got.stdout)

    def test_lossless_batched_verification_and_rollback(self) -> None:
        baseline = run_qwen("qwen_tiny", MTP="0", NGEN="32")
        speculative = run_qwen("qwen_tiny", MTP="1", NGEN="32", DRAFT="3")
        rejected = run_qwen(
            "qwen_tiny", MTP="1", NGEN="32", DRAFT="3", SPEC_FORCE_REJECT="1"
        )
        for got in (baseline, speculative, rejected):
            self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        expected = tokens(baseline.stdout)
        self.assertEqual(tokens(speculative.stdout), expected)
        self.assertEqual(tokens(rejected.stdout), expected)
        self.assertRegex(speculative.stderr, r"accepted=24/24 .*target_forwards=8")
        self.assertIn("drafting paused", rejected.stderr)

    def test_multitoken_prompt_uses_lossless_target_prefill(self) -> None:
        # The tiny numerical model intentionally has a 512-token vocabulary
        # while using the official tokenizer. Alternating printable and control
        # bytes prevents BPE merges into vocabulary IDs above the tiny limit.
        prompt = "!\x01!\x02!\x03!\x04!\x05!\x06!\x07!\x08"
        baseline = run_qwen(
            "qwen_tiny", MTP="0", PROMPT=prompt, NGEN="32"
        )
        speculative = run_qwen(
            "qwen_tiny",
            MTP="1",
            PROMPT=prompt,
            NGEN="32",
            DRAFT="1",
            MTP_MIN_ACCEPT="0",
        )
        for got in (baseline, speculative):
            self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        self.assertEqual(tokens(speculative.stdout), tokens(baseline.stdout))

    def test_confidence_admission_falls_back_losslessly(self) -> None:
        baseline = run_qwen("qwen_tiny", MTP="0", NGEN="32")
        conservative = run_qwen(
            "qwen_tiny",
            MTP="1",
            NGEN="32",
            DRAFT="1",
            MTP_MIN_ACCEPT="0",
            MTP_MIN_MARGIN="1000000",
        )
        for got in (baseline, conservative):
            self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        self.assertEqual(tokens(conservative.stdout), tokens(baseline.stdout))
        self.assertRegex(
            conservative.stderr,
            r"\[MTP_ADMIT\] min_margin=1e\+06 skipped=[1-9][0-9]*",
        )
        self.assertIn("timing=", conservative.stderr)
        self.assertIn("/0.000s draft/verify/replay", conservative.stderr)

    def test_ram_budget_derives_cache_and_rejects_oom_plan(self) -> None:
        fitted = run_qwen(
            "qwen_tiny_i4",
            MTP="0",
            LOAD_ONLY="1",
            RAM_GB="0.001",
            RAM_HEADROOM_GB="0",
        )
        self.assertEqual(fitted.returncode, 0, fitted.stdout + fitted.stderr)
        self.assertRegex(
            fitted.stderr,
            r"\[RAM_PLAN\].*expert-bytes=[1-9][0-9]* cap/layer=[1-9][0-9]*",
        )
        rejected = run_qwen(
            "qwen_tiny_i4", MTP="0", LOAD_ONLY="1", RAM_GB="100"
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("RAM plan exceeds physical memory", rejected.stderr)

    def test_tiered_direct_io_and_persistent_expert_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            emap = Path(tmp) / "expert-map.bin"
            options = {
                "MTP": "0",
                "TF": "1",
                "EXPERT_RAM": "3",
                "PIPE": "1",
                "URING": "1",
                "DIRECT": "1",
                "PREFETCH_LOAD": "1",
                "PREFETCH_EXTRA": "1",
                "PREFETCH_MIN_CONF": "0",
                "AUTOPIN": "1",
                "EMAP_PATH": str(emap),
            }
            first = run_qwen("qwen_tiny_i4", **options)
            second = run_qwen("qwen_tiny_i4", **options)
            for got in (first, second):
                self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
                self.assertIn("[ORACLE] 32/32", got.stdout)
                self.assertIn("[GREEDY] 32/32", got.stdout)
                self.assertRegex(got.stderr, r"\[TIERS\].*misses=[1-9][0-9]*")
                self.assertRegex(
                    got.stderr, r"\[PREFETCH\] predictions=[1-9][0-9]*"
                )
                self.assertRegex(got.stderr, r"uring-(batches|fallbacks)=[1-9][0-9]*")
            self.assertIn("[EMAP]", first.stderr)
            self.assertIn("loaded=0 saved=1", first.stderr)
            self.assertIn("loaded=1 saved=1", second.stderr)
            self.assertTrue(emap.exists())


if __name__ == "__main__":
    unittest.main()
