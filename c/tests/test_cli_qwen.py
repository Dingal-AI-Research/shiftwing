import json
import os
import runpy
import socket
import subprocess
import time
import unittest
from unittest import mock
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "shiftwing"


class ShiftwingCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(CLI), *args],
            cwd=ROOT.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

    def test_help_lists_gate9_commands(self) -> None:
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("chat", "serve", "web", "doctor", "convert"):
            self.assertIn(command, result.stdout)

    def test_version_matches_release_package(self) -> None:
        result = self.run_cli("--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "shiftwing 0.1.0")

    def test_doctor_accepts_tiny_container(self) -> None:
        result = self.run_cli(
            "doctor",
            "--model",
            str(ROOT / "qwen_tiny_i4"),
            "--kv-slots",
            "2",
            "--context",
            "64",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(all(item["status"] in {"pass", "skip"} for item in report["checks"]))

    def test_convert_help_uses_workspace_environment(self) -> None:
        result = self.run_cli("convert", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--min-free-gb", result.stdout)
        self.assertIn("--selftest", result.stdout)

    def test_server_uses_workspace_environment(self) -> None:
        namespace = runpy.run_path(str(CLI), run_name="shiftwing_cli_test")
        parser = namespace["build_parser"]()
        args = parser.parse_args(
            ["serve", "--model", str(ROOT / "qwen_tiny_i4")]
        )
        argv = namespace["_server_argv"](args)
        self.assertEqual(Path(argv[0]), namespace["PYTHON"])
        self.assertTrue(Path(argv[0]).is_file())

    def test_runtime_defaults_to_int4_despite_ambient_sidecars(self) -> None:
        namespace = runpy.run_path(str(CLI), run_name="shiftwing_cli_test")
        parser = namespace["build_parser"]()
        args = parser.parse_args(
            ["serve", "--model", str(ROOT / "qwen_tiny_i4")]
        )
        ambient = {
            "EXPERT_Q2": "1",
            "EXPERT_Q3": "1",
            "Q3_NATIVE": "1",
            "Q3_ROUTE_ATLAS": "1",
        }
        with mock.patch.dict(os.environ, ambient):
            env = namespace["_runtime_env"](args)
        for name in ambient:
            self.assertEqual(env[name], "0", name)

    def test_runtime_enables_q3_only_when_requested(self) -> None:
        namespace = runpy.run_path(str(CLI), run_name="shiftwing_cli_test")
        parser = namespace["build_parser"]()
        args = parser.parse_args(
            [
                "serve",
                "--model",
                str(ROOT / "qwen_tiny_i4"),
                "--cuda",
                "--expert-q3",
            ]
        )
        env = namespace["_runtime_env"](args)
        self.assertEqual(env["EXPERT_Q3"], "1")
        self.assertEqual(env["Q3_NATIVE"], "0")
        self.assertEqual(env["Q3_ROUTE_ATLAS"], "0")

    def test_route_atlas_uses_fast_expanded_q4_unless_native_is_explicit(self) -> None:
        namespace = runpy.run_path(str(CLI), run_name="shiftwing_cli_test")
        parser = namespace["build_parser"]()
        common = [
            "serve", "--model", str(ROOT / "qwen_tiny_i4"), "--cuda",
            "--expert-q3", "--q3-route-atlas",
        ]
        expanded = namespace["_runtime_env"](parser.parse_args(common))
        self.assertEqual(expanded["Q3_ROUTE_ATLAS"], "1")
        self.assertEqual(expanded["Q3_NATIVE"], "0")
        native = namespace["_runtime_env"](
            parser.parse_args([*common, "--q3-native"])
        )
        self.assertEqual(native["Q3_ROUTE_ATLAS"], "1")
        self.assertEqual(native["Q3_NATIVE"], "1")


    def test_runtime_refuses_missing_selected_q3_tensor(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "SNAP": str(ROOT / "qwen_tiny_i4"),
                "EXPERT_Q2": "0",
                "EXPERT_Q3": "1",
                "LOAD_ONLY": "1",
            }
        )
        result = subprocess.run(
            [str(ROOT / "qwen")],
            cwd=ROOT.parent,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing selected int3 expert tensor", result.stderr)

    def test_web_serves_bundle_and_real_tiny_chat(self) -> None:
        if not (ROOT.parent / "web" / "dist" / "index.html").is_file():
            self.skipTest("web bundle has not been built")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = os.environ.copy()
        env.update({"EXPERT_RAM": "4", "PREFETCH_THREADS": "0"})
        process = subprocess.Popen(
            [
                str(CLI),
                "web",
                "--no-build",
                "--model",
                str(ROOT / "qwen_tiny_i4"),
                "--port",
                str(port),
                "--max-tokens",
                "2",
                "--kv-slots",
                "2",
            ],
            cwd=ROOT.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 15
            while True:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                        break
                except OSError:
                    if process.poll() is not None:
                        stderr = process.stderr.read().decode(errors="replace")
                        self.fail(f"web server exited early: {stderr}")
                    if time.monotonic() >= deadline:
                        self.fail("web server did not become ready")
                    time.sleep(0.05)
            with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                page = response.read()
            self.assertIn(b"<title>shiftwing</title>", page)
            request = Request(
                f"http://127.0.0.1:{port}/v1/completions",
                data=json.dumps(
                    {
                        "model": "qwen3.5-shiftwing",
                        "prompt": "!",
                        "temperature": 0,
                        "top_p": 1,
                        "max_tokens": 2,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=10) as response:
                    payload = json.load(response)
            except HTTPError as error:
                body = error.read().decode(errors="replace")
                process.terminate()
                process.wait(timeout=5)
                stderr = process.stderr.read().decode(errors="replace")
                self.fail(f"chat request failed: {body}\n{stderr}")
            self.assertTrue(payload["choices"][0]["text"])
            self.assertEqual(payload["usage"]["completion_tokens"], 2)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
