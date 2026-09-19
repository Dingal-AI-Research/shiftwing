"""Native GLM progress and terminal SSE errors used by blocking reviews."""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from openai_server import Engine, APIError
import test_openai_server_qwen as qwen_tests


class GlmNativeProgressTests(unittest.TestCase):
    def test_native_prefill_and_decode_progress_precede_completion(self):
        env = dict(os.environ, OMP_NUM_THREADS="2", GLM53_MAXT="128",
                   GLM53_PREFILL_CHUNK="2", GLM53_EXPERT_GB="0.01")
        engine = Engine(ROOT / "glm53", ROOT / "glm53_tiny_i4", env=env,
                        max_tokens=4, kv_slots=1)
        events = []
        try:
            self.assertEqual(engine.first_model_output_timeout_ms, 3600000)
            self.assertEqual(engine.prefill_first_progress_timeout_ms, 900000)
            self.assertEqual(engine.prefill_progress_stall_timeout_ms, 900000)
            stats = engine.generate("<t001><t002><t003><t004><t005><t006><t007><t008>", 4, 0, 1,
                lambda text: events.append(("text", text)), cache_slot=0,
                on_progress=lambda event: events.append((event["phase"], event)))
        finally:
            engine.close()
        prefill = [value for kind, value in events if kind == "prefill"]
        self.assertTrue(prefill, "GLM must keep the progress watchdog informed")
        self.assertEqual(prefill[0]["event"], "begin")
        self.assertEqual(prefill[-1]["event"], "end")
        self.assertEqual(prefill[-1]["prompt_tokens_prefilled"], stats["prompt_tokens"])
        counts = [event["prompt_tokens_prefilled"] for event in prefill]
        self.assertEqual(counts, sorted(counts))
        self.assertGreater(len(set(counts)), 2, "report intermediate completed chunks")
        percentages = [value["progress_percent"] for value in prefill if "progress_percent" in value]
        self.assertEqual(percentages, sorted(percentages))
        self.assertTrue(any(value["prompt_tokens_prefilled"] == 0 and value.get("progress_percent", 0) > 0 for value in prefill),
                        "report completed layers before the first whole chunk finishes")
        first_text = next(i for i, event in enumerate(events) if event[0] == "text")
        self.assertTrue(all(kind == "prefill" for kind, _ in events[:first_text]))
        decode = [value for kind, value in events if kind == "decode"]
        self.assertEqual([event["completion_tokens"] for event in decode], list(range(1, stats["completion_tokens"] + 1)))
        self.assertTrue(all(event["max_tokens"] == 4 for event in decode))


class GlmContextLimitTests(unittest.TestCase):
    def start_engine(self, context, **overrides):
        env = dict(os.environ, OMP_NUM_THREADS="2", CTX=str(context),
                   GLM53_PREFILL_CHUNK="16", GLM53_EXPERT_GB="0.01")
        env.pop("GLM53_MAXT", None)
        env.update(overrides)
        return Engine(ROOT / "glm53", ROOT / "glm53_tiny_i4", env=env,
                      max_tokens=2, kv_slots=1)

    def test_request_beyond_old_4096_limit_runs_at_configured_64k(self):
        prompt = "<t002>" * 4097
        old = self.start_engine(4096)
        try:
            with self.assertRaises(APIError) as failure:
                old.generate(prompt, 2, 0, 1, lambda text: None)
            self.assertEqual(failure.exception.code, "context_length_exceeded")
            self.assertIn("4096", failure.exception.message)
        finally:
            old.close()
        engine = self.start_engine(65536)
        progress = []
        try:
            stats = engine.generate(prompt, 2, 0, 1, lambda text: None,
                                    on_progress=progress.append)
        finally:
            engine.close()
        self.assertEqual(stats["prompt_tokens"], 4097)
        self.assertTrue(any(p.get("phase") == "prefill" and p.get("event") == "end"
                            and p["prompt_tokens_prefilled"] == 4097 for p in progress))

    def test_explicit_glm_limit_rejects_full_prompt_before_prefill(self):
        engine = self.start_engine(65536, GLM53_MAXT="64")
        progress = []
        try:
            with self.assertRaises(APIError) as failure:
                engine.generate("<t002>" * 64, 2, 0, 1, lambda text: None,
                                on_progress=progress.append)
            self.assertEqual(failure.exception.status, 400)
            self.assertEqual(failure.exception.code, "context_length_exceeded")
            self.assertIn("64", failure.exception.message)
            self.assertEqual(progress, [])
        finally:
            engine.close()


class StreamingFailureTests(qwen_tests.QwenHTTPGatewayTests):
    def check_failure(self, error, expected_code):
        def generate(*args, **kwargs):
            raise error
        self.engine.generate = generate
        with self.post({"model": "qwen-tiny", "messages": [{"role": "user", "content": "Review"}], "stream": True}) as response:
            self.assertEqual(response.status, 200)
            wire = response.read()
        self.assertIn(f'"code":"{expected_code}"'.encode(), wire)
        self.assertNotIn(b"HTTP/1.1", wire, "never send another HTTP header inside SSE")
        self.assertNotIn(b'"finish_reason":"stop"', wire)
        self.assertTrue(wire.endswith(b"data: [DONE]\n\n"))

    def test_native_request_error_is_an_sse_error(self):
        self.check_failure(APIError(400, "Prompt exceeds context", "messages", "context_length_exceeded"), "context_length_exceeded")

    def test_engine_failure_is_an_sse_error(self):
        self.check_failure(RuntimeError("native engine exited unexpectedly"), "engine_error")


if __name__ == "__main__":
    unittest.main()
