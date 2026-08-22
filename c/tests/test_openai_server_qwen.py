import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai_server import (  # noqa: E402
    APIError,
    APIServer,
    ClientCancelled,
    Engine,
    FirstModelOutputTimeoutError,
    generation_options,
    parse_tool_calls,
    render_chat,
)


BINARY = ROOT / ("qwen.exe" if os.name == "nt" else "qwen")


class QwenChatProtocolTests(unittest.TestCase):
    @staticmethod
    def official_render(messages, tools=None, enable_thinking=False) -> str:
        from transformers.utils.chat_template_utils import render_jinja_template

        config = json.loads((ROOT / "qwen_tiny" / "tokenizer_config.json").read_text())
        rendered, _ = render_jinja_template(
            [messages],
            tools=tools,
            chat_template=config["chat_template"],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            add_vision_id=False,
        )
        return rendered[0]

    def test_renders_text_chatml_and_thinking_prefix(self) -> None:
        rendered = render_chat(
            [
                {"role": "system", "content": "System"},
                {"role": "developer", "content": "Developer"},
                {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Again"},
            ],
            enable_thinking=False,
        )
        self.assertEqual(
            rendered,
            "<|im_start|>system\nSystem\n\nDeveloper<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\nHello<|im_end|>\n"
            "<|im_start|>user\nAgain<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_tool_history_uses_qwen3_xml(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        rendered = render_chat(
            [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"city":"London"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": '{"temperature":21}'},
            ],
            tools=tools,
        )
        self.assertIn("# Tools", rendered)
        self.assertIn("<function=weather>", rendered)
        self.assertIn("<parameter=city>\nLondon\n</parameter>", rendered)
        self.assertIn(
            "<|im_start|>user\n<tool_response>\n"
            '{"temperature":21}\n</tool_response><|im_end|>\n',
            rendered,
        )

    def test_renderer_byte_matches_official_qwen_template(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I should call the tool.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": {"city": "London"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temperature":21}'},
        ]
        self.assertEqual(
            render_chat(messages, tools=tools, enable_thinking=False),
            self.official_render(messages, tools=tools, enable_thinking=False),
        )

    def test_cache_renderer_is_exact_extension_of_prior_generation(self) -> None:
        first = render_chat(
            [{"role": "user", "content": "Return exactly: colib ready"}],
            enable_thinking=False,
            cache_prefix_compatible=True,
        )
        continued = render_chat(
            [
                {"role": "user", "content": "Return exactly: colib ready"},
                {"role": "assistant", "content": "colib ready"},
                {"role": "user", "content": "Return that exact text again"},
            ],
            enable_thinking=False,
            cache_prefix_compatible=True,
        )
        cached_transcript = first + "colib ready<|im_end|>\n"
        self.assertTrue(continued.startswith(cached_transcript))

    def test_parses_qwen3_xml_and_removes_reasoning(self) -> None:
        text, calls = parse_tool_calls(
            "<think>private</think>\nAnswer first\n"
            "<tool_call>\n<function=weather>\n"
            "<parameter=city>\nLondon\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        self.assertEqual(text, "Answer first")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "weather")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"city": "London", "days": 3},
        )

    def test_phase9_gateway_is_explicitly_greedy_only(self) -> None:
        self.assertEqual(generation_options({}, 8), (8, 0.0, 1.0, None))
        with self.assertRaisesRegex(APIError, "greedy"):
            generation_options({"temperature": 0.7}, 8)
        with self.assertRaisesRegex(APIError, "Structured"):
            generation_options({"response_format": {"type": "json_object"}}, 8)


class QwenEngineTelemetryParserTests(unittest.TestCase):
    def test_pfpipe_updates_the_profile_turn_with_matching_request_id(self) -> None:
        payload = (
            b"PERF request-a 9 1 2 3 4 5 6\n"
            b"PERF request-b 8 7 6 5 4 3 2\n"
            b"PFPIPE request-a 1 7 23 4096 1.25 0.5 6.75 8.5\n"
        )
        engine = Engine.__new__(Engine)
        engine.process = type(
            "StubProcess", (), {"stdout": io.BytesIO(payload)}
        )()
        engine.trace = False
        engine.profile = []
        engine.profile_seq = 0
        engine.closed = True

        engine._dispatch_stdout()

        self.assertEqual(engine.profile_seq, 2)
        turn = engine.profile[0]
        self.assertEqual(turn["request_id"], "request-a")
        self.assertIs(turn["prefill_load_pipeline_active"], True)
        self.assertEqual(turn["prefill_pipeline_batches"], 7)
        self.assertEqual(turn["prefill_pipeline_experts"], 23)
        self.assertEqual(turn["prefill_pipeline_bytes"], 4096)
        self.assertEqual(turn["prefill_pipeline_producer_load_s"], 1.25)
        self.assertEqual(turn["prefill_pipeline_consumer_wait_s"], 0.5)
        self.assertEqual(turn["prefill_pipeline_consumer_compute_s"], 6.75)
        self.assertEqual(turn["prefill_pipeline_wall_s"], 8.5)
        self.assertNotIn("prefill_pipeline_batches", engine.profile[1])


class QwenEngineMuxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not BINARY.exists():
            raise unittest.SkipTest("qwen binary has not been built")

    def test_gateway_dispatches_real_mux_data_and_done(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "EXPERT_RAM": "4",
                "PREFETCH_THREADS": "0",
                "SERVE_RESIDENT": "1",
            }
        )
        engine = Engine(
            BINARY,
            ROOT / "qwen_tiny_i4",
            max_tokens=2,
            env=env,
            kv_slots=2,
        )
        chunks: list[str] = []
        try:
            stats = engine.generate("!", 2, 0.0, 1.0, chunks.append, cache_slot=1)
        finally:
            engine.close()
        self.assertTrue("".join(chunks))
        self.assertEqual(stats["completion_tokens"], 2)
        self.assertEqual(stats["prompt_tokens"], 1)
        self.assertGreater(engine.hwinfo["cores"], 0)
        self.assertEqual(engine.emap["rows"], 5)
        self.assertEqual(engine.emap["cols"], 8)
        self.assertEqual(len(engine.hits), 2 * ((5 * 8 + 7) // 8))
        self.assertEqual(engine.profile_seq, 1)
        self.assertEqual(engine.profile[-1]["request_id"], "1")
        self.assertGreaterEqual(engine.profile[-1]["wall_s"], 0.0)
        self.assertEqual(engine.profile[-1]["prompt_tokens"], 1)
        self.assertEqual(engine.profile[-1]["completion_tokens"], 2)
        self.assertGreater(
            engine.profile[-1]["expert_cpu_hits"]
            + engine.profile[-1]["expert_gpu_hits"]
            + engine.profile[-1]["expert_misses"],
            0,
        )
        self.assertGreaterEqual(engine.profile[-1]["expert_read_bytes"], 0)

        self.assertGreaterEqual(engine.profile[-1]["expert_direct_bytes"], 0)
        self.assertGreaterEqual(engine.profile[-1]["expert_uring_batches"], 0)
        self.assertGreaterEqual(engine.profile[-1]["expert_uring_reads"], 0)
        self.assertGreater(
            engine.profile[-1]["decode_expert_cpu_hits"]
            + engine.profile[-1]["decode_expert_gpu_hits"]
            + engine.profile[-1]["decode_expert_misses"],
            0,
        )
        self.assertGreaterEqual(engine.profile[-1]["decode_expert_read_bytes"], 0)
        self.assertGreaterEqual(engine.profile[-1]["decode_expert_direct_bytes"], 0)
        self.assertGreaterEqual(engine.profile[-1]["decode_wall_s"], 0.0)
        self.assertGreaterEqual(
            engine.profile[-1]["decode_expert_uring_batches"], 0
        )
        self.assertGreaterEqual(engine.profile[-1]["decode_expert_uring_reads"], 0)
        self.assertGreaterEqual(engine.profile[-1]["decode_expert_disk_s"], 0.0)
        self.assertGreaterEqual(engine.profile[-1]["decode_expert_matmul_s"], 0.0)
        self.assertGreaterEqual(engine.profile[-1]["decode_attention_s"], 0.0)
        self.assertGreaterEqual(engine.profile[-1]["decode_lm_head_s"], 0.0)
        self.assertGreaterEqual(stats["cache_hit_percent"], 0.0)
        self.assertLessEqual(stats["cache_hit_percent"], 100.0)
        if os.environ.get("COLI_CUDA") == "1":
            self.assertGreater(engine.resident["layers"], 0)
            self.assertGreater(engine.resident["device_moe"], 0)
            self.assertEqual(engine.resident["host_moe"], 0)
            self.assertGreater(engine.resident["router_d2h_bytes"], 0)
            self.assertGreater(engine.tiers["vram"], 0)
            self.assertEqual(
                engine.tiers["vram"]
                + engine.tiers["ram"]
                + engine.tiers["disk"],
                engine.emap["rows"] * engine.emap["cols"],
            )

    def test_close_does_not_discard_result_when_killed_child_stays_in_io(self) -> None:
        class FakePipe:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = FakePipe()
                self.stdout = FakePipe()
                self.wait_timeouts: list[int] = []
                self.terminated = False
                self.killed = False

            def poll(self):
                return None

            def wait(self, timeout):
                self.wait_timeouts.append(timeout)
                raise subprocess.TimeoutExpired(["qwen"], timeout)

            def terminate(self) -> None:
                self.terminated = True

            def kill(self) -> None:
                self.killed = True

        class FakeDispatcher:
            def join(self, timeout) -> None:
                self.timeout = timeout

        engine = Engine.__new__(Engine)
        engine.pending_lock = threading.Lock()
        engine.write_lock = threading.Lock()
        engine.closed = False
        engine.process = FakeProcess()
        engine.dispatcher = FakeDispatcher()
        engine._fail_pending = lambda error: None

        engine.close()

        self.assertEqual(engine.process.wait_timeouts, [10, 5, 30])
        self.assertTrue(engine.process.terminated)
        self.assertTrue(engine.process.killed)
        self.assertTrue(engine.process.stdin.closed)
        self.assertTrue(engine.process.stdout.closed)

    def test_graceful_close_persists_learned_expert_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            emap = Path(directory) / "expert_map.bin"
            env = os.environ.copy()
            env.update(
                {
                    "EXPERT_RAM": "2",
                    "PREFETCH_THREADS": "0",
                    "EMAP_PATH": str(emap),
                }
            )
            engine = Engine(
                BINARY,
                ROOT / "qwen_tiny_i4",
                max_tokens=4,
                env=env,
                kv_slots=1,
            )
            try:
                engine.generate("!", 4, 0.0, 1.0, lambda _piece: None)
            finally:
                engine.close()
            payload = emap.read_bytes()
            self.assertEqual(payload[:8], b"COLIEMAP")
            self.assertGreater(len(payload), 20)

    def test_persistent_pipe_accepts_two_simultaneous_submits(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "EXPERT_RAM": "4",
                "PREFETCH_THREADS": "0",
                "SERVE_RESIDENT": "1",
            }
        )
        engine = Engine(
            BINARY,
            ROOT / "qwen_tiny_i4",
            max_tokens=2,
            env=env,
            kv_slots=2,
        )
        barrier = threading.Barrier(2)
        results = [None, None]
        errors = []

        def request(slot):
            try:
                barrier.wait()
                pieces = []
                results[slot] = engine.generate(
                    "!", 2, 0.0, 1.0, pieces.append, cache_slot=slot
                )
            except Exception as error:
                errors.append(error)

        workers = [
            threading.Thread(target=request, args=(slot,)) for slot in range(2)
        ]
        try:
            for worker in workers:
                worker.start()
            deadline = time.monotonic() + 10
            for worker in workers:
                worker.join(max(0, deadline - time.monotonic()))
            self.assertFalse(
                any(worker.is_alive() for worker in workers),
                "simultaneous persistent-pipe submissions stalled",
            )
            self.assertEqual(errors, [])
            self.assertTrue(all(result is not None for result in results))
            self.assertTrue(
                all(result["completion_tokens"] == 2 for result in results)
            )
        finally:
            engine.close()
            for worker in workers:
                worker.join(timeout=2)

    def test_gateway_cancels_real_mux_after_first_data(self) -> None:
        env = os.environ.copy()
        env.update({"EXPERT_RAM": "4", "PREFETCH_THREADS": "0"})
        engine = Engine(
            BINARY,
            ROOT / "qwen_tiny_i4",
            max_tokens=64,
            env=env,
            kv_slots=1,
        )
        chunks: list[str] = []
        try:
            with self.assertRaises(ClientCancelled):
                engine.generate(
                    "!",
                    64,
                    0.0,
                    1.0,
                    chunks.append,
                    cache_slot=0,
                    cancelled=lambda: bool(chunks),
                )
        finally:
            engine.close()
        self.assertEqual(len(chunks), 1)

    def test_cancelled_slot_releases_while_peer_completes(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "EXPERT_RAM": "4",
                "PREFETCH_THREADS": "0",
                "SERVE_RESIDENT": "1",
            }
        )
        engine = Engine(
            BINARY,
            ROOT / "qwen_tiny_i4",
            max_tokens=32,
            env=env,
            kv_slots=2,
        )
        barrier = threading.Barrier(2)
        outcome = {}
        errors = []

        def cancel_slot():
            pieces = []
            try:
                barrier.wait()
                engine.generate(
                    "!",
                    32,
                    0.0,
                    1.0,
                    pieces.append,
                    cache_slot=0,
                    cancelled=lambda: bool(pieces),
                )
                errors.append("cancelled request completed")
            except ClientCancelled:
                outcome["cancelled_pieces"] = len(pieces)
            except Exception as error:
                errors.append(str(error))

        def peer_slot():
            pieces = []
            try:
                barrier.wait()
                outcome["peer"] = engine.generate(
                    "!", 4, 0.0, 1.0, pieces.append, cache_slot=1
                )
            except Exception as error:
                errors.append(str(error))

        workers = [
            threading.Thread(target=cancel_slot),
            threading.Thread(target=peer_slot),
        ]
        try:
            for worker in workers:
                worker.start()
            deadline = time.monotonic() + 30
            for worker in workers:
                worker.join(max(0, deadline - time.monotonic()))
            self.assertFalse(any(worker.is_alive() for worker in workers))
            self.assertEqual(errors, [])
            self.assertEqual(outcome["cancelled_pieces"], 1)
            self.assertEqual(outcome["peer"]["completion_tokens"], 4)
            reuse = engine.generate("!", 1, 0.0, 1.0, lambda _: None, cache_slot=0)
            self.assertEqual(reuse["completion_tokens"], 1)
        finally:
            engine.close()
            for worker in workers:
                worker.join(timeout=2)


class FakeEngine:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.chunks = ["Hel", "lo"]
        self.first_output_timeout = False

    def generate(
        self,
        prompt,
        maximum,
        temperature,
        top_p,
        on_text,
        cache_slot=0,
        cancelled=None,
        grammar=None,
        on_progress=None,
    ):
        self.prompts.append(prompt)
        if on_progress:
            on_progress({
                "phase": "prefill", "event": "progress",
                "prompt_tokens_total": 3,
                "prompt_tokens_cached": 0,
                "prompt_tokens_prefilled": 2,
                "elapsed_ms": 12,
            })
        if self.first_output_timeout:
            raise FirstModelOutputTimeoutError("No real model output within 180000 ms")
        for chunk in self.chunks:
            on_text(chunk)
        return {
            "completion_tokens": 2,
            "tokens_per_second": 10.0,
            "cache_hit_percent": 0.0,
            "rss_gb": 0.1,
            "prompt_tokens": 3,
            "length_limited": False,
            "ttft_ms": 25.0,
            "prefill_time_ms": 20.0,
            "decode_time_ms": 200.0,
            "wall_time_ms": 225.0,
            "profile": {
                "expert_cpu_hits": 2,
                "expert_gpu_hits": 3,
                "expert_misses": 1,
                "expert_read_bytes": 4096,
                "expert_direct_bytes": 4096,
            },
        }


class QwenHTTPGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeEngine()
        self.server = APIServer(
            ("127.0.0.1", 0),
            self.engine,
            "qwen-tiny",
            max_tokens=8,
            kv_slots=2,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def post(self, body):
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urlopen(request, timeout=5)

    def test_nonstreaming_openai_chat_response(self) -> None:
        with self.post(
            {
                "model": "qwen-tiny",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0,
                "top_p": 1,
                "cache_slot": 1,
            }
        ) as response:
            payload = json.load(response)
        self.assertEqual(payload["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(payload["usage"]["total_tokens"], 5)
        metrics = payload["colib_metrics"]
        self.assertEqual(metrics["object"], "colib.metrics")
        self.assertEqual(metrics["decode_tokens_per_second"], 10.0)
        self.assertEqual(metrics["completion_tokens"], 2)
        self.assertEqual(metrics["cache_hits"], 5)
        self.assertEqual(metrics["disk_bytes"], 4096)
        self.assertIn("<|im_start|>user\nHi<|im_end|>", self.engine.prompts[0])

    def test_streaming_openai_chat_finishes_with_done(self) -> None:
        with self.post(
            {
                "model": "qwen-tiny",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "temperature": 0,
                "top_p": 1,
            }
        ) as response:
            wire = response.read()
        self.assertIn(b'"content":"Hel"', wire)
        self.assertIn(b'"content":"lo"', wire)
        self.assertIn(b'"object":"colib.progress"', wire)
        self.assertIn(b'"object":"colib.metrics"', wire)
        self.assertIn(b'"decode_tokens_per_second":10.0', wire)
        self.assertIn(b'"prompt_tokens_prefilled":2', wire)
        self.assertIn(b'"schema_version":1', wire)
        self.assertTrue(wire.endswith(b"data: [DONE]\n\n"))

    def test_streaming_timeout_is_reported_in_band_after_headers(self) -> None:
        self.engine.first_output_timeout = True
        with self.post(
            {
                "model": "qwen-tiny",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "temperature": 0,
                "top_p": 1,
            }
        ) as response:
            self.assertEqual(response.status, 200)
            wire = response.read()
        self.assertIn(b'"code":"first_model_output_timeout"', wire)
        self.assertTrue(wire.endswith(b"data: [DONE]\n\n"))

    def test_nonstreaming_separates_qwen_reasoning(self) -> None:
        self.engine.chunks = ["private chain", "</think>\n\nPublic answer"]
        with self.post(
            {
                "model": "qwen-tiny",
                "messages": [{"role": "user", "content": "Think"}],
                "enable_thinking": True,
                "temperature": 0,
                "top_p": 1,
            }
        ) as response:
            payload = json.load(response)
        message = payload["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "private chain")
        self.assertEqual(message["content"], "Public answer")

    def test_streaming_separates_split_reasoning_close_tag(self) -> None:
        self.engine.chunks = ["private", " chain</thi", "nk>\n\nPublic"]
        with self.post(
            {
                "model": "qwen-tiny",
                "messages": [{"role": "user", "content": "Think"}],
                "enable_thinking": True,
                "stream": True,
                "temperature": 0,
                "top_p": 1,
            }
        ) as response:
            wire = response.read()
        events = [
            json.loads(line[6:])
            for line in wire.decode().splitlines()
            if line.startswith("data: {")
        ]
        progress = [event for event in events if event["object"] == "colib.progress"]
        metrics = [event for event in events if event["object"] == "colib.metrics"]
        self.assertTrue(progress)
        self.assertEqual(len(metrics), 1)
        deltas = [event["choices"][0]["delta"] for event in events
                  if event["object"] not in ("colib.progress", "colib.metrics")
                  and event.get("choices")]
        reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)
        content = "".join(delta.get("content", "") for delta in deltas)
        self.assertEqual(reasoning, "private chain")
        self.assertEqual(content, "\n\nPublic")
        self.assertNotIn(b"</think>", wire)


if __name__ == "__main__":
    unittest.main()
