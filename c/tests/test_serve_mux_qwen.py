import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / ("qwen.exe" if os.name == "nt" else "qwen")
ADVISORY = {
    "HWINFO",
    "TIERS",
    "EMAP",
    "HITS",
    "PERF",
    "DPERF",
    "RESIDENT",
    "CACHE",
    "DCACHE",
    "PFPIPE",
    "Q3NATIVE",
    "Q3ATLAS",
}


def run_engine(
    wire: bytes,
    *,
    env_overrides: dict[str, str] | None = None,
    resident: bool = True,
    session_dir: Path | None = None,
    suffix_block: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env.update(
        {
            "SNAP": str(ROOT / "qwen_tiny_i4"),
            "MTP": "0",
            "SERVE_BATCH": "1",
            "SERVE_RESIDENT": "1" if resident else "0",
            "KV_SLOTS": "2",
            "EXPERT_RAM": "4",
            "PREFETCH_THREADS": "0",
        }
    )
    if session_dir is not None:
        env["SESSION_DIR"] = str(session_dir)
    if env_overrides:
        env.update(env_overrides)
    if suffix_block:
        env["SERVE_SUFFIX_BLOCK"] = "1"
    return subprocess.run(
        [str(BINARY)],
        input=wire,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=30,
        check=False,
    )


def parse_output(payload: bytes) -> list[tuple[str, int, bytes | str]]:
    cursor = 0
    events: list[tuple[str, int, bytes | str]] = []
    ready = b"\x01\x01READY\x01\x01\n"
    if not payload.startswith(ready):
        raise AssertionError(f"missing READY: {payload[:100]!r}")
    cursor = len(ready)
    while cursor < len(payload):
        end = payload.find(b"\n", cursor)
        if end < 0:
            raise AssertionError("unterminated response header")
        header = payload[cursor:end].decode("ascii")
        cursor = end + 1
        fields = header.split()
        if fields[0] == "DATA":
            request_id, size = int(fields[1]), int(fields[2])
            body = payload[cursor : cursor + size]
            cursor += size
            if payload[cursor : cursor + 1] != b"\n":
                raise AssertionError("DATA terminator")
            cursor += 1
            events.append(("DATA", request_id, body))
        elif fields[0] == "DONE":
            events.append(("DONE", int(fields[1]), header))
        elif fields[0] == "ERROR":
            events.append(("ERROR", int(fields[1]), fields[2]))
        elif fields[0] in ADVISORY:
            events.append((fields[0], 0, header))
        else:
            raise AssertionError(f"unexpected mux line: {header}")
    return events


class QwenMuxIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not BINARY.exists():
            raise unittest.SkipTest("qwen binary has not been built")

    def test_streams_byte_counted_tokens_and_done(self) -> None:
        got = run_engine(b"SUBMIT 42 0 1 3 0 1\n!\n")
        self.assertEqual(got.returncode, 0, got.stderr.decode(errors="replace"))
        events = parse_output(got.stdout)
        self.assertEqual(sum(event[0] == "DATA" for event in events), 3)
        self.assertEqual(events[-1][0:2], ("DONE", 42))
        self.assertIn(" STAT 3 ", events[-1][2])
        kinds = {event[0] for event in events}
        self.assertTrue((ADVISORY - {"RESIDENT", "Q3NATIVE", "Q3ATLAS"}) <= kinds)
        if os.environ.get("COLI_CUDA") == "1":
            resident = next(
                str(event[2]) for event in events if event[0] == "RESIDENT"
            )
            self.assertEqual(len(resident.split()), 9)
            fields = resident.split()
            self.assertGreater(int(fields[2]), 0)
            self.assertGreater(int(fields[3]), 0)
            self.assertEqual(int(fields[4]), 0)
        perf = next(str(event[2]) for event in events if event[0] == "PERF")
        self.assertEqual(perf.split()[1], "42")
        self.assertGreaterEqual(len(perf.split()), 9)
        emap = next(str(event[2]) for event in events if event[0] == "EMAP")
        _, rows, cols, payload = emap.split()
        self.assertEqual(len(payload), 2 * int(rows) * int(cols))
        hits = next(str(event[2]) for event in events if event[0] == "HITS")
        _, rows, cols, payload = hits.split()
        cache = next(str(event[2]) for event in events if event[0] == "CACHE")
        dcache = next(str(event[2]) for event in events if event[0] == "DCACHE")
        pfpipe = next(str(event[2]) for event in events if event[0] == "PFPIPE")
        self.assertEqual(len(cache.split()), 9)
        self.assertEqual(len(dcache.split()), 9)
        self.assertEqual(len(pfpipe.split()), 10)
        self.assertEqual(pfpipe.split()[1:3], ["42", "0"])
        self.assertEqual(len(payload), 2 * ((int(rows) * int(cols) + 7) // 8))
        if os.environ.get("COLI_CUDA") == "1":
            stderr = got.stderr.decode(errors="replace")
            self.assertRegex(stderr, r"GDN-calls=[1-9][0-9]*")
            self.assertRegex(stderr, r"GQA-calls=[1-9][0-9]*")
            if os.environ.get("CUDA_PROFILE_STAGES") == "1":
                self.assertEqual(len(perf.split()), 18)
                self.assertGreaterEqual(int(perf.split()[9]), 0)
                for value in perf.split()[10:18]:
                    self.assertGreaterEqual(float(value), 0.0)

    def test_ordered_prefill_expert_io_batch_preserves_exact_output(self) -> None:
        # Tiny vocabulary-safe byte-fallback tokens; ordinary multi-byte text
        # can merge to a tokenizer id above this fixture's truncated vocab.
        prompt = b"!\x00\x01"
        wire = (
            f"SUBMIT 43 0 {len(prompt)} 5 0 1\n".encode()
            + prompt
            + b"\n"
        )
        common = {
            "PIPE": "1",
            "URING": "1",
            "URING_PERSIST": "1",
        }
        control = run_engine(
            wire,
            env_overrides={**common, "PREFILL_EXPERT_BATCH": "1"},
        )
        candidate = run_engine(
            wire,
            env_overrides={**common, "PREFILL_EXPERT_BATCH": "4"},
        )
        self.assertEqual(
            control.returncode, 0, control.stderr.decode(errors="replace")
        )
        self.assertEqual(
            candidate.returncode, 0, candidate.stderr.decode(errors="replace")
        )

        def semantic(payload: bytes):
            events = parse_output(payload)
            return [
                event for event in events if event[0] in {"DATA", "ERROR"}
            ] + [
                (event[0], event[1])
                for event in events
                if event[0] == "DONE"
            ]

        self.assertEqual(semantic(control.stdout), semantic(candidate.stdout))
        control_uring = re.search(
            rb"uring-batches=([0-9]+)", control.stderr
        )
        candidate_uring = re.search(
            rb"uring-batches=([0-9]+)", candidate.stderr
        )
        self.assertIsNotNone(control_uring)
        self.assertIsNotNone(candidate_uring)
        self.assertLess(
            int(candidate_uring.group(1)), int(control_uring.group(1))
        )

    def test_cold_prefill_device_threshold_preserves_exact_output(self) -> None:
        # A cold prefill expert may execute on CUDA once it serves enough rows
        # to amortize uploading its weights. The upload uses scratch outside the
        # weight cache, so decode residents are never evicted. Output must not
        # change, and a threshold no batch can reach must keep the host path.
        prompt = b"!\x00\x01"
        wire = (
            f"SUBMIT 44 0 {len(prompt)} 5 0 1\n".encode()
            + prompt
            + b"\n"
        )
        common = {"PIPE": "1", "URING": "1", "URING_PERSIST": "1"}
        control = run_engine(
            wire, env_overrides={**common, "PREFILL_COLD_DEVICE": "0"}
        )
        unreachable = run_engine(
            wire, env_overrides={**common, "PREFILL_COLD_DEVICE": "100000"}
        )
        self.assertEqual(
            control.returncode, 0, control.stderr.decode(errors="replace")
        )
        self.assertEqual(
            unreachable.returncode, 0, unreachable.stderr.decode(errors="replace")
        )

        def semantic(payload: bytes):
            events = parse_output(payload)
            return [
                event for event in events if event[0] in {"DATA", "ERROR"}
            ] + [
                (event[0], event[1])
                for event in events
                if event[0] == "DONE"
            ]

        self.assertEqual(semantic(control.stdout), semantic(unreachable.stdout))
        # A threshold beyond any achievable batch must leave the device path
        # unused, which is what makes the flag safe to default off.
        for payload in (control.stderr, unreachable.stderr):
            match = re.search(rb"\[COLD_PREFILL\] experts=([0-9]+)", payload)
            if match is not None:
                self.assertEqual(int(match.group(1)), 0)

    def test_prefill_load_pipeline_emits_deltas_and_preserves_output(self) -> None:
        prompt = b"!\x00\x01"
        wire = (
            f"SUBMIT 45 0 {len(prompt)} 5 0 1\n".encode()
            + prompt
            + b"\n"
        )
        common = {
            "PIPE": "1",
            "URING": "0",
            "DIRECT": "0",
            "PREFETCH_LOAD": "0",
            "PREFILL_EXPERT_BATCH": "1",
            "EXPERT_RAM": "2",
        }
        control = run_engine(wire, env_overrides=common)
        candidate = run_engine(
            wire,
            env_overrides={
                **common,
                "PREFILL_CACHE_BYPASS": "1",
                "PREFILL_LOAD_PIPELINE": "1",
            },
        )
        self.assertEqual(
            control.returncode, 0, control.stderr.decode(errors="replace")
        )
        self.assertEqual(
            candidate.returncode, 0, candidate.stderr.decode(errors="replace")
        )

        def semantic(payload: bytes):
            events = parse_output(payload)
            return [
                event
                for event in events
                if event[0] in {"DATA", "ERROR"}
            ] + [(event[0], event[1]) for event in events if event[0] == "DONE"]

        self.assertEqual(semantic(control.stdout), semantic(candidate.stdout))
        control_row = next(
            str(event[2]).split()
            for event in parse_output(control.stdout)
            if event[0] == "PFPIPE"
        )
        candidate_row = next(
            str(event[2]).split()
            for event in parse_output(candidate.stdout)
            if event[0] == "PFPIPE"
        )
        self.assertEqual(control_row[1:6], ["45", "0", "0", "0", "0"])
        self.assertEqual(candidate_row[1:3], ["45", "1"])
        self.assertGreater(int(candidate_row[3]), 1)
        self.assertGreaterEqual(int(candidate_row[4]), int(candidate_row[3]))
        self.assertGreater(int(candidate_row[5]), 0)
        for value in candidate_row[6:10]:
            self.assertGreaterEqual(float(value), 0.0)
        self.assertGreater(float(candidate_row[6]), 0.0)
        self.assertGreater(float(candidate_row[8]), 0.0)
        self.assertGreater(float(candidate_row[9]), 0.0)

    def test_two_active_slots_keep_prefill_pipeline_deltas_request_exact(self) -> None:
        prompt_a = b"!"
        prompt_b = b"?\x00\x01"

        def submit(request_id: int, slot: int, prompt: bytes) -> bytes:
            return (
                f"SUBMIT {request_id} {slot} {len(prompt)} 2 0 1\n".encode()
                + prompt
                + b"\n"
            )

        env = {
            "PIPE": "1",
            "URING": "0",
            "DIRECT": "0",
            "PREFETCH_LOAD": "0",
            "PREFILL_EXPERT_BATCH": "4",
            "PREFILL_CACHE_BYPASS": "1",
            "PREFILL_LOAD_PIPELINE": "1",
        }
        wire_a = submit(51, 0, prompt_a)
        wire_b = submit(52, 1, prompt_b)
        standalone_a = run_engine(wire_a, env_overrides=env)
        standalone_b = run_engine(wire_b, env_overrides=env)
        combined = run_engine(wire_a + wire_b, env_overrides=env)
        for result in (standalone_a, standalone_b, combined):
            self.assertEqual(
                result.returncode, 0, result.stderr.decode(errors="replace")
            )

        def rows(payload: bytes) -> dict[int, list[str]]:
            parsed = {}
            for event in parse_output(payload):
                if event[0] == "PFPIPE":
                    fields = str(event[2]).split()
                    parsed[int(fields[1])] = fields
            return parsed

        row_a = rows(standalone_a.stdout)[51]
        row_b = rows(standalone_b.stdout)[52]
        combined_rows = rows(combined.stdout)
        # Active flag, batch count, expert count and actual bytes are exact;
        # wall-clock fields are intentionally not compared across processes.
        self.assertEqual(combined_rows[51][2:6], row_a[2:6])
        self.assertEqual(combined_rows[52][2:6], row_b[2:6])

    def test_queued_cancel_prevents_decode_and_releases_slot(self) -> None:
        wire = (
            b"SUBMIT 7 0 1 8 0 1\n!\n"
            b"CANCEL 7\n"
            b"SUBMIT 8 0 1 1 0 1\n?\n"
        )
        got = run_engine(wire)
        self.assertEqual(got.returncode, 0, got.stderr.decode(errors="replace"))
        events = parse_output(got.stdout)
        self.assertIn(("ERROR", 7, "CANCELLED"), events)
        self.assertFalse(any(event[0] == "DATA" and event[1] == 7 for event in events))
        self.assertTrue(any(event[0] == "DONE" and event[1] == 8 for event in events))

    def test_two_active_slots_stream_to_independent_request_ids(self) -> None:
        wire = (
            b"SUBMIT 21 0 1 2 0 1\n!\n"
            b"SUBMIT 22 1 1 2 0 1\n?\n"
        )
        got = run_engine(wire)
        self.assertEqual(got.returncode, 0, got.stderr.decode(errors="replace"))
        events = parse_output(got.stdout)
        for request_id in (21, 22):
            self.assertEqual(
                sum(event[0] == "DATA" and event[1] == request_id for event in events),
                2,
            )
            self.assertTrue(
                any(event[0] == "DONE" and event[1] == request_id for event in events)
            )

    def test_resident_and_snapshot_mux_emit_identical_token_bytes(self) -> None:
        wire = (
            b"SUBMIT 31 0 1 5 0 1\n!\n"
            b"SUBMIT 32 1 1 5 0 1\n?\n"
        )
        resident = run_engine(wire, resident=True)
        snapshot = run_engine(wire, resident=False)
        self.assertEqual(resident.returncode, 0, resident.stderr.decode(errors="replace"))
        self.assertEqual(snapshot.returncode, 0, snapshot.stderr.decode(errors="replace"))

        def semantic_events(payload):
            return [
                event
                for event in parse_output(payload)
                if event[0] in {"DATA", "ERROR"}
            ] + [
                (event[0], event[1])
                for event in parse_output(payload)
                if event[0] == "DONE"
            ]

        self.assertEqual(semantic_events(resident.stdout), semantic_events(snapshot.stdout))

    def test_resident_session_warm_reload_exact_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory)
            first = run_engine(
                b"SUBMIT 71 0 1 8 0 1\n!\nCANCEL 71\n",
                resident=True,
                session_dir=session_dir,
            )
            self.assertEqual(first.returncode, 0, first.stderr.decode(errors="replace"))
            first_events = parse_output(first.stdout)
            self.assertIn(("ERROR", 71, "CANCELLED"), first_events)
            self.assertFalse(
                any(event[0] == "DATA" and event[1] == 71 for event in first_events)
            )
            extended = b"!a"
            wire = (
                f"SUBMIT 72 0 {len(extended)} 3 0 1\n".encode()
                + extended
                + b"\n"
            )
            restored = run_engine(wire, resident=True, session_dir=session_dir)
            fresh = run_engine(wire, resident=True)
            self.assertEqual(
                restored.returncode, 0, restored.stderr.decode(errors="replace")
            )
            self.assertEqual(fresh.returncode, 0, fresh.stderr.decode(errors="replace"))
            restored_events = parse_output(restored.stdout)
            fresh_events = parse_output(fresh.stdout)
            restored_data = [
                event[2] for event in restored_events if event[0] == "DATA"
            ]
            fresh_data = [event[2] for event in fresh_events if event[0] == "DATA"]
            self.assertEqual(restored_data, fresh_data)
            stderr = restored.stderr.decode(errors="replace")
            self.assertIn("[SESSION] restored slot=0", stderr)
            self.assertIn("[SESSION] exact-extension slot=0", stderr)

    def test_resident_multi_token_suffix_matches_fresh_prefill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory)
            first = run_engine(
                b"SUBMIT 81 0 1 8 0 1\n!\nCANCEL 81\n",
                resident=True,
                session_dir=session_dir,
            )
            self.assertEqual(first.returncode, 0, first.stderr.decode(errors="replace"))
            # NUL/SOH are byte-fallback tokens below the tiny model's
            # deliberately truncated 512-entry vocabulary and cannot merge
            # with the cached leading exclamation token.
            extended = b"!\x00\x01"
            wire = (
                f"SUBMIT 82 0 {len(extended)} 4 0 1\n".encode()
                + extended
                + b"\n"
            )
            restored = run_engine(
                wire,
                resident=True,
                session_dir=session_dir,
                suffix_block=True,
            )
            fresh = run_engine(wire, resident=True)
            self.assertEqual(
                restored.returncode, 0, restored.stderr.decode(errors="replace")
            )
            self.assertEqual(fresh.returncode, 0, fresh.stderr.decode(errors="replace"))
            restored_data = [
                event[2] for event in parse_output(restored.stdout) if event[0] == "DATA"
            ]
            fresh_data = [
                event[2] for event in parse_output(fresh.stdout) if event[0] == "DATA"
            ]
            self.assertEqual(restored_data, fresh_data)
            self.assertIn(
                "[SESSION] exact-extension slot=0",
                restored.stderr.decode(errors="replace"),
            )


if __name__ == "__main__":
    unittest.main()
