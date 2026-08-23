#!/usr/bin/env python3
"""Wait for Ornith397 conversion, then run the preregistered Gate 8 sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SOURCE = (
    "hf://deepreinforce-ai/Ornith-1.0-397B-FP8@"
    "8b61f97a8512d9d01bff1a9625c9a16730e115bb"
)
EXPECTED_FINGERPRINT = (
    "4b1c5ab0824e25b3768e3c6df8392394194fce86c4e9f4f3da24e3134ccf4c94"
)
EXPECTED_MANIFEST = {
    "complete": True,
    "source": EXPECTED_SOURCE,
    "source_fingerprint": EXPECTED_FINGERPRINT,
    "source_shards": 122,
    "output_shards": 122,
    "logical_tensor_count": 93078,
    "tensor_count": 278152,
    "data_bytes": 212634789241,
    "xbits": "int4g128",
    "io_bits": 8,
    "shared_bits": 8,
    "group_size": 128,
    "include_mtp": False,
}


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    artifact: Path
    acceptance_path: tuple[str, ...]
    capture_stdout: bool = False


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root is not an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for name in path:
        if not isinstance(current, dict):
            return None
        current = current.get(name)
    return current


def validate_manifest(model: Path) -> dict[str, Any]:
    path = model / "quantization.json"
    manifest = _read_object(path)
    mismatches = {
        name: {"expected": expected, "observed": manifest.get(name)}
        for name, expected in EXPECTED_MANIFEST.items()
        if manifest.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"{path}: pinned manifest mismatch: {mismatches}")
    return manifest


def process_matches(pid: int, needle: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return needle.encode() in command


def wait_for_conversion(
    model: Path,
    *,
    converter_pid: int,
    poll_seconds: float,
) -> dict[str, Any]:
    manifest_path = model / "quantization.json"
    while True:
        active = process_matches(converter_pid, "convert_qwen.py")
        if manifest_path.is_file():
            manifest = validate_manifest(model)
            if not active:
                return manifest
        elif not active:
            raise RuntimeError(
                f"converter PID {converter_pid} exited without {manifest_path}"
            )
        print(
            f"[wait] converter PID {converter_pid} active; "
            f"complete_manifest={manifest_path.is_file()}",
            flush=True,
        )
        time.sleep(poll_seconds)


def build_cuda_engine(root: Path) -> str:
    argv = (
        "make",
        "-C",
        str(root / "c"),
        "CUDA=1",
        "CUDA_ARCH=native",
        "qwen",
    )
    print(f"[build] {' '.join(argv)}", flush=True)
    completed = subprocess.run(argv, cwd=root, check=False)
    binary = root / "c" / "qwen"
    if completed.returncode != 0:
        raise RuntimeError(
            f"CUDA engine build failed with exit code {completed.returncode}"
        )
    if not binary.is_file():
        raise RuntimeError(f"CUDA engine build did not produce {binary}")
    return _sha256(binary)


def gate8_steps(root: Path) -> tuple[Step, ...]:
    python = root / ".venv" / "bin" / "python"
    model = root / "c" / "ornith397"
    return (
        Step(
            "doctor",
            (
                str(root / "c" / "shiftwing"),
                "doctor",
                "--model",
                str(model),
                "--kv-slots",
                "1",
                "--context",
                "4096",
                "--cuda-expert-gb",
                "6",
                "--ram-cache-gb",
                "18",
                "--runtime-headroom-gb",
                "1",
                "--verify-hashes",
                "--expect-source",
                EXPECTED_SOURCE,
                "--expect-source-fingerprint",
                EXPECTED_FINGERPRINT,
                "--expect-source-shards",
                "122",
                "--expect-output-shards",
                "122",
                "--expect-logical-tensors",
                "93078",
                "--expect-physical-tensors",
                "278152",
                "--expect-data-bytes",
                "212634789241",
                "--json",
            ),
            root / "c" / "ornith397_doctor.json",
            ("status",),
            capture_stdout=True,
        ),
        Step(
            "ppl",
            (
                str(python),
                str(root / "c" / "tools" / "eval_qwen.py"),
                "--snapshot",
                str(model),
                "--corpus",
                str(root / "c" / "bench" / "qwen35_eval.txt"),
                "--max-tokens",
                "1024",
                "--ctx-size",
                "512",
                "--ram-gb",
                "18",
                "--ram-headroom-gb",
                "1",
                "--max-ppl",
                "50",
                "--require-complete-manifest",
                "--output",
                str(root / "c" / "ornith397_ppl_smoke.json"),
            ),
            root / "c" / "ornith397_ppl_smoke.json",
            ("acceptance", "passed"),
        ),
        Step(
            "tier",
            (
                str(python),
                str(root / "c" / "tools" / "qualify_tiered_model.py"),
                "--model",
                str(model),
                "--max-tokens",
                "64",
                "--warmup-passes",
                "2",
                "--measured-passes",
                "1",
                "--expert-ram-gb",
                "18",
                "--cuda-expert-gb",
                "6",
                "--ram-headroom-gb",
                "1",
                "--cuda-headroom-gb",
                "1",
                "--minimum-tps",
                "0.85",
                "--uring-persist",
                "1",
                "--pinned-upload",
                "1",
                "--decode-protect",
                "1",
                "--decode-protect-prewarm",
                "1",
                "--output",
                str(root / "c" / "ornith397_qualification.json"),
            ),
            root / "c" / "ornith397_qualification.json",
            ("acceptance", "passed"),
        ),
        Step(
            "tools",
            (
                str(python),
                str(root / "c" / "tools" / "qualify_ornith_tools.py"),
                "--model",
                str(model),
                "--ram-gb",
                "18",
                "--ram-headroom-gb",
                "1",
                "--cuda-expert-gb",
                "6",
                "--cuda-headroom-gb",
                "1",
                "--startup-timeout",
                "1800",
                "--request-timeout",
                "1800",
                "--output",
                str(root / "c" / "ornith397_tool_qualification.json"),
            ),
            root / "c" / "ornith397_tool_qualification.json",
            ("passed",),
        ),
    )


def _step_is_complete(step: Step, record: Any) -> bool:
    if not isinstance(record, dict) or record.get("status") != "passed":
        return False
    if record.get("argv") != list(step.argv) or not step.artifact.is_file():
        return False
    return record.get("artifact_sha256") == _sha256(step.artifact)


def run_pipeline(
    root: Path,
    state_path: Path,
    *,
    manifest_sha256: str,
    engine_sha256: str,
) -> int:
    if state_path.is_file():
        state = _read_object(state_path)
        bound_manifest = state.get("manifest_sha256")
        if bound_manifest != manifest_sha256:
            raise ValueError(
                f"{state_path}: qualification state is bound to manifest "
                f"{bound_manifest!r}, not {manifest_sha256!r}"
            )
        if state.get("engine_sha256") != engine_sha256:
            raise ValueError(
                f"{state_path}: qualification state is bound to engine "
                f"{state.get('engine_sha256')!r}, not {engine_sha256!r}"
            )
    else:
        state = {
            "schema_version": 1,
            "status": "running",
            "manifest_sha256": manifest_sha256,
            "engine_sha256": engine_sha256,
            "expected_manifest": EXPECTED_MANIFEST,
            "steps": {},
        }
    records = state.setdefault("steps", {})
    for step in gate8_steps(root):
        if _step_is_complete(step, records.get(step.name)):
            print(f"[resume] {step.name}: verified", flush=True)
            continue
        started = time.time()
        print(f"[run] {step.name}: {' '.join(step.argv)}", flush=True)
        if step.capture_stdout:
            completed = subprocess.run(
                step.argv,
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            print(completed.stdout, end="", flush=True)
            if completed.returncode == 0:
                _atomic_text(step.artifact, completed.stdout)
        else:
            completed = subprocess.run(step.argv, cwd=root, check=False)
        record = {
            "argv": list(step.argv),
            "artifact": str(step.artifact),
            "started_unix": started,
            "finished_unix": time.time(),
            "returncode": completed.returncode,
            "status": "failed",
        }
        try:
            artifact = _read_object(step.artifact)
            accepted = _nested(artifact, step.acceptance_path)
            if step.name == "doctor":
                accepted = accepted == "ok"
            else:
                accepted = accepted is True
            if completed.returncode == 0 and accepted:
                record["artifact_sha256"] = _sha256(step.artifact)
                record["status"] = "passed"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            record["artifact_error"] = str(error)
        records[step.name] = record
        state["status"] = (
            "running" if record["status"] == "passed" else "failed"
        )
        _atomic_json(state_path, state)
        if record["status"] != "passed":
            print(f"[fail] {step.name}", flush=True)
            return 1
    state["status"] = "passed"
    state["finished_unix"] = time.time()
    _atomic_json(state_path, state)
    print("[pass] Ornith397 automatic Gate 8 sequence complete", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--converter-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "c" / "bench" / "ornith397_gate8_pipeline.json",
    )
    args = parser.parse_args()
    if args.converter_pid <= 0 or args.poll_seconds <= 0:
        parser.error("converter PID and poll interval must be positive")
    root = args.root.resolve()
    model = root / "c" / "ornith397"
    wait_for_conversion(
        model,
        converter_pid=args.converter_pid,
        poll_seconds=args.poll_seconds,
    )
    manifest_sha256 = _sha256(model / "quantization.json")
    engine_sha256 = build_cuda_engine(root)
    return run_pipeline(
        root,
        args.state.resolve(),
        manifest_sha256=manifest_sha256,
        engine_sha256=engine_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
