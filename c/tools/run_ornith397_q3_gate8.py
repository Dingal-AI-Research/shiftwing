#!/usr/bin/env python3
"""Run the owner-authorized, manifest-bound Ornith397 q3 Gate-8 sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from runtime_env import engine_source_sha256  # noqa: E402
EXPECTED_BASE = {
    "complete": True,
    "source": (
        "hf://deepreinforce-ai/Ornith-1.0-397B-FP8@"
        "8b61f97a8512d9d01bff1a9625c9a16730e115bb"
    ),
    "source_fingerprint": (
        "4b1c5ab0824e25b3768e3c6df8392394194fce86c4e9f4f3da24e3134ccf4c94"
    ),
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
EXPECTED_Q3 = {
    "format": "colib-routed-expert-int3-sidecar-v1",
    "complete": True,
    "config_sha256": (
        "c31964d2d920c10228a40d77afe02b19122b66f7bf3f6eb4b0809ba42527b0d6"
    ),
    "signature": {
        "source_identity": (
            "93f6769bc6d8e2f7d669a8571be1471905a8e6eb487b596dd6836ee08fd2a813"
        ),
        "group_size": 128,
        "iterations": 3,
        "experts_per_file": 64,
    },
    "layers": 60,
    "file_count": 480,
    "tensor_count": 276480,
}
INT4_PPL = 1.050067545
Q3_PPL_WAIVED = True
Q3_MINIMUM_TPS = 0.70
Q3_POLICY_BASIS = (
    "owner accepted Ornith397 routed-expert q3 without a perplexity gate and "
    "selected the measured 0.718432662 tok/s expanded-q4 profile"
)
PPL_CORPUS_SHA256 = (
    "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e"
)


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    artifact: Path
    acceptance_path: tuple[str, ...]
    capture_stdout: bool = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root is not an object")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for name in path:
        if not isinstance(current, dict):
            return None
        current = current.get(name)
    return current


def exact_subset(
    path: Path, observed: dict[str, Any], expected: dict[str, Any]
) -> None:
    mismatches = {
        name: {"expected": value, "observed": observed.get(name)}
        for name, value in expected.items()
        if observed.get(name) != value
    }
    if mismatches:
        raise ValueError(f"{path}: pinned manifest mismatch: {mismatches}")


def validate_manifests(model: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base_path = model / "quantization.json"
    base = read_object(base_path)
    exact_subset(base_path, base, EXPECTED_BASE)
    q3_path = model / "expert-q3.json"
    q3 = read_object(q3_path)
    exact_subset(q3_path, q3, EXPECTED_Q3)
    data_bytes = q3.get("data_bytes")
    files = q3.get("files")
    if not isinstance(data_bytes, int) or data_bytes <= 0:
        raise ValueError(f"{q3_path}: data_bytes must be positive")
    if not isinstance(files, list) or len(files) != EXPECTED_Q3["file_count"]:
        raise ValueError(f"{q3_path}: files does not contain 480 records")
    return base, q3


def process_matches(pid: int, needle: str) -> bool:
    if pid <= 0:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return needle.encode() in command


def wait_for_q3(model: Path, *, converter_pid: int, poll_seconds: float) -> None:
    q3_path = model / "expert-q3.json"
    while True:
        active = process_matches(converter_pid, "requantize_expert_q2.py")
        complete = False
        if q3_path.is_file():
            try:
                complete = read_object(q3_path).get("complete") is True
            except (OSError, ValueError, json.JSONDecodeError):
                complete = False
        if complete and not active:
            validate_manifests(model)
            return
        if not active:
            raise RuntimeError(
                f"q3 converter PID {converter_pid} is inactive and {q3_path} "
                "is not complete"
            )
        print(
            f"[wait] q3 converter PID {converter_pid} active; complete={complete}",
            flush=True,
        )
        time.sleep(poll_seconds)


def validate_int4_ppl_baseline(root: Path) -> tuple[Path, str]:
    path = root / "c" / "ornith397_ppl_smoke.json"
    value = read_object(path)
    if value.get("ppl") != INT4_PPL:
        raise ValueError(f"{path}: pinned int4 PPL changed")
    if value.get("corpus_sha256") != PPL_CORPUS_SHA256:
        raise ValueError(f"{path}: corpus identity changed")
    if value.get("expert_lowbit_manifest") is not None:
        raise ValueError(f"{path}: baseline unexpectedly selected a low-bit sidecar")
    if nested(value, ("acceptance", "passed")) is not True:
        raise ValueError(f"{path}: baseline was not accepted")
    model_manifest = value.get("model_manifest")
    if not isinstance(model_manifest, dict):
        raise ValueError(f"{path}: model manifest is missing")
    exact_subset(path, model_manifest, {k: v for k, v in EXPECTED_BASE.items() if k != "complete"})
    return path, sha256(path)


def build_cuda_engine(root: Path) -> str:
    argv = ("make", "-C", str(root / "c"), "CUDA=1", "CUDA_ARCH=native", "qwen")
    print(f"[build] {' '.join(argv)}", flush=True)
    completed = subprocess.run(argv, cwd=root, check=False)
    binary = root / "c" / "qwen"
    if completed.returncode != 0 or not binary.is_file():
        raise RuntimeError(f"CUDA engine build failed with exit code {completed.returncode}")
    return sha256(binary)


def gate8_steps(root: Path) -> tuple[Step, ...]:
    python = root / ".venv" / "bin" / "python"
    model = root / "c" / "ornith397"
    prompts = root / "c" / "fixtures" / "qwen35_prompts.json"
    reference = root / "c" / "ornith397_int4_prefix_reference.json"
    common_tier = (
        "--model", str(model),
        "--expert-ram-gb", "18",
        "--cuda-expert-gb", "6",
        "--ram-headroom-gb", "1",
        "--cuda-headroom-gb", "1",
        "--threads", "8",
        "--uring-persist", "1",
        "--pinned-upload", "1",
        "--decode-protect", "1",
        "--decode-protect-prewarm", "1",
        "--expert-q3", "1",
        "--q3-native", "0",
    )
    return (
        Step(
            "q3_doctor",
            (
                str(python),
                str(root / "c" / "tools" / "audit_expert_sidecar.py"),
                "--snapshot", str(model),
                "--bits", "3",
                "--verify-hashes",
                "--hash-workers", "8",
            ),
            root / "c" / "ornith397_q3_doctor.json",
            ("passed",),
            capture_stdout=True,
        ),
        Step(
            "int4_reference",
            (
                str(python),
                str(root / "c" / "tools" / "compare_qwen_prefix.py"),
                "--snapshot", str(model),
                "--prompts", str(prompts),
                "--tokens", "64",
                "--threads", "8",
                "--ram-gb", "18",
                "--ram-headroom-gb", "1",
                "--prefetch-threads", "0",
                "--production-cuda",
                "--cuda-expert-gb", "6",
                "--cuda-headroom-gb", "1",
                "--require-complete-manifest",
                "--c-only",
                "--output", str(reference),
            ),
            reference,
            ("acceptance", "passed"),
        ),
        Step(
            "q3_teacher_forced",
            (
                str(python),
                str(root / "c" / "tools" / "compare_qwen_prefix.py"),
                "--snapshot", str(model),
                "--prompts", str(prompts),
                "--tokens", "64",
                "--threads", "8",
                "--ram-gb", "18",
                "--ram-headroom-gb", "1",
                "--prefetch-threads", "0",
                "--production-cuda",
                "--cuda-expert-gb", "6",
                "--cuda-headroom-gb", "1",
                "--require-complete-manifest",
                "--reference-json", str(reference),
                "--teacher-forced-only",
                "--expert-q3",
                "--min-tf-prompt-agreement", "0.85",
                "--min-tf-aggregate-agreement", "0.90",
                "--output", str(root / "c" / "ornith397_q3_prefix_gate.json"),
            ),
            root / "c" / "ornith397_q3_prefix_gate.json",
            ("acceptance", "passed"),
        ),
        Step(
            "q3_coherence",
            (
                str(python),
                str(root / "c" / "tools" / "qualify_tiered_model.py"),
                *common_tier,
                "--max-tokens", "64",
                "--warmup-passes", "0",
                "--measured-passes", "1",
                "--minimum-tps", "0.000001",
                "--output", str(root / "c" / "ornith397_q3_coherence.json"),
            ),
            root / "c" / "ornith397_q3_coherence.json",
            ("acceptance", "passed"),
        ),
        Step(
            "q3_tools",
            (
                str(python),
                str(root / "c" / "tools" / "qualify_ornith_tools.py"),
                "--model", str(model),
                "--ram-gb", "18",
                "--ram-headroom-gb", "1",
                "--cuda-expert-gb", "6",
                "--cuda-headroom-gb", "1",
                "--startup-timeout", "1800",
                "--request-timeout", "1800",
                "--threads", "8",
                "--expert-q3",
                "--output", str(root / "c" / "ornith397_q3_tool_gate.json"),
            ),
            root / "c" / "ornith397_q3_tool_gate.json",
            ("passed",),
        ),
        Step(
            "q3_tier",
            (
                str(python),
                str(root / "c" / "tools" / "qualify_tiered_model.py"),
                *common_tier,
                "--max-tokens", "64",
                "--warmup-passes", "2",
                "--measured-passes", "1",
                "--minimum-tps", f"{Q3_MINIMUM_TPS:.2f}",
                "--output", str(root / "c" / "ornith397_q3_qualification.json"),
            ),
            root / "c" / "ornith397_q3_qualification.json",
            ("acceptance", "passed"),
        ),
    )


def step_complete(step: Step, record: Any) -> bool:
    return bool(
        isinstance(record, dict)
        and record.get("status") == "passed"
        and record.get("argv") == list(step.argv)
        and step.artifact.is_file()
        and record.get("artifact_sha256") == sha256(step.artifact)
    )


def run_pipeline(
    root: Path,
    state_path: Path,
    *,
    base_manifest_sha256: str,
    q3_manifest_sha256: str,
    baseline_sha256: str,
    engine_sha256: str,
    acknowledge_coherence: bool,
    engine_source_fingerprint: str = "",
) -> int:
    bindings = {
        "base_manifest_sha256": base_manifest_sha256,
        "q3_manifest_sha256": q3_manifest_sha256,
        "int4_ppl_baseline_sha256": baseline_sha256,
        "engine_source_sha256": engine_source_fingerprint,
    }
    if state_path.is_file():
        state = read_object(state_path)
        mismatches = {
            key: {"expected": value, "observed": state.get(key)}
            for key, value in bindings.items()
            if state.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{state_path}: qualification binding mismatch {mismatches}")
    else:
        state = {
            "schema_version": 2,
            "status": "running",
            **bindings,
            "engine_sha256": engine_sha256,
            "expected_base": EXPECTED_BASE,
            "expected_q3": EXPECTED_Q3,
            "int4_ppl": INT4_PPL,
            "q3_ppl_waived": Q3_PPL_WAIVED,
            "q3_minimum_tps": Q3_MINIMUM_TPS,
            "q3_policy_basis": Q3_POLICY_BASIS,
            "steps": {},
        }
    state["schema_version"] = 2
    state["q3_ppl_waived"] = Q3_PPL_WAIVED
    state["q3_minimum_tps"] = Q3_MINIMUM_TPS
    state["q3_policy_basis"] = Q3_POLICY_BASIS
    state.pop("q3_max_relative_ppl_delta", None)
    state.pop("q3_max_ppl", None)
    if acknowledge_coherence:
        state["coherence_review_acknowledged"] = True
    records = state.setdefault("steps", {})
    for step in gate8_steps(root):
        if step.name == "q3_tools" and state.get("coherence_review_acknowledged") is not True:
            state["status"] = "awaiting_coherence_review"
            atomic_json(state_path, state)
            print(
                "[review] q3 coherence artifact passed machine checks; inspect all "
                "four outputs, then rerun with --acknowledge-q3-coherence",
                flush=True,
            )
            return 2
        if step_complete(step, records.get(step.name)):
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
                step.artifact.write_text(completed.stdout, encoding="utf-8")
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
            artifact = read_object(step.artifact)
            accepted = nested(artifact, step.acceptance_path) is True
            if completed.returncode == 0 and accepted:
                record["artifact_sha256"] = sha256(step.artifact)
                record["status"] = "passed"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            record["artifact_error"] = str(error)
        records[step.name] = record
        state["status"] = "running" if record["status"] == "passed" else "failed"
        atomic_json(state_path, state)
        if record["status"] != "passed":
            print(f"[fail] {step.name}", flush=True)
            return 1
    state["status"] = "passed"
    state["finished_unix"] = time.time()
    atomic_json(state_path, state)
    print("[pass] Ornith397 q3 Gate 8 sequence complete", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--converter-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--acknowledge-q3-coherence", action="store_true")
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "c" / "bench" / "ornith397_q3_gate8_pipeline.json",
    )
    args = parser.parse_args()
    if args.converter_pid < 0 or args.poll_seconds <= 0:
        parser.error("converter PID cannot be negative and poll interval must be positive")
    root = args.root.resolve()
    model = root / "c" / "ornith397"
    wait_for_q3(
        model,
        converter_pid=args.converter_pid,
        poll_seconds=args.poll_seconds,
    )
    validate_manifests(model)
    _, baseline_hash = validate_int4_ppl_baseline(root)
    engine_hash = build_cuda_engine(root)
    return run_pipeline(
        root,
        args.state.resolve(),
        base_manifest_sha256=sha256(model / "quantization.json"),
        q3_manifest_sha256=sha256(model / "expert-q3.json"),
        baseline_sha256=baseline_hash,
        engine_sha256=engine_hash,
        engine_source_fingerprint=engine_source_sha256(root),
        acknowledge_coherence=args.acknowledge_q3_coherence,
    )


if __name__ == "__main__":
    raise SystemExit(main())
