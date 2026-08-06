#!/usr/bin/env python3
"""Run the preregistered Ornith35/397 Gate 9 production sequence."""

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

from audit_release import audit_release  # noqa: E402
from expert_lowbit import load_complete_expert_sidecar  # noqa: E402
from runtime_env import engine_source_sha256  # noqa: E402


@dataclass(frozen=True)
class ModelProfile:
    name: str
    source: str
    ram_gb: int
    cuda_expert_gb: int
    cancel_limit_s: float
    expert_q3: bool


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    artifact: Path


PROFILES = (
    ModelProfile(
        "ornith35",
        (
            "hf://deepreinforce-ai/Ornith-1.0-35B-FP8@"
            "1ab57ce0b44950e498a88756f40ad1ed4d0f30ca"
        ),
        8,
        6,
        1.0,
        False,
    ),
    ModelProfile(
        "ornith397",
        (
            "hf://deepreinforce-ai/Ornith-1.0-397B-FP8@"
            "8b61f97a8512d9d01bff1a9625c9a16730e115bb"
        ),
        18,
        5,
        3.0,
        True,
    ),
)
GATE9_CHECK_IDS = {
    f"{profile.name}.{kind}"
    for profile in PROFILES
    for kind in ("batch", "cancel", "web")
}


def _atomic_json(path: Path, value: Any) -> None:
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


def manifest_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for profile in PROFILES:
        model = root / "c" / profile.name
        result[profile.name] = _sha256(model / "quantization.json")
        if profile.expert_q3:
            load_complete_expert_sidecar(model, bits=3)
            result[f"{profile.name}.expert_q3"] = _sha256(
                model / "expert-q3.json"
            )
    return result


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


def validate_gate8(root: Path) -> dict[str, Any]:
    audit = audit_release(root, ornith397_expert_q3=True)
    failures = set(audit["summary"]["failures"])
    non_gate9 = failures - GATE9_CHECK_IDS
    if non_gate9:
        raise ValueError(
            "Gate 9 cannot start; pre-Gate-9 release checks failed: "
            + ", ".join(sorted(non_gate9))
        )
    return audit


def gate9_steps(root: Path) -> tuple[Step, ...]:
    python = str(root / ".venv" / "bin" / "python")
    tools = root / "c" / "tools"
    cdir = root / "c"
    steps: list[Step] = []
    for profile in PROFILES:
        model = cdir / profile.name
        prefix = cdir / profile.name
        lowbit_args = ("--expert-q3",) if profile.expert_q3 else ()
        common_batch = (
            python,
            str(tools / "bench_serve_batch.py"),
            "--model",
            str(model),
            "--prompt",
            "Reply with exactly: colib batch ready",
            "--expect-exact",
            "colib batch ready",
            "--requests",
            "2",
            "--max-tokens",
            "8",
            "--context",
            "4096",
            "--threads",
            "8",
            "--warmup-passes",
            "2",
        )
        resource_args = (
            "--ram-gb",
            str(profile.ram_gb),
            "--ram-headroom-gb",
            "1",
            "--cuda",
            "--cuda-expert-gb",
            str(profile.cuda_expert_gb),
            "--cuda-headroom-gb",
            "1",
        )
        ab = prefix.with_name(f"{profile.name}_batch_ab.json")
        ba = prefix.with_name(f"{profile.name}_batch_ba.json")
        abba = prefix.with_name(f"{profile.name}_batch_abba.json")
        cancel = prefix.with_name(f"{profile.name}_cancel_gate.json")
        web = prefix.with_name(f"{profile.name}_web_gate.json")
        steps.extend(
            (
                Step(
                    f"{profile.name}.batch_ab",
                    common_batch
                    + (
                        "--order",
                        "sequential,concurrent",
                    )
                    + resource_args
                    + lowbit_args
                    + (
                        "--request-timeout",
                        "1800",
                        "--output",
                        str(ab),
                    ),
                    ab,
                ),
                Step(
                    f"{profile.name}.batch_ba",
                    common_batch
                    + (
                        "--order",
                        "concurrent,sequential",
                    )
                    + resource_args
                    + lowbit_args
                    + (
                        "--request-timeout",
                        "1800",
                        "--output",
                        str(ba),
                    ),
                    ba,
                ),
                Step(
                    f"{profile.name}.batch_abba",
                    (
                        python,
                        str(tools / "compare_batch_ab.py"),
                        "--ab",
                        str(ab),
                        "--ba",
                        str(ba),
                        "--expect-family",
                        "ornith-1.0",
                        "--expect-source",
                        profile.source,
                        "--minimum-each",
                        "0.95",
                        "--minimum-geomean",
                        "1.0",
                        "--output",
                        str(abba),
                    ),
                    abba,
                ),
                Step(
                    f"{profile.name}.cancel",
                    (
                        python,
                        str(tools / "bench_cancel_mux.py"),
                        "--model",
                        str(model),
                        "--prompt",
                        "Reply briefly: cancellation probe",
                        "--max-tokens",
                        "8",
                        "--context",
                        "4096",
                        "--threads",
                        "8",
                        "--warmup-passes",
                        "2",
                        "--trials",
                        "20",
                    )
                    + resource_args
                    + lowbit_args
                    + (
                        "--request-timeout",
                        "1800",
                        "--max-cancel-ack-s",
                        str(profile.cancel_limit_s),
                        "--output",
                        str(cancel),
                    ),
                    cancel,
                ),
                Step(
                    f"{profile.name}.web",
                    (
                        python,
                        str(tools / "smoke_web_qwen.py"),
                        "--model",
                        str(model),
                        "--requests",
                        "2",
                        "--kv-slots",
                        "2",
                        "--max-tokens",
                        "16",
                        "--context",
                        "4096",
                        "--threads",
                        "8",
                    )
                    + resource_args
                    + lowbit_args
                    + (
                        "--timeout",
                        "1800",
                        "--expect-exact",
                        "colib ready",
                        "--output",
                        str(web),
                    ),
                    web,
                ),
            )
        )
    return tuple(steps)


def _step_complete(step: Step, record: Any) -> bool:
    return (
        isinstance(record, dict)
        and record.get("status") == "passed"
        and record.get("argv") == list(step.argv)
        and step.artifact.is_file()
        and record.get("artifact_sha256") == _sha256(step.artifact)
    )


def run_pipeline(
    root: Path,
    state_path: Path,
    *,
    bound_manifests: dict[str, str],
    engine_sha256: str,
    engine_source_fingerprint: str = "",
) -> int:
    if state_path.is_file():
        state = _read_object(state_path)
        if state.get("manifest_sha256") != bound_manifests:
            raise ValueError(
                f"{state_path}: qualification state is bound to different manifests"
            )
        if state.get("engine_source_sha256") != engine_source_fingerprint:
            raise ValueError(
                f"{state_path}: qualification state is bound to different engine sources"
            )
    else:
        state = {
            "schema_version": 1,
            "status": "running",
            "manifest_sha256": bound_manifests,
            "engine_sha256": engine_sha256,
            "engine_source_sha256": engine_source_fingerprint,
            "steps": {},
        }
    records = state.setdefault("steps", {})
    for step in gate9_steps(root):
        if _step_complete(step, records.get(step.name)):
            print(f"[resume] {step.name}: verified", flush=True)
            continue
        started = time.time()
        print(f"[run] {step.name}: {' '.join(step.argv)}", flush=True)
        completed = subprocess.run(step.argv, cwd=root, check=False)
        record: dict[str, Any] = {
            "argv": list(step.argv),
            "artifact": str(step.artifact),
            "started_unix": started,
            "finished_unix": time.time(),
            "returncode": completed.returncode,
            "status": "failed",
        }
        try:
            artifact = _read_object(step.artifact)
            if (
                completed.returncode == 0
                and (artifact.get("acceptance") or {}).get("passed") is True
            ):
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
    final_audit = audit_release(root, ornith397_expert_q3=True)
    state["release_audit_summary"] = final_audit["summary"]
    state["status"] = (
        "passed" if final_audit["summary"]["passed"] is True else "failed"
    )
    state["finished_unix"] = time.time()
    _atomic_json(state_path, state)
    if state["status"] != "passed":
        print(
            "[fail] Gate 9 artifacts did not pass the release evidence audit",
            flush=True,
        )
        return 1
    print("[pass] Ornith35/397 Gate 9 sequence complete", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "c" / "bench" / "ornith_gate9_pipeline.json",
    )
    parser.add_argument(
        "--acknowledge-gate8",
        action="store_true",
        help="confirm the completed Ornith397 Gate-8 evidence was reviewed",
    )
    args = parser.parse_args()
    if not args.acknowledge_gate8:
        parser.error("--acknowledge-gate8 is required")
    root = args.root.resolve()
    validate_gate8(root)
    engine_sha256 = build_cuda_engine(root)
    return run_pipeline(
        root,
        args.state.resolve(),
        bound_manifests=manifest_hashes(root),
        engine_sha256=engine_sha256,
        engine_source_fingerprint=engine_source_sha256(root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
