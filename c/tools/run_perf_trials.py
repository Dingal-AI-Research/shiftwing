#!/usr/bin/env python3
"""Run independent, hash-bound benchmark trials with atomic resume state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "colib.performance-trials.v1"
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*argv: str) -> str:
        result = subprocess.run(
            argv,
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()

    try:
        commit = run("git", "rev-parse", "HEAD")
        branch = run("git", "branch", "--show-current")
        dirty = bool(run("git", "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}
    return {"commit": commit, "branch": branch, "dirty": dirty}


def environment_snapshot() -> dict[str, Any]:
    prefixes = (
        "COLI_",
        "CUDA_",
        "DECODE_",
        "DIRECT",
        "EXPERT_",
        "OMP_",
        "PREFETCH_",
        "Q3_",
        "RAM_",
        "URING",
    )
    selected = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(prefixes)
        and not any(token in name for token in ("KEY", "TOKEN", "SECRET"))
    }
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "selected": selected,
    }


def binding_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"binding is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def command_for_trial(template: list[str], output: Path) -> list[str]:
    replaced = [value.replace("{output}", str(output)) for value in template]
    if sum(value.count("{output}") for value in template) != 1:
        raise ValueError("command must contain exactly one {output} placeholder")
    return replaced


def dotted_value(value: Any, key: str) -> Any:
    current = value
    for component in key.split("."):
        if not isinstance(current, dict) or component not in current:
            raise ValueError(f"result is missing {key}")
        current = current[component]
    return current


def signature(
    *,
    label: str,
    trials: int,
    command: list[str],
    acceptance_key: str,
    bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    record = {
        "label": label,
        "trials": trials,
        "command_template": command,
        "acceptance_key": acceptance_key,
        "bindings": bindings,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


def load_or_create_state(
    path: Path,
    expected: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("schema") != SCHEMA:
            raise ValueError("trial state schema does not match")
        if state.get("signature") != expected:
            raise ValueError("trial state signature does not match current inputs")
        changed = False
        for attempt in state.get("attempts", []):
            if attempt.get("status") == "running":
                attempt["status"] = "interrupted"
                attempt["failed_at"] = utc_now()
                attempt["reason"] = "stale running attempt recovered"
                changed = True
        if changed:
            state["updated_at"] = utc_now()
            atomic_json(path, state)
        return state
    state = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "repository": git_metadata(root),
        "environment": environment_snapshot(),
        "signature": expected,
        "attempts": [],
        "completed": {},
        "status": "running",
    }
    atomic_json(path, state)
    return state


def verified_completed(
    state: dict[str, Any], trial: int, output: Path
) -> bool:
    record = state.get("completed", {}).get(str(trial))
    if not isinstance(record, dict) or not output.is_file():
        return False
    return (
        output.stat().st_size == record.get("bytes")
        and sha256_file(output) == record.get("sha256")
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--bind", type=Path, action="append", default=[])
    parser.add_argument(
        "--recover-only",
        action="store_true",
        help="mark stale attempts interrupted and exit without starting work",
    )
    parser.add_argument("--acceptance-key", default="acceptance.passed")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if args.trials < 1:
        parser.error("--trials must be positive")
    if not LABEL_RE.fullmatch(args.label):
        parser.error("--label contains unsafe filename characters")
    if not args.command:
        parser.error("a command template is required after --")
    try:
        command_for_trial(args.command, Path("probe"))
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve(strict=True)
    state_path = args.state.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bindings = [binding_record(path) for path in args.bind]
    expected = signature(
        label=args.label,
        trials=args.trials,
        command=args.command,
        acceptance_key=args.acceptance_key,
        bindings=bindings,
    )
    try:
        state = load_or_create_state(state_path, expected, root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"trial state error: {error}", file=sys.stderr)
        return 2
    if args.recover_only:
        state["status"] = (
            "complete"
            if len(state.get("completed", {})) == args.trials
            else "interrupted"
        )
        state["updated_at"] = utc_now()
        atomic_json(state_path, state)
        print(
            json.dumps(
                {"status": state["status"], "state": str(state_path)},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    for trial in range(1, args.trials + 1):
        output = output_dir / f"{args.label}-trial-{trial:02d}.json"
        if verified_completed(state, trial, output):
            print(f"[resume {trial}/{args.trials}] verified {output}", flush=True)
            continue
        partial = output_dir / f".{args.label}-trial-{trial:02d}.partial.json"
        stdout_path = output_dir / f"{args.label}-trial-{trial:02d}.stdout.log"
        stderr_path = output_dir / f"{args.label}-trial-{trial:02d}.stderr.log"
        partial.unlink(missing_ok=True)
        command = command_for_trial(args.command, partial)
        attempt = {
            "attempt": len(state["attempts"]) + 1,
            "trial": trial,
            "status": "running",
            "started_at": utc_now(),
            "command": command,
            "command_shell": shlex.join(command),
            "partial": str(partial),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        state["attempts"].append(attempt)
        state["updated_at"] = utc_now()
        atomic_json(state_path, state)
        print(
            f"[trial {trial}/{args.trials}] {attempt['command_shell']}",
            flush=True,
        )
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
        except KeyboardInterrupt:
            attempt["status"] = "interrupted"
            attempt["interrupted_at"] = utc_now()
            attempt["reason"] = "benchmark controller received an interrupt"
            state["status"] = "interrupted"
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
            return 130
        attempt["exit_code"] = completed.returncode
        if completed.returncode:
            attempt["status"] = "failed"
            attempt["failed_at"] = utc_now()
            attempt["reason"] = "benchmark command returned nonzero"
            state["status"] = "failed"
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
            return completed.returncode
        try:
            result = json.loads(partial.read_text(encoding="utf-8"))
            if dotted_value(result, args.acceptance_key) is not True:
                raise ValueError(
                    f"{args.acceptance_key} is not true in trial result"
                )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            attempt["status"] = "failed"
            attempt["failed_at"] = utc_now()
            attempt["reason"] = str(error)
            state["status"] = "failed"
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
            return 2
        os.replace(partial, output)
        directory = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        record = {
            "trial": trial,
            "output": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "completed_at": utc_now(),
            "summary": result.get("summary"),
            "hardware": result.get("hardware"),
            "configuration": result.get("configuration"),
            "model_manifest": result.get("model_manifest"),
            "expert_lowbit_manifest": result.get("expert_lowbit_manifest"),
        }
        state["completed"][str(trial)] = record
        attempt["status"] = "complete"
        attempt["completed_at"] = record["completed_at"]
        attempt["output_sha256"] = record["sha256"]
        state["status"] = (
            "complete" if len(state["completed"]) == args.trials else "running"
        )
        state["updated_at"] = utc_now()
        atomic_json(state_path, state)
        print(
            f"[trial {trial}/{args.trials}] complete {record['sha256']}",
            flush=True,
        )

    state["status"] = "complete"
    state["completed_at"] = utc_now()
    state["updated_at"] = state["completed_at"]
    atomic_json(state_path, state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "trials": len(state["completed"]),
                "state": str(state_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
