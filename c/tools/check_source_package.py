#!/usr/bin/env python3
"""Reject generated binaries, model weights, and build output in Git."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path, PurePosixPath
from typing import Iterable


FORBIDDEN_SUFFIXES = {
    ".a",
    ".dll",
    ".exp",
    ".gguf",
    ".lib",
    ".o",
    ".safetensors",
    ".so",
}
FORBIDDEN_PREFIXES = (
    "web/dist/",
    "c/qwen35/",
    "c/qwen397/",
    "c/ornith35/",
    "c/ornith397/",
)
FORBIDDEN_EXACT = {"c/qwen", "c/qwen.exe"}


def _native_executable(path: Path) -> bool:
    with path.open("rb") as handle:
        header = handle.read(64)
        if header[:4] == b"\x7fELF":
            return True
        if len(header) < 64 or header[:2] != b"MZ":
            return False
        pe_offset = int.from_bytes(header[60:64], "little")
        if pe_offset < 64 or pe_offset > path.stat().st_size - 4:
            return False
        handle.seek(pe_offset)
        return handle.read(4) == b"PE\0\0"


def tracked_paths(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.decode("utf-8", errors="replace").strip()
            or "git ls-files failed"
        )
    return [
        item.decode("utf-8", errors="strict")
        for item in completed.stdout.split(b"\0")
        if item
    ]


def check_source_package(root: Path, tracked: Iterable[str]) -> list[str]:
    root = root.resolve()
    failures: list[str] = []
    for raw in tracked:
        path = PurePosixPath(raw)
        normalized = path.as_posix()
        suffix = path.suffix.lower()
        if normalized in FORBIDDEN_EXACT:
            failures.append(f"tracked engine binary: {normalized}")
        if normalized.startswith(FORBIDDEN_PREFIXES):
            failures.append(f"tracked generated/model path: {normalized}")
        if suffix in FORBIDDEN_SUFFIXES:
            failures.append(f"tracked generated/model suffix: {normalized}")
        local = root.joinpath(*path.parts)
        if local.is_file() and _native_executable(local):
            failures.append(f"tracked native executable: {normalized}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    root = args.root.resolve()
    tracked = tracked_paths(root)
    failures = check_source_package(root, tracked)
    print(f"source package: tracked={len(tracked)} failures={len(failures)}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
