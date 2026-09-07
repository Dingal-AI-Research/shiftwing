#!/usr/bin/env python3
"""Fetch the pinned DeepSeek-V4 release with resumable, hash-bound state."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import sys
import urllib.error
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from deepseek_v4_spec import SOURCE_REPO, SOURCE_REVISION, WEIGHT_SHARDS


STATE_SCHEMA = "colib.deepseek-v4.fetch-state.v1"
STATE_FILE = "fetch-state.json"
SOURCE_MARKER = "colib-source.json"
API_FILE = "upstream-api.json"
COPY_CHUNK = 8 * 1024 * 1024

REQUIRED_METADATA = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

REFERENCE_CANDIDATES = (
    "encoding/README.md",
    "encoding/encoding_dsv4.py",
    "encoding/test_encoding_dsv4.py",
    "inference/config.json",
    "inference/convert.py",
    "inference/kernel.py",
    "inference/model.py",
    "README.md",
    "LICENSE",
    "special_tokens_map.json",
    "added_tokens.json",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(COPY_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe upstream filename: {name!r}")
    return path.as_posix()


def api_url(repo: str = SOURCE_REPO, revision: str = SOURCE_REVISION) -> str:
    return f"https://huggingface.co/api/models/{repo}/revision/{revision}"


def resolve_url(name: str, repo: str = SOURCE_REPO, revision: str = SOURCE_REVISION) -> str:
    quoted = urllib.parse.quote(safe_relative(name), safe="/")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{quoted}"


def _request(url: str, headers: dict[str, str] | None = None):
    combined = {"User-Agent": "colib-deepseek-v4-fetch/1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        combined["Authorization"] = f"Bearer {token}"
    if headers:
        combined.update(headers)
    return urllib.request.urlopen(urllib.request.Request(url, headers=combined), timeout=120)


def fetch_api_inventory(url: str | None = None) -> dict[str, Any]:
    with _request(url or api_url()) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("Hugging Face model API response is not a mapping")
    if value.get("sha") != SOURCE_REVISION:
        raise ValueError(
            f"Hugging Face resolved {value.get('sha')!r}, expected {SOURCE_REVISION}"
        )
    siblings = value.get("siblings")
    if not isinstance(siblings, list):
        raise ValueError("Hugging Face model API response has no sibling inventory")
    names = []
    for sibling in siblings:
        if isinstance(sibling, dict) and isinstance(sibling.get("rfilename"), str):
            names.append(safe_relative(sibling["rfilename"]))
    value["colib_files"] = sorted(set(names))
    return value


def metadata_files(inventory: Iterable[str]) -> list[str]:
    available = set(inventory)
    missing = sorted(set(REQUIRED_METADATA) - available)
    if missing:
        raise ValueError(f"pinned release is missing required metadata: {missing}")
    return list(REQUIRED_METADATA) + [
        name for name in REFERENCE_CANDIDATES if name in available
    ]


def shard_files(index: dict[str, Any]) -> list[str]:
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    shards = sorted({safe_relative(value) for value in weight_map.values()})
    if len(shards) != WEIGHT_SHARDS:
        raise ValueError(f"expected {WEIGHT_SHARDS} weight shards, found {len(shards)}")
    return shards


def download_one(url: str, output: Path) -> dict[str, Any]:
    """Download one file, resuming a durable .partial with HTTP Range."""
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    existing = partial.stat().st_size if partial.is_file() else 0
    request_headers = {"Range": f"bytes={existing}-"} if existing else {}
    response = _request(url, request_headers)
    status = getattr(response, "status", None) or response.getcode()
    if existing and status != 206:
        response.close()
        partial.unlink()
        existing = 0
        response = _request(url)
        status = getattr(response, "status", None) or response.getcode()
    digest = hashlib.sha256()
    if existing:
        with partial.open("rb") as handle:
            while block := handle.read(COPY_CHUNK):
                digest.update(block)
    mode = "ab" if existing else "wb"
    try:
        with partial.open(mode) as handle:
            while block := response.read(COPY_CHUNK):
                handle.write(block)
                digest.update(block)
            handle.flush()
            os.fsync(handle.fileno())
        headers = response.headers
        etag = headers.get("ETag") or headers.get("X-Linked-Etag")
        resolved_url = response.geturl()
    finally:
        response.close()
    os.replace(partial, output)
    return {
        "size": output.stat().st_size,
        "sha256": digest.hexdigest(),
        "etag": etag,
        "url": url,
        "resolved_url": resolved_url,
        "completed_at": utc_now(),
    }


def _plan_sha(files: Iterable[str]) -> str:
    payload = json.dumps(
        {"repository": SOURCE_REPO, "revision": SOURCE_REVISION, "files": list(files)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _state(output: Path, plan_sha: str, files: list[str], command: str) -> dict[str, Any]:
    path = output / STATE_FILE
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        compatible_plan = value.get("plan_sha256") == plan_sha or value.get(
            "metadata_plan_sha256"
        ) == plan_sha
        old_metadata = {
            name for name in value.get("files", [])
            if not (name.startswith("model-") and name.endswith(".safetensors"))
        }
        append_only_metadata = (
            value.get("schema") == STATE_SCHEMA
            and value.get("repository") == SOURCE_REPO
            and value.get("revision") == SOURCE_REVISION
            and old_metadata.issubset(files)
        )
        if value.get("schema") != STATE_SCHEMA or not (compatible_plan or append_only_metadata):
            raise ValueError("fetch state belongs to a different pinned file plan")
        if append_only_metadata and not compatible_plan:
            value["metadata_plan_sha256"] = plan_sha
        return value
    value = {
        "schema": STATE_SCHEMA,
        "repository": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "plan_sha256": plan_sha,
        "metadata_plan_sha256": plan_sha,
        "files": files,
        "completed": {},
        "command": command,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
    }
    atomic_json(path, value)
    return value


def _verify_completed(name: str, path: Path, evidence: dict[str, Any]) -> str:
    if not path.is_file() or path.stat().st_size != evidence.get("size"):
        raise ValueError(f"completed download size changed: {name}")
    if sha256_file(path) != evidence.get("sha256"):
        raise ValueError(f"completed download hash changed: {name}")
    return name


def _fetch_weights(output: Path, weights: list[str], state: dict[str, Any],
                   workers: int) -> None:
    state_path = output / STATE_FILE
    completed = [name for name in weights if name in state["completed"]]
    pending = [name for name in weights if name not in state["completed"]]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        verification = {
            executor.submit(
                _verify_completed, name, output / name, state["completed"][name]
            ): name
            for name in completed
        }
        for future in as_completed(verification):
            future.result()
        downloads = {
            executor.submit(download_one, resolve_url(name), output / name): name
            for name in pending
        }
        try:
            for future in as_completed(downloads):
                name = downloads[future]
                state["completed"][name] = future.result()
                state["updated_at"] = utc_now()
                atomic_json(state_path, state)
                print(
                    f"[fetched {len(state['completed'])}/{len(state['files'])}] {name}",
                    flush=True,
                )
        except BaseException:
            for future in downloads:
                future.cancel()
            raise


def _fetch_attempt(output: Path, metadata_only: bool = False,
                   api: dict[str, Any] | None = None,
                   workers: int = 1) -> dict[str, Any]:
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = api or fetch_api_inventory()
    if inventory.get("sha") != SOURCE_REVISION:
        raise ValueError("inventory is not bound to the pinned revision")
    atomic_json(output / API_FILE, inventory)
    files = metadata_files(inventory["colib_files"])
    # Fetch metadata first. It contains the authoritative shard inventory.
    plan = _plan_sha(files)
    state = _state(output, plan, files, shlex.join(sys.argv))
    attempts = state.setdefault("attempts", [])
    recovered_at = utc_now()
    for previous in attempts:
        if previous.get("status") == "running":
            previous["status"] = "interrupted"
            previous["completed_after"] = len(state.get("completed", {}))
            previous["failed_at"] = recovered_at
            previous["reason"] = "stale running attempt recovered by next invocation"

    attempt = {
        "attempt": len(state.setdefault("attempts", [])) + 1,
        "command": shlex.join(sys.argv),
        "pid": os.getpid(),
        "workers": workers,
        "metadata_only": metadata_only,
        "started_at": utc_now(),
        "completed_before": len(state.get("completed", {})),
        "status": "running",
    }
    state["attempts"].append(attempt)
    state["status"] = "running"
    state.pop("completed_at", None)
    state["updated_at"] = utc_now()
    atomic_json(output / STATE_FILE, state)
    for name in files:
        completed = state["completed"].get(name)
        path = output.joinpath(*PurePosixPath(name).parts)
        if completed:
            if path.stat().st_size != completed["size"] or sha256_file(path) != completed["sha256"]:
                raise ValueError(f"completed download changed: {name}")
            continue
        state["completed"][name] = download_one(resolve_url(name), path)
        state["updated_at"] = utc_now()
        atomic_json(output / STATE_FILE, state)

    weights = shard_files(json.loads((output / "model.safetensors.index.json").read_text()))
    if not set(weights).issubset(set(inventory["colib_files"])):
        raise ValueError("model index names a shard absent from the pinned API inventory")
    if not metadata_only:
        full_files = files + weights
        full_plan = _plan_sha(full_files)
        if state["plan_sha256"] != full_plan:
            # Metadata state becomes the initial durable boundary of the full plan.
            state["plan_sha256"] = full_plan
            state["files"] = full_files
            state["updated_at"] = utc_now()
            atomic_json(output / STATE_FILE, state)
        _fetch_weights(output, weights, state, workers)
    state["status"] = "metadata_complete" if metadata_only else "complete"
    state["updated_at"] = utc_now()
    state["completed_at"] = utc_now()
    state["weight_shards"] = weights
    attempt["status"] = state["status"]
    attempt["completed_after"] = len(state["completed"])
    attempt["completed_at"] = state["completed_at"]
    atomic_json(output / STATE_FILE, state)
    atomic_json(
        output / SOURCE_MARKER,
        {
            "repository": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "api_sha": inventory["sha"],
            "fetch_state_sha256": sha256_file(output / STATE_FILE),
            "status": state["status"],
        },
    )
    return state


def _record_attempt_failure(output: Path, error: BaseException) -> None:
    state_path = output.resolve() / STATE_FILE
    try:
        state = json.loads(state_path.read_text())
        attempt = next(
            (
                item
                for item in reversed(state.get("attempts", []))
                if item.get("status") == "running"
            ),
            None,
        )
        if attempt is None:
            return
        failed_at = utc_now()
        status = (
            "interrupted"
            if isinstance(error, (KeyboardInterrupt, SystemExit))
            else "failed"
        )
        attempt["status"] = status
        attempt["completed_after"] = len(state.get("completed", {}))
        attempt["failed_at"] = failed_at
        attempt["error"] = {"type": type(error).__name__}
        if isinstance(error, OSError) and error.errno is not None:
            attempt["error"]["errno"] = error.errno
        state["status"] = status
        state["updated_at"] = failed_at
        state.pop("completed_at", None)
        atomic_json(state_path, state)
    except (OSError, ValueError):
        # A hard filesystem failure may prevent the terminal write. The next
        # invocation closes any surviving "running" attempt before it starts.
        return


def fetch(output: Path, metadata_only: bool = False,
          api: dict[str, Any] | None = None, workers: int = 1) -> dict[str, Any]:
    try:
        return _fetch_attempt(
            output, metadata_only=metadata_only, api=api, workers=workers
        )
    except BaseException as error:
        _record_attempt_failure(output, error)
        raise



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel shard verification/download workers (1-8)")
    args = parser.parse_args()
    try:
        state = fetch(
            args.output, metadata_only=args.metadata_only, workers=args.workers
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": state["status"],
                "revision": state["revision"],
                "completed_files": len(state["completed"]),
                "weight_shards": len(state.get("weight_shards", [])),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
