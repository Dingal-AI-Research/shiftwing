#!/usr/bin/env python3
"""Recover pinned native weights with bounded, verified, resumable staging."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import stat
import struct
import threading
import urllib.error
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import convert_deepseek_v4 as converter
import fetch_deepseek_v4 as fetcher
import validate_deepseek_v4_conversion as validator
from deepseek_v4_spec import MIN_FINAL_FREE_BYTES, SOURCE_REVISION

RANGE_BYTES = 8 * 1024**2
WORKERS = 6
NETWORK_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 1
JOURNAL_RESERVE = 1024**2


def merge_spans(spans):
    merged = []
    for lo, hi in sorted(spans):
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def missing_spans(wanted, verified):
    """Subtract verified intervals without discarding a partially covered range."""
    result = []
    intervals = merge_spans(verified)
    for lo, hi in merge_spans(wanted):
        cursor = lo
        for start, end in intervals:
            if start >= hi:
                break
            if end <= cursor:
                continue
            if start > cursor:
                result.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < hi:
            result.append((cursor, hi))
    return result


def allocation_bytes(spans, block=4096):
    return sum(hi - lo for lo, hi in merge_spans(
        ((lo // block * block, converter._align(hi, block)) for lo, hi in spans)
    ))


def identity(value):
    return {"device": value.st_dev, "inode": value.st_ino}


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_all(fd, data, offset):
    view = memoryview(data)
    while view:
        count = os.pwrite(fd, view, offset)
        if count <= 0:
            raise OSError("short staging write")
        offset += count
        view = view[count:]


@contextlib.contextmanager
def recovery_lock(source):
    """One journal writer/converter per source, released by the OS on a crash."""
    source.mkdir(parents=True, exist_ok=True)
    fd = os.open(source / "recovery.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another checkpoint recovery owns this source") from exc
        yield
    finally:
        os.close(fd)


class Recovery:
    def __init__(self, source: Path, target: Path, floor=MIN_FINAL_FREE_BYTES):
        self.source, self.target = source.resolve(), target.resolve()
        self.floor = floor
        self.cache_path = self.source / "recovery-headers.json"
        self.headers = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        if self.headers and self.headers.get("revision") != SOURCE_REVISION:
            raise ValueError("recovery headers belong to another revision")
        self.headers.setdefault("revision", SOURCE_REVISION)
        self.headers.setdefault("shards", {})
        self.owned_path = self.source / "recovery-staging.json"
        self.owned = json.loads(self.owned_path.read_text()) if self.owned_path.exists() else {}
        self.validation_path = self.target / "recovery-validation.json"
        self.validation = json.loads(self.validation_path.read_text()) if self.validation_path.exists() else {}
        self.expected_shards = set(fetcher.shard_files(json.loads((self.source / converter.INDEX_FILE).read_text())))
        for name in self.expected_shards:
            converter._validate_relative_file(name)
        self.output_reserve = 0

    def _path(self, name):
        if name not in self.expected_shards:
            raise ValueError("unexpected source shard")
        converter._validate_relative_file(name)
        path = self.source / name
        if path.is_symlink():
            raise ValueError("source staging must not be a symlink")
        return path

    def header(self, name):
        self._path(name)
        item = self.headers["shards"].get(name)
        if item is None:
            # A bounded header read also permits a server returning the full body.
            with fetcher._request(fetcher.resolve_url(name), {"Range": "bytes=0-7"}) as response:
                raw = response.read(8)
            if len(raw) != 8:
                raise ValueError("truncated remote header")
            size = struct.unpack("<Q", raw)[0]
            if not 0 < size <= 32 * 1024**2:
                raise ValueError("remote header exceeds bound")
            with fetcher._request(fetcher.resolve_url(name), {"Range": f"bytes=0-{size+7}"}) as response:
                payload = response.read(size + 8)
            if len(payload) != size + 8 or payload[:8] != raw:
                raise ValueError("remote header changed or was truncated")
            header = json.loads(payload[8:])
            length = size + 8 + max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
            item = {"start": size + 8, "header": header, "size": length}
            self.headers["shards"][name] = item
            converter.atomic_json(self.cache_path, self.headers)
            print(f"[header {len(self.headers['shards'])}/{len(self.expected_shards)}] {name}", flush=True)
        return item["start"], item["header"]

    def _wanted(self, name, records):
        self.header(name)
        item = self.headers["shards"][name]
        if len({r.group for r in records}) != 1:
            raise ValueError("staging must belong to exactly one conversion group")
        for record in records:
            if (record.source_shard != name or record.nbytes < 0
                    or record.source_offset < item["start"]
                    or record.source_offset + record.nbytes > item["size"]):
                raise ValueError("source record extent outside shard payload")
        return merge_spans((r.source_offset, r.source_offset + r.nbytes) for r in records)

    def _header_payload(self, name):
        item = self.headers["shards"][name]
        raw = json.dumps(item["header"], separators=(",", ":"), ensure_ascii=False).encode()
        length = item["start"] - 8
        if not 0 < length <= 32 * 1024**2 or len(raw) > length:
            raise ValueError("canonical source header exceeds original header")
        return struct.pack("<Q", length) + raw.ljust(length, b" ")

    def _check_owner(self, path, evidence):
        value = path.lstat()
        if (not stat.S_ISREG(value.st_mode) or value.st_nlink != 1
                or evidence.get("identity") != identity(value)):
            raise ValueError("staging ownership identity mismatch; preserving file")

    def _verified_ranges(self, path, evidence):
        item = self.headers["shards"][path.name]
        if (evidence.get("mode") != "ranges" or evidence.get("revision") != SOURCE_REVISION
                or evidence.get("bytes") != item["size"]
                or evidence.get("header_sha256") != json_sha(item)):
            raise ValueError("staging identity does not match pinned header")
        self._check_owner(path, evidence)
        if not evidence.get("initialized"):
            if evidence.get("ranges") or path.stat().st_size not in (0, item["size"]):
                raise ValueError("invalid interrupted staging initialization")
            return []
        if path.stat().st_size != item["size"]:
            raise ValueError("source length mismatch")
        if converter.read_safetensors_header(path) != self.header(path.name):
            raise ValueError("source header mismatch")
        intervals = []
        with path.open("rb") as handle:
            for span in sorted(evidence["ranges"], key=lambda s: s["offset"]):
                lo, size = span["offset"], span["bytes"]
                if (type(lo) is not int or type(size) is not int or size <= 0
                        or size > RANGE_BYTES or lo < item["start"] or lo + size > item["size"]
                        or (intervals and lo < intervals[-1][1])):
                    raise ValueError("invalid or overlapping staging range evidence")
                handle.seek(lo)
                data = handle.read(size)
                if len(data) != size or hashlib.sha256(data).hexdigest() != span["sha256"]:
                    raise ValueError("source range integrity mismatch; preserving file")
                intervals.append((lo, lo + size))
        return intervals

    def verify_ranges(self, path, evidence, records):
        wanted = self._wanted(path.name, records)
        intervals = self._verified_ranges(path, evidence)
        if not evidence.get("initialized") or missing_spans(wanted, intervals):
            raise ValueError("staged ranges do not cover converted tensor")
        return intervals

    def _preflight(self, source_bytes, output_bytes):
        # Budget sparse extents rounded to filesystem blocks on both filesystems.
        source_bytes += JOURNAL_RESERVE
        if self.source.stat().st_dev == self.target.stat().st_dev:
            converter.preflight_storage(self.target, source_bytes + output_bytes, self.floor)
        else:
            converter.preflight_storage(self.source, source_bytes, self.floor)
            converter.preflight_storage(self.target, output_bytes + JOURNAL_RESERVE, self.floor)

    def stage_ranges(self, name, records):
        """Resume only unverified spans; a range journal never precedes data fsync."""
        path = self._path(name)
        wanted = self._wanted(name, records)
        item = self.headers["shards"][name]
        group = records[0].group
        evidence = self.owned.get(name)
        exists = path.exists()
        if evidence is not None and (evidence.get("mode") != "ranges"
                or evidence.get("revision") != SOURCE_REVISION
                or evidence.get("header_sha256") != json_sha(item)
                or evidence.get("group") != group):
            raise ValueError("existing staging belongs to a different group or identity")
        if exists:
            if evidence is None:
                raise ValueError("refusing to adopt an existing source file")
            verified = self._verified_ranges(path, evidence)
        else:
            if evidence and (evidence.get("ranges") or evidence.get("identity")):
                raise ValueError("owned staging disappeared; preserving its integrity evidence")
            verified = []
        missing = missing_spans(wanted, verified)
        initialize = not evidence or not evidence.get("initialized")
        self._preflight(allocation_bytes(missing + ([(0, item["start"])] if initialize else [])), self.output_reserve)
        if evidence is None:
            evidence = {"mode": "ranges", "revision": SOURCE_REVISION, "group": group,
                        "bytes": item["size"], "header_sha256": json_sha(item),
                        "ranges": [], "complete": False, "initialized": False}
            # Record intent BEFORE exclusive creation. A collision is never adopted.
            self.owned[name] = evidence
            converter.atomic_json(self.owned_path, self.owned)
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | (0 if exists else os.O_CREAT | os.O_EXCL), 0o600)
        try:
            if not exists:
                evidence["identity"] = identity(os.fstat(fd))
                converter.atomic_json(self.owned_path, self.owned)
            elif identity(os.fstat(fd)) != evidence["identity"]:
                raise ValueError("staging changed while opening")
            if initialize:
                os.ftruncate(fd, item["size"])
                write_all(fd, self._header_payload(name), 0)
                os.fsync(fd)
                converter._fsync_directory(self.source)
                evidence["initialized"] = True
            evidence["complete"] = False
            converter.atomic_json(self.owned_path, self.owned)
            tasks = [(lo, min(lo + RANGE_BYTES, hi))
                     for start, hi in missing for lo in range(start, hi, RANGE_BYTES)]
            self._transfer_ranges(fd, name, tasks, evidence)
            self.verify_ranges(path, evidence, records)
            evidence["complete"] = True
            converter.atomic_json(self.owned_path, self.owned)
        finally:
            os.close(fd)

    def _transfer_ranges(self, fd, name, tasks, evidence):
        item = self.headers["shards"][name]
        stopped = threading.Event()

        def transfer(span):
            lo, hi = span
            if stopped.is_set():
                return None
            # Distinct URLs prevent cached redirects reusing a different Range.
            url = fetcher.resolve_url(name) + f"?range={lo}-{hi-1}"
            for attempt in range(NETWORK_ATTEMPTS):
                try:
                    with fetcher._request(url, {"Range": f"bytes={lo}-{hi-1}", "Accept-Encoding": "identity"}) as response:
                        if (response.status != 206
                                or response.headers.get("Content-Range") != f"bytes {lo}-{hi-1}/{item['size']}"
                                or response.headers.get("Content-Encoding", "identity") != "identity"):
                            raise ValueError("server did not honor exact source byte range")
                        data = response.read(hi - lo + 1)
                    break
                except (TimeoutError, ConnectionError, urllib.error.URLError) as error:
                    retryable = not isinstance(error, urllib.error.HTTPError) or error.code in (429, 500, 502, 503, 504)
                    if not retryable or attempt + 1 == NETWORK_ATTEMPTS:
                        raise
                    print(f"[range retry {attempt + 1}/{NETWORK_ATTEMPTS - 1}] {name} bytes={lo}-{hi-1} {type(error).__name__}", flush=True)
                    if stopped.wait(RETRY_WAIT_SECONDS * (2 ** attempt)):
                        return None
            if len(data) != hi - lo:
                raise ValueError("truncated or oversized source range")
            write_all(fd, data, lo)
            return {"offset": lo, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

        def commit(future):
            record = future.result()
            if record is not None:
                os.fsync(fd)
                evidence["ranges"].append(record)
                converter.atomic_json(self.owned_path, self.owned)

        pool = ThreadPoolExecutor(max_workers=WORKERS)
        pending = {}
        next_task = 0
        try:
            while next_task < len(tasks) or pending:
                # Include in-flight writes when rechecking the floor.
                outstanding = tasks[next_task:] + list(pending.values())
                self._preflight(allocation_bytes(outstanding), self.output_reserve)
                while next_task < len(tasks) and len(pending) < WORKERS:
                    span = tasks[next_task]
                    pending[pool.submit(transfer, span)] = span
                    next_task += 1
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    del pending[future]
                    commit(future)
                print(f"[ranges] {name} verified={sum(s['bytes'] for s in evidence['ranges'])} bytes", flush=True)
        finally:
            stopped.set()
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True)
            # Preserve successes concurrent with failure/cancellation. Submit no more.
            for future in pending:
                if not future.cancelled() and future.exception() is None:
                    commit(future)

    def prepare(self, records):
        by_name = {n: [r for r in records if r.source_shard == n]
                   for n in sorted({r.source_shard for r in records})}
        need = 0
        for name, shard_records in by_name.items():
            path = self._path(name)
            wanted = self._wanted(name, shard_records)
            evidence = self.owned.get(name)
            if path.exists() and evidence and evidence.get("mode") == "ranges":
                verified = self._verified_ranges(path, evidence)
                need += allocation_bytes(missing_spans(wanted, verified))
            elif not path.exists():
                need += allocation_bytes([(0, self.headers["shards"][name]["start"])] + wanted)
        self.output_reserve = converter.projected_output_bytes({"group": records}, converter.DEFAULT_ALIGNMENT)
        self._preflight(need, self.output_reserve)
        try:
            for name, shard_records in by_name.items():
                path = self._path(name)
                evidence = self.owned.get(name)
                if not path.exists() or (evidence and evidence.get("mode") == "ranges"):
                    self.stage_ranges(name, shard_records)
                    self.verify_ranges(path, self.owned[name], shard_records)
                else:
                    # Pre-existing full shards need hash evidence but are never claimed.
                    fetch_state = self.source / fetcher.STATE_FILE
                    completed = json.loads(fetch_state.read_text()).get("completed", {}) if fetch_state.exists() else {}
                    full_evidence = evidence or completed.get(name)
                    if not full_evidence:
                        raise ValueError("existing source has no download integrity evidence")
                    fetcher._verify_completed(name, path, full_evidence)
                if path.stat().st_size != self.headers["shards"][name]["size"]:
                    raise ValueError(f"source length mismatch: {name}")
                if converter.read_safetensors_header(path) != self.header(name):
                    raise ValueError(f"source header mismatch: {name}")
                print(f"[staged] {name}", flush=True)
        finally:
            self.output_reserve = 0

    def release(self, records, segment, inventory):
        group = records[0].group
        # Earlier completed groups may share a shard with an interrupted later group.
        owned = {name: self.owned[name] for name in sorted({r.source_shard for r in records})
                 if name in self.owned and self.owned[name].get("group") == group
                 and self.owned[name].get("identity")}
        if not owned:
            return
        state = json.loads((self.target / converter.STATE_FILE).read_text())
        if (state.get("completed", {}).get(group) != segment
                or state.get("inventory", {}).get(group) != inventory):
            raise ValueError("converted output has not been durably committed")
        binding = {"revision": SOURCE_REVISION, "segment": segment,
                   "inventory_sha256": json_sha(inventory)}
        prior = self.validation.get(group)
        if prior is None:
            for name, evidence in owned.items():
                self.verify_ranges(self._path(name), evidence, [r for r in records if r.source_shard == name])
            result = validator._validate_group(self.source, self.target, group, records, inventory, segment)
            prior = {"binding": binding, "validation": result}
            self.validation[group] = prior
            converter.atomic_json(self.validation_path, self.validation)
        else:
            output = self.target / group
            if (prior.get("binding") != binding or output.stat().st_size != segment["size"]
                    or converter.sha256_file(output) != segment["sha256"]):
                raise ValueError("recovery validation does not bind current converted output")
        for name, evidence in owned.items():
            path = self._path(name)
            if path.exists():
                self._check_owner(path, evidence)
                if validator.stat_identity(path) != prior["validation"]["source_identities"][name]:
                    raise ValueError("source staging changed after independent validation")
                path.unlink()
                converter._fsync_directory(self.source)
            del self.owned[name]
            converter.atomic_json(self.owned_path, self.owned)


def main():
    global WORKERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headers-only", action="store_true")
    parser.add_argument("--download-workers", type=int, choices=range(1, 25), default=WORKERS,
                        help="bounded concurrent 8 MiB source ranges (default: 6)")
    parser.add_argument("--stop-after-groups", type=int,
                        help="stop at a durable boundary after this many new groups")
    args = parser.parse_args()
    WORKERS = args.download_workers
    if args.stop_after_groups is not None and args.stop_after_groups < 1:
        parser.error("--stop-after-groups must be positive")
    with recovery_lock(args.source):
        fetcher.fetch(args.source, metadata_only=True)
        args.output.mkdir(parents=True, exist_ok=True)
        recovery = Recovery(args.source, args.output)
        groups, _ = converter.build_plan(args.source, header_provider=recovery.header)
        remaining = sum(converter.projected_output_bytes({g: r}, converter.DEFAULT_ALIGNMENT)
                        for g, r in groups.items() if not (args.output / g).exists())
        largest_staging = max(sum(allocation_bytes(
            [(0, recovery.headers["shards"][name]["start"])] + recovery._wanted(
                name, [r for r in records if r.source_shard == name]))
            for name in {r.source_shard for r in records}) for records in groups.values())
        recovery._preflight(largest_staging, remaining)
        if args.headers_only:
            print(json.dumps({"groups": len(groups), "output_bytes": converter.projected_output_bytes(groups, converter.DEFAULT_ALIGNMENT),
                              "maximum_staging_bytes": largest_staging}))
            return 0
        state = converter.convert(args.source, args.output, header_provider=recovery.header,
                                  prepare_group=recovery.prepare, release_group=recovery.release,
                                  stop_after_groups=args.stop_after_groups)
        print(json.dumps({"status": state["status"], "manifest": state.get("manifest")}))
        return 0 if state["status"] == "complete" or args.stop_after_groups else 1


if __name__ == "__main__":
    raise SystemExit(main())
