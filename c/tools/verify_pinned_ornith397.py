#!/usr/bin/env python3
"""Compare a reconstructed Ornith397 snapshot with its preserved manifests.

`colib doctor --verify-hashes` proves the physical shards match the live
conversion ledger, and `audit_expert_sidecar.py --verify-hashes` proves the
same for the routed-expert sidecar. Neither knows about the retirement
evidence, so neither can prove the reconstruction reproduced the *pinned*
model rather than merely a self-consistent one.

This tool closes that loop. It compares the live ledger, quantization
manifest, and sidecar manifest field by field with
`docs/research/artifacts/ornith397_retirement/`, and can independently rehash
every physical file so the conclusion does not depend on either audit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_perf_trials import atomic_json, sha256_file, utc_now  # noqa: E402


SCHEMA = "colib.ornith397-pinned-verification.v1"
BASE_SCHEMA = "colib.ornith397-base-shards.v1"
LEDGER_FILE = ".conversion-state.json"
QUANTIZATION_FILE = "quantization.json"

SHARD_FIELDS = ("output", "sha256", "file_size", "data_bytes", "tensor_count")
SIDECAR_FIELDS = (
    "file",
    "sha256",
    "bytes",
    "tensor_count",
    "layer",
    "experts",
    "first_expert",
    "last_expert",
)
SIGNATURE_FIELDS = (
    "source",
    "source_fingerprint",
    "xbits",
    "io_bits",
    "shared_bits",
    "group_size",
    "include_mtp",
)
TOTAL_FIELDS = (
    "data_bytes",
    "tensor_count",
    "logical_tensor_count",
    "output_shards",
)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} root is not an object")
    return value


def digest_many(paths: list[Path], workers: int) -> list[str]:
    if workers < 2 or len(paths) < 2:
        return [sha256_file(path) for path in paths]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(sha256_file, paths))


def compare_records(
    kind: str,
    expected: dict[str, dict[str, Any]],
    observed: dict[str, dict[str, Any]],
    fields: tuple[str, ...],
    failures: list[str],
) -> int:
    matched = 0
    for name in sorted(set(expected) | set(observed)):
        want, got = expected.get(name), observed.get(name)
        if want is None:
            failures.append(f"{kind} {name} is not in the preserved manifest")
            continue
        if got is None:
            failures.append(f"{kind} {name} is missing from the snapshot")
            continue
        differing = [
            field for field in fields if want.get(field) != got.get(field)
        ]
        if differing:
            for field in differing:
                failures.append(
                    f"{kind} {name} {field} is {got.get(field)!r}, "
                    f"expected {want.get(field)!r}"
                )
            continue
        matched += 1
    return matched


def verify_physical(
    kind: str,
    snapshot: Path,
    expected: dict[str, dict[str, Any]],
    size_field: str,
    rehash: bool,
    workers: int,
    failures: list[str],
) -> dict[str, Any]:
    names = sorted(expected)
    present: list[str] = []
    for name in names:
        path = snapshot / name
        if not path.is_file():
            failures.append(f"{kind} {name} is absent from the snapshot")
            continue
        size = path.stat().st_size
        if size != expected[name].get(size_field):
            failures.append(
                f"{kind} {name} is {size} bytes, "
                f"expected {expected[name].get(size_field)}"
            )
            continue
        present.append(name)
    record: dict[str, Any] = {
        "files": len(names),
        "size_verified": len(present),
        "rehashed": 0,
        "hash_verified": 0,
    }
    if not rehash:
        return record
    digests = digest_many([snapshot / name for name in present], workers)
    record["rehashed"] = len(digests)
    for name, digest in zip(present, digests):
        if digest != expected[name].get("sha256"):
            failures.append(
                f"{kind} {name} hashes to {digest}, "
                f"expected {expected[name].get('sha256')}"
            )
            continue
        record["hash_verified"] += 1
    return record


def verify_base(
    snapshot: Path,
    manifest_path: Path,
    *,
    rehash: bool,
    workers: int,
) -> dict[str, Any]:
    failures: list[str] = []
    manifest = read_object(manifest_path)
    if manifest.get("schema") != BASE_SCHEMA:
        failures.append(
            f"preserved base manifest schema is {manifest.get('schema')!r}"
        )
    expected = manifest.get("shards")
    if not isinstance(expected, dict):
        raise ValueError("preserved base manifest has no shard object")

    ledger_path = snapshot / LEDGER_FILE
    quantization_path = snapshot / QUANTIZATION_FILE
    ledger_sha256 = sha256_file(ledger_path) if ledger_path.is_file() else None
    quantization_sha256 = (
        sha256_file(quantization_path) if quantization_path.is_file() else None
    )
    if ledger_sha256 is None:
        failures.append("the live conversion ledger is absent")
    elif ledger_sha256 != manifest.get("conversion_state_sha256"):
        failures.append(
            f"live ledger hashes to {ledger_sha256}, expected "
            f"{manifest.get('conversion_state_sha256')}"
        )
    if quantization_sha256 is None:
        failures.append("the live quantization manifest is absent")
    elif quantization_sha256 != manifest.get("quantization_sha256"):
        failures.append(
            f"live quantization manifest hashes to {quantization_sha256}, "
            f"expected {manifest.get('quantization_sha256')}"
        )

    observed: dict[str, dict[str, Any]] = {}
    if ledger_path.is_file():
        ledger = read_object(ledger_path)
        completed = ledger.get("completed")
        if not isinstance(completed, dict):
            failures.append("the live ledger has no completed object")
        else:
            observed = {
                str(record.get("output")): record
                for record in completed.values()
                if isinstance(record, dict)
            }
            if len(observed) != len(completed):
                failures.append("live ledger outputs are not unique")
        signature = ledger.get("signature")
        if isinstance(signature, dict):
            preserved_signature = manifest.get("signature") or {}
            for field in SIGNATURE_FIELDS:
                if field not in preserved_signature:
                    continue
                if signature.get(field) != preserved_signature.get(field):
                    failures.append(
                        f"ledger signature {field} is {signature.get(field)!r}, "
                        f"expected {preserved_signature.get(field)!r}"
                    )

    if quantization_path.is_file():
        quantization = read_object(quantization_path)
        if quantization.get("complete") is not True:
            failures.append("the live quantization manifest is incomplete")
        totals = manifest.get("totals") or {}
        for field in TOTAL_FIELDS:
            if field not in totals:
                continue
            if quantization.get(field) != totals.get(field):
                failures.append(
                    f"quantization {field} is {quantization.get(field)!r}, "
                    f"expected {totals.get(field)!r}"
                )

    matched = compare_records(
        "base shard", expected, observed, SHARD_FIELDS, failures
    )
    physical = verify_physical(
        "base shard",
        snapshot,
        expected,
        "file_size",
        rehash,
        workers,
        failures,
    )
    return {
        "manifest": str(manifest_path),
        "expected_shards": len(expected),
        "ledger_shards": len(observed),
        "records_matched": matched,
        "ledger_sha256": ledger_sha256,
        "quantization_sha256": quantization_sha256,
        "physical": physical,
        "failures": failures,
        "passed": not failures
        and matched == len(expected)
        and physical["size_verified"] == len(expected)
        and (not rehash or physical["hash_verified"] == len(expected)),
    }


def verify_sidecar(
    snapshot: Path,
    manifest_path: Path,
    *,
    bits: int,
    rehash: bool,
    workers: int,
) -> dict[str, Any]:
    failures: list[str] = []
    preserved = read_object(manifest_path)
    records = preserved.get("files")
    if not isinstance(records, list):
        raise ValueError("preserved sidecar manifest has no file list")
    expected = {str(record.get("file")): record for record in records}
    if len(expected) != len(records):
        failures.append("preserved sidecar files are not unique")

    live_path = snapshot / f"expert-q{bits}.json"
    live_sha256 = sha256_file(live_path) if live_path.is_file() else None
    observed: dict[str, dict[str, Any]] = {}
    if live_sha256 is None:
        failures.append(f"the live expert-q{bits} manifest is absent")
    else:
        live = read_object(live_path)
        for field in ("format", "complete", "layers", "file_count",
                      "tensor_count", "data_bytes", "config_sha256"):
            if field not in preserved:
                continue
            if live.get(field) != preserved.get(field):
                failures.append(
                    f"sidecar {field} is {live.get(field)!r}, "
                    f"expected {preserved.get(field)!r}"
                )
        live_signature = live.get("signature") or {}
        preserved_signature = preserved.get("signature") or {}
        for field in sorted(preserved_signature):
            if live_signature.get(field) != preserved_signature.get(field):
                failures.append(
                    f"sidecar signature {field} is "
                    f"{live_signature.get(field)!r}, expected "
                    f"{preserved_signature.get(field)!r}"
                )
        live_records = live.get("files")
        if not isinstance(live_records, list):
            failures.append("the live sidecar manifest has no file list")
        else:
            observed = {
                str(record.get("file")): record for record in live_records
            }
            if len(observed) != len(live_records):
                failures.append("live sidecar files are not unique")

    matched = compare_records(
        f"q{bits} file", expected, observed, SIDECAR_FIELDS, failures
    )
    physical = verify_physical(
        f"q{bits} file", snapshot, expected, "bytes", rehash, workers, failures
    )
    return {
        "manifest": str(manifest_path),
        "expected_files": len(expected),
        "live_files": len(observed),
        "records_matched": matched,
        "live_manifest_sha256": live_sha256,
        "physical": physical,
        "failures": failures,
        "passed": not failures
        and matched == len(expected)
        and physical["size_verified"] == len(expected)
        and (not rehash or physical["hash_verified"] == len(expected)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--sidecar-manifest", type=Path)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument(
        "--rehash",
        action="store_true",
        help="independently recompute every physical file digest",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.base_manifest is None and args.sidecar_manifest is None:
        parser.error("at least one of --base-manifest or --sidecar-manifest")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.bits < 1:
        parser.error("--bits must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = args.snapshot.resolve()
    if not snapshot.is_dir():
        print(f"snapshot is not a directory: {snapshot}", file=sys.stderr)
        return 2
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "snapshot": str(snapshot),
        "rehash": args.rehash,
    }
    try:
        if args.base_manifest is not None:
            report["base"] = verify_base(
                snapshot,
                args.base_manifest.resolve(strict=True),
                rehash=args.rehash,
                workers=args.workers,
            )
        if args.sidecar_manifest is not None:
            report[f"expert_q{args.bits}"] = verify_sidecar(
                snapshot,
                args.sidecar_manifest.resolve(strict=True),
                bits=args.bits,
                rehash=args.rehash,
                workers=args.workers,
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"verification error: {error}", file=sys.stderr)
        return 2
    sections = [
        value
        for key, value in report.items()
        if isinstance(value, dict) and "passed" in value
    ]
    report["passed"] = bool(sections) and all(
        section["passed"] for section in sections
    )
    if args.output is not None:
        atomic_json(args.output, report)
        report["output"] = str(args.output)
    summary = {
        "passed": report["passed"],
        "snapshot": report["snapshot"],
        "rehash": report["rehash"],
    }
    for key, section in report.items():
        if not isinstance(section, dict) or "passed" not in section:
            continue
        summary[key] = {
            "passed": section["passed"],
            "records_matched": section["records_matched"],
            "physical": section["physical"],
            "failures": section["failures"][:20],
            "failure_count": len(section["failures"]),
        }
    if args.output is not None:
        summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
