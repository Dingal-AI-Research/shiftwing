#!/usr/bin/env python3
"""Compare reversed-order batch artifacts and enforce the warm AB/BA gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_ORDERS = (
    ("sequential", "concurrent"),
    ("concurrent", "sequential"),
)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root is not an object")
    return value


def _order(artifact: dict[str, Any]) -> tuple[str, ...]:
    runs = artifact.get("runs")
    if not isinstance(runs, list):
        raise ValueError("runs are missing")
    return tuple(
        run.get("mode") if isinstance(run, dict) else None
        for run in runs
    )


def _outputs(artifact: dict[str, Any]) -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {}
    for run in artifact["runs"]:
        mode = run["mode"]
        requests = run.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError(f"{mode} requests are missing")
        texts = []
        for request in requests:
            if not isinstance(request, dict) or not isinstance(
                request.get("text"), str
            ):
                raise ValueError(f"{mode} request output is invalid")
            texts.append(request["text"])
        outputs[mode] = texts
    return outputs


def compare(
    ab: dict[str, Any],
    ba: dict[str, Any],
    *,
    minimum_geomean: float,
    minimum_each: float,
    expected_family: str = "ornith-1.0",
    expected_source: str | None = None,
) -> dict[str, Any]:
    """Validate two artifacts and return a schema-versioned gate result."""
    failures: list[str] = []
    for label, artifact, expected_order in (
        ("AB", ab, EXPECTED_ORDERS[0]),
        ("BA", ba, EXPECTED_ORDERS[1]),
    ):
        try:
            order = _order(artifact)
        except ValueError as error:
            failures.append(f"{label}: {error}")
            order = ()
        if order != expected_order:
            failures.append(
                f"{label}: mode order {order!r} != {expected_order!r}"
            )
        acceptance = artifact.get("acceptance")
        if not isinstance(acceptance, dict) or acceptance.get("passed") is not True:
            failures.append(f"{label}: source artifact acceptance did not pass")
        if artifact.get("cuda") is not True:
            failures.append(f"{label}: CUDA was not enabled")
        prompt = artifact.get("prompt")
        if not isinstance(prompt, dict) or prompt.get("raw") is not False:
            failures.append(f"{label}: production prompt was not family-rendered")

    identity_fields = (
        "model_family",
        "model_manifest",
        "expert_lowbit_manifest",
        "prompt",
        "tier_configuration",
    )
    for name in identity_fields:
        if ab.get(name) != ba.get(name):
            failures.append(f"AB/BA {name} differs")
    if ab.get("model_family") != expected_family:
        failures.append(
            f"model family {ab.get('model_family')!r} != {expected_family!r}"
        )
    observed_source = (ab.get("model_manifest") or {}).get("source")
    if expected_source is not None and observed_source != expected_source:
        failures.append(
            f"model source {observed_source!r} != {expected_source!r}"
        )

    speedups: list[float] = []
    outputs: list[dict[str, list[str]]] = []
    for label, artifact in (("AB", ab), ("BA", ba)):
        comparison = artifact.get("comparison")
        try:
            value = float(comparison["aggregate_tps_speedup"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError
            speedups.append(value)
        except (KeyError, TypeError, ValueError):
            failures.append(f"{label}: aggregate TPS speedup is invalid")
        if not isinstance(comparison, dict) or comparison.get("outputs_match") is not True:
            failures.append(f"{label}: sequential/concurrent outputs differ")
        try:
            outputs.append(_outputs(artifact))
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{label}: {error}")

    if len(outputs) == 2:
        reference = outputs[0].get("sequential")
        if any(
            mode_outputs != reference
            for artifact_outputs in outputs
            for mode_outputs in artifact_outputs.values()
        ):
            failures.append("deterministic outputs differ across AB/BA modes")

    geomean = (
        math.sqrt(speedups[0] * speedups[1])
        if len(speedups) == 2
        else None
    )
    if len(speedups) == 2 and min(speedups) < minimum_each:
        failures.append(
            f"minimum order speedup {min(speedups):.6f} < {minimum_each:.6f}"
        )
    if geomean is None or geomean < minimum_geomean:
        failures.append(
            f"geometric-mean speedup {geomean!r} < {minimum_geomean:.6f}"
        )

    return {
        "schema_version": 1,
        "model_family": ab.get("model_family"),
        "model_manifest": ab.get("model_manifest"),
        "expert_lowbit_manifest": ab.get("expert_lowbit_manifest"),
        "prompt": ab.get("prompt"),
        "speedups": {
            "ab": speedups[0] if len(speedups) > 0 else None,
            "ba": speedups[1] if len(speedups) > 1 else None,
            "geometric_mean": geomean,
            "minimum": min(speedups) if len(speedups) == 2 else None,
        },
        "acceptance": {
            "minimum_each": minimum_each,
            "minimum_geometric_mean": minimum_geomean,
            "expected_family": expected_family,
            "expected_source": expected_source,
            "passed": not failures,
            "failures": failures,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab", type=Path, required=True)
    parser.add_argument("--ba", type=Path, required=True)
    parser.add_argument("--minimum-geomean", type=float, default=1.0)
    parser.add_argument("--minimum-each", type=float, default=0.95)
    parser.add_argument("--expect-family", default="ornith-1.0")
    parser.add_argument("--expect-source")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name, value in (
        ("--minimum-geomean", args.minimum_geomean),
        ("--minimum-each", args.minimum_each),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be positive and finite")

    result = compare(
        _load(args.ab),
        _load(args.ba),
        minimum_geomean=args.minimum_geomean,
        minimum_each=args.minimum_each,
        expected_family=args.expect_family,
        expected_source=args.expect_source,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
