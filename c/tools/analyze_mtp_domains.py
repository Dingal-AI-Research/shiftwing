#!/usr/bin/env python3
"""Summarize and validate a bench_mtp_domains.py result."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

LOGIT_RE = re.compile(r"^\[LOGITS (\d+)\](.*)$", re.MULTILINE)
MOE_CHECK_RE = re.compile(
    r"^\[CUDA_MOE_CHECK\] layer=(\d+) batch=(\d+) maxdiff=([0-9.eE+-]+)$",
    re.MULTILINE,
)

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        default=root / "c" / "bench" / "mtp_domains.json",
    )
    return parser.parse_args()


def first_difference(left: list[int], right: list[int]) -> tuple[int, int, int] | None:
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if lhs != rhs:
            return index, lhs, rhs
    if len(left) != len(right):
        lhs = left[len(right)] if len(left) > len(right) else -1
        rhs = right[len(left)] if len(right) > len(left) else -1
        return min(len(left), len(right)), lhs, rhs
    return None


def logit_lines(result: dict[str, Any], center: int) -> list[str]:
    lines = []
    for match in LOGIT_RE.finditer(result.get("stderr", "")):
        step = int(match.group(1))
        if center - 1 <= step <= center + 1:
            lines.append(match.group(0))
    return lines


def main() -> None:
    args = parse_args()
    payload = json.loads(args.result.read_text())
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for record in payload["runs"]:
        grouped.setdefault(record["prompt_id"], {})[int(record["depth"])] = record[
            "result"
        ]
    depths = sorted({int(record["depth"]) for record in payload["runs"]})

    print(
        "prompt\tdepth\ttok/s\tspeedup\taccept%\ttargets\tskipped\t"
        "accepted_margin\tskipped_margin\treplay_s\t"
        "overlap%\tmoe_s\tmisses\tdraft_miss\tverify_miss\treplay_miss"
    )
    for prompt in payload["prompts"]:
        prompt_id = prompt["id"]
        runs = grouped[prompt_id]
        baseline_depth = 0 if 0 in runs else depths[0]
        baseline = runs[baseline_depth]["decode_tok_s"]
        for depth in depths:
            result = runs[depth]
            mtp = result.get("mtp", {})
            batch = result.get("cuda_batch", {})
            cache = result.get("mtp_cache", {})
            confidence = result.get("mtp_confidence", {})
            admission = result.get("mtp_admission", {})
            print(
                f"{prompt_id}\tD{depth}\t{result['decode_tok_s']:.2f}\t"
                f"{result['decode_tok_s'] / baseline:.3f}\t"
                f"{mtp.get('acceptance_pct', 0):.1f}\t"
                f"{mtp.get('target_forwards', 0)}\t"
                f"{admission.get('skipped', 0)}\t"
                f"{confidence.get('accepted_mean', 0):.3f}\t"
                f"{confidence.get('unverified_mean', 0):.3f}\t"
                f"{mtp.get('replay_s', 0):.3f}\t"
                f"{batch.get('overlap_pct', 0):.2f}\t"
                f"{result['profile']['moe_s']:.3f}\t"
                f"{result['profile']['expert_misses']}\t"
                f"{cache.get('draft_misses', 0)}\t"
                f"{cache.get('verify_misses', 0)}\t"
                f"{cache.get('replay_misses', 0)}"
            )

    print("\naggregate")
    print(
        "depth\tmean tok/s\tgeomean speedup\tweighted accept%\t"
        "weighted overlap%\tskipped"
    )
    for depth in depths:
        results = [grouped[prompt["id"]][depth] for prompt in payload["prompts"]]
        speeds = [result["decode_tok_s"] for result in results]
        ratios = [
            grouped[prompt["id"]][depth]["decode_tok_s"]
            / grouped[prompt["id"]][
                0 if 0 in grouped[prompt["id"]] else depth
            ]["decode_tok_s"]
            for prompt in payload["prompts"]
        ]
        geomean = statistics.geometric_mean(ratios)
        if depth:
            accepted = sum(result["mtp"]["accepted"] for result in results)
            proposed = sum(result["mtp"]["proposed"] for result in results)
            routes = sum(
                result.get("cuda_batch", {}).get("routes", 0) for result in results
            )
            unique = sum(
                result.get("cuda_batch", {}).get("unique_experts", 0)
                for result in results
            )
            acceptance = 100.0 * accepted / proposed
            overlap = 100.0 * (1.0 - unique / routes) if routes else 0.0
            skipped = sum(
                result.get("mtp_admission", {}).get("skipped", 0)
                for result in results
            )
        else:
            acceptance = overlap = 0.0
            skipped = 0
        print(
            f"D{depth}\t{statistics.mean(speeds):.2f}\t{geomean:.3f}\t"
            f"{acceptance:.2f}\t{overlap:.2f}\t{skipped}"
        )

    print("\ntoken identity")
    any_difference = False
    if 0 not in depths:
        print("not evaluated (result has no D0 baseline)")
    else:
        for prompt in payload["prompts"]:
            prompt_id = prompt["id"]
            baseline = grouped[prompt_id][0]["tokens"]
            for depth in depths:
                if depth == 0:
                    continue
                difference = first_difference(
                    baseline, grouped[prompt_id][depth]["tokens"]
                )
                if difference:
                    any_difference = True
                    index, baseline_token, speculative_token = difference
                    print(
                        f"{prompt_id} D{depth}: first difference at generated token "
                        f"{index}: D0={baseline_token}, D{depth}={speculative_token}"
                    )
                    for line in logit_lines(grouped[prompt_id][0], index):
                        print(f"  D0 {line}")
                    for line in logit_lines(grouped[prompt_id][depth], index):
                        print(f"  D{depth} {line}")
                else:
                    print(f"{prompt_id} D{depth}: identical")
    checks = []
    for record in payload["runs"]:
        for match in MOE_CHECK_RE.finditer(record["result"].get("stderr", "")):
            checks.append(
                (
                    float(match.group(3)),
                    record["prompt_id"],
                    int(record["depth"]),
                    int(match.group(1)),
                    int(match.group(2)),
                )
            )
    if checks:
        print("\nCUDA MoE batch checks")
        print(f"comparisons={len(checks)} maximum={max(item[0] for item in checks):.8g}")
        for difference, prompt_id, depth, layer, batch in sorted(
            checks, reverse=True
        )[:10]:
            print(
                f"{prompt_id} D{depth} layer={layer} batch={batch} "
                f"maxdiff={difference:.8g}"
            )
    if any_difference:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
