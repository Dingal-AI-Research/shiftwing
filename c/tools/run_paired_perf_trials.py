#!/usr/bin/env python3
"""Run alternating AB/BA paired benchmark trials with atomic resume state.

Each pair executes both arms back to back and alternates their launch order,
so a monotonic drift in machine state cannot be attributed to either arm. The
controller binds the engine, fixtures, and manifests by hash, refuses to
resume across any binding drift, verifies every published artifact, and then
reports paired confidence bounds for decode throughput and time to first
token together with the preregistered exactness gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_perf_trials import (  # noqa: E402
    LABEL_RE,
    atomic_json,
    binding_record,
    command_for_trial,
    dotted_value,
    environment_snapshot,
    git_metadata,
    sha256_file,
    utc_now,
)
from runtime_env import engine_source_sha256  # noqa: E402


SCHEMA = "colib.paired-performance-trials.v1"
REPORT_SCHEMA = "colib.paired-performance-report.v1"
ARM_SEPARATOR = ":::"
ARMS = ("control", "candidate")

# Fields that must be identical across every trial. Available RAM is excluded
# deliberately: the preserved five-trial control varies from 2.289 to 2.312
# GiB with no intervention, so gating on it would fail every real comparison.
# Its range is reported instead.
HARDWARE_IDENTITY = (
    "cores",
    "cpu",
    "gpu",
    "gpus",
    "ram_total_gb",
    "vram_total_gb",
)
HARDWARE_VOLATILE = ("ram_avail_gb",)


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction with Lentz's method."""

    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    raise ValueError("incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees: int) -> float:
    """Cumulative distribution of Student's t with `degrees` freedom."""

    if degrees < 1:
        raise ValueError("degrees of freedom must be positive")
    tail = 0.5 * regularized_incomplete_beta(
        degrees / 2.0, 0.5, degrees / (degrees + value * value)
    )
    return 1.0 - tail if value > 0.0 else tail


def student_t_critical(degrees: int, confidence: float) -> float:
    """Two-sided critical value for the requested confidence level."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    target = 1.0 - (1.0 - confidence) / 2.0
    low, high = 0.0, 1.0
    while student_t_cdf(high, degrees) < target:
        high *= 2.0
        if high > 1e12:
            raise ValueError("critical value search diverged")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if student_t_cdf(middle, degrees) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def paired_ratio_interval(
    ratios: list[float], confidence: float
) -> dict[str, Any]:
    """Geometric-mean ratio and its paired log-scale confidence interval."""

    if not ratios or any(value <= 0.0 for value in ratios):
        return {
            "count": len(ratios),
            "geometric_mean": None,
            "lower": None,
            "upper": None,
            "confidence": confidence,
            "reason": "ratios must be positive and non-empty",
        }
    logs = [math.log(value) for value in ratios]
    center = statistics.fmean(logs)
    record: dict[str, Any] = {
        "count": len(ratios),
        "geometric_mean": math.exp(center),
        "minimum": min(ratios),
        "maximum": max(ratios),
        "confidence": confidence,
    }
    if len(logs) < 2:
        record.update({"lower": None, "upper": None, "reason": "one pair only"})
        return record
    spread = statistics.stdev(logs)
    critical = student_t_critical(len(logs) - 1, confidence)
    half = critical * spread / math.sqrt(len(logs))
    record.update(
        {
            "lower": math.exp(center - half),
            "upper": math.exp(center + half),
            "log_mean": center,
            "log_stdev": spread,
            "t_critical": critical,
        }
    )
    return record


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def trial_metrics(result: dict[str, Any], tps_key: str) -> dict[str, Any]:
    """Reduce one qualifier artifact to the values the paired report needs."""

    measured = result.get("measured") or []
    if not isinstance(measured, list) or not measured:
        raise ValueError("trial result has no measured turns")
    turns: list[dict[str, Any]] = []
    for turn in measured:
        stats = turn.get("stats") or {}
        profile = stats.get("profile") or {}
        text = turn.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("measured turn has empty output text")
        turns.append(
            {
                "prompt_index": turn.get("prompt_index"),
                "prompt_sha256": digest_text(str(turn.get("prompt", ""))),
                "text_sha256": digest_text(text),
                "text_chars": len(text),
                "ttft_s": turn.get("ttft_s"),
                "wall_s": turn.get("wall_s"),
                "tokens_per_second": stats.get("tokens_per_second"),
                "completion_tokens": stats.get("completion_tokens"),
                "prompt_tokens": stats.get("prompt_tokens"),
                "decode_expert_read_bytes": profile.get(
                    "decode_expert_read_bytes"
                ),
                "decode_expert_misses": profile.get("decode_expert_misses"),
                "decode_expert_cpu_hits": profile.get("decode_expert_cpu_hits"),
                "decode_expert_gpu_hits": profile.get("decode_expert_gpu_hits"),
            }
        )
    ttfts = [float(turn["ttft_s"]) for turn in turns if turn["ttft_s"] is not None]
    if len(ttfts) != len(turns):
        raise ValueError("measured turns are missing time-to-first-token")
    summary = result.get("summary") or {}
    resident = result.get("resident") or {}

    def total(field: str) -> int | None:
        values = [turn[field] for turn in turns]
        if any(value is None for value in values):
            return None
        return sum(int(value) for value in values)

    return {
        "sustained_tps": float(dotted_value(result, tps_key)),
        "median_ttft_s": statistics.median(ttfts),
        "mean_ttft_s": statistics.fmean(ttfts),
        "turn_count": len(turns),
        "turns": turns,
        "output_sha256": digest_text(
            " ".join(turn["text_sha256"] for turn in turns)
        ),
        "prompt_set_sha256": digest_text(
            " ".join(turn["prompt_sha256"] for turn in turns)
        ),
        "decode_expert_read_bytes": total("decode_expert_read_bytes"),
        "decode_expert_misses": total("decode_expert_misses"),
        "summary": {
            "outputs_nonempty": summary.get("outputs_nonempty"),
            "telemetry_complete": summary.get("telemetry_complete"),
            "cuda_active": summary.get("cuda_active"),
            "resident_cuda_graph": summary.get("resident_cuda_graph"),
            "native_q3_active": summary.get("native_q3_active"),
            "q3_route_atlas_active": summary.get("q3_route_atlas_active"),
            "automatic_gate_pass": summary.get("automatic_gate_pass"),
            "median_turn_tps": summary.get("median_turn_tps"),
            "minimum_turn_tps": summary.get("minimum_turn_tps"),
        },
        "host_moe": resident.get("host_moe"),
        "device_moe": resident.get("device_moe"),
        "hardware": result.get("hardware"),
        "configuration": result.get("configuration"),
        "model_manifest": result.get("model_manifest"),
        "expert_lowbit_manifest": result.get("expert_lowbit_manifest"),
    }


def expected_trials(args: argparse.Namespace) -> int:
    return args.prime_trials + args.pairs * len(ARMS)


def prime_arm(index: int) -> str:
    """Priming trials alternate arms so neither is systematically first."""

    return ARMS[(index - 1) % len(ARMS)]


def pair_order(pair: int) -> tuple[str, str]:
    """Odd pairs run control first (AB); even pairs run candidate first (BA)."""

    return ARMS if pair % 2 else (ARMS[1], ARMS[0])


def signature(
    *,
    pairs: int,
    prime_trials: int,
    labels: dict[str, str],
    commands: dict[str, list[str]],
    acceptance_key: str,
    tps_key: str,
    settle_seconds: float,
    bindings: list[dict[str, Any]],
    engine_source: str | None,
) -> dict[str, Any]:
    record = {
        "pairs": pairs,
        "prime_trials": prime_trials,
        "arms": [
            {
                "arm": arm,
                "label": labels[arm],
                "command_template": commands[arm],
            }
            for arm in ARMS
        ],
        "acceptance_key": acceptance_key,
        "tps_key": tps_key,
        "settle_seconds": settle_seconds,
        "bindings": bindings,
        "engine_source_sha256": engine_source,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


def load_or_create_state(
    path: Path, expected: dict[str, Any], root: Path
) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("schema") != SCHEMA:
            raise ValueError("paired trial state schema does not match")
        if state.get("signature") != expected:
            raise ValueError(
                "paired trial state signature does not match current inputs"
            )
        changed = False
        for attempt in state.get("attempts", []):
            if attempt.get("status") == "running":
                attempt["status"] = "interrupted"
                attempt["failed_at"] = utc_now()
                attempt["reason"] = "stale running attempt recovered"
                changed = True
        if changed:
            state["status"] = "interrupted"
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


def verified_completed(state: dict[str, Any], key: str, output: Path) -> bool:
    record = state.get("completed", {}).get(key)
    if not isinstance(record, dict) or not output.is_file():
        return False
    return (
        output.stat().st_size == record.get("bytes")
        and sha256_file(output) == record.get("sha256")
    )


def _flag(value: Any) -> bool:
    return value is True


def hardware_identity(hardware: Any) -> dict[str, Any]:
    """Stable machine identity, excluding values that drift between trials."""

    if not isinstance(hardware, dict):
        return {}
    return {name: hardware.get(name) for name in HARDWARE_IDENTITY}


def analyze(
    state: dict[str, Any],
    *,
    confidence: float,
    per_prompt_floor: float,
) -> dict[str, Any]:
    """Compute paired bounds and preregistered exactness gates."""

    completed = state.get("completed", {})
    pairs = state["signature"]["pairs"]
    failures: list[str] = []
    pair_records: list[dict[str, Any]] = []
    tps_ratios: list[float] = []
    ttft_ratios: list[float] = []
    per_prompt_minimum: float | None = None
    hardware_reference: Any = None
    volatile: dict[str, list[float]] = {name: [] for name in HARDWARE_VOLATILE}
    arm_outputs: dict[str, set[str]] = {arm: set() for arm in ARMS}

    for pair in range(1, pairs + 1):
        entries = {}
        for arm in ARMS:
            record = completed.get(f"{pair}:{arm}")
            if not isinstance(record, dict) or "metrics" not in record:
                failures.append(f"pair {pair} {arm} trial is incomplete")
                break
            entries[arm] = record
        if len(entries) != len(ARMS):
            continue
        control = entries["control"]["metrics"]
        candidate = entries["candidate"]["metrics"]
        arm_outputs["control"].add(control["output_sha256"])
        arm_outputs["candidate"].add(candidate["output_sha256"])

        if control["prompt_set_sha256"] != candidate["prompt_set_sha256"]:
            failures.append(f"pair {pair} arms did not run the same prompts")
        outputs_identical = control["output_sha256"] == candidate["output_sha256"]
        if not outputs_identical:
            failures.append(f"pair {pair} candidate output differs from control")
        for arm, metrics in (("control", control), ("candidate", candidate)):
            flags = metrics["summary"]
            for name in (
                "outputs_nonempty",
                "telemetry_complete",
                "cuda_active",
                "automatic_gate_pass",
            ):
                if not _flag(flags.get(name)):
                    failures.append(f"pair {pair} {arm} {name} is not true")
            if metrics.get("host_moe") != 0:
                failures.append(
                    f"pair {pair} {arm} recorded host-MoE fallback "
                    f"{metrics.get('host_moe')!r}"
                )
        identities = {
            arm: hardware_identity(metrics["hardware"])
            for arm, metrics in (("control", control), ("candidate", candidate))
        }
        if identities["control"] != identities["candidate"]:
            failures.append(f"pair {pair} hardware identity differs between arms")
        if hardware_reference is None:
            hardware_reference = identities["control"]
        elif identities["control"] != hardware_reference:
            failures.append(f"pair {pair} hardware identity differs from pair 1")
        for metrics in (control, candidate):
            hardware = metrics["hardware"]
            if not isinstance(hardware, dict):
                continue
            for name in HARDWARE_VOLATILE:
                value = hardware.get(name)
                if isinstance(value, (int, float)):
                    volatile[name].append(float(value))

        tps_ratio = candidate["sustained_tps"] / control["sustained_tps"]
        ttft_ratio = candidate["median_ttft_s"] / control["median_ttft_s"]
        tps_ratios.append(tps_ratio)
        ttft_ratios.append(ttft_ratio)

        prompt_ratios: list[dict[str, Any]] = []
        control_turns = {turn["prompt_index"]: turn for turn in control["turns"]}
        for turn in candidate["turns"]:
            reference = control_turns.get(turn["prompt_index"])
            if reference is None:
                failures.append(
                    f"pair {pair} prompt {turn['prompt_index']} is missing "
                    "from the control arm"
                )
                continue
            rate = turn.get("tokens_per_second")
            baseline = reference.get("tokens_per_second")
            if not rate or not baseline:
                failures.append(
                    f"pair {pair} prompt {turn['prompt_index']} has no rate"
                )
                continue
            ratio = float(rate) / float(baseline)
            prompt_ratios.append(
                {
                    "prompt_index": turn["prompt_index"],
                    "control_tps": float(baseline),
                    "candidate_tps": float(rate),
                    "ratio": ratio,
                    "control_ttft_s": reference.get("ttft_s"),
                    "candidate_ttft_s": turn.get("ttft_s"),
                }
            )
            if per_prompt_minimum is None or ratio < per_prompt_minimum:
                per_prompt_minimum = ratio

        pair_records.append(
            {
                "pair": pair,
                "order": "AB" if pair % 2 else "BA",
                "launch_order": list(pair_order(pair)),
                "control_tps": control["sustained_tps"],
                "candidate_tps": candidate["sustained_tps"],
                "tps_ratio": tps_ratio,
                "control_median_ttft_s": control["median_ttft_s"],
                "candidate_median_ttft_s": candidate["median_ttft_s"],
                "ttft_ratio": ttft_ratio,
                "outputs_identical": outputs_identical,
                "control_output_sha256": control["output_sha256"],
                "candidate_output_sha256": candidate["output_sha256"],
                "control_decode_expert_read_bytes": control[
                    "decode_expert_read_bytes"
                ],
                "candidate_decode_expert_read_bytes": candidate[
                    "decode_expert_read_bytes"
                ],
                "control_decode_expert_misses": control["decode_expert_misses"],
                "candidate_decode_expert_misses": candidate[
                    "decode_expert_misses"
                ],
                "prompts": prompt_ratios,
            }
        )

    for arm, digests in arm_outputs.items():
        if len(digests) > 1:
            failures.append(f"{arm} outputs are not identical across pairs")

    tps_interval = paired_ratio_interval(tps_ratios, confidence)
    ttft_interval = paired_ratio_interval(ttft_ratios, confidence)
    complete = len(pair_records) == pairs
    tps_lower = tps_interval.get("lower")
    ttft_upper = ttft_interval.get("upper")
    gates = {
        "pairs_complete": complete,
        "outputs_identical": all(
            record["outputs_identical"] for record in pair_records
        )
        and complete,
        "no_per_prompt_slowdown": per_prompt_minimum is not None
        and per_prompt_minimum >= per_prompt_floor,
        "decode_lower_bound_above_one": tps_lower is not None and tps_lower > 1.0,
        "ttft_upper_bound_below_one": ttft_upper is not None and ttft_upper < 1.0,
        "no_structural_failure": not failures,
    }
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": utc_now(),
        "state_signature_sha256": state["signature"]["sha256"],
        "confidence": confidence,
        "per_prompt_floor": per_prompt_floor,
        "pairs": pair_records,
        "decode_tps_ratio": tps_interval,
        "ttft_ratio": ttft_interval,
        "minimum_per_prompt_ratio": per_prompt_minimum,
        "hardware_identity": hardware_reference,
        "hardware_volatile": {
            name: {"minimum": min(values), "maximum": max(values)}
            for name, values in volatile.items()
            if values
        },
        "gates": gates,
        "failures": failures,
        "decision": "promote" if all(gates.values()) else "hold",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument(
        "--prime-trials",
        type=int,
        default=0,
        help=(
            "unmeasured alternating trials run before pair 1 so the first "
            "recorded pair is not charged for cold caches; their artifacts "
            "are hashed and retained but excluded from the paired analysis"
        ),
    )
    parser.add_argument("--bind", type=Path, action="append", default=[])
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.0,
        help="idle interval between trials so machine state can settle",
    )
    parser.add_argument("--acceptance-key", default="acceptance.passed")
    parser.add_argument("--tps-key", default="summary.sustained_tps")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--per-prompt-floor", type=float, default=1.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--bind-engine-source",
        action="store_true",
        help="bind resume to the engine source fingerprint",
    )
    parser.add_argument(
        "--recover-only",
        action="store_true",
        help="mark stale attempts interrupted and exit without starting work",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="recompute the paired report from recorded trials only",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.prime_trials < 0:
        parser.error("--prime-trials must not be negative")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must lie in (0, 1)")
    for label in (args.control_label, args.candidate_label):
        if not LABEL_RE.fullmatch(label):
            parser.error("arm labels contain unsafe filename characters")
    if args.control_label == args.candidate_label:
        parser.error("arm labels must differ")
    if args.settle_seconds < 0.0:
        parser.error("--settle-seconds must not be negative")
    if not args.command:
        parser.error(
            "two command templates are required after --, separated by "
            f"{ARM_SEPARATOR}"
        )
    if args.command.count(ARM_SEPARATOR) != 1:
        parser.error(
            f"command templates must be separated by exactly one {ARM_SEPARATOR}"
        )
    split = args.command.index(ARM_SEPARATOR)
    args.commands = {
        "control": args.command[:split],
        "candidate": args.command[split + 1 :],
    }
    for arm, template in args.commands.items():
        if not template:
            parser.error(f"the {arm} command template is empty")
        try:
            command_for_trial(template, Path("probe"))
        except ValueError as error:
            parser.error(f"{arm} command: {error}")
    return args


def _publish_report(
    report: dict[str, Any], path: Path | None
) -> dict[str, Any]:
    if path is not None:
        atomic_json(path, report)
    return report


def _print_report(report: dict[str, Any], path: Path | None) -> None:
    payload = {
        "decision": report["decision"],
        "gates": report["gates"],
        "decode_tps_ratio": {
            key: report["decode_tps_ratio"].get(key)
            for key in ("geometric_mean", "lower", "upper", "count")
        },
        "ttft_ratio": {
            key: report["ttft_ratio"].get(key)
            for key in ("geometric_mean", "lower", "upper", "count")
        },
        "minimum_per_prompt_ratio": report["minimum_per_prompt_ratio"],
        "failures": report["failures"],
    }
    if path is not None:
        payload["report"] = str(path)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve(strict=True)
    state_path = args.state.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report.resolve() if args.report else None
    labels = {"control": args.control_label, "candidate": args.candidate_label}
    try:
        bindings = [binding_record(path) for path in args.bind]
        engine_source = (
            engine_source_sha256(root) if args.bind_engine_source else None
        )
    except (OSError, ValueError) as error:
        print(f"binding error: {error}", file=sys.stderr)
        return 2
    expected = signature(
        pairs=args.pairs,
        prime_trials=args.prime_trials,
        labels=labels,
        commands=args.commands,
        acceptance_key=args.acceptance_key,
        tps_key=args.tps_key,
        settle_seconds=args.settle_seconds,
        bindings=bindings,
        engine_source=engine_source,
    )
    try:
        state = load_or_create_state(state_path, expected, root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"paired trial state error: {error}", file=sys.stderr)
        return 2

    if args.recover_only:
        state["status"] = (
            "complete"
            if len(state.get("completed", {})) == expected_trials(args)
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

    if args.analyze_only:
        report = analyze(
            state,
            confidence=args.confidence,
            per_prompt_floor=args.per_prompt_floor,
        )
        _publish_report(report, report_path)
        _print_report(report, report_path)
        return 0 if report["decision"] == "promote" else 1

    schedule: list[dict[str, Any]] = []
    for index in range(1, args.prime_trials + 1):
        arm = prime_arm(index)
        schedule.append(
            {
                "key": f"prime:{index}:{arm}",
                "arm": arm,
                "prime": index,
                "pair": None,
                "order": None,
                "stem": f"{labels[arm]}-prime-{index:02d}",
                "tag": f"prime {index}/{args.prime_trials} {arm}",
            }
        )
    for pair in range(1, args.pairs + 1):
        order = "AB" if pair % 2 else "BA"
        for arm in pair_order(pair):
            schedule.append(
                {
                    "key": f"{pair}:{arm}",
                    "arm": arm,
                    "prime": None,
                    "pair": pair,
                    "order": order,
                    "stem": f"{labels[arm]}-pair-{pair:02d}",
                    "tag": f"pair {pair}/{args.pairs} {order} {arm}",
                }
            )

    executed = 0
    for entry in schedule:
        key, arm, stem, tag = (
            entry["key"],
            entry["arm"],
            entry["stem"],
            entry["tag"],
        )
        output = output_dir / f"{stem}.json"
        if verified_completed(state, key, output):
            print(f"[resume {tag}] verified {output}", flush=True)
            continue
        if executed and args.settle_seconds:
            time.sleep(args.settle_seconds)
        partial = output_dir / f".{stem}.partial.json"
        stdout_path = output_dir / f"{stem}.stdout.log"
        stderr_path = output_dir / f"{stem}.stderr.log"
        partial.unlink(missing_ok=True)
        command = command_for_trial(args.commands[arm], partial)
        attempt = {
            "attempt": len(state["attempts"]) + 1,
            "trial_key": key,
            "pair": entry["pair"],
            "prime": entry["prime"],
            "arm": arm,
            "order": entry["order"],
            "status": "running",
            "started_at": utc_now(),
            "command": command,
            "command_shell": shlex.join(command),
            "partial": str(partial),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        state["attempts"].append(attempt)
        state["status"] = "running"
        state["updated_at"] = utc_now()
        atomic_json(state_path, state)
        print(f"[{tag}] {attempt['command_shell']}", flush=True)
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
            attempt["reason"] = "paired controller received an interrupt"
            state["status"] = "interrupted"
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
            return 130
        executed += 1
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
            metrics = trial_metrics(result, args.tps_key)
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
            "pair": entry["pair"],
            "prime": entry["prime"],
            "arm": arm,
            "label": labels[arm],
            "output": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "completed_at": utc_now(),
            "metrics": metrics,
        }
        state["completed"][key] = record
        attempt["status"] = "complete"
        attempt["completed_at"] = record["completed_at"]
        attempt["output_sha256"] = record["sha256"]
        state["status"] = (
            "complete"
            if len(state["completed"]) == expected_trials(args)
            else "running"
        )
        state["updated_at"] = utc_now()
        atomic_json(state_path, state)
        print(
            f"[{tag}] complete {metrics['sustained_tps']:.9f} tok/s "
            f"{record['sha256']}",
            flush=True,
        )

    state["status"] = "complete"
    state["completed_at"] = utc_now()
    state["updated_at"] = state["completed_at"]
    report = analyze(
        state,
        confidence=args.confidence,
        per_prompt_floor=args.per_prompt_floor,
    )
    state["report"] = {
        "decision": report["decision"],
        "gates": report["gates"],
        "path": str(report_path) if report_path else None,
    }
    atomic_json(state_path, state)
    _publish_report(report, report_path)
    _print_report(report, report_path)
    return 0 if report["decision"] == "promote" else 1


if __name__ == "__main__":
    raise SystemExit(main())
