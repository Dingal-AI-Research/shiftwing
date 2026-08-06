#!/usr/bin/env python3
"""Join pinned model manifests and Gate 8/9 artifacts into one release audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from runtime_env import ENGINE_SOURCE_FILES, engine_source_sha256  # noqa: E402


MANIFEST_FIELDS = (
    "source",
    "source_fingerprint",
    "source_shards",
    "output_shards",
    "data_bytes",
    "tensor_count",
    "logical_tensor_count",
    "xbits",
    "io_bits",
    "shared_bits",
    "group_size",
    "include_mtp",
)

MODEL_REQUIREMENTS = {
    "ornith35": {
        "source": (
            "hf://deepreinforce-ai/Ornith-1.0-35B-FP8@"
            "1ab57ce0b44950e498a88756f40ad1ed4d0f30ca"
        ),
        "source_fingerprint": (
            "15281dc0352464f68ec93e805283a7c070aeb06f22622d975648381009cfe1d6"
        ),
        "source_shards": 16,
        "output_shards": 16,
        "tensor_count": 93277,
        "logical_tensor_count": 31333,
        "data_bytes": 19081810684,
        "xbits": "int4g128",
        "io_bits": 8,
        "shared_bits": 8,
        "group_size": 128,
        "include_mtp": False,
    },
    "ornith397": {
        "source": (
            "hf://deepreinforce-ai/Ornith-1.0-397B-FP8@"
            "8b61f97a8512d9d01bff1a9625c9a16730e115bb"
        ),
        "source_fingerprint": (
            "4b1c5ab0824e25b3768e3c6df8392394194fce86c4e9f4f3da24e3134ccf4c94"
        ),
        "source_shards": 122,
        "output_shards": 122,
        "tensor_count": 278152,
        "logical_tensor_count": 93078,
        "data_bytes": 212634789241,
        "xbits": "int4g128",
        "io_bits": 8,
        "shared_bits": 8,
        "group_size": 128,
        "include_mtp": False,
    },
}

Q3_REQUIREMENTS = {
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
Q3_ARTIFACT_FIELDS = (
    "format",
    "complete",
    "config_sha256",
    "signature",
    "layers",
    "file_count",
    "tensor_count",
    "data_bytes",
)
ORNITH397_INT4_PPL = 1.050067545
ORNITH397_Q3_MINIMUM_TPS = 0.70
ORNITH397_Q3_POLICY_BASIS = (
    "owner accepted Ornith397 routed-expert q3 without a perplexity gate and "
    "selected the measured 0.718432662 tok/s expanded-q4 profile"
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid or unavailable JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root is not an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish a report without exposing a truncated destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {name: value.get(name) for name in MANIFEST_FIELDS}


def _q3_identity(value: dict[str, Any], *, include_hash: str | None = None) -> dict[str, Any]:
    result = {name: value.get(name) for name in Q3_ARTIFACT_FIELDS}
    if include_hash is not None:
        result["manifest_sha256"] = include_hash
    return result


def _artifact_q3_matches(value: dict[str, Any], expected: dict[str, Any]) -> bool:
    observed = value.get("expert_lowbit_manifest")
    return isinstance(observed, dict) and all(
        observed.get(name) == expected.get(name) for name in Q3_ARTIFACT_FIELDS
    )


def _q3_doctor_accepted(
    value: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    return (
        value.get("schema_version") == 1
        and value.get("status") == "ok"
        and value.get("passed") is True
        and value.get("bits") == 3
        and value.get("manifest_sha256") == expected.get("manifest_sha256")
        and value.get("source_identity")
        == expected.get("signature", {}).get("source_identity")
        and value.get("layers") == expected.get("layers")
        and value.get("experts_per_layer") == 512
        and value.get("group_size")
        == expected.get("signature", {}).get("group_size")
        and value.get("iterations")
        == expected.get("signature", {}).get("iterations")
        and value.get("experts_per_file")
        == expected.get("signature", {}).get("experts_per_file")
        and value.get("file_count") == expected.get("file_count")
        and value.get("tensor_count") == expected.get("tensor_count")
        and value.get("data_bytes") == expected.get("data_bytes")
        and value.get("headers_verified") == expected.get("file_count")
        and value.get("hashes_verified") == expected.get("file_count")
    )


def _q3_pipeline_accepted(
    value: dict[str, Any],
    *,
    cdir: Path,
    base_manifest_sha256: str,
    q3_manifest_sha256: str,
) -> bool:
    steps = value.get("steps")
    required = {
        "q3_doctor": cdir / "ornith397_q3_doctor.json",
        "int4_reference": cdir / "ornith397_int4_prefix_reference.json",
        "q3_teacher_forced": cdir / "ornith397_q3_prefix_gate.json",
        "q3_coherence": cdir / "ornith397_q3_coherence.json",
        "q3_tools": cdir / "ornith397_q3_tool_gate.json",
        "q3_tier": cdir / "ornith397_q3_qualification.json",
    }
    step_bindings_pass = isinstance(steps, dict)
    if step_bindings_pass:
        for name, path in required.items():
            record = steps.get(name)
            if (
                not isinstance(record, dict)
                or record.get("status") != "passed"
                or record.get("artifact") != str(path.resolve())
                or not isinstance(record.get("argv"), list)
                or not path.is_file()
                or record.get("artifact_sha256") != _sha256(path)
            ):
                step_bindings_pass = False
                break
    baseline = cdir / "ornith397_ppl_smoke.json"
    engine = cdir / "qwen"
    recorded_engine_sha256 = value.get("engine_sha256")
    try:
        current_engine_source_sha256 = engine_source_sha256(cdir.parent)
    except OSError:
        return False
    return (
        value.get("schema_version") == 2
        and value.get("status") == "passed"
        and value.get("q3_ppl_waived") is True
        and value.get("q3_minimum_tps") == ORNITH397_Q3_MINIMUM_TPS
        and value.get("q3_policy_basis") == ORNITH397_Q3_POLICY_BASIS
        and value.get("coherence_review_acknowledged") is True
        and value.get("base_manifest_sha256") == base_manifest_sha256
        and value.get("q3_manifest_sha256") == q3_manifest_sha256
        and baseline.is_file()
        and value.get("int4_ppl_baseline_sha256") == _sha256(baseline)
        and engine.is_file()
        and isinstance(recorded_engine_sha256, str)
        and len(recorded_engine_sha256) == 64
        and value.get("engine_source_sha256")
        == current_engine_source_sha256
        and step_bindings_pass
    )


def _nested_true(value: dict[str, Any], *path: str) -> bool:
    current: Any = value
    for name in path:
        if not isinstance(current, dict):
            return False
        current = current.get(name)
    return current is True


def _finite_at_least(value: Any, minimum: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= minimum


def _finite_at_most(value: Any, maximum: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number <= maximum


def _integer_equals(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _doctor_accepted(
    value: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    checks = value.get("checks")
    plan = value.get("plan")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "ok"
        or not isinstance(checks, list)
        or not isinstance(plan, dict)
    ):
        return False
    by_id = {
        item.get("id"): item
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    required_passes = {
        "model.path",
        "model.config",
        "model.tokenizer",
        "model.container",
        "model.quantization",
        "model.ledger",
        "engine.binary",
        "accelerator.cuda",
        "runtime.state",
        "runtime.resources",
    }
    if any((by_id.get(name) or {}).get("status") != "pass" for name in required_passes):
        return False
    container = (by_id["model.container"].get("details") or {})
    ledger = (by_id["model.ledger"].get("details") or {})
    resources = (by_id["runtime.resources"].get("details") or {})
    preregistered = ledger.get("preregistered_expectations") or {}
    expected_preregistered = {
        "source": expected["source"],
        "source_fingerprint": expected["source_fingerprint"],
        "source_shards": expected["source_shards"],
        "output_shards": expected["output_shards"],
        "logical_tensors": expected["logical_tensor_count"],
        "physical_tensors": expected["tensor_count"],
        "data_bytes": expected["data_bytes"],
    }
    return (
        container.get("conversion_complete") is True
        and _integer_equals(container.get("shards"), expected["output_shards"])
        and _integer_equals(
            container.get("parsed_shards"),
            expected["output_shards"],
        )
        and _integer_equals(
            container.get("header_tensors"),
            expected["tensor_count"],
        )
        and _integer_equals(
            container.get("tensor_bytes"),
            expected["data_bytes"],
        )
        and ledger.get("source") == expected["source"]
        and ledger.get("source_fingerprint") == expected["source_fingerprint"]
        and _integer_equals(
            ledger.get("committed_sources"),
            expected["source_shards"],
        )
        and _integer_equals(
            ledger.get("output_shards"),
            expected["output_shards"],
        )
        and _integer_equals(
            ledger.get("logical_tensors"),
            expected["logical_tensor_count"],
        )
        and _integer_equals(
            ledger.get("physical_tensors"),
            expected["tensor_count"],
        )
        and _integer_equals(ledger.get("data_bytes"), expected["data_bytes"])
        and ledger.get("hashes_requested") is True
        and _integer_equals(
            ledger.get("hashes_checked"),
            expected["output_shards"],
        )
        and preregistered == expected_preregistered
        and isinstance(resources.get("checks"), dict)
        and all(
            resources["checks"].get(name) is True
            for name in ("disk", "ram", "vram")
        )
        and _nested_true(plan, "checks", "disk")
        and _nested_true(plan, "checks", "ram")
        and _nested_true(plan, "checks", "vram")
        and plan.get("safe") is True
    )


def _prefix_accepted(value: dict[str, Any]) -> bool:
    rows = value.get("rows")
    if (
        not _nested_true(value, "acceptance", "passed")
        or not _integer_equals(value.get("prompts"), 20)
        or not isinstance(rows, list)
        or len(rows) != 20
    ):
        return False
    matching = 0
    compared = 0
    for row in rows:
        if not isinstance(row, dict):
            return False
        row_matching = row.get("tf_matches")
        row_compared = row.get("tf_compared")
        if (
            isinstance(row_matching, bool)
            or not isinstance(row_matching, int)
            or isinstance(row_compared, bool)
            or not isinstance(row_compared, int)
            or row_compared != 64
            or row_matching < 0
            or row_matching > row_compared
        ):
            return False
        agreement = row_matching / row_compared
        try:
            reported_agreement = float(row.get("tf_agreement"))
        except (TypeError, ValueError):
            return False
        if agreement < 0.85 or not math.isclose(
            reported_agreement, agreement, rel_tol=0, abs_tol=1e-12
        ):
            return False
        matching += row_matching
        compared += row_compared
    aggregate = matching / compared
    try:
        reported_aggregate = float(value.get("tf_agreement"))
    except (TypeError, ValueError):
        return False
    return (
        aggregate >= 0.90
        and _integer_equals(value.get("tf_matching_tokens"), matching)
        and _integer_equals(value.get("tf_compared_tokens"), compared)
        and _integer_equals(value.get("tf_prompts_at_85_percent"), 20)
        and math.isclose(
            reported_aggregate, aggregate, rel_tol=0, abs_tol=1e-12
        )
    )


def _ppl_accepted(
    value: dict[str, Any],
    *,
    maximum_ppl: float,
    maximum_relative_delta: float | None = None,
) -> bool:
    if not _nested_true(value, "acceptance", "passed"):
        return False
    try:
        ppl = float(value.get("ppl"))
    except (TypeError, ValueError):
        return False
    if (
        not math.isfinite(ppl)
        or ppl <= 0
        or ppl > maximum_ppl
        or not _finite_at_least(value.get("tokens"), 1)
    ):
        return False
    if maximum_relative_delta is None:
        return True
    try:
        reference = float(value.get("llama_ppl"))
        reported = float(value.get("relative_ppl_delta"))
    except (TypeError, ValueError):
        return False
    if not math.isfinite(reference) or reference <= 0 or not math.isfinite(reported):
        return False
    recomputed = (ppl - reference) / reference
    return abs(recomputed) <= maximum_relative_delta and math.isclose(
        reported,
        recomputed,
        rel_tol=0,
        abs_tol=1e-12,
    )


def _tool_accepted(value: dict[str, Any]) -> bool:
    tool_response = value.get("tool_call_response") or {}
    final_response = value.get("final_response") or {}
    tool_choices = tool_response.get("choices")
    final_choices = final_response.get("choices")
    if (
        value.get("passed") is not True
        or not isinstance(tool_choices, list)
        or len(tool_choices) != 1
        or not isinstance(final_choices, list)
        or len(final_choices) != 1
    ):
        return False
    tool_choice = tool_choices[0]
    final_choice = final_choices[0]
    if not isinstance(tool_choice, dict) or not isinstance(final_choice, dict):
        return False
    tool_message = tool_choice.get("message")
    final_message = final_choice.get("message")
    if not isinstance(tool_message, dict) or not isinstance(final_message, dict):
        return False
    tool_calls = tool_message.get("tool_calls")
    if (
        tool_choice.get("finish_reason") != "tool_calls"
        or not isinstance(tool_calls, list)
        or len(tool_calls) != 1
        or final_choice.get("finish_reason") != "stop"
        or final_message.get("tool_calls") not in (None, [])
    ):
        return False
    if not isinstance(tool_calls[0], dict):
        return False
    function = tool_calls[0].get("function") or {}
    if not isinstance(function, dict):
        return False
    try:
        arguments = json.loads(function.get("arguments"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    content = final_message.get("content")
    resident = ((value.get("profile") or {}).get("resident") or {})
    return (
        function.get("name") == "get_weather"
        and arguments == {"city": "Paris"}
        and isinstance(content, str)
        and bool(content.strip())
        and "18" in content
        and _finite_at_least(
            (tool_response.get("usage") or {}).get("completion_tokens"),
            1,
        )
        and _finite_at_least(
            (final_response.get("usage") or {}).get("completion_tokens"),
            1,
        )
        and _finite_at_least(resident.get("device_moe"), 1)
        and _integer_equals(resident.get("host_moe"), 0)
        and _finite_at_least(resident.get("router_d2h_bytes"), 1)
        and _finite_at_least(resident.get("logits_d2h_bytes"), 1)
    )


def _tier_accepted(
    value: dict[str, Any],
    *,
    q3_expected: dict[str, Any] | None = None,
    warmup_passes: int = 2,
    minimum_tps: float = 0.85,
) -> bool:
    summary = value.get("summary") or {}
    configuration = value.get("configuration") or {}
    tiers = value.get("tiers") or {}
    emap = value.get("emap") or {}
    telemetry = value.get("telemetry_summary") or {}
    resident = value.get("resident") or {}
    hardware = value.get("hardware") or {}
    measured = value.get("measured")
    if not all(
        isinstance(item, dict)
        for item in (
            summary,
            configuration,
            tiers,
            emap,
            telemetry,
            resident,
            hardware,
        )
    ):
        return False
    if (
        not _nested_true(value, "acceptance", "passed")
        or not _nested_true(value, "summary", "automatic_gate_pass")
        or not _nested_true(value, "summary", "outputs_nonempty")
        or not _nested_true(value, "summary", "telemetry_complete")
        or not _nested_true(value, "summary", "cuda_active")
        or not _nested_true(value, "summary", "resident_cuda_graph")
        or not _nested_true(value, "summary", "throughput_pass")
        or value.get("model_family") != "ornith-1.0"
        or (
            value.get("expert_lowbit_manifest") is not None
            if q3_expected is None
            else not _artifact_q3_matches(value, q3_expected)
        )
        or value.get("telemetry_error") is not None
        or not isinstance(measured, list)
        or len(measured) != 4
    ):
        return False
    expected_configuration = {
        "warmup_passes": warmup_passes,
        "measured_passes": 1,
        "max_tokens": 64,
        "context": 4096,
        "expert_ram_gb": 18.0,
        "cuda_expert_gb": 6.0,
        "ram_headroom_gb": 1.0,
        "cuda_headroom_gb": 1.0,
        "minimum_tps": minimum_tps,
        "uring_persist": True,
        "pinned_upload": True,
        "decode_protect": True,
        "decode_protect_prewarm": True,
        "predictive_prefetch": False,
    }
    if any(configuration.get(name) != expected for name, expected in expected_configuration.items()):
        return False
    if q3_expected is not None and (
        configuration.get("expert_q3") is not True
        or configuration.get("expert_q2") is not False
        or configuration.get("q3_native") is not False
    ):
        return False
    total_tokens = 0
    total_decode_s = 0.0
    for item in measured:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            return False
        stats = item.get("stats")
        if not isinstance(stats, dict):
            return False
        tokens = stats.get("completion_tokens")
        try:
            rate = float(stats.get("tokens_per_second"))
        except (TypeError, ValueError):
            return False
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
            or not math.isfinite(rate)
            or rate <= 0
        ):
            return False
        total_tokens += tokens
        total_decode_s += tokens / rate
    sustained = total_tokens / total_decode_s
    try:
        reported_sustained = float(summary.get("sustained_tps"))
    except (TypeError, ValueError):
        return False
    if (
        sustained < minimum_tps
        or not math.isclose(
            reported_sustained,
            sustained,
            rel_tol=0,
            abs_tol=1e-9,
        )
        or not _finite_at_least(tiers.get("vram"), 1)
        or not _finite_at_least(tiers.get("vram_gb"), 0.001)
        or not _finite_at_least(hardware.get("gpus"), 1)
        or not _finite_at_least(hardware.get("vram_total_gb"), 1)
        or not _integer_equals(emap.get("rows"), 60)
        or not _integer_equals(emap.get("cols"), 512)
        or not _integer_equals(telemetry.get("rows"), 60)
        or not _integer_equals(telemetry.get("cols"), 512)
        or not _integer_equals(telemetry.get("experts"), 60 * 512)
    ):
        return False
    tier_counts = telemetry.get("tier_counts")
    if not isinstance(tier_counts, dict):
        return False
    for name in ("disk", "ram", "vram"):
        if not _integer_equals(tier_counts.get(name), tiers.get(name)):
            return False
    return (
        sum(int(tiers.get(name, -1)) for name in ("disk", "ram", "vram"))
        == 60 * 512
        and _integer_equals(resident.get("layers"), resident.get("device_moe"))
        and _finite_at_least(resident.get("device_moe"), 1)
        and _integer_equals(resident.get("host_moe"), 0)
        and _finite_at_least(resident.get("activation_h2d_bytes"), 1)
        and _integer_equals(
            resident.get("activation_d2h_bytes"),
            resident.get("activation_h2d_bytes"),
        )
        and _finite_at_least(resident.get("router_d2h_bytes"), 1)
        and _finite_at_least(resident.get("logits_d2h_bytes"), 1)
    )


def _batch_accepted(value: dict[str, Any]) -> bool:
    speedups = value.get("speedups") or {}
    acceptance = value.get("acceptance") or {}
    prompt = value.get("prompt") or {}
    manifest = value.get("model_manifest") or {}
    if not all(isinstance(item, dict) for item in (speedups, acceptance, prompt, manifest)):
        return False
    try:
        ab = float(speedups.get("ab"))
        ba = float(speedups.get("ba"))
        minimum = float(speedups.get("minimum"))
        geomean = float(speedups.get("geometric_mean"))
    except (TypeError, ValueError):
        return False
    recomputed_geomean = math.sqrt(ab * ba) if ab > 0 and ba > 0 else math.nan
    return (
        acceptance.get("passed") is True
        and value.get("model_family") == "ornith-1.0"
        and acceptance.get("expected_family") == "ornith-1.0"
        and acceptance.get("expected_source") == manifest.get("source")
        and acceptance.get("minimum_each") == 0.95
        and acceptance.get("minimum_geometric_mean") == 1.0
        and prompt.get("raw") is False
        and prompt.get("user_text") == "Reply with exactly: colib batch ready"
        and prompt.get("expected_exact_content") == "colib batch ready"
        and math.isfinite(ab)
        and math.isfinite(ba)
        and min(ab, ba) >= 0.95
        and recomputed_geomean >= 1.0
        and math.isclose(minimum, min(ab, ba), rel_tol=0, abs_tol=1e-12)
        and math.isclose(
            geomean,
            recomputed_geomean,
            rel_tol=0,
            abs_tol=1e-12,
        )
    )


def _cancel_accepted(
    value: dict[str, Any],
    limit: float,
    *,
    expected_ram_gb: float,
    expected_cuda_expert_gb: float,
    expected_expert_lowbit: str | None = None,
) -> bool:
    trials = value.get("trials")
    summary = value.get("summary") or {}
    acceptance = value.get("acceptance") or {}
    prompt = value.get("prompt") or {}
    tier_configuration = value.get("tier_configuration") or {}
    tiers = value.get("tiers") or {}
    hardware = value.get("hardware") or {}
    resident = value.get("resident") or {}
    if not all(
        isinstance(item, dict)
        for item in (
            summary,
            acceptance,
            prompt,
            tier_configuration,
            tiers,
            hardware,
            resident,
        )
    ):
        return False
    samples = summary.get("cancel_ack_samples_s")
    valid_samples = (
        isinstance(samples, list)
        and len(samples) == 20
        and all(_finite_at_least(sample, 0) for sample in samples)
    )
    recomputed_p95 = None
    if valid_samples:
        ordered = sorted(float(sample) for sample in samples)
        recomputed_p95 = ordered[18]

    def valid_trial(trial: Any) -> bool:
        if not isinstance(trial, dict):
            return False
        cancelled = trial.get("cancelled")
        peer = trial.get("peer")
        slot_reuse = trial.get("slot_reuse")
        if not all(
            isinstance(item, dict)
            for item in (cancelled, peer, slot_reuse)
        ):
            return False
        peer_stats = peer.get("stats")
        return (
            isinstance(peer_stats, dict)
            and cancelled.get("pieces_before_cancel") == 1
            and _finite_at_least(cancelled.get("cancel_ack_s"), 0)
            and bool(peer.get("text"))
            and _finite_at_least(peer_stats.get("completion_tokens"), 1)
            and bool(slot_reuse.get("text"))
        )

    lifecycle_ok = (
        isinstance(trials, list)
        and len(trials) == 20
        and all(valid_trial(trial) for trial in trials)
    )
    trial_samples_match = (
        valid_samples
        and lifecycle_ok
        and all(
            math.isclose(
                float(sample),
                float(trial["cancelled"]["cancel_ack_s"]),
                rel_tol=0,
                abs_tol=1e-12,
            )
            for sample, trial in zip(samples, trials)
        )
    )
    expected_tier_configuration = {
        "warmup_passes": 2,
        "trials": 20,
        "context": 4096,
        "ram_gb": expected_ram_gb,
        "ram_headroom_gb": 1.0,
        "cuda_expert_gb": expected_cuda_expert_gb,
        "cuda_headroom_gb": 1.0,
        "tiered_io": True,
        "predictive_prefetch": False,
        "uring_persist": False,
        "pinned_upload": False,
        "decode_protect": False,
        "decode_protect_prewarm": False,
        "expert_lowbit": expected_expert_lowbit,
    }
    return (
        _nested_true(value, "acceptance", "passed")
        and value.get("model_family") == "ornith-1.0"
        and value.get("cuda") is True
        and prompt.get("user_text") == "Reply briefly: cancellation probe"
        and prompt.get("raw") is False
        and all(
            tier_configuration.get(name) == expected
            for name, expected in expected_tier_configuration.items()
        )
        and acceptance.get("maximum_cancel_ack_s") == limit
        and acceptance.get("measured_statistic") == "p95_cancel_ack_s"
        and lifecycle_ok
        and valid_samples
        and trial_samples_match
        and summary.get("p95_method") == "nearest-rank"
        and _finite_at_most(summary.get("p95_cancel_ack_s"), limit)
        and recomputed_p95 is not None
        and math.isclose(
            float(summary.get("p95_cancel_ack_s")),
            recomputed_p95,
            rel_tol=0,
            abs_tol=1e-9,
        )
        and _finite_at_least(tiers.get("vram"), 1)
        and _finite_at_least(hardware.get("gpus"), 1)
        and _finite_at_least(hardware.get("vram_total_gb"), 1)
        and _integer_equals(resident.get("layers"), resident.get("device_moe"))
        and _finite_at_least(resident.get("device_moe"), 1)
        and _integer_equals(resident.get("host_moe"), 0)
        and _finite_at_least(resident.get("router_d2h_bytes"), 1)
        and _finite_at_least(resident.get("logits_d2h_bytes"), 1)
    )


def _web_accepted(
    value: dict[str, Any],
    *,
    expected_ram_gb: float,
    expected_cuda_expert_gb: float,
    expected_expert_lowbit: str | None = None,
) -> bool:
    requests = value.get("requests")
    resident = ((value.get("profile") or {}).get("resident") or {})
    final_health = value.get("final_health") or {}
    scheduler = final_health.get("scheduler") or {}
    hwinfo = final_health.get("hwinfo") or {}
    tiers = final_health.get("tiers") or {}
    configuration = value.get("configuration") or {}
    if not all(
        isinstance(item, dict)
        for item in (
            resident,
            final_health,
            scheduler,
            hwinfo,
            tiers,
            configuration,
        )
    ):
        return False
    expected_configuration = {
        "expected_exact_content": "colib ready",
        "requests": 2,
        "kv_slots": 2,
        "max_tokens": 16,
        "context": 4096,
        "ram_gb": expected_ram_gb,
        "ram_headroom_gb": 1.0,
        "cuda_expert_gb": expected_cuda_expert_gb,
        "cuda_headroom_gb": 1.0,
        "predictive_prefetch": False,
        "uring_persist": False,
        "pinned_upload": False,
        "decode_protect": False,
        "decode_protect_prewarm": False,
        "expert_lowbit": expected_expert_lowbit,
    }
    return (
        _nested_true(value, "acceptance", "passed")
        and value.get("model_family") == "ornith-1.0"
        and value.get("cuda") is True
        and value.get("web_bundle") is True
        and all(
            configuration.get(name) == expected
            for name, expected in expected_configuration.items()
        )
        and isinstance(requests, list)
        and len(requests) == 2
        and all(
            isinstance(request, dict)
            and str(request.get("content", "")).strip() == "colib ready"
            and _finite_at_least(
                (request.get("usage") or {}).get("completion_tokens"),
                1,
            )
            for request in requests
        )
        and _finite_at_least(resident.get("device_moe"), 1)
        and _integer_equals(resident.get("layers"), resident.get("device_moe"))
        and _integer_equals(resident.get("host_moe"), 0)
        and _finite_at_least(resident.get("router_d2h_bytes"), 1)
        and _finite_at_least(resident.get("logits_d2h_bytes"), 1)
        and _integer_equals(scheduler.get("active"), 0)
        and _integer_equals(scheduler.get("queued"), 0)
        and _finite_at_least(scheduler.get("completed"), 2)
        and _finite_at_least(hwinfo.get("gpus"), 1)
        and _finite_at_least(hwinfo.get("vram_total_gb"), 1)
        and _finite_at_least(tiers.get("vram"), 1)
    )


def audit_release(
    root: Path,
    *,
    ornith397_expert_q3: bool = False,
) -> dict[str, Any]:
    """Return all release-evidence checks without mutating the workspace."""
    root = root.resolve()
    cdir = root / "c"
    checks: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    q3_identity: dict[str, Any] | None = None

    def record(identifier: str, passed: bool, summary: str, **details: Any) -> None:
        item = {
            "id": identifier,
            "passed": bool(passed),
            "summary": summary,
        }
        if details:
            item["details"] = details
        checks.append(item)

    for model, expected in MODEL_REQUIREMENTS.items():
        path = cdir / model / "quantization.json"
        try:
            raw = _read_object(path)
        except ValueError as error:
            record(f"{model}.manifest", False, str(error))
            continue
        identity = _manifest_identity(raw)
        errors = []
        if raw.get("complete") is not True:
            errors.append("complete is not true")
        for name, required in expected.items():
            if raw.get(name) != required:
                errors.append(f"{name} {raw.get(name)!r} != {required!r}")
        manifests[model] = identity
        record(
            f"{model}.manifest",
            not errors,
            "complete pinned model manifest matches release identity"
            if not errors
            else f"manifest identity failed ({len(errors)} issue(s))",
            path=str(path),
            errors=errors,
            identity=identity,
        )

    if ornith397_expert_q3:
        q3_path = cdir / "ornith397" / "expert-q3.json"
        try:
            q3_raw = _read_object(q3_path)
        except ValueError as error:
            record("ornith397.q3_manifest", False, str(error))
        else:
            q3_errors = [
                f"{name} {q3_raw.get(name)!r} != {required!r}"
                for name, required in Q3_REQUIREMENTS.items()
                if q3_raw.get(name) != required
            ]
            if not isinstance(q3_raw.get("data_bytes"), int) or q3_raw.get("data_bytes", 0) <= 0:
                q3_errors.append("data_bytes is not positive")
            files = q3_raw.get("files")
            if not isinstance(files, list) or len(files) != Q3_REQUIREMENTS["file_count"]:
                q3_errors.append("files does not contain 480 records")
            q3_identity = _q3_identity(
                q3_raw,
                include_hash=_sha256(q3_path),
            )
            record(
                "ornith397.q3_manifest",
                not q3_errors,
                (
                    "complete pinned q3 sidecar manifest matches release identity"
                    if not q3_errors
                    else f"q3 manifest identity failed ({len(q3_errors)} issue(s))"
                ),
                path=str(q3_path),
                errors=q3_errors,
                identity=q3_identity,
            )

    artifact_specs: list[
        tuple[str, str, str, Callable[[dict[str, Any]], bool], str]
    ] = [
        (
            "ornith35.prefix",
            "ornith35",
            "ornith35_prefix_gate.json",
            _prefix_accepted,
            "teacher-forced numerical gate",
        ),
        (
            "ornith35.ppl",
            "ornith35",
            "ornith35_ppl_gate.json",
            lambda value: _ppl_accepted(
                value,
                maximum_ppl=50.0,
                maximum_relative_delta=0.05,
            ),
            "perplexity gate",
        ),
        (
            "ornith35.tools",
            "ornith35",
            "ornith35_tool_gate.json",
            _tool_accepted,
            "generated HTTP tool gate",
        ),
    ]
    if ornith397_expert_q3:
        artifact_specs.extend(
            [
                (
                    "ornith397.prefix",
                    "ornith397",
                    "ornith397_q3_prefix_gate.json",
                    _prefix_accepted,
                    "q3 teacher-forced numerical gate",
                ),
                (
                    "ornith397.coherence",
                    "ornith397",
                    "ornith397_q3_coherence.json",
                    lambda value: _tier_accepted(
                        value,
                        q3_expected=q3_identity,
                        warmup_passes=0,
                        minimum_tps=0.000001,
                    ),
                    "q3 four-output coherence/CUDA gate",
                ),
                (
                    "ornith397.tier",
                    "ornith397",
                    "ornith397_q3_qualification.json",
                    lambda value: _tier_accepted(
                        value,
                        q3_expected=q3_identity,
                        minimum_tps=ORNITH397_Q3_MINIMUM_TPS,
                    ),
                    "q3 tier/coherence/throughput gate",
                ),
                (
                    "ornith397.tools",
                    "ornith397",
                    "ornith397_q3_tool_gate.json",
                    _tool_accepted,
                    "q3 generated HTTP tool gate",
                ),
            ]
        )
    else:
        artifact_specs.extend(
            [
                (
                    "ornith397.ppl",
                    "ornith397",
                    "ornith397_ppl_smoke.json",
                    lambda value: _ppl_accepted(value, maximum_ppl=50.0),
                    "perplexity corruption smoke",
                ),
                (
                    "ornith397.tier",
                    "ornith397",
                    "ornith397_qualification.json",
                    _tier_accepted,
                    "direct-profile tier/coherence/throughput gate",
                ),
                (
                    "ornith397.tools",
                    "ornith397",
                    "ornith397_tool_qualification.json",
                    _tool_accepted,
                    "generated HTTP tool gate",
                ),
            ]
        )

    doctor_path = cdir / "ornith397_doctor.json"
    try:
        doctor = _read_object(doctor_path)
    except ValueError as error:
        record("ornith397.doctor", False, str(error))
    else:
        doctor_passed = _doctor_accepted(
            doctor,
            MODEL_REQUIREMENTS["ornith397"],
        )
        record(
            "ornith397.doctor",
            doctor_passed,
            (
                "independent container/header/ledger/hash/resource audit passed"
                if doctor_passed
                else "independent structural doctor evidence is false or incomplete"
            ),
            path=str(doctor_path),
        )

    if ornith397_expert_q3:
        q3_doctor_path = cdir / "ornith397_q3_doctor.json"
        try:
            q3_doctor = _read_object(q3_doctor_path)
        except ValueError as error:
            record("ornith397.q3_doctor", False, str(error))
        else:
            q3_doctor_passed = (
                q3_identity is not None
                and _q3_doctor_accepted(q3_doctor, q3_identity)
            )
            record(
                "ornith397.q3_doctor",
                q3_doctor_passed,
                (
                    "independent q3 inventory/header/hash audit passed"
                    if q3_doctor_passed
                    else "independent q3 audit evidence is false or incomplete"
                ),
                path=str(q3_doctor_path),
            )

        pipeline_path = cdir / "bench" / "ornith397_q3_gate8_pipeline.json"
        try:
            q3_pipeline = _read_object(pipeline_path)
        except ValueError as error:
            record("ornith397.coherence_review", False, str(error))
        else:
            ppl_waiver_passed = (
                q3_pipeline.get("schema_version") == 2
                and q3_pipeline.get("q3_ppl_waived") is True
                and q3_pipeline.get("q3_minimum_tps")
                == ORNITH397_Q3_MINIMUM_TPS
                and q3_pipeline.get("q3_policy_basis")
                == ORNITH397_Q3_POLICY_BASIS
                and "q3_ppl" not in (q3_pipeline.get("steps") or {})
            )
            record(
                "ornith397.ppl",
                ppl_waiver_passed,
                (
                    "owner-approved q3 perplexity waiver is explicit and bound"
                    if ppl_waiver_passed
                    else "q3 perplexity waiver is missing, stale, or ambiguous"
                ),
                path=str(pipeline_path),
                engine_sha256=q3_pipeline.get("engine_sha256"),
                engine_source_sha256=q3_pipeline.get("engine_source_sha256"),
            )
            try:
                base_hash = _sha256(cdir / "ornith397" / "quantization.json")
                q3_hash = _sha256(cdir / "ornith397" / "expert-q3.json")
            except OSError:
                pipeline_passed = False
            else:
                pipeline_passed = _q3_pipeline_accepted(
                    q3_pipeline,
                    cdir=cdir,
                    base_manifest_sha256=base_hash,
                    q3_manifest_sha256=q3_hash,
                )
            record(
                "ornith397.coherence_review",
                pipeline_passed,
                (
                    "q3 coherence was explicitly reviewed in the bound pipeline"
                    if pipeline_passed
                    else "q3 coherence review acknowledgement is missing or unbound"
                ),
                path=str(pipeline_path),
                engine_sha256=q3_pipeline.get("engine_sha256"),
                engine_source_sha256=q3_pipeline.get("engine_source_sha256"),
            )

    for model in ("ornith35", "ornith397"):
        artifact_specs.extend(
            [
                (
                    f"{model}.batch",
                    model,
                    f"{model}_batch_abba.json",
                    _batch_accepted,
                    "reversed-order continuous-batch gate",
                ),
                (
                    f"{model}.cancel",
                    model,
                    f"{model}_cancel_gate.json",
                    (
                        (
                            lambda value: _cancel_accepted(
                                value,
                                1.0,
                                expected_ram_gb=8.0,
                                expected_cuda_expert_gb=6.0,
                            )
                        )
                        if model == "ornith35"
                        else (
                            lambda value: _cancel_accepted(
                                value,
                                3.0,
                                expected_ram_gb=18.0,
                                expected_cuda_expert_gb=5.0,
                                expected_expert_lowbit=(
                                    "int3g128" if ornith397_expert_q3 else None
                                ),
                            )
                        )
                    ),
                    "20-trial cancellation p95 gate",
                ),
                (
                    f"{model}.web",
                    model,
                    f"{model}_web_gate.json",
                    (
                        (
                            lambda value: _web_accepted(
                                value,
                                expected_ram_gb=8.0,
                                expected_cuda_expert_gb=6.0,
                            )
                        )
                        if model == "ornith35"
                        else (
                            lambda value: _web_accepted(
                                value,
                                expected_ram_gb=18.0,
                                expected_cuda_expert_gb=5.0,
                                expected_expert_lowbit=(
                                    "int3g128" if ornith397_expert_q3 else None
                                ),
                            )
                        )
                    ),
                    "exact web/gateway/CUDA streaming gate",
                ),
            ]
        )

    for identifier, model, filename, predicate, description in artifact_specs:
        path = cdir / filename
        try:
            artifact = _read_object(path)
        except ValueError as error:
            record(identifier, False, str(error))
            continue
        expected_manifest = manifests.get(model)
        observed_manifest = artifact.get("model_manifest")
        manifest_matches = (
            expected_manifest is not None
            and isinstance(observed_manifest, dict)
            and _manifest_identity(observed_manifest) == expected_manifest
        )
        if ornith397_expert_q3 and model == "ornith397":
            lowbit_matches = (
                q3_identity is not None
                and _artifact_q3_matches(artifact, q3_identity)
            )
        else:
            lowbit_matches = artifact.get("expert_lowbit_manifest") is None
        accepted = predicate(artifact)
        passed = manifest_matches and lowbit_matches and accepted
        errors = []
        if not manifest_matches:
            errors.append("artifact model_manifest does not match the container")
        if not lowbit_matches:
            errors.append("artifact expert_lowbit_manifest does not match the selected profile")
        if not accepted:
            errors.append(f"{description} acceptance is false or incomplete")
        record(
            identifier,
            passed,
            f"{description} is accepted and manifest-bound"
            if passed
            else f"{description} failed release audit",
            path=str(path),
            errors=errors,
        )

    failures = [item["id"] for item in checks if not item["passed"]]
    return {
        "schema_version": 1,
        "root": str(root),
        "checks": checks,
        "summary": {
            "passed": not failures,
            "check_count": len(checks),
            "failure_count": len(failures),
            "failures": failures,
            "requires_make_check": True,
            "requires_clean_release_commit_and_v0_1_tag": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ornith397-expert-q3",
        action="store_true",
        help="audit the owner-authorized Ornith397 q3 Gate-8/9 evidence profile",
    )
    args = parser.parse_args()
    result = audit_release(
        args.root,
        ornith397_expert_q3=args.ornith397_expert_q3,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        _atomic_write_text(args.output, rendered)
    print(rendered, end="")
    return 0 if result["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
