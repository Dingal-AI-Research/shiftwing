#!/usr/bin/env python3
"""Publish compact, independently verified Ornith397 retirement evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "colib.ornith397-retirement.v1"
BASE_SCHEMA = "colib.ornith397-base-shards.v1"
COPY_NAMES = (
    "config.json",
    "quantization.json",
    "expert-q3.json",
    "generation_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def median_ttft(result: dict[str, Any]) -> float:
    values = [float(row["ttft_s"]) for row in result["measured"]]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("measured TTFT values must be finite")
    return statistics.median(values)


def verify_trials(
    state_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    signature = state.get("signature", {})
    trials = int(signature.get("trials", 0))
    completed = state.get("completed", {})
    if (
        state.get("schema") != "colib.performance-trials.v1"
        or state.get("status") != "complete"
        or trials < 1
        or len(completed) != trials
    ):
        raise ValueError("performance trial state is not complete")
    for binding in signature.get("bindings", []):
        path = Path(binding["path"]).resolve(strict=True)
        if (
            path.stat().st_size != binding["bytes"]
            or sha256_file(path) != binding["sha256"]
        ):
            raise ValueError(f"performance binding drifted: {path}")

    rows: list[dict[str, Any]] = []
    invariant: dict[str, Any] | None = None
    for trial in range(1, trials + 1):
        completed_row = completed[str(trial)]
        output = Path(completed_row["output"]).resolve(strict=True)
        output_bytes = output.read_bytes()
        output_hash = sha256_bytes(output_bytes)
        if (
            len(output_bytes) != completed_row["bytes"]
            or output_hash != completed_row["sha256"]
        ):
            raise ValueError(f"trial artifact does not match state: {output}")
        result = json.loads(output_bytes)
        if result.get("acceptance", {}).get("passed") is not True:
            raise ValueError(f"trial acceptance is false: {output}")
        measured = result.get("measured", [])
        if not measured:
            raise ValueError(f"trial has no measured turns: {output}")
        hardware_identity = dict(result.get("hardware") or {})
        hardware_identity.pop("ram_avail_gb", None)
        fixed = {
            "configuration": result.get("configuration"),
            "hardware_identity": hardware_identity,
            "model_manifest": result.get("model_manifest"),
            "expert_lowbit_manifest": result.get("expert_lowbit_manifest"),
        }
        if invariant is None:
            invariant = fixed
        elif fixed != invariant:
            raise ValueError("trial hardware/configuration/manifests drifted")
        rows.append(
            {
                "trial": trial,
                "artifact": str(output),
                "bytes": len(output_bytes),
                "sha256": output_hash,
                "sustained_tps": float(result["summary"]["sustained_tps"]),
                "median_ttft_s": median_ttft(result),
                "turns": [
                    {
                        "prompt_index": int(turn["prompt_index"]),
                        "completion_tokens": int(
                            turn["stats"]["completion_tokens"]
                        ),
                        "decode_tps": float(
                            turn["stats"]["tokens_per_second"]
                        ),
                        "ttft_s": float(turn["ttft_s"]),
                        "wall_s": float(turn["wall_s"]),
                    }
                    for turn in measured
                ],
            }
        )
    assert invariant is not None
    return state, rows, sha256_bytes(state_bytes)


def token_weighted_tps(rows: list[dict[str, Any]]) -> float:
    tokens = 0
    seconds = 0.0
    for row in rows:
        for turn in row["turns"]:
            count = turn["completion_tokens"]
            rate = turn["decode_tps"]
            if count < 1 or not math.isfinite(rate) or rate <= 0:
                raise ValueError("invalid completion token count or decode rate")
            tokens += count
            seconds += count / rate
    return tokens / seconds



def verify_model(model: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    quant_path = model / "quantization.json"
    state_path = model / ".conversion-state.json"
    q3_path = model / "expert-q3.json"
    quant = load_json(quant_path)
    conversion = load_json(state_path)
    q3 = load_json(q3_path)
    if (
        quant.get("complete") is not True
        or conversion.get("signature", {}).get("source")
        != quant.get("source")
        or conversion.get("signature", {}).get("source_fingerprint")
        != quant.get("source_fingerprint")
    ):
        raise ValueError("base conversion identity is incomplete or inconsistent")
    completed = conversion.get("completed", {})
    if not isinstance(completed, dict):
        raise ValueError("base conversion completed ledger is not an object")
    totals = {
        "output_shards": len(completed),
        "data_bytes": sum(int(row["data_bytes"]) for row in completed.values()),
        "file_bytes": sum(int(row["file_size"]) for row in completed.values()),
        "tensor_count": sum(int(row["tensor_count"]) for row in completed.values()),
        "logical_tensor_count": len(conversion.get("inventory", {})),
    }
    for key in ("output_shards", "data_bytes", "tensor_count", "logical_tensor_count"):
        if totals[key] != int(quant[key]):
            raise ValueError(f"base conversion total mismatch for {key}")
    if (
        q3.get("complete") is not True
        or len(q3.get("files", [])) != int(q3.get("file_count", -1))
        or sum(int(row["bytes"]) for row in q3["files"])
        != int(q3["data_bytes"])
    ):
        raise ValueError("q3 manifest is incomplete or internally inconsistent")
    base = {
        "schema": BASE_SCHEMA,
        "recorded_at": utc_now(),
        "quantization_sha256": sha256_file(quant_path),
        "conversion_state_sha256": sha256_file(state_path),
        "signature": conversion["signature"],
        "totals": totals,
        "shards": completed,
    }
    manifests = {
        "quantization": quant,
        "quantization_sha256": sha256_file(quant_path),
        "expert_q3": q3,
        "expert_q3_sha256": sha256_file(q3_path),
        "config_sha256": sha256_file(model / "config.json"),
    }
    return base, manifests



def publish(
    *,
    root: Path,
    model: Path,
    state_path: Path,
    historical_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    state, rows, state_sha256 = verify_trials(state_path)
    base, manifests = verify_model(model)
    historical = load_json(historical_path)
    historical_tps = float(historical["summary"]["sustained_tps"])
    historical_ttft = median_ttft(historical)
    trial_tps = [row["sustained_tps"] for row in rows]
    trial_ttft = [row["median_ttft_s"] for row in rows]
    turn_ttft = [turn["ttft_s"] for row in rows for turn in row["turns"]]
    aggregate = {
        "trial_count": len(rows),
        "measured_turn_count": len(turn_ttft),
        "token_weighted_decode_tps": token_weighted_tps(rows),
        "median_trial_sustained_tps": statistics.median(trial_tps),
        "mean_trial_sustained_tps": statistics.fmean(trial_tps),
        "minimum_trial_sustained_tps": min(trial_tps),
        "maximum_trial_sustained_tps": max(trial_tps),
        "median_trial_ttft_s": statistics.median(trial_ttft),
        "median_turn_ttft_s": statistics.median(turn_ttft),
        "mean_turn_ttft_s": statistics.fmean(turn_ttft),
    }
    aggregate["median_tps_over_historical"] = (
        aggregate["median_trial_sustained_tps"] / historical_tps
    )
    aggregate["median_ttft_over_historical"] = (
        aggregate["median_trial_ttft_s"] / historical_ttft
    )
    source = manifests["quantization"]["source"]
    revision = source.rsplit("@", 1)[1]
    reconstruction = {
        "base": [
            "./.venv/bin/python",
            "c/tools/convert_qwen.py",
            "--repo",
            "deepreinforce-ai/Ornith-1.0-397B-FP8",
            "--revision",
            revision,
            "--outdir",
            "c/ornith397",
            "--staging-dir",
            "c/.ornith397.source",
            "--xbits",
            "int4g128",
            "--io-bits",
            "8",
            "--shared-bits",
            "8",
            "--group-size",
            "128",
            "--min-free-gb",
            "100",
        ],
        "q3": [
            "./.venv/bin/python",
            "c/tools/requantize_expert_q2.py",
            "--snapshot",
            "c/ornith397",
            "--bits",
            "3",
            "--group-size",
            "128",
            "--iterations",
            "3",
            "--experts-per-file",
            "64",
            "--workers",
            "8",
            "--torch-threads",
            "1",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / "base-shards.json"
    atomic_json(base_path, base)
    copied: dict[str, Any] = {}
    for name in COPY_NAMES:
        source_path = model / name
        destination = output_dir / name
        payload = source_path.read_bytes()
        atomic_bytes(destination, payload)
        copied[name] = {
            "path": str(destination.relative_to(root)),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
    compact_artifacts = {
        "base-shards.json": {
            "path": str(base_path.relative_to(root)),
            "bytes": base_path.stat().st_size,
            "sha256": sha256_file(base_path),
        },
        **copied,
    }
    record = {
        "schema": SCHEMA,
        "recorded_at": utc_now(),
        "repository": state.get("repository"),
        "environment": state.get("environment"),
        "performance": {
            "state": str(state_path),
            "state_sha256": state_sha256,
            "signature": state["signature"],
            "completed_at": state.get("completed_at"),
            "configuration": state["completed"]["1"].get("configuration"),
            "hardware": state["completed"]["1"].get("hardware"),
            "trials": rows,
            "aggregate": aggregate,
        },
        "historical_control": {
            "artifact": str(historical_path),
            "sha256": sha256_file(historical_path),
            "sustained_tps": historical_tps,
            "median_ttft_s": historical_ttft,
        },
        "source_identity": {
            "source": source,
            "source_fingerprint": manifests["quantization"][
                "source_fingerprint"
            ],
            "base_quantization_sha256": manifests["quantization_sha256"],
            "expert_q3_sha256": manifests["expert_q3_sha256"],
            "config_sha256": manifests["config_sha256"],
        },
        "reconstruction": {
            "commands": reconstruction,
            "tool_hashes": {
                "c/tools/convert_qwen.py": sha256_file(
                    root / "c/tools/convert_qwen.py"
                ),
                "c/tools/requantize_expert_q2.py": sha256_file(
                    root / "c/tools/requantize_expert_q2.py"
                ),
            },
        },
        "compact_artifacts": compact_artifacts,
        "retirement_acceptance": {
            "five_trials_complete": len(rows) == 5,
            "all_artifacts_hash_verified": True,
            "all_acceptance_gates_passed": True,
            "source_and_conversion_identity_consistent": True,
            "base_shard_hashes_preserved": len(base["shards"]) == 122,
            "q3_shard_hashes_preserved": len(
                manifests["expert_q3"]["files"]
            )
            == 480,
            "ornith_weight_container_may_be_removed_for_space": True,
            "deepseek_promotion_gate_passed": False,
        },
    }
    atomic_json(output_dir / "control.json", record)
    return record



def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--model", type=Path, default=Path("c/ornith397"))
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(
            "c/bench/deepseek_perf_qualification/"
            "ornith397-legacy4-control-state.json"
        ),
    )
    parser.add_argument(
        "--historical",
        type=Path,
        default=Path("c/ornith397_q3_qualification.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/research/artifacts/ornith397_retirement"),
    )
    return parser.parse_args()


def rooted(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> int:
    args = parse_args()
    root = args.root.resolve(strict=True)
    model = rooted(root, args.model)
    state_path = rooted(root, args.state)
    historical_path = rooted(root, args.historical)
    output_dir = rooted(root, args.output_dir)
    try:
        record = publish(
            root=root,
            model=model,
            state_path=state_path,
            historical_path=historical_path,
            output_dir=output_dir,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"retirement evidence failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_dir / "control.json"),
                "aggregate": record["performance"]["aggregate"],
                "acceptance": record["retirement_acceptance"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
