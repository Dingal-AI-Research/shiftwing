#!/usr/bin/env python3
"""Validate one real FP8 weight/scale pair against Transformers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.integrations.finegrained_fp8 import Fp8Dequantize


def _converter():
    path = Path(__file__).with_name("convert_qwen.py")
    spec = importlib.util.spec_from_file_location("convert_qwen_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("weight")
    args = parser.parse_args()
    converter = _converter()
    scale_name = converter.fp8_scale_name(args.weight)
    with safe_open(args.shard, framework="pt", device="cpu") as handle:
        if args.weight not in handle.keys() or scale_name not in handle.keys():
            raise SystemExit("weight or sibling scale is absent from shard")
        weight = handle.get_tensor(args.weight)
        scale = handle.get_tensor(scale_name)
    got = converter.dequantize_fp8_blocks(weight, scale)
    reference_op = object.__new__(Fp8Dequantize)
    reference = reference_op._dequantize_one(
        weight, scale, output_dtype=torch.float32
    )
    difference = (got - reference).abs()
    mode = converter.precision_for_tensor(args.weight, got)
    if mode == "int8":
        packed, target_scale = converter.quantize_int8_rows(got)
        reconstructed = converter.dequantize_int8_rows(packed, target_scale)
    elif mode == "int4g128":
        packed, target_scale = converter.quantize_int4_grouped(got)
        reconstructed = converter.dequantize_int4_grouped(
            packed, target_scale, columns=got.shape[-1]
        )
    else:
        packed, target_scale, reconstructed = got, torch.empty(0), got
    error = reconstructed - got
    payload = {
        "weight": args.weight,
        "weight_dtype": str(weight.dtype),
        "weight_shape": list(weight.shape),
        "scale_dtype": str(scale.dtype),
        "scale_shape": list(scale.shape),
        "transformers_exact": torch.equal(got, reference),
        "transformers_max_abs_diff": float(difference.max()),
        "decoded_sha256": hashlib.sha256(
            got.contiguous().numpy().tobytes()
        ).hexdigest(),
        "target_mode": mode,
        "target_payload_bytes": packed.numel() * packed.element_size()
        + target_scale.numel() * target_scale.element_size()
        + (1 if mode != "f32" else 0),
        "target_max_abs_error": float(error.abs().max()),
        "target_rmse": float(error.square().mean().sqrt()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
