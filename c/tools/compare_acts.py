#!/usr/bin/env python3
"""Compare C per-layer hidden dumps with the deterministic HF tiny oracle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from make_qwen_oracle import rounded_model


ROOT = Path(__file__).resolve().parents[1]


def default_ref(snapshot: Path) -> Path:
    name = snapshot.name
    ref = "ref_qwen_int8.json" if "int8" in name else "ref_qwen_i4.json" if "i4" in name else "ref_qwen.json"
    return snapshot.parent / ref


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "qwen_tiny")
    parser.add_argument("--ref", type=Path)
    parser.add_argument("--engine", type=Path, default=ROOT / "qwen")
    parser.add_argument("--acts-dir", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--no-run-c", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.snapshot = args.snapshot.resolve()
    args.engine = args.engine.resolve()
    refpath = (args.ref or default_ref(args.snapshot)).resolve()
    reference = json.loads(refpath.read_text())
    temporary = None
    acts_dir = args.acts_dir
    if acts_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="qwen-acts-")
        acts_dir = Path(temporary.name)
    acts_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_run_c:
        env = os.environ.copy()
        env.update(
            {
                "SNAP": str(args.snapshot),
                "REF": str(refpath),
                "TF": "1",
                "DUMP_ACTS": "1",
                "ACTS_DIR": str(acts_dir),
                # Layer-wise diagnostics compare against dequantized HF math;
                # keep the optional activation-int8 expert approximation out.
                "IDOT": "0",
            }
        )
        subprocess.run([str(args.engine)], cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)

    torch.set_num_threads(1)
    model = rounded_model(reference["quant"], reference["seed"])
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        handles.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, index=index: captured.__setitem__(index, output.detach().cpu().float())
            )
        )
    input_ids = torch.tensor([reference["full_ids"][:-1]], dtype=torch.long)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    for handle in handles:
        handle.remove()

    checked = 0
    worst = (0.0, -1, -1)
    for layer, values in sorted(captured.items()):
        for token in range(values.shape[1]):
            path = acts_dir / f"layer-{layer:03d}-token-{token:06d}.f32"
            if not path.is_file():
                raise SystemExit(f"missing C activation dump: {path}")
            c_value = np.fromfile(path, dtype=np.float32)
            hf_value = values[0, token].numpy()
            if c_value.shape != hf_value.shape:
                raise SystemExit(f"shape mismatch at layer {layer} token {token}: {c_value.shape} vs {hf_value.shape}")
            diff = float(np.max(np.abs(c_value - hf_value)))
            checked += 1
            if diff > worst[0]:
                worst = (diff, layer, token)
            if diff > args.atol:
                index = int(np.argmax(np.abs(c_value - hf_value)))
                raise SystemExit(
                    f"first divergent activation: layer={layer} token={token} index={index} "
                    f"C={c_value[index]:.9g} HF={hf_value[index]:.9g} max_abs={diff:.3g}"
                )
    print(
        f"activation parity: {checked} layer-token states within {args.atol:g}; "
        f"worst={worst[0]:.3g} at layer={worst[1]} token={worst[2]}"
    )
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
