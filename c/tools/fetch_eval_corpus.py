#!/usr/bin/env python3
"""Fetch and verify the fixed public-domain Gate-4 perplexity corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "fixtures" / "qwen35_eval_corpus.json")
    parser.add_argument("--output", type=Path, default=root / "bench" / "qwen35_eval.txt")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    with urllib.request.urlopen(manifest["url"], timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["sha256"]:
        raise SystemExit(f"corpus SHA256 mismatch: got {digest}, expected {manifest['sha256']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_bytes(data)
    temp.replace(args.output)
    print(f"{args.output}: {len(data)} bytes sha256={digest}")


if __name__ == "__main__":
    main()
