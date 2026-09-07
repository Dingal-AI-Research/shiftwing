#!/usr/bin/env python3
"""The expert cache must not change what the model says.

Whatever the eviction policy is, and however small the cache, the tokens have
to be identical: the cache decides what is read from disk, never what comes
out. This runs the same prompt at several cache sizes and compares.

The sizes matter. A block of the expert union is at most as large as the cache,
so a cache smaller than the union forces several blocks per layer -- and that
is where a policy bug lives. Frequency-based eviction shipped with a
reservation window scoped to the layer instead of the block, so the second
block of every layer found every slot reserved and the engine aborted. The
fixture has four experts and the default cache holds all of them, so only the
small caps reach that path.

Needs the converted (streaming) fixture: an f32 fixture keeps its experts
resident and never touches the cache at all.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CAPS = (1, 2, 3, 0)          # 0 = size it from the measured budget
IDS = "5,17,42"
NGEN = 4


def tokens_at(binary: Path, fixture: Path, cap: int) -> str:
    result = subprocess.run(
        # Path("./glm53") normalises to "glm53", which subprocess looks for on
        # PATH rather than in the current directory.
        [str(binary.resolve()), "--model", str(fixture.resolve()), "--ids", IDS,
         "--greedy", str(NGEN), str(cap)],
        capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"FAIL cache={cap}: engine exited {result.returncode}\n"
            f"{result.stderr.strip()[-500:]}")
    for line in result.stdout.splitlines():
        if line.startswith("greedy"):
            return line.strip()
    raise SystemExit(f"FAIL cache={cap}: no greedy line\n{result.stdout[-500:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True,
                        help="a CONVERTED tiny model (quantised experts)")
    arguments = parser.parse_args()

    if not (arguments.fixture / "model.safetensors").exists():
        print(f"SKIP: no converted fixture at {arguments.fixture}")
        return 0

    baseline = None
    for cap in CAPS:
        got = tokens_at(arguments.binary, arguments.fixture, cap)
        if baseline is None:
            baseline = got
        elif got != baseline:
            print(f"FAIL: cache size changed the output\n"
                  f"  cache={CAPS[0]}: {baseline}\n  cache={cap}: {got}")
            return 1
    print(f"PASS expert cache: identical tokens at cache sizes "
          f"{', '.join(str(c) for c in CAPS)} ({baseline})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
