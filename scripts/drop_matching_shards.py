#!/usr/bin/env python3
"""Drop container shards holding tensors that match a pattern, so they re-convert.

`group_size_for_name` now gives the n-gram embedding one quantization group per
row instead of the default 128, which stores it at 4.20 bits/parameter rather
than 6.80 -- 25.0 GiB instead of 40.5 GiB. The shards already on disk predate
that, and the conversion-state signature covers CLI options only, so a resume
will not notice the drift on its own.

Each shard's logical tensor list is read from its own safetensors header rather
than guessed from a name prefix: an earlier attempt dropped every
`layers.1.ple.*` inventory entry on the assumption they shared one shard, and
left the inventory 131 entries short.

Idempotent: shards already absent from disk and state are skipped, so an
interrupted run can simply be repeated.
"""
from __future__ import annotations
import json, os, struct, sys, glob

ROOT = "/home/dinga/Projects/shiftwing"
OUT = os.path.join(ROOT, "c/qwen38fn")
STATE = os.path.join(OUT, ".conversion-state.json")
DRY = "--dry-run" in sys.argv
PATTERN = next((a.split("=",1)[1] for a in sys.argv if a.startswith("--match=")),
               "ngram_embedding.shard_")


def shard_logical_tensors(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return {k for k in hdr if k != "__metadata__" and not k.endswith((".qs", ".qtype"))}


def main() -> int:
    state = json.load(open(STATE))
    completed, inventory = state["completed"], state["inventory"]
    targets = []
    for path in sorted(glob.glob(os.path.join(OUT, "model-*.safetensors"))):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        if any(PATTERN in k for k in hdr):
            targets.append(path)
    print(f"shards matching {PATTERN!r}: {len(targets)}")
    if not targets:
        print("nothing to do (already re-converted)")
        return 0
    dropped_tensors = 0
    for path in targets:
        name = os.path.basename(path)
        for key in shard_logical_tensors(path):
            if key in inventory:
                del inventory[key]
                dropped_tensors += 1
        completed.pop(name, None)
        if not DRY:
            os.remove(path)
    print(f"dropped {len(targets)} shards, {dropped_tensors} inventory entries")
    print(f"completed {len(completed)} (was 131), inventory {len(inventory)} (was 76491)")
    if DRY:
        print("dry run: no changes written")
        return 0
    tmp = STATE + ".tmp"
    json.dump(state, open(tmp, "w"))
    os.replace(tmp, STATE)
    print("state updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
