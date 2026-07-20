#!/usr/bin/env python3
"""Preflight checks before downloading/converting/running a model.

Usage:  python3 tools/preflight.py [35b|397b] [--dir PATH] [--cuda] [--iobench]

Hard-gates the expensive steps (PLAN.md risks #5–#7): free disk for the
download→convert→delete loop, RAM, VRAM + CUDA toolkit (Phase 5+), and an
optional NVMe O_DIRECT read benchmark via the iobench binary.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

# free-disk needed for the streaming download→convert→delete loop:
# ≈ int4 output container + one source shard of headroom. (PLAN.md Phase 4/7.)
REQ = {
    "35b":  {"disk_gb": 35,  "note": "bf16 source ~70 GB streamed; output ~20 GB"},
    "397b": {"disk_gb": 250, "note": "FP8 source ~400 GB streamed; output ~210 GB"},
}
MIN_RAM_GB = 28          # 32 GB box: leave headroom for OS
MIN_CUDA_MAJOR, MIN_CUDA_MINOR = 12, 8   # sm_120 (RTX 5070 Ti) needs ≥12.8
NVME_ADVISE_GBS = 3.0    # sustained O_DIRECT read advised for 397B streaming


def gb(n):
    return n / (1024 ** 3)


def check_disk(path, need_gb):
    total, used, free = shutil.disk_usage(path)
    ok = gb(free) >= need_gb
    print(f"[{'OK' if ok else 'FAIL'}] disk: {gb(free):.0f} GB free at {path} (need ≥{need_gb} GB)")
    return ok


def check_ram():
    try:
        with open("/proc/meminfo") as f:
            kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
        ok = kb / 1024 / 1024 >= MIN_RAM_GB
        print(f"[{'OK' if ok else 'WARN'}] ram: {kb/1024/1024:.1f} GB (want ≥{MIN_RAM_GB} GB)")
        return True  # warn-only: engine has its own RAM guard
    except Exception as e:
        print(f"[WARN] ram: could not read /proc/meminfo ({e})")
        return True


def check_cuda():
    ok = True
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        print(f"[OK] gpu: {out}")
    else:
        print("[FAIL] gpu: nvidia-smi not found")
        ok = False
    nvcc = shutil.which("nvcc") or (os.path.exists("/usr/local/cuda/bin/nvcc") and "/usr/local/cuda/bin/nvcc")
    if nvcc:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release (\d+)\.(\d+)", out)
        if m and (int(m.group(1)), int(m.group(2))) >= (MIN_CUDA_MAJOR, MIN_CUDA_MINOR):
            print(f"[OK] nvcc: release {m.group(1)}.{m.group(2)} (≥{MIN_CUDA_MAJOR}.{MIN_CUDA_MINOR} for sm_120)")
        else:
            print(f"[FAIL] nvcc: {m.group(0) if m else 'unknown version'} — need ≥{MIN_CUDA_MAJOR}.{MIN_CUDA_MINOR} for sm_120 (RTX 5070 Ti)")
            ok = False
    else:
        print(f"[FAIL] nvcc: not found — install CUDA toolkit ≥{MIN_CUDA_MAJOR}.{MIN_CUDA_MINOR} (WSL2: cuda-toolkit, NOT cuda-drivers)")
        ok = False
    return ok


def check_iobench(path):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe = os.path.join(here, "iobench")
    if not os.path.exists(exe):
        print("[WARN] iobench not built (make -C c iobench) — skipping NVMe benchmark")
        return True
    out = subprocess.run([exe, path], capture_output=True, text=True).stdout
    print(out.strip())
    m = re.search(r"([\d.]+)\s*GB/s", out)
    if m and float(m.group(1)) < NVME_ADVISE_GBS:
        print(f"[WARN] NVMe sustained read {m.group(1)} GB/s < advised {NVME_ADVISE_GBS} GB/s for 397B streaming")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="35b", choices=sorted(REQ))
    ap.add_argument("--dir", default=".", help="directory that will hold downloads/containers")
    ap.add_argument("--cuda", action="store_true", help="also require GPU + CUDA toolkit (Phase 5+)")
    ap.add_argument("--iobench", action="store_true", help="run the NVMe O_DIRECT read benchmark")
    a = ap.parse_args()

    req = REQ[a.model]
    print(f"== preflight {a.model} — {req['note']} ==")
    ok = check_disk(a.dir, req["disk_gb"])
    ok &= check_ram()
    if a.cuda:
        ok &= check_cuda()
    if a.iobench:
        check_iobench(a.dir)
    print("== PASS ==" if ok else "== FAIL ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
