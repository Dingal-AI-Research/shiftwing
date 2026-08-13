"""Construct reproducible colib engine environments for qualification tools."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path


# Files whose contents determine the native/CUDA inference engine.  The raw
# CUDA executable is intentionally not a stable release identity: nvcc emits
# byte-different fatbins for identical forced rebuilds.  Qualification ledgers
# retain the exact executable hash for historical provenance and use this
# source fingerprint to prove that a later clean rebuild came from the same
# implementation.
ENGINE_SOURCE_FILES = (
    "c/Makefile",
    "c/qwen.c",
    "c/backend_cuda.cu",
    "c/backend_cuda.h",
    "c/compat.h",
    "c/decode_batch.h",
    "c/json.h",
    "c/serve_mux.h",
    "c/serve_scheduler.h",
    "c/st.h",
    "c/tier.h",
    "c/tok.h",
    "c/tok_nfc.h",
    "c/tok_unicode.h",
    "c/uring.h",
)


# Every engine-specific variable that can change control flow, numerical mode,
# storage behavior, profiling, or the selected model/sidecar. Qualification
# harnesses start without these values and then install their recorded profile.
ENGINE_ENV_KEYS = frozenset(
    {
        "ACTS_DIR",
        "AUTOPIN",
        "CHAT",
        "COLI_CUDA",
        "COLI_ENGINE_TRACE",
        "COLI_GPU",
        "COLI_MMAP",
        "CTX",
        "CUDA_ASYNC_ALLOC",
        "CUDA_ATTN",
        "CUDA_DENSE",
        "CUDA_EXPERT_GB",
        "CUDA_EXPERTS",
        "CUDA_F16",
        "CUDA_GDN",
        "CUDA_GDN_CHECK",
        "CUDA_GROUPED",
        "CUDA_GROUPED_KERNEL",
        "CUDA_HEADROOM_GB",
        "CUDA_MLP",
        "CUDA_MTP",
        "CUDA_PINNED_UPLOAD",
        "CUDA_PRELOAD",
        "CUDA_PROFILE_STAGES",
        "CUDA_PROJECTIONS",
        "CUDA_RESIDENT_ACTIVATIONS",
        "CUDA_RESIDENT_KV",
        "CUDA_SHARED_FUSED",
        "CUDA_SPEC_BATCH",
        "CUDA_SPEC_FULL",
        "CUDA_SPEC_GDN",
        "CUDA_SPEC_GDN_BATCH",
        "CUDA_SPEC_MOE_BATCH",
        "CUDA_SPEC_MOE_CHECK",
        "CUDA_SPEC_MOE_EXACT",
        "DEBUG_LOGITS",
        "DECODE_PROTECT",
        "DECODE_PROTECT_PREWARM",
        "DIRECT",
        "DRAFT",
        "DUMP_ACTS",
        "EMAP_PATH",
        "EVAL_CHUNK",
        "EVAL_GROUPED",
        "EVAL_IDS",
        "EVAL_LM_BATCH",
        "EXPERT_Q2",
        "EXPERT_Q3",
        "EXPERT_Q3_MAX_LAYER",
        "EXPERT_Q3_MIN_LAYER",
        "EXPERT_RAM",
        "GDN_CHUNK",
        "IDOT",
        "KV16",
        "KVSAVE",
        "KV_SLOTS",
        "LOAD_ONLY",
        "MLOCK",
        "MTP",
        "MTP_MIN_ACCEPT",
        "MTP_MIN_MARGIN",
        "NGEN",
        "OMP_MIN_WORK",
        "OMP_NUM_THREADS",
        "OMP_PLACES",
        "OMP_PROC_BIND",
        "PIN",
        "PIN_GB",
        "PIPE",
        "PREFETCH_EXTRA",
        "PREFETCH_LOAD",
        "PREFETCH_MIN_CONF",
        "PREFETCH_THREADS",
        "Q3_ROUTE_ATLAS",
        "Q3_NATIVE",
        "PREFIX_IDS",
        "PROF",
        "PROF_DETAIL",
        "PROMPT",
        "RAM_GB",
        "RAM_HEADROOM_GB",
        "REF",
        "SERVE",
        "SERVE_BATCH",
        "SERVE_RESIDENT",
        "SERVE_SUFFIX_BLOCK",
        "SERVE_TOPK",
        "SESSION_DIR",
        "SNAP",
        "SPEC_BATCH",
        "SPEC_FORCE_REJECT",
        "TEXT",
        "TF",
        "TFPREFIX_IDS",
        "TIER_TRACE",
        "TOPK",
        "NUCLEUS",
        "URING",
        "URING_PERSIST",
        "URING_WORKERS",
        "WARMUP",
    }
)


def isolated_engine_env(
    source: Mapping[str, str] | None = None,
    *,
    preserve: tuple[str, ...] = (),
) -> dict[str, str]:
    """Copy an environment after removing unrecorded engine controls."""

    original = os.environ if source is None else source
    retained = {name: original[name] for name in preserve if name in original}
    env = {name: value for name, value in original.items() if name not in ENGINE_ENV_KEYS}
    env.update(retained)
    return env


def engine_source_sha256(root: Path) -> str:
    """Hash the complete repository-local source of the qualification engine."""

    root = root.resolve()
    digest = hashlib.sha256()
    for relative in ENGINE_SOURCE_FILES:
        path = root / relative
        data = path.read_bytes()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()
