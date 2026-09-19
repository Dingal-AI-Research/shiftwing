#!/usr/bin/env bash
cd /home/dinga/Projects/shiftwing
W=docs/research/deepseek-full-parity-2026-09-11
S=/tmp/claude-1000/-home-dinga/b22c312d-1bc8-4082-8eb2-0aeba5f8f9d0/scratchpad
DSV4_EXPERIMENTAL=1 DIRECT=1 URING=1 URING_PERSIST=1 \
  timeout 3000 c/tests/probe_deepseek_v4_prefill c/deepseek-v4-flash-0731 \
  "$W/prompt8192.tokens" 8192 1024 "$S/pf8192" 2>&1 \
  | grep -E "^PROBE_RESULT|^PROBE_PHASE" > "$S/pf8192.log"
grep -oE "seconds=[0-9.]+ load_seconds|read_bytes=[0-9]+" "$S/pf8192.log" | head -2
