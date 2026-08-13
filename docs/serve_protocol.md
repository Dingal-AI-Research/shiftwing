# The serve protocols — engine ⇄ server wire format

The engine speaks two line-oriented protocols over stdin/stdout. Both are plain text
plus byte-counted payload frames; every outbound line is written with a trailing
`fflush`, one line per write. Windows implementations must switch both ends of
the pipe to binary mode before the handshake—the CRT's CRLF translation would
otherwise corrupt sentinels and stall byte-counted reads. That platform hook
is still pending in the current preflight implementation.

| protocol | entry | selected by | used by |
|---|---|---|---|
| **mux** (continuous batching, up to 16 KV slots) | `run_serve_mux` | `SERVE_BATCH=1` | `openai_server.py`, `coli web` |
| **legacy** (single slot, interactive) | `run_serve` | `SERVE=1` (without `SERVE_BATCH`) | `coli chat` |

This document is the reference for the **mux** protocol; the legacy protocol
is summarized at the end. It defines the target contract for `qwen.c`. The
framing primitive lives in `c/serve_mux.h`; until `run_serve_mux` is connected,
this document is a specification rather than a claim that the complete mux
engine already ships.

The current `SERVE_BATCH=1` implementation supports two target-only execution
paths. The default correctness reference switches saved model state and
evaluates rows sequentially. `SERVE_RESIDENT=1` instead keeps recurrent/KV
state per slot and evaluates all active rows through batched projections and
grouped MoE without per-token snapshots. CUDA builds keep eligible GDN
recurrence and GQA KV slot-major on the device when the guarded allocation
fits. Both emit the startup telemetry below, request frames, per-turn telemetry,
and `DONE`; warm production performance qualification remains Phase-9 work.

## Startup handshake (engine → server)

```
\x01\x01READY\x01\x01
HWINFO <cores> <ram_total_gb> <ram_avail_gb> <ngpu> <vram_total_gb> <cpu_name>|<gpu_name>
TIERS <vram_experts> <ram_experts> <disk_experts> <vram_gb> <ram_gb>
Q3NATIVE <uploads> <upload_bytes> <gemv_calls> <grouped_calls>
Q3ATLAS <active> <refreshes> <routes> <host_entries> <device_entries> <device_capacity> <loads> <device_bytes> <uncovered>
EMAP <rows> <cols> <hex>
```

The server must not send requests before `READY`. `HWINFO`/`TIERS`/`EMAP`
follow it and may grow — **servers must ignore line kinds they do not
recognize**; that is the protocol's forward-compatibility rule.

## Requests (server → engine)

```
SUBMIT <id> <slot> <bytes> <max_tokens> <temperature> <top_p>\n<payload>\n
CANCEL <id>\n
```

- `id` — non-zero u64, unique among in-flight requests.
- `slot` — KV slot index, `0 … KV_SLOTS-1` (`KV_SLOTS` env, 1–16, default 1). A slot
  holds one conversation's KV; the engine reuses it only when the tokenized
  payload is an exact extension of the slot's complete stored history. A
  divergent or shortened payload is fully re-prefilled.
- `bytes` — exact byte length of `payload` (UTF-8, may contain newlines). The engine
  reads exactly that many bytes after the header line, then one trailing `\n`.
- `payload` — the fully rendered prompt (the server owns the chat template).
- EOF on stdin = graceful shutdown: in-flight requests finish first.
- `SESSION_DIR=/path` persists each slot after `DONE` or acknowledged
  cancellation. A later engine process restores the slot and may reuse it only
  when the new tokenized payload is an exact extension of the stored history.
  Invalid or incompatible checkpoint files are ignored.

With `SERVE_RESIDENT=1`, a one-slot CUDA server uses transactional incremental
prefill by default. `PREFILL_BATCH` selects 1-8 causal rows (default 8); rows
use an internal staging slot and the live conversation is replaced only after
the final batch succeeds. Intermediate batches do not calculate vocabulary
logits, and the final batch calculates the LM head only for its last row.
`SERVE_PREFILL_BACKEND=serial` retains the prior correctness path. The mux
returns to input polling between microbatches, allowing `CANCEL` to discard
staging state while preserving the previous live conversation. Other server
shapes currently retain serial prefill. During decode, every active slot
contributes one row to the resident forward. CUDA builds use device-resident
GDN/GQA state when eligible.

## Responses (engine → server)

Per request, in order:

```
PREFILL_BEGIN <id> <total> <cached>
PREFILL_PROGRESS <id> <completed> <total> <elapsed_ms>
PREFILL_END <id> <total> <elapsed_ms>
DATA <id> <n>\n<n bytes of UTF-8>\n        # a decoded token's text; repeated
TOPK <id> 5 <logprob> <hextext> ... ×5     # candidates for the sampled token (SERVE_TOPK=1)
HITS <rows> <cols> <hex>                   # ~every 6 tokens: routed-expert bitmap since last HITS
RESIDENT <id> <layers> <device_moe> <host_moe> <activation_h2d> <activation_d2h> <logits_d2h> <router_d2h>
DPERF <id> <dt> <t_edisk> <t_ewait> <t_emm> <t_attn> <t_kvb> <t_head> [...]
CACHE <id> <cpu_hits> <gpu_hits> <misses> <read_bytes> <direct_bytes>
DCACHE <id> <cpu_hits> <gpu_hits> <misses> <read_bytes> <direct_bytes>
REPIN <layer> <eid> <old_tier> <gpu>       # live re-pin swap events, as they happen
...
DONE <id> STAT <emitted> <tok_s> <hit_pct> <rss_gb> <prompt_tokens> <length_limited>
```

`PREFILL_BEGIN`, monotonic `PREFILL_PROGRESS`, and `PREFILL_END` are
request-control frames emitted before the first `DATA`. `completed` includes
exact-prefix cached tokens. Gateways should surface them as progress, but must
not treat them as model output or use them to disarm a first-output deadline.

Errors replace the stream: `ERROR <id> <CODE>` with codes `BAD_FRAME`, `BAD_REQUEST`,
`SLOT_BUSY`, `DUPLICATE_ID`, `EMPTY_PROMPT`, `NOT_FOUND` (CANCEL of unknown id),
`CANCELLED`. A `CANCEL` is acknowledged by `ERROR <id> CANCELLED` after the slot's KV
is persisted.

A malformed header line is recoverable and may receive `BAD_FRAME`. A
truncated payload, missing payload terminator, or rejected untrusted payload
length is connection-fatal because the next frame boundary cannot be
established safely.

The CUDA implementation also emits cumulative `RESIDENT` transfer and
execution-path counters. The implementation emits `PERF`, `DPERF`, `RESIDENT`
(CUDA), `CACHE`, `DCACHE`, `TIERS`, `EMAP`, and `HITS` immediately before
each `DONE`. `ENTROPY`, `GPUS`, and `.coli_usage` persistence remain
future extensions.

## Telemetry lines

| line | format | meaning |
|---|---|---|
| `TIERS` | `TIERS <vram> <ram> <disk> <vram_gb> <ram_gb>` | expert count per tier + resident bytes |
| `HWINFO` | `HWINFO <cores> <ram_total> <ram_avail> <ngpu> <vram_total> <cpu>\|<gpu>` | host snapshot (GBs are floats) |
| `Q3NATIVE` | `Q3NATIVE <uploads> <upload_bytes> <gemv_calls> <grouped_calls>` | cumulative proof that packed-q3 CUDA upload and execution paths ran |
| `Q3ATLAS` | `Q3ATLAS <active> <refreshes> <routes> <host_entries> <device_entries> <device_capacity> <loads> <device_bytes> <uncovered>` | unvalidated adaptive q3 hot-route partition state; emitted only when explicitly enabled |
| `EMAP` | `EMAP <rows> <cols> <hex>` | one byte per expert, row-major over `rows×cols` (sparse layers +MTP × experts): `byte = (tier<<6) \| heat` — 2-bit tier (0 disk / 1 RAM / 2 VRAM), 6-bit log₂-bucketed usage heat |
| `HITS` | `HITS <rows> <cols> <hex>` | 1 bit per expert, experts routed since the previous `HITS` |
| `PERF` | `PERF <id> <dt> <t_edisk> <t_ewait> <t_emm> <t_attn> <t_kvb> <t_head> [<cuda_tx> <cuda_setup> <cuda_routed_hidden> <cuda_routed_down> <cuda_reduce> <cuda_shared_hidden> <cuda_shared_scale> <cuda_shared_down> <cuda_download>]` | inclusive request-interval deltas, seconds; the optional suffix is CUDA-event time for fused MoE transactions; concurrent rows can share batched work, so phase values are diagnostic rather than additive |
| `DPERF` | same fields and optional CUDA suffix as `PERF` | decode-only interval measured after prefill; use this record to explain `DONE` decode tok/s and retain `PERF` for inclusive request/TTFT diagnosis |
| `RESIDENT` | `RESIDENT <id> <layers> <device_moe> <host_moe> <activation_h2d> <activation_d2h> <logits_d2h> <router_d2h>` | cumulative CUDA resident-layer/MoE transactions and transfer bytes; a qualified graph requires positive device counts and zero host-MoE fallback |
| `CACHE` | `CACHE <id> <cpu_hits> <gpu_hits> <misses> <read_bytes> <direct_bytes>` | inclusive request-interval routed-expert cache counts and physical expert-I/O bytes; overlapping requests may share work, so per-request deltas are diagnostic under concurrency |
| `DCACHE` | `DCACHE <id> <cpu_hits> <gpu_hits> <misses> <read_bytes> <direct_bytes>` | decode-only counterpart measured after prefill; `DONE` cache-hit percentage uses this interval so it aligns with decode tok/s |
| `ENTROPY` | `ENTROPY <h0> <h1> …` | per-sparse-layer routing entropy of the turn, bits |
| `GPUS` | `GPUS <n> (<used_gb> <total_gb> <experts>)×n` | per-device VRAM + resident expert count (CUDA builds) |
| `TOPK` | `TOPK <id> 5 (<logprob> <hextext>)×5` | token text hex-encoded so the line stays line-shaped |
| `REPIN` | `REPIN <layer> <eid> <old_tier> <gpu>` | one line per hot-store swap (`REPIN=n` mode) |

All telemetry is advisory: servers render what they know and skip the rest.

## HTTP surface (`openai_server.py`)

- `POST /v1/chat/completions` — OpenAI-compatible; streaming responses emit one extra
  SSE frame `data: {"colibri": {stats, perf, topk, entropy, gpus, repin}}` immediately
  before `data: [DONE]`; non-streaming responses attach the same object as a
  `"colibri"` field.
- `GET /experts` — the latest `EMAP`/`HITS` state: `{rows, cols, map, hits, seq,
  gpus, entropy, repin}`.
- `GET /*` — static hosting of `web/dist` (SPA fallback, path-traversal-safe), plus
  `experts.json` if published there (the measured expert atlas, #175/#218).

## Legacy protocol (`run_serve`, `coli chat`)

Interactive lines are prompts; control frames: `\x02RESET` (clear history),
`\x02MORE` (continue an NGEN-truncated answer), and
`\x02PROMPT <bytes> <max_tokens> <temperature> <top_p> [kv_slot]\n<prompt>\n`
(the pre-mux API mode). Responses are raw text terminated by `\x01\x01END\x01\x01`
plus a `STAT` line. New integrations should use the mux protocol.
