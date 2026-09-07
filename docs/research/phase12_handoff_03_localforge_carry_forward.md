# Phase 12 handoff 03: LocalForge carry-forward

Date: 2026-08-18

## Scope

This is a future integration contract only. No LocalForge repository,
configuration, deployment, service, or runtime has been modified or probed
in Phase 12.

**Superseded 2026-08-23.** The owner subsequently directed changes to the
LocalForge repository: the code-review gate now runs on the colib 397B lane for
tasks classified complex at triage, and the `explore` SWE phase was removed. The
scope statement above remains as written because it was true at this report's
decision boundary; see `CURRENT_STATUS_HANDOFF.md` for the current state.

## Model identity and readiness

After Ornith397 reconstruction and performance qualification, LocalForge
should target an Ornith-specific model ID such as
`ornith-1.0-397b-q3-shiftwing`. The binding must include:

- pinned Hugging Face revision
  `8b61f97a8512d9d01bff1a9625c9a16730e115bb`;
- base `quantization.json` fingerprint and every base shard hash;
- q3 sidecar manifest fingerprint and every sidecar hash;
- engine-source fingerprint and CUDA binary hash;
- exact qualified cache, I/O, CUDA, context, and sampling environment.

Readiness must fail closed on any fingerprint mismatch, incomplete sidecar,
CPU/host-MoE fallback, missing CUDA device, or absent model telemetry.

## API and streaming contract

Use the existing OpenAI-compatible chat endpoint and SSE framing. The final
`shiftwing.metrics` event is authoritative for (while `colib.metrics` remains
accepted as a legacy compatibility event):

- TTFT, queue wait, prefill time, decode time, and total time;
- true completion-token decode tok/s;
- prompt and completion token counts;
- expert cache hits, misses, hit percentage, disk bytes, and direct-I/O
  bytes.

LocalForge must not estimate speed from text chunks. It should tolerate
keepalive comments, consume the final metrics event, then accept `[DONE]`.
Cancellation must close the client request and receive a bounded engine
acknowledgement without corrupting the resident session.

## Prompt, tools, and sampling

Use the pinned Ornith tokenizer and its native Jinja/tool protocol; do not
replace it with a generic chat template. Before integration, run
`c/tools/probe_localforge_prompt.py` against a captured prompt and verify
system, user, assistant, tool-definition, tool-call, and tool-result
boundaries.

The initial deterministic profile is temperature 0 and top-p 1. Agent/tool
sampling may use the previously validated Ornith profile only after exact
tool-call structure passes. LocalForge should send explicit values rather
than inheriting ambient server settings.

## Context, sessions, and cache slots

Start with context 4096, matching the device performance baseline. The
Shiftwing CLI accepts up to 65,536 but LocalForge must not advertise a larger
limit until 16K and 64K continuation, cancellation, memory, and latency
soaks pass on this exact device.

Persist the model fingerprint with every versioned session. A resumed
session must bind its cache slot, position, recurrent DeltaNet state,
full-attention KV state, sampler state, and exact prompt prefix. Reject
incompatible snapshots instead of truncating or silently starting a
different model. Cache-slot allocation and release must remain bounded under
batching and cancellation.

## Qualification handoff

Before any LocalForge change:

1. Finish byte-exact Ornith397 base and q3 reconstruction.
2. Complete the paired expanded-q4 route-atlas A/B and select only a passing
   configuration.
3. Re-run batch, cancellation, session reload, prefix reuse, tools, web, and
   context-boundary controls.
4. Capture a release audit with the final model, engine, environment, and
   metrics schema fingerprints.

Only those qualified values may be copied into a later LocalForge task.
