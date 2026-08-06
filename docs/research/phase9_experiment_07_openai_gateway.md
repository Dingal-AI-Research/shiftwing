# Phase 9 Experiment 7: OpenAI Gateway and Qwen Protocol Parity

## Abstract

This experiment connected the Phase 9 multiplexed inference protocol to an
OpenAI-compatible HTTP gateway. The gateway was adapted from the dependency-free
server in JustVugg/colibri, pinned at commit
`81f08a09e5651ce52616dc720f68810f9021c0be`. The inherited GLM-5.2 prompt and
tool-call protocol was replaced with the text-only Qwen3.5 ChatML protocol and
Qwen3 XML function calls. Seven protocol/engine checks and four HTTP checks passed.
The result is an executable API reference, although it is not yet a production
server because the engine remains greedy-only and evaluates active slots
sequentially.

## Research question

Can the existing Colibrì HTTP gateway be connected to the new Qwen mux without
changing model-visible prompt bytes or overstating unsupported generation
features?

## Method

The experiment separated three boundaries:

1. **Model protocol.** A local renderer was implemented for the official
   Qwen3.5 text-only template. It supports leading system/developer
   instructions, user and assistant turns, reasoning markers, tool histories,
   and grouped tool responses. Qwen3 XML responses are converted to OpenAI
   `tool_calls`.
2. **Engine protocol.** Gateway startup now reads exactly the mux `READY`
   sentinel. Requests are sent as byte-counted `SUBMIT` frames, and one
   dispatcher maps `DATA`, `DONE`, and `ERROR` frames back to concurrent HTTP
   requests.
3. **HTTP protocol.** Both ordinary JSON responses and server-sent-event
   streaming were exercised through a real local TCP listener.

For prompt parity, a representative conversation containing a system message,
assistant reasoning, a tool call, and a tool result was rendered twice: once by
the gateway and once by Transformers 5.14.1 using the snapshot's own Jinja
template. The complete strings were compared byte for byte. Engine integration
used the actual `qwen_tiny_i4` snapshot rather than a mocked wire process.
The C reference loop was also changed to keep one vocabulary-logits buffer per
slot. For the 248,320-token vocabulary this avoids approximately 0.95 MiB of
allocation and release for every active slot-token; it does not change logits
or batching semantics.

## Results

| Test group | Result |
|---|---:|
| Basic ChatML and thinking-prefix rendering | pass |
| Tool declaration/history rendering | pass |
| Official Jinja byte comparison | pass |
| Qwen3 XML parsing and reasoning removal | pass |
| Unsupported-feature validation | pass |
| Real mux `DATA`/`DONE` dispatch | pass |
| Real mux cancellation after first `DATA` | pass |
| Non-streaming OpenAI chat completion | pass |
| Streaming completion ending in `[DONE]` | pass |
| Non-streaming reasoning/content separation | pass |
| Split-token streaming `</think>` separation | pass |

The first real-engine attempt identified a protocol mismatch: the inherited
gateway used a legacy helper that waited for a `STAT` line immediately after
`READY`, while the mux emits request-scoped statistics inside `DONE`. Replacing
that helper with an exact sentinel read removed the deadlock. This observation
also demonstrates why a real subprocess check was required in addition to
renderer unit tests.

After the per-slot allocation change, all 17 C tests, all 47 Python tests, and
the standalone CUDA/session checks remained green.

## Validity and limitations

The parity comparison covers the released tiny Qwen3.5 template and a
representative multi-step tool history. More randomized histories should be
added before Gate 9 closes. The gateway currently rejects non-greedy
`temperature`/`top_p` values and non-text `response_format` requests because the
C mux does not implement sampling or per-request grammars. This is an explicit
failure rather than silent parameter substitution.

The test establishes HTTP and framing correctness, not continuous batching.
The engine still restores one slot, evaluates one row, saves it, and then moves
to the next row. Telemetry fields remain advisory targets and are not yet all
emitted.

## Conclusion

The HTTP boundary is now functional and model-protocol-correct for the supported
greedy text subset. The engine adapter also cancels a real mux request after its
first `DATA` frame. The next experiment should verify cancellation from an
actual disconnected HTTP socket and then replace sequential slot switching with
resident batched state and batched GDN decode.
