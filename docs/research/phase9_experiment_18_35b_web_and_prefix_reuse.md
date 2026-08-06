# Phase 9 Experiment 18: Real 35B Web Qualification and Prefix Reuse

## Abstract

This experiment exercised the installed-style browser, OpenAI gateway, mux
engine, tokenizer, tier manager, and CUDA kernels together on the converted
Qwen3.5-35B-A3B container. Two greedy chat turns both returned the requested
text exactly and the scheduler released and reused slot zero correctly. The
experiment also found a template/cache boundary that prevented exact-prefix
reuse: Qwen's official history template removes the empty thinking envelope
that was present in the original generation prompt. A serving-only renderer
now preserves that envelope while the public renderer remains byte-identical
to the official template.

Correct prefix recognition did not by itself reduce short-conversation time to
first token. The uncached 20-token suffix was replayed through decode-oriented
resident kernels, which was slower than the highly batched fresh prefill for
this 43-token transcript. A causal block continuation was implemented and
proved token-exact on the tiny model, but was also slower on the tested
hardware. It remains opt-in rather than becoming an unmeasured default.

## Research questions

1. Does the complete production web path generate coherent, deterministic
   output on the real 35B container?
2. Does a second OpenAI chat request form an exact extension of the token
   history retained by its mux slot?
3. If it does, is suffix-only continuation faster than fresh prefill for a
   short second turn?

The initial hypothesis was that exact reuse would reduce second-turn TTFT.

## Method

The qualification helper `c/tools/smoke_web_qwen.py` started:

```
colib web --no-build --model c/qwen35 --kv-slots 2 --cuda
```

It waited for `/health`, fetched the production browser root, then sent two
streaming `/v1/chat/completions` requests to `cache_slot=0`. Temperature was
zero and each user instruction requested exactly `colib ready`. The second
request included the first user/assistant exchange as OpenAI message history.
TTFT was measured at the first nonempty content delta rather than the initial
assistant-role SSE frame.

The machine was an AMD Ryzen 7 7700X with 29.38 GiB visible RAM and an NVIDIA
GeForce RTX 5070 Ti with 15.92 GiB visible VRAM. The long-running 397B
converter remained active during these trials, so cold page-cache and storage
timings are not treated as isolated benchmark results.

The prompt boundary was analyzed with the official tokenizer and chat
template. The original no-thinking generation prompt ended in:

```
<|im_start|>assistant
<think>

</think>

```

The official rendering of that assistant as an earlier history turn omitted
the thinking envelope. Only 15 leading tokens therefore matched. The
`cache_prefix_compatible` renderer mode retains the generation-time envelope
for historical assistants. Its default remains false, preserving the existing
byte-for-byte official-template test. The HTTP gateway and interactive CLI use
the cache-compatible mode internally.

For suffix execution, two mechanisms were compared:

- the production resident path, consuming each uncached token across slot-major
  CUDA GDN/GQA state;
- a new causal block path that exports one resident slot, consumes the suffix
  through `forward_decode_block`, and imports the final state.

The latter is enabled only by `SERVE_SUFFIX_BLOCK=1`. A multi-token tiny-model
test compares its continuation with fresh full prefill.

## Results

### Functional qualification

Every complete 35B trial passed these invariants:

- the production web bundle was served;
- both requests returned exactly `colib ready`;
- each response reported four completion tokens;
- the scheduler reported two admissions and two completions with zero rejects,
  timeouts, or cancellations;
- the final expert tier held 320 experts in 5.76 GiB VRAM;
- no server or CUDA error occurred.

### Prompt analysis

| Representation | Prompt/history property |
|---|---:|
| official first-turn prompt | 19 tokens |
| official continued prompt | 39 tokens |
| official common prefix | 15 tokens |
| cache-compatible continued prompt | 43 tokens |
| cached transcript prefix | exact |
| uncached cache-compatible suffix | 20 tokens |

The extra four tokens are the historical empty thinking envelope. They are
necessary to represent the actual bytes that produced the cached state.

### Timing observations

| Path | Startup (s) | turn 1 TTFT (s) | turn 2 TTFT (s) | Interpretation |
|---|---:|---:|---:|---|
| official history, warm-page-cache observation | 4.05 | 10.47 | 24.85 | no exact reuse |
| exact cache renderer, resident token replay | 31.65 | 27.69 | 28.88 | exact, but 20 decode-style suffix steps |
| exact cache renderer, causal CPU-oriented block | 19.04 | 24.69 | 32.77 | exact block is slower |
| full CUDA speculative block flags | — | — | >150 | stopped; decisively slower for this workload |

These are qualification observations, not a controlled performance A/B:
startup/page-cache state and concurrent conversion differed. The direction of
the causal-block result is nevertheless unambiguous enough to reject making it
the default, because it exceeded both alternative paths substantially.

Focused regression tests after restoring the production default passed 22/22
across mux integration, Qwen gateway protocol, and CLI/browser coverage. The
complete regression suite later passed 21 C executables and 59 Python tests.

## Interpretation

The original reuse failure was model-template-specific, not GPU-specific.
Qwen's public conversation template intentionally suppresses prior reasoning,
whereas a state cache needs the literal tokens previously evaluated. Separate
public and cache-canonical renderings resolve those competing requirements.

The timing crossover is hardware- and workload-specific. Fresh prefill exposes
many prompt rows to batched projections and grouped expert work. A 20-token
suffix processed as 20 resident decode steps loses that parallelism, even
though it avoids recomputing 23 cached tokens. Reuse should become increasingly
valuable as the cached history grows relative to the new suffix, but this
experiment does not establish that crossover point.

The existing causal-block implementation proves a useful mathematical seam,
but its speculative CUDA mode was designed for very short MTP verification
blocks, not arbitrary chat suffixes. Reusing it wholesale is therefore not a
production optimization.

## Decision

- Retain the cache-compatible internal renderer.
- Retain exact-prefix validation and per-slot token history.
- Keep resident per-token suffix execution as the default.
- Keep `SERVE_SUFFIX_BLOCK=1` as an experimental, parity-tested mechanism.
- Do not claim a TTFT improvement for short second turns.
- Add a dedicated continuation-prefill CUDA kernel and controlled
  history/suffix crossover study before enabling an adaptive policy.

## Limitations and next experiment

The 397B converter contended for storage and page cache, and only one GPU/model
combination was tested. The browser document was fetched and the same gateway
used by that document was exercised, but browser automation did not click the
form. Sampling remains intentionally unsupported.

Gate 9 still requires the 397B production web run and production-size
continuous-batch measurements. For 35B, the next performance study should hold
weights/page cache constant and sweep cached-history lengths and suffix lengths
while recording prefill, state-transfer, and first-decode CUDA events
separately.
