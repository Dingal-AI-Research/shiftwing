# Phase 8 Experiment 7: Ornith-35B Numerical and Tool-Use Qualification

**Model:** `deepreinforce-ai/Ornith-1.0-35B-FP8`
**Converted revision:** `1ab57ce0b44950e498a88756f40ad1ed4d0f30ca`
**Reference revision:** `383064f72a1ef3087b779f268d3ca117eb989aac`
**Date:** 2026-07-28
**Status:** Ornith-35B sub-gate passed

## Abstract

This experiment tested whether the converted Ornith-1.0-35B container
preserves the behavior of the publisher's independently quantized GGUF and
whether it can complete a real OpenAI-compatible tool interaction. Three
complementary tests were used: next-token agreement while both models were
given the same history, perplexity on a fixed text sample, and an HTTP
tool-call round trip.

The converted model matched 1,232 of 1,280 next-token decisions (96.25%).
All 20 prompts exceeded the preregistered 85% per-prompt threshold. Its
perplexity was 1.108495 versus 1.1086 for the official Q4_K_M reference, a
-0.0094% relative difference. In the live protocol test, the model selected
`get_weather`, supplied `Paris`, consumed a deterministic 18 °C result,
produced a coherent final answer, and terminated with `finish_reason=stop`.
All preregistered Ornith-35B acceptance criteria therefore passed.

## 1. Research question

Does the structurally complete Ornith-35B conversion behave like an
independent official quantization and obey the complete model-to-tool-to-model
protocol used by an application?

## 2. Terms

**Teacher forcing** means that both models receive the same known token
history before predicting the next token. This prevents one early,
near-equal choice from changing the entire later conversation and lets the
test compare the models at every controlled position.

**Perplexity** measures how surprised a model is by known text. Lower is
better, but the important quantity here is agreement with the independent
reference. Two implementations with nearly equal perplexity assign nearly
equal total probability to the test text even if an occasional top token
differs.

**Tool call** means that the model requests an external function instead of
inventing the function's result. The application executes that function and
returns its result to the model.

**Clean termination** means that the model emits the protocol's `stop`
condition after its final answer. This prevents an application from waiting
for an answer that never ends.

## 3. Method

### 3.1 Identity controls

The converted container manifest fixes:

```text
hf://deepreinforce-ai/Ornith-1.0-35B-FP8@1ab57ce0b44950e498a88756f40ad1ed4d0f30ca
index fingerprint:
15281dc0352464f68ec93e805283a7c070aeb06f22622d975648381009cfe1d6
payload bytes: 19,081,810,684
```

The external reference was the publisher's
`ornith-1.0-35b-Q4_K_M.gguf`, exactly 21,166,757,760 bytes, with SHA-256:

```text
ff25291b2599fb927a835e624d2b3540106af61761c3fa57ac4264046dbec002
```

The reference runner was llama.cpp commit `e920c52`, built with GCC 13.3.
The colib runtime used an RTX 5070 Ti for dense and expert computation, an
8 GiB host expert budget, and a 6 GiB device expert budget.

### 3.2 Next-token comparison

Twenty deterministic mixed-domain prompts were evaluated for 64 positions
each. The preregistered rules required:

- at least 90% aggregate teacher-forced agreement;
- at least 85% agreement for every individual prompt;
- exactly 64 controlled positions for every prompt.

Free-running continuations were also compared, but only as a diagnostic.
They are not an acceptance statistic because a single early choice changes
the later input seen by each model.

### 3.3 Perplexity comparison

A hash-pinned 1,024-token source window was split into two 512-token chunks.
There were 510 scored next-token positions. The maximum accepted relative
perplexity difference was 5%.

### 3.4 HTTP tool interaction

The real OpenAI-compatible gateway started the C engine and rendered Ornith's
official chat template. The first request required a weather tool call for
Paris. The harness then supplied:

```json
{"city":"Paris","temperature_c":18,"condition":"clear"}
```

The second request required a nonempty final answer that used the returned
temperature, made no second tool call, leaked no raw tool XML, and terminated
with `finish_reason=stop`.

## 4. Results

### 4.1 Numerical agreement

| Measurement | Result | Requirement | Decision |
|---|---:|---:|---|
| teacher-forced matches | 1,232 / 1,280 | at least 90% | pass |
| aggregate agreement | 96.25% | at least 90% | pass |
| prompts at or above 85% | 20 / 20 | 20 / 20 | pass |
| weakest prompt | 58 / 64 = 90.625% | at least 85% | pass |
| free-running agreement | 746 / 1,280 = 58.28% | diagnostic | recorded |
| colib perplexity | 1.108495 | reference-relative | pass |
| llama.cpp perplexity | 1.1086 | reference | reference |
| relative perplexity difference | -0.00943% | absolute value at most 5% | pass |

The perplexity corpus SHA-256 was
`01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`.
The evaluated token-ID sequence SHA-256 was
`90d23bdd12555dfb041ee5166026b47e1f726d64f3e3a6b3dec0c8603c1f2c5e`.

### 4.2 Tool behavior and termination

The first response contained exactly:

```json
{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}
```

and ended with `finish_reason=tool_calls`. After receiving the deterministic
tool result, the model answered:

```text
The current weather in Paris is clear with a temperature of 18°C.
```

The final response ended with `finish_reason=stop`. It contained no second
call and no raw XML.

Startup took 207.61 seconds. The tool-selection request took 146.54 seconds
for 297 prompt tokens and 26 generated tokens; the final-answer request took
40.32 seconds for 81 prompt tokens and 17 generated tokens. These are
end-to-end latency observations under a deliberately bounded 8 GiB host /
6 GiB device expert cache, not decode-throughput benchmarks.

The resident CUDA telemetry recorded 1,720 layer-forwards, all 1,720 using
device MoE and none using host MoE. Router and final logits were the intended
device-to-host boundaries.

## 5. Interpretation

The three tests address different failure modes:

```text
container bytes
     |
     v
teacher-forced decisions ----> local numerical behavior
     |
     v
perplexity -------------------> probability distribution over a corpus
     |
     v
HTTP tool round trip ---------> application-visible behavior and termination
```

Passing all three is stronger than a successful model load. The agreement
test constrains individual predictions, perplexity constrains aggregate
probability quality, and the HTTP test verifies that tokenizer, chat
template, tool parser, engine, and stop-token handling cooperate in the
deployed path.

## 6. Limitations

The official comparison is between two different low-bit quantizations, not
against the original full-precision checkpoint. Exact token identity is
therefore neither expected nor required. The fixed corpus is deliberately
small enough to run repeatedly and does not measure every capability.

The latency observations include startup, prompt processing, and expert
cache misses. They must not be interpreted as a sustained decode benchmark.
MTP was not included in this target-only container.

## 7. Decision

Accept the Ornith-35B numerical, tool-use, and clean-termination sub-gate.
Proceed directly to Ornith-397B conversion and qualification. Do not resume
the retired Qwen397 3-bit conversion. If Ornith-397B cannot meet its bounded
memory profile at grouped int4, first measure the same grouped 3-bit
treatment on this accepted Ornith-35B control, then generate low-bit
sidecars only for Ornith-397B.

Primary artifacts:

- `c/ornith35_prefix_gate.json`
- `c/ornith35_ppl_gate.json`
- `c/ornith35_tool_gate.json`
