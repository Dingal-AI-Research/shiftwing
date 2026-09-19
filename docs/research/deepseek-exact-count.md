# DeepSeek exact prompt counting

The counter uses the pinned DeepSeek-V4-Flash-0731 tokenizer and the same official
chat renderer as the generation gateway. It needs tokenizer metadata and a small
CPU executable; it does not load weights or create a CUDA context.

Build the helper:

```sh
make -C c CUDA=0 deepseek_v4_tokenize
```

When explicitly needed, a separate counter can run while the orchestrator remains
loaded. Use an available port; this command is not installed or started by default:

```sh
python3 c/openai_server.py \
  --model c/.deepseek-v4-flash-0731.source \
  --model-id deepseek-v4-flash-0731-colib \
  --tokenizer-only --host 127.0.0.1 --port 8767
```

POST `/v1/chat/count_tokens` with the complete chat request:

```json
{
  "model": "deepseek-v4-flash-0731-colib",
  "reasoning_effort": "low",
  "messages": [
    {"role": "system", "content": "All mandatory review instructions."},
    {"role": "user", "content": "The original request, complete selected evidence and exact diff."}
  ]
}
```

The response includes `prompt` (the exact rendered string), `prompt_sha256`,
`prompt_bytes`, `prompt_tokens`, `input_limit: 92160`, `output_budget: 8192`,
`review_context: 100352`, and `fits`. An oversized prompt is returned intact with
`fits: false`; the caller must split the review and recount its final packets.
The 90k input cap is separate from the 8k response reserve (98k total context).
LocalForge uses this profile for both code and scaffold reviews.
The counter never selects, summarizes, or removes evidence. `fits` describes the
input budget only; it does not qualify or approve a review.

Checkpoint revision, tokenizer SHA256, encoder SHA256, native helper SHA256, and
count duration are returned with the result. Counting checks the pinned tokenizer
hash before running the helper. The gateway shares rendering, tools, tool-choice
and effort handling between counting and chat generation. Omitted DeepSeek effort
stays non-thinking; `low` opens the thinking block without the high/max prefix.

In tokenizer-only mode, generation endpoints return HTTP 503. Authentication and
host restrictions remain those of the existing gateway. LocalForge invokes the CPU-only counter helper before swapping in DeepSeek
for code and scaffold reviews. Counting and the configured input cap are
independent of native serving qualification and measured review performance.

## Native integration

Use `dsv4_tok_load` and `dsv4_tok_encode` from `c/deepseek_v4_tokenizer.h` for
DeepSeek. Generic `tok_encode` does not implement the checkpoint's split sequence:
382 of 3,287 initial reference cases produced different IDs despite exact decode
round trips. The new path preserves digit and CJK isolation boundaries, the third
regex's punctuation/word rules, and every added token. Qwen/GLM keep their existing
path. A destination buffer smaller than the input byte count is rejected, so the
DeepSeek wrapper cannot silently return a truncated token sequence.

The helper takes length-framed UTF-8 bytes on stdin:

```text
COUNT byte_length\n<exact bytes>\n
```

It returns one JSON object containing `tokens`. `ENCODE` additionally returns
`ids` and `decoded_hex` for reference checks. Frames are bounded to 16 MiB;
malformed, truncated, oversized or unsupported-recipe inputs fail closed.

## Recorded checks, 2026-09-09

- 7,630 native/reference cases matched every token ID and decoded byte, including
  all 1,283 added tokens, multilingual/code inputs, Unicode boundary samples,
  NUL/empty payloads, all effort modes, and long fully rendered prompts.
- The actual HTTP endpoint returned 32,768 tokens with `fits: true` and 32,769 with
  `fits: false`, preserving the exact rendered prompt. Counts took about 0.4 s on
  this host while checkpoint recovery was running. No inference engine loaded.
- These checks validate tokenization and budgeting only. They do not establish
  model correctness, review quality, GPU memory fit, or the 20-minute review gate.

Reproduce reference validation with the repository virtual environment:

```sh
.venv/bin/python c/tools/verify_deepseek_v4_tokenizer.py \
  --tokenizer c/.deepseek-v4-flash-0731.source/tokenizer.json \
  --output /tmp/deepseek-tokenizer-parity.json
```

Evidence: `deepseek-tokenizer-parity-2026-09-09.json` records the initial failure;
`deepseek-native-tokenizer-parity-2026-09-09.json` records the corrected comparison;
`deepseek-count-endpoint-2026-09-09.json` records the actual HTTP boundary checks.
