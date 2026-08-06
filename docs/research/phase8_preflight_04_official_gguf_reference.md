# Phase 8 Preflight 4: Official Ornith-35B Numerical Reference

## Abstract

The Phase 8 roadmap requires the converted Ornith-35B model to pass the same
statistical controls used for Qwen35 against its own GGUF. A metadata-only
registry check confirms that the publisher now provides an official GGUF
repository. This preflight pins its immutable revision, exact Q4_K_M object,
size, and SHA-256 digest before any download and freezes the numerical
comparison criteria.

## 1. Reference artifact

The authoritative repository is
[`deepreinforce-ai/Ornith-1.0-35B-GGUF`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF).
The pinned state is:

| Property | Value |
|---|---|
| repository revision | `383064f72a1ef3087b779f268d3ca117eb989aac` |
| file | `ornith-1.0-35b-Q4_K_M.gguf` |
| exact bytes | 21,166,757,760 |
| file SHA-256 | `ff25291b2599fb927a835e624d2b3540106af61761c3fa57ac4264046dbec002` |
| Hugging Face blob ID | `fe22dc3ac4939a48113ce5c419550fb4c627d994` |

These values were read through the Hub metadata API with file metadata
enabled. No GGUF payload was downloaded while the 397B conversion was active.
The same official repository also lists Q5_K_M, Q6_K, Q8_0, and BF16 files;
Q4_K_M is selected because it matches the cross-quantizer control used in
Phase 4 and has the lowest additional storage cost.

## 2. Frozen comparison

After Ornith-35B conversion, the official Q4_K_M file will be downloaded at
the pinned revision and its complete SHA-256 verified. The existing fixed
twenty-prompt corpus is tokenized once with the converted snapshot tokenizer.
Both engines then receive the same 64 teacher-forced token IDs per prompt.

```sh
.venv/bin/hf download deepreinforce-ai/Ornith-1.0-35B-GGUF \
  ornith-1.0-35b-Q4_K_M.gguf \
  --revision 383064f72a1ef3087b779f268d3ca117eb989aac \
  --local-dir c/reference
sha256sum -c <<'EOF'
ff25291b2599fb927a835e624d2b3540106af61761c3fa57ac4264046dbec002  c/reference/ornith-1.0-35b-Q4_K_M.gguf
EOF
```

Acceptance requires:

- at least 85% next-token argmax agreement on every prompt;
- at least 90% aggregate agreement over all 1,280 positions;
- finite perplexity from both engines on the fixed 1,024-token corpus window;
- absolute relative perplexity difference no greater than 5%;
- coherent greedy ChatML output with clean termination.

Free-running token equality remains diagnostic because an early
quantizer-dependent near-tie can create two valid but different continuations.
The teacher-forced comparison isolates local predictive agreement without
allowing one early rank flip to replace the rest of the input sequence.

## 3. Controls

- Both artifacts originate from the same publisher and model family.
- The colib artifact must have a complete converter manifest, which is copied
  into both numerical result files.
- The GGUF revision and file digest are immutable inputs to the report.
- The corpus bytes and token-ID hashes are retained with the result.
- Greedy temperature and context are identical.
- Per-prompt scores are retained; the aggregate cannot hide a failed prompt.
- The 397B converter and other storage-heavy jobs are stopped before timing.

`eval_qwen.py` accepts `--max-relative-ppl-delta 0.05` together with the
pinned `--llama-model`; Gate 8 also supplies `--require-complete-manifest`.
The result embeds the converter source/fingerprint, tensor and byte totals,
and precision map, writes the artifact first, and exits nonero if the
absolute relative difference exceeds the frozen bound.

The comparison is between different quantizers, so it establishes statistical
agreement rather than bit equality with the FP8 source.

The external runner is pinned to the clean sibling checkout
`/home/dinga/Projects/llama.cpp` at commit
`e920c523e3b8a0163fe498af5bf90df35ff51d25`. Its `llama-cli`,
`llama-server`, and `llama-perplexity` binaries all report `e920c52`, built
with GCC 13.3.0 for Linux x86-64. The checkout is used read-only, and both
numerical harnesses capture the invoked binary's `--version` output in their
JSON artifact.

The perplexity acceptance command is:

```sh
.venv/bin/python c/tools/eval_qwen.py \
  --snapshot c/ornith35 \
  --corpus c/bench/qwen35_eval.txt \
  --max-tokens 1024 --ctx-size 512 \
  --ram-gb 8 --ram-headroom-gb 1 \
  --llama-model c/reference/ornith-1.0-35b-Q4_K_M.gguf \
  --llama-perplexity /home/dinga/Projects/llama.cpp/build/bin/llama-perplexity \
  --max-relative-ppl-delta 0.05 \
  --require-complete-manifest \
  --output c/ornith35_ppl_gate.json
```

The teacher-forced thresholds are executable CLI conditions rather than a
manual reading of summary fields:

```sh
.venv/bin/python c/tools/compare_qwen_prefix.py \
  --snapshot c/ornith35 \
  --gguf c/reference/ornith-1.0-35b-Q4_K_M.gguf \
  --require-complete-manifest \
  --expected-gguf-sha256 ff25291b2599fb927a835e624d2b3540106af61761c3fa57ac4264046dbec002 \
  --llama-server /home/dinga/Projects/llama.cpp/build/bin/llama-server \
  --tokens 64 \
  --min-tf-prompt-agreement 0.85 \
  --min-tf-aggregate-agreement 0.90 \
  --output c/ornith35_prefix_gate.json
```

The tool writes per-prompt and aggregate evidence before exiting nonzero when
either bound fails.

## 4. Decision

Use the official Q4_K_M artifact as Gate 8's external numerical control. Do
not substitute a community conversion or silently follow the repository's
moving `main` revision. Download it only after the current Qwen397 conversion
finishes, then run the frozen Phase-4-style teacher-forced and perplexity
checks before the generated HTTP tool gate.
