# DeepSeek V4 Flash repetition investigation — 17 September 2026

The local weight files pass a complete integrity check. The captured review was generated with greedy decoding and thinking disabled. The production repetition guard misses this exact captured output. Separately, the custom inference implementation has unresolved numerical disagreement with the repository's adapted reference. These findings explain why parameter count alone is a poor guide to the observed failure; they do not establish one exclusive cause.

At the user's request, DeepSeek review sampling now uses temperature 1 and top_p 0.95. The change is in `/home/dinga/Projects/localforge/src/inference/openai-adapter.ts`, selected by `route.reviewProfile === "deepseek-v4"`. Both initial and recovery requests use this builder. The existing mode and output budget are preserved. Fifty-one focused review/routing/spiral tests passed, and a direct call to the request builder returned the new settings. This verifies configuration, not successful completion of the previously failing review.

**The captured incident is identifiable.**

The saved request is `/home/dinga/Projects/localforge/.localforge/reviews/04138cb6ec225910/1789647622895-6696afcd-5445-4ab6-87e7-e6c4da80b4ae.request.json`. Its metadata records 26,517 actual input tokens, exactly matching the pasted UI. Fresh rendering through the pinned encoder reproduces both that count and its expected prompt hash:

`38a3cb1f3d53e5990a9fb5d5d243d2bffb24c65c21f4b317e3e55fb22252a3f7`.

| Request property | Captured value |
|---|---|
| Model alias | deepseek-flash |
| Temperature / top_p | 0 / 1 |
| Thinking | disabled; no reasoning_effort |
| Output ceiling | 8,192 tokens |
| Actual input | 26,517 tokens |
| Rendered assistant prefix | `<｜Assistant｜></think>` |

This is visible-answer degeneration, rather than an endless hidden reasoning trace. Of 35 complete action strings in the pasted excerpt, only 28 are exactly distinct. Actions 29–35 exactly repeat actions 19–25. Several other entries restate earlier claims with different wording. The output also includes self-retracted allegations and speculative requirements. Those are review-quality failures even before repetition becomes extensive.

The saved request uses the official nonthinking prefix intentionally; the count and rendering evidence do not suggest accidental prompt truncation or an arbitrary generic chat template. The incident log shows context 40,960, enough for this prompt plus its 8,192-token reserve. The checkpoint's advertised context is a separate property from this local serving limit. There is no normal terminal request record for this incident; the API log ends with engine shutdown. Consequently the incident's exact process binary hash cannot be recovered from a completed DSV4_REQUEST line.

**The complete weight check passed.**

I ran the repository's recovery validator against all current model segments. It rebuilds the expected tensor plan from the pinned source index and cached shard headers, verifies manifest/state/proof bindings and metadata, hashes every current segment, and checks for changes during validation. This freshly verifies the stored bytes against retained checksums that were independently compared with source bytes during conversion. It does not redownload and recompare the entire upstream checkpoint today.

| Check | Result |
|---|---|
| Pinned source | deepseek-ai/DeepSeek-V4-Flash-0731 |
| Revision | 9e165c30e2704aec5d9d593cce3eebd58bbef1cb |
| Verified native groups | 91 / 91 |
| Tensor records | 72,317 |
| Weight payload covered | 166,878,536,440 bytes |
| Base-layer expert coverage | all 43 × 256 experts, all three projections |
| Manifest SHA-256 | c84796f773e5b3942e4ba5200a480b7d1ba1c7add7e87cbb7e959271a1270a47 |

The fresh report is `weight-integrity.json`. This rules against missing experts, partial recovery, or changed stored weight bytes relative to the retained source proofs. It does not establish that the runtime interprets and executes every tensor correctly.

**The approximately 304B package contains a 284B target model plus draft weights.**

Counting logical learned scalar slots in the inventory, unpacking FP4 values and excluding encoded quantization scales and integer routing maps, gives:

| Part | Logical learned scalars | Stored payload |
|---|---:|---:|
| Base routed experts | 277,025,390,592 | 147.170 GB |
| Other base parameters | 7,306,849,879 | 8.846 GB |
| DSpark draft component | 19,845,850,983 | 10.863 GB |
| Total | 304,178,091,454 | 166.879 GB |

The base target therefore has approximately 284.33B parameters. Its 43 layers select six of 256 routed experts per token, plus a shared expert. DeepSeek describes the Flash target as approximately 13B activated parameters. That does not make it equivalent to a 13B dense model: the selected experts vary and the much larger stored parameter pool matters. It also does not mean all 304B parameters participate in each output token. The native reviewer explicitly runs with DSpark off. [Official Flash architecture description](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/README.md)

The precision is native mixed precision: packed FP4 routed experts, predominantly FP8 other matrix weights, with BF16/FP32 tensors where specified. The local file size is consistent with those encodings. This is not evidence that a third-party aggressive quantization was substituted. The release config specifies 4 residual streams, 20 Sinkhorn iterations, a 128-token local attention window, and alternating compression ratios 4/128 after the initial uncompressed layers. These details make the correctness of scaling, rounding, routing and compressed attention consequential. [Official config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json)

**Greedy sampling was a real configuration mismatch.**

The original LocalForge adapter hardcoded temperature 0 and top_p 1 for every Colib request. Native `dsv4_sample` returns the largest-logit token immediately when temperature is zero. Once a repetitive continuation becomes the highest-scoring choice, that path offers no sampling escape. Raising temperature permits alternatives, including ending the list, but does not guarantee that the model will choose a correct or concise answer.

DeepSeek recommends temperature 1.0, using top_p 0.95 for agentic scenarios and 1.0 otherwise. Its reported code-agent evaluations use max reasoning. Your captured nonthinking, greedy review should not be expected to reproduce those evaluation conditions. [0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/README.md)

I selected the supported pair 1.0/0.95 for the requested change because this local gateway accepts only 0/1 or 1/0.95. Changing temperature alone to 1 while retaining top_p 1 would be rejected by the gateway. The native sampler implements actual temperature/top-p sampling; the previous greedy restriction in the adapter was stale for this engine.

There are two remaining operational details. The native random seed is derived deterministically from the rendered prompt, so repeating an identical prompt with identical settings can still produce identical sampled output. Also, this gateway explicitly rejects nonzero frequency/presence penalties and per-request seeds. Advice to add those API fields would currently produce an error rather than an effective mitigation.

**The runtime numerical discrepancy is still unresolved.**

I recomputed the saved 43-layer, 32-token comparison from the raw float files rather than copying its pass/fail summary. Native and traced-native logits are identical. Against the saved reference:

| Metric | Recomputed result |
|---|---:|
| Relative L2 error of logits | 0.102731 |
| Cosine similarity | 0.994712 |
| Maximum absolute logit difference | 2.994095 |
| Native top token | 979, `:\n\n` |
| Reference top token | 270, ` the` |
| Top-two native margin | 0.457428 |
| Top-two reference margin | 0.257580 |
| Total variation of temperature-1 softmax distributions | 0.170729 |

For example, the native top token has probability 38.16% under the native logits versus 22.95% under the reference logits. The reference top token has probability 31.29% under the reference versus 19.45% under native. These are measurements on one saved prefix, without top-p truncation. They are not a measured quality-loss percentage or a prediction for the 26,517-token incident.

The saved layer traces already differ after layer 0 (0.631% relative L2) and layer 1 (1.450%). Those layers precede compressed attention. That argues against explaining *all* the numerical disagreement as exclusively a very-long-context compression problem. It does not localize a defective operation. Later discrepancies need not grow monotonically because residual mixing and normalization change the scale.

The reference uses the pinned official model.py blocks with locally written CPU replacements for TileLang primitives and lazy weight loading. Thus this is an unresolved disagreement between two implementations, not proof that the local engine alone is wrong. Floating-point reduction/rounding, reference-adapter fidelity, routing near selection boundaries and native implementation errors remain possibilities. Existing checks of six dense projections and three routed experts are encouraging but cannot certify whole-model behavior.

These logits were generated on September 11. The current serving binary hash is `696433bb32b0d33d69da01d0d66f9051ce7e7a61bd5738f69f710e60c31de49a`; earlier completed production logs identify `a066822435d644e6ca7df2b446d460657bc23d84bb6edf09c581336d6700d7eb`. I did not run new GPU inference while the user was preparing to resume their review. Accordingly, the historical discrepancy remains a concrete investigation lead, not a fresh acceptance result for the current executable.

**The guard demonstrably misses this failure pattern.**

I replayed the supplied completion through the actual production `createTextSpiralGuard` with chunk sizes 1, 4, 16, 64 and 256 characters. Every replay returned `detected: false`, including final flush. See `guard-replay.json` and `replay-guard.mjs`.

The guard requires four exact trailing word cycles, with a maximum cycle size of 256 words, or four exact paragraph cycles separated by blank lines. The captured JSON list repeats seven complete items with intervening material and contains additional paraphrased repetitions. The excerpt is only 11,432 characters, below the request's 81,920-character cap. The existing guard therefore allows it to keep streaming; the failure is reproducible independently of model weights.

The review prompt also promotes every actionable suggestion and regression risk into a required action, without a distinctness requirement or a finding limit. The supplied orchestrator brief foregrounds locator ambiguity and serial-test ordering, and the output repeatedly elaborates those concerns. This is a plausible amplifier of speculative over-review, not a demonstrated cause of the loop. A useful future prompt change would require distinct, evidence-supported defects and consolidate duplicate findings. Output grammar alone would ensure valid JSON, not truth or progress.

**Large-model loops are documented, including this exact checkpoint.**

In a firsthand report on the model repository, an operator reproduced repetition on a roughly 240K-token tool-use conversation across Ollama Cloud, DashScope and a community vLLM deployment at temperature 1/top_p 0.95. They report loops with speculative decoding removed and with official message encoding. Those observations are evidence that raising temperature and disabling DSpark cannot universally eliminate 0731 loops. Their much longer, tool-heavy workload differs from this tool-free review; it does not diagnose this incident. [Portable reproduction report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/discussions/58)

Another firsthand discussion contains multiple distinct failure mechanisms, including a speculation-specific implementation issue and looping that persists with speculation disabled. It also contains an edited retraction of a claimed CUDA-version cure. This is why a generic “update CUDA” or “disable DSpark” answer would be unjustified for the native reviewer, which already disables DSpark. [Reasoning-loop investigation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/discussions/39)

Research on reasoning-model loops finds that larger models loop less, not that parameter count guarantees immunity. It distinguishes repeated learned errors from mere sampling behavior; higher temperature can reduce loops while leaving inefficient or incorrect reasoning. Separately, experimental work on sentence repetition documents that previous repetitions can raise the probability of another repetition. This is the relevant feedback mechanism: generated text becomes input to the next prediction. Neither paper establishes the causal internals of this particular checkpoint. [Reasoning-loop research](https://arxiv.org/abs/2512.12895), [sentence-level self-reinforcement](https://arxiv.org/abs/2206.02369)

**The diagnosis supported by the evidence is intact weights plus a vulnerable generation setup, with runtime correctness still needing resolution.**

| Candidate explanation | Evidence and status |
|---|---|
| Corrupt or missing stored weights | Complete fresh integrity check passed; strongly disfavored |
| Greedy decoding contributes | Confirmed incident setting; plausible causal contributor; changed as requested |
| Disabled reasoning hurts review quality | Confirmed setting; earlier 2,048-token fixture missed a defect off and found it with reasoning; limited sample |
| Native numerical execution contributes | Historical whole-model discrepancy confirmed; incident causality and current-binary behavior unresolved |
| Guard lets degeneration continue | Directly reproduced on captured output |
| Inherent model vulnerability | Supported by external reproductions; exact contribution here unmeasured |
| DSpark bug | Inapplicable to this native path because speculation is disabled |
| Prompt/context overflow | Exact request fits the observed serving window; no supporting evidence |

The next decisive experiment is a controlled replay of the same review evidence at 0/1 and 1/0.95, keeping thinking, prompt, engine, and weight hashes fixed and recording duplication, valid JSON, verdict quality and termination. A separate thinking-mode comparison can follow. Before attributing residual failures to the learned weights, run fresh teacher-forced logit comparisons on the current binary at multiple positions, including just before the first duplicate; validate the CPU reference adapters against official kernels; and trace the first material layer/operation divergence. These experiments require additional inference and were not performed in this audit.

A repetition guard that catches repeated findings and nonadjacent repeated spans can bound wasted review time independently. It should terminate or retry a failed review rather than treating a forced stop as an approval. That is a separate implementation change from the temperature adjustment requested here.

All new evidence is in this directory: `weight-integrity.json`, `analysis.json`, `guard-replay.json`, their scripts, and the two temperature-test logs. Only the requested sampling change was made to production source during this investigation.
