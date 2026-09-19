# GLM review streaming and standalone smoke test

Shiftwing emits progress while GLM reads the prompt, then streams thinking and the final answer separately. A client must wait for a normal completion before accepting a review.

## Progress contract

Every progress frame is an SSE `data:` object with `object: "shiftwing.progress"`, `schema_version: 1`, and the HTTP request ID. LocalForge also accepts the older `colib.progress` name.

| Phase | Fields | Meaning |
| --- | --- | --- |
| `prefill` | `event`, `prompt_tokens_total`, `prompt_tokens_cached`, `prompt_tokens_prefilled`, `elapsed_ms` | `begin`, intermediate `progress`, then `end`. Prefilled tokens include the cached prefix. The count increases only after all processing for a chunk finishes. |
| `prefill` | optional `progress_percent` | Progress within a chunk. GLM counts each completed transformer layer plus the output projection as one unit per token. This measures completed processing stages, not elapsed or remaining time. It does not claim partially processed tokens are fully cached. |
| `decode` | `completion_tokens`, `max_tokens`, `elapsed_ms` | Native generated-token count and the effective output cap. Thinking and response tokens share this cap. `100 * completion_tokens / max_tokens` is **output budget used**, not the percentage of the eventual answer or time remaining. |

The native pipe uses `PREFILL_BEGIN`, `PREFILL_PROGRESS`, `PREFILL_END`, and `DECODE_PROGRESS`. The optional sixth field on `PREFILL_PROGRESS` is percentage in thousandths of a percentage point, from 0 to 100000. The original five-field format remains supported. Progress is emitted after each completed layer and whole chunk without changing the batch size or model calculations.

Actual thoughts arrive in `choices[].delta.reasoning_content`; final answer text arrives in `choices[].delta.content`. A legacy leading `.` reasoning delta is a keepalive. Repeated progress snapshots also keep the HTTP stream active; they do not extend the native stall deadline unless computation advances.

## Timeouts and completion

The defaults are:

- `FIRST_MODEL_OUTPUT_TIMEOUT_MS`: **3600000 ms / one hour** until the first generated model bytes.
- `PREFILL_FIRST_PROGRESS_TIMEOUT_MS`: **900000 ms / 15 minutes** until prompt processing advances.
- `PREFILL_PROGRESS_STALL_TIMEOUT_MS`: **900000 ms / 15 minutes** between genuine prompt-processing advances.

Operators can override these environment variables. They are first-output and prefill-stall limits; cancellation of an entire review remains controlled by the caller.

An engine failure after SSE headers produces an in-band `error` object followed by `[DONE]`. It must never write a second HTTP response into the stream. A successful review needs nonempty final content, `finish_reason: "stop"`, and `[DONE]`. Reject `length`, disconnects, engine errors, missing finish reasons, and incomplete or invalid review JSON.

## Run a real smoke test without LocalForge

Use one terminal to start only Shiftwing. This example uses the actual CPU GLM model, 14 GB expert cache, one slot and 64K context. Ensure other large managed models have been unloaded first.

```sh
cd /home/dinga/Projects/shiftwing
make -C c glm53
SHIFTWING_ENGINE="$PWD/c/glm53" \
FIRST_MODEL_OUTPUT_TIMEOUT_MS=3600000 \
PREFILL_FIRST_PROGRESS_TIMEOUT_MS=900000 \
PREFILL_PROGRESS_STALL_TIMEOUT_MS=900000 \
c/shiftwing serve --model "$PWD/c/glm53_i4" \
  --host 127.0.0.1 --port 18080 --model-id glm53-flash \
  --context 65536 --max-tokens 16384 --kv-slots 1 \
  --ram-gb 14 --cuda-expert-gb 0 \
  --session-dir /tmp/shiftwing-review-smoke-sessions
```

After `/v1/models` is ready, run in another terminal with a new output directory:

```sh
python3 c/tests/smoke_glm53_review.py \
  --base-url http://127.0.0.1:18080 \
  --output-dir /tmp/shiftwing-review-smoke-001
```

The script requests thinking and a short structured review of an intentionally unsafe approval rule. It makes no project edits, runs no orchestrator, and imports no LocalForge code. It checks the native prefill/decode counts, effective output cap, thinking, final JSON, normal finish, and `[DONE]`. Its process exit code is nonzero on failure.

The output directory contains:

- `request.json`: exact model request, including thinking and token cap.
- `wire.sse`: complete received SSE bytes.
- `timeline.jsonl`: every parsed event with elapsed seconds.
- `thinking.txt` and `response.txt`: separate streamed text.
- `summary.json`: timings, token usage, terminal status and pass/fail result.

The intentionally unsafe scaffold must receive `changes_required` with correction notes. The test does not interpret that verdict as a transport failure.

## Fast regressions

```sh
.venv/bin/python -m unittest discover -s c/tests -p test_glm53_review_stream.py -v
cd c/tests
../../.venv/bin/python -m unittest test_openai_server_glm53 test_openai_server_qwen -v
```

The first command uses the small native GLM fixture to check real pipe telemetry and HTTP failure framing. It does not replace the real-model smoke test.

## Review context mismatch fixed on 2026-09-08

The LocalForge managed launcher supplied `CTX=65536` in the environment but omitted `--context` from `shiftwing serve`. The Shiftwing CLI defaults that argument to 4096 and explicitly writes both `CTX` and `GLM53_MAXT` from it. Consequently the review prompt builder budgeted for 65,536 tokens while the native GLM slot had only 4,096. An oversized request was rejected before `PREFILL_BEGIN`, and its generic `BAD_REQUEST` became the unhelpful engine-failed message in the accordion. This code path matches the reported zero prefill/decode progress after model loading; the old generic error did not retain the actual token count.

LocalForge now passes the configured context as an explicit `--context` argument. The GLM engine also accepts `CTX` when `GLM53_MAXT` is absent for direct API launches. Requests that fill or exceed the native slot emit `CONTEXT_EXCEEDED` with the token-count lower bound and actual context limit; the existing API translation carries that explanation into the review SSE error. Known malformed requests and protocol frames now retain their specific reasons instead of becoming a generic server error.

These edits do not approve or bypass a failed audit. At initial implementation, the first-output/progress deadlines remained 15 minutes. Normal-completion requirements remain in place. At initial implementation, no review, test suite, or application restart was run. The subsequently authorized tests are recorded below. The native engine was rebuilt separately for the next launch; the LocalForge launcher change takes effect at its next normal restart.

## Authorized test runs on 2026-09-08

After the user requested test runs, 52 focused tests passed: 42 LocalForge tests covering launch configuration, production inference, audit handling, and streamed reviews; and 10 Shiftwing tests covering the native engine and SSE transport. LocalForge TypeScript checking also passed. The updated launch test explicitly checks that a GLM context of 65,536 is passed on the command line. The native fixture rejects a 4,097-token prompt at a 4,096-token context and completes it at 65,536, reproducing the old configuration failure and verifying the fix.

The separate real GLM run used the actual LocalForge launch builder with an isolated port/session directory, thinking enabled, an 8,192-token output budget, and all three timeout settings at 900,000 ms. The CLI translation and running native process both confirmed a 65,536-token context. Its 5,258-token prompt was accepted and emitted 528 prefill events, but the review **failed** after 902.62 seconds with `first_model_output_timeout`. The last reading progress was 17.569% (896 fully processed tokens; percentage also includes progress within the current chunk). No thinking or answer tokens arrived.

The SSE stream returned the specific timeout error and `[DONE]`, with no successful finish reason. No approval was accepted. This is a separate remaining blocker: the absolute deadline for the first generated output expires even while CPU prompt processing continues to advance. The context fix does not make this real review pass, and the 15-minute settings were not changed.

The owned smoke-test server was stopped. LocalForge was not started or restarted, and no LocalForge end-to-end task ran. The launcher change still takes effect at the next normal LocalForge launch.

## First-output timeout increased to one hour on 2026-09-08

After the real review timed out during active prompt reading, the user requested a one-hour timeout. Shiftwing now defaults `FIRST_MODEL_OUTPUT_TIMEOUT_MS` to `3600000` (one hour), and LocalForge's saved environment explicitly uses the same value. The standalone smoke client's socket timeout is also one hour. First-progress and stalled-progress limits remain 15 minutes, and LocalForge's configured overall audit limit is three hours. Thus a review may spend up to an hour reading an advancing prompt before its first generated output.

The 15-minute smoke result above is historical evidence from before this change; it is not a one-hour benchmark. No model or LocalForge restart was performed. The setting takes effect when the processes are next normally launched.

## One-hour rerun stopped for performance investigation

The user stopped the real-model rerun after 48m 44.8s at 51.968% reading progress. The first-output hour did not expire; the terminal engine error was caused by terminating the owned test process at the user's request. The 5,258-token synthetic prompt produced no thinking or answer text. The process reported about 2.3 TB of storage reads, including model loading, and no process swap. Full findings and ranked proposals are in Shiftwing's `docs/research/glm53-prefill-bottleneck-2026-09-08.md`. LocalForge was not restarted, and no additional model run was started.

## Batched GPU prompt reading

The GLM CUDA build now accelerates batched prompt calculations while preserving
the same native prefill/progress protocol. See
[GPU prefill controls and measured limitations](research/glm53-gpu-prefill-2026-09-08.md).
`GLM53_CUDA=0 GLM53_BATCH_MATH=0` restores scalar CPU math without rebuilding.
The one-hour first-output timeout and strict completion/approval checks are unchanged.

## Larger prompt chunks

The native GLM default is now 1024 prompt tokens per chunk, with only
the final position scored during served prefill. Layer progress and strict
review completion checks are preserved. See [measurements and overrides](research/glm53-larger-chunks-2026-09-08.md).
