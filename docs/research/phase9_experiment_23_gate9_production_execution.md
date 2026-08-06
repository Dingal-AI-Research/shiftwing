# Experiment 23: Gate-9 Production Serving Execution

## Abstract

This experiment executes the manifest-bound Gate-9 serving protocol after the
selected Ornith397 q3 model passed Gate 8. Both Ornith35 and Ornith397 q3 pass
sequential/concurrent order reversal, exact output, cooperative cancellation,
CUDA residency, and two-turn web history. The first Ornith35 web attempt
exposed an environment defect: the launcher selected system Python, which
lacked Transformers and returned HTTP 500 before inference. The launcher now
selects the workspace virtual environment, and the repeated web step passes.
Ornith397 execution was then interrupted twice before an A/B artifact could be
published; each resume correctly hash-verified completed evidence and restarted
the atomic A/B step. The final uninterrupted run published all five Ornith397
artifacts. Its reversed-order batch geometric mean is
1.1098070927775951, cancellation p95 is 1.9357337760002338 seconds, both web
turns return exactly `colib ready`, all measured MoE forwards remain on CUDA,
and the final q3 release audit passes 20/20. Gate 9 therefore passes.

## 1. Research question

Does the production server preserve exact model output and request lifecycle
while serving multiple slots, reversing sequential/concurrent benchmark order,
cancelling one live request without harming its peer, reusing the released
slot, and maintaining a two-turn web conversation on both production models?

The hypothesis is that the resident CUDA graph and mux scheduler satisfy these
properties without host-MoE fallback. Gate 9 is not a raw single-request speed
test: it tests whether batching, cancellation, history reuse, and HTTP serving
remain correct together.

## 2. Methods

The controller is bound to the repaired CUDA engine SHA-256
`d33ee2123ad0190b356daa3a19d07af561057f57b41ddfef23da420f15f26c5b`,
the exact Ornith35 manifest, the Ornith397 base manifest, and q3 sidecar
manifest SHA-256
`5231fbe79bc9edf27b86222e4df503066a0b8f04f02fc612ab993696096b0180`.
An explicit Gate-8 acknowledgement was supplied only after the complete q3
tier artifact passed at 0.836866773 token/s sustained.

For each model, the frozen sequence runs batch A/B, batch B/A, an ABBA
comparator, 20 cancellation trials, and a two-turn exact web control. The batch
comparators require at least 0.95 times speedup in each ordering and at least
1.00 geometric-mean speedup. Cancellation p95 must be no more than 1.0 second
for Ornith35 and 3.0 seconds for Ornith397. Every exact-output and CUDA
residency check must also pass.

Ornith35 uses 8 GiB expert RAM and 6 GiB expert VRAM. Ornith397 uses its
complete q3 sidecar, 18 GiB expert RAM, and 5 GiB expert VRAM for the batch
protocol. The controller atomically publishes an artifact only after a whole
step succeeds and hash-binds each passing artifact before resuming.

## 3. Ornith35 batch results

Batch A/B produced a concurrent/sequential speedup of 1.008884810. Reversing
the order produced 1.036471812. The ABBA comparator therefore reports a
1.022585286 geometric mean and 1.008884810 minimum. Both outputs match the
required `colib batch ready` text. The comparator passes its 1.00 geometric
mean and 0.95 per-order limits.

The three artifacts and their SHA-256 values are:

| Artifact | SHA-256 |
|---|---|
| `ornith35_batch_ab.json` | `81158b57cc83488cd0c97b9fb91ecbf390cd71b916e96322b3312a28be00f75f` |
| `ornith35_batch_ba.json` | `52079bb6f39149aedba51ce56c0d13050ea7c5d81e30e9ec9555fe5ca1c19d1a` |
| `ornith35_batch_abba.json` | `ec98cf98c18a3348440ce272ef7cddd0c5921373838ff080d9c2ebcde58f8381` |

## 4. Ornith35 cancellation result

All 20 trials observe exactly one streamed piece before cancellation, allow the
peer to finish, and reuse the cancelled slot immediately. Cancel-to-acknowledge
samples range from approximately 0.032 to 0.170 seconds. The nearest-rank p95
is 0.162146312 seconds, well below the 1.0-second ceiling. All 8,480 measured
MoE layer-forwards use CUDA and none use host MoE. The artifact SHA-256 is
`0091fe4173c1de26b1416fefe845d9170d1426a51bd64833d755805e38b904ea`.

## 5. Web failure, diagnosis, and repair

The first Ornith35 web attempt returned HTTP 500 before any model matrix
multiplication. The smoke harness originally attached server stderr to an
unread pipe and discarded it on request exceptions. It now writes stderr to a
seekable temporary file and prints it only when the step fails. The repeated
failure then reported `No module named 'transformers'` and zero inference calls.

The `colib` launcher already computed the workspace Python path for conversion,
but `_server_argv` and `command_serve` used `sys.executable`. Because the CLI's
shebang selected system Python, the Ornith renderer could not import the pinned
Transformers package. The server launcher now uses the same workspace Python
selection as conversion. A regression test asserts that the server argument
vector begins with the selected workspace interpreter. Eight focused CLI/web
tests pass after the change.

The repeated two-turn web control passes. Both responses equal `colib ready`,
the scheduler returns to zero active and queued requests, and all 1,120 MoE
layer-forwards use CUDA with zero host fallback. History reuse reduces observed
TTFT from 9.252 to 1.739 seconds. The artifact SHA-256 is
`b174733a6d9cd3a7b2c2f31ea4a22ef479ccdad0d0155df39d580c802dbaec81`.

## 6. Ornith397 interruption history and completed batching

The Ornith397 q3 A/B step began with a 13.20 GiB VRAM requirement against
14.66 GiB startup availability. Its first internal ordering completed 1,800 of
1,800 MoE layer-forwards on CUDA with zero host fallback and a 61.93% combined
RAM/VRAM expert-hit rate. The project owner then requested a pause while the
paired ordering was starting. The controller and model processes exited, GPU
utilization returned to zero, and no `ornith397_batch_ab.json` was published.

The incomplete internal ordering is diagnostic only. On continuation, the
controller must hash-verify the five completed Ornith35 artifacts and restart
the complete Ornith397 A/B step. It must not synthesize an artifact from the
single internal ordering or skip directly to B/A.

A later resume verified an idle GPU and all five Ornith35 hashes, then restarted
the complete A/B step as required. The owner requested another pause during its
first internal ordering. Again, the process exited before the step published
`ornith397_batch_ab.json`; therefore the evidence boundary and required restart
behavior were unchanged at that interruption boundary.

The final resume again verified the bound manifests, q3 sidecar, CUDA engine,
commands, and every previously published artifact, then restarted A/B from its
beginning. The uninterrupted A/B and B/A steps both returned the exact required
`colib batch ready` output and passed CUDA-residency checks. Their speedups and
the comparator result are:

| Measurement | Result |
|---|---:|
| A/B speedup | 1.1234715068953394 |
| B/A speedup | 1.0963088744307583 |
| Geometric mean | 1.1098070927775951 |
| Minimum ordering | 1.0963088744307583 |

The minimum exceeds the 0.95 per-order floor and the geometric mean exceeds
the 1.00 aggregate floor. The atomically published batch artifacts are:

| Artifact | SHA-256 |
|---|---|
| `ornith397_batch_ab.json` | `e703bc9343d0f78794049fc96c52ba4386c77a84ae94a4a5f2e4e3709b2cc1ee` |
| `ornith397_batch_ba.json` | `625554c513a6b5875ba6456a6d3e9d09c60e2aa7cc7184fd34e0735e36c82eb6` |
| `ornith397_batch_abba.json` | `b9f95d9462f33d530370b5929a9bb11044ccbf71692782b525f9b5870ea41032` |

The two interruptions remain part of the execution history, but neither
contributes partial data to these results: no artifact existed at either pause,
and the accepted evidence comes only from the later complete atomic steps.

## 7. Ornith397 cancellation result

All 20 trials cancel one request after exactly one streamed piece, allow its
peer to complete, and immediately reuse the released slot. Cancel-to-acknowledge
latency ranges from 1.05909308700029 to 1.95773938999992 seconds. The
nearest-rank p95 is 1.9357337760002338 seconds, below the 3.0-second ceiling.
All 12,720 measured MoE layer-forwards use CUDA and none use host MoE. The
artifact SHA-256 is
`606c4383e341e09db746f117b95f2bc1c83286837df19a6f87b667cb858d69ce`.

## 8. Ornith397 web result

The exact two-turn production web control passes. Both responses equal
`colib ready`, and the scheduler finishes with zero active or queued requests.
Startup takes 303.2187 seconds; observed TTFT is 160.1793 seconds on the first
turn and 74.1480 seconds on the history-bearing second turn. All 480 measured
MoE layer-forwards use CUDA with zero host fallback. The artifact SHA-256 is
`e5aa4d5ee2211686bfeb4af00b2bacfa50a3c1ef12b4256d085f1fb91a0dbb36`.

## 9. Final audit and conclusion

The completed `ornith_gate9_pipeline.json` records `status: passed`, all ten
frozen production steps as passed, and the same manifest, q3-sidecar, engine,
command, and artifact bindings used by the controller. Its final q3 release
audit passes 20/20 checks with zero failures, including every Ornith397 batch,
cancellation, and web evidence requirement represented by the five published
artifacts.

Gate 9 passes. The production server preserves exact output, peer isolation,
cancelled-slot reuse, two-turn history, and device-resident MoE execution on
both production models while meeting every frozen batch and cancellation
threshold. The earlier launcher failure and owner interruptions are retained
as provenance, but the repaired launcher and later complete atomic runs replace
the provisional state with final accepted evidence. The audit still declares
green `make check` and an intentional clean release commit/tag as separate
release conditions; those conditions do not change this Gate-9 result.
