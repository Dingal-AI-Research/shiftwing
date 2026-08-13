# colib Research Log

This directory contains focused experimental reports that support the phase
papers and gate decisions.

## Retrospective phase papers

The completed Phase 0–5 studies are grouped in
[past_phases/](past_phases/README.md):

1. [Reproducible inference foundation](past_phases/phase0_reproducible_foundation.md)
2. [Independent oracle and tokenizer](past_phases/phase1_oracle_and_tokenizer.md)
3. [CPU numerical correctness](past_phases/phase2_cpu_correctness.md)
4. [Exact CPU optimization](past_phases/phase3_cpu_optimization.md)
5. [Full-model conversion and validation](past_phases/phase4_full_model_validation.md)
6. [Persistent CUDA acceleration](past_phases/phase5_cuda_acceleration.md)

## Reporting rule

Every performance or numerical experiment that may change a phase decision
must record:

1. the research question and hypothesis;
2. the exact implementation or instrumentation change;
3. hardware, model, prompt/data, warmup, and environment controls;
4. correctness checks proving that the intended path executed;
5. measured results, including negative results;
6. limitations and possible confounding variables;
7. the engineering decision: retain, revise, or revert.

Run-to-run timing differences are not attributed to a feature unless telemetry
confirms that the feature executed.

Status statements inside an individual report are contemporaneous: “gate
open” means open at that experiment's decision boundary. Later experiments
and the phase summaries below provide the authoritative retrospective
disposition. Earlier negative or pending statements are preserved rather
than rewritten after the fact.

## Phase 6 reports

The main paper is [Lossless MTP speculative decoding](../phase6_mtp.md).

1. [Expert-route overlap and mixed-precision MoE batching](phase6_experiment_01_route_overlap.md)
2. [Multi-domain MTP acceptance and exact CUDA batching](phase6_experiment_02_multidomain.md)
3. [Expert-cache attribution and MTP pinning](phase6_experiment_03_cache_attribution.md)
4. [Confidence-aware speculative admission](phase6_experiment_04_confidence_admission.md)
5. [Accepted-path verification profile](phase6_experiment_05_accepted_path_profile.md)
6. [Batched int8 shared expert and Gate-6 closure](phase6_experiment_06_shared_expert_gate_closure.md)

Phase 6 is complete. Experiment 6 records the CUDA event study, format audit,
bit-exact int8 shared-expert batch kernel, and the passing 1.394-fold gate
result.

## Phase 7 reports

1. [397B FP8 checkpoint inventory](phase7_preflight_01_checkpoint_inventory.md)
2. [397B tier and state resource plan](phase7_preflight_02_resource_plan.md)
3. [Real official FP8 tensor probe](phase7_preflight_03_real_fp8_probe.md)
4. [Production conversion pilot](phase7_preflight_04_conversion_pilot.md)
5. [Tiered direct-storage pipeline and learning cache](phase7_experiment_05_tiered_storage_pipeline.md)
6. [Real-container storage throughput](phase7_experiment_06_real_container_io.md)
7. [Equivalent-capacity cache-locality proxy](phase7_experiment_07_cache_locality_proxy.md)
8. [Predictive read/compute overlap A/B](phase7_experiment_08_predictive_prefetch.md)
9. [Preregistered 397B qualification protocol](phase7_preflight_05_qualification_protocol.md)
10. [Complete 397B streaming conversion](phase7_experiment_09_conversion_completion.md)
11. [Complete-model qualification](phase7_experiment_10_complete_model_qualification.md)
12. [Corrected cache telemetry and stream-ordered CUDA allocation](phase7_experiment_11_cache_telemetry_and_async_allocator.md)
13. [I/O, upload staging, and decode-only attribution](phase7_experiment_12_io_upload_and_decode_attribution.md)
14. [Decode-set protection and the capacity boundary](phase7_experiment_13_decode_cache_protection.md)
15. [Low-bit routed-expert sidecars and the 32 GiB no-go](phase7_experiment_14_low_bit_expert_sidecars.md)
16. [User-authorized relaxed 3-bit profile](phase7_experiment_15_relaxed_q3_profile.md)

The preflights pin the official revision, validate all 94,078 FP8 scale grids,
and replace the initial memory estimate with an executable plan. Gate 7 uses a
no-MTP, one-slot profile because the original MTP + 7 GiB cache plan would
require 23.65 GiB VRAM.

Experiment 5 validates the layer-batched `io_uring`/`O_DIRECT` path, exact
fallbacks, storage telemetry, and persistent expert heat map on the tiny
oracle. Experiment 6 tests real converted expert payloads and revises the
necessary warmed non-disk hit rate to approximately 83–91% at the bandwidth
observed during conversion. Experiment 7 uses Qwen35 as a controlled
equivalent-capacity proxy: the Gate-7-sized fraction predicts only about
71–73% warmed hits, placing the throughput target at risk. Full-model
throughput remains a Gate-7 measurement. Experiment 8 rejects unfiltered
route-transition prefetch: despite exact tokens and real overlap, its 12.85%
precision increased I/O and slowed the 35B control. The mechanism remains
opt-in behind a confidence threshold and is disabled for the initial Gate-7
run. Preflight 5 freezes the complete-model integrity, perplexity, memory,
warm-up, telemetry, coherence, and sustained-throughput criteria before the
94-shard conversion completes. Experiment 9 records the completed
212,634,789,241-byte container, exact preflight-byte agreement, 90-shard
resume-integrity sweep, MTP-only source shard, source/output manifest
distinction, and passing structural/resource diagnostic.
Experiment 10 records the first complete-model result: exact integrity,
PPL 1.0191, coherent output, guarded memory, and resident CUDA all pass, but
the preregistered measured decode rate is only 0.3701 tok/s. Experiment 11
corrects tier and decode-only cache accounting, persists the learned atlas on
graceful mux shutdown, rejects a host-exclusive cache policy, and isolates a
2.22-fold allocator improvement under identical routes and read bytes.
Stream-ordered allocation raises the bounded result to 0.7468 tok/s, so Gate
7 remains open on storage/cache locality rather than correctness.
Experiment 12 rejects persistent `io_uring` and pinned upload staging as
defaults, then corrects the phase boundary with decode-only `DPERF`
telemetry. Storage consumes 3.758 of 5.251 decode seconds and expert
upload/compute consumes 1.451 seconds; attention plus the language-model head
consume only 0.041 seconds. At the measured compute cost and storage rate,
the 2 tok/s target requires approximately 96% non-disk hits.
Experiment 13 proves that prefill pollution is actionable for short repeated
turns: phase-aware protection plus host materialization eliminates all
four-token decode reads and reaches 2.736 tok/s. The same policy reaches only
0.772 tok/s over 64 tokens, with 12,454 misses and 83.25 GB read. The
reference machine therefore remains below Gate 7 because its sustained route
working set exceeds the available 18 GiB/6 GiB cache union.
Experiment 14 implements and quality-gates grouped 2-bit and 3-bit expert
sidecars before spending the much larger 397B conversion cost. Two-bit
agreement is only 9.38%. Three-bit reaches 95.23% aggregate agreement but
fails one preregistered prompt at 82.81%; early/late mixed-precision splits
also fail. The formats remain reproducible experiments, but neither is
accepted. With unchanged precision requiring more RAM, Gate 7 is closed as
a measured 32 GiB hardware no-go, not passed.
After reviewing the measured 0.94-point aggregate and 4.69-point weakest-
prompt reductions, the project owner accepted this trade-off in principle.
Experiment 15 records the concurrency pilot and two completed atomic files,
then retires the full Qwen397 conversion because Ornith is the production
target. The quantizer and acceptance controls move forward only as a possible
Ornith35-gated treatment for Ornith397.

## Phase 8 reports

1. [Ornith registry, FP8 layout, and agent protocol](phase8_preflight_01_ornith_registry_and_protocol.md)
2. [OpenAI-to-Ornith tool-call history round trip](phase8_experiment_02_tool_history_round_trip.md)
3. [End-to-end Ornith tool gate](phase8_preflight_03_e2e_tool_gate.md)
4. [Official Ornith-35B GGUF numerical reference](phase8_preflight_04_official_gguf_reference.md)
5. [One-million-token stretch disposition](phase8_preflight_05_1m_stretch_disposition.md)
6. [Complete Ornith-35B streaming conversion](phase8_experiment_06_ornith35_conversion.md)
7. [Ornith-35B numerical and tool-use qualification](phase8_experiment_07_ornith35_numerical_and_tool_gate.md)
8. [Ornith-397B streaming conversion pilot](phase8_experiment_08_ornith397_streaming_pilot.md)
9. [Ornith-397B qualification protocol](phase8_preflight_06_ornith397_qualification_protocol.md)
10. [Ornith-35B grouped-3-bit quality control](phase8_experiment_09_ornith35_q3_control.md)
11. [Lossless Ornith-397B warm route atlas](phase8_experiment_10_warm_route_atlas.md)
12. [CUDA dense-host release](phase8_experiment_11_cuda_dense_host_release.md)
13. [Disjoint expanded warm atlas](phase8_experiment_12_disjoint_expanded_atlas.md)
14. [Revised Ornith397 production threshold](phase8_preflight_07_revised_production_threshold.md)
15. [Revised-threshold production qualification](phase8_experiment_13_revised_threshold_qualification.md)
16. [Owner-accepted Ornith397 grouped 3-bit profile](phase8_preflight_08_owner_accepted_q3.md)
17. [Ornith397 q3 qualification automation](phase8_preflight_09_q3_qualification_automation.md)
18. [Native q3 and adaptive hot-route atlas](phase8_preflight_10_native_q3_hot_route_atlas.md)
19. [Q3 release-policy amendment and reproducible CUDA path](phase8_experiment_14_q3_release_amendment.md)
20. [Q3 cold-start coherence failure](phase8_experiment_15_q3_cold_start_coherence.md)
21. [Repaired Q3 Gate-8 controls and throughput retry](phase8_experiment_16_repaired_q3_gate8_retry.md)

The first preflight pins both official model revisions, validates their
compressed-tensors FP8 inventories, and corrects the roadmap from Hermes JSON
to the released Qwen3 XML tool protocol. Experiment 2 closes a protocol
boundary discovered before model conversion: OpenAI wire-format argument
strings are normalized to mappings only while rendering Ornith's official
Jinja history. The official 35B template now passes a call/result/continuation
round trip. Preflight 3 connects that exact snapshot renderer to the live
OpenAI/Anthropic gateway and CLI and freezes a real HTTP generated-call,
tool-result, and final-answer harness; model-generated tool selection remains
part of Gate 8. Preflight 4 pins the publisher's official Q4_K_M artifact and
freezes per-prompt teacher-forced and perplexity thresholds without
downloading its 21.17 GB payload during the Qwen397 conversion. Preflight 5
closes the optional 1M stretch as a non-target: no official checkpoint or
YaRN configuration exists, and one 397B slot would require 60 GiB fp32 / 30
GiB bf16 KV before weights and caches.
Experiment 6 converts all 16 pinned Ornith-35B FP8 shards into an independently
audited 19,081,810,684-byte container. Exact agreement with the metadata-only
prediction closes structural conversion. Experiment 7 closes the 35B
numerical and deployment-behavior sub-gate: teacher-forced agreement is
96.25% with all 20 prompts above threshold, perplexity differs from the
publisher GGUF by -0.00943%, and the live HTTP model selects a Paris weather
tool, consumes its 18 °C result, answers coherently, and stops cleanly.
Gate 8 now proceeds directly to Ornith-397B. Experiment 8 commits the first
of 122 production shards, validates the real per-channel FP8 conversion at a
14.1 GiB peak, and proves atomic source cleanup plus hash-bound resume before
the remaining long transfer. By the 50-shard checkpoint it has committed
87,624,947,869 payload bytes and 114,360 physical tensors. Its post-conversion
supervisor is manifest-bound and resumable, runs the hash audit, perplexity,
tiered-throughput, and generated-tool checks sequentially, and intentionally
stops before the separately controlled Gate-9 serving experiments.
Experiment 9 follows the qualification protocol's conditional lower-bit
branch after the direct profile and two non-lossy locality pilots establish
expert-I/O capacity as the remaining bottleneck. It preregisters a relaxed
Ornith35 token, perplexity, coherence, and generated-tool control before any
Ornith397 3-bit sidecar may be created.
Experiments 10 and 11 then reject two lossless capacity shortcuts. Complete
route instrumentation raises the deterministic working set from the partial
2,934-entry visualization to 4,779 pairs. Dense-host release safely frees
5.266 GiB and grows RAM to 64 experts/layer, but independent tiers duplicate
880 entries and throughput collapses despite fewer reads. Experiment 12 is the
bounded synthesis: its exact guard finds that 66 host slots per layer still
require 1,088 pinned device experts, 135 more than the safe device capacity.
It rejects before preload and closes residency-only remediation negatively on
the reference machine.
After that negative closure, the project owner revised the Ornith397 product
requirement from 2.0 to 0.85 token/s. Preflight 7 records this explicitly as a
post-result protocol amendment. It does not promote the short 0.867369 pilot;
it freezes a new full four-prompt optimized qualification and leaves every
numerical, tool, Gate-9, and release requirement unchanged.
Experiment 13 executes that amendment. The optimized lossless profile improves
the original full baseline from 0.527355 to 0.631964 token/s, but the four
measured turns remain below the revised 0.85 threshold. All structural,
coherence, resource, persistent-I/O, pinned-upload, and CUDA-residency controls
pass. Gate 8 closes negative again, so the ordered tool and Gate-9 sequences
remain prohibited.
Preflight 8 records the owner's subsequent decision to accept the measured
Ornith35 int3 quality class in exchange for capacity. It prospectively replaces
the 5% PPL rule only for the explicitly lossy Ornith397 int3 profile with a
12% ceiling, retains the token/coherence/tool/CUDA controls, and keeps the
0.85 token/s production threshold. It also authorizes deletion of retired
full-size Qwen artifacts while preserving the tiny correctness oracle.
Preflight 9 turns that amended product rule into a fail-closed experiment. An
independent program re-derives all 480 files and 276,480 tensor headers before
hashing them, while a separate controller binds quality and speed artifacts to
the base container, q3 sidecar, int4 PPL baseline, CUDA executable, and exact
commands. It also inserts a mandatory semantic review after four fixed outputs
because nonempty text alone does not establish coherence.
Preflight 10 records the owner's production selection of q3 after the complete
expanded-q4 profile reached 0.718433 token/s sustained and 0.865836 token/s at
the median. It implements native packed-q3 CUDA storage and an opt-in adaptive,
disjoint RAM/VRAM hot-route atlas. Its CUDA kernels now pass focused tests, but
full-model validation remains open and the accepted expanded-q4 path stays the
release default.
Experiment 14 encodes the later owner-approved Ornith397 PPL waiver and 0.70
token/s sustained floor without rewriting historical evidence. It adds an
explicit expanded-q4/native execution switch, validates the amended controller
and 20-check auditor, and launches the resumable final Gate-8 sequence.
Experiment 15 records the first semantic-review stop in that sequence. The
structural and teacher-forced controls pass, but the zero-warmup four-output
run is garbled despite complete CUDA-residency telemetry. The controller is
correctly left unacknowledged while a single-variable two-warmup diagnostic
tests the strongest observed cold/warm difference.

## Phase 9 reports

1. [Mux byte-framing preflight](phase9_preflight_01_mux_framing.md)
2. [Slot scheduler state preflight](phase9_preflight_02_scheduler_state.md)
3. [Hybrid recurrent/KV session-state preflight](phase9_preflight_03_session_state.md)
4. [Atomic disk checkpoint preflight](phase9_preflight_04_disk_checkpoint.md)
5. [Exact-extension prefix-reuse preflight](phase9_preflight_05_exact_extension.md)
6. [Functional mux engine preflight](phase9_preflight_06_functional_mux.md)
7. [OpenAI gateway and Qwen protocol parity](phase9_experiment_07_openai_gateway.md)
8. [Session-switch cost and reusable buffers](phase9_experiment_08_session_switch_cost.md)
9. [Resident batched GDN rows](phase9_experiment_09_resident_gdn_rows.md)
10. [Resident batched GQA rows](phase9_experiment_10_resident_gqa_rows.md)
11. [Resident hybrid layer](phase9_experiment_11_resident_hybrid_layer.md)
12. [Whole-model resident batch](phase9_experiment_12_whole_model_resident_batch.md)
13. [Warm session reload](phase9_experiment_13_warm_session_reload.md)
14. [Runtime telemetry](phase9_experiment_14_runtime_telemetry.md)
15. [CUDA-resident multi-slot GDN](phase9_experiment_15_cuda_resident_gdn.md)
16. [CUDA-resident multi-slot GQA](phase9_experiment_16_cuda_resident_gqa.md)
17. [Web and CLI surface](phase9_experiment_17_web_cli_surface.md)
18. [Real 35B web qualification and prefix reuse](phase9_experiment_18_35b_web_and_prefix_reuse.md)
19. [CUDA events and live continuous-batch admission](phase9_experiment_19_cuda_events_and_live_batch.md)
20. [Cancellation latency with a live peer slot](phase9_experiment_20_cancellation_latency.md)
21. [Resident CUDA activation graph](phase9_experiment_21_resident_cuda_graph.md)
22. [Production cancellation qualifier](phase9_experiment_22_cancellation_qualifier.md)
23. [Ornith production serving qualification](phase9_preflight_07_ornith_production_qualification.md)
24. [Q3 manifest-bound production harnesses](phase9_preflight_08_q3_manifest_binding.md)
25. [Gate-9 production serving execution](phase9_experiment_23_gate9_production_execution.md)

The first server preflight implements and tests the transport boundary without
claiming that continuous batching exists. Exact byte counts allow prompts to
contain newlines, while payload-boundary failures close the connection instead
of parsing payload bytes as a later command. The second preflight proves
request identity and slot lifecycle independently of inference, including the
required persist-before-release cancellation boundary. The third preflight
saves and restores the CPU target model's hybrid recurrence/KV state and
regenerates an eight-token continuation exactly on CPU and CUDA. Device state
uses explicit transfers; MTP state remains explicitly rejected until the
drafter section is complete. The fourth preflight serializes that target state
with version/config guards, a checksum, fsync, and atomic rename; exact reload
passes and deliberate corruption is rejected. The fifth preflight proves that
restoring an exact prefix and consuming only its suffix gives the same
eight-token continuation as fresh full prefill on CPU and CUDA. The sixth
preflight connects these components to `SERVE_BATCH=1` and passes streaming,
cancellation, and slot-reuse integration tests on the tiny engine; its decode
rows are still sequential, so continuous batching remains open. The seventh
experiment adapts the upstream dependency-free HTTP gateway, byte-compares its
Qwen3.5 tool-history rendering with the official snapshot template, and tests
both ordinary and streamed OpenAI responses over a real local socket. The
eighth experiment removes per-token snapshot allocation and quantifies why the
remaining state copies must be replaced by resident batched state. The ninth
adds the first such compute seam: multiple GDN rows share batched projections
while retaining independent recurrent state. The tenth gives full attention
the same row ownership with independent GQA KV caches. The eleventh composes
normalization, attention, residuals, and grouped MoE into a complete resident
transformer-layer operation. The twelfth extends that ownership through the
whole model and connects it to the mux as an opt-in CPU continuous-batch path.
The thirteenth persists token history and resident state atomically and proves
an exact-extension continuation across engine processes.
The fourteenth emits hardware, tier, route-heat, hit, and request-timing
telemetry through both CPU and CUDA mux builds and connects it to the gateway's
observability state.
The fifteenth moves slot-major GDN convolution/recurrent state onto CUDA,
batches the active rows' projections, and preserves exact checkpoint reload.
The sixteenth moves slot-major GQA KV onto CUDA with a memory guard, completing
device ownership of persistent target-session state.
The seventeenth vendors and rebrands the browser client, adds the installable
launcher/doctor surface, and tests bundle-to-gateway-to-engine operation.
The eighteenth qualifies that complete path on the real 35B container, fixes
the official-history versus cached-token rendering boundary, and records the
negative short-suffix block-replay result instead of claiming a TTFT gain.
The nineteenth fixes a persistent-pipe stdio/poll deadlock found by live
simultaneous submissions, adds opt-in CUDA-event profile suffixes and a batch
benchmark harness, and records the contended 35B negative result.
The twentieth measures cooperative cancellation while another slot continues,
proves immediate slot reuse, and records a 9.88-second contended 35B
cancel-to-ack observation for later warm comparison.
The twenty-first keeps the mux residual stream, normalizations, hybrid
attention, routed/shared experts, residual additions, final norm, and LM head
on CUDA. Only router logits return to CPU for top-k/tier control before final
model outputs; explicit transfer counters distinguish this graph-resident path
from state residency.
The twenty-second turns the cancellation probe into a manifest-pinned,
warm-up-aware, machine-checked qualifier. Its first real 35B validation proves
one-piece cancellation, peer isolation, slot reuse, and 120/120 device-MoE
layer-forwards, while explicitly deferring uncontended p95 timing.
The production preflight also provides a manifest-bound ten-step controller
for both Ornith models. It refuses to start before the independent Gate-8
evidence is green, requires explicit review acknowledgement, and stops before
Phase-10 release actions.
Experiment 23 completes that authorized sequence. Ornith35 passes its ABBA
batch, 20-trial cancellation, exact web-history, and CUDA-residency controls.
A system-Python launcher defect initially prevents Ornith rendering, is
diagnosed from durable server stderr, and is repaired by selecting the
workspace virtual environment. After two owner pauses at atomic evidence
boundaries, Ornith397 restarts A/B from the beginning and passes both orderings,
the geometric-mean comparator, 20-trial cancellation, exact two-turn web
history, and zero-host-fallback CUDA residency. The controller and independent
q3 release audit finish passed at 20/20 checks.

## Phase 10 reports

1. [Header-exact container diagnostics](phase10_experiment_01_container_doctor.md)
2. [Reproducible build and benchmark environment](phase10_experiment_02_reproducible_environment.md)
3. [Composite release gate](phase10_experiment_03_release_gate.md)
4. [Release evidence audit](phase10_preflight_04_release_evidence_audit.md)
5. [Negative release disposition](phase10_experiment_05_negative_release_disposition.md)
6. [Q3 release evidence audit](phase10_preflight_06_q3_release_audit.md)
7. [Q3 positive release disposition](phase10_experiment_07_q3_release_disposition.md)
8. [Ornith35 explicit-int4 default and throughput record](phase10_experiment_08_ornith35_int4_reversion.md)

The first hardening experiment upgrades `colib doctor` from presence checks to
index/header ownership, bounds, overlap, shard, and exact payload-byte
validation without reading model data.
The second records the exact compiler, CUDA, Python, web, hardware, build,
test, and benchmark-control environment and aligns README status with measured
Gate-9 progress.
The third expands `make check` into a clean portable, tokenizer, Python, web,
audit, and CUDA gate. Its first execution exposed and corrected a
conditional-scoping bug that had silently skipped CUDA. The final post-Gate-9
rerun passes 21 C executables, 137 Python tests, 10,000 tokenizer cases,
18 web tests, the production build, a zero-vulnerability audit, and both CUDA
suites. It also rejects broken local Markdown links and unindexed research
reports; the current live audit covers 91 documents, 111 links, and 75 reports.
An index-driven source-package step rejects tracked ELF/PE executables, model
weights, compiled objects, engine binaries, and generated web/model
directories; four fixtures and the complete 273-path prospective source index
after final evidence staging pass. The corresponding cached diff also passes
Git's whitespace check.
The fourth joins both pinned Ornith manifests to 13 Gate 8/9 artifacts,
including the 122-shard header/ledger/SHA-256 doctor result, recomputes the
production thresholds, and preserves `make check`, the release commit, and
`v0.1` as separate final requirements.
The fifth applies that frozen audit after the complete Ornith397 study. Seven
of 15 checks pass, but direct throughput fails at 0.527355 token/s and the
ordered tool and Gate-9 steps remain unavailable. It verifies that the Gate-9
controller refuses to bypass those prerequisites, closes the roadmap without
weakening the 2 token/s threshold, and deliberately withholds `v0.1`.
This is the historical disposition for the original requirement. The later
owner-approved 0.85 token/s amendment is recorded prospectively in Phase-8
Preflight 7; it reopens qualification without altering Experiment 5's result.
The sixth adds a separate 20-check q3 audit profile without modifying the
historical 15-check result. It binds the complete sidecar, independent doctor,
numerical/coherence/tool/tier stages, explicit coherence review, and every
Gate-9 artifact to the exact q3 manifest.
The seventh applies that profile to the completed q3 evidence chain. It retains
the historical int4 rejection, records the passing Gate-8 and Gate-9 results,
regenerates the q3 audit at 20/20, and records the green final repository
check. It separates this evidence acceptance from the remaining clean release
commit and `v0.1` tag actions.
The eighth makes the Ornith35 launcher's int4 default immune to ambient
low-bit research variables and records a fresh three-turn, 64-token
directional measurement at 40.416843 token/s sustained with zero host-MoE
fallback.
