# Shiftwing Research Log

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

## Phase 11 reports

1. [DeepSeek-V4 pinned specification and storage preflight](phase11_preflight_01_deepseek_v4_spec_and_storage.md)
2. [DeepSeek-V4 native loader, CUDA correctness, and serving metrics](phase11_preflight_02_native_loader_cuda_and_metrics.md)
3. [Host-storage recovery and Ornith35 reconstruction record](phase11_experiment_03_storage_recovery.md)
4. [Pinned source fetch completion](phase11_experiment_04_pinned_fetch_completion.md)
5. [Fresh Ornith397 control and retirement record](phase11_experiment_05_ornith397_control_and_retirement.md)
6. [Real conversion, resume, and native-byte audit](phase11_experiment_06_real_conversion_and_validation.md)
7. [First real 43-layer CUDA forward](phase11_experiment_07_first_real_forward.md)
8. [Cold/warm real-model profile](phase11_experiment_08_cold_warm_profile.md)
9. [Pinned/direct matched A/B](phase11_experiment_09_pinned_direct_ab.md)

The first preflight freezes the DeepSeek-V4-Flash-0731 source identity and
inference-critical configuration, adds a secret-redacted storage/hardware
gate, and implements native-byte aligned conversion with atomic resumability.
Its fixture suite passes, while the real 48-shard inventory, engine,
correctness, quality, paired-performance, and promotion gates remain open.

The second preflight records the dtype/shape-aware native segment loader, the
complete scalar expert oracle, the SM120 grouped FP4 expert path with the
official route-before-`w2` quantization boundary, and authoritative
`colib.metrics` web consumption. These are fixture-level correctness results;
the real 43-layer path, tensor-core performance qualification, and promotion
remain gated.

The third report identifies Windows host-volume exhaustion as the cause of the
interrupted source fetch, preserves exact Ornith35 source/manifests/hashes, and
records deletion of only its regenerable weights. An ext4 trim plus direct
`CompactVirtualDisk` call reclaims 53,822,357,504 host bytes without enabling
WSL's unsafe sparse mode; Ornith397 remains available for its fresh control.

The fourth report closes the real pinned-source sub-gate. A third hash-bound
attempt resumes from 58/62 and commits all 48 shards; a separate full-file pass
accepts 72,317 tensors, all 43 base expert layers, all three bundled DSpark
expert layers, and 166,878,536,440 indexed payload bytes. Full conversion is
not authorized at the resulting host free-space level; Ornith397 remains
protected until its fresh control is recorded.

The fifth report completes five process-fresh current-branch Ornith397
controls on the exact historical four-prompt subset. All automatic CUDA,
resident-graph, output, and telemetry gates pass; median sustained decode is
0.824033698 token/s and median trial TTFT is 43.508444 seconds. It preserves
the exact trial/controller hashes, 122 base-shard hashes, 480 q3-shard hashes,
source identity, environment, and reconstruction commands in compact tracked
artifacts. This is the Ornith side only; no DeepSeek paired speedup or
promotion is claimed.

The sixth report closes Gate 11.0 on the real pinned model. A deliberate
nine-segment interruption resumes to all 91 segments, and a dependency-bound
validator compares all 166,878,536,440 native payload bytes with the source
while rehashing 72,317 records and 166,881,088,004 aligned segment bytes. It
also corrects the official BF16 confidence-head contract and makes full
descriptor validation mandatory before future real conversion writes. Live
runtime, quality, paired-performance, and promotion gates remain open.

The seventh report records the first manifest-bound forward through all 43
base layers of the real converted checkpoint. The SM120 runtime returns finite
logits after loading exactly 258 routed experts and attributes every dense,
storage, and CUDA-upload counter. This closes only the narrow real-weight
forward smoke: initialization takes 334.191103 seconds and cold decode takes
118.610136 seconds (0.008431 tok/s), so the performance gate fails and the
default remains unchanged. The next experiment must isolate the warm cached
decode and profile host work between CUDA calls before mux or paired
qualification claims.

The eighth report retains the process for a second token and adds per-layer,
per-kernel-family, storage, and cache telemetry. The resident token improves
from 134.734045 to 84.532588 seconds, but changes 230 of 258 routed experts
and reads another 3.075 GB. Dense CUDA projections account for only 1.913
seconds. The evidence therefore rejects cache sizing alone and selects pinned
upload staging plus persistent direct I/O as the next controlled optimization.

The ninth report A/B tests that intervention on the identical two-token
sequence. Pinned staging and direct/persistent I/O preserve exact tokens and
logits, halve initialization, and improve the cold token, but the resident
token regresses from 84.532588 to 96.703443 seconds. The combined path is
therefore rejected for decode qualification; a short storage-mode benchmark
and coalesced expert extents are required before another full-model run.

## Phase 12 reports

1. [Ornith397 reactivation preflight](phase12_preflight_01_ornith397_reactivation.md)
2. [Expanded-q4 route-atlas candidate](phase12_experiment_02_expanded_q4_route_atlas.md)
3. [LocalForge carry-forward](phase12_handoff_03_localforge_carry_forward.md)
4. [Ordered grouped-prefill expert I/O](phase12_experiment_04_ordered_prefill_expert_io.md)
5. [Paired AB/BA performance controller](phase12_preflight_05_paired_ab_controller.md)
6. [Frozen expert-map cold-start preload](phase12_preflight_06_frozen_expert_map_preload.md)
7. [Storage-bound optimization candidates](phase12_preflight_07_storage_bound_optimization_candidates.md)
8. [Long-context ingestion](phase12_preflight_08_long_context_ingestion.md)
9. [Prefill compute candidates for 10-16k context](phase12_preflight_09_prefill_compute_candidates.md)
10. [8,451-token best-stack long-context qualification](phase12_experiment_10_long_context_best_stack.md)
11. [Batched full-attention prefill](phase12_experiment_11_batched_prefill_attention.md)

The first Phase-12 report records the owner's DeepSeek rejection, revalidates
the compact Ornith397 reconstruction bundle, and starts the exact pinned,
shard-resumable reconstruction. LocalForge remains documentation-only
carryover and is not modified.

The second report records the first RTX 5070 Ti optimization candidate.
Routed experts remain q3 on NVMe and in RAM, but atlas-selected VRAM copies
expand to the faster q4 CUDA representation. Focused Python and SM120 CUDA
correctness tests pass; the preregistered five-pair real-model comparison
waits for byte-exact reconstruction.

The third report fixes the future LocalForge integration contract without
changing any LocalForge source, configuration, or deployment. It names the
model identity, streaming metrics event, context and sampling defaults,
session/cancellation requirements, and prompt-probe boundary that must be
revalidated after Ornith performance qualification.

The fourth report records a second reversible RTX 5070 Ti candidate. Grouped
prefill preserves expert-id order and CPU q3 math while submitting up to four
consecutive cold experts to the persistent ring together. A tiny exact-output
control keeps 44 hits, 30 misses, and 300 reads unchanged while reducing ring
submissions from 50 to 41. The default remains one expert per submission; the
five-pair real Ornith397 TTFT protocol waits for reconstruction and cannot
inherit the route-atlas result.

The fifth report adds the controller the preregistered five-pair protocol
requires and that the repository did not have. Pairs alternate AB/BA, optional
priming trials absorb cold-cache warm-up, every artifact is hash-verified and
resumable, and promotion needs identical outputs plus a 95% decode lower bound
above 1.0 and a TTFT upper bound below 1.0. Seventeen focused tests pass and
the extractor reproduces the preserved control aggregates exactly; no
real-model result is claimed.

The sixth report records the owner's cold-start preload design and measures
what it can deliver. The cache holds 12.85% of experts covering 51.8% of
routing mass, so preloading 80% of experts is not physically possible and the
achievable target is filling every cache slot with the hottest experts. Once
warm, cache contents are already within ~1.5 points of a perfect same-size
selection, so the win is cold start rather than throughput: the 5.22-point
first-trial gap. The engine's heat map, which previously never survived a run
because `SIGTERM` skips `atexit`, now checkpoints on turn boundaries and can be
frozen as a read-only benchmark input.

The seventh report measures where decode time actually goes and proposes two
candidates against it. Expert disk I/O is 70.2% of decode, 1,195 MB per token
at an effective 1.40 GB/s with 5.11 MB per miss, so removing all disk time
would still leave a 2.77 tok/s ceiling. Candidate 4 asks whether the read path
leaves throughput unused and is gated on an O_DIRECT sweep before any work.
Candidate 5 demotes the rarely-routed tail to 2-bit to cut bytes per miss by a
third, at a quality cost that must clear the teacher-forcing and perplexity
gates. A hardware note records that the board has two DIMM slots, both
occupied, and what cache capacity each upgrade would buy.

The eighth report addresses the owner's intent to load 10,000-16,000 tokens on
the first request, which changes the dominant cost. A 38-token turn touches
2,684 distinct experts, 8.7% of the total, but a 10k prompt saturates the
expert set, so prefill must read nearly the whole 157 GB sidecar: about 137 GB
of misses, a floor of roughly 98 seconds at the measured 1.40 GB/s. Expert-major
grouped prefill turns out to be on by default already. The report adds prefix KV
caching as candidate 6, rescopes the ordered-prefill flag to long context, and
notes that the frozen preload helps far less here because prefill needs every
expert regardless of heat.

The ninth report measures the 10-16k reviewer target and plans against it. Time
to first token is 11.2 minutes at 2,335 tokens and 33.0 minutes at 8,451, with
16k projecting past an hour, so the gap to a 10-minute budget is 4x to 7x.
Prefill is 91% compute: cold-expert matmul on the host is 14.0 minutes, the 45
linear-attention layers 9.0, the 15 full-attention layers 6.8, and disk only
3.0 and flat past ~2k tokens. It records the split timers and recorded CUDA
flags that make the three candidates testable, including why environment
variables could not work, and withdraws an attention fix that would have moved
work off the GPU.

The tenth report qualifies the stacked long-context configuration on the frozen
8,451-token fixture. Time to first token falls from 32m57.830s to 19m19.300s,
41.3853% lower, with byte-identical output and the load pipeline hiding 94.91%
of producer load time behind consumer compute. It is a single stacked arm
against a preserved baseline, not a factorial attribution, and it misses the
strict 18-minute threshold by 79.299940 seconds.

The eleventh report reopens the attention candidate the ninth had closed.
Prefill was calling the single-token decode path once per position, launching 32
blocks on a 70-multiprocessor device at about 0.1% of roofline. Batching the
rows without touching the arithmetic cuts full attention 9.709x on the real
model, from 421.850426 to 43.450325 seconds, and lowers TTFT 33.75% from
19m19.300s to 12m48.086s with byte-identical output. Attention is the only stage
that moves materially, making it a cleaner attribution than the stacked
best-stack arm. It also finds that the attention score buffer overflowed the
48 KB shared-memory cap at position 12,256, which disabled CUDA for the whole
run and made the 10-16k target workload unservable. The first full-model attempt
was a negative result in which the new code never executed; it is preserved and
reported.

## GLM-5.3-Flash

1. [Engine qualification and LocalForge wiring](glm53_flash_engine_qualification.md)

The deep review lane moves off Ornith-397B, which was proven correct on the
review task and then measured at 139 minutes for a single 16K review. The GLM
engine is checked against a fixture from HuggingFace's own `Glm5NextTextModel`
and keeps its tokens exact at int4 -- the property Qwen3.8-Flash-Next lacked --
and the converter's on-disk format, the failure point of the Qwen attempt, is
checked rather than assumed. Four faults were found on the way, three of them
in wiring rather than in the model: no build rule, a link that survived only
because the compiler deleted the call, a settings normaliser that silently
discarded every colib lane but Ornith, and an engine budget knob whose name
differs per engine. Speed on the real checkpoint is still unmeasured.
