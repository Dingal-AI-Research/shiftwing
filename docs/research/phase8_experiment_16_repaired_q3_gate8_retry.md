# Experiment 16: Repaired Q3 Gate-8 Controls and Throughput Retry

## Abstract

This experiment re-runs the Ornith397 grouped-three-bit (q3) Gate-8 controls
after repairing a representation-dispatch defect in resident CUDA prefill. The
new binary-bound teacher-forced, free-running coherence, and generated-tool
controls pass. The first complete production-tier run remains coherent but
fails the owner-approved 0.70 token/s sustained threshold because one of four
measured turns slows to 0.243102 token/s between three turns at 0.785044 to
0.945905 token/s. Component timing shows simultaneous storage, matrix
multiplication, and language-model-head slowdowns, followed by immediate
recovery. This report preregisters one unchanged, complete retry after an idle
system-health check. The retry is not allowed to change prompts, omit a slow
turn, use the median as the gate, or lower the threshold.

## 1. Research question

After correcting the expanded-q4 prefill dispatch, does the selected
Ornith397 q3 profile satisfy all frozen Gate-8 correctness, tool-use, CUDA
residency, and sustained-throughput requirements on the reference machine?

The hypothesis is that the isolated slow measured turn in the first repaired
production run is transient host or storage contention rather than a stable
model-route regression. This is testable because a stable regression should
persist into later turns, while transient contention may recover without a
configuration change.

## 2. Methods

The tested engine is the repaired CUDA executable with SHA-256
`d33ee2123ad0190b356daa3a19d07af561057f57b41ddfef23da420f15f26c5b`.
The base Ornith397 container and complete q3 sidecar remain fixed. The q3
sidecar manifest has SHA-256
`5231fbe79bc9edf27b86222e4df503066a0b8f04f02fc612ab993696096b0180`
and contains 480 files, 276,480 tensors, and 157,073,113,440 data bytes.

The production profile uses eight CPU threads, 18 GiB routed-expert RAM,
6 GiB routed-expert VRAM, persistent `io_uring`, pinned upload, decode-cache
protection and prewarming, and q3 experts expanded to the validated q4 CUDA
representation. Native packed-q3 and the adaptive route atlas remain disabled.
Each complete tier trial performs two four-prompt warm-up passes and then one
four-prompt measured pass. Every prompt generates 64 tokens. The acceptance
metric is total measured completion tokens divided by total measured wall
time; it must be at least 0.70 token/s.

The first repaired production trial is retained unchanged as
`c/ornith397_q3_qualification.json`, SHA-256
`1a5a178a2d744dcfb4d220b6cb436e8e69f4d70ed0722189c1f069a69e61cd5e`.
Before the retry, the reference WSL machine showed 28 GiB available RAM,
513 GiB free storage, load averages 0.11/1.14/1.68, an idle RTX 5070 Ti at
47 degrees Celsius, and no recent NVMe, GPU-reset, or out-of-memory kernel
errors. No model server or benchmark process remained active.

Only one full retry is authorized. It must use the controller's exact frozen
command and overwrite the tier artifact atomically. The controller may reuse
earlier passing artifacts only when their recorded inputs and hashes match.
If the retry fails again, Gate 8 remains open for reliability diagnosis rather
than repeated attempts until a favorable sample appears.

## 3. Correctness results before the tier retry

The regenerated teacher-forced q3 comparison matches 1,243 of 1,280 positions
(97.109375%). All 20 prompts exceed the 85% per-prompt floor; the weakest
prompt matches 59 of 64 positions (92.1875%). Its artifact SHA-256 is
`462dce23cdf046d08cc7f263f97fd8c3a086bcc925add361acddb632c136c5ae`.

The repaired zero-warmup four-prompt coherence run answers all four prompts
meaningfully. It sustains 0.665842 token/s during this cold correctness control,
executes 15,360 of 15,360 MoE layer-forwards on CUDA, and records zero host-MoE
fallback. Its artifact SHA-256 is
`21bc37fd15102e60eb2960163057608eb25654e5298a4e97b0b84e15f3dacd83`.
The semantic review was explicitly acknowledged.

The generated HTTP tool control selects
`get_weather({"city":"Paris"})`, consumes the deterministic result of 18
degrees Celsius and clear conditions, answers that Paris is clear at 18
degrees Celsius, and stops without a second tool call. It executes 2,580 of
2,580 MoE layer-forwards on CUDA with no host fallback. Its artifact SHA-256
is `051917aa2e3fda7c5f0e4651ce590c1d65753ab155905648e5376b96c04e2e75`.

These results close the correctness regression found in Experiment 15. They
do not by themselves satisfy the production throughput requirement.

## 4. First repaired production-tier result

The first complete repaired run produced the following measured turns:

| Prompt | Token/s | TTFT (s) | Wall time (s) | Cache hit (%) |
|---|---:|---:|---:|---:|
| Hash-table explanation | 0.945905 | 76.647 | 144.347 | 65.2422 |
| TypeScript grouping | 0.243102 | 68.918 | 332.263 | 63.2682 |
| Service diagnosis | 0.807838 | 138.210 | 217.478 | 63.6797 |
| RAM versus NVMe | 0.785044 | 91.606 | 173.178 | 58.0417 |

The aggregate sustained rate is 0.520672 token/s, the median turn rate is
0.796441 token/s, and the minimum is 0.243102 token/s. Therefore the frozen
sustained gate fails even though the median exceeds the threshold. All text
remains coherent, all 46,080 MoE layer-forwards execute on CUDA, and host-MoE
fallback remains zero.

The slow TypeScript turn spends 178.963 seconds reading experts and 119.139
seconds in expert matrix multiplication, compared with approximately 76--110
and 32--60 seconds, respectively, on surrounding stable turns. Its decode-only
storage and matrix times are 150.872 and 110.081 seconds. The language-model
head also rises to 1.432 seconds from approximately 0.21 seconds. The cache-hit
rate and bytes read are not unusually adverse, and the next turn recovers to
0.807838 token/s. Because storage, GPU computation, and the language-model head
slow together, the evidence is consistent with system-wide transient
contention or throttling rather than an additional q3 representation defect.

## 5. Decision rule

The first run is a real failure and remains part of the research record. One
unchanged complete rerun is justified by the isolated, multi-component outlier,
immediate recovery, previously accepted 0.718433 token/s full profile, and idle
pre-run health check. Gate 8 passes only if the new complete artifact satisfies
the same 0.70 token/s sustained rule and all correctness/residency checks. A
second failure is evidence that this hardware profile has not established the
required production reliability.

## 6. Retry result

A first operational attempt was deliberately interrupted at the project
owner's request after three of eight warm-up turns and during the fourth
warm-up turn. The completed warm-up rates were 0.590, 0.810, and 0.769
token/s. No measured turn ran, no qualification artifact was published, and
the prior failed artifact remained byte-identical. Consequently that
interruption supplied no acceptance evidence and the complete protocol was
restarted from its beginning.

The complete retry began only after Dota 2 was closed and Windows GPU-engine
activity fell from approximately 32% to less than 5%. At the final pre-run
check, Linux reported 28 GiB available RAM, the GPU reported 1% utilization,
222 MiB used VRAM, and no stale model process.

The second warm-up pass produced 0.809576, 0.841949, 0.743861, and 0.703504
token/s. The measured results were:

| Prompt | Token/s | TTFT (s) | Wall time (s) | Cache hit (%) |
|---|---:|---:|---:|---:|
| Hash-table explanation | 0.858887 | 82.880 | 157.435 | 65.3021 |
| TypeScript grouping | 0.857416 | 76.494 | 151.177 | 63.3125 |
| Service diagnosis | 0.865132 | 91.217 | 165.238 | 63.7240 |
| RAM versus NVMe | 0.773247 | 81.065 | 163.872 | 58.1953 |

The complete aggregate sustained rate is 0.836866773 token/s, the median turn
rate is 0.8581515 token/s, and the minimum turn rate is 0.773247 token/s. All
four measured turns individually exceed the 0.70 token/s production floor.
The TypeScript prompt that previously fell to 0.243102 token/s instead reaches
0.857416 token/s, so the earlier isolated slowdown does not reproduce.

All 46,080 MoE layer-forwards execute on CUDA and none fall back to host MoE.
The run records complete telemetry, an active resident graph, coherent text,
and no acceptance failure. The production artifact SHA-256 is
`7d4d1fe8add387fbdd274a6bd287629023b76c7562a67ac328ffc731507c4106`;
the completed Gate-8 pipeline ledger SHA-256 is
`88f96c755132bfba9117ce4ab74876ec56bb041a70f918acb58c56b32cce8f29`.

## 7. Conclusion

The repaired Ornith397 q3 profile passes Gate 8 under the frozen production
protocol. The result does not erase the first failed run; instead, the paired
evidence shows that this storage-heavy profile can experience severe transient
slowdowns while its uncontended complete retry exceeds the owner-approved
floor with a 19.6% sustained margin. Production monitoring should therefore
retain per-turn latency and component timing even though the acceptance gate
is now closed. Gate 9 is authorized against this exact engine and q3 manifest.
