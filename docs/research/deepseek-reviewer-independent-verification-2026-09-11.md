# Independent verification of overnight DeepSeek results

Date: 2026-09-11.

**Verified: the current native engine produced five complete fixture reviews that
identify the planted defects, including the exact 32,768-token input. It is not
qualified under the agreed requirements and is not the active LocalForge reviewer.**

This audit reads the original native logs and saved completions, independently
recounts all five prompts, recomputes saved logit comparisons, reruns 18 focused
oracle/qualification tests, and executes the current binary's ordinary startup
gate. It does not repeat the roughly three-hour capability inference run.
Machine-readable results are in
[deepseek-reviewer-independent-verification-2026-09-11.json](deepseek-reviewer-independent-verification-2026-09-11.json).

## What happened overnight

Checkpoint recovery finished. The saved independent byte validation covers
91 groups and 166,878,536,440 payload bytes. Its manifest, state and proof-ledger
hashes still match; every segment's size/inode/device/mtime matches the saved
validation. This audit did not rehash all 167 GB.

Early review attempts failed on a stale executable and then an invalid READY
handshake: a human banner was written to protocol stdout. The banner now goes to
stderr. The _GNU_SOURCE/O_DIRECT include-order fix is present in the entry point
and execution header.

The formal 20-minute run subsequently completed the 512-token case and cancelled
the 2,048-token case. Its report is explicitly failed. The later capability run
allowed 90 minutes per case, used low reasoning, 40,960 context and 8,192 output
budget, and completed all five on one loaded 43-layer engine.

## Independently checked capability results

| Input tokens | Prefill seconds | Decode tokens/s | Complete review minutes | Defect found; EOS |
| ---: | ---: | ---: | ---: | --- |
| 512 | 136.7 | 0.349 | 16.0 | Yes |
| 2,048 | 221.4 | 0.359 | 28.2 | Yes |
| 8,192 | 552.8 | 0.354 | 32.8 | Yes |
| 16,384 | 985.2 | 0.396 | 43.1 | Yes |
| 32,768 | 1936.1 | 0.405 | 51.1 | Yes |

All native input counts agree with fresh native-tokenizer counts and the pinned
reference tokenizer's complete token-ID sequence. Saved request rendering and
input hashes agree with native payload hashes. All five native records bind to
the current engine SHA-256
`a066822435d644e6ca7df2b446d460657bc23d84bb6edf09c581336d6700d7eb`
and current checkpoint manifest
`c84796f773e5b3942e4ba5200a480b7d1ba1c7add7e87cbb7e959271a1270a47`.

I inspected the final JSON answers as well as rerunning their defect oracle.
They correctly explain the equal-to-length index, negative transfer amount,
missing tax-rate cache key, sibling-directory prefix bypass, and unconditional
commit after partial failure. All five native records terminate on EOS.

The table's prefill column is **not an exact client TTFT measurement**: that
statistic was discarded by the capability script. The earlier formal 512-token
run retained an actual TTFT of **129.0 seconds** and decode speed of **0.354
tokens/s**, against a previous engine hash. The capability run's current-engine
prefill ranges from 136.7 seconds to 1,936.1 seconds (32.3 minutes). Speeds include
thinking and the final answer, using generated tokens divided by native decode
seconds; EOS is counted.

The native allocation records pass the existing memory checker: peak device
accounting about 11.98 GiB, scratch 0.437 GiB, peak process RSS 22.93 GiB versus
27.73 GiB initially available. These are reported native peaks, not a separate
external continuous memory trace.

## Where the other model's interpretation overreaches

**“Quality does not degrade with context length” is not established.** There is
one simple planted defect at each size; the longer context is largely unrelated
helper functions and the selected source is at the end. This establishes five
successful examples, not broad review accuracy, resistance to distractors, or
stable performance across positions and lengths. There are no clean-code control
cases in this ladder to assess false rejection.

**Changing the time bar alone would not finish qualification.** All five observed
wall times are indeed below one hour, so the review timing condition would pass
under that alternative. But full-model numerical parity is still failed, the
required full prefill/chunk acceptance sweep has no completed report, and there
is no aggregate qualification artifact.

**The logit discrepancy remains real.** I recomputed it from both saved native
logit files against the saved reference: relative L2 0.102731, cosine 0.994712,
maximum absolute error 2.994095, and native top token 979 versus reference 270.
The near-tie explanation is a hypothesis about the cause; it does not make the
existing parity gate pass or demonstrate the absence of an implementation error.
The queue script exits zero after writing this failed comparison, so a file in
`.nightqueue/done/` means the measurement finished, not that parity passed.

**“Already fits production” describes observed request duration, not a tested
production workflow.** The capability job directly constructs the experimental
native Engine. It does not exercise LocalForge's managed orchestrator unload,
reviewer handoff, GPU release, cancellation recovery and restoration.

**The one-hour production premise is stale in the current workspace.** The
handoff required keeping one hour, but current LocalForge source defaults and
the saved `colib.auditTimeoutMs` are both 10,800,000 ms (three hours). The source
default was already present in the repository baseline; this audit does not
attribute the discrepancy to the overnight DeepSeek changes. All five measured
durations still happen to be below one hour.

## Current production state, freshly verified

- Running the current binary with the normal DeepSeek profile and
  `DSV4_EXPERIMENTAL=0` returns **78**, reports production locked, reads zero
  weight bytes and does not load GPU weights.
- `c/deepseek-v4-flash-0731/review-qualification.json` is absent; the qualification
  loader rejects it.
- LocalForge's saved `auditAgentModel` is **glm53-flash**, with the GLM engine
  path. Its DeepSeek candidate profile explicitly forces
  `DSV4_EXPERIMENTAL=0`, so selecting it currently encounters that lock.
- At the process check, there was no DeepSeek inference process and GPU usage
  was zero. No services, saved settings, startup lock or acceptance limits were
  modified during this audit.

The nonthinking shortcut is also unsuitable on the measured example: its saved
complete JSON approves the negative-transfer defect. That demonstrates one miss,
not a universal conclusion about nonthinking mode.

## Evidence

- [Overnight capability native log](../../.nightqueue/logs/160-deepseek-full-ladder-capability.sh.log)
- [Capability run script](../../.nightqueue/done/160-deepseek-full-ladder-capability.sh)
- [Failed formal review report](deepseek-review-acceptance-2026-09-11.json/reviews.json)
- [Failed full-model logit comparison](deepseek-full-logit-parity-2026-09-11.json)
- [Checkpoint byte-validation report](deepseek-recovery-validation-2026-09-10.json)
- [Original overnight findings](deepseek-reviewer-acceptance-findings-2026-09-11.md)

The defensible conclusion is **demonstrated experimental review capability,
with 16–51 minute fixture runs**. Production readiness requires further work;
changing a time limit or approving the startup patch alone is insufficient.
