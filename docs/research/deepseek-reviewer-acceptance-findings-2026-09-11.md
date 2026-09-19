# DeepSeek reviewer: measured acceptance status

Date: 2026-09-11. Supersedes the performance sections of
`deepseek-reviewer-model-handoff-2026-09-09.md`, which had no complete-model
numbers. Everything below is measured on the complete 43-layer model.

## Headline: the model reviews correctly at every required length

Measured end to end with the 20-minute cancel removed (a capability
measurement, not an acceptance run -- the acceptance driver is untouched):

| case | prompt | wall clock | output tokens | planted defect found |
| --- | ---: | ---: | ---: | --- |
| upper-bound | 512 | 16.0 min | 286 | yes |
| negative-transfer | 2048 | 28.2 min | 528 | yes |
| incomplete-cache-key | 8192 | 32.8 min | 501 | yes |
| path-containment | 16384 | 43.1 min | 634 | yes |
| partial-transaction | 32768 | **51.1 min** | 457 | yes |

**All five find their planted defect, including the actual 32,768-token case.**
The reviewer works. What it misses is the 20-minute qualification bar, not the
task.

Two consequences.

The capability question is settled: quality does **not** degrade with context
length, so "accept a longer bar" is a real option rather than a hopeful one.
Until this ran it was unverified, and offering it as an option was premature.

Every case fits the **one-hour production timeout the handoff says to keep**
(51.1 minutes is the worst case). So a reviewer qualified against the
production timeout rather than a separate 20-minute gate passes the whole
ladder today, with no engineering change at all.

What remains true is that against the stated 20-minute bar, only 512 tokens
passes. That is a specification decision, and it is now the only thing standing
between this model and production use.

## Result against the 20-minute bar: one of five cases passes

| case | prompt | prefill | decode | total | outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| upper-bound | 512 | 128.6 s | 807.4 s / 286 tok | **936.4 s** | **passes**, terminated on EOS |
| negative-transfer | 2048 | 220.1 s | 982.1 s / 367 tok | 1202.4 s | cancelled at the 20-minute limit |
| incomplete-cache-key | 8192 | 536.1 s | — | ~25 min projected | not reached |
| path-containment | 16384 | ~17 min projected | — | ~33 min projected | not reached |
| partial-transaction | 32768 | ~32 min projected | — | ~48 min projected | not reached |

The requirement is five complete reviews, each within 20 minutes, including an
actual 32,768-token input. That is **not met on this hardware.**

Quality is not the problem. The 512-token case found its planted defect, named
the file and line, and terminated on EOS inside the budget. The 2048-token case
was generating a correct review when the clock cut it off.

## The blocking bug that was fixed

`deepseek_v4.c` included system headers before `st.h` could define
`_GNU_SOURCE`. glibc locks its feature-test macros at the first system header,
so `O_DIRECT` was never visible in that translation unit, the O_DIRECT twin
descriptors became -1, and **every expert read silently fell back to buffered**:

    before: direct_bytes=0   direct_fallbacks=2140  uring_batches=0  uring_fallbacks=340
    after:  direct_bytes=26.84 GB  direct_fallbacks=0  uring_batches=340  uring_fallbacks=0

`qwen.c` and `glm53.c` define `_GNU_SOURCE` before any include; this file did
not. The same defect reached every probe through `deepseek_v4_execute.h`, which
includes `<sys/resource.h>` ahead of the chain to `st.h`, so the partial-model
numbers in the older handoff were also taken on buffered reads.

Worth 2.95x on decode (0.1105 -> 0.3259 tok/s) and 1.25x on prefill
(243.8 -> 195.1 s at 2048 tokens), with **output bit-identical** before and
after: `top=273 logit=26.759634` to nine significant figures.

## Why the remaining gap is not I/O

Measured prefill, 43 layers, direct I/O:

    512 tokens -> 127.1 s     2048 -> 193.9 s     8192 -> 536.1 s

Phase split at 8192 tokens:

    attention   283.0 s  52.8%
    routed      137.9 s  25.7%
    read         79.2 s  14.8%
    route/shared 35.9 s   6.7%

Scaling 2048 -> 8192 (4x tokens): attention 4.19x, routed 2.74x, read 1.18x.
Reads plateau because a long prompt touches all 256 experts in every layer, so
there is nothing left to read; attention is linear and already the majority.

At 32,768 tokens reads are about 87 s of a ~32-minute prefill. Eliminating
**all** remaining I/O changes no verdict in the table. The gap is roughly 2.5x
of attention and routed-expert compute.

## Levers tested and rejected

| lever | result |
| --- | --- |
| io_uring workers 4 -> 12 | worse: 0.281 -> 0.252 tok/s |
| prefill chunk 512 / 1024 / 2048 | no effect: 195.1 / 195.1 / 202.8 s |
| host expert cache 8 -> 12 -> 14 GiB | more hits, fewer bytes, **slower**: 0.281 -> 0.204 tok/s |

The cache result is the instructive one: the routed working set is far larger
than any cache this host can hold, and the added memory pressure costs more
than the avoided reads save.

## Complete-model logit parity

Native vs the unchanged pinned reference on identical tokens:
relative L2 0.1027, cosine 0.9947, **different argmax** (native 979, reference
270). Before reading that as a broken head: hidden states track the reference
at cosine >= 0.9993 through all 43 layers (0.99996 at layer 42), the scales
agree (std 5.52 vs 5.56), and both rank the same three candidates --

    reference  270 (24.060)  264 (23.802)  979 (23.750)
    native     979 (24.246)  264 (23.789)  270 (23.572)

The reference's top-1 margin is 0.258 logits against a 5.56 standard
deviation, about 0.05 sigma. This is accumulated arithmetic difference flipping
a genuine near-tie, not a structural fault. It remains a real obstacle to
bit-level agreement and should not be called resolved.

## Nonthinking mode: fast, and wrong

The profile permits omitting reasoning effort ("Omitted effort is
nonthinking"), and cost here is per token, so fewer output tokens means less
time. Measured on the failing 2048-token case, against 367 tokens / 982 s
decode / cancelled with `reasoning_effort=low`:

    seconds 262.6   within the 20-minute budget
    output_tokens 15
    completion: {"verdict": "APPROVE", "issues": []}

It finishes in a fifth of the budget and **approves a change containing a
planted defect**. The output is valid JSON, not truncated -- an earlier
"malformed completion framing" label was an artefact of parsing nonthinking
output with the thinking-mode parser, and is withdrawn.

A reviewer that silently approves defective code is worse than one that is too
slow, because the slow one fails visibly. This option is closed: the speed came
from the model not doing the work, and the acceptance criteria were not altered
to accommodate it.

## Where the remaining 2.5x would have to come from

The attention stage is 52.8% of prefill at 8192 tokens. Measured inside it at
2048 tokens, with a temporary timer since removed:

    attention stage total   67.8 s
    serial per-token loop   20.7 s   30.5%
    CUDA sparse kernel etc  47.1 s   69.5%

The serial loop is the one the older handoff flagged ("scalar state transitions
still construct each token's query, compression and causal selected indices in
order"). It is real, but it is not the wall. Parallelising it perfectly across
six threads saves about 17 s of a 194 s prefill -- roughly 9%, or 13% at 8192 --
and the loop is causal by construction, so the change carries correctness risk
for a gain that moves no verdict in the table above. It was not made.

**The wall is the batched CUDA sparse-attention kernel**, at roughly 70% of the
attention stage. That kernel is already batched and was already optimised once
(an isolated 1024x128 case improved from 100 ms scalar to 24.7 ms batched).
Any further gain has to come from it, from the routed-expert GEMMs, or from
hardware. That is a kernel engineering project with its own qualification, not
a configuration change, and it should be scoped deliberately rather than
started opportunistically at the end of an acceptance run.

## A kernel attempt that did not pay (recorded so it is not repeated)

Two arithmetic-exact changes were made to `dsv4_sparse_attention_kernel` and
measured: staging the query row in shared memory (it was re-read from global
once per dimension per scored key) and keeping the output accumulator in shared
memory (it was read-modify-written in global once per 64-key block, so a
2048-key row made 32 round trips per thread instead of one).

Both preserved every operation and its order, and the fixtures confirmed it:
whole-prompt `max_error=0` at chunks 1/35/69/103/137, and the independent
pinned-kernel comparison still exact at all 64-key boundaries.

**It produced no speedup at all**: 195.2 s against a 193.9 s baseline, with the
attention stage unchanged at 67.6 s. The change was reverted.

Two things this establishes. The query and accumulator traffic was evidently
already served by L1/L2, so the kernel is not bound by those accesses; the
remaining candidate inside it is the scattered `kv[indices[t]*dim+d]` gather,
which is memory-bound. And the earlier "47.1 s outside the serial loop" figure
is not the kernel alone -- that timer spans projections, the collector, RoPE and
output projections too, so the kernel's true share is unmeasured and probably
much smaller than it appeared. Anyone resuming this should instrument inside
the attention stage before optimising any part of it.

The online softmax was deliberately left serial on thread 0. Its running sum is
an order-dependent float reduction, and reordering it is precisely the class of
change that previously moved a layer by 1.41e-5 and altered a token.

## There is no hotspot: the attention stage measured from the inside

After the failed kernel attempt, the attention stage was instrumented into its
four sections (2048 tokens, 43 layers, timers since removed):

| section | seconds | share |
| --- | ---: | ---: |
| serial per-token state loop (CPU) | 20.9 | 32.7% |
| input projections (HC, rmsnorm, wq_a/q_norm/wq_b, wkv -- FP8 GEMMs) | 19.3 | 30.2% |
| CUDA sparse attention + RoPE | 13.9 | 21.7% |
| output projections (wo_a groups, wo_b) | 9.8 | 15.4% |

This explains the null result above: **the sparse-attention kernel is only
about 22% of the attention stage**, roughly 7% of prefill, so even a large
bit-exact win inside it cannot show up in a whole-prefill measurement. The
earlier "47.1 s outside the serial loop" figure conflated all four sections and
should not be used.

It also answers the optimisation question directly. The cost is **broadly
distributed**, not concentrated: roughly a third CPU state transitions, a third
FP8 dense projections, a fifth sparse attention, a sixth output projections.
Reaching 2.6x would mean removing about 62% of *everything*, not fixing one
hot spot. The largest single item, the serial CPU loop, is worth at most ~13%
of prefill if parallelised perfectly, and it is causal by construction.

That is the strongest statement this work can make: no single change available
here closes the gap, and the evidence is measurement rather than judgement.

## The decision this needs

Production keeps a one-hour timeout; the 20 minutes is the qualification bar.
Every case fits inside an hour -- 32,768 tokens lands at roughly 48 minutes.

1. Accept the model with a longer qualification bar for long inputs.
2. Keep 20 minutes and cap reviewer input far below 32,768 tokens. 512 passes
   today; 2048 is close and would need about 1.3x.
3. Fund an attention and expert-compute optimisation for a ~2.5x target. That
   is a GPU-kernel project, not a configuration change.

No further configuration tuning closes this. The checkpoint is byte-validated
(91/91 groups, 166.9 GB compared), the engine passes its native fixtures and
79/79 Python tests, and the pending startup patch remains unapplied.
