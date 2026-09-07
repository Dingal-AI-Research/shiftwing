# GLM-5.3-Flash performance roadmap

Date: 2026-09-07
Status: planned; nothing in phases 1-3 is implemented yet

Baseline is the activated review lane: one planted-defect review, correct
verdict, **436.7 s** (prefill 100.0 s, decode 336.7 s for 192 tokens =
0.570 tok/s). Ornith-397B does the same review in 8350 s.

Roadmap order as directed: **io_uring, then MTP speculative decoding, then
int3 experts.** Section 5 records a measurement that argues for a different
order; the decision is the owner's and the order above stands until changed.

## 1. The governing constraint

Everything here is bounded by one number. Measured on this host's own shards,
O_DIRECT, engine not running:

| streams | throughput |
| ---: | ---: |
| 1 (bs 1M) | 1.1 GB/s |
| 1 (bs 16M) | 1.9 GB/s |
| 4 | **2.6 GB/s** |
| 8 | 2.15 GB/s |

Throughput peaks near four streams and *falls* at eight. Against that ceiling,
the lane currently achieves:

| stage | bytes moved | time | rate |
| --- | ---: | ---: | ---: |
| decode | 858 GB | 336.7 s | **2.55 GB/s** |
| prefill | 172 GB | 100.0 s | 1.72 GB/s |

**Decode is already at the disk ceiling.** No amount of extra parallelism can
speed it up; only moving fewer bytes can. Prefill has roughly 1.5x of headroom,
though not all of its 100 s is I/O -- 172 GB at 2.6 GB/s is 66 s, so about 34 s
is compute.

The unit that matters for decode is therefore **bytes per accepted token**,
currently 4.77 GB (8 experts x 42 sparse layers x 14.2 MB), less a 6.3% cache
hit rate.

## 2. Phase 1 -- io_uring batched expert reads

**What changes.** `expert_read` issues one blocking `pread` per expert inside an
OpenMP loop. `st.h` already has `st_read_raw_batch_uring`, but it addresses
tensors by name and the expert loader works in raw `(fd, offset, length)`
triples. So: a raw-range batch helper alongside the O_DIRECT one already added
to `glm53_compat.h`, then restructure the block loop in the MoE forward to
collect all misses and submit them as a single batch.

**Files.** `c/glm53_compat.h` (new helper), `c/glm53.c` (block loop),
`c/Makefile` (nothing expected).

**Gate.** `make -C c test-glm53` unchanged at f32 9.54e-07 and int4 0.144;
one scored review run.

**Expected.** Prefill only, since decode is already saturated. Upper bound is
prefill 100 s -> 66 s, and the real figure is lower because ~34 s of prefill is
compute. Call it **5-8% of total**, and treat anything above that as a
surprise worth understanding.

**Risks.** io_uring is available on this kernel (`io_uring_setup` returns
features 0x3fff on 6.6.87.2-microsoft-standard-WSL2), so the path exists; the
fallback to the current per-read path must stay, because a filesystem or
kernel that refuses it is a normal condition. Partial completions and short
reads are the fiddly part.

## 3. Phase 2 -- MTP speculative decoding

**What exists already.** The checkpoint carries the MTP layer in full: all 1753
layer-45 tensors including `eh_proj`, `enorm`, `hnorm`, plus its own 288 routed
experts. `glm53.c` parses `n_mtp` and reserves struct fields, and the loader
reaches the layer, but there is **no draft/verify loop**. Separately, `qwen.c`
has a complete, gate-passed implementation of exactly this (draft, verify,
recurrent-state rollback, telemetry, confidence-gated admission), documented in
[Phase 6](phase6_mtp.md). Phase 2 is a port of a proven design, not an
invention.

**What changes.** A draft step using layer 45; verification via the existing
`forward_span`, which already accepts *n* tokens; an acceptance test; and
rollback of KV plus the KDA recurrent and convolution state on rejection.

**Gate.** Speculative and non-speculative decoding must produce **identical
token streams**. Phase 6 records two traps worth inheriting: MTP and non-MTP
modes had used different prompt-prefill algorithms, and a batched MoE kernel
changed floating-point operation order by up to 1.41e-5 per layer, which
altered one code-domain token while leaving `Hello` identical. An oracle that
only checks easy prompts will not catch that.

**Measured, 2026-09-07: acceptance is 39.0%, and the phase should stop here.**

The head is loaded and drafting. Observation mode -- drafts scored against the
token the model actually chose, output untouched -- gave **23 of 59, 39.0%**,
on a review-shaped prompt. Against the cost model in section 5:

| acceptance | bytes per accepted token | vs baseline |
| ---: | ---: | ---: |
| 39% (measured) | 1.76B | **+76% slower** |
| 90% | 1.02B | +2% |
| 94.5% (Qwen's) | 0.97B | -3% |

Enabling acceptance at 39% would make decoding roughly three quarters slower.

**A defect of mine probably depresses that number, and fixing it does not
change the decision.** Prefill never runs the MTP layer, so at the first draft
its KV cache holds nothing: the block can attend only to the position it is
drafting from, where a trained head expects the whole prompt behind it.
Populating it means running the MTP layer over the prompt -- one layer, but
with its own 288 routed experts, so about 4 GB of reads per prompt. Worth
roughly 1.5 s, and it would raise acceptance by an unknown amount.

It is not worth doing, because the ceiling does not move. Even at Qwen's 94.5%
this is a **3%** gain, and anything below 90% is a loss. Chasing a 3% best case
through a bug fix whose payoff is unmeasured, on a path that is negative if the
rate lands anywhere below 90%, is a bad trade.

**What is left in the tree.** The head loads, `mtp_step` runs, and observation
mode reports the rate; all of it is behind `GLM53_MTP` and off by default, and
the engine is byte-identical with it off. The accept path -- snapshot, verify,
rollback, replay -- is deliberately not written.

**Risks, had it been built.** Rollback is the subtle part: GLM's KDA recurrent
state differs from Qwen's GDN state, so that piece does not port directly.

## 4. Phase 3 -- int3 experts

**What changes.** An int3 packing in `convert_glm53.py` and a matching dequant
path in the engine's expert matmul. `qwen.c` already has an int3 expert format
(`EXPERT_Q3`) to follow.

**Expected.** Expert bytes fall from 14.2 MB to about 10.9 MB, so decode bytes
fall about **23%**: decode 336.7 s -> ~260 s, total ~360 s.

**Cost that is easy to miss.** Conversion streams and deletes each source shard
after converting it, so the 182 GB of source is gone. Re-converting means
re-downloading it at the ~5 MB/s the unauthenticated HF endpoint gives, which
is **about 10 hours of wall time** before any measurement can start. An HF
token would change that.

**Risks.** int3 quality on GLM is unmeasured. The int4 tiny-oracle margin is
0.144 against an int8 margin of 0.0086, so int3 needs its own quality gate on
real prompts, not just the tiny oracle.

## 5. A measurement that argues for reordering

MTP's benefit depends on what the bottleneck is, and it is not the same here as
in Phase 6.

Phase 6 measured, for Qwen at depth 1: **94.48% acceptance** and **15.59% route
overlap**, for a final 1.394x. That gain came on a compute-bound configuration,
where the cost unit is the forward pass.

Here the cost unit is bytes read. Verifying two tokens in one pass needs the
*union* of their expert routes. At 15.59% overlap that union is about
`8 + 8 x 0.844 = 14.75` experts, or 1.84x the bytes of a single token, in
exchange for about 1.94 accepted tokens:

    bytes per accepted token = 4.77 GB x 1.84 / 1.94 = 4.52 GB
    against 4.77 GB today, a saving of about 5%

So on this hardware MTP is worth roughly **5%**, not the ~25% estimated before
this overlap figure was found. It is the largest engineering effort of the
three and, on these numbers, the smallest return.

Ranked by measured return per unit of effort, the order would be:

| | phase | expected | effort |
| --- | --- | ---: | --- |
| 1 | int3 experts | ~18% of total | ~1 day + ~10 h re-download |
| 2 | io_uring | 5-8% of total | ~half a day |
| 3 | MTP | ~5% of total | 1-2 days |

The directed order (io_uring, MTP, int3) stands unless changed. Phase 1 is
cheap either way, so it is not a costly place to start.

**Both estimates above proved wrong once measured, in the same direction.**
io_uring came in slower than the code it replaced across three attempts, and
MTP measured 39% acceptance against a break-even of 90%. The common cause is
in section 1: decode already runs at 2.55 GB/s against a 2.6 GB/s disk. There
is no idle capacity for a scheduling change to recover, so any rewrite that
adds bookkeeping loses, and anything that does not reduce bytes cannot win.
Only int3, which actually removes bytes, is still expected to help.

## 6. What none of these fix

Together the three phases are worth perhaps 30%, and they do not touch the
constraint: 4.77 GB of expert reads per token against 2.6 GB/s of disk. The
expert cache holds 23 of 288 experts per layer and returns a 6.3% hit rate,
and hit rate tracks cache fraction almost linearly here, so routing is close to
uniform and there is no skew to exploit.

More RAM is the only change that removes reads rather than reshuffling them.
The GPU does not help: 16 GB of VRAM would take the cache from 8% to ~18% of
experts, worth about 7%, and GPU *compute* is worth nothing while the machine
sits 77-94% idle waiting on storage.

## 7. Verification protocol, every phase

1. `make -C c test-glm53` -- f32 9.54e-07, int8 0.0086, int4 0.144, tokens
   exact. Any movement here means the change altered the model, not its speed.
2. `make -C c test-c` and the Python suite, because `st.h` and `tok.h` are
   shared with the Ornith lane that is still installed.
3. One scored review through `review_quality_bench.py --endpoint`, reporting
   prefill, decode, tok/s and the expert cache hit rate.
4. Record the measurement in `docs/research/glm53_flash_engine_qualification.md`
   whether or not it helped. A phase that does nothing is a result.
