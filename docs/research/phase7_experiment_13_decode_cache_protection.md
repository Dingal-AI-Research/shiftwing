# Phase 7 Experiment 13: Decode-Set Protection and Capacity Boundary

**Target:** Qwen3.5-397B-A17B tiered decode
**Hardware:** Ryzen 7 7700X, RTX 5070 Ti, external NVMe, 29.4 GiB WSL RAM
**Date:** 2026-07-28
**Status:** short-turn treatment retained as opt-in; sustained gate not met

## Abstract

Decode-only attribution showed that prompt prefill displaced experts learned
during the preceding warm decode. This experiment added a phase-aware host
cache: experts used during decode accumulate a separate heat score, up to 38
of each layer's 48 host slots are protected, and ten slots remain available
for a complete top-10 route set during the next prefill. An optional
inter-request warm step also materializes selected experts that survived only
in the independent device cache.

For a repeated four-token prompt, protection alone reduced decode misses from
726 to 292 and raised throughput to 1.296 tokens per second. Materializing
device-only selections eliminated all 726 original decode misses, produced
zero decode storage bytes, and reached 2.736 tokens per second with identical
text. This proves that the accepted CUDA compute path is fast enough when the
decode working set fits the protected cache.

The result does not scale to the preregistered sustained workload. On the same
prompt with 64 generated tokens, the strengthened policy measured 0.772
tokens per second, a 67.57% non-disk hit rate, 12,454 misses, and 83.25 GB of
decode reads. The long route set greatly exceeds 38 protected experts per
layer. Gate 7 therefore remains unmet on the 32 GB reference workstation;
the four-token result is not substituted for the frozen 64-token criterion.

## 1. Definitions

- **Cache pollution** occurs when work from one phase evicts data that a
  later phase would reuse. Here, the next prompt's prefill evicted experts
  retained from the previous decode.
- A **protected slot** cannot be selected as a normal replacement victim.
- The **working set** is the collection of distinct experts needed over an
  interval. It can be much larger than the ten experts used by one layer for
  one token.
- **Materialization** copies or reloads a selected expert into host RAM when
  its only surviving copy is in VRAM.
- A **capacity boundary** is a performance limit caused by how much useful
  data fits in memory, rather than by arithmetic kernel speed.

## 2. Research questions

1. Is prefill pollution responsible for a material part of bounded decode
   misses?
2. Can a phase-aware replacement policy cross 2 tok/s without changing
   weights, routing, generated text, or the RAM/VRAM budgets?
3. Does a short-turn result remain valid over the official 64-token sustained
   interval?

## 3. Mechanism

`DECODE_PROTECT=1` allocates a separate per-layer decode heat table. Only
expert requests made after prefill update it. At the end of a request, the
engine selects at most:

```text
host slots - router top-k = 48 - 10 = 38 experts per layer
```

This leaves enough unprotected slots to load any complete top-10 route set
without violating the cache's in-use protection rule.

`DECODE_PROTECT_PREWARM=1` additionally ensures that every selected expert
has a host copy between requests. This is necessary because the CUDA expert
LRU is independent: a QMat can lose its host owner while the device copy
remains usable. A later device eviction would otherwise expose a storage
miss despite the expert's decode score.

Inter-request loads are emitted after `DONE`; they are cache warm-up rather
than request decode. They are tracked separately and do not increment route
heat, request CPU hits, or request storage misses. The next request still
counts its complete prefill and every decode operation.

Both switches are disabled by default. They favor repeated or highly similar
turns and can impose substantial work between unrelated requests.

## 4. Controlled method

Every arm restored the same expert atlas:

```text
SHA-256 3ceb220c640bcfd1efcc58c7bc99fe44316e8ebf61b754b47d89464e9501b216
```

The model, prompt, context 512, one-slot resident CUDA graph, 18 GiB host
expert cache, 6 GiB device expert cache, direct I/O, stream-ordered CUDA
allocator, and CUDA event profiling were fixed. Each arm used one warm
request followed by one measured request. Output equality and exactly 600
routes per generated token were required.

The first treatment generated four tokens with protection but no
materialization. The second added materialization. The scaling arm repeated
the strengthened treatment with 64 generated tokens.

## 5. Results

### 5.1 Four-token ablation

| Policy | Decode rate | Host hits | GPU hits | Misses | Read bytes | Non-disk hits |
|---|---:|---:|---:|---:|---:|---:|
| accepted allocator baseline | 0.759–0.810 | 1,157 | 517 | 726 | 4.853 GB | 69.75% |
| protect resident host set | 1.296 | 1,591 | 517 | 292 | 1.952 GB | 87.83% |
| protect + materialize | **2.736** | 1,883 | 517 | **0** | **0** | **100.00%** |

All arms emitted the identical prefix `A **cache miss`. The strengthened
arm's decode-only profile was:

| Phase, four tokens | Seconds |
|---|---:|
| storage | 0.000000 |
| expert upload/compute | 1.412993 |
| GDN/GQA | 0.023528 |
| language-model head | 0.008750 |
| total | 1.446568 |

This is direct evidence that the CUDA path itself exceeds the 2 tok/s
threshold when all decode experts are warm.

### 5.2 Sixty-four-token scaling arm

The output remained identical across warm and measured requests. The measured
decode produced 64 tokens in 82.86 seconds:

| Counter or phase | Value |
|---|---:|
| decode rate | **0.772 tok/s** |
| host hits | 12,399 |
| GPU hits | 13,547 |
| storage misses | 12,454 |
| total routes | 38,400 |
| non-disk hit rate | 67.57% |
| storage bytes | 83.251 GB |
| storage time | 61.489 s |
| expert upload/compute | 20.668 s |
| GDN/GQA | 0.424 s |
| language-model head | 0.253 s |

Protection improves the warm arm's 0.482 tok/s to 0.772 tok/s, but long
decode still spends 74.2% of its time reading experts.

### 5.3 Capacity analysis

The 64-token target at 2 tok/s allows 32 seconds. Its measured non-storage
work consumes 21.346 seconds, leaving 10.654 seconds for I/O. At the observed
1.354 GB/s effective read rate, the request may read approximately 14.43 GB,
or about 2,158 equal-size experts. Thus at least:

```text
1 - 2,158 / 38,400 = 94.38% non-disk hits
```

are required on this machine.

The saved atlas gives the following ideal static frequency coverage:

| Experts retained per layer | Route coverage |
|---:|---:|
| 48 | 78.67% |
| 64 | 86.20% |
| 80 | 92.88% |
| 88 | 95.33% |
| 96 | 97.04% |

Per-layer redistribution under the same 2,880 host slots raises 78.67% only
to 79.25%, so unequal layer allocation cannot bridge the gap.

An ideal disjoint 88-entry RAM/VRAM union would need roughly 72 host experts
plus 16 device experts per layer. The host portion alone is 26.89 GiB of
routed weights. With the current 6.78 GiB host-resident dense/shared payload,
0.21 GiB state, and 1 GiB guard, that idealized plan needs about 34.9 GiB of
WSL RAM. The machine exposes only 29.4 GiB and has 31.1 GiB physically
installed. Current device/host duplication makes the practical requirement
higher still.

## 6. Lossless-compression probe

A deterministic sample of 180 routed int4 matrices (360 MiB) was tested
before proposing a compressed host tier. Nibble entropy was 3.180 bits per
four-bit value. Fast zlib produced a 75.03% aggregate size, but the median
matrix retained 86.16% of its original bytes. Decompression on every expert
use would add CPU work, and the resulting effective capacity still would not
guarantee the required disjoint 88-entry union. A compressed tier is
therefore not implemented from this probe alone.

## 7. Decision

The phase-aware mechanism is retained as an opt-in optimization for repeated
short turns. It is not enabled by default and does not close Gate 7. The
short result demonstrates compute sufficiency; the long result demonstrates
a reference-hardware capacity and storage limit.

The next Gate-7 decision must be explicit:

1. qualify unchanged precision on a machine with at least 48 GB RAM and a
   larger disjoint expert union;
2. authorize a separately quality-gated lower-bit routed-expert format; or
3. close the reference 32 GB profile as a measured no-go while preserving the
   original ≥2 tok/s threshold as unmet.

Artifacts:

- `c/qwen397_bounded_decode_protect.json`
- `c/qwen397_bounded_decode_prewarm.json`
- `c/qwen397_bounded_decode_prewarm_64.json`
