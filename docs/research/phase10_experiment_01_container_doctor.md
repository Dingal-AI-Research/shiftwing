# Phase 10 Experiment 1: Header-Exact Container Diagnostics

## Abstract

This experiment hardened `colib doctor` from a file-presence check into a
read-only safetensors inventory validator. It parses every shard header,
validates tensor offset bounds and non-overlap, matches index ownership in both
directions, rejects unsafe shard paths, detects duplicates and missing shards,
and verifies the index `metadata.total_size` against the exact tensor payload
sum. The converted Qwen3.5-35B container passes with 64,646 physical tensors in
14 shards and an exact 19,929,665,806-byte payload.

## Method

For each safetensors file, the diagnostic reads:

1. the eight-byte little-endian header length;
2. the bounded JSON header (maximum accepted size 64 MiB);
3. no tensor payload bytes.

Each non-metadata entry must contain a string dtype, a nonnegative integer
shape, and two ordered integer offsets inside the file's data section. Sorted
payload ranges may not overlap.

For sharded containers, `model.safetensors.index.json` must contain a nonempty
`weight_map`. Shard values must be plain filenames rather than absolute or
parent-relative paths. The validator compares:

- every indexed name and its expected shard;
- every name physically present in a shard header;
- duplicate names across shards;
- the sum of all header payload ranges;
- optional `metadata.total_size`.

When a converter-produced `quantization.json` is present, doctor independently
checks its completion flag, payload byte count, physical tensor count,
source-shard count, and loader-inventory counts against the parsed container.
Oracle snapshots without this optional manifest remain valid.

An index-less single-file container is also valid. If a conversion state exists
without a final index, doctor reports a resumable warning rather than claiming
completion.

Two tests were added. The real tiny single-shard oracle must report a positive
tensor count and payload size. A synthetic index pointing to a missing shard
must make the overall doctor result an error.

## Results

The Qwen35 audit reported:

| Property | Value |
|---|---:|
| indexed physical tensors | 64,646 |
| header physical tensors | 64,646 |
| referenced/parsed shards | 14 / 14 |
| exact tensor payload | 19,929,665,806 bytes |
| shard files including headers | 19,938,400,006 bytes |
| index metadata total | 19,929,665,806 bytes |
| audit status | pass |

The single-file tiny int4 oracle passes. The missing-shard fixture fails with a
specific `missing shard` inventory error and its deliberately incomplete
manifest also fails. Qwen35's manifest matches all 19,929,665,806 bytes,
64,646 physical tensors, 14 source shards, and the complete logical loader
inventory. The staged CLI test continues to pass, and the CUDA-linked engine
is still recognized through `ldd`.
The complete suite passes 21 C executables and 59 Python tests.

## Interpretation

File presence is insufficient for a 19–213 GB model artifact. An interrupted
copy can leave every expected filename present while truncating a header or
payload; an inconsistent index can silently direct the loader to the wrong
shard. Header-exact validation catches these structural failures without the
I/O cost of hashing every tensor.

This check complements rather than replaces numerical qualification. Valid
offsets prove container integrity and loader addressability, not correct
quantization values or model output.

Resource diagnostics query the first NVIDIA device's actual total/free memory
through `nvidia-smi`, accept an explicit runtime-headroom budget, and fail when
the computed disk, host-RAM, or device-VRAM plan exceeds measured capacity.
VRAM is checked against currently free rather than nominal total bytes; the C
runtime repeats a stricter check after creating its CUDA context.
They no longer assume a nominal 16.0 GiB device or report an unsafe plan only
as inert JSON.

## Decision

- Retain header-exact inventory as the default `doctor` behavior.
- Keep it read-only and payload-free so production containers audit quickly.
- Require a present converter completion manifest to agree with the observed
  container totals and loader inventory.
- Treat a malformed finalized index or missing indexed shard as a hard error.
- Treat an index-less resumable conversion as a warning.
- Use the same audit immediately after the 397B converter finalizes its index.

## Limitations

Payload contents are not checksummed, so in-place bit corruption inside a
correctly sized range is outside this diagnostic. Full cryptographic hashes
would require reading the entire container and should be a separate opt-in
command. The 64 MiB header ceiling is a defensive implementation limit; no
current project container approaches it.
