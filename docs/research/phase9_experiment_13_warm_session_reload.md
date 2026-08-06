# Phase 9 Experiment 13: Warm Session Reload

## Abstract

This experiment completed target-session persistence at the mux boundary.
`SESSION_DIR` stores one atomic checkpoint per slot containing token history,
last hidden state, every GDN recurrent matrix and convolution tail, and used GQA
KV. A new engine process validates and restores the checkpoint, then reuses it
only for a token-exact extension. The integration test restored a cancelled
prompt, consumed a known extension, and emitted the same continuation bytes as
fresh full prefill.

## Method

The new `COLIMUXS` envelope records:

- a version and model dimensions;
- KV precision and position;
- history, recurrent, and KV element counts;
- token IDs;
- flattened target recurrent/conv state;
- used KV and last hidden vector;
- an FNV-1a checksum over all semantic payloads.

Writes use a process-specific temporary file, `fflush`, file `fsync`, atomic
rename, and directory `fsync`. They occur before `DONE` and before a
`CANCELLED` acknowledgement. Startup reads `slot-NN.colimux`, validates model
geometry and checksum, then imports it into either the snapshot reference or
`ResidentBatchState`.

The subprocess test used two engine lifetimes:

1. process A prefetched `!`, received `CANCEL`, persisted slot 0, and exited;
2. process B loaded the same directory and submitted `!a`.

The tiny tokenizer maps these strings to `[0]` and `[0, 64]`, so the second
request is a known exact token extension. A third process generated from `!a`
without a checkpoint. The restored and fresh `DATA` byte sequences were
compared.

## Results

Process B reported:

```
[SESSION] restored slot=0 tokens=1
[SESSION] exact-extension slot=0 cached=1 prompt=2
```

Its continuation matched fresh prefill exactly. The resident state also
survived a direct `COLISESS` export/write/read/import round trip and generated
the same next eight tokens.

## Interpretation

The server no longer needs an in-memory Python conversation object to preserve
model state across engine restarts. Token history remains authoritative:
shortened, divergent, or differently tokenized text cannot reuse the cache and
falls back to the established chunked prefill path.

Persisting before cancellation acknowledgement preserves the protocol's
release-before-reuse invariant. Persisting only at turn boundaries avoids the
per-token storage traffic rejected in Experiment 8.

## Limitations

The format is native/local and target-only. It does not contain MTP drafter
state, a source-model content hash, endian conversion, or cross-version schema
migration. Checkpoints are keyed by numeric slot rather than an authenticated
application-level conversation ID; the HTTP server is responsible for stable,
private slot assignment.

## Conclusion

The Phase 9 warm-reload requirement is now implemented for target decoding:
history and hybrid state survive a process restart, exact extensions reuse
state, and divergent prompts still re-prefill.
