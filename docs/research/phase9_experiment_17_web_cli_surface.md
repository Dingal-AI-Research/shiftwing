# Phase 9 Experiment 17: Web and CLI Surface

## Abstract

This experiment connected the mux engine and Qwen-aware HTTP gateway to the
planned user-facing surface. The pinned upstream React/Vite client was vendored
and rebranded, its model and decoding defaults were changed to Qwen3.5 greedy
service, and its profiler was aligned with `PERF`. A new `colib` launcher now
provides chat, serve, web, conversion, and doctor commands in both source and
installed layouts. The subsequently expanded suite passes 21 C, 59 Python,
and 18 web tests.

## Method

The web directory was imported from colibri commit
`81f08a09e5651ce52616dc720f68810f9021c0be`, the same immutable revision used
for the HTTP gateway. Visible branding and package metadata were changed to
colib. Protocol compatibility keys such as `x-colibri-queue-wait-ms` and
legacy browser-storage cleanup were intentionally retained.

Gate 9 currently supports greedy generation only. The UI therefore defaults
to `qwen3.5-colib`, sends temperature zero, and displays a disabled greedy
control instead of advertising unsupported sampling. The profile type accepts
the new request ID and KV timing field. On `DONE`, the gateway joins the
matching `PERF` record with prompt/completion counts so the existing throughput
and batching views remain meaningful.

The dependency-free `c/colib` launcher implements:

- `chat`: one persistent mux engine with Qwen ChatML history;
- `serve`: the OpenAI-compatible HTTP gateway;
- `web`: deterministic npm build when necessary, then the same gateway plus
  its static bundle;
- `convert`: delegation to the shared resumable Qwen converter;
- `doctor`: read-only model, engine, CUDA-linkage, and state-resource checks.

The Makefile gained web build/test targets and an install layout:

```
bin/colib
libexec/colib/qwen
libexec/colib/{openai_server.py,doctor.py}
libexec/colib/tools/
libexec/colib/web/dist/
```

The gateway searches both the source and installed bundle locations. A staged
`DESTDIR` install was executed and its installed launcher diagnosed the tiny
container successfully.

The browser/API end-to-end test starts `colib web`, loads the built root page,
checks the colib title, submits a real two-token completion through the HTTP
gateway and tiny C engine, and terminates the process cleanly. A raw completion
is used because the deliberately small oracle vocabulary does not contain the
official production ChatML special-token IDs.

## Results

| Check | Result |
|---|---:|
| complete C suite | 21/21 |
| complete Python suite | 59/59 |
| web unit tests | 18/18 |
| TypeScript + Vite production build | pass |
| production dependency audit | 0 vulnerabilities |
| staged install + installed doctor | pass |
| static bundle + gateway + real C engine | pass |

The production bundle is 226.50 kB JavaScript and 27.62 kB CSS before gzip
(74.13 kB and 6.58 kB after gzip, respectively).

The imported lockfile initially resolved a development-only PostCSS version
affected by a source-map path disclosure advisory. Pinning PostCSS 8.5.23
removed the audit finding; the production dependency audit and full audit both
then reported zero vulnerabilities.

## Interpretation

The server is no longer only a protocol primitive. A user can now diagnose a
converted model, start persistent local chat, expose an authenticated API, or
open the browser UI through one command while using the same tested engine and
session semantics.

Locking the UI to greedy mode is an interface-correctness choice: unsupported
controls do not silently promise behavior the mux rejects. Sampling can later
be enabled in the UI when the wire protocol implements it.

## Limitations

The end-to-end browser test uses the tiny raw-completion path. Production
Qwen ChatML rendering is independently byte-compared with the official
template, but a complete browser chat must still be qualified on 35B and 397B.

`doctor.py` is intentionally a first Qwen-specific diagnostic. Phase 10 must
expand platform coverage, validate actual container inventories rather than
only their index presence, and document installation dependencies for
conversion. The UI currently has no MTP or sampling controls.

## Conclusion

The planned Phase 9 gateway/web/CLI surface is implemented, installable, and
covered by unit plus real-process tests. Gate 9 now depends primarily on
production-model browser qualification and the remaining end-to-end CUDA
activation/performance work.
