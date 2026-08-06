# colib web

React/Vite interface for an OpenAI-compatible colib server.

```sh
npm install
npm run dev
```

The default endpoint is `http://127.0.0.1:8000/v1`. Run
`./c/colib web --model c/qwen35 --cuda` to build the bundle and serve it from
the same process as the API. Gate 9 uses fixed greedy decoding, so the
temperature control is intentionally locked at zero.

Local validation:

```sh
npm test
npm run build
```

Besides Chat and Brain, the **Profiling** tab charts where the engine spent
each turn's wall time (I/O wait, expert matmul, attention, LM head) from the
server's `/profile` endpoint — a rolling window of per-turn `PERF` snapshots
emitted by the engine.

The test suite stays browser-light: API requests use a mocked `fetch`, while
runtime capability and storage behavior are covered through pure helpers. It
checks that `/health` and `/profile` are resolved next to (not below) the OpenAI `/v1` prefix,
supports both boolean and numeric `scheduler.active` responses, and sends the
colib-specific `cache_slot` field only when KV-slot support was advertised.

The endpoint and selected model are persisted locally. API keys are intentionally
memory-only; startup/persistence also removes the legacy `colibri.apiKey` value.
