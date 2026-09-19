# DeepSeek reviewer thinking experiment

capability and performance only; not acceptance or activation.

Status: **running**. Updated 2026-09-15T15:25:05.686039+00:00. Active: 32768-low.

Native token counts; thinking/final counts exclude `</think>` and EOS. TTFT includes thinking. First final is the first non-whitespace final-answer content. Load time is separate. Timeouts are censored observations, never completed reviews.

| Input | Effort | Status | Prefill min | First final min | Total min | Thinking tok | Final tok | Decode tok/s | Defect found |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 2048 | off | eos | 4.82 | 4.83 | 6.41 | 0 | 14 | 0.16 | False |
| 2048 | low | eos | 5.09 | 26.08 | 31.34 | 419 | 107 | 0.34 | True |
| 2048 | high | eos | 3.82 | 16.51 | 20.50 | 269 | 76 | 0.35 | True |
| 2048 | max | eos | 4.90 | 22.34 | 28.67 | 321 | 104 | 0.30 | True |
| 32768 | off | eos | 33.97 | 33.98 | 38.02 | 0 | 87 | 0.36 | True |
| 32768 | low | pending | — | — | — | — | — | — | — |
| 32768 | high | pending | — | — | — | — | — | — | — |
| 32768 | max | pending | — | — | — | — | — | — | — |
| 8192 | off | pending | — | — | — | — | — | — | — |
| 8192 | low | pending | — | — | — | — | — | — | — |
| 8192 | high | pending | — | — | — | — | — | — | — |
| 8192 | max | pending | — | — | — | — | — | — | — |
| 16384 | off | pending | — | — | — | — | — | — | — |
| 16384 | low | pending | — | — | — | — | — | — | — |
| 16384 | high | pending | — | — | — | — | — | — | — |
| 16384 | max | pending | — | — | — | — | — | — | — |
| 512 | off | pending | — | — | — | — | — | — | — |
| 512 | low | pending | — | — | — | — | — | — | — |
| 512 | high | pending | — | — | — | — | — | — | — |
| 512 | max | pending | — | — | — | — | — | — | — |

Limitations:

- one deterministic sample per cell
- small planted defects with synthetic reference helpers
- one distinct defect per length; compare efforts within each length
- effort prefixes consume context; helper padding is adjusted to exact total
- expert and OS caches are not flushed; per-request KV starts empty
- existing full-model numerical parity gate is still failing
