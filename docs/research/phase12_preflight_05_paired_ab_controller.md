# Phase 12 preflight 05: paired AB/BA performance controller

Date: 2026-08-19

## Outcome

The preregistered five-pair protocol in
[expanded-q4 route atlas](phase12_experiment_02_expanded_q4_route_atlas.md)
and [ordered grouped-prefill expert I/O](phase12_experiment_04_ordered_prefill_expert_io.md)
had no controller that could actually execute it. `c/tools/run_perf_trials.py`
runs one arm as independent hash-bound trials; it cannot alternate arms, pair
their results, compare outputs across arms, or produce a confidence bound.

`c/tools/run_paired_perf_trials.py` adds that layer. It is a separate tool and
does not modify the single-arm controller, which keeps every historical trial
artifact and its recorded controller hash valid.

No real-model performance claim is made here. This report records the
mechanism and its own validation only; the model reconstruction is still
running.

## Design

Each pair runs both arms back to back and alternates their launch order: odd
pairs run control first (AB), even pairs run candidate first (BA). Alternation
is what makes a monotone drift in machine state — page cache, thermals, or
fragmentation — fall on both arms rather than on whichever arm happens to run
second.

The measured five-trial Ornith397 control makes that risk concrete rather than
theoretical. Its own trials rise from 0.773577249 to 0.838411793 tok/s as the
session warms, an 8.4% spread with no intervention at all. A naive
control-then-candidate sequence would report most of that warm-up as candidate
speedup.

For the same reason the controller accepts `--prime-trials N`: N unmeasured
trials, alternating arms, run before pair 1. They are executed, verified,
hashed, and retained in the state ledger exactly like measured trials, and
they are excluded from every ratio and bound.

## Binding and resume

The signature covers the pair count, prime count, both arm labels and command
templates, the acceptance key, the throughput key, the settle interval, every
`--bind` hash, and optionally the engine source fingerprint
(`--bind-engine-source`). Any drift refuses the resume instead of mixing
evidence from two configurations.

A trial counts as complete only when its published artifact still matches the
recorded size and SHA-256. A stale `running` attempt becomes `interrupted` on
the next launch, and execution restarts at the first unverified boundary. An
artifact is published only after its acceptance key is true and its metrics
parse; a failed arm leaves no output file.

## Analysis

Per pair the controller computes the candidate/control ratio for sustained
decode tok/s and for median turn TTFT. Bounds are paired t-intervals on the
log ratios, so the reported interval is for the geometric mean and cannot
place a nonpositive ratio inside the bound.

Student's t comes from the regularized incomplete beta with a Lentz continued
fraction; no SciPy dependency is added. The implementation reproduces
published critical values to five decimals: 12.706205 at one degree of
freedom, 2.776445 at four, 2.042272 at thirty, and 4.604095 at four for the
99% level.

Promotion requires every gate:

1. `pairs_complete` — all requested pairs recorded on both arms.
2. `outputs_identical` — the candidate reproduces the control text exactly,
   per prompt, in every pair, and each arm is self-identical across pairs.
3. `no_per_prompt_slowdown` — no prompt in any pair falls below the
   `--per-prompt-floor` ratio, default 1.0.
4. `decode_lower_bound_above_one` — the 95% lower bound on the decode ratio
   exceeds 1.0.
5. `ttft_upper_bound_below_one` — the 95% upper bound on the TTFT ratio is
   below 1.0.
6. `no_structural_failure` — identical prompt sets and hardware identity,
   both arms non-empty, telemetry complete, CUDA active, automatic qualifier
   gate passed, and zero host-MoE fallback in every trial.

Hardware equality is checked on stable identity only: cores, CPU, GPU, GPU
count, total RAM, and total VRAM. Available RAM is excluded and reported as a
range instead. The preserved five-trial control varies from 2.289 to 2.312 GiB
available with no intervention at all, so gating on that field would fail
every real comparison. A replay of those artifacts is what exposed this; the
first implementation did gate on the whole hardware block and reported four
spurious failures.

The exit code is 0 only for `promote`; `hold` exits 1 and a controller or
acceptance failure exits 2. Decode expert read bytes and miss counts are
carried per pair so the storage-side claim in each candidate can be checked
against the same evidence.

Two pairs cannot promote anything in practice: at one degree of freedom the
95% critical value is 12.706205, so the interval swamps the effect. The
preregistered five pairs remain the requirement, and the controller reports
`hold` rather than a bound it cannot support.

## Validation

- 20 focused tests pass, covering the statistics against published values,
  AB/BA alternation, prime-trial alternation and exclusion, hash-verified
  resume, binding drift, pair-count drift, divergent output, per-prompt
  slowdown, host-MoE fallback, drifting available RAM, differing GPU identity,
  false acceptance, stale-attempt recovery, reanalysis without execution, and
  incomplete state.
- `trial_metrics` was run against the five preserved Ornith397 control
  artifacts. It reproduces the recorded aggregates exactly: median sustained
  0.824033698 tok/s, minimum 0.773577249, maximum 0.838411793, and median
  trial TTFT 43.508444 s. All five outputs hash identically, which is the
  premise the exact-output gate depends on.
- Two end-to-end replays drive the controller with the real preserved
  artifacts rather than a synthetic schema. Replaying each control trial
  against itself gives a ratio of exactly 1.0 and reports `hold` with no
  structural failure, so a null effect cannot be promoted. Replaying the same
  artifacts with a uniform 1.12 throughput and 0.93 TTFT scaling reports
  `promote` with matching bounds, five pairs, two priming trials, and
  identical outputs.
- Python compilation, `c/tools/check_docs.py`, and `git diff --check` pass.
- `c/tests/test_run_perf_trials.py` still passes unchanged.

## Launch form

The two arm templates are separated by `:::` and each must contain exactly one
`{output}` placeholder:

```sh
./.venv/bin/python c/tools/run_paired_perf_trials.py \
  --state c/bench/ornith397_atlas_pairs/state.json \
  --output-dir c/bench/ornith397_atlas_pairs \
  --report c/bench/ornith397_atlas_pairs/report.json \
  --pairs 5 --prime-trials 1 --bind-engine-source \
  --bind c/qwen --bind c/fixtures/ornith397_legacy_prompts.json \
  --bind c/tools/qualify_tiered_model.py \
  --bind c/ornith397/expert-q3.json \
  --bind c/ornith397/.conversion-state.json \
  -- {control qualifier template with --q3-route-atlas 0 --output {output}} \
  ::: {candidate qualifier template with --q3-route-atlas 1 --output {output}}
```

Both qualifier templates are the preregistered ones from the candidate
reports and are not restated here, so they cannot drift between documents.

## Identity

- repository HEAD at implementation:
  `35d9ab86fa2b8f8acbf56c8438413dfb08f2a2a7`;
- controller SHA-256:
  `4310ff622b5a9a142e607f3e5d176f2ee0a6defc4ea767ba30f097e2adb2f718`;
- test SHA-256:
  `07ff23cc27bb4f027a46d21d619a9b15264720e8826c45a3c2845cfca64d2087`.

The working tree is intentionally dirty. Any later edit to either file
invalidates these hashes, so the real run must rebind and restamp them before
the first pair executes.
