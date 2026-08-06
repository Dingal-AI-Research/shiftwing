# Phase 8 Experiment 15: Ornith397 Q3 Cold-Start Coherence Failure

## Abstract

This experiment tests whether the selected Ornith397 grouped-3-bit runtime
produces meaningful free-running text before Gate 8 is approved. Structural
integrity and teacher-forced prediction passed, but a four-prompt run started
without inference warm-up produced garbled or incomplete text. Automatic
checks passed because they verify execution, telemetry, and nonempty output,
not meaning. Semantic review therefore rejected the artifact. Warm-up and
upload-staging controls did not restore meaning. A native-q3 control was
coherent, an actual-dimension kernel comparison was exact, and code inspection
then found a representation-dispatch regression in resident batched prefill.
After repair, the expanded-q4 control produced the expected coherent answer.

## 1. Research question

Does the exact release candidate produce coherent text from a cold start, and
if not, is warm-up history the main observed variable separating the failed
run from the earlier coherent q3 profile?

Teacher forcing and free-running generation measure different properties.
Teacher forcing supplies the correct previous token at every position, so a
small prediction error does not alter later inputs. Free-running generation
feeds each selected token back into the model. One early error can therefore
change the entire continuation. Both controls are necessary.

## 2. Method

The schema-2 Gate-8 controller bound the base manifest, complete q3 sidecar,
current CUDA executable, fixed prompts, exact commands, and output hashes. It
ran these ordered controls:

1. verify all 480 q3 files, 480 headers, and 276,480 tensors;
2. capture 20 int4 prompts at 64 prediction positions each;
3. compare q3 against those 1,280 int4 positions;
4. generate four q3 outputs with zero warm-up passes; and
5. stop for explicit semantic review before tool or tier qualification.

The coherence profile used eight CPU threads, 18 GiB of expert RAM, 6 GiB of
expert VRAM, expanded-q4 CUDA execution for the q3 sidecar, persistent direct
I/O, pinned upload, decode protection, and one measured 64-token pass. The
diagnostic kept all of these variables fixed and changed only `warmup_passes`
from zero to two.

## 3. Results

| Control | Result |
|---|---:|
| Q3 sidecar audit | 480/480 files and headers; 276,480 tensors; pass |
| Int4 reference | 1,280/1,280 positions; pass; `a79219e2…9bf0` |
| Q3 teacher forcing | 1,243/1,280 matches; 97.109375%; pass |
| Weakest q3 prompt | 59/64 matches; 92.1875%; pass |
| Zero-warmup coherence machine checks | pass; `616ad712…def2` |
| Zero-warmup semantic review | reject |
| Device-MoE layer-forwards | 12,060/12,060; zero host fallback |
| Coherence-stage sustained rate | 0.735595 tok/s |
| Two-warmup diagnostic | semantic reject; `eb73fd0d…70a4c` |
| Warmed measured sustained rate | 1.237648 tok/s |
| Measured text versus first pass | four of four byte-identical |
| Current-engine int4 control | coherent; `28b2a2ab…075f` |
| Int4 control device-MoE | 3,840/3,840; zero host fallback |
| Native packed-q3 control | coherent; 0.477183 tok/s cold decode |
| Expanded q3, pinned staging disabled | same malformed text; 0.632182 tok/s |
| Actual-shape native-q3 versus expanded-q4 CUDA | zero maximum difference |
| Fixed expanded-q4 cold control | coherent; `5c8cdbf8…f660` |
| Fixed control timing | 0.563045 tok/s; 177.392 s TTFT |

The four retained outputs began with:

- `Here is the answer in the **) "006".` followed by repeated quotation marks;
- `Here **code for the TypeScript:` followed by disconnected numbers;
- `Hereed you!h of the 2 1000` followed by disconnected numbers; and
- `The trade between memory is a data:` with no completed explanation.

These outputs do not answer their prompts. Nonempty text and valid CUDA
telemetry are therefore insufficient evidence of language-model correctness.

## 4. Interpretation

The result does not look like missing weights or a broadly broken q3
conversion: the independent inventory passes, teacher-forced agreement is
97.11%, and an earlier q3 artifact answered the same four prompts coherently.
The strongest observed difference is runtime history. The coherent artifact
ran two complete warm-up passes (eight hidden turns) before measurement; the
failed artifact ran none.

The controlled diagnostic falsified the warm-up hypothesis. It executed two
complete warm-up passes followed by one measured pass. Performance improved as
route residency accumulated: the measured aggregate reached 1.237648 token/s.
Meaning did not improve. For each prompt, both warm-up outputs and the measured
output were byte-identical to the original invalid text. The result therefore
cannot be explained by a merely cold route cache or insufficient startup
prewarming.

The earlier coherent artifact must differ in some other way. Its manifests and
prompts appear equivalent, but it predates the current rebuilt executable and
some loader/runtime hardening. One possibility is that the older run did not
exercise every selected q3 tensor and silently used int4 base tensors; the
current loader now fails closed instead of allowing that mixture.

The current int4 control answered the first frozen prompt coherently: `Here is
an explanation tailored for an undergraduate programmer...`. It used the same
current executable, prompt renderer, tokenizer, dense layers, CUDA-resident
graph, memory budgets, and zero-warmup profile. All 3,840 MoE layer-forwards
ran on device with no host fallback. Its artifact SHA-256 is
`28b2a2abfd78b040119c49d090ec69627dbee814f114ea59c427f2894bf4075f`.
This isolates the observed failure to q3 expert execution or q3 precision, not
the shared engine.

## 5. Isolation and repair

The native packed-q3 control generated the same coherent opening as the int4
control and the earlier accepted q3 profile. This proved that the q3 sidecar,
its scales, and the common model graph retained useful language behavior. A
second expanded-q4 control disabled pinned upload staging but reproduced the
malformed text exactly, falsifying a host-staging lifetime hypothesis.

The CUDA tests were then extended to compare native packed q3 directly with
its exact expanded-q4 representation at Ornith397's routed-expert dimensions:
4,096 hidden channels, 1,024 intermediate channels, group size 128, and ten
routes. Their grouped-MoE outputs had zero maximum difference. Thus neither
the expansion nor the CUDA arithmetic explained the full-model divergence.

The remaining difference was dispatch. Resident batched prefill uploaded an
expanded-q4 buffer when `Q3_NATIVE=0`, but selected the q3 kernel whenever the
source matrix had `fmt==3`. The kernel consequently interpreted four-bit
nibbles as packed three-bit triplets. This corrupted prompt state before the
ordinary one-token decode path began. The fix requires both `fmt==3` and
`Q3_NATIVE=1` before selecting the q3 batch kernel; otherwise it supplies the
expanded row stride to the q4 batch kernel. Native-q3 telemetry is likewise
counted only when the native route is active.

The rebuilt CUDA and session suites passed. A cold 64-token production-route
control then generated the expected undergraduate hash-table explanation,
used device MoE for all 3,840 layer-forwards, and had no host fallback. Its
artifact SHA-256 is
`5c8cdbf8b43b745466dccdc2c739260366e036c3dc7c76a5f5b851dd88f8f660`.
Because the executable changed, the q3 teacher-forced and frozen four-prompt
coherence controls must be regenerated before semantic acknowledgement.

## 6. Conclusion

Gate 8 remains open pending regenerated binary-bound controls, but the observed
coherence failure now has a reproduced cause and a validated repair. It was a
runtime representation-dispatch defect, not evidence that three-bit Ornith397
weights are intrinsically incoherent. The manual review boundary prevented a
fast, nonempty, but meaningless result from being promoted automatically.
