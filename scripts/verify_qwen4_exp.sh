#!/usr/bin/env bash
# End-to-end Qwen4-Exp correctness check against a transformers reference.
#
# The unit fixtures (hyper-connection, PLE n-gram, PLE layer, QSA selection)
# prove each algorithm in isolation. They cannot catch a composition error, and
# the port shipped three of those: an int4-hardcoded PLE gather, layer
# classification testing the wrong label, and a missing QSA indexer. This builds
# small real models through transformers, converts them with the production
# converter, runs the engine, and requires the engine's argmax to match.
#
# int8 is used for the comparison: int4-g128 on random Gaussian weights carries
# 10-30% reconstruction error at these dimensions, which swamps the signal. The
# precision question for real weights is a separate measurement.
#
# Deliberately excluded: real dimensions *combined with* a restricting indexer.
# The QSA selection is discrete, and its relu-sum over index heads amplifies
# near-zero crossings, so upstream quantization error (1-3% by the third layer)
# can flip which block is admitted. Verified as precision rather than logic: at
# a position with a partial tail the engine reproduced the reference's admitted
# set exactly, all 19 tokens, and quantizing the reference's own index_qk_proj
# does not reproduce the engine's choice. The effect is inflated at these sizes
# -- flipping one of four admitted blocks moves 25% of the visible context,
# where the real model admits 512 of ~4096 blocks at 16K and one flip moves
# 0.2%.
#
# Needs the transformers overlay (main has qwen4_exp, the pinned venv does not):
#   scripts/verify_qwen4_exp.sh
set -uo pipefail
ROOT=/home/dinga/Projects/shiftwing
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/.tf-main/site:$ROOT/.tf-main/transformers-src/src"
PY="$ROOT/.venv/bin/python"

if [ ! -d "$ROOT/.tf-main/transformers-src" ]; then
  echo "missing transformers overlay at .tf-main/transformers-src" >&2
  exit 2
fi

fail=0
run_case() {
  local name="$1"; shift
  local src=/tmp/vq_src out=/tmp/vq_out acts=/tmp/vq_acts
  rm -rf "$src" "$out" "$acts"; mkdir -p "$acts"
  "$PY" c/tools/make_qwen4_exp_oracle.py --outdir "$src" "$@" >/dev/null 2>&1 || {
    echo "  $name: GENERATE FAILED"; fail=$((fail+1)); return; }
  "$PY" -u c/tools/convert_qwen.py --indir "$src" --outdir "$out" \
      --xbits int8 --min-free-gb 1 >/dev/null 2>&1 || {
    echo "  $name: CONVERT FAILED"; fail=$((fail+1)); return; }
  "$PY" -c "
import json;t=json.load(open('$src/reference.json'))['tokens']
open('/tmp/vq_ids.txt','w').write(' '.join(map(str,t))+'\n')"
  local got
  got=$(env SNAP="$out" MTP=0 CTX=256 EXPERT_RAM=64 DUMP_ACTS=1 ACTS_DIR="$acts" \
            PREFIX_IDS=/tmp/vq_ids.txt NGEN=1 ./c/qwen 2>/dev/null \
        | sed -n 's/^PREFIX 0: *//p')
  "$PY" - "$name" "$got" <<'PY'
import json,sys,numpy as np
name,got=sys.argv[1],sys.argv[2].strip()
ref=json.load(open('/tmp/vq_src/reference.json'))
want=str(ref['argmax_per_position'][-1])
W,L,T=ref['hidden_width'],ref['layers'],len(ref['tokens'])
worst=0.0
for li in range(L):
    R=np.fromfile(f'/tmp/vq_src/reference_acts/layer-{li:03d}.f32',dtype=np.float32).reshape(T,W)
    E=np.stack([np.fromfile(f'/tmp/vq_acts/layer-{li:03d}-token-{t:06d}.f32',dtype=np.float32)
                for t in range(T)])
    worst=max(worst,float(np.linalg.norm(R-E)/np.linalg.norm(R)))
    # argmax agreement is the real assertion. The activation bound only catches
    # gross divergence: int8 error grows with hidden_size (~0.03 at hidden 64,
    # ~0.19 at 2560 over eight layers), so a tight bound would flag precision,
    # not correctness.
    ok = got==want and worst<0.25
print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} argmax {got} (want {want})  worst_layer_rel {worst:.4f}")
sys.exit(0 if ok else 1)
PY
  [ $? -ne 0 ] && fail=$((fail+1))
}

echo "Qwen4-Exp engine vs transformers reference (int8)"
run_case "baseline 4L"                 --layers 4  --experts 8
run_case "deep 16L"                    --layers 16 --experts 32
run_case "expert streaming 16L"        --layers 16 --experts 32
run_case "real dimensions 4L"          --layers 4  --experts 16 --real-dims
run_case "QSA indexer restricting"     --layers 8  --experts 8 --tokens 64 --indexer-budget 8
run_case "real dims 8L"                --layers 8  --experts 16 --tokens 64 --real-dims

echo
if [ "$fail" -eq 0 ]; then echo "all cases passed"; else echo "$fail case(s) FAILED"; fi
exit $((fail > 0))
