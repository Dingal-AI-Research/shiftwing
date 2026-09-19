"""Read-only forensics of the checkpoint, captured request, and saved traces.

Run from the Shiftwing repository root using .venv/bin/python.
This recomputes saved numerical comparisons; it does not run new inference.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer

ROOT = Path.cwd()
OUT = ROOT / 'docs/research/deepseek-loop-investigation-2026-09-17'
MODEL = ROOT / 'c/deepseek-v4-flash-0731'
REVIEW = ROOT.parent / 'localforge/.localforge/reviews/04138cb6ec225910/1789647622895-6696afcd-5445-4ab6-87e7-e6c4da80b4ae.request.json'
ATTACHMENT = Path('/mnt/c/Users/dinga/.codex/attachments/fc4de0f0-afc0-4af9-b3e4-f926abaceb97/pasted-text.txt')
TRACE = ROOT / 'docs/research/deepseek-full-parity-2026-09-11'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compare(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    diff = a - b
    return {'relative_l2': float(np.linalg.norm(diff) / np.linalg.norm(b)),
            'cosine': float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b))),
            'max_absolute_error': float(np.abs(diff).max()),
            'exact_fraction': float((a == b).mean())}

manifest = json.loads((MODEL / 'model-manifest.json').read_text())
records = [r for values in manifest['inventory'].values() for r in values]
params, sizes, dtypes = Counter(), Counter(), Counter()
coverage = defaultdict(set)
for r in records:
    is_draft = r['file'].startswith('dspark/')
    category = 'draft' if is_draft else 'base_routed' if r['expert'] is not None else 'base_other'
    sizes[category] += r['nbytes']
    dtypes[r['dtype']] += r['nbytes']
    # Quantization scale encodings and integer routing maps are not learned scalars.
    if r['dtype'] not in ('F8_E8M0', 'I64'):
        params[category] += math.prod(r['shape']) * (2 if r['dtype'] == 'I8' else 1)
    if not is_draft and r['expert'] is not None and r['name'].endswith('.weight'):
        coverage[r['layer']].add((r['expert'], r['projection']))

text = ATTACHMENT.read_text()
completion = text[text.index('{\n"verdict"'):].strip()
actions = []
for line in completion.splitlines():
    if line.startswith('"Fix the Playwright'):
        actions.append(json.loads(line.rstrip(',')))
duplicates = [{'action_numbers': [i+1 for i, a in enumerate(actions) if a == action], 'text': action}
              for action, count in Counter(actions).items() if count > 1]
blocks = []
for width in range(2, len(actions)//2+1):
    for start in range(len(actions)-2*width+1):
        if actions[start:start+width] == actions[start+width:start+2*width]:
            block = '\n'.join(actions[start:start+width])
            blocks.append({'first_start': start+1, 'second_start': start+width+1,
                           'actions': width, 'words': len(block.split()), 'characters': len(block)})
nonadjacent_blocks = []
for first in range(len(actions)):
    for second in range(first+1, len(actions)):
        width = 0
        while first+width < second and second+width < len(actions) and actions[first+width] == actions[second+width]:
            width += 1
        if width >= 2:
            nonadjacent_blocks.append({'first_start':first+1,'second_start':second+1,'actions':width,
                                       'words':len('\n'.join(actions[first:first+width]).split())})

request = json.loads(REVIEW.read_text())
sys.path.insert(0, str(ROOT/'c'))
from tools.deepseek_v4_protocol import render_deepseek_chat
rendered = render_deepseek_chat(request['messages'], reasoning_effort=request.get('reasoning_effort'))
tokenizer = Tokenizer.from_file(str(MODEL/'tokenizer.json'))
ids = tokenizer.encode(rendered, add_special_tokens=False).ids
rendered_hash = hashlib.sha256(rendered.encode()).hexdigest()

native = np.fromfile(TRACE/'native.logits', dtype=np.float32)
ref = np.fromfile(TRACE/'reference.logits', dtype=np.float32)
logits = compare(native, ref)
for name, vector in [('native', native), ('reference', ref)]:
    top = np.argsort(vector)[-10:][::-1]
    logits[name+'_top10'] = [{'token':int(i), 'text':tokenizer.decode([int(i)]),
                            'logit':float(vector[i])} for i in top]
    logits[name+'_margin'] = float(vector[top[0]]-vector[top[1]])
logits['native_traced_identical'] = bool(np.array_equal(native, np.fromfile(TRACE/'traced.logits', np.float32)))
def softmax(values):
    values = values.astype(np.float64)
    exp = np.exp(values-values.max())
    return exp/exp.sum()
a,b = softmax(native),softmax(ref)
middle = (a+b)/2
probabilities = {
    'scope':'saved 32-token fixture, probabilities at temperature 1 without top-p truncation',
    'total_variation':float(np.abs(a-b).sum()/2),
    'kl_reference_to_native_nats':float(np.sum(b*np.log(b/a))),
    'js_nats':float((np.sum(a*np.log(a/middle))+np.sum(b*np.log(b/middle)))/2),
    'native_probability_of_reference_top':float(a[ref.argmax()]),
    'reference_probability_of_reference_top':float(b[ref.argmax()]),
    'native_probability_of_native_top':float(a[native.argmax()]),
    'reference_probability_of_native_top':float(b[native.argmax()]),
}
layers = []
for layer in range(43):
    a = np.fromfile(TRACE/f'traced.layer{layer}', np.float32)
    b = np.fromfile(TRACE/f'reference.layer{layer}', np.float32)
    layers.append({'layer':layer, **compare(a,b)})

report = {
    'scope': 'fresh metadata/request inspection and recomputation of saved 2026-09-11 numerical traces; no fresh model inference',
    'source': manifest['source'],
    'manifest_sha256': sha(MODEL/'model-manifest.json'),
    'current_binary_sha256': sha(ROOT/'c/deepseek_v4'),
    'tokenizer_sha256': sha(MODEL/'tokenizer.json'),
    'record_count': len(records), 'segment_count': len(manifest['segments']),
    'payload_bytes': sum(sizes.values()), 'payload_by_category': dict(sizes),
    'payload_by_dtype': dict(dtypes), 'learned_scalar_counts': dict(params),
    'scalar_count_method': 'exclude F8_E8M0 quantization scales and I64 routing maps; unpack I8 FP4 weights as two values per byte',
    'base_routed_expert_projection_coverage': {str(k):len(v) for k,v in sorted(coverage.items())},
    'request': {'path':str(REVIEW), 'sha256':sha(REVIEW),
                **{k:request.get(k) for k in ('model','temperature','top_p','max_tokens','enable_thinking','reasoning_effort')},
                'rendered_tokens':len(ids), 'rendered_sha256':rendered_hash,
                'expected_prompt_sha256':request.get('expected_prompt_sha256'),
                'hash_matches':rendered_hash == request.get('expected_prompt_sha256'),
                'rendered_suffix':rendered[-100:]},
    'pasted_output': {'sha256':sha(ATTACHMENT), 'complete_action_count':len(actions),
                      'unique_exact_actions':len(set(actions)), 'duplicate_actions':duplicates,
                      'exact_repeated_adjacent_blocks':blocks,
                      'repeated_nonoverlapping_blocks':nonadjacent_blocks, 'characters':len(completion)},
    'saved_logit_comparison':logits, 'saved_layer_comparisons':layers,
    'saved_probability_comparison':probabilities,
}
(OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('pasted_output','saved_layer_comparisons')},indent=2))
print(json.dumps({'actions':len(actions),'unique':len(set(actions)),'blocks':blocks,
                  'layers':[{k:r[k] for k in ('layer','relative_l2')} for r in layers]},indent=2))
