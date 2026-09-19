#!/usr/bin/env python3
"""Compare released dense records with independent CPU Torch equations."""
import argparse, hashlib, json, os, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import torch
from verify_deepseek_v4_expert import activation
from deepseek_v4_spec import SOURCE_REVISION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','source','probe','report'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();state=json.loads((args.model/'conversion-state.json').read_text())
    proofs={}
    for file in ('historical-validation.json','recovery-validation.json'):proofs.update(json.loads((args.model/file).read_text()))
    inventory={r['name']:(g,r) for g,items in state['inventory'].items() for r in items if g.startswith('dense/')}
    torch.set_num_threads(4);results=[]
    with tempfile.TemporaryDirectory(prefix='dsv4-dense-reference-') as tmp:
        root=Path(tmp)
        for case in ('layers.0.attn.wq_a','layers.0.attn.wq_b','layers.0.attn.wo_a','layers.2.attn.compressor.wkv','layers.2.attn.indexer.weights_proj','layers.0.ffn.gate'):
            group,record=inventory[case+'.weight'];records=[record]
            if record['dtype']=='F8_E4M3':records.append(inventory[case+'.scale'][1])
            if proofs[group]['binding']['segment']!=state['completed'][group]:raise ValueError('source proof mismatch')
            target=root/group;target.parent.mkdir(exist_ok=True)
            if not target.exists():target.symlink_to((args.model/group).resolve())
            (root/'model-manifest.json').write_text(json.dumps({'schema':'colib.deepseek-v4.model-manifest.v1','source':{'revision':SOURCE_REVISION},'inventory':{group:records}}))
            def read(r):
                with (args.model/group).open('rb') as f:f.seek(r['offset']);data=f.read(r['nbytes'])
                if hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('record checksum mismatch')
                return torch.from_numpy(np.frombuffer(data,dtype=np.uint8).copy())
            bf16=record['dtype']=='BF16'
            weight=read(record).view(torch.bfloat16 if bf16 else torch.float8_e4m3fn).reshape(record['shape']).float()
            x=(torch.randn(17,weight.shape[1],generator=torch.Generator().manual_seed(712))*torch.tensor([.03,.25,1.5])[torch.arange(17)%3,None]).to(torch.bfloat16).float()
            if bf16:expected=(x@weight.T).numpy()
            elif case.endswith('.wo_a'):
                ws=torch.pow(2.,read(records[1]).reshape(records[1]['shape']).float()-127)
                dequant=(weight*ws.repeat_interleave(128,0).repeat_interleave(128,1)[:weight.shape[0],:weight.shape[1]]).to(torch.bfloat16).float()
                expected=(x@dequant.T).to(torch.bfloat16).float().numpy()
            else:
                a,ascale=activation(x);wscale=torch.pow(2.0,read(records[1]).reshape(records[1]['shape']).float()-127)
                expected=torch.zeros(17,weight.shape[0]);row_scale=wscale[torch.arange(weight.shape[0])//128]
                for block in range(weight.shape[1]//128):expected+=(a[:,block*128:(block+1)*128]@weight[:,block*128:(block+1)*128].T)*ascale[:,block,None]*row_scale[None,:,block]
                expected=expected.to(torch.bfloat16).float().numpy()
            for mode,batch in (('cpu',1),('cuda',17)):
                inp=root/'input.bin';out=root/'output.bin';inp.write_bytes(x[:batch].numpy().tobytes())
                run=subprocess.run([str(args.probe.resolve()),str(root),case,str(batch),str(inp),str(out)],env={**os.environ,'COLI_CUDA':str(int(mode=='cuda')),'OMP_NUM_THREADS':'4'},capture_output=True,text=True,timeout=180,check=True)
                actual=np.fromfile(out,np.float32).reshape(batch,-1);ref=expected[:batch]
                relative=float(np.linalg.norm(actual-ref)/max(1e-20,np.linalg.norm(ref)))
                result={'case':case,'mode':mode,'batch':batch,'dtype':record['dtype'],'relative_l2_error':relative,'max_absolute_error':float(np.max(np.abs(actual-ref))),'exact_fraction':float(np.mean(actual==ref)),'passed':bool(np.isfinite(actual).all() and relative <= (2e-5 if bf16 else .002)),'probe':run.stdout.strip()}
                results.append(result);print(json.dumps(result),flush=True)
    report={'source_revision':SOURCE_REVISION,'probe_sha256':hashlib.sha256(args.probe.read_bytes()).hexdigest(),'reference_model_sha256':hashlib.sha256((args.source/'inference/model.py').read_bytes()).hexdigest(),'passed':all(r['passed'] for r in results),'results':results,'scope':'released dense projections, independent Torch equations; not complete model acceptance'}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise ValueError('released dense comparison failed')


if __name__=='__main__':main()
