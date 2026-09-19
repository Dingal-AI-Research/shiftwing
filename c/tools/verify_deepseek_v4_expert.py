#!/usr/bin/env python3
"""Independent CPU Torch equations vs native CPU/CUDA on released FP4 experts.

Implements the pinned inference/model.py Expert.forward and kernel.py fp4_gemm
block accumulation using Torch operations, without calling the native reference.
"""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from deepseek_v4_spec import SOURCE_REVISION


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def activation(x):
    blocks=x.reshape(x.shape[0],-1,128)
    scale=torch.pow(2.0,torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4)/448.0))).float()
    quant=(blocks/scale[...,None]).clamp(-448,448).to(torch.float8_e4m3fn).float()
    return quant.reshape(x.shape),scale


def linear(x,weight,scale):
    a,ascale=activation(x)
    out=torch.zeros((x.shape[0],weight.shape[0]),dtype=torch.float32)
    for block in range(x.shape[1]//32):
        inner=a[:,block*32:(block+1)*32] @ weight[:,block*32:(block+1)*32].T
        out += inner * ascale[:,block//4,None] * scale[None,:,block]
    return out.to(torch.bfloat16).float()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--batch',type=int,default=3);p.add_argument('--probe',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    args=p.parse_args(); model=args.model.resolve();batch=args.batch
    if not 1<=batch<=128:raise ValueError('reference batch must be 1..128')
    state=json.loads((model/'conversion-state.json').read_text())
    validation=json.loads((model/'recovery-validation.json').read_text())
    torch.set_num_threads(4)
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])
    results=[]
    with tempfile.TemporaryDirectory(prefix='dsv4-released-experts-') as tmp:
        root=Path(tmp)
        groups={f'experts/layer-{layer:02d}.bin':state['inventory'][f'experts/layer-{layer:02d}.bin'] for layer in (0,1)}
        for group in groups:
            if validation[group]['binding']['segment'] != state['completed'][group]:
                raise ValueError('source comparison does not bind expert group')
            target=root/group;target.parent.mkdir(exist_ok=True);target.symlink_to(model/group)
        (root/'model-manifest.json').write_text(json.dumps({'schema':'colib.deepseek-v4.model-manifest.v1','source':{'revision':SOURCE_REVISION},'inventory':groups}))
        for layer,expert in ((0,0),(0,127),(1,255)):
            group=f'experts/layer-{layer:02d}.bin'
            inventory={r['name']:r for r in groups[group]}
            prefix=f'layers.{layer}.ffn.experts.{expert}'
            def read(suffix):
                r=inventory[prefix+suffix]
                with (model/group).open('rb') as f:
                    f.seek(r['offset']);data=f.read(r['nbytes'])
                if hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('record checksum mismatch')
                return torch.from_numpy(np.frombuffer(data,dtype=np.uint8).copy()).reshape(r['shape'])
            weights={}
            for projection in ('w1','w2','w3'):
                packed=read('.'+projection+'.weight')
                code=torch.stack((packed&15,packed>>4),dim=-1).reshape(packed.shape[0],-1).long()
                weights[projection]=(lut[code],torch.pow(2.0,read('.'+projection+'.scale').float()-127))
            generator=torch.Generator().manual_seed(layer*256+expert+17)
            x=(torch.randn(batch,4096,generator=generator)*torch.tensor([.03,.25,1.5])[torch.arange(batch)%3,None]).to(torch.bfloat16).float()
            routes=torch.tensor([.03125,.1875,.625])[torch.arange(batch)%3]
            started=time.monotonic()
            gate=linear(x,*weights['w1']).clamp(max=10)
            up=linear(x,*weights['w3']).clamp(-10,10)
            middle=(routes[:,None]*(torch.nn.functional.silu(gate)*up)).to(torch.bfloat16).float()
            expected=linear(middle,*weights['w2']).numpy()
            reference_seconds=time.monotonic()-started
            input_path=root/'input.bin';input_path.write_bytes(x.numpy().tobytes()+routes.numpy().tobytes())
            for mode in ('cpu','cuda'):
                output=root/f'{mode}.bin'
                process=subprocess.run([str(args.probe.resolve()),str(root),str(layer),str(expert),str(batch),str(input_path),str(output)],env={**os.environ,'COLI_CUDA':'1' if mode=='cuda' else '0','OMP_NUM_THREADS':'4'},capture_output=True,text=True,timeout=120,check=True)
                actual=np.frombuffer(output.read_bytes(),dtype=np.float32).reshape(expected.shape)
                difference=np.abs(actual-expected)
                relative_l2=float(np.linalg.norm(actual-expected)/max(1e-20,np.linalg.norm(expected)))
                passed=bool(np.isfinite(actual).all() and relative_l2<=0.002)
                result={'layer':layer,'expert':expert,'batch':batch,'mode':mode,'pass':passed,'max_absolute_error':float(difference.max()),'relative_l2_error':relative_l2,'exact_fraction':float(np.mean(actual==expected)),'reference_seconds':reference_seconds,'input_sha256':digest(input_path),'output_sha256':digest(output),'probe_output':process.stdout.strip()}
                results.append(result);print(json.dumps(result),flush=True)
                if not passed: raise ValueError('released expert numerical comparison failed')
    report={'source_revision':SOURCE_REVISION,'probe_sha256':digest(args.probe),'reference_model_sha256':digest(args.source/'inference/model.py'),'reference_kernel_sha256':digest(args.source/'inference/kernel.py'),'torch':torch.__version__,'passed':all(r['pass'] for r in results),'results':results,'scope':'three released experts; independent Torch equations; not full-model quality qualification'}
    args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
