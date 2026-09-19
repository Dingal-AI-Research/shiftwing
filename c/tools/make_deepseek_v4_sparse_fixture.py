#!/usr/bin/env python3
"""Pinned 64-key online-softmax oracle at every block boundary."""
import hashlib,json
from pathlib import Path
import torch
from reference_deepseek_v4_prefix import sparse
from make_deepseek_v4_fixture import REFERENCE_HASHES

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    kernel=args.source/'inference/kernel.py';digest=hashlib.sha256(kernel.read_bytes()).hexdigest()
    if digest!=REFERENCE_HASHES['inference/kernel.py']:raise ValueError('pinned kernel identity changed')
    q=torch.tensor([((i*7)%23-11)/8 for i in range(16)],dtype=torch.bfloat16).reshape(1,1,2,8)
    kv=torch.tensor([((t*13+d*3)%37-18)/16*(1+t//64) for t in range(137) for d in range(8)],dtype=torch.bfloat16).reshape(1,137,8)
    sink=torch.tensor([0.,-1.],dtype=torch.float32)
    ids=torch.tensor([t if t%13 or t==0 else -1 for t in range(137)],dtype=torch.int32)
    cases=[]
    for size in (1,63,64,65,127,128,137):
        expected=sparse(q,kv,sink,ids[:size].reshape(1,1,size),.125).float().flatten().tolist()
        cases.append({'selected':size,'output':expected})
    args.output.write_text(json.dumps({'reference_kernel_sha256':digest,'cases':cases},indent=2)+'\n')
if __name__=='__main__':main()
