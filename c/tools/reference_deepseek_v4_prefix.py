#!/usr/bin/env python3
"""Execute the pinned model.py Block on CPU with independent Torch kernel adapters.

The official block/attention/compressor/indexer/HC code is loaded unchanged.
Only TileLang primitives and lazy released-weight loading are adapted to CPU.
This bounded reference never instantiates the full model or loads all experts.
"""
import argparse, hashlib, importlib.util, json, math, sys, types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from verify_deepseek_v4_expert import activation, linear as fp4_linear
from deepseek_v4_spec import EXPECTED_COMPRESS_RATIOS

def simulate_fp8(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
    shape=x.shape;b=x.float().reshape(-1,block_size)
    scale=torch.pow(2.0,torch.ceil(torch.log2(b.abs().amax(-1).clamp_min(1e-4)/448)))
    q=(b/scale[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).float()
    if inplace:x.copy_((q*scale[:,None]).reshape(shape).to(x.dtype));return x
    return q.reshape(shape),scale.reshape(*shape[:-1],shape[-1]//block_size)

def simulate_fp4(x, block_size=32, inplace=False):
    shape=x.shape;b=x.float().reshape(-1,block_size)
    scale=torch.pow(2.,torch.ceil(torch.log2(b.abs().amax(-1).clamp_min(6*2**-126)/6)))
    v=b/scale[:,None];lut=torch.tensor([0,.5,1,1.5,2,3,4,6],dtype=torch.float32)
    # Ties to even mantissa, with explicit deterministic preference.
    order=torch.tensor([0,2,4,6,1,3,5,7]);distance=(v.abs()[...,None]-lut[order]).abs()
    q=lut[order[distance.argmin(-1)]]*v.sign();decoded=(q*scale[:,None]).reshape(shape).to(x.dtype)
    if inplace:x.copy_(decoded);return x
    return decoded,scale

def sinkhorn(mixes, scale, base, hc=4, iterations=20, eps=1e-6):
    pre=torch.sigmoid(mixes[...,:hc]*scale[0]+base[:hc])+eps
    post=2*torch.sigmoid(mixes[...,hc:2*hc]*scale[1]+base[hc:2*hc])
    comb=(mixes[...,2*hc:]*scale[2]+base[2*hc:]).reshape(*mixes.shape[:-1],hc,hc).softmax(-1)+eps
    comb=comb/(comb.sum(-2,keepdim=True)+eps)
    for _ in range(iterations-1):
        comb=comb/(comb.sum(-1,keepdim=True)+eps);comb=comb/(comb.sum(-2,keepdim=True)+eps)
    return pre,post,comb

def sparse(q,kv,sink,indices,scale):
    # Pinned kernel.py rounds the unnormalized online probabilities to BF16
    # before each 64-key value GEMM. The denominator remains FP32.
    out=torch.empty_like(q)
    for b in range(q.shape[0]):
        for t in range(q.shape[1]):
            ids=indices[b,t].long();query=q[b,t].float()
            maximum=torch.full((q.shape[2],),-torch.inf,dtype=torch.float32)
            denominator=torch.zeros_like(maximum);value=torch.zeros_like(query)
            for start in range(0,len(ids),64):
                group=ids[start:start+64];valid=group>=0
                selected=kv[b,group.clamp_min(0)].float()*valid[:,None]
                scores=(query@selected.T)*scale;scores[:,~valid]=-torch.inf
                previous=maximum;maximum=torch.maximum(maximum,scores.amax(-1))
                factor=torch.exp(previous-maximum)
                weights=torch.exp(scores-maximum[:,None])
                denominator=denominator*factor+weights.sum(-1)
                value=value*factor[:,None]+weights.bfloat16().float()@selected
            denominator+=torch.exp(sink-maximum)
            out[b,t]=(value/denominator[:,None]).to(q.dtype)
    return out

def rotate(x):
    data=x.float();n=data.shape[-1];width=1
    while width<n:
        shape=data.shape;blocks=data.reshape(*shape[:-1],-1,2,width)
        left,right=blocks[...,0,:],blocks[...,1,:]
        data=torch.stack((left+right,left-right),-2).reshape(shape);width*=2
    return (data/math.sqrt(n)).to(x.dtype)

class Reader:
    def __init__(self,model):
        self.model=model;state=json.loads((model/'conversion-state.json').read_text())
        self.records={r['name']:(g,r) for g,items in state['inventory'].items() for r in items};self.cache={}
    def raw(self,name):
        group,r=self.records[name]
        with (self.model/group).open('rb') as f:f.seek(r['offset']);data=f.read(r['nbytes'])
        if hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('released checksum mismatch: '+name)
        dtype={'BF16':torch.bfloat16,'F32':torch.float32,'I64':torch.int64,'F8_E4M3':torch.float8_e4m3fn,'F8_E8M0':torch.uint8,'I8':torch.uint8}[r['dtype']]
        return torch.from_numpy(np.frombuffer(data,np.uint8).copy()).view(dtype).reshape(r['shape'])
    def weight(self,name):
        if name in self.cache:return self.cache[name]
        value=self.raw(name);dtype=self.records[name][1]['dtype']
        if dtype in ('F8_E4M3','I8'):
            scale=torch.pow(2.,self.raw(name.replace('.weight','.scale')).float()-127)
            if dtype=='I8':
                lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float32)
                code=torch.stack((value&15,value>>4),-1).reshape(value.shape[0],-1).long();value=(lut[code],scale)
            else:value=(value.float(),scale)
        if '.experts.' not in name:self.cache[name]=value
        return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','model','tokens','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--layers',type=int,default=4);p.add_argument('--stage-input',type=Path);p.add_argument('--logits',action='store_true');args=p.parse_args()
    if not 1<=args.layers<=43 or (args.logits and (args.layers!=43 or args.stage_input)):raise ValueError('logits require an unconditioned complete 43-layer reference')
    torch.set_num_threads(4);torch.set_default_dtype(torch.bfloat16)
    stub=types.ModuleType('kernel');stub.act_quant=simulate_fp8;stub.fp4_act_quant=simulate_fp4;stub.hc_split_sinkhorn=sinkhorn;stub.sparse_attn=sparse
    stub.fp8_gemm=stub.fp4_gemm=lambda *a: (_ for _ in ()).throw(RuntimeError('unexpected eager quantized primitive'))
    sys.modules['kernel']=stub
    spec=importlib.util.spec_from_file_location('deepseek_pinned_cpu_reference',args.source/'inference/model.py');official=importlib.util.module_from_spec(spec);sys.modules[spec.name]=official;spec.loader.exec_module(official)
    official.rotate_activation=rotate;official.default_dtype=torch.float8_e4m3fn;official.scale_fmt='ue8m0';official.scale_dtype=torch.float8_e8m0fnu
    reader=Reader(args.model)
    class LazyLinear(nn.Module):
        def __init__(self,in_features,out_features,bias=False,dtype=None):
            super().__init__();self.prefix='';self.dtype=dtype or official.default_dtype
        @property
        def weight(self):
            value=reader.weight(self.prefix+'.weight')
            if isinstance(value,tuple):
                w,s=value;return (w*s.repeat_interleave(128,0).repeat_interleave(128,1)[:w.shape[0],:w.shape[1]]).to(torch.bfloat16)
            return value.to(self.dtype)
        def forward(self,x):
            value=reader.weight(self.prefix+'.weight')
            if not isinstance(value,tuple):return torch.nn.functional.linear(x,value.to(x.dtype))
            weight,scale=value;shape=x.shape;flat=x.float().reshape(-1,shape[-1])
            if reader.records[self.prefix+'.weight'][1]['dtype']=='I8':out=fp4_linear(flat,weight,scale)
            else:
                a,ascale=activation(flat);out=torch.zeros(flat.shape[0],weight.shape[0],dtype=torch.float32)
                rowscale=scale[torch.arange(weight.shape[0])//128]
                for block in range(flat.shape[1]//128):out+=(a[:,block*128:(block+1)*128]@weight[:,block*128:(block+1)*128].T)*ascale[:,block,None]*rowscale[None,:,block]
            return out.reshape(*shape[:-1],weight.shape[0]).to(x.dtype)
    official.Linear=official.ColumnParallelLinear=official.RowParallelLinear=LazyLinear
    ids=torch.from_numpy(np.fromfile(args.tokens,np.int32).astype(np.int64))[None,:];count=ids.shape[1]
    params=official.ModelArgs(compress_ratios=tuple(EXPECTED_COMPRESS_RATIOS),max_batch_size=1,max_seq_len=max(128,math.ceil(count/128)*128),dim=4096,moe_inter_dim=2048,n_layers=43,n_hash_layers=3,n_routed_experts=256,n_activated_experts=6,expert_dtype='fp4',route_scale=1.5,swiglu_limit=10,compress_rope_theta=160000,original_seq_len=65536,rope_factor=16)
    with torch.inference_mode():
        h=reader.raw('embed.weight')[ids].unsqueeze(2).repeat(1,1,4,1)
        for layer in range(args.layers):
            block=official.Block(layer,params)
            for name,module in block.named_modules():
                if isinstance(module,LazyLinear):module.prefix=f'layers.{layer}.'+name
            for name,param in list(block.named_parameters()):
                parent_name,_,field=name.rpartition('.');parent=block.get_submodule(parent_name) if parent_name else block
                data=reader.raw(f'layers.{layer}.'+name).to(param.dtype)
                setattr(parent,field,nn.Parameter(data,requires_grad=False))
            if args.stage_input:
                def replace_input(mod,inp,label):
                    data=np.fromfile(str(args.stage_input)+f'.layer{layer}.{label}',np.float32)
                    return (torch.from_numpy(data).reshape(1,count,4096).to(torch.bfloat16),*inp[1:])
                block.attn.register_forward_pre_hook(lambda mod,inp:replace_input(mod,inp,'attention_input'))
                block.ffn.register_forward_pre_hook(lambda mod,inp:replace_input(mod,inp,'ffn_input'))
            for label,module in (('attention',block.attn),('ffn',block.ffn),('qa',block.attn.wq_a),('qnorm',block.attn.q_norm),('query',block.attn.wq_b),('kv',block.attn.wkv)):

                def save(mod,inp,out,label=label):
                    out.float().numpy().tofile(str(args.output)+f'.layer{layer}.{label}')
                module.register_forward_hook(save)
            def save_input(mod,inp,label):inp[0].float().numpy().tofile(str(args.output)+f'.layer{layer}.{label}')
            block.attn.register_forward_pre_hook(lambda mod,inp:save_input(mod,inp,'attention_input'))
            block.ffn.register_forward_pre_hook(lambda mod,inp:save_input(mod,inp,'ffn_input'))
            block.attn.wo_b.register_forward_pre_hook(lambda mod,inp:save_input(mod,inp,'orank'))
            h=block(h,0,ids);h.float().numpy().tofile(str(args.output)+f'.layer{layer}')
            print(json.dumps({'layer':layer,'tokens':count,'finite':bool(torch.isfinite(h).all()),'norm':float(h.float().norm())}),flush=True)
            del block;reader.cache.clear()
        h[0,-1].float().numpy().tofile(args.output)
        if args.logits:
            # Reuse the upstream HC head and RMSNorm; only weight loading is lazy.
            reduced=official.Block.hc_head(types.SimpleNamespace(hc_eps=params.hc_eps,norm_eps=params.norm_eps),h,
                reader.raw('hc_head_fn').float(),reader.raw('hc_head_scale').float(),reader.raw('hc_head_base').float())
            norm=official.RMSNorm(params.dim,params.norm_eps);norm.weight=nn.Parameter(reader.raw('norm.weight').float(),requires_grad=False)
            logits=torch.nn.functional.linear(norm(reduced)[:,-1].float(),reader.raw('head.weight').float())
            logits.float().numpy().tofile(str(args.output)+'.logits')
            print(json.dumps({'logits_finite':bool(torch.isfinite(logits).all()),'top_token':int(logits.argmax(-1)[0])}),flush=True)

if __name__=='__main__':main()
