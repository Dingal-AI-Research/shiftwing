#define QWEN_NO_MAIN
#include "../qwen.c"

int main(void){
    setenv("MTP","0",1);setenv("KV16","0",1);setenv("EXPERT_RAM","4",1);setenv("PREFETCH_THREADS","0",1);
    static Model m;model_init(&m,"qwen_tiny_i4");Oracle o=load_oracle("ref_qwen_i4.json");
    int alt[]={9,8,7,6},*prompt[2]={o.prompt,alt},plen[2]={o.nprompt,4};enum{B=2,STEPS=8};
    ResidentBatchState resident={0};if(!resident_batch_init(&m,&resident,B)){fprintf(stderr,"resident init failed\n");return 1;}
    float*logits=falloc(m.c.vocab);
    for(int slot=0;slot<B;slot++){model_reset(&m);prefill_dispatch(&m,prompt[slot],plen[slot],logits);if(!resident_import_model_slot(&m,&resident,slot,logits)){fprintf(stderr,"resident import failed\n");return 1;}}
    int slots[B]={0,1},tokens[B],got[B][STEPS];
    for(int step=0;step<STEPS;step++){for(int slot=0;slot<B;slot++){tokens[slot]=argmax(resident.logits+(int64_t)slot*m.c.vocab,m.c.vocab);got[slot][step]=tokens[slot];}if(step+1<STEPS&&!resident_forward_tokens(&m,&resident,slots,tokens,B)){fprintf(stderr,"resident forward failed\n");return 1;}}
    int token_bad=0,state_bad=0;float hidden_diff=0.f,logit_diff=0.f,recurrent_diff=0.f,kv_diff=0.f;
    int cd=2*m.c.lin_k_heads*m.c.lin_k_dim+m.c.lin_v_heads*m.c.lin_v_dim,kvrows=m.c.n_kv_heads*m.c.head_dim;
    for(int slot=0;slot<B;slot++){
        model_reset(&m);prefill_dispatch(&m,prompt[slot],plen[slot],logits);
        for(int step=0;step<STEPS;step++){int token=argmax(logits,m.c.vocab);if(token!=got[slot][step])token_bad++;if(step+1<STEPS)forward_token(&m,token,logits);}
        if(m.pos!=resident.pos[slot])state_bad++;
        for(int h=0;h<m.c.hidden;h++){float d=fabsf(m.last_hidden[h]-resident.last_hidden[(int64_t)slot*m.c.hidden+h]);if(d>hidden_diff)hidden_diff=d;}
        for(int v=0;v<m.c.vocab;v++){float d=fabsf(logits[v]-resident.logits[(int64_t)slot*m.c.vocab+v]);if(d>logit_diff)logit_diff=d;}
        for(int li=0;li<m.c.n_layers;li++){ResidentLayerState*r=&resident.layer[li];
            if(m.layer[li].type==LT_LINEAR){GdnW*w=&m.layer[li].gdn;size_t nc=(size_t)m.c.conv_kernel*cd,ns=(size_t)m.c.lin_v_heads*m.c.lin_k_dim*m.c.lin_v_dim;for(size_t i=0;i<nc;i++){float d=fabsf(w->conv_state[i]-r->conv[(size_t)slot*nc+i]);if(d>recurrent_diff)recurrent_diff=d;}for(size_t i=0;i<ns;i++){float d=fabsf(w->state[i]-r->gdn[(size_t)slot*ns+i]);if(d>recurrent_diff)recurrent_diff=d;}}
            else{AttnW*w=&m.layer[li].attn;size_t n=(size_t)m.pos*kvrows,base=(size_t)slot*resident.max_seq*kvrows;for(size_t i=0;i<n;i++){float d=fabsf(w->k_cache[i]-r->k[base+i]);if(d>kv_diff)kv_diff=d;d=fabsf(w->v_cache[i]-r->v[base+i]);if(d>kv_diff)kv_diff=d;}}
        }
    }
    SessionState exported={0},disk={0};ResidentBatchState restored={0};char path[256];snprintf(path,sizeof(path),"/tmp/colib-resident-%ld.bin",(long)getpid());
    int disk_bad=!resident_export_session(&m,&resident,0,&exported)||!session_state_write(path,&exported)||!session_state_read(&m,path,&disk)||!resident_batch_init(&m,&restored,1)||!resident_import_session(&m,&restored,0,&disk);
    int continuation_bad=0,one=0;
    if(!disk_bad)for(int step=0;step<STEPS;step++){int a=argmax(resident.logits,m.c.vocab),b=argmax(restored.logits,m.c.vocab);if(a!=b)continuation_bad++;if(step+1<STEPS){if(!resident_forward_tokens(&m,&resident,&one,&a,1)||!resident_forward_tokens(&m,&restored,&one,&b,1)){continuation_bad++;break;}}}
    unlink(path);session_state_free(&exported);session_state_free(&disk);resident_batch_free(&m,&restored);
    printf("resident whole-model batch: rows=%d tokens=%s hidden=%.3g logits=%.3g recurrent=%.3g kv=%.3g disk=%s\n",B,token_bad?"mismatch":"exact",hidden_diff,logit_diff,recurrent_diff,kv_diff,(disk_bad||continuation_bad)?"mismatch":"exact");
    resident_batch_free(&m,&resident);free(logits);
    return(token_bad||state_bad||hidden_diff>1e-5f||logit_diff>1e-4f||recurrent_diff>1e-5f||kv_diff>1e-6f||disk_bad||continuation_bad)?1:0;
}
