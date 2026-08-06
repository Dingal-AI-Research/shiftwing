#define QWEN_NO_MAIN
#include "../qwen.c"

static void baseline(Model*m,const int*prompt,int n,int*out,int steps,SessionState*checkpoint){
    float*logits=falloc(m->c.vocab);model_reset(m);prefill_dispatch(m,prompt,n,logits);
    for(int i=0;i<steps;i++){if(checkpoint&&!session_state_save(m,&checkpoint[i])){fprintf(stderr,"baseline checkpoint failed\n");exit(1);}out[i]=argmax(logits,m->c.vocab);if(i+1<steps)forward_token(m,out[i],logits);}free(logits);
}

int main(void){
    setenv("MTP","0",1);setenv("EXPERT_RAM","4",1);setenv("PREFETCH_THREADS","0",1);
    static Model m;model_init(&m,"qwen_tiny_i4");
#ifdef COLI_CUDA
    if(getenv("SESSION_CUDA")){setenv("COLI_CUDA","1",1);setenv("CUDA_DENSE","1",1);setenv("CUDA_F16","1",1);setenv("CUDA_EXPERTS","1",1);setenv("CUDA_EXPERT_GB","0.01",1);if(!cuda_backend_start()){fprintf(stderr,"CUDA session backend unavailable\n");return 1;}cuda_model_preload_dense(&m);}
#endif
    Oracle o=load_oracle("ref_qwen_i4.json");
    float*logits=falloc(m.c.vocab);model_reset(&m);prefill_dispatch(&m,o.prompt,o.nprompt,logits);
    SessionState state={0};if(!session_state_save(&m,&state)){fprintf(stderr,"session save failed\n");return 1;}
    int first[8],second[8];for(int i=0;i<8;i++){first[i]=argmax(logits,m.c.vocab);if(i+1<8)forward_token(&m,first[i],logits);}
    if(!session_state_restore(&m,&state,logits)){fprintf(stderr,"session restore failed\n");return 1;}
    for(int i=0;i<8;i++){second[i]=argmax(logits,m.c.vocab);if(i+1<8)forward_token(&m,second[i],logits);}
    int bad=0;for(int i=0;i<8;i++)bad+=first[i]!=second[i];printf("session recurrent+KV restore: %s pos=%d recurrent=%zu kv=%zu\n",bad?"mismatch":"exact",state.pos,state.recurrent_n,state.kv_n);
    SessionState reuse={0};if(!session_state_save(&m,&reuse)){fprintf(stderr,"reuse save failed\n");return 1;}void*reuse_r=reuse.recurrent,*reuse_k=m.kv16?(void*)reuse.kv_bf16:(void*)reuse.kv,*reuse_h=reuse.last_hidden;int reuse_token=argmax(logits,m.c.vocab);forward_token(&m,reuse_token,logits);if(!session_state_save(&m,&reuse)){fprintf(stderr,"reuse advance save failed\n");return 1;}int reuse_bad=reuse_r!=reuse.recurrent||reuse_k!=(m.kv16?(void*)reuse.kv_bf16:(void*)reuse.kv)||reuse_h!=reuse.last_hidden;printf("session snapshot buffers: %s recurrent-cap=%zu kv-cap=%zu\n",reuse_bad?"reallocated":"reused",reuse.recurrent_cap,reuse.kv_cap);session_state_free(&reuse);
    char session_path[256];snprintf(session_path,sizeof(session_path),"/tmp/colib-session-%ld.bin",(long)getpid());SessionState disk={0};int disk_ids[8];
    if(!session_state_write(session_path,&state)||!session_state_read(&m,session_path,&disk)||!session_state_restore(&m,&disk,logits)){fprintf(stderr,"disk session round trip failed\n");return 1;}
    for(int i=0;i<8;i++){disk_ids[i]=argmax(logits,m.c.vocab);if(i+1<8)forward_token(&m,disk_ids[i],logits);}int disk_bad=0;for(int i=0;i<8;i++)disk_bad+=disk_ids[i]!=first[i];FILE*corrupt=fopen(session_path,"r+b");if(!corrupt){fprintf(stderr,"open checkpoint for corruption failed\n");return 1;}fseek(corrupt,-1,SEEK_END);int byte=fgetc(corrupt);fseek(corrupt,-1,SEEK_END);fputc(byte^1,corrupt);fclose(corrupt);SessionState rejected={0};int corrupt_accepted=session_state_read(&m,session_path,&rejected);unlink(session_path);session_state_free(&rejected);session_state_free(&disk);printf("session disk checkpoint: %s corruption=%s\n",disk_bad?"mismatch":"exact",corrupt_accepted?"accepted":"rejected");
    session_state_free(&state);

    int alternate_prompt[]={9,8,7,6},expected[2][8],got[2][8];SessionState checkpoint[2][8]={{{0}}};baseline(&m,o.prompt,o.nprompt,expected[0],8,checkpoint[0]);baseline(&m,alternate_prompt,4,expected[1],8,checkpoint[1]);
    SessionState slots[2]={{0}};const int*prompt[2]={o.prompt,alternate_prompt};int plen[2]={o.nprompt,4};
    for(int s=0;s<2;s++){model_reset(&m);prefill_dispatch(&m,prompt[s],plen[s],logits);if(!session_state_save(&m,&slots[s])){fprintf(stderr,"slot save failed\n");return 1;}}
    int state_bad=0;for(int step=0;step<8;step++)for(int s=0;s<2;s++){if(!session_state_restore(&m,&slots[s],logits)){fprintf(stderr,"slot restore failed\n");return 1;}SessionState*want=&checkpoint[s][step];int rb=slots[s].recurrent_n!=want->recurrent_n||memcmp(slots[s].recurrent,want->recurrent,want->recurrent_n*sizeof(float)),kb=slots[s].kv_n!=want->kv_n||memcmp(slots[s].kv,want->kv,want->kv_n*sizeof(float));if(rb||kb){if(!state_bad)fprintf(stderr,"first state mismatch slot=%d step=%d recurrent=%d kv=%d\n",s,step,rb,kb);state_bad++;}got[s][step]=argmax(logits,m.c.vocab);if(step+1<8){forward_token(&m,got[s][step],logits);if(!session_state_save(&m,&slots[s])){fprintf(stderr,"slot advance save failed\n");return 1;}}}
    int slot_bad=0;for(int s=0;s<2;s++)for(int i=0;i<8;i++)slot_bad+=got[s][i]!=expected[s][i];printf("two-slot alternating restore: %s state=%s\n",slot_bad?"mismatch":"exact",state_bad?"mismatch":"exact");if(slot_bad)for(int s=0;s<2;s++){fprintf(stderr,"slot%d expected/got:",s);for(int i=0;i<8;i++)fprintf(stderr," %d/%d",expected[s][i],got[s][i]);fputc('\n',stderr);}for(int s=0;s<2;s++){session_state_free(&slots[s]);for(int i=0;i<8;i++)session_state_free(&checkpoint[s][i]);}

    int history[32],nh=0;for(int i=0;i<o.nprompt;i++)history[nh++]=o.prompt[i];model_reset(&m);prefill_dispatch(&m,history,nh,logits);for(int i=0;i<4;i++){int token=argmax(logits,m.c.vocab);history[nh++]=token;forward_token(&m,token,logits);}SessionState prefix={0};if(!session_state_save(&m,&prefix)||!session_state_restore(&m,&prefix,logits)){fprintf(stderr,"prefix save/restore failed\n");return 1;}int extension[]={17,18};for(int i=0;i<2;i++){history[nh++]=extension[i];forward_token(&m,extension[i],logits);}int reused[8],fresh[8];for(int i=0;i<8;i++){reused[i]=argmax(logits,m.c.vocab);if(i+1<8)forward_token(&m,reused[i],logits);}model_reset(&m);prefill_dispatch(&m,history,nh,logits);for(int i=0;i<8;i++){fresh[i]=argmax(logits,m.c.vocab);if(i+1<8)forward_token(&m,fresh[i],logits);}int prefix_bad=0;for(int i=0;i<8;i++)prefix_bad+=reused[i]!=fresh[i];printf("exact-prefix extension reuse: %s\n",prefix_bad?"mismatch":"exact");session_state_free(&prefix);
    free(logits);return(bad||reuse_bad||disk_bad||corrupt_accepted||slot_bad||state_bad||prefix_bad)?1:0;
}
