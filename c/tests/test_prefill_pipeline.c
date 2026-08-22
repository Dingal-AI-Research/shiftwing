#define QWEN_NO_MAIN
#define QWEN_PREFILL_PIPE_TEST_HOOKS
#include "../qwen.c"

static void invalidate_host_cache(Model*m){
    for(int li=0;li<m->c.n_layers;li++){
        MoeW*w=&m->layer[li].moe;
        for(int slot=0;slot<w->cap;slot++)w->expert[slot].eid=-1;
    }
}

static int cache_ids_unchanged(Model*m,const int*before){
    int at=0;
    for(int li=0;li<m->c.n_layers;li++){
        MoeW*w=&m->layer[li].moe;
        for(int slot=0;slot<w->cap;slot++,at++)
            if(w->expert[slot].eid!=before[at])return 0;
    }
    return 1;
}

static int model_state_exact(Model*a,Model*b){
    Cfg*c=&a->c;if(memcmp(&a->c,&b->c,sizeof(Cfg))||a->pos!=b->pos||
       memcmp(a->last_hidden,b->last_hidden,(size_t)c->hidden*sizeof(float)))
        return 0;
    int cd=2*c->lin_k_heads*c->lin_k_dim+c->lin_v_heads*c->lin_v_dim;
    int kvrows=c->n_kv_heads*c->head_dim;
    for(int li=0;li<c->n_layers;li++)if(a->layer[li].type==LT_LINEAR){
        GdnW*x=&a->layer[li].gdn,*y=&b->layer[li].gdn;
        size_t nc=(size_t)c->conv_kernel*cd;
        size_t ns=(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim;
        if(memcmp(x->conv_state,y->conv_state,nc*sizeof(float))||
           memcmp(x->state,y->state,ns*sizeof(float)))return 0;
    }else{
        AttnW*x=&a->layer[li].attn,*y=&b->layer[li].attn;
        size_t n=(size_t)a->pos*kvrows;
        if(memcmp(x->k_cache,y->k_cache,n*sizeof(float))||
           memcmp(x->v_cache,y->v_cache,n*sizeof(float)))return 0;
    }
    return 1;
}

int main(void){
    setenv("MTP","0",1);setenv("KV16","0",1);setenv("CTX","64",1);
    setenv("EXPERT_RAM","4",1);setenv("PREFETCH_THREADS","0",1);
    setenv("PREFETCH_LOAD","0",1);setenv("PREFILL_COLD_DEVICE","0",1);
    setenv("PREFILL_EXPERT_BATCH","1",1);setenv("PROF_DETAIL","1",1);
    setenv("PIPE","1",1);setenv("URING","0",1);setenv("DIRECT","0",1);
    setenv("COLI_MMAP","0",1);unsetenv("PREFILL_CACHE_BYPASS");
    unsetenv("PREFILL_LOAD_PIPELINE");unsetenv("PREFILL_PIPE_TEST_LOAD_US");
    unsetenv("PREFILL_PIPE_TEST_COMPUTE_US");

    static Model control,synchronous,candidate;
    Oracle o=load_oracle("ref_qwen_i4.json");
    model_init(&control,"qwen_tiny_i4");invalidate_host_cache(&control);
    float*control_logits=falloc(control.c.vocab);
    prefill_dispatch(&control,o.prompt,o.nprompt,control_logits);

    setenv("PREFILL_CACHE_BYPASS","1",1);
    model_init(&synchronous,"qwen_tiny_i4");invalidate_host_cache(&synchronous);
    int sync_cache_n=synchronous.c.n_layers*synchronous.expert_cap;
    int*sync_cache=xcalloc(sync_cache_n,sizeof(int));int sync_at=0;
    for(int li=0;li<synchronous.c.n_layers;li++){
        MoeW*w=&synchronous.layer[li].moe;
        for(int slot=0;slot<w->cap;slot++)sync_cache[sync_at++]=w->expert[slot].eid;
    }
    float*sync_logits=falloc(synchronous.c.vocab);
    prefill_dispatch(&synchronous,o.prompt,o.nprompt,sync_logits);
    int sync_exact=!memcmp(control_logits,sync_logits,
        (size_t)synchronous.c.vocab*sizeof(float))&&
        model_state_exact(&control,&synchronous);
    int sync_topology=cache_ids_unchanged(&synchronous,sync_cache)&&
        !synchronous.prefill_pipe.started&&synchronous.pfpipe_batches==0;

    /* Exercise the real st_read_raw_batch/load_expert_set(n>1) producer path,
     * while the synchronous bypass above remains the batch=1 control. */
    setenv("PREFILL_EXPERT_BATCH","4",1);
    setenv("PREFILL_LOAD_PIPELINE","1",1);
    setenv("PREFILL_PIPE_TEST_LOAD_US","20000",1);
    setenv("PREFILL_PIPE_TEST_COMPUTE_US","20000",1);
    model_init(&candidate,"qwen_tiny_i4");invalidate_host_cache(&candidate);
    int cache_n=candidate.c.n_layers*candidate.expert_cap;
    int*cache_before=xcalloc(cache_n,sizeof(int));int at=0;
    for(int li=0;li<candidate.c.n_layers;li++){
        MoeW*w=&candidate.layer[li].moe;
        for(int slot=0;slot<w->cap;slot++)cache_before[at++]=w->expert[slot].eid;
    }
    float*candidate_logits=falloc(candidate.c.vocab);
    prefill_dispatch(&candidate,o.prompt,o.nprompt,candidate_logits);

    int exact=!memcmp(control_logits,candidate_logits,
                      (size_t)candidate.c.vocab*sizeof(float))&&
              model_state_exact(&control,&candidate);
    int topology=cache_ids_unchanged(&candidate,cache_before);
    PrefillPipe*p=&candidate.prefill_pipe;pthread_mutex_lock(&p->lock);
    int drained=p->busy==0&&p->job_slot<0&&
        p->slot[0].state==PREFILL_PIPE_EMPTY&&
        p->slot[1].state==PREFILL_PIPE_EMPTY;
    pthread_mutex_unlock(&p->lock);
    uint64_t pipeline_batches=candidate.pfpipe_batches;
    uint64_t pipeline_experts=candidate.pfpipe_experts;
    uint64_t pipeline_bytes=candidate.pfpipe_bytes;
    double pipeline_producer=candidate.pfpipe_producer_s;
    double pipeline_compute=candidate.pfpipe_compute_s;
    double pipeline_wall=candidate.pfpipe_wall_s;
    double serial=pipeline_producer+pipeline_compute;
    double overlap=serial-pipeline_wall;
    int instrumented=p->started&&candidate.pfpipe_batches>=
        (uint64_t)2*candidate.c.n_layers&&candidate.pfpipe_experts>=
        candidate.pfpipe_batches&&candidate.pfpipe_bytes>0;
    int materialized_batch=pipeline_experts>pipeline_batches;
    int overlapped=overlap>0.05;

    uint64_t batches_before_decode=candidate.pfpipe_batches;
    for(int step=0;step<4&&exact;step++){
        int a=argmax(control_logits,control.c.vocab);
        int b=argmax(candidate_logits,candidate.c.vocab);
        if(a!=b){exact=0;break;}
        forward_token(&control,a,control_logits);
        forward_token(&candidate,b,candidate_logits);
        if(memcmp(control_logits,candidate_logits,
                  (size_t)candidate.c.vocab*sizeof(float))||
           !model_state_exact(&control,&candidate))exact=0;
    }
    int decode_isolated=candidate.pfpipe_batches==batches_before_decode;
    /* A stop racing a submitted but not-yet-dequeued load must drain and
     * publish it rather than strand the slot in LOADING. */
    invalidate_host_cache(&candidate);int shutdown_eid=0;
    uint64_t shutdown_generation=prefill_pipe_submit(
        &candidate,&candidate.layer[0].moe,0,&shutdown_eid,1);
    prefill_pipe_stop(&candidate);pthread_mutex_lock(&p->lock);
    int shutdown_published=p->job_slot<0&&
        p->slot[0].generation==shutdown_generation&&
        p->slot[0].state==PREFILL_PIPE_READY;
    pthread_mutex_unlock(&p->lock);
    if(shutdown_published){
        prefill_pipe_wait(&candidate,0,shutdown_generation);
        prefill_pipe_release(&candidate,0,shutdown_generation);
    }
    pthread_mutex_lock(&p->lock);
    int shutdown_drained=shutdown_published&&!p->started&&p->job_slot<0&&
        p->slot[0].state==PREFILL_PIPE_EMPTY;
    pthread_mutex_unlock(&p->lock);
    printf("prefill pipeline: sync-exact=%d sync-topology=%d exact=%d topology=%d "
           "drained=%d decode-isolated=%d batch4=%d shutdown-drained=%d "
           "batches=%llu experts=%llu bytes=%llu producer=%.3f compute=%.3f "
           "wall=%.3f overlap=%.3f\n",sync_exact,sync_topology,exact,topology,
           drained,decode_isolated,materialized_batch,shutdown_drained,
           (unsigned long long)pipeline_batches,
           (unsigned long long)pipeline_experts,
           (unsigned long long)pipeline_bytes,
           pipeline_producer,pipeline_compute,pipeline_wall,overlap);
    free(sync_cache);free(cache_before);
    free(control_logits);free(sync_logits);free(candidate_logits);
    free(o.prompt);free(o.full);free(o.tf);
    free(o.mtp_ids);free(o.mtp_pred);free(o.mtp_logits);
    return sync_exact&&sync_topology&&exact&&topology&&drained&&decode_isolated&&
        instrumented&&materialized_batch&&overlapped&&shutdown_drained?0:1;
}
