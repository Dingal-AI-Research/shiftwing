#ifndef COLIB_DEEPSEEK_V4_EXECUTE_H
#define COLIB_DEEPSEEK_V4_EXECUTE_H
/* Before this header pulls in any system header: glibc locks its
 * feature-test macros at the first one it sees, and st.h defining
 * _GNU_SOURCE further down the include chain is then too late -- O_DIRECT
 * stays invisible and every expert read silently falls back to buffered.
 * Guarded so a translation unit that already defined it is untouched. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/resource.h>
#include <unistd.h>
#include "deepseek_v4_memory.h"
#include "deepseek_v4_server.h"
#include "deepseek_v4_tokenizer.h"
#define DSV4_TOKENIZER_SHA256 "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"

typedef struct {
    dsv4_store *store;
    dsv4_dense_arena dense;
    dsv4_expert_cache experts;
    dsv4_memory_plan memory;
    Tok tokenizer;
    char manifest_sha256[65], binary_sha256[65];
#ifdef COLI_CUDA
    ColiCuda *cuda;
#endif
    double load_seconds;
    char error[256];
} dsv4_execution;

static inline double dsv4_budget_gb(const char *name,double fallback) {
    const char *s=getenv(name); if (!s || !*s) return fallback;
    char *end; double value=strtod(s,&end);
    return !*end && isfinite(value) && value>0 && value<=1024 ? value : -1;
}
static inline void dsv4_execution_close(dsv4_execution *e) {
    dsv4_expert_cache_close(&e->experts);
    dsv4_dense_arena_close(&e->dense);
#ifdef COLI_CUDA
    if (e->cuda) coli_cuda_destroy(e->cuda);
#endif
}
static inline int dsv4_execution_init(dsv4_execution *e,dsv4_store *store,int context,int chunk) {
    memset(e,0,sizeof(*e)); e->store=store;
    double start=dsv4_server_seconds();
    int use_cuda=getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))!=0;
    uint64_t device_free=0;
    double host_gb=dsv4_budget_gb("RAM_GB",8),device_gb=dsv4_budget_gb("CUDA_EXPERT_GB",2),headroom=dsv4_budget_gb("CUDA_HEADROOM_GB",1.5);
    if (host_gb<0 || device_gb<0 || headroom<1.5) {
        snprintf(e->error,sizeof(e->error),"invalid memory budgets (GPU headroom must be >=1.5 GiB)");goto fail;
    }
#ifdef COLI_CUDA
    if (use_cuda) {
        size_t available,total;
        if (coli_cuda_create(&e->cuda,0) || coli_cuda_memory_info(e->cuda,&available,&total)) {
            snprintf(e->error,sizeof(e->error),"CUDA memory query: %s",coli_cuda_last_error());goto fail;
        }
        device_free=available;
    }
#else
    if (use_cuda) { snprintf(e->error,sizeof(e->error),"binary has no CUDA support");goto fail; }
#endif
    /* A launcher that has just unloaded another model starts this engine
     * before the kernel and driver have returned that model's memory, so a
     * single MemAvailable sample can be several GiB short. Re-plan every
     * second for a bounded time rather than fail on the first reading. */
    uint64_t released=dsv4_dense_fp8_page_bytes(store,0);
    const char *wait_text=getenv("DSV4_MEMORY_WAIT_S");
    int wait_limit=wait_text && *wait_text ? atoi(wait_text) : 90;
    uint64_t layer_bytes=(uint64_t)(DSV4_BASE_LAYERS+DSV4_DSPARK_LAYERS)*dsv4_expert_payload_bytes();
    uint64_t wanted_host=(uint64_t)(host_gb*DSV4_GIB)/layer_bytes*layer_bytes;
    if (wanted_host>DSV4_EXPERTS*layer_bytes) wanted_host=DSV4_EXPERTS*layer_bytes;
    uint64_t wanted_device=use_cuda ? (uint64_t)(device_gb*DSV4_GIB)/dsv4_expert_payload_bytes()*dsv4_expert_payload_bytes() : 0;
    uint64_t last_available=0; int stable=0, announced=0;
    for (int waited=0;;waited++) {
        uint64_t host_available=dsv4_host_available_bytes();
        int fits=dsv4_plan_memory(&e->memory,context,chunk,host_available,device_free,use_cuda,
                (uint64_t)(host_gb*DSV4_GIB),(uint64_t)(device_gb*DSV4_GIB),(uint64_t)(headroom*DSV4_GIB),released);
        /* Memory returns over several seconds after another model unloads.
         * Accept the plan once it grants the full cache budgets, or once
         * availability has stopped rising; never accept the first fitting
         * sample, which would pin a small expert cache for the process life. */
        if (fits && e->memory.host_cache>=wanted_host && e->memory.device_cache>=wanted_device) break;
        if (fits && host_available<=last_available+last_available/100) { if (++stable>=3) break; } else stable=0;
        if (waited>=wait_limit) {
            if (fits) break;
            snprintf(e->error,sizeof(e->error),"%s",e->memory.error);goto fail;
        }
        if (!fits && !strncmp(e->memory.error,"invalid",7)) { snprintf(e->error,sizeof(e->error),"%s",e->memory.error);goto fail; }
        if (!announced++) fprintf(stderr,"DSV4_MEMORY_WAIT host_available=%llu device_available=%llu fits=%d limit_s=%d\n",
            (unsigned long long)host_available,(unsigned long long)device_free,fits,wait_limit);
        last_available=host_available;
        sleep(1);
#ifdef COLI_CUDA
        if (use_cuda) { size_t available,total; if (!coli_cuda_memory_info(e->cuda,&available,&total)) device_free=available; }
#endif
    }
    if (announced) fprintf(stderr,"DSV4_MEMORY_WAIT done waited_s=%d\n",announced);
    dsv4_memory_plan *p=&e->memory;
    fprintf(stderr,"DSV4_MEMORY context=%d chunk=%d host_required=%llu host_available=%llu device_required=%llu device_available=%llu dense=%llu dense_released=%llu state=%llu host_cache=%llu device_cache=%llu host_scratch=%llu device_scratch=%llu staging=%llu\n",
        context,p->chunk,(unsigned long long)p->host_required,(unsigned long long)p->host_available,
        (unsigned long long)p->device_required,(unsigned long long)p->device_available,(unsigned long long)p->dense_bytes,
        (unsigned long long)p->dense_released,
        (unsigned long long)p->state_bytes,(unsigned long long)p->host_cache,(unsigned long long)p->device_cache,
        (unsigned long long)p->host_scratch,(unsigned long long)p->device_scratch,(unsigned long long)p->staging_bytes);
    if (!dsv4_store_require_integrity(store)) { snprintf(e->error,sizeof(e->error),"%.255s",store->error);goto fail; }
    if (!dsv4_dense_arena_init(&e->dense,store,0,1)) { snprintf(e->error,sizeof(e->error),"%s",e->dense.error);goto fail; }
    if (!dsv4_expert_cache_init(&e->experts,store,p->host_capacity)) { snprintf(e->error,sizeof(e->error),"%s",e->experts.error);goto fail; }
#ifdef COLI_CUDA
    if (use_cuda && (!dsv4_dense_arena_enable_cuda(&e->dense,e->cuda) ||
        !dsv4_expert_cache_enable_cuda(&e->experts,e->cuda,(size_t)p->device_cache))) {
        snprintf(e->error,sizeof(e->error),"CUDA native cache initialization failed: %.100s %.100s",e->dense.error,e->experts.error);goto fail;
    }
#endif
    e->load_seconds=dsv4_server_seconds()-start;
    return 1;
fail:
    dsv4_execution_close(e); return 0;
}

/* Capacities are actual allocated bytes and only grow within one process.
 * Pool high-water marks include stream-ordered allocation and fragmentation. */
static inline void dsv4_execution_memory_report(dsv4_execution *e,const char *id) {
#ifdef COLI_CUDA
    if (e->cuda) {
        unsigned long long used,reserved,backend;size_t available,total;
        if (coli_cuda_dsv4_memory_stats(e->cuda,&used,&reserved,&backend,&available,&total)) return;
        dsv4_dense_arena *d=&e->dense;dsv4_expert_cache *c=&e->experts;
        uint64_t scratch=backend+d->cuda_activation_cap+d->cuda_activation_scale_cap+d->cuda_output_cap+
            d->cuda_attention_cache_cap+d->cuda_index_cache_cap+d->cuda_attention_query_cap+d->cuda_attention_out_cap+
            d->cuda_attention_indices_cap+d->cuda_attention_aux_cap+c->cuda_activation_cap+c->cuda_activation_scale_cap+c->cuda_output_cap;
        fprintf(stderr,"DSV4_ALLOCATIONS id=%s pool_used_peak=%llu pool_reserved_peak=%llu scratch_allocated=%llu device_free=%zu device_total=%zu dense_allocated=%zu pinned_expert_staging=%zu pinned_read_staging=%zu\n",
            id,used,reserved,(unsigned long long)scratch,available,total,d->arena_bytes,c->cuda_upload_staging_cap,c->read_staging_pinned?c->read_staging_bytes:0);
    }
#endif
}

typedef struct { double probability; int id; } dsv4_candidate;
static int dsv4_candidate_compare(const void *a,const void *b) {
    const dsv4_candidate *x=a,*y=b;
    if (x->probability!=y->probability) return x->probability>y->probability ? -1 : 1;
    return x->id-y->id;
}
static inline uint64_t dsv4_random(uint64_t *state) {
    uint64_t x=*state;x^=x>>12;x^=x<<25;x^=x>>27;*state=x;return x*UINT64_C(2685821657736338717);
}
static inline int dsv4_sample(const float *logits,int count,double temperature,double top_p,uint64_t *rng,dsv4_candidate *candidates) {
    int best=-1;
    for (int i=0;i<count;i++) {
        if (!isfinite(logits[i])) return -1;
        if (best<0 || logits[i]>logits[best]) best=i;
    }
    if (best<0 || temperature==0) return best;
    double total=0;
    for (int i=0;i<count;i++) {
        candidates[i]=(dsv4_candidate){exp(((double)logits[i]-logits[best])/temperature),i};
        total+=candidates[i].probability;
    }
    qsort(candidates,(size_t)count,sizeof(*candidates),dsv4_candidate_compare);
    double mass=0;int kept=0;
    do {mass+=candidates[kept++].probability;} while (kept<count && mass<top_p*total);
    double target=(double)(dsv4_random(rng)>>11)*0x1.0p-53*mass;
    for (int i=0;i<kept;i++) { target-=candidates[i].probability;if (target<0) return candidates[i].id; }
    return candidates[kept-1].id;
}

typedef struct { dsv4_server *server; const char *id; int total; double start,last; int decode; dsv4_runtime *runtime; } dsv4_activity;
static int dsv4_execution_progress(void *opaque,int committed,int chunk,int layer,int layers,int activity) {
    dsv4_activity *p=opaque;
    if (dsv4_cancelled(p->server)) return 0;
    double now=dsv4_server_seconds();
    if (!p->decode && now-p->last>=0.25) {
        dsv4_frame(p->server,"PREFILL_PROGRESS %s %d %d %lld\n",p->id,committed,p->total,(long long)((now-p->start)*1000));
        dsv4_frame(p->server,"PREFILL_ACTIVITY %s %d %d %d %d %d %d\n",p->id,committed,chunk,layer,layers,activity,p->runtime->prefill_chunk_start);
        p->last=now;
    }
    return 1;
}
static dsv4_result dsv4_execute_request(dsv4_server *server,const dsv4_request *request,void *opaque) {
    dsv4_execution *e=opaque;
    dsv4_result result={0}; dsv4_runtime runtime={0};
    int *tokens=NULL;float *logits=NULL;dsv4_candidate *candidates=NULL;
    uint64_t hits=e->experts.hits,misses=e->experts.misses;
    uint64_t reads=__atomic_load_n(&e->store->raw.read_bytes,__ATOMIC_RELAXED);
    double start=dsv4_server_seconds(),decode_start=0,prefill_seconds=0;
    double decode_read_s=0,decode_hash_s=0,decode_copy_s=0,decode_pin_s=0,decode_upload_s=0,decode_kernel_s=0;
    uint64_t decode_read_experts=0,decode_hits=0,decode_misses=0,decode_reads=0,decode_batches=0,
        decode_cuda_hits=0,decode_cuda_misses=0;
    (void)decode_cuda_hits;(void)decode_cuda_misses;
    char payload_sha[65];dsv4_sha256_hex(request->payload,request->bytes,payload_sha);
    result.error="FORWARD_FAILED";
    if (!request->bytes) { result.error="EMPTY_PROMPT";goto done; }
    tokens=malloc(request->bytes*sizeof(*tokens));
    if (!tokens) { result.error="OUT_OF_MEMORY";goto done; }
    int count=dsv4_tok_encode(&e->tokenizer,request->payload,(int)request->bytes,tokens,(int)request->bytes);
    result.prompt_tokens=count;
    if (count<1) { result.error="EMPTY_PROMPT";goto done; }
    if (count>DSV4_REVIEW_INPUT_TOKENS || request->maximum>e->memory.context-count) {
        result.error="CONTEXT_EXCEEDED";
        snprintf(result.detail,sizeof(result.detail),"%d %d",count,count>DSV4_REVIEW_INPUT_TOKENS ? DSV4_REVIEW_INPUT_TOKENS : e->memory.context-request->maximum);goto done;
    }
    if (dsv4_cancelled(server)) goto done;
    logits=malloc((size_t)DSV4_VOCAB*sizeof(*logits));
    candidates=malloc((size_t)DSV4_VOCAB*sizeof(*candidates));
    if (!logits || !candidates || !dsv4_runtime_init(&runtime,e->store,&e->dense,&e->experts,e->memory.context)) {
        result.error="OUT_OF_MEMORY";goto done;
    }
    dsv4_activity progress={server,request->id,count,dsv4_server_seconds(),0,0,&runtime};
    dsv4_frame(server,"PREFILL_BEGIN %s %d 0\n",request->id,count);
    if (!dsv4_runtime_prefill_prompt(&runtime,tokens,count,e->memory.chunk,logits,dsv4_execution_progress,&progress)) goto done;
    dsv4_frame(server,"PREFILL_PROGRESS %s %d %d %lld\n",request->id,count,count,(long long)((dsv4_server_seconds()-progress.start)*1000));
    prefill_seconds=dsv4_server_seconds()-progress.start;
    dsv4_frame(server,"PREFILL_END %s %d %lld\n",request->id,count,(long long)(prefill_seconds*1000));
    decode_start=dsv4_server_seconds();progress.decode=1;
    /* Decode-phase attribution starts here; prefill traffic is excluded. */
    decode_read_s=e->experts.read_seconds;decode_hash_s=e->experts.hash_seconds;decode_copy_s=e->experts.copy_seconds;
    decode_pin_s=e->experts.pin_seconds;decode_upload_s=e->experts.upload_seconds;decode_kernel_s=e->experts.kernel_seconds;
    decode_read_experts=e->experts.read_experts;decode_hits=e->experts.hits;decode_misses=e->experts.misses;
    decode_reads=__atomic_load_n(&e->store->raw.read_bytes,__ATOMIC_RELAXED);
    decode_batches=__atomic_load_n(&e->store->raw.uring_batches,__ATOMIC_RELAXED);
#ifdef COLI_CUDA
    decode_cuda_hits=e->experts.cuda_hits;decode_cuda_misses=e->experts.cuda_misses;
#endif
    uint64_t rng=UINT64_C(0x6d25314a295e73b1);
    for (size_t i=0;i<request->bytes;i++) rng=(rng^(unsigned char)request->payload[i])*UINT64_C(1099511628211);
    if (!rng) rng=1;
    result.length_limited=1;
    /* DSV4_DECODE_TRACE=1: one stderr line per step with the chosen token, the
     * top two logits and a logit checksum, so two runs can be compared step by
     * step and the first divergence located together with its margin. */
    int trace=getenv("DSV4_DECODE_TRACE") && atoi(getenv("DSV4_DECODE_TRACE"))!=0;
    for (int step=0;step<request->maximum;step++) {
        if (dsv4_cancelled(server)) goto done;
        int token=dsv4_sample(logits,DSV4_VOCAB,request->temperature,request->top_p,&rng,candidates);
        if (token<0) { result.error="NONFINITE_LOGITS";goto done; }
        if (trace) {
            int best=0,second=-1;double sum=0;
            for (int i=0;i<DSV4_VOCAB;i++) {
                sum+=logits[i];
                if (logits[i]>logits[best]) { second=best;best=i; }
                else if (second<0 || logits[i]>logits[second]) second=i;
            }
            fprintf(stderr,"DSV4_DECODE_TRACE step=%d token=%d top=%d logit=%.9g second=%d margin=%.9g sum=%.9g\n",
                step,token,best,logits[best],second,logits[best]-logits[second],sum);
        }
        result.completion_tokens++;
        if (token==1) { result.length_limited=0;break; }
        if (token>=e->tokenizer.n_ids || !e->tokenizer.id2str[token]) { result.error="INVALID_TOKEN";goto done; }
        char text[65536];
        if (strlen(e->tokenizer.id2str[token])>=sizeof(text)) { result.error="INVALID_TOKEN";goto done; }
        int bytes=tok_decode(&e->tokenizer,&token,1,text,sizeof(text));
        dsv4_data(server,request->id,text,(size_t)bytes);
        dsv4_frame(server,"DECODE_PROGRESS %s %d %d %lld\n",request->id,step+1,request->maximum,(long long)((dsv4_server_seconds()-decode_start)*1000));
        if (step+1<request->maximum && !dsv4_runtime_prefill_chunk(&runtime,&token,1,logits,dsv4_execution_progress,&progress)) goto done;
    }
    result.error=NULL;
done:
    if (result.error && !result.detail[0] && !dsv4_cancelled(server))
        snprintf(result.detail,sizeof(result.detail),"%.80s %.80s %.80s",runtime.error,e->dense.error,e->experts.error);
    if (decode_start && runtime.decode_steps) {
        dsv4_expert_cache *cache=&e->experts;
        uint64_t cuda_hits=0,cuda_misses=0;
#ifdef COLI_CUDA
        cuda_hits=cache->cuda_hits-decode_cuda_hits;cuda_misses=cache->cuda_misses-decode_cuda_misses;
#endif
        /* Per-step seconds by phase. read/hash/copy/pin are the storage miss path,
         * upload is host staging plus H2D enqueue, kernel is expert launches
         * through their sync (which also absorbs queued upload DMA). */
        fprintf(stderr,"DSV4_DECODE_PROFILE id=%s grouped=%d steps=%llu step_s=%.6f attention_s=%.6f route_s=%.6f "
            "routed_s=%.6f shared_s=%.6f head_s=%.6f read_s=%.6f hash_s=%.6f copy_s=%.6f pin_s=%.6f upload_s=%.6f kernel_s=%.6f "
            "read_experts=%.3f read_bytes=%.0f uring_batches=%.3f host_hits=%.3f host_misses=%.3f cuda_hits=%.3f cuda_misses=%.3f pinned_cache=%d direct_uploads=%llu pin_failures=%llu\n",
            request->id,runtime.decode_grouped,(unsigned long long)runtime.decode_steps,
            runtime.decode_step_seconds/runtime.decode_steps,runtime.decode_attention_seconds/runtime.decode_steps,
            runtime.decode_route_seconds/runtime.decode_steps,runtime.decode_routed_seconds/runtime.decode_steps,
            runtime.decode_shared_seconds/runtime.decode_steps,runtime.decode_head_seconds/runtime.decode_steps,
            (cache->read_seconds-decode_read_s)/runtime.decode_steps,(cache->hash_seconds-decode_hash_s)/runtime.decode_steps,
            (cache->copy_seconds-decode_copy_s)/runtime.decode_steps,(cache->pin_seconds-decode_pin_s)/runtime.decode_steps,
            (cache->upload_seconds-decode_upload_s)/runtime.decode_steps,
            (cache->kernel_seconds-decode_kernel_s)/runtime.decode_steps,
            (double)(cache->read_experts-decode_read_experts)/runtime.decode_steps,
            (double)(__atomic_load_n(&e->store->raw.read_bytes,__ATOMIC_RELAXED)-decode_reads)/runtime.decode_steps,
            (double)(__atomic_load_n(&e->store->raw.uring_batches,__ATOMIC_RELAXED)-decode_batches)/runtime.decode_steps,
            (double)(cache->hits-decode_hits)/runtime.decode_steps,(double)(cache->misses-decode_misses)/runtime.decode_steps,
            (double)cuda_hits/runtime.decode_steps,(double)cuda_misses/runtime.decode_steps,
#ifdef COLI_CUDA
            cache->pinned_cache,(unsigned long long)cache->cuda_direct_uploads,(unsigned long long)cache->pin_failures
#else
            0,0ull,0ull
#endif
            );
    }
    dsv4_runtime_close(&runtime);free(tokens);free(logits);free(candidates);
    double now=dsv4_server_seconds();
    if (decode_start && now>decode_start) result.tokens_per_second=result.completion_tokens/(now-decode_start);
    uint64_t h=e->experts.hits-hits,m=e->experts.misses-misses;
    result.cache_hit_percent=h+m ? 100.0*h/(h+m) : 0;
    struct rusage usage;if (!getrusage(RUSAGE_SELF,&usage)) result.rss_gb=(double)usage.ru_maxrss*1024/DSV4_GIB;
    fprintf(stderr,"DSV4_REQUEST id=%s payload_sha256=%s manifest_sha256=%s binary_sha256=%s tokenizer_sha256=%s prompt=%d generated=%d prefill_s=%.6f decode_s=%.6f total_s=%.6f peak_rss_gb=%.6f read_bytes=%llu termination=%s\n",
        request->id,payload_sha,e->manifest_sha256,e->binary_sha256,DSV4_TOKENIZER_SHA256,result.prompt_tokens,result.completion_tokens,
        prefill_seconds,decode_start ? now-decode_start : 0,now-start,result.rss_gb,
        (unsigned long long)(__atomic_load_n(&e->store->raw.read_bytes,__ATOMIC_RELAXED)-reads),
        dsv4_cancelled(server) ? "cancelled" : result.error ? result.error : result.length_limited ? "length" : "eos");
    dsv4_execution_memory_report(e,request->id);
    return result;
}
#endif
