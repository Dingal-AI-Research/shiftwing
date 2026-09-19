/* Bounded populated prefix probe. Compile-time layer count is explicit in every
 * record; a prefix result cannot qualify the complete 43-layer model. */
#include "../deepseek_v4_execute.h"
static double probe_start,probe_last;
static int probe_progress(void *opaque,int committed,int chunk,int layer,int layers,int activity) {
    (void)opaque;double now=dsv4_server_seconds();
    if (now-probe_last>=2) {
        fprintf(stderr,"PROBE_PROGRESS layers=%d layer=%d activity=%d chunk=%d seconds=%.3f\n",layers,layer,activity,chunk,now-probe_start);
        probe_last=now;
    }
    return now-probe_start<600;
}
typedef struct {const char *path;dsv4_runtime *runtime;} probe_trace_context;
static int probe_trace(void *opaque,const char *stage,int layer,const float *data,int count,int width) {
    probe_trace_context *context=opaque;int position=context->runtime->prefill_chunk_start-context->runtime->position;
    char path[4096];snprintf(path,sizeof(path),"%s.layer%d.%s",context->path,layer,stage);
    FILE *f=fopen(path,position?"r+b":"wb");if (!f || fseek(f,(long)((size_t)position*width*sizeof(float)),SEEK_SET)) {if(f) fclose(f);return 0;}
    int ok=fwrite(data,sizeof(float),(size_t)count*width,f)==(size_t)count*width;
    if (fclose(f)) ok=0;return ok;
}
int main(int argc,char **argv) {
    if (argc!=6) { fprintf(stderr,"usage: probe SNAP TOKENS_BIN COUNT CHUNK OUTPUT_HC\n");return 2; }
    int count=atoi(argv[3]),chunk=atoi(argv[4]);
    if (count<1 || count>DSV4_REVIEW_INPUT_TOKENS || chunk<1 || chunk>2048) return 2;
    int *tokens=malloc(count*sizeof(int));
    FILE *f=fopen(argv[2],"rb");
    if (!tokens || !f || fread(tokens,sizeof(int),count,f)!=(size_t)count || fgetc(f)!=EOF) return 2;
    fclose(f);
    dsv4_store store={0};dsv4_execution execution={0};dsv4_runtime runtime={0};
    if (!dsv4_store_init(&store,argv[1])) { fprintf(stderr,"%s\n",store.error);return 2; }
    double load_start=dsv4_server_seconds();
#ifdef COLI_CUDA
    size_t available,total;
    if (coli_cuda_create(&execution.cuda,0) || coli_cuda_memory_info(execution.cuda,&available,&total)) return 2;
    if (!dsv4_plan_memory(&execution.memory,count,chunk,dsv4_host_available_bytes(),available,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,0)) {
        fprintf(stderr,"%s\n",execution.memory.error);return 2;
    }
    if (!dsv4_store_require_integrity(&store) || !dsv4_dense_arena_init(&execution.dense,&store,0,0) ||
        !dsv4_expert_cache_init(&execution.experts,&store,execution.memory.host_capacity) ||
        !dsv4_dense_arena_enable_cuda(&execution.dense,execution.cuda) ||
        !dsv4_expert_cache_enable_cuda(&execution.experts,execution.cuda,execution.memory.device_cache)) return 2;
#else
    fprintf(stderr,"populated prefix probe requires a CUDA build\n");return 2;
#endif
    execution.load_seconds=dsv4_server_seconds()-load_start;
    if (!dsv4_runtime_init(&runtime,&store,&execution.dense,&execution.experts,count)) return 2;
    probe_trace_context trace_context={argv[5],&runtime};
    if (getenv("PROBE_TRACE") && strcmp(getenv("PROBE_TRACE"),"layers")) {runtime.trace=probe_trace;runtime.trace_context=&trace_context;}
    uint64_t reads=store.raw.read_bytes;probe_start=dsv4_server_seconds();
    int ok=1;
    size_t width=(size_t)DSV4_HC_MULT*DSV4_ATTN_HIDDEN;
    float *bank=malloc((size_t)count*width*sizeof(float));
    if (!bank || !dsv4_expert_prefill_begin(&execution.experts)) return 2;
    for (int b=0;ok && b<count;b++) ok=dsv4_dense_embed_token(&execution.dense,tokens[b],bank+(size_t)b*width);
    for (int layer=0;ok && layer<DSV4_RUNTIME_LAYERS;layer++) {
        ok=dsv4_expert_prefill_layer(&execution.experts,layer);
        if (ok && !getenv("PROBE_LEGACY_CHUNKS")) ok=dsv4_prefill_layer_prompt(&runtime,tokens,count,execution.memory.chunk,layer,bank,probe_progress,NULL);
        for (int pos=0;ok && getenv("PROBE_LEGACY_CHUNKS") && pos<count;pos+=execution.memory.chunk) {
            int take=count-pos;if (take>execution.memory.chunk) take=execution.memory.chunk;
            runtime.prefill_chunk_start=pos;
            ok=dsv4_prefill_layer_chunk(&runtime,tokens+pos,take,layer,bank+(size_t)pos*width,probe_progress,NULL);
        }
        if (ok && !getenv("PROBE_LEGACY_CHUNKS")) {
            fprintf(stderr,"PROBE_PHASE layer=%d attention=%.6f route=%.6f read=%.6f routed=%.6f shared_post=%.6f total=%.6f\n",
                layer,runtime.last_attention_stage_seconds[layer],runtime.last_route_seconds[layer],runtime.last_prefill_read_seconds[layer],
                runtime.last_routed_seconds[layer],runtime.last_shared_seconds[layer],runtime.last_layer_seconds[layer]);
        }
        if (ok && getenv("PROBE_TRACE")) {
            char path[4096];snprintf(path,sizeof(path),"%s.layer%d",argv[5],layer);
            FILE *trace=fopen(path,"wb");
            if (!trace || fwrite(bank,sizeof(float),(size_t)count*width,trace)!=(size_t)count*width || fclose(trace)) ok=0;
        }
    }
    float *logits=getenv("PROBE_LOGITS")?malloc(DSV4_VOCAB*sizeof(float)):NULL;
    if (getenv("PROBE_LOGITS") && (DSV4_RUNTIME_LAYERS!=43 || !logits)) ok=0;
    if (ok) ok=dsv4_prefill_commit(&runtime,tokens,count,bank,logits);
    if (ok && logits) {
        int top=0;
        for (int i=0;i<DSV4_VOCAB;i++) {
            if (!isfinite(logits[i])) ok=0;
            if (logits[i]>logits[top]) top=i;
        }
        char path[4096];snprintf(path,sizeof(path),"%s.logits",argv[5]);
        FILE *out=fopen(path,"wb");
        if (!out) ok=0;
        else {if (fwrite(logits,sizeof(float),DSV4_VOCAB,out)!=DSV4_VOCAB) ok=0;if (fclose(out)) ok=0;}
        fprintf(stderr,"PROBE_LOGITS finite=%d top=%d logit=%.9g\n",ok,top,logits[top]);
    }
    free(logits);
    dsv4_expert_prefill_end(&execution.experts);free(bank);
    double elapsed=dsv4_server_seconds()-probe_start;
    if (ok) {
        f=fopen(argv[5],"wb");
        ok=f && fwrite(runtime.hc,sizeof(float),DSV4_HC_MULT*DSV4_ATTN_HIDDEN,f)==DSV4_HC_MULT*DSV4_ATTN_HIDDEN;
        if (f) ok=ok && !fclose(f);
    }
    struct rusage usage;getrusage(RUSAGE_SELF,&usage);
    fprintf(stdout,"PROBE_RESULT ok=%d layers=%d tokens=%d chunk=%d seconds=%.6f load_seconds=%.6f read_bytes=%llu peak_rss=%llu",
        ok,DSV4_RUNTIME_LAYERS,count,execution.memory.chunk,elapsed,execution.load_seconds,
        (unsigned long long)(store.raw.read_bytes-reads),(unsigned long long)usage.ru_maxrss*1024);
#ifdef COLI_CUDA
    fprintf(stdout," dense_fp8_seconds=%.6f dense_bf16_seconds=%.6f dense_calls=%llu expert_upload_bytes=%llu",execution.dense.cuda_fp8_seconds,execution.dense.cuda_bf16_seconds,(unsigned long long)execution.dense.cuda_calls,(unsigned long long)execution.experts.cuda_upload_bytes);
#endif
    fprintf(stdout," expert_schedule=%s\n",getenv("PROBE_LEGACY_CHUNKS")?"legacy_chunks":"whole_prompt");fflush(stdout);
    if (!ok) fprintf(stderr,"%s %s %s\n",runtime.error,execution.dense.error,execution.experts.error);
    dsv4_execution_memory_report(&execution,"prefix-probe");
    dsv4_runtime_close(&runtime);dsv4_execution_close(&execution);dsv4_store_close(&store);free(tokens);return ok?0:2;
}
