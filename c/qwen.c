/* colib — from-scratch C inference engine for Qwen3.5 MoE + Ornith-1.0.
 *
 * Phase 5 hybrid engine: the complete Phase-4 CPU path plus opt-in CUDA
 * quantized projections and grouped experts. Formulas mirror
 * docs/qwen35_arch.md.
 *
 * Generic infrastructure follows colibri (github.com/JustVugg/colibri,
 * Apache-2.0); see NOTICE.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <poll.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "compat.h"
#include "st.h"
#include "json.h"
#include "tok.h"
#include "tier.h"
#include "serve_mux.h"
#include "serve_scheduler.h"
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif

#define QW_MAX_LAYERS 128
#define QW_MAX_TOPK 64
#define QW_NAME 512

enum { LT_LINEAR = 0, LT_FULL = 1 };

typedef struct {
    int hidden, n_layers, vocab, max_position, eos_token, eos_token2, pad_token;
    int n_heads, n_kv_heads, head_dim;
    float partial_rotary, theta, eps;
    int n_experts, topk, moe_inter, shared_inter;
    int norm_topk;
    int lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel;
    int mtp_layers, full_interval;
    signed char layer_type[QW_MAX_LAYERS];
} Cfg;

typedef struct {
    /* fmt 2 expands losslessly to q4 on CUDA; fmt 3 remains packed end to end. */
    int fmt, O, I, gs, rb, drb, ng; /* 0=f32, 1=int8, 2=int2, 3=int3, 4=int4 */
    float *f, *s;
    int8_t *q8;
    uint8_t *q4;
    void *map_q,*map_s;size_t map_q_len,map_s_len;
#ifdef COLI_CUDA
    void *d_q,*d_s;
    void *cuda_cache;
    char cuda_key[QW_NAME];
    int cuda_eligible,cuda_ready;
#endif
} QMat;
typedef struct { QMat gate, up, down; int eid; } Expert;
typedef struct {
    QMat router;
    Expert *expert;
    QMat shared_gate, shared_up, shared_down, shared_scale;
    uint32_t *heat,*last,*decode_heat,clock;uint16_t*transition;
    unsigned char*prefetched,*decode_pinned;
    int prev_route[QW_MAX_TOPK],prev_n,pf_pending;int cap,layer;
    pthread_mutex_t lock;pthread_cond_t pf_done;
#ifdef COLI_CUDA
    float *d_router_weight,*d_scale_weight;
#endif
} MoeW;
typedef struct {
    QMat qkv, z, b, a, out;
    float *conv, *A_log, *dt_bias, *norm;
    float *conv_state, *state;
#ifdef COLI_CUDA
    float *d_conv,*d_A_log,*d_dt_bias,*d_norm,*d_conv_state,*d_state,*d_conv_backup,*d_state_backup;
    float *d_a_weight,*d_b_weight;
    int cuda_aux_ready,cuda_state_pos;
#endif
} GdnW;
typedef struct {
    QMat q, k, v, o;
    float *q_norm, *k_norm;
    float *k_cache, *v_cache;
    uint16_t *k_cache16, *v_cache16;
#ifdef COLI_CUDA
    float *d_q_norm,*d_k_norm,*d_k_cache,*d_v_cache;
    int cuda_aux_ready,cuda_state_pos;
#endif
} AttnW;
typedef struct {
    int type,index;
    float *input_norm, *post_norm;
#ifdef COLI_CUDA
    float *d_input_norm,*d_post_norm;
#endif
    GdnW gdn;
    AttnW attn;
    MoeW moe;
} Layer;
typedef struct {
    int enabled, pos;
    Layer layer;
    QMat fc;
    float *pre_embed_norm, *pre_hidden_norm, *norm;
    uint64_t proposed, accepted, target_forwards, emitted;
    uint64_t draft_misses,verify_misses,replay_misses;
    uint64_t confidence_skips;
    uint64_t conf_accepted[5],conf_rejected[5],conf_unverified[5];
    double conf_accepted_sum,conf_rejected_sum,conf_unverified_sum;
    double verify_detail[4],replay_detail[4],fallback_detail[4];
    double draft_s,verify_s,replay_s,fallback_s;
} MtpW;
typedef struct {
    Cfg c;
    shards S;
    Tok T; int has_tok;
    QMat embed, lm_head;
    float *final_norm;
    Layer *layer;
    MtpW mtp;
    float *last_hidden;
    int pos, max_seq, quant_mode, kv16, expert_cap; /* 0=floating, 8=int8, 4=grouped int4 */
    int matrix_f32, matrix_i8, matrix_i2, matrix_i3, matrix_i4;
    int dump_acts, debug_logits,route_record_suppress,decode_phase;
    int decode_prewarmer;
    int prof_detail;
    double prof_gdn,prof_attn,prof_moe,prof_lm,prof_expert_load;uint64_t prof_expert_misses;
    double dense_load_s;
    pthread_t pf_thread[4];pthread_mutex_t pf_lock;pthread_cond_t pf_cond;
    struct{int layer,n,load,eid[QW_MAX_TOPK];}pf_job[128];
    int pf_head,pf_tail,pf_nthread;
    uint64_t pf_predictions,pf_loads,pf_useful,pf_wasted,pf_gpu_skips,pf_dropped;
    uint32_t *emap_seed;char emap_path[2048];int emap_loaded,emap_saved;
    uint64_t tier_hits,tier_misses,tier_gpu_hits;
    uint64_t decode_pin_refreshes,decode_pin_fallbacks;
    uint64_t decode_prewarm_loads,decode_prewarm_bytes;
    int q3_route_atlas;
    uint64_t q3_atlas_host_entries,q3_atlas_uncovered;
    unsigned char *turn_hits;
#ifdef COLI_CUDA
    float *d_final_norm;
#endif
} Model;

typedef struct {
    int pos,hidden,layers,kv16;
    float *recurrent,*kv,*last_hidden;
    uint16_t *kv_bf16;
    size_t recurrent_n,kv_n,recurrent_cap,kv_cap,hidden_cap;
} SessionState;

typedef struct {
    float *conv,*gdn,*k,*v;
    uint16_t *k16,*v16;
#ifdef COLI_CUDA
    float *d_conv,*d_gdn;
    float *d_k,*d_v;
    int cuda_gdn,cuda_attn;
#endif
} ResidentLayerState;

typedef struct {
    int nslots,max_seq,kv16;
    int *pos;
    float *last_hidden,*logits;
    ResidentLayerState *layer;
#ifdef COLI_CUDA
    float *d_x,*d_n,*d_mix,*d_moe,*d_a,*d_b,*d_router,*d_scale,*d_logits;
    unsigned short*d_n16;
    int cuda_activations;
#endif
} ResidentBatchState;

typedef struct {
    char magic[8];
    uint32_t version,hidden,layers,kv16;
    uint32_t pos,reserved;
    uint64_t recurrent_n,kv_n,checksum;
} SessionDiskHeader;

typedef struct {
    int *prompt, nprompt, *full, nfull, *tf, ntf;
    int *mtp_ids, nmtp, *mtp_pred, nmtp_pred;
    float *mtp_logits; int mtp_rows, mtp_cols;
} Oracle;

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void die(const char *msg){ fprintf(stderr,"%s\n",msg); exit(1); }
static void *xcalloc(size_t n,size_t z){ void *p=calloc(n,z); if(!p){ fprintf(stderr,"OOM %zu bytes\n",n*z); exit(1); } return p; }
static float *falloc(int64_t n){ return xcalloc((size_t)n,sizeof(float)); }
static float sigmoidf_stable(float x){ if(x>=0){ float z=expf(-x); return 1.f/(1.f+z); } float z=expf(x); return z/(1.f+z); }
static float siluf(float x){ return x*sigmoidf_stable(x); }
static float softplusf_stable(float x){ return x>20.f?x:(x<-20.f?expf(x):log1pf(expf(x))); }
static uint16_t f32_to_bf16(float x){uint32_t u;memcpy(&u,&x,4);u+=0x7fffu+((u>>16)&1u);return(uint16_t)(u>>16);}

typedef struct {
    char magic[8];
    uint32_t version,layers,experts;
} ExpertMapHeader;
static Model *tier_atexit_model;

static void expert_map_configure(Model*m,const char*snap){
    const char*p=getenv("EMAP_PATH");
    if(p&&*p)snprintf(m->emap_path,sizeof(m->emap_path),"%s",p);
    else if(st_env_enabled("AUTOPIN"))
        snprintf(m->emap_path,sizeof(m->emap_path),"%s/expert_map.bin",snap);
    if(!m->emap_path[0])return;
    size_t n=(size_t)m->c.n_layers*m->c.n_experts;
    m->emap_seed=xcalloc(n,sizeof(uint32_t));
    FILE*f=fopen(m->emap_path,"rb");if(!f)return;
    ExpertMapHeader h;
    if(fread(&h,1,sizeof(h),f)==sizeof(h)&&!memcmp(h.magic,"COLIEMAP",8)&&
       h.version==1&&h.layers==(uint32_t)m->c.n_layers&&
       h.experts==(uint32_t)m->c.n_experts&&
       fread(m->emap_seed,sizeof(uint32_t),n,f)==n)m->emap_loaded=1;
    else memset(m->emap_seed,0,n*sizeof(uint32_t));
    fclose(f);
}
static void expert_map_save(Model*m){
    if(!m||!m->emap_path[0]||!m->layer)return;
    size_t n=(size_t)m->c.n_layers*m->c.n_experts;
    uint32_t*heat=xcalloc(n,sizeof(uint32_t));
    for(int l=0;l<m->c.n_layers;l++)
        memcpy(heat+(size_t)l*m->c.n_experts,m->layer[l].moe.heat,
               (size_t)m->c.n_experts*sizeof(uint32_t));
    char tmp[2304];snprintf(tmp,sizeof(tmp),"%s.tmp.%ld",m->emap_path,(long)getpid());
    FILE*f=fopen(tmp,"wb");if(f){
        ExpertMapHeader h={{'C','O','L','I','E','M','A','P'},1,
                           (uint32_t)m->c.n_layers,(uint32_t)m->c.n_experts};
        int ok=fwrite(&h,1,sizeof(h),f)==sizeof(h)&&
               fwrite(heat,sizeof(uint32_t),n,f)==n&&fflush(f)==0&&fsync(fileno(f))==0;
        if(fclose(f)!=0)ok=0;
        if(ok&&rename(tmp,m->emap_path)==0)m->emap_saved=1;else unlink(tmp);
    }
    free(heat);
}
static void tier_report_and_save(void){
    Model*m=tier_atexit_model;if(!m)return;expert_map_save(m);
    uint64_t h=m->tier_hits,g=m->tier_gpu_hits,miss=m->tier_misses,total=h+g+miss;
    fprintf(stderr,"[TIERS] cap/layer=%d cpu-hits=%llu gpu-hits=%llu misses=%llu hit-rate=%.2f%% read=%.3fGiB direct=%.3fGiB direct-fallbacks=%llu uring-batches=%llu uring-reads=%llu uring-setups=%llu uring-reuses=%llu uring-fallbacks=%llu\n",
            m->expert_cap,(unsigned long long)h,(unsigned long long)g,
            (unsigned long long)miss,total?100.0*(double)(h+g)/(double)total:0.,
            (double)m->S.read_bytes/(1024.*1024.*1024.),
            (double)m->S.direct_bytes/(1024.*1024.*1024.),
            (unsigned long long)m->S.direct_fallbacks,
            (unsigned long long)m->S.uring_batches,
            (unsigned long long)m->S.uring_reads,
            (unsigned long long)m->S.uring_setups,
            (unsigned long long)m->S.uring_reuses,
            (unsigned long long)m->S.uring_fallbacks);
    fprintf(stderr,"[PREFETCH] predictions=%llu loads=%llu useful=%llu wasted=%llu gpu-skips=%llu dropped=%llu precision=%.2f%%\n",
            (unsigned long long)m->pf_predictions,
            (unsigned long long)m->pf_loads,
            (unsigned long long)m->pf_useful,
            (unsigned long long)m->pf_wasted,
            (unsigned long long)m->pf_gpu_skips,
            (unsigned long long)m->pf_dropped,
            m->pf_loads?100.0*(double)m->pf_useful/(double)m->pf_loads:0.);
    if(st_env_enabled("DECODE_PROTECT"))
        fprintf(stderr,"[DECODE_PROTECT] refreshes=%llu pin-fallbacks=%llu prewarm-loads=%llu prewarm=%.3fGiB\n",
                (unsigned long long)m->decode_pin_refreshes,
                (unsigned long long)m->decode_pin_fallbacks,
                (unsigned long long)m->decode_prewarm_loads,
                (double)m->decode_prewarm_bytes/(1024.*1024.*1024.));
    if(m->emap_path[0])fprintf(stderr,"[EMAP] path=%s loaded=%d saved=%d routes=%llu\n",
                              m->emap_path,m->emap_loaded,m->emap_saved,
                              (unsigned long long)total);
}
static void tier_counters_reset(Model*m){
    m->tier_hits=m->tier_misses=m->tier_gpu_hits=0;
    m->pf_predictions=m->pf_loads=m->pf_useful=m->pf_wasted=0;
    m->pf_gpu_skips=m->pf_dropped=0;
    m->S.read_bytes=m->S.direct_bytes=m->S.direct_fallbacks=0;
    m->S.uring_batches=m->S.uring_reads=m->S.uring_fallbacks=0;
    m->S.uring_setups=m->S.uring_reuses=0;
}

/* A packed int3 triplet contains eight consecutive unsigned codes.  Two
 * 12-bit lookups replace the old per-value shift loop for both CPU dot
 * products and the q4 staging representation used by the CUDA kernels. */
static pthread_once_t q3_lut_once=PTHREAD_ONCE_INIT;
static uint32_t q3_i8_lut[4096];
static uint16_t q3_q4_lut[4096];
static void q3_lut_init(void){
    for(unsigned word=0;word<4096;word++){
        uint32_t i8=0;uint16_t q4=0;
        for(int i=0;i<4;i++){
            unsigned code=(word>>(3*i))&7u;
            i8|=(uint32_t)(uint8_t)((int)code-4)<<(8*i);
            q4|=(uint16_t)(code+4u)<<(4*i);
        }
        q3_i8_lut[word]=i8;q3_q4_lut[word]=q4;
    }
}
static inline uint32_t q3_word24(const uint8_t*src){
    return (uint32_t)src[0]|((uint32_t)src[1]<<8)|
           ((uint32_t)src[2]<<16);
}
static inline uint64_t q3_unpack8_i8(uint32_t word){
    return (uint64_t)q3_i8_lut[word&4095u]|
           ((uint64_t)q3_i8_lut[word>>12]<<32);
}
static void q3_expand_q4(const QMat*w,uint8_t*out){
    pthread_once(&q3_lut_once,q3_lut_init);
    int triplets=w->rb/3;
    #pragma omp parallel for schedule(static) if((int64_t)w->O*w->rb>=1048576)
    for(int row=0;row<w->O;row++){
        const uint8_t*src=w->q4+(size_t)row*w->rb;
        uint8_t*dst=out+(size_t)row*w->drb;
        for(int triplet=0;triplet<triplets;triplet++){
            uint32_t word=q3_word24(src+3*triplet);
            uint32_t packed=(uint32_t)q3_q4_lut[word&4095u]|
                            ((uint32_t)q3_q4_lut[word>>12]<<16);
            memcpy(dst+4*triplet,&packed,sizeof(packed));
        }
    }
}

#ifdef COLI_CUDA
typedef struct CudaWeightCache {
    char *key;void*d_q,*d_s;size_t bytes;uint64_t stamp;
    int expert,pinned,mtp,atlas;
    int fmt,O,I,gs,rb,drb,ng;
    QMat *owner;struct CudaWeightCache *hnext,*prev,*next;
} CudaWeightCache;
#define CUDA_WEIGHT_BUCKETS 16384
typedef struct {
    ColiCuda *ctx;
    float *x,*y,*gate,*up,*tmp,*proj[4];
    unsigned short *x16,*gate16;
    size_t xcap,ycap,gatecap,upcap,tmpcap,projcap[4],x16cap,gate16cap;
    pthread_mutex_t lock;
    int active,failed,use_f16;
    uint64_t calls,uploads,mlp_calls,grouped_calls,grouped_kernel_calls,projection_groups,gdn_calls,attn_calls;
    uint64_t q3_uploads,q3_upload_bytes,q3_gemv_calls,q3_grouped_calls;
    uint64_t q3_atlas_refreshes,q3_atlas_loads,q3_atlas_bytes;
    int q3_atlas_entries,q3_atlas_capacity,q3_atlas_routes;
    uint64_t batch_transactions,batch_routes,batch_unique_experts;
    uint64_t resident_h2d,resident_d2h,resident_logits_d2h,resident_router_d2h;
    uint64_t resident_layers,resident_device_moe,resident_host_moe;
    void *upload_stage;size_t upload_stage_cap,upload_stage_used;
    int upload_stage_active;
    uint64_t pinned_upload_batches,pinned_upload_bytes;
    CudaWeightCache *weight_hash[CUDA_WEIGHT_BUCKETS],*lru_head,*lru_tail;
    size_t expert_bytes,mtp_expert_bytes,expert_budget;uint64_t weight_clock,cache_hits,evictions;
} CudaRuntime;
static CudaRuntime cuda_rt={.lock=PTHREAD_MUTEX_INITIALIZER};
/* Phase-6 target verification remains on the exact CPU batch path while the
 * one-layer MTP drafter may use CUDA. */
static int cuda_suppress;

static void cuda_backend_stop(void){
    if(!cuda_rt.ctx)return;
    fprintf(stderr,"[CUDA] qmat calls=%llu fused-mlp calls=%llu grouped-expert calls=%llu grouped-kernel calls=%llu projection-groups=%llu GDN-calls=%llu GQA-calls=%llu uploads=%llu cache-hits=%llu evictions=%llu expert-VRAM=%.2fGiB mtp-expert-VRAM=%.2fGiB\n",(unsigned long long)cuda_rt.calls,(unsigned long long)cuda_rt.mlp_calls,(unsigned long long)cuda_rt.grouped_calls,(unsigned long long)cuda_rt.grouped_kernel_calls,(unsigned long long)cuda_rt.projection_groups,(unsigned long long)cuda_rt.gdn_calls,(unsigned long long)cuda_rt.attn_calls,(unsigned long long)cuda_rt.uploads,(unsigned long long)cuda_rt.cache_hits,(unsigned long long)cuda_rt.evictions,(double)cuda_rt.expert_bytes/(1024.*1024.*1024.),(double)cuda_rt.mtp_expert_bytes/(1024.*1024.*1024.));
    if(cuda_rt.q3_uploads)fprintf(stderr,"[CUDA_Q3] native-uploads=%llu upload=%.3fGiB gemv-calls=%llu grouped-calls=%llu\n",(unsigned long long)cuda_rt.q3_uploads,(double)cuda_rt.q3_upload_bytes/(1024.*1024.*1024.),(unsigned long long)cuda_rt.q3_gemv_calls,(unsigned long long)cuda_rt.q3_grouped_calls);
    if(cuda_rt.q3_atlas_refreshes)fprintf(stderr,
        "[CUDA_Q3_ATLAS] refreshes=%llu routes=%d entries=%d capacity=%d loads=%llu payload=%.3fGiB\n",
        (unsigned long long)cuda_rt.q3_atlas_refreshes,
        cuda_rt.q3_atlas_routes,cuda_rt.q3_atlas_entries,
        cuda_rt.q3_atlas_capacity,
        (unsigned long long)cuda_rt.q3_atlas_loads,
        (double)cuda_rt.q3_atlas_bytes/(1024.*1024.*1024.));
    if(cuda_rt.pinned_upload_batches)fprintf(stderr,
        "[CUDA_UPLOAD] pinned-batches=%llu staged=%.3fGiB capacity=%.3fGiB\n",
        (unsigned long long)cuda_rt.pinned_upload_batches,
        (double)cuda_rt.pinned_upload_bytes/(1024.*1024.*1024.),
        (double)cuda_rt.upload_stage_cap/(1024.*1024.*1024.));
    if(cuda_rt.batch_transactions){double reuse=1.0-(double)cuda_rt.batch_unique_experts/(double)cuda_rt.batch_routes;fprintf(stderr,"[CUDA_BATCH] transactions=%llu routes=%llu unique=%llu overlap=%.2f%% routes/transaction=%.2f unique/transaction=%.2f\n",(unsigned long long)cuda_rt.batch_transactions,(unsigned long long)cuda_rt.batch_routes,(unsigned long long)cuda_rt.batch_unique_experts,100.0*reuse,(double)cuda_rt.batch_routes/(double)cuda_rt.batch_transactions,(double)cuda_rt.batch_unique_experts/(double)cuda_rt.batch_transactions);}
    if(cuda_rt.resident_layers)fprintf(stderr,"[CUDA_RESIDENT] layers=%llu device-moe=%llu host-moe=%llu activation-h2d=%llu activation-d2h=%llu logits-d2h=%llu router-d2h=%llu bytes\n",
        (unsigned long long)cuda_rt.resident_layers,
        (unsigned long long)cuda_rt.resident_device_moe,
        (unsigned long long)cuda_rt.resident_host_moe,
        (unsigned long long)cuda_rt.resident_h2d,
        (unsigned long long)cuda_rt.resident_d2h,
        (unsigned long long)cuda_rt.resident_logits_d2h,
        (unsigned long long)cuda_rt.resident_router_d2h);
    coli_cuda_free(cuda_rt.ctx,cuda_rt.x);coli_cuda_free(cuda_rt.ctx,cuda_rt.y);coli_cuda_free(cuda_rt.ctx,cuda_rt.gate);coli_cuda_free(cuda_rt.ctx,cuda_rt.up);coli_cuda_free(cuda_rt.ctx,cuda_rt.tmp);coli_cuda_free(cuda_rt.ctx,cuda_rt.x16);coli_cuda_free(cuda_rt.ctx,cuda_rt.gate16);
    for(int i=0;i<4;i++)coli_cuda_free(cuda_rt.ctx,cuda_rt.proj[i]);
    coli_cuda_free_host(cuda_rt.ctx,cuda_rt.upload_stage);
    for(int i=0;i<CUDA_WEIGHT_BUCKETS;i++){CudaWeightCache*e=cuda_rt.weight_hash[i];while(e){CudaWeightCache*n=e->hnext;coli_cuda_free(cuda_rt.ctx,e->d_q);coli_cuda_free(cuda_rt.ctx,e->d_s);free(e->key);free(e);e=n;}}
    coli_cuda_destroy(cuda_rt.ctx);cuda_rt.ctx=NULL;
}

static int cuda_backend_start(void){
    const char*ce=getenv("COLI_CUDA"),*de=getenv("CUDA_DENSE");
    if(!ce||atoi(ce)==0||!de||atoi(de)==0)return 0;
    int device=getenv("CUDA_DEVICE")?atoi(getenv("CUDA_DEVICE")):0;
    if(coli_cuda_create(&cuda_rt.ctx,device)){fprintf(stderr,"[CUDA] disabled: %s\n",coli_cuda_last_error());cuda_rt.failed=1;return 0;}
    cuda_rt.active=1;cuda_rt.use_f16=getenv("CUDA_F16")&&atoi(getenv("CUDA_F16"))!=0;
    double expert_gb=getenv("CUDA_EXPERT_GB")?atof(getenv("CUDA_EXPERT_GB")):8.;if(expert_gb<0.)expert_gb=0.;cuda_rt.expert_budget=(size_t)(expert_gb*1024.*1024.*1024.);
    atexit(cuda_backend_stop);
    int major=0,minor=0;coli_cuda_compute_capability(cuda_rt.ctx,&major,&minor);
    fprintf(stderr,"[CUDA] dense backend: %s (sm_%d%d), activations=%s, allocator=%s\n",coli_cuda_device_name(cuda_rt.ctx),major,minor,cuda_rt.use_f16?"fp16":"fp32",coli_cuda_async_alloc_enabled(cuda_rt.ctx)?"stream-ordered":"synchronous");
    return 1;
}
static void cuda_backend_fail(const char*where){
    if(!cuda_rt.failed)fprintf(stderr,"[CUDA] %s failed; falling back to CPU: %s\n",where,coli_cuda_last_error());
    cuda_rt.failed=1;cuda_rt.active=0;
}
static int cuda_scratch(void**p,size_t*cap,size_t need){
    if(*cap>=need)return 0;
    coli_cuda_free(cuda_rt.ctx,*p);*p=NULL;*cap=0;
    if(coli_cuda_malloc(cuda_rt.ctx,p,need))return -1;*cap=need;return 0;
}
static uint32_t cuda_key_hash(const char*s){uint32_t h=2166136261u;for(;*s;s++)h=(h^(unsigned char)*s)*16777619u;return h&(CUDA_WEIGHT_BUCKETS-1);}
static CudaWeightCache*cuda_cache_find(const char*key){uint32_t h=cuda_key_hash(key);for(CudaWeightCache*e=cuda_rt.weight_hash[h];e;e=e->hnext)if(!strcmp(e->key,key))return e;return NULL;}
static void cuda_lru_unlink(CudaWeightCache*e){if(e->prev)e->prev->next=e->next;else cuda_rt.lru_head=e->next;if(e->next)e->next->prev=e->prev;else cuda_rt.lru_tail=e->prev;e->prev=e->next=NULL;}
static void cuda_lru_touch(CudaWeightCache*e){if(!e->expert||e->pinned)return;if(e==cuda_rt.lru_head){e->stamp=++cuda_rt.weight_clock;return;}if(e->prev||e->next||cuda_rt.lru_tail==e)cuda_lru_unlink(e);e->prev=NULL;e->next=cuda_rt.lru_head;if(cuda_rt.lru_head)cuda_rt.lru_head->prev=e;else cuda_rt.lru_tail=e;cuda_rt.lru_head=e;e->stamp=++cuda_rt.weight_clock;}
static void cuda_cache_remove(CudaWeightCache*e){uint32_t h=cuda_key_hash(e->key);CudaWeightCache**p=&cuda_rt.weight_hash[h];while(*p&&*p!=e)p=&(*p)->hnext;if(*p)*p=e->hnext;if(e->expert&&!e->pinned)cuda_lru_unlink(e);if(e->expert){if(e->mtp)cuda_rt.mtp_expert_bytes-=e->bytes;else cuda_rt.expert_bytes-=e->bytes;}if(e->owner){e->owner->cuda_ready=0;e->owner->d_q=e->owner->d_s=e->owner->cuda_cache=NULL;}coli_cuda_free(cuda_rt.ctx,e->d_q);coli_cuda_free(cuda_rt.ctx,e->d_s);free(e->key);free(e);cuda_rt.evictions++;}
static void cuda_cache_make_room(size_t bytes){while(cuda_rt.expert_budget&&cuda_rt.expert_bytes+bytes>cuda_rt.expert_budget&&cuda_rt.lru_tail)cuda_cache_remove(cuda_rt.lru_tail);}
static int cuda_q3_native_enabled(void){
    const char*e=getenv("Q3_NATIVE");
    return e&&atoi(e)!=0;
}
static size_t cuda_qmat_device_qbytes(const QMat*w){
    return (size_t)w->O*(w->fmt==3&&cuda_q3_native_enabled()?w->rb:w->drb);
}
static void q2_expand_q4(const QMat*w,uint8_t*out){
    size_t count=(size_t)w->O*w->rb;
    for(size_t i=0;i<count;i++){
        uint8_t b=w->q4[i];
        uint8_t c0=(uint8_t)(5+2*(b&3)),c1=(uint8_t)(5+2*((b>>2)&3));
        uint8_t c2=(uint8_t)(5+2*((b>>4)&3)),c3=(uint8_t)(5+2*(b>>6));
        out[2*i]=(uint8_t)(c0|(c1<<4));out[2*i+1]=(uint8_t)(c2|(c3<<4));
    }
}
static int cuda_qmat_upload_locked(QMat*w){
    if(w->cuda_ready){CudaWeightCache*e=(CudaWeightCache*)w->cuda_cache;if(e)cuda_lru_touch(e);return 0;}
    uint32_t h=cuda_key_hash(w->cuda_key);CudaWeightCache*hit=cuda_cache_find(w->cuda_key);if(hit){w->d_q=hit->d_q;w->d_s=hit->d_s;w->cuda_cache=hit;w->cuda_ready=1;hit->owner=w;cuda_lru_touch(hit);cuda_rt.cache_hits++;return 0;}
    size_t qbytes=cuda_qmat_device_qbytes(w),sbytes=(size_t)w->O*(w->fmt==1?1:w->ng)*sizeof(float),bytes=qbytes+sbytes;int expert=strstr(w->cuda_key,".experts.")!=NULL,pinned=expert&&!strncmp(w->cuda_key,"mtp.",4);if(expert&&!pinned){cuda_cache_make_room(bytes);if(cuda_rt.expert_budget&&cuda_rt.expert_bytes+bytes>cuda_rt.expert_budget)return -1;}
    const void*qsrc=w->fmt==1?(const void*)w->q8:(const void*)w->q4;
    const void*ssrc=w->s;uint8_t*temporary=NULL;
    if(cuda_rt.upload_stage_active){
        if(cuda_rt.upload_stage_used+qbytes+sbytes>cuda_rt.upload_stage_cap)
            return -1;
        unsigned char*stage=(unsigned char*)cuda_rt.upload_stage+
                            cuda_rt.upload_stage_used;
        if(w->fmt==2)q2_expand_q4(w,stage);
        else if(w->fmt==3&&!cuda_q3_native_enabled())q3_expand_q4(w,stage);
        else memcpy(stage,qsrc,qbytes);
        memcpy(stage+qbytes,ssrc,sbytes);
        qsrc=stage;ssrc=stage+qbytes;
        cuda_rt.upload_stage_used+=qbytes+sbytes;
        cuda_rt.pinned_upload_bytes+=qbytes+sbytes;
    }else if(w->fmt==2||(w->fmt==3&&!cuda_q3_native_enabled())){
        temporary=xcalloc(qbytes,1);
        if(w->fmt==2)q2_expand_q4(w,temporary);else q3_expand_q4(w,temporary);
        qsrc=temporary;
    }
    void*dq=NULL,*ds=NULL;if(coli_cuda_malloc(cuda_rt.ctx,&dq,qbytes)||coli_cuda_malloc(cuda_rt.ctx,&ds,sbytes)||
       coli_cuda_upload(cuda_rt.ctx,dq,qsrc,qbytes)||coli_cuda_upload(cuda_rt.ctx,ds,ssrc,sbytes)){coli_cuda_free(cuda_rt.ctx,dq);coli_cuda_free(cuda_rt.ctx,ds);free(temporary);return -1;}
    if(temporary&&coli_cuda_sync(cuda_rt.ctx)){coli_cuda_free(cuda_rt.ctx,dq);coli_cuda_free(cuda_rt.ctx,ds);free(temporary);return -1;}free(temporary);
    CudaWeightCache*e=xcalloc(1,sizeof(*e));e->key=strdup(w->cuda_key);e->d_q=dq;e->d_s=ds;e->bytes=bytes;e->expert=expert;e->pinned=pinned;e->mtp=pinned;e->fmt=w->fmt;e->O=w->O;e->I=w->I;e->gs=w->gs;e->rb=w->rb;e->drb=w->drb;e->ng=w->ng;e->owner=w;e->hnext=cuda_rt.weight_hash[h];cuda_rt.weight_hash[h]=e;if(expert){if(e->mtp)cuda_rt.mtp_expert_bytes+=bytes;else{cuda_rt.expert_bytes+=bytes;cuda_lru_touch(e);}}w->d_q=dq;w->d_s=ds;w->cuda_cache=e;w->cuda_ready=1;cuda_rt.uploads++;if(w->fmt==3&&cuda_q3_native_enabled()){cuda_rt.q3_uploads++;cuda_rt.q3_upload_bytes+=qbytes;}return 0;
}
static int cuda_experts_upload_locked(Expert*const*expert,int routes){
    int enabled=0;const char*env=getenv("CUDA_PINNED_UPLOAD");
    if(env&&atoi(env)!=0)enabled=1;
    QMat*unique[QW_MAX_TOPK*3];int n=0;size_t reserve=0;
    for(int r=0;r<routes;r++){QMat*q[3]={&expert[r]->gate,&expert[r]->up,
        &expert[r]->down};for(int j=0;j<3;j++){int seen=0;
        for(int p=0;p<n;p++)if(unique[p]==q[j]){seen=1;break;}
        if(seen)continue;if(n>=QW_MAX_TOPK*3)return -1;unique[n++]=q[j];
        if(q[j]->fmt==2||q[j]->fmt==3)enabled=1;
    }}
    for(int i=0;i<n;i++){
        QMat*q=unique[i];
        if(enabled&&!q->cuda_ready&&!cuda_cache_find(q->cuda_key))
            reserve+=cuda_qmat_device_qbytes(q)+
                     (size_t)q->O*(q->fmt==1?1:q->ng)*sizeof(float);
    }
    if(reserve){
        if(coli_cuda_sync(cuda_rt.ctx))return -1;
        if(cuda_rt.upload_stage_cap<reserve){
            void*next=NULL;
            if(coli_cuda_malloc_host(cuda_rt.ctx,&next,reserve))return -1;
            coli_cuda_free_host(cuda_rt.ctx,cuda_rt.upload_stage);
            cuda_rt.upload_stage=next;cuda_rt.upload_stage_cap=reserve;
        }
        cuda_rt.upload_stage_used=0;cuda_rt.upload_stage_active=1;
        cuda_rt.pinned_upload_batches++;
    }
    int error=0;for(int i=0;i<n&&!error;i++)
        error=cuda_qmat_upload_locked(unique[i]);
    cuda_rt.upload_stage_active=0;cuda_rt.upload_stage_used=0;
    return error;
}
static void cuda_qmat_detach(QMat*w){CudaWeightCache*e=(CudaWeightCache*)w->cuda_cache;if(e&&e->owner==w)e->owner=NULL;w->d_q=w->d_s=w->cuda_cache=NULL;w->cuda_ready=0;}
static int cuda_qmat_launch_locked(float*dy,const void*dx,QMat*w,int input_f16){
    if(w->fmt==1){
        if(input_f16)return -1;
        return coli_cuda_q8_gemv(cuda_rt.ctx,dy,(const float*)dx,(const signed char*)w->d_q,(const float*)w->d_s,w->O,w->I,w->rb);
    }
    if(w->fmt==3){
        if(cuda_q3_native_enabled()){
            int rc=input_f16?coli_cuda_q3_gemv_f16(cuda_rt.ctx,dy,(const unsigned short*)dx,(const unsigned char*)w->d_q,(const float*)w->d_s,w->O,w->I,w->gs,w->rb,w->ng):coli_cuda_q3_gemv(cuda_rt.ctx,dy,(const float*)dx,(const unsigned char*)w->d_q,(const float*)w->d_s,w->O,w->I,w->gs,w->rb,w->ng);if(!rc)cuda_rt.q3_gemv_calls++;return rc;
        }
        if(input_f16)return coli_cuda_q4_gemv_f16(cuda_rt.ctx,dy,(const unsigned short*)dx,(const unsigned char*)w->d_q,(const float*)w->d_s,w->O,w->I,w->gs,w->drb,w->ng);
        return coli_cuda_q4_gemv(cuda_rt.ctx,dy,(const float*)dx,(const unsigned char*)w->d_q,(const float*)w->d_s,w->O,w->I,w->gs,w->drb,w->ng);
    }
    if(input_f16)return coli_cuda_q4_gemv_f16(cuda_rt.ctx,dy,(const unsigned short*)dx,(const unsigned char*)w->d_q,(const float*)w->d_s,w->O,w->I,w->gs,w->drb,w->ng);
    return coli_cuda_q4_gemv(cuda_rt.ctx,dy,(const float*)dx,(const unsigned char*)w->d_q,(const float*)w->d_s,w->O,w->I,w->gs,w->drb,w->ng);
}
static int cuda_qmat_batch_device_locked(float*dy,const float*dx,QMat*w,int B){
    if(B<=0||!w->cuda_eligible||(w->fmt!=1&&w->fmt!=2&&w->fmt!=3&&w->fmt!=4))return -1;
    if(cuda_qmat_upload_locked(w))return -1;
    if(w->fmt==1)return coli_cuda_q8_gemm(
        cuda_rt.ctx,dy,dx,B,(const signed char*)w->d_q,
        (const float*)w->d_s,w->O,w->I,w->rb);
    if(w->fmt==3&&cuda_q3_native_enabled())return -1;
    return coli_cuda_q4_gemm(
        cuda_rt.ctx,dy,dx,B,(const unsigned char*)w->d_q,
        (const float*)w->d_s,w->O,w->I,w->gs,w->drb,w->ng);
}
static int cuda_qmat_try(float*y,const float*x,const QMat*wc){
    if(cuda_suppress||!cuda_rt.active||!wc->cuda_eligible||(wc->fmt!=1&&wc->fmt!=2&&wc->fmt!=3&&wc->fmt!=4))return 0;
    QMat*w=(QMat*)wc;int used=0;pthread_mutex_lock(&cuda_rt.lock);
    if(!cuda_rt.active)goto out;
    if(cuda_qmat_upload_locked(w)){cuda_backend_fail("matrix upload");goto out;}
    if(cuda_scratch((void**)&cuda_rt.x,&cuda_rt.xcap,(size_t)w->I*sizeof(float))||
       cuda_scratch((void**)&cuda_rt.y,&cuda_rt.ycap,(size_t)w->O*sizeof(float))){cuda_backend_fail("activation allocation");goto out;}
    if(coli_cuda_upload(cuda_rt.ctx,cuda_rt.x,x,(size_t)w->I*sizeof(float))){cuda_backend_fail("activation upload");goto out;}
    int rc,input_f16=0;
    if(w->fmt!=1&&cuda_rt.use_f16){
        if(cuda_scratch((void**)&cuda_rt.x16,&cuda_rt.x16cap,(size_t)w->I*sizeof(unsigned short))||
           coli_cuda_f32_to_f16(cuda_rt.ctx,cuda_rt.x16,cuda_rt.x,w->I)){cuda_backend_fail("fp16 activation conversion");goto out;}
        input_f16=1;
    }
    rc=cuda_qmat_launch_locked(cuda_rt.y,input_f16?(const void*)cuda_rt.x16:(const void*)cuda_rt.x,w,input_f16);
    if(rc||coli_cuda_download(cuda_rt.ctx,y,cuda_rt.y,(size_t)w->O*sizeof(float))||coli_cuda_sync(cuda_rt.ctx)){cuda_backend_fail("matrix execution");goto out;}
    cuda_rt.calls++;used=1;
out:pthread_mutex_unlock(&cuda_rt.lock);return used;
}
static int cuda_qmat_batch_try(float*y,const float*x,int B,int ldx,const QMat*wc){
    const char*be=getenv("CUDA_SPEC_BATCH");if(!be||atoi(be)==0||!cuda_rt.active||B<2||B>8||ldx!=wc->I||wc->fmt!=4||!wc->cuda_eligible||!strstr(wc->cuda_key,".linear_attn."))return 0;
    QMat*w=(QMat*)wc;int used=0;pthread_mutex_lock(&cuda_rt.lock);if(!cuda_rt.active)goto out;
    if(cuda_qmat_upload_locked(w)){cuda_backend_fail("batch matrix upload");goto out;}
    if(cuda_scratch((void**)&cuda_rt.x,&cuda_rt.xcap,(size_t)B*w->I*sizeof(float))||cuda_scratch((void**)&cuda_rt.y,&cuda_rt.ycap,(size_t)B*w->O*sizeof(float))){cuda_backend_fail("batch activation allocation");goto out;}
    if(coli_cuda_upload(cuda_rt.ctx,cuda_rt.x,x,(size_t)B*w->I*sizeof(float))||
       coli_cuda_q4_gemm(cuda_rt.ctx,cuda_rt.y,cuda_rt.x,B,(const unsigned char*)w->d_q,(const float*)w->d_s,w->O,w->I,w->gs,w->rb,w->ng)||
       coli_cuda_download(cuda_rt.ctx,y,cuda_rt.y,(size_t)B*w->O*sizeof(float))||coli_cuda_sync(cuda_rt.ctx)){cuda_backend_fail("batch matrix execution");goto out;}
    cuda_rt.calls+=(uint64_t)B;used=1;
out:pthread_mutex_unlock(&cuda_rt.lock);return used;
}
static int cuda_mlp_try(float*y,const float*x,const QMat*gc,const QMat*uc,const QMat*dc){
    if(cuda_suppress)return 0;
    static int enabled=-1;if(enabled<0){const char*e=getenv("CUDA_MLP");enabled=!e||atoi(e)!=0;}
    if(!enabled||!cuda_rt.active||!gc->cuda_eligible||!uc->cuda_eligible||!dc->cuda_eligible)return 0;
    if((gc->fmt!=1&&gc->fmt!=2&&gc->fmt!=3&&gc->fmt!=4)||uc->fmt!=gc->fmt||dc->fmt!=gc->fmt||gc->O!=uc->O||gc->I!=uc->I||dc->I!=gc->O)return 0;
    QMat*g=(QMat*)gc,*u=(QMat*)uc,*d=(QMat*)dc;int used=0,H=g->I,I=g->O;pthread_mutex_lock(&cuda_rt.lock);
    if(!cuda_rt.active)goto out;
    if(cuda_qmat_upload_locked(g)||cuda_qmat_upload_locked(u)||cuda_qmat_upload_locked(d)){cuda_backend_fail("MLP matrix upload");goto out;}
    if(cuda_scratch((void**)&cuda_rt.x,&cuda_rt.xcap,(size_t)H*sizeof(float))||
       cuda_scratch((void**)&cuda_rt.y,&cuda_rt.ycap,(size_t)d->O*sizeof(float))||
       cuda_scratch((void**)&cuda_rt.gate,&cuda_rt.gatecap,(size_t)I*sizeof(float))||
       cuda_scratch((void**)&cuda_rt.up,&cuda_rt.upcap,(size_t)I*sizeof(float))){cuda_backend_fail("MLP activation allocation");goto out;}
    if(coli_cuda_upload(cuda_rt.ctx,cuda_rt.x,x,(size_t)H*sizeof(float))){cuda_backend_fail("MLP activation upload");goto out;}
    const void*din=cuda_rt.x;int input_f16=0;
    if(g->fmt!=1&&cuda_rt.use_f16){
        if(cuda_scratch((void**)&cuda_rt.x16,&cuda_rt.x16cap,(size_t)H*sizeof(unsigned short))||
           coli_cuda_f32_to_f16(cuda_rt.ctx,cuda_rt.x16,cuda_rt.x,H)){cuda_backend_fail("MLP fp16 input conversion");goto out;}
        din=cuda_rt.x16;input_f16=1;
    }
    if(cuda_qmat_launch_locked(cuda_rt.gate,din,g,input_f16)||cuda_qmat_launch_locked(cuda_rt.up,din,u,input_f16)||
       coli_cuda_silu_mul(cuda_rt.ctx,cuda_rt.gate,cuda_rt.gate,cuda_rt.up,I)){cuda_backend_fail("MLP hidden projection");goto out;}
    din=cuda_rt.gate;input_f16=0;
    if(d->fmt!=1&&cuda_rt.use_f16){
        if(cuda_scratch((void**)&cuda_rt.gate16,&cuda_rt.gate16cap,(size_t)I*sizeof(unsigned short))||
           coli_cuda_f32_to_f16(cuda_rt.ctx,cuda_rt.gate16,cuda_rt.gate,I)){cuda_backend_fail("MLP fp16 hidden conversion");goto out;}
        din=cuda_rt.gate16;input_f16=1;
    }
    if(cuda_qmat_launch_locked(cuda_rt.y,din,d,input_f16)||coli_cuda_download(cuda_rt.ctx,y,cuda_rt.y,(size_t)d->O*sizeof(float))||coli_cuda_sync(cuda_rt.ctx)){cuda_backend_fail("MLP down projection");goto out;}
    cuda_rt.calls+=3;cuda_rt.mlp_calls++;used=1;
out:pthread_mutex_unlock(&cuda_rt.lock);return used;
}
static int cuda_projection_group_try(float*const*y,const float*x,const QMat*const*matrix,int n){
    if(cuda_suppress)return 0;
    static int enabled=-1;if(enabled<0){const char*e=getenv("CUDA_PROJECTIONS");enabled=!e||atoi(e)!=0;}
    if(!enabled||!cuda_rt.active||n<2||n>4)return 0;
    int fmt=matrix[0]->fmt,I=matrix[0]->I;if(fmt!=1&&fmt!=2&&fmt!=3&&fmt!=4)return 0;
    for(int j=0;j<n;j++)if(!matrix[j]->cuda_eligible||matrix[j]->fmt!=fmt||matrix[j]->I!=I)return 0;
    int used=0;pthread_mutex_lock(&cuda_rt.lock);if(!cuda_rt.active)goto out;
    for(int j=0;j<n;j++)if(cuda_qmat_upload_locked((QMat*)matrix[j])){cuda_backend_fail("projection matrix upload");goto out;}
    if(cuda_scratch((void**)&cuda_rt.x,&cuda_rt.xcap,(size_t)I*sizeof(float))){cuda_backend_fail("projection input allocation");goto out;}
    for(int j=0;j<n;j++)if(cuda_scratch((void**)&cuda_rt.proj[j],&cuda_rt.projcap[j],(size_t)matrix[j]->O*sizeof(float))){cuda_backend_fail("projection output allocation");goto out;}
    if(coli_cuda_upload(cuda_rt.ctx,cuda_rt.x,x,(size_t)I*sizeof(float))){cuda_backend_fail("projection input upload");goto out;}
    const void*input=cuda_rt.x;int input_f16=0;
    if(fmt!=1&&cuda_rt.use_f16){
        if(cuda_scratch((void**)&cuda_rt.x16,&cuda_rt.x16cap,(size_t)I*sizeof(unsigned short))||coli_cuda_f32_to_f16(cuda_rt.ctx,cuda_rt.x16,cuda_rt.x,I)){cuda_backend_fail("projection fp16 conversion");goto out;}
        input=cuda_rt.x16;input_f16=1;
    }
    for(int j=0;j<n;j++)if(cuda_qmat_launch_locked(cuda_rt.proj[j],input,(QMat*)matrix[j],input_f16)){cuda_backend_fail("projection kernel");goto out;}
    for(int j=0;j<n;j++)if(coli_cuda_download(cuda_rt.ctx,y[j],cuda_rt.proj[j],(size_t)matrix[j]->O*sizeof(float))){cuda_backend_fail("projection download");goto out;}
    if(coli_cuda_sync(cuda_rt.ctx)){cuda_backend_fail("projection synchronization");goto out;}
    cuda_rt.calls+=(uint64_t)n;cuda_rt.projection_groups++;used=1;
out:pthread_mutex_unlock(&cuda_rt.lock);return used;
}
static int cuda_gdn_eligible(GdnW*w){static int enabled=-1;if(enabled<0){const char*e=getenv("CUDA_GDN");enabled=!e||atoi(e)!=0;}return enabled&&cuda_rt.active&&cuda_rt.use_f16&&w->qkv.fmt==4&&w->z.fmt==4&&w->out.fmt==4&&w->qkv.cuda_eligible&&w->z.cuda_eligible&&w->out.cuda_eligible;}
static int cuda_gdn_prepare_locked(GdnW*w,const Cfg*c,int pos){
    int kh=c->lin_k_heads,vh=c->lin_v_heads,dk=c->lin_k_dim,dv=c->lin_v_dim,kd=kh*dk,vd=vh*dv,cd=2*kd+vd,K=c->conv_kernel;
    if(cuda_qmat_upload_locked(&w->qkv)||cuda_qmat_upload_locked(&w->z)||cuda_qmat_upload_locked(&w->out)){cuda_backend_fail("GDN matrix upload");goto done;}
    if(!w->cuda_aux_ready){
        size_t conv_bytes=(size_t)cd*K*sizeof(float),head_bytes=(size_t)vh*sizeof(float),norm_bytes=(size_t)dv*sizeof(float),conv_state_bytes=(size_t)cd*K*sizeof(float),state_bytes=(size_t)vh*dk*dv*sizeof(float);
        if(coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_conv,conv_bytes)||coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_A_log,head_bytes)||coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_dt_bias,head_bytes)||coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_norm,norm_bytes)||coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_conv_state,conv_state_bytes)||coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_state,state_bytes)||coli_cuda_upload(cuda_rt.ctx,w->d_conv,w->conv,conv_bytes)||coli_cuda_upload(cuda_rt.ctx,w->d_A_log,w->A_log,head_bytes)||coli_cuda_upload(cuda_rt.ctx,w->d_dt_bias,w->dt_bias,head_bytes)||coli_cuda_upload(cuda_rt.ctx,w->d_norm,w->norm,norm_bytes)||coli_cuda_upload(cuda_rt.ctx,w->d_conv_state,w->conv_state,conv_state_bytes)||coli_cuda_upload(cuda_rt.ctx,w->d_state,w->state,state_bytes)){cuda_backend_fail("GDN state allocation");goto done;}
        w->cuda_aux_ready=1;w->cuda_state_pos=pos;
    }
    if(w->cuda_state_pos!=pos){size_t conv_state_bytes=(size_t)cd*K*sizeof(float),state_bytes=(size_t)vh*dk*dv*sizeof(float);if(coli_cuda_upload(cuda_rt.ctx,w->d_conv_state,w->conv_state,conv_state_bytes)||coli_cuda_upload(cuda_rt.ctx,w->d_state,w->state,state_bytes)){cuda_backend_fail("GDN state synchronization");goto done;}w->cuda_state_pos=pos;}
    return 1;
done:return 0;
}
static int cuda_gdn_try(float*out,const float*x,const float*a,const float*b,GdnW*w,const Cfg*c,int pos){
    if(cuda_suppress||!cuda_gdn_eligible(w))return 0;
    int used=0,kh=c->lin_k_heads,vh=c->lin_v_heads,dk=c->lin_k_dim,dv=c->lin_v_dim,K=c->conv_kernel;pthread_mutex_lock(&cuda_rt.lock);if(!cuda_rt.active||!cuda_gdn_prepare_locked(w,c,pos))goto done;
    if(coli_cuda_gdn_decode_q4_f16(cuda_rt.ctx,out,x,a,b,(const unsigned char*)w->qkv.d_q,(const float*)w->qkv.d_s,w->qkv.rb,w->qkv.ng,(const unsigned char*)w->z.d_q,(const float*)w->z.d_s,w->z.rb,w->z.ng,(const unsigned char*)w->out.d_q,(const float*)w->out.d_s,w->out.rb,w->out.ng,w->d_conv,w->d_A_log,w->d_dt_bias,w->d_norm,w->d_conv_state,w->d_state,pos,c->hidden,kh,vh,dk,dv,K,w->qkv.gs,c->eps)){cuda_backend_fail("GDN decode");goto done;}
    w->cuda_state_pos=pos+1;cuda_rt.calls+=3;cuda_rt.gdn_calls++;used=1;
done:pthread_mutex_unlock(&cuda_rt.lock);return used;
}
static int cuda_gdn_block_try(float*out,const float*x,const float*a,const float*b,int T,GdnW*w,const Cfg*c,int pos){
    if(cuda_suppress||!cuda_gdn_eligible(w)||T<2||T>8)return 0;int used=0,kh=c->lin_k_heads,vh=c->lin_v_heads,dk=c->lin_k_dim,dv=c->lin_v_dim,K=c->conv_kernel;pthread_mutex_lock(&cuda_rt.lock);if(!cuda_rt.active||!cuda_gdn_prepare_locked(w,c,pos))goto done;
    if(coli_cuda_gdn_block_q4_f16(cuda_rt.ctx,out,x,a,b,T,(const unsigned char*)w->qkv.d_q,(const float*)w->qkv.d_s,w->qkv.rb,w->qkv.ng,(const unsigned char*)w->z.d_q,(const float*)w->z.d_s,w->z.rb,w->z.ng,(const unsigned char*)w->out.d_q,(const float*)w->out.d_s,w->out.rb,w->out.ng,w->d_conv,w->d_A_log,w->d_dt_bias,w->d_norm,w->d_conv_state,w->d_state,pos,c->hidden,kh,vh,dk,dv,K,w->qkv.gs,c->eps)){cuda_backend_fail("GDN block decode");goto done;}w->cuda_state_pos=pos+T;cuda_rt.calls+=(uint64_t)3*T;cuda_rt.gdn_calls+=(uint64_t)T;used=1;
done:pthread_mutex_unlock(&cuda_rt.lock);return used;
}
static int cuda_gdn_slots_try(float*out,const float*x,const float*a,const float*b,
                              int B,const int*slot,const int*pos,
                              ResidentLayerState*r,int nslots,GdnW*w,const Cfg*c){
    if(cuda_suppress||!r||!r->cuda_gdn||!cuda_gdn_eligible(w))return 0;
    int used=0,kh=c->lin_k_heads,vh=c->lin_v_heads,dk=c->lin_k_dim,dv=c->lin_v_dim,K=c->conv_kernel;
    pthread_mutex_lock(&cuda_rt.lock);
    if(!cuda_rt.active||!cuda_gdn_prepare_locked(w,c,w->cuda_state_pos<0?0:w->cuda_state_pos))goto done;
    if(coli_cuda_gdn_slots_q4_f16(cuda_rt.ctx,out,x,a,b,B,slot,pos,
        (const unsigned char*)w->qkv.d_q,(const float*)w->qkv.d_s,w->qkv.rb,w->qkv.ng,
        (const unsigned char*)w->z.d_q,(const float*)w->z.d_s,w->z.rb,w->z.ng,
        (const unsigned char*)w->out.d_q,(const float*)w->out.d_s,w->out.rb,w->out.ng,
        w->d_conv,w->d_A_log,w->d_dt_bias,w->d_norm,r->d_conv,r->d_gdn,nslots,
        c->hidden,kh,vh,dk,dv,K,w->qkv.gs,c->eps)){
        cuda_backend_fail("resident GDN slot batch");used=-1;goto done;
    }
    cuda_rt.calls+=(uint64_t)3*B;cuda_rt.gdn_calls+=(uint64_t)B;used=1;
done:
    pthread_mutex_unlock(&cuda_rt.lock);return used;
}
static int cuda_spec_gdn_enabled(void){const char*e=getenv("CUDA_SPEC_GDN");return e&&atoi(e)!=0&&cuda_rt.active&&cuda_rt.use_f16;}
static int cuda_spec_gdn_batch_enabled(void){const char*e=getenv("CUDA_SPEC_GDN_BATCH");return !e||atoi(e)!=0;}
static int cuda_spec_full_enabled(void){const char*e=getenv("CUDA_SPEC_FULL");return e&&atoi(e)!=0&&cuda_rt.active;}
static int cuda_recurrent_snapshot(Model*m,int pos){
    if(!cuda_spec_gdn_enabled())return 0;Cfg*c=&m->c;int kd=c->lin_k_heads*c->lin_k_dim,vd=c->lin_v_heads*c->lin_v_dim,cd=2*kd+vd;size_t conv_bytes=(size_t)c->conv_kernel*cd*sizeof(float),state_bytes=(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim*sizeof(float);int ok=1;pthread_mutex_lock(&cuda_rt.lock);
    for(int li=0;li<c->n_layers&&ok;li++)if(m->layer[li].type==LT_LINEAR){GdnW*w=&m->layer[li].gdn;if(!cuda_gdn_eligible(w)||!cuda_gdn_prepare_locked(w,c,pos)){ok=0;break;}if((!w->d_conv_backup&&coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_conv_backup,conv_bytes))||(!w->d_state_backup&&coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_state_backup,state_bytes))||coli_cuda_copy(cuda_rt.ctx,w->d_conv_backup,w->d_conv_state,conv_bytes)||coli_cuda_copy(cuda_rt.ctx,w->d_state_backup,w->d_state,state_bytes)){cuda_backend_fail("GDN speculative snapshot");ok=0;break;}}
    if(ok&&coli_cuda_sync(cuda_rt.ctx)){cuda_backend_fail("GDN speculative snapshot sync");ok=0;}pthread_mutex_unlock(&cuda_rt.lock);return ok;
}
static void cuda_recurrent_restore(Model*m,int pos){
    Cfg*c=&m->c;int kd=c->lin_k_heads*c->lin_k_dim,vd=c->lin_v_heads*c->lin_v_dim,cd=2*kd+vd;size_t conv_bytes=(size_t)c->conv_kernel*cd*sizeof(float),state_bytes=(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim*sizeof(float);pthread_mutex_lock(&cuda_rt.lock);
    for(int li=0;li<c->n_layers;li++)if(m->layer[li].type==LT_LINEAR){GdnW*w=&m->layer[li].gdn;if(!w->d_conv_backup||!w->d_state_backup||coli_cuda_copy(cuda_rt.ctx,w->d_conv_state,w->d_conv_backup,conv_bytes)||coli_cuda_copy(cuda_rt.ctx,w->d_state,w->d_state_backup,state_bytes)){cuda_backend_fail("GDN speculative restore");break;}w->cuda_state_pos=pos;}if(coli_cuda_sync(cuda_rt.ctx))cuda_backend_fail("GDN speculative restore sync");pthread_mutex_unlock(&cuda_rt.lock);
}
static void cuda_attention_rewind(Model*m,int pos){for(int li=0;li<m->c.n_layers;li++)if(m->layer[li].type==LT_FULL&&m->layer[li].attn.cuda_aux_ready)m->layer[li].attn.cuda_state_pos=pos;}
static int cuda_attn_eligible(AttnW*w,const Cfg*c,int kv16){
    static int enabled=-1;if(enabled<0){const char*e=getenv("CUDA_ATTN");enabled=!e||atoi(e)!=0;}
    return enabled&&cuda_rt.active&&cuda_rt.use_f16&&!kv16&&w->q.fmt==4&&w->k.fmt==4&&w->v.fmt==4&&(w->o.fmt==1||w->o.fmt==4);
}
static int cuda_attn_prepare_locked(AttnW*w,const Cfg*c,int max_seq){
    int hd=c->head_dim,kvrows=c->n_kv_heads*hd;
    if(cuda_qmat_upload_locked(&w->q)||cuda_qmat_upload_locked(&w->k)||cuda_qmat_upload_locked(&w->v)||cuda_qmat_upload_locked(&w->o)){cuda_backend_fail("attention matrix upload");return 0;}
    if(!w->cuda_aux_ready){size_t nb=(size_t)hd*4,kb=(size_t)max_seq*kvrows*4;if(coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_q_norm,nb)||coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_k_norm,nb)||coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_k_cache,kb)||coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_v_cache,kb)||coli_cuda_upload(cuda_rt.ctx,w->d_q_norm,w->q_norm,nb)||coli_cuda_upload(cuda_rt.ctx,w->d_k_norm,w->k_norm,nb)){cuda_backend_fail("attention state allocation");return 0;}w->cuda_aux_ready=1;w->cuda_state_pos=0;}
    return 1;
}
static int cuda_attn_try(float*out,const float*x,AttnW*w,const Cfg*c,int pos,int max_seq,int kv16){
    if(cuda_suppress)return 0;
    if(!cuda_attn_eligible(w,c,kv16))return 0;
    int used=0,hd=c->head_dim,kvrows=c->n_kv_heads*hd,rd=(int)(hd*c->partial_rotary);pthread_mutex_lock(&cuda_rt.lock);if(!cuda_attn_prepare_locked(w,c,max_seq))goto done;
    if(w->cuda_state_pos!=pos){size_t bytes=(size_t)pos*kvrows*4;if(bytes&&(coli_cuda_upload(cuda_rt.ctx,w->d_k_cache,w->k_cache,bytes)||coli_cuda_upload(cuda_rt.ctx,w->d_v_cache,w->v_cache,bytes))){cuda_backend_fail("attention state synchronization");goto done;}w->cuda_state_pos=pos;}
    if(coli_cuda_gqa_decode_q4_f16(cuda_rt.ctx,out,x,(const unsigned char*)w->q.d_q,(const float*)w->q.d_s,w->q.rb,w->q.ng,(const unsigned char*)w->k.d_q,(const float*)w->k.d_s,w->k.rb,w->k.ng,(const unsigned char*)w->v.d_q,(const float*)w->v.d_s,w->v.rb,w->v.ng,w->o.d_q,(const float*)w->o.d_s,w->o.fmt,w->o.rb,w->o.ng,w->d_q_norm,w->d_k_norm,w->d_k_cache,w->d_v_cache,pos,c->hidden,c->n_heads,c->n_kv_heads,hd,rd,w->q.gs,c->theta,c->eps)){cuda_backend_fail("attention decode");goto done;}w->cuda_state_pos=pos+1;cuda_rt.calls+=4;cuda_rt.attn_calls++;used=1;
done:pthread_mutex_unlock(&cuda_rt.lock);return used;
}
static int cuda_attn_slots_try(float*out,const float*x,int B,const int*slot,
                               const int*pos,ResidentLayerState*r,int nslots,
                               AttnW*w,const Cfg*c,int max_seq,int kv16){
    if(cuda_suppress||!r||!r->cuda_attn||!cuda_attn_eligible(w,c,kv16))return 0;
    int used=0,hd=c->head_dim,rd=(int)(hd*c->partial_rotary);pthread_mutex_lock(&cuda_rt.lock);
    if(!cuda_rt.active||!cuda_attn_prepare_locked(w,c,max_seq))goto done;
    if(coli_cuda_gqa_slots_q4_f16(cuda_rt.ctx,out,x,B,slot,pos,
        (const unsigned char*)w->q.d_q,(const float*)w->q.d_s,w->q.rb,w->q.ng,
        (const unsigned char*)w->k.d_q,(const float*)w->k.d_s,w->k.rb,w->k.ng,
        (const unsigned char*)w->v.d_q,(const float*)w->v.d_s,w->v.rb,w->v.ng,
        w->o.d_q,(const float*)w->o.d_s,w->o.fmt,w->o.rb,w->o.ng,w->d_q_norm,w->d_k_norm,
        r->d_k,r->d_v,nslots,max_seq,c->hidden,c->n_heads,c->n_kv_heads,hd,rd,w->q.gs,c->theta,c->eps)){
        cuda_backend_fail("resident GQA slot batch");used=-1;goto done;
    }
    cuda_rt.calls+=(uint64_t)4*B;cuda_rt.attn_calls+=(uint64_t)B;used=1;
done:
    pthread_mutex_unlock(&cuda_rt.lock);return used;
}

/* Keep the residual stream, both normalizations, and the hybrid attention
 * transaction on the device. Routing/expert loading is still CPU-directed, so
 * this first graph-residency seam downloads only the post-attention normalized
 * rows for MoE and uploads the resulting MoE rows for the second residual. */
static int cuda_resident_core_try(Model*m,Layer*l,ResidentBatchState*s,
                                  ResidentLayerState*r,float*host_norm,int B,
                                  const int*slot,const int*pos){
    if(cuda_suppress||!cuda_rt.active||!s||!s->cuda_activations||!r)return 0;
    Cfg*c=&m->c;int H=c->hidden,used=0;
    pthread_mutex_lock(&cuda_rt.lock);
    if(!cuda_rt.active)goto done;
    if(!l->d_input_norm){
        size_t nb=(size_t)H*sizeof(float);
        if(coli_cuda_malloc(cuda_rt.ctx,(void**)&l->d_input_norm,nb)||
           coli_cuda_malloc(cuda_rt.ctx,(void**)&l->d_post_norm,nb)||
           coli_cuda_upload(cuda_rt.ctx,l->d_input_norm,l->input_norm,nb)||
           coli_cuda_upload(cuda_rt.ctx,l->d_post_norm,l->post_norm,nb)){
            cuda_backend_fail("resident layer norm upload");used=-1;goto done;
        }
    }
    if(coli_cuda_rmsnorm_zero_batch(cuda_rt.ctx,s->d_n,s->d_x,
                                    l->d_input_norm,B,H,c->eps)){
        cuda_backend_fail("resident input normalization");used=-1;goto done;
    }
    if(l->type==LT_LINEAR){
        GdnW*w=&l->gdn;int vh=c->lin_v_heads;
        if(!r->cuda_gdn||!cuda_gdn_prepare_locked(w,c,0)){
            used=0;goto done;
        }
        int gate_error=0;
        if(w->a.f&&w->b.f&&!w->d_a_weight){
            size_t wb=(size_t)vh*H*sizeof(float);
            if(coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_a_weight,wb)||
               coli_cuda_malloc(cuda_rt.ctx,(void**)&w->d_b_weight,wb)||
               coli_cuda_upload(cuda_rt.ctx,w->d_a_weight,w->a.f,wb)||
               coli_cuda_upload(cuda_rt.ctx,w->d_b_weight,w->b.f,wb)){
                cuda_backend_fail("resident GDN gate upload");used=-1;goto done;
            }
        }
        if(w->a.f&&w->b.f)
            gate_error=coli_cuda_f32_gemm(cuda_rt.ctx,s->d_a,s->d_n,
                                          w->d_a_weight,B,vh,H)||
                       coli_cuda_f32_gemm(cuda_rt.ctx,s->d_b,s->d_n,
                                          w->d_b_weight,B,vh,H);
        else
            gate_error=cuda_qmat_batch_device_locked(s->d_a,s->d_n,&w->a,B)||
                       cuda_qmat_batch_device_locked(s->d_b,s->d_n,&w->b,B);
        if(gate_error||
           coli_cuda_gdn_slots_q4_f16_device(
               cuda_rt.ctx,s->d_mix,s->d_n,s->d_a,s->d_b,B,slot,pos,
               (const unsigned char*)w->qkv.d_q,(const float*)w->qkv.d_s,
               w->qkv.rb,w->qkv.ng,(const unsigned char*)w->z.d_q,
               (const float*)w->z.d_s,w->z.rb,w->z.ng,
               (const unsigned char*)w->out.d_q,(const float*)w->out.d_s,
               w->out.rb,w->out.ng,w->d_conv,w->d_A_log,w->d_dt_bias,
               w->d_norm,r->d_conv,r->d_gdn,s->nslots,H,c->lin_k_heads,
               vh,c->lin_k_dim,c->lin_v_dim,c->conv_kernel,w->qkv.gs,c->eps)){
            cuda_backend_fail("resident device GDN");used=-1;goto done;
        }
        cuda_rt.calls+=(uint64_t)3*B;cuda_rt.gdn_calls+=(uint64_t)B;
    }else{
        AttnW*w=&l->attn;int hd=c->head_dim,rd=(int)(hd*c->partial_rotary);
        if(!r->cuda_attn||!cuda_attn_prepare_locked(w,c,s->max_seq)){
            used=0;goto done;
        }
        if(coli_cuda_gqa_slots_q4_f16_device(
               cuda_rt.ctx,s->d_mix,s->d_n,B,slot,pos,
               (const unsigned char*)w->q.d_q,(const float*)w->q.d_s,
               w->q.rb,w->q.ng,(const unsigned char*)w->k.d_q,
               (const float*)w->k.d_s,w->k.rb,w->k.ng,
               (const unsigned char*)w->v.d_q,(const float*)w->v.d_s,
               w->v.rb,w->v.ng,w->o.d_q,(const float*)w->o.d_s,w->o.fmt,
               w->o.rb,w->o.ng,w->d_q_norm,w->d_k_norm,r->d_k,r->d_v,
               s->nslots,s->max_seq,H,c->n_heads,c->n_kv_heads,hd,rd,
               w->q.gs,c->theta,c->eps)){
            cuda_backend_fail("resident device GQA");used=-1;goto done;
        }
        cuda_rt.calls+=(uint64_t)4*B;cuda_rt.attn_calls+=(uint64_t)B;
    }
    int boundary_error=coli_cuda_axpy(cuda_rt.ctx,s->d_x,s->d_mix,1.f,B*H)||
        coli_cuda_rmsnorm_zero_batch(cuda_rt.ctx,s->d_n,s->d_x,
                                     l->d_post_norm,B,H,c->eps);
    if(!boundary_error&&host_norm)
        boundary_error=coli_cuda_download(cuda_rt.ctx,host_norm,s->d_n,
                                          (size_t)B*H*sizeof(float))||
                       coli_cuda_sync(cuda_rt.ctx);
    if(boundary_error){
        cuda_backend_fail("resident post-attention boundary");used=-1;goto done;
    }
    if(host_norm)cuda_rt.resident_d2h+=(uint64_t)B*H*sizeof(float);
    used=1;
done:
    pthread_mutex_unlock(&cuda_rt.lock);return used;
}

static int cuda_resident_moe_commit(Model*m,ResidentBatchState*s,
                                    const float*host_moe,int B){
    if(!cuda_rt.active||!s||!s->cuda_activations)return 0;
    int ok=1,H=m->c.hidden;pthread_mutex_lock(&cuda_rt.lock);
    if(coli_cuda_upload(cuda_rt.ctx,s->d_moe,host_moe,
                        (size_t)B*H*sizeof(float))||
       coli_cuda_axpy(cuda_rt.ctx,s->d_x,s->d_moe,1.f,B*H)){
        cuda_backend_fail("resident MoE boundary");ok=0;
    }
    pthread_mutex_unlock(&cuda_rt.lock);return ok;
}

static int cuda_resident_output_try(Model*m,ResidentBatchState*s,float*hidden,
                                    float*logits,int B){
    if(!cuda_rt.active||!s||!s->cuda_activations||B<1)return 0;
    QMat*head=&m->lm_head;Cfg*c=&m->c;
    if(!head->cuda_eligible||(head->fmt!=1&&head->fmt!=4))return 0;
    int used=0;pthread_mutex_lock(&cuda_rt.lock);
    if(!m->d_final_norm){
        size_t bytes=(size_t)c->hidden*sizeof(float);
        if(coli_cuda_malloc(cuda_rt.ctx,(void**)&m->d_final_norm,bytes)||
           coli_cuda_upload(cuda_rt.ctx,m->d_final_norm,m->final_norm,bytes)){
            cuda_backend_fail("resident final norm upload");used=-1;goto done;
        }
    }
    if(coli_cuda_rmsnorm_zero_batch(cuda_rt.ctx,s->d_n,s->d_x,
                                    m->d_final_norm,B,c->hidden,c->eps)||
       cuda_qmat_batch_device_locked(s->d_logits,s->d_n,head,B)||
       coli_cuda_download(cuda_rt.ctx,hidden,s->d_n,
                          (size_t)B*c->hidden*sizeof(float))||
       coli_cuda_download(cuda_rt.ctx,logits,s->d_logits,
                          (size_t)B*c->vocab*sizeof(float))||
       coli_cuda_sync(cuda_rt.ctx)){
        cuda_backend_fail("resident output projection");used=-1;goto done;
    }
    cuda_rt.calls+=(uint64_t)B;
    cuda_rt.resident_d2h+=(uint64_t)B*c->hidden*sizeof(float);
    cuda_rt.resident_logits_d2h+=(uint64_t)B*c->vocab*sizeof(float);
    used=1;
done:
    pthread_mutex_unlock(&cuda_rt.lock);return used;
}

static void cuda_expert_preload(Expert*e){
    if(cuda_suppress||!cuda_rt.active||!e->gate.cuda_eligible)return;pthread_mutex_lock(&cuda_rt.lock);
    Expert*one[1]={e};
    if(cuda_experts_upload_locked(one,1))cuda_backend_fail("expert preload");
    pthread_mutex_unlock(&cuda_rt.lock);
}
static int cuda_cached_expert(Model*m,int layer,int eid,Expert*out){
    if(cuda_suppress||!cuda_rt.active||!(getenv("CUDA_EXPERTS")&&atoi(getenv("CUDA_EXPERTS"))!=0))return 0;
    char key[3][QW_NAME];const char*suf[3]={"gate_proj","up_proj","down_proj"};CudaWeightCache*entry[3];pthread_mutex_lock(&cuda_rt.lock);
    for(int j=0;j<3;j++){if(layer==m->c.n_layers)snprintf(key[j],sizeof(key[j]),"mtp.layers.0.mlp.experts.%d.%s.weight",eid,suf[j]);else snprintf(key[j],sizeof(key[j]),"layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[j]);entry[j]=cuda_cache_find(key[j]);if(!entry[j]){pthread_mutex_unlock(&cuda_rt.lock);return 0;}}
    QMat*q[3]={&out->gate,&out->up,&out->down};memset(out,0,sizeof(*out));
    for(int j=0;j<3;j++){q[j]->fmt=entry[j]->fmt;q[j]->O=entry[j]->O;q[j]->I=entry[j]->I;q[j]->gs=entry[j]->gs;q[j]->ng=entry[j]->ng;q[j]->rb=entry[j]->rb;q[j]->drb=entry[j]->drb;q[j]->cuda_eligible=q[j]->cuda_ready=1;q[j]->d_q=entry[j]->d_q;q[j]->d_s=entry[j]->d_s;q[j]->cuda_cache=entry[j];size_t z=strlen(key[j])+1;if(z>sizeof(q[j]->cuda_key))z=sizeof(q[j]->cuda_key);memcpy(q[j]->cuda_key,key[j],z);q[j]->cuda_key[sizeof(q[j]->cuda_key)-1]=0;cuda_lru_touch(entry[j]);}
    out->eid=eid;cuda_rt.cache_hits+=3;pthread_mutex_unlock(&cuda_rt.lock);return 1;
}
static int cuda_has_expert(Model*m,int layer,int eid){
    if(cuda_suppress||!cuda_rt.active||!(getenv("CUDA_EXPERTS")&&atoi(getenv("CUDA_EXPERTS"))!=0))return 0;
    char key[QW_NAME];const char*suf[3]={"gate_proj","up_proj","down_proj"};int found=1;pthread_mutex_lock(&cuda_rt.lock);
    for(int j=0;j<3;j++){if(layer==m->c.n_layers)snprintf(key,sizeof(key),"mtp.layers.0.mlp.experts.%d.%s.weight",eid,suf[j]);else snprintf(key,sizeof(key),"layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[j]);if(!cuda_cache_find(key)){found=0;break;}}
    pthread_mutex_unlock(&cuda_rt.lock);return found;
}
static void cuda_expert_cache_key(char*out,size_t cap,int layer,int eid,
                                  const char*suffix){
    snprintf(out,cap,"layers.%d.mlp.experts.%d.%s.weight",layer,eid,suffix);
}
static void cuda_q3_atlas_unpin_all(void){
    pthread_mutex_lock(&cuda_rt.lock);
    for(int bucket=0;bucket<CUDA_WEIGHT_BUCKETS;bucket++)
        for(CudaWeightCache*e=cuda_rt.weight_hash[bucket];e;e=e->hnext)
            if(e->atlas){
                e->atlas=0;e->pinned=0;cuda_lru_touch(e);
            }
    cuda_rt.q3_atlas_entries=0;cuda_rt.q3_atlas_bytes=0;
    pthread_mutex_unlock(&cuda_rt.lock);
}
static void cuda_q3_atlas_drop_expert(int layer,int eid){
    static const char*suffix[3]={"gate_proj","up_proj","down_proj"};
    pthread_mutex_lock(&cuda_rt.lock);
    for(int j=0;j<3;j++){
        char key[QW_NAME];cuda_expert_cache_key(key,sizeof(key),layer,eid,suffix[j]);
        CudaWeightCache*e=cuda_cache_find(key);
        if(e&&!e->mtp)cuda_cache_remove(e);
    }
    pthread_mutex_unlock(&cuda_rt.lock);
}
static int cuda_q3_atlas_pin_expert(int layer,int eid,size_t*bytes){
    static const char*suffix[3]={"gate_proj","up_proj","down_proj"};
    CudaWeightCache*entry[3]={0};size_t total=0;int ok=1;
    pthread_mutex_lock(&cuda_rt.lock);
    for(int j=0;j<3;j++){
        char key[QW_NAME];cuda_expert_cache_key(key,sizeof(key),layer,eid,suffix[j]);
        entry[j]=cuda_cache_find(key);
        if(!entry[j]||entry[j]->fmt!=3||entry[j]->mtp){ok=0;break;}
        total+=entry[j]->bytes;
    }
    if(ok)for(int j=0;j<3;j++)if(!entry[j]->atlas){
        if(!entry[j]->pinned)cuda_lru_unlink(entry[j]);
        entry[j]->pinned=1;entry[j]->atlas=1;
    }
    pthread_mutex_unlock(&cuda_rt.lock);
    if(ok&&bytes)*bytes=total;return ok;
}
static void cuda_model_preload_dense(Model*m){
    const char*pe=getenv("CUDA_PRELOAD");if(!cuda_rt.active||(pe&&atoi(pe)==0))return;
    pthread_mutex_lock(&cuda_rt.lock);QMat*list[16];
#define PRELOAD_ONE(q) do{QMat*_q=(q);if(_q->cuda_eligible&&(_q->fmt==1||_q->fmt==4)&&cuda_qmat_upload_locked(_q)){cuda_backend_fail("dense preload");goto done;}}while(0)
    PRELOAD_ONE(&m->lm_head);
    for(int li=0;li<m->c.n_layers;li++){Layer*l=&m->layer[li];int n=0;if(l->type==LT_LINEAR){list[n++]=&l->gdn.qkv;list[n++]=&l->gdn.z;list[n++]=&l->gdn.b;list[n++]=&l->gdn.a;list[n++]=&l->gdn.out;}else{list[n++]=&l->attn.q;list[n++]=&l->attn.k;list[n++]=&l->attn.v;list[n++]=&l->attn.o;}list[n++]=&l->moe.router;list[n++]=&l->moe.shared_gate;list[n++]=&l->moe.shared_up;list[n++]=&l->moe.shared_down;list[n++]=&l->moe.shared_scale;for(int j=0;j<n;j++)PRELOAD_ONE(list[j]);}
done:pthread_mutex_unlock(&cuda_rt.lock);
#undef PRELOAD_ONE
}
static int cuda_shared_add_locked(MoeW*w,int H){
    QMat*g=&w->shared_gate,*u=&w->shared_up,*d=&w->shared_down,*scale=&w->shared_scale;int I=g->O;
    if((g->fmt!=1&&g->fmt!=4)||u->fmt!=g->fmt||d->fmt!=g->fmt||
       (scale->fmt!=1&&scale->fmt!=4)||g->I!=H||u->I!=H||u->O!=I||
       d->I!=I||d->O!=H||scale->I!=H||scale->O!=1)return 0;
    if(cuda_qmat_upload_locked(g)||cuda_qmat_upload_locked(u)||
       cuda_qmat_upload_locked(d)||cuda_qmat_upload_locked(scale))return -1;
    if(g->fmt==4&&scale->fmt==4&&cuda_rt.use_f16){
        if(coli_cuda_shared_q4_mlp_f16(cuda_rt.ctx,cuda_rt.y,cuda_rt.x16,
           (const unsigned char*)g->d_q,(const float*)g->d_s,
           (const unsigned char*)u->d_q,(const float*)u->d_s,
           (const unsigned char*)d->d_q,(const float*)d->d_s,
           (const unsigned char*)scale->d_q,(const float*)scale->d_s,
           H,I,g->gs,g->rb,g->ng,d->rb,d->ng,scale->rb,scale->ng))return -1;
    }else{
        if(cuda_scratch((void**)&cuda_rt.gate,&cuda_rt.gatecap,(size_t)I*sizeof(float))||
           cuda_scratch((void**)&cuda_rt.up,&cuda_rt.upcap,(size_t)I*sizeof(float))||
           cuda_scratch((void**)&cuda_rt.tmp,&cuda_rt.tmpcap,(size_t)H*sizeof(float))||
           cuda_scratch((void**)&cuda_rt.proj[0],&cuda_rt.projcap[0],sizeof(float)))return -1;
        const void*input=cuda_rt.x;int input_f16=0;
        if(g->fmt==4&&cuda_rt.use_f16){input=cuda_rt.x16;input_f16=1;}
        if(cuda_qmat_launch_locked(cuda_rt.gate,input,g,input_f16)||
           cuda_qmat_launch_locked(cuda_rt.up,input,u,input_f16)||
           coli_cuda_silu_mul(cuda_rt.ctx,cuda_rt.gate,cuda_rt.gate,cuda_rt.up,I))return -1;
        const void*hidden=cuda_rt.gate;int hidden_f16=0;
        if(d->fmt==4&&cuda_rt.use_f16){
            if(cuda_scratch((void**)&cuda_rt.gate16,&cuda_rt.gate16cap,(size_t)I*sizeof(unsigned short))||
               coli_cuda_f32_to_f16(cuda_rt.ctx,cuda_rt.gate16,cuda_rt.gate,I))return -1;
            hidden=cuda_rt.gate16;hidden_f16=1;
        }
        if(cuda_qmat_launch_locked(cuda_rt.tmp,hidden,d,hidden_f16))return -1;
        input=scale->fmt==4&&cuda_rt.use_f16?(const void*)cuda_rt.x16:(const void*)cuda_rt.x;
        input_f16=scale->fmt==4&&cuda_rt.use_f16;
        if(cuda_qmat_launch_locked(cuda_rt.proj[0],input,scale,input_f16)||
           coli_cuda_sigmoid_axpy(cuda_rt.ctx,cuda_rt.y,cuda_rt.tmp,cuda_rt.proj[0],H))return -1;
    }
    cuda_rt.calls+=4;cuda_rt.mlp_calls++;return 1;
}
static int cuda_experts_try(float*y,const float*x,Expert*const*expert,const float*weight,int K,MoeW*moe,int*includes_shared){
    if(cuda_suppress)return 0;
    static int enabled=-1;if(enabled<0){const char*e=getenv("CUDA_GROUPED");enabled=!e||atoi(e)!=0;}
    *includes_shared=0;
    if(!enabled||!cuda_rt.active||K<=0)return 0;
    int fmt=expert[0]->gate.fmt,H=expert[0]->gate.I,I=expert[0]->gate.O;
    if((fmt!=1&&fmt!=2&&fmt!=3&&fmt!=4)||expert[0]->down.O!=H)return 0;
    for(int j=0;j<K;j++)if(!expert[j]->gate.cuda_eligible||!expert[j]->up.cuda_eligible||!expert[j]->down.cuda_eligible||expert[j]->gate.fmt!=fmt||expert[j]->up.fmt!=fmt||expert[j]->down.fmt!=fmt||expert[j]->gate.I!=H||expert[j]->gate.O!=I||expert[j]->up.I!=H||expert[j]->up.O!=I||expert[j]->down.I!=I||expert[j]->down.O!=H)return 0;
    int used=0;pthread_mutex_lock(&cuda_rt.lock);if(!cuda_rt.active)goto out;
    if(cuda_experts_upload_locked(expert,K)){
        cuda_backend_fail("grouped expert matrix upload");goto out;
    }
    if(cuda_scratch((void**)&cuda_rt.x,&cuda_rt.xcap,(size_t)H*sizeof(float))||
       cuda_scratch((void**)&cuda_rt.y,&cuda_rt.ycap,(size_t)H*sizeof(float))||
       cuda_scratch((void**)&cuda_rt.tmp,&cuda_rt.tmpcap,(size_t)H*sizeof(float))||
       cuda_scratch((void**)&cuda_rt.gate,&cuda_rt.gatecap,(size_t)I*sizeof(float))||
       cuda_scratch((void**)&cuda_rt.up,&cuda_rt.upcap,(size_t)I*sizeof(float))){cuda_backend_fail("grouped expert activation allocation");goto out;}
    if(coli_cuda_upload(cuda_rt.ctx,cuda_rt.x,x,(size_t)H*sizeof(float))||coli_cuda_memset(cuda_rt.ctx,cuda_rt.y,0,(size_t)H*sizeof(float))){cuda_backend_fail("grouped expert input");goto out;}
    const void*xin=cuda_rt.x;int x_f16=0;
    if(fmt!=1&&cuda_rt.use_f16){
        if(cuda_scratch((void**)&cuda_rt.x16,&cuda_rt.x16cap,(size_t)H*sizeof(unsigned short))||coli_cuda_f32_to_f16(cuda_rt.ctx,cuda_rt.x16,cuda_rt.x,H)){cuda_backend_fail("grouped expert fp16 input conversion");goto out;}
        xin=cuda_rt.x16;x_f16=1;
    }
    const char*gke=getenv("CUDA_GROUPED_KERNEL");
    if(fmt!=1&&x_f16&&(!gke||atoi(gke)!=0)){
        const unsigned char*gq[K],*uq[K],*dq[K];const float*gs[K],*us[K],*ds[K];
        for(int j=0;j<K;j++){gq[j]=(const unsigned char*)expert[j]->gate.d_q;gs[j]=(const float*)expert[j]->gate.d_s;uq[j]=(const unsigned char*)expert[j]->up.d_q;us[j]=(const float*)expert[j]->up.d_s;dq[j]=(const unsigned char*)expert[j]->down.d_q;ds[j]=(const float*)expert[j]->down.d_s;}
        QMat*g=&expert[0]->gate,*d=&expert[0]->down;
        int native_q3=fmt==3&&cuda_q3_native_enabled();
        int grouped_error=native_q3?
            coli_cuda_grouped_q3_mlp_f16(cuda_rt.ctx,cuda_rt.y,cuda_rt.x16,gq,gs,uq,us,dq,ds,weight,K,H,I,g->gs,g->rb,g->ng,d->rb,d->ng):
            coli_cuda_grouped_q4_mlp_f16(cuda_rt.ctx,cuda_rt.y,cuda_rt.x16,gq,gs,uq,us,dq,ds,weight,K,H,I,g->gs,g->drb,g->ng,d->drb,d->ng);
        if(grouped_error){cuda_backend_fail("grouped expert kernel");goto out;}if(native_q3)cuda_rt.q3_grouped_calls++;
        const char*sfe=getenv("CUDA_SHARED_FUSED");int allow_shared=!sfe||atoi(sfe)!=0;
        int sf=allow_shared?cuda_shared_add_locked(moe,H):0;if(sf<0){cuda_backend_fail("shared expert fusion");goto out;}*includes_shared=sf;
        if(coli_cuda_download(cuda_rt.ctx,y,cuda_rt.y,(size_t)H*sizeof(float))||coli_cuda_sync(cuda_rt.ctx)){cuda_backend_fail("grouped MoE result");goto out;}
        cuda_rt.calls+=(uint64_t)3*K;cuda_rt.mlp_calls+=K;cuda_rt.grouped_calls++;cuda_rt.grouped_kernel_calls++;used=1;goto out;
    }
    for(int j=0;j<K;j++){
        QMat*g=&expert[j]->gate,*u=&expert[j]->up,*d=&expert[j]->down;
        if(cuda_qmat_launch_locked(cuda_rt.gate,xin,g,x_f16)||cuda_qmat_launch_locked(cuda_rt.up,xin,u,x_f16)||coli_cuda_silu_mul(cuda_rt.ctx,cuda_rt.gate,cuda_rt.gate,cuda_rt.up,I)){cuda_backend_fail("grouped expert hidden projection");goto out;}
        const void*hidden=cuda_rt.gate;int hidden_f16=0;
        if(fmt!=1&&cuda_rt.use_f16){
            if(cuda_scratch((void**)&cuda_rt.gate16,&cuda_rt.gate16cap,(size_t)I*sizeof(unsigned short))||coli_cuda_f32_to_f16(cuda_rt.ctx,cuda_rt.gate16,cuda_rt.gate,I)){cuda_backend_fail("grouped expert fp16 hidden conversion");goto out;}
            hidden=cuda_rt.gate16;hidden_f16=1;
        }
        if(cuda_qmat_launch_locked(cuda_rt.tmp,hidden,d,hidden_f16)||coli_cuda_axpy(cuda_rt.ctx,cuda_rt.y,cuda_rt.tmp,weight[j],H)){cuda_backend_fail("grouped expert down projection");goto out;}
    }
    {const char*sfe=getenv("CUDA_SHARED_FUSED");int allow_shared=!sfe||atoi(sfe)!=0;int sf=allow_shared?cuda_shared_add_locked(moe,H):0;if(sf<0){cuda_backend_fail("shared expert fusion");goto out;}*includes_shared=sf;}
    if(coli_cuda_download(cuda_rt.ctx,y,cuda_rt.y,(size_t)H*sizeof(float))||coli_cuda_sync(cuda_rt.ctx)){cuda_backend_fail("grouped expert result");goto out;}
    cuda_rt.calls+=(uint64_t)3*K;cuda_rt.mlp_calls+=K;cuda_rt.grouped_calls++;used=1;
out:pthread_mutex_unlock(&cuda_rt.lock);return used;
}

static int cuda_experts_batch_try(float *y, const float *x,
                                  Expert *const *expert,
                                  const float *weight, int B, int K,
                                  MoeW *moe, int *includes_shared) {
    if (cuda_suppress || !cuda_rt.active || !cuda_rt.use_f16 || B < 2 ||
        B > 8 || K <= 0 || B * K > 64)
        return 0;
    int routes = B * K, H = expert[0]->gate.I, I = expert[0]->gate.O;
    QMat *sg = &moe->shared_gate, *su = &moe->shared_up;
    QMat *sd = &moe->shared_down, *ss = &moe->shared_scale;
    *includes_shared = 0;
    int routed_fmt = expert[0]->gate.fmt;
    if ((routed_fmt != 2 && routed_fmt != 3 && routed_fmt != 4) ||
        expert[0]->down.O != H)
        return 0;
    int batch_shared_q4 =
        sg->fmt == 4 && su->fmt == 4 && sd->fmt == 4 && ss->fmt == 4 &&
        sg->cuda_eligible && su->cuda_eligible && sd->cuda_eligible &&
        ss->cuda_eligible && sg->I == H && su->I == H &&
        su->O == sg->O && sd->I == sg->O && sd->O == H &&
        ss->I == H && ss->O == 1;
    int batch_shared_q8 =
        sg->fmt == 1 && su->fmt == 1 && sd->fmt == 1 && ss->fmt == 0 &&
        sg->cuda_eligible && su->cuda_eligible && sd->cuda_eligible &&
        sg->I == H && su->I == H && su->O == sg->O &&
        sd->I == sg->O && sd->O == H && ss->I == H && ss->O == 1;
    int batch_shared = batch_shared_q4 || batch_shared_q8;
    float shared_scale[8];
    if (batch_shared_q8) {
        for (int t = 0; t < B; t++) {
            float value = 0.0f;
            for (int h = 0; h < H; h++)
                value += x[(int64_t)t * H + h] * ss->f[h];
            shared_scale[t] = sigmoidf_stable(value);
        }
    }
    size_t active_bytes = 0;
    for (int r = 0; r < routes; r++) {
        Expert *e = expert[r];
        if (!e->gate.cuda_eligible || !e->up.cuda_eligible ||
            !e->down.cuda_eligible || e->gate.fmt != routed_fmt ||
            e->up.fmt != routed_fmt || e->down.fmt != routed_fmt ||
            e->gate.I != H || e->gate.O != I ||
            e->up.I != H || e->up.O != I || e->down.I != I ||
            e->down.O != H)
            return 0;
        int first = 1;
        for (int p = 0; p < r; p++)
            if (expert[p]->eid == e->eid) first = 0;
        if (first) {
            QMat *q[3] = {&e->gate, &e->up, &e->down};
            for (int j = 0; j < 3; j++)
                active_bytes += (size_t)q[j]->O * q[j]->drb +
                                (size_t)q[j]->O * q[j]->ng * sizeof(float);
        }
    }
    if (cuda_rt.expert_budget && active_bytes > cuda_rt.expert_budget)
        return 0;

    int used = 0;
    pthread_mutex_lock(&cuda_rt.lock);
    if (!cuda_rt.active) goto out;
    if (cuda_experts_upload_locked(expert, routes)) {
        cuda_backend_fail("batched expert matrix upload");
        goto out;
    }
    if (batch_shared &&
        (cuda_qmat_upload_locked(sg) || cuda_qmat_upload_locked(su) ||
         cuda_qmat_upload_locked(sd) ||
         (batch_shared_q4 && cuda_qmat_upload_locked(ss)))) {
        cuda_backend_fail("batched shared expert upload");
        goto out;
    }
    if (cuda_scratch((void **)&cuda_rt.x, &cuda_rt.xcap,
                     (size_t)B * H * sizeof(float)) ||
        cuda_scratch((void **)&cuda_rt.y, &cuda_rt.ycap,
                     (size_t)B * H * sizeof(float)) ||
        cuda_scratch((void **)&cuda_rt.x16, &cuda_rt.x16cap,
                     (size_t)B * H * sizeof(unsigned short))) {
        cuda_backend_fail("batched expert activation allocation");
        goto out;
    }
    if (coli_cuda_upload(cuda_rt.ctx, cuda_rt.x, x,
                         (size_t)B * H * sizeof(float)) ||
        coli_cuda_f32_to_f16(cuda_rt.ctx, cuda_rt.x16, cuda_rt.x, B * H)) {
        cuda_backend_fail("batched expert input conversion");
        goto out;
    }
    const unsigned char *gq[64], *uq[64], *dq[64];
    const float *gs[64], *us[64], *ds[64];
    for (int r = 0; r < routes; r++) {
        gq[r] = (const unsigned char *)expert[r]->gate.d_q;
        gs[r] = (const float *)expert[r]->gate.d_s;
        uq[r] = (const unsigned char *)expert[r]->up.d_q;
        us[r] = (const float *)expert[r]->up.d_s;
        dq[r] = (const unsigned char *)expert[r]->down.d_q;
        ds[r] = (const float *)expert[r]->down.d_s;
    }
    QMat *g = &expert[0]->gate, *d = &expert[0]->down;
    int native_q3 = g->fmt == 3 && cuda_q3_native_enabled();
    int grouped_error = native_q3 ?
        coli_cuda_grouped_q3_mlp_batch_f16(
            cuda_rt.ctx, cuda_rt.y, cuda_rt.x16, gq, gs, uq, us, dq, ds,
            weight, B, K, H, I, g->gs, g->rb, g->ng, d->rb, d->ng) :
        coli_cuda_grouped_q4_mlp_batch_f16(
            cuda_rt.ctx, cuda_rt.y, cuda_rt.x16, gq, gs, uq, us, dq, ds,
            weight, B, K, H, I, g->gs, g->drb, g->ng, d->drb, d->ng);
    if (grouped_error) {
        cuda_backend_fail("batched routed MoE execution");
        goto out;
    }if(native_q3)cuda_rt.q3_grouped_calls++;
    if (batch_shared_q4 &&
        coli_cuda_shared_q4_mlp_batch_f16(
            cuda_rt.ctx, cuda_rt.y, cuda_rt.x16,
            (const unsigned char *)sg->d_q, (const float *)sg->d_s,
            (const unsigned char *)su->d_q, (const float *)su->d_s,
            (const unsigned char *)sd->d_q, (const float *)sd->d_s,
            (const unsigned char *)ss->d_q, (const float *)ss->d_s, B, H,
            sg->O, sg->gs, sg->rb, sg->ng, sd->rb, sd->ng, ss->rb,
            ss->ng)) {
        cuda_backend_fail("batched shared MoE execution");
        goto out;
    }
    if (batch_shared_q8 &&
        coli_cuda_shared_q8_mlp_batch(
            cuda_rt.ctx, cuda_rt.y, cuda_rt.x, shared_scale,
            (const signed char *)sg->d_q, (const float *)sg->d_s, sg->rb,
            (const signed char *)su->d_q, (const float *)su->d_s, su->rb,
            (const signed char *)sd->d_q, (const float *)sd->d_s, sd->rb,
            B, H, sg->O)) {
        cuda_backend_fail("batched shared q8 MoE execution");
        goto out;
    }
    if (coli_cuda_download(cuda_rt.ctx, y, cuda_rt.y,
                           (size_t)B * H * sizeof(float)) ||
        coli_cuda_sync(cuda_rt.ctx)) {
        cuda_backend_fail("batched MoE execution");
        goto out;
    }
    cuda_rt.calls += (uint64_t)3 * routes +
                     (batch_shared ? (uint64_t)4 * B : 0);
    cuda_rt.mlp_calls += (uint64_t)routes +
                         (batch_shared ? (uint64_t)B : 0);
    cuda_rt.grouped_calls += (uint64_t)B;
    cuda_rt.grouped_kernel_calls++;
    *includes_shared = batch_shared;
    used = 1;
out:
    pthread_mutex_unlock(&cuda_rt.lock);
    return used;
}
#endif

static size_t telemetry_qmat_bytes(const QMat*q){
    if(!q)return 0;if(q->fmt==0)return(size_t)q->O*q->I*sizeof(float);
    size_t scales=(size_t)q->O*(q->fmt==1?1:q->ng)*sizeof(float);return(size_t)q->O*q->rb+scales;
}
static size_t telemetry_expert_bytes(const Expert*e){return telemetry_qmat_bytes(&e->gate)+telemetry_qmat_bytes(&e->up)+telemetry_qmat_bytes(&e->down);}
static void telemetry_hw_probe(char*cpu,size_t cap,int*cores,double*total,double*avail){
    cpu[0]=0;*cores=0;*total=*avail=0.;
#ifndef _WIN32
    FILE*f=fopen("/proc/cpuinfo","r");if(f){char line[512];while(fgets(line,sizeof(line),f))if(!strncmp(line,"model name",10)){char*p=strchr(line,':');if(p){while(*++p==' ');p[strcspn(p,"\r\n")]=0;snprintf(cpu,cap,"%s",p);}break;}fclose(f);}
    long n=sysconf(_SC_NPROCESSORS_ONLN);if(n>0)*cores=(int)n;f=fopen("/proc/meminfo","r");if(f){char line[256];double mt=0.,ma=0.;while(fgets(line,sizeof(line),f)){if(sscanf(line,"MemTotal: %lf",&mt)==1)*total=mt/1048576.;if(sscanf(line,"MemAvailable: %lf",&ma)==1)*avail=ma/1048576.;}fclose(f);}
#else
    SYSTEM_INFO si;GetSystemInfo(&si);*cores=(int)si.dwNumberOfProcessors;compat_meminfo(total,avail);
#endif
}
static void telemetry_hwinfo_emit(void){
    char cpu[256],gpu[128]="";int cores=0,ngpu=0;double total=0.,avail=0.,vram=0.;telemetry_hw_probe(cpu,sizeof(cpu),&cores,&total,&avail);
#ifdef COLI_CUDA
    if(cuda_rt.active){size_t free_bytes=0,total_bytes=0;ngpu=1;snprintf(gpu,sizeof(gpu),"%s",coli_cuda_device_name(cuda_rt.ctx));if(!coli_cuda_memory_info(cuda_rt.ctx,&free_bytes,&total_bytes))vram=(double)total_bytes/(1024.*1024.*1024.);}
#endif
    printf("HWINFO %d %.3f %.3f %d %.3f %s|%s\n",cores,total,avail,ngpu,vram,cpu,gpu);
}
static void telemetry_tiers_emit(Model*m){
    int ram=0,gpu=0,total=m->c.n_layers*m->c.n_experts;size_t rb=0;
    for(int li=0;li<m->c.n_layers;li++){MoeW*w=&m->layer[li].moe;
        for(int j=0;j<w->cap;j++)if(w->expert[j].eid>=0)
            rb+=telemetry_expert_bytes(&w->expert[j]);
        for(int eid=0;eid<m->c.n_experts;eid++){
#ifdef COLI_CUDA
            /*
             * The CUDA LRU is independent of the host Expert slots.  When a
             * host slot is reused, free_qmat() only detaches its QMat owner;
             * the device entry remains addressable by tensor name.  Inspect
             * that name cache first so device-only experts are not reported
             * as disk, and classify host/device duplicates at their fastest
             * tier.  rb remains the physical host-cache payload, including
             * duplicates, while the expert counts are disjoint.
             */
            if(cuda_has_expert(m,li,eid)){gpu++;continue;}
#endif
            for(int j=0;j<w->cap;j++)if(w->expert[j].eid==eid){ram++;break;}
        }
    }
    int disk=total-ram-gpu;if(disk<0)disk=0;double gb=0.;
#ifdef COLI_CUDA
    gb=(double)cuda_rt.expert_bytes/(1024.*1024.*1024.);
#endif
    printf("TIERS %d %d %d %.6f %.6f\n",gpu,ram,disk,gb,(double)rb/(1024.*1024.*1024.));
#ifdef COLI_CUDA
    printf("Q3NATIVE %llu %llu %llu %llu\n",
        (unsigned long long)cuda_rt.q3_uploads,
        (unsigned long long)cuda_rt.q3_upload_bytes,
        (unsigned long long)cuda_rt.q3_gemv_calls,
        (unsigned long long)cuda_rt.q3_grouped_calls);
    if(m->q3_route_atlas)printf("Q3ATLAS %d %llu %d %llu %d %d %llu %llu %llu\n",
        cuda_rt.q3_atlas_refreshes>0,
        (unsigned long long)cuda_rt.q3_atlas_refreshes,
        cuda_rt.q3_atlas_routes,
        (unsigned long long)m->q3_atlas_host_entries,
        cuda_rt.q3_atlas_entries,cuda_rt.q3_atlas_capacity,
        (unsigned long long)cuda_rt.q3_atlas_loads,
        (unsigned long long)cuda_rt.q3_atlas_bytes,
        (unsigned long long)m->q3_atlas_uncovered);
#endif
}
static void telemetry_emap_emit(Model*m){
    int rows=m->c.n_layers,cols=m->c.n_experts;size_t n=(size_t)rows*cols;char*hex=xcalloc(n*2+1,1);
    for(int li=0;li<rows;li++){MoeW*w=&m->layer[li].moe;for(int eid=0;eid<cols;eid++){int tier=0;
#ifdef COLI_CUDA
        if(cuda_has_expert(m,li,eid))tier=2;
#endif
        if(!tier)for(int j=0;j<w->cap;j++)if(w->expert[j].eid==eid){tier=1;break;}
        uint32_t u=w->heat? w->heat[eid]:0;int heat=0;while(u){heat++;u>>=1;}if(heat>63)heat=63;int v=(tier<<6)|heat;size_t off=((size_t)li*cols+eid)*2;hex[off]="0123456789abcdef"[v>>4];hex[off+1]="0123456789abcdef"[v&15];}}
    printf("EMAP %d %d %s\n",rows,cols,hex);free(hex);
}
static void telemetry_hits_emit(Model*m){
    int rows=m->c.n_layers,cols=m->c.n_experts;size_t bits=(size_t)rows*cols,nb=(bits+7)/8;unsigned char*bm=xcalloc(nb,1);
    if(m->turn_hits)for(size_t i=0;i<bits;i++)if(m->turn_hits[i])bm[i>>3]|=(unsigned char)(1u<<(i&7));
    char*hex=xcalloc(nb*2+1,1);for(size_t i=0;i<nb;i++){hex[i*2]="0123456789abcdef"[bm[i]>>4];hex[i*2+1]="0123456789abcdef"[bm[i]&15];}
    printf("HITS %d %d %s\n",rows,cols,hex);if(m->turn_hits)memset(m->turn_hits,0,bits);free(hex);free(bm);
}
static void telemetry_startup_emit(Model*m){telemetry_hwinfo_emit();telemetry_tiers_emit(m);telemetry_emap_emit(m);fflush(stdout);}
static void telemetry_turn_emit(Model*m,uint64_t id,double started,const double*base,
                                double decode_started,
                                const double*decode_base,
                                uint64_t cpu_hit_base,uint64_t gpu_hit_base,
                                uint64_t miss_base,uint64_t read_base,
                                uint64_t direct_base,uint64_t decode_cpu_hit_base,
                                uint64_t decode_gpu_hit_base,
                                uint64_t decode_miss_base,
                                uint64_t decode_read_base,
                                uint64_t decode_direct_base,
                                uint64_t cuda_tx_base,
                                const double*cuda_base,
                                uint64_t decode_cuda_tx_base,
                                const double*decode_cuda_base){
    double dt=now_s()-started,edisk=m->prof_expert_load-base[0],moe=m->prof_moe-base[1],sequence=(m->prof_gdn-base[2])+(m->prof_attn-base[3]),head=m->prof_lm-base[4];
    double emm=moe-edisk;if(emm<0.)emm=0.;
    printf("PERF %llu %.6f %.6f %.6f %.6f %.6f %.6f %.6f",(unsigned long long)id,dt,edisk,0.,emm,sequence,0.,head);
#ifdef COLI_CUDA
    unsigned long long cuda_tx=0;double cuda_ms[8]={0};
    if(cuda_rt.active&&!coli_cuda_profile_snapshot(cuda_rt.ctx,&cuda_tx,cuda_ms)){
        printf(" %llu",(unsigned long long)(cuda_tx-(unsigned long long)cuda_tx_base));
        for(int i=0;i<8;i++)printf(" %.6f",(cuda_ms[i]-cuda_base[i])/1000.);
    }
    printf("\nRESIDENT %llu %llu %llu %llu %llu %llu %llu %llu",
        (unsigned long long)id,
        (unsigned long long)cuda_rt.resident_layers,
        (unsigned long long)cuda_rt.resident_device_moe,
        (unsigned long long)cuda_rt.resident_host_moe,
        (unsigned long long)cuda_rt.resident_h2d,
        (unsigned long long)cuda_rt.resident_d2h,
        (unsigned long long)cuda_rt.resident_logits_d2h,
        (unsigned long long)cuda_rt.resident_router_d2h);
#else
    (void)cuda_tx_base;(void)cuda_base;
#endif
    putchar('\n');
    double ddt=now_s()-decode_started;
    double dedisk=m->prof_expert_load-decode_base[0];
    double dmoe=m->prof_moe-decode_base[1];
    double dsequence=(m->prof_gdn-decode_base[2])+
                     (m->prof_attn-decode_base[3]);
    double dhead=m->prof_lm-decode_base[4],demm=dmoe-dedisk;
    if(demm<0.)demm=0.;
    printf("DPERF %llu %.6f %.6f %.6f %.6f %.6f %.6f %.6f",
           (unsigned long long)id,ddt,dedisk,0.,demm,dsequence,0.,dhead);
#ifdef COLI_CUDA
    unsigned long long dcuda_tx=0;double dcuda_ms[8]={0};
    if(cuda_rt.active&&!coli_cuda_profile_snapshot(cuda_rt.ctx,&dcuda_tx,
                                                   dcuda_ms)){
        printf(" %llu",(unsigned long long)(
            dcuda_tx-(unsigned long long)decode_cuda_tx_base));
        for(int i=0;i<8;i++)
            printf(" %.6f",(dcuda_ms[i]-decode_cuda_base[i])/1000.);
    }
#else
    (void)decode_cuda_tx_base;(void)decode_cuda_base;
#endif
    putchar('\n');
    uint64_t cpu_hits=m->tier_hits-cpu_hit_base;
    uint64_t gpu_hits=m->tier_gpu_hits-gpu_hit_base;
    uint64_t misses=m->tier_misses-miss_base;
    uint64_t read_bytes=m->S.read_bytes-read_base;
    uint64_t direct_bytes=m->S.direct_bytes-direct_base;
    printf("CACHE %llu %llu %llu %llu %llu %llu\n",
           (unsigned long long)id,
           (unsigned long long)cpu_hits,
           (unsigned long long)gpu_hits,
           (unsigned long long)misses,
           (unsigned long long)read_bytes,
           (unsigned long long)direct_bytes);
    printf("DCACHE %llu %llu %llu %llu %llu %llu\n",
           (unsigned long long)id,
           (unsigned long long)(m->tier_hits-decode_cpu_hit_base),
           (unsigned long long)(m->tier_gpu_hits-decode_gpu_hit_base),
           (unsigned long long)(m->tier_misses-decode_miss_base),
           (unsigned long long)(m->S.read_bytes-decode_read_base),
           (unsigned long long)(m->S.direct_bytes-decode_direct_base));
    telemetry_tiers_emit(m);telemetry_emap_emit(m);telemetry_hits_emit(m);fflush(stdout);
}

static float dot(const float *a,const float *b,int n){ float s=0.f; for(int i=0;i<n;i++) s+=a[i]*b[i]; return s; }
static int parallel_work(int64_t work){
    static int64_t threshold=-1;if(threshold<0){const char *e=getenv("OMP_MIN_WORK");threshold=e?atoll(e):262144;}
#ifdef _OPENMP
    return work>=threshold&&!omp_in_parallel();
#else
    (void)work;return 0;
#endif
}
#if defined(__AVX2__)
static inline float hsum256f(__m256 v){__m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);lo=_mm_add_ps(lo,hi);lo=_mm_hadd_ps(lo,lo);lo=_mm_hadd_ps(lo,lo);return _mm_cvtss_f32(lo);}
static inline int hsum256_i32(__m256i v){__m128i lo=_mm256_castsi256_si128(v),hi=_mm256_extracti128_si256(v,1);lo=_mm_add_epi32(lo,hi);lo=_mm_hadd_epi32(lo,lo);lo=_mm_hadd_epi32(lo,lo);return _mm_cvtsi128_si32(lo);}
#endif
static float dot_q8f(const int8_t *w,const float *x,int n){
    float sum=0.f;int i=0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512 acc=_mm512_setzero_ps();for(;i+16<=n;i+=16){__m128i b=_mm_loadu_si128((const __m128i*)(w+i));__m512 wf=_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b));acc=_mm512_fmadd_ps(_mm512_loadu_ps(x+i),wf,acc);}sum=_mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
    __m256 acc=_mm256_setzero_ps();for(;i+8<=n;i+=8){__m128i b=_mm_loadl_epi64((const __m128i*)(w+i));__m256 wf=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),wf,acc);}sum=hsum256f(acc);
#endif
    for(;i<n;i++)sum+=x[i]*(float)w[i];return sum;
}
static float qrow_i8(const float*x,int8_t*q,int n){float amax=0.f;for(int i=0;i<n;i++){float a=fabsf(x[i]);if(a>amax)amax=a;}float s=amax/127.f;if(s<1e-12f)s=1e-12f;float inv=1.f/s;for(int i=0;i<n;i++)q[i]=(int8_t)lrintf(x[i]*inv);return s;}
static int32_t dot_i8i8(const int8_t*w,const int8_t*x,int n){
    int32_t sum=0;int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    __m512i acc=_mm512_setzero_si512();for(;i+64<=n;i+=64){__m512i wv=_mm512_loadu_si512((const void*)(w+i)),xv=_mm512_loadu_si512((const void*)(x+i));__mmask64 neg=_mm512_movepi8_mask(wv);__m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);}sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVX2__)
    __m256i acc=_mm256_setzero_si256(),ones=_mm256_set1_epi16(1);for(;i+32<=n;i+=32){__m256i wv=_mm256_loadu_si256((const __m256i*)(w+i)),xv=_mm256_loadu_si256((const __m256i*)(x+i));__m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));}sum=hsum256_i32(acc);
#endif
    for(;i<n;i++)sum+=(int32_t)w[i]*x[i];return sum;
}
static float dot_q4_group(const uint8_t *w,const float *x,int n){
    float sum=0.f;int i=0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m128i mask=_mm_set1_epi8(15);const __m512i eight=_mm512_set1_epi32(8);__m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps();
    for(;i+32<=n;i+=32){__m128i by=_mm_loadu_si128((const __m128i*)(w+(i>>1)));__m128i lo=_mm_and_si128(by,mask),hi=_mm_and_si128(_mm_srli_epi16(by,4),mask);__m128i n0=_mm_unpacklo_epi8(lo,hi),n1=_mm_unpackhi_epi8(lo,hi);__m512 f0=_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n0),eight));__m512 f1=_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n1),eight));a0=_mm512_fmadd_ps(_mm512_loadu_ps(x+i),f0,a0);a1=_mm512_fmadd_ps(_mm512_loadu_ps(x+i+16),f1,a1);}sum=_mm512_reduce_add_ps(_mm512_add_ps(a0,a1));
#elif defined(__AVX2__)
    const __m128i mask=_mm_set1_epi8(15);const __m256i eight=_mm256_set1_epi32(8);__m256 acc=_mm256_setzero_ps();
    for(;i+16<=n;i+=16){__m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));__m128i lo=_mm_and_si128(by,mask),hi=_mm_and_si128(_mm_srli_epi16(by,4),mask),ni=_mm_unpacklo_epi8(lo,hi);__m256 f0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(ni),eight));__m256 f1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(ni,8)),eight));acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),f0,acc);acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),f1,acc);}sum=hsum256f(acc);
#endif
    for(;i<n;i++){uint8_t b=w[i>>1];sum+=x[i]*(float)(((i&1)?(b>>4):(b&15))-8);}return sum;
}
static float dot_q2_group(const uint8_t*w,const float*x,int n){
    float sum=0.f;
    for(int i=0;i<n;i++){
        uint8_t code=(uint8_t)((w[i>>2]>>(2*(i&3)))&3);
        sum+=x[i]*(float)(2*(int)code-3);
    }
    return sum;
}
static int q3_value(const uint8_t*w,int i){
    int bit=3*i,byte=bit>>3,shift=bit&7;uint16_t word=w[byte];
    if(shift>5)word|=(uint16_t)w[byte+1]<<8;
    return (int)((word>>shift)&7)-4;
}
static float dot_q3_group(const uint8_t*w,const float*x,int n){
    pthread_once(&q3_lut_once,q3_lut_init);
    float sum=0.f;int i=0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512 acc=_mm512_setzero_ps();
    for(;i+16<=n;i+=16){
        uint64_t a=q3_unpack8_i8(q3_word24(w+(i*3/8)));
        uint64_t b=q3_unpack8_i8(q3_word24(w+(i*3/8)+3));
        __m128i q=_mm_set_epi64x((long long)b,(long long)a);
        __m512 qf=_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q));
        acc=_mm512_fmadd_ps(_mm512_loadu_ps(x+i),qf,acc);
    }
    sum=_mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
    __m256 acc=_mm256_setzero_ps();
    for(;i+8<=n;i+=8){
        uint64_t packed=q3_unpack8_i8(q3_word24(w+(i*3/8)));
        __m128i q=_mm_cvtsi64_si128((long long)packed);
        __m256 qf=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
        acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),qf,acc);
    }
    sum=hsum256f(acc);
#endif
    for(;i<n;i++)sum+=x[i]*(float)q3_value(w,i);return sum;
}
static float qmat_dot_row(const QMat *w,int row,const float *x){
    if(w->fmt==0)return dot(x,w->f+(int64_t)row*w->I,w->I);
    if(w->fmt==1)return dot_q8f(w->q8+(int64_t)row*w->rb,x,w->I)*w->s[row];
    const uint8_t *q=w->q4+(int64_t)row*w->rb;const float *s=w->s+(int64_t)row*w->ng;float out=0.f;
    for(int g=0;g<w->ng;g++){int base=g*w->gs,n=w->gs;if(base+n>w->I)n=w->I-base;if(n>0){float v=w->fmt==2?dot_q2_group(q+(base>>2),x+base,n):(w->fmt==3?dot_q3_group(q+(base*3/8),x+base,n):dot_q4_group(q+(base>>1),x+base,n));out+=v*s[g];}}return out;
}
static void qmat_mul_ex(float *y,const float *x,const QMat *w,int allow_idot){
#ifdef COLI_CUDA
    if(cuda_qmat_try(y,x,w))return;
#endif
    static int idot=-1;if(idot<0){const char*e=getenv("IDOT");idot=e?atoi(e)!=0:1;}
    if(w->fmt==1&&idot&&allow_idot){static _Thread_local int8_t*qbuf;static _Thread_local int cap;if(cap<w->I){int8_t*p=realloc(qbuf,(size_t)w->I);if(!p)die("OOM idot activation");qbuf=p;cap=w->I;}float sx=qrow_i8(x,qbuf,w->I);int8_t*shared_qbuf=qbuf;int par=parallel_work((int64_t)w->O*w->I);
        #pragma omp parallel for schedule(static) if(par)
        for(int o=0;o<w->O;o++)y[o]=(float)dot_i8i8(w->q8+(int64_t)o*w->rb,shared_qbuf,w->I)*w->s[o]*sx;return;
    }
    int par=parallel_work((int64_t)w->O*w->I);
    #pragma omp parallel for schedule(static) if(par)
    for(int o=0;o<w->O;o++)y[o]=qmat_dot_row(w,o,x);
}
static void qmat_mul(float*y,const float*x,const QMat*w){qmat_mul_ex(y,x,w,1);}
static void qmat_row(float *out,const QMat *w,int row){
    if(row<0||row>=w->O)die("matrix row out of range");if(w->fmt==0){memcpy(out,w->f+(int64_t)row*w->I,(size_t)w->I*sizeof(float));return;}
    if(w->fmt==1){const int8_t*q=w->q8+(int64_t)row*w->rb;for(int i=0;i<w->I;i++)out[i]=(float)q[i]*w->s[row];return;}
    const uint8_t*q=w->q4+(int64_t)row*w->rb;const float*s=w->s+(int64_t)row*w->ng;for(int i=0;i<w->I;i++){int v;if(w->fmt==2){uint8_t b=q[i>>2];v=2*((b>>(2*(i&3)))&3)-3;}else if(w->fmt==3)v=q3_value(q,i);else{uint8_t b=q[i>>1];v=((i&1)?(b>>4):(b&15))-8;}out[i]=(float)v*s[i/w->gs];}
}
static void rmsnorm_zero(float *out,const float *x,const float *w,int n,float eps){
    float ms=0.f; for(int i=0;i<n;i++) ms += x[i]*x[i];
    float r=1.f/sqrtf(ms/(float)n+eps);
    for(int i=0;i<n;i++) out[i]=x[i]*r*(1.f+w[i]);
}

/* ---------- config ---------- */
static char *read_file(const char *path,long *n_out){
    FILE *f=fopen(path,"rb"); if(!f){ perror(path); exit(1); }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=xcalloc((size_t)n+1,1); if(fread(b,1,n,f)!=(size_t)n) die("short read"); fclose(f); if(n_out)*n_out=n; return b;
}
static int jint(jval *o,const char *k,int d){ jval *v=json_get(o,k); return v&&v->t==J_NUM?(int)v->num:d; }
static float jnum(jval *o,const char *k,float d){ jval *v=json_get(o,k); return v&&v->t==J_NUM?(float)v->num:d; }
static int cfg_is_eos(const Cfg*c,int token){return token==c->eos_token||(c->eos_token2>=0&&token==c->eos_token2);}
static void load_cfg(Cfg *c,const char *snap){
    memset(c,0,sizeof(*c));c->eos_token=c->eos_token2=c->pad_token=-1; char path[2048]; snprintf(path,sizeof(path),"%s/config.json",snap);
    long n; char *buf=read_file(path,&n),*arena=NULL; jval *root=json_parse(buf,&arena); (void)n;
    jval *tc=json_get(root,"text_config"); if(!tc||tc->t!=J_OBJ) tc=root;
    c->hidden=jint(tc,"hidden_size",0); c->n_layers=jint(tc,"num_hidden_layers",0);
    c->vocab=jint(tc,"vocab_size",0); c->max_position=jint(tc,"max_position_embeddings",32768);
    c->eos_token=jint(tc,"eos_token_id",-1);c->pad_token=jint(tc,"pad_token_id",c->eos_token);
    c->eos_token=jint(root,"eos_token_id",c->eos_token);c->pad_token=jint(root,"pad_token_id",c->pad_token);
    c->n_heads=jint(tc,"num_attention_heads",0); c->n_kv_heads=jint(tc,"num_key_value_heads",0);
    c->head_dim=jint(tc,"head_dim",c->n_heads?c->hidden/c->n_heads:0); c->eps=jnum(tc,"rms_norm_eps",1e-6f);
    c->n_experts=jint(tc,"num_experts",0); c->topk=jint(tc,"num_experts_per_tok",0);
    c->moe_inter=jint(tc,"moe_intermediate_size",0); c->shared_inter=jint(tc,"shared_expert_intermediate_size",0);
    c->norm_topk=1;
    c->lin_k_heads=jint(tc,"linear_num_key_heads",0); c->lin_v_heads=jint(tc,"linear_num_value_heads",0);
    c->lin_k_dim=jint(tc,"linear_key_head_dim",0); c->lin_v_dim=jint(tc,"linear_value_head_dim",0);
    c->conv_kernel=jint(tc,"linear_conv_kernel_dim",4); c->mtp_layers=jint(tc,"mtp_num_hidden_layers",0);
    c->full_interval=jint(tc,"full_attention_interval",4);
    jval *rp=json_get(tc,"rope_parameters"); if(!rp||rp->t!=J_OBJ) rp=tc;
    c->theta=jnum(rp,"rope_theta",10000000.f); c->partial_rotary=jnum(rp,"partial_rotary_factor",0.25f);
    if(c->n_layers<=0||c->n_layers>QW_MAX_LAYERS||c->hidden<=0||c->vocab<=0) die("invalid Qwen config");
    if(c->topk<=0||c->topk>QW_MAX_TOPK||c->topk>c->n_experts)die("invalid routed expert top-k");
    if(c->lin_v_heads%c->lin_k_heads) die("linear value heads must be divisible by key heads");
    jval *lt=json_get(tc,"layer_types");
    if(lt&&lt->t==J_ARR&&lt->len==c->n_layers){
        for(int i=0;i<c->n_layers;i++) c->layer_type[i]=(lt->kids[i]->t==J_STR&&!strcmp(lt->kids[i]->str,"full_attention"))?LT_FULL:LT_LINEAR;
    }else for(int i=0;i<c->n_layers;i++) c->layer_type[i]=((i+1)%c->full_interval==0)?LT_FULL:LT_LINEAR;
    free(buf); free(arena);
    snprintf(path,sizeof(path),"%s/generation_config.json",snap);FILE*gf=fopen(path,"rb");
    if(gf){fclose(gf);char*gb=read_file(path,&n),*ga=NULL;jval*gr=json_parse(gb,&ga);jval*ev=json_get(gr,"eos_token_id");
        if(ev&&ev->t==J_NUM){c->eos_token=(int)ev->num;c->eos_token2=-1;}
        else if(ev&&ev->t==J_ARR&&ev->len){c->eos_token=(int)ev->kids[0]->num;c->eos_token2=ev->len>1?(int)ev->kids[1]->num:-1;}
        c->pad_token=jint(gr,"pad_token_id",c->pad_token);free(gb);free(ga);}
}

/* ---------- tensor loading; quantized matrices stay packed ---------- */
static int64_t find_named(shards *S,const char *base,char *out,size_t outsz){
    const char *pre[]={"","model.","model.language_model."};
    for(size_t i=0;i<sizeof(pre)/sizeof(pre[0]);i++){ snprintf(out,outsz,"%s%s",pre[i],base); int64_t n=st_numel(S,out); if(n>=0)return n; }
    return -1;
}
static int q3_expert_enabled(const char*base){
    const char*e3=getenv("EXPERT_Q3");
    if(!e3||atoi(e3)==0||!strstr(base,".experts."))return 0;
    const char*p=strstr(base,"layers.");
    if(!p)return 0;
    p+=7;char*layer_end=NULL;long layer=strtol(p,&layer_end,10);
    if(layer_end==p||layer<0||layer>=QW_MAX_LAYERS||*layer_end!='.')
        die("cannot parse expert layer for int3 selection");
    const char*minimum=getenv("EXPERT_Q3_MIN_LAYER");
    if(minimum&&*minimum){
        char*end=NULL;long min_layer=strtol(minimum,&end,10);
        if(end==minimum||*end||min_layer<0||min_layer>=QW_MAX_LAYERS)
            die("EXPERT_Q3_MIN_LAYER is outside the supported layer range");
        if(layer<min_layer)return 0;
    }
    const char*limit=getenv("EXPERT_Q3_MAX_LAYER");
    if(limit&&*limit){
        char*end=NULL;long max_layer=strtol(limit,&end,10);
        if(end==limit||*end||max_layer<-1||max_layer>=QW_MAX_LAYERS)
            die("EXPERT_Q3_MAX_LAYER is outside the supported layer range");
        if(layer>max_layer)return 0;
    }
    return 1;
}
static int64_t find_qmat_named(Model*m,const char*base,char*out,size_t outsz){
    if(q3_expert_enabled(base)){
        char q3base[QW_NAME];int n=snprintf(q3base,sizeof(q3base),"%s.q3",base);
        if(n<0||(size_t)n>=sizeof(q3base))die("int3 expert tensor name too long");
        int64_t count=find_named(&m->S,q3base,out,outsz);
        if(count>=0)return count;
        fprintf(stderr,"missing selected int3 expert tensor %s\n",q3base);
        exit(1);
    }
    const char*e=getenv("EXPERT_Q2");
    if(e&&atoi(e)!=0&&strstr(base,".experts.")){
        char q2base[QW_NAME];int n=snprintf(q2base,sizeof(q2base),"%s.q2",base);
        if(n<0||(size_t)n>=sizeof(q2base))die("int2 expert tensor name too long");
        int64_t count=find_named(&m->S,q2base,out,outsz);
        if(count>=0)return count;
    }
    return find_named(&m->S,base,out,outsz);
}
static float *load_vec(Model *m,const char *base,int n){
    char name[QW_NAME]; if(find_named(&m->S,base,name,sizeof(name))<0){ fprintf(stderr,"missing tensor %s\n",base); exit(1); }
    st_tensor *t=st_find(&m->S,name); if(t->dtype==3||t->numel!=n){fprintf(stderr,"bad vector %s\n",name);exit(1);}
    float *out=falloc(n);st_read_f32(&m->S,name,out,0);return out;
}
static void*map_tensor(Model*m,const char*name,void**map_base,size_t*map_len){
    st_tensor*t=st_find(&m->S,name);if(!t)die("cannot mmap missing tensor");long page=sysconf(_SC_PAGESIZE);int64_t off=t->off&~((int64_t)page-1),delta=t->off-off;size_t len=(size_t)(delta+t->nbytes);void*p=mmap(NULL,len,PROT_READ,MAP_PRIVATE,t->fd,off);if(p==MAP_FAILED){perror("mmap tensor");exit(1);}*map_base=p;*map_len=len;return(char*)p+delta;
}
static QMat load_qmat_mode(Model *m,const char *base,int rows,int cols,int use_mmap){
    char name[QW_NAME];if(find_qmat_named(m,base,name,sizeof(name))<0){fprintf(stderr,"missing tensor %s\n",base);exit(1);}
    st_tensor *t=st_find(&m->S,name);QMat q={.O=rows,.I=cols};int64_t expected=(int64_t)rows*cols;
#ifdef COLI_CUDA
    q.cuda_eligible=strstr(base,".experts.")==NULL||(getenv("CUDA_EXPERTS")&&atoi(getenv("CUDA_EXPERTS"))!=0);
    snprintf(q.cuda_key,sizeof(q.cuda_key),"%s",base);
#endif
    if(t->dtype!=3){
        if(t->numel!=expected){fprintf(stderr,"shape mismatch %s: got %lld expected %lld\n",name,(long long)t->numel,(long long)expected);exit(1);}
        q.f=falloc(expected);st_read_f32(&m->S,name,q.f,0);m->matrix_f32++;return q;
    }
    char qsname[QW_NAME+8]; snprintf(qsname,sizeof(qsname),"%s.qs",name);
    int64_t ns=st_numel(&m->S,qsname); if(ns<0) die("quantized tensor missing .qs scales");
    st_tensor*qst=st_find(&m->S,qsname);if(use_mmap){if(!qst||qst->dtype!=2)die("mmap scales must be f32");q.s=map_tensor(m,qsname,&q.map_s,&q.map_s_len);}else{q.s=falloc(ns);st_read_f32(&m->S,qsname,q.s,0);}
    char qtname[QW_NAME+8];snprintf(qtname,sizeof(qtname),"%s.qtype",name);int qtype=0;if(st_numel(&m->S,qtname)==1){uint8_t tag=0;st_read_raw(&m->S,qtname,&tag,0);qtype=tag;}
    /* The tiny all-int4 fixture has 64-column matrices whose padded g128 row
     * is also 64 bytes, making size alone ambiguous with int8. Its embedding
     * declares the snapshot-wide mode; real mixed-model int4 inputs are >=512. */
    if(qtype==8||(qtype==0&&m->quant_mode!=4&&t->nbytes==expected&&ns==rows)){
        q.fmt=1;q.rb=q.drb=cols;if(use_mmap)q.q8=map_tensor(m,name,&q.map_q,&q.map_q_len);else{q.q8=xcalloc((size_t)t->nbytes,1);st_read_raw(&m->S,name,q.q8,0);}m->matrix_i8++;
    }else if(qtype==2){
        if(ns%rows||t->nbytes%rows)die("bad int2 matrix payload");q.fmt=2;q.ng=(int)(ns/rows);q.rb=(int)(t->nbytes/rows);q.drb=q.rb*2;q.gs=(q.rb*4)/q.ng;
        if(q.gs<=0||q.ng<=0||q.gs%4)die("bad int2 group geometry");if(use_mmap)q.q4=map_tensor(m,name,&q.map_q,&q.map_q_len);else{q.q4=xcalloc((size_t)t->nbytes,1);st_read_raw(&m->S,name,q.q4,0);}m->matrix_i2++;
    }else if(qtype==3){
        if(ns%rows||t->nbytes%rows)die("bad int3 matrix payload");q.fmt=3;q.ng=(int)(ns/rows);q.rb=(int)(t->nbytes/rows);
        if(q.rb%3)die("bad int3 packed row");q.drb=(q.rb/3)*4;q.gs=((q.rb/3)*8)/q.ng;
        if(q.gs<=0||q.ng<=0||q.gs%8)die("bad int3 group geometry");if(use_mmap)q.q4=map_tensor(m,name,&q.map_q,&q.map_q_len);else{q.q4=xcalloc((size_t)t->nbytes,1);st_read_raw(&m->S,name,q.q4,0);}m->matrix_i3++;
    }else{
        if(qtype&&qtype!=4)die("unknown matrix qtype tag");
        if(ns%rows||t->nbytes%rows)die("bad int4 matrix payload");q.fmt=4;q.ng=(int)(ns/rows);q.rb=q.drb=(int)(t->nbytes/rows);q.gs=(q.rb*2)/q.ng;
        if(q.gs<=0||q.ng<=0)die("bad int4 group geometry");if(use_mmap)q.q4=map_tensor(m,name,&q.map_q,&q.map_q_len);else{q.q4=xcalloc((size_t)t->nbytes,1);st_read_raw(&m->S,name,q.q4,0);}m->matrix_i4++;
    }
    return q;
}
static QMat load_qmat(Model*m,const char*base,int rows,int cols){return load_qmat_mode(m,base,rows,cols,0);}
static void prepare_quant_qmat(Model*m,const char*base,int rows,int cols,QMat*q,
                               char*name,size_t name_cap,char*qsname,size_t qs_cap,
                               void**payload){
    if(find_qmat_named(m,base,name,name_cap)<0){fprintf(stderr,"missing tensor %s\n",base);exit(1);}
    st_tensor*t=st_find(&m->S,name);int64_t expected=(int64_t)rows*cols;
    if(!t||t->dtype!=3){fprintf(stderr,"bad quantized matrix %s\n",name);exit(1);}
    memset(q,0,sizeof(*q));q->O=rows;q->I=cols;
#ifdef COLI_CUDA
    q->cuda_eligible=strstr(base,".experts.")==NULL||(getenv("CUDA_EXPERTS")&&atoi(getenv("CUDA_EXPERTS"))!=0);
    snprintf(q->cuda_key,sizeof(q->cuda_key),"%s",base);
#endif
    snprintf(qsname,qs_cap,"%s.qs",name);int64_t ns=st_numel(&m->S,qsname);
    st_tensor*qst=st_find(&m->S,qsname);if(ns<0||!qst||qst->dtype!=2)die("quantized tensor missing f32 .qs scales");
    q->s=falloc(ns);
    char qtname[QW_NAME+8];snprintf(qtname,sizeof(qtname),"%s.qtype",name);int qtype=0;
    if(st_numel(&m->S,qtname)==1){uint8_t tag=0;st_read_raw(&m->S,qtname,&tag,0);qtype=tag;}
    if(qtype==8||(qtype==0&&m->quant_mode!=4&&t->nbytes==expected&&ns==rows)){
        q->fmt=1;q->rb=q->drb=cols;q->q8=xcalloc((size_t)t->nbytes,1);*payload=q->q8;m->matrix_i8++;
    }else if(qtype==2){
        if(ns%rows||t->nbytes%rows)die("bad int2 matrix payload");
        q->fmt=2;q->ng=(int)(ns/rows);q->rb=(int)(t->nbytes/rows);q->drb=q->rb*2;q->gs=(q->rb*4)/q->ng;
        if(q->gs<=0||q->ng<=0||q->gs%4)die("bad int2 group geometry");
        q->q4=xcalloc((size_t)t->nbytes,1);*payload=q->q4;m->matrix_i2++;
    }else if(qtype==3){
        if(ns%rows||t->nbytes%rows)die("bad int3 matrix payload");
        q->fmt=3;q->ng=(int)(ns/rows);q->rb=(int)(t->nbytes/rows);
        if(q->rb%3)die("bad int3 packed row");q->drb=(q->rb/3)*4;q->gs=((q->rb/3)*8)/q->ng;
        if(q->gs<=0||q->ng<=0||q->gs%8)die("bad int3 group geometry");
        q->q4=xcalloc((size_t)t->nbytes,1);*payload=q->q4;m->matrix_i3++;
    }else{
        if(qtype&&qtype!=4)die("unknown matrix qtype tag");
        if(ns%rows||t->nbytes%rows)die("bad int4 matrix payload");
        q->fmt=4;q->ng=(int)(ns/rows);q->rb=q->drb=(int)(t->nbytes/rows);q->gs=(q->rb*2)/q->ng;
        if(q->gs<=0||q->ng<=0)die("bad int4 group geometry");
        q->q4=xcalloc((size_t)t->nbytes,1);*payload=q->q4;m->matrix_i4++;
    }
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static int qmat_source_is_quantized(Model*m,const char*base){
    char found[QW_NAME];if(find_qmat_named(m,base,found,sizeof(found))<0)return 0;
    st_tensor*t=st_find(&m->S,found);return t&&t->dtype==3;
}
static void free_qmat(QMat*q){
#ifdef COLI_CUDA
    if(cuda_rt.ctx){pthread_mutex_lock(&cuda_rt.lock);cuda_qmat_detach(q);pthread_mutex_unlock(&cuda_rt.lock);}
#endif
    free(q->f);if(q->map_s)munmap(q->map_s,q->map_s_len);else free(q->s);if(q->map_q)munmap(q->map_q,q->map_q_len);else{free(q->q8);free(q->q4);}memset(q,0,sizeof(*q));}
static void lname(Model*m,char *out,size_t cap,int layer,const char *suffix){
    if(layer==m->c.n_layers)snprintf(out,cap,"mtp.layers.0.%s",suffix);
    else snprintf(out,cap,"layers.%d.%s",layer,suffix);
}

static void expert_prefetch_load_set(Model*m,MoeW*w,const int*eid,int n);
static void *expert_prefetch_worker(void*arg){
    Model*m=arg;for(;;){int li,n,load,eid[QW_MAX_TOPK];pthread_mutex_lock(&m->pf_lock);while(m->pf_head==m->pf_tail)pthread_cond_wait(&m->pf_cond,&m->pf_lock);li=m->pf_job[m->pf_head].layer;n=m->pf_job[m->pf_head].n;load=m->pf_job[m->pf_head].load;memcpy(eid,m->pf_job[m->pf_head].eid,(size_t)n*sizeof(int));m->pf_head=(m->pf_head+1)&127;pthread_mutex_unlock(&m->pf_lock);
        if(load){MoeW*w=li==m->c.n_layers?&m->mtp.layer.moe:&m->layer[li].moe;expert_prefetch_load_set(m,w,eid,n);pthread_mutex_lock(&w->lock);w->pf_pending=0;pthread_cond_broadcast(&w->pf_done);pthread_mutex_unlock(&w->lock);continue;}
#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
        const char*suf[]={"gate_proj","up_proj","down_proj"};for(int i=0;i<n;i++)for(int k=0;k<3;k++){char base[QW_NAME],name[QW_NAME],tail[160];snprintf(tail,sizeof(tail),"mlp.experts.%d.%s.weight",eid[i],suf[k]);lname(m,base,sizeof(base),li,tail);if(find_qmat_named(m,base,name,sizeof(name))>=0){st_tensor*t=st_find(&m->S,name);if(t)posix_fadvise(t->fd,t->off,t->nbytes,POSIX_FADV_WILLNEED);char qs[QW_NAME+8];snprintf(qs,sizeof(qs),"%s.qs",name);t=st_find(&m->S,qs);if(t)posix_fadvise(t->fd,t->off,t->nbytes,POSIX_FADV_WILLNEED);}}
#else
        (void)li;(void)eid;
#endif
    }return NULL;
}
static void expert_prefetch_start(Model*m){int n=getenv("PREFETCH_THREADS")?atoi(getenv("PREFETCH_THREADS")):1;if(n<0)n=0;if(n>4)n=4;m->pf_nthread=n;if(!n)return;pthread_mutex_init(&m->pf_lock,NULL);pthread_cond_init(&m->pf_cond,NULL);for(int i=0;i<n;i++)pthread_create(&m->pf_thread[i],NULL,expert_prefetch_worker,m);}
static int expert_prefetch_submit_set(Model*m,int li,const int*eid,int n,int load){
    if(!m->pf_nthread||n<=0)return 0;if(n>QW_MAX_TOPK)n=QW_MAX_TOPK;int queued=0;
    pthread_mutex_lock(&m->pf_lock);int next=(m->pf_tail+1)&127;
    if(next!=m->pf_head){m->pf_job[m->pf_tail].layer=li;m->pf_job[m->pf_tail].n=n;m->pf_job[m->pf_tail].load=load;memcpy(m->pf_job[m->pf_tail].eid,eid,(size_t)n*sizeof(int));m->pf_tail=next;pthread_cond_signal(&m->pf_cond);queued=1;}
    else __atomic_fetch_add(&m->pf_dropped,1,__ATOMIC_RELAXED);
    pthread_mutex_unlock(&m->pf_lock);return queued;
}
static void expert_prefetch_submit(Model*m,int li,int eid){(void)expert_prefetch_submit_set(m,li,&eid,1,0);}

static void load_expert(Model*m,Expert*e,int li,int eid){
    Cfg*c=&m->c;const char*mm=getenv("COLI_MMAP");
    int batch=st_env_enabled("PIPE")||st_env_enabled("URING")||st_env_enabled("DIRECT");
    int use_mmap=m->expert_cap<c->n_experts&&mm&&atoi(mm)!=0&&!batch;
    char base[3][QW_NAME],name[3][QW_NAME],qs[3][QW_NAME+8],tail[160];
    const char*suf[3]={"gate_proj","up_proj","down_proj"};
    int rows[3]={c->moe_inter,c->moe_inter,c->hidden};
    int cols[3]={c->hidden,c->hidden,c->moe_inter};
    for(int k=0;k<3;k++){snprintf(tail,sizeof(tail),"mlp.experts.%d.%s.weight",eid,suf[k]);lname(m,base[k],sizeof(base[k]),li,tail);}
    if(!use_mmap&&batch){
        int quantized=1;
        for(int k=0;k<3;k++)if(!qmat_source_is_quantized(m,base[k]))quantized=0;
        if(quantized){
            QMat*q[3]={&e->gate,&e->up,&e->down};void*payload[3];
            const char*read_name[6];void*read_out[6];
            for(int k=0;k<3;k++){
                prepare_quant_qmat(m,base[k],rows[k],cols[k],q[k],name[k],sizeof(name[k]),qs[k],sizeof(qs[k]),&payload[k]);
                read_name[2*k]=qs[k];read_out[2*k]=q[k]->s;
                read_name[2*k+1]=name[k];read_out[2*k+1]=payload[k];
            }
            st_read_raw_batch(&m->S,read_name,read_out,6,1);e->eid=eid;return;
        }
    }
    e->gate=load_qmat_mode(m,base[0],rows[0],cols[0],use_mmap);
    e->up=load_qmat_mode(m,base[1],rows[1],cols[1],use_mmap);
    e->down=load_qmat_mode(m,base[2],rows[2],cols[2],use_mmap);e->eid=eid;
}
static void load_expert_set(Model*m,Expert**expert,int li,const int*eid,int n){
    if(n<=0)return;if(n==1||!st_env_enabled("PIPE")){for(int i=0;i<n;i++)load_expert(m,expert[i],li,eid[i]);return;}
    Cfg*c=&m->c;int quantized=1;char base[QW_NAME],tail[160];
    const char*suf[3]={"gate_proj","up_proj","down_proj"};
    for(int i=0;i<n&&quantized;i++)for(int k=0;k<3;k++){snprintf(tail,sizeof(tail),"mlp.experts.%d.%s.weight",eid[i],suf[k]);lname(m,base,sizeof(base),li,tail);if(!qmat_source_is_quantized(m,base)){quantized=0;break;}}
    if(!quantized){for(int i=0;i<n;i++)load_expert(m,expert[i],li,eid[i]);return;}
    int nr=n*6;char(*resolved)[QW_NAME+8]=xcalloc((size_t)nr,sizeof(*resolved));
    const char**read_name=xcalloc(nr,sizeof(*read_name));void**read_out=xcalloc(nr,sizeof(*read_out));
    int rows[3]={c->moe_inter,c->moe_inter,c->hidden};
    int cols[3]={c->hidden,c->hidden,c->moe_inter};
    for(int i=0;i<n;i++){
        QMat*q[3]={&expert[i]->gate,&expert[i]->up,&expert[i]->down};
        for(int k=0;k<3;k++){void*payload=NULL;snprintf(tail,sizeof(tail),"mlp.experts.%d.%s.weight",eid[i],suf[k]);lname(m,base,sizeof(base),li,tail);
            prepare_quant_qmat(m,base,rows[k],cols[k],q[k],resolved[6*i+2*k+1],QW_NAME+8,resolved[6*i+2*k],QW_NAME+8,&payload);
            read_name[6*i+2*k]=resolved[6*i+2*k];read_out[6*i+2*k]=q[k]->s;
            read_name[6*i+2*k+1]=resolved[6*i+2*k+1];read_out[6*i+2*k+1]=payload;
        }
        expert[i]->eid=eid[i];
    }
    st_read_raw_batch(&m->S,read_name,read_out,nr,1);
    free(resolved);free(read_name);free(read_out);
}
static void load_moe(Model *m,Layer *l,int li){
    Cfg *c=&m->c; char n[QW_NAME]; MoeW *w=&l->moe;
    lname(m,n,sizeof(n),li,"mlp.gate.weight"); w->router=load_qmat(m,n,c->n_experts,c->hidden);
    w->cap=m->expert_cap;w->layer=li;w->expert=xcalloc(w->cap,sizeof(Expert));w->heat=xcalloc(c->n_experts,sizeof(uint32_t));w->last=xcalloc(c->n_experts,sizeof(uint32_t));if(st_env_enabled("PREFETCH_LOAD"))w->transition=xcalloc((size_t)c->n_experts*c->n_experts,sizeof(uint16_t));w->prefetched=xcalloc(c->n_experts,1);if(st_env_enabled("DECODE_PROTECT")){w->decode_heat=xcalloc(c->n_experts,sizeof(uint32_t));w->decode_pinned=xcalloc(c->n_experts,1);}pthread_mutex_init(&w->lock,NULL);pthread_cond_init(&w->pf_done,NULL);
    if(li<c->n_layers&&m->emap_seed)memcpy(w->heat,m->emap_seed+(size_t)li*c->n_experts,(size_t)c->n_experts*sizeof(uint32_t));
    int*initial=xcalloc(w->cap,sizeof(int));unsigned char*chosen=xcalloc(c->n_experts,1);
    for(int s=0;s<w->cap;s++){int best=-1;if(m->emap_loaded&&st_env_enabled("AUTOPIN"))for(int e=0;e<c->n_experts;e++)if(!chosen[e]&&(best<0||w->heat[e]>w->heat[best]))best=e;if(best<0||(!w->heat[best]&&s<c->n_experts&&!chosen[s]))best=s;while(best<c->n_experts&&chosen[best])best++;if(best>=c->n_experts)for(best=0;best<c->n_experts&&chosen[best];best++);if(best>=c->n_experts)die("cannot seed expert cache");chosen[best]=1;initial[s]=best;}
    for(int s=0;s<w->cap;s++){w->expert[s].eid=-1;load_expert(m,&w->expert[s],li,initial[s]);}
    free(initial);free(chosen);
    lname(m,n,sizeof(n),li,"mlp.shared_expert.gate_proj.weight"); w->shared_gate=load_qmat(m,n,c->shared_inter,c->hidden);
    lname(m,n,sizeof(n),li,"mlp.shared_expert.up_proj.weight"); w->shared_up=load_qmat(m,n,c->shared_inter,c->hidden);
    lname(m,n,sizeof(n),li,"mlp.shared_expert.down_proj.weight"); w->shared_down=load_qmat(m,n,c->hidden,c->shared_inter);
    lname(m,n,sizeof(n),li,"mlp.shared_expert_gate.weight"); w->shared_scale=load_qmat(m,n,1,c->hidden);
}
static void load_layer_into(Model *m,Layer*l,int li,int type){
    Cfg *c=&m->c; char n[QW_NAME]; l->type=type;l->index=li;
    lname(m,n,sizeof(n),li,"input_layernorm.weight"); l->input_norm=load_vec(m,n,c->hidden);
    lname(m,n,sizeof(n),li,"post_attention_layernorm.weight"); l->post_norm=load_vec(m,n,c->hidden);
    if(l->type==LT_LINEAR){
        GdnW *w=&l->gdn; int kd=c->lin_k_heads*c->lin_k_dim, vd=c->lin_v_heads*c->lin_v_dim, cd=2*kd+vd;
        lname(m,n,sizeof(n),li,"linear_attn.in_proj_qkv.weight"); w->qkv=load_qmat(m,n,cd,c->hidden);
        lname(m,n,sizeof(n),li,"linear_attn.in_proj_z.weight"); w->z=load_qmat(m,n,vd,c->hidden);
        lname(m,n,sizeof(n),li,"linear_attn.in_proj_b.weight"); w->b=load_qmat(m,n,c->lin_v_heads,c->hidden);
        lname(m,n,sizeof(n),li,"linear_attn.in_proj_a.weight"); w->a=load_qmat(m,n,c->lin_v_heads,c->hidden);
        lname(m,n,sizeof(n),li,"linear_attn.conv1d.weight"); w->conv=load_vec(m,n,cd*c->conv_kernel);
        lname(m,n,sizeof(n),li,"linear_attn.A_log"); w->A_log=load_vec(m,n,c->lin_v_heads);
        lname(m,n,sizeof(n),li,"linear_attn.dt_bias"); w->dt_bias=load_vec(m,n,c->lin_v_heads);
        lname(m,n,sizeof(n),li,"linear_attn.norm.weight"); w->norm=load_vec(m,n,c->lin_v_dim);
        lname(m,n,sizeof(n),li,"linear_attn.out_proj.weight"); w->out=load_qmat(m,n,c->hidden,vd);
        w->conv_state=falloc((int64_t)c->conv_kernel*cd);
        w->state=falloc((int64_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim);
    }else{
        AttnW *w=&l->attn; int qrows=c->n_heads*c->head_dim*2, kvrows=c->n_kv_heads*c->head_dim;
        lname(m,n,sizeof(n),li,"self_attn.q_proj.weight"); w->q=load_qmat(m,n,qrows,c->hidden);
        lname(m,n,sizeof(n),li,"self_attn.k_proj.weight"); w->k=load_qmat(m,n,kvrows,c->hidden);
        lname(m,n,sizeof(n),li,"self_attn.v_proj.weight"); w->v=load_qmat(m,n,kvrows,c->hidden);
        lname(m,n,sizeof(n),li,"self_attn.o_proj.weight"); w->o=load_qmat(m,n,c->hidden,c->n_heads*c->head_dim);
        lname(m,n,sizeof(n),li,"self_attn.q_norm.weight"); w->q_norm=load_vec(m,n,c->head_dim);
        lname(m,n,sizeof(n),li,"self_attn.k_norm.weight"); w->k_norm=load_vec(m,n,c->head_dim);
        if(m->kv16){w->k_cache16=xcalloc((int64_t)m->max_seq*kvrows,sizeof(uint16_t));w->v_cache16=xcalloc((int64_t)m->max_seq*kvrows,sizeof(uint16_t));}
        else{w->k_cache=falloc((int64_t)m->max_seq*kvrows);w->v_cache=falloc((int64_t)m->max_seq*kvrows);}
    }
    load_moe(m,l,li);
}
static void load_layer(Model*m,int li){load_layer_into(m,&m->layer[li],li,m->c.layer_type[li]);}
static size_t expert_packed_bytes(Model*m){
    const char*suf[]={"gate_proj","up_proj","down_proj"};size_t total=0;
    for(int j=0;j<3;j++){char base[QW_NAME],name[QW_NAME],tail[160],qs[QW_NAME+8];
        snprintf(tail,sizeof(tail),"mlp.experts.0.%s.weight",suf[j]);lname(m,base,sizeof(base),0,tail);
        if(find_qmat_named(m,base,name,sizeof(name))<0)die("cannot size expert cache");
        st_tensor*t=st_find(&m->S,name);if(!t)die("cannot find expert payload");
        total+=(size_t)t->nbytes;snprintf(qs,sizeof(qs),"%s.qs",name);t=st_find(&m->S,qs);
        if(t)total+=(size_t)t->nbytes;
    }
    return total;
}
static size_t dense_container_bytes(Model*m,int use_mtp){
    size_t total=0;for(int i=0;i<m->S.n;i++){const char*n=m->S.t[i].name;
        if(strstr(n,".mlp.experts."))continue;
        if(!use_mtp&&!strncmp(n,"mtp.",4))continue;
        total+=(size_t)m->S.t[i].nbytes;
    }return total;
}
static size_t runtime_state_bytes(Model*m,int ctx,int use_mtp){
    Cfg*c=&m->c;size_t total=0;int full=0;
    for(int i=0;i<c->n_layers;i++)if(c->layer_type[i]==LT_LINEAR){
        int cd=2*c->lin_k_heads*c->lin_k_dim+c->lin_v_heads*c->lin_v_dim;
        total+=(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim*sizeof(float);
        total+=(size_t)cd*c->conv_kernel*sizeof(float);
    }else full++;
    size_t scalar=m->kv16?sizeof(uint16_t):sizeof(float);
    total+=(size_t)ctx*full*2*c->n_kv_heads*c->head_dim*scalar;
    if(use_mtp)total+=(size_t)ctx*2*c->n_kv_heads*c->head_dim*scalar;
    return total;
}
#ifdef COLI_CUDA
static void cuda_validate_vram_plan(Model*m){
    if(!cuda_rt.active)return;size_t free_bytes=0,total_bytes=0;
    if(coli_cuda_memory_info(cuda_rt.ctx,&free_bytes,&total_bytes)){cuda_backend_fail("VRAM query");return;}
    double head_gb=getenv("CUDA_HEADROOM_GB")?atof(getenv("CUDA_HEADROOM_GB")):1.5;
    size_t dense=dense_container_bytes(m,m->mtp.enabled),state=runtime_state_bytes(m,m->max_seq,m->mtp.enabled);
    double required=(double)dense+(double)cuda_rt.expert_budget+(double)state+head_gb*1024.*1024.*1024.;
    fprintf(stderr,"[VRAM_PLAN] dense=%.2fGiB expert-budget=%.2fGiB state=%.2fGiB headroom=%.2fGiB required=%.2fGiB free=%.2fGiB total=%.2fGiB\n",(double)dense/(1024.*1024.*1024.),(double)cuda_rt.expert_budget/(1024.*1024.*1024.),(double)state/(1024.*1024.*1024.),head_gb,required/(1024.*1024.*1024.),(double)free_bytes/(1024.*1024.*1024.),(double)total_bytes/(1024.*1024.*1024.));
    if(required>(double)free_bytes)die("VRAM plan exceeds currently free device memory");
}
#endif
static void model_init(Model *m,const char *snap){
    memset(m,0,sizeof(*m)); load_cfg(&m->c,snap); int ctx=getenv("CTX")?atoi(getenv("CTX")):512;
    m->kv16=getenv("KV16")&&atoi(getenv("KV16"))!=0;
    if(ctx<=0)ctx=512; if(ctx>m->c.max_position)ctx=m->c.max_position; m->max_seq=ctx;
    st_init(&m->S,snap);
    expert_map_configure(m,snap);
    const char*mtp_env_early=getenv("MTP");int use_mtp_early=!mtp_env_early||atoi(mtp_env_early)!=0;char mtp_probe[QW_NAME];if(find_named(&m->S,"mtp.fc.weight",mtp_probe,sizeof(mtp_probe))<0)use_mtp_early=0;
    const char*ram_gb=getenv("RAM_GB");
    if(getenv("EXPERT_RAM"))m->expert_cap=atoi(getenv("EXPERT_RAM"));
    else if(ram_gb){
        double gb=atof(ram_gb);if(gb<=0.)die("RAM_GB must be positive");
        size_t one=expert_packed_bytes(m);int cache_layers=m->c.n_layers+(use_mtp_early?1:0);
        m->expert_cap=(int)((gb*1024.*1024.*1024.)/((double)cache_layers*one));
    }else m->expert_cap=m->c.n_experts;
    if(m->expert_cap<m->c.topk)m->expert_cap=m->c.topk;if(m->expert_cap>m->c.n_experts)m->expert_cap=m->c.n_experts;
    if(ram_gb){
        double gb=atof(ram_gb),head_gb=getenv("RAM_HEADROOM_GB")?atof(getenv("RAM_HEADROOM_GB")):1.5;
        size_t dense=dense_container_bytes(m,use_mtp_early),state=runtime_state_bytes(m,ctx,use_mtp_early),one=expert_packed_bytes(m);
        double required=(double)dense+gb*1024.*1024.*1024.+(double)state+head_gb*1024.*1024.*1024.;
        long pages=sysconf(_SC_PHYS_PAGES),page=sysconf(_SC_PAGESIZE);double physical=pages>0&&page>0?(double)pages*page:0.;
        fprintf(stderr,"[RAM_PLAN] dense=%.2fGiB expert-budget=%.2fGiB state=%.2fGiB headroom=%.2fGiB required=%.2fGiB physical=%.2fGiB expert-bytes=%zu cap/layer=%d\n",(double)dense/(1024.*1024.*1024.),gb,(double)state/(1024.*1024.*1024.),head_gb,required/(1024.*1024.*1024.),physical/(1024.*1024.*1024.),one,m->expert_cap);
        if(physical>0.&&required>physical)die("RAM plan exceeds physical memory");
    }
    char ename[QW_NAME]; if(find_named(&m->S,"embed_tokens.weight",ename,sizeof(ename))<0)die("missing embedding tensor");
    st_tensor *et=st_find(&m->S,ename); if(et->dtype==3)m->quant_mode=et->nbytes==(int64_t)m->c.vocab*m->c.hidden?8:4;
    char tp[2048]; snprintf(tp,sizeof(tp),"%s/tokenizer.json",snap);
    FILE *f=fopen(tp,"rb"); if(f){fclose(f);tok_load(&m->T,tp);m->has_tok=1;}
    double t0=now_s(); Cfg *c=&m->c;
    m->embed=load_qmat(m,"embed_tokens.weight",c->vocab,c->hidden); m->final_norm=load_vec(m,"norm.weight",c->hidden);
    char name[QW_NAME]; m->lm_head=find_named(&m->S,"lm_head.weight",name,sizeof(name))>=0?load_qmat(m,"lm_head.weight",c->vocab,c->hidden):m->embed;
    m->layer=xcalloc(c->n_layers,sizeof(Layer)); for(int i=0;i<c->n_layers;i++) load_layer(m,i);
    m->last_hidden=falloc(c->hidden);
    const char*mtp_env=getenv("MTP");int mtp_requested=!mtp_env||atoi(mtp_env)!=0;
    const char*mtp_req[]={"mtp.fc.weight","mtp.pre_fc_norm_embedding.weight","mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight","mtp.layers.0.input_layernorm.weight","mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_proj.weight","mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight","mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.gate.weight","mtp.layers.0.mlp.experts.0.gate_proj.weight"};
    int mtp_complete=c->mtp_layers>0;
    for(size_t i=0;i<sizeof(mtp_req)/sizeof(mtp_req[0])&&mtp_complete;i++){char found[QW_NAME];if(find_named(&m->S,mtp_req[i],found,sizeof(found))<0)mtp_complete=0;}
    if(mtp_requested&&mtp_complete){
        m->mtp.enabled=1;m->mtp.fc=load_qmat(m,"mtp.fc.weight",c->hidden,2*c->hidden);
        m->mtp.pre_embed_norm=load_vec(m,"mtp.pre_fc_norm_embedding.weight",c->hidden);
        m->mtp.pre_hidden_norm=load_vec(m,"mtp.pre_fc_norm_hidden.weight",c->hidden);
        m->mtp.norm=load_vec(m,"mtp.norm.weight",c->hidden);
        load_layer_into(m,&m->mtp.layer,c->n_layers,LT_FULL);
    }
    m->q3_route_atlas=st_env_enabled("Q3_ROUTE_ATLAS");
    if(m->q3_route_atlas){
#ifndef COLI_CUDA
        die("Q3_ROUTE_ATLAS requires a CUDA build");
#else
        if(!st_env_enabled("EXPERT_Q3")||m->matrix_i3<=0)
            die("Q3_ROUTE_ATLAS requires complete grouped-int3 experts");
        if(!cuda_q3_native_enabled())
            die("Q3_ROUTE_ATLAS requires Q3_NATIVE=1");
        if(!st_env_enabled("DECODE_PROTECT"))
            die("Q3_ROUTE_ATLAS requires DECODE_PROTECT=1");
        if(!st_env_enabled("CUDA_EXPERTS"))
            die("Q3_ROUTE_ATLAS requires CUDA_EXPERTS=1");
        fprintf(stderr,
            "[Q3_ATLAS] adaptive disjoint hot-route cache enabled; "
            "validation pending\n");
#endif
    }
    expert_prefetch_start(m);
    m->dense_load_s=now_s()-t0; m->dump_acts=getenv("DUMP_ACTS")&&strcmp(getenv("DUMP_ACTS"),"0");
    m->debug_logits=getenv("DEBUG_LOGITS")&&strcmp(getenv("DEBUG_LOGITS"),"0");
    m->prof_detail=getenv("PROF_DETAIL")&&atoi(getenv("PROF_DETAIL"))!=0;
    tier_atexit_model=m;atexit(tier_report_and_save);
}

/* ---------- hybrid forward ---------- */
static void model_reset(Model *m){
    m->pos=0;m->mtp.pos=0;memset(m->last_hidden,0,(size_t)m->c.hidden*sizeof(float)); Cfg *c=&m->c;
    if(m->pf_nthread){pthread_mutex_lock(&m->pf_lock);m->pf_head=m->pf_tail;pthread_mutex_unlock(&m->pf_lock);}
    for(int i=0;i<c->n_layers;i++){MoeW*w=&m->layer[i].moe;pthread_mutex_lock(&w->lock);w->prev_n=w->pf_pending=0;memset(w->prefetched,0,(size_t)c->n_experts);pthread_cond_broadcast(&w->pf_done);pthread_mutex_unlock(&w->lock);}
    if(m->mtp.enabled){MoeW*w=&m->mtp.layer.moe;pthread_mutex_lock(&w->lock);w->prev_n=w->pf_pending=0;memset(w->prefetched,0,(size_t)c->n_experts);pthread_cond_broadcast(&w->pf_done);pthread_mutex_unlock(&w->lock);}
    for(int i=0;i<c->n_layers;i++) if(m->layer[i].type==LT_LINEAR){
        int kd=c->lin_k_heads*c->lin_k_dim, vd=c->lin_v_heads*c->lin_v_dim, cd=2*kd+vd;
        memset(m->layer[i].gdn.conv_state,0,(size_t)c->conv_kernel*cd*sizeof(float));
        memset(m->layer[i].gdn.state,0,(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim*sizeof(float));
#ifdef COLI_CUDA
        GdnW*w=&m->layer[i].gdn;if(w->cuda_aux_ready){coli_cuda_memset(cuda_rt.ctx,w->d_conv_state,0,(size_t)c->conv_kernel*cd*sizeof(float));coli_cuda_memset(cuda_rt.ctx,w->d_state,0,(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim*sizeof(float));w->cuda_state_pos=0;}
#endif
    }
#ifdef COLI_CUDA
    for(int i=0;i<c->n_layers;i++)if(m->layer[i].type==LT_FULL&&m->layer[i].attn.cuda_aux_ready)m->layer[i].attn.cuda_state_pos=0;
    if(m->mtp.enabled&&m->mtp.layer.attn.cuda_aux_ready)m->mtp.layer.attn.cuda_state_pos=0;
#endif
}
static void gdn_rule_step(float *core,float *state,const float *q,const float *k,const float *v,const float *z,
                          const float *a,const float *b,const float *A_log,const float *dt_bias,const float *norm,
                          int kh,int vh,int dk,int dv,float eps){
    int ratio=vh/kh; float invsqrt=1.f/sqrtf((float)dk);
    for(int h=0;h<vh;h++){
        int hk=h/ratio; float *S=state+(int64_t)h*dk*dv; const float *qh=q+(int64_t)hk*dk,*khv=k+(int64_t)hk*dk,*vv=v+(int64_t)h*dv;
        float qn=0.f,kn=0.f; for(int i=0;i<dk;i++){qn+=qh[i]*qh[i];kn+=khv[i]*khv[i];}
        qn=1.f/sqrtf(qn+1e-6f); kn=1.f/sqrtf(kn+1e-6f);
        float beta=sigmoidf_stable(b[h]); float decay=expf(-expf(A_log[h])*softplusf_stable(a[h]+dt_bias[h]));
        for(int i=0;i<dk*dv;i++) S[i]*=decay;
        for(int j=0;j<dv;j++){
            float mem=0.f; for(int i=0;i<dk;i++) mem+=S[(int64_t)i*dv+j]*(khv[i]*kn);
            float delta=(vv[j]-mem)*beta; for(int i=0;i<dk;i++) S[(int64_t)i*dv+j]+=(khv[i]*kn)*delta;
        }
        for(int j=0;j<dv;j++){ float sum=0.f; for(int i=0;i<dk;i++)sum+=S[(int64_t)i*dv+j]*(qh[i]*qn*invsqrt); core[(int64_t)h*dv+j]=sum; }
        float ms=0.f; for(int j=0;j<dv;j++){float v0=core[(int64_t)h*dv+j];ms+=v0*v0;} float r=1.f/sqrtf(ms/(float)dv+eps);
        for(int j=0;j<dv;j++) core[(int64_t)h*dv+j]=core[(int64_t)h*dv+j]*r*norm[j]*siluf(z[(int64_t)h*dv+j]);
    }
}
/* Raw Gated DeltaNet rules used to validate the 64-token WY prefill path.
 * q/k are normalized here and q receives the 1/sqrt(dk) attention scale. */
static void gdn_prefill_seq(float *out,float *state,const float *q,const float *k,const float *v,const float *g,const float *beta,int T,int H,int dk,int dv){
    float *qn=falloc(dk),*kn=falloc(dk);float scale=1.f/sqrtf((float)dk);
    for(int t=0;t<T;t++)for(int h=0;h<H;h++){
        const float*qi=q+((int64_t)t*H+h)*dk,*ki=k+((int64_t)t*H+h)*dk,*vi=v+((int64_t)t*H+h)*dv;float*S=state+(int64_t)h*dk*dv,*yo=out+((int64_t)t*H+h)*dv;
        float q2=0.f,k2=0.f;for(int d=0;d<dk;d++){q2+=qi[d]*qi[d];k2+=ki[d]*ki[d];}q2=1.f/sqrtf(q2+1e-6f);k2=1.f/sqrtf(k2+1e-6f);for(int d=0;d<dk;d++){qn[d]=qi[d]*q2*scale;kn[d]=ki[d]*k2;}
        float decay=expf(g[(int64_t)t*H+h]);for(int i=0;i<dk*dv;i++)S[i]*=decay;
        for(int j=0;j<dv;j++){float mem=0.f;for(int d=0;d<dk;d++)mem+=S[(int64_t)d*dv+j]*kn[d];float delta=(vi[j]-mem)*beta[(int64_t)t*H+h];for(int d=0;d<dk;d++)S[(int64_t)d*dv+j]+=kn[d]*delta;}
        for(int j=0;j<dv;j++){float z0=0.f;for(int d=0;d<dk;d++)z0+=S[(int64_t)d*dv+j]*qn[d];yo[j]=z0;}
    }free(qn);free(kn);
}
static void gdn_prefill_chunked(float *out,float *state,const float *q,const float *k,const float *v,const float *g,const float *beta,int T,int H,int dk,int dv){
    enum{C=64};int nc=(T+C-1)/C;float scale=1.f/sqrtf((float)dk);
    float *qq=falloc((int64_t)C*dk),*kk=falloc((int64_t)C*dk),*vv=falloc((int64_t)C*dv),*bb=falloc(C),*gc=falloc(C);
    float *A=falloc(C*C),*tmp=falloc(C),*vp=falloc((int64_t)C*dv),*kc=falloc((int64_t)C*dk),*vn=falloc((int64_t)C*dv);
    for(int h=0;h<H;h++){float*S=state+(int64_t)h*dk*dv;
        for(int ch=0;ch<nc;ch++){int n=T-ch*C;if(n>C)n=C;memset(qq,0,(size_t)C*dk*sizeof(float));memset(kk,0,(size_t)C*dk*sizeof(float));memset(vv,0,(size_t)C*dv*sizeof(float));memset(bb,0,C*sizeof(float));memset(gc,0,C*sizeof(float));memset(A,0,C*C*sizeof(float));
            for(int i=0;i<n;i++){int t=ch*C+i;const float*qi=q+((int64_t)t*H+h)*dk,*ki=k+((int64_t)t*H+h)*dk;float q2=0.f,k2=0.f;for(int d=0;d<dk;d++){q2+=qi[d]*qi[d];k2+=ki[d]*ki[d];}q2=1.f/sqrtf(q2+1e-6f);k2=1.f/sqrtf(k2+1e-6f);for(int d=0;d<dk;d++){qq[(int64_t)i*dk+d]=qi[d]*q2*scale;kk[(int64_t)i*dk+d]=ki[d]*k2;}memcpy(vv+(int64_t)i*dv,v+((int64_t)t*H+h)*dv,(size_t)dv*sizeof(float));bb[i]=beta[(int64_t)t*H+h];gc[i]=g[(int64_t)t*H+h]+(i?gc[i-1]:0.f);}
            for(int i=n;i<C;i++)gc[i]=i?gc[i-1]:0.f;
            for(int i=1;i<C;i++)for(int j=0;j<i;j++)A[(int64_t)i*C+j]=-bb[i]*dot(kk+(int64_t)i*dk,kk+(int64_t)j*dk,dk)*expf(gc[i]-gc[j]);
            for(int i=1;i<C;i++){for(int j=0;j<i;j++){float z0=A[(int64_t)i*C+j];for(int p=0;p<i;p++)z0+=A[(int64_t)i*C+p]*A[(int64_t)p*C+j];tmp[j]=z0;}for(int j=0;j<i;j++)A[(int64_t)i*C+j]=tmp[j];}
            for(int i=0;i<C;i++)A[(int64_t)i*C+i]=1.f;
            for(int i=0;i<C;i++){for(int j=0;j<dv;j++){float z0=0.f;for(int p=0;p<C;p++)z0+=A[(int64_t)i*C+p]*vv[(int64_t)p*dv+j]*bb[p];vp[(int64_t)i*dv+j]=z0;}for(int d=0;d<dk;d++){float z0=0.f;for(int p=0;p<C;p++)z0+=A[(int64_t)i*C+p]*kk[(int64_t)p*dk+d]*bb[p]*expf(gc[p]);kc[(int64_t)i*dk+d]=z0;}}
            for(int i=0;i<C;i++)for(int j=0;j<dv;j++){float z0=0.f;for(int d=0;d<dk;d++)z0+=kc[(int64_t)i*dk+d]*S[(int64_t)d*dv+j];vn[(int64_t)i*dv+j]=vp[(int64_t)i*dv+j]-z0;}
            for(int i=0;i<n;i++){float*yo=out+((int64_t)(ch*C+i)*H+h)*dv;for(int j=0;j<dv;j++){float z0=0.f;for(int d=0;d<dk;d++)z0+=qq[(int64_t)i*dk+d]*expf(gc[i])*S[(int64_t)d*dv+j];for(int p=0;p<=i;p++)z0+=dot(qq+(int64_t)i*dk,kk+(int64_t)p*dk,dk)*expf(gc[i]-gc[p])*vn[(int64_t)p*dv+j];yo[j]=z0;}}
            float last=gc[C-1];for(int d=0;d<dk;d++)for(int j=0;j<dv;j++){float z0=S[(int64_t)d*dv+j]*expf(last);for(int i=0;i<C;i++)z0+=kk[(int64_t)i*dk+d]*expf(last-gc[i])*vn[(int64_t)i*dv+j];S[(int64_t)d*dv+j]=z0;}
        }
    }free(qq);free(kk);free(vv);free(bb);free(gc);free(A);free(tmp);free(vp);free(kc);free(vn);
}
static void gdn_forward(Model *m,Layer *l,const float *x,float *out){
    Cfg *c=&m->c; GdnW *w=&l->gdn; int kh=c->lin_k_heads,vh=c->lin_v_heads,dk=c->lin_k_dim,dv=c->lin_v_dim;
    int kd=kh*dk,vd=vh*dv,cd=2*kd+vd,K=c->conv_kernel,pos=m->pos;
    float *raw=falloc(cd),*mix=falloc(cd),*z=falloc(vd),*a=falloc(vh),*b=falloc(vh),*core=falloc(vd);
    qmat_mul_ex(b,x,&w->b,0);qmat_mul_ex(a,x,&w->a,0);
#ifdef COLI_CUDA
    float*cuda_check=NULL;if(cuda_gdn_try(out,x,a,b,w,c,pos)){const char*ce=getenv("CUDA_GDN_CHECK");if(!ce||atoi(ce)==0){free(raw);free(mix);free(z);free(a);free(b);free(core);return;}cuda_check=falloc(c->hidden);memcpy(cuda_check,out,(size_t)c->hidden*sizeof(float));}
#endif
    int projected=0;
#ifdef COLI_CUDA
    float*py[2]={raw,z};const QMat*pm[2]={&w->qkv,&w->z};projected=cuda_projection_group_try(py,x,pm,2);
#endif
    if(!projected){qmat_mul_ex(raw,x,&w->qkv,0);qmat_mul_ex(z,x,&w->z,0);}
    memcpy(w->conv_state+(int64_t)(pos%K)*cd,raw,(size_t)cd*sizeof(float));
    for(int ch=0;ch<cd;ch++){
        float acc=0.f;
        for(int tap=0;tap<K;tap++){ int src=pos-(K-1-tap); if(src>=0) acc+=w->conv[(int64_t)ch*K+tap]*w->conv_state[(int64_t)(src%K)*cd+ch]; }
        mix[ch]=siluf(acc);
    }
    gdn_rule_step(core,w->state,mix,mix+kd,mix+2*kd,z,a,b,w->A_log,w->dt_bias,w->norm,kh,vh,dk,dv,c->eps);
    qmat_mul_ex(out,core,&w->out,0);
#ifdef COLI_CUDA
    if(cuda_check){float md=0.f;for(int i=0;i<c->hidden;i++){float d=fabsf(out[i]-cuda_check[i]);if(d>md)md=d;}fprintf(stderr,"[CUDA_GDN_CHECK] layer=%d pos=%d maxdiff=%.8g\n",l->index,pos,md);free(cuda_check);}
#endif
    free(raw);free(mix);free(z);free(a);free(b);free(core);
}
static void rope_head(float *x,int hd,int rd,int pos,float theta){
    int half=rd/2; for(int i=0;i<half;i++){ float angle=(float)pos*powf(theta,-2.f*(float)i/(float)rd),co=cosf(angle),si=sinf(angle); float a=x[i],b=x[i+half]; x[i]=a*co-b*si; x[i+half]=b*co+a*si; } (void)hd;
}
static void attn_forward(Model *m,Layer *l,const float *x,float *out){
    Cfg *c=&m->c; AttnW *w=&l->attn; int nh=c->n_heads,nkv=c->n_kv_heads,hd=c->head_dim,pos=m->pos,rd=(int)(hd*c->partial_rotary);
    int qrows=nh*hd*2,kvrows=nkv*hd; float *qp=falloc(qrows),*q=falloc(nh*hd),*gate=falloc(nh*hd),*k=falloc(kvrows),*v=falloc(kvrows),*ctx=falloc(nh*hd),*scores=falloc((int64_t)nh*(pos+1));
#ifdef COLI_CUDA
    if(cuda_attn_try(out,x,w,c,pos,m->max_seq,m->kv16)){free(qp);free(q);free(gate);free(k);free(v);free(ctx);free(scores);return;}
#endif
    int projected=0;
#ifdef COLI_CUDA
    float*py[3]={qp,k,v};const QMat*pm[3]={&w->q,&w->k,&w->v};projected=cuda_projection_group_try(py,x,pm,3);
#endif
    if(!projected){qmat_mul_ex(qp,x,&w->q,0);qmat_mul_ex(k,x,&w->k,0);qmat_mul_ex(v,x,&w->v,0);}
    for(int h=0;h<nh;h++){ memcpy(q+(int64_t)h*hd,qp+(int64_t)h*2*hd,(size_t)hd*sizeof(float)); memcpy(gate+(int64_t)h*hd,qp+(int64_t)h*2*hd+hd,(size_t)hd*sizeof(float)); rmsnorm_zero(q+(int64_t)h*hd,q+(int64_t)h*hd,w->q_norm,hd,c->eps); rope_head(q+(int64_t)h*hd,hd,rd,pos,c->theta); }
    for(int h=0;h<nkv;h++){ rmsnorm_zero(k+(int64_t)h*hd,k+(int64_t)h*hd,w->k_norm,hd,c->eps); rope_head(k+(int64_t)h*hd,hd,rd,pos,c->theta); }
    if(m->kv16)for(int i=0;i<kvrows;i++){w->k_cache16[(int64_t)pos*kvrows+i]=f32_to_bf16(k[i]);w->v_cache16[(int64_t)pos*kvrows+i]=f32_to_bf16(v[i]);}else{memcpy(w->k_cache+(int64_t)pos*kvrows,k,(size_t)kvrows*sizeof(float));memcpy(w->v_cache+(int64_t)pos*kvrows,v,(size_t)kvrows*sizeof(float));}
    int rep=nh/nkv; float scale=1.f/sqrtf((float)hd);
    int par=parallel_work((int64_t)nh*(pos+1)*hd);
    #pragma omp parallel for schedule(static) if(par)
    for(int h=0;h<nh;h++){
        int hk=h/rep;float*sh=scores+(int64_t)h*(pos+1);float mx=-INFINITY;
        for(int t=0;t<=pos;t++){float z0=0.f;int64_t off=(int64_t)t*kvrows+(int64_t)hk*hd;for(int j=0;j<hd;j++)z0+=q[(int64_t)h*hd+j]*(m->kv16?bf16_to_f32(w->k_cache16[off+j]):w->k_cache[off+j]);sh[t]=z0*scale;if(sh[t]>mx)mx=sh[t];}
        float den=0.f;for(int t=0;t<=pos;t++){sh[t]=expf(sh[t]-mx);den+=sh[t];}
        for(int j=0;j<hd;j++){float sum=0.f;for(int t=0;t<=pos;t++){int64_t off=(int64_t)t*kvrows+(int64_t)hk*hd+j;sum+=(sh[t]/den)*(m->kv16?bf16_to_f32(w->v_cache16[off]):w->v_cache[off]);}ctx[(int64_t)h*hd+j]=sum*sigmoidf_stable(gate[(int64_t)h*hd+j]);}
    }
    qmat_mul_ex(out,ctx,&w->o,0); free(qp);free(q);free(gate);free(k);free(v);free(ctx);free(scores);
}
/* y[B,O] = X[B,I] @ W^T with one pass over the weights for the whole batch.
 * The per-token GEMV path re-reads an expert's matrices for every routed token,
 * which is pure memory traffic on a bandwidth-bound CPU; hoisting the weight-row
 * loop outside the batch reads each row once and reuses it for all B tokens. */
static void qmat_mul_batch(float *y,const float *x,int B,int ldx,const QMat *w,int allow_idot){
    if(B<=0)return;
    if(B==1){qmat_mul_ex(y,x,w,allow_idot);return;}
#ifdef COLI_CUDA
    if(cuda_qmat_batch_try(y,x,B,ldx,w))return;
#endif
    static int idot=-1;if(idot<0){const char*e=getenv("IDOT");idot=e?atoi(e)!=0:1;}
    int par=parallel_work((int64_t)w->O*w->I*B);
    if(w->fmt==1&&idot&&allow_idot){
        int8_t *q=xcalloc((int64_t)B*w->I,1);float *sx=falloc(B);
        for(int b=0;b<B;b++)sx[b]=qrow_i8(x+(int64_t)b*ldx,q+(int64_t)b*w->I,w->I);
        #pragma omp parallel for schedule(static) if(par)
        for(int o=0;o<w->O;o++){const int8_t*row=w->q8+(int64_t)o*w->rb;float s=w->s[o];
            for(int b=0;b<B;b++)y[(int64_t)b*w->O+o]=(float)dot_i8i8(row,q+(int64_t)b*w->I,w->I)*s*sx[b];}
        free(q);free(sx);return;
    }
    if(w->fmt==4){
        /* Keep each token's exact SIMD reduction order while placing tokens
         * inside the output-row loop.  The packed row remains hot for all B
         * activations without replacing AVX dot products with scalar nibble
         * expansion (which was both slower and numerically different). */
        #pragma omp parallel for schedule(static) if(par)
        for(int o=0;o<w->O;o++)for(int b=0;b<B;b++)y[(int64_t)b*w->O+o]=qmat_dot_row(w,o,x+(int64_t)b*ldx);
        return;
    }
    #pragma omp parallel for schedule(static) if(par)
    for(int o=0;o<w->O;o++)for(int b=0;b<B;b++)y[(int64_t)b*w->O+o]=qmat_dot_row(w,o,x+(int64_t)b*ldx);
}

/* One Gated DeltaNet layer for independent decode slots. State is already
 * resident in row-major [slot,...] form; no SessionState save/restore occurs.
 * This CPU seam establishes the Phase-9 row geometry before the CUDA kernels
 * receive the same leading slot dimension. */
static void gdn_forward_slot_batch(Model*m,Layer*l,const float*x,float*out,int B,
                                   const int*slot,const int*pos,ResidentLayerState*resident,
                                   int resident_slots,float*conv_state,float*state){
    if(B<=0)return;Cfg*c=&m->c;GdnW*w=&l->gdn;int kh=c->lin_k_heads,vh=c->lin_v_heads,dk=c->lin_k_dim,dv=c->lin_v_dim;
    int kd=kh*dk,vd=vh*dv,cd=2*kd+vd,K=c->conv_kernel,H=c->hidden;size_t cs=(size_t)K*cd,ss=(size_t)vh*dk*dv;
    float*a=falloc((int64_t)B*vh),*bp=falloc((int64_t)B*vh);
    qmat_mul_batch(bp,x,B,H,&w->b,0);qmat_mul_batch(a,x,B,H,&w->a,0);
#ifdef COLI_CUDA
    int cuda_used=cuda_gdn_slots_try(out,x,a,bp,B,slot,pos,resident,resident_slots,w,c);
    if(cuda_used>0){free(a);free(bp);return;}
    if(cuda_used<0)die("resident CUDA GDN state became unavailable");
#endif
    float*raw=falloc((int64_t)B*cd),*mix=falloc((int64_t)B*cd),*z=falloc((int64_t)B*vd),*core=falloc((int64_t)B*vd);
    qmat_mul_batch(raw,x,B,H,&w->qkv,0);qmat_mul_batch(z,x,B,H,&w->z,0);
    #pragma omp parallel for schedule(static) if(B>1)
    for(int row=0;row<B;row++){
        int sid=slot?slot[row]:row,p=pos[row];float*cr=conv_state+(size_t)sid*cs,*mr=mix+(int64_t)row*cd;
        memcpy(cr+(int64_t)(p%K)*cd,raw+(int64_t)row*cd,(size_t)cd*sizeof(float));
        for(int ch=0;ch<cd;ch++){float acc=0.f;for(int tap=0;tap<K;tap++){int src=p-(K-1-tap);if(src>=0)acc+=w->conv[(int64_t)ch*K+tap]*cr[(int64_t)(src%K)*cd+ch];}mr[ch]=siluf(acc);}
        gdn_rule_step(core+(int64_t)row*vd,state+(size_t)sid*ss,mr,mr+kd,mr+2*kd,z+(int64_t)row*vd,
                      a+(int64_t)row*vh,bp+(int64_t)row*vh,w->A_log,w->dt_bias,w->norm,kh,vh,dk,dv,c->eps);
    }
    qmat_mul_batch(out,core,B,vd,&w->out,0);
    free(raw);free(mix);free(z);free(a);free(bp);free(core);
}

/* Full-attention counterpart to gdn_forward_slot_batch. Each row owns a
 * [max_seq,kv_heads,head_dim] cache slice and may have a different position. */
static void attn_forward_slot_batch(Model*m,Layer*l,const float*x,float*out,int B,
                                    const int*slot,const int*pos,ResidentLayerState*resident,
                                    int resident_slots,float*k_cache,float*v_cache,
                                    uint16_t*k_cache16,uint16_t*v_cache16,int max_seq){
    if(B<=0)return;Cfg*c=&m->c;AttnW*w=&l->attn;int nh=c->n_heads,nkv=c->n_kv_heads,hd=c->head_dim;
    int rd=(int)(hd*c->partial_rotary),qrows=nh*hd*2,kvrows=nkv*hd,H=c->hidden,maxp=0;
    for(int row=0;row<B;row++){if(pos[row]<0||pos[row]>=max_seq)die("slot attention position outside context");if(pos[row]>maxp)maxp=pos[row];}
#ifdef COLI_CUDA
    int cuda_used=cuda_attn_slots_try(out,x,B,slot,pos,resident,resident_slots,w,c,max_seq,m->kv16);
    if(cuda_used>0)return;
    if(cuda_used<0)die("resident CUDA GQA state became unavailable");
#endif
    float*qp=falloc((int64_t)B*qrows),*q=falloc((int64_t)B*nh*hd),*gate=falloc((int64_t)B*nh*hd);
    float*k=falloc((int64_t)B*kvrows),*v=falloc((int64_t)B*kvrows),*ctx=falloc((int64_t)B*nh*hd);
    float*scores=falloc((int64_t)B*nh*(maxp+1));
    qmat_mul_batch(qp,x,B,H,&w->q,0);qmat_mul_batch(k,x,B,H,&w->k,0);qmat_mul_batch(v,x,B,H,&w->v,0);
    #pragma omp parallel for schedule(static) if(B>1)
    for(int row=0;row<B;row++){
        int sid=slot?slot[row]:row;
        float*qr=q+(int64_t)row*nh*hd,*gr=gate+(int64_t)row*nh*hd,*kr=k+(int64_t)row*kvrows;
        for(int h=0;h<nh;h++){memcpy(qr+(int64_t)h*hd,qp+(int64_t)row*qrows+(int64_t)h*2*hd,(size_t)hd*sizeof(float));memcpy(gr+(int64_t)h*hd,qp+(int64_t)row*qrows+(int64_t)h*2*hd+hd,(size_t)hd*sizeof(float));rmsnorm_zero(qr+(int64_t)h*hd,qr+(int64_t)h*hd,w->q_norm,hd,c->eps);rope_head(qr+(int64_t)h*hd,hd,rd,pos[row],c->theta);}
        for(int h=0;h<nkv;h++){rmsnorm_zero(kr+(int64_t)h*hd,kr+(int64_t)h*hd,w->k_norm,hd,c->eps);rope_head(kr+(int64_t)h*hd,hd,rd,pos[row],c->theta);}
        size_t base=(size_t)sid*max_seq*kvrows+(size_t)pos[row]*kvrows;
        if(m->kv16)for(int i=0;i<kvrows;i++){k_cache16[base+i]=f32_to_bf16(kr[i]);v_cache16[base+i]=f32_to_bf16(v[(int64_t)row*kvrows+i]);}
        else{memcpy(k_cache+base,kr,(size_t)kvrows*sizeof(float));memcpy(v_cache+base,v+(int64_t)row*kvrows,(size_t)kvrows*sizeof(float));}
    }
    int rep=nh/nkv;float scale=1.f/sqrtf((float)hd);int stride=maxp+1;
    #pragma omp parallel for schedule(static) if(B*nh>1)
    for(int rh=0;rh<B*nh;rh++){
        int row=rh/nh,sid=slot?slot[row]:row,h=rh%nh,hk=h/rep,p=pos[row];float*qr=q+((int64_t)row*nh+h)*hd,*gr=gate+((int64_t)row*nh+h)*hd;
        float*cr=ctx+((int64_t)row*nh+h)*hd,*sh=scores+(int64_t)rh*stride,mx=-INFINITY;size_t cache_base=(size_t)sid*max_seq*kvrows;
        for(int t=0;t<=p;t++){float z0=0.f;size_t off=cache_base+(size_t)t*kvrows+(size_t)hk*hd;for(int j=0;j<hd;j++)z0+=qr[j]*(m->kv16?bf16_to_f32(k_cache16[off+j]):k_cache[off+j]);sh[t]=z0*scale;if(sh[t]>mx)mx=sh[t];}
        float den=0.f;for(int t=0;t<=p;t++){sh[t]=expf(sh[t]-mx);den+=sh[t];}
        for(int j=0;j<hd;j++){float sum=0.f;for(int t=0;t<=p;t++){size_t off=cache_base+(size_t)t*kvrows+(size_t)hk*hd+j;sum+=(sh[t]/den)*(m->kv16?bf16_to_f32(v_cache16[off]):v_cache[off]);}cr[j]=sum*sigmoidf_stable(gr[j]);}
    }
    qmat_mul_batch(out,ctx,B,nh*hd,&w->o,0);
    free(qp);free(q);free(gate);free(k);free(v);free(ctx);free(scores);
}

/* Batched SwiGLU expert: out[B,hidden] from x[B,ldx]. */
static void mlp_batch(float *out,const float *x,int B,int ldx,const QMat *gate,const QMat *up,const QMat *down,int I){
    float *g=falloc((int64_t)B*I),*u=falloc((int64_t)B*I);
    qmat_mul_batch(g,x,B,ldx,gate,1); qmat_mul_batch(u,x,B,ldx,up,1);
    for(int64_t i=0;i<(int64_t)B*I;i++)g[i]=siluf(g[i])*u[i];
    qmat_mul_batch(out,g,B,I,down,1);
    free(g);free(u);
}

static void mlp_one(float *out,const float *x,const QMat *gate,const QMat *up,const QMat *down,int I){
#ifdef COLI_CUDA
    if(cuda_mlp_try(out,x,gate,up,down))return;
#endif
    float *g=falloc(I),*u=falloc(I); qmat_mul(g,x,gate); qmat_mul(u,x,up); for(int i=0;i<I;i++)g[i]=siluf(g[i])*u[i]; qmat_mul(out,g,down); free(g);free(u);
}
static void router_topk(const float *logit,int E,int K,int *idx,float *weight){
    float mx=-INFINITY;for(int e=0;e<E;e++)if(logit[e]>mx)mx=logit[e];float *prob=falloc(E),den=0.f;
    for(int e=0;e<E;e++){prob[e]=expf(logit[e]-mx);den+=prob[e];}for(int e=0;e<E;e++)prob[e]/=den;
    for(int j=0;j<K;j++){int best=-1;float bv=-1.f;for(int e=0;e<E;e++){int used=0;for(int p=0;p<j;p++)if(idx[p]==e)used=1;if(!used&&prob[e]>bv){bv=prob[e];best=e;}}idx[j]=best;weight[j]=bv;}
    float topden=0.f;for(int j=0;j<K;j++)topden+=weight[j];for(int j=0;j<K;j++)weight[j]/=topden;free(prob);
}
static void expert_decode_heat_touch(Model*m,MoeW*w,int eid){
    if(m->decode_phase&&w->decode_heat&&w->decode_heat[eid]!=UINT32_MAX)
        w->decode_heat[eid]++;
}
static int expert_victim_slot(Model*m,MoeW*w,const int*protect,int np){
    int slot=-1;uint64_t cold=UINT64_MAX;
    for(int pass=0;pass<2&&slot<0;pass++){
        for(int s=0;s<w->cap;s++){
            int old=w->expert[s].eid,held=0;
            for(int j=0;j<np;j++)if(protect[j]==old)held=1;
            if(held||(pass==0&&old>=0&&w->decode_pinned&&
                      w->decode_pinned[old]))continue;
            uint64_t score=old<0?0:
                tier_lfru_score(w->heat[old],w->last[old],w->clock);
            if(slot<0||score<cold){slot=s;cold=score;}
        }
        if(slot<0&&pass==0&&w->decode_pinned)
            __atomic_fetch_add(&m->decode_pin_fallbacks,1,
                               __ATOMIC_RELAXED);
        if(!w->decode_pinned)break;
    }
    return slot;
}
static Expert *expert_load_impl(Model*m,MoeW*w,int eid,const int*protect,int np);
typedef struct {int layer,eid;uint32_t heat,last;} Q3AtlasCandidate;
static int q3_atlas_candidate_cmp(const void*av,const void*bv){
    const Q3AtlasCandidate*a=av,*b=bv;
    if(a->heat!=b->heat)return a->heat>b->heat?-1:1;
    if(a->last!=b->last)return a->last>b->last?-1:1;
    if(a->layer!=b->layer)return a->layer-b->layer;
    return a->eid-b->eid;
}
static void expert_decode_pins_refresh(Model*m){
    if(!st_env_enabled("DECODE_PROTECT"))return;
    int E=m->c.n_experts,atlas=m->q3_route_atlas;
    int prewarm=st_env_enabled("DECODE_PROTECT_PREWARM")||atlas;
    uint64_t read_before=m->S.read_bytes;
    size_t candidate_cap=(size_t)m->c.n_layers*E,ncandidate=0;
    Q3AtlasCandidate*candidate=atlas?xcalloc(candidate_cap,sizeof(*candidate)):NULL;
#ifdef COLI_CUDA
    if(atlas)cuda_q3_atlas_unpin_all();
#endif
    m->q3_atlas_host_entries=0;m->q3_atlas_uncovered=0;
    m->decode_prewarmer=prewarm;
    for(int li=0;li<m->c.n_layers;li++){
        MoeW*w=&m->layer[li].moe;if(!w->decode_pinned)continue;
        int limit=w->cap-m->c.topk;if(limit<0)limit=0;
        int*selected=xcalloc((size_t)(limit?limit:1),sizeof(int)),nselected=0;
        pthread_mutex_lock(&w->lock);
        memset(w->decode_pinned,0,(size_t)E);
        for(int pick=0;pick<limit;pick++){
            int best=-1;
            for(int eid=0;eid<E;eid++)
                if(!w->decode_pinned[eid]&&w->decode_heat[eid]&&
                   (best<0||w->decode_heat[eid]>w->decode_heat[best]))
                    best=eid;
            if(best<0)break;w->decode_pinned[best]=1;
            selected[nselected++]=best;
        }
        if(atlas)for(int eid=0;eid<E;eid++)
            if(w->decode_heat[eid]&&!w->decode_pinned[eid]){
                candidate[ncandidate++]=(Q3AtlasCandidate){
                    .layer=li,.eid=eid,.heat=w->decode_heat[eid],
                    .last=w->last[eid]};
            }
        pthread_mutex_unlock(&w->lock);
        m->q3_atlas_host_entries+=(uint64_t)nselected;
#ifdef COLI_CUDA
        if(atlas)for(int i=0;i<nselected;i++)
            cuda_q3_atlas_drop_expert(li,selected[i]);
#endif
        /*
         * A decode expert can survive only in the independent device LRU
         * after its host owner was detached. Materialize the selected set in
         * host RAM now, between requests, so the following prompt cannot turn
         * a later device eviction into a decode-time disk miss.
         */
        if(prewarm)
            for(int i=0;i<nselected;i++){
                int present=0;pthread_mutex_lock(&w->lock);
                for(int s=0;s<w->cap;s++)
                    if(w->expert[s].eid==selected[i]){present=1;break;}
                pthread_mutex_unlock(&w->lock);
                if(!present)expert_load_impl(m,w,selected[i],NULL,0);
            }
        free(selected);
    }
#ifdef COLI_CUDA
    if(atlas&&cuda_rt.active){
        size_t one=expert_packed_bytes(m);
        size_t reserve=(size_t)m->c.topk*one;
        size_t usable=cuda_rt.expert_budget>reserve?
            cuda_rt.expert_budget-reserve:0;
        size_t capacity=one?usable/one:0;
        if(capacity>candidate_cap)capacity=candidate_cap;
        qsort(candidate,ncandidate,sizeof(*candidate),q3_atlas_candidate_cmp);
        cuda_rt.q3_atlas_capacity=(int)capacity;
        cuda_rt.q3_atlas_routes=(int)(m->q3_atlas_host_entries+ncandidate);
        size_t chosen=ncandidate<capacity?ncandidate:capacity;
        for(size_t i=0;i<chosen;i++){
            int li=candidate[i].layer,eid=candidate[i].eid;
            MoeW*w=&m->layer[li].moe;size_t bytes=0;
            int pinned=cuda_q3_atlas_pin_expert(li,eid,&bytes);
            if(!pinned){
                Expert*e=expert_load_impl(m,w,eid,NULL,0);
                cuda_expert_preload(e);
                pinned=cuda_q3_atlas_pin_expert(li,eid,&bytes);
                if(pinned)cuda_rt.q3_atlas_loads++;
            }
            if(!pinned){m->q3_atlas_uncovered++;continue;}
            cuda_rt.q3_atlas_entries++;
            cuda_rt.q3_atlas_bytes+=bytes;
            pthread_mutex_lock(&w->lock);
            for(int s=0;s<w->cap;s++)if(w->expert[s].eid==eid){
                free_qmat(&w->expert[s].gate);free_qmat(&w->expert[s].up);
                free_qmat(&w->expert[s].down);w->expert[s].eid=-1;break;
            }
            pthread_mutex_unlock(&w->lock);
        }
        if(ncandidate>chosen)m->q3_atlas_uncovered+=ncandidate-chosen;
        cuda_rt.q3_atlas_refreshes++;
    }else if(atlas)m->q3_atlas_uncovered+=ncandidate;
#else
    if(atlas)m->q3_atlas_uncovered+=ncandidate;
#endif
    m->decode_prewarmer=0;
    if(prewarm)m->decode_prewarm_bytes+=m->S.read_bytes-read_before;
    __atomic_fetch_add(&m->decode_pin_refreshes,1,__ATOMIC_RELAXED);
    free(candidate);
}
static void expert_heat_touch(Model*m,MoeW*w,int eid){
    if(!m->turn_hits)m->turn_hits=xcalloc((size_t)(m->c.n_layers+1)*m->c.n_experts,1);
    if(w->layer>=0&&w->layer<=m->c.n_layers&&eid>=0&&eid<m->c.n_experts)m->turn_hits[(size_t)w->layer*m->c.n_experts+eid]=1;
    pthread_mutex_lock(&w->lock);w->clock++;if(w->heat[eid]!=UINT32_MAX)w->heat[eid]++;expert_decode_heat_touch(m,w,eid);w->last[eid]=w->clock;
    if((w->clock&4095u)==0)tier_decay(w->heat,m->c.n_experts);pthread_mutex_unlock(&w->lock);
}
static void expert_prefetch_consume(Model*m,MoeW*w,int eid){
    if(w->prefetched[eid]){w->prefetched[eid]=0;__atomic_fetch_add(&m->pf_useful,1,__ATOMIC_RELAXED);}
}
static void expert_prefetch_evict(Model*m,MoeW*w,int eid){
    if(eid>=0&&w->prefetched[eid]){w->prefetched[eid]=0;__atomic_fetch_add(&m->pf_wasted,1,__ATOMIC_RELAXED);}
}
static Expert *expert_load_impl(Model*m,MoeW*w,int eid,const int*protect,int np){
    pthread_mutex_lock(&w->lock);w->clock++;if(!m->decode_prewarmer){if(w->heat[eid]!=UINT32_MAX)w->heat[eid]++;expert_decode_heat_touch(m,w,eid);}w->last[eid]=w->clock;
    if(!m->decode_prewarmer&&(w->clock&4095u)==0)tier_decay(w->heat,m->c.n_experts);
    for(int s=0;s<w->cap;s++)if(w->expert[s].eid==eid){Expert*e=&w->expert[s];expert_prefetch_consume(m,w,eid);__atomic_fetch_add(&m->tier_hits,1,__ATOMIC_RELAXED);pthread_mutex_unlock(&w->lock);return e;}
    int slot=expert_victim_slot(m,w,protect,np);
    if(slot<0)die("expert cache smaller than routed top-k");if(m->decode_prewarmer)__atomic_fetch_add(&m->decode_prewarm_loads,1,__ATOMIC_RELAXED);else __atomic_fetch_add(&m->tier_misses,1,__ATOMIC_RELAXED);Expert*e=&w->expert[slot];expert_prefetch_evict(m,w,e->eid);free_qmat(&e->gate);free_qmat(&e->up);free_qmat(&e->down);double t0=m->prof_detail?now_s():0.;load_expert(m,e,w->layer,eid);if(m->prof_detail){m->prof_expert_load+=now_s()-t0;m->prof_expert_misses++;}
    if(getenv("TIER_TRACE"))fprintf(stderr,"[TIER] layer=%d slot=%d expert=%d\n",w->layer,slot,eid);pthread_mutex_unlock(&w->lock);return e;
}
static void expert_load_many(Model*m,MoeW*w,const int*eid,int n,Expert**out){
    if(n<=0)return;if(n==1){out[0]=expert_load_impl(m,w,eid[0],eid,n);return;}
    Expert**miss=xcalloc(n,sizeof(*miss));int*mid=xcalloc(n,sizeof(*mid));int nm=0;
    pthread_mutex_lock(&w->lock);
    for(int i=0;i<n;i++){
        w->clock++;if(w->heat[eid[i]]!=UINT32_MAX)w->heat[eid[i]]++;expert_decode_heat_touch(m,w,eid[i]);w->last[eid[i]]=w->clock;
        if((w->clock&4095u)==0)tier_decay(w->heat,m->c.n_experts);
        for(int s=0;s<w->cap;s++)if(w->expert[s].eid==eid[i]){out[i]=&w->expert[s];break;}
        if(out[i]){expert_prefetch_consume(m,w,eid[i]);__atomic_fetch_add(&m->tier_hits,1,__ATOMIC_RELAXED);continue;}
        int slot=expert_victim_slot(m,w,eid,n);
        if(slot<0)die("expert cache smaller than routed top-k");
        Expert*e=&w->expert[slot];expert_prefetch_evict(m,w,e->eid);free_qmat(&e->gate);free_qmat(&e->up);free_qmat(&e->down);e->eid=eid[i];
        out[i]=e;miss[nm]=e;mid[nm++]=eid[i];__atomic_fetch_add(&m->tier_misses,1,__ATOMIC_RELAXED);
    }
    double t0=m->prof_detail?now_s():0.;load_expert_set(m,miss,w->layer,mid,nm);
    if(m->prof_detail){m->prof_expert_load+=now_s()-t0;m->prof_expert_misses+=(uint64_t)nm;}
    if(getenv("TIER_TRACE"))for(int i=0;i<nm;i++)fprintf(stderr,"[TIER] layer=%d expert=%d batch=%d\n",w->layer,mid[i],nm);
    pthread_mutex_unlock(&w->lock);free(miss);free(mid);
}
static void expert_prefetch_load_set(Model*m,MoeW*w,const int*eid,int n){
    if(n<=0)return;Expert**miss=xcalloc(n,sizeof(*miss));int*mid=xcalloc(n,sizeof(*mid));int nm=0;
    pthread_mutex_lock(&w->lock);
    for(int i=0;i<n;i++){
        int present=0;for(int s=0;s<w->cap;s++)if(w->expert[s].eid==eid[i]){present=1;break;}
        if(present)continue;
        int slot=expert_victim_slot(m,w,eid,n);
        if(slot<0)continue;
        Expert*e=&w->expert[slot];expert_prefetch_evict(m,w,e->eid);free_qmat(&e->gate);free_qmat(&e->up);free_qmat(&e->down);e->eid=eid[i];w->prefetched[eid[i]]=1;
        miss[nm]=e;mid[nm++]=eid[i];
    }
    load_expert_set(m,miss,w->layer,mid,nm);
    pthread_mutex_unlock(&w->lock);
    if(nm)__atomic_fetch_add(&m->pf_loads,(uint64_t)nm,__ATOMIC_RELAXED);
    free(miss);free(mid);
}
static int expert_predict_enabled(void){
    static int enabled=-1;if(enabled<0){const char*e=getenv("PREFETCH_LOAD");enabled=e&&atoi(e)!=0;}return enabled;
}
static void expert_route_record_block(Model*m,MoeW*w,const int*idx,int T,int K){
    if(m->route_record_suppress||!w->transition)return;
    int E=m->c.n_experts;pthread_mutex_lock(&w->lock);
    for(int t=0;t<T;t++){
        const int*current=idx+(int64_t)t*K;
        if(w->prev_n)for(int p=0;p<w->prev_n;p++)for(int q=0;q<K;q++){uint16_t*v=&w->transition[(size_t)w->prev_route[p]*E+current[q]];if(*v!=UINT16_MAX)(*v)++;}
        memcpy(w->prev_route,current,(size_t)K*sizeof(int));w->prev_n=K;
    }
    pthread_mutex_unlock(&w->lock);
}
static void expert_predict_submit(Model*m,MoeW*w){
    if(!expert_predict_enabled()||!m->pf_nthread)return;
    int E=m->c.n_experts,limit=getenv("PREFETCH_EXTRA")?atoi(getenv("PREFETCH_EXTRA")):2;
    double min_conf=getenv("PREFETCH_MIN_CONF")?atof(getenv("PREFETCH_MIN_CONF")):0.5;
    if(min_conf<0.)min_conf=0.;if(min_conf>1.)min_conf=1.;
    int spare=w->cap-m->c.topk;if(limit<=0||spare<=0)return;if(limit>spare)limit=spare;if(limit>QW_MAX_TOPK)limit=QW_MAX_TOPK;
    int predicted[QW_MAX_TOPK],n=0,prev[QW_MAX_TOPK],np=0;uint64_t*score=xcalloc(E,sizeof(*score));unsigned char*resident=xcalloc(E,1);
    pthread_mutex_lock(&w->lock);np=w->prev_n;if(np>QW_MAX_TOPK)np=QW_MAX_TOPK;memcpy(prev,w->prev_route,(size_t)np*sizeof(int));
    for(int s=0;s<w->cap;s++)if(w->expert[s].eid>=0)resident[w->expert[s].eid]=1;
    uint64_t score_total=0;for(int e=0;e<E;e++){for(int p=0;p<np;p++)score[e]+=w->transition[(size_t)prev[p]*E+e];score_total+=score[e];}
    pthread_mutex_unlock(&w->lock);
    if(!np||!score_total){free(score);free(resident);return;}
    for(int pick=0;pick<limit;pick++){int best=-1;for(int e=0;e<E;e++)if(!resident[e]&&score[e]&&(best<0||score[e]>score[best]))best=e;if(best<0||(double)m->c.topk*(double)score[best]/(double)score_total<min_conf)break;resident[best]=1;
#ifdef COLI_CUDA
        if(cuda_has_expert(m,w->layer,best)){__atomic_fetch_add(&m->pf_gpu_skips,1,__ATOMIC_RELAXED);continue;}
#endif
        predicted[n++]=best;
    }
    free(score);free(resident);if(!n)return;__atomic_fetch_add(&m->pf_predictions,(uint64_t)n,__ATOMIC_RELAXED);
    pthread_mutex_lock(&w->lock);while(w->pf_pending)pthread_cond_wait(&w->pf_done,&w->lock);w->pf_pending=1;pthread_mutex_unlock(&w->lock);
    if(!expert_prefetch_submit_set(m,w->layer,predicted,n,1)){pthread_mutex_lock(&w->lock);w->pf_pending=0;pthread_cond_broadcast(&w->pf_done);pthread_mutex_unlock(&w->lock);}
}
static void expert_predict_wait(MoeW*w){
    pthread_mutex_lock(&w->lock);while(w->pf_pending)pthread_cond_wait(&w->pf_done,&w->lock);pthread_mutex_unlock(&w->lock);
}
static void moe_forward(Model *m,Layer *l,const float *x,float *out){
    Cfg *c=&m->c; MoeW *w=&l->moe;expert_predict_wait(w);int E=c->n_experts,K=c->topk,H=c->hidden; float *logit=falloc(E),*tmp=falloc((int64_t)K*H),*shared=falloc(H);
    int *idx=xcalloc(K,sizeof(int)); float *weight=falloc(K);Expert**chosen=xcalloc(K,sizeof(Expert*));Expert*cached=xcalloc(K,sizeof(Expert));unsigned char*from_cache=xcalloc(K,1);qmat_mul_ex(logit,x,&w->router,0); router_topk(logit,E,K,idx,weight);
    expert_route_record_block(m,w,idx,1,K);
    int*nid=xcalloc(K,sizeof(int)),*npos=xcalloc(K,sizeof(int));int nn=0;
    for(int j=0;j<K;j++){
#ifdef COLI_CUDA
        if(cuda_cached_expert(m,w->layer,idx[j],&cached[j])){expert_heat_touch(m,w,idx[j]);__atomic_fetch_add(&m->tier_gpu_hits,1,__ATOMIC_RELAXED);chosen[j]=&cached[j];from_cache[j]=1;continue;}
#endif
        if(!expert_predict_enabled())expert_prefetch_submit(m,w->layer,idx[j]);nid[nn]=idx[j];npos[nn]=j;nn++;
    }
    if(nn){Expert**loaded=xcalloc(nn,sizeof(*loaded));expert_load_many(m,w,nid,nn,loaded);for(int i=0;i<nn;i++)chosen[npos[i]]=loaded[i];free(loaded);}
    memset(out,0,(size_t)H*sizeof(float));int grouped=0,shared_fused=0;
#ifdef COLI_CUDA
    grouped=cuda_experts_try(out,x,chosen,weight,K,w,&shared_fused);
#endif
    if(!grouped){for(int j=0;j<K;j++)if(from_cache[j])chosen[j]=expert_load_impl(m,w,idx[j],idx,K);int par=parallel_work((int64_t)K*(2*c->moe_inter*H+c->moe_inter*H));
        #pragma omp parallel for schedule(static) if(par)
        for(int j=0;j<K;j++)mlp_one(tmp+(int64_t)j*H,x,&chosen[j]->gate,&chosen[j]->up,&chosen[j]->down,c->moe_inter);
        for(int j=0;j<K;j++)for(int h=0;h<H;h++)out[h]+=tmp[(int64_t)j*H+h]*weight[j];}
    if(!shared_fused){mlp_one(shared,x,&w->shared_gate,&w->shared_up,&w->shared_down,c->shared_inter); float sglog;qmat_mul_ex(&sglog,x,&w->shared_scale,0);float sg=sigmoidf_stable(sglog);for(int h=0;h<H;h++)out[h]+=shared[h]*sg;}
    free(logit);free(tmp);free(shared);free(idx);free(weight);free(chosen);free(cached);free(from_cache);free(nid);free(npos);
}
#ifdef COLI_CUDA
static int moe_cuda_block(Model *m, Layer *l, const float *x, int T,
                          float *out) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("CUDA_SPEC_MOE_BATCH");
        /* The mixed routed-batch/shared-token path wins at D1 on the 35B
         * target. Preserve an explicit fallback for other devices. */
        enabled = !e || atoi(e) != 0;
    }
    Cfg *c = &m->c;
    MoeW *w = &l->moe;
    expert_predict_wait(w);
    int E = c->n_experts, K = c->topk, H = c->hidden, routes = T * K;
    if (!enabled || !cuda_rt.active || !cuda_rt.use_f16 || T < 2 || T > 8 ||
        K <= 0 || routes > 64)
        return 0;
    int *idx = xcalloc(routes, sizeof(int));
    float *weight = falloc(routes);
    Expert **chosen = xcalloc(routes, sizeof(Expert *));
    Expert *cached = xcalloc(routes, sizeof(Expert));
    unsigned char *unique = xcalloc(E, 1);
    int n_unique = 0;
    for (int t = 0; t < T; t++) {
        float *logit = falloc(E);
        qmat_mul_ex(logit, x + (int64_t)t * H, &w->router, 0);
        router_topk(logit, E, K, idx + (int64_t)t * K,
                    weight + (int64_t)t * K);
        free(logit);
        for (int j = 0; j < K; j++) {
            int eid = idx[(int64_t)t * K + j];
            if (!unique[eid]) {
                unique[eid] = 1;
                n_unique++;
            }
        }
    }
    int used = 0, shared_fused = 0;
    if (n_unique <= w->cap) {
        for (int r = 0; r < routes; r++) {
            int duplicate = -1;
            for (int p = 0; p < r; p++)
                if (idx[p] == idx[r]) {
                    duplicate = p;
                    break;
                }
            if (duplicate >= 0) {
                chosen[r] = chosen[duplicate];
                continue;
            }
            if (cuda_cached_expert(m, w->layer, idx[r], &cached[r])) {
                expert_heat_touch(m,w,idx[r]);__atomic_fetch_add(&m->tier_gpu_hits,1,__ATOMIC_RELAXED);
                chosen[r] = &cached[r];
                continue;
            }
            if(!expert_predict_enabled())expert_prefetch_submit(m, w->layer, idx[r]);
            chosen[r] = expert_load_impl(m, w, idx[r], idx, routes);
        }
        used = cuda_experts_batch_try(out, x, chosen, weight, T, K, w,
                                      &shared_fused);
        if(used&&!shared_fused){float*shared=falloc(H);for(int t=0;t<T;t++){mlp_one(shared,x+(int64_t)t*H,&w->shared_gate,&w->shared_up,&w->shared_down,c->shared_inter);float sglog;qmat_mul_ex(&sglog,x+(int64_t)t*H,&w->shared_scale,0);float sg=sigmoidf_stable(sglog);for(int h=0;h<H;h++)out[(int64_t)t*H+h]+=shared[h]*sg;}free(shared);}
        if(used){expert_route_record_block(m,w,idx,T,K);cuda_rt.batch_transactions++;cuda_rt.batch_routes+=(uint64_t)routes;cuda_rt.batch_unique_experts+=(uint64_t)n_unique;}
    }
    free(idx);
    free(weight);
    free(chosen);
    free(cached);
    free(unique);
    return used;
}

static int cuda_resident_moe_eligible(Model*m,Layer*l,int B){
    if(!cuda_rt.active||!cuda_rt.use_f16||B<1||B>8||
       m->c.topk<=0||B*m->c.topk>64||
       !(getenv("CUDA_EXPERTS")&&atoi(getenv("CUDA_EXPERTS"))!=0))
        return 0;
    MoeW*w=&l->moe;QMat*r=&w->router,*g=&w->shared_gate,*u=&w->shared_up;
    QMat*d=&w->shared_down,*s=&w->shared_scale;
    int router_ok=r->f||(r->cuda_eligible&&(r->fmt==1||r->fmt==4));
    int shared_q4=g->fmt==4&&u->fmt==4&&d->fmt==4&&s->fmt==4;
    int scale_ok=s->f||(s->cuda_eligible&&(s->fmt==1||s->fmt==4));
    int shared_q8=g->fmt==1&&u->fmt==1&&d->fmt==1&&scale_ok;
    return router_ok&&(shared_q4||shared_q8);
}

static int cuda_resident_moe_try(Model*m,Layer*l,ResidentBatchState*s,int B){
    if(!cuda_resident_moe_eligible(m,l,B))return 0;
    Cfg*c=&m->c;MoeW*w=&l->moe;int E=c->n_experts,K=c->topk,H=c->hidden;
    int routes=B*K,used=0;
    expert_predict_wait(w);
    float*logit=falloc((int64_t)B*E),*weight=falloc(routes);
    int*idx=xcalloc(routes,sizeof(int));Expert**chosen=xcalloc(routes,sizeof(*chosen));
    Expert*cached=xcalloc(routes,sizeof(*cached));unsigned char*unique=xcalloc(E,1);

    pthread_mutex_lock(&cuda_rt.lock);
    int router_error=0;
    if(w->router.f){
        if(!w->d_router_weight){
            size_t bytes=(size_t)E*H*sizeof(float);
            router_error=coli_cuda_malloc(cuda_rt.ctx,
                                           (void**)&w->d_router_weight,bytes)||
                         coli_cuda_upload(cuda_rt.ctx,w->d_router_weight,
                                          w->router.f,bytes);
        }
        if(!router_error)router_error=coli_cuda_f32_gemm(
            cuda_rt.ctx,s->d_router,s->d_n,w->d_router_weight,B,E,H);
    }else router_error=cuda_qmat_batch_device_locked(
        s->d_router,s->d_n,&w->router,B);
    if(!router_error)router_error=coli_cuda_download(
        cuda_rt.ctx,logit,s->d_router,(size_t)B*E*sizeof(float))||
        coli_cuda_sync(cuda_rt.ctx);
    if(!router_error)cuda_rt.resident_router_d2h+=(uint64_t)B*E*sizeof(float);
    if(router_error){cuda_backend_fail("resident router boundary");used=-1;}
    pthread_mutex_unlock(&cuda_rt.lock);
    if(used<0)goto done;

    int n_unique=0;
    for(int row=0;row<B;row++){
        router_topk(logit+(int64_t)row*E,E,K,idx+(int64_t)row*K,
                    weight+(int64_t)row*K);
        for(int j=0;j<K;j++){int eid=idx[(int64_t)row*K+j];
            if(!unique[eid]){unique[eid]=1;n_unique++;}}
    }
    if(n_unique>w->cap)goto done;
    for(int route=0;route<routes;route++){
        int duplicate=-1;
        for(int p=0;p<route;p++)if(idx[p]==idx[route]){duplicate=p;break;}
        if(duplicate>=0){chosen[route]=chosen[duplicate];continue;}
        if(cuda_cached_expert(m,w->layer,idx[route],&cached[route])){
            expert_heat_touch(m,w,idx[route]);
            __atomic_fetch_add(&m->tier_gpu_hits,1,__ATOMIC_RELAXED);
            chosen[route]=&cached[route];continue;
        }
        if(!expert_predict_enabled())expert_prefetch_submit(m,w->layer,
                                                            idx[route]);
        chosen[route]=expert_load_impl(m,w,idx[route],idx,routes);
    }

    pthread_mutex_lock(&cuda_rt.lock);
    int error=0;
    for(int route=0;route<routes&&!error;route++){Expert*e=chosen[route];
        if(!e||(e->gate.fmt!=2&&e->gate.fmt!=3&&e->gate.fmt!=4)||
           e->up.fmt!=e->gate.fmt||e->down.fmt!=e->gate.fmt)error=1;}
    if(!error)error=cuda_experts_upload_locked(chosen,routes);
    QMat*sg=&w->shared_gate,*su=&w->shared_up,*sd=&w->shared_down;
    QMat*ss=&w->shared_scale;
    if(!error)error=cuda_qmat_upload_locked(sg)||
                    cuda_qmat_upload_locked(su)||
                    cuda_qmat_upload_locked(sd);
    if(!error&&ss->fmt==4)error=cuda_qmat_upload_locked(ss);
    if(!error)error=coli_cuda_f32_to_f16(cuda_rt.ctx,s->d_n16,s->d_n,B*H);
    const unsigned char*gq[64],*uq[64],*dq[64];
    const float*gs[64],*us[64],*ds[64];
    for(int route=0;route<routes;route++){Expert*e=chosen[route];
        if(!e)continue;gq[route]=(const unsigned char*)e->gate.d_q;
        gs[route]=(const float*)e->gate.d_s;
        uq[route]=(const unsigned char*)e->up.d_q;
        us[route]=(const float*)e->up.d_s;
        dq[route]=(const unsigned char*)e->down.d_q;
        ds[route]=(const float*)e->down.d_s;}
    if(!error){QMat*g=&chosen[0]->gate,*d=&chosen[0]->down;
        int native_q3=g->fmt==3&&cuda_q3_native_enabled();
        error=native_q3?coli_cuda_grouped_q3_mlp_batch_f16(
            cuda_rt.ctx,s->d_moe,s->d_n16,gq,gs,uq,us,dq,ds,weight,B,K,H,
            c->moe_inter,g->gs,g->rb,g->ng,d->rb,d->ng):
            coli_cuda_grouped_q4_mlp_batch_f16(
            cuda_rt.ctx,s->d_moe,s->d_n16,gq,gs,uq,us,dq,ds,weight,B,K,H,
            c->moe_inter,g->gs,g->drb,g->ng,d->drb,d->ng);}
    if(!error&&chosen[0]->gate.fmt==3&&cuda_q3_native_enabled())
        cuda_rt.q3_grouped_calls++;
    if(!error&&sg->fmt==4)
        error=coli_cuda_shared_q4_mlp_batch_f16(
            cuda_rt.ctx,s->d_moe,s->d_n16,
            (const unsigned char*)sg->d_q,(const float*)sg->d_s,
            (const unsigned char*)su->d_q,(const float*)su->d_s,
            (const unsigned char*)sd->d_q,(const float*)sd->d_s,
            (const unsigned char*)ss->d_q,(const float*)ss->d_s,B,H,
            c->shared_inter,sg->gs,sg->rb,sg->ng,sd->rb,sd->ng,ss->rb,
            ss->ng);
    if(!error&&sg->fmt==1){
        if(ss->f){
            if(!w->d_scale_weight){
                size_t bytes=(size_t)H*sizeof(float);
                error=coli_cuda_malloc(cuda_rt.ctx,
                                        (void**)&w->d_scale_weight,bytes)||
                      coli_cuda_upload(cuda_rt.ctx,w->d_scale_weight,
                                       ss->f,bytes);
            }
            if(!error)error=coli_cuda_f32_gemm(
                cuda_rt.ctx,s->d_scale,s->d_n,w->d_scale_weight,B,1,H);
        }else error=cuda_qmat_batch_device_locked(
            s->d_scale,s->d_n,ss,B);
        if(!error)error=coli_cuda_sigmoid(
            cuda_rt.ctx,s->d_scale,s->d_scale,B);
        if(!error)error=coli_cuda_shared_q8_mlp_batch_device_scale(
            cuda_rt.ctx,s->d_moe,s->d_n,s->d_scale,
            (const signed char*)sg->d_q,(const float*)sg->d_s,sg->rb,
            (const signed char*)su->d_q,(const float*)su->d_s,su->rb,
            (const signed char*)sd->d_q,(const float*)sd->d_s,sd->rb,
            B,H,c->shared_inter);
    }
    if(!error)error=coli_cuda_axpy(cuda_rt.ctx,s->d_x,s->d_moe,1.f,B*H);
    if(error){cuda_backend_fail("resident device MoE");used=-1;}
    else{cuda_rt.calls+=(uint64_t)3*routes+(uint64_t)4*B;
        cuda_rt.mlp_calls+=(uint64_t)routes+(uint64_t)B;
        cuda_rt.grouped_calls+=(uint64_t)B;cuda_rt.grouped_kernel_calls++;
        cuda_rt.batch_transactions++;cuda_rt.batch_routes+=(uint64_t)routes;
        cuda_rt.batch_unique_experts+=(uint64_t)n_unique;
        cuda_rt.resident_device_moe++;used=1;}
    pthread_mutex_unlock(&cuda_rt.lock);
    if(used>0)expert_route_record_block(m,w,idx,B,K);
done:
    free(logit);free(weight);free(idx);free(chosen);free(cached);free(unique);
    return used;
}
#endif
static void moe_prefill_grouped(Model*m,Layer*l,const float*x,int T,float*out){
    Cfg*c=&m->c;MoeW*w=&l->moe;expert_predict_wait(w);int E=c->n_experts,K=c->topk,H=c->hidden;int*idx=xcalloc((int64_t)T*K,sizeof(int));float*weight=falloc((int64_t)T*K),*tmp=falloc((int64_t)T*H),*routed=falloc((int64_t)T*K*H),*shared=falloc((int64_t)T*H);unsigned char*used=xcalloc(E,1);
    #pragma omp parallel for schedule(static) if(T>1)
    for(int t=0;t<T;t++){float*logit=falloc(E);qmat_mul_ex(logit,x+(int64_t)t*H,&w->router,0);router_topk(logit,E,K,idx+(int64_t)t*K,weight+(int64_t)t*K);free(logit);}
    for(int t=0;t<T;t++)for(int j=0;j<K;j++)used[idx[(int64_t)t*K+j]]=1;
    expert_route_record_block(m,w,idx,T,K);
    /* The shared expert sees every token, so this is the highest-reuse batch in
     * the layer. Keep its weights hot across T activations as well. */
    mlp_batch(shared,x,T,H,&w->shared_gate,&w->shared_up,&w->shared_down,c->shared_inter);
    #pragma omp parallel for schedule(static) if(T>1)
    for(int t=0;t<T;t++){float*ot=shared+(int64_t)t*H,sg;qmat_mul_ex(&sg,x+(int64_t)t*H,&w->shared_scale,0);sg=sigmoidf_stable(sg);for(int h=0;h<H;h++)ot[h]*=sg;}
    if(!expert_predict_enabled())for(int eid=0;eid<E;eid++)if(used[eid])expert_prefetch_submit(m,w->layer,eid);
    /* Gather the tokens routed to each expert, run one batched SwiGLU, then
     * scatter the weighted results back. The expert's weights are streamed once
     * per layer instead of once per routed token. */
    int *tok=xcalloc(T,sizeof(int));float *wt=falloc(T),*xb=falloc((int64_t)T*H);
    for(int eid=0;eid<E;eid++)if(used[eid]){Expert*e=expert_load_impl(m,w,eid,NULL,0);
#ifdef COLI_CUDA
        cuda_expert_preload(e);
#endif
        int B=0;
        for(int t=0;t<T;t++)for(int j=0;j<K;j++)if(idx[(int64_t)t*K+j]==eid){tok[B]=t;wt[B]=weight[(int64_t)t*K+j];B++;break;}
        if(!B)continue;
        for(int b=0;b<B;b++)memcpy(xb+(int64_t)b*H,x+(int64_t)tok[b]*H,(size_t)H*sizeof(float));
        mlp_batch(tmp,xb,B,H,&e->gate,&e->up,&e->down,c->moe_inter);
        for(int b=0;b<B;b++){int t=tok[b],route=0;while(route<K&&idx[(int64_t)t*K+route]!=eid)route++;if(route==K)die("grouped route scatter mismatch");float*ot=routed+((int64_t)t*K+route)*H,*tt=tmp+(int64_t)b*H,g=wt[b];for(int h=0;h<H;h++)ot[h]=tt[h]*g;}
    }
    for(int t=0;t<T;t++){float*ot=out+(int64_t)t*H;memset(ot,0,(size_t)H*sizeof(float));for(int j=0;j<K;j++){float*rt=routed+((int64_t)t*K+j)*H;for(int h=0;h<H;h++)ot[h]+=rt[h];}for(int h=0;h<H;h++)ot[h]+=shared[(int64_t)t*H+h];}
    free(tok);free(wt);free(xb);
    free(idx);free(weight);free(tmp);free(routed);free(shared);free(used);
}
static void layer_forward_slot_batch(Model*m,Layer*l,float*x,int B,const int*slot,const int*pos,
                                     ResidentLayerState*resident,int resident_slots,float*conv_state,float*gdn_state,float*k_cache,float*v_cache,
                                     uint16_t*k_cache16,uint16_t*v_cache16,int max_seq){
    Cfg*c=&m->c;int H=c->hidden;float*n=falloc((int64_t)B*H),*mix=falloc((int64_t)B*H),*moe=falloc((int64_t)B*H);
    expert_predict_submit(m,&l->moe);
    #pragma omp parallel for schedule(static) if(B>1)
    for(int row=0;row<B;row++)rmsnorm_zero(n+(int64_t)row*H,x+(int64_t)row*H,l->input_norm,H,c->eps);
    double core_t0=m->prof_detail?now_s():0.;
    if(l->type==LT_LINEAR)gdn_forward_slot_batch(m,l,n,mix,B,slot,pos,resident,resident_slots,conv_state,gdn_state);
    else attn_forward_slot_batch(m,l,n,mix,B,slot,pos,resident,resident_slots,k_cache,v_cache,k_cache16,v_cache16,max_seq);
    if(m->prof_detail){if(l->type==LT_LINEAR)m->prof_gdn+=now_s()-core_t0;else m->prof_attn+=now_s()-core_t0;}
    #pragma omp parallel for schedule(static) if(B>1)
    for(int row=0;row<B;row++){float*xr=x+(int64_t)row*H,*mr=mix+(int64_t)row*H,*nr=n+(int64_t)row*H;for(int h=0;h<H;h++)xr[h]+=mr[h];rmsnorm_zero(nr,xr,l->post_norm,H,c->eps);}
    double moe_t0=m->prof_detail?now_s():0.;moe_prefill_grouped(m,l,n,B,moe);
    if(m->prof_detail)m->prof_moe+=now_s()-moe_t0;
    #pragma omp parallel for schedule(static) if(B>1)
    for(int row=0;row<B;row++){float*xr=x+(int64_t)row*H,*er=moe+(int64_t)row*H;for(int h=0;h<H;h++)xr[h]+=er[h];}
    free(n);free(mix);free(moe);
}
static void resident_batch_free(Model*m,ResidentBatchState*s);
static int resident_batch_init(Model*m,ResidentBatchState*s,int nslots){
    if(!m||!s||nslots<1||nslots>MUX_MAX_SLOTS)return 0;memset(s,0,sizeof(*s));Cfg*c=&m->c;
    s->nslots=nslots;s->max_seq=m->max_seq;s->kv16=m->kv16;s->pos=xcalloc(nslots,sizeof(int));
    s->last_hidden=falloc((int64_t)nslots*c->hidden);s->logits=falloc((int64_t)nslots*c->vocab);
    s->layer=xcalloc(c->n_layers,sizeof(ResidentLayerState));
    int cd=2*c->lin_k_heads*c->lin_k_dim+c->lin_v_heads*c->lin_v_dim,kvrows=c->n_kv_heads*c->head_dim;
#ifdef COLI_CUDA
    int cuda_kv=cuda_rt.active&&(!getenv("CUDA_RESIDENT_KV")||atoi(getenv("CUDA_RESIDENT_KV"))!=0);size_t cuda_kv_need=0;
    if(cuda_kv)for(int li=0;li<c->n_layers;li++)if(m->layer[li].type==LT_FULL&&cuda_attn_eligible(&m->layer[li].attn,c,m->kv16))cuda_kv_need+=(size_t)2*nslots*m->max_seq*kvrows*sizeof(float);
    if(cuda_kv&&cuda_kv_need){size_t free_bytes=0,total_bytes=0;if(coli_cuda_memory_info(cuda_rt.ctx,&free_bytes,&total_bytes)||cuda_kv_need+(size_t)512*1024*1024>free_bytes){fprintf(stderr,"[CUDA] resident GQA KV needs %.2fGiB; retaining host KV\n",(double)cuda_kv_need/(1024.*1024.*1024.));cuda_kv=0;}}
#endif
    for(int li=0;li<c->n_layers;li++){ResidentLayerState*r=&s->layer[li];
        if(m->layer[li].type==LT_LINEAR){size_t nc=(size_t)nslots*c->conv_kernel*cd,ns=(size_t)nslots*c->lin_v_heads*c->lin_k_dim*c->lin_v_dim;r->conv=nc?falloc((int64_t)nc):NULL;r->gdn=ns?falloc((int64_t)ns):NULL;
#ifdef COLI_CUDA
            if(cuda_rt.active&&cuda_gdn_eligible(&m->layer[li].gdn)){
                if(coli_cuda_malloc(cuda_rt.ctx,(void**)&r->d_conv,nc*sizeof(float))||coli_cuda_malloc(cuda_rt.ctx,(void**)&r->d_gdn,ns*sizeof(float))||coli_cuda_memset(cuda_rt.ctx,r->d_conv,0,nc*sizeof(float))||coli_cuda_memset(cuda_rt.ctx,r->d_gdn,0,ns*sizeof(float))){resident_batch_free(m,s);return 0;}r->cuda_gdn=1;
            }
#endif
        }
        else{size_t nk=(size_t)nslots*m->max_seq*kvrows;if(m->kv16){r->k16=nk?xcalloc(nk,sizeof(uint16_t)):NULL;r->v16=nk?xcalloc(nk,sizeof(uint16_t)):NULL;}else{r->k=nk?falloc((int64_t)nk):NULL;r->v=nk?falloc((int64_t)nk):NULL;
#ifdef COLI_CUDA
            if(cuda_kv&&cuda_attn_eligible(&m->layer[li].attn,c,m->kv16)){if(coli_cuda_malloc(cuda_rt.ctx,(void**)&r->d_k,nk*sizeof(float))||coli_cuda_malloc(cuda_rt.ctx,(void**)&r->d_v,nk*sizeof(float))||coli_cuda_memset(cuda_rt.ctx,r->d_k,0,nk*sizeof(float))||coli_cuda_memset(cuda_rt.ctx,r->d_v,0,nk*sizeof(float))){resident_batch_free(m,s);return 0;}r->cuda_attn=1;}
#endif
        }}
    }
#ifdef COLI_CUDA
    if(cuda_rt.active&&coli_cuda_sync(cuda_rt.ctx)){resident_batch_free(m,s);return 0;}
    int want_activations=!getenv("CUDA_RESIDENT_ACTIVATIONS")||
                         atoi(getenv("CUDA_RESIDENT_ACTIVATIONS"))!=0;
    s->cuda_activations=want_activations&&cuda_rt.active&&cuda_rt.use_f16;
    int missing_gdn=0,missing_gqa=0,missing_gate=0;
    for(int li=0;li<c->n_layers&&s->cuda_activations;li++){
        ResidentLayerState*r=&s->layer[li];Layer*l=&m->layer[li];
        if(l->type==LT_LINEAR&&!r->cuda_gdn)missing_gdn++;
        if(l->type==LT_LINEAR){
            int aok=l->gdn.a.f||(l->gdn.a.cuda_eligible&&
                                  (l->gdn.a.fmt==1||l->gdn.a.fmt==4));
            int bok=l->gdn.b.f||(l->gdn.b.cuda_eligible&&
                                  (l->gdn.b.fmt==1||l->gdn.b.fmt==4));
            if(!aok||!bok)missing_gate++;
        }
        if(l->type==LT_FULL&&!r->cuda_attn)missing_gqa++;
        if(missing_gdn||missing_gate||missing_gqa)
            s->cuda_activations=0;
    }
    if(s->cuda_activations){
        size_t hb=(size_t)nslots*c->hidden*sizeof(float);
        size_t gb=(size_t)nslots*c->lin_v_heads*sizeof(float);
        size_t rb=(size_t)nslots*c->n_experts*sizeof(float);
        size_t lb=(size_t)nslots*c->vocab*sizeof(float);
        if(coli_cuda_malloc(cuda_rt.ctx,(void**)&s->d_x,hb)||
           coli_cuda_malloc(cuda_rt.ctx,(void**)&s->d_n,hb)||
           coli_cuda_malloc(cuda_rt.ctx,(void**)&s->d_n16,hb/2)||
           coli_cuda_malloc(cuda_rt.ctx,(void**)&s->d_mix,hb)||
           coli_cuda_malloc(cuda_rt.ctx,(void**)&s->d_moe,hb)||
           coli_cuda_malloc(cuda_rt.ctx,(void**)&s->d_a,gb)||
           coli_cuda_malloc(cuda_rt.ctx,(void**)&s->d_b,gb)||
           coli_cuda_malloc(cuda_rt.ctx,(void**)&s->d_router,rb)||
           coli_cuda_malloc(cuda_rt.ctx,(void**)&s->d_scale,
                            (size_t)nslots*sizeof(float))||
           coli_cuda_malloc(cuda_rt.ctx,(void**)&s->d_logits,lb)){
            resident_batch_free(m,s);return 0;
        }
        fprintf(stderr,"[CUDA] resident residual stream enabled; "
                       "router metadata is the only per-layer host boundary "
                       "when routed experts are device-eligible\n");
    }else if(want_activations&&cuda_rt.active){
        fprintf(stderr,"[CUDA] resident residual stream unavailable: "
                       "f16=%d gdn=%d gdn-gate=%d gqa=%d\n",
                       cuda_rt.use_f16,missing_gdn,missing_gate,missing_gqa);
    }
#endif
    return 1;
}
static void resident_batch_free(Model*m,ResidentBatchState*s){
    if(!s)return;if(s->layer&&m)for(int li=0;li<m->c.n_layers;li++){ResidentLayerState*r=&s->layer[li];
#ifdef COLI_CUDA
        if(cuda_rt.ctx){coli_cuda_free(cuda_rt.ctx,r->d_conv);coli_cuda_free(cuda_rt.ctx,r->d_gdn);coli_cuda_free(cuda_rt.ctx,r->d_k);coli_cuda_free(cuda_rt.ctx,r->d_v);}
#endif
        free(r->conv);free(r->gdn);free(r->k);free(r->v);free(r->k16);free(r->v16);}
#ifdef COLI_CUDA
    if(cuda_rt.ctx){coli_cuda_free(cuda_rt.ctx,s->d_x);coli_cuda_free(cuda_rt.ctx,s->d_n);
        coli_cuda_free(cuda_rt.ctx,s->d_n16);
        coli_cuda_free(cuda_rt.ctx,s->d_mix);coli_cuda_free(cuda_rt.ctx,s->d_moe);
        coli_cuda_free(cuda_rt.ctx,s->d_a);coli_cuda_free(cuda_rt.ctx,s->d_b);
        coli_cuda_free(cuda_rt.ctx,s->d_router);coli_cuda_free(cuda_rt.ctx,s->d_scale);
        coli_cuda_free(cuda_rt.ctx,s->d_logits);}
#endif
    free(s->layer);free(s->pos);free(s->last_hidden);free(s->logits);memset(s,0,sizeof(*s));
}
static int resident_import_model_slot(Model*m,ResidentBatchState*s,int slot,const float*logits){
    if(!m||!s||slot<0||slot>=s->nslots||m->pos<0||m->pos>s->max_seq||s->kv16!=m->kv16)return 0;Cfg*c=&m->c;
    s->pos[slot]=m->pos;memcpy(s->last_hidden+(int64_t)slot*c->hidden,m->last_hidden,(size_t)c->hidden*sizeof(float));
    if(logits)memcpy(s->logits+(int64_t)slot*c->vocab,logits,(size_t)c->vocab*sizeof(float));
    int cd=2*c->lin_k_heads*c->lin_k_dim+c->lin_v_heads*c->lin_v_dim,kvrows=c->n_kv_heads*c->head_dim;
#ifdef COLI_CUDA
    int cuda_locked=cuda_rt.active,cuda_ok=1;if(cuda_locked)pthread_mutex_lock(&cuda_rt.lock);
#endif
    for(int li=0;li<c->n_layers;li++){ResidentLayerState*r=&s->layer[li];
        if(m->layer[li].type==LT_LINEAR){GdnW*w=&m->layer[li].gdn;size_t nc=(size_t)c->conv_kernel*cd,ns=(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim;memcpy(r->conv+(size_t)slot*nc,w->conv_state,nc*sizeof(float));memcpy(r->gdn+(size_t)slot*ns,w->state,ns*sizeof(float));
#ifdef COLI_CUDA
            if(cuda_locked&&r->cuda_gdn){float*dc=r->d_conv+(size_t)slot*nc,*ds=r->d_gdn+(size_t)slot*ns;if(w->cuda_aux_ready&&w->cuda_state_pos==m->pos){if(coli_cuda_copy(cuda_rt.ctx,dc,w->d_conv_state,nc*sizeof(float))||coli_cuda_copy(cuda_rt.ctx,ds,w->d_state,ns*sizeof(float)))cuda_ok=0;}else if(coli_cuda_upload(cuda_rt.ctx,dc,w->conv_state,nc*sizeof(float))||coli_cuda_upload(cuda_rt.ctx,ds,w->state,ns*sizeof(float)))cuda_ok=0;}
#endif
        }
        else{AttnW*w=&m->layer[li].attn;size_t n=(size_t)m->pos*kvrows,base=(size_t)slot*s->max_seq*kvrows;if(m->kv16){memcpy(r->k16+base,w->k_cache16,n*sizeof(uint16_t));memcpy(r->v16+base,w->v_cache16,n*sizeof(uint16_t));}else{
#ifdef COLI_CUDA
                int source_cuda=cuda_locked&&w->cuda_aux_ready&&w->cuda_state_pos==m->pos;
                if(source_cuda){if(n&&(coli_cuda_download(cuda_rt.ctx,r->k+base,w->d_k_cache,n*sizeof(float))||coli_cuda_download(cuda_rt.ctx,r->v+base,w->d_v_cache,n*sizeof(float))))cuda_ok=0;}
                else
#endif
                {memcpy(r->k+base,w->k_cache,n*sizeof(float));memcpy(r->v+base,w->v_cache,n*sizeof(float));}
#ifdef COLI_CUDA
                if(cuda_locked&&r->cuda_attn){float*dk=r->d_k+base,*dv=r->d_v+base;if(source_cuda){if(n&&(coli_cuda_copy(cuda_rt.ctx,dk,w->d_k_cache,n*sizeof(float))||coli_cuda_copy(cuda_rt.ctx,dv,w->d_v_cache,n*sizeof(float))))cuda_ok=0;}else if(n&&(coli_cuda_upload(cuda_rt.ctx,dk,w->k_cache,n*sizeof(float))||coli_cuda_upload(cuda_rt.ctx,dv,w->v_cache,n*sizeof(float))))cuda_ok=0;}
#endif
            }}
    }
#ifdef COLI_CUDA
    if(cuda_locked){if(coli_cuda_sync(cuda_rt.ctx))cuda_ok=0;pthread_mutex_unlock(&cuda_rt.lock);if(!cuda_ok)return 0;}
#endif
    return 1;
}
static int resident_forward_tokens(Model*m,ResidentBatchState*s,const int*slot,const int*token,int B){
    if(!m||!s||!slot||!token||B<1||B>s->nslots)return 0;Cfg*c=&m->c;
    float*x=falloc((int64_t)B*c->hidden),*n=falloc((int64_t)B*c->hidden),*bl=falloc((int64_t)B*c->vocab);int*pos=xcalloc(B,sizeof(int));
    unsigned char seen[MUX_MAX_SLOTS]={0};int ok=1,output_done=0;
    for(int row=0;row<B;row++){int sid=slot[row];if(sid<0||sid>=s->nslots||seen[sid]||token[row]<0||token[row]>=c->vocab||s->pos[sid]>=s->max_seq){ok=0;break;}seen[sid]=1;pos[row]=s->pos[sid];qmat_row(x+(int64_t)row*c->hidden,&m->embed,token[row]);}
#ifdef COLI_CUDA
    if(ok&&s->cuda_activations){
        float*moe=falloc((int64_t)B*c->hidden);
        pthread_mutex_lock(&cuda_rt.lock);
        if(coli_cuda_upload(cuda_rt.ctx,s->d_x,x,
                            (size_t)B*c->hidden*sizeof(float))){
            cuda_backend_fail("resident embedding boundary");ok=0;
        }else cuda_rt.resident_h2d+=(uint64_t)B*c->hidden*sizeof(float);
        pthread_mutex_unlock(&cuda_rt.lock);
        for(int li=0;ok&&li<c->n_layers;li++){
            Layer*l=&m->layer[li];ResidentLayerState*r=&s->layer[li];
            expert_predict_submit(m,&l->moe);
            int device_moe=cuda_resident_moe_eligible(m,l,B);
            double core_t0=m->prof_detail?now_s():0.;
            int used=cuda_resident_core_try(m,l,s,r,device_moe?NULL:n,
                                            B,slot,pos);
            if(used<=0){if(!used)fprintf(stderr,"[CUDA] resident activation "
                "eligibility changed at layer %d\n",li);ok=0;break;}
            if(m->prof_detail){if(l->type==LT_LINEAR)m->prof_gdn+=now_s()-core_t0;
                              else m->prof_attn+=now_s()-core_t0;}
            cuda_rt.resident_layers++;
            double moe_t0=m->prof_detail?now_s():0.;
            int device_used=device_moe?cuda_resident_moe_try(m,l,s,B):0;
            if(device_used<0){ok=0;break;}
            if(!device_used){
                if(device_moe){
                    pthread_mutex_lock(&cuda_rt.lock);
                    int boundary_error=coli_cuda_download(
                        cuda_rt.ctx,n,s->d_n,(size_t)B*c->hidden*sizeof(float))||
                        coli_cuda_sync(cuda_rt.ctx);
                    pthread_mutex_unlock(&cuda_rt.lock);
                    if(boundary_error){cuda_backend_fail(
                        "resident MoE fallback boundary");ok=0;break;}
                    cuda_rt.resident_d2h+=(uint64_t)B*c->hidden*sizeof(float);
                }
                moe_prefill_grouped(m,l,n,B,moe);
                if(!cuda_resident_moe_commit(m,s,moe,B)){ok=0;break;}
                cuda_rt.resident_h2d+=(uint64_t)B*c->hidden*sizeof(float);
                cuda_rt.resident_host_moe++;
            }
            if(m->prof_detail)m->prof_moe+=now_s()-moe_t0;
        }
        if(ok){
            double lm_t0=m->prof_detail?now_s():0.;
            int output_used=cuda_resident_output_try(m,s,n,bl,B);
            if(output_used<0)ok=0;
            else if(output_used>0)output_done=1;
            else{
                pthread_mutex_lock(&cuda_rt.lock);
                if(coli_cuda_download(cuda_rt.ctx,x,s->d_x,
                                      (size_t)B*c->hidden*sizeof(float))||
                   coli_cuda_sync(cuda_rt.ctx)){
                    cuda_backend_fail("resident final-norm boundary");ok=0;
                }else cuda_rt.resident_d2h+=(uint64_t)B*c->hidden*sizeof(float);
                pthread_mutex_unlock(&cuda_rt.lock);
            }
            if(m->prof_detail)m->prof_lm+=now_s()-lm_t0;
        }
        free(moe);
    }else
#endif
    for(int li=0;ok&&li<c->n_layers;li++){ResidentLayerState*r=&s->layer[li];layer_forward_slot_batch(m,&m->layer[li],x,B,slot,pos,r,s->nslots,r->conv,r->gdn,r->k,r->v,r->k16,r->v16,s->max_seq);}
    if(ok&&!output_done){
        #pragma omp parallel for schedule(static) if(B>1)
        for(int row=0;row<B;row++)rmsnorm_zero(n+(int64_t)row*c->hidden,x+(int64_t)row*c->hidden,m->final_norm,c->hidden,c->eps);
        double lm_t0=m->prof_detail?now_s():0.;qmat_mul_batch(bl,n,B,c->hidden,&m->lm_head,1);
        if(m->prof_detail)m->prof_lm+=now_s()-lm_t0;
    }
    if(ok){
        for(int row=0;row<B;row++){int sid=slot[row];memcpy(s->last_hidden+(int64_t)sid*c->hidden,n+(int64_t)row*c->hidden,(size_t)c->hidden*sizeof(float));memcpy(s->logits+(int64_t)sid*c->vocab,bl+(int64_t)row*c->vocab,(size_t)c->vocab*sizeof(float));s->pos[sid]++;}
    }
    free(x);free(n);free(bl);free(pos);return ok;
}
static void dump_hidden(Model *m,int li,const float *x){
    if(!m->dump_acts)return; const char *dir=getenv("ACTS_DIR");if(!dir)dir="acts"; mkdir(dir,0755); char p[2048];snprintf(p,sizeof(p),"%s/layer-%03d-token-%06d.f32",dir,li,m->pos);FILE*f=fopen(p,"wb");if(f){fwrite(x,sizeof(float),m->c.hidden,f);fclose(f);}
}
static void layer_forward_one(Model*m,Layer*l,float*x,int pos){
    Cfg*c=&m->c;int old_pos=m->pos;m->pos=pos;
    float*n=falloc(c->hidden),*mix=falloc(c->hidden),*moe=falloc(c->hidden);
    expert_predict_submit(m,&l->moe);
    rmsnorm_zero(n,x,l->input_norm,c->hidden,c->eps);
    if(l->type==LT_LINEAR)gdn_forward(m,l,n,mix);else attn_forward(m,l,n,mix);
    for(int i=0;i<c->hidden;i++)x[i]+=mix[i];
    rmsnorm_zero(n,x,l->post_norm,c->hidden,c->eps);moe_forward(m,l,n,moe);
    for(int i=0;i<c->hidden;i++)x[i]+=moe[i];
    m->pos=old_pos;free(n);free(mix);free(moe);
}
static int mtp_argmax(const float*x,int n){int b=0;for(int i=1;i<n;i++)if(x[i]>x[b])b=i;return b;}
/* One Qwen3.5 MTP step, matching vLLM's executable checkpoint equation:
 * fc([pre_fc_norm_embedding(embed(next)); pre_fc_norm_hidden(target_hidden)])
 * -> one full-attention+MoE decoder layer -> mtp.norm -> shared lm_head. */
static int mtp_step(Model*m,int next_token,const float*target_hidden,int pos,float*logits,float*out_hidden){
    if(!m->mtp.enabled)return -1;Cfg*c=&m->c;if(next_token<0||next_token>=c->vocab)return -1;
#ifdef COLI_CUDA
    int prior_cuda_suppress=cuda_suppress;const char*cuda_mtp=getenv("CUDA_MTP");
    if(cuda_mtp&&atoi(cuda_mtp)==0)cuda_suppress=1;
#endif
    float*embed=falloc(c->hidden),*hn=falloc(c->hidden),*cat=falloc(2*c->hidden),*x=falloc(c->hidden);
    qmat_row(embed,&m->embed,next_token);
    rmsnorm_zero(embed,embed,m->mtp.pre_embed_norm,c->hidden,c->eps);
    rmsnorm_zero(hn,target_hidden,m->mtp.pre_hidden_norm,c->hidden,c->eps);
    memcpy(cat,embed,(size_t)c->hidden*sizeof(float));memcpy(cat+c->hidden,hn,(size_t)c->hidden*sizeof(float));
    qmat_mul_ex(x,cat,&m->mtp.fc,1);layer_forward_one(m,&m->mtp.layer,x,pos);
    rmsnorm_zero(hn,x,m->mtp.norm,c->hidden,c->eps);qmat_mul_ex(logits,hn,&m->lm_head,1);
    if(out_hidden)memcpy(out_hidden,hn,(size_t)c->hidden*sizeof(float));
    int pred=mtp_argmax(logits,c->vocab);free(embed);free(hn);free(cat);free(x);
#ifdef COLI_CUDA
    cuda_suppress=prior_cuda_suppress;
#endif
    return pred;
}
static void forward_token(Model *m,int token,float *logits){
    Cfg *c=&m->c; if(token<0||token>=c->vocab){fprintf(stderr,"token %d outside model vocab %d\n",token,c->vocab);exit(1);} if(m->pos>=m->max_seq)die("CTX exhausted");
    float *x=falloc(c->hidden),*n=falloc(c->hidden),*mix=falloc(c->hidden),*moe=falloc(c->hidden); qmat_row(x,&m->embed,token);
    for(int li=0;li<c->n_layers;li++){
        Layer *l=&m->layer[li];expert_predict_submit(m,&l->moe);rmsnorm_zero(n,x,l->input_norm,c->hidden,c->eps);double t0=m->prof_detail?now_s():0.;if(l->type==LT_LINEAR)gdn_forward(m,l,n,mix);else attn_forward(m,l,n,mix);if(m->prof_detail){if(l->type==LT_LINEAR)m->prof_gdn+=now_s()-t0;else m->prof_attn+=now_s()-t0;}
        for(int i=0;i<c->hidden;i++)x[i]+=mix[i]; rmsnorm_zero(n,x,l->post_norm,c->hidden,c->eps);t0=m->prof_detail?now_s():0.;moe_forward(m,l,n,moe);if(m->prof_detail)m->prof_moe+=now_s()-t0;for(int i=0;i<c->hidden;i++)x[i]+=moe[i]; dump_hidden(m,li,x);
    }
    rmsnorm_zero(n,x,m->final_norm,c->hidden,c->eps);memcpy(m->last_hidden,n,(size_t)c->hidden*sizeof(float));double t0=m->prof_detail?now_s():0.;qmat_mul_ex(logits,n,&m->lm_head,1);if(m->prof_detail)m->prof_lm+=now_s()-t0;m->pos++;free(x);free(n);free(mix);free(moe);
}
static int use_gdn_chunk(void){const char*e=getenv("GDN_CHUNK");return !e||atoi(e)!=0;}
static void gdn_prefill_layer(Model*m,Layer*l,const float*x,int T,float*out){
    Cfg*c=&m->c;GdnW*w=&l->gdn;int kh=c->lin_k_heads,vh=c->lin_v_heads,dk=c->lin_k_dim,dv=c->lin_v_dim,kd=kh*dk,vd=vh*dv,cd=2*kd+vd,K=c->conv_kernel,ratio=vh/kh;
    float*raw=falloc((int64_t)T*cd),*mix=falloc((int64_t)T*cd),*z=falloc((int64_t)T*vd),*aa=falloc((int64_t)T*vh),*bb=falloc((int64_t)T*vh),*q=falloc((int64_t)T*vh*dk),*k=falloc((int64_t)T*vh*dk),*v=falloc((int64_t)T*vd),*gg=falloc((int64_t)T*vh),*beta=falloc((int64_t)T*vh),*core=falloc((int64_t)T*vd);
    int par=parallel_work((int64_t)T*c->hidden*cd);
    #pragma omp parallel for schedule(static) if(par)
    for(int t=0;t<T;t++){qmat_mul_ex(raw+(int64_t)t*cd,x+(int64_t)t*c->hidden,&w->qkv,0);qmat_mul_ex(z+(int64_t)t*vd,x+(int64_t)t*c->hidden,&w->z,0);qmat_mul_ex(bb+(int64_t)t*vh,x+(int64_t)t*c->hidden,&w->b,0);qmat_mul_ex(aa+(int64_t)t*vh,x+(int64_t)t*c->hidden,&w->a,0);}
    for(int t=0;t<T;t++){memcpy(w->conv_state+(int64_t)(t%K)*cd,raw+(int64_t)t*cd,(size_t)cd*sizeof(float));for(int ch=0;ch<cd;ch++){float acc=0.f;for(int tap=0;tap<K;tap++){int src=t-(K-1-tap);if(src>=0)acc+=w->conv[(int64_t)ch*K+tap]*raw[(int64_t)src*cd+ch];}mix[(int64_t)t*cd+ch]=siluf(acc);}for(int h=0;h<vh;h++){int hk=h/ratio;memcpy(q+((int64_t)t*vh+h)*dk,mix+(int64_t)t*cd+(int64_t)hk*dk,(size_t)dk*sizeof(float));memcpy(k+((int64_t)t*vh+h)*dk,mix+(int64_t)t*cd+kd+(int64_t)hk*dk,(size_t)dk*sizeof(float));memcpy(v+((int64_t)t*vh+h)*dv,mix+(int64_t)t*cd+2*kd+(int64_t)h*dv,(size_t)dv*sizeof(float));gg[(int64_t)t*vh+h]=-expf(w->A_log[h])*softplusf_stable(aa[(int64_t)t*vh+h]+w->dt_bias[h]);beta[(int64_t)t*vh+h]=sigmoidf_stable(bb[(int64_t)t*vh+h]);}}
    if(use_gdn_chunk())gdn_prefill_chunked(core,w->state,q,k,v,gg,beta,T,vh,dk,dv);else gdn_prefill_seq(core,w->state,q,k,v,gg,beta,T,vh,dk,dv);
    for(int t=0;t<T;t++){for(int h=0;h<vh;h++){float*co=core+((int64_t)t*vh+h)*dv;float ms=0.f;for(int j=0;j<dv;j++)ms+=co[j]*co[j];float r=1.f/sqrtf(ms/(float)dv+c->eps);for(int j=0;j<dv;j++)co[j]=co[j]*r*w->norm[j]*siluf(z[((int64_t)t*vh+h)*dv+j]);}qmat_mul_ex(out+(int64_t)t*c->hidden,core+(int64_t)t*vd,&w->out,0);}
    free(raw);free(mix);free(z);free(aa);free(bb);free(q);free(k);free(v);free(gg);free(beta);free(core);
}

/* Continue a GDN layer from its current recurrent/causal-convolution state for
 * a short speculative block.  Unlike the fresh-prompt prefill above, ring
 * indices are absolute and the existing state is deliberately preserved. */
static void gdn_decode_block(Model*m,Layer*l,const float*x,int T,int base,float*out){
    Cfg*c=&m->c;GdnW*w=&l->gdn;int kh=c->lin_k_heads,vh=c->lin_v_heads,dk=c->lin_k_dim,dv=c->lin_v_dim,kd=kh*dk,vd=vh*dv,cd=2*kd+vd,K=c->conv_kernel,ratio=vh/kh;
#ifdef COLI_CUDA
    if(cuda_spec_gdn_enabled()&&cuda_gdn_eligible(w)){
        int use_block=cuda_spec_gdn_batch_enabled();float*aa=falloc((int64_t)T*vh),*bb=falloc((int64_t)T*vh);
        if(use_block){qmat_mul_batch(aa,x,T,c->hidden,&w->a,0);qmat_mul_batch(bb,x,T,c->hidden,&w->b,0);}
        else for(int t=0;t<T;t++){qmat_mul_ex(aa+(int64_t)t*vh,x+(int64_t)t*c->hidden,&w->a,0);qmat_mul_ex(bb+(int64_t)t*vh,x+(int64_t)t*c->hidden,&w->b,0);}
        int prior=cuda_suppress;cuda_suppress=0;
        int block_used=use_block&&cuda_gdn_block_try(out,x,aa,bb,T,w,c,base);
        if(!block_used)for(int t=0;t<T;t++)if(!cuda_gdn_try(out+(int64_t)t*c->hidden,x+(int64_t)t*c->hidden,aa+(int64_t)t*vh,bb+(int64_t)t*vh,w,c,base+t))die("CUDA speculative GDN failed");
        cuda_suppress=prior;free(aa);free(bb);return;
    }
#endif
    float*raw=falloc((int64_t)T*cd),*mix=falloc((int64_t)T*cd),*z=falloc((int64_t)T*vd),*aa=falloc((int64_t)T*vh),*bb=falloc((int64_t)T*vh),*q=falloc((int64_t)T*vh*dk),*k=falloc((int64_t)T*vh*dk),*v=falloc((int64_t)T*vd),*gg=falloc((int64_t)T*vh),*beta=falloc((int64_t)T*vh),*core=falloc((int64_t)T*vd);
    qmat_mul_batch(raw,x,T,c->hidden,&w->qkv,0);qmat_mul_batch(z,x,T,c->hidden,&w->z,0);qmat_mul_batch(aa,x,T,c->hidden,&w->a,0);qmat_mul_batch(bb,x,T,c->hidden,&w->b,0);
    for(int t=0;t<T;t++){int pos=base+t;memcpy(w->conv_state+(int64_t)(pos%K)*cd,raw+(int64_t)t*cd,(size_t)cd*sizeof(float));for(int ch=0;ch<cd;ch++){float acc=0.f;for(int tap=0;tap<K;tap++){int src=pos-(K-1-tap);if(src>=0)acc+=w->conv[(int64_t)ch*K+tap]*w->conv_state[(int64_t)(src%K)*cd+ch];}mix[(int64_t)t*cd+ch]=siluf(acc);}for(int h=0;h<vh;h++){int hk=h/ratio;memcpy(q+((int64_t)t*vh+h)*dk,mix+(int64_t)t*cd+(int64_t)hk*dk,(size_t)dk*sizeof(float));memcpy(k+((int64_t)t*vh+h)*dk,mix+(int64_t)t*cd+kd+(int64_t)hk*dk,(size_t)dk*sizeof(float));memcpy(v+((int64_t)t*vh+h)*dv,mix+(int64_t)t*cd+2*kd+(int64_t)h*dv,(size_t)dv*sizeof(float));gg[(int64_t)t*vh+h]=-expf(w->A_log[h])*softplusf_stable(aa[(int64_t)t*vh+h]+w->dt_bias[h]);beta[(int64_t)t*vh+h]=sigmoidf_stable(bb[(int64_t)t*vh+h]);}}
    gdn_prefill_seq(core,w->state,q,k,v,gg,beta,T,vh,dk,dv);
    for(int t=0;t<T;t++)for(int h=0;h<vh;h++){float*co=core+((int64_t)t*vh+h)*dv;float ms=0.f;for(int j=0;j<dv;j++)ms+=co[j]*co[j];float r=1.f/sqrtf(ms/(float)dv+c->eps);for(int j=0;j<dv;j++)co[j]=co[j]*r*w->norm[j]*siluf(z[((int64_t)t*vh+h)*dv+j]);}
    qmat_mul_batch(out,core,T,vd,&w->out,0);
    free(raw);free(mix);free(z);free(aa);free(bb);free(q);free(k);free(v);free(gg);free(beta);free(core);
}

/* Target-model block verification.  The returned row t is the distribution
 * after consuming token[t], and hidden[t] is its post-final-norm state. */
static void forward_decode_block(Model*m,const int*token,int T,float*logits,float*hidden){
    Cfg*c=&m->c;if(T<1||m->pos+T>m->max_seq)die("invalid speculative target block");int base=m->pos;
    float*x=falloc((int64_t)T*c->hidden),*n=falloc((int64_t)T*c->hidden),*mix=falloc((int64_t)T*c->hidden);
    for(int t=0;t<T;t++){if(token[t]<0||token[t]>=c->vocab)die("speculative token outside vocab");qmat_row(x+(int64_t)t*c->hidden,&m->embed,token[t]);}
    for(int li=0;li<c->n_layers;li++){
        Layer*l=&m->layer[li];expert_predict_submit(m,&l->moe);for(int t=0;t<T;t++)rmsnorm_zero(n+(int64_t)t*c->hidden,x+(int64_t)t*c->hidden,l->input_norm,c->hidden,c->eps);
        double core_t0=m->prof_detail?now_s():0.;if(l->type==LT_LINEAR)gdn_decode_block(m,l,n,T,base,mix);else{
#ifdef COLI_CUDA
            int prior=cuda_suppress;if(cuda_spec_full_enabled())cuda_suppress=0;
#endif
            for(int t=0;t<T;t++){m->pos=base+t;attn_forward(m,l,n+(int64_t)t*c->hidden,mix+(int64_t)t*c->hidden);}
#ifdef COLI_CUDA
            cuda_suppress=prior;
#endif
        }if(m->prof_detail){if(l->type==LT_LINEAR)m->prof_gdn+=now_s()-core_t0;else m->prof_attn+=now_s()-core_t0;}
        for(int t=0;t<T;t++){float*xt=x+(int64_t)t*c->hidden;for(int i=0;i<c->hidden;i++)xt[i]+=mix[(int64_t)t*c->hidden+i];rmsnorm_zero(n+(int64_t)t*c->hidden,xt,l->post_norm,c->hidden,c->eps);}
        double moe_t0=m->prof_detail?now_s():0.;
#ifdef COLI_CUDA
        if(cuda_spec_full_enabled()){
            int prior=cuda_suppress;cuda_suppress=0;int batch_used=moe_cuda_block(m,l,n,T,mix);
            if(!batch_used)for(int t=0;t<T;t++)moe_forward(m,l,n+(int64_t)t*c->hidden,mix+(int64_t)t*c->hidden);
            const char*check_env=getenv("CUDA_SPEC_MOE_CHECK"),*exact_env=getenv("CUDA_SPEC_MOE_EXACT");
            int check=check_env&&atoi(check_env)!=0,exact=exact_env&&atoi(exact_env)!=0;
            if(batch_used&&(check||exact)){float*ref=falloc((int64_t)T*c->hidden);int suppress=m->route_record_suppress;m->route_record_suppress=1;for(int t=0;t<T;t++)moe_forward(m,l,n+(int64_t)t*c->hidden,ref+(int64_t)t*c->hidden);m->route_record_suppress=suppress;float md=0.f;for(int64_t i=0;i<(int64_t)T*c->hidden;i++){float d=fabsf(mix[i]-ref[i]);if(d>md)md=d;}if(check)fprintf(stderr,"[CUDA_MOE_CHECK] layer=%d batch=%d maxdiff=%.8g\n",li,T,md);if(exact)memcpy(mix,ref,(size_t)T*c->hidden*sizeof(float));free(ref);}
            cuda_suppress=prior;
        }else
#endif
        moe_prefill_grouped(m,l,n,T,mix);if(m->prof_detail)m->prof_moe+=now_s()-moe_t0;for(int64_t i=0;i<(int64_t)T*c->hidden;i++)x[i]+=mix[i];
    }
    for(int t=0;t<T;t++)rmsnorm_zero(n+(int64_t)t*c->hidden,x+(int64_t)t*c->hidden,m->final_norm,c->hidden,c->eps);
    double lm_t0=m->prof_detail?now_s():0.;
#ifdef COLI_CUDA
    if(cuda_spec_full_enabled()){int prior=cuda_suppress;cuda_suppress=0;for(int t=0;t<T;t++)qmat_mul_ex(logits+(int64_t)t*c->vocab,n+(int64_t)t*c->hidden,&m->lm_head,1);cuda_suppress=prior;}else
#endif
    qmat_mul_batch(logits,n,T,c->hidden,&m->lm_head,1);if(m->prof_detail)m->prof_lm+=now_s()-lm_t0;if(hidden)memcpy(hidden,n,(size_t)T*c->hidden*sizeof(float));memcpy(m->last_hidden,n+(int64_t)(T-1)*c->hidden,(size_t)c->hidden*sizeof(float));m->pos=base+T;
    free(x);free(n);free(mix);
}

static void target_forward_one(Model*m,int token,float*logits,int exact_cpu){
#ifdef COLI_CUDA
    int prior=cuda_suppress;if(exact_cpu)cuda_suppress=1;
#else
    (void)exact_cpu;
#endif
    forward_token(m,token,logits);
#ifdef COLI_CUDA
    cuda_suppress=prior;
#endif
}
static void target_forward_block(Model*m,const int*token,int T,float*logits){
#ifdef COLI_CUDA
    int prior=cuda_suppress;cuda_suppress=1;
#endif
    forward_decode_block(m,token,T,logits,NULL);
#ifdef COLI_CUDA
    cuda_suppress=prior;
#endif
}

static size_t recurrent_state_floats(Model*m){Cfg*c=&m->c;size_t n=0;int cd=2*c->lin_k_heads*c->lin_k_dim+c->lin_v_heads*c->lin_v_dim;for(int li=0;li<c->n_layers;li++)if(m->layer[li].type==LT_LINEAR)n+=(size_t)c->conv_kernel*cd+(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim;return n;}
static void recurrent_state_copy(Model*m,float*buf,int restore){Cfg*c=&m->c;size_t off=0;int cd=2*c->lin_k_heads*c->lin_k_dim+c->lin_v_heads*c->lin_v_dim;for(int li=0;li<c->n_layers;li++)if(m->layer[li].type==LT_LINEAR){GdnW*w=&m->layer[li].gdn;size_t nc=(size_t)c->conv_kernel*cd,ns=(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim;if(restore){memcpy(w->conv_state,buf+off,nc*sizeof(float));memcpy(w->state,buf+off+nc,ns*sizeof(float));}else{memcpy(buf+off,w->conv_state,nc*sizeof(float));memcpy(buf+off+nc,w->state,ns*sizeof(float));}off+=nc+ns;}}
static size_t session_kv_scalars(Model*m,int pos){size_t n=0;int rows=m->c.n_kv_heads*m->c.head_dim;for(int li=0;li<m->c.n_layers;li++)if(m->layer[li].type==LT_FULL)n+=(size_t)2*pos*rows;return n;}
static void session_state_free(SessionState*s){if(!s)return;free(s->recurrent);free(s->kv);free(s->kv_bf16);free(s->last_hidden);memset(s,0,sizeof(*s));}
static int session_buffer_reserve(void**ptr,size_t*cap,size_t need,size_t elem){
    if(need<=*cap)return 1;if(need>SIZE_MAX/elem)return 0;size_t next=*cap?*cap:1;
    while(next<need){if(next>SIZE_MAX/2){next=need;break;}next*=2;}
    if(next>SIZE_MAX/elem)return 0;void*p=realloc(*ptr,next*elem);if(!p)return 0;*ptr=p;*cap=next;return 1;
}
static int session_state_reserve(Model*m,SessionState*s,size_t rn,size_t kn){
    if(s->kv16!=m->kv16&&(s->kv||s->kv_bf16)){free(s->kv);free(s->kv_bf16);s->kv=NULL;s->kv_bf16=NULL;s->kv_cap=0;}
    if(!session_buffer_reserve((void**)&s->recurrent,&s->recurrent_cap,rn,sizeof(float)))return 0;
    if(!session_buffer_reserve((void**)&s->last_hidden,&s->hidden_cap,(size_t)m->c.hidden,sizeof(float)))return 0;
    return m->kv16?session_buffer_reserve((void**)&s->kv_bf16,&s->kv_cap,kn,sizeof(uint16_t)):
        session_buffer_reserve((void**)&s->kv,&s->kv_cap,kn,sizeof(float));
}
static int session_state_save(Model*m,SessionState*s){
    if(!m||!s||m->mtp.enabled)return 0;
    size_t rn=recurrent_state_floats(m),kn=session_kv_scalars(m,m->pos);
    if(!session_state_reserve(m,s,rn,kn))return 0;
    s->pos=m->pos;s->hidden=m->c.hidden;s->layers=m->c.n_layers;s->kv16=m->kv16;s->recurrent_n=rn;s->kv_n=kn;
    memcpy(s->last_hidden,m->last_hidden,(size_t)m->c.hidden*sizeof(float));
    size_t roff=0,koff=0;int rows=m->c.n_kv_heads*m->c.head_dim,ok=1,cd=2*m->c.lin_k_heads*m->c.lin_k_dim+m->c.lin_v_heads*m->c.lin_v_dim;
#ifdef COLI_CUDA
    if(cuda_rt.active)pthread_mutex_lock(&cuda_rt.lock);
#endif
    for(int li=0;li<m->c.n_layers;li++)if(m->layer[li].type==LT_LINEAR){GdnW*w=&m->layer[li].gdn;size_t nc=(size_t)m->c.conv_kernel*cd,ns=(size_t)m->c.lin_v_heads*m->c.lin_k_dim*m->c.lin_v_dim;
#ifdef COLI_CUDA
        if(cuda_rt.active&&w->cuda_aux_ready&&w->cuda_state_pos==m->pos){if(coli_cuda_download(cuda_rt.ctx,s->recurrent+roff,w->d_conv_state,nc*sizeof(float))||coli_cuda_download(cuda_rt.ctx,s->recurrent+roff+nc,w->d_state,ns*sizeof(float)))ok=0;}
        else
#endif
        {memcpy(s->recurrent+roff,w->conv_state,nc*sizeof(float));memcpy(s->recurrent+roff+nc,w->state,ns*sizeof(float));}roff+=nc+ns;
    }else{AttnW*w=&m->layer[li].attn;size_t n=(size_t)m->pos*rows;if(m->kv16){memcpy(s->kv_bf16+koff,w->k_cache16,n*sizeof(uint16_t));memcpy(s->kv_bf16+koff+n,w->v_cache16,n*sizeof(uint16_t));}else{
#ifdef COLI_CUDA
        if(cuda_rt.active&&w->cuda_aux_ready&&w->cuda_state_pos==m->pos){if(coli_cuda_download(cuda_rt.ctx,s->kv+koff,w->d_k_cache,n*sizeof(float))||coli_cuda_download(cuda_rt.ctx,s->kv+koff+n,w->d_v_cache,n*sizeof(float)))ok=0;}
        else
#endif
        {memcpy(s->kv+koff,w->k_cache,n*sizeof(float));memcpy(s->kv+koff+n,w->v_cache,n*sizeof(float));}}koff+=2*n;}
#ifdef COLI_CUDA
    if(cuda_rt.active){if(coli_cuda_sync(cuda_rt.ctx))ok=0;pthread_mutex_unlock(&cuda_rt.lock);}
#endif
    return ok&&roff==rn&&koff==kn;
}
static int session_state_restore(Model*m,const SessionState*s,float*logits){
    if(!m||!s||m->mtp.enabled||s->pos<0||s->pos>m->max_seq||s->hidden!=m->c.hidden||s->layers!=m->c.n_layers||s->kv16!=m->kv16||s->recurrent_n!=recurrent_state_floats(m)||s->kv_n!=session_kv_scalars(m,s->pos))return 0;
    model_reset(m);memcpy(m->last_hidden,s->last_hidden,(size_t)m->c.hidden*sizeof(float));
    size_t roff=0,koff=0;int rows=m->c.n_kv_heads*m->c.head_dim,ok=1,cd=2*m->c.lin_k_heads*m->c.lin_k_dim+m->c.lin_v_heads*m->c.lin_v_dim;
#ifdef COLI_CUDA
    if(cuda_rt.active)pthread_mutex_lock(&cuda_rt.lock);
#endif
    for(int li=0;li<m->c.n_layers;li++)if(m->layer[li].type==LT_LINEAR){GdnW*w=&m->layer[li].gdn;size_t nc=(size_t)m->c.conv_kernel*cd,ns=(size_t)m->c.lin_v_heads*m->c.lin_k_dim*m->c.lin_v_dim;memcpy(w->conv_state,s->recurrent+roff,nc*sizeof(float));memcpy(w->state,s->recurrent+roff+nc,ns*sizeof(float));
#ifdef COLI_CUDA
        if(cuda_rt.active&&w->cuda_aux_ready){if(coli_cuda_upload(cuda_rt.ctx,w->d_conv_state,s->recurrent+roff,nc*sizeof(float))||coli_cuda_upload(cuda_rt.ctx,w->d_state,s->recurrent+roff+nc,ns*sizeof(float)))ok=0;else w->cuda_state_pos=s->pos;}
#endif
        roff+=nc+ns;
    }else{AttnW*w=&m->layer[li].attn;size_t n=(size_t)s->pos*rows;if(m->kv16){memcpy(w->k_cache16,s->kv_bf16+koff,n*sizeof(uint16_t));memcpy(w->v_cache16,s->kv_bf16+koff+n,n*sizeof(uint16_t));}else{memcpy(w->k_cache,s->kv+koff,n*sizeof(float));memcpy(w->v_cache,s->kv+koff+n,n*sizeof(float));
#ifdef COLI_CUDA
        if(cuda_rt.active&&w->cuda_aux_ready){if(coli_cuda_upload(cuda_rt.ctx,w->d_k_cache,s->kv+koff,n*sizeof(float))||coli_cuda_upload(cuda_rt.ctx,w->d_v_cache,s->kv+koff+n,n*sizeof(float)))ok=0;else w->cuda_state_pos=s->pos;}
#endif
        }koff+=2*n;}
#ifdef COLI_CUDA
    if(cuda_rt.active){if(coli_cuda_sync(cuda_rt.ctx))ok=0;pthread_mutex_unlock(&cuda_rt.lock);}
#endif
    m->pos=s->pos;if(ok&&logits)qmat_mul_ex(logits,m->last_hidden,&m->lm_head,1);return ok&&roff==s->recurrent_n&&koff==s->kv_n;
}
static int resident_export_session(Model*m,const ResidentBatchState*r,int slot,SessionState*s){
    if(!m||!r||!s||slot<0||slot>=r->nslots||r->kv16!=m->kv16)return 0;Cfg*c=&m->c;
    size_t rn=recurrent_state_floats(m),kn=session_kv_scalars(m,r->pos[slot]);if(!session_state_reserve(m,s,rn,kn))return 0;
    s->pos=r->pos[slot];s->hidden=c->hidden;s->layers=c->n_layers;s->kv16=m->kv16;s->recurrent_n=rn;s->kv_n=kn;
    memcpy(s->last_hidden,r->last_hidden+(int64_t)slot*c->hidden,(size_t)c->hidden*sizeof(float));
    size_t roff=0,koff=0;int cd=2*c->lin_k_heads*c->lin_k_dim+c->lin_v_heads*c->lin_v_dim,kvrows=c->n_kv_heads*c->head_dim;
#ifdef COLI_CUDA
    int cuda_locked=cuda_rt.active,cuda_ok=1;if(cuda_locked)pthread_mutex_lock(&cuda_rt.lock);
#endif
    for(int li=0;li<c->n_layers;li++){const ResidentLayerState*lr=&r->layer[li];
        if(m->layer[li].type==LT_LINEAR){size_t nc=(size_t)c->conv_kernel*cd,ns=(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim;
#ifdef COLI_CUDA
            if(cuda_locked&&lr->cuda_gdn){if(coli_cuda_download(cuda_rt.ctx,s->recurrent+roff,lr->d_conv+(size_t)slot*nc,nc*sizeof(float))||coli_cuda_download(cuda_rt.ctx,s->recurrent+roff+nc,lr->d_gdn+(size_t)slot*ns,ns*sizeof(float)))cuda_ok=0;}
            else
#endif
            {memcpy(s->recurrent+roff,lr->conv+(size_t)slot*nc,nc*sizeof(float));memcpy(s->recurrent+roff+nc,lr->gdn+(size_t)slot*ns,ns*sizeof(float));}roff+=nc+ns;}
        else{size_t n=(size_t)s->pos*kvrows,base=(size_t)slot*r->max_seq*kvrows;if(s->kv16){memcpy(s->kv_bf16+koff,lr->k16+base,n*sizeof(uint16_t));memcpy(s->kv_bf16+koff+n,lr->v16+base,n*sizeof(uint16_t));}else{
#ifdef COLI_CUDA
            if(cuda_locked&&lr->cuda_attn){if(n&&(coli_cuda_download(cuda_rt.ctx,s->kv+koff,lr->d_k+base,n*sizeof(float))||coli_cuda_download(cuda_rt.ctx,s->kv+koff+n,lr->d_v+base,n*sizeof(float))))cuda_ok=0;}
            else
#endif
            {memcpy(s->kv+koff,lr->k+base,n*sizeof(float));memcpy(s->kv+koff+n,lr->v+base,n*sizeof(float));}}koff+=2*n;}
    }
#ifdef COLI_CUDA
    if(cuda_locked){if(coli_cuda_sync(cuda_rt.ctx))cuda_ok=0;pthread_mutex_unlock(&cuda_rt.lock);if(!cuda_ok)return 0;}
#endif
    return roff==rn&&koff==kn;
}
static int resident_import_session(Model*m,ResidentBatchState*r,int slot,const SessionState*s){
    if(!m||!r||!s||slot<0||slot>=r->nslots||s->pos<0||s->pos>r->max_seq||s->hidden!=m->c.hidden||s->layers!=m->c.n_layers||s->kv16!=r->kv16||s->recurrent_n!=recurrent_state_floats(m)||s->kv_n!=session_kv_scalars(m,s->pos))return 0;Cfg*c=&m->c;
    r->pos[slot]=s->pos;memcpy(r->last_hidden+(int64_t)slot*c->hidden,s->last_hidden,(size_t)c->hidden*sizeof(float));
    size_t roff=0,koff=0;int cd=2*c->lin_k_heads*c->lin_k_dim+c->lin_v_heads*c->lin_v_dim,kvrows=c->n_kv_heads*c->head_dim;
#ifdef COLI_CUDA
    int cuda_locked=cuda_rt.active,cuda_ok=1;if(cuda_locked)pthread_mutex_lock(&cuda_rt.lock);
#endif
    for(int li=0;li<c->n_layers;li++){ResidentLayerState*lr=&r->layer[li];
        if(m->layer[li].type==LT_LINEAR){size_t nc=(size_t)c->conv_kernel*cd,ns=(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim;memcpy(lr->conv+(size_t)slot*nc,s->recurrent+roff,nc*sizeof(float));memcpy(lr->gdn+(size_t)slot*ns,s->recurrent+roff+nc,ns*sizeof(float));
#ifdef COLI_CUDA
            if(cuda_locked&&lr->cuda_gdn&&(coli_cuda_upload(cuda_rt.ctx,lr->d_conv+(size_t)slot*nc,s->recurrent+roff,nc*sizeof(float))||coli_cuda_upload(cuda_rt.ctx,lr->d_gdn+(size_t)slot*ns,s->recurrent+roff+nc,ns*sizeof(float))))cuda_ok=0;
#endif
            roff+=nc+ns;}
        else{size_t n=(size_t)s->pos*kvrows,base=(size_t)slot*r->max_seq*kvrows;if(s->kv16){memcpy(lr->k16+base,s->kv_bf16+koff,n*sizeof(uint16_t));memcpy(lr->v16+base,s->kv_bf16+koff+n,n*sizeof(uint16_t));}else{memcpy(lr->k+base,s->kv+koff,n*sizeof(float));memcpy(lr->v+base,s->kv+koff+n,n*sizeof(float));
#ifdef COLI_CUDA
            if(cuda_locked&&lr->cuda_attn&&n&&(coli_cuda_upload(cuda_rt.ctx,lr->d_k+base,s->kv+koff,n*sizeof(float))||coli_cuda_upload(cuda_rt.ctx,lr->d_v+base,s->kv+koff+n,n*sizeof(float))))cuda_ok=0;
#endif
        }koff+=2*n;}
    }
#ifdef COLI_CUDA
    if(cuda_locked){if(coli_cuda_sync(cuda_rt.ctx))cuda_ok=0;pthread_mutex_unlock(&cuda_rt.lock);if(!cuda_ok)return 0;}
#endif
    if(roff!=s->recurrent_n||koff!=s->kv_n)return 0;qmat_mul_ex(r->logits+(int64_t)slot*c->vocab,r->last_hidden+(int64_t)slot*c->hidden,&m->lm_head,1);return 1;
}
/* Extend one resident slot through the existing block-verification path.
 * A chat continuation commonly contributes 10-100 uncached template/user
 * tokens.  Running those as independent decode steps throws away the same
 * grouped projection/MoE work that makes fresh prefill fast.  Exporting one
 * slot, consuming the suffix as a causal block, and importing it back keeps
 * the exact recurrent/KV state while amortizing matrix-launch overhead. */
static int resident_forward_suffix(Model*m,ResidentBatchState*r,int slot,
                                   const int*token,int T){
    if(T<1)return 1;
    if(T==1)return resident_forward_tokens(m,r,&slot,token,1);
    SessionState state={0};float*logits=NULL;int ok=resident_export_session(m,r,slot,&state);
    if(ok)ok=session_state_restore(m,&state,NULL);
    if(ok){
        logits=falloc((int64_t)T*m->c.vocab);
        forward_decode_block(m,token,T,logits,NULL);
        ok=resident_import_model_slot(m,r,slot,logits+(int64_t)(T-1)*m->c.vocab);
    }
    free(logits);session_state_free(&state);return ok;
}
static uint64_t session_hash_add(uint64_t h,const void*data,size_t n){const unsigned char*p=data;for(size_t i=0;i<n;i++){h^=p[i];h*=1099511628211ULL;}return h;}
static uint64_t session_state_checksum(const SessionState*s){
    uint64_t h=1469598103934665603ULL;h=session_hash_add(h,s->recurrent,s->recurrent_n*sizeof(float));
    if(s->kv16)h=session_hash_add(h,s->kv_bf16,s->kv_n*sizeof(uint16_t));else h=session_hash_add(h,s->kv,s->kv_n*sizeof(float));
    return session_hash_add(h,s->last_hidden,(size_t)s->hidden*sizeof(float));
}
static int session_state_write(const char*path,const SessionState*s){
    if(!path||!s||s->hidden<=0||s->layers<=0||s->pos<0||s->recurrent_n>SIZE_MAX/sizeof(float)||s->kv_n>SIZE_MAX/(s->kv16?sizeof(uint16_t):sizeof(float)))return 0;
    SessionDiskHeader h={{'C','O','L','I','S','E','S','S'},1,(uint32_t)s->hidden,(uint32_t)s->layers,(uint32_t)s->kv16,(uint32_t)s->pos,0,(uint64_t)s->recurrent_n,(uint64_t)s->kv_n,session_state_checksum(s)};
    char tmp[2304];snprintf(tmp,sizeof(tmp),"%s.tmp.%ld",path,(long)getpid());FILE*f=fopen(tmp,"wb");if(!f)return 0;int ok=fwrite(&h,1,sizeof(h),f)==sizeof(h)&&fwrite(s->recurrent,sizeof(float),s->recurrent_n,f)==s->recurrent_n;
    if(ok)ok=s->kv16?fwrite(s->kv_bf16,sizeof(uint16_t),s->kv_n,f)==s->kv_n:fwrite(s->kv,sizeof(float),s->kv_n,f)==s->kv_n;
    if(ok)ok=fwrite(s->last_hidden,sizeof(float),(size_t)s->hidden,f)==(size_t)s->hidden&&fflush(f)==0&&fsync(fileno(f))==0;
    if(fclose(f)!=0)ok=0;if(ok&&rename(tmp,path)==0){char dir[2304];snprintf(dir,sizeof(dir),"%s",path);char*slash=strrchr(dir,'/');if(slash){if(slash==dir)slash[1]=0;else*slash=0;}else snprintf(dir,sizeof(dir),".");int fd=open(dir,O_RDONLY|O_DIRECTORY);if(fd>=0){int synced=fsync(fd)==0;close(fd);return synced;}return 0;}unlink(tmp);return 0;
}
static int session_state_read(Model*m,const char*path,SessionState*s){
    if(!m||!path||!s||m->mtp.enabled)return 0;FILE*f=fopen(path,"rb");if(!f)return 0;SessionDiskHeader h;int ok=fread(&h,1,sizeof(h),f)==sizeof(h)&&!memcmp(h.magic,"COLISESS",8)&&h.version==1&&h.hidden==(uint32_t)m->c.hidden&&h.layers==(uint32_t)m->c.n_layers&&h.kv16==(uint32_t)m->kv16&&h.pos<=(uint32_t)m->max_seq&&h.recurrent_n==(uint64_t)recurrent_state_floats(m)&&h.kv_n==(uint64_t)session_kv_scalars(m,(int)h.pos)&&h.recurrent_n<=SIZE_MAX/sizeof(float)&&h.kv_n<=SIZE_MAX/(m->kv16?sizeof(uint16_t):sizeof(float));
    if(!ok){fclose(f);return 0;}session_state_free(s);s->pos=(int)h.pos;s->hidden=(int)h.hidden;s->layers=(int)h.layers;s->kv16=(int)h.kv16;s->recurrent_n=(size_t)h.recurrent_n;s->kv_n=(size_t)h.kv_n;s->recurrent=s->recurrent_n?falloc((int64_t)s->recurrent_n):NULL;s->last_hidden=falloc(s->hidden);if(s->kv16)s->kv_bf16=s->kv_n?xcalloc(s->kv_n,sizeof(uint16_t)):NULL;else s->kv=s->kv_n?falloc((int64_t)s->kv_n):NULL;s->recurrent_cap=s->recurrent_n;s->kv_cap=s->kv_n;s->hidden_cap=(size_t)s->hidden;
    ok=fread(s->recurrent,sizeof(float),s->recurrent_n,f)==s->recurrent_n;if(ok)ok=s->kv16?fread(s->kv_bf16,sizeof(uint16_t),s->kv_n,f)==s->kv_n:fread(s->kv,sizeof(float),s->kv_n,f)==s->kv_n;if(ok)ok=fread(s->last_hidden,sizeof(float),(size_t)s->hidden,f)==(size_t)s->hidden&&fgetc(f)==EOF;fclose(f);
    if(!ok||session_state_checksum(s)!=h.checksum){session_state_free(s);return 0;}return 1;
}
static double forward_prefill_core(Model*m,const int*token,int T,float*logits,int score_first,int*score_count,int grouped_moe,float*hidden_all){
    Cfg*c=&m->c;if(T<=0||m->pos!=0||T>m->max_seq)die("prefill requires a fresh model and valid length");float*x=falloc((int64_t)T*c->hidden),*n=falloc((int64_t)T*c->hidden),*mix=falloc((int64_t)T*c->hidden),*moe=falloc(c->hidden);
    for(int t=0;t<T;t++){if(token[t]<0||token[t]>=c->vocab)die("prefill token outside vocab");qmat_row(x+(int64_t)t*c->hidden,&m->embed,token[t]);}
    for(int li=0;li<c->n_layers;li++){
        Layer*l=&m->layer[li];
        for(int t=0;t<T;t++)rmsnorm_zero(n+(int64_t)t*c->hidden,x+(int64_t)t*c->hidden,l->input_norm,c->hidden,c->eps);
        double core_t0=m->prof_detail?now_s():0.;
        if(l->type==LT_LINEAR)gdn_prefill_layer(m,l,n,T,mix);
        else for(int t=0;t<T;t++){m->pos=t;attn_forward(m,l,n+(int64_t)t*c->hidden,mix+(int64_t)t*c->hidden);}
        if(m->prof_detail){if(l->type==LT_LINEAR)m->prof_gdn+=now_s()-core_t0;else m->prof_attn+=now_s()-core_t0;}
        for(int t=0;t<T;t++){float*xt=x+(int64_t)t*c->hidden;for(int i=0;i<c->hidden;i++)xt[i]+=mix[(int64_t)t*c->hidden+i];rmsnorm_zero(n+(int64_t)t*c->hidden,xt,l->post_norm,c->hidden,c->eps);}
        double moe_t0=m->prof_detail?now_s():0.;
        if(grouped_moe){moe_prefill_grouped(m,l,n,T,mix);for(int t=0;t<T;t++){float*xt=x+(int64_t)t*c->hidden;for(int i=0;i<c->hidden;i++)xt[i]+=mix[(int64_t)t*c->hidden+i];}}
        else for(int t=0;t<T;t++){float*xt=x+(int64_t)t*c->hidden;moe_forward(m,l,n+(int64_t)t*c->hidden,moe);for(int i=0;i<c->hidden;i++)xt[i]+=moe[i];}
        if(m->prof_detail)m->prof_moe+=now_s()-moe_t0;
    }
    if(hidden_all)for(int t=0;t<T;t++)rmsnorm_zero(hidden_all+(int64_t)t*c->hidden,x+(int64_t)t*c->hidden,m->final_norm,c->hidden,c->eps);
    m->pos=T;double nll=0.;int scored=0;
    double lm_t0=m->prof_detail?now_s():0.;
    if(score_first<0){if(hidden_all)memcpy(n,hidden_all+(int64_t)(T-1)*c->hidden,(size_t)c->hidden*sizeof(float));else rmsnorm_zero(n,x+(int64_t)(T-1)*c->hidden,m->final_norm,c->hidden,c->eps);memcpy(m->last_hidden,n,(size_t)c->hidden*sizeof(float));qmat_mul_ex(logits,n,&m->lm_head,1);}
    else{
        int total=T-1-score_first,lmb=32;const char*be=getenv("EVAL_LM_BATCH");if(be)lmb=atoi(be);if(lmb<1)lmb=1;if(lmb>total)lmb=total;
        float*bn=falloc((int64_t)lmb*c->hidden),*bl=falloc((int64_t)lmb*c->vocab);
        for(int base=0;base<total;base+=lmb){int B=total-base;if(B>lmb)B=lmb;
            for(int b=0;b<B;b++){int t=score_first+base+b;rmsnorm_zero(bn+(int64_t)b*c->hidden,x+(int64_t)t*c->hidden,m->final_norm,c->hidden,c->eps);}
            qmat_mul_batch(bl,bn,B,c->hidden,&m->lm_head,1);
            for(int b=0;b<B;b++){int t=score_first+base+b;float*lb=bl+(int64_t)b*c->vocab,mx=-INFINITY;for(int j=0;j<c->vocab;j++)if(lb[j]>mx)mx=lb[j];double den=0.;for(int j=0;j<c->vocab;j++)den+=exp((double)lb[j]-mx);nll+=(double)mx+log(den)-lb[token[t+1]];scored++;}
        }
        free(bn);free(bl);
    }
    if(m->prof_detail)m->prof_lm+=now_s()-lm_t0;
    if(score_count)*score_count=scored;free(x);free(n);free(mix);free(moe);return nll;
}
static void forward_prefill(Model*m,const int*token,int T,float*logits){(void)forward_prefill_core(m,token,T,logits,-1,NULL,0,NULL);}
static void prefill_dispatch(Model*m,const int*token,int T,float*logits){if(m->dump_acts)for(int i=0;i<T;i++)forward_token(m,token[i],logits);else forward_prefill(m,token,T,logits);}
static int argmax(const float *x,int n){int b=0;for(int i=1;i<n;i++)if(x[i]>x[b])b=i;return b;}
static void print_top5(const float *x,int n,int step){ int id[5]={-1,-1,-1,-1,-1};for(int j=0;j<5;j++)for(int i=0;i<n;i++){int used=0;for(int p=0;p<j;p++)if(id[p]==i)used=1;if(!used&&(id[j]<0||x[i]>x[id[j]]))id[j]=i;}fprintf(stderr,"[LOGITS %d]",step);for(int j=0;j<5;j++)fprintf(stderr," %d:%.7g",id[j],x[id[j]]);fputc('\n',stderr); }

static int mtp_margin_bin(float margin){
    if(margin<0.25f)return 0;if(margin<0.5f)return 1;
    if(margin<1.f)return 2;if(margin<2.f)return 3;return 4;
}
static void mtp_conf_record(MtpW*w,const float*margin,int n,int accepted,int rejected){
    for(int i=0;i<n;i++){int b=mtp_margin_bin(margin[i]);if(i<accepted){w->conf_accepted[b]++;w->conf_accepted_sum+=margin[i];}else if(rejected&&i==accepted){w->conf_rejected[b]++;w->conf_rejected_sum+=margin[i];}else{w->conf_unverified[b]++;w->conf_unverified_sum+=margin[i];}}
}
static uint64_t mtp_conf_total(const uint64_t*v){uint64_t n=0;for(int i=0;i<5;i++)n+=v[i];return n;}
static void mtp_conf_print(const MtpW*w){
    uint64_t na=mtp_conf_total(w->conf_accepted),nr=mtp_conf_total(w->conf_rejected),nu=mtp_conf_total(w->conf_unverified);
    fprintf(stderr,"[MTP_CONF] margin_bins=<0.25,<0.5,<1,<2,>=2 accepted=%llu,%llu,%llu,%llu,%llu rejected=%llu,%llu,%llu,%llu,%llu unverified=%llu,%llu,%llu,%llu,%llu mean=%.6f/%.6f/%.6f\n",(unsigned long long)w->conf_accepted[0],(unsigned long long)w->conf_accepted[1],(unsigned long long)w->conf_accepted[2],(unsigned long long)w->conf_accepted[3],(unsigned long long)w->conf_accepted[4],(unsigned long long)w->conf_rejected[0],(unsigned long long)w->conf_rejected[1],(unsigned long long)w->conf_rejected[2],(unsigned long long)w->conf_rejected[3],(unsigned long long)w->conf_rejected[4],(unsigned long long)w->conf_unverified[0],(unsigned long long)w->conf_unverified[1],(unsigned long long)w->conf_unverified[2],(unsigned long long)w->conf_unverified[3],(unsigned long long)w->conf_unverified[4],na?w->conf_accepted_sum/(double)na:0.,nr?w->conf_rejected_sum/(double)nr:0.,nu?w->conf_unverified_sum/(double)nu:0.);
}
static void mtp_profile_mark(const Model*m,double*v){v[0]=m->prof_gdn;v[1]=m->prof_attn;v[2]=m->prof_moe;v[3]=m->prof_lm;}
static void mtp_profile_add(const Model*m,double*out,const double*before){out[0]+=m->prof_gdn-before[0];out[1]+=m->prof_attn-before[1];out[2]+=m->prof_moe-before[2];out[3]+=m->prof_lm-before[3];}
static int mtp_draft(Model*m,int next_token,int cap,int*draft,float*margin){
    if(!m->mtp.enabled||cap<1)return 0;Cfg*c=&m->c;if(cap>8)cap=8;
    if(m->mtp.pos+cap>m->max_seq)cap=m->max_seq-m->mtp.pos;if(cap<1)return 0;
    float*h=falloc(c->hidden),*next_h=falloc(c->hidden),*logits=falloc(c->vocab);memcpy(h,m->last_hidden,(size_t)c->hidden*sizeof(float));
    int token=next_token,n=0;
    for(int i=0;i<cap;i++){int pred=mtp_step(m,token,h,m->mtp.pos+i,logits,next_h);if(pred<0)break;float second=-INFINITY;for(int j=0;j<c->vocab;j++)if(j!=pred&&logits[j]>second)second=logits[j];draft[n]=pred;margin[n]=logits[pred]-second;n++;token=pred;memcpy(h,next_h,(size_t)c->hidden*sizeof(float));}
    free(h);free(next_h);free(logits);return n;
}

static int suppress_emit;
static void emit_generated(Model*m,int token,int textout){
    if(suppress_emit)return;
    if(textout){if(cfg_is_eos(&m->c,token))return;char piece[4096];int z=tok_decode(&m->T,&token,1,piece,sizeof(piece)-1);fwrite(piece,1,z,stdout);fflush(stdout);}
    else printf(" %d",token);
}

/* Prime both the target recurrence and the MTP decoder cache from the prompt.
 * MTP position i consumes the target hidden state after prompt[i] together
 * with prompt[i+1].  The final prompt hidden is retained for the first draft. */
static void prefill_mtp(Model*m,const int*token,int T,float*logits){
    if(T<=0)die("MTP prefill requires at least one token");Cfg*c=&m->c;float*mtp_logits=falloc(c->vocab),*hidden=falloc((int64_t)T*c->hidden);
    (void)forward_prefill_core(m,token,T,logits,-1,NULL,0,hidden);
    for(int i=0;i+1<T;i++)if(mtp_step(m,token[i+1],hidden+(int64_t)i*c->hidden,i,mtp_logits,NULL)<0)die("MTP prompt prefill failed");
    m->mtp.pos=T-1;free(mtp_logits);free(hidden);
}

/* Lossless greedy speculative loop.  The first implementation verifies MTP
 * proposals one token at a time, which establishes state alignment for both
 * GDN recurrence and full-attention KV before the batched verifier is enabled.
 * A rejected token is never fed to the target, so no recurrent rollback is
 * required on this exact fallback path. */
static int generate_mtp_greedy(Model*m,float*logits,int ngen,int textout){
    Cfg*c=&m->c;int made=0,draft_cap=getenv("DRAFT")?atoi(getenv("DRAFT")):3;if(draft_cap<1)draft_cap=1;if(draft_cap>8)draft_cap=8;
    float min_margin=getenv("MTP_MIN_MARGIN")?(float)atof(getenv("MTP_MIN_MARGIN")):0.f;
    if(!suppress_emit&&m->mtp.proposed==0){memset(m->mtp.conf_accepted,0,sizeof(m->mtp.conf_accepted));memset(m->mtp.conf_rejected,0,sizeof(m->mtp.conf_rejected));memset(m->mtp.conf_unverified,0,sizeof(m->mtp.conf_unverified));memset(m->mtp.verify_detail,0,sizeof(m->mtp.verify_detail));memset(m->mtp.replay_detail,0,sizeof(m->mtp.replay_detail));memset(m->mtp.fallback_detail,0,sizeof(m->mtp.fallback_detail));m->mtp.conf_accepted_sum=m->mtp.conf_rejected_sum=m->mtp.conf_unverified_sum=0.;m->mtp.confidence_skips=0;m->mtp.fallback_s=0.;}
    int batched=!getenv("SPEC_BATCH")||atoi(getenv("SPEC_BATCH"))!=0;
    int target_exact_cpu=batched;
#ifdef COLI_CUDA
    if(cuda_spec_full_enabled())target_exact_cpu=0;
#endif
    float min_accept=getenv("MTP_MIN_ACCEPT")?(float)atof(getenv("MTP_MIN_ACCEPT")):0.5f;int paused=0;uint64_t groups=0;
    size_t recurrent_n=batched?recurrent_state_floats(m):0;float*recurrent=recurrent_n?falloc((int64_t)recurrent_n):NULL;
    float*block_logits=batched?falloc((int64_t)9*c->vocab):NULL,margin[8];int draft[8],target[9];
    while(made<ngen){
        int next=argmax(logits,c->vocab),stop=cfg_is_eos(c,next);emit_generated(m,next,textout);if(m->debug_logits)print_top5(logits,c->vocab,made);made++;m->mtp.emitted++;
        if(stop||made>=ngen)break;
        if(paused){uint64_t miss0=m->prof_expert_misses;double detail0[4],fallback_t0=now_s();mtp_profile_mark(m,detail0);target_forward_one(m,next,logits,target_exact_cpu);m->mtp.fallback_s+=now_s()-fallback_t0;mtp_profile_add(m,m->mtp.verify_detail,detail0);mtp_profile_add(m,m->mtp.fallback_detail,detail0);m->mtp.verify_misses+=m->prof_expert_misses-miss0;m->mtp.target_forwards++;continue;}
        int cap=draft_cap;if(cap>ngen-made)cap=ngen-made;int start=m->mtp.pos;uint64_t miss0=m->prof_expert_misses;double draft_t0=now_s();int g=mtp_draft(m,next,cap,draft,margin);m->mtp.draft_s+=now_s()-draft_t0;m->mtp.draft_misses+=m->prof_expert_misses-miss0;const char*force_reject=getenv("SPEC_FORCE_REJECT");if(g>0&&force_reject&&atoi(force_reject)!=0)draft[0]=(draft[0]+1)%c->vocab;groups++;
        if(g>0&&min_margin>0.f&&margin[0]<min_margin){mtp_conf_record(&m->mtp,margin,g,0,0);m->mtp.confidence_skips++;miss0=m->prof_expert_misses;double detail0[4],fallback_t0=now_s();mtp_profile_mark(m,detail0);target_forward_one(m,next,logits,target_exact_cpu);m->mtp.fallback_s+=now_s()-fallback_t0;mtp_profile_add(m,m->mtp.verify_detail,detail0);mtp_profile_add(m,m->mtp.fallback_detail,detail0);m->mtp.verify_misses+=m->prof_expert_misses-miss0;m->mtp.target_forwards++;m->mtp.pos=start+1;
#ifdef COLI_CUDA
            if(m->mtp.layer.attn.cuda_aux_ready)m->mtp.layer.attn.cuda_state_pos=start+1;
#endif
            continue;}
        m->mtp.proposed+=(uint64_t)g;
        if(!batched||g<1){miss0=m->prof_expert_misses;double detail0[4];mtp_profile_mark(m,detail0);target_forward_one(m,next,logits,target_exact_cpu);mtp_profile_add(m,m->mtp.verify_detail,detail0);m->mtp.verify_misses+=m->prof_expert_misses-miss0;m->mtp.target_forwards++;if(g<1)continue;
            int accepted=0,rejected=0;
            while(accepted<g&&made<ngen){int verified=argmax(logits,c->vocab);if(verified!=draft[accepted]){rejected=1;break;}int token=draft[accepted],draft_stop=cfg_is_eos(c,token);emit_generated(m,token,textout);if(m->debug_logits)print_top5(logits,c->vocab,made);made++;accepted++;m->mtp.accepted++;m->mtp.emitted++;if(draft_stop||made>=ngen)break;miss0=m->prof_expert_misses;mtp_profile_mark(m,detail0);forward_token(m,token,logits);mtp_profile_add(m,m->mtp.verify_detail,detail0);m->mtp.verify_misses+=m->prof_expert_misses-miss0;m->mtp.target_forwards++;}
            mtp_conf_record(&m->mtp,margin,g,accepted,rejected);
            m->mtp.pos=start+1+accepted;
        }else{
            int base=m->pos;target[0]=next;for(int i=0;i<g;i++)target[i+1]=draft[i];int gpu_recurrent=0;
#ifdef COLI_CUDA
            gpu_recurrent=cuda_recurrent_snapshot(m,base);
#endif
            if(!gpu_recurrent)recurrent_state_copy(m,recurrent,0);miss0=m->prof_expert_misses;double detail0[4];mtp_profile_mark(m,detail0);double verify_t0=now_s();target_forward_block(m,target,g+1,block_logits);m->mtp.verify_s+=now_s()-verify_t0;mtp_profile_add(m,m->mtp.verify_detail,detail0);m->mtp.verify_misses+=m->prof_expert_misses-miss0;m->mtp.target_forwards++;
            int accepted=0,halt=0;while(accepted<g&&made<ngen){float*row=block_logits+(int64_t)accepted*c->vocab;if(argmax(row,c->vocab)!=draft[accepted])break;int token=draft[accepted],draft_stop=cfg_is_eos(c,token);emit_generated(m,token,textout);if(m->debug_logits)print_top5(row,c->vocab,made);made++;accepted++;m->mtp.accepted++;m->mtp.emitted++;if(draft_stop||made>=ngen){halt=1;break;}}
            mtp_conf_record(&m->mtp,margin,g,accepted,accepted<g&&!halt);
            if(accepted==g){memcpy(logits,block_logits+(int64_t)g*c->vocab,(size_t)c->vocab*sizeof(float));}
            else if(!halt){
#ifdef COLI_CUDA
                if(gpu_recurrent){cuda_recurrent_restore(m,base);if(cuda_spec_full_enabled())cuda_attention_rewind(m,base);}else
#endif
                recurrent_state_copy(m,recurrent,1);m->pos=base;miss0=m->prof_expert_misses;mtp_profile_mark(m,detail0);double replay_t0=now_s();target_forward_block(m,target,accepted+1,block_logits);m->mtp.replay_s+=now_s()-replay_t0;mtp_profile_add(m,m->mtp.replay_detail,detail0);m->mtp.replay_misses+=m->prof_expert_misses-miss0;m->mtp.target_forwards++;memcpy(logits,block_logits+(int64_t)accepted*c->vocab,(size_t)c->vocab*sizeof(float));}
            m->mtp.pos=start+1+accepted;
        }
        if(m->mtp.proposed>=24&&m->mtp.proposed&&((double)m->mtp.accepted/(double)m->mtp.proposed)<min_accept)paused=1;
    }
    if(paused)fprintf(stderr,"[MTP] drafting paused after %llu groups: acceptance below %.0f%%\n",(unsigned long long)groups,100.0*min_accept);
    if(!suppress_emit)mtp_conf_print(&m->mtp);
    if(!suppress_emit&&min_margin>0.f)fprintf(stderr,"[MTP_ADMIT] min_margin=%.6g skipped=%llu\n",min_margin,(unsigned long long)m->mtp.confidence_skips);
    if(!suppress_emit)fprintf(stderr,"[MTP_VERIFY_DETAIL] verify=%.6f/%.6f/%.6f/%.6f fallback=%.6f/%.6f/%.6f/%.6f replay=%.6f/%.6f/%.6f/%.6f fallback_s=%.6f gdn/attn/moe/lm\n",m->mtp.verify_detail[0],m->mtp.verify_detail[1],m->mtp.verify_detail[2],m->mtp.verify_detail[3],m->mtp.fallback_detail[0],m->mtp.fallback_detail[1],m->mtp.fallback_detail[2],m->mtp.fallback_detail[3],m->mtp.replay_detail[0],m->mtp.replay_detail[1],m->mtp.replay_detail[2],m->mtp.replay_detail[3],m->mtp.fallback_s);
    free(recurrent);free(block_logits);
    return made;
}

/* ---------- oracle replay ---------- */
static int *jints(jval *root,const char *key,int *n){ jval *a=json_get(root,key);if(!a||a->t!=J_ARR)die("bad oracle array");int *v=xcalloc(a->len,sizeof(int));for(int i=0;i<a->len;i++)v[i]=(int)a->kids[i]->num;*n=a->len;return v; }
static float*jfloats2(jval*root,const char*key,int*rows,int*cols){jval*a=json_get(root,key);if(!a||a->t!=J_ARR||a->len<1||a->kids[0]->t!=J_ARR)return NULL;*rows=a->len;*cols=a->kids[0]->len;float*v=falloc((int64_t)*rows**cols);for(int i=0;i<*rows;i++){jval*r=a->kids[i];if(r->t!=J_ARR||r->len!=*cols)die("ragged oracle float matrix");for(int j=0;j<*cols;j++)v[(int64_t)i**cols+j]=(float)r->kids[j]->num;}return v;}
static Oracle load_oracle(const char *path){ long n;char*b=read_file(path,&n),*arena=NULL;jval*r=json_parse(b,&arena);(void)n;Oracle o={0};o.prompt=jints(r,"prompt_ids",&o.nprompt);o.full=jints(r,"full_ids",&o.nfull);o.tf=jints(r,"tf_pred",&o.ntf);jval*mi=json_get(r,"mtp_input_ids"),*mp=json_get(r,"mtp_pred");if(mi&&mi->t==J_ARR)o.mtp_ids=jints(r,"mtp_input_ids",&o.nmtp);if(mp&&mp->t==J_ARR)o.mtp_pred=jints(r,"mtp_pred",&o.nmtp_pred);o.mtp_logits=jfloats2(r,"mtp_logits",&o.mtp_rows,&o.mtp_cols);free(b);free(arena);return o; }
static void default_ref_path(char *out,size_t cap,const char *snap){
    const char *base=strrchr(snap,'/');base=base?base+1:snap;const char *ref=strstr(base,"int8")?"ref_qwen_int8.json":(strstr(base,"i4")?"ref_qwen_i4.json":"ref_qwen.json");
    const char *slash=strrchr(snap,'/');if(slash){int n=(int)(slash-snap);snprintf(out,cap,"%.*s/%s",n,snap,ref);}else snprintf(out,cap,"%s",ref);
}
static int run_oracle(Model *m,const char *snap){
    char path[2048];const char *rp=getenv("REF");if(rp)snprintf(path,sizeof(path),"%s",rp);else default_ref_path(path,sizeof(path),snap);Oracle o=load_oracle(path);if(o.ntf!=32||o.nfull!=o.nprompt+o.ntf)die("oracle must contain 32 generated tokens");
    int mtp_ok=1;
    if(m->mtp.enabled&&o.mtp_ids&&o.mtp_logits){
        if(o.nmtp_pred!=o.nmtp-1||o.mtp_rows!=o.nmtp-1||o.mtp_cols!=m->c.vocab)die("bad MTP oracle dimensions");
        float*main_logits=falloc(m->c.vocab),*mtp_logits=falloc(m->c.vocab);int matched=0;float maxdiff=0.f;model_reset(m);
        for(int i=0;i<o.nmtp-1;i++){forward_token(m,o.mtp_ids[i],main_logits);int pred=mtp_step(m,o.mtp_ids[i+1],m->last_hidden,i,mtp_logits,NULL);if(pred==o.mtp_pred[i])matched++;for(int j=0;j<m->c.vocab;j++){float d=fabsf(mtp_logits[j]-o.mtp_logits[(int64_t)i*m->c.vocab+j]);if(d>maxdiff)maxdiff=d;}}
        float tol=(m->matrix_i8||m->matrix_i2||m->matrix_i3||m->matrix_i4)?1.5e-2f:2e-3f;
        mtp_ok=matched==o.nmtp_pred&&maxdiff<tol;printf("[MTP_ORACLE] %d/%d logits_maxdiff=%.8g tol=%.4g\n",matched,o.nmtp_pred,maxdiff,tol);free(main_logits);free(mtp_logits);
    }else if(m->mtp.enabled){printf("[MTP_ORACLE] missing reference\n");mtp_ok=0;}
    float *logits=falloc(m->c.vocab);model_reset(m);prefill_dispatch(m,o.prompt,o.nprompt,logits);int pass=0;
    for(int i=0;i<o.ntf;i++){int p=argmax(logits,m->c.vocab);if(m->debug_logits)print_top5(logits,m->c.vocab,i);if(p==o.tf[i])pass++;if(i+1<o.ntf)forward_token(m,o.full[o.nprompt+i],logits);}printf("[ORACLE] %d/%d\n",pass,o.ntf);
    model_reset(m);prefill_dispatch(m,o.prompt,o.nprompt,logits);int gp=0;
    for(int i=0;i<o.ntf;i++){int p=argmax(logits,m->c.vocab);if(p==o.full[o.nprompt+i])gp++;if(i+1<o.ntf)forward_token(m,p,logits);}printf("[GREEDY] %d/%d\n",gp,o.ntf);free(logits);return(pass==o.ntf&&gp==o.ntf&&mtp_ok)?0:2;
}
static int *read_token_ids(const char*path,int*n){FILE*f=fopen(path,"rb");if(!f){perror(path);exit(1);}int cap=4096,*ids=xcalloc(cap,sizeof(int)),v;*n=0;while(fscanf(f,"%d",&v)==1){if(*n==cap){cap*=2;int*p=realloc(ids,(size_t)cap*sizeof(int));if(!p)die("OOM token ids");ids=p;}ids[(*n)++]=v;}fclose(f);return ids;}
static int run_eval_ids(Model*m,const char*path){
    int n;int*ids=read_token_ids(path,&n);if(n<2)die("EVAL_IDS requires at least two token IDs");
    int chunk=getenv("EVAL_CHUNK")?atoi(getenv("EVAL_CHUNK")):n;if(chunk<2)die("EVAL_CHUNK must be at least two");if(chunk>m->max_seq)die("EVAL_CHUNK exceeds CTX");
    int nchunk=n/chunk;if(!getenv("EVAL_CHUNK")){nchunk=1;chunk=n;}if(nchunk<1)die("EVAL_IDS has no complete evaluation chunk");
    /* Match llama-perplexity: reset each non-overlapping context and score
     * only its second half, so every scored token has substantial context. */
    int first=getenv("EVAL_CHUNK")?chunk/2:0,scored=0;float*logits=falloc(m->c.vocab);double nll=0.,t0=now_s();
    int grouped=getenv("EVAL_GROUPED")?atoi(getenv("EVAL_GROUPED"))!=0:1;
    for(int c=0;c<nchunk;c++){int count=0,base=c*chunk;model_reset(m);nll+=forward_prefill_core(m,ids+base,chunk,logits,first,&count,grouped,NULL);scored+=count;}
    double sec=now_s()-t0,ppl=exp(nll/scored);printf("[PPL] tokens=%d nll=%.9f ppl=%.9f tok_s=%.3f\n",scored,nll,ppl,scored/sec);free(ids);free(logits);return 0;
}
static int run_prefix_ids(Model*m,const char*path,int ngen){FILE*f=fopen(path,"rb");if(!f){perror(path);return 1;}char*line=NULL;size_t cap=0;ssize_t z;float*logits=falloc(m->c.vocab);int row=0;while((z=getline(&line,&cap,f))>=0){int*ids=xcalloc(m->max_seq,sizeof(int)),n=0;char*p=line,*end;while(*p){long v=strtol(p,&end,10);if(end==p){p++;continue;}if(n>=m->max_seq)die("PREFIX_IDS prompt exceeds CTX");ids[n++]=(int)v;p=end;}if(!n){free(ids);continue;}model_reset(m);prefill_dispatch(m,ids,n,logits);printf("PREFIX %d:",row);for(int i=0;i<ngen;i++){int id=argmax(logits,m->c.vocab);printf(" %d",id);if(m->debug_logits){fprintf(stderr,"[ROW %d]",row);print_top5(logits,m->c.vocab,i);}if(cfg_is_eos(&m->c,id))break;if(i+1<ngen)forward_token(m,id,logits);}printf("\n");row++;free(ids);}free(line);free(logits);fclose(f);return 0;}

/* Teacher-forced agreement against a reference continuation. Each input line is
 * "nprompt tok tok ...": prefill the prompt once, compare its argmax, then feed
 * the reference token through the normal decode recurrence. This is the same
 * oracle procedure used by TF=1 and is far cheaper than recomputing logits for
 * every position in one evaluation-prefill block. */
static int run_tfprefix_ids(Model*m,const char*path){
    FILE*f=fopen(path,"rb");if(!f){perror(path);return 1;}char*line=NULL;size_t cap=0;ssize_t z;
    float*logits=falloc(m->c.vocab);int row=0;long tot=0,hit=0;
    while((z=getline(&line,&cap,f))>=0){
        int*ids=xcalloc(m->max_seq,sizeof(int)),n=0;char*p=line,*end;
        while(*p){long v=strtol(p,&end,10);if(end==p){p++;continue;}if(n>=m->max_seq)die("TFPREFIX_IDS line exceeds CTX");ids[n++]=(int)v;p=end;}
        if(n<3){free(ids);continue;}
        int nprompt=ids[0],*seq=ids+1,T=n-1;
        if(nprompt<1||nprompt>=T)die("TFPREFIX_IDS needs 1 <= nprompt < sequence length");
        int scored=T-nprompt,match=0;model_reset(m);prefill_dispatch(m,seq,nprompt,logits);
        for(int i=0;i<scored;i++){if(argmax(logits,m->c.vocab)==seq[nprompt+i])match++;if(i+1<scored)forward_token(m,seq[nprompt+i],logits);}
        printf("TFPREFIX %d: %d/%d\n",row++,match,scored);fflush(stdout);
        tot+=scored;hit+=match;free(ids);
    }
    printf("[TFAGREE] %ld/%ld %.4f\n",hit,tot,tot?(double)hit/tot:0.);
    free(line);free(logits);fclose(f);return 0;
}

typedef struct {
    SessionState state;int state_valid;
    int*history,nhistory,history_cap;
    float*logits;
    double started,request_started,prof_start[5],decode_prof_start[5];
    double cuda_prof_start[8],decode_cuda_prof_start[8];
    uint64_t tier_hit_start,tier_gpu_hit_start,tier_miss_start;
    uint64_t read_start,direct_start,cuda_tx_start,decode_cuda_tx_start;
    uint64_t decode_tier_hit_start,decode_tier_gpu_hit_start;
    uint64_t decode_tier_miss_start,decode_read_start,decode_direct_start;
} MuxRuntimeSlot;

typedef struct {
    char magic[8];
    uint32_t version,hidden,layers,kv16,pos,nhistory,reserved;
    uint64_t recurrent_n,kv_n,checksum;
} MuxSessionDiskHeader;

static int mux_history_set(MuxRuntimeSlot*r,const int*ids,int n){
    if(r->history_cap<n){int cap=r->history_cap?r->history_cap:256;while(cap<n)cap*=2;int*p=realloc(r->history,(size_t)cap*sizeof(int));if(!p)return 0;r->history=p;r->history_cap=cap;}
    memcpy(r->history,ids,(size_t)n*sizeof(int));r->nhistory=n;return 1;
}
static int mux_history_append(MuxRuntimeSlot*r,int id){
    if(r->nhistory==r->history_cap){int cap=r->history_cap?r->history_cap*2:256;int*p=realloc(r->history,(size_t)cap*sizeof(int));if(!p)return 0;r->history=p;r->history_cap=cap;}
    r->history[r->nhistory++]=id;return 1;
}
static int mux_exact_prefix(const MuxRuntimeSlot*r,const int*ids,int n){return r->state_valid&&r->nhistory<=n&&!memcmp(r->history,ids,(size_t)r->nhistory*sizeof(int));}
static double mux_cache_hit_percent(const Model*m,const MuxRuntimeSlot*r){
    uint64_t cpu=m->tier_hits-r->decode_tier_hit_start;
    uint64_t gpu=m->tier_gpu_hits-r->decode_tier_gpu_hit_start;
    uint64_t miss=m->tier_misses-r->decode_tier_miss_start,total=cpu+gpu+miss;
    return total?100.0*(double)(cpu+gpu)/(double)total:0.;
}
static uint64_t mux_session_checksum(const SessionState*s,const int*history,int n){
    uint64_t h=1469598103934665603ULL;h=session_hash_add(h,history,(size_t)n*sizeof(int));h=session_hash_add(h,s->recurrent,s->recurrent_n*sizeof(float));
    if(s->kv16)h=session_hash_add(h,s->kv_bf16,s->kv_n*sizeof(uint16_t));else h=session_hash_add(h,s->kv,s->kv_n*sizeof(float));
    return session_hash_add(h,s->last_hidden,(size_t)s->hidden*sizeof(float));
}
static int mux_session_write(const char*path,const SessionState*s,const int*history,int n){
    if(!path||!s||!history||n<1||n!=s->pos)return 0;MuxSessionDiskHeader h={{'C','O','L','I','M','U','X','S'},1,(uint32_t)s->hidden,(uint32_t)s->layers,(uint32_t)s->kv16,(uint32_t)s->pos,(uint32_t)n,0,(uint64_t)s->recurrent_n,(uint64_t)s->kv_n,mux_session_checksum(s,history,n)};
    char tmp[2304];snprintf(tmp,sizeof(tmp),"%s.tmp.%ld",path,(long)getpid());FILE*f=fopen(tmp,"wb");if(!f)return 0;int ok=fwrite(&h,1,sizeof(h),f)==sizeof(h)&&fwrite(history,sizeof(int),(size_t)n,f)==(size_t)n&&fwrite(s->recurrent,sizeof(float),s->recurrent_n,f)==s->recurrent_n;
    if(ok)ok=s->kv16?fwrite(s->kv_bf16,sizeof(uint16_t),s->kv_n,f)==s->kv_n:fwrite(s->kv,sizeof(float),s->kv_n,f)==s->kv_n;
    if(ok)ok=fwrite(s->last_hidden,sizeof(float),(size_t)s->hidden,f)==(size_t)s->hidden&&fflush(f)==0&&fsync(fileno(f))==0;
    if(fclose(f)!=0)ok=0;if(ok&&rename(tmp,path)==0){char dir[2304];snprintf(dir,sizeof(dir),"%s",path);char*slash=strrchr(dir,'/');if(slash){if(slash==dir)slash[1]=0;else*slash=0;}else snprintf(dir,sizeof(dir),".");int fd=open(dir,O_RDONLY|O_DIRECTORY);if(fd>=0){int synced=fsync(fd)==0;close(fd);return synced;}return 0;}unlink(tmp);return 0;
}
static int mux_session_read(Model*m,const char*path,SessionState*s,int**history,int*n){
    if(!m||!path||!s||!history||!n)return 0;FILE*f=fopen(path,"rb");if(!f)return 0;MuxSessionDiskHeader h;
    int ok=fread(&h,1,sizeof(h),f)==sizeof(h)&&!memcmp(h.magic,"COLIMUXS",8)&&h.version==1&&h.hidden==(uint32_t)m->c.hidden&&h.layers==(uint32_t)m->c.n_layers&&h.kv16==(uint32_t)m->kv16&&h.pos<=(uint32_t)m->max_seq&&h.nhistory==h.pos&&h.nhistory>0&&h.recurrent_n==(uint64_t)recurrent_state_floats(m)&&h.kv_n==(uint64_t)session_kv_scalars(m,(int)h.pos);
    if(!ok){fclose(f);return 0;}SessionState loaded={0};size_t rn=(size_t)h.recurrent_n,kn=(size_t)h.kv_n;if(!session_state_reserve(m,&loaded,rn,kn)){fclose(f);return 0;}
    loaded.pos=(int)h.pos;loaded.hidden=(int)h.hidden;loaded.layers=(int)h.layers;loaded.kv16=(int)h.kv16;loaded.recurrent_n=rn;loaded.kv_n=kn;int*ids=xcalloc((size_t)h.nhistory,sizeof(int));
    ok=fread(ids,sizeof(int),(size_t)h.nhistory,f)==(size_t)h.nhistory&&fread(loaded.recurrent,sizeof(float),rn,f)==rn;
    if(ok)ok=loaded.kv16?fread(loaded.kv_bf16,sizeof(uint16_t),kn,f)==kn:fread(loaded.kv,sizeof(float),kn,f)==kn;
    if(ok)ok=fread(loaded.last_hidden,sizeof(float),(size_t)loaded.hidden,f)==(size_t)loaded.hidden&&fgetc(f)==EOF;fclose(f);
    if(!ok||mux_session_checksum(&loaded,ids,(int)h.nhistory)!=h.checksum){free(ids);session_state_free(&loaded);return 0;}
    session_state_free(s);*s=loaded;free(*history);*history=ids;*n=(int)h.nhistory;return 1;
}
static void mux_session_path(char*out,size_t cap,const char*dir,int slot){snprintf(out,cap,"%s/slot-%02d.colimux",dir,slot);}
static int mux_checkpoint_slot(Model*m,ResidentBatchState*resident,int resident_mode,MuxRuntimeSlot*r,int slot,const char*dir){
    if(!dir||!r->state_valid||r->nhistory<1)return 1;char path[2304];mux_session_path(path,sizeof(path),dir,slot);
    if(resident_mode){SessionState state={0};int ok=resident_export_session(m,resident,slot,&state)&&mux_session_write(path,&state,r->history,r->nhistory);session_state_free(&state);return ok;}
    return mux_session_write(path,&r->state,r->history,r->nhistory);
}

/* Functional Phase-9 reference loop. Slots are switched through exact
 * SessionState save/restore; continuous batched kernels replace the sequential
 * decode-row loop after protocol/session correctness is fixed. */
static int run_serve_mux(Model*m){
    if(m->mtp.enabled){fprintf(stderr,"SERVE_BATCH v1 requires MTP=0\n");return 2;}
    /* poll(2) observes the file descriptor, not bytes read ahead by stdio.
     * With a persistent gateway pipe, buffered fread/getline could consume a
     * second SUBMIT into FILE's private buffer and then leave poll waiting on
     * an empty descriptor forever. */
    setvbuf(stdin,NULL,_IONBF,0);
    m->prof_detail=1;
    int nslots=getenv("KV_SLOTS")?atoi(getenv("KV_SLOTS")):1;mux_scheduler sched;if(!mux_scheduler_init(&sched,nslots)){fprintf(stderr,"KV_SLOTS must be 1..16\n");return 2;}
    int resident_mode=getenv("SERVE_RESIDENT")&&atoi(getenv("SERVE_RESIDENT"))!=0;ResidentBatchState resident={0};
    if(resident_mode&&!resident_batch_init(m,&resident,nslots)){fprintf(stderr,"resident batch state allocation failed\n");return 2;}
    MuxRuntimeSlot runtime[MUX_MAX_SLOTS];memset(runtime,0,sizeof(runtime));int eof=0;const char*session_dir=getenv("SESSION_DIR");
    if(session_dir&&*session_dir){if(mkdir(session_dir,0755)!=0&&errno!=EEXIST){fprintf(stderr,"SESSION_DIR create failed: %s\n",strerror(errno));resident_batch_free(m,&resident);return 2;}
        for(int slot=0;slot<nslots;slot++){char path[2304];mux_session_path(path,sizeof(path),session_dir,slot);int n=0;
            if(mux_session_read(m,path,&runtime[slot].state,&runtime[slot].history,&n)){runtime[slot].nhistory=runtime[slot].history_cap=n;int ok=!resident_mode||resident_import_session(m,&resident,slot,&runtime[slot].state);if(ok){runtime[slot].state_valid=1;fprintf(stderr,"[SESSION] restored slot=%d tokens=%d\n",slot,n);}else{free(runtime[slot].history);runtime[slot].history=NULL;runtime[slot].nhistory=runtime[slot].history_cap=0;}if(resident_mode)session_state_free(&runtime[slot].state);}
        }
    }
    if(!mux_write_ready(stdout))return 2;telemetry_startup_emit(m);
    for(;;){
        int rows[MUX_MAX_SLOTS],active=mux_scheduler_decode_rows(&sched,rows,MUX_MAX_SLOTS);
        struct pollfd pfd={.fd=fileno(stdin),.events=POLLIN};int pr=eof?0:poll(&pfd,1,active?0:-1);
        if(pr<0&&errno==EINTR)continue;if(pr<0)return 2;
        if(pr>0&&(pfd.revents&(POLLIN|POLLHUP))){
            mux_frame frame={0};mux_parse_error pe={0};mux_frame_kind kind=mux_read_frame(stdin,&frame,0,&pe);
            if(kind==MUX_FRAME_EOF){eof=1;mux_frame_clear(&frame);}
            else if(kind==MUX_FRAME_ERROR){mux_write_error(stdout,0,pe.code?pe.code:"BAD_FRAME");mux_frame_clear(&frame);if(pe.fatal)return 2;continue;}
            else if(kind==MUX_FRAME_CANCEL){const char*code=mux_scheduler_cancel(&sched,frame.id);if(code)mux_write_error(stdout,frame.id,code);else{int slot=mux_scheduler_find(&sched,frame.id);if(!mux_checkpoint_slot(m,&resident,resident_mode,&runtime[slot],slot,session_dir))mux_write_error(stdout,frame.id,"INTERNAL");else mux_write_error(stdout,frame.id,"CANCELLED");mux_scheduler_release(&sched,slot,NULL);}mux_frame_clear(&frame);continue;}
            else{
                const char*code=mux_scheduler_submit(&sched,frame.id,frame.slot,frame.max_tokens,frame.temperature,frame.top_p);
                if(!code&&(frame.temperature!=0.f||frame.top_p!=1.f))code="BAD_REQUEST";
                if(code){mux_write_error(stdout,frame.id,code);mux_frame_clear(&frame);continue;}
                m->decode_phase=0;
                int*ids=xcalloc(m->max_seq,sizeof(int));int n=m->has_tok?tok_encode(&m->T,(const char*)frame.payload,(int)frame.nbytes,ids,m->max_seq):0;
                if(n<=0){mux_write_error(stdout,frame.id,"EMPTY_PROMPT");mux_scheduler_release(&sched,frame.slot,NULL);free(ids);mux_frame_clear(&frame);continue;}
                MuxRuntimeSlot*r=&runtime[frame.slot];int reuse=mux_exact_prefix(r,ids,n),prepared=1;
                r->request_started=now_s();r->prof_start[0]=m->prof_expert_load;r->prof_start[1]=m->prof_moe;r->prof_start[2]=m->prof_gdn;r->prof_start[3]=m->prof_attn;r->prof_start[4]=m->prof_lm;
                r->tier_hit_start=m->tier_hits;r->tier_gpu_hit_start=m->tier_gpu_hits;
                r->tier_miss_start=m->tier_misses;r->read_start=m->S.read_bytes;
                r->direct_start=m->S.direct_bytes;
#ifdef COLI_CUDA
                {unsigned long long tx=0;if(cuda_rt.active&&!coli_cuda_profile_snapshot(cuda_rt.ctx,&tx,r->cuda_prof_start))r->cuda_tx_start=(uint64_t)tx;else{r->cuda_tx_start=0;memset(r->cuda_prof_start,0,sizeof(r->cuda_prof_start));}}
#endif
                if(reuse&&session_dir)fprintf(stderr,"[SESSION] exact-extension slot=%d cached=%d prompt=%d\n",frame.slot,r->nhistory,n);
                if(resident_mode){
                    if(reuse){
                        int suffix=n-r->nhistory;
                        const char*block=getenv("SERVE_SUFFIX_BLOCK");
                        if(block&&atoi(block)!=0)prepared=resident_forward_suffix(m,&resident,frame.slot,ids+r->nhistory,suffix);
                        else for(int i=r->nhistory;i<n;i++){int sid=frame.slot;if(!resident_forward_tokens(m,&resident,&sid,ids+i,1)){prepared=0;break;}}
                    }
                    else{float*prefill_logits=falloc(m->c.vocab);model_reset(m);prefill_dispatch(m,ids,n,prefill_logits);prepared=resident_import_model_slot(m,&resident,frame.slot,prefill_logits);free(prefill_logits);}
                }else{
                    if(!r->logits)r->logits=falloc(m->c.vocab);float*logits=r->logits;
                    if(reuse){if(!session_state_restore(m,&r->state,logits))reuse=0;else for(int i=r->nhistory;i<n;i++)forward_token(m,ids[i],logits);}
                    if(!reuse){model_reset(m);prefill_dispatch(m,ids,n,logits);}
                    prepared=session_state_save(m,&r->state);
                }
                if(!prepared||!mux_history_set(r,ids,n)){r->state_valid=0;r->nhistory=0;mux_write_error(stdout,frame.id,"INTERNAL");mux_scheduler_release(&sched,frame.slot,NULL);free(ids);mux_frame_clear(&frame);continue;}
                r->state_valid=1;
                r->decode_tier_hit_start=m->tier_hits;
                r->decode_tier_gpu_hit_start=m->tier_gpu_hits;
                r->decode_tier_miss_start=m->tier_misses;
                r->decode_read_start=m->S.read_bytes;
                r->decode_direct_start=m->S.direct_bytes;
                r->decode_prof_start[0]=m->prof_expert_load;
                r->decode_prof_start[1]=m->prof_moe;
                r->decode_prof_start[2]=m->prof_gdn;
                r->decode_prof_start[3]=m->prof_attn;
                r->decode_prof_start[4]=m->prof_lm;
#ifdef COLI_CUDA
                {unsigned long long tx=0;
                    if(cuda_rt.active&&!coli_cuda_profile_snapshot(
                        cuda_rt.ctx,&tx,r->decode_cuda_prof_start))
                        r->decode_cuda_tx_start=(uint64_t)tx;
                    else{r->decode_cuda_tx_start=0;memset(
                        r->decode_cuda_prof_start,0,
                        sizeof(r->decode_cuda_prof_start));}}
#endif
                r->started=now_s();mux_scheduler_prefill_done(&sched,frame.slot,n);free(ids);mux_frame_clear(&frame);continue;
            }
        }
        active=mux_scheduler_decode_rows(&sched,rows,MUX_MAX_SLOTS);
        if(!active){if(eof)break;continue;}
        if(resident_mode){
            int token[MUX_MAX_SLOTS],stop[MUX_MAX_SLOTS],limited[MUX_MAX_SLOTS],did_advance[MUX_MAX_SLOTS]={0},advance_slot[MUX_MAX_SLOTS],advance_token[MUX_MAX_SLOTS],nadvance=0;
            for(int ri=0;ri<active;ri++){int sid=rows[ri];mux_slot*q=&sched.slot[sid];MuxRuntimeSlot*r=&runtime[sid];float*logits=resident.logits+(int64_t)sid*m->c.vocab;
                token[ri]=argmax(logits,m->c.vocab);stop[ri]=cfg_is_eos(&m->c,token[ri]);char piece[4096];int z=stop[ri]?0:tok_decode(&m->T,token+ri,1,piece,sizeof(piece));
                if(!stop[ri]&&!mux_write_data(stdout,q->id,piece,(size_t)z))return 2;limited[ri]=mux_scheduler_emitted(&sched,sid);
                if(resident.pos[sid]<resident.max_seq){did_advance[ri]=1;advance_slot[nadvance]=sid;advance_token[nadvance]=token[ri];nadvance++;}else limited[ri]=1;
                (void)r;
            }
            m->decode_phase=1;
            int advanced=!nadvance||resident_forward_tokens(m,&resident,advance_slot,advance_token,nadvance);
            m->decode_phase=0;
            for(int ri=0;ri<active;ri++){int sid=rows[ri];mux_slot*q=&sched.slot[sid];MuxRuntimeSlot*r=&runtime[sid];
                if(!advanced||(did_advance[ri]&&!mux_history_append(r,token[ri]))){r->state_valid=0;r->nhistory=0;mux_write_error(stdout,q->id,"INTERNAL");mux_scheduler_release(&sched,sid,NULL);continue;}
                if(stop[ri]||limited[ri]){if(!limited[ri])mux_scheduler_done(&sched,sid);if(!mux_checkpoint_slot(m,&resident,resident_mode,r,sid,session_dir)){mux_write_error(stdout,q->id,"INTERNAL");mux_scheduler_release(&sched,sid,NULL);continue;}telemetry_turn_emit(m,q->id,r->request_started,r->prof_start,r->started,r->decode_prof_start,r->tier_hit_start,r->tier_gpu_hit_start,r->tier_miss_start,r->read_start,r->direct_start,r->decode_tier_hit_start,r->decode_tier_gpu_hit_start,r->decode_tier_miss_start,r->decode_read_start,r->decode_direct_start,r->cuda_tx_start,r->cuda_prof_start,r->decode_cuda_tx_start,r->decode_cuda_prof_start);double dt=now_s()-r->started;mux_write_done(stdout,q->id,q->emitted,dt>0?q->emitted/dt:0.,mux_cache_hit_percent(m,r),0.,q->prompt_tokens,limited[ri]);expert_decode_pins_refresh(m);mux_scheduler_release(&sched,sid,NULL);}
            }
            continue;
        }
        for(int ri=0;ri<active;ri++){
            int slot=rows[ri];mux_slot*q=&sched.slot[slot];MuxRuntimeSlot*r=&runtime[slot];float*logits=r->logits;
            if(!logits||!session_state_restore(m,&r->state,logits)){
                r->state_valid=0;r->nhistory=0;mux_write_error(stdout,q->id,"INTERNAL");mux_scheduler_release(&sched,slot,NULL);continue;
            }
            int token=argmax(logits,m->c.vocab),stop=cfg_is_eos(&m->c,token);char piece[4096];
            int z=stop?0:tok_decode(&m->T,&token,1,piece,sizeof(piece));
            if(!stop&&!mux_write_data(stdout,q->id,piece,(size_t)z))return 2;
            int limited=mux_scheduler_emitted(&sched,slot);
            if(m->pos<m->max_seq){
                m->decode_phase=1;
                forward_token(m,token,logits);
                m->decode_phase=0;
                if(!mux_history_append(r,token)||!session_state_save(m,&r->state)){
                    r->state_valid=0;r->nhistory=0;mux_write_error(stdout,q->id,"INTERNAL");mux_scheduler_release(&sched,slot,NULL);continue;
                }
            }else limited=1;
            if(stop||limited){
                if(!limited)mux_scheduler_done(&sched,slot);
                if(!mux_checkpoint_slot(m,&resident,resident_mode,r,slot,session_dir)){mux_write_error(stdout,q->id,"INTERNAL");mux_scheduler_release(&sched,slot,NULL);continue;}
                telemetry_turn_emit(m,q->id,r->request_started,r->prof_start,r->started,r->decode_prof_start,r->tier_hit_start,r->tier_gpu_hit_start,r->tier_miss_start,r->read_start,r->direct_start,r->decode_tier_hit_start,r->decode_tier_gpu_hit_start,r->decode_tier_miss_start,r->decode_read_start,r->decode_direct_start,r->cuda_tx_start,r->cuda_prof_start,r->decode_cuda_tx_start,r->decode_cuda_prof_start);
                double dt=now_s()-r->started;
                mux_write_done(stdout,q->id,q->emitted,dt>0?q->emitted/dt:0.,mux_cache_hit_percent(m,r),0.,q->prompt_tokens,limited);
                expert_decode_pins_refresh(m);
                mux_scheduler_release(&sched,slot,NULL);
            }
        }
    }
    resident_batch_free(m,&resident);
    for(int i=0;i<nslots;i++){session_state_free(&runtime[i].state);free(runtime[i].history);free(runtime[i].logits);}return 0;
}

#ifndef QWEN_NO_MAIN
int main(void){
    const char *snap=getenv("SNAP");if(!snap){fprintf(stderr,"set SNAP=<model directory>\n");return 1;}static Model m;model_init(&m,snap);
#ifdef COLI_CUDA
    cuda_backend_start();
    cuda_validate_vram_plan(&m);
    cuda_model_preload_dense(&m);
#else
    if(getenv("COLI_CUDA")&&atoi(getenv("COLI_CUDA"))!=0)fprintf(stderr,"[CUDA] binary was built without CUDA=1; using CPU\n");
#endif
    if(getenv("SERVE_BATCH")&&atoi(getenv("SERVE_BATCH"))!=0)return run_serve_mux(&m);
    Cfg*c=&m.c;int nf=0;for(int i=0;i<c->n_layers;i++)nf+=c->layer_type[i]==LT_FULL;
    const char*format=m.matrix_i3?(m.matrix_i8||m.matrix_i4?"mixed-with-int3-g128":"int3-g128"):(m.matrix_i2?(m.matrix_i8||m.matrix_i4?"mixed-with-int2-g128":"int2-g128"):(m.matrix_i8&&m.matrix_i4?"mixed-int8/int4-g128":(m.matrix_i8?"int8":(m.matrix_i4?"int4-g128":"bf16/fp32"))));
    const char*phase=m.mtp.enabled?"PHASE 6 MTP speculative decoding":"PHASE 5 target decoding";
#ifdef COLI_CUDA
    if(cuda_rt.active&&!m.mtp.enabled)phase="PHASE 5 CUDA target decoding";
#endif
    printf("colib qwen engine — %s\n",phase);printf("model: hidden=%d layers=%d (%d full / %d GDN) vocab=%d ctx=%d\n",c->hidden,c->n_layers,nf,c->n_layers-nf,c->vocab,m.max_seq);
    printf("weights loaded in %.2fs, tokenizer=%s, format=%s, matrices=%d/%d/%d/%d/%d f32/i8/i2/i3/i4, experts/layer=%d, KV=%s, MTP=%s\n",m.dense_load_s,m.has_tok?"yes":"no",format,m.matrix_f32,m.matrix_i8,m.matrix_i2,m.matrix_i3,m.matrix_i4,m.expert_cap,m.kv16?"bf16":"fp32",m.mtp.enabled?"active":"off");if(getenv("LOAD_ONLY")&&atoi(getenv("LOAD_ONLY"))!=0)return 0;if(getenv("TF")&&strcmp(getenv("TF"),"0"))return run_oracle(&m,snap);if(getenv("EVAL_IDS"))return run_eval_ids(&m,getenv("EVAL_IDS"));if(getenv("PREFIX_IDS"))return run_prefix_ids(&m,getenv("PREFIX_IDS"),getenv("NGEN")?atoi(getenv("NGEN")):64);if(getenv("TFPREFIX_IDS"))return run_tfprefix_ids(&m,getenv("TFPREFIX_IDS"));
    int ids[4096],nids=0;const char *prompt=getenv("PROMPT");if(!prompt)prompt=c->vocab<1000?"!":"Hello";char*chatbuf=NULL;if(getenv("CHAT")&&atoi(getenv("CHAT"))!=0){size_t z=strlen(prompt)+128;chatbuf=xcalloc(z,1);snprintf(chatbuf,z,"<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n",prompt);prompt=chatbuf;}
    if(m.has_tok)nids=tok_encode(&m.T,prompt,(int)strlen(prompt),ids,4096);else ids[nids++]=1;if(nids<=0)die("empty prompt");
    float *logits=falloc(c->vocab);model_reset(&m);double pt=now_s();if(m.mtp.enabled)prefill_mtp(&m,ids,nids,logits);else prefill_dispatch(&m,ids,nids,logits);pt=now_s()-pt;int warm=getenv("WARMUP")?atoi(getenv("WARMUP")):0;if(warm>0){if(m.mtp.enabled){suppress_emit=1;generate_mtp_greedy(&m,logits,warm,0);suppress_emit=0;m.mtp.proposed=m.mtp.accepted=m.mtp.target_forwards=m.mtp.emitted=0;m.mtp.draft_misses=m.mtp.verify_misses=m.mtp.replay_misses=0;m.mtp.draft_s=m.mtp.verify_s=m.mtp.replay_s=0.;}else for(int i=0;i<warm;i++){int p=argmax(logits,c->vocab);if(cfg_is_eos(c,p))break;if(i+1<warm)forward_token(&m,p,logits);}model_reset(&m);double wt=now_s();if(m.mtp.enabled)prefill_mtp(&m,ids,nids,logits);else prefill_dispatch(&m,ids,nids,logits);pt=now_s()-wt;fprintf(stderr,"[WARMUP] %d-token route/kernel warmup complete\n",warm);}m.prof_gdn=m.prof_attn=m.prof_moe=m.prof_lm=m.prof_expert_load=0.;m.prof_expert_misses=0;tier_counters_reset(&m);
#ifdef COLI_CUDA
    cuda_rt.batch_transactions=cuda_rt.batch_routes=cuda_rt.batch_unique_experts=0;
    coli_cuda_profile_reset(cuda_rt.ctx);
#endif
    int ngen=getenv("NGEN")?atoi(getenv("NGEN")):32;int textout=m.has_tok&&getenv("TEXT")&&atoi(getenv("TEXT"))!=0;printf(textout?"text:":"tokens:");double dt=now_s();int made=0;
    if(m.mtp.enabled)made=generate_mtp_greedy(&m,logits,ngen,textout);else for(int i=0;i<ngen;i++){int p=argmax(logits,c->vocab),stop=cfg_is_eos(c,p);made++;if(textout&&!stop){char piece[4096];int z=tok_decode(&m.T,&p,1,piece,sizeof(piece)-1);fwrite(piece,1,z,stdout);fflush(stdout);}else if(!textout)printf(" %d",p);if(m.debug_logits)print_top5(logits,c->vocab,i);if(stop)break;if(i+1<ngen)forward_token(&m,p,logits);}dt=now_s()-dt;printf("\n");if(m.mtp.enabled){fprintf(stderr,"[MTP] accepted=%llu/%llu (%.1f%%), target_forwards=%llu, emitted=%llu, draft=%s timing=%.3f/%.3f/%.3fs draft/verify/replay\n",(unsigned long long)m.mtp.accepted,(unsigned long long)m.mtp.proposed,m.mtp.proposed?100.0*(double)m.mtp.accepted/(double)m.mtp.proposed:0.0,(unsigned long long)m.mtp.target_forwards,(unsigned long long)m.mtp.emitted,getenv("DRAFT")?getenv("DRAFT"):"3",m.mtp.draft_s,m.mtp.verify_s,m.mtp.replay_s);uint64_t attributed=m.mtp.draft_misses+m.mtp.verify_misses+m.mtp.replay_misses,other=m.prof_expert_misses>attributed?m.prof_expert_misses-attributed:0;fprintf(stderr,"[MTP_CACHE] draft_misses=%llu verify_misses=%llu replay_misses=%llu other_misses=%llu total_misses=%llu\n",(unsigned long long)m.mtp.draft_misses,(unsigned long long)m.mtp.verify_misses,(unsigned long long)m.mtp.replay_misses,(unsigned long long)other,(unsigned long long)m.prof_expert_misses);}if(getenv("PROF")&&atoi(getenv("PROF"))!=0)fprintf(stderr,"[PERF] load=%.3fs prefill=%.3fs (%.2f tok/s) decode=%.3fs (%.2f tok/s)\n",m.dense_load_s,pt,nids/pt,dt,made/dt);if(m.prof_detail){double known=m.prof_gdn+m.prof_attn+m.prof_moe+m.prof_lm;fprintf(stderr,"[PERF_DETAIL] gdn=%.3fs attn=%.3fs moe=%.3fs (load=%.3fs misses=%llu) lm=%.3fs other=%.3fs\n",m.prof_gdn,m.prof_attn,m.prof_moe,m.prof_expert_load,(unsigned long long)m.prof_expert_misses,m.prof_lm,dt-known);}free(chatbuf);free(logits);return 0;
}
#endif
