/* colib — from-scratch C inference engine for Qwen3.5 MoE + Ornith-1.0.
 *
 * Phase 4 CPU engine: hybrid Gated DeltaNet + gated GQA, chunked WY prefill,
 * packed int8/int4-g128 AVX kernels, softmax-top-k MoE, expert LFRU streaming,
 * and teacher-forced oracle replay. Formulas mirror docs/qwen35_arch.md.
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

#define QW_MAX_LAYERS 128
#define QW_NAME 512

enum { LT_LINEAR = 0, LT_FULL = 1 };

typedef struct {
    int hidden, n_layers, vocab, max_position, eos_token, pad_token;
    int n_heads, n_kv_heads, head_dim;
    float partial_rotary, theta, eps;
    int n_experts, topk, moe_inter, shared_inter;
    int norm_topk;
    int lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel;
    int mtp_layers, full_interval;
    signed char layer_type[QW_MAX_LAYERS];
} Cfg;

typedef struct {
    int fmt, O, I, gs, rb, ng; /* fmt: 0=f32 materialized, 1=int8, 4=grouped int4 */
    float *f, *s;
    int8_t *q8;
    uint8_t *q4;
    void *map_q,*map_s;size_t map_q_len,map_s_len;
} QMat;
typedef struct { QMat gate, up, down; int eid; } Expert;
typedef struct {
    QMat router;
    Expert *expert;
    QMat shared_gate, shared_up, shared_down, shared_scale;
    uint32_t *heat,*last,clock;int cap,layer;
    pthread_mutex_t lock;
} MoeW;
typedef struct {
    QMat qkv, z, b, a, out;
    float *conv, *A_log, *dt_bias, *norm;
    float *conv_state, *state;
} GdnW;
typedef struct {
    QMat q, k, v, o;
    float *q_norm, *k_norm;
    float *k_cache, *v_cache;
    uint16_t *k_cache16, *v_cache16;
} AttnW;
typedef struct {
    int type,index;
    float *input_norm, *post_norm;
    GdnW gdn;
    AttnW attn;
    MoeW moe;
} Layer;
typedef struct {
    Cfg c;
    shards S;
    Tok T; int has_tok;
    QMat embed, lm_head;
    float *final_norm;
    Layer *layer;
    int pos, max_seq, quant_mode, kv16, expert_cap; /* 0=floating, 8=int8, 4=grouped int4 */
    int matrix_f32, matrix_i8, matrix_i4;
    int dump_acts, debug_logits;
    int prof_detail;
    double prof_gdn,prof_attn,prof_moe,prof_lm,prof_expert_load;uint64_t prof_expert_misses;
    double dense_load_s;
    pthread_t pf_thread[4];pthread_mutex_t pf_lock;pthread_cond_t pf_cond;
    struct{int layer,eid;}pf_job[128];int pf_head,pf_tail,pf_nthread;
} Model;

typedef struct { int *prompt, nprompt, *full, nfull, *tf, ntf; } Oracle;

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void die(const char *msg){ fprintf(stderr,"%s\n",msg); exit(1); }
static void *xcalloc(size_t n,size_t z){ void *p=calloc(n,z); if(!p){ fprintf(stderr,"OOM %zu bytes\n",n*z); exit(1); } return p; }
static float *falloc(int64_t n){ return xcalloc((size_t)n,sizeof(float)); }
static float sigmoidf_stable(float x){ if(x>=0){ float z=expf(-x); return 1.f/(1.f+z); } float z=expf(x); return z/(1.f+z); }
static float siluf(float x){ return x*sigmoidf_stable(x); }
static float softplusf_stable(float x){ return x>20.f?x:(x<-20.f?expf(x):log1pf(expf(x))); }
static uint16_t f32_to_bf16(float x){uint32_t u;memcpy(&u,&x,4);u+=0x7fffu+((u>>16)&1u);return(uint16_t)(u>>16);}

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
static float qmat_dot_row(const QMat *w,int row,const float *x){
    if(w->fmt==0)return dot(x,w->f+(int64_t)row*w->I,w->I);
    if(w->fmt==1)return dot_q8f(w->q8+(int64_t)row*w->rb,x,w->I)*w->s[row];
    const uint8_t *q=w->q4+(int64_t)row*w->rb;const float *s=w->s+(int64_t)row*w->ng;float out=0.f;
    for(int g=0;g<w->ng;g++){int base=g*w->gs,n=w->gs;if(base+n>w->I)n=w->I-base;if(n>0)out+=dot_q4_group(q+(base>>1),x+base,n)*s[g];}return out;
}
static void qmat_mul_ex(float *y,const float *x,const QMat *w,int allow_idot){
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
    const uint8_t*q=w->q4+(int64_t)row*w->rb;const float*s=w->s+(int64_t)row*w->ng;for(int i=0;i<w->I;i++){uint8_t b=q[i>>1];out[i]=(float)(((i&1)?(b>>4):(b&15))-8)*s[i/w->gs];}
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
static void load_cfg(Cfg *c,const char *snap){
    memset(c,0,sizeof(*c)); char path[2048]; snprintf(path,sizeof(path),"%s/config.json",snap);
    long n; char *buf=read_file(path,&n),*arena=NULL; jval *root=json_parse(buf,&arena); (void)n;
    jval *tc=json_get(root,"text_config"); if(!tc||tc->t!=J_OBJ) tc=root;
    c->hidden=jint(tc,"hidden_size",0); c->n_layers=jint(tc,"num_hidden_layers",0);
    c->vocab=jint(tc,"vocab_size",0); c->max_position=jint(tc,"max_position_embeddings",32768);
    c->eos_token=jint(tc,"eos_token_id",-1);c->pad_token=jint(tc,"pad_token_id",c->eos_token);
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
    if(c->lin_v_heads%c->lin_k_heads) die("linear value heads must be divisible by key heads");
    jval *lt=json_get(tc,"layer_types");
    if(lt&&lt->t==J_ARR&&lt->len==c->n_layers){
        for(int i=0;i<c->n_layers;i++) c->layer_type[i]=(lt->kids[i]->t==J_STR&&!strcmp(lt->kids[i]->str,"full_attention"))?LT_FULL:LT_LINEAR;
    }else for(int i=0;i<c->n_layers;i++) c->layer_type[i]=((i+1)%c->full_interval==0)?LT_FULL:LT_LINEAR;
    free(buf); free(arena);
}

/* ---------- tensor loading; quantized matrices stay packed ---------- */
static int64_t find_named(shards *S,const char *base,char *out,size_t outsz){
    const char *pre[]={"","model.","model.language_model."};
    for(size_t i=0;i<sizeof(pre)/sizeof(pre[0]);i++){ snprintf(out,outsz,"%s%s",pre[i],base); int64_t n=st_numel(S,out); if(n>=0)return n; }
    return -1;
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
    char name[QW_NAME];if(find_named(&m->S,base,name,sizeof(name))<0){fprintf(stderr,"missing tensor %s\n",base);exit(1);}
    st_tensor *t=st_find(&m->S,name);QMat q={.O=rows,.I=cols};int64_t expected=(int64_t)rows*cols;
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
        q.fmt=1;q.rb=cols;if(use_mmap)q.q8=map_tensor(m,name,&q.map_q,&q.map_q_len);else{q.q8=xcalloc((size_t)t->nbytes,1);st_read_raw(&m->S,name,q.q8,0);}m->matrix_i8++;
    }else{
        if(qtype&&qtype!=4)die("unknown matrix qtype tag");
        if(ns%rows||t->nbytes%rows)die("bad int4 matrix payload");q.fmt=4;q.ng=(int)(ns/rows);q.rb=(int)(t->nbytes/rows);q.gs=(q.rb*2)/q.ng;
        if(q.gs<=0||q.ng<=0)die("bad int4 group geometry");if(use_mmap)q.q4=map_tensor(m,name,&q.map_q,&q.map_q_len);else{q.q4=xcalloc((size_t)t->nbytes,1);st_read_raw(&m->S,name,q.q4,0);}m->matrix_i4++;
    }
    return q;
}
static QMat load_qmat(Model*m,const char*base,int rows,int cols){return load_qmat_mode(m,base,rows,cols,0);}
static void free_qmat(QMat*q){free(q->f);if(q->map_s)munmap(q->map_s,q->map_s_len);else free(q->s);if(q->map_q)munmap(q->map_q,q->map_q_len);else{free(q->q8);free(q->q4);}memset(q,0,sizeof(*q));}
static void lname(char *out,size_t cap,int layer,const char *suffix){ snprintf(out,cap,"layers.%d.%s",layer,suffix); }

static void *expert_prefetch_worker(void*arg){
    Model*m=arg;for(;;){pthread_mutex_lock(&m->pf_lock);while(m->pf_head==m->pf_tail)pthread_cond_wait(&m->pf_cond,&m->pf_lock);int li=m->pf_job[m->pf_head].layer,eid=m->pf_job[m->pf_head].eid;m->pf_head=(m->pf_head+1)&127;pthread_mutex_unlock(&m->pf_lock);
#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
        const char*suf[]={"gate_proj","up_proj","down_proj"};for(int k=0;k<3;k++){char base[QW_NAME],name[QW_NAME];snprintf(base,sizeof(base),"layers.%d.mlp.experts.%d.%s.weight",li,eid,suf[k]);if(find_named(&m->S,base,name,sizeof(name))>=0){st_tensor*t=st_find(&m->S,name);if(t)posix_fadvise(t->fd,t->off,t->nbytes,POSIX_FADV_WILLNEED);char qs[QW_NAME+8];snprintf(qs,sizeof(qs),"%s.qs",name);t=st_find(&m->S,qs);if(t)posix_fadvise(t->fd,t->off,t->nbytes,POSIX_FADV_WILLNEED);}}
#else
        (void)li;(void)eid;
#endif
    }return NULL;
}
static void expert_prefetch_start(Model*m){int n=getenv("PREFETCH_THREADS")?atoi(getenv("PREFETCH_THREADS")):1;if(n<0)n=0;if(n>4)n=4;m->pf_nthread=n;if(!n)return;pthread_mutex_init(&m->pf_lock,NULL);pthread_cond_init(&m->pf_cond,NULL);for(int i=0;i<n;i++)pthread_create(&m->pf_thread[i],NULL,expert_prefetch_worker,m);}
static void expert_prefetch_submit(Model*m,int li,int eid){if(!m->pf_nthread)return;pthread_mutex_lock(&m->pf_lock);int next=(m->pf_tail+1)&127;if(next!=m->pf_head){m->pf_job[m->pf_tail].layer=li;m->pf_job[m->pf_tail].eid=eid;m->pf_tail=next;pthread_cond_signal(&m->pf_cond);}pthread_mutex_unlock(&m->pf_lock);}

static void load_expert(Model*m,Expert*e,int li,int eid){Cfg*c=&m->c;const char*mm=getenv("COLI_MMAP");int use_mmap=m->expert_cap<c->n_experts&&mm&&atoi(mm)!=0;char n[QW_NAME];snprintf(n,sizeof(n),"layers.%d.mlp.experts.%d.gate_proj.weight",li,eid);e->gate=load_qmat_mode(m,n,c->moe_inter,c->hidden,use_mmap);snprintf(n,sizeof(n),"layers.%d.mlp.experts.%d.up_proj.weight",li,eid);e->up=load_qmat_mode(m,n,c->moe_inter,c->hidden,use_mmap);snprintf(n,sizeof(n),"layers.%d.mlp.experts.%d.down_proj.weight",li,eid);e->down=load_qmat_mode(m,n,c->hidden,c->moe_inter,use_mmap);e->eid=eid;}
static void load_moe(Model *m,Layer *l,int li){
    Cfg *c=&m->c; char n[QW_NAME]; MoeW *w=&l->moe;
    lname(n,sizeof(n),li,"mlp.gate.weight"); w->router=load_qmat(m,n,c->n_experts,c->hidden);
    w->cap=m->expert_cap;w->layer=li;w->expert=xcalloc(w->cap,sizeof(Expert));w->heat=xcalloc(c->n_experts,sizeof(uint32_t));w->last=xcalloc(c->n_experts,sizeof(uint32_t));pthread_mutex_init(&w->lock,NULL);for(int s=0;s<w->cap;s++){w->expert[s].eid=-1;load_expert(m,&w->expert[s],li,s);}
    lname(n,sizeof(n),li,"mlp.shared_expert.gate_proj.weight"); w->shared_gate=load_qmat(m,n,c->shared_inter,c->hidden);
    lname(n,sizeof(n),li,"mlp.shared_expert.up_proj.weight"); w->shared_up=load_qmat(m,n,c->shared_inter,c->hidden);
    lname(n,sizeof(n),li,"mlp.shared_expert.down_proj.weight"); w->shared_down=load_qmat(m,n,c->hidden,c->shared_inter);
    lname(n,sizeof(n),li,"mlp.shared_expert_gate.weight"); w->shared_scale=load_qmat(m,n,1,c->hidden);
}
static void load_layer(Model *m,int li){
    Cfg *c=&m->c; Layer *l=&m->layer[li]; char n[QW_NAME]; l->type=c->layer_type[li];l->index=li;
    lname(n,sizeof(n),li,"input_layernorm.weight"); l->input_norm=load_vec(m,n,c->hidden);
    lname(n,sizeof(n),li,"post_attention_layernorm.weight"); l->post_norm=load_vec(m,n,c->hidden);
    if(l->type==LT_LINEAR){
        GdnW *w=&l->gdn; int kd=c->lin_k_heads*c->lin_k_dim, vd=c->lin_v_heads*c->lin_v_dim, cd=2*kd+vd;
        lname(n,sizeof(n),li,"linear_attn.in_proj_qkv.weight"); w->qkv=load_qmat(m,n,cd,c->hidden);
        lname(n,sizeof(n),li,"linear_attn.in_proj_z.weight"); w->z=load_qmat(m,n,vd,c->hidden);
        lname(n,sizeof(n),li,"linear_attn.in_proj_b.weight"); w->b=load_qmat(m,n,c->lin_v_heads,c->hidden);
        lname(n,sizeof(n),li,"linear_attn.in_proj_a.weight"); w->a=load_qmat(m,n,c->lin_v_heads,c->hidden);
        lname(n,sizeof(n),li,"linear_attn.conv1d.weight"); w->conv=load_vec(m,n,cd*c->conv_kernel);
        lname(n,sizeof(n),li,"linear_attn.A_log"); w->A_log=load_vec(m,n,c->lin_v_heads);
        lname(n,sizeof(n),li,"linear_attn.dt_bias"); w->dt_bias=load_vec(m,n,c->lin_v_heads);
        lname(n,sizeof(n),li,"linear_attn.norm.weight"); w->norm=load_vec(m,n,c->lin_v_dim);
        lname(n,sizeof(n),li,"linear_attn.out_proj.weight"); w->out=load_qmat(m,n,c->hidden,vd);
        w->conv_state=falloc((int64_t)c->conv_kernel*cd);
        w->state=falloc((int64_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim);
    }else{
        AttnW *w=&l->attn; int qrows=c->n_heads*c->head_dim*2, kvrows=c->n_kv_heads*c->head_dim;
        lname(n,sizeof(n),li,"self_attn.q_proj.weight"); w->q=load_qmat(m,n,qrows,c->hidden);
        lname(n,sizeof(n),li,"self_attn.k_proj.weight"); w->k=load_qmat(m,n,kvrows,c->hidden);
        lname(n,sizeof(n),li,"self_attn.v_proj.weight"); w->v=load_qmat(m,n,kvrows,c->hidden);
        lname(n,sizeof(n),li,"self_attn.o_proj.weight"); w->o=load_qmat(m,n,c->hidden,c->n_heads*c->head_dim);
        lname(n,sizeof(n),li,"self_attn.q_norm.weight"); w->q_norm=load_vec(m,n,c->head_dim);
        lname(n,sizeof(n),li,"self_attn.k_norm.weight"); w->k_norm=load_vec(m,n,c->head_dim);
        if(m->kv16){w->k_cache16=xcalloc((int64_t)m->max_seq*kvrows,sizeof(uint16_t));w->v_cache16=xcalloc((int64_t)m->max_seq*kvrows,sizeof(uint16_t));}
        else{w->k_cache=falloc((int64_t)m->max_seq*kvrows);w->v_cache=falloc((int64_t)m->max_seq*kvrows);}
    }
    load_moe(m,l,li);
}
static void model_init(Model *m,const char *snap){
    memset(m,0,sizeof(*m)); load_cfg(&m->c,snap); int ctx=getenv("CTX")?atoi(getenv("CTX")):512;
    m->kv16=getenv("KV16")&&atoi(getenv("KV16"))!=0;
    m->expert_cap=getenv("EXPERT_RAM")?atoi(getenv("EXPERT_RAM")):m->c.n_experts;if(m->expert_cap<m->c.topk)m->expert_cap=m->c.topk;if(m->expert_cap>m->c.n_experts)m->expert_cap=m->c.n_experts;
    if(ctx<=0)ctx=512; if(ctx>m->c.max_position)ctx=m->c.max_position; m->max_seq=ctx;
    st_init(&m->S,snap);
    char ename[QW_NAME]; if(find_named(&m->S,"embed_tokens.weight",ename,sizeof(ename))<0)die("missing embedding tensor");
    st_tensor *et=st_find(&m->S,ename); if(et->dtype==3)m->quant_mode=et->nbytes==(int64_t)m->c.vocab*m->c.hidden?8:4;
    char tp[2048]; snprintf(tp,sizeof(tp),"%s/tokenizer.json",snap);
    FILE *f=fopen(tp,"rb"); if(f){fclose(f);tok_load(&m->T,tp);m->has_tok=1;}
    double t0=now_s(); Cfg *c=&m->c;
    m->embed=load_qmat(m,"embed_tokens.weight",c->vocab,c->hidden); m->final_norm=load_vec(m,"norm.weight",c->hidden);
    char name[QW_NAME]; m->lm_head=find_named(&m->S,"lm_head.weight",name,sizeof(name))>=0?load_qmat(m,"lm_head.weight",c->vocab,c->hidden):m->embed;
    m->layer=xcalloc(c->n_layers,sizeof(Layer)); for(int i=0;i<c->n_layers;i++) load_layer(m,i);
    expert_prefetch_start(m);
    m->dense_load_s=now_s()-t0; m->dump_acts=getenv("DUMP_ACTS")&&strcmp(getenv("DUMP_ACTS"),"0");
    m->debug_logits=getenv("DEBUG_LOGITS")&&strcmp(getenv("DEBUG_LOGITS"),"0");
    m->prof_detail=getenv("PROF_DETAIL")&&atoi(getenv("PROF_DETAIL"))!=0;
}

/* ---------- hybrid forward ---------- */
static void model_reset(Model *m){
    m->pos=0; Cfg *c=&m->c;
    for(int i=0;i<c->n_layers;i++) if(m->layer[i].type==LT_LINEAR){
        int kd=c->lin_k_heads*c->lin_k_dim, vd=c->lin_v_heads*c->lin_v_dim, cd=2*kd+vd;
        memset(m->layer[i].gdn.conv_state,0,(size_t)c->conv_kernel*cd*sizeof(float));
        memset(m->layer[i].gdn.state,0,(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim*sizeof(float));
    }
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
    qmat_mul_ex(raw,x,&w->qkv,0); qmat_mul_ex(z,x,&w->z,0); qmat_mul_ex(b,x,&w->b,0); qmat_mul_ex(a,x,&w->a,0);
    memcpy(w->conv_state+(int64_t)(pos%K)*cd,raw,(size_t)cd*sizeof(float));
    for(int ch=0;ch<cd;ch++){
        float acc=0.f;
        for(int tap=0;tap<K;tap++){ int src=pos-(K-1-tap); if(src>=0) acc+=w->conv[(int64_t)ch*K+tap]*w->conv_state[(int64_t)(src%K)*cd+ch]; }
        mix[ch]=siluf(acc);
    }
    gdn_rule_step(core,w->state,mix,mix+kd,mix+2*kd,z,a,b,w->A_log,w->dt_bias,w->norm,kh,vh,dk,dv,c->eps);
    qmat_mul_ex(out,core,&w->out,0); free(raw);free(mix);free(z);free(a);free(b);free(core);
}
static void rope_head(float *x,int hd,int rd,int pos,float theta){
    int half=rd/2; for(int i=0;i<half;i++){ float angle=(float)pos*powf(theta,-2.f*(float)i/(float)rd),co=cosf(angle),si=sinf(angle); float a=x[i],b=x[i+half]; x[i]=a*co-b*si; x[i+half]=b*co+a*si; } (void)hd;
}
static void attn_forward(Model *m,Layer *l,const float *x,float *out){
    Cfg *c=&m->c; AttnW *w=&l->attn; int nh=c->n_heads,nkv=c->n_kv_heads,hd=c->head_dim,pos=m->pos,rd=(int)(hd*c->partial_rotary);
    int qrows=nh*hd*2,kvrows=nkv*hd; float *qp=falloc(qrows),*q=falloc(nh*hd),*gate=falloc(nh*hd),*k=falloc(kvrows),*v=falloc(kvrows),*ctx=falloc(nh*hd),*scores=falloc((int64_t)nh*(pos+1));
    qmat_mul_ex(qp,x,&w->q,0); qmat_mul_ex(k,x,&w->k,0); qmat_mul_ex(v,x,&w->v,0);
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
static void mlp_one(float *out,const float *x,const QMat *gate,const QMat *up,const QMat *down,int I){
    float *g=falloc(I),*u=falloc(I); qmat_mul(g,x,gate); qmat_mul(u,x,up); for(int i=0;i<I;i++)g[i]=siluf(g[i])*u[i]; qmat_mul(out,g,down); free(g);free(u);
}
static void router_topk(const float *logit,int E,int K,int *idx,float *weight){
    float mx=-INFINITY;for(int e=0;e<E;e++)if(logit[e]>mx)mx=logit[e];float *prob=falloc(E),den=0.f;
    for(int e=0;e<E;e++){prob[e]=expf(logit[e]-mx);den+=prob[e];}for(int e=0;e<E;e++)prob[e]/=den;
    for(int j=0;j<K;j++){int best=-1;float bv=-1.f;for(int e=0;e<E;e++){int used=0;for(int p=0;p<j;p++)if(idx[p]==e)used=1;if(!used&&prob[e]>bv){bv=prob[e];best=e;}}idx[j]=best;weight[j]=bv;}
    float topden=0.f;for(int j=0;j<K;j++)topden+=weight[j];for(int j=0;j<K;j++)weight[j]/=topden;free(prob);
}
static Expert *expert_load_impl(Model*m,MoeW*w,int eid,const int*protect,int np){
    pthread_mutex_lock(&w->lock);w->clock++;if(w->heat[eid]!=UINT32_MAX)w->heat[eid]++;w->last[eid]=w->clock;
    if((w->clock&4095u)==0)tier_decay(w->heat,m->c.n_experts);
    for(int s=0;s<w->cap;s++)if(w->expert[s].eid==eid){Expert*e=&w->expert[s];pthread_mutex_unlock(&w->lock);return e;}
    int slot=-1;uint64_t cold=UINT64_MAX;for(int s=0;s<w->cap;s++){int old=w->expert[s].eid,held=0;for(int j=0;j<np;j++)if(protect[j]==old)held=1;if(held)continue;uint64_t score=old<0?0:tier_lfru_score(w->heat[old],w->last[old],w->clock);if(slot<0||score<cold){slot=s;cold=score;}}
    if(slot<0)die("expert cache smaller than routed top-k");Expert*e=&w->expert[slot];free_qmat(&e->gate);free_qmat(&e->up);free_qmat(&e->down);double t0=m->prof_detail?now_s():0.;load_expert(m,e,w->layer,eid);if(m->prof_detail){m->prof_expert_load+=now_s()-t0;m->prof_expert_misses++;}
    if(getenv("TIER_TRACE"))fprintf(stderr,"[TIER] layer=%d slot=%d expert=%d\n",w->layer,slot,eid);pthread_mutex_unlock(&w->lock);return e;
}
static void moe_forward(Model *m,Layer *l,const float *x,float *out){
    Cfg *c=&m->c; MoeW *w=&l->moe; int E=c->n_experts,K=c->topk,H=c->hidden; float *logit=falloc(E),*tmp=falloc((int64_t)K*H),*shared=falloc(H);
    int *idx=xcalloc(K,sizeof(int)); float *weight=falloc(K);Expert**chosen=xcalloc(K,sizeof(Expert*)); qmat_mul_ex(logit,x,&w->router,0); router_topk(logit,E,K,idx,weight);for(int j=0;j<K;j++)expert_prefetch_submit(m,w->layer,idx[j]);for(int j=0;j<K;j++)chosen[j]=expert_load_impl(m,w,idx[j],idx,K);
    memset(out,0,(size_t)H*sizeof(float));
    int par=parallel_work((int64_t)K*(2*c->moe_inter*H+c->moe_inter*H));
    #pragma omp parallel for schedule(static) if(par)
    for(int j=0;j<K;j++)mlp_one(tmp+(int64_t)j*H,x,&chosen[j]->gate,&chosen[j]->up,&chosen[j]->down,c->moe_inter);
    for(int j=0;j<K;j++)for(int h=0;h<H;h++)out[h]+=tmp[(int64_t)j*H+h]*weight[j];
    mlp_one(shared,x,&w->shared_gate,&w->shared_up,&w->shared_down,c->shared_inter); float sglog;qmat_mul_ex(&sglog,x,&w->shared_scale,0);float sg=sigmoidf_stable(sglog);
    for(int h=0;h<H;h++)out[h]+=shared[h]*sg;
    free(logit);free(tmp);free(shared);free(idx);free(weight);free(chosen);
}
static void moe_prefill_grouped(Model*m,Layer*l,const float*x,int T,float*out){
    Cfg*c=&m->c;MoeW*w=&l->moe;int E=c->n_experts,K=c->topk,H=c->hidden;int*idx=xcalloc((int64_t)T*K,sizeof(int));float*weight=falloc((int64_t)T*K),*tmp=falloc((int64_t)T*H);unsigned char*used=xcalloc(E,1);
    #pragma omp parallel for schedule(static) if(T>1)
    for(int t=0;t<T;t++){float*logit=falloc(E);qmat_mul_ex(logit,x+(int64_t)t*H,&w->router,0);router_topk(logit,E,K,idx+(int64_t)t*K,weight+(int64_t)t*K);free(logit);}
    for(int t=0;t<T;t++)for(int j=0;j<K;j++)used[idx[(int64_t)t*K+j]]=1;
    #pragma omp parallel for schedule(static) if(T>1)
    for(int t=0;t<T;t++){float*ot=out+(int64_t)t*H;mlp_one(ot,x+(int64_t)t*H,&w->shared_gate,&w->shared_up,&w->shared_down,c->shared_inter);float sg;qmat_mul_ex(&sg,x+(int64_t)t*H,&w->shared_scale,0);sg=sigmoidf_stable(sg);for(int h=0;h<H;h++)ot[h]*=sg;}
    for(int eid=0;eid<E;eid++)if(used[eid])expert_prefetch_submit(m,w->layer,eid);
    for(int eid=0;eid<E;eid++)if(used[eid]){Expert*e=expert_load_impl(m,w,eid,NULL,0);
        #pragma omp parallel for schedule(static) if(T>1)
        for(int t=0;t<T;t++){float wt=0.f;for(int j=0;j<K;j++)if(idx[(int64_t)t*K+j]==eid){wt=weight[(int64_t)t*K+j];break;}if(wt==0.f)continue;float*tt=tmp+(int64_t)t*H;mlp_one(tt,x+(int64_t)t*H,&e->gate,&e->up,&e->down,c->moe_inter);float*ot=out+(int64_t)t*H;for(int h=0;h<H;h++)ot[h]+=tt[h]*wt;}
    }
    free(idx);free(weight);free(tmp);free(used);
}
static void dump_hidden(Model *m,int li,const float *x){
    if(!m->dump_acts)return; const char *dir=getenv("ACTS_DIR");if(!dir)dir="acts"; mkdir(dir,0755); char p[2048];snprintf(p,sizeof(p),"%s/layer-%03d-token-%06d.f32",dir,li,m->pos);FILE*f=fopen(p,"wb");if(f){fwrite(x,sizeof(float),m->c.hidden,f);fclose(f);}
}
static void forward_token(Model *m,int token,float *logits){
    Cfg *c=&m->c; if(token<0||token>=c->vocab){fprintf(stderr,"token %d outside model vocab %d\n",token,c->vocab);exit(1);} if(m->pos>=m->max_seq)die("CTX exhausted");
    float *x=falloc(c->hidden),*n=falloc(c->hidden),*mix=falloc(c->hidden),*moe=falloc(c->hidden); qmat_row(x,&m->embed,token);
    for(int li=0;li<c->n_layers;li++){
        Layer *l=&m->layer[li]; rmsnorm_zero(n,x,l->input_norm,c->hidden,c->eps);double t0=m->prof_detail?now_s():0.;if(l->type==LT_LINEAR)gdn_forward(m,l,n,mix);else attn_forward(m,l,n,mix);if(m->prof_detail){if(l->type==LT_LINEAR)m->prof_gdn+=now_s()-t0;else m->prof_attn+=now_s()-t0;}
        for(int i=0;i<c->hidden;i++)x[i]+=mix[i]; rmsnorm_zero(n,x,l->post_norm,c->hidden,c->eps);t0=m->prof_detail?now_s():0.;moe_forward(m,l,n,moe);if(m->prof_detail)m->prof_moe+=now_s()-t0;for(int i=0;i<c->hidden;i++)x[i]+=moe[i]; dump_hidden(m,li,x);
    }
    rmsnorm_zero(n,x,m->final_norm,c->hidden,c->eps);double t0=m->prof_detail?now_s():0.;qmat_mul_ex(logits,n,&m->lm_head,1);if(m->prof_detail)m->prof_lm+=now_s()-t0;m->pos++;free(x);free(n);free(mix);free(moe);
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
static double forward_prefill_core(Model*m,const int*token,int T,float*logits,int score_first,int*score_count,int grouped_moe){
    Cfg*c=&m->c;if(T<=0||m->pos!=0||T>m->max_seq)die("prefill requires a fresh model and valid length");float*x=falloc((int64_t)T*c->hidden),*n=falloc((int64_t)T*c->hidden),*mix=falloc((int64_t)T*c->hidden),*moe=falloc(c->hidden);
    for(int t=0;t<T;t++){if(token[t]<0||token[t]>=c->vocab)die("prefill token outside vocab");qmat_row(x+(int64_t)t*c->hidden,&m->embed,token[t]);}
    for(int li=0;li<c->n_layers;li++){Layer*l=&m->layer[li];for(int t=0;t<T;t++)rmsnorm_zero(n+(int64_t)t*c->hidden,x+(int64_t)t*c->hidden,l->input_norm,c->hidden,c->eps);if(l->type==LT_LINEAR)gdn_prefill_layer(m,l,n,T,mix);else for(int t=0;t<T;t++){m->pos=t;attn_forward(m,l,n+(int64_t)t*c->hidden,mix+(int64_t)t*c->hidden);}for(int t=0;t<T;t++){float*xt=x+(int64_t)t*c->hidden;for(int i=0;i<c->hidden;i++)xt[i]+=mix[(int64_t)t*c->hidden+i];rmsnorm_zero(n+(int64_t)t*c->hidden,xt,l->post_norm,c->hidden,c->eps);}if(grouped_moe){moe_prefill_grouped(m,l,n,T,mix);for(int t=0;t<T;t++){float*xt=x+(int64_t)t*c->hidden;for(int i=0;i<c->hidden;i++)xt[i]+=mix[(int64_t)t*c->hidden+i];}}else for(int t=0;t<T;t++){float*xt=x+(int64_t)t*c->hidden;moe_forward(m,l,n+(int64_t)t*c->hidden,moe);for(int i=0;i<c->hidden;i++)xt[i]+=moe[i];}}
    m->pos=T;double nll=0.;int scored=0;
    if(score_first<0){rmsnorm_zero(n,x+(int64_t)(T-1)*c->hidden,m->final_norm,c->hidden,c->eps);qmat_mul_ex(logits,n,&m->lm_head,1);}
    else for(int t=score_first;t<T-1;t++){rmsnorm_zero(n,x+(int64_t)t*c->hidden,m->final_norm,c->hidden,c->eps);qmat_mul_ex(logits,n,&m->lm_head,1);float mx=-INFINITY;for(int j=0;j<c->vocab;j++)if(logits[j]>mx)mx=logits[j];double den=0.;for(int j=0;j<c->vocab;j++)den+=exp((double)logits[j]-mx);nll+=(double)mx+log(den)-logits[token[t+1]];scored++;}
    if(score_count)*score_count=scored;free(x);free(n);free(mix);free(moe);return nll;
}
static void forward_prefill(Model*m,const int*token,int T,float*logits){(void)forward_prefill_core(m,token,T,logits,-1,NULL,0);}
static void prefill_dispatch(Model*m,const int*token,int T,float*logits){if(m->dump_acts)for(int i=0;i<T;i++)forward_token(m,token[i],logits);else forward_prefill(m,token,T,logits);}
static int argmax(const float *x,int n){int b=0;for(int i=1;i<n;i++)if(x[i]>x[b])b=i;return b;}
static void print_top5(const float *x,int n,int step){ int id[5]={-1,-1,-1,-1,-1};for(int j=0;j<5;j++)for(int i=0;i<n;i++){int used=0;for(int p=0;p<j;p++)if(id[p]==i)used=1;if(!used&&(id[j]<0||x[i]>x[id[j]]))id[j]=i;}fprintf(stderr,"[LOGITS %d]",step);for(int j=0;j<5;j++)fprintf(stderr," %d:%.7g",id[j],x[id[j]]);fputc('\n',stderr); }

/* ---------- oracle replay ---------- */
static int *jints(jval *root,const char *key,int *n){ jval *a=json_get(root,key);if(!a||a->t!=J_ARR)die("bad oracle array");int *v=xcalloc(a->len,sizeof(int));for(int i=0;i<a->len;i++)v[i]=(int)a->kids[i]->num;*n=a->len;return v; }
static Oracle load_oracle(const char *path){ long n;char*b=read_file(path,&n),*arena=NULL;jval*r=json_parse(b,&arena);(void)n;Oracle o={0};o.prompt=jints(r,"prompt_ids",&o.nprompt);o.full=jints(r,"full_ids",&o.nfull);o.tf=jints(r,"tf_pred",&o.ntf);free(b);free(arena);return o; }
static void default_ref_path(char *out,size_t cap,const char *snap){
    const char *base=strrchr(snap,'/');base=base?base+1:snap;const char *ref=strstr(base,"int8")?"ref_qwen_int8.json":(strstr(base,"i4")?"ref_qwen_i4.json":"ref_qwen.json");
    const char *slash=strrchr(snap,'/');if(slash){int n=(int)(slash-snap);snprintf(out,cap,"%.*s/%s",n,snap,ref);}else snprintf(out,cap,"%s",ref);
}
static int run_oracle(Model *m,const char *snap){
    char path[2048];const char *rp=getenv("REF");if(rp)snprintf(path,sizeof(path),"%s",rp);else default_ref_path(path,sizeof(path),snap);Oracle o=load_oracle(path);if(o.ntf!=32||o.nfull!=o.nprompt+o.ntf)die("oracle must contain 32 generated tokens");
    float *logits=falloc(m->c.vocab);model_reset(m);prefill_dispatch(m,o.prompt,o.nprompt,logits);int pass=0;
    for(int i=0;i<o.ntf;i++){int p=argmax(logits,m->c.vocab);if(m->debug_logits)print_top5(logits,m->c.vocab,i);if(p==o.tf[i])pass++;if(i+1<o.ntf)forward_token(m,o.full[o.nprompt+i],logits);}printf("[ORACLE] %d/%d\n",pass,o.ntf);
    model_reset(m);prefill_dispatch(m,o.prompt,o.nprompt,logits);int gp=0;
    for(int i=0;i<o.ntf;i++){int p=argmax(logits,m->c.vocab);if(p==o.full[o.nprompt+i])gp++;if(i+1<o.ntf)forward_token(m,p,logits);}printf("[GREEDY] %d/%d\n",gp,o.ntf);free(logits);return(pass==o.ntf&&gp==o.ntf)?0:2;
}
static int *read_token_ids(const char*path,int*n){FILE*f=fopen(path,"rb");if(!f){perror(path);exit(1);}int cap=4096,*ids=xcalloc(cap,sizeof(int)),v;*n=0;while(fscanf(f,"%d",&v)==1){if(*n==cap){cap*=2;int*p=realloc(ids,(size_t)cap*sizeof(int));if(!p)die("OOM token ids");ids=p;}ids[(*n)++]=v;}fclose(f);return ids;}
static int run_eval_ids(Model*m,const char*path){
    int n;int*ids=read_token_ids(path,&n);if(n<2)die("EVAL_IDS requires at least two token IDs");
    int chunk=getenv("EVAL_CHUNK")?atoi(getenv("EVAL_CHUNK")):n;if(chunk<2)die("EVAL_CHUNK must be at least two");if(chunk>m->max_seq)die("EVAL_CHUNK exceeds CTX");
    int nchunk=n/chunk;if(!getenv("EVAL_CHUNK")){nchunk=1;chunk=n;}if(nchunk<1)die("EVAL_IDS has no complete evaluation chunk");
    /* Match llama-perplexity: reset each non-overlapping context and score
     * only its second half, so every scored token has substantial context. */
    int first=getenv("EVAL_CHUNK")?chunk/2:0,scored=0;float*logits=falloc(m->c.vocab);double nll=0.,t0=now_s();
    for(int c=0;c<nchunk;c++){int count=0,base=c*chunk;model_reset(m);nll+=forward_prefill_core(m,ids+base,chunk,logits,first,&count,1);scored+=count;}
    double sec=now_s()-t0,ppl=exp(nll/scored);printf("[PPL] tokens=%d nll=%.9f ppl=%.9f tok_s=%.3f\n",scored,nll,ppl,scored/sec);free(ids);free(logits);return 0;
}
static int run_prefix_ids(Model*m,const char*path,int ngen){FILE*f=fopen(path,"rb");if(!f){perror(path);return 1;}char*line=NULL;size_t cap=0;ssize_t z;float*logits=falloc(m->c.vocab);int row=0;while((z=getline(&line,&cap,f))>=0){int*ids=xcalloc(m->max_seq,sizeof(int)),n=0;char*p=line,*end;while(*p){long v=strtol(p,&end,10);if(end==p){p++;continue;}if(n>=m->max_seq)die("PREFIX_IDS prompt exceeds CTX");ids[n++]=(int)v;p=end;}if(!n){free(ids);continue;}model_reset(m);prefill_dispatch(m,ids,n,logits);printf("PREFIX %d:",row++);for(int i=0;i<ngen;i++){int id=argmax(logits,m->c.vocab);printf(" %d",id);if(id==m->c.eos_token)break;if(i+1<ngen)forward_token(m,id,logits);}printf("\n");free(ids);}free(line);free(logits);fclose(f);return 0;}

#ifndef QWEN_NO_MAIN
int main(void){
    const char *snap=getenv("SNAP");if(!snap){fprintf(stderr,"set SNAP=<model directory>\n");return 1;}Model m;model_init(&m,snap);Cfg*c=&m.c;int nf=0;for(int i=0;i<c->n_layers;i++)nf+=c->layer_type[i]==LT_FULL;
    const char*format=m.matrix_i8&&m.matrix_i4?"mixed-int8/int4-g128":(m.matrix_i8?"int8":(m.matrix_i4?"int4-g128":"bf16/fp32"));
    printf("colib qwen engine — PHASE 4 CPU validation\n");printf("model: hidden=%d layers=%d (%d full / %d GDN) vocab=%d ctx=%d\n",c->hidden,c->n_layers,nf,c->n_layers-nf,c->vocab,m.max_seq);
    printf("weights loaded in %.2fs, tokenizer=%s, format=%s, matrices=%d/%d/%d f32/i8/i4, experts/layer=%d, KV=%s\n",m.dense_load_s,m.has_tok?"yes":"no",format,m.matrix_f32,m.matrix_i8,m.matrix_i4,m.expert_cap,m.kv16?"bf16":"fp32");if(getenv("LOAD_ONLY")&&atoi(getenv("LOAD_ONLY"))!=0)return 0;if(getenv("TF")&&strcmp(getenv("TF"),"0"))return run_oracle(&m,snap);if(getenv("EVAL_IDS"))return run_eval_ids(&m,getenv("EVAL_IDS"));if(getenv("PREFIX_IDS"))return run_prefix_ids(&m,getenv("PREFIX_IDS"),getenv("NGEN")?atoi(getenv("NGEN")):64);
    int ids[4096],nids=0;const char *prompt=getenv("PROMPT");if(!prompt)prompt=c->vocab<1000?"!":"Hello";char*chatbuf=NULL;if(getenv("CHAT")&&atoi(getenv("CHAT"))!=0){size_t z=strlen(prompt)+128;chatbuf=xcalloc(z,1);snprintf(chatbuf,z,"<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n",prompt);prompt=chatbuf;}
    if(m.has_tok)nids=tok_encode(&m.T,prompt,(int)strlen(prompt),ids,4096);else ids[nids++]=1;if(nids<=0)die("empty prompt");
    float *logits=falloc(c->vocab);model_reset(&m);double pt=now_s();prefill_dispatch(&m,ids,nids,logits);pt=now_s()-pt;m.prof_gdn=m.prof_attn=m.prof_moe=m.prof_lm=m.prof_expert_load=0.;m.prof_expert_misses=0;int ngen=getenv("NGEN")?atoi(getenv("NGEN")):32;int textout=m.has_tok&&getenv("TEXT")&&atoi(getenv("TEXT"))!=0;printf(textout?"text:":"tokens:");double dt=now_s();int made=0;
    for(int i=0;i<ngen;i++){int p=argmax(logits,c->vocab),stop=p==c->eos_token;made++;if(textout&&!stop){char piece[4096];int z=tok_decode(&m.T,&p,1,piece,sizeof(piece)-1);fwrite(piece,1,z,stdout);fflush(stdout);}else if(!textout)printf(" %d",p);if(m.debug_logits)print_top5(logits,c->vocab,i);if(stop)break;if(i+1<ngen)forward_token(&m,p,logits);}dt=now_s()-dt;printf("\n");if(getenv("PROF")&&atoi(getenv("PROF"))!=0)fprintf(stderr,"[PERF] load=%.3fs prefill=%.3fs (%.2f tok/s) decode=%.3fs (%.2f tok/s)\n",m.dense_load_s,pt,nids/pt,dt,made/dt);if(m.prof_detail){double known=m.prof_gdn+m.prof_attn+m.prof_moe+m.prof_lm;fprintf(stderr,"[PERF_DETAIL] gdn=%.3fs attn=%.3fs moe=%.3fs (load=%.3fs misses=%llu) lm=%.3fs other=%.3fs\n",m.prof_gdn,m.prof_attn,m.prof_moe,m.prof_expert_load,(unsigned long long)m.prof_expert_misses,m.prof_lm,dt-known);}free(chatbuf);free(logits);return 0;
}
#endif
