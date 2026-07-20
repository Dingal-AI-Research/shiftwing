/* colib — from-scratch C inference engine for Qwen3.5 MoE + Ornith-1.0.
 *
 * Architecture (qwen3_5_moe text_config): hybrid stack of 3× Gated DeltaNet
 * (linear attention, recurrent fp32 state, no KV cache) + 1× gated GQA full
 * attention (partial interleaved RoPE), every layer followed by softmax-top-k
 * MoE with one always-on shared expert; optional 1-layer MTP draft head.
 *
 * Design and generic subsystems follow colibri (github.com/JustVugg/colibri,
 * Apache-2.0) — see NOTICE. This file is written fresh for Qwen3.5; nothing
 * GLM-specific (MLA, DSA, sigmoid router) is carried over.
 *
 * PHASE 0 SKELETON: config/tokenizer/safetensors plumbing + embed→norm→lm_head
 * no-op forward (layers not yet implemented). See PLAN.md for the phase gates.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "compat.h"
#include "st.h"
#include "json.h"
#include "tok.h"

#define QW_MAX_LAYERS 128

/* per-layer mixer type (layer_types[] in config.json) */
enum { LT_LINEAR = 0, LT_FULL = 1 };

typedef struct {
    int hidden, n_layers, vocab;
    int n_heads, n_kv_heads, head_dim;        /* full attention (GQA, gated) */
    float partial_rotary;                     /* 0.25 → RoPE on head_dim/4 dims */
    float theta, eps;
    int n_experts, topk, moe_inter, shared_inter;
    int norm_topk;                            /* renorm top-k probs (verify in Phase 1) */
    int lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel;   /* Gated DeltaNet */
    int mtp_layers;                           /* MTP draft head depth (0 = absent) */
    int full_interval;                        /* every Nth layer is full attention */
    signed char layer_type[QW_MAX_LAYERS];    /* LT_LINEAR / LT_FULL */
} Cfg;

typedef struct {
    Cfg c;
    shards S;
    Tok T; int has_tok;
    float *embed, *final_norm, *lm_head;
    double dense_load_s;
} Model;

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static float *falloc(int64_t n){ float *p = malloc(n*sizeof(float)); if(!p){ fprintf(stderr,"OOM %lld floats\n",(long long)n); exit(1);} return p; }

static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps){
    double ms = 0; for(int i=0;i<D;i++) ms += (double)x[i]*x[i];
    float r = 1.f/sqrtf((float)(ms/D)+eps);
    for(int i=0;i<D;i++) out[i] = x[i]*r*w[i];
}

/* y[O] = x[I] @ W^T, W row-major [O,I] */
static void matmul_row(float *y, const float *x, const float *W, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const float *w = W + (int64_t)o*I;
        float acc = 0.f; for(int i=0;i<I;i++) acc += x[i]*w[i];
        y[o] = acc;
    }
}

/* ---------- config ----------
 * Accepts both the full multimodal config (vision_config + text_config — the
 * shape HF ships) and a bare text config (what the converter may write).
 * rope params may live flat or under rope_parameters (transformers ≥5). */
static char *read_file(const char *path, long *n_out){
    FILE *f = fopen(path,"rb"); if(!f){ perror(path); exit(1);}
    fseek(f,0,SEEK_END); long n = ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){ fprintf(stderr,"short read %s\n",path); exit(1);} buf[n]=0; fclose(f);
    if(n_out) *n_out = n;
    return buf;
}

static int jint(jval *o, const char *k, int def){ jval *v = json_get(o,k); return (v && v->t==J_NUM) ? (int)v->num : def; }
static float jnum(jval *o, const char *k, float def){ jval *v = json_get(o,k); return (v && v->t==J_NUM) ? (float)v->num : def; }

static void load_cfg(Cfg *c, const char *snap){
    char path[2048]; snprintf(path,sizeof(path),"%s/config.json",snap);
    long n; char *buf = read_file(path,&n);
    char *arena = NULL; jval *root = json_parse(buf,&arena);
    jval *tc = json_get(root,"text_config"); if(!tc || tc->t!=J_OBJ) tc = root;

    c->hidden      = jint(tc,"hidden_size",0);
    c->n_layers    = jint(tc,"num_hidden_layers",0);
    c->vocab       = jint(tc,"vocab_size",0);
    c->n_heads     = jint(tc,"num_attention_heads",0);
    c->n_kv_heads  = jint(tc,"num_key_value_heads",0);
    c->head_dim    = jint(tc,"head_dim", c->n_heads ? c->hidden/c->n_heads : 0);
    c->eps         = jnum(tc,"rms_norm_eps",1e-6f);
    c->n_experts   = jint(tc,"num_experts",0);
    c->topk        = jint(tc,"num_experts_per_tok",0);
    c->moe_inter   = jint(tc,"moe_intermediate_size",0);
    c->shared_inter= jint(tc,"shared_expert_intermediate_size",0);
    jval *nt = json_get(tc,"norm_topk_prob"); c->norm_topk = (nt && nt->t==J_BOOL) ? nt->boolean : 1;
    c->lin_k_heads = jint(tc,"linear_num_key_heads",0);
    c->lin_v_heads = jint(tc,"linear_num_value_heads",0);
    c->lin_k_dim   = jint(tc,"linear_key_head_dim",0);
    c->lin_v_dim   = jint(tc,"linear_value_head_dim",0);
    c->conv_kernel = jint(tc,"linear_conv_kernel_dim",4);
    c->mtp_layers  = jint(tc,"mtp_num_hidden_layers",0);
    c->full_interval = jint(tc,"full_attention_interval",4);

    jval *rp = json_get(tc,"rope_parameters"); if(!rp || rp->t!=J_OBJ) rp = tc;
    c->theta          = jnum(rp,"rope_theta",10000000.f);
    c->partial_rotary = jnum(rp,"partial_rotary_factor",0.25f);

    if(c->n_layers <= 0 || c->n_layers > QW_MAX_LAYERS){ fprintf(stderr,"bad num_hidden_layers %d\n",c->n_layers); exit(1); }
    jval *lt = json_get(tc,"layer_types");
    if(lt && lt->t==J_ARR && lt->len==c->n_layers){
        for(int i=0;i<c->n_layers;i++)
            c->layer_type[i] = (lt->kids[i]->t==J_STR && strcmp(lt->kids[i]->str,"full_attention")==0) ? LT_FULL : LT_LINEAR;
    } else {
        for(int i=0;i<c->n_layers;i++) c->layer_type[i] = ((i+1)%c->full_interval==0) ? LT_FULL : LT_LINEAR;
    }
    free(buf); free(arena);
}

/* ---------- loading ---------- */
/* Tensor names: HF full model prefixes text weights with model.language_model.*;
 * the converter strips that to model.* (PLAN.md). Accept both plus bare names. */
static int64_t find_named(shards *S, const char *base, char *out, size_t outsz){
    const char *pre[] = { "", "model.", "model.language_model." };
    for(size_t i=0;i<sizeof(pre)/sizeof(pre[0]);i++){
        snprintf(out,outsz,"%s%s",pre[i],base);
        int64_t n = st_numel(S,out);
        if(n >= 0) return n;
    }
    return -1;
}

static float *load_named(Model *m, const char *base){
    char nm[512];
    int64_t n = find_named(&m->S, base, nm, sizeof(nm));
    if(n < 0){ fprintf(stderr,"missing tensor: %s (tried bare/model./model.language_model. prefixes)\n",base); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, nm, p, 0);
    return p;
}

static void model_init(Model *m, const char *snap){
    memset(m,0,sizeof(*m));
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    char tokpath[2048]; snprintf(tokpath,sizeof(tokpath),"%s/tokenizer.json",snap);
    FILE *tf = fopen(tokpath,"rb");
    if(tf){ fclose(tf); tok_load(&m->T, tokpath); m->has_tok = 1; }
    double t0 = now_s();
    m->embed      = load_named(m, "embed_tokens.weight");
    m->final_norm = load_named(m, "norm.weight");
    /* untied lm_head (vocab 248320, tie_word_embeddings false) */
    char nm[512];
    if(find_named(&m->S,"lm_head.weight",nm,sizeof(nm)) >= 0){
        m->lm_head = falloc((int64_t)m->c.vocab*m->c.hidden);
        st_read_f32(&m->S,nm,m->lm_head,0);
    } else {
        m->lm_head = m->embed;   /* tied fallback (tiny fixtures) */
    }
    m->dense_load_s = now_s()-t0;
}

/* ---------- PHASE 0 no-op forward: embed → final norm → lm_head ----------
 * Proves config/tokenizer/safetensors plumbing end-to-end. Replaced by the
 * real hybrid layer stack in Phase 2. */
static int noop_next_token(Model *m, int tok){
    Cfg *c = &m->c;
    float *h = falloc(c->hidden), *logit = falloc(c->vocab);
    rmsnorm_row(h, m->embed + (int64_t)tok*c->hidden, m->final_norm, c->hidden, c->eps);
    matmul_row(logit, h, m->lm_head, c->hidden, c->vocab);
    int best = 0; float bv = logit[0];
    for(int i=1;i<c->vocab;i++) if(logit[i]>bv){ bv=logit[i]; best=i; }
    free(h); free(logit);
    return best;
}

int main(void){
    const char *snap = getenv("SNAP");
    if(!snap){ fprintf(stderr,"set SNAP=<model directory>\n"); return 1; }
    Model m; model_init(&m, snap);
    Cfg *c = &m.c;
    int n_full = 0; for(int i=0;i<c->n_layers;i++) if(c->layer_type[i]==LT_FULL) n_full++;
    printf("colib qwen engine — PHASE 0 skeleton\n");
    printf("model: hidden=%d layers=%d (%d full-attn / %d deltanet) vocab=%d\n",
           c->hidden, c->n_layers, n_full, c->n_layers-n_full, c->vocab);
    printf("attn: %dQ/%dKV hd=%d rope=%.2f×%d theta=%.0f | gdn: %dK/%dV ×%d conv=%d\n",
           c->n_heads, c->n_kv_heads, c->head_dim, c->partial_rotary,
           (int)(c->partial_rotary*c->head_dim), c->theta,
           c->lin_k_heads, c->lin_v_heads, c->lin_v_dim, c->conv_kernel);
    printf("moe: %d experts top-%d inter=%d shared=%d norm_topk=%d | mtp_layers=%d\n",
           c->n_experts, c->topk, c->moe_inter, c->shared_inter, c->norm_topk, c->mtp_layers);
    printf("dense skeleton weights loaded in %.2fs, tokenizer=%s\n", m.dense_load_s, m.has_tok?"yes":"no");

    const char *prompt = getenv("PROMPT"); if(!prompt) prompt = "Hello";
    int ids[512]; int nid = 0;
    if(m.has_tok){
        nid = tok_encode(&m.T, prompt, (int)strlen(prompt), ids, 512);
        printf("prompt \"%s\" → %d tokens:", prompt, nid);
        for(int i=0;i<nid;i++) printf(" %d", ids[i]);
        printf("\n");
    } else { ids[nid++] = 1; }
    int tok = ids[nid-1];
    printf("noop-forward greedy continuation (NOT real inference):");
    for(int i=0;i<8;i++){ tok = noop_next_token(&m, tok); printf(" %d", tok); }
    printf("\n[OK] phase-0 plumbing works\n");
    return 0;
}
