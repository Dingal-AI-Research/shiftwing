/* Opt-in loaded-model comparison. Uses normal served prefill, not generation. */
#define GLM53_NO_MAIN
#include "../glm53.c"
#include <assert.h>

typedef struct { int chunk; double start; } BenchProgress;
static void progress(int completed,double partial,void *context) {
    BenchProgress *p=context;
    printf("{\"event\":\"progress\",\"chunk\":%d,\"completed\":%d,\"partial\":%.4f,\"elapsed_s\":%.3f}\n",p->chunk,completed,partial,now_s()-p->start);
    fflush(stdout);
}
static void warm_matrix(Mat *w) {
#ifdef GLM53_CUDA_BACKEND
    if(w->rows<=0 || w->columns<=0)return;
    GlmCudaMatrix d=gpu_matrix(w,1);
    float *x=calloc((size_t)2*w->columns,sizeof(float)),*y=batch_floats((size_t)2*w->rows);
    if(!glm_cuda_mm(g_glm_cuda,y,x,2,&d,0,w->rows)) {
        fprintf(stderr,"resident warmup failed: %s\n",glm_cuda_error());exit(1);
    }
    free(y);free(x);
#else
    (void)w;
#endif
}
static void reset_experts(GModel *m) {
    for(int i=0;i<m->n_held;i++) {
        LCache *c=&m->ecache[i];c->n=0;
        for(int j=0;j<c->cap;j++) {c->s[j].eid=-1;c->s[j].used=0;}
    }
    memset(m->euse,0,(size_t)m->n_held*m->c.n_experts*sizeof(*m->euse));
    m->clock=0;m->hits=0;m->miss=0;m->ebytes=0;
}
int main(int argc,char **argv) {
    if(argc<5) {fprintf(stderr,"usage: %s MODEL IDS_FILE OUTPUT_DIR CHUNK...\n",argv[0]);return 2;}
    FILE *f=fopen(argv[2],"r");if(!f) {perror(argv[2]);return 2;}
    int ids[16384],n=0;while(n<16384 && fscanf(f,"%d",&ids[n])==1)n++;
    fclose(f);if(n<=0) {fprintf(stderr,"empty input\n");return 2;}
    double load=now_s();GModel m={0};model_load(&m,argv[1]);
#ifdef GLM53_CUDA_BACKEND
    assert(g_glm_cuda && g_batch_prefill);
#else
    fprintf(stderr,"CUDA build required\n");return 2;
#endif
    if(!m.streaming) {fprintf(stderr,"a streaming expert model is required\n");return 2;}
    double loaded=now_s();
    // Give each chunk configuration the same warm shared weights. Expert
    // cache contents/frequencies and all sequence state are reset per run.
    for(int i=0;i<m.n_held;i++) {
        GLayer *l=&m.layer[i];
        Mat *mat[]={&l->kq,&l->kk,&l->kv,&l->ko,&l->kga,&l->kgb,&l->kfa,&l->kfb,&l->kb,
            &l->qa,&l->qb,&l->kva,&l->kvb_kt,&l->kvb_v,&l->o,&l->iwq,&l->iwk,&l->iwp,&l->ikpg,
            &l->dg,&l->du,&l->dd,&l->rg,&l->ru,&l->rd};
        for(size_t j=0;j<sizeof(mat)/sizeof(*mat);j++)warm_matrix(mat[j]);
    }
    printf("{\"event\":\"ready\",\"load_s\":%.3f,\"warmup_s\":%.3f,\"tokens\":%d}\n",loaded-load,now_s()-loaded,n);fflush(stdout);
    for(int k=4;k<argc;k++) {
        int chunk=atoi(argv[k]);if(chunk<=0) {fprintf(stderr,"chunk must be positive\n");return 2;}setenv("GLM53_PREFILL_CHUNK",argv[k],1);
        reset_experts(&m);GSession *s=session_open(&m,n+1);
        BenchProgress p={chunk,now_s()};
        printf("{\"event\":\"begin\",\"chunk\":%d}\n",chunk);fflush(stdout);
        float *logits=forward_prefill_progress(&m,s,ids,n,NULL,0,0,progress,&p);
        double seconds=now_s()-p.start;
        char path[4096];snprintf(path,sizeof(path),"%s/logits-%d.f32",argv[3],chunk);
        f=fopen(path,"wb");if(!f) {perror(path);return 2;}
        size_t written=fwrite(logits,sizeof(float),m.c.vocab,f);
        int closed=fclose(f);
        if(written!=(size_t)m.c.vocab || closed) {fprintf(stderr,"could not save logits\n");return 2;}
        printf("{\"event\":\"result\",\"chunk\":%d,\"tokens\":%d,\"seconds\":%.3f,\"tokens_per_s\":%.4f,\"next_token\":%d,\"expert_bytes\":%llu,\"expert_misses\":%ld,\"expert_hits\":%ld,\"rss_gb\":%.4f}\n",chunk,n,seconds,n/seconds,argmax(logits,m.c.vocab),(unsigned long long)m.ebytes,m.miss,m.hits,rss_gb());fflush(stdout);
        free(logits);session_close(&m,s);
    }
    gpu_release();return 0;
}
