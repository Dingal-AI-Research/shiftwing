#define GLM53_NO_MAIN
#include "../glm53.c"
#include <assert.h>

typedef struct { int n, ends; double last; } TestProgress;
static void progress_check(int completed, double partial, void *context) {
    TestProgress *p=context;
    double position=completed+partial;
    assert(position+1e-8>=p->last && position<=p->n+1e-8);
    if (partial==0 && completed>p->last) p->ends++;
    p->last=position;
}
static void near(const float *a,const float *b,int n) {
    for (int i=0;i<n;i++) {
        if (!isfinite(a[i]) || fabsf(a[i]-b[i])>1e-4f) {
            fprintf(stderr,"prefill mismatch %d: %.9g vs %.9g\n",i,a[i],b[i]);abort();
        }
    }
}
int main(int argc,char **argv) {
    assert(argc==2);
    enum {N=1031,STEPS=3};
    int ids[N];for(int i=0;i<N;i++)ids[i]=(i*13+5)%90;
    g_cap_override=1;
    GModel m={0};model_load(&m,argv[1]);
    setenv("GLM53_PREFILL_CHUNK","128",1);
    GSession *base=session_open(&m,N+STEPS+1);
    if (!base->last_hidden) base->last_hidden=calloc(m.c.hidden,sizeof(float));
    float *all=forward_prefill(&m,base,ids,N,NULL,0,1);
    float *last=batch_floats(m.c.vocab),*hidden=batch_floats(m.c.hidden);
    memcpy(last,all+(size_t)(N-1)*m.c.vocab,m.c.vocab*sizeof(float));
    memcpy(hidden,base->last_hidden,m.c.hidden*sizeof(float));
    int continuation[STEPS];
    float *row=last;
    for(int i=0;i<STEPS;i++) {
        continuation[i]=argmax(row,m.c.vocab);
        float *next=forward_span(&m,base,&continuation[i],1,NULL,0);
        if(row!=last)free(row);row=next;
    }
    free(row);session_close(&m,base);
    const int chunks[]={512,1024,0};
    for(int k=0;k<3;k++) {
        char setting[32];snprintf(setting,sizeof(setting),"%d",chunks[k]);
        if(chunks[k])setenv("GLM53_PREFILL_CHUNK",setting,1);else unsetenv("GLM53_PREFILL_CHUNK");
        for(int keep_all=0;keep_all<=1;keep_all++) {
            GSession *s=session_open(&m,N+STEPS+1);
            if(!s->last_hidden)s->last_hidden=calloc(m.c.hidden,sizeof(float));
            TestProgress p={.n=N};
            float *got=forward_prefill_progress(&m,s,ids,N,NULL,0,keep_all,progress_check,&p);
            assert(s->filled==N && p.last==N);
            if(chunks[k])assert(p.ends==(N+chunks[k]-1)/chunks[k]);
            if(keep_all) {
                for(int t=0;t<N;t++)assert(argmax(got+(size_t)t*m.c.vocab,m.c.vocab)==argmax(all+(size_t)t*m.c.vocab,m.c.vocab));
                near(got,all,N*m.c.vocab);
            }
            near(got+(keep_all?(size_t)(N-1)*m.c.vocab:0),last,m.c.vocab);
            near(s->last_hidden,hidden,m.c.hidden);
            row=got+(keep_all?(size_t)(N-1)*m.c.vocab:0);
            for(int i=0;i<STEPS;i++) {
                int token=argmax(row,m.c.vocab);assert(token==continuation[i]);
                float *next=forward_span(&m,s,&token,1,NULL,0);
                free(got);got=next;row=got;
            }
            free(got);session_close(&m,s);
            printf("PASS prefill chunk=%s all_logits=%d: 1031 tokens, partial tail, continuation, MTP hidden state and progress\n",chunks[k]?setting:"default",keep_all);
        }
    }
    free(hidden);free(last);free(all);model_release(&m);
    return 0;
}
