#include "../backend_cuda.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CHECK(c) do {if (!(c)) {fprintf(stderr,"sparse bench:%d: %s\n",__LINE__,#c);return 1;}} while(0)
static double now(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void) {
    int batch=1024,heads=64,dim=512,selected=128,rows=batch+selected;
    size_t n=(size_t)batch*heads*dim;
    float *q=malloc(n*sizeof(float)),*kv=malloc((size_t)rows*dim*sizeof(float)),sink[64]={0};
    int *ids=malloc((size_t)batch*selected*sizeof(int)),*counts=malloc(batch*sizeof(int));
    CHECK(q && kv && ids && counts);
    for (size_t i=0;i<n;i++) q[i]=(float)((int)(i%29)-14)/16;
    for (int i=0;i<rows*dim;i++) kv[i]=(float)(i%23-11)/16;
    for (int b=0;b<batch;b++) {counts[b]=selected;for (int k=0;k<selected;k++) ids[b*selected+k]=b+k;}
    ColiCuda *cuda=NULL;void *dq,*dkv,*di,*dc,*ds,*out;
    CHECK(!coli_cuda_create(&cuda,0));
#define UPLOAD(dst,src,bytes) CHECK(!coli_cuda_malloc(cuda,&dst,bytes));CHECK(!coli_cuda_upload(cuda,dst,src,bytes))
    UPLOAD(dq,q,n*sizeof(float));UPLOAD(dkv,kv,(size_t)rows*dim*sizeof(float));
    UPLOAD(di,ids,(size_t)batch*selected*sizeof(int));UPLOAD(dc,counts,batch*sizeof(int));UPLOAD(ds,sink,sizeof(sink));
    CHECK(!coli_cuda_malloc(cuda,&out,n*sizeof(float)));CHECK(!coli_cuda_sync(cuda));
    for (int repetition=0;repetition<3;repetition++) for (int mode=0;mode<2;mode++) {
        double start=now();
        if (mode) CHECK(!coli_cuda_dsv4_sparse_attention_batch(cuda,out,dq,dkv,di,dc,ds,batch,heads,dim,selected,1/sqrtf(dim)));
        else for (int b=0;b<batch;b++) {
            CHECK(!coli_cuda_dsv4_sparse_attention(cuda,(float*)out+(size_t)b*heads*dim,(float*)dq+(size_t)b*heads*dim,dkv,(int*)di+(size_t)b*selected,ds,heads,dim,selected,1/sqrtf(dim)));
            CHECK(!coli_cuda_sync(cuda));
        }
        CHECK(!coli_cuda_sync(cuda));
        printf("SPARSE_BENCH repetition=%d batch=%d selected=%d mode=%s seconds=%.6f\n",repetition,batch,selected,mode?"batch":"scalar_sync",now()-start);fflush(stdout);
    }
    coli_cuda_free(cuda,dq);coli_cuda_free(cuda,dkv);coli_cuda_free(cuda,di);coli_cuda_free(cuda,dc);coli_cuda_free(cuda,ds);coli_cuda_free(cuda,out);coli_cuda_destroy(cuda);
    free(q);free(kv);free(ids);free(counts);return 0;
}
