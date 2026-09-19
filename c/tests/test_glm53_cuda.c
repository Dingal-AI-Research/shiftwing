#include "../glm53_cuda.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double value(const GlmCudaMatrix *m, int r, int c) {
    if (!m->fmt) return ((const float *)m->weights)[(size_t)r*m->cols+c];
    if (m->fmt == 1) return ((const int8_t *)m->weights)[(size_t)r*m->cols+c] * (double)m->scales[r];
    int b = ((const uint8_t *)m->weights)[(size_t)r*((m->cols+1)/2)+c/2];
    int q = ((c&1) ? b>>4 : b&15)-8;
    return q * (double)m->scales[(size_t)r*((m->cols+m->group-1)/m->group)+c/m->group];
}
static void reference(float *out, const float *x, int batch, const GlmCudaMatrix *m, int first, int rows) {
    for (int t=0;t<batch;t++) for (int r=0;r<rows;r++) {
        double sum=0;
        for (int c=0;c<m->cols;c++) sum += value(m,first+r,c)*x[(size_t)t*m->cols+c];
        out[(size_t)t*rows+r]=(float)sum;
    }
}
static void close_to(const float *got,const float *want,size_t n) {
    for (size_t i=0;i<n;i++) if (!isfinite(got[i]) || fabsf(got[i]-want[i]) > 2e-4f*(1+fabsf(want[i]))) {
        fprintf(stderr,"mismatch at %zu: %.9g vs %.9g\n",i,got[i],want[i]); abort();
    }
}
static void matrix_case(GlmCuda *ctx,int fmt,int rows,int cols,int batch,int first,int count) {
    size_t stride=fmt==4?(size_t)(cols+1)/2:(size_t)cols*(fmt==0?4:1);
    unsigned char *data=malloc((size_t)rows*stride);
    int group=7,groups=(cols+group-1)/group;
    float *scales=malloc((size_t)rows*groups*sizeof(float));
    for (size_t i=0;i<(size_t)rows*stride;i++) data[i]=(unsigned char)((i*37+11)%256);
    if (!fmt) for (int i=0;i<rows*cols;i++) ((float *)data)[i]=(float)((i*17)%31-15)/64;
    for (int i=0;i<rows*groups;i++) scales[i]=(float)(1+i%7)/128;
    float *x=malloc((size_t)batch*cols*4),*y=malloc((size_t)batch*count*4),*ref=malloc((size_t)batch*count*4);
    for (int i=0;i<batch*cols;i++) x[i]=(float)(i%19-9)/32;
    void *cache=NULL;
    GlmCudaMatrix m={fmt,rows,cols,group,data,scales,&cache};
    reference(ref,x,batch,&m,first,count);
    assert(glm_cuda_mm(ctx,y,x,batch,&m,first,count)); close_to(y,ref,(size_t)batch*count);
    GlmCudaStats before,after; glm_cuda_stats(ctx,&before);
    assert(glm_cuda_mm(ctx,y,x,batch,&m,first,count)); close_to(y,ref,(size_t)batch*count);
    glm_cuda_stats(ctx,&after);
    assert(after.uploaded_bytes-before.uploaded_bytes == (unsigned long long)batch*cols*4);
    free(ref); free(y); free(x); free(scales); free(data);
}
static void experts(GlmCuda *ctx) {
    enum { B=5,I=9,H=13 };
    uint8_t g[H*((I+1)/2)],u[sizeof(g)],d[I*((H+1)/2)];
    float gs[H*3],us[H*3],ds[I*4],x[B*I],y[B*I],ref[B*I],gate[B*H],up[B*H],previous[B*I];
    for (int i=0;i<H*3;i++) gs[i]=us[i]=0.125f;
    for (int i=0;i<I*4;i++) ds[i]=0.0625f;
    for (int i=0;i<B*I;i++) x[i]=(i%7-3)*2.0f;
    GlmCudaMatrix mg={4,H,I,4,g,gs,NULL},mu={4,H,I,4,u,us,NULL},md={4,I,H,4,d,ds,NULL};
    for (int pass=0;pass<2;pass++) {
        // Same host addresses, new expert contents, including partial groups.
        for (size_t i=0;i<sizeof(g);i++) {g[i]=(i*37+pass*13)%256;u[i]=(i*29+pass*31)%256;}
        for (size_t i=0;i<sizeof(d);i++) d[i]=(i*17+pass*7)%256;
        reference(gate,x,B,&mg,0,H); reference(up,x,B,&mu,0,H);
        for (int i=0;i<B*H;i++) {
            double a=fmin(gate[i],1.25),b=fmax(-1.25,fmin(up[i],1.25));
            gate[i]=(float)(a/(1+exp(-a))*b);
        }
        reference(ref,gate,B,&md,0,I);
        assert(glm_cuda_mlp(ctx,y,x,B,&mg,&mu,&md,1.25f)); close_to(y,ref,B*I);
        if (!pass) memcpy(previous,y,sizeof(y)); else assert(memcmp(previous,y,sizeof(y))!=0);
    }
}
int main(void) {
    GlmCuda *ctx=glm_cuda_create(0,(size_t)128<<20);
    if (!ctx) {fprintf(stderr,"CUDA required: %s\n",glm_cuda_error());return 1;}
    matrix_case(ctx,0,29,17,7,3,19);
    matrix_case(ctx,1,29,17,7,3,19);
    matrix_case(ctx,4,29,17,7,3,19);
    // More than one 32 MiB row tile; verifies the output stride between tiles.
    matrix_case(ctx,4,2200,4097,3,3,2195);
    experts(ctx);
    GlmCudaStats stats; glm_cuda_stats(ctx,&stats);
    assert(stats.batched_calls==10 && stats.mlp_calls==2 && stats.peak_bytes<=stats.budget_bytes);
    glm_cuda_destroy(ctx);
    ctx=glm_cuda_create(0,128);
    assert(ctx);
    float w[256]={0},x[32]={0},y[32];
    for (int i=0;i<32;i++) y[i]=123.0f;
    GlmCudaMatrix too_large={0,16,16,0,w,NULL,NULL};
    assert(!glm_cuda_mm(ctx,y,x,2,&too_large,0,16));
    for (int i=0;i<32;i++) assert(y[i]==123.0f);
    glm_cuda_stats(ctx,&stats); assert(stats.peak_bytes<=128 && stats.fallbacks==1);
    glm_cuda_destroy(ctx);
    assert(!glm_cuda_create(9999,(size_t)128<<20));
    puts("PASS GLM CUDA: FP32/int8/int4, partial rows/groups, tiled outputs, resident reuse, changed experts, clamp rules and budget fallback");
    return 0;
}
