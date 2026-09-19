#include "../backend_cuda.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CHECK(c) do {if (!(c)) {fprintf(stderr,"gemm bench:%d: %s\n",__LINE__,#c);return 1;}} while(0)
static double now(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void) {
    ColiCuda *cuda=NULL;CHECK(!coli_cuda_create(&cuda,0));
    for (int mode=0;mode<2;mode++) {
        int batch=mode?192:1024,rows=mode?2048:32768,cols=mode?4096:1024;
        size_t a=(size_t)batch*cols,w=(size_t)rows*cols/(mode?2:1),ws=mode?(size_t)rows*(cols/32):(size_t)((rows+127)/128)*(cols/128),out_bytes=(size_t)batch*rows*sizeof(float);
        unsigned char *act=malloc(a),*weight=malloc(w),*ascale=malloc(a/128),*wscale=malloc(ws);float *out=malloc(out_bytes),*pinned=NULL;
        CHECK(act && weight && ascale && wscale && out);memset(act,0x38,a);memset(weight,mode?0x22:0x28,w);memset(ascale,127,a/128);memset(wscale,127,ws);
        void *da,*dw,*das,*dws,*dout;
#define UPLOAD(dst,src,bytes) CHECK(!coli_cuda_malloc(cuda,&dst,bytes));CHECK(!coli_cuda_upload(cuda,dst,src,bytes))
        UPLOAD(da,act,a);UPLOAD(dw,weight,w);UPLOAD(das,ascale,a/128);UPLOAD(dws,wscale,ws);CHECK(!coli_cuda_malloc(cuda,&dout,out_bytes));CHECK(!coli_cuda_malloc_host(cuda,(void**)&pinned,out_bytes));CHECK(!coli_cuda_sync(cuda));
        for (int repetition=0;repetition<4;repetition++) {
            double start=now();
            CHECK(!(mode?coli_cuda_dsv4_fp4_gemm(cuda,dout,da,das,dw,dws,batch,rows,cols):coli_cuda_dsv4_fp8_gemm(cuda,dout,da,das,dw,dws,batch,rows,cols)));
            CHECK(!coli_cuda_sync(cuda));double kernel=now()-start;
            start=now();CHECK(!coli_cuda_download(cuda,out,dout,out_bytes));CHECK(!coli_cuda_sync(cuda));double pageable=now()-start;
            start=now();CHECK(!coli_cuda_download(cuda,pinned,dout,out_bytes));CHECK(!coli_cuda_sync(cuda));double pin=now()-start;
            CHECK(isfinite(out[0]) && !memcmp(out,pinned,out_bytes));
            printf("GEMM_BENCH fp=%d repetition=%d batch=%d rows=%d cols=%d kernel_s=%.6f pageable_s=%.6f pinned_s=%.6f tflops=%.4f\n",mode?4:8,repetition,batch,rows,cols,kernel,pageable,pin,2.0*batch*rows*cols/kernel/1e12);fflush(stdout);
        }
        coli_cuda_free(cuda,da);coli_cuda_free(cuda,dw);coli_cuda_free(cuda,das);coli_cuda_free(cuda,dws);coli_cuda_free(cuda,dout);coli_cuda_free_host(cuda,pinned);
        free(act);free(weight);free(ascale);free(wscale);free(out);
    }
    coli_cuda_destroy(cuda);return 0;
}
