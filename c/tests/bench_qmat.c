#define QWEN_NO_MAIN
#include "../qwen.c"

static void fill_i4(QMat *w,int O,int I,int gs){
    int ng=(I+gs-1)/gs,rb=ng*(gs/2);*w=(QMat){.fmt=4,.O=O,.I=I,.gs=gs,.rb=rb,.ng=ng};w->q4=xcalloc((size_t)O*rb,1);w->s=falloc((int64_t)O*ng);
    for(int64_t i=0;i<(int64_t)O*rb;i++)w->q4[i]=(uint8_t)(i*29+17);for(int64_t i=0;i<(int64_t)O*ng;i++)w->s[i]=.015625f;
}

/* Batched throughput at a given token count.  The point of the B sweep is that a
 * GEMM amortizes the weight read and the int4 dequant across all B activations,
 * so GMAC/s must RISE with B.  A flat curve means the batch path is really B
 * independent GEMVs, and a falling GB/s means the activation matrix is being
 * re-streamed once per output row. */
static void run(QMat *w,int O,int I,int iters,int B){
    float *x=falloc((int64_t)B*I),*y=falloc((int64_t)B*O);
    for(int64_t i=0;i<(int64_t)B*I;i++)x[i]=sinf((float)i);
    if(B==1)qmat_mul(y,x,w);else qmat_mul_batch(y,x,B,I,w,1);
    double t=now_s();
    for(int k=0;k<iters;k++){if(B==1)qmat_mul(y,x,w);else qmat_mul_batch(y,x,B,I,w,1);}
    t=now_s()-t;
    double mac=(double)iters*(double)B*(double)O*(double)I;
    double wbytes=(double)iters*O*((double)w->rb+(double)w->ng*sizeof(float));
    double xbytes=(double)iters*(double)B*I*sizeof(float);
    printf("  B=%-5d %8.2f GMAC/s   weights %7.2f GB/s   acts %8.2f GB/s   %8.3f ms/call   checksum %.7g\n",
           B,mac/t/1e9,wbytes/t/1e9,xbytes/t/1e9,t*1e3/iters,y[(int64_t)(B/2)*O+O/2]);
    free(x);free(y);
}

int main(int argc,char **argv){
    int O=argc>1?atoi(argv[1]):4096,I=argc>2?atoi(argv[2]):4096,iters=argc>3?atoi(argv[3]):20;
    int gs=getenv("BENCH_GS")?atoi(getenv("BENCH_GS")):128;
    if(O<=0||I<=0||iters<=0)return 2;
    QMat w;fill_i4(&w,O,I,gs);
    printf("int4-g%d %dx%d, %d iters, %d threads\n",gs,O,I,iters,
#ifdef _OPENMP
           omp_get_max_threads()
#else
           1
#endif
           );
    if(argc>4){int B=atoi(argv[4]);if(B<1)return 2;run(&w,O,I,iters,B);}
    else{int bs[]={1,8,64,512,3560};for(size_t i=0;i<sizeof(bs)/sizeof(bs[0]);i++)run(&w,O,I,iters,bs[i]);}
    return 0;
}
