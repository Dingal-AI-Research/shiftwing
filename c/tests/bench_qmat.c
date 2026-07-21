#define QWEN_NO_MAIN
#include "../qwen.c"

static void fill_i4(QMat *w,int O,int I){
    int ng=(I+127)/128,rb=ng*64;*w=(QMat){.fmt=4,.O=O,.I=I,.gs=128,.rb=rb,.ng=ng};w->q4=xcalloc((size_t)O*rb,1);w->s=falloc((int64_t)O*ng);
    for(int64_t i=0;i<(int64_t)O*rb;i++)w->q4[i]=(uint8_t)(i*29+17);for(int64_t i=0;i<(int64_t)O*ng;i++)w->s[i]=.015625f;
}
int main(int argc,char **argv){
    int O=argc>1?atoi(argv[1]):4096,I=argc>2?atoi(argv[2]):4096,iters=argc>3?atoi(argv[3]):20;if(O<=0||I<=0||iters<=0)return 2;
    QMat w;fill_i4(&w,O,I);float*x=falloc(I),*y=falloc(O);for(int i=0;i<I;i++)x[i]=sinf((float)i);
    qmat_mul(y,x,&w);double t=now_s();for(int k=0;k<iters;k++)qmat_mul(y,x,&w);t=now_s()-t;
    double bytes=(double)iters*O*((double)w.rb+(double)w.ng*sizeof(float));printf("int4-g128 %dx%d: %.2f GB/s, %.2f GEMV/s, checksum %.7g\n",O,I,bytes/t/1e9,iters/t,y[O/2]);return 0;
}
