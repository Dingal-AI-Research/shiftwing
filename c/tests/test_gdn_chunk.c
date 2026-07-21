#define QWEN_NO_MAIN
#include "../qwen.c"

static int run_case(int T){
    enum{H=2,DK=4,DV=5};float*q=falloc((int64_t)T*H*DK),*k=falloc((int64_t)T*H*DK),*v=falloc((int64_t)T*H*DV),*g=falloc(T*H),*b=falloc(T*H),*a=falloc((int64_t)T*H*DV),*z=falloc((int64_t)T*H*DV);float sa[H*DK*DV],sz[H*DK*DV];
    for(int64_t i=0;i<(int64_t)T*H*DK;i++){q[i]=sinf((float)(i*7+1))*.3f;k[i]=cosf((float)(i*11+2))*.25f;}
    for(int64_t i=0;i<(int64_t)T*H*DV;i++)v[i]=sinf((float)(i*5+3))*.4f;
    for(int i=0;i<T*H;i++){g[i]=-.003f*(float)(1+i%13);b[i]=.05f+.9f*(float)(i%17)/16.f;}
    for(int i=0;i<H*DK*DV;i++)sa[i]=sz[i]=sinf((float)(i+4))*.02f;
    gdn_prefill_seq(a,sa,q,k,v,g,b,T,H,DK,DV);gdn_prefill_chunked(z,sz,q,k,v,g,b,T,H,DK,DV);
    float od=0.f,sd=0.f;for(int i=0;i<T*H*DV;i++){float d=fabsf(a[i]-z[i]);if(d>od)od=d;}for(int i=0;i<H*DK*DV;i++){float d=fabsf(sa[i]-sz[i]);if(d>sd)sd=d;}
    printf("gdn chunk T=%d output=%.3g state=%.3g\n",T,od,sd);free(q);free(k);free(v);free(g);free(b);free(a);free(z);return od<=1e-4f&&sd<=1e-4f;
}
int main(void){int ok=1;int n[]={1,5,63,64,65,127};for(size_t i=0;i<sizeof(n)/sizeof(n[0]);i++)ok&=run_case(n[i]);return ok?0:1;}
