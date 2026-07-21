#define QWEN_NO_MAIN
#include "../qwen.c"

static QMat make_i4(int O,int I,uint32_t seed){int ng=(I+127)/128,rb=ng*64;QMat w={.fmt=4,.O=O,.I=I,.gs=128,.rb=rb,.ng=ng};w.q4=xcalloc((size_t)O*rb,1);w.s=falloc((int64_t)O*ng);for(int64_t i=0;i<(int64_t)O*rb;i++)w.q4[i]=(uint8_t)(seed+i*29);for(int64_t i=0;i<(int64_t)O*ng;i++)w.s[i]=.015625f;return w;}
int main(void){
    int L=getenv("STACK_LAYERS")?atoi(getenv("STACK_LAYERS")):40,A=getenv("STACK_ACTIVE")?atoi(getenv("STACK_ACTIVE")):9,H=getenv("STACK_HIDDEN")?atoi(getenv("STACK_HIDDEN")):2048,I=getenv("STACK_INTER")?atoi(getenv("STACK_INTER")):512,N=getenv("STACK_ITERS")?atoi(getenv("STACK_ITERS")):3;if(L<1||A<1||H<1||I<1||N<1)return 2;
    int64_t ne=(int64_t)L*A;QMat*g=xcalloc(ne,sizeof(QMat)),*u=xcalloc(ne,sizeof(QMat)),*d=xcalloc(ne,sizeof(QMat));double packed=0;
    for(int64_t e=0;e<ne;e++){g[e]=make_i4(I,H,(uint32_t)e);u[e]=make_i4(I,H,(uint32_t)e+71);d[e]=make_i4(H,I,(uint32_t)e+149);packed+=g[e].O*(g[e].rb+g[e].ng*4.0)+u[e].O*(u[e].rb+u[e].ng*4.0)+d[e].O*(d[e].rb+d[e].ng*4.0);}
    float*x=falloc(H),*ga=falloc(I),*up=falloc(I),*down=falloc(H),*y=falloc(H);for(int i=0;i<H;i++)x[i]=sinf((float)i)*.1f;double t=now_s();
    for(int it=0;it<N;it++)for(int l=0;l<L;l++){memset(y,0,(size_t)H*sizeof(float));for(int a=0;a<A;a++){int64_t e=(int64_t)l*A+a;qmat_mul(ga,x,&g[e]);qmat_mul(up,x,&u[e]);for(int j=0;j<I;j++)ga[j]=siluf(ga[j])*up[j];qmat_mul(down,ga,&d[e]);for(int j=0;j<H;j++)y[j]+=down[j];}for(int j=0;j<H;j++)x[j]=y[j]/A;}
    t=now_s()-t;printf("synthetic stack L=%d active=%d H=%d I=%d: %.2f tok/s, %.2f GB/s active packed, %.2f GiB weights, checksum %.7g\n",L,A,H,I,N/t,N*packed/t/1e9,packed/1073741824.0,x[H/2]);return 0;
}
