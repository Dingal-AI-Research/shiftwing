#define QWEN_NO_MAIN
#include "../qwen.c"

static float ref_row(const QMat *w,int r,const float *x){
    float y=0.f;
    if(w->fmt==1){for(int i=0;i<w->I;i++)y+=x[i]*(float)w->q8[(int64_t)r*w->rb+i]*w->s[r];return y;}
    if(w->fmt==2){for(int i=0;i<w->I;i++){uint8_t b=w->q4[(int64_t)r*w->rb+i/4];int q=2*((b>>(2*(i&3)))&3)-3;y+=x[i]*(float)q*w->s[(int64_t)r*w->ng+i/w->gs];}return y;}
    if(w->fmt==3){for(int i=0;i<w->I;i++)y+=x[i]*(float)q3_value(w->q4+(int64_t)r*w->rb,i)*w->s[(int64_t)r*w->ng+i/w->gs];return y;}
    for(int i=0;i<w->I;i++){uint8_t b=w->q4[(int64_t)r*w->rb+i/2];int q=((i&1)?b>>4:b&15)-8;y+=x[i]*(float)q*w->s[(int64_t)r*w->ng+i/w->gs];}
    return y;
}
static int check_i8(int I){
    enum{O=17};QMat w={.fmt=1,.O=O,.I=I,.rb=I};w.q8=xcalloc((size_t)O*I,1);w.s=falloc(O);float*x=falloc(I),y[O];
    for(int i=0;i<I;i++)x[i]=sinf((float)(i*13+1))*.03125f;
    for(int r=0;r<O;r++){w.s[r]=r==0?0.f:ldexpf(1.f,(r%9)-10);for(int i=0;i<I;i++)w.q8[(int64_t)r*I+i]=(int8_t)(((r*37+i*19)%255)-127);}
    qmat_mul_ex(y,x,&w,0);float md=0.f;for(int r=0;r<O;r++){float d=fabsf(y[r]-ref_row(&w,r,x));if(d>md)md=d;}
    free(w.q8);free(w.s);free(x);printf("qmat int8 I=%d maxdiff=%.3g\n",I,md);return md<=1e-5f;
}
static int check_i4(int I){
    enum{O=17,GS=128};int ng=(I+GS-1)/GS,rb=ng*(GS/2);QMat w={.fmt=4,.O=O,.I=I,.gs=GS,.rb=rb,.ng=ng};w.q4=xcalloc((size_t)O*rb,1);w.s=falloc(O*ng);float*x=falloc(I),*row=falloc(I),y[O];
    for(int i=0;i<I;i++)x[i]=cosf((float)(i*11+3))*.0625f;
    for(int r=0;r<O;r++)for(int g=0;g<ng;g++)w.s[(int64_t)r*ng+g]=(r==0)?0.f:ldexpf(1.f,((r+g)%8)-8);
    for(int r=0;r<O;r++)for(int i=0;i<rb*2;i++){int q=((r*5+i*7)%16);uint8_t*p=&w.q4[(int64_t)r*rb+i/2];if(i&1)*p|=(uint8_t)(q<<4);else*p=(uint8_t)q;}
    qmat_mul(y,x,&w);float md=0.f,rd=0.f;for(int r=0;r<O;r++){float d=fabsf(y[r]-ref_row(&w,r,x));if(d>md)md=d;qmat_row(row,&w,r);for(int i=0;i<I;i++){uint8_t b=w.q4[(int64_t)r*rb+i/2];float e=(float)(((i&1)?b>>4:b&15)-8)*w.s[(int64_t)r*ng+i/GS];d=fabsf(row[i]-e);if(d>rd)rd=d;}}
    free(w.q4);free(w.s);free(x);free(row);printf("qmat int4 I=%d maxdiff=%.3g row=%.3g\n",I,md,rd);return md<=2e-6f&&rd==0.f;
}
static int check_i2(int I){
    enum{O=17,GS=128};int ng=(I+GS-1)/GS,rb=ng*(GS/4);QMat w={.fmt=2,.O=O,.I=I,.gs=GS,.rb=rb,.ng=ng};w.q4=xcalloc((size_t)O*rb,1);w.s=falloc(O*ng);float*x=falloc(I),*row=falloc(I),y[O];
    for(int i=0;i<I;i++)x[i]=cosf((float)(i*17+5))*.0625f;
    for(int r=0;r<O;r++)for(int g=0;g<ng;g++)w.s[(int64_t)r*ng+g]=(r==0)?0.f:ldexpf(1.f,((r+g)%8)-8);
    for(int r=0;r<O;r++)for(int i=0;i<rb*4;i++){int code=(r*3+i*5)%4;w.q4[(int64_t)r*rb+i/4]|=(uint8_t)(code<<(2*(i&3)));}
    qmat_mul(y,x,&w);float md=0.f,rd=0.f;for(int r=0;r<O;r++){float d=fabsf(y[r]-ref_row(&w,r,x));if(d>md)md=d;qmat_row(row,&w,r);for(int i=0;i<I;i++){uint8_t b=w.q4[(int64_t)r*rb+i/4];float e=(float)(2*((b>>(2*(i&3)))&3)-3)*w.s[(int64_t)r*ng+i/GS];d=fabsf(row[i]-e);if(d>rd)rd=d;}}
    free(w.q4);free(w.s);free(x);free(row);printf("qmat int2 I=%d maxdiff=%.3g row=%.3g\n",I,md,rd);return md<=2e-6f&&rd==0.f;
}
static void set_q3(uint8_t*w,int i,int code){
    int bit=3*i,byte=bit>>3,shift=bit&7;uint16_t word=(uint16_t)code<<shift;
    w[byte]|=(uint8_t)word;if(shift>5)w[byte+1]|=(uint8_t)(word>>8);
}
static int check_i3_expand(void){
    enum{O=19,I=384,GS=128};int ng=I/GS,rb=I*3/8,drb=I/2;
    QMat w={.fmt=3,.O=O,.I=I,.gs=GS,.rb=rb,.drb=drb,.ng=ng};
    w.q4=xcalloc((size_t)O*rb,1);uint8_t*out=xcalloc((size_t)O*drb,1);
    for(int r=0;r<O;r++)for(int i=0;i<I;i++)set_q3(w.q4+(int64_t)r*rb,i,(r*3+i*5)%8);
    q3_expand_q4(&w,out);int ok=1;
    for(int r=0;r<O;r++)for(int i=0;i<I;i++){
        uint8_t b=out[(int64_t)r*drb+i/2];int got=((i&1)?b>>4:b&15)-8;
        if(got!=q3_value(w.q4+(int64_t)r*rb,i)){ok=0;break;}
    }
    printf("qmat int3->int4 expansion=%s\n",ok?"exact":"FAIL");
    free(w.q4);free(out);return ok;
}
static int check_i3(int I){
    enum{O=17,GS=128};int ng=(I+GS-1)/GS,rb=ng*(GS*3/8);QMat w={.fmt=3,.O=O,.I=I,.gs=GS,.rb=rb,.ng=ng};w.q4=xcalloc((size_t)O*rb,1);w.s=falloc(O*ng);float*x=falloc(I),*row=falloc(I),y[O];
    for(int i=0;i<I;i++)x[i]=cosf((float)(i*23+7))*.0625f;
    for(int r=0;r<O;r++)for(int g=0;g<ng;g++)w.s[(int64_t)r*ng+g]=(r==0)?0.f:((r+g)&1?-1.f:1.f)*ldexpf(1.f,((r+g)%8)-8);
    for(int r=0;r<O;r++)for(int i=0;i<ng*GS;i++)set_q3(w.q4+(int64_t)r*rb,i,(r*3+i*5)%8);
    qmat_mul(y,x,&w);float md=0.f,rd=0.f;for(int r=0;r<O;r++){float d=fabsf(y[r]-ref_row(&w,r,x));if(d>md)md=d;qmat_row(row,&w,r);for(int i=0;i<I;i++){float e=(float)q3_value(w.q4+(int64_t)r*rb,i)*w.s[(int64_t)r*ng+i/GS];d=fabsf(row[i]-e);if(d>rd)rd=d;}}
    free(w.q4);free(w.s);free(x);free(row);printf("qmat int3 I=%d maxdiff=%.3g row=%.3g\n",I,md,rd);return md<=2e-6f&&rd==0.f;
}
static int check_idot(void){
    enum{N=1027};int8_t*w=xcalloc(N,1),*qx=xcalloc(N,1);float*x=falloc(N);int64_t ref=0;
    for(int i=0;i<N;i++){w[i]=(int8_t)(((i*43)%256)-128);x[i]=sinf((float)(i*17+1))*3.f;}
    float sx=qrow_i8(x,qx,N);for(int i=0;i<N;i++)ref+=(int32_t)w[i]*qx[i];int32_t got=dot_i8i8(w,qx,N);float exact=dot_q8f(w,x,N),approx=(float)got*sx,err=fabsf(exact-approx)/(fabsf(exact)+1e-6f);
    printf("VNNI idot integer=%s activation-relerr=%.3g\n",got==ref?"exact":"FAIL",err);free(w);free(qx);free(x);return got==ref&&err<.05f;
}
static int check_i8_parallel_idot(void){
    enum{O=512,I=2048};QMat w={.fmt=1,.O=O,.I=I,.rb=I};w.q8=xcalloc((size_t)O*I,1);w.s=falloc(O);float*x=falloc(I),*y=falloc(O);
    for(int i=0;i<I;i++)x[i]=sinf((float)(i*17+3))*2.f;for(int r=0;r<O;r++){w.s[r]=.00390625f;for(int i=0;i<I;i++)w.q8[(int64_t)r*I+i]=(int8_t)(((r*31+i*7)%255)-127);}
    qmat_mul(y,x,&w);float ref=ref_row(&w,O-1,x),rel=fabsf(y[O-1]-ref)/(fabsf(ref)+1e-6f);int ok=isfinite(y[0])&&isfinite(y[O-1])&&rel<.05f;
    printf("qmat int8 OpenMP IDOT relerr=%.3g\n",rel);free(w.q8);free(w.s);free(x);free(y);return ok;
}
static int check_batch(int fmt){
    enum{B=5,O=23,I=257,LDX=263,GS=128};int ng=(I+GS-1)/GS,rb=fmt==1?I:(fmt==2?ng*GS/4:(fmt==3?ng*GS*3/8:ng*GS/2));
    QMat w={.fmt=fmt,.O=O,.I=I,.gs=fmt==1?0:GS,.rb=rb,.ng=fmt==1?0:ng};
    if(fmt==1){w.q8=xcalloc((size_t)O*rb,1);w.s=falloc(O);for(int r=0;r<O;r++){w.s[r]=ldexpf(1.f,(r%7)-9);for(int i=0;i<I;i++)w.q8[(int64_t)r*rb+i]=(int8_t)(((r*29+i*11)%255)-127);}}
    else{w.q4=xcalloc((size_t)O*rb,1);w.s=falloc(O*ng);for(int r=0;r<O;r++)for(int g=0;g<ng;g++)w.s[(int64_t)r*ng+g]=ldexpf(1.f,((r+g)%7)-7);for(int r=0;r<O;r++)for(int i=0;i<ng*GS;i++){int q=(r*3+i*13)%(fmt==2?4:(fmt==3?8:16));if(fmt==2)w.q4[(int64_t)r*rb+i/4]|=(uint8_t)(q<<(2*(i&3)));else if(fmt==3)set_q3(w.q4+(int64_t)r*rb,i,q);else{uint8_t*p=&w.q4[(int64_t)r*rb+i/2];if(i&1)*p|=(uint8_t)(q<<4);else*p=(uint8_t)q;}}}
    float*x=falloc((int64_t)B*LDX),*yb=falloc((int64_t)B*O),*yr=falloc((int64_t)B*O);for(int b=0;b<B;b++)for(int i=0;i<LDX;i++)x[(int64_t)b*LDX+i]=sinf((float)(b*101+i*17+5))*.25f;
    qmat_mul_batch(yb,x,B,LDX,&w,1);for(int b=0;b<B;b++)qmat_mul_ex(yr+(int64_t)b*O,x+(int64_t)b*LDX,&w,1);
    float md=0.f;for(int i=0;i<B*O;i++){float d=fabsf(yb[i]-yr[i]);if(d>md)md=d;}printf("qmat batch fmt=%d maxdiff=%.3g\n",fmt,md);
    /* Bit-exact for every format, not merely within tolerance.  The batched path
     * must reproduce the per-token GEMV's reduction structure exactly: two zmm
     * accumulators per group over i+=32, reduce, scale, then sum groups in index
     * order.  All four formats already satisfy this, so any drift is a real
     * regression rather than an accepted quantization difference. */
    free(w.q8);free(w.q4);free(w.s);free(x);free(yb);free(yr);return md==0.f;
}
int main(void){
    setenv("IDOT","1",1);
    int ok=check_i3_expand()&&check_idot()&&check_i8_parallel_idot()&&check_batch(1)&&check_batch(2)&&check_batch(3)&&check_batch(4);int i8[]={1,31,64,128,257},i2[]={1,31,64,128,129,256,257},i3[]={1,31,64,128,129,256,257},i4[]={1,31,64,128,129,256,257};
    for(size_t i=0;i<sizeof(i8)/sizeof(i8[0]);i++)ok&=check_i8(i8[i]);
    for(size_t i=0;i<sizeof(i2)/sizeof(i2[0]);i++)ok&=check_i2(i2[i]);
    for(size_t i=0;i<sizeof(i3)/sizeof(i3[0]);i++)ok&=check_i3(i3[i]);
    for(size_t i=0;i<sizeof(i4)/sizeof(i4[0]);i++)ok&=check_i4(i4[i]);return ok?0:1;
}
