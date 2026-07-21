#define QWEN_NO_MAIN
#include "../qwen.c"
#include "fixture_util.h"

int main(void){
    char *buf,*arena;jval *r=fixture_load("fixtures/deltanet.json",&buf,&arena);
    enum{T=5,H=2,D=4,C=12};float mix[T*H*C],z[T*H*D],a[T*H],b[T*H],A[H],dt[H],norm[D],expected[T*H*D],states[T*H*D*D];int at=0;
    flatten_f32(json_get(r,"conv_silu_qkv"),mix,&at);at=0;flatten_f32(json_get(r,"z"),z,&at);at=0;flatten_f32(json_get(r,"a"),a,&at);at=0;flatten_f32(json_get(r,"b"),b,&at);
    at=0;flatten_f32(json_get(r,"A_log"),A,&at);at=0;flatten_f32(json_get(r,"dt_bias"),dt,&at);at=0;flatten_f32(json_get(r,"norm_weight"),norm,&at);
    at=0;flatten_f32(json_get(r,"output"),expected,&at);at=0;flatten_f32(json_get(r,"state_trajectory"),states,&at);
    float state[H*D*D]={0},out[H*D],q[H*D],k[H*D],v[H*D],max_out=0.f,max_state=0.f;
    for(int t=0;t<T;t++){
        const float *m=mix+t*H*C;for(int h=0;h<H;h++){memcpy(q+h*D,m+h*C,D*sizeof(float));memcpy(k+h*D,m+h*C+D,D*sizeof(float));memcpy(v+h*D,m+h*C+2*D,D*sizeof(float));}
        gdn_rule_step(out,state,q,k,v,z+t*H*D,a+t*H,b+t*H,A,dt,norm,H,H,D,D,1e-6f);
        float d=fixture_maxdiff(out,expected+t*H*D,H*D);if(d>max_out)max_out=d;d=fixture_maxdiff(state,states+t*H*D*D,H*D*D);if(d>max_state)max_state=d;
    }
    float cq[T*H*D],ck[T*H*D],cv[T*H*D],cg[T*H],cb[T*H],co[T*H*D],cs[H*D*D]={0};
    for(int t=0;t<T;t++)for(int h=0;h<H;h++){const float*m=mix+t*H*C+h*C;memcpy(cq+(t*H+h)*D,m,D*sizeof(float));memcpy(ck+(t*H+h)*D,m+D,D*sizeof(float));memcpy(cv+(t*H+h)*D,m+2*D,D*sizeof(float));cg[t*H+h]=-expf(A[h])*softplusf_stable(a[t*H+h]+dt[h]);cb[t*H+h]=sigmoidf_stable(b[t*H+h]);}
    gdn_prefill_chunked(co,cs,cq,ck,cv,cg,cb,T,H,D,D);for(int t=0;t<T;t++)for(int h=0;h<H;h++){float*p=co+(t*H+h)*D,ms=0.f;for(int j=0;j<D;j++)ms+=p[j]*p[j];float rr=1.f/sqrtf(ms/D+1e-6f);for(int j=0;j<D;j++)p[j]=p[j]*rr*norm[j]*siluf(z[(t*H+h)*D+j]);}
    float chunk_out=fixture_maxdiff(co,expected,T*H*D),chunk_state=fixture_maxdiff(cs,states+(T-1)*H*D*D,H*D*D);
    printf("deltanet fixture: output %.3g state %.3g chunk-output %.3g chunk-state %.3g\n",max_out,max_state,chunk_out,chunk_state);free(buf);free(arena);return(max_out<=1e-4f&&max_state<=1e-5f&&chunk_out<=1e-4f&&chunk_state<=1e-5f)?0:1;
}
