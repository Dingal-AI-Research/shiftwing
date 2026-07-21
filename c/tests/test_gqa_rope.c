#define QWEN_NO_MAIN
#include "../qwen.c"
#include "fixture_util.h"

int main(void){
    char *buf,*arena;jval *r=fixture_load("fixtures/rope_partial.json",&buf,&arena);enum{T=6,H=2,D=64};float q[T*H*D],expected[T*H*D];int at=0;
    flatten_f32(json_get(r,"q"),q,&at);at=0;flatten_f32(json_get(r,"q_out"),expected,&at);int rd=(int)json_get(r,"rotary_dim")->num;float theta=(float)json_get(r,"theta")->num;
    for(int t=0;t<T;t++)for(int h=0;h<H;h++)rope_head(q+(t*H+h)*D,D,rd,t,theta);float d=fixture_maxdiff(q,expected,T*H*D);
    printf("partial RoPE fixture: maxdiff %.3g\n",d);free(buf);free(arena);return d<=2e-6f?0:1;
}
