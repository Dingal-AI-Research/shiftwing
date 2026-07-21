#define QWEN_NO_MAIN
#include "../qwen.c"
#include "fixture_util.h"

int main(void){
    char *buf,*arena;jval *r=fixture_load("fixtures/router.json",&buf,&arena);enum{T=7,E=8,K=2};float logits[T*E],expected[T*K],got[T*K];int indices[T*K],expidx[T*K],at=0;
    flatten_f32(json_get(r,"logits"),logits,&at);at=0;flatten_f32(json_get(r,"weights"),expected,&at);at=0;flatten_i32(json_get(r,"indices"),expidx,&at);
    for(int t=0;t<T;t++)router_topk(logits+t*E,E,K,indices+t*K,got+t*K);float d=fixture_maxdiff(got,expected,T*K);int bad=0;for(int i=0;i<T*K;i++)bad+=indices[i]!=expidx[i];
    printf("router fixture: maxdiff %.3g index_mismatches %d\n",d,bad);free(buf);free(arena);return(d<=2e-6f&&!bad)?0:1;
}
