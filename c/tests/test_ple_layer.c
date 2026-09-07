/* Qwen4-Exp PLE layer parity against a PyTorch reference.
 * Fixture: tools/make_ple_layer_fixture.py
 *
 * Exercises the sign-preserving sqrt gate (a plain sqrt would drop negative
 * scores) and the dilated depthwise convolution, including the zero history
 * before position 0 that the reference produces by left padding. */
#define QWEN_NO_MAIN
#include "../qwen.c"
#include "fixture_util.h"

int main(void){
    char *buf,*arena;jval *r=fixture_load("fixtures/ple_layer.json",&buf,&arena);
    enum{T=5,HC=4,H=6,D=8,K=4,DIL=3,WIDE=HC*H};
    jval *cfg=json_get(r,"config");
    if((int)json_get(cfg,"tokens")->num!=T||(int)json_get(cfg,"hc_count")->num!=HC||
       (int)json_get(cfg,"hidden")->num!=H||(int)json_get(cfg,"embed_dim")->num!=D||
       (int)json_get(cfg,"kernel")->num!=K||(int)json_get(cfg,"dilation")->num!=DIL){
        fprintf(stderr,"fixture dimensions changed; regenerate the test\n");return 1;
    }
    float eps=(float)json_get(cfg,"eps")->num;
    float hidden[T*WIDE],emb[T*D],key_w[WIDE*D],value_w[H*D];
    float nk[WIDE],nq[WIDE],nc[WIDE],conv_w[WIDE*K],expected[T*WIDE],got[T*WIDE];
    int at=0;flatten_f32(json_get(r,"hidden"),hidden,&at);
    at=0;flatten_f32(json_get(r,"embeddings"),emb,&at);
    at=0;flatten_f32(json_get(r,"key_w"),key_w,&at);
    at=0;flatten_f32(json_get(r,"value_w"),value_w,&at);
    at=0;flatten_f32(json_get(r,"norm_key"),nk,&at);
    at=0;flatten_f32(json_get(r,"norm_query"),nq,&at);
    at=0;flatten_f32(json_get(r,"norm_conv"),nc,&at);
    at=0;flatten_f32(json_get(r,"conv_w"),conv_w,&at);
    at=0;flatten_f32(json_get(r,"expected_output"),expected,&at);

    ple_layer_forward(T,HC,H,D,K,DIL,eps,hidden,emb,key_w,value_w,nk,nq,nc,conv_w,got);
    float d=fixture_maxdiff(got,expected,T*WIDE);
    printf("ple layer fixture: maxdiff %.3g\n",d);
    free(buf);free(arena);
    return d<=2e-6f?0:1;
}
