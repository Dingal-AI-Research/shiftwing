/* Qwen4-Exp gated residual (hyper-connection) parity against a PyTorch
 * reference. Fixture: tools/make_hyper_connection_fixture.py
 *
 * Covers all three pieces the decoder layer needs: the collapse to a single
 * hidden vector, the per-stream injection weights, the scatter back into the
 * raw stream, and the use_combine=False mixer that has no injection weights. */
#define QWEN_NO_MAIN
#include "../qwen.c"
#include "fixture_util.h"

int main(void){
    char *buf,*arena;jval *r=fixture_load("fixtures/hyper_connection.json",&buf,&arena);
    enum{T=3,HC=4,H=8,R=6,WIDE=HC*H};
    jval *cfg=json_get(r,"config");
    if((int)json_get(cfg,"tokens")->num!=T||(int)json_get(cfg,"hc_count")->num!=HC||
       (int)json_get(cfg,"hidden")->num!=H||(int)json_get(cfg,"lowrank")->num!=R){
        fprintf(stderr,"fixture dimensions changed; regenerate the test\n");return 1;
    }
    float eps=(float)json_get(cfg,"eps")->num;
    float hyper[T*WIDE],norm_w[WIDE],wd[R*WIDE],wu[WIDE*R],wi[HC*WIDE],block[T*H];
    float exp_mixed[T*H],exp_inj[T*HC],exp_injected[T*WIDE],exp_mixer[T*H];
    int at=0;flatten_f32(json_get(r,"hyper_input"),hyper,&at);
    at=0;flatten_f32(json_get(r,"hc_norm_weight"),norm_w,&at);
    at=0;flatten_f32(json_get(r,"mix_down"),wd,&at);
    at=0;flatten_f32(json_get(r,"mix_up"),wu,&at);
    at=0;flatten_f32(json_get(r,"block_inject"),wi,&at);
    at=0;flatten_f32(json_get(r,"block_out"),block,&at);
    at=0;flatten_f32(json_get(r,"expected_mixed"),exp_mixed,&at);
    at=0;flatten_f32(json_get(r,"expected_injection"),exp_inj,&at);
    at=0;flatten_f32(json_get(r,"expected_injected"),exp_injected,&at);
    at=0;flatten_f32(json_get(r,"expected_mixer_only"),exp_mixer,&at);

    float mixed[T*H],inj[T*HC],injected[T*WIDE],mixer[T*H];
    float s_normed[WIDE],s_low[R],s_gate[WIDE];
    for(int t=0;t<T;t++){
        hc_gated_residual(hyper+t*WIDE,HC,H,R,norm_w,eps,wd,wu,wi,
                          mixed+t*H,inj+t*HC,s_normed,s_low,s_gate);
        memcpy(injected+t*WIDE,hyper+t*WIDE,sizeof(float)*WIDE);
        hc_inject(injected+t*WIDE,block+t*H,inj+t*HC,HC,H);
        hc_gated_residual(hyper+t*WIDE,HC,H,R,norm_w,eps,wd,wu,NULL,
                          mixer+t*H,NULL,s_normed,s_low,s_gate);
    }
    float d_mixed=fixture_maxdiff(mixed,exp_mixed,T*H);
    float d_inj=fixture_maxdiff(inj,exp_inj,T*HC);
    float d_injected=fixture_maxdiff(injected,exp_injected,T*WIDE);
    float d_mixer=fixture_maxdiff(mixer,exp_mixer,T*H);
    printf("hyper-connection fixture: mixed %.3g injection %.3g injected %.3g mixer_only %.3g\n",
           d_mixed,d_inj,d_injected,d_mixer);
    free(buf);free(arena);
    float tol=2e-6f;
    return(d_mixed<=tol&&d_inj<=tol&&d_injected<=tol&&d_mixer<=tol)?0:1;
}
