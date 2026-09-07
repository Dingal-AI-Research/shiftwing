/* Qwen4-Exp PLE n-gram index parity against a Python reference.
 * Fixture: tools/make_ple_ngram_fixture.py
 *
 * Checks both halves: the EOS-segment-aware shift window, and the multiply-xor
 * hash reduced per head. The fixture sequence deliberately contains two EOS
 * tokens, a repeated token, and both vocabulary extremes. */
#define QWEN_NO_MAIN
#include "../qwen.c"
#include "fixture_util.h"

/* Multipliers reach ~2.4e13, well past int32; they are exact as doubles
 * (< 2^53) so the JSON round-trip is lossless, but flatten_i32 would truncate. */
static void flatten_i64(jval *v,int64_t *out,int *at){
    if(v->t==J_NUM){out[(*at)++]=(int64_t)v->num;return;}
    if(v->t!=J_ARR){fprintf(stderr,"fixture value is not numeric/array\n");exit(1);}
    for(int i=0;i<v->len;i++)flatten_i64(v->kids[i],out,at);
}

int main(void){
    char *buf,*arena;jval *r=fixture_load("fixtures/ple_ngram.json",&buf,&arena);
    enum{T=10,NGRAM=3,HPN=8,HEADS=(NGRAM-1)*HPN};
    jval *cfg=json_get(r,"config");
    if((int)json_get(cfg,"tokens")->num!=T||(int)json_get(cfg,"ngram_size")->num!=NGRAM||
       (int)json_get(cfg,"heads_per_ngram")->num!=HPN||(int)json_get(cfg,"ngram_heads")->num!=HEADS){
        fprintf(stderr,"fixture dimensions changed; regenerate the test\n");return 1;
    }
    int eos=(int)json_get(cfg,"eos")->num;
    int tokens[T],exp_shift[NGRAM*T];int64_t mult[NGRAM],vocab[HEADS],offset[HEADS],exp_ids[T*HEADS];
    int at=0;flatten_i32(json_get(r,"input_ids"),tokens,&at);
    at=0;flatten_i32(json_get(r,"expected_shifted"),exp_shift,&at);
    at=0;flatten_i64(json_get(r,"layer_multipliers"),mult,&at);
    at=0;flatten_i64(json_get(r,"ngram_heads_vocab_sizes"),vocab,&at);
    at=0;flatten_i64(json_get(r,"ngram_heads_offsets"),offset,&at);
    at=0;flatten_i64(json_get(r,"expected_ngram_ids"),exp_ids,&at);

    int shift_bad=0,id_bad=0;
    for(int position=0;position<T;position++){
        int window[NGRAM];int64_t got[HEADS];
        ple_shift_window(tokens,position,NGRAM,eos,window);
        for(int s=0;s<NGRAM;s++) shift_bad += window[s]!=exp_shift[s*T+position];
        ple_ngram_ids(window,NGRAM,HPN,mult,vocab,offset,got);
        for(int h=0;h<HEADS;h++) id_bad += got[h]!=exp_ids[(size_t)position*HEADS+h];
    }
    printf("ple n-gram fixture: shift_mismatches %d id_mismatches %d (of %d/%d)\n",
           shift_bad,id_bad,NGRAM*T,T*HEADS);
    free(buf);free(arena);
    return(!shift_bad&&!id_bad)?0:1;
}
