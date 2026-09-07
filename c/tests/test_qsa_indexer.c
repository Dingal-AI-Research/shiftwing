/* Qwen4-Exp QSA indexer selection parity against a PyTorch reference.
 * Fixture: tools/make_qsa_indexer_fixture.py
 *
 * Checks the block scores and the admitted-key mask. Selection feeds an
 * additive attention mask, so the admitted set is what carries meaning, not the
 * top-k ordering -- the fixture compares masks for that reason. The trailing
 * incomplete block must be admitted without being scored. */
#define QWEN_NO_MAIN
#include "../qwen.c"
#include "fixture_util.h"

int main(void){
    char *buf,*arena;jval *r=fixture_load("fixtures/qsa_indexer.json",&buf,&arena);
    enum{VIS=19,HEADS=4,DIM=8,RATIO=4,TOPK=3,BLOCKS=VIS/RATIO};
    jval *cfg=json_get(r,"config");
    if((int)json_get(cfg,"visible")->num!=VIS||(int)json_get(cfg,"n_heads")->num!=HEADS||
       (int)json_get(cfg,"head_dim")->num!=DIM||(int)json_get(cfg,"compress_ratio")->num!=RATIO||
       (int)json_get(cfg,"block_topk")->num!=TOPK||(int)json_get(cfg,"complete_blocks")->num!=BLOCKS){
        fprintf(stderr,"fixture dimensions changed; regenerate the test\n");return 1;
    }
    float query[HEADS*DIM],keys[BLOCKS*DIM],exp_scores[BLOCKS];
    int exp_mask[VIS];
    int at=0;flatten_f32(json_get(r,"query"),query,&at);
    at=0;flatten_f32(json_get(r,"block_keys"),keys,&at);
    at=0;flatten_f32(json_get(r,"expected_scores"),exp_scores,&at);
    at=0;flatten_i32(json_get(r,"expected_mask"),exp_mask,&at);

    float scores[BLOCKS];uint8_t mask[VIS];int heap[TOPK];
    qsa_block_scores(query,HEADS,DIM,keys,BLOCKS,scores);
    qsa_select_mask(scores,BLOCKS,TOPK,RATIO,VIS,heap,mask);

    float d=fixture_maxdiff(scores,exp_scores,BLOCKS);
    int bad=0,admitted=0;
    for(int i=0;i<VIS;i++){bad += mask[i]!=(uint8_t)exp_mask[i];admitted += mask[i];}
    printf("qsa indexer fixture: score maxdiff %.3g mask_mismatches %d admitted %d/%d\n",
           d,bad,admitted,VIS);
    free(buf);free(arena);
    return(d<=2e-6f&&!bad)?0:1;
}
