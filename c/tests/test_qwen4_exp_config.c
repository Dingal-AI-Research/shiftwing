/* Qwen4-Exp config detection and parsing.
 *
 * The fixture config is trimmed verbatim from Qwen/Qwen3.8-Flash-Next, so this
 * pins the real values rather than a hand-written approximation. It guards the
 * two things that are easy to get silently wrong: family detection (Ornith
 * keeps Qwen3.5's architecture strings, so detection must not fire on it) and
 * the one-indexed ple_layer_ids mapping. */
#define QWEN_NO_MAIN
#include "../qwen.c"

static int fail;
static void expect(const char *what,long got,long want){
    if(got!=want){fprintf(stderr,"  %-28s got %ld want %ld\n",what,got,want);fail++;}
}

int main(void){
    Cfg c;
    load_cfg(&c,"fixtures/qwen4_exp_config");
    expect("is_qwen4_exp",c.is_qwen4_exp,1);
    expect("hidden",c.hidden,2560);
    expect("n_layers",c.n_layers,48);
    expect("vocab",c.vocab,248320);
    expect("n_experts",c.n_experts,512);
    expect("topk",c.topk,10);
    expect("moe_inter",c.moe_inter,640);
    expect("hc_count",c.hc_count,4);
    expect("hc_lowrank",c.hc_lowrank,320);
    expect("idx_budget",c.idx_budget,2048);
    expect("idx_ratio",c.idx_ratio,4);
    expect("idx_head_dim",c.idx_head_dim,128);
    expect("idx_n_heads",c.idx_n_heads,4);
    expect("ngram_size",c.ngram_size,3);
    expect("heads_per_ngram",c.heads_per_ngram,8);
    expect("ple_embed_dim",c.ple_embed_dim,2560);
    expect("ngram_shards",c.ngram_shards,128);

    int full=0,ple=0,ple_index=-1;
    for(int i=0;i<c.n_layers;i++){
        full += c.layer_type[i]==LT_FULL;
        if(c.ple_layer[i]){ple++;ple_index=i;}
    }
    expect("full_attention layers",full,12);
    expect("ple layers",ple,1);
    /* ple_layer_ids is [2], one-indexed, so the tensors live on layer 1. */
    expect("ple layer index",ple_index,1);
    expect("ple layer is linear",c.layer_type[ple_index]==LT_LINEAR,1);

    /* Negative case, trimmed from the live Ornith397 container: this family
     * reports Qwen3_5MoeForConditionalGeneration on purpose, so detection must
     * not fire and none of the Qwen4-Exp fields may be populated. */
    Cfg legacy;
    load_cfg(&legacy,"fixtures/qwen3_5_config");
    expect("legacy is_qwen4_exp",legacy.is_qwen4_exp,0);
    expect("legacy n_layers",legacy.n_layers,60);
    expect("legacy hc_count",legacy.hc_count,0);
    expect("legacy idx_budget",legacy.idx_budget,0);
    expect("legacy ngram_shards",legacy.ngram_shards,0);
    int legacy_ple=0;
    for(int i=0;i<legacy.n_layers;i++) legacy_ple += legacy.ple_layer[i];
    expect("legacy ple layers",legacy_ple,0);

    printf("qwen4-exp config: %s (%d checks failed)\n",fail?"FAIL":"ok",fail);
    return fail?1:0;
}
