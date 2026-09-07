#ifndef COLIB_DEEPSEEK_V4_BLOCK_H
#define COLIB_DEEPSEEK_V4_BLOCK_H

#include "deepseek_v4_compressed_attention.h"
#include "deepseek_v4_indexed_attention.h"
#include "deepseek_v4_tier.h"

typedef struct {
    float *reduced;
    float *post;
    float *combination;
    float *mixes;
    float *module_output;
    float *expanded;
    dsv4_attention_scratch attention;
    float *router_logits;
    float *routed;
    float *shared;
    float *expert_output;
    float *moe_gate;
    float *moe_up;
    uint8_t *moe_activation;
    uint8_t *moe_activation_scale;
    double last_attention_stage_seconds;
    double last_ffn_stage_seconds;
} dsv4_block_scratch;

typedef struct {
    const dsv4_dense_arena *dense;
    int layer;
    dsv4_sliding_attention_state *state;
    dsv4_attention_scratch *scratch;
} dsv4_block_attention_context;

typedef struct {
    const dsv4_dense_arena *dense;
    int layer;
    dsv4_compressed_attention_state *state;
    dsv4_compressed_attention_scratch *scratch;
} dsv4_block_nonoverlap_attention_context;

typedef struct {
    const dsv4_dense_arena *dense;
    int layer;
    dsv4_indexed_attention_state *state;
    dsv4_indexed_attention_scratch *scratch;
} dsv4_block_overlap_attention_context;

typedef struct {
    dsv4_expert_cache *cache;
    const dsv4_dense_arena *dense;
    int layer;
    int token;
    dsv4_block_scratch *scratch;
} dsv4_block_moe_context;

static inline int dsv4_block_attention_module(void *opaque,
                                               const float *input,
                                               float *output) {
    dsv4_block_attention_context *context =
        (dsv4_block_attention_context *)opaque;
    return dsv4_attention_decode_sliding(
        context->dense, context->layer, input, context->state,
        context->scratch, output);
}

static inline int dsv4_block_nonoverlap_attention_module(
    void *opaque, const float *input, float *output) {
    dsv4_block_nonoverlap_attention_context *context =
        (dsv4_block_nonoverlap_attention_context *)opaque;
    return dsv4_attention_decode_compressed_nonoverlap(
        context->dense, context->layer, input, context->state,
        context->scratch, output);
}

static inline int dsv4_block_overlap_attention_module(
    void *opaque, const float *input, float *output) {
    dsv4_block_overlap_attention_context *context =
        (dsv4_block_overlap_attention_context *)opaque;
    return dsv4_attention_decode_compressed_overlap(
        context->dense, context->layer, input, context->state,
        context->scratch, output);
}

static inline int dsv4_block_moe_module(void *opaque, const float *input,
                                        float *output) {
    dsv4_block_moe_context *context = (dsv4_block_moe_context *)opaque;
    dsv4_block_scratch *scratch = context->scratch;
    return dsv4_moe_forward(
        context->cache, context->dense, context->layer, context->token,
        input, output, scratch->router_logits, scratch->routed,
        scratch->shared, scratch->expert_output, scratch->moe_gate,
        scratch->moe_up, scratch->moe_activation,
        scratch->moe_activation_scale);
}

static inline int dsv4_block_bind_stage(
    const dsv4_dense_arena *dense, int layer, const char *operation,
    const float **function_weight, const float **scale, const float **base,
    const uint16_t **norm_weight) {
    char prefix[64], suffix[64];
    if (!dsv4_dense_layer_prefix(prefix, sizeof(prefix), layer)) return 0;
    snprintf(suffix, sizeof(suffix), ".hc_%s_fn", operation);
    *function_weight = (const float *)dsv4_dense_named(
        dense, prefix, suffix, DSV4_DTYPE_F32, DSV4_HC_MIX,
        DSV4_HC_MULT * DSV4_ATTN_HIDDEN);
    snprintf(suffix, sizeof(suffix), ".hc_%s_scale", operation);
    *scale = (const float *)dsv4_attention_vector(
        dense, prefix, suffix, DSV4_DTYPE_F32, 3);
    snprintf(suffix, sizeof(suffix), ".hc_%s_base", operation);
    *base = (const float *)dsv4_attention_vector(
        dense, prefix, suffix, DSV4_DTYPE_F32, DSV4_HC_MIX);
    snprintf(suffix, sizeof(suffix), ".%s_norm.weight", operation);
    *norm_weight = (const uint16_t *)dsv4_attention_vector(
        dense, prefix, suffix, DSV4_DTYPE_BF16, DSV4_ATTN_HIDDEN);
    return *function_weight && *scale && *base && *norm_weight;
}

static inline int dsv4_block_decode_common(
    const dsv4_dense_arena *dense, dsv4_expert_cache *cache, int layer,
    int token, float *hc_state, dsv4_hc_module_fn attention_module,
    void *attention_context, dsv4_block_scratch *scratch) {
    if (!dense || !cache || !hc_state || !attention_module ||
        !attention_context || !scratch || layer < 0 ||
        DSV4_ATTN_HIDDEN != DSV4_EXPERT_HIDDEN)
        return 0;
    const float *function_weight, *scale, *base;
    const uint16_t *norm_weight;
    if (!dsv4_block_bind_stage(
            dense, layer, "attn", &function_weight, &scale, &base,
            &norm_weight))
        return 0;
    double stage_started = dsv4_dense_now_seconds();
    if (!dsv4_hc_stage(
            hc_state, DSV4_ATTN_HIDDEN, function_weight, scale, base,
            norm_weight, 1e-6f, attention_module, attention_context,
            scratch->reduced, scratch->post, scratch->combination,
            scratch->mixes, scratch->module_output, scratch->expanded))
        return 0;
    scratch->last_attention_stage_seconds =
        dsv4_dense_now_seconds() - stage_started;
    if (!dsv4_block_bind_stage(
            dense, layer, "ffn", &function_weight, &scale, &base,
            &norm_weight))
        return 0;
    dsv4_block_moe_context moe = {cache, dense, layer, token, scratch};
    stage_started = dsv4_dense_now_seconds();
    int ok = dsv4_hc_stage(
        hc_state, DSV4_ATTN_HIDDEN, function_weight, scale, base,
        norm_weight, 1e-6f, dsv4_block_moe_module, &moe,
        scratch->reduced, scratch->post, scratch->combination,
        scratch->mixes, scratch->module_output, scratch->expanded);
    scratch->last_ffn_stage_seconds =
        dsv4_dense_now_seconds() - stage_started;
    return ok;
}

/* Complete block entry points for the pinned layer schedule: layers 0-1 use
 * pure sliding attention, even layers >=2 use ratio-4 learned selection, and
 * odd layers >=3 use exhaustive ratio-128 compressed history. */
static inline int dsv4_block_decode_sliding(
    const dsv4_dense_arena *dense, dsv4_expert_cache *cache, int layer,
    int token, float *hc_state, dsv4_sliding_attention_state *attention_state,
    dsv4_block_scratch *scratch) {
    if (!attention_state || !scratch || layer < 0 || layer > 1) return 0;
    dsv4_block_attention_context attention = {
        dense, layer, attention_state, &scratch->attention,
    };
    return dsv4_block_decode_common(
        dense, cache, layer, token, hc_state, dsv4_block_attention_module,
        &attention, scratch);
}

static inline int dsv4_block_decode_compressed_nonoverlap(
    const dsv4_dense_arena *dense, dsv4_expert_cache *cache, int layer,
    int token, float *hc_state,
    dsv4_compressed_attention_state *attention_state,
    dsv4_compressed_attention_scratch *attention_scratch,
    dsv4_block_scratch *scratch) {
    if (!attention_state || !attention_scratch || !scratch || layer < 3 ||
        !(layer & 1) || layer > 41)
        return 0;
    attention_scratch->attention = scratch->attention;
    dsv4_block_nonoverlap_attention_context attention = {
        dense, layer, attention_state, attention_scratch,
    };
    return dsv4_block_decode_common(
        dense, cache, layer, token, hc_state,
        dsv4_block_nonoverlap_attention_module, &attention, scratch);
}

static inline int dsv4_block_decode_compressed_overlap(
    const dsv4_dense_arena *dense, dsv4_expert_cache *cache, int layer,
    int token, float *hc_state,
    dsv4_indexed_attention_state *attention_state,
    dsv4_indexed_attention_scratch *attention_scratch,
    dsv4_block_scratch *scratch) {
    if (!attention_state || !attention_scratch || !scratch || layer < 2 ||
        (layer & 1) || layer > 42)
        return 0;
    attention_scratch->attention = scratch->attention;
    dsv4_block_overlap_attention_context attention = {
        dense, layer, attention_state, attention_scratch,
    };
    return dsv4_block_decode_common(
        dense, cache, layer, token, hc_state,
        dsv4_block_overlap_attention_module, &attention, scratch);
}

#endif
