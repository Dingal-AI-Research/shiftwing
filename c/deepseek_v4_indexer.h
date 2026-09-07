#ifndef COLIB_DEEPSEEK_V4_INDEXER_H
#define COLIB_DEEPSEEK_V4_INDEXER_H

#include "deepseek_v4_attention.h"

#ifndef DSV4_INDEX_HEADS
#define DSV4_INDEX_HEADS 64
#endif
#ifndef DSV4_INDEX_DIM
#define DSV4_INDEX_DIM 128
#endif
#ifndef DSV4_INDEX_TOPK
#define DSV4_INDEX_TOPK 512
#endif
#ifndef DSV4_INDEX_RATIO
#define DSV4_INDEX_RATIO 4
#endif

typedef struct {
    int position;
    int max_compressed;
    float *kv_cache;
    dsv4_compressor_state compressor;
} dsv4_indexer_state;

typedef struct {
    float *query;
    float *compressor_kv;
    float *compressor_score;
    float *head_weights;
    float *scores;
    uint8_t *activation;
    uint8_t *activation_scale;
} dsv4_indexer_scratch;

static inline int dsv4_indexer_state_init(
    dsv4_indexer_state *state, int max_context, float *kv_cache,
    float *compressor_kv_state, float *compressor_score_state) {
    if (!state || max_context < 1 || !kv_cache ||
        !compressor_kv_state || !compressor_score_state)
        return 0;
    state->position = 0;
    state->max_compressed = (max_context + DSV4_INDEX_RATIO - 1) / DSV4_INDEX_RATIO;
    state->kv_cache = kv_cache;
    memset(kv_cache, 0, (size_t)state->max_compressed * DSV4_INDEX_DIM *
                         sizeof(*kv_cache));
    return dsv4_compressor_state_init(
        &state->compressor, DSV4_INDEX_RATIO, DSV4_INDEX_DIM, 1,
        compressor_kv_state, compressor_score_state);
}

/* Returns selected count, zero before a compressed position is visible, and
 * -1 on invalid state/weights. Output indices already include `offset`. */
static inline int dsv4_indexer_decode(
    const dsv4_dense_arena *dense, int layer, const float *input,
    const float *q_rank, dsv4_indexer_state *state,
    dsv4_indexer_scratch *scratch, int offset, int *indices) {
    if (!dense || !input || !q_rank || !state || !state->kv_cache ||
        !scratch || !scratch->query || !scratch->compressor_kv ||
        !scratch->compressor_score || !scratch->head_weights ||
        !scratch->scores || !scratch->activation ||
        !scratch->activation_scale || !indices || offset < 0 || layer < 2 ||
        state->position != state->compressor.next_position)
        return -1;
    char layer_prefix[64], indexer[128], projection[160], name[192];
    if (!dsv4_dense_layer_prefix(layer_prefix, sizeof(layer_prefix), layer) ||
        snprintf(indexer, sizeof(indexer), "%s.attn.indexer", layer_prefix) >=
            (int)sizeof(indexer))
        return -1;
    const uint8_t *weight, *scale;
    snprintf(projection, sizeof(projection), "%s.wq_b", indexer);
    if (!dsv4_dense_fp8_pair(
            dense, projection, DSV4_INDEX_HEADS * DSV4_INDEX_DIM,
            DSV4_ATTN_Q_RANK, &weight, &scale) ||
        !dsv4_dense_linear_fp8(dense,
            scratch->query, q_rank, weight, scale, 1,
            DSV4_INDEX_HEADS * DSV4_INDEX_DIM, DSV4_ATTN_Q_RANK,
            scratch->activation, scratch->activation_scale))
        return -1;
    for (int head = 0; head < DSV4_INDEX_HEADS; head++) {
        float *query = scratch->query + (size_t)head * DSV4_INDEX_DIM;
        dsv4_rope(query + DSV4_INDEX_DIM - DSV4_ATTN_ROPE_DIM,
                   DSV4_ATTN_ROPE_DIM, state->position, DSV4_ORIGINAL_CONTEXT,
                   DSV4_COMPRESS_ROPE_THETA, DSV4_ROPE_FACTOR, 32, 1, 0);
        dsv4_round_bf16_array(query, DSV4_INDEX_DIM);
        if (!dsv4_hadamard(query, DSV4_INDEX_DIM) ||
            !dsv4_fp4_simulate(query, DSV4_INDEX_DIM, 32))
            return -1;
    }
    snprintf(projection, sizeof(projection), "%s.compressor", indexer);
    snprintf(name, sizeof(name), "%s.wkv.weight", projection);
    const dsv4_tensor_desc *descriptor = NULL;
    const uint16_t *compressor_wkv =
        (const uint16_t *)dsv4_dense_find(dense, name, &descriptor);
    if (!compressor_wkv ||
        !dsv4_model_tensor_shape(descriptor, DSV4_DTYPE_BF16, 2,
                                 2 * DSV4_INDEX_DIM, DSV4_ATTN_HIDDEN))
        return -1;
    snprintf(name, sizeof(name), "%s.wgate.weight", projection);
    const uint16_t *compressor_wgate =
        (const uint16_t *)dsv4_dense_find(dense, name, &descriptor);
    if (!compressor_wgate ||
        !dsv4_model_tensor_shape(descriptor, DSV4_DTYPE_BF16, 2,
                                 2 * DSV4_INDEX_DIM, DSV4_ATTN_HIDDEN))
        return -1;
    dsv4_bf16_gemm(scratch->compressor_kv, input, compressor_wkv, 1,
                    2 * DSV4_INDEX_DIM, DSV4_ATTN_HIDDEN);
    dsv4_bf16_gemm(scratch->compressor_score, input, compressor_wgate, 1,
                    2 * DSV4_INDEX_DIM, DSV4_ATTN_HIDDEN);
    const float *ape = (const float *)dsv4_dense_named(
        dense, projection, ".ape", DSV4_DTYPE_F32, DSV4_INDEX_RATIO,
        2 * DSV4_INDEX_DIM);
    const uint16_t *compressor_norm =
        (const uint16_t *)dsv4_attention_vector(
            dense, projection, ".norm.weight", DSV4_DTYPE_BF16,
            DSV4_INDEX_DIM);
    if (!ape || !compressor_norm) return -1;
    int compressed_index = state->position / DSV4_INDEX_RATIO;
    if (compressed_index >= state->max_compressed) return -1;
    float *compressed = state->kv_cache +
        (size_t)compressed_index * DSV4_INDEX_DIM;
    int emitted = dsv4_compressor_step(
        &state->compressor, state->position, scratch->compressor_kv,
        scratch->compressor_score, ape, compressed);
    if (emitted < 0) return -1;
    if (emitted) {
        dsv4_round_bf16_array(compressed, DSV4_INDEX_DIM);
        dsv4_rmsnorm(compressed, compressed, compressor_norm,
                      DSV4_INDEX_DIM, 1e-6f);
        dsv4_rope(compressed + DSV4_INDEX_DIM - DSV4_ATTN_ROPE_DIM,
                   DSV4_ATTN_ROPE_DIM,
                   state->position + 1 - DSV4_INDEX_RATIO,
                   DSV4_ORIGINAL_CONTEXT, DSV4_COMPRESS_ROPE_THETA,
                   DSV4_ROPE_FACTOR, 32, 1, 0);
        dsv4_round_bf16_array(compressed, DSV4_INDEX_DIM);
        if (!dsv4_hadamard(compressed, DSV4_INDEX_DIM) ||
            !dsv4_fp4_simulate(compressed, DSV4_INDEX_DIM, 32))
            return -1;
    }
    snprintf(name, sizeof(name), "%s.weights_proj.weight", indexer);
    const uint16_t *weights_projection =
        (const uint16_t *)dsv4_dense_find(dense, name, &descriptor);
    if (!weights_projection ||
        !dsv4_model_tensor_shape(descriptor, DSV4_DTYPE_BF16, 2,
                                 DSV4_INDEX_HEADS, DSV4_ATTN_HIDDEN))
        return -1;
    dsv4_bf16_gemm(scratch->head_weights, input, weights_projection, 1,
                    DSV4_INDEX_HEADS, DSV4_ATTN_HIDDEN);
    dsv4_round_bf16_array(scratch->head_weights, DSV4_INDEX_HEADS);
    float head_scale = 1.0f /
        sqrtf((float)DSV4_INDEX_DIM * DSV4_INDEX_HEADS);
    for (int head = 0; head < DSV4_INDEX_HEADS; head++)
        scratch->head_weights[head] *= head_scale;
    int visible = (state->position + 1) / DSV4_INDEX_RATIO;
    int topk = visible < DSV4_INDEX_TOPK ? visible : DSV4_INDEX_TOPK;
    state->position++;
    if (!topk) return 0;
    return dsv4_indexer_topk(
        scratch->query, state->kv_cache, scratch->head_weights,
        DSV4_INDEX_HEADS, DSV4_INDEX_DIM, visible, topk, offset,
        scratch->scores, indices);
}

#endif
