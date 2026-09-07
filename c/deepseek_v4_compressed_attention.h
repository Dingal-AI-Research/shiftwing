#ifndef COLIB_DEEPSEEK_V4_COMPRESSED_ATTENTION_H
#define COLIB_DEEPSEEK_V4_COMPRESSED_ATTENTION_H

#include "deepseek_v4_attention.h"

#ifndef DSV4_COMPRESS_RATIO
#define DSV4_COMPRESS_RATIO 128
#endif

typedef struct {
    int position;
    int max_compressed;
    float *kv_cache; /* [window + max_compressed, head_dim] */
    dsv4_compressor_state compressor;
} dsv4_compressed_attention_state;

typedef struct {
    dsv4_attention_scratch attention;
    float *compressor_kv;
    float *compressor_score;
} dsv4_compressed_attention_scratch;

static inline int dsv4_compressed_attention_state_init(
    dsv4_compressed_attention_state *state, int max_context,
    float *kv_cache, float *compressor_kv_state,
    float *compressor_score_state) {
    if (!state || max_context < 1 || !kv_cache ||
        !compressor_kv_state || !compressor_score_state)
        return 0;
    state->position = 0;
    state->max_compressed = (max_context + DSV4_COMPRESS_RATIO - 1) / DSV4_COMPRESS_RATIO;
    state->kv_cache = kv_cache;
    memset(kv_cache, 0,
           (size_t)(DSV4_ATTN_WINDOW + state->max_compressed) *
               DSV4_ATTN_HEAD_DIM * sizeof(*kv_cache));
    return dsv4_compressor_state_init(
        &state->compressor, DSV4_COMPRESS_RATIO, DSV4_ATTN_HEAD_DIM, 0,
        compressor_kv_state, compressor_score_state);
}

/* Non-overlapping compressor mode used by every 128x layer. This deliberately
 * mirrors the sliding reference instead of hiding the extra state transition:
 * the real oracle can attribute q/kv, compression, selection, and output
 * projection independently. */
static inline int dsv4_attention_decode_compressed_nonoverlap(
    const dsv4_dense_arena *dense, int layer, const float *input,
    dsv4_compressed_attention_state *state,
    dsv4_compressed_attention_scratch *scratch, float *output) {
    dsv4_attention_scratch *work = scratch ? &scratch->attention : NULL;
    if (!dense || !input || !state || !state->kv_cache || !scratch ||
        !scratch->compressor_kv || !scratch->compressor_score || !work ||
        !work->q_rank || !work->query || !work->kv || !work->context ||
        !work->o_rank || !work->indices || !work->activation ||
        !work->activation_scale || !output || layer < 2)
        return 0;
    char layer_prefix[64], attention[96], projection[128];
    if (!dsv4_dense_layer_prefix(layer_prefix, sizeof(layer_prefix), layer) ||
        snprintf(attention, sizeof(attention), "%s.attn", layer_prefix) >=
            (int)sizeof(attention))
        return 0;
    const uint8_t *weight, *scale;
    snprintf(projection, sizeof(projection), "%s.wq_a", attention);
    if (!dsv4_dense_fp8_pair(dense, projection, DSV4_ATTN_Q_RANK,
                              DSV4_ATTN_HIDDEN, &weight, &scale) ||
        !dsv4_dense_linear_fp8(dense, work->q_rank, input, weight, scale, 1,
                          DSV4_ATTN_Q_RANK, DSV4_ATTN_HIDDEN,
                          work->activation, work->activation_scale))
        return 0;
    const uint16_t *q_norm = (const uint16_t *)dsv4_attention_vector(
        dense, attention, ".q_norm.weight", DSV4_DTYPE_BF16,
        DSV4_ATTN_Q_RANK);
    if (!q_norm) return 0;
    dsv4_rmsnorm(work->q_rank, work->q_rank, q_norm, DSV4_ATTN_Q_RANK,
                  1e-6f);
    snprintf(projection, sizeof(projection), "%s.wq_b", attention);
    if (!dsv4_dense_fp8_pair(
            dense, projection, DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM,
            DSV4_ATTN_Q_RANK, &weight, &scale) ||
        !dsv4_dense_linear_fp8(dense,
            work->query, work->q_rank, weight, scale, 1,
            DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM, DSV4_ATTN_Q_RANK,
            work->activation, work->activation_scale))
        return 0;
    for (int head = 0; head < DSV4_ATTN_HEADS; head++) {
        float *query = work->query + (size_t)head * DSV4_ATTN_HEAD_DIM;
        float square_sum = 0.0f;
        for (int axis = 0; axis < DSV4_ATTN_HEAD_DIM; axis++)
            square_sum += query[axis] * query[axis];
        float inverse = 1.0f / sqrtf(
            square_sum / DSV4_ATTN_HEAD_DIM + 1e-6f);
        for (int axis = 0; axis < DSV4_ATTN_HEAD_DIM; axis++)
            query[axis] *= inverse;
        dsv4_round_bf16_array(query, DSV4_ATTN_HEAD_DIM);
        dsv4_rope(query + DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM,
                   DSV4_ATTN_ROPE_DIM, state->position, DSV4_ORIGINAL_CONTEXT,
                   DSV4_COMPRESS_ROPE_THETA, DSV4_ROPE_FACTOR, 32, 1, 0);
        dsv4_round_bf16_array(query, DSV4_ATTN_HEAD_DIM);
    }
    snprintf(projection, sizeof(projection), "%s.wkv", attention);
    if (!dsv4_dense_fp8_pair(dense, projection, DSV4_ATTN_HEAD_DIM,
                              DSV4_ATTN_HIDDEN, &weight, &scale) ||
        !dsv4_dense_linear_fp8(dense, work->kv, input, weight, scale, 1,
                          DSV4_ATTN_HEAD_DIM, DSV4_ATTN_HIDDEN,
                          work->activation, work->activation_scale))
        return 0;
    const uint16_t *kv_norm = (const uint16_t *)dsv4_attention_vector(
        dense, attention, ".kv_norm.weight", DSV4_DTYPE_BF16,
        DSV4_ATTN_HEAD_DIM);
    if (!kv_norm) return 0;
    dsv4_rmsnorm(work->kv, work->kv, kv_norm, DSV4_ATTN_HEAD_DIM, 1e-6f);
    dsv4_rope(work->kv + DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM,
               DSV4_ATTN_ROPE_DIM, state->position, DSV4_ORIGINAL_CONTEXT,
               DSV4_COMPRESS_ROPE_THETA, DSV4_ROPE_FACTOR, 32, 1, 0);
    dsv4_round_bf16_array(work->kv, DSV4_ATTN_HEAD_DIM);
    int non_rope = DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM;
    if (non_rope && !dsv4_fp8_simulate(work->kv, non_rope, 64)) return 0;
    memcpy(state->kv_cache +
               (size_t)(state->position % DSV4_ATTN_WINDOW) *
                   DSV4_ATTN_HEAD_DIM,
           work->kv, DSV4_ATTN_HEAD_DIM * sizeof(*work->kv));

    char compressor[128], name[160];
    snprintf(compressor, sizeof(compressor), "%s.compressor", attention);
    snprintf(name, sizeof(name), "%s.wkv.weight", compressor);
    const dsv4_tensor_desc *descriptor = NULL;
    const uint16_t *compressor_wkv =
        (const uint16_t *)dsv4_dense_find(dense, name, &descriptor);
    if (!compressor_wkv ||
        !dsv4_model_tensor_shape(descriptor, DSV4_DTYPE_BF16, 2,
                                 DSV4_ATTN_HEAD_DIM, DSV4_ATTN_HIDDEN))
        return 0;
    snprintf(name, sizeof(name), "%s.wgate.weight", compressor);
    const uint16_t *compressor_wgate =
        (const uint16_t *)dsv4_dense_find(dense, name, &descriptor);
    if (!compressor_wgate ||
        !dsv4_model_tensor_shape(descriptor, DSV4_DTYPE_BF16, 2,
                                 DSV4_ATTN_HEAD_DIM, DSV4_ATTN_HIDDEN))
        return 0;
    dsv4_bf16_gemm(scratch->compressor_kv, input, compressor_wkv, 1,
                    DSV4_ATTN_HEAD_DIM, DSV4_ATTN_HIDDEN);
    dsv4_bf16_gemm(scratch->compressor_score, input, compressor_wgate, 1,
                    DSV4_ATTN_HEAD_DIM, DSV4_ATTN_HIDDEN);
    const float *ape = (const float *)dsv4_dense_named(
        dense, compressor, ".ape", DSV4_DTYPE_F32,
        DSV4_COMPRESS_RATIO, DSV4_ATTN_HEAD_DIM);
    const uint16_t *compressor_norm =
        (const uint16_t *)dsv4_attention_vector(
            dense, compressor, ".norm.weight", DSV4_DTYPE_BF16,
            DSV4_ATTN_HEAD_DIM);
    if (!ape || !compressor_norm) return 0;
    float *compressed = state->kv_cache +
        (size_t)(DSV4_ATTN_WINDOW +
                 state->position / DSV4_COMPRESS_RATIO) *
            DSV4_ATTN_HEAD_DIM;
    int emitted = dsv4_compressor_step(
        &state->compressor, state->position, scratch->compressor_kv,
        scratch->compressor_score, ape, compressed);
    if (emitted < 0) return 0;
    if (emitted) {
        int compressed_index = state->position / DSV4_COMPRESS_RATIO;
        if (compressed_index >= state->max_compressed) return 0;
        dsv4_round_bf16_array(compressed, DSV4_ATTN_HEAD_DIM);
        dsv4_rmsnorm(compressed, compressed, compressor_norm,
                      DSV4_ATTN_HEAD_DIM, 1e-6f);
        dsv4_rope(compressed + DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM,
                   DSV4_ATTN_ROPE_DIM,
                   state->position + 1 - DSV4_COMPRESS_RATIO,
                   DSV4_ORIGINAL_CONTEXT, DSV4_COMPRESS_ROPE_THETA,
                   DSV4_ROPE_FACTOR, 32, 1, 0);
        dsv4_round_bf16_array(compressed, DSV4_ATTN_HEAD_DIM);
        if (non_rope && !dsv4_fp8_simulate(compressed, non_rope, 64))
            return 0;
    }
    int window_count = dsv4_window_indices(
        DSV4_ATTN_WINDOW, 1, state->position, 0, work->indices);
    int compressed_count = dsv4_compress_indices(
        DSV4_COMPRESS_RATIO, 1, state->position, DSV4_ATTN_WINDOW, 0,
        work->indices + window_count);
    int selected = window_count + compressed_count;
    const float *sink = (const float *)dsv4_attention_vector(
        dense, attention, ".attn_sink", DSV4_DTYPE_F32,
        DSV4_ATTN_HEADS);
    if (!sink || selected < 1) return 0;
    dsv4_sparse_attention(
        work->context, work->query, state->kv_cache, DSV4_ATTN_HEADS,
        DSV4_ATTN_HEAD_DIM, work->indices, selected, sink,
        1.0f / sqrtf((float)DSV4_ATTN_HEAD_DIM));
    for (int head = 0; head < DSV4_ATTN_HEADS; head++)
        dsv4_rope(work->context +
                       (size_t)head * DSV4_ATTN_HEAD_DIM +
                       DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM,
                   DSV4_ATTN_ROPE_DIM, state->position, DSV4_ORIGINAL_CONTEXT,
                   DSV4_COMPRESS_ROPE_THETA, DSV4_ROPE_FACTOR, 32, 1, 1);
    dsv4_round_bf16_array(
        work->context, (size_t)DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM);
    int group_width = DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM /
                      DSV4_ATTN_O_GROUPS;
    snprintf(projection, sizeof(projection), "%s.wo_a", attention);
    const uint8_t *wo_a_weight, *wo_a_scale;
    if (!dsv4_dense_fp8_pair(
            dense, projection, DSV4_ATTN_O_GROUPS * DSV4_ATTN_O_RANK,
            group_width, &wo_a_weight, &wo_a_scale))
        return 0;
    size_t group_weight = (size_t)DSV4_ATTN_O_RANK * group_width;
    size_t group_scale =
        (size_t)((DSV4_ATTN_O_RANK + 127) / 128) *
        ((group_width + 127) / 128);
    for (int group = 0; group < DSV4_ATTN_O_GROUPS; group++)
        if (!dsv4_dense_linear_fp8(dense,
                work->o_rank + (size_t)group * DSV4_ATTN_O_RANK,
                work->context + (size_t)group * group_width,
                wo_a_weight + (size_t)group * group_weight,
                wo_a_scale + (size_t)group * group_scale, 1,
                DSV4_ATTN_O_RANK, group_width, work->activation,
                work->activation_scale))
            return 0;
    snprintf(projection, sizeof(projection), "%s.wo_b", attention);
    if (!dsv4_dense_fp8_pair(
            dense, projection, DSV4_ATTN_HIDDEN,
            DSV4_ATTN_O_GROUPS * DSV4_ATTN_O_RANK, &weight, &scale) ||
        !dsv4_dense_linear_fp8(dense,
            output, work->o_rank, weight, scale, 1, DSV4_ATTN_HIDDEN,
            DSV4_ATTN_O_GROUPS * DSV4_ATTN_O_RANK, work->activation,
            work->activation_scale))
        return 0;
    state->position++;
    return 1;
}

#endif
