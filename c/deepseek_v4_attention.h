#ifndef COLIB_DEEPSEEK_V4_ATTENTION_H
#define COLIB_DEEPSEEK_V4_ATTENTION_H

/* Scalar decode reference for the pure sliding-window MLA mode used by base
 * layers 0 and 1. Compressed layers reuse these exact q/kv/output projections
 * and add the separately tested compressor/indexer selections. */

#include "deepseek_v4_dense.h"

#ifndef DSV4_ATTN_HIDDEN
#define DSV4_ATTN_HIDDEN DSV4_DIM
#endif
#ifndef DSV4_ATTN_HEADS
#define DSV4_ATTN_HEADS 64
#endif
#ifndef DSV4_ATTN_HEAD_DIM
#define DSV4_ATTN_HEAD_DIM 512
#endif
#ifndef DSV4_ATTN_ROPE_DIM
#define DSV4_ATTN_ROPE_DIM 64
#endif
#ifndef DSV4_ATTN_Q_RANK
#define DSV4_ATTN_Q_RANK 1024
#endif
#ifndef DSV4_ATTN_O_GROUPS
#define DSV4_ATTN_O_GROUPS 8
#endif
#ifndef DSV4_ATTN_O_RANK
#define DSV4_ATTN_O_RANK 1024
#endif
#ifndef DSV4_ATTN_WINDOW
#define DSV4_ATTN_WINDOW 128
#endif
#ifndef DSV4_COMPRESS_ROPE_THETA
#define DSV4_COMPRESS_ROPE_THETA 160000.0f
#endif
#ifndef DSV4_ORIGINAL_CONTEXT
#define DSV4_ORIGINAL_CONTEXT 65536
#endif
#ifndef DSV4_ROPE_FACTOR
#define DSV4_ROPE_FACTOR 16.0f
#endif

typedef struct {
    int position;
    float *kv_cache;
} dsv4_sliding_attention_state;

typedef struct {
    float *q_rank;
    float *query;
    float *kv;
    float *context;
    float *o_rank;
    float *scores;
    int *indices;
    uint8_t *activation;
    uint8_t *activation_scale;
} dsv4_attention_scratch;

static inline int dsv4_sliding_attention_state_init(
    dsv4_sliding_attention_state *state, float *kv_cache) {
    if (!state || !kv_cache) return 0;
    state->position = 0;
    state->kv_cache = kv_cache;
    memset(kv_cache, 0, (size_t)DSV4_ATTN_WINDOW * DSV4_ATTN_HEAD_DIM *
                         sizeof(*kv_cache));
    return 1;
}

static inline const void *dsv4_attention_vector(
    const dsv4_dense_arena *dense, const char *prefix, const char *suffix,
    dsv4_dtype dtype, int64_t dimension) {
    char name[160];
    int length = snprintf(name, sizeof(name), "%s%s", prefix, suffix);
    if (length < 0 || length >= (int)sizeof(name)) return NULL;
    const dsv4_tensor_desc *descriptor = NULL;
    const void *value = dsv4_dense_find(dense, name, &descriptor);
    return value && descriptor && descriptor->dtype == dtype &&
           descriptor->rank == 1 && descriptor->shape[0] == dimension
        ? value : NULL;
}

static inline int dsv4_attention_decode_sliding(
    const dsv4_dense_arena *dense, int layer, const float *input,
    dsv4_sliding_attention_state *state, dsv4_attention_scratch *scratch,
    float *output) {
    if (!dense || !input || !state || !state->kv_cache || !scratch ||
        !scratch->q_rank || !scratch->query || !scratch->kv ||
        !scratch->context || !scratch->o_rank || !scratch->scores ||
        !scratch->indices || !scratch->activation ||
        !scratch->activation_scale || !output || layer < 0 || layer > 1)
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
        !dsv4_dense_linear_fp8(dense, scratch->q_rank, input, weight, scale, 1,
                          DSV4_ATTN_Q_RANK, DSV4_ATTN_HIDDEN,
                          scratch->activation, scratch->activation_scale))
        return 0;
    const uint16_t *q_norm = (const uint16_t *)dsv4_attention_vector(
        dense, attention, ".q_norm.weight", DSV4_DTYPE_BF16,
        DSV4_ATTN_Q_RANK);
    if (!q_norm) return 0;
    dsv4_rmsnorm(scratch->q_rank, scratch->q_rank, q_norm,
                  DSV4_ATTN_Q_RANK, 1e-6f);
    snprintf(projection, sizeof(projection), "%s.wq_b", attention);
    if (!dsv4_dense_fp8_pair(
            dense, projection, DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM,
            DSV4_ATTN_Q_RANK, &weight, &scale) ||
        !dsv4_dense_linear_fp8(dense,
            scratch->query, scratch->q_rank, weight, scale, 1,
            DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM, DSV4_ATTN_Q_RANK,
            scratch->activation, scratch->activation_scale))
        return 0;
    for (int head = 0; head < DSV4_ATTN_HEADS; head++) {
        float square_sum = 0.0f;
        float *query = scratch->query + (size_t)head * DSV4_ATTN_HEAD_DIM;
        for (int axis = 0; axis < DSV4_ATTN_HEAD_DIM; axis++)
            square_sum += query[axis] * query[axis];
        float inverse = 1.0f / sqrtf(
            square_sum / DSV4_ATTN_HEAD_DIM + 1e-6f);
        for (int axis = 0; axis < DSV4_ATTN_HEAD_DIM; axis++)
            query[axis] *= inverse;
        dsv4_round_bf16_array(query, DSV4_ATTN_HEAD_DIM);
        dsv4_rope(query + DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM,
                   DSV4_ATTN_ROPE_DIM, state->position, 0, 10000.0f,
                   1.0f, 0, 0, 0);
        dsv4_round_bf16_array(query, DSV4_ATTN_HEAD_DIM);
    }
    snprintf(projection, sizeof(projection), "%s.wkv", attention);
    if (!dsv4_dense_fp8_pair(dense, projection, DSV4_ATTN_HEAD_DIM,
                              DSV4_ATTN_HIDDEN, &weight, &scale) ||
        !dsv4_dense_linear_fp8(dense, scratch->kv, input, weight, scale, 1,
                          DSV4_ATTN_HEAD_DIM, DSV4_ATTN_HIDDEN,
                          scratch->activation, scratch->activation_scale))
        return 0;
    const uint16_t *kv_norm = (const uint16_t *)dsv4_attention_vector(
        dense, attention, ".kv_norm.weight", DSV4_DTYPE_BF16,
        DSV4_ATTN_HEAD_DIM);
    if (!kv_norm) return 0;
    dsv4_rmsnorm(scratch->kv, scratch->kv, kv_norm,
                  DSV4_ATTN_HEAD_DIM, 1e-6f);
    dsv4_rope(scratch->kv + DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM,
               DSV4_ATTN_ROPE_DIM, state->position, 0, 10000.0f,
               1.0f, 0, 0, 0);
    dsv4_round_bf16_array(scratch->kv, DSV4_ATTN_HEAD_DIM);
    int non_rope = DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM;
    if (non_rope && !dsv4_fp8_simulate(scratch->kv, non_rope, 64)) return 0;
    memcpy(state->kv_cache +
               (size_t)(state->position % DSV4_ATTN_WINDOW) *
                   DSV4_ATTN_HEAD_DIM,
           scratch->kv, DSV4_ATTN_HEAD_DIM * sizeof(*scratch->kv));
    int selected = dsv4_window_indices(
        DSV4_ATTN_WINDOW, 1, state->position, 0, scratch->indices);
    const float *sink = (const float *)dsv4_attention_vector(
        dense, attention, ".attn_sink", DSV4_DTYPE_F32,
        DSV4_ATTN_HEADS);
    if (!sink || selected < 1) return 0;
    dsv4_sparse_attention(
        scratch->context, scratch->query, state->kv_cache,
        DSV4_ATTN_HEADS, DSV4_ATTN_HEAD_DIM, scratch->indices, selected,
        sink, 1.0f / sqrtf((float)DSV4_ATTN_HEAD_DIM));
    for (int head = 0; head < DSV4_ATTN_HEADS; head++)
        dsv4_rope(scratch->context +
                       (size_t)head * DSV4_ATTN_HEAD_DIM +
                       DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM,
                   DSV4_ATTN_ROPE_DIM, state->position, 0, 10000.0f,
                   1.0f, 0, 0, 1);
    dsv4_round_bf16_array(
        scratch->context, (size_t)DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM);
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
                scratch->o_rank + (size_t)group * DSV4_ATTN_O_RANK,
                scratch->context + (size_t)group * group_width,
                wo_a_weight + (size_t)group * group_weight,
                wo_a_scale + (size_t)group * group_scale, 1,
                DSV4_ATTN_O_RANK, group_width, scratch->activation,
                scratch->activation_scale))
            return 0;
    snprintf(projection, sizeof(projection), "%s.wo_b", attention);
    if (!dsv4_dense_fp8_pair(
            dense, projection, DSV4_ATTN_HIDDEN,
            DSV4_ATTN_O_GROUPS * DSV4_ATTN_O_RANK, &weight, &scale) ||
        !dsv4_dense_linear_fp8(dense,
            output, scratch->o_rank, weight, scale, 1,
            DSV4_ATTN_HIDDEN, DSV4_ATTN_O_GROUPS * DSV4_ATTN_O_RANK,
            scratch->activation, scratch->activation_scale))
        return 0;
    state->position++;
    return 1;
}

#endif
