#ifndef COLIB_DEEPSEEK_V4_INDEXED_ATTENTION_H
#define COLIB_DEEPSEEK_V4_INDEXED_ATTENTION_H

/* Complete incremental attention path for ratio-4 layers.  These layers use
 * the normal 128-token ring plus an overlapping learned compressor, and a
 * separate compressed indexer which selects at most 512 historical entries. */

#include "deepseek_v4_indexer.h"

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
    int max_compressed;
    float *kv_cache; /* [window + max_compressed, head_dim] */
    dsv4_compressor_state compressor;
    dsv4_indexer_state indexer;
} dsv4_indexed_attention_state;

typedef struct {
    dsv4_attention_scratch attention;
    float *compressor_kv;
    float *compressor_score;
    dsv4_indexer_scratch indexer;
} dsv4_indexed_attention_scratch;

static inline int dsv4_indexed_attention_state_init(
    dsv4_indexed_attention_state *state, int max_context, float *kv_cache,
    float *compressor_kv_state, float *compressor_score_state,
    float *index_kv_cache, float *index_compressor_kv_state,
    float *index_compressor_score_state) {
    if (!state || max_context < 1 || !kv_cache ||
        !compressor_kv_state || !compressor_score_state ||
        !index_kv_cache || !index_compressor_kv_state ||
        !index_compressor_score_state)
        return 0;
    state->position = 0;
    state->max_compressed = (max_context + DSV4_INDEX_RATIO - 1) / DSV4_INDEX_RATIO;
    state->kv_cache = kv_cache;
    memset(kv_cache, 0,
           (size_t)(DSV4_ATTN_WINDOW + state->max_compressed) *
               DSV4_ATTN_HEAD_DIM * sizeof(*kv_cache));
    return dsv4_compressor_state_init(
               &state->compressor, DSV4_INDEX_RATIO,
               DSV4_ATTN_HEAD_DIM, 1, compressor_kv_state,
               compressor_score_state) &&
           dsv4_indexer_state_init(
               &state->indexer, max_context, index_kv_cache,
               index_compressor_kv_state,
               index_compressor_score_state);
}

static inline int dsv4_attention_decode_compressed_overlap(
    const dsv4_dense_arena *dense, int layer, const float *input,
    dsv4_indexed_attention_state *state,
    dsv4_indexed_attention_scratch *scratch, float *output) {
    dsv4_attention_scratch *work = scratch ? &scratch->attention : NULL;
    if (!dense || !input || !state || !state->kv_cache || !scratch ||
        !scratch->compressor_kv || !scratch->compressor_score || !work ||
        !work->q_rank || !work->query || !work->kv || !work->context ||
        !work->o_rank || !work->indices || !work->activation ||
        !work->activation_scale || !output || layer < 2 ||
        state->position != state->compressor.next_position ||
        state->position != state->indexer.position)
        return 0;
    char layer_prefix[64], attention[96], projection[160], name[192];
    if (!dsv4_dense_layer_prefix(layer_prefix, sizeof(layer_prefix), layer) ||
        snprintf(attention, sizeof(attention), "%s.attn", layer_prefix) >=
            (int)sizeof(attention))
        return 0;
    const uint8_t *weight, *scale;
    snprintf(projection, sizeof(projection), "%s.wq_a", attention);
    if (!dsv4_dense_fp8_pair(dense, projection, DSV4_ATTN_Q_RANK,
                              DSV4_ATTN_HIDDEN, &weight, &scale) ||
        !dsv4_attention_project_fp8(dense, work->q_rank, input, weight, scale, 1,
                          DSV4_ATTN_Q_RANK, DSV4_ATTN_HIDDEN,
                          work->activation, work->activation_scale, work->prefill_q_rank))
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
        !dsv4_attention_project_fp8(dense, work->query, work->q_rank, weight, scale, 1,
            DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM, DSV4_ATTN_Q_RANK,
            work->activation, work->activation_scale, work->prefill_query))
        return 0;
    for (int head = 0; head < DSV4_ATTN_HEADS; head++) {
        float *query = work->query + (size_t)head * DSV4_ATTN_HEAD_DIM;
        dsv4_query_norm_bf16(query,DSV4_ATTN_HEAD_DIM,1e-6f);
        dsv4_rope(query + DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM,
                   DSV4_ATTN_ROPE_DIM, state->position,
                   DSV4_ORIGINAL_CONTEXT, DSV4_COMPRESS_ROPE_THETA,
                   DSV4_ROPE_FACTOR, 32, 1, 0);
        dsv4_round_bf16_array(query, DSV4_ATTN_HEAD_DIM);
    }

    int window_count = dsv4_window_indices(
        DSV4_ATTN_WINDOW, 1, state->position, 0, work->indices);
    int indexed_count = dsv4_indexer_decode(
        dense, layer, input, work->q_rank, &state->indexer,
        &scratch->indexer, DSV4_ATTN_WINDOW,
        work->indices + window_count);
    if (indexed_count < 0) return 0;

    snprintf(projection, sizeof(projection), "%s.wkv", attention);
    if (!dsv4_dense_fp8_pair(dense, projection, DSV4_ATTN_HEAD_DIM,
                              DSV4_ATTN_HIDDEN, &weight, &scale) ||
        !dsv4_attention_project_fp8(dense, work->kv, input, weight, scale, 1,
                          DSV4_ATTN_HEAD_DIM, DSV4_ATTN_HIDDEN,
                          work->activation, work->activation_scale, work->prefill_kv))
        return 0;
    const uint16_t *kv_norm = (const uint16_t *)dsv4_attention_vector(
        dense, attention, ".kv_norm.weight", DSV4_DTYPE_BF16,
        DSV4_ATTN_HEAD_DIM);
    if (!kv_norm) return 0;
    dsv4_rmsnorm(work->kv, work->kv, kv_norm, DSV4_ATTN_HEAD_DIM, 1e-6f);
    dsv4_rope(work->kv + DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM,
               DSV4_ATTN_ROPE_DIM, state->position,
               DSV4_ORIGINAL_CONTEXT, DSV4_COMPRESS_ROPE_THETA,
               DSV4_ROPE_FACTOR, 32, 1, 0);
    dsv4_round_bf16_array(work->kv, DSV4_ATTN_HEAD_DIM);
    int non_rope = DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM;
    if (non_rope && !dsv4_fp8_simulate(work->kv, non_rope, 64)) return 0;
    memcpy(state->kv_cache +
               (size_t)(state->position % DSV4_ATTN_WINDOW) *
                   DSV4_ATTN_HEAD_DIM,
           work->kv, DSV4_ATTN_HEAD_DIM * sizeof(*work->kv));

    char compressor[128];
    snprintf(compressor, sizeof(compressor), "%s.compressor", attention);
    const dsv4_tensor_desc *descriptor = NULL;
    snprintf(name, sizeof(name), "%s.wkv.weight", compressor);
    const uint16_t *compressor_wkv =
        (const uint16_t *)dsv4_dense_find(dense, name, &descriptor);
    if (!compressor_wkv ||
        !dsv4_model_tensor_shape(descriptor, DSV4_DTYPE_BF16, 2,
                                 2 * DSV4_ATTN_HEAD_DIM,
                                 DSV4_ATTN_HIDDEN))
        return 0;
    snprintf(name, sizeof(name), "%s.wgate.weight", compressor);
    const uint16_t *compressor_wgate =
        (const uint16_t *)dsv4_dense_find(dense, name, &descriptor);
    if (!compressor_wgate ||
        !dsv4_model_tensor_shape(descriptor, DSV4_DTYPE_BF16, 2,
                                 2 * DSV4_ATTN_HEAD_DIM,
                                 DSV4_ATTN_HIDDEN))
        return 0;
    dsv4_attention_project_bf16(scratch->compressor_kv, input, compressor_wkv, 1,
                    2 * DSV4_ATTN_HEAD_DIM, DSV4_ATTN_HIDDEN, work->prefill_compressor_kv);
    dsv4_attention_project_bf16(scratch->compressor_score, input, compressor_wgate, 1,
                    2 * DSV4_ATTN_HEAD_DIM, DSV4_ATTN_HIDDEN, work->prefill_compressor_score);
    const float *ape = (const float *)dsv4_dense_named(
        dense, compressor, ".ape", DSV4_DTYPE_F32, DSV4_INDEX_RATIO,
        2 * DSV4_ATTN_HEAD_DIM);
    const uint16_t *compressor_norm =
        (const uint16_t *)dsv4_attention_vector(
            dense, compressor, ".norm.weight", DSV4_DTYPE_BF16,
            DSV4_ATTN_HEAD_DIM);
    if (!ape || !compressor_norm) return 0;
    int compressed_index = state->position / DSV4_INDEX_RATIO;
    if (compressed_index >= state->max_compressed) return 0;
    float *compressed = state->kv_cache +
        (size_t)(DSV4_ATTN_WINDOW + compressed_index) *
            DSV4_ATTN_HEAD_DIM;
    int emitted = dsv4_compressor_step(
        &state->compressor, state->position, scratch->compressor_kv,
        scratch->compressor_score, ape, compressed);
    if (emitted < 0) return 0;
    if (emitted) {
        dsv4_round_bf16_array(compressed, DSV4_ATTN_HEAD_DIM);
        dsv4_rmsnorm(compressed, compressed, compressor_norm,
                      DSV4_ATTN_HEAD_DIM, 1e-6f);
        dsv4_rope(compressed + DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM,
                   DSV4_ATTN_ROPE_DIM,
                   state->position + 1 - DSV4_INDEX_RATIO,
                   DSV4_ORIGINAL_CONTEXT, DSV4_COMPRESS_ROPE_THETA,
                   DSV4_ROPE_FACTOR, 32, 1, 0);
        dsv4_round_bf16_array(compressed, DSV4_ATTN_HEAD_DIM);
        if (non_rope && !dsv4_fp8_simulate(compressed, non_rope, 64))
            return 0;
    }

    int selected = window_count + indexed_count;
    const float *sink = (const float *)dsv4_attention_vector(
        dense, attention, ".attn_sink", DSV4_DTYPE_F32,
        DSV4_ATTN_HEADS);
    if (!sink || selected < 1) return 0;
    if (work->prefill_collect) {
        if (!work->prefill_collect(work->prefill_collect_context,work->query,state->kv_cache,
                work->indices,selected,sink,state->position)) return 0;
        state->position++;
        return 1;
    }
    if (!dsv4_dense_sparse_attention(dense,
        work->context, work->query, state->kv_cache, DSV4_ATTN_HEADS,
        DSV4_ATTN_HEAD_DIM, work->indices, selected, sink,
        1.0f / sqrtf((float)DSV4_ATTN_HEAD_DIM),
        state->position,DSV4_ATTN_WINDOW,DSV4_INDEX_RATIO,DSV4_ATTN_WINDOW+state->max_compressed)) return 0;
    for (int head = 0; head < DSV4_ATTN_HEADS; head++)
        dsv4_rope(work->context + (size_t)head * DSV4_ATTN_HEAD_DIM +
                       DSV4_ATTN_HEAD_DIM - DSV4_ATTN_ROPE_DIM,
                   DSV4_ATTN_ROPE_DIM, state->position,
                   DSV4_ORIGINAL_CONTEXT, DSV4_COMPRESS_ROPE_THETA,
                   DSV4_ROPE_FACTOR, 32, 1, 1);
    dsv4_round_bf16_array(
        work->context, (size_t)DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM);
    if (work->prefill_context) {
        memcpy(work->prefill_context,work->context,
            (size_t)DSV4_ATTN_HEADS*DSV4_ATTN_HEAD_DIM*sizeof(float));
        state->position++;
        return 1;
    }
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
        if (!dsv4_dense_fp8_weight_bf16(dense,
                work->o_rank + (size_t)group * DSV4_ATTN_O_RANK,
                work->context + (size_t)group * group_width,
                wo_a_weight + (size_t)group * group_weight,
                wo_a_scale + (size_t)group * group_scale, 1,
                DSV4_ATTN_O_RANK, group_width))
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
