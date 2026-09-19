#ifndef COLIB_DEEPSEEK_V4_RUNTIME_H
#define COLIB_DEEPSEEK_V4_RUNTIME_H

/* Bounded ownership and single-token scalar execution for the pinned base
 * model.  The mux scheduler will own one instance per live slot; a failed
 * token poisons the instance so partially advanced layer state is never
 * reused as a valid prefix. */

#include "deepseek_v4_block.h"

#ifndef DSV4_RUNTIME_LAYERS
#define DSV4_RUNTIME_LAYERS 43
#endif

typedef enum {
    DSV4_LAYER_SLIDING = 0,
    DSV4_LAYER_OVERLAP = 4,
    DSV4_LAYER_NONOVERLAP = 128,
} dsv4_layer_mode;

typedef struct {
    dsv4_layer_mode mode;
    union {
        dsv4_sliding_attention_state sliding;
        dsv4_compressed_attention_state nonoverlap;
        dsv4_indexed_attention_state overlap;
    } attention;
} dsv4_runtime_layer;

typedef struct {
    dsv4_store *store;
    dsv4_dense_arena *dense;
    dsv4_expert_cache *experts;
    int context;
    int position;
    int prefill_chunk_start;
    int poisoned;
    int (*trace)(void *, const char *, int, const float *, int, int);
    void *trace_context;
    int32_t *history;
    float *hc;
    float *head_hidden;
    dsv4_runtime_layer layers[DSV4_RUNTIME_LAYERS];
    dsv4_block_scratch block;
    dsv4_compressed_attention_scratch nonoverlap_scratch;
    dsv4_indexed_attention_scratch overlap_scratch;
    double last_layer_seconds[DSV4_RUNTIME_LAYERS];
    double last_attention_stage_seconds[DSV4_RUNTIME_LAYERS];
    double last_ffn_stage_seconds[DSV4_RUNTIME_LAYERS];
    double last_route_seconds[DSV4_RUNTIME_LAYERS];
    double last_routed_seconds[DSV4_RUNTIME_LAYERS];
    double last_shared_seconds[DSV4_RUNTIME_LAYERS];
    double last_prefill_read_seconds[DSV4_RUNTIME_LAYERS];
    double last_head_seconds;
    /* Decode (single-token chunk) attribution, cumulative over the request. */
    uint64_t decode_steps;
    double decode_step_seconds, decode_attention_seconds, decode_route_seconds,
        decode_routed_seconds, decode_shared_seconds, decode_head_seconds;
    /* One grouped six-expert submission per layer instead of six serial
     * single-expert fetches. Defaults on; DSV4_DECODE_GROUPED=0 restores the
     * serial path for paired comparison. */
    int decode_grouped;
    char error[256];
} dsv4_runtime;

static inline dsv4_layer_mode dsv4_runtime_layer_mode(int layer) {
    if (layer < 2) return DSV4_LAYER_SLIDING;
    return (layer & 1) ? DSV4_LAYER_NONOVERLAP : DSV4_LAYER_OVERLAP;
}

static inline void *dsv4_runtime_calloc(size_t count, size_t size) {
    if (!count || !size || count > SIZE_MAX / size) return NULL;
    return calloc(count, size);
}

static inline uint64_t dsv4_runtime_state_bytes(int context) {
    if (context < 1 || context > DSV4_MAX_CONTEXT) return 0;
    uint64_t floats = (uint64_t)DSV4_HC_MULT * DSV4_ATTN_HIDDEN +
        DSV4_ATTN_HIDDEN;
    uint64_t bytes = (uint64_t)context * sizeof(int32_t);
    for (int layer = 0; layer < DSV4_RUNTIME_LAYERS; layer++) {
        int mode = dsv4_runtime_layer_mode(layer);
        uint64_t compressed = mode == DSV4_LAYER_SLIDING ? 0 :
            ((uint64_t)context + mode - 1) / mode;
        floats += ((uint64_t)DSV4_ATTN_WINDOW + compressed) *
                  DSV4_ATTN_HEAD_DIM;
        if (mode == DSV4_LAYER_NONOVERLAP) {
            floats += 2 * dsv4_compressor_state_floats(
                128, DSV4_ATTN_HEAD_DIM, 0);
        } else if (mode == DSV4_LAYER_OVERLAP) {
            floats += 2 * dsv4_compressor_state_floats(
                DSV4_INDEX_RATIO, DSV4_ATTN_HEAD_DIM, 1);
            floats += compressed * DSV4_INDEX_DIM;
            floats += 2 * dsv4_compressor_state_floats(
                DSV4_INDEX_RATIO, DSV4_INDEX_DIM, 1);
        }
    }
    return floats > (UINT64_MAX - bytes) / sizeof(float)
        ? 0 : bytes + floats * sizeof(float);
}

static inline int dsv4_runtime_init_layer(dsv4_runtime *runtime,
                                           int layer) {
    if (!runtime || layer<0 || layer>=DSV4_RUNTIME_LAYERS) return 0;
    dsv4_runtime_layer *current = &runtime->layers[layer];
    current->mode = dsv4_runtime_layer_mode(layer);
    int context = runtime->context;
    if (current->mode == DSV4_LAYER_SLIDING) {
        float *cache = (float *)dsv4_runtime_calloc(
            (size_t)DSV4_ATTN_WINDOW * DSV4_ATTN_HEAD_DIM,
            sizeof(float));
        return cache && dsv4_sliding_attention_state_init(
                            &current->attention.sliding, cache);
    }
    int compressed = (context + current->mode - 1) / current->mode;
    float *cache = (float *)dsv4_runtime_calloc(
        ((size_t)DSV4_ATTN_WINDOW + compressed) * DSV4_ATTN_HEAD_DIM,
        sizeof(float));
    if (!cache) return 0;
    if (current->mode == DSV4_LAYER_NONOVERLAP) {
        size_t state_n = dsv4_compressor_state_floats(
            128, DSV4_ATTN_HEAD_DIM, 0);
        float *kv = (float *)dsv4_runtime_calloc(state_n, sizeof(float));
        float *score = (float *)dsv4_runtime_calloc(state_n, sizeof(float));
        if (!kv || !score || !dsv4_compressed_attention_state_init(
                &current->attention.nonoverlap, context, cache, kv, score)) {
            free(cache); free(kv); free(score); return 0;
        }
        return 1;
    }
    size_t main_n = dsv4_compressor_state_floats(
        DSV4_INDEX_RATIO, DSV4_ATTN_HEAD_DIM, 1);
    size_t index_n = dsv4_compressor_state_floats(
        DSV4_INDEX_RATIO, DSV4_INDEX_DIM, 1);
    float *main_kv = (float *)dsv4_runtime_calloc(main_n, sizeof(float));
    float *main_score = (float *)dsv4_runtime_calloc(main_n, sizeof(float));
    float *index_cache = (float *)dsv4_runtime_calloc(
        (size_t)compressed * DSV4_INDEX_DIM, sizeof(float));
    float *index_kv = (float *)dsv4_runtime_calloc(index_n, sizeof(float));
    float *index_score = (float *)dsv4_runtime_calloc(index_n, sizeof(float));
    if (!main_kv || !main_score || !index_cache || !index_kv ||
        !index_score || !dsv4_indexed_attention_state_init(
            &current->attention.overlap, context, cache, main_kv,
            main_score, index_cache, index_kv, index_score)) {
        free(cache); free(main_kv); free(main_score); free(index_cache);
        free(index_kv); free(index_score); return 0;
    }
    return 1;
}

static inline int dsv4_runtime_init_scratch(dsv4_runtime *runtime) {
    dsv4_block_scratch *block = &runtime->block;
#define DSV4_ALLOC_FLOAT(field, count) \
    ((field) = (float *)dsv4_runtime_calloc((count), sizeof(float)))
#define DSV4_ALLOC_BYTE(field, count) \
    ((field) = (uint8_t *)dsv4_runtime_calloc((count), sizeof(uint8_t)))
    size_t query_n = (size_t)DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM;
    size_t output_rank_n = (size_t)DSV4_ATTN_O_GROUPS * DSV4_ATTN_O_RANK;
    size_t selection_n = DSV4_ATTN_WINDOW +
        (DSV4_INDEX_TOPK > (runtime->context + 127) / 128
             ? DSV4_INDEX_TOPK : (runtime->context + 127) / 128);
    size_t activation_n = query_n > DSV4_ATTN_HIDDEN
        ? query_n : DSV4_ATTN_HIDDEN;
    if (activation_n < DSV4_MOE_INTERMEDIATE)
        activation_n = DSV4_MOE_INTERMEDIATE;
    if (!DSV4_ALLOC_FLOAT(runtime->hc,
                          (size_t)DSV4_HC_MULT * DSV4_ATTN_HIDDEN) ||
        !DSV4_ALLOC_FLOAT(runtime->head_hidden, DSV4_ATTN_HIDDEN) ||
        !DSV4_ALLOC_FLOAT(block->reduced, DSV4_ATTN_HIDDEN) ||
        !DSV4_ALLOC_FLOAT(block->post, DSV4_HC_MULT) ||
        !DSV4_ALLOC_FLOAT(block->combination,
                          DSV4_HC_MULT * DSV4_HC_MULT) ||
        !DSV4_ALLOC_FLOAT(block->mixes, DSV4_HC_MIX) ||
        !DSV4_ALLOC_FLOAT(block->module_output, DSV4_ATTN_HIDDEN) ||
        !DSV4_ALLOC_FLOAT(block->expanded,
                          (size_t)DSV4_HC_MULT * DSV4_ATTN_HIDDEN) ||
        !DSV4_ALLOC_FLOAT(block->attention.q_rank, DSV4_ATTN_Q_RANK) ||
        !DSV4_ALLOC_FLOAT(block->attention.query, query_n) ||
        !DSV4_ALLOC_FLOAT(block->attention.kv, DSV4_ATTN_HEAD_DIM) ||
        !DSV4_ALLOC_FLOAT(block->attention.context, query_n) ||
        !DSV4_ALLOC_FLOAT(block->attention.o_rank, output_rank_n) ||
        !DSV4_ALLOC_FLOAT(block->attention.scores, selection_n) ||
        !(block->attention.indices = (int *)dsv4_runtime_calloc(
              selection_n, sizeof(int))) ||
        !DSV4_ALLOC_BYTE(block->attention.activation, activation_n) ||
        !DSV4_ALLOC_BYTE(block->attention.activation_scale,
                          (activation_n + 127) / 128) ||
        !DSV4_ALLOC_FLOAT(block->router_logits, DSV4_EXPERTS) ||
        !DSV4_ALLOC_FLOAT(block->routed, DSV4_ATTN_HIDDEN) ||
        !DSV4_ALLOC_FLOAT(block->shared, DSV4_ATTN_HIDDEN) ||
        !DSV4_ALLOC_FLOAT(block->expert_output, DSV4_ATTN_HIDDEN) ||
        !DSV4_ALLOC_FLOAT(block->moe_gate, DSV4_MOE_INTERMEDIATE) ||
        !DSV4_ALLOC_FLOAT(block->moe_up, DSV4_MOE_INTERMEDIATE) ||
        !DSV4_ALLOC_BYTE(block->moe_activation, activation_n) ||
        !DSV4_ALLOC_BYTE(block->moe_activation_scale,
                          (activation_n + 31) / 32))
        return 0;
    runtime->nonoverlap_scratch.attention = block->attention;
    if (!DSV4_ALLOC_FLOAT(runtime->nonoverlap_scratch.compressor_kv,
                          DSV4_ATTN_HEAD_DIM) ||
        !DSV4_ALLOC_FLOAT(runtime->nonoverlap_scratch.compressor_score,
                          DSV4_ATTN_HEAD_DIM))
        return 0;
    runtime->overlap_scratch.attention = block->attention;
    if (!DSV4_ALLOC_FLOAT(runtime->overlap_scratch.compressor_kv,
                          2 * DSV4_ATTN_HEAD_DIM) ||
        !DSV4_ALLOC_FLOAT(runtime->overlap_scratch.compressor_score,
                          2 * DSV4_ATTN_HEAD_DIM) ||
        !DSV4_ALLOC_FLOAT(runtime->overlap_scratch.indexer.query,
                          (size_t)DSV4_INDEX_HEADS * DSV4_INDEX_DIM) ||
        !DSV4_ALLOC_FLOAT(runtime->overlap_scratch.indexer.compressor_kv,
                          2 * DSV4_INDEX_DIM) ||
        !DSV4_ALLOC_FLOAT(runtime->overlap_scratch.indexer.compressor_score,
                          2 * DSV4_INDEX_DIM) ||
        !DSV4_ALLOC_FLOAT(runtime->overlap_scratch.indexer.head_weights,
                          DSV4_INDEX_HEADS) ||
        !DSV4_ALLOC_FLOAT(runtime->overlap_scratch.indexer.scores,
                          (runtime->context + DSV4_INDEX_RATIO - 1) /
                              DSV4_INDEX_RATIO) ||
        !DSV4_ALLOC_BYTE(runtime->overlap_scratch.indexer.activation,
                          DSV4_ATTN_Q_RANK) ||
        !DSV4_ALLOC_BYTE(runtime->overlap_scratch.indexer.activation_scale,
                          (DSV4_ATTN_Q_RANK + 127) / 128))
        return 0;
#undef DSV4_ALLOC_FLOAT
#undef DSV4_ALLOC_BYTE
    return 1;
}

static inline void dsv4_runtime_close(dsv4_runtime *runtime) {
    if (!runtime) return;
    for (int layer = 0; layer < DSV4_RUNTIME_LAYERS; layer++) {
        dsv4_runtime_layer *current = &runtime->layers[layer];
        if (current->mode == DSV4_LAYER_SLIDING) {
            free(current->attention.sliding.kv_cache);
        } else if (current->mode == DSV4_LAYER_NONOVERLAP) {
            free(current->attention.nonoverlap.kv_cache);
            free(current->attention.nonoverlap.compressor.kv_state);
            free(current->attention.nonoverlap.compressor.score_state);
        } else if (current->mode == DSV4_LAYER_OVERLAP) {
            free(current->attention.overlap.kv_cache);
            free(current->attention.overlap.compressor.kv_state);
            free(current->attention.overlap.compressor.score_state);
            free(current->attention.overlap.indexer.kv_cache);
            free(current->attention.overlap.indexer.compressor.kv_state);
            free(current->attention.overlap.indexer.compressor.score_state);
        }
    }
    free(runtime->history); free(runtime->hc); free(runtime->head_hidden);
    dsv4_block_scratch *block = &runtime->block;
    free(block->reduced); free(block->post); free(block->combination);
    free(block->mixes); free(block->module_output); free(block->expanded);
    free(block->attention.q_rank); free(block->attention.query);
    free(block->attention.kv); free(block->attention.context);
    free(block->attention.o_rank); free(block->attention.scores);
    free(block->attention.indices); free(block->attention.activation);
    free(block->attention.activation_scale); free(block->router_logits);
    free(block->routed); free(block->shared); free(block->expert_output);
    free(block->moe_gate); free(block->moe_up);
    free(block->moe_activation); free(block->moe_activation_scale);
    free(runtime->nonoverlap_scratch.compressor_kv);
    free(runtime->nonoverlap_scratch.compressor_score);
    free(runtime->overlap_scratch.compressor_kv);
    free(runtime->overlap_scratch.compressor_score);
    free(runtime->overlap_scratch.indexer.query);
    free(runtime->overlap_scratch.indexer.compressor_kv);
    free(runtime->overlap_scratch.indexer.compressor_score);
    free(runtime->overlap_scratch.indexer.head_weights);
    free(runtime->overlap_scratch.indexer.scores);
    free(runtime->overlap_scratch.indexer.activation);
    free(runtime->overlap_scratch.indexer.activation_scale);
    memset(runtime, 0, sizeof(*runtime));
}

static inline int dsv4_runtime_init(dsv4_runtime *runtime,
                                     dsv4_store *store,
                                     dsv4_dense_arena *dense,
                                     dsv4_expert_cache *experts,
                                     int context) {
    if (!runtime || !store || !dense || !experts || context < 1 ||
        context > DSV4_MAX_CONTEXT || !dsv4_runtime_state_bytes(context))
        return 0;
    memset(runtime, 0, sizeof(*runtime));
    runtime->store = store; runtime->dense = dense;
    runtime->experts = experts; runtime->context = context;
    const char *grouped = getenv("DSV4_DECODE_GROUPED");
    runtime->decode_grouped = !grouped || !*grouped || atoi(grouped) != 0;
    runtime->history = (int32_t *)dsv4_runtime_calloc(
        (size_t)context, sizeof(*runtime->history));
    if (!runtime->history || !dsv4_runtime_init_scratch(runtime)) {
        dsv4_runtime_close(runtime);
        snprintf(runtime->error, sizeof(runtime->error),
                 "cannot allocate runtime scratch");
        return 0;
    }
    for (int layer = 0; layer < DSV4_RUNTIME_LAYERS; layer++)
        if (!dsv4_runtime_init_layer(runtime, layer)) {
            dsv4_runtime_close(runtime);
            snprintf(runtime->error, sizeof(runtime->error),
                     "cannot allocate layer %d state", layer);
            return 0;
        }
    return 1;
}

static inline int dsv4_runtime_head(const dsv4_dense_arena *dense,
                                     const float *hc, float *hidden) {
    const float *function_weight = (const float *)dsv4_dense_named(
        dense, "hc_head", "_fn", DSV4_DTYPE_F32, DSV4_HC_MULT,
        DSV4_HC_MULT * DSV4_ATTN_HIDDEN);
    const float *base = (const float *)dsv4_attention_vector(
        dense, "hc_head", "_base", DSV4_DTYPE_F32, DSV4_HC_MULT);
    const float *scale = (const float *)dsv4_attention_vector(
        dense, "hc_head", "_scale", DSV4_DTYPE_F32, 1);
    const uint16_t *norm = (const uint16_t *)dsv4_attention_vector(
        dense, "norm", ".weight", DSV4_DTYPE_BF16, DSV4_ATTN_HIDDEN);
    if (!function_weight || !base || !scale || !norm) return 0;
    double square_sum = 0.0;
    for (int index = 0; index < DSV4_HC_MULT * DSV4_ATTN_HIDDEN; index++)
        square_sum += (double)hc[index] * hc[index];
    float inverse = 1.0f / sqrtf(
        (float)(square_sum /
                (DSV4_HC_MULT * DSV4_ATTN_HIDDEN)) + 1e-6f);
    float pre[DSV4_HC_MULT];
    for (int copy = 0; copy < DSV4_HC_MULT; copy++) {
        double sum = 0.0;
        const float *row = function_weight +
            (size_t)copy * DSV4_HC_MULT * DSV4_ATTN_HIDDEN;
        for (int index = 0;
             index < DSV4_HC_MULT * DSV4_ATTN_HIDDEN; index++)
            sum += (double)row[index] * hc[index];
        pre[copy] = dsv4_sigmoid((float)sum * inverse * scale[0] +
                                  base[copy]) + 1e-6f;
    }
    for (int axis = 0; axis < DSV4_ATTN_HIDDEN; axis++) {
        float sum = 0.0f;
        for (int copy = 0; copy < DSV4_HC_MULT; copy++)
            sum += pre[copy] *
                   hc[(size_t)copy * DSV4_ATTN_HIDDEN + axis];
        hidden[axis] = sum;
    }
    dsv4_round_bf16_array(hidden, DSV4_ATTN_HIDDEN);
    dsv4_rmsnorm(hidden, hidden, norm, DSV4_ATTN_HIDDEN, 1e-6f);
    return 1;
}

static inline int dsv4_runtime_decode_token(dsv4_runtime *runtime,
                                             int token, float *logits,
                                             int head_rows_per_read,
                                             int direct) {
    (void)head_rows_per_read;
    (void)direct;
    if (!runtime || runtime->poisoned || !runtime->store ||
        !runtime->dense || !runtime->experts || !logits ||
        token < 0 || token >= DSV4_VOCAB ||
        runtime->position >= runtime->context)
        return 0;
    if (!dsv4_dense_embed_token(runtime->dense, token, runtime->hc))
        goto fail;
    for (int layer = 0; layer < DSV4_RUNTIME_LAYERS; layer++) {
        double layer_start = dsv4_dense_now_seconds();
        dsv4_runtime_layer *current = &runtime->layers[layer];
        int ok = current->mode == DSV4_LAYER_SLIDING
            ? dsv4_block_decode_sliding(
                  runtime->dense, runtime->experts, layer, token,
                  runtime->hc, &current->attention.sliding, &runtime->block)
            : current->mode == DSV4_LAYER_OVERLAP
                ? dsv4_block_decode_compressed_overlap(
                      runtime->dense, runtime->experts, layer, token,
                      runtime->hc, &current->attention.overlap,
                      &runtime->overlap_scratch, &runtime->block)
                : dsv4_block_decode_compressed_nonoverlap(
                      runtime->dense, runtime->experts, layer, token,
                      runtime->hc, &current->attention.nonoverlap,
                      &runtime->nonoverlap_scratch, &runtime->block);
        runtime->last_layer_seconds[layer] =
            dsv4_dense_now_seconds() - layer_start;
        runtime->last_attention_stage_seconds[layer] =
            runtime->block.last_attention_stage_seconds;
        runtime->last_ffn_stage_seconds[layer] =
            runtime->block.last_ffn_stage_seconds;
        runtime->last_route_seconds[layer] =
            runtime->experts->last_route_seconds;
        runtime->last_routed_seconds[layer] =
            runtime->experts->last_routed_seconds;
        runtime->last_shared_seconds[layer] =
            runtime->experts->last_shared_seconds;
        if (!ok) goto fail;
    }
    double head_start = dsv4_dense_now_seconds();
    const uint16_t *head = (const uint16_t *)dsv4_dense_named(
        runtime->dense, "head", ".weight", DSV4_DTYPE_BF16,
        DSV4_VOCAB, DSV4_ATTN_HIDDEN);
    if (!head ||
        !dsv4_runtime_head(runtime->dense, runtime->hc,
                           runtime->head_hidden) ||
        !dsv4_dense_linear_bf16(
            runtime->dense, logits, runtime->head_hidden, head,
            1, DSV4_VOCAB, DSV4_ATTN_HIDDEN))
        goto fail;
    runtime->last_head_seconds = dsv4_dense_now_seconds() - head_start;
    runtime->history[runtime->position++] = token;
    return 1;
fail:
    runtime->poisoned = 1;
    snprintf(runtime->error, sizeof(runtime->error),
             "token %d failed after position %d; state poisoned",
             token, runtime->position);
    return 0;
}

#endif
