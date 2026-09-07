#ifndef COLIB_DEEPSEEK_V4_RUNTIME_SESSION_H
#define COLIB_DEEPSEEK_V4_RUNTIME_SESSION_H

/* Deterministic production-layout mapping between the live scalar runtime and
 * the versioned session container. Runtime BF16 values are held as exactly
 * representable floats; packing therefore loses no model state. Compressor
 * accumulators remain FP32. */

#include "deepseek_v4_runtime.h"
#include "deepseek_v4_session.h"

typedef struct {
    uint64_t mhc;
    uint64_t window_kv;
    uint64_t compressed_kv;
    uint64_t compressor;
} dsv4_runtime_session_shape;

static inline int dsv4_runtime_session_shape_for(
    int context, dsv4_runtime_session_shape *shape) {
    if (!shape || context < 1 || context > 65536 ||
        DSV4_RUNTIME_LAYERS != 43 || DSV4_ATTN_HIDDEN != 4096)
        return 0;
    memset(shape, 0, sizeof(*shape));
    shape->mhc = (uint64_t)DSV4_HC_MULT * DSV4_ATTN_HIDDEN;
    for (int layer = 0; layer < DSV4_RUNTIME_LAYERS; layer++) {
        int mode = dsv4_runtime_layer_mode(layer);
        shape->window_kv +=
            (uint64_t)DSV4_ATTN_WINDOW * DSV4_ATTN_HEAD_DIM;
        if (mode == DSV4_LAYER_NONOVERLAP) {
            uint64_t compressed = ((uint64_t)context + 127) / 128;
            shape->compressed_kv += compressed * DSV4_ATTN_HEAD_DIM;
            shape->compressor += 2 * dsv4_compressor_state_floats(
                128, DSV4_ATTN_HEAD_DIM, 0);
        } else if (mode == DSV4_LAYER_OVERLAP) {
            uint64_t compressed =
                ((uint64_t)context + DSV4_INDEX_RATIO - 1) /
                DSV4_INDEX_RATIO;
            shape->compressed_kv += compressed *
                (DSV4_ATTN_HEAD_DIM + DSV4_INDEX_DIM);
            shape->compressor += 2 * dsv4_compressor_state_floats(
                DSV4_INDEX_RATIO, DSV4_ATTN_HEAD_DIM, 1);
            shape->compressor += 2 * dsv4_compressor_state_floats(
                DSV4_INDEX_RATIO, DSV4_INDEX_DIM, 1);
        }
    }
    return 1;
}

static inline void dsv4_runtime_pack_bf16(uint16_t *output,
                                           const float *input,
                                           size_t count) {
    for (size_t index = 0; index < count; index++)
        output[index] = dsv4_float_to_bf16(input[index]);
}

static inline void dsv4_runtime_unpack_bf16(float *output,
                                             const uint16_t *input,
                                             size_t count) {
    for (size_t index = 0; index < count; index++)
        output[index] = dsv4_bf16(input[index]);
}

static inline int dsv4_runtime_export_session(
    const dsv4_runtime *runtime, uint32_t sampler_profile,
    float temperature, float top_p, const uint64_t sampler_rng[4],
    dsv4_session_state *state) {
    dsv4_runtime_session_shape shape;
    if (!runtime || !state || !sampler_rng || runtime->poisoned ||
        runtime->position < 1 || runtime->position > runtime->context ||
        !dsv4_runtime_session_shape_for(runtime->context, &shape))
        return 0;
    dsv4_session_state packed = {0};
    packed.position = (uint32_t)runtime->position;
    packed.context = (uint32_t)runtime->context;
    packed.layers = DSV4_RUNTIME_LAYERS;
    packed.hidden = DSV4_ATTN_HIDDEN;
    packed.sampler_profile = sampler_profile;
    packed.temperature = temperature; packed.top_p = top_p;
    memcpy(packed.sampler_rng, sampler_rng, sizeof(packed.sampler_rng));
    packed.history_n = (uint64_t)runtime->position;
    packed.mhc_n = shape.mhc;
    packed.window_kv_n = shape.window_kv;
    packed.compressed_kv_n = shape.compressed_kv;
    packed.compressor_n = shape.compressor;
    packed.history = (int32_t *)dsv4_runtime_calloc(
        (size_t)packed.history_n, sizeof(*packed.history));
    packed.mhc = (uint16_t *)dsv4_runtime_calloc(
        (size_t)packed.mhc_n, sizeof(*packed.mhc));
    packed.window_kv = (uint16_t *)dsv4_runtime_calloc(
        (size_t)packed.window_kv_n, sizeof(*packed.window_kv));
    packed.compressed_kv = (uint16_t *)dsv4_runtime_calloc(
        (size_t)packed.compressed_kv_n, sizeof(*packed.compressed_kv));
    packed.compressor = (float *)dsv4_runtime_calloc(
        (size_t)packed.compressor_n, sizeof(*packed.compressor));
    if (!packed.history || !packed.mhc || !packed.window_kv ||
        !packed.compressed_kv || !packed.compressor) {
        dsv4_session_free(&packed); return 0;
    }
    memcpy(packed.history, runtime->history,
           (size_t)packed.history_n * sizeof(*packed.history));
    dsv4_runtime_pack_bf16(packed.mhc, runtime->hc,
                            (size_t)packed.mhc_n);
    uint64_t window_at = 0, compressed_at = 0, compressor_at = 0;
    for (int layer = 0; layer < DSV4_RUNTIME_LAYERS; layer++) {
        const dsv4_runtime_layer *current = &runtime->layers[layer];
        const float *cache = current->mode == DSV4_LAYER_SLIDING
            ? current->attention.sliding.kv_cache
            : current->mode == DSV4_LAYER_OVERLAP
                ? current->attention.overlap.kv_cache
                : current->attention.nonoverlap.kv_cache;
        size_t window_n = (size_t)DSV4_ATTN_WINDOW * DSV4_ATTN_HEAD_DIM;
        dsv4_runtime_pack_bf16(packed.window_kv + window_at, cache,
                                window_n);
        window_at += window_n;
        if (current->mode == DSV4_LAYER_NONOVERLAP) {
            size_t compressed_n =
                (size_t)current->attention.nonoverlap.max_compressed *
                DSV4_ATTN_HEAD_DIM;
            dsv4_runtime_pack_bf16(
                packed.compressed_kv + compressed_at, cache + window_n,
                compressed_n);
            compressed_at += compressed_n;
            size_t state_n = dsv4_compressor_state_floats(
                128, DSV4_ATTN_HEAD_DIM, 0);
            memcpy(packed.compressor + compressor_at,
                   current->attention.nonoverlap.compressor.kv_state,
                   state_n * sizeof(float));
            compressor_at += state_n;
            memcpy(packed.compressor + compressor_at,
                   current->attention.nonoverlap.compressor.score_state,
                   state_n * sizeof(float));
            compressor_at += state_n;
        } else if (current->mode == DSV4_LAYER_OVERLAP) {
            size_t compressed_n =
                (size_t)current->attention.overlap.max_compressed *
                DSV4_ATTN_HEAD_DIM;
            dsv4_runtime_pack_bf16(
                packed.compressed_kv + compressed_at, cache + window_n,
                compressed_n);
            compressed_at += compressed_n;
            size_t main_n = dsv4_compressor_state_floats(
                DSV4_INDEX_RATIO, DSV4_ATTN_HEAD_DIM, 1);
            memcpy(packed.compressor + compressor_at,
                   current->attention.overlap.compressor.kv_state,
                   main_n * sizeof(float));
            compressor_at += main_n;
            memcpy(packed.compressor + compressor_at,
                   current->attention.overlap.compressor.score_state,
                   main_n * sizeof(float));
            compressor_at += main_n;
            size_t index_cache_n =
                (size_t)current->attention.overlap.indexer.max_compressed *
                DSV4_INDEX_DIM;
            dsv4_runtime_pack_bf16(
                packed.compressed_kv + compressed_at,
                current->attention.overlap.indexer.kv_cache,
                index_cache_n);
            compressed_at += index_cache_n;
            size_t index_n = dsv4_compressor_state_floats(
                DSV4_INDEX_RATIO, DSV4_INDEX_DIM, 1);
            memcpy(packed.compressor + compressor_at,
                   current->attention.overlap.indexer.compressor.kv_state,
                   index_n * sizeof(float));
            compressor_at += index_n;
            memcpy(packed.compressor + compressor_at,
                   current->attention.overlap.indexer.compressor.score_state,
                   index_n * sizeof(float));
            compressor_at += index_n;
        }
    }
    if (window_at != shape.window_kv || compressed_at != shape.compressed_kv ||
        compressor_at != shape.compressor ||
        !dsv4_session_layout_valid(&packed, NULL)) {
        dsv4_session_free(&packed); return 0;
    }
    dsv4_session_free(state);
    *state = packed;
    return 1;
}

static inline int dsv4_runtime_restore_session(
    dsv4_runtime *runtime, const dsv4_session_state *state) {
    dsv4_runtime_session_shape shape;
    if (!runtime || !state || runtime->position || runtime->poisoned ||
        state->context != (uint32_t)runtime->context ||
        !dsv4_runtime_session_shape_for(runtime->context, &shape) ||
        !dsv4_session_layout_valid(state, NULL) ||
        state->mhc_n != shape.mhc ||
        state->window_kv_n != shape.window_kv ||
        state->compressed_kv_n != shape.compressed_kv ||
        state->compressor_n != shape.compressor)
        return 0;
    memcpy(runtime->history, state->history,
           (size_t)state->history_n * sizeof(*runtime->history));
    dsv4_runtime_unpack_bf16(runtime->hc, state->mhc,
                              (size_t)state->mhc_n);
    uint64_t window_at = 0, compressed_at = 0, compressor_at = 0;
    for (int layer = 0; layer < DSV4_RUNTIME_LAYERS; layer++) {
        dsv4_runtime_layer *current = &runtime->layers[layer];
        float *cache = current->mode == DSV4_LAYER_SLIDING
            ? current->attention.sliding.kv_cache
            : current->mode == DSV4_LAYER_OVERLAP
                ? current->attention.overlap.kv_cache
                : current->attention.nonoverlap.kv_cache;
        size_t window_n = (size_t)DSV4_ATTN_WINDOW * DSV4_ATTN_HEAD_DIM;
        dsv4_runtime_unpack_bf16(cache, state->window_kv + window_at,
                                  window_n);
        window_at += window_n;
        if (current->mode == DSV4_LAYER_SLIDING) {
            current->attention.sliding.position = (int)state->position;
        } else if (current->mode == DSV4_LAYER_NONOVERLAP) {
            size_t compressed_n =
                (size_t)current->attention.nonoverlap.max_compressed *
                DSV4_ATTN_HEAD_DIM;
            dsv4_runtime_unpack_bf16(cache + window_n,
                                      state->compressed_kv + compressed_at,
                                      compressed_n);
            compressed_at += compressed_n;
            size_t state_n = dsv4_compressor_state_floats(
                128, DSV4_ATTN_HEAD_DIM, 0);
            memcpy(current->attention.nonoverlap.compressor.kv_state,
                   state->compressor + compressor_at,
                   state_n * sizeof(float));
            compressor_at += state_n;
            memcpy(current->attention.nonoverlap.compressor.score_state,
                   state->compressor + compressor_at,
                   state_n * sizeof(float));
            compressor_at += state_n;
            current->attention.nonoverlap.position = (int)state->position;
            current->attention.nonoverlap.compressor.next_position =
                (int)state->position;
        } else {
            size_t compressed_n =
                (size_t)current->attention.overlap.max_compressed *
                DSV4_ATTN_HEAD_DIM;
            dsv4_runtime_unpack_bf16(cache + window_n,
                                      state->compressed_kv + compressed_at,
                                      compressed_n);
            compressed_at += compressed_n;
            size_t main_n = dsv4_compressor_state_floats(
                DSV4_INDEX_RATIO, DSV4_ATTN_HEAD_DIM, 1);
            memcpy(current->attention.overlap.compressor.kv_state,
                   state->compressor + compressor_at,
                   main_n * sizeof(float));
            compressor_at += main_n;
            memcpy(current->attention.overlap.compressor.score_state,
                   state->compressor + compressor_at,
                   main_n * sizeof(float));
            compressor_at += main_n;
            size_t index_cache_n =
                (size_t)current->attention.overlap.indexer.max_compressed *
                DSV4_INDEX_DIM;
            dsv4_runtime_unpack_bf16(
                current->attention.overlap.indexer.kv_cache,
                state->compressed_kv + compressed_at, index_cache_n);
            compressed_at += index_cache_n;
            size_t index_n = dsv4_compressor_state_floats(
                DSV4_INDEX_RATIO, DSV4_INDEX_DIM, 1);
            memcpy(current->attention.overlap.indexer.compressor.kv_state,
                   state->compressor + compressor_at,
                   index_n * sizeof(float));
            compressor_at += index_n;
            memcpy(current->attention.overlap.indexer.compressor.score_state,
                   state->compressor + compressor_at,
                   index_n * sizeof(float));
            compressor_at += index_n;
            current->attention.overlap.position = (int)state->position;
            current->attention.overlap.compressor.next_position =
                (int)state->position;
            current->attention.overlap.indexer.position =
                (int)state->position;
            current->attention.overlap.indexer.compressor.next_position =
                (int)state->position;
        }
    }
    if (window_at != shape.window_kv || compressed_at != shape.compressed_kv ||
        compressor_at != shape.compressor)
        return 0;
    runtime->position = (int)state->position;
    return 1;
}

#endif
