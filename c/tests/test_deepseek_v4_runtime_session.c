#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../deepseek_v4_runtime_session.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

static void seed_runtime(dsv4_runtime *runtime) {
    runtime->position = 1;
    runtime->history[0] = 123;
    for (int index = 0; index < DSV4_HC_MULT * DSV4_ATTN_HIDDEN; index++)
        runtime->hc[index] = dsv4_round_bf16((index % 17 - 8) / 8.0f);
    for (int layer = 0; layer < DSV4_RUNTIME_LAYERS; layer++) {
        dsv4_runtime_layer *current = &runtime->layers[layer];
        float *cache = current->mode == DSV4_LAYER_SLIDING
            ? current->attention.sliding.kv_cache
            : current->mode == DSV4_LAYER_OVERLAP
                ? current->attention.overlap.kv_cache
                : current->attention.nonoverlap.kv_cache;
        cache[0] = dsv4_round_bf16((layer + 1) / 16.0f);
        if (current->mode == DSV4_LAYER_SLIDING) {
            current->attention.sliding.position = 1;
        } else if (current->mode == DSV4_LAYER_NONOVERLAP) {
            current->attention.nonoverlap.position = 1;
            current->attention.nonoverlap.compressor.next_position = 1;
            current->attention.nonoverlap.compressor.kv_state[0] =
                layer + 0.25f;
            current->attention.nonoverlap.compressor.score_state[0] =
                layer + 0.5f;
            cache[(size_t)DSV4_ATTN_WINDOW * DSV4_ATTN_HEAD_DIM] =
                dsv4_round_bf16(layer + 0.75f);
        } else {
            current->attention.overlap.position = 1;
            current->attention.overlap.compressor.next_position = 1;
            current->attention.overlap.indexer.position = 1;
            current->attention.overlap.indexer.compressor.next_position = 1;
            current->attention.overlap.compressor.kv_state[0] =
                layer + 0.25f;
            current->attention.overlap.compressor.score_state[0] =
                layer + 0.5f;
            current->attention.overlap.indexer.compressor.kv_state[0] =
                layer + 1.25f;
            current->attention.overlap.indexer.compressor.score_state[0] =
                layer + 1.5f;
            cache[(size_t)DSV4_ATTN_WINDOW * DSV4_ATTN_HEAD_DIM] =
                dsv4_round_bf16(layer + 0.75f);
            current->attention.overlap.indexer.kv_cache[0] =
                dsv4_round_bf16(layer + 1.75f);
        }
    }
}

int main(void) {
    dsv4_runtime_session_shape large_shape;
    CHECK(dsv4_runtime_session_shape_for(100352, &large_shape));
    CHECK(!dsv4_runtime_session_shape_for(100353, &large_shape));
    CHECK((large_shape.compressed_kv + large_shape.window_kv + large_shape.mhc)*sizeof(uint16_t)
          + large_shape.compressor*sizeof(float) + 100352*sizeof(int32_t) < DSV4_SESSION_MAX_BYTES);
    dsv4_store store = {0};
    dsv4_dense_arena dense = {0};
    dsv4_expert_cache experts = {0};
    dsv4_runtime source = {0}, restored = {0};
    CHECK(dsv4_runtime_init(&source, &store, &dense, &experts, 1));
    CHECK(dsv4_runtime_init(&restored, &store, &dense, &experts, 1));
    seed_runtime(&source);
    dsv4_runtime_session_shape expected;
    CHECK(dsv4_runtime_session_shape_for(1, &expected));
    CHECK(expected.mhc == 4 * 4096 && expected.window_kv > 0 &&
          expected.compressed_kv > 0 && expected.compressor > 0);
    uint64_t rng[4] = {1, 2, 3, 4};
    dsv4_session_state snapshot = {0};
    CHECK(dsv4_runtime_export_session(
        &source, 0, 0.0f, 1.0f, rng, &snapshot));
    CHECK(snapshot.window_kv_n == expected.window_kv &&
          snapshot.compressed_kv_n == expected.compressed_kv &&
          snapshot.compressor_n == expected.compressor);

    dsv4_session_identity identity = {0};
    memset(identity.model, 1, sizeof(identity.model));
    memset(identity.tokenizer_protocol, 2, sizeof(identity.tokenizer_protocol));
    memset(identity.engine, 3, sizeof(identity.engine));
    char path[] = "/tmp/colib-dsv4-runtime-session-XXXXXX";
    int descriptor = mkstemp(path); CHECK(descriptor >= 0);
    close(descriptor); unlink(path);
    CHECK(dsv4_session_write(path, &identity, &snapshot));
    dsv4_session_state loaded = {0};
    CHECK(dsv4_session_read(path, &identity, 1, &loaded));
    CHECK(dsv4_runtime_restore_session(&restored, &loaded));
    CHECK(restored.position == 1 && restored.history[0] == 123);
    CHECK(!memcmp(source.hc, restored.hc,
                  DSV4_HC_MULT * DSV4_ATTN_HIDDEN * sizeof(float)));
    for (int layer = 0; layer < DSV4_RUNTIME_LAYERS; layer++) {
        dsv4_runtime_layer *left = &source.layers[layer];
        dsv4_runtime_layer *right = &restored.layers[layer];
        const float *left_cache = left->mode == DSV4_LAYER_SLIDING
            ? left->attention.sliding.kv_cache
            : left->mode == DSV4_LAYER_OVERLAP
                ? left->attention.overlap.kv_cache
                : left->attention.nonoverlap.kv_cache;
        const float *right_cache = right->mode == DSV4_LAYER_SLIDING
            ? right->attention.sliding.kv_cache
            : right->mode == DSV4_LAYER_OVERLAP
                ? right->attention.overlap.kv_cache
                : right->attention.nonoverlap.kv_cache;
        CHECK(left_cache[0] == right_cache[0]);
        int position = right->mode == DSV4_LAYER_SLIDING
            ? right->attention.sliding.position
            : right->mode == DSV4_LAYER_OVERLAP
                ? right->attention.overlap.position
                : right->attention.nonoverlap.position;
        CHECK(position == 1);
        if (right->mode == DSV4_LAYER_OVERLAP) {
            CHECK(left->attention.overlap.indexer.kv_cache[0] ==
                  right->attention.overlap.indexer.kv_cache[0]);
            CHECK(left->attention.overlap.indexer.compressor.score_state[0] ==
                  right->attention.overlap.indexer.compressor.score_state[0]);
        }
    }
    CHECK(!dsv4_runtime_restore_session(&restored, &loaded));
    loaded.context = 2;
    dsv4_runtime fresh = {0};
    CHECK(dsv4_runtime_init(&fresh, &store, &dense, &experts, 1));
    CHECK(!dsv4_runtime_restore_session(&fresh, &loaded));
    dsv4_runtime_close(&fresh);

    unlink(path);
    dsv4_session_free(&loaded); dsv4_session_free(&snapshot);
    dsv4_runtime_close(&restored); dsv4_runtime_close(&source);
    puts("DeepSeek-V4 live-runtime session round-trip tests: ok");
    return 0;
}
