#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DSV4_DIM 128
#define DSV4_VOCAB 3
#define DSV4_EXPERTS 8
#define DSV4_TOPK 2
#define DSV4_ATTN_HIDDEN 128
#define DSV4_ATTN_HEADS 2
#define DSV4_ATTN_HEAD_DIM 128
#define DSV4_ATTN_ROPE_DIM 64
#define DSV4_ATTN_Q_RANK 128
#define DSV4_ATTN_O_GROUPS 1
#define DSV4_ATTN_O_RANK 128
#define DSV4_ATTN_WINDOW 4
#define DSV4_INDEX_HEADS 2
#define DSV4_INDEX_DIM 128
#define DSV4_INDEX_TOPK 2
#define DSV4_INDEX_RATIO 4
#define DSV4_MOE_INTERMEDIATE 128
#define DSV4_RUNTIME_LAYERS 5
#include "../deepseek_v4_runtime.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

int main(void) {
    dsv4_store store = {0};
    dsv4_dense_arena dense = {0};
    dsv4_expert_cache experts = {0};
    dsv4_runtime runtime = {0};
    CHECK(dsv4_runtime_state_bytes(1) > 0);
    CHECK(dsv4_runtime_state_bytes(65536) >
          dsv4_runtime_state_bytes(1));
    CHECK(dsv4_runtime_state_bytes(0) == 0);
    CHECK(dsv4_runtime_state_bytes(65537) == 0);
    CHECK(dsv4_runtime_init(&runtime, &store, &dense, &experts, 1));
    CHECK(runtime.context == 1 && runtime.position == 0 &&
          !runtime.poisoned);
    CHECK(runtime.layers[0].mode == DSV4_LAYER_SLIDING);
    CHECK(runtime.layers[1].mode == DSV4_LAYER_SLIDING);
    CHECK(runtime.layers[2].mode == DSV4_LAYER_OVERLAP);
    CHECK(runtime.layers[2].attention.overlap.max_compressed == 1);
    CHECK(runtime.layers[3].mode == DSV4_LAYER_NONOVERLAP);
    CHECK(runtime.layers[3].attention.nonoverlap.max_compressed == 1);
    CHECK(runtime.layers[4].mode == DSV4_LAYER_OVERLAP);
    CHECK(runtime.block.attention.indices != NULL);
    CHECK(runtime.overlap_scratch.indexer.scores != NULL);
    float invalid_logits[DSV4_VOCAB];
    CHECK(!dsv4_runtime_decode_token(
        &runtime, 0, invalid_logits, 1, 0));
    CHECK(runtime.poisoned && runtime.position == 0);
    CHECK(!dsv4_runtime_decode_token(
        &runtime, 0, invalid_logits, 1, 0));
    dsv4_runtime_close(&runtime);
    CHECK(runtime.context == 0 && runtime.hc == NULL);

    CHECK(dsv4_runtime_init(&runtime, &store, &dense, &experts, 13));
    CHECK(runtime.layers[2].attention.overlap.max_compressed == 4);
    CHECK(runtime.layers[3].attention.nonoverlap.max_compressed == 1);
    CHECK(runtime.layers[4].attention.overlap.indexer.max_compressed == 4);
    CHECK(runtime.layers[2].attention.overlap.compressor.next_position == 0);
    CHECK(runtime.layers[2].attention.overlap.indexer.position == 0);
    dsv4_runtime_close(&runtime);

    char root[] = "/tmp/colib-dsv4-runtime-head-XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    char segment[1024], manifest[1024];
    snprintf(segment, sizeof(segment), "%s/dense.bin", root);
    snprintf(manifest, sizeof(manifest), "%s/model-manifest.json", root);
    unsigned char payload[8468] = {0};
    for (int axis = 0; axis < DSV4_ATTN_HIDDEN; axis++)
        ((uint16_t *)(payload + 8212))[axis] = 0x3f80;
    FILE *file = fopen(segment, "wb"); CHECK(file != NULL);
    CHECK(fwrite(payload, 1, sizeof(payload), file) == sizeof(payload) &&
          fclose(file) == 0);
    file = fopen(manifest, "wb"); CHECK(file != NULL);
    CHECK(fprintf(file,
        "{\"schema\":\"colib.deepseek-v4.model-manifest.v1\","
        "\"source\":{\"revision\":"
        "\"9e165c30e2704aec5d9d593cce3eebd58bbef1cb\"},"
        "\"inventory\":{\"dense.bin\":["
        "{\"name\":\"hc_head_fn\",\"file\":\"dense.bin\","
        "\"offset\":0,\"nbytes\":8192,\"dtype\":\"F32\","
        "\"shape\":[4,512]},"
        "{\"name\":\"hc_head_base\",\"file\":\"dense.bin\","
        "\"offset\":8192,\"nbytes\":16,\"dtype\":\"F32\","
        "\"shape\":[4]},"
        "{\"name\":\"hc_head_scale\",\"file\":\"dense.bin\","
        "\"offset\":8208,\"nbytes\":4,\"dtype\":\"F32\","
        "\"shape\":[1]},"
        "{\"name\":\"norm.weight\",\"file\":\"dense.bin\","
        "\"offset\":8212,\"nbytes\":256,\"dtype\":\"BF16\","
        "\"shape\":[128]}]}}") > 0 && fclose(file) == 0);
    dsv4_store head_store; CHECK(dsv4_store_init(&head_store, root));
    dsv4_dense_arena head_dense;
    CHECK(dsv4_dense_arena_init(&head_dense, &head_store, 0, 0));
    float hc[DSV4_HC_MULT * DSV4_ATTN_HIDDEN], hidden[DSV4_ATTN_HIDDEN];
    for (int copy = 0; copy < DSV4_HC_MULT; copy++)
        for (int axis = 0; axis < DSV4_ATTN_HIDDEN; axis++)
            hc[copy * DSV4_ATTN_HIDDEN + axis] = (float)(copy + 1);
    CHECK(dsv4_runtime_head(&head_dense, hc, hidden));
    for (int axis = 0; axis < DSV4_ATTN_HIDDEN; axis++)
        CHECK(fabsf(hidden[axis] - 1.0f) < 1e-5f);
    dsv4_dense_arena_close(&head_dense);
    dsv4_store_close(&head_store);
    unlink(manifest); unlink(segment); rmdir(root);
    puts("DeepSeek-V4 bounded runtime ownership tests: ok");
    return 0;
}
