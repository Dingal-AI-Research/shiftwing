#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DSV4_DIM 128
#define DSV4_VOCAB 3
#define DSV4_EXPERTS 8
#define DSV4_TOPK 2
#define DSV4_ATTN_HIDDEN 128
#define DSV4_ATTN_Q_RANK 128
#define DSV4_ATTN_ROPE_DIM 64
#define DSV4_INDEX_HEADS 2
#define DSV4_INDEX_DIM 128
#define DSV4_INDEX_TOPK 2
#define DSV4_INDEX_RATIO 4
#include "../deepseek_v4_indexer.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

typedef struct {
    const char *name, *dtype;
    int rank, rows, columns;
    size_t offset, bytes;
} record;

static int element_bytes(const char *dtype) {
    if (!strcmp(dtype, "BF16")) return 2;
    if (!strcmp(dtype, "F32")) return 4;
    return 1;
}

int main(void) {
    record records[] = {
        {"layers.2.attn.indexer.wq_b.weight", "F8_E4M3", 2, 256, 128, 0, 0},
        {"layers.2.attn.indexer.wq_b.scale", "F8_E8M0", 2, 2, 1, 0, 0},
        {"layers.2.attn.indexer.weights_proj.weight", "BF16", 2, 2, 128, 0, 0},
        {"layers.2.attn.indexer.compressor.ape", "F32", 2, 4, 256, 0, 0},
        {"layers.2.attn.indexer.compressor.norm.weight", "BF16", 1, 128, 0, 0, 0},
        {"layers.2.attn.indexer.compressor.wgate.weight", "BF16", 2, 256, 128, 0, 0},
        {"layers.2.attn.indexer.compressor.wkv.weight", "BF16", 2, 256, 128, 0, 0},
    };
    int count = (int)(sizeof(records) / sizeof(records[0]));
    size_t total = 0;
    for (int index = 0; index < count; index++) {
        records[index].offset = total;
        size_t elements = (size_t)records[index].rows *
            (records[index].rank == 2 ? records[index].columns : 1);
        records[index].bytes = elements * element_bytes(records[index].dtype);
        total += records[index].bytes;
    }
    unsigned char *payload = calloc(total, 1); CHECK(payload != NULL);
    for (int index = 0; index < count; index++) {
        unsigned char *value = payload + records[index].offset;
        if (!strcmp(records[index].dtype, "F8_E4M3"))
            memset(value, 0x38, records[index].bytes);
        else if (!strcmp(records[index].dtype, "F8_E8M0"))
            memset(value, 127, records[index].bytes);
        else if (!strcmp(records[index].dtype, "BF16"))
            for (size_t item = 0; item < records[index].bytes / 2; item++)
                ((uint16_t *)value)[item] = 0x3f80;
    }
    char root[] = "/tmp/colib-dsv4-indexer-XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    char segment_path[1024], manifest_path[1024];
    snprintf(segment_path, sizeof(segment_path), "%s/dense.bin", root);
    snprintf(manifest_path, sizeof(manifest_path), "%s/model-manifest.json",
             root);
    FILE *file = fopen(segment_path, "wb"); CHECK(file != NULL);
    CHECK(fwrite(payload, 1, total, file) == total && fclose(file) == 0);
    file = fopen(manifest_path, "wb"); CHECK(file != NULL);
    CHECK(fputs("{\"schema\":\"colib.deepseek-v4.model-manifest.v1\","
                "\"source\":{\"revision\":"
                "\"9e165c30e2704aec5d9d593cce3eebd58bbef1cb\"},"
                "\"inventory\":{\"dense.bin\":[", file) >= 0);
    for (int index = 0; index < count; index++) {
        record *item = &records[index];
        CHECK(fprintf(file,
            "{\"name\":\"%s\",\"file\":\"dense.bin\","
            "\"offset\":%zu,\"nbytes\":%zu,\"dtype\":\"%s\","
            "\"shape\":[%d", item->name, item->offset, item->bytes,
            item->dtype, item->rows) > 0);
        if (item->rank == 2) CHECK(fprintf(file, ",%d", item->columns) > 0);
        CHECK(fprintf(file, "]}%s", index + 1 < count ? "," : "") > 0);
    }
    CHECK(fputs("]}}", file) >= 0 && fclose(file) == 0);

    dsv4_store store; CHECK(dsv4_store_init(&store, root));
    dsv4_dense_arena dense; CHECK(dsv4_dense_arena_init(&dense, &store, 0, 0));
    float kv_cache[3 * DSV4_INDEX_DIM];
    float kv_state[2 * DSV4_INDEX_RATIO * 2 * DSV4_INDEX_DIM];
    float score_state[2 * DSV4_INDEX_RATIO * 2 * DSV4_INDEX_DIM];
    dsv4_indexer_state state;
    CHECK(dsv4_indexer_state_init(
        &state, 12, kv_cache, kv_state, score_state));
    float query[DSV4_INDEX_HEADS * DSV4_INDEX_DIM];
    float compressor_kv[2 * DSV4_INDEX_DIM];
    float compressor_score[2 * DSV4_INDEX_DIM];
    float head_weights[DSV4_INDEX_HEADS], scores[3];
    uint8_t activation[DSV4_ATTN_Q_RANK], activation_scale[1];
    dsv4_indexer_scratch scratch = {
        query, compressor_kv, compressor_score, head_weights, scores,
        activation, activation_scale,
    };
    float input[DSV4_ATTN_HIDDEN] = {0}; input[0] = 1.0f;
    float q_rank[DSV4_ATTN_Q_RANK] = {0}; q_rank[0] = 1.0f;
    int indices[DSV4_INDEX_TOPK], first_index = -1;
    for (int position = 0; position < 12; position++) {
        int selected = dsv4_indexer_decode(
            &dense, 2, input, q_rank, &state, &scratch, 7, indices);
        int visible = (position + 1) / DSV4_INDEX_RATIO;
        int expected = visible < DSV4_INDEX_TOPK ? visible : DSV4_INDEX_TOPK;
        CHECK(selected == expected);
        for (int item = 0; item < selected; item++)
            CHECK(indices[item] >= 7 && indices[item] < 7 + visible);
        if (position == 3) { CHECK(indices[0] == 7); first_index = indices[0]; }
        if (position == 7) CHECK(indices[0] != indices[1]);
    }
    CHECK(state.position == 12 && state.compressor.next_position == 12);
    CHECK(dsv4_indexer_state_init(&state, 12, kv_cache, kv_state,
                                   score_state));
    for (int position = 0; position < 4; position++)
        CHECK(dsv4_indexer_decode(
            &dense, 2, input, q_rank, &state, &scratch, 7, indices) ==
            (position == 3));
    CHECK(indices[0] == first_index);
    state.position++;
    CHECK(dsv4_indexer_decode(
        &dense, 2, input, q_rank, &state, &scratch, 7, indices) == -1);

    dsv4_dense_arena_close(&dense); dsv4_store_close(&store);
    free(payload); unlink(manifest_path); unlink(segment_path); rmdir(root);
    puts("DeepSeek-V4 learned compressed indexer tests: ok");
    return 0;
}
