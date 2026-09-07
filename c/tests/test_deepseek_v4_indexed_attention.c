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
#include "../deepseek_v4_indexed_attention.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

typedef struct {
    const char *name, *dtype;
    int rank, rows, columns;
    size_t offset, bytes;
} record;

static int dtype_bytes(const char *dtype) {
    if (!strcmp(dtype, "BF16")) return 2;
    if (!strcmp(dtype, "F32")) return 4;
    return 1;
}

int main(void) {
    record records[] = {
        {"layers.2.attn.wq_a.weight", "F8_E4M3", 2, 128, 128, 0, 0},
        {"layers.2.attn.wq_a.scale", "F8_E8M0", 2, 1, 1, 0, 0},
        {"layers.2.attn.wq_b.weight", "F8_E4M3", 2, 256, 128, 0, 0},
        {"layers.2.attn.wq_b.scale", "F8_E8M0", 2, 2, 1, 0, 0},
        {"layers.2.attn.wkv.weight", "F8_E4M3", 2, 128, 128, 0, 0},
        {"layers.2.attn.wkv.scale", "F8_E8M0", 2, 1, 1, 0, 0},
        {"layers.2.attn.wo_a.weight", "F8_E4M3", 2, 128, 256, 0, 0},
        {"layers.2.attn.wo_a.scale", "F8_E8M0", 2, 1, 2, 0, 0},
        {"layers.2.attn.wo_b.weight", "F8_E4M3", 2, 128, 128, 0, 0},
        {"layers.2.attn.wo_b.scale", "F8_E8M0", 2, 1, 1, 0, 0},
        {"layers.2.attn.q_norm.weight", "BF16", 1, 128, 0, 0, 0},
        {"layers.2.attn.kv_norm.weight", "BF16", 1, 128, 0, 0, 0},
        {"layers.2.attn.attn_sink", "F32", 1, 2, 0, 0, 0},
        {"layers.2.attn.compressor.ape", "F32", 2, 4, 256, 0, 0},
        {"layers.2.attn.compressor.norm.weight", "BF16", 1, 128, 0, 0, 0},
        {"layers.2.attn.compressor.wgate.weight", "BF16", 2, 256, 128, 0, 0},
        {"layers.2.attn.compressor.wkv.weight", "BF16", 2, 256, 128, 0, 0},
        {"layers.2.attn.indexer.wq_b.weight", "F8_E4M3", 2, 256, 128, 0, 0},
        {"layers.2.attn.indexer.wq_b.scale", "F8_E8M0", 2, 2, 1, 0, 0},
        {"layers.2.attn.indexer.weights_proj.weight", "BF16", 2, 2, 128, 0, 0},
        {"layers.2.attn.indexer.compressor.ape", "F32", 2, 4, 256, 0, 0},
        {"layers.2.attn.indexer.compressor.norm.weight", "BF16", 1, 128, 0, 0, 0},
        {"layers.2.attn.indexer.compressor.wgate.weight", "BF16", 2, 256, 128, 0, 0},
        {"layers.2.attn.indexer.compressor.wkv.weight", "BF16", 2, 256, 128, 0, 0},
    };
    const int count = (int)(sizeof(records) / sizeof(records[0]));
    size_t total = 0;
    for (int index = 0; index < count; index++) {
        records[index].offset = total;
        size_t elements = (size_t)records[index].rows *
            (records[index].rank == 2 ? records[index].columns : 1);
        records[index].bytes = elements * dtype_bytes(records[index].dtype);
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
    char root[] = "/tmp/colib-dsv4-indexed-attention-XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    char segment[1024], manifest[1024];
    snprintf(segment, sizeof(segment), "%s/dense.bin", root);
    snprintf(manifest, sizeof(manifest), "%s/model-manifest.json", root);
    FILE *file = fopen(segment, "wb"); CHECK(file != NULL);
    CHECK(fwrite(payload, 1, total, file) == total && fclose(file) == 0);
    file = fopen(manifest, "wb"); CHECK(file != NULL);
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
    float kv_cache[(DSV4_ATTN_WINDOW + 3) * DSV4_ATTN_HEAD_DIM];
    float compressor_kv_state[2 * DSV4_INDEX_RATIO *
                              2 * DSV4_ATTN_HEAD_DIM];
    float compressor_score_state[2 * DSV4_INDEX_RATIO *
                                 2 * DSV4_ATTN_HEAD_DIM];
    float index_kv_cache[3 * DSV4_INDEX_DIM];
    float index_kv_state[2 * DSV4_INDEX_RATIO * 2 * DSV4_INDEX_DIM];
    float index_score_state[2 * DSV4_INDEX_RATIO * 2 * DSV4_INDEX_DIM];
    dsv4_indexed_attention_state state;
    CHECK(dsv4_indexed_attention_state_init(
        &state, 12, kv_cache, compressor_kv_state,
        compressor_score_state, index_kv_cache, index_kv_state,
        index_score_state));

    float q_rank[DSV4_ATTN_Q_RANK];
    float query[DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM];
    float kv[DSV4_ATTN_HEAD_DIM];
    float context[DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM];
    float o_rank[DSV4_ATTN_O_GROUPS * DSV4_ATTN_O_RANK];
    float scores[DSV4_ATTN_WINDOW + DSV4_INDEX_TOPK];
    int indices[DSV4_ATTN_WINDOW + DSV4_INDEX_TOPK];
    uint8_t activation[DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM];
    uint8_t activation_scale[2];
    float compressor_kv[2 * DSV4_ATTN_HEAD_DIM];
    float compressor_score[2 * DSV4_ATTN_HEAD_DIM];
    float index_query[DSV4_INDEX_HEADS * DSV4_INDEX_DIM];
    float index_compressor_kv[2 * DSV4_INDEX_DIM];
    float index_compressor_score[2 * DSV4_INDEX_DIM];
    float head_weights[DSV4_INDEX_HEADS], index_scores[3];
    uint8_t index_activation[DSV4_ATTN_Q_RANK], index_scale[1];
    dsv4_indexed_attention_scratch scratch = {
        {q_rank, query, kv, context, o_rank, scores, indices,
         activation, activation_scale},
        compressor_kv, compressor_score,
        {index_query, index_compressor_kv, index_compressor_score,
         head_weights, index_scores, index_activation, index_scale},
    };
    float input[DSV4_ATTN_HIDDEN] = {0}; input[0] = 1.0f;
    float output[DSV4_ATTN_HIDDEN], first[DSV4_ATTN_HIDDEN];
    for (int position = 0; position < 10; position++) {
        CHECK(dsv4_attention_decode_compressed_overlap(
            &dense, 2, input, &state, &scratch, output));
        for (int axis = 0; axis < DSV4_ATTN_HIDDEN; axis++)
            CHECK(isfinite(output[axis]));
        if (position == 0) memcpy(first, output, sizeof(first));
        if (position == 3) CHECK(indices[DSV4_ATTN_WINDOW] == DSV4_ATTN_WINDOW);
        if (position == 7)
            CHECK(indices[DSV4_ATTN_WINDOW] !=
                  indices[DSV4_ATTN_WINDOW + 1]);
    }
    CHECK(state.position == 10 && state.compressor.next_position == 10 &&
          state.indexer.position == 10 &&
          state.indexer.compressor.next_position == 10);
    CHECK(dsv4_indexed_attention_state_init(
        &state, 12, kv_cache, compressor_kv_state,
        compressor_score_state, index_kv_cache, index_kv_state,
        index_score_state));
    CHECK(dsv4_attention_decode_compressed_overlap(
        &dense, 2, input, &state, &scratch, output));
    CHECK(!memcmp(first, output, sizeof(first)));

    dsv4_dense_arena_close(&dense); dsv4_store_close(&store);
    free(payload); unlink(manifest); unlink(segment); rmdir(root);
    puts("DeepSeek-V4 ratio-4 indexed attention tests: ok");
    return 0;
}
