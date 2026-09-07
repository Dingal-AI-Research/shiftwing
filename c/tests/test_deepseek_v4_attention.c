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
#include "../deepseek_v4_compressed_attention.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

typedef struct {
    const char *name;
    const char *dtype;
    int rank;
    int rows;
    int columns;
    size_t offset;
    size_t bytes;
} record;

static int dtype_bytes(const char *dtype) {
    if (!strcmp(dtype, "BF16")) return 2;
    if (!strcmp(dtype, "F32")) return 4;
    return 1;
}

int main(void) {
    record records[] = {
        {"layers.0.attn.wq_a.weight", "F8_E4M3", 2, 128, 128, 0, 0},
        {"layers.0.attn.wq_a.scale", "F8_E8M0", 2, 1, 1, 0, 0},
        {"layers.0.attn.wq_b.weight", "F8_E4M3", 2, 256, 128, 0, 0},
        {"layers.0.attn.wq_b.scale", "F8_E8M0", 2, 2, 1, 0, 0},
        {"layers.0.attn.wkv.weight", "F8_E4M3", 2, 128, 128, 0, 0},
        {"layers.0.attn.wkv.scale", "F8_E8M0", 2, 1, 1, 0, 0},
        {"layers.0.attn.wo_a.weight", "F8_E4M3", 2, 128, 256, 0, 0},
        {"layers.0.attn.wo_a.scale", "F8_E8M0", 2, 1, 2, 0, 0},
        {"layers.0.attn.wo_b.weight", "F8_E4M3", 2, 128, 128, 0, 0},
        {"layers.0.attn.wo_b.scale", "F8_E8M0", 2, 1, 1, 0, 0},
        {"layers.0.attn.q_norm.weight", "BF16", 1, 128, 0, 0, 0},
        {"layers.0.attn.kv_norm.weight", "BF16", 1, 128, 0, 0, 0},
        {"layers.0.attn.attn_sink", "F32", 1, 2, 0, 0, 0},
        {"layers.3.attn.wq_a.weight", "F8_E4M3", 2, 128, 128, 0, 0},
        {"layers.3.attn.wq_a.scale", "F8_E8M0", 2, 1, 1, 0, 0},
        {"layers.3.attn.wq_b.weight", "F8_E4M3", 2, 256, 128, 0, 0},
        {"layers.3.attn.wq_b.scale", "F8_E8M0", 2, 2, 1, 0, 0},
        {"layers.3.attn.wkv.weight", "F8_E4M3", 2, 128, 128, 0, 0},
        {"layers.3.attn.wkv.scale", "F8_E8M0", 2, 1, 1, 0, 0},
        {"layers.3.attn.wo_a.weight", "F8_E4M3", 2, 128, 256, 0, 0},
        {"layers.3.attn.wo_a.scale", "F8_E8M0", 2, 1, 2, 0, 0},
        {"layers.3.attn.wo_b.weight", "F8_E4M3", 2, 128, 128, 0, 0},
        {"layers.3.attn.wo_b.scale", "F8_E8M0", 2, 1, 1, 0, 0},
        {"layers.3.attn.q_norm.weight", "BF16", 1, 128, 0, 0, 0},
        {"layers.3.attn.kv_norm.weight", "BF16", 1, 128, 0, 0, 0},
        {"layers.3.attn.attn_sink", "F32", 1, 2, 0, 0, 0},
        {"layers.3.attn.compressor.ape", "F32", 2, 128, 128, 0, 0},
        {"layers.3.attn.compressor.norm.weight", "BF16", 1, 128, 0, 0, 0},
        {"layers.3.attn.compressor.wgate.weight", "BF16", 2, 128, 128, 0, 0},
        {"layers.3.attn.compressor.wkv.weight", "BF16", 2, 128, 128, 0, 0},
    };
    const int nrecords = (int)(sizeof(records) / sizeof(records[0]));
    size_t total = 0;
    for (int index = 0; index < nrecords; index++) {
        records[index].offset = total;
        size_t elements = (size_t)records[index].rows *
            (records[index].rank == 2 ? records[index].columns : 1);
        records[index].bytes = elements * dtype_bytes(records[index].dtype);
        total += records[index].bytes;
    }
    unsigned char *payload = calloc(total, 1); CHECK(payload != NULL);
    for (int index = 0; index < nrecords; index++) {
        unsigned char *value = payload + records[index].offset;
        if (!strcmp(records[index].dtype, "F8_E4M3"))
            memset(value, 0x38, records[index].bytes);
        else if (!strcmp(records[index].dtype, "F8_E8M0"))
            memset(value, 127, records[index].bytes);
        else if (!strcmp(records[index].dtype, "BF16"))
            for (size_t item = 0; item < records[index].bytes / 2; item++)
                ((uint16_t *)value)[item] = 0x3f80;
    }
    char root[] = "/tmp/colib-dsv4-attention-XXXXXX";
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
    for (int index = 0; index < nrecords; index++) {
        record *item = &records[index];
        CHECK(fprintf(file,
            "{\"name\":\"%s\",\"file\":\"dense.bin\","
            "\"offset\":%zu,\"nbytes\":%zu,\"dtype\":\"%s\","
            "\"shape\":[%d", item->name, item->offset, item->bytes,
            item->dtype, item->rows) > 0);
        if (item->rank == 2) CHECK(fprintf(file, ",%d", item->columns) > 0);
        CHECK(fprintf(file, "]}%s", index + 1 < nrecords ? "," : "") > 0);
    }
    CHECK(fputs("]}}", file) >= 0 && fclose(file) == 0);

    setenv("DIRECT", "1", 1);
    dsv4_store store; CHECK(dsv4_store_init(&store, root));
    dsv4_dense_arena dense;
    CHECK(dsv4_dense_arena_init(&dense, &store, 0, 0));
    CHECK(dense.records == nrecords && dense.payload_bytes == total);
    float kv_cache[DSV4_ATTN_WINDOW * DSV4_ATTN_HEAD_DIM];
    dsv4_sliding_attention_state state;
    CHECK(dsv4_sliding_attention_state_init(&state, kv_cache));
    float q_rank[DSV4_ATTN_Q_RANK];
    float query[DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM];
    float kv[DSV4_ATTN_HEAD_DIM];
    float context[DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM];
    float o_rank[DSV4_ATTN_O_GROUPS * DSV4_ATTN_O_RANK];
    float scores[DSV4_ATTN_WINDOW + 2];
    int indices[DSV4_ATTN_WINDOW + 2];
    uint8_t activation[DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM];
    uint8_t activation_scale[
        DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM / 128];
    dsv4_attention_scratch scratch = {
        q_rank, query, kv, context, o_rank, scores, indices,
        activation, activation_scale,
    };
    float input[DSV4_ATTN_HIDDEN] = {0}; input[0] = 1.0f;
    float output[DSV4_ATTN_HIDDEN], first[DSV4_ATTN_HIDDEN];
    for (int position = 0; position < 6; position++) {
        CHECK(dsv4_attention_decode_sliding(
            &dense, 0, input, &state, &scratch, output));
        for (int index = 0; index < DSV4_ATTN_HIDDEN; index++)
            CHECK(isfinite(output[index]));
        if (position == 0) {
            memcpy(first, output, sizeof(first));
            for (int index = 0; index < DSV4_ATTN_HIDDEN; index++)
                CHECK(fabsf(output[index] - 32768.0f) < 1e-2f);
        }
        if (position == 4)
            CHECK(indices[0] == 1 && indices[1] == 2 &&
                  indices[2] == 3 && indices[3] == 0);
    }
    CHECK(state.position == 6);
    CHECK(dsv4_sliding_attention_state_init(&state, kv_cache));
    CHECK(dsv4_attention_decode_sliding(
        &dense, 0, input, &state, &scratch, output));
    CHECK(!memcmp(first, output, sizeof(first)));

    float compressed_cache[(DSV4_ATTN_WINDOW + 2) * DSV4_ATTN_HEAD_DIM];
    float compressor_kv_state[DSV4_COMPRESS_RATIO * DSV4_ATTN_HEAD_DIM];
    float compressor_score_state[DSV4_COMPRESS_RATIO * DSV4_ATTN_HEAD_DIM];
    dsv4_compressed_attention_state compressed_state;
    CHECK(dsv4_compressed_attention_state_init(
        &compressed_state, 256, compressed_cache, compressor_kv_state,
        compressor_score_state));
    float compressor_kv[DSV4_ATTN_HEAD_DIM];
    float compressor_score[DSV4_ATTN_HEAD_DIM];
    dsv4_compressed_attention_scratch compressed_scratch = {
        scratch, compressor_kv, compressor_score,
    };
    float compressed_first[DSV4_ATTN_HIDDEN];
    for (int position = 0; position < 130; position++) {
        CHECK(dsv4_attention_decode_compressed_nonoverlap(
            &dense, 3, input, &compressed_state, &compressed_scratch,
            output));
        for (int index = 0; index < DSV4_ATTN_HIDDEN; index++)
            CHECK(isfinite(output[index]));
        if (position == 0) {
            memcpy(compressed_first, output, sizeof(compressed_first));
            for (int index = 0; index < DSV4_ATTN_HIDDEN; index++)
                CHECK(fabsf(output[index] - 32768.0f) < 1e-2f);
        }
        if (position == 127) {
            CHECK(indices[4] == DSV4_ATTN_WINDOW);
            for (int axis = 0; axis < DSV4_ATTN_HEAD_DIM; axis++)
                CHECK(isfinite(compressed_cache[
                    DSV4_ATTN_WINDOW * DSV4_ATTN_HEAD_DIM + axis]));
        }
    }
    CHECK(compressed_state.position == 130 &&
          compressed_state.compressor.next_position == 130);
    CHECK(dsv4_compressed_attention_state_init(
        &compressed_state, 256, compressed_cache, compressor_kv_state,
        compressor_score_state));
    CHECK(dsv4_attention_decode_compressed_nonoverlap(
        &dense, 3, input, &compressed_state, &compressed_scratch, output));
    CHECK(!memcmp(compressed_first, output, sizeof(compressed_first)));

    dsv4_dense_arena_close(&dense); dsv4_store_close(&store);
    free(payload); unlink(manifest_path); unlink(segment_path); rmdir(root);
    puts("DeepSeek-V4 sliding MLA attention tests: ok");
    return 0;
}
