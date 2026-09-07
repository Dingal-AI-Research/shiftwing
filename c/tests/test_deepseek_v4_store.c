#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DSV4_DIM 4
#define DSV4_VOCAB 3
#define DSV4_EXPERTS 8
#define DSV4_TOPK 2
#include "../deepseek_v4_model.h"
#include "../deepseek_v4_dense.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

int main(void) {
    char root[] = "/tmp/colib-dsv4-store-XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    char experts[1024], segment[1024], manifest[1024];
    snprintf(experts, sizeof(experts), "%s/experts", root);
    snprintf(segment, sizeof(segment), "%s/experts/layer-00.bin", root);
    snprintf(manifest, sizeof(manifest), "%s/model-manifest.json", root);
    CHECK(mkdir(experts, 0700) == 0);
    unsigned char source[12288], first[4096], second[4096];
    for (int i = 0; i < 12288; i++) source[i] = (unsigned char)(i * 17 + 3);
    uint16_t embedding[3][4] = {
        {0x3f80, 0x4000, 0x4040, 0x4080},
        {0x40a0, 0x40c0, 0x40e0, 0x4100},
        {0x4110, 0x4120, 0x4130, 0x4140},
    };
    int64_t routes[3][2] = {{0, 1}, {7, 3}, {2, 4}};
    uint16_t head[3][4] = {
        {0x3f80, 0, 0, 0},
        {0, 0x3f80, 0, 0},
        {0, 0, 0x3f80, 0x3f80},
    };
    memcpy(source + 8192, embedding, sizeof(embedding));
    memcpy(source + 8192 + sizeof(embedding), routes, sizeof(routes));
    memcpy(source + 8192 + sizeof(embedding) + sizeof(routes), head,
           sizeof(head));
    FILE *file = fopen(segment, "wb"); CHECK(file != NULL);
    CHECK(fwrite(source, 1, sizeof(source), file) == sizeof(source));
    CHECK(fclose(file) == 0);
    file = fopen(manifest, "wb"); CHECK(file != NULL);
    const char *prefix =
        "{\"schema\":\"colib.deepseek-v4.model-manifest.v1\","
        "\"source\":{\"revision\":\"9e165c30e2704aec5d9d593cce3eebd58bbef1cb\"},"
        "\"inventory\":{\"experts/layer-00.bin\":[";
    CHECK(fputs(prefix, file) >= 0);
    CHECK(fputs("{\"name\":\"layers.0.ffn.experts.0.w1.weight\","
                "\"file\":\"experts/layer-00.bin\",\"offset\":0,"
                "\"nbytes\":4096,\"dtype\":\"I8\",\"shape\":[64,64],"
                "\"layer\":0,\"expert\":0,\"projection\":\"w1\"},", file) >= 0);
    CHECK(fputs("{\"name\":\"layers.0.ffn.experts.0.w1.scale\","
                "\"file\":\"experts/layer-00.bin\",\"offset\":4096,"
                "\"nbytes\":4096,\"dtype\":\"F8_E8M0\",\"shape\":[64,64],"
                "\"layer\":0,\"expert\":0,\"projection\":\"w1\"},", file) >= 0);
    CHECK(fputs("{\"name\":\"embed.weight\","
                "\"file\":\"experts/layer-00.bin\",\"offset\":8192,"
                "\"nbytes\":24,\"dtype\":\"BF16\",\"shape\":[3,4]},", file) >= 0);
    CHECK(fputs("{\"name\":\"layers.1.ffn.gate.tid2eid\","
                "\"file\":\"experts/layer-00.bin\",\"offset\":8216,"
                "\"nbytes\":48,\"dtype\":\"I64\",\"shape\":[3,2],"
                "\"layer\":1},", file) >= 0);
    CHECK(fputs("{\"name\":\"head.weight\","
                "\"file\":\"experts/layer-00.bin\",\"offset\":8264,"
                "\"nbytes\":24,\"dtype\":\"BF16\",\"shape\":[3,4]}]}}", file) >= 0);
    CHECK(fclose(file) == 0);

    dsv4_store store;
    CHECK(dsv4_store_init(&store, root));
    CHECK(store.records == 5 && store.segments == 1);
    CHECK(st_has(&store.raw, "layers.0.ffn.experts.0.w1.weight"));
    CHECK(st_nbytes(&store.raw, "layers.0.ffn.experts.0.w1.scale") == 4096);
    const dsv4_tensor_desc *weight = dsv4_store_find(
        &store, "layers.0.ffn.experts.0.w1.weight");
    const dsv4_tensor_desc *scale = dsv4_store_find(
        &store, "layers.0.ffn.experts.0.w1.scale");
    CHECK(weight && weight->dtype == DSV4_DTYPE_I8 && weight->rank == 2);
    CHECK(weight->shape[0] == 64 && weight->shape[1] == 64);
    CHECK(weight->layer == 0 && weight->expert == 0 &&
          !strcmp(weight->projection, "w1"));
    CHECK(scale && scale->dtype == DSV4_DTYPE_UE8M0 && scale->rank == 2);
    setenv("DIRECT", "1", 1); setenv("URING", "1", 1); setenv("URING_PERSIST", "1", 1);
    st_read_raw(&store.raw, "layers.0.ffn.experts.0.w1.weight", first, 0);
    const char *names[2] = {"layers.0.ffn.experts.0.w1.weight",
                            "layers.0.ffn.experts.0.w1.scale"};
    void *outputs[2] = {first, second};
    st_read_raw_batch(&store.raw, names, outputs, 2, 1);
    CHECK(!memcmp(first, source, 4096));
    CHECK(!memcmp(second, source + 4096, 4096));
    unsigned char slice[73];
    CHECK(dsv4_store_read_slice(
        &store, "layers.0.ffn.experts.0.w1.weight", 17, sizeof(slice),
        slice, 1));
    CHECK(!memcmp(slice, source + 17, sizeof(slice)));
    CHECK(!dsv4_store_read_slice(
        &store, "layers.0.ffn.experts.0.w1.weight", 4090, 7, slice, 0));
    uint16_t embedding_row[4];
    CHECK(dsv4_model_read_bf16_row(&store, "embed.weight", 1, 3, 4,
                                    embedding_row, 1));
    CHECK(!memcmp(embedding_row, embedding[1], sizeof(embedding_row)));
    CHECK(!dsv4_model_read_bf16_row(&store, "embed.weight", 3, 3, 4,
                                     embedding_row, 0));
    float hc_embedding[DSV4_HC_MULT * DSV4_DIM];
    CHECK(dsv4_model_embed_token(&store, 1, hc_embedding, 1));
    for (int copy = 0; copy < DSV4_HC_MULT; copy++)
        for (int column = 0; column < DSV4_DIM; column++)
            CHECK(hc_embedding[copy * DSV4_DIM + column] ==
                  (float)(5 + column));
    int route[DSV4_TOPK];
    CHECK(dsv4_model_hash_route(&store, 1, 1, route, 1));
    CHECK(route[0] == 7 && route[1] == 3);
    CHECK(!dsv4_model_hash_route(&store, 3, 1, route, 0));
    float hidden[4] = {1, 2, 3, 4}, logits[3];
    CHECK(dsv4_model_head_logits(&store, hidden, logits, 2, 1));
    CHECK(logits[0] == 1 && logits[1] == 2 && logits[2] == 7);
    CHECK(store.raw.read_bytes >= 3 * 4096 + sizeof(slice) +
          2 * sizeof(embedding_row) + sizeof(routes[0]) + sizeof(head));
    uint64_t dense_read_start = store.raw.read_bytes;
    dsv4_dense_arena dense;
    CHECK(dsv4_dense_arena_init(&dense, &store, 0, 0));
    CHECK(dense.records == 3 && dense.payload_bytes ==
          sizeof(embedding) + sizeof(routes) + sizeof(head));
    const dsv4_tensor_desc *dense_descriptor = NULL;
    const uint16_t *dense_embedding = dsv4_dense_find(
        &dense, "embed.weight", &dense_descriptor);
    CHECK(dense_embedding && dense_descriptor &&
          dense_descriptor->dtype == DSV4_DTYPE_BF16);
    CHECK((uintptr_t)dense_embedding % DSV4_DENSE_ALIGNMENT == 0);
    CHECK(!memcmp(dense_embedding, embedding, sizeof(embedding)));
    CHECK(dsv4_dense_find(&dense, "layers.0.ffn.experts.0.w1.weight",
                          NULL) == NULL);
    CHECK(store.raw.read_bytes - dense_read_start == dense.payload_bytes);
    dsv4_dense_arena_close(&dense);
    CHECK(!dsv4_dense_arena_init(&dense, &store, 0, 1));
    CHECK(strstr(dense.error, "dense contract expected") != NULL);
    dsv4_dense_arena_close(&dense);
    dsv4_store_close(&store);

    unlink(manifest); unlink(segment); rmdir(experts); rmdir(root);
    puts("DeepSeek-V4 manifest store tests: ok");
    return 0;
}
