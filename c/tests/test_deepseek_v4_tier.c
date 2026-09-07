#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DSV4_DIM 4
#define DSV4_VOCAB 3
#define DSV4_EXPERTS 8
#define DSV4_TOPK 2
#define DSV4_EXPERT_HIDDEN 128
#define DSV4_MOE_INTERMEDIATE 128
#define DSV4_BASE_LAYERS 1
#define DSV4_DSPARK_LAYERS 0
#include "../deepseek_v4_tier.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

static int write_record(FILE *file, int expert, int record, int comma) {
    const char *suffix[6] = {"w1.weight", "w1.scale", "w2.weight",
                             "w2.scale", "w3.weight", "w3.scale"};
    const char *dtype[6] = {"I8", "F8_E8M0", "I8", "F8_E8M0",
                            "I8", "F8_E8M0"};
    const int offset[6] = {0, 8192, 8704, 16896, 17408, 25600};
    const int bytes[6] = {8192, 512, 8192, 512, 8192, 512};
    const int rows[6] = {128, 128, 128, 128, 128, 128};
    const int columns[6] = {64, 4, 64, 4, 64, 4};
    return fprintf(file,
        "{\"name\":\"layers.0.ffn.experts.%d.%s\","
        "\"file\":\"experts.bin\",\"offset\":%d,\"nbytes\":%d,"
        "\"dtype\":\"%s\",\"shape\":[%d,%d],\"layer\":0,"
        "\"expert\":%d}%s", expert, suffix[record], offset[record],
        bytes[record], dtype[record], rows[record], columns[record], expert,
        comma ? "," : "") > 0;
}

static int write_dense_record(FILE *file, const char *name, int offset,
                              int bytes, const char *dtype, int rows,
                              int columns, int comma) {
    return fprintf(file,
        "{\"name\":\"%s\",\"file\":\"dense.bin\","
        "\"offset\":%d,\"nbytes\":%d,\"dtype\":\"%s\","
        "\"shape\":[%d,%d]}%s", name, offset, bytes, dtype, rows,
        columns, comma ? "," : "") > 0;
}

int main(void) {
    char root[] = "/tmp/colib-dsv4-tier-XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    char segment_path[1024], dense_path[1024], manifest_path[1024];
    snprintf(segment_path, sizeof(segment_path), "%s/experts.bin", root);
    snprintf(dense_path, sizeof(dense_path), "%s/dense.bin", root);
    snprintf(manifest_path, sizeof(manifest_path), "%s/model-manifest.json",
             root);
    unsigned char source[26112];
    memset(source, 0x22, sizeof(source));
    memset(source + 8192, 127, 512);
    memset(source + 16896, 127, 512);
    memset(source + 25600, 127, 512);
    FILE *file = fopen(segment_path, "wb"); CHECK(file != NULL);
    CHECK(fwrite(source, 1, sizeof(source), file) == sizeof(source));
    CHECK(fclose(file) == 0);
    unsigned char dense_source[51251] = {0};
    uint16_t *router = (uint16_t *)dense_source;
    for (int expert = 0; expert < DSV4_EXPERTS; expert++)
        router[expert * DSV4_EXPERT_HIDDEN] = 0x3f80;
    int64_t hash_routes[3][2] = {{0, 1}, {2, 3}, {4, 5}};
    memcpy(dense_source + 2048, hash_routes, sizeof(hash_routes));
    memset(dense_source + 2096, 0x38, 16384);
    dense_source[18480] = 127;
    memset(dense_source + 18481, 0x38, 16384);
    dense_source[34865] = 127;
    memset(dense_source + 34866, 0x38, 16384);
    dense_source[51250] = 127;
    file = fopen(dense_path, "wb"); CHECK(file != NULL);
    CHECK(fwrite(dense_source, 1, sizeof(dense_source), file) ==
          sizeof(dense_source));
    CHECK(fclose(file) == 0);
    file = fopen(manifest_path, "wb"); CHECK(file != NULL);
    CHECK(fputs("{\"schema\":\"colib.deepseek-v4.model-manifest.v1\","
                "\"source\":{\"revision\":"
                "\"9e165c30e2704aec5d9d593cce3eebd58bbef1cb\"},"
                "\"inventory\":{\"experts.bin\":[", file) >= 0);
    for (int expert = 0; expert < DSV4_EXPERTS; expert++)
        for (int record = 0; record < 6; record++)
            CHECK(write_record(file, expert, record,
                               expert + 1 < DSV4_EXPERTS || record < 5));
    CHECK(fputs("],\"dense.bin\":[", file) >= 0);
    const char *dense_name[8] = {
        "layers.0.ffn.gate.weight", "layers.0.ffn.gate.tid2eid",
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.0.ffn.shared_experts.w1.scale",
        "layers.0.ffn.shared_experts.w2.weight",
        "layers.0.ffn.shared_experts.w2.scale",
        "layers.0.ffn.shared_experts.w3.weight",
        "layers.0.ffn.shared_experts.w3.scale",
    };
    const int dense_offset[8] = {0, 2048, 2096, 18480, 18481, 34865,
                                 34866, 51250};
    const int dense_bytes[8] = {2048, 48, 16384, 1, 16384, 1, 16384, 1};
    const char *dense_dtype[8] = {"BF16", "I64", "F8_E4M3", "F8_E8M0",
                                  "F8_E4M3", "F8_E8M0", "F8_E4M3",
                                  "F8_E8M0"};
    const int dense_rows[8] = {8, 3, 128, 1, 128, 1, 128, 1};
    const int dense_columns[8] = {128, 2, 128, 1, 128, 1, 128, 1};
    for (int record = 0; record < 8; record++)
        CHECK(write_dense_record(file, dense_name[record],
                                 dense_offset[record], dense_bytes[record],
                                 dense_dtype[record], dense_rows[record],
                                 dense_columns[record], record < 7));
    CHECK(fputs("]}}", file) >= 0 && fclose(file) == 0);

    setenv("DIRECT", "1", 1); setenv("URING", "1", 1);
    setenv("URING_PERSIST", "1", 1);
    dsv4_store store; CHECK(dsv4_store_init(&store, root));
    CHECK(store.records == 56);
    dsv4_expert_cache cache;
    CHECK(dsv4_expert_cache_init(&cache, &store, 2));
    CHECK(dsv4_expert_payload_bytes() == sizeof(source));
    int prefetch_ids[2] = {0, 1};
    CHECK(dsv4_expert_cache_prefetch(&cache, 0, prefetch_ids, 2));

    dsv4_expert_entry *zero = dsv4_expert_cache_acquire(&cache, 0, 0);
    CHECK(zero && cache.misses == 1 && cache.hits == 0);
    uint64_t first_read = store.raw.read_bytes;
    CHECK(first_read == sizeof(source));
    float input[128] = {0}, gate[128], up[128], output[128];
    uint8_t activation[128], activation_scale[1]; input[0] = 1.0f;
    CHECK(dsv4_expert_fp4(output, input, zero->w1, zero->s1, zero->w2,
                          zero->s2, zero->w3, zero->s3, 128, 128, 0.5f,
                          gate, up, activation, activation_scale));
    for (int index = 0; index < 128; index++)
        CHECK(fabsf(output[index] - 48.0f) < 1e-4f);
    CHECK(dsv4_expert_cache_release(&cache, zero));
    zero = dsv4_expert_cache_acquire(&cache, 0, 0);
    CHECK(zero && cache.hits == 1 && store.raw.read_bytes == first_read);
    CHECK(dsv4_expert_cache_release(&cache, zero));
    zero = dsv4_expert_cache_acquire(&cache, 0, 0);
    CHECK(zero && dsv4_expert_cache_release(&cache, zero));
    dsv4_expert_entry *one = dsv4_expert_cache_acquire(&cache, 0, 1);
    CHECK(one && dsv4_expert_cache_release(&cache, one));
    dsv4_expert_entry *two = dsv4_expert_cache_acquire(&cache, 0, 2);
    CHECK(two && cache.evictions == 1 && dsv4_expert_cache_release(&cache, two));
    zero = dsv4_expert_cache_acquire(&cache, 0, 0);
    CHECK(zero && cache.hits == 3 && dsv4_expert_cache_release(&cache, zero));

    zero = dsv4_expert_cache_acquire(&cache, 0, 0);
    two = dsv4_expert_cache_acquire(&cache, 0, 2);
    CHECK(zero && two);
    CHECK(dsv4_expert_cache_acquire(&cache, 0, 3) == NULL);
    CHECK(strstr(cache.error, "all expert slots are active") != NULL);
    CHECK(dsv4_expert_cache_release(&cache, zero));
    CHECK(dsv4_expert_cache_release(&cache, two));

    dsv4_expert_cache_close(&cache);

    CHECK(dsv4_expert_cache_init(&cache, &store, 2));
    uint64_t grouped_read_start = store.raw.read_bytes;
    uint64_t grouped_batch_start = store.raw.uring_batches;
    uint64_t grouped_fallback_start = store.raw.uring_fallbacks;
    dsv4_expert_entry *grouped[2];
    CHECK(dsv4_expert_cache_acquire_many(
        &cache, 0, prefetch_ids, 2, grouped));
    CHECK(grouped[0] && grouped[1] && cache.misses == 2 && cache.hits == 0);
    CHECK(store.raw.read_bytes - grouped_read_start == 2 * sizeof(source));
    CHECK((store.raw.uring_batches - grouped_batch_start) +
          (store.raw.uring_fallbacks - grouped_fallback_start) == 1);
    CHECK(dsv4_expert_cache_release(&cache, grouped[0]));
    CHECK(dsv4_expert_cache_release(&cache, grouped[1]));
    uint64_t grouped_hit_read = store.raw.read_bytes;
    grouped_batch_start = store.raw.uring_batches;
    CHECK(dsv4_expert_cache_acquire_many(
        &cache, 0, prefetch_ids, 2, grouped));
    CHECK(cache.hits == 2 && store.raw.read_bytes == grouped_hit_read &&
          store.raw.uring_batches == grouped_batch_start);
    CHECK(dsv4_expert_cache_release(&cache, grouped[0]));
    CHECK(dsv4_expert_cache_release(&cache, grouped[1]));
    float route_weights[2] = {0.25f, 0.75f};
    float routed_temp[128];
    CHECK(dsv4_routed_experts_forward(
        &cache, 0, prefetch_ids, route_weights, input, output, routed_temp,
        gate, up, activation, activation_scale));
    for (int index = 0; index < 128; index++)
        CHECK(fabsf(output[index] - 96.0f) < 1e-4f);
    CHECK(cache.hits == 4 && store.raw.read_bytes == grouped_hit_read);
    dsv4_dense_arena dense;
    CHECK(dsv4_dense_arena_init(&dense, &store, 0, 0));
    CHECK(dense.records == 8 && dense.payload_bytes == sizeof(dense_source));
    float full_output[128], router_logits[8], routed[128], shared[128];
    CHECK(dsv4_moe_forward(
        &cache, &dense, 0, 0, input, full_output, router_logits, routed,
        shared, routed_temp, gate, up, activation, activation_scale));
    for (int index = 0; index < 128; index++)
        CHECK(fabsf(full_output[index] - 240.0f) < 1e-4f);
    CHECK(cache.hits == 6);
    dsv4_dense_arena_close(&dense);
    int duplicate_ids[2] = {0, 0};
    CHECK(!dsv4_expert_cache_acquire_many(
        &cache, 0, duplicate_ids, 2, grouped));
    CHECK(strstr(cache.error, "must be unique") != NULL);
    dsv4_expert_cache_close(&cache);
    dsv4_store_close(&store);
    unlink(manifest_path); unlink(dense_path); unlink(segment_path); rmdir(root);
    puts("DeepSeek-V4 native expert tier tests: ok");
    return 0;
}
