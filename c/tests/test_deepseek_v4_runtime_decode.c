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
#define DSV4_RUNTIME_LAYERS 1
#define DSV4_BASE_LAYERS 1
#define DSV4_DSPARK_LAYERS 0
#include "../deepseek_v4_runtime.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

typedef struct {
    char name[192];
    const char *dtype;
    int rank, rows, columns;
    size_t offset, bytes;
} record;

typedef struct {
    record items[96];
    int count;
    size_t bytes;
} layout;

static int dtype_bytes(const char *dtype) {
    if (!strcmp(dtype, "BF16")) return 2;
    if (!strcmp(dtype, "F32")) return 4;
    if (!strcmp(dtype, "I64")) return 8;
    return 1;
}

static int add_record(layout *plan, const char *name, const char *dtype,
                      int rank, int rows, int columns) {
    if (!plan || plan->count >= (int)(sizeof(plan->items) /
                                      sizeof(plan->items[0]))) return 0;
    record *item = &plan->items[plan->count++];
    if (snprintf(item->name, sizeof(item->name), "%s", name) >=
        (int)sizeof(item->name)) return 0;
    item->dtype = dtype; item->rank = rank;
    item->rows = rows; item->columns = columns;
    item->offset = plan->bytes;
    item->bytes = (size_t)rows * (rank == 2 ? columns : 1) *
                  dtype_bytes(dtype);
    plan->bytes += item->bytes;
    return 1;
}

static int add1(layout *plan, const char *name, const char *dtype, int rows) {
    return add_record(plan, name, dtype, 1, rows, 0);
}

static int add2(layout *plan, const char *name, const char *dtype,
                int rows, int columns) {
    return add_record(plan, name, dtype, 2, rows, columns);
}

static int add_fp8(layout *plan, const char *prefix, int rows, int columns) {
    char name[192];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (!add2(plan, name, "F8_E4M3", rows, columns)) return 0;
    snprintf(name, sizeof(name), "%s.scale", prefix);
    return add2(plan, name, "F8_E8M0", (rows + 127) / 128,
                (columns + 127) / 128);
}

static int add_expert(layout *plan, int expert) {
    char prefix[160], name[192];
    snprintf(prefix, sizeof(prefix), "layers.0.ffn.experts.%d", expert);
    const char *projection[3] = {"w1", "w2", "w3"};
    for (int index = 0; index < 3; index++) {
        snprintf(name, sizeof(name), "%s.%s.weight", prefix,
                 projection[index]);
        if (!add2(plan, name, "I8", 128, 64)) return 0;
        snprintf(name, sizeof(name), "%s.%s.scale", prefix,
                 projection[index]);
        if (!add2(plan, name, "F8_E8M0", 128, 4)) return 0;
    }
    return 1;
}

static int build_layout(layout *plan) {
    memset(plan, 0, sizeof(*plan));
    if (!add2(plan, "embed.weight", "BF16", 3, 128) ||
        !add2(plan, "head.weight", "BF16", 3, 128) ||
        !add1(plan, "norm.weight", "BF16", 128) ||
        !add1(plan, "hc_head_base", "F32", 4) ||
        !add2(plan, "hc_head_fn", "F32", 4, 512) ||
        !add1(plan, "hc_head_scale", "F32", 1) ||
        !add_fp8(plan, "layers.0.attn.wq_a", 128, 128) ||
        !add_fp8(plan, "layers.0.attn.wq_b", 256, 128) ||
        !add_fp8(plan, "layers.0.attn.wkv", 128, 128) ||
        !add_fp8(plan, "layers.0.attn.wo_a", 128, 256) ||
        !add_fp8(plan, "layers.0.attn.wo_b", 128, 128) ||
        !add1(plan, "layers.0.attn.q_norm.weight", "BF16", 128) ||
        !add1(plan, "layers.0.attn.kv_norm.weight", "BF16", 128) ||
        !add1(plan, "layers.0.attn.attn_sink", "F32", 2) ||
        !add1(plan, "layers.0.attn_norm.weight", "BF16", 128) ||
        !add1(plan, "layers.0.ffn_norm.weight", "BF16", 128) ||
        !add1(plan, "layers.0.hc_attn_base", "F32", 24) ||
        !add2(plan, "layers.0.hc_attn_fn", "F32", 24, 512) ||
        !add1(plan, "layers.0.hc_attn_scale", "F32", 3) ||
        !add1(plan, "layers.0.hc_ffn_base", "F32", 24) ||
        !add2(plan, "layers.0.hc_ffn_fn", "F32", 24, 512) ||
        !add1(plan, "layers.0.hc_ffn_scale", "F32", 3) ||
        !add2(plan, "layers.0.ffn.gate.weight", "BF16", 8, 128) ||
        !add2(plan, "layers.0.ffn.gate.tid2eid", "I64", 3, 2) ||
        !add_fp8(plan, "layers.0.ffn.shared_experts.w1", 128, 128) ||
        !add_fp8(plan, "layers.0.ffn.shared_experts.w2", 128, 128) ||
        !add_fp8(plan, "layers.0.ffn.shared_experts.w3", 128, 128))
        return 0;
    for (int expert = 0; expert < DSV4_EXPERTS; expert++)
        if (!add_expert(plan, expert)) return 0;
    return 1;
}

static int write_fixture(const char *root, const layout *plan) {
    char segment[1024], manifest[1024];
    snprintf(segment, sizeof(segment), "%s/model.bin", root);
    snprintf(manifest, sizeof(manifest), "%s/model-manifest.json", root);
    unsigned char *payload = (unsigned char *)calloc(plan->bytes, 1);
    if (!payload) return 0;
    for (int index = 0; index < plan->count; index++) {
        const record *item = &plan->items[index];
        unsigned char *value = payload + item->offset;
        if (!strcmp(item->dtype, "BF16")) {
            for (size_t element = 0; element < item->bytes / 2; element++)
                ((uint16_t *)value)[element] = 0x3f80;
        } else if (!strcmp(item->dtype, "F8_E4M3")) {
            memset(value, 0x38, item->bytes);
        } else if (!strcmp(item->dtype, "F8_E8M0")) {
            memset(value, 127, item->bytes);
        } else if (!strcmp(item->dtype, "I8")) {
            memset(value, 0x22, item->bytes);
        } else if (!strcmp(item->dtype, "I64")) {
            for (int row = 0; row < item->rows; row++) {
                ((int64_t *)value)[row * 2] = 0;
                ((int64_t *)value)[row * 2 + 1] = 1;
            }
        }
    }
    FILE *file = fopen(segment, "wb");
    int ok = file && fwrite(payload, 1, plan->bytes, file) == plan->bytes &&
             fclose(file) == 0;
    free(payload);
    if (!ok) return 0;
    file = fopen(manifest, "wb");
    if (!file || fputs(
        "{\"schema\":\"colib.deepseek-v4.model-manifest.v1\","
        "\"source\":{\"revision\":"
        "\"9e165c30e2704aec5d9d593cce3eebd58bbef1cb\"},"
        "\"inventory\":{\"model.bin\":[", file) < 0) return 0;
    for (int index = 0; index < plan->count; index++) {
        const record *item = &plan->items[index];
        if (fprintf(file,
            "{\"name\":\"%s\",\"file\":\"model.bin\","
            "\"offset\":%zu,\"nbytes\":%zu,\"dtype\":\"%s\","
            "\"shape\":[%d", item->name, item->offset, item->bytes,
            item->dtype, item->rows) < 0) return 0;
        if (item->rank == 2 && fprintf(file, ",%d", item->columns) < 0)
            return 0;
        if (fprintf(file, "]}%s", index + 1 < plan->count ? "," : "") < 0)
            return 0;
    }
    return fputs("]}}", file) >= 0 && fclose(file) == 0;
}

int main(void) {
    layout plan; CHECK(build_layout(&plan));
    char root[] = "/tmp/colib-dsv4-runtime-decode-XXXXXX";
    CHECK(mkdtemp(root) != NULL && write_fixture(root, &plan));
    dsv4_store store; CHECK(dsv4_store_init(&store, root));
    CHECK(store.records == plan.count);
    dsv4_dense_arena dense;
    CHECK(dsv4_dense_arena_init(&dense, &store, 0, 0));
    dsv4_expert_cache experts;
    CHECK(dsv4_expert_cache_init(&experts, &store, DSV4_TOPK));
    uint64_t decode_read_bytes =
        __atomic_load_n(&store.raw.read_bytes, __ATOMIC_RELAXED);
    uint64_t expected_expert_bytes =
        (uint64_t)DSV4_TOPK * dsv4_expert_payload_bytes();
#ifdef COLI_CUDA
    ColiCuda *cuda = NULL;
    CHECK(!coli_cuda_create(&cuda, 0));
#endif
    dsv4_runtime runtime;
    CHECK(dsv4_runtime_init(&runtime, &store, &dense, &experts, 2));
    float logits[DSV4_VOCAB], first[DSV4_VOCAB];
    CHECK(dsv4_runtime_decode_token(&runtime, 0, logits, 2, 0));
    memcpy(first, logits, sizeof(first));
    CHECK(runtime.position == 1 && runtime.history[0] == 0 &&
          runtime.layers[0].attention.sliding.position == 1);
    CHECK(experts.misses == 2 && !runtime.poisoned);
    CHECK(__atomic_load_n(&store.raw.read_bytes, __ATOMIC_RELAXED) ==
          decode_read_bytes + expected_expert_bytes);
    for (int item = 0; item < DSV4_VOCAB; item++)
        CHECK(isfinite(logits[item]) && fabsf(logits[item] - logits[0]) < 1e-5f);
    CHECK(dsv4_runtime_decode_token(&runtime, 1, logits, 2, 0));
    CHECK(runtime.position == 2 && runtime.history[1] == 1 &&
          runtime.layers[0].attention.sliding.position == 2);
    CHECK(experts.hits == 2 && experts.misses == 2);
    CHECK(__atomic_load_n(&store.raw.read_bytes, __ATOMIC_RELAXED) ==
          decode_read_bytes + expected_expert_bytes);
    CHECK(!dsv4_runtime_decode_token(&runtime, 2, logits, 2, 0));
    CHECK(!runtime.poisoned && runtime.position == 2);
    dsv4_runtime_close(&runtime);

#ifdef COLI_CUDA
    CHECK(dsv4_dense_arena_enable_cuda(&dense, cuda));
    CHECK(dsv4_expert_cache_enable_cuda(
        &experts, cuda,
        (size_t)DSV4_TOPK * dsv4_expert_payload_bytes()));
#endif

    CHECK(dsv4_runtime_init(&runtime, &store, &dense, &experts, 2));
    CHECK(dsv4_runtime_decode_token(&runtime, 0, logits, 2, 0));
    CHECK(!memcmp(first, logits, sizeof(first)));
    CHECK(__atomic_load_n(&store.raw.read_bytes, __ATOMIC_RELAXED) ==
          decode_read_bytes + expected_expert_bytes);
#ifdef COLI_CUDA
    CHECK(dense.cuda_calls == 9);
    CHECK(experts.cuda_misses == 2 && experts.cuda_hits == 0);
    CHECK(dsv4_runtime_decode_token(&runtime, 1, logits, 2, 0));
    CHECK(dense.cuda_calls == 18);
    CHECK(experts.cuda_misses == 2 && experts.cuda_hits == 2);
    CHECK(dense.cuda_upload_bytes > dense.arena_bytes);
#endif
    dsv4_runtime_close(&runtime);
    dsv4_expert_cache_close(&experts);
    dsv4_dense_arena_close(&dense); dsv4_store_close(&store);
#ifdef COLI_CUDA
    coli_cuda_destroy(cuda);
#endif
    char manifest[1024], segment[1024];
    snprintf(manifest, sizeof(manifest), "%s/model-manifest.json", root);
    snprintf(segment, sizeof(segment), "%s/model.bin", root);
    unlink(manifest); unlink(segment); rmdir(root);
    puts("DeepSeek-V4 one-layer embedding-to-logits runtime test: ok");
    return 0;
}
