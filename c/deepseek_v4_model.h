#ifndef COLIB_DEEPSEEK_V4_MODEL_H
#define COLIB_DEEPSEEK_V4_MODEL_H

/* Model-semantic readers over the manifest-bound DeepSeek store.
 *
 * The converted container preserves checkpoint bytes, so semantic operations
 * must validate dtype and physical shape before interpreting a byte range.
 * These helpers deliberately use bounded row reads: embeddings and the first
 * three hash routers never fault their complete, multi-gigabyte tensors into
 * RAM, while the CPU reference head scans predictable contiguous row blocks.
 */

#include <float.h>
#include <stdarg.h>
#include <stdint.h>

#include "deepseek_v4.h"
#include "deepseek_v4_store.h"

#ifndef DSV4_DIM
#define DSV4_DIM 4096
#endif
#ifndef DSV4_HC_MULT
#define DSV4_HC_MULT 4
#endif
#ifndef DSV4_VOCAB
#define DSV4_VOCAB 129280
#endif
#ifndef DSV4_EXPERTS
#define DSV4_EXPERTS 256
#endif
#ifndef DSV4_TOPK
#define DSV4_TOPK 6
#endif

static int dsv4_model_tensor_shape(const dsv4_tensor_desc *descriptor,
                                    dsv4_dtype dtype, int rank,
                                    int64_t rows, int64_t columns) {
    return descriptor && descriptor->dtype == dtype &&
           descriptor->rank == rank && descriptor->shape[0] == rows &&
           descriptor->shape[1] == columns;
}

static int dsv4_model_read_bf16_row(dsv4_store *store, const char *name,
                                     int64_t row, int64_t rows,
                                     int64_t columns, uint16_t *output,
                                     int direct) {
    const dsv4_tensor_desc *descriptor = dsv4_store_find(store, name);
    if (!output || row < 0 || row >= rows || columns <= 0 ||
        !dsv4_model_tensor_shape(descriptor, DSV4_DTYPE_BF16, 2,
                                 rows, columns))
        return 0;
    if (columns > INT64_MAX / (int64_t)sizeof(uint16_t)) return 0;
    int64_t bytes = columns * (int64_t)sizeof(uint16_t);
    return dsv4_store_read_slice(store, name, row * bytes, bytes, output,
                                 direct);
}

static int dsv4_model_read_i64_row(dsv4_store *store, const char *name,
                                    int64_t row, int64_t rows,
                                    int64_t columns, int64_t *output,
                                    int direct) {
    const dsv4_tensor_desc *descriptor = dsv4_store_find(store, name);
    if (!output || row < 0 || row >= rows || columns <= 0 ||
        !dsv4_model_tensor_shape(descriptor, DSV4_DTYPE_I64, 2,
                                 rows, columns))
        return 0;
    if (columns > INT64_MAX / (int64_t)sizeof(int64_t)) return 0;
    int64_t bytes = columns * (int64_t)sizeof(int64_t);
    return dsv4_store_read_slice(store, name, row * bytes, bytes, output,
                                 direct);
}

static int dsv4_model_embed_token(dsv4_store *store, int token,
                                   float *hc_output, int direct) {
    if (!store || !hc_output || token < 0 || token >= DSV4_VOCAB) return 0;
    uint16_t *row = (uint16_t *)malloc(DSV4_DIM * sizeof(*row));
    if (!row) return 0;
    int ok = dsv4_model_read_bf16_row(store, "embed.weight", token,
                                      DSV4_VOCAB, DSV4_DIM, row, direct);
    if (ok) {
        for (int copy = 0; copy < DSV4_HC_MULT; copy++)
            for (int index = 0; index < DSV4_DIM; index++)
                hc_output[(int64_t)copy * DSV4_DIM + index] =
                    dsv4_bf16(row[index]);
    }
    free(row);
    return ok;
}

static int dsv4_model_hash_route(dsv4_store *store, int layer, int token,
                                  int indices[DSV4_TOPK], int direct) {
    if (!store || !indices || layer < 0 || layer >= 3 || token < 0 ||
        token >= DSV4_VOCAB)
        return 0;
    char name[96];
    if (snprintf(name, sizeof(name), "layers.%d.ffn.gate.tid2eid", layer) >=
        (int)sizeof(name))
        return 0;
    int64_t raw[DSV4_TOPK];
    if (!dsv4_model_read_i64_row(store, name, token, DSV4_VOCAB,
                                 DSV4_TOPK, raw, direct))
        return 0;
    for (int index = 0; index < DSV4_TOPK; index++) {
        if (raw[index] < 0 || raw[index] >= DSV4_EXPERTS) return 0;
        indices[index] = (int)raw[index];
    }
    return 1;
}

/* CPU correctness fallback for the output head. Rows are read in bounded,
 * contiguous blocks so callers can choose a memory/throughput tradeoff without
 * changing logits. The production CUDA path can consume the same row blocks. */
static int dsv4_model_head_logits(dsv4_store *store, const float *hidden,
                                   float *logits, int rows_per_read,
                                   int direct) {
    const dsv4_tensor_desc *descriptor =
        dsv4_store_find(store, "head.weight");
    if (!store || !hidden || !logits || rows_per_read < 1 ||
        !dsv4_model_tensor_shape(descriptor, DSV4_DTYPE_BF16, 2,
                                 DSV4_VOCAB, DSV4_DIM))
        return 0;
    if (rows_per_read > DSV4_VOCAB) rows_per_read = DSV4_VOCAB;
    size_t capacity = (size_t)rows_per_read * DSV4_DIM;
    if (capacity > SIZE_MAX / sizeof(uint16_t)) return 0;
    uint16_t *weights = (uint16_t *)malloc(capacity * sizeof(*weights));
    if (!weights) return 0;
    for (int base = 0; base < DSV4_VOCAB; base += rows_per_read) {
        int count = DSV4_VOCAB - base;
        if (count > rows_per_read) count = rows_per_read;
        int64_t elements = (int64_t)count * DSV4_DIM;
        int64_t offset = (int64_t)base * DSV4_DIM * sizeof(uint16_t);
        int64_t bytes = elements * (int64_t)sizeof(uint16_t);
        if (!dsv4_store_read_slice(store, "head.weight", offset, bytes,
                                   weights, direct)) {
            free(weights);
            return 0;
        }
#pragma omp parallel for schedule(static)
        for (int row = 0; row < count; row++) {
            double sum = 0.0;
            const uint16_t *weight = weights + (int64_t)row * DSV4_DIM;
            for (int column = 0; column < DSV4_DIM; column++)
                sum += (double)hidden[column] * dsv4_bf16(weight[column]);
            logits[base + row] = (float)sum;
        }
    }
    free(weights);
    return 1;
}


typedef struct {
    dsv4_store *store;
    int base_layers;
    int dspark_layers;
    char error[512];
} dsv4_model_contract;

static int dsv4_contract_fail(dsv4_model_contract *contract,
                              const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(contract->error, sizeof(contract->error), format, arguments);
    va_end(arguments);
    return 0;
}

static int dsv4_contract_name(char *output, size_t capacity,
                              const char *prefix, const char *suffix) {
    int length = snprintf(output, capacity, "%s%s", prefix, suffix);
    return length >= 0 && (size_t)length < capacity;
}

static int dsv4_contract_expect1(dsv4_model_contract *contract,
                                 const char *prefix, const char *suffix,
                                 dsv4_dtype dtype, int64_t dimension) {
    char name[192];
    if (!dsv4_contract_name(name, sizeof(name), prefix, suffix))
        return dsv4_contract_fail(contract, "tensor name is too long: %s%s",
                                  prefix, suffix);
    const dsv4_tensor_desc *descriptor =
        dsv4_store_find(contract->store, name);
    if (!descriptor || descriptor->dtype != dtype || descriptor->rank != 1 ||
        descriptor->shape[0] != dimension)
        return dsv4_contract_fail(contract,
                                  "tensor contract mismatch: %s", name);
    return 1;
}

static int dsv4_contract_expect2(dsv4_model_contract *contract,
                                 const char *prefix, const char *suffix,
                                 dsv4_dtype dtype, int64_t rows,
                                 int64_t columns) {
    char name[192];
    if (!dsv4_contract_name(name, sizeof(name), prefix, suffix))
        return dsv4_contract_fail(contract, "tensor name is too long: %s%s",
                                  prefix, suffix);
    const dsv4_tensor_desc *descriptor =
        dsv4_store_find(contract->store, name);
    if (!dsv4_model_tensor_shape(descriptor, dtype, 2, rows, columns))
        return dsv4_contract_fail(contract,
                                  "tensor contract mismatch: %s", name);
    return 1;
}

static int dsv4_contract_fp8(dsv4_model_contract *contract,
                             const char *prefix, int64_t rows,
                             int64_t columns) {
    return dsv4_contract_expect2(contract, prefix, ".weight",
                                 DSV4_DTYPE_FP8_E4M3, rows, columns) &&
           dsv4_contract_expect2(contract, prefix, ".scale",
                                 DSV4_DTYPE_UE8M0, (rows + 127) / 128,
                                 (columns + 127) / 128);
}

static int dsv4_contract_fp4_expert(dsv4_model_contract *contract,
                                    const char *prefix) {
    return dsv4_contract_expect2(contract, prefix, ".w1.weight",
                                 DSV4_DTYPE_I8, 2048, 2048) &&
           dsv4_contract_expect2(contract, prefix, ".w1.scale",
                                 DSV4_DTYPE_UE8M0, 2048, 128) &&
           dsv4_contract_expect2(contract, prefix, ".w2.weight",
                                 DSV4_DTYPE_I8, 4096, 1024) &&
           dsv4_contract_expect2(contract, prefix, ".w2.scale",
                                 DSV4_DTYPE_UE8M0, 4096, 64) &&
           dsv4_contract_expect2(contract, prefix, ".w3.weight",
                                 DSV4_DTYPE_I8, 2048, 2048) &&
           dsv4_contract_expect2(contract, prefix, ".w3.scale",
                                 DSV4_DTYPE_UE8M0, 2048, 128);
}

static int dsv4_contract_block(dsv4_model_contract *contract,
                               const char *prefix, int hashed) {
    if (!dsv4_contract_expect1(contract, prefix, ".attn.attn_sink",
                               DSV4_DTYPE_F32, 64) ||
        !dsv4_contract_expect1(contract, prefix, ".attn.kv_norm.weight",
                               DSV4_DTYPE_BF16, 512) ||
        !dsv4_contract_expect1(contract, prefix, ".attn.q_norm.weight",
                               DSV4_DTYPE_BF16, 1024))
        return 0;
    const struct { const char *suffix; int64_t rows, columns; } projections[] = {
        {".attn.wkv", 512, 4096},
        {".attn.wo_a", 8192, 4096},
        {".attn.wo_b", 4096, 8192},
        {".attn.wq_a", 1024, 4096},
        {".attn.wq_b", 32768, 1024},
        {".ffn.shared_experts.w1", 2048, 4096},
        {".ffn.shared_experts.w2", 4096, 2048},
        {".ffn.shared_experts.w3", 2048, 4096},
    };
    char projection[192];
    for (size_t index = 0;
         index < sizeof(projections) / sizeof(projections[0]); index++) {
        if (!dsv4_contract_name(projection, sizeof(projection), prefix,
                                projections[index].suffix) ||
            !dsv4_contract_fp8(contract, projection,
                               projections[index].rows,
                               projections[index].columns))
            return 0;
    }
    if (!dsv4_contract_expect1(contract, prefix, ".attn_norm.weight",
                               DSV4_DTYPE_BF16, 4096) ||
        !dsv4_contract_expect2(contract, prefix, ".ffn.gate.weight",
                               DSV4_DTYPE_BF16, 256, 4096) ||
        !dsv4_contract_expect1(contract, prefix, ".ffn_norm.weight",
                               DSV4_DTYPE_BF16, 4096))
        return 0;
    if (hashed) {
        if (!dsv4_contract_expect2(contract, prefix,
                                   ".ffn.gate.tid2eid", DSV4_DTYPE_I64,
                                   129280, 6))
            return 0;
    } else if (!dsv4_contract_expect1(contract, prefix, ".ffn.gate.bias",
                                      DSV4_DTYPE_F32, 256)) {
        return 0;
    }
    for (const char *operation = "attn"; operation;
         operation = !strcmp(operation, "attn") ? "ffn" : NULL) {
        char suffix[48];
        snprintf(suffix, sizeof(suffix), ".hc_%s_base", operation);
        if (!dsv4_contract_expect1(contract, prefix, suffix,
                                   DSV4_DTYPE_F32, 24))
            return 0;
        snprintf(suffix, sizeof(suffix), ".hc_%s_fn", operation);
        if (!dsv4_contract_expect2(contract, prefix, suffix,
                                   DSV4_DTYPE_F32, 24, 16384))
            return 0;
        snprintf(suffix, sizeof(suffix), ".hc_%s_scale", operation);
        if (!dsv4_contract_expect1(contract, prefix, suffix,
                                   DSV4_DTYPE_F32, 3))
            return 0;
    }
    for (int expert = 0; expert < DSV4_EXPERTS; expert++) {
        char expert_prefix[192];
        int length = snprintf(expert_prefix, sizeof(expert_prefix),
                              "%s.ffn.experts.%d", prefix, expert);
        if (length < 0 || length >= (int)sizeof(expert_prefix) ||
            !dsv4_contract_fp4_expert(contract, expert_prefix))
            return 0;
    }
    return 1;
}

static int dsv4_contract_compressor(dsv4_model_contract *contract,
                                    const char *prefix, int ratio,
                                    int dimension) {
    int coefficient = ratio == 4 ? 2 : 1;
    return dsv4_contract_expect2(contract, prefix, ".ape", DSV4_DTYPE_F32,
                                 ratio, coefficient * dimension) &&
           dsv4_contract_expect1(contract, prefix, ".norm.weight",
                                 DSV4_DTYPE_BF16, dimension) &&
           dsv4_contract_expect2(contract, prefix, ".wgate.weight",
                                 DSV4_DTYPE_BF16,
                                 coefficient * dimension, DSV4_DIM) &&
           dsv4_contract_expect2(contract, prefix, ".wkv.weight",
                                 DSV4_DTYPE_BF16,
                                 coefficient * dimension, DSV4_DIM);
}

static int dsv4_model_bind_contract(dsv4_model_contract *contract,
                                    dsv4_store *store) {
    if (!contract || !store) return 0;
    memset(contract, 0, sizeof(*contract));
    contract->store = store;
    if (store->records != 72317)
        return dsv4_contract_fail(contract,
                                  "expected 72317 tensors, found %d",
                                  store->records);
    if (!dsv4_contract_expect2(contract, "embed", ".weight",
                               DSV4_DTYPE_BF16, DSV4_VOCAB, DSV4_DIM) ||
        !dsv4_contract_expect2(contract, "head", ".weight",
                               DSV4_DTYPE_BF16, DSV4_VOCAB, DSV4_DIM) ||
        !dsv4_contract_expect1(contract, "norm", ".weight",
                               DSV4_DTYPE_BF16, DSV4_DIM) ||
        !dsv4_contract_expect1(contract, "hc_head", "_base",
                               DSV4_DTYPE_F32, 4) ||
        !dsv4_contract_expect2(contract, "hc_head", "_fn",
                               DSV4_DTYPE_F32, 4, 16384) ||
        !dsv4_contract_expect1(contract, "hc_head", "_scale",
                               DSV4_DTYPE_F32, 1))
        return 0;
    for (int layer = 0; layer < 43; layer++) {
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "layers.%d", layer);
        if (!dsv4_contract_block(contract, prefix, layer < 3)) return 0;
        int ratio = layer < 2 ? 0 : (layer % 2 ? 128 : 4);
        if (ratio) {
            char compressor[96];
            snprintf(compressor, sizeof(compressor),
                     "%s.attn.compressor", prefix);
            if (!dsv4_contract_compressor(contract, compressor, ratio, 512))
                return 0;
            if (ratio == 4) {
                snprintf(compressor, sizeof(compressor),
                         "%s.attn.indexer.compressor", prefix);
                if (!dsv4_contract_compressor(contract, compressor, 4, 128) ||
                    !dsv4_contract_expect2(contract, prefix,
                         ".attn.indexer.weights_proj.weight",
                         DSV4_DTYPE_BF16, 64, 4096))
                    return 0;
                char indexer[96];
                snprintf(indexer, sizeof(indexer), "%s.attn.indexer.wq_b",
                         prefix);
                if (!dsv4_contract_fp8(contract, indexer, 8192, 1024))
                    return 0;
            }
        }
        contract->base_layers++;
    }
    for (int stage = 0; stage < 3; stage++) {
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "mtp.%d", stage);
        if (!dsv4_contract_block(contract, prefix, 0)) return 0;
        contract->dspark_layers++;
    }
    if (!dsv4_contract_expect1(contract, "mtp.0.main_norm", ".weight",
                               DSV4_DTYPE_BF16, DSV4_DIM) ||
        !dsv4_contract_fp8(contract, "mtp.0.main_proj", 4096, 12288) ||
        !dsv4_contract_expect2(contract, "mtp.2.confidence_head.proj",
                               ".weight", DSV4_DTYPE_BF16, 1, 4352) ||
        !dsv4_contract_expect1(contract, "mtp.2.hc_head", "_base",
                               DSV4_DTYPE_F32, 4) ||
        !dsv4_contract_expect2(contract, "mtp.2.hc_head", "_fn",
                               DSV4_DTYPE_F32, 4, 16384) ||
        !dsv4_contract_expect1(contract, "mtp.2.hc_head", "_scale",
                               DSV4_DTYPE_F32, 1) ||
        !dsv4_contract_expect2(contract, "mtp.2.markov_head.markov_w1",
                               ".weight", DSV4_DTYPE_BF16, DSV4_VOCAB, 256) ||
        !dsv4_contract_expect2(contract, "mtp.2.markov_head.markov_w2",
                               ".weight", DSV4_DTYPE_BF16, DSV4_VOCAB, 256) ||
        !dsv4_contract_expect1(contract, "mtp.2.norm", ".weight",
                               DSV4_DTYPE_BF16, DSV4_DIM))
        return 0;
    return 1;
}

#endif
