#ifndef COLIB_DEEPSEEK_V4_DENSE_H
#define COLIB_DEEPSEEK_V4_DENSE_H

/* One-time host-RAM arena for non-routed weights. The 0731 physical layout is
 * 8.238 GiB for the base dense set and 0.554 GiB for optional DSpark dense
 * weights; the 146.625 GiB of routed experts remain in the bounded tier.
 * Payloads are preserved byte-for-byte and 64-byte aligned for CPU/CUDA use. */

#include "deepseek_v4_model.h"
#include <time.h>
#ifdef COLI_CUDA
#include <pthread.h>
#include "backend_cuda.h"
#endif

#define DSV4_BASE_DENSE_RECORDS 1564
#define DSV4_BASE_DENSE_BYTES UINT64_C(8845959388)
#define DSV4_DSPARK_DENSE_RECORDS 97
#define DSV4_DSPARK_DENSE_BYTES UINT64_C(595190812)
#define DSV4_DENSE_ALIGNMENT 64u
#ifndef DSV4_EXPERT_HIDDEN
#define DSV4_EXPERT_HIDDEN DSV4_DIM
#endif
#ifndef DSV4_MOE_INTERMEDIATE
#define DSV4_MOE_INTERMEDIATE 2048
#endif
#ifndef DSV4_BASE_LAYERS
#define DSV4_BASE_LAYERS 43
#endif

typedef struct {
    dsv4_store *store;
    unsigned char *arena;
    size_t arena_bytes;
    size_t *offsets;
    int records;
    int include_dspark;
    uint64_t payload_bytes;
#ifdef COLI_CUDA
    ColiCuda *cuda;
    unsigned char *cuda_arena;
    void *cuda_activation;
    void *cuda_activation_scale;
    void *cuda_output;
    size_t cuda_activation_cap;
    size_t cuda_activation_scale_cap;
    size_t cuda_output_cap;
    uint64_t cuda_calls;
    uint64_t cuda_upload_bytes;
    uint64_t cuda_download_bytes;
    uint64_t cuda_pinned_upload_bytes;
    uint64_t cuda_pageable_upload_bytes;
    double cuda_fp8_seconds;
    double cuda_bf16_seconds;
    int cuda_lock_initialized;
    pthread_mutex_t cuda_lock;
#endif

    char error[256];
} dsv4_dense_arena;

static inline double dsv4_dense_now_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static inline int dsv4_dense_is_expert(const char *name) {
    return name && strstr(name, ".ffn.experts.") != NULL;
}

static inline int dsv4_dense_is_dspark(const char *name) {
    return name && !strncmp(name, "mtp.", 4);
}

static inline size_t dsv4_dense_align(size_t value) {
    return (value + DSV4_DENSE_ALIGNMENT - 1) &
           ~(size_t)(DSV4_DENSE_ALIGNMENT - 1);
}

static inline int dsv4_dense_arena_init(dsv4_dense_arena *dense,
                                         dsv4_store *store,
                                         int include_dspark, int strict) {
    if (!dense || !store || store->raw.n < 1) return 0;
    memset(dense, 0, sizeof(*dense));
    dense->store = store;
    dense->include_dspark = include_dspark != 0;
    dense->offsets = (size_t *)malloc((size_t)store->raw.n *
                                      sizeof(*dense->offsets));
    if (!dense->offsets) {
        snprintf(dense->error, sizeof(dense->error), "out of memory");
        return 0;
    }
    size_t cursor = 0;
    for (int index = 0; index < store->raw.n; index++) {
        st_tensor *tensor = &store->raw.t[index];
        int selected = !dsv4_dense_is_expert(tensor->name) &&
            (dense->include_dspark || !dsv4_dense_is_dspark(tensor->name));
        dense->offsets[index] = SIZE_MAX;
        if (!selected) continue;
        if (tensor->nbytes < 0 ||
            (uint64_t)tensor->nbytes > SIZE_MAX - cursor) {
            snprintf(dense->error, sizeof(dense->error),
                     "dense arena size overflow at %s", tensor->name);
            free(dense->offsets); dense->offsets = NULL;
            return 0;
        }
        cursor = dsv4_dense_align(cursor);
        dense->offsets[index] = cursor;
        cursor += (size_t)tensor->nbytes;
        dense->payload_bytes += (uint64_t)tensor->nbytes;
        dense->records++;
    }
    dense->arena_bytes = dsv4_dense_align(cursor);
    int expected_records = DSV4_BASE_DENSE_RECORDS +
        (dense->include_dspark ? DSV4_DSPARK_DENSE_RECORDS : 0);
    uint64_t expected_bytes = DSV4_BASE_DENSE_BYTES +
        (dense->include_dspark ? DSV4_DSPARK_DENSE_BYTES : 0);
    if (strict && (dense->records != expected_records ||
                   dense->payload_bytes != expected_bytes)) {
        snprintf(dense->error, sizeof(dense->error),
                 "dense contract expected %d/%llu records/bytes, got %d/%llu",
                 expected_records, (unsigned long long)expected_bytes,
                 dense->records, (unsigned long long)dense->payload_bytes);
        free(dense->offsets); dense->offsets = NULL;
        return 0;
    }
    if (!dense->arena_bytes ||
        posix_memalign((void **)&dense->arena, DSV4_DENSE_ALIGNMENT,
                       dense->arena_bytes) != 0) {
        snprintf(dense->error, sizeof(dense->error),
                 "cannot allocate %zu-byte dense arena", dense->arena_bytes);
        free(dense->offsets); dense->offsets = NULL;
        return 0;
    }
    for (int index = 0; index < store->raw.n; index++)
        if (dense->offsets[index] != SIZE_MAX)
            st_read_raw(&store->raw, store->raw.t[index].name,
                        dense->arena + dense->offsets[index], 1);
    return 1;
}
#ifdef COLI_CUDA
static inline int dsv4_dense_cuda_buffer(
    dsv4_dense_arena *dense, void **buffer, size_t *capacity,
    size_t bytes, const char *label) {
    if (*capacity >= bytes) return 1;
    if (*buffer) coli_cuda_free(dense->cuda, *buffer);
    *buffer = NULL;
    *capacity = 0;
    if (coli_cuda_malloc(dense->cuda, buffer, bytes)) {
        snprintf(dense->error, sizeof(dense->error), "%s: %s", label,
                 coli_cuda_last_error());
        return 0;
    }
    *capacity = bytes;
    return 1;
}

static inline int dsv4_dense_arena_enable_cuda(
    dsv4_dense_arena *dense, ColiCuda *cuda) {
    if (!dense || !dense->arena || !dense->arena_bytes || dense->cuda ||
        !cuda)
        return 0;
    unsigned char *device_arena = NULL;
    int upload_error =
        coli_cuda_malloc(cuda, (void **)&device_arena, dense->arena_bytes);
    void *staging = NULL;
    size_t staging_bytes = 64u * 1024u * 1024u;
    const char *staging_mb = getenv("DSV4_DENSE_PINNED_MB");
    if (staging_mb && *staging_mb) {
        long requested = strtol(staging_mb, NULL, 10);
        if (requested >= 16 && requested <= 512)
            staging_bytes = (size_t)requested * 1024u * 1024u;
    }
    const char *pinned_env = getenv("DSV4_PINNED_UPLOAD");
    int use_pinned = !pinned_env || atoi(pinned_env) != 0;
    int pinned = use_pinned && !upload_error &&
        !coli_cuda_malloc_host(cuda, &staging, staging_bytes);
    if (!upload_error && pinned) {
        for (size_t offset = 0; offset < dense->arena_bytes;) {
            size_t bytes = dense->arena_bytes - offset;
            if (bytes > staging_bytes) bytes = staging_bytes;
            memcpy(staging, dense->arena + offset, bytes);
            if (coli_cuda_upload(
                    cuda, device_arena + offset, staging, bytes) ||
                coli_cuda_sync(cuda)) {
                upload_error = 1;
                break;
            }
            offset += bytes;
        }
    } else if (!upload_error) {
        upload_error =
            coli_cuda_upload(cuda, device_arena, dense->arena,
                             dense->arena_bytes) ||
            coli_cuda_sync(cuda);
    }
    if (staging) coli_cuda_free_host(cuda, staging);
    if (upload_error) {
        if (device_arena) coli_cuda_free(cuda, device_arena);
        coli_cuda_sync(cuda);
        snprintf(dense->error, sizeof(dense->error),
                 "CUDA dense arena upload: %s", coli_cuda_last_error());
        return 0;
    }
    if (pthread_mutex_init(&dense->cuda_lock, NULL)) {
        coli_cuda_free(cuda, device_arena);
        coli_cuda_sync(cuda);
        snprintf(dense->error, sizeof(dense->error),
                 "cannot initialize CUDA dense-arena lock");
        return 0;
    }
    dense->cuda = cuda;
    dense->cuda_arena = device_arena;
    dense->cuda_lock_initialized = 1;
    dense->cuda_upload_bytes = dense->arena_bytes;
    dense->cuda_pinned_upload_bytes =
        pinned ? dense->arena_bytes : 0;
    dense->cuda_pageable_upload_bytes =
        pinned ? 0 : dense->arena_bytes;
    return 1;
}

static inline const unsigned char *dsv4_dense_cuda_pointer(
    const dsv4_dense_arena *dense, const void *host, size_t bytes) {
    uintptr_t begin = (uintptr_t)dense->arena;
    uintptr_t value = (uintptr_t)host;
    if (value < begin || value - begin > dense->arena_bytes ||
        bytes > dense->arena_bytes - (value - begin))
        return NULL;
    return dense->cuda_arena + (value - begin);
}
#endif

static inline int dsv4_dense_linear_fp8(
    const dsv4_dense_arena *view, float *output, const float *input,
    const uint8_t *weight, const uint8_t *weight_scale,
    int batch, int rows, int columns, uint8_t *activation,
    uint8_t *activation_scale) {
    if (!view || !output || !input || !weight || !weight_scale ||
        !activation || !activation_scale || batch < 1 || rows < 1 ||
        columns < 1 || columns % 128)
        return 0;
#ifndef COLI_CUDA
    return dsv4_linear_fp8(
        output, input, weight, weight_scale, batch, rows, columns,
        activation, activation_scale);
#else
    if (!view->cuda)
        return dsv4_linear_fp8(
            output, input, weight, weight_scale, batch, rows, columns,
            activation, activation_scale);
    double cuda_start = dsv4_dense_now_seconds();
    if ((size_t)batch > SIZE_MAX / (size_t)columns ||
        (size_t)rows > SIZE_MAX / (size_t)columns ||
        (size_t)batch > SIZE_MAX / (size_t)rows)
        return 0;
    size_t activation_bytes = (size_t)batch * columns;
    size_t activation_scale_bytes =
        (size_t)batch * (size_t)(columns / 128);
    size_t output_bytes = (size_t)batch * rows * sizeof(float);
    size_t weight_bytes = (size_t)rows * columns;
    size_t weight_scale_bytes =
        (size_t)((rows + 127) / 128) * (size_t)(columns / 128);
    const unsigned char *device_weight =
        dsv4_dense_cuda_pointer(view, weight, weight_bytes);
    const unsigned char *device_scale =
        dsv4_dense_cuda_pointer(view, weight_scale, weight_scale_bytes);
    if (!device_weight || !device_scale) return 0;
    for (int item = 0; item < batch; item++)
        if (!dsv4_act_quant_mxfp(
                input + (size_t)item * columns, columns,
                activation + (size_t)item * columns,
                activation_scale + (size_t)item * (columns / 128)))
            return 0;
    dsv4_dense_arena *dense = (dsv4_dense_arena *)view;
    pthread_mutex_lock(&dense->cuda_lock);
    dense->error[0] = '\0';
    int ok = dsv4_dense_cuda_buffer(
                 dense, &dense->cuda_activation,
                 &dense->cuda_activation_cap, activation_bytes,
                 "CUDA dense activation allocation") &&
             dsv4_dense_cuda_buffer(
                 dense, &dense->cuda_activation_scale,
                 &dense->cuda_activation_scale_cap,
                 activation_scale_bytes,
                 "CUDA dense scale allocation") &&
             dsv4_dense_cuda_buffer(
                 dense, &dense->cuda_output, &dense->cuda_output_cap,
                 output_bytes, "CUDA dense output allocation");
    if (ok)
        ok = !coli_cuda_upload(
                 dense->cuda, dense->cuda_activation,
                 activation, activation_bytes) &&
             !coli_cuda_upload(
                 dense->cuda, dense->cuda_activation_scale,
                 activation_scale, activation_scale_bytes) &&
             !coli_cuda_dsv4_fp8_gemm(
                 dense->cuda, (float *)dense->cuda_output,
                 (const unsigned char *)dense->cuda_activation,
                 (const unsigned char *)dense->cuda_activation_scale,
                 device_weight, device_scale, batch, rows, columns) &&
             !coli_cuda_download(
                 dense->cuda, output, dense->cuda_output, output_bytes) &&
             !coli_cuda_sync(dense->cuda);
    if (ok) {
        dense->cuda_calls++;
        dense->cuda_upload_bytes +=
            activation_bytes + activation_scale_bytes;
        dense->cuda_download_bytes += output_bytes;
    } else if (!dense->error[0]) {
        snprintf(dense->error, sizeof(dense->error),
                 "CUDA dense FP8 projection: %s",
                 coli_cuda_last_error());
    }
    dense->cuda_fp8_seconds += dsv4_dense_now_seconds() - cuda_start;
    pthread_mutex_unlock(&dense->cuda_lock);
    if (ok)
        dsv4_round_bf16_array(output, (size_t)batch * rows);
    return ok;
#endif
}
static inline int dsv4_dense_linear_bf16(
    const dsv4_dense_arena *view, float *output, const float *input,
    const uint16_t *weight, int batch, int rows, int columns) {
    if (!view || !output || !input || !weight || batch < 1 ||
        rows < 1 || columns < 1)
        return 0;
#ifndef COLI_CUDA
    dsv4_bf16_gemm(output, input, weight, batch, rows, columns);
    return 1;
#else
    if (!view->cuda) {
        dsv4_bf16_gemm(output, input, weight, batch, rows, columns);
        return 1;
    }
    double cuda_start = dsv4_dense_now_seconds();
    if ((size_t)rows > SIZE_MAX / (size_t)columns ||
        (size_t)batch > SIZE_MAX / (size_t)columns ||
        (size_t)batch > SIZE_MAX /
            ((size_t)rows * sizeof(float)))
        return 0;
    size_t input_bytes =
        (size_t)batch * columns * sizeof(float);
    size_t output_bytes =
        (size_t)batch * rows * sizeof(float);
    size_t weight_bytes =
        (size_t)rows * columns * sizeof(uint16_t);
    const unsigned char *device_weight =
        dsv4_dense_cuda_pointer(view, weight, weight_bytes);
    if (!device_weight) return 0;
    dsv4_dense_arena *dense = (dsv4_dense_arena *)view;
    pthread_mutex_lock(&dense->cuda_lock);
    dense->error[0] = '\0';
    int ok = dsv4_dense_cuda_buffer(
                 dense, &dense->cuda_activation,
                 &dense->cuda_activation_cap, input_bytes,
                 "CUDA dense BF16 input allocation") &&
             dsv4_dense_cuda_buffer(
                 dense, &dense->cuda_output,
                 &dense->cuda_output_cap, output_bytes,
                 "CUDA dense BF16 output allocation");
    if (ok)
        ok = !coli_cuda_upload(
                 dense->cuda, dense->cuda_activation,
                 input, input_bytes) &&
             !coli_cuda_dsv4_bf16_gemm(
                 dense->cuda, (float *)dense->cuda_output,
                 (const float *)dense->cuda_activation,
                 (const unsigned short *)device_weight,
                 batch, rows, columns) &&
             !coli_cuda_download(
                 dense->cuda, output, dense->cuda_output,
                 output_bytes) &&
             !coli_cuda_sync(dense->cuda);
    if (ok) {
        dense->cuda_calls++;
        dense->cuda_upload_bytes += input_bytes;
        dense->cuda_download_bytes += output_bytes;
    } else if (!dense->error[0]) {
        snprintf(dense->error, sizeof(dense->error),
                 "CUDA dense BF16 projection: %s",
                 coli_cuda_last_error());
    }
    dense->cuda_bf16_seconds += dsv4_dense_now_seconds() - cuda_start;
    pthread_mutex_unlock(&dense->cuda_lock);
    return ok;
#endif
}



static inline const void *dsv4_dense_find(const dsv4_dense_arena *dense,
                                           const char *name,
                                           const dsv4_tensor_desc **descriptor) {
    if (!dense || !dense->arena || !name) return NULL;
    st_tensor *tensor = st_find(&dense->store->raw, name);
    if (!tensor) return NULL;
    ptrdiff_t index = tensor - dense->store->raw.t;
    if (index < 0 || index >= dense->store->raw.n ||
        dense->offsets[index] == SIZE_MAX)
        return NULL;
    if (descriptor) *descriptor = &dense->store->descriptors[index];
    return dense->arena + dense->offsets[index];
}

static inline int dsv4_dense_layer_prefix(char *output, size_t capacity,
                                           int layer) {
    if (!output || !capacity || layer < 0) return 0;
    int length = layer < DSV4_BASE_LAYERS
        ? snprintf(output, capacity, "layers.%d", layer)
        : snprintf(output, capacity, "mtp.%d", layer - DSV4_BASE_LAYERS);
    return length >= 0 && (size_t)length < capacity;
}

static inline const void *dsv4_dense_named(
    const dsv4_dense_arena *dense, const char *prefix, const char *suffix,
    dsv4_dtype dtype, int64_t rows, int64_t columns) {
    char name[192];
    int length = snprintf(name, sizeof(name), "%s%s", prefix, suffix);
    if (length < 0 || length >= (int)sizeof(name)) return NULL;
    const dsv4_tensor_desc *descriptor = NULL;
    const void *value = dsv4_dense_find(dense, name, &descriptor);
    return value && dsv4_model_tensor_shape(descriptor, dtype, 2,
                                             rows, columns) ? value : NULL;
}

static inline int dsv4_dense_embed_token(
    const dsv4_dense_arena *dense, int token, float *hc_output) {
    if (!dense || !hc_output || token < 0 || token >= DSV4_VOCAB)
        return 0;
    const uint16_t *embedding = (const uint16_t *)dsv4_dense_named(
        dense, "embed", ".weight", DSV4_DTYPE_BF16,
        DSV4_VOCAB, DSV4_DIM);
    if (!embedding) return 0;
    const uint16_t *row = embedding + (size_t)token * DSV4_DIM;
    for (int copy = 0; copy < DSV4_HC_MULT; copy++)
        for (int index = 0; index < DSV4_DIM; index++)
            hc_output[(size_t)copy * DSV4_DIM + index] =
                dsv4_bf16(row[index]);
    return 1;
}

static inline int dsv4_dense_fp8_pair(
    const dsv4_dense_arena *dense, const char *prefix,
    int64_t rows, int64_t columns, const uint8_t **weight,
    const uint8_t **scale) {
    *weight = (const uint8_t *)dsv4_dense_named(
        dense, prefix, ".weight", DSV4_DTYPE_FP8_E4M3, rows, columns);
    *scale = (const uint8_t *)dsv4_dense_named(
        dense, prefix, ".scale", DSV4_DTYPE_UE8M0,
        (rows + 127) / 128, (columns + 127) / 128);
    return *weight && *scale;
}

static inline int dsv4_dense_route(
    const dsv4_dense_arena *dense, int layer, int token, const float *input,
    int *indices, float *weights, float *logits) {
    if (!dense || !input || !indices || !weights || !logits || token < 0 ||
        token >= DSV4_VOCAB)
        return 0;
    char layer_prefix[64], gate_prefix[96];
    if (!dsv4_dense_layer_prefix(layer_prefix, sizeof(layer_prefix), layer) ||
        snprintf(gate_prefix, sizeof(gate_prefix), "%s.ffn.gate",
                 layer_prefix) >= (int)sizeof(gate_prefix))
        return 0;
    const uint16_t *gate = (const uint16_t *)dsv4_dense_named(
        dense, gate_prefix, ".weight", DSV4_DTYPE_BF16,
        DSV4_EXPERTS, DSV4_EXPERT_HIDDEN);
    if (!gate) return 0;
    dsv4_bf16_gemm(logits, input, gate, 1, DSV4_EXPERTS,
                    DSV4_EXPERT_HIDDEN);
    int hash_indices[DSV4_TOPK];
    const int *hash = NULL;
    const float *bias = NULL;
    if (layer < 3) {
        const int64_t *table = (const int64_t *)dsv4_dense_named(
            dense, gate_prefix, ".tid2eid", DSV4_DTYPE_I64,
            DSV4_VOCAB, DSV4_TOPK);
        if (!table) return 0;
        for (int route = 0; route < DSV4_TOPK; route++) {
            int64_t expert = table[(size_t)token * DSV4_TOPK + route];
            if (expert < 0 || expert >= DSV4_EXPERTS) return 0;
            hash_indices[route] = (int)expert;
        }
        hash = hash_indices;
    } else {
        char name[128];
        if (snprintf(name, sizeof(name), "%s.bias", gate_prefix) >=
            (int)sizeof(name)) return 0;
        const dsv4_tensor_desc *descriptor = NULL;
        bias = (const float *)dsv4_dense_find(dense, name, &descriptor);
        if (!bias || !descriptor || descriptor->dtype != DSV4_DTYPE_F32 ||
            descriptor->rank != 1 || descriptor->shape[0] != DSV4_EXPERTS)
            return 0;
    }
    return dsv4_route_topk(logits, bias, DSV4_EXPERTS, hash,
                            DSV4_TOPK, indices, weights);
}

static inline int dsv4_dense_shared_expert(
    const dsv4_dense_arena *dense, int layer, const float *input,
    float *output, float *gate, float *up, uint8_t *activation,
    uint8_t *activation_scale) {
    char layer_prefix[64], prefix[128];
    if (!dsv4_dense_layer_prefix(layer_prefix, sizeof(layer_prefix), layer))
        return 0;
    const char *projection[3] = {"w1", "w2", "w3"};
    const uint8_t *weight[3], *scale[3];
    const int64_t rows[3] = {DSV4_MOE_INTERMEDIATE,
                             DSV4_EXPERT_HIDDEN,
                             DSV4_MOE_INTERMEDIATE};
    const int64_t columns[3] = {DSV4_EXPERT_HIDDEN,
                                DSV4_MOE_INTERMEDIATE,
                                DSV4_EXPERT_HIDDEN};
    for (int index = 0; index < 3; index++) {
        int length = snprintf(prefix, sizeof(prefix),
                              "%s.ffn.shared_experts.%s", layer_prefix,
                              projection[index]);
        if (length < 0 || length >= (int)sizeof(prefix) ||
            !dsv4_dense_fp8_pair(dense, prefix, rows[index], columns[index],
                                  &weight[index], &scale[index]))
            return 0;
    }
    if (!dsv4_dense_linear_fp8(
            dense, gate, input, weight[0], scale[0], 1,
            DSV4_MOE_INTERMEDIATE, DSV4_EXPERT_HIDDEN,
            activation, activation_scale) ||
        !dsv4_dense_linear_fp8(
            dense, up, input, weight[2], scale[2], 1,
            DSV4_MOE_INTERMEDIATE, DSV4_EXPERT_HIDDEN,
            activation, activation_scale))
        return 0;
    for (int index = 0; index < DSV4_MOE_INTERMEDIATE; index++)
        gate[index] = dsv4_round_bf16(
            dsv4_clamped_swiglu(gate[index], up[index]));
    return dsv4_dense_linear_fp8(
        dense, output, gate, weight[1], scale[1], 1,
        DSV4_EXPERT_HIDDEN, DSV4_MOE_INTERMEDIATE,
        activation, activation_scale);
}

static inline void dsv4_dense_arena_close(dsv4_dense_arena *dense) {
    if (!dense) return;
#ifdef COLI_CUDA
    if (dense->cuda) {
        if (dense->cuda_lock_initialized)
            pthread_mutex_lock(&dense->cuda_lock);
        if (dense->cuda_arena)
            coli_cuda_free(dense->cuda, dense->cuda_arena);
        if (dense->cuda_activation)
            coli_cuda_free(dense->cuda, dense->cuda_activation);
        if (dense->cuda_activation_scale)
            coli_cuda_free(dense->cuda, dense->cuda_activation_scale);
        if (dense->cuda_output)
            coli_cuda_free(dense->cuda, dense->cuda_output);
        coli_cuda_sync(dense->cuda);
        if (dense->cuda_lock_initialized) {
            pthread_mutex_unlock(&dense->cuda_lock);
            pthread_mutex_destroy(&dense->cuda_lock);
        }
    }
#endif

    free(dense->arena); free(dense->offsets);
    memset(dense, 0, sizeof(*dense));
}

#endif
